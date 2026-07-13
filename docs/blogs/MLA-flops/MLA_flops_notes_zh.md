---
title: "MLA FLOPs、RoPE、Query Compression 与 MTP 笔记"
description: "一篇关于 MLA FLOPs、矩阵乘顺序、RoPE decoupling、query compression，以及为什么 MTP 不能免费复用 main MLA cache 的技术笔记。"
date: 2026-07-06
outline: deep
---

# MLA FLOPs、RoPE、Query Compression 与 MTP 笔记

本文关心的是 DeepSeek-V2 MLA 里一个很关键的问题：MLA 到底省了什么，FLOPs 又被转移到了哪里，RoPE等其他部分在MLA中的计算方式会有什么变化？本文会先比较 MHA 和 MLA 在 prefill/training/decode 中的 FLOPs 差异，代入论文中真实参数分析FLOPs计算的具体细节，再顺着同一套记号讨论 RoPE、query compression 和 MTP。这样做可以让很多看起来像“架构设计”的问题，其实可以直接从矩阵乘顺序和 cache format 中看出来。

基本上，全文可以分成四部分：

- MHA 和 MLA 在 prefill/training/decode 中的 FLOPs 对比。
- 为什么 RoPE 会阻碍 MLA absorption。
- 为什么 MLA 里还要压缩 $Q$。
- 为什么 MTP 会削弱 MLA 的加速收益。


![MLA sketch](MLA_baseline.png)

## Notation

为方便推导，本文把一次 multiply-add 记为 $2$ FLOPs， 并且只统计矩阵乘的 FLOPs，暂时忽略 softmax、mask、normalization、RoPE elementwise rotation 和 bias term。MLA 推导里用到的矩阵的主要维度如下：

| Symbol | Meaning | Shape / role |
| --- | --- | --- |
| $T$ | prefill sequence length | number of tokens in prefill or training |
| $t$ | decode KV-cache length | number of cached historical tokens |
| $d$ | hidden size | $x_i \in \mathbb{R}^{d}$ |
| $h$ | number of attention heads | head index $s = 1,\ldots,h$ |
| $d_k$ | per-head query/key dimension | $q_i^s,k_i^s \in \mathbb{R}^{d_k}$ |
| $d_v$ | per-head value dimension | $v_i^s,o_i^s \in \mathbb{R}^{d_v}$ |
| $d_c$ | MLA KV latent dimension | $c_i^{KV} \in \mathbb{R}^{d_c}$ |
| $d_{qc}$ | MLA query latent dimension | $c_i^Q \in \mathbb{R}^{d_{qc}}$ |

为了避免 head index 掩盖 shared latent，下面更有用的 tensor 视角是：

$$
\begin{aligned}
W_Q,W_K&:(h,d,d_k),&
W_V&:(h,d,d_v),&
W_o&:(h,d_v,d),\\
W_{DKV}&:(d,d_c),&
W_{UK}&:(h,d_c,d_k),&
W_{UV}&:(h,d_c,d_v),\\
W_{DQ}&:(d,d_{qc}),&
W_{UQ}&:(h,d_{qc},d_k).
\end{aligned}
$$

因此 $C^{KV}:(T,d_c)$ 和 $C^Q:(T,d_{qc})$ 是所有 heads 共享的 latent，而 $Q,K,V$ 位于 head space，例如 $Q:(T,h,d_k)$，$V:(T,h,d_v)$。下面有些公式仍然保留单 head slice 来方便阅读，但真正的 storage/computation convention 应该按这里的 tensor shape 理解。

## Baseline FLOPs

先从 baseline multi-head attention (MHA) 开始， 推导每一步的矩阵乘和 FLOPs。便于与MLA对比，理解MLA的优化到底在哪里。

### MHA Prefill
首先从输入 $X$ 计算 $Q,K,V$ projection：$Q=XW_Q,\quad K=XW_K,\quad V=XW_V.$

对于 MHA prefill，完整 score 表达式 $S$ ，attention output $O$ ，output projection $Y$ 计算公式如下，其中 $P$ 是 softmax 之后的 attention matrix：

$$
S
=
QK^\top
=
\underset{(T,d)}{X}
\underset{(h,d,d_k)}{W_Q}
\underset{(h,d_k,d)}{W_K^\top}
\underset{(d,T)}{X^\top}.
$$

$$
O=PV=
\underset{(h,T,T)}{P}
\underset{(T,d)}{X}
\underset{(h,d,d_v)}{W_V}.
$$

$$
Y
=
\underset{(T,h,d_v)}{O}
\underset{(h,d_v,d)}{W_o}.
$$

对于attention score $S$ 的计算，从 FLOPs 角度看，关键是让 dominating 的 $T^2$ 项系数尽可能小。由于 $d_k<d$，合理的计算顺序是：

$$
\underset{(T,d)}{X}
\underset{(h,d,d_k)}{W_Q}
\rightarrow
\underset{(T,h,d_k)}{Q},
\qquad
\mathrm{FLOPs}=2Thdd_k.
$$
$$
\underset{(T,d)}{X}
\underset{(h,d,d_k)}{W_K}
\rightarrow
\underset{(T,h,d_k)}{K},
\qquad
\mathrm{FLOPs}=2Thdd_k.
$$
$$
\underset{(T,h,d_k)}{Q}
\underset{(h,d_k,T)}{K^\top}
\rightarrow
\underset{(h,T,T)}{S},
\qquad
\mathrm{FLOPs}=2hT^2d_k,
$$

然后计算 $O$，$Y$ 的 FLOPs，这里可以把 value/output path 合在一起看。按 tensor shape 写，可以理解为 head-wise 的四矩阵乘法：

$$
\underset{(h,T,T)}{P}
\underset{(T,d)}{X}
\underset{(h,d,d_v)}{W_V}
\underset{(h,d_v,d)}{W_o}
\rightarrow
\underset{(T,d)}{Y}.
$$

最优顺序是先把 $X$ 投影到 $d_v$，再乘 $P$。对所有 heads 来说，标准顺序是：

$$
XW_V \rightarrow V,\qquad PV \rightarrow O,\qquad OW_o \rightarrow Y,
\quad
F=2Thdd_v+2hT^2d_v+2Thd_vd.
$$

如果先算 $PX$，那么 cost 变成 $2hT^2d+2Thdd_v+2Thd_vd$。如果先合并 $W_VW_o$，后面的 attention 乘法二次项系数也会变成 $d$ 而不是 $d_v$。由于通常 $d_v\ll d$，标准顺序更优，因为 quadratic term 是 $2hT^2d_v$，而不是 $2hT^2d$。

> 但对于 causal prefill/training，实现上只需要计算 lower triangle。可见的 query-key pair 数量是： $\frac{T(T+1)}{2}.$
> 所以 causal-aware attention FLOPs 是：
> $F_{\mathrm{attn,causal}}=hT(T+1)(d_k+d_v).$
> 因此， causal-aware prefill FLOPs 是：
>$$
>F_{\mathrm{MHA,prefill}}=2Tdh(2d_k+d_v)
>+2hT^2(d_k+d_v)
>+2Thd_vd.
>$$
> 如果实现中显式考虑 causal mask，把中间项替换为 $hT(T+1)(d_k+d_v)$。

### MLA Prefill

对于 MLA prefill，首先计算共享的 KV latent：$C^{KV}=XW_{DKV}$。这里的 $C^{KV}$ 是所有 heads 共用的 latent 表示。

**Without Query Compression.**

对于不使用 query compression 的 MLA，完整 score 表达式 $S$ 和 attention output $O$ 如下：

$$
S
=
QK^\top
=
\underset{(T,d)}{X}
\underset{(h,d,d_k)}{W_Q}
\underset{(h,d_k,d_c)}{W_{UK}^\top}
\underset{(d_c,d)}{W_{DKV}^\top}
\underset{(d,T)}{X^\top}.
$$


$$
O=PV
=
\underset{(h,T,T)}{P}
\underset{(T,d)}{X}
\underset{(d,d_c)}{W_{DKV}}
\underset{(h,d_c,d_v)}{W_{UV}}.
$$

从 FLOPs 角度看，仍然要盯住 $T^2$ 项的系数。通常 $d_k<d_c$，所以 prefill 里不应该先算 $Q^s(W_{UK}^s)^\top$ 再乘 $(C^{KV})^\top$；这样会把 dominating 的 $QK^\top$ 项变成 $2T^2d_c$。

key side 还有另一个计算顺序问题。按 tensor shape 看，score 里的 key side 是：

$$
\underset{(T,d)}{X}
\underset{(d,d_c)}{W_{DKV}}
\underset{(h,d_c,d_k)}{W_{UK}}
\rightarrow
\underset{(T,h,d_k)}{K}.
$$

shared-latent 顺序是：

$$
XW_{DKV}\rightarrow C^{KV},\qquad C^{KV}W_{UK}\rightarrow K,
\quad
F=2Tdd_c+2Thd_cd_k.
$$

如果先形成 effective per-head key projection $W_{DKV}W_{UK}$，再和 $X$ 相乘，那么 activation-side cost 是：

$$
X(W_{DKV}W_{UK})\rightarrow K,
\quad
F=2Thdd_k,
$$

这里还没有算额外的一次性 weight precomposition cost $2hdd_cd_k$。代入 DeepSeek-V2 参数 $d=5120,\ d_c=512,\ d_k=128,\ h=128$，shared-latent 顺序约为 $22.0\text{M}T$ FLOPs，而 effective-$W_K$ 顺序是 $167.8\text{M}T$ FLOPs。也就是说，便宜的路径是先算 $XW_{DKV}$，先利用共享的 $d_c$ bottleneck，再扩展到各个 heads。

因此计算顺序是：

$$
\underset{(T,d)}{X}
\underset{(d,d_c)}{W_{DKV}}
\rightarrow
\underset{(T,d_c)}{C^{KV}},
\qquad
\mathrm{FLOPs}=2Tdd_c
\quad \text{shared across heads}.
$$

$$
\underset{(T,d)}{X}
\underset{(h,d,d_k)}{W_Q}
\rightarrow
\underset{(T,h,d_k)}{Q},
\qquad
\mathrm{FLOPs}=2Thdd_k.
$$

$$
\underset{(T,d_c)}{C^{KV}}
\underset{(h,d_c,d_k)}{W_{UK}}
\rightarrow
\underset{(T,h,d_k)}{K},
\qquad
\mathrm{FLOPs}=2Thd_cd_k.
$$

$$
\underset{(T,h,d_k)}{Q}
\underset{(h,d_k,T)}{K^\top}
\rightarrow
\underset{(h,T,T)}{S},
\qquad
\mathrm{FLOPs}=2hT^2d_k.
$$

然后计算 $O$ 的 FLOPs，其中 $O=PV$，$V=C^{KV}W_{UV}$：

$$
\underset{(T,d_c)}{C^{KV}}
\underset{(h,d_c,d_v)}{W_{UV}}
\rightarrow
\underset{(T,h,d_v)}{V},
\qquad
\mathrm{FLOPs}=2Thd_cd_v.
$$

$$
\underset{(h,T,T)}{P}
\underset{(T,h,d_v)}{V}
\rightarrow
\underset{(T,h,d_v)}{O},
\qquad
\mathrm{FLOPs}=2hT^2d_v.
$$

这个顺序在 kernel 层面是可实现的。最直接的做法是先 materialize $K,V$，然后调用标准 FlashAttention kernel；更节省内存的做法则是在每个 attention tile 内从 $C^{KV}$ 计算 $K,V$。

不使用 query compression 时，causal-aware MLA prefill FLOPs 是：

$$
F_{\mathrm{MLA,prefill,noQ}}
=
2Tdd_c
+2Thdd_k
+2Thd_c(d_k+d_v)
+2hT^2(d_k+d_v)
+2Thd_vd.
$$

> 如果实现中显式考虑 causal mask，把中间的 attention 项替换为 $hT(T+1)(d_k+d_v)$。

**With Query Compression.**

使用 query compression 后，完整 score 表达式变成：

$$
S
=
QK^\top
=
\underset{(T,d)}{X}
\underset{(d,d_{qc})}{W_{DQ}}
\underset{(h,d_{qc},d_k)}{W_{UQ}}
\underset{(h,d_k,d_c)}{W_{UK}^\top}
\underset{(d_c,d)}{W_{DKV}^\top}
\underset{(d,T)}{X^\top}.
$$

attention output $O$ 保持不变：

$$
O=PV
=
\underset{(h,T,T)}{P}
\underset{(T,d)}{X}
\underset{(d,d_c)}{W_{DKV}}
\underset{(h,d_c,d_v)}{W_{UV}}.
$$

最优计算顺序是：

$$
\underset{(T,d)}{X}
\underset{(d,d_{qc})}{W_{DQ}}
\rightarrow
\underset{(T,d_{qc})}{C^Q},
\qquad
\mathrm{FLOPs}=2Tdd_{qc}
\quad \text{shared across heads}.
$$

$$
\underset{(T,d_{qc})}{C^Q}
\underset{(h,d_{qc},d_k)}{W_{UQ}}
\rightarrow
\underset{(T,h,d_k)}{Q},
\qquad
\mathrm{FLOPs}=2Thd_{qc}d_k.
$$

$C^{KV},K,V,S,O$ 路径和 no-Q-compression 情况相同。使用 query compression 时，non-causal-aware MLA prefill 总 FLOPs 是：

$$
F_{\mathrm{MLA,prefill,Qcomp}}
=
2Tdd_c
+2Tdd_{qc}
+2Thd_{qc}d_k
+2Thd_c(d_k+d_v)
+2hT^2(d_k+d_v)
+2Thd_vd.
$$

> 如果实现中显式考虑 causal mask，把中间的 attention 项替换为 $hT(T+1)(d_k+d_v)$。

### Prefill Comparison

从上面的公式可以看到，如果二者都显式计算同样形状的 $QK^\top$ 和 $PV$，那么 MHA 和 MLA 的 quadratic attention FLOPs 是相同的；prefill 阶段真正的差异主要来自 linear projection part。

MHA 的 linear projection FLOPs 是 $2Tdh(2d_k+d_v)$。MLA without query compression 是 $2Tdd_c+2Thdd_k+2Thd_c(d_k+d_v)$。MLA with query compression 是 $2Tdd_c+2Tdd_{qc}+2Thd_{qc}d_k+2Thd_c(d_k+d_v)$。

代入 DeepSeek-V2 attention 配置 $d=5120,\ h=128,\ d_k=d_v=128,\ d_c=512,\ d_{qc}=1536$，MHA linear projection cost 是 $503.3\text{M}T$，MLA without query compression 是 $206.6\text{M}T$，MLA with query compression 是 $104.9\text{M}T$。也就是说，MLA no-Q 约为 MHA 的 $41.1\%$，MLA Q-compression 约为 MHA 的 $20.8\%$。

**MLA prefill 不会降低 quadratic attention FLOPs，但会显著降低 linear projection FLOPs；with query compression 时这一点尤其明显**。在 DeepSeek-V2 配置下，MLA with query compression 的 projection FLOPs 只有 MHA 的约 $20.8\%$。

### MHA Decode

有 KV cache 时，历史的 $K_{\le t},V_{\le t}$ 已经在之前的 decode steps 中算好。当前 step 只需要计算并 append 新的 $q,k,v$：

$$
\underset{(1,d)}{x}
\underset{(h,d,d_k)}{W_Q}
\rightarrow
\underset{(1,h,d_k)}{q},
\qquad
\underset{(1,d)}{x}
\underset{(h,d,d_k)}{W_K}
\rightarrow
\underset{(1,h,d_k)}{k},
\qquad
\underset{(1,d)}{x}
\underset{(h,d,d_v)}{W_V}
\rightarrow
\underset{(1,h,d_v)}{v}.
$$

对所有 heads 来说，当前 token projection cost 是：$F_{qkv}=2dh(2d_k+d_v).$

$$
\underset{(1,d)}{x}
\underset{(h,d,d_k)}{W_Q,W_K}
\rightarrow
\underset{(1,h,d_k)}{q,k},
\qquad
\underset{(1,d)}{x}
\underset{(h,d,d_v)}{W_V}
\rightarrow
\underset{(1,h,d_v)}{v}.
$$

这里的计算顺序实际上由 KV cache 决定：先从当前 token 的 $x$ 生成 $q,k,v$，其中 $k,v$ 被 append 到 cache；然后只用当前 $q$ 去读历史 $K_{\le t}$ 计算 scores。这样 decode-length 相关的乘法发生在 $d_k,d_v$ 维度上，而不是重新从 hidden dimension $d$ 处理整段 $X_{\le t}$。然后 attention 读取 cached $K,V$：

$$
score=qK_{\le t}^\top,\qquad o=pV_{\le t}.
$$

$$
\underset{(1,h,d_k)}{q}
\underset{(h,d_k,t)}{K_{\le t}^\top}
\rightarrow
\underset{(1,h,t)}{score},
\qquad
\mathrm{FLOPs}=2htd_k.
$$

$$
\underset{(1,h,t)}{p}
\underset{(t,h,d_v)}{V_{\le t}}
\rightarrow
\underset{(1,h,d_v)}{o},
\qquad
\mathrm{FLOPs}=2htd_v.
$$

$$
\underset{(1,h,d_v)}{o}
\underset{(h,d_v,d)}{W_o}
\rightarrow
\underset{(1,d)}{y},
\qquad
\mathrm{FLOPs}=2hd_vd.
$$

所以 cached decode FLOPs 是：$F_{\mathrm{MHA,decode,cached}}=
2dh(2d_k+d_v)
+2ht(d_k+d_v)
+2hd_vd .$


### MLA Decode

对于 MLA decode，cache 中存的不是 full per-head $K,V$，而是共享的 latent cache。在当前 step，我们 append 一个新的 latent vector：

$$
\underset{(t,d_c)}{C^{KV}_{\le t}},
\qquad
\underset{(1,d)}{x}
\underset{(d,d_c)}{W_{DKV}}
\rightarrow
\underset{(1,d_c)}{c^{KV}},
\qquad
\mathrm{FLOPs}=2dd_c.
$$

因此当前 token 的 linear projection cost 是明确包含在 decode 里的。没有 query compression 时：

$$
F_{\mathrm{MLA,current,noQ}}=2dd_c+2hdd_k.
$$

有 query compression 时：

$$
F_{\mathrm{MLA,current,Qcomp}}=2dd_c+2dd_{qc}+2hd_{qc}d_k.
$$

MLA decode 不需要额外 append full $k,v$；真正写入 cache 的是 shared $c^{KV}$ latent，$K,V$ 只在 attention 需要时 reconstruct 或 absorb。

完整的 score 和 value 表达式如下。这里保留两个变体：MLA without query compression 和 MLA with query compression。它们的区别主要在 current-token query path，而 latent-cache attention path 是共享的。

$$
\begin{aligned}
score_{\mathrm{noQ}}
&=
\underset{(1,d)}{x}
\underset{(h,d,d_k)}{W_Q}
\underset{(h,d_k,d_c)}{W_{UK}^\top}
\underset{(d_c,t)}{(C^{KV}_{\le t})^\top},
\\
score_{\mathrm{Qcomp}}
&=
\underset{(1,d)}{x}
\underset{(d,d_{qc})}{W_{DQ}}
\underset{(h,d_{qc},d_k)}{W_{UQ}}
\underset{(h,d_k,d_c)}{W_{UK}^\top}
\underset{(d_c,t)}{(C^{KV}_{\le t})^\top},
\\
o
&=
\underset{(1,h,t)}{p}
\underset{(t,d_c)}{C^{KV}_{\le t}}
\underset{(h,d_c,d_v)}{W_{UV}}.
\end{aligned}
$$

对于输入 query $x$，cached execution 首先构造当前 query。没有 query compression 时：

$$
\underset{(1,d)}{x}
\underset{(h,d,d_k)}{W_Q}
\rightarrow
\underset{(1,h,d_k)}{q},
\qquad
\mathrm{FLOPs}=2hdd_k.
$$

有 query compression 时，$q$ 的计算被 factorize 为：

$$
\begin{aligned}
\underset{(1,d)}{x}
\underset{(d,d_{qc})}{W_{DQ}}
\rightarrow
\underset{(1,d_{qc})}{c^Q},
\qquad
&\mathrm{FLOPs}=2dd_{qc}
\quad \text{shared across heads},
\\
\underset{(1,d_{qc})}{c^Q}
\underset{(h,d_{qc},d_k)}{W_{UQ}}
\rightarrow
\underset{(1,h,d_k)}{q},
\qquad
&\mathrm{FLOPs}=2hd_{qc}d_k.
\end{aligned}
$$

之后两个变体使用相同的 cached attention path。score side 的顺序是先把 $W_{UK}^\top$ 吸收到 current query 上，得到 $q_{\mathrm{abs}}=qW_{UK}^\top$，再和 latent cache 相乘；这样 $W_{UK}$ 只作用在长度为 $1$ 的 current token 上，而不是作用在长度为 $t$ 的整个 cache 上。value side 也是同样的原则：先算 $z=pC^{KV}_{\le t}$ 得到 latent weighted sum，再乘 $W_{UV}$，而不是先为所有 cached tokens reconstruct full $V$。

$$
\begin{aligned}
\underset{(1,h,d_k)}{q}
\underset{(h,d_k,d_c)}{W_{UK}^\top}
&\rightarrow
\underset{(1,h,d_c)}{q_{\mathrm{abs}}},
\qquad
\mathrm{FLOPs}=2hd_kd_c,
\\
\underset{(1,h,d_c)}{q_{\mathrm{abs}}}
\underset{(d_c,t)}{(C^{KV}_{\le t})^\top}
&\rightarrow
\underset{(1,h,t)}{score},
\qquad
\mathrm{FLOPs}=2htd_c,
\\
\underset{(1,h,t)}{p}
\underset{(t,d_c)}{C^{KV}_{\le t}}
\underset{(h,d_c,d_v)}{W_{UV}}
&\rightarrow
\underset{(1,h,d_v)}{o},
\qquad
\mathrm{FLOPs}=2htd_c + 2hd_cd_v.
\end{aligned}
$$

output projection 与 MHA 相同：

$$
\underset{(1,h,d_v)}{o}
\underset{(h,d_v,d)}{W_o}
\rightarrow
\underset{(1,d)}{y},
\qquad
\mathrm{FLOPs}=2hd_vd.
$$

所以 cached all-head decode attention FLOPs 是：$F_{\mathrm{MLA,decode,attn}}=2hd_kd_c+4htd_c+2hd_cd_v.$

对于整个 MLA decode layer：

$$
\begin{aligned}
F_{\mathrm{MLA,decode,cached,noQ}}
&=
2dd_c
+2hdd_k
+2hd_kd_c
+4htd_c
+2hd_cd_v
+2hd_vd,
\\
F_{\mathrm{MLA,decode,cached,Qcomp}}
&=
2dd_c
+2dd_{qc}
+2hd_{qc}d_k
+2hd_kd_c
+4htd_c
+2hd_cd_v
+2hd_vd .
\end{aligned}
$$

其中 $4htd_c$ 是 decode-length part：一个 $2htd_c$ 用来读取 latent cache 计算 scores，另一个 $2htd_c$ 用来再次读取 latent cache 计算 weighted latent value。这个顺序在 kernel 层面可实现：一个 fused MLA decode kernel 可以计算 $q_{\mathrm{abs}}$，stream $C^{KV}_{\le t}$ 的 cache blocks，做 online softmax，累积 $z^s=p^sC^{KV}_{\le t}$，最后再乘 $W_{UV}^s$。

### Decode Comparison

把 MHA 和 MLA 的 decode 公式放在一起看，MHA cached decode 是：

$$
F_{\mathrm{MHA,decode,cached}}
=
2dh(2d_k+d_v)
+2ht(d_k+d_v)
+2hd_vd.
$$

MLA without query compression 是：

$$
F_{\mathrm{MLA,decode,cached,noQ}}
=
2dd_c
+2hdd_k
+2hd_kd_c
+4htd_c
+2hd_cd_v
+2hd_vd.
$$

如果只看 FLOPs，MLA decode 不一定比 MHA 更小。比如 decode-length 相关项里，MHA 是 $2ht(d_k+d_v)$，而 MLA latent path 里是 $4htd_c$。在 DeepSeek-V2 的维度 $d_k=d_v=128,\ d_c=512$ 下，后者的算术量并不小。但 decode 阶段通常是一个 memory-bound 的问题，MHA 每个 cached token 要存和读 full per-head cache：$K_{\le t}:(t,h,d_k),\quad V_{\le t}:(t,h,d_v),$ 所以每个 token 的 cache 规模大约是 $h(d_k+d_v)$。MLA 存的是 shared latent：$C^{KV}_{\le t}:(t,d_c),$ 再加上较小的 decoupled RoPE cache。代入 $h=128,\ d_k=d_v=128,\ d_c=512$，MHA 的 full KV cache 每个 token 是 $32768$ 维，而 MLA 的 latent cache 是 $512$ 维。即使 MLA 在 kernel 内多做一些 $q_{\mathrm{abs}}$、latent score 和 latent value 的计算，它避免了反复从显存读取 full per-head $K,V$。

所以 MLA decode 的加速主要来自 **cache 的变化**：把长上下文 decode 中最重的 full KV 读写，换成了 shared latent cache 的读写。FLOPs 被转移到一些更小、更适合 fused kernel 的计算上；MLA 在这部分的本质还是用计算来换显存。


## Why RoPE Gets in the Way

MLA 的 absorption trick 成立，有一个隐含前提：从 latent cache 到 key 的映射在不同 position 上是固定的。不考虑 RoPE 时，对于 cached token $i$：

$$
\begin{aligned}
k_i^s
&=
c_i^{KV}W_{UK}^s,
\\
score_i^s
&=
q^s(k_i^s)^\top
=
(q^s(W_{UK}^s)^\top)(c_i^{KV})^\top .
\end{aligned}
$$

于是可以先计算一个 current-token vector $q_{\mathrm{abs}}^s=q^s(W_{UK}^s)^\top$，再对所有 cached positions 复用它。这就是 latent-cache decode 背后最干净的代数结构。

问题在于，RoPE 会在 key path 中插入 position-dependent rotation。令 $R_t$ 表示当前 position 的 query rotation，$R_i$ 表示 cached position $i$ 的 key rotation：

$$
\begin{aligned}
score_i^s
&=
(R_tq^s)(R_ik_i^s)^\top
\\
&=
q^sR_t^\top R_i(W_{UK}^s)^\top(c_i^{KV})^\top .
\end{aligned}
$$

现在矩阵 $R_t^\top R_i(W_{UK}^s)^\top$ 依赖于 $i$。这意味着我们不能只形成一个 $q_{\mathrm{abs}}^s$，然后对所有 cached tokens 复用它。重构并缓存所有 rotated per-head keys 当然可行，但这样基本又回到了普通 KV cache 的方向。

实际做法是 decouple content channel 和 positional channel：

$$
score_i^s
=
q_{\mathrm{content,abs}}^s(c_i^{KV})^\top
+
(R_tq_{\mathrm{rope}}^s)(R_ik_{i,\mathrm{rope}}^s)^\top .
$$

content channel 仍然走 MLA latent-cache path；RoPE channel 则单独存，并且维度保持较小。所以 RoPE 并不会让 MLA 不可能实现，它只是阻止了“整个 key 都隐藏在一个 shared latent cache 后面”的最干净版本。

## Why Compress Q Too

KV compression 和 Q compression 解决的是系统里的不同问题。KV compression 改变 decode 时存什么、读什么；Q compression 则 factorize query projection：

$$
C^Q=XW_{DQ},\qquad Q^s=C^QW_{UQ}^s.
$$

- **Activation memory.** 没有 query compression 时，query activation 规模是 $T h d_k$；有 query compression 时，compressed query state 规模是 $T d_{qc}$。关键条件很简单：

  $$
  d_{qc}<hd_k .
  $$

  对于 DeepSeek-V2，$h=128$，$d_k=128$，$d_{qc}=1536$，所以 $hd_k=16384$。这会显著减少需要保存的 query activations，在 long-context prefill/training 中尤其重要。

- **Prefill/training FLOPs.** Q compression 也可能带来 compute win。query projection 从：

  $$
  F_Q=2Tdh d_k
  \qquad\Longrightarrow\qquad
  F_{Q,\mathrm{comp}}=2Tdd_{qc}+2Thd_{qc}d_k .
  $$

  当 $d_{qc}(d+hd_k)<hdd_k$ 时，Q compression 会降低 query-projection FLOPs。在 DeepSeek-V2 维度下，$F_Q=167.8\text{M}T$，而 $F_{Q,\mathrm{comp}}=66.1\text{M}T$。这也解释了为什么前面的 prefill comparison 中，MLA no-Q 和 MLA Q-compression 的差距很明显。

- **Decode.** Q compression 只影响 current-token query path；它本身不会降低 latent cache-length term。score path 是：

  $$
  \underset{(1,d)}{x}
  \underset{(d,d_{qc})}{W_{DQ}}
  \underset{(h,d_{qc},d_k)}{W_{UQ}}
  \underset{(h,d_k,d_c)}{W_{UK}^\top}
  \underset{(d_c,t)}{(C_{\le t}^{KV})^\top}.
  $$

  实际实现中通常仍然经过较小的 per-head bottleneck $d_k$。score-side cost 是：

  $$
  2d_{qc}d_k+2d_kd_c+2td_c
  \quad
  \text{instead of}
  \quad
  2d_{qc}d_c+2td_c .
  $$

  因为 $d_k$ 远小于 $d_c$ 和 $d_{qc}$，所以 two-step path 通常更便宜。

**Q compression 会节省 activation memory，也可能节省 prefill projection FLOPs，但对 decode compute 的改变相对有限；KV compression 仍然是主要的 decode-cache bandwidth win。**

## Why MTP Weakens MLA's Decode Acceleration Gain

![MTP](MTP.png)

[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) 中的 MTP 是额外 training objective。对于 prediction depth $k$，MTP module 会把上一 depth 的 representation 和 shifted token embedding 拼接起来：

$$
h_i^{\prime k}
=
M_k[
\operatorname{RMSNorm}(h_i^{k-1});
\operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))
],
\qquad
h_{1:T-k}^{k}
=
\operatorname{TRM}_k(h_{1:T-k}^{\prime k}).
$$

然后 shared output head 预测第 $k$ 个额外 future token：

$$
P_{i+k+1}^{k}=\operatorname{OutHead}(h_i^k).
$$

正常 inference 时，DeepSeek-V3 可以直接丢弃这些 MTP modules，此时 MLA 的 decode path 不受 MTP 影响。真正需要讨论的是把 MTP modules 拿来做 speculative decoding：MTP path 先 draft 多个 future tokens，再由 main model verify。

先看 plain MLA 的收益。前面推导的 MLA decode 加速，主要来自 main decode path 的 cache format 改变：普通 MHA 每个 token 大约读取 $h(d_k+d_v)$ 维 cached $K,V$，而 MLA 主要读取 $d_c$ 维 latent cache，再加上较小的 decoupled RoPE 部分。以 DeepSeek-V2/V3 风格的参数为例，$h(d_k+d_v)=128(128+128)=32768$，而 $d_c=512$。也就是说，MLA 的优势可以粗略理解为：

$$
G_{\mathrm{MLA}}
=
F_{\mathrm{MHA,decode}}
-
F_{\mathrm{MLA,decode}} .
$$

引入 MTP 做 speculative decoding 后，实际每生成一组 accepted tokens 的成本不再只是 $F_{\mathrm{MLA,decode}}$，而是 draft 加 verify 的成本。假设一次 draft $D$ 个 tokens，最终接受 $A$ 个 tokens，那么平均到每个 accepted token 上，大致是：

$$
\bar F_{\mathrm{MTP}}
\approx
\frac{
F_{\mathrm{draft}}(1{:}D)+F_{\mathrm{verify}}(D)
}{A}.
$$

如果要让 MTP 不吃掉 MLA 原本的加速收益，就需要这个平均成本不要比 plain MLA decode 高太多；更强一点，如果希望 MTP 在 MLA 上继续带来 decode 加速，则需要：

$$
F_{\mathrm{draft}}(1{:}D)+F_{\mathrm{verify}}(D)
<
A\cdot F_{\mathrm{MLA,decode}}.
$$

问题在于，MTP draft path 并不是免费的。第 $k$ 个 draft depth 至少包含 fusion projection、一个额外的 $\operatorname{TRM}_k$ block，以及 output head：

$$
F_{\mathrm{draft}}(k)
\approx
4d^2
+F_{\operatorname{TRM}_k,\mathrm{decode}}^{\mathrm{MLA}}
+2dV.
$$

其中 $4d^2$ 来自把 $[h^{k-1};\operatorname{Emb}]$ 从 $2d$ 投影回 $d$，$F_{\operatorname{TRM}_k,\mathrm{decode}}^{\mathrm{MLA}}$ 是 MTP Transformer block 的额外 decode cost，$2dV$ 是计算完整 logits 时 shared output head 的 cost。

关键点很简单：$\operatorname{TRM}_k$ 跑在 shifted hidden stream $h^k$ 上，和 main model 的 hidden states、token alignment 都不同，所以不能直接复用 main $C^{KV}$ cache。如果 MTP block 内部也使用 MLA，就要为这条 stream 维护一份自己的 latent cache。

因此，**MTP 会削弱 MLA 在 decode 阶段的加速收益**。MLA 把 main decode path 的 cache 读写压低了，但 MTP 又多出一条 draft stream、verify cost，以及一份不能直接复用 main cache 的 shifted latent states。从 FLOPs/cache 角度看，这些额外成本会吃掉一部分 MLA 的收益。

