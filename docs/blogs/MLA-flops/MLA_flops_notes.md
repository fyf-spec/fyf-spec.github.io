---
title: "MLA FLOPs, RoPE, Query Compression, and MTP Notes"
description: "A technical note on MLA FLOPs, matrix multiplication order, RoPE decoupling, query compression, and why MTP cannot freely reuse the main MLA cache."
date: 2026-07-06
outline: deep
---

# A Detailed FLOPs Perspective of MLA

<p class="language-switch"><strong>Language:</strong> English | <a href="./MLA_flops_notes_zh.html">Chinese</a></p>

This note is about a key question in DeepSeek-V2 MLA: what exactly does MLA save, where do the FLOPs move, and how do RoPE and the other components change the computation? I first compare MHA and MLA in prefill/training/decode, plug in the paper's actual dimensions to make the FLOPs concrete, and then use the same notation to discuss RoPE, query compression, and MTP. The point is that many things that look like "architecture design" can be read directly from matrix multiplication order and cache format.

Basically, the blog can be divided into four parts:

- FLOPs comparison between MHA and MLA in prefill/training/decode.
- Why RoPE gets in the way of MLA absorption.
- Why MLA also compresses $Q$.
- Why MTP weakens MLA's decode acceleration gain.

![MLA sketch](MLA_baseline.png)

## Notation

For the derivations below, I count one multiply-add as $2$ FLOPs, and I only count matrix multiplication FLOPs. I temporarily ignore softmax, masks, normalization, RoPE elementwise rotation, and bias terms. The main dimensions used in the MLA derivation are:

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

To avoid hiding the shared latent behind a head index, the more useful tensor view is:

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

Therefore $C^{KV}:(T,d_c)$ and $C^Q:(T,d_{qc})$ are shared latents across all heads, while $Q,K,V$ live in head space, for example $Q:(T,h,d_k)$ and $V:(T,h,d_v)$. Some equations below still keep a single-head slice for readability, but the actual storage/computation convention should be understood by the tensor shapes here.

## Baseline FLOPs

I start from baseline multi-head attention (MHA), derive each matrix multiplication and its FLOPs, and then compare it with MLA to see where the optimization actually comes from.

### MHA Prefill

First compute the $Q,K,V$ projections from input $X$: $Q=XW_Q,\quad K=XW_K,\quad V=XW_V.$

For MHA prefill, in tensor form, the full score expression $S$, attention output $O$, and output projection $Y$ are as follows. Here $P$ is the attention matrix after softmax:

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

For the attention score $S$, the FLOPs perspective is simple: make the coefficient of the dominating $T^2$ term as small as possible. Since $d_k<d$, the reasonable computation order is:

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

Then compute the FLOPs of $O$ and $Y$. Here it is useful to view the value/output path as one head-wise four-matrix product:

$$
\underset{(h,T,T)}{P}
\underset{(T,d)}{X}
\underset{(h,d,d_v)}{W_V}
\underset{(h,d_v,d)}{W_o}
\rightarrow
\underset{(T,d)}{Y}.
$$

The best order is to project $X$ down to $d_v$ first, and then multiply by $P$. Across all heads, the standard order is:

$$
XW_V \rightarrow V,\qquad PV \rightarrow O,\qquad OW_o \rightarrow Y,
\quad
F=2Thdd_v+2hT^2d_v+2Thd_vd.
$$

If we compute $PX$ first, the cost becomes $2hT^2d+2Thdd_v+2Thd_vd$. If we precompose $W_VW_o$ first, the quadratic attention multiplication also has coefficient $d$ instead of $d_v$. Since usually $d_v\ll d$, the standard order is better: the quadratic term is $2hT^2d_v$ instead of $2hT^2d$.

> For causal prefill/training, an implementation only needs to compute the lower triangle. The number of visible query-key pairs is:
> $\frac{T(T+1)}{2}.$
> So the causal-aware attention FLOPs are:
> $F_{\mathrm{attn,causal}}=hT(T+1)(d_k+d_v).$
> Therefore, the causal-aware prefill FLOPs are:
> $$
> F_{\mathrm{MHA,prefill}}=2Tdh(2d_k+d_v)
> +2hT^2(d_k+d_v)
> +2Thd_vd.
> $$
> If the implementation explicitly uses a causal mask, replace the middle term by $hT(T+1)(d_k+d_v)$.

### MLA Prefill

For MLA prefill, first compute the shared KV latent: $C^{KV}=XW_{DKV}$. Here $C^{KV}$ is a latent representation shared by all heads.

**Without Query Compression.**

For MLA without query compression, the full score expression $S$ and attention output $O$ are:

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

From a FLOPs perspective, we still watch the coefficient of the $T^2$ term. Usually $d_k<d_c$, so in prefill we should not first compute $Q^s(W_{UK}^s)^\top$ and then multiply by $(C^{KV})^\top$; that would make the dominating $QK^\top$ term become $2T^2d_c$.

There is another computation-order issue on the key side. In tensor shape, the key side in the score is:

$$
\underset{(T,d)}{X}
\underset{(d,d_c)}{W_{DKV}}
\underset{(h,d_c,d_k)}{W_{UK}}
\rightarrow
\underset{(T,h,d_k)}{K}.
$$

The shared-latent order is:

$$
XW_{DKV}\rightarrow C^{KV},\qquad C^{KV}W_{UK}\rightarrow K,
\quad
F=2Tdd_c+2Thd_cd_k.
$$

If we first form the effective per-head key projection $W_{DKV}W_{UK}$ and then multiply it by $X$, the activation-side cost is:

$$
X(W_{DKV}W_{UK})\rightarrow K,
\quad
F=2Thdd_k,
$$

not counting the extra one-time weight precomposition cost $2hdd_cd_k$. Plugging in DeepSeek-V2 values $d=5120,\ d_c=512,\ d_k=128,\ h=128$, the shared-latent order is about $22.0\text{M}T$ FLOPs, while the effective-$W_K$ order is $167.8\text{M}T$ FLOPs. In other words, the cheap path is to compute $XW_{DKV}$ first, use the shared $d_c$ bottleneck first, and only then expand to all heads.

Therefore the computation order is:

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

Then compute the FLOPs of $O$, where $O=PV$ and $V=C^{KV}W_{UV}$:

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

This order is implementable at the kernel level. The most direct implementation materializes $K,V$ first and then calls a standard FlashAttention kernel; a more memory-saving implementation computes $K,V$ from $C^{KV}$ inside each attention tile.

Without query compression, the causal-aware MLA prefill FLOPs are:

$$
F_{\mathrm{MLA,prefill,noQ}}
=
2Tdd_c
+2Thdd_k
+2Thd_c(d_k+d_v)
+2hT^2(d_k+d_v)
+2Thd_vd.
$$

> If the implementation explicitly uses a causal mask, replace the middle attention term by $hT(T+1)(d_k+d_v)$.

**With Query Compression.**

With query compression, the full score expression becomes:

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

The attention output $O$ is unchanged:

$$
O=PV
=
\underset{(h,T,T)}{P}
\underset{(T,d)}{X}
\underset{(d,d_c)}{W_{DKV}}
\underset{(h,d_c,d_v)}{W_{UV}}.
$$

The optimal computation order is:

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

The $C^{KV},K,V,S,O$ path is the same as the no-Q-compression case. With query compression, the non-causal-aware MLA prefill total FLOPs are:

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

> If the implementation explicitly uses a causal mask, replace the middle attention term by $hT(T+1)(d_k+d_v)$.

### Prefill Comparison

From the formulas above, if both explicitly compute the same-shaped $QK^\top$ and $PV$, then MHA and MLA have the same quadratic attention FLOPs. In prefill, the real difference mainly comes from the linear projection part.

The MHA linear projection FLOPs are $2Tdh(2d_k+d_v)$. MLA without query compression costs $2Tdd_c+2Thdd_k+2Thd_c(d_k+d_v)$. MLA with query compression costs $2Tdd_c+2Tdd_{qc}+2Thd_{qc}d_k+2Thd_c(d_k+d_v)$.

Plugging in the DeepSeek-V2 attention configuration $d=5120,\ h=128,\ d_k=d_v=128,\ d_c=512,\ d_{qc}=1536$, the MHA linear projection cost is $503.3\text{M}T$, MLA without query compression is $206.6\text{M}T$, and MLA with query compression is $104.9\text{M}T$. That is, MLA no-Q is about $41.1\%$ of MHA, while MLA Q-compression is about $20.8\%$ of MHA.

**MLA prefill does not reduce quadratic attention FLOPs, but it significantly reduces linear projection FLOPs; this is especially clear with query compression.** Under the DeepSeek-V2 configuration, MLA with query compression has only about $20.8\%$ of the MHA projection FLOPs.

### MHA Decode

With KV cache, historical $K_{\le t},V_{\le t}$ have already been computed in previous decode steps. The current step only needs to compute and append the new $q,k,v$:

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

Across all heads, the current-token projection cost is: $F_{qkv}=2dh(2d_k+d_v).$

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

The computation order here is determined by KV cache: first generate $q,k,v$ from the current token $x$, append $k,v$ to the cache, and then use only the current $q$ to read historical $K_{\le t}$ and compute scores. In this way, decode-length multiplications happen in dimensions $d_k,d_v$, rather than reprocessing the whole $X_{\le t}$ from hidden dimension $d$.

Then attention reads cached $K,V$:

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

So the cached decode FLOPs are: $F_{\mathrm{MHA,decode,cached}}=
2dh(2d_k+d_v)
+2ht(d_k+d_v)
+2hd_vd .$

### MLA Decode

For MLA decode, the cache stores the shared latent cache instead of full per-head $K,V$. At the current step, we append one new latent vector:

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

Therefore the current-token linear projection cost is explicitly included in decode. Without query compression:

$$
F_{\mathrm{MLA,current,noQ}}=2dd_c+2hdd_k.
$$

With query compression:

$$
F_{\mathrm{MLA,current,Qcomp}}=2dd_c+2dd_{qc}+2hd_{qc}d_k.
$$

MLA decode does not need to append full $k,v$ separately; what is actually written into cache is the shared $c^{KV}$ latent, while $K,V$ are reconstructed or absorbed only when attention needs them.

The full score and value expressions are below. I keep the two variants, MLA without query compression and MLA with query compression. They differ mainly in the current-token query path, while the latent-cache attention path is shared.

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

For input query $x$, cached execution first constructs the current query. Without query compression:

$$
\underset{(1,d)}{x}
\underset{(h,d,d_k)}{W_Q}
\rightarrow
\underset{(1,h,d_k)}{q},
\qquad
\mathrm{FLOPs}=2hdd_k.
$$

With query compression, the computation of $q$ is factorized as:

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

After that, both variants use the same cached attention path. On the score side, the order is to first absorb $W_{UK}^\top$ into the current query to get $q_{\mathrm{abs}}=qW_{UK}^\top$, and then multiply it by the latent cache. This makes $W_{UK}$ act only on the length-$1$ current token, instead of on the length-$t$ whole cache. The value side follows the same principle: first compute $z=pC^{KV}_{\le t}$ to get a latent weighted sum, and then multiply by $W_{UV}$, rather than reconstructing full $V$ for all cached tokens first.

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

The output projection is the same as MHA:

$$
\underset{(1,h,d_v)}{o}
\underset{(h,d_v,d)}{W_o}
\rightarrow
\underset{(1,d)}{y},
\qquad
\mathrm{FLOPs}=2hd_vd.
$$

So the cached all-head decode attention FLOPs are: $F_{\mathrm{MLA,decode,attn}}=2hd_kd_c+4htd_c+2hd_cd_v.$

For the whole MLA decode layer:

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

Here $4htd_c$ is the decode-length part: one $2htd_c$ reads the latent cache to compute scores, and the other $2htd_c$ reads the latent cache again to compute the weighted latent value. This order is implementable at the kernel level: a fused MLA decode kernel can compute $q_{\mathrm{abs}}$, stream the cache blocks of $C^{KV}_{\le t}$, run online softmax, accumulate $z^s=p^sC^{KV}_{\le t}$, and finally multiply by $W_{UV}^s$.

### Decode Comparison

Putting the MHA and MLA decode formulas side by side, MHA cached decode is:

$$
F_{\mathrm{MHA,decode,cached}}
=
2dh(2d_k+d_v)
+2ht(d_k+d_v)
+2hd_vd.
$$

MLA without query compression is:

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

If we only look at FLOPs, MLA decode is not necessarily smaller than MHA. For the decode-length terms, MHA has $2ht(d_k+d_v)$, while the MLA latent path has $4htd_c$. Under DeepSeek-V2 dimensions $d_k=d_v=128,\ d_c=512$, the arithmetic work of the latter is not small. But decode is usually memory-bound. MHA needs to store and read full per-head cache for each cached token: $K_{\le t}:(t,h,d_k),\quad V_{\le t}:(t,h,d_v),$ so the cache size per token is roughly $h(d_k+d_v)$. MLA stores the shared latent: $C^{KV}_{\le t}:(t,d_c),$ plus a smaller decoupled RoPE cache. Plugging in $h=128,\ d_k=d_v=128,\ d_c=512$, the full MHA KV cache is $32768$ dimensions per token, while the MLA latent cache is $512$ dimensions. Even if MLA does extra work inside the kernel for $q_{\mathrm{abs}}$, latent scores, and latent values, it avoids repeatedly reading full per-head $K,V$ from GPU memory.

So MLA decode acceleration mainly comes from **the cache change**: it replaces the heaviest full-KV reads and writes in long-context decode with shared-latent cache reads and writes. FLOPs are shifted into smaller computations that fit fused kernels better; in this part, MLA is essentially trading computation for memory.

## Why RoPE Gets in the Way

MLA's absorption trick has an implicit prerequisite: the map from latent cache to key is fixed across positions. Without RoPE, for cached token $i$:

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

Then we can first compute a current-token vector $q_{\mathrm{abs}}^s=q^s(W_{UK}^s)^\top$, and reuse it for all cached positions. This is the clean algebraic structure behind latent-cache decode.

The problem is that RoPE inserts a position-dependent rotation into the key path. Let $R_t$ denote the query rotation at the current position, and $R_i$ denote the key rotation at cached position $i$:

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

Now the matrix $R_t^\top R_i(W_{UK}^s)^\top$ depends on $i$. This means we cannot form one $q_{\mathrm{abs}}^s$ and reuse it for all cached tokens. Reconstructing and caching all rotated per-head keys is of course possible, but that basically moves us back toward a normal KV cache.

The practical solution is to decouple the content channel and the positional channel:

$$
score_i^s
=
q_{\mathrm{content,abs}}^s(c_i^{KV})^\top
+
(R_tq_{\mathrm{rope}}^s)(R_ik_{i,\mathrm{rope}}^s)^\top .
$$

The content channel still uses the MLA latent-cache path; the RoPE channel is stored separately and kept small. Therefore RoPE does not make MLA impossible. It only blocks the cleanest version where the whole key is hidden behind one shared latent cache.

## Why Compress Q Too

KV compression and Q compression solve different system problems. KV compression changes what we store and read during decode; Q compression factorizes query projection:

$$
C^Q=XW_{DQ},\qquad Q^s=C^QW_{UQ}^s.
$$

- **Activation memory.** Without query compression, the query activation scale is $T h d_k$; with query compression, the compressed query state scale is $T d_{qc}$. The key condition is simple:

  $$
  d_{qc}<hd_k .
  $$

  For DeepSeek-V2, $h=128$, $d_k=128$, and $d_{qc}=1536$, so $hd_k=16384$. This significantly reduces the query activations that need to be saved, which is especially important in long-context prefill/training.

- **Prefill/training FLOPs.** Q compression can also bring a compute win. The query projection changes from:

  $$
  F_Q=2Tdh d_k
  \qquad\Longrightarrow\qquad
  F_{Q,\mathrm{comp}}=2Tdd_{qc}+2Thd_{qc}d_k .
  $$

  When $d_{qc}(d+hd_k)<hdd_k$, Q compression reduces query-projection FLOPs. Under DeepSeek-V2 dimensions, $F_Q=167.8\text{M}T$, while $F_{Q,\mathrm{comp}}=66.1\text{M}T$. This also explains why the previous prefill comparison shows a clear gap between MLA no-Q and MLA Q-compression.

- **Decode.** Q compression only affects the current-token query path; it does not reduce the latent cache-length term by itself. The score path is:

  $$
  \underset{(1,d)}{x}
  \underset{(d,d_{qc})}{W_{DQ}}
  \underset{(h,d_{qc},d_k)}{W_{UQ}}
  \underset{(h,d_k,d_c)}{W_{UK}^\top}
  \underset{(d_c,t)}{(C_{\le t}^{KV})^\top}.
  $$

  In actual implementations, it usually still goes through the smaller per-head bottleneck $d_k$. The score-side cost is:

  $$
  2d_{qc}d_k+2d_kd_c+2td_c
  \quad
  \text{instead of}
  \quad
  2d_{qc}d_c+2td_c .
  $$

  Since $d_k$ is much smaller than $d_c$ and $d_{qc}$, the two-step path is usually cheaper.

**Q compression saves activation memory and may also save prefill projection FLOPs, but it changes decode compute only modestly; KV compression is still the main decode-cache bandwidth win.**

## Why MTP Weakens MLA's Decode Acceleration Gain

![MTP](MTP.png)

The [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) uses MTP as an additional training objective. For prediction depth $k$, the MTP module concatenates the previous-depth representation and a shifted token embedding:

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

Then the shared output head predicts the $k$-th additional future token:

$$
P_{i+k+1}^{k}=\operatorname{OutHead}(h_i^k).
$$

During normal inference, DeepSeek-V3 can directly discard these MTP modules, so the MLA decode path is not affected by MTP. The real issue appears when we repurpose MTP modules for speculative decoding: the MTP path first drafts several future tokens, and then the main model verifies them.

First consider the gain from plain MLA. The decode acceleration derived above mainly comes from changing the cache format of the main decode path: normal MHA reads roughly $h(d_k+d_v)$ cached $K,V$ values per token, while MLA mainly reads a $d_c$-dimensional latent cache, plus a smaller decoupled RoPE part. With DeepSeek-V2/V3-style dimensions, $h(d_k+d_v)=128(128+128)=32768$, while $d_c=512$. Roughly speaking, the MLA gain is:

$$
G_{\mathrm{MLA}}
=
F_{\mathrm{MHA,decode}}
-
F_{\mathrm{MLA,decode}} .
$$

After introducing MTP for speculative decoding, the cost of generating a group of accepted tokens is no longer just $F_{\mathrm{MLA,decode}}$. It becomes the draft cost plus the verification cost. If we draft $D$ tokens and finally accept $A$ tokens, the average cost per accepted token is roughly:

$$
\bar F_{\mathrm{MTP}}
\approx
\frac{
F_{\mathrm{draft}}(1{:}D)+F_{\mathrm{verify}}(D)
}{A}.
$$

For MTP not to eat away the original MLA gain, this average cost should not be much larger than plain MLA decode. A stronger condition, if we want MTP to further accelerate decode on top of MLA, is:

$$
F_{\mathrm{draft}}(1{:}D)+F_{\mathrm{verify}}(D)
<
A\cdot F_{\mathrm{MLA,decode}}.
$$

The problem is that the MTP draft path is not free. The $k$-th draft depth at least contains a fusion projection, an extra $\operatorname{TRM}_k$ block, and the output head:

$$
F_{\mathrm{draft}}(k)
\approx
4d^2
+F_{\operatorname{TRM}_k,\mathrm{decode}}^{\mathrm{MLA}}
+2dV.
$$

Here $4d^2$ comes from projecting $[h^{k-1};\operatorname{Emb}]$ from $2d$ back to $d$, $F_{\operatorname{TRM}_k,\mathrm{decode}}^{\mathrm{MLA}}$ is the extra decode cost of the MTP Transformer block, and $2dV$ is the shared output-head cost for full logits.

The key point is simple: $\operatorname{TRM}_k$ runs on a shifted hidden stream $h^k$, whose hidden states and token alignment differ from the main model. So it cannot directly reuse the main $C^{KV}$ cache. If the MTP block also uses MLA internally, it needs its own latent cache for this stream.

Therefore, **MTP weakens MLA's decode-stage acceleration gain**. MLA reduces cache traffic on the main decode path, but MTP adds a draft stream, verification cost, and shifted latent states that cannot directly reuse the main cache. From the FLOPs/cache perspective, these extra costs spend part of MLA's gain.
