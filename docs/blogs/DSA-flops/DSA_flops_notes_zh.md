---
title: "DeepSeek DSA FLOPs"
description: "沿用 MLA FLOPs 的计数口径，推导 DeepSeek Sparse Attention 的核心稀疏注意力与 Lightning Indexer FLOPs。"
date: 2026-07-28
outline: deep
---

# DeepSeek DSA FLOPs

一句话概括：DSA 先用一个低维但仍为二次复杂度的 Lightning Indexer 找到 Top-$k$ 个 token，再让 MLA 的核心注意力只处理这些 token，把主注意力从 $O(T^2)$ 降为 $O(Tk)$。

本文只推导两个部分：**核心稀疏注意力**和 **Lightning Indexer**。公式依据 [DeepSeek-V3.2 Technical Report](https://arxiv.org/abs/2512.02556v1)；具体张量维度和计算路径以 DeepSeek 官方的 [V3.2-Exp 配置](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/config_671B_v3.2.json) 与 [Indexer 实现](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py#L435-L486) 为准。

## Notation

沿用 MLA FLOPs 笔记的口径：一次 multiply-add 记作 $2$ FLOPs，并且只统计矩阵乘。暂时忽略 softmax、mask、normalization、RoPE、ReLU、逐元素乘法、Top-$k$ selection 和 cache 读写。

这里还需要明确计算边界：

- 核心注意力从已经形成的 absorbed content query $Q^A$、RoPE query $Q^R$ 和 MLA cache 开始计算，不重复统计 MLA 的 query/KV projection。
- Indexer 从 MLA 已有的 query latent $C^Q$ 和输入 hidden states $X$ 开始计算。官方实现直接把 MLA 的 query latent 传给 Indexer，因此不再重复统计 $XW_{DQ}$。
- 不统计核心注意力之后的 latent value up-projection 和 output projection。

主要记号如下：

| Symbol | Meaning | DeepSeek-V3.2 |
| --- | --- | ---: |
| $T$ | prefill / training sequence length | 最长 $128\mathrm{K}$ |
| $t$ | decode 时的 KV-cache length | variable |
| $k$ | 每个 query 选出的 token 数 | $2048$ |
| $d$ | hidden size | $7168$ |
| $h$ | 核心注意力的 query heads | $128$ |
| $d_c$ | MLA KV latent / absorbed content dimension | $512$ |
| $d_r$ | decoupled RoPE dimension | $64$ |
| $d_{qc}$ | MLA query latent dimension | $1536$ |
| $H^I$ | Indexer heads | $64$ |
| $d^I$ | 每个 Indexer head 的维度 | $128$ |

核心注意力涉及的 tensor 可以写成：

$$ Q^A:(T,h,d_c),\qquad Q^R:(T,h,d_r),\qquad C^{KV}:(T,d_c),\qquad K^R:(T,d_r). $$

Indexer 涉及的 tensor 是：

$$ C^Q:(T,d_{qc}),\qquad Q^I:(T,H^I,d^I),\qquad K^I:(T,d^I),\qquad W^I:(T,H^I). $$

最容易误解的一点是：$K^I$ 在所有 $H^I$ 个 Indexer heads 之间共享。因此 key projection 的 FLOPs 不乘 $H^I$；但每个 head 都要和这份 shared key 做点积，所以 pairwise scoring 的 FLOPs 仍然乘 $H^I$。

### Causal token-pair 数

Dense causal attention 实际可见的 query-key pair 数是：

$$ N_{\mathrm{dense}}(T)=\frac{T(T+1)}{2}. $$

DSA 中第 $s$ 个 query 最多保留 $\min(k,s)$ 个 key，因此核心稀疏注意力实际处理的 pair 数是：

$$ N_{\mathrm{sel}}(T,k)=\sum_{s=1}^{T}\min(k,s)=\begin{cases}\frac{T(T+1)}{2},&T\le k,\\kT-\frac{k(k-1)}{2},&T>k.\end{cases} $$

第二项 $k(k-1)/2$ 是开头不足 $k$ 个历史 token 的 causal 修正。长序列下才可以近似写成 $N_{\mathrm{sel}}\approx Tk$。

## Core Sparse Attention FLOPs

论文明确说明 DSA 的核心注意力采用 MLA 的 MQA mode：同一位置的 $C^{KV}$ 和 $K^R$ 被所有 query heads 共享。对 query $s$，令 $\mathcal{S}_s$ 是 Indexer 选出的 token 集合，$m_s=|\mathcal{S}_s|=\min(k,s)$。

### Attention score

content score 在 absorbed latent space 中计算：

$$ \underset{(h,d_c)}{Q_s^A}\underset{(d_c,m_s)}{(C_{\mathcal{S}_s}^{KV})^\top}\rightarrow\underset{(h,m_s)}{S_s^A},\qquad \mathrm{FLOPs}=2hm_sd_c. $$

RoPE score 单独走较小的 positional channel：

$$ \underset{(h,d_r)}{Q_s^R}\underset{(d_r,m_s)}{(K_{\mathcal{S}_s}^{R})^\top}\rightarrow\underset{(h,m_s)}{S_s^R},\qquad \mathrm{FLOPs}=2hm_sd_r. $$

两部分相加后再做 softmax。相加和 softmax 不属于本文的矩阵乘 FLOPs。

### Weighted latent value

softmax 得到每个 query head 各自的权重 $P_s:(h,m_s)$。所有 heads 读取同一份 latent value cache，但因为各 head 的权重不同，weighted sum 仍然要按 head 计算：

$$ \underset{(h,m_s)}{P_s}\underset{(m_s,d_c)}{C_{\mathcal{S}_s}^{KV}}\rightarrow\underset{(h,d_c)}{Z_s},\qquad \mathrm{FLOPs}=2hm_sd_c. $$

因此，单个 query 的 core attention FLOPs 是：

$$ F_{\mathrm{core},s}=2hm_s(2d_c+d_r). $$

其中两个 $d_c$ 分别来自 content score 和 weighted latent value；$d_r$ 来自 RoPE score。

### Prefill / training

对所有 causal query 求和：

$$ F_{\mathrm{DSA,core,prefill}}=2h(2d_c+d_r)N_{\mathrm{sel}}(T,k). $$

当 $T>k$ 时：

$$ F_{\mathrm{DSA,core,prefill}}=2h(2d_c+d_r)\left(kT-\frac{k(k-1)}{2}\right). $$

如果把 $N_{\mathrm{sel}}$ 换成 $N_{\mathrm{dense}}$，就得到相同 MQA/absorbed 路径下的 dense MLA core attention FLOPs。也就是说，DSA 并没有改变每个保留 pair 的计算，而是把 pair 数从约 $T^2/2$ 降到了约 $Tk$。

### Decode

decode 时只有一个当前 query。令 $m=\min(k,t)$，则：

$$ F_{\mathrm{DSA,core,decode}}=2hm(2d_c+d_r). $$

当 cache 已经长于 $k$ 时，core attention 对上下文长度不再线性增长，而是固定为：

$$ F_{\mathrm{DSA,core,decode}}=2hk(2d_c+d_r),\qquad t\ge k. $$

代入 $h=128$、$d_c=512$、$d_r=64$、$k=2048$：

$$ 2h(2d_c+d_r)=278{,}528\ \mathrm{FLOPs/pair}. $$

- $128\mathrm{K}=131072$ prefill 时，$N_{\mathrm{sel}}=266{,}339{,}328$，core attention 约为 $74.18$ TFLOPs / layer。
- 当 $t\ge 2048$ 时，每个 decode token 的 core attention 是 $570.43$ MFLOPs / layer。

## Lightning Indexer FLOPs

论文给出的 index score 是：

$$ I_{s,r}=\sum_{j=1}^{H^I}w_{s,j}^I\operatorname{ReLU}\left(q_{s,j}^I\cdot k_r^I\right). $$

$q_{s,j}^I\in\mathbb{R}^{d^I}$ 是第 $j$ 个 Indexer head 的 query，$k_r^I\in\mathbb{R}^{d^I}$ 是位置 $r$ 的 shared key，$w_{s,j}^I$ 是当前 query 对各 Indexer heads 的组合权重。作者指出，选择 ReLU 是出于吞吐考虑；较少的 heads 和 FP8 实现使这个二次复杂度模块远轻于原始 dense attention。

### Indexer projections

官方代码复用 MLA 的 query latent $C^Q$ 来产生 Indexer queries：

$$ \underset{(T,d_{qc})}{C^Q}\underset{(d_{qc},H^Id^I)}{W_{IQ}}\rightarrow\underset{(T,H^I,d^I)}{Q^I},\qquad \mathrm{FLOPs}=2Td_{qc}H^Id^I. $$

shared Indexer key 直接从 hidden states 投影：

$$ \underset{(T,d)}{X}\underset{(d,d^I)}{W_{IK}}\rightarrow\underset{(T,d^I)}{K^I},\qquad \mathrm{FLOPs}=2Tdd^I. $$

每个 query 还要产生 $H^I$ 个 head weights：

$$ \underset{(T,d)}{X}\underset{(d,H^I)}{W_{IW}}\rightarrow\underset{(T,H^I)}{W^I},\qquad \mathrm{FLOPs}=2TdH^I. $$

所以 Indexer 的线性 projection part 是：

$$ F_{\mathrm{index,proj}}=2T\left(d_{qc}H^Id^I+dd^I+dH^I\right). $$

### Full-prefix index scoring

每个 Indexer head 都要让所有 queries 与 shared keys 做点积：

$$ \underset{(T,d^I)}{Q_j^I}\underset{(d^I,T)}{(K^I)^\top}\rightarrow\underset{(T,T)}{A_j^I},\qquad \mathrm{FLOPs}=2T^2d^I\quad\text{per head}. $$

如果只计算 causal lower triangle，所有 $H^I$ 个 heads 的 FLOPs 是：

$$ F_{\mathrm{index,score,causal}}=H^Id^IT(T+1). $$

如果实现先形成完整的 $T\times T$ score matrix 再 mask，则是：

$$ F_{\mathrm{index,score,square}}=2H^IT^2d^I. $$

ReLU、乘以 $w_{s,j}^I$、跨 heads 求和以及 Top-$k$ 都不是这里统计的矩阵乘 FLOPs。特别要注意，Top-$k$ 不会让 Indexer 的打分也变成 $O(Tk)$：Indexer 必须先扫描整个 causal prefix，才能知道哪些位置属于 Top-$k$，所以它仍然是 $O(T^2)$。

### Prefill / training

采用 causal-aware 计数时，Indexer 总 FLOPs 是：

$$ F_{\mathrm{index,prefill}}=2T\left(d_{qc}H^Id^I+dd^I+dH^I\right)+H^Id^IT(T+1). $$

代入 $d=7168$、$d_{qc}=1536$、$H^I=64$、$d^I=128$：

$$ F_{\mathrm{index,prefill}}=27{,}918{,}336T+8192T(T+1). $$

对于 $T=131072$：

- 三条 Indexer projection 共约 $3.66$ TFLOPs / layer；
- full-prefix causal scoring 约为 $140.74$ TFLOPs / layer；
- Indexer 合计约为 $144.40$ TFLOPs / layer，不包含 ReLU、reduction 和 Top-$k$。

### Decode

decode 时只生成当前 token 的 $q^I$、$k^I$ 和 head weights，然后让当前 $H^I$ 个 queries 扫描长度为 $t$ 的 shared Indexer key cache：

$$ F_{\mathrm{index,decode}}=2d_{qc}H^Id^I+2dd^I+2dH^I+2H^Itd^I. $$

前三项是当前 token 的 projections，最后一项是随 cache length 增长的 full-prefix scoring。代入 DeepSeek-V3.2 参数：

$$ F_{\mathrm{index,decode}}=27{,}918{,}336+16{,}384t. $$

当 $t=131072$ 时，Indexer 约为 $2.175$ GFLOPs / token / layer，其中 $2.147$ GFLOPs 来自扫描历史 key cache。

## 当前范围内的总 FLOPs

只把本文讨论的 core attention 与 Indexer 相加：

$$ F_{\mathrm{DSA,prefill}}^{\mathrm{core+index}}=2h(2d_c+d_r)N_{\mathrm{sel}}(T,k)+2T\left(d_{qc}H^Id^I+dd^I+dH^I\right)+H^Id^IT(T+1). $$

$$ F_{\mathrm{DSA,decode}}^{\mathrm{core+index}}=2h\min(k,t)(2d_c+d_r)+2d_{qc}H^Id^I+2dd^I+2dH^I+2H^Itd^I. $$

在 $128\mathrm{K}$ 上，前向计算约为 $218.58$ TFLOPs / layer；在 $t=128\mathrm{K}$ 的单 token decode 中约为 $2.746$ GFLOPs / layer。这里的结论边界很明确：**核心注意力已经从 $O(T^2)$ 变成 $O(Tk)$，但 Lightning Indexer 仍保留 $O(T^2)$ 的全前缀打分。**
