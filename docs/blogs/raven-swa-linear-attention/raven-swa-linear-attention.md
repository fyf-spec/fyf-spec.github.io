---
title: "Sparse linear attention ：从 Linear Attention 角度看 SWA"
description: "从 memory slot、FIFO 与稀疏路由出发，理解 SWA、状态式 Linear Attention 与 Raven 之间的关系。"
date: 2026-08-09
lang: zh-CN
outline: false
---

# Sparse linear attention ：从 Linear Attention 角度看 SWA

由于 Linear attention 中衰减系数的存在，Linear attention的上下文能力通常受到一定限制，可以将其看做一种窗口很长的 Sliding window attention(SWA)。但从另一个角度来说，SWA 也可以看做一种高度离散化的 linear attention，SWA保存下来的 KV cache 可以看做 linear attention 中高度离散化的状态空间。接下来我们从两种视角分别讨论，并且介绍一些相关的论文，比如Raven。

## 前情提要：Linear attention
正常的 linear state model 会维护一个固定大小的状态矩阵 $\mathbf S_t$。每个新 token 到来时，模型先衰减旧状态，再用当前 key-value 的外积写入新信息，最后用 query 读取：

$$ \mathbf S_t=\mathbf S_{t-1}\mathbf A_t+\mathbf v_t\mathbf k_t^\top,\qquad \mathbf o_t=\mathbf S_t\mathbf q_t. $$

其中 $\mathbf A_t$ 控制历史信息的保留程度，$\mathbf v_t\mathbf k_t^\top$ 负责写入，$\mathbf q_t$ 负责读取。下面从这个固定状态更新出发，重新理解 SWA。

![SSM、SWA 与 Raven 的状态更新对比](image.png)
*图 1：SSM 稠密更新整个状态，SWA 按 FIFO 更新一行 KV cache，Raven 则通过路由选择要更新的固定槽位。*

## 从 linear state 角度看 SWA
在 SWA 中，每添加一个 token， KV Cache 中会就会添加一行，其他行保持不变。如果把 SWA 的每一行 KV cache 看成一个 slot，其更新可以写成

$$ \mathbf S_t=(\mathbf 1-\mathbf e_t)\odot\mathbf S_{t-1}+\mathbf e_t\mathbf u_t^\top. $$

$\mathbf e_t$ 是周期变化的 one-hot 向量，每次只选择一行：清除旧 K/V 后写入新 token，其余行保持不变；由于只保持窗口内的 KV cache 可见，当 cache 长度达到窗口长度后，最早内容被踢出，这就是 FIFO (First in first out)。

为方便与 linear attention 的 state 对比，我们不妨设 window 大小为 $d$ ，此时 KV cache 固定为了 $d$ x $d$ 大小的状态空间！因此，可将 $t+1$ 步写入的 token 看做写一个 $t+1$ 行处的 slot，体现了对状态空间的写稀疏。

## 从 SWA 角度看 linear state

反过来，也可以用 SWA 的 slots 理解上面的 linear attention / SSM 更新。

若继续把 $\mathbf S_t$ 的每一行看做一个 slot，SWA 的写入向量是 one-hot $\mathbf e_t$，而 linear state 的 $\mathbf v_t$ 通常是 dense 的：每个 token 都会同时写入所有 slots。$\mathbf A_t$ 则不断衰减旧状态，使历史 token 的贡献逐渐变小，而不是像 SWA 一样在窗口边界被一次性踢出。因此，带 decay 的 linear attention 可以理解为一种没有硬边界、窗口长度由 decay 动态决定的“soft SWA”。

> 注意，decay 并非所有 linear attention 都具备；朴素 linear attention 仍然可以只累加。


## Sparse linear memory：两者的中间态

因此，我们可以初步建立感觉：SWA 是每步更新一个slot，传统 Linear attention 是每步更新所有 slot，那么有没有折中的方案，让我们能既保持linear attention的表达能力，又保持SWA的稀疏更新？

于是我们想到，我们每步可以只更新一部分 slots，这样状态空间的更新就是稀疏的，这就是 sparse linear memory 的来源。其中 slots 的选取有很多方法，比如 [Raven](https://goombalab.github.io/blog/2026/raven-part2/#from-framework-to-model) 的 $Top-K$ 路由，[SDM](https://arxiv.org/abs/2607.07386) 的 product-key。我们设选取 $K$ 个 slots，当 $K=1$、周期轮转且完全覆盖时，它退化为 SWA；当 $K=M$ 时，它只在写入行为上接近稠密 SSM；$1<K<M$ 则形成两者之间的稀疏 linear memory。它可以用 router 与 sparse kernel 的复杂度，换取更少的写入干扰和更长的关键记忆。

## Sparse linear memory 的长期问题

对于这个思路，我仍抱有一些问题：稀疏写入虽减少干扰，却未消除有限容量，也许可以外推到5M，但再之后呢？模型能用这种方法获得真正的长上下文记忆能力吗？

另外，文本越长，固定 slots 越容易碰撞；扩大 memory 后，又必须用有限读取预算找到目标。并且对于更大的状态空间，对于显存的消耗和 infra 要求也大幅提高。

更根本的是，写入早于未来 query，模型无法预知哪些普通 token 日后会变得重要。pattern 在训练时 data-independent 已经定下来了，怎么持续学习而不只是把 state 做大？
