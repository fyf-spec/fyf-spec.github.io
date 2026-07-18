---
title: "Linear Attention：From DeltaNet to KDA"
description: "从固定状态的线性注意力出发，逐步推导 DeltaNet、Parallel DeltaNet、Gated DeltaNet、KDA 与 Gated DeltaNet-2。"
date: 2026-07-18
lang: zh-CN
outline: deep
---

# Linear Attention：From DeltaNet to KDA

在普通的 decoder-only Transformer 生成过程中，对第 $t$ 个 token 产生的 $\mathbf q_t,\mathbf k_t,\mathbf v_t$。其中 query 只用于当前这一次读取，但新 key 和 value 则要留下来，供之后每一个 token 查询。下图采用“每个 token 占一行”的矩阵记号：旧 cache 已有 $t-1$ 行，新 token 到来后，$\mathbf k_t^\top$ 与 $\mathbf v_t^\top$ 分别成为 $\mathbf K$ 和 $\mathbf V$ 的第 $t$ 行。因此 KV cache 的大小随序列长度线性增长，给显存带来巨大压力。

![当前 query 读取 K/V cache，而新 key 和 value 分别追加为一个新行](./qkv-cache-append-diagram.png)

*图 1：新 token 产生当前 query、key 和 value；$\mathbf q_t$ 只负责读取，$\mathbf k_t^\top$ 与 $\mathbf v_t^\top$ 各向已有 cache 追加一行。若实现采用转置布局，同一操作会表现为追加一列。*

设旧 cache 已经有 $t-1$ 行，新 token 到来后的完整过程只需写成

$$ \begin{aligned}
\mathbf q_t&=\mathbf x_t\mathbf W_Q,\qquad \mathbf k_t=\mathbf x_t\mathbf W_K,\qquad \mathbf v_t=\mathbf x_t\mathbf W_V,\\
\mathbf K_{\le t}&=\begin{bmatrix}\mathbf K_{<t}\\\mathbf k_t^\top\end{bmatrix},\qquad
\mathbf V_{\le t}=\begin{bmatrix}\mathbf V_{<t}\\\mathbf v_t^\top\end{bmatrix},\qquad
\mathbf o_t^\top=\operatorname{softmax}\!\left(\frac{\mathbf q_t^\top\mathbf K_{\le t}^\top}{\sqrt{d_k}}\right)\mathbf V_{\le t}.
\end{aligned} $$


Linear attention 改变的正是这种 “每来一个 token，就给 K/V 再加一行”的存储方式。它认为可以用一个固定大小的状态来保存历史信息，使每个新的 key-value 对可以当场合并进同一个固定大小的矩阵状态。

![KDA 论文中固定状态矩阵的衰减、擦除、写入与读取](./kda-state-update-paper.png)

*图 2：裁自 [Kimi Linear 第 3 节](https://arxiv.org/abs/2510.26692v2)。蓝色矩阵是固定大小状态，橙色向量表示当前 token 对状态的写入与读取。*

新 token 不再追加一行 K 和一行 V，而是用外积 $\phi(\mathbf k_t)\mathbf v_t^\top$ 修改同一个 $d_\phi\times d_v$ 状态；query 也不再扫描所有历史行，而是直接与该状态相乘。在通道维度固定时，存储与单步计算不再依赖已有多少 token，总计算因而对 $L$ 线性。

这个改变也是代价的来源：Transformer 保留每个 token 的 K/V 行，可以精确访问某个历史位置；linear attention 却让所有历史共享一个有限状态，本质上是**有损的固定状态压缩**。纯加法更新的问题是只会累加。如果状态已经记住“键 A 对应值 2”，后来又收到“键 A 现在对应值 8”，新旧答案会被叠在一起，而不是完成替换。DeltaNet 到 Gated DeltaNet-2 的整条路线，正是在回答固定状态的三个问题：

- 怎样覆盖一条旧关联
- 怎样在 GPU 上并行完成许多次覆盖
- 遗忘、擦除、写入究竟应该控制到多细

## DeltaNet

普通线性注意力解决了 KV cache 增长，却暴露出第一个记忆管理问题：状态只会累加，不会替换。假设状态已经记住“键 A 对应 2”，后来同一个键 A 又对应 8，加法更新会让两次写入叠在一起。DeltaNet 的核心主张是：**不要无条件写入新 value，而要先读出当前答案，只把新旧答案的误差写回同一个 key 地址。**

把状态矩阵记为 $\mathbf S_{t-1}\in\mathbb R^{d_k\times d_v}$。当前 key 为 $\mathbf k_t\in\mathbb R^{d_k}$，目标 value 为 $\mathbf v_t\in\mathbb R^{d_v}$，旧状态给出的预测是

$$ \widehat{\mathbf v}_t=\mathbf S_{t-1}^\top\mathbf k_t. $$

若希望这个预测接近 $\mathbf v_t$，最直接的局部目标就是平方误差

$$ \mathcal L_t(\mathbf S)=\frac12\left\|\mathbf S^\top\mathbf k_t-\mathbf v_t\right\|_2^2. $$

这里的目标不是训练整个网络，而是解释当前 token 如何编辑 fast-weight state。令 $\mathbf e_t=\mathbf S^\top\mathbf k_t-\mathbf v_t$。状态元素 $S_{ab}$ 只通过第 $b$ 个读出中的 $S_{ab}k_{t,a}$ 影响损失，因此矩阵梯度为

$$ \nabla_{\mathbf S}\mathcal L_t=\mathbf k_t\left(\mathbf S^\top\mathbf k_t-\mathbf v_t\right)^\top=\mathbf k_t\mathbf e_t^\top. $$

从 $\mathbf S_{t-1}$ 出发，用当前 token 预测的写入强度 $\beta_t\in(0,1)$ 做一步梯度下降，得到

$$ \mathbf S_t=\mathbf S_{t-1}+\beta_t\mathbf k_t\left(\mathbf v_t-\mathbf S_{t-1}^\top\mathbf k_t\right)^\top. $$

这就是 Delta Rule。括号里的 residual 回答“旧答案错了多少”，左侧的 $\mathbf k_t$ 回答“修改哪个地址”，$\beta_t$ 回答“这次修改多强”。$\beta_t$ 不是训练优化器的全局学习率，而是慢网络根据当前输入生成的逐 token gate；典型写法是 $\beta_t=\sigma(\mathbf w_\beta^\top\mathbf x_t)$。如果删除旧读出 $\mathbf S_{t-1}^\top\mathbf k_t$，更新便退化为只写不擦的加法线性注意力。

展开 residual 后，可以更清楚地看到“先擦后写”：

$$ \mathbf S_t=(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\top)\mathbf S_{t-1}+\beta_t\mathbf k_t\mathbf v_t^\top. $$

第一项沿当前 key 擦除旧读出，第二项沿同一 key 写入新 value。对任意测试方向 $\mathbf x$，

$$ (\mathbf S_t-\mathbf S_{t-1})^\top\mathbf x=\beta_t\left(\mathbf v_t-\mathbf S_{t-1}^\top\mathbf k_t\right)(\mathbf k_t^\top\mathbf x). $$

若 $\mathbf x\perp\mathbf k_t$，右侧为零，所以这次 edit 不改变该方向。若再令 $\|\mathbf k_t\|_2=1$，更新后的当前-key读出为

$$ \mathbf S_t^\top\mathbf k_t=(1-\beta_t)\widehat{\mathbf v}_t+\beta_t\mathbf v_t. $$

因此归一化后的 $\beta_t$ 有精确的插值含义。若 key 未归一化，有效步长会变成 $\beta_t\|\mathbf k_t\|_2^2$，过大的 key 范数可能导致过度擦写。

用一个最小例子检查计算。令 $d_k=2,d_v=1$，$\mathbf S_{t-1}=[2,4]^\top$，$\mathbf k_t=[1,0]^\top$，$v_t=8$，$\beta_t=0.25$。旧读出是 $2$，residual 是 $8-2=6$，写回量为

$$ 0.25\begin{bmatrix}1\\0\end{bmatrix}6=\begin{bmatrix}1.5\\0\end{bmatrix},\qquad \mathbf S_t=\begin{bmatrix}3.5\\4\end{bmatrix}. $$

当前 key 方向从 2 向 8 移动四分之一，正交方向仍为 4。作者在 2021 年论文的重复赋值任务中直接验证了这种覆盖能力；WikiText-103 small 上，加法 Linear Transformer 的 test PPL 为 38.3，Delta Network 为 35.5，但 Transformer 为 34.1。证据支持“误差写入优于对应的纯累加”，并不支持“早期 DeltaNet 已全面超过 Transformer”。

DeltaNet 仍留下两个彼此独立的问题。

- 第一，它的语义虽然好，却必须按 $\mathbf S_1\rightarrow\mathbf S_2\rightarrow\cdots$ 串行执行，训练时难以吃满 GPU；
- 第二，它只修改当前 key 命中的方向，与当前 key 无关的陈旧内容不会主动消失。前一个问题导向 Parallel DeltaNet，后一个问题导向 Gated DeltaNet。


## Parallel DeltaNet

Parallel DeltaNet 要解决的不是“记忆规则不够好”，而是“同一条规则无法高效训练”。直接并行计算每个 token 的 residual 是错误的，因为第 $r$ 个 residual 必须读取已经经过前 $r-1$ 次编辑的状态；如果所有位置都读取 chunk 入口 $\mathbf S_0$，就改变了 Delta Rule 的模型语义。论文的核心主张因此非常具体：**保持逐 token Delta update 完全不变，只把一个 chunk 内的串行依赖精确重排为少量矩阵乘法和一个单位下三角求解。**

先把局部位置记为 $r=1,\ldots,C$，并定义

$$ \mathbf A_r=\mathbf I-\beta_r\mathbf k_r\mathbf k_r^\top,\qquad \mathbf S_r=\mathbf A_r\mathbf S_{r-1}+\beta_r\mathbf k_r\mathbf v_r^\top. $$

这里 $\mathbf A_r$ 是“擦除转移”，第二项是新写入。理解后续符号的关键不是背论文的最终式，而是先问：连续执行 $r$ 次仿射变换后，结果由哪两部分组成？前两步直接展开：

$$ \mathbf S_1=\mathbf A_1\mathbf S_0+\beta_1\mathbf k_1\mathbf v_1^\top, $$

$$ \mathbf S_2=\mathbf A_2\mathbf A_1\mathbf S_0+\mathbf A_2\beta_1\mathbf k_1\mathbf v_1^\top+\beta_2\mathbf k_2\mathbf v_2^\top. $$

结果自然分成“入口状态经过连续擦除后还剩什么”和“chunk 内新写入经过后续擦除后还剩什么”。因此定义

$$ \mathbf S_r=\mathbf P^r\mathbf S_0+\mathbf H^r, $$

$$ \mathbf P^r=\mathbf A_r\mathbf A_{r-1}\cdots\mathbf A_1,\qquad \mathbf H^r=\sum_{i=1}^{r}\left(\mathbf A_r\cdots\mathbf A_{i+1}\right)\beta_i\mathbf k_i\mathbf v_i^\top. $$

当 $i=r$ 时，括号中的空乘积按单位阵处理。$\mathbf P^r\in\mathbb R^{d_k\times d_k}$ 描述入口历史的转移，$\mathbf H^r\in\mathbb R^{d_k\times d_v}$ 描述块内写入的累计贡献。它们不是额外记忆，而是展开原递推后必然出现的两项。

问题还没有解决：若为每个 $r$ 显式构造 $\mathbf P^r$ 和 $\mathbf H^r$，仍会产生许多 $d\times d$ 小矩阵。为什么想到 WY 形式？因为每个 $\mathbf A_r$ 都是“单位阵减 rank-one”，而每次写入也以 $\mathbf k_r$ 为左因子；所以所有变化始终落在 chunk 的 key 张成的子空间里。最自然的目标便是只保存这些 key 的外积系数：

$$ \mathbf P^r=\mathbf I-\sum_{i=1}^{r}\mathbf k_i\mathbf w_i^\top,\qquad \mathbf H^r=\sum_{i=1}^{r}\mathbf k_i\mathbf u_i^\top. $$

下面不能把 $\mathbf w_r,\mathbf u_r$ 当作凭空定义的辅助变量；它们是为了让这个外积形式在第 $r$ 步之后仍然成立而被系数匹配出来的。假设前 $r-1$ 步已有 $\mathbf P^{r-1}=\mathbf I-\sum_{i<r}\mathbf k_i\mathbf w_i^\top$，则

$$ \begin{aligned}\mathbf P^r&=(\mathbf I-\beta_r\mathbf k_r\mathbf k_r^\top)\mathbf P^{r-1}\\&=\mathbf P^{r-1}-\mathbf k_r\left(\beta_r\mathbf k_r^\top\mathbf P^{r-1}\right).\end{aligned} $$

为了让最后一项继续写成 $\mathbf k_r\mathbf w_r^\top$，右因子只能取

$$ \mathbf w_r^\top=\beta_r\mathbf k_r^\top\mathbf P^{r-1}, $$

也就是

$$ \mathbf w_r=\beta_r\left(\mathbf k_r-\sum_{i<r}\mathbf w_i(\mathbf k_i^\top\mathbf k_r)\right). $$

同理，新写入部分满足

$$ \mathbf H^r=\mathbf A_r\mathbf H^{r-1}+\beta_r\mathbf k_r\mathbf v_r^\top. $$

代入 $\mathbf H^{r-1}=\sum_{i<r}\mathbf k_i\mathbf u_i^\top$ 并收集新出现的 $\mathbf k_r$ 系数：

$$ \begin{aligned}\mathbf H^r&=\mathbf H^{r-1}-\beta_r\mathbf k_r\mathbf k_r^\top\sum_{i<r}\mathbf k_i\mathbf u_i^\top+\beta_r\mathbf k_r\mathbf v_r^\top\\&=\mathbf H^{r-1}+\mathbf k_r\left[\beta_r\left(\mathbf v_r-\sum_{i<r}\mathbf u_i(\mathbf k_i^\top\mathbf k_r)\right)\right]^\top.\end{aligned} $$

因此

$$ \mathbf u_r=\beta_r\left(\mathbf v_r-\sum_{i<r}\mathbf u_i(\mathbf k_i^\top\mathbf k_r)\right). $$

$\mathbf u_r$ 被称为 pseudo-value，因为它不是原始 value，而是已经扣除了此前有效写入在当前 key 上造成的旧读出；它同时携带“写入新内容”和“抵消旧关联”两种作用。$\mathbf w_r$ 则描述入口旧状态应怎样被这些 edit 擦除。两条递推共享 $\mathbf k_i^\top\mathbf k_r$，原因不是巧合：$\mathbf P$ 与 $\mathbf H$ 都受到同一个 $\mathbf A_r$ 作用。若两把 key 正交，该内积为零，第 $r$ 次编辑便无需修正第 $i$ 次编辑。

一个两步例子可以看出 pseudo-value 为什么会出现负数。令 $\mathbf k_1=\mathbf k_2=[1,0]^\top$，$\beta_1=1$，$\mathbf v_1=[2,0]^\top$，再令 $\beta_2=0.5$，$\mathbf v_2=[0,2]^\top$。第一步 $\mathbf u_1=[2,0]^\top$；第二步因为两把 key 完全重合，

$$ \mathbf u_2=0.5\left(\begin{bmatrix}0\\2\end{bmatrix}-\begin{bmatrix}2\\0\end{bmatrix}\right)=\begin{bmatrix}-1\\1\end{bmatrix}. $$

最终同一 key 上的有效 value 是 $\mathbf u_1+\mathbf u_2=[1,1]^\top$，正好是旧值与新值的 50% 插值。负分量不是“负记忆”，而是覆盖旧答案所需的抵消量。

到这里，$C$ 个 $\mathbf u_r,\mathbf w_r$ 仍然按 $r$ 递推。下一步为什么会想到下三角系统？因为第 $r$ 个量只依赖 $i<r$，把这些因果方程按行堆起来，系数天然就是严格下三角。定义按行堆叠

$$ \mathbf K[r,:]=\mathbf k_r^\top,\quad \mathbf V[r,:]=\mathbf v_r^\top,\quad \mathbf U[r,:]=\mathbf u_r^\top,\quad \mathbf W[r,:]=\mathbf w_r^\top, $$

其中 $\mathbf K,\mathbf W\in\mathbb R^{C\times d_k}$，$\mathbf V,\mathbf U\in\mathbb R^{C\times d_v}$。再令

$$ \mathbf D=\operatorname{Diag}(\beta_1,\ldots,\beta_C),\qquad \mathbf G=\mathbf K\mathbf K^\top,\qquad \mathbf L=\operatorname{tril}(\mathbf D\mathbf G,-1). $$

于是 $L_{ri}=\beta_r\mathbf k_r^\top\mathbf k_i$ 当且仅当 $i<r$，其余为零。注意 $\mathbf D$ 必须放在 Gram 矩阵左侧，因为 $\beta_r$ 属于第 $r$ 条方程、应缩放第 $r$ 行；若放在右侧就会错误地使用 $\beta_i$。将 $\mathbf u_r$ 的递推移项并转置：

$$ \mathbf u_r^\top+\sum_{i<r}\beta_r(\mathbf k_r^\top\mathbf k_i)\mathbf u_i^\top=\beta_r\mathbf v_r^\top. $$

这正是矩阵方程第 $r$ 行，因此全部递推可写为

$$ (\mathbf I+\mathbf L)\mathbf U=\mathbf D\mathbf V,\qquad (\mathbf I+\mathbf L)\mathbf W=\mathbf D\mathbf K. $$

以 $C=3$ 为例，记 $g_{ri}=\mathbf k_r^\top\mathbf k_i$：

$$ \mathbf I+\mathbf L=\begin{bmatrix}1&0&0\\\beta_2g_{21}&1&0\\\beta_3g_{31}&\beta_3g_{32}&1\end{bmatrix}. $$

第一行得到 $\mathbf u_1^\top=\beta_1\mathbf v_1^\top$；第二行扣除位置 1 对位置 2 的影响；第三行扣除位置 1、2 对位置 3 的影响。下三角系统没有加入新的计算语义，只是把 $C$ 条因果递推改写成一个可批量求解的对象。令

$$ \mathbf T=(\mathbf I+\mathbf L)^{-1}\mathbf D, $$

便有

$$ \mathbf U=\mathbf T\mathbf V,\qquad \mathbf W=\mathbf T\mathbf K. $$

$\mathbf T\in\mathbb R^{C\times C}$ 不是转置符号，也不是可学习参数；$T_{ri}$ 表示原始第 $i$ 个 key/value 对第 $r$ 个有效系数的贡献。公式写成逆是为了简洁，实现不会调用通用矩阵求逆，而会利用 $\mathbf I+\mathbf L$ 是单位下三角矩阵做 forward substitution，再把主要工作交给矩阵乘法。

最后把紧凑因子重新组装成 chunk 更新。因为

$$ \mathbf P^C=\mathbf I-\mathbf K^\top\mathbf W,\qquad \mathbf H^C=\mathbf K^\top\mathbf U, $$

所以

$$ \begin{aligned}\mathbf S_C&=(\mathbf I-\mathbf K^\top\mathbf W)\mathbf S_0+\mathbf K^\top\mathbf U\\&=\mathbf S_0+\mathbf K^\top(\mathbf U-\mathbf W\mathbf S_0).\end{aligned} $$

定义净更新

$$ \mathbf R=\mathbf U-\mathbf W\mathbf S_0\in\mathbb R^{C\times d_v}, $$

其中 $\mathbf U$ 是块内有效新写入，$\mathbf W\mathbf S_0$ 是同一组 edit 应从具体入口状态中擦掉的旧读出。于是出口状态与全部 chunk 输出分别为

$$ \mathbf S_C=\mathbf S_0+\mathbf K^\top\mathbf R, $$

$$ \mathbf O=\mathbf Q\mathbf S_0+(\mathbf Q\mathbf K^\top\odot\mathbf M)\mathbf R. $$

$\mathbf Q\mathbf S_0$ 读取 chunk 之前的历史；$\mathbf Q\mathbf K^\top\odot\mathbf M$ 计算 chunk 内带因果 mask 的 query–key 相关性；再乘 $\mathbf R$，便读取当前位置之前已经发生的净编辑。这样，chunk 内 token 通过大矩阵运算并行，只有 chunk 出口状态仍按块递推。令 $C=1$ 会退化为逐 token recurrence；令 $C$ 过大则会让 $C\times C$ 相关矩阵与下三角求解变贵，因此 chunk size 是并行度、局部二次项与硬件利用率之间的工程折中。

复杂度也能从形状直接看出。单个 chunk 的 token–token 相关和下三角路径约为 $O(C^2d)$，与 $d\times d$ 状态交互约为 $O(Cd^2)$；共有 $L/C$ 个 chunk，因此

$$ \frac LC\left[O(C^2d)+O(Cd^2)\right]=O(LCd+Ld^2). $$

“Parallel”不表示总 FLOPs 必然更少，$Ld^2$ 主项仍在；真正收益是把 $L$ 次小而串行的更新改成约 $L/C$ 个边界递推，并把块内工作变成 GPU 擅长的 GEMM。作者报告 chunkwise kernel 相对其递归 kernel 约有 4–36 倍加速，但端到端 1.3B 模型在 2K 时仍比 Transformer++ 慢约 19%，到 16K 才快约 28%。训练和长提示 prefill 适合 chunkwise 形式；自回归 decode 每次只有一个新 token，直接维护原始 $\mathbf S_t$ 更自然。

Parallel DeltaNet 至此解决了“怎样高效执行 Delta Rule”，却没有改变 Delta Rule 的记忆策略：只有当前 key 命中的方向会被覆盖，与当前 key 无关的陈旧内容仍可能永久占用固定状态。这正是下一篇工作要处理的问题。


## Gated DeltaNet

Parallel DeltaNet 让定向覆盖可以高效训练，但没有回答“什么时候应该主动清空旧背景”。Delta Rule 很精确：它只改当前 key 方向；代价是主题已经切换时，其他方向的旧内容仍留在有限状态里。Mamba2 式 scalar decay 恰好相反：它能迅速缩小全部历史，却无法根据当前 key 的旧读出精确替换某一条关联。Gated DeltaNet 的核心主张是：**让 scalar decay 负责全局清理，让 Delta Rule 负责局部纠错，两者不是竞争关系，而是互补的记忆操作。**

先用当前 head 的保留率 $\alpha_t\in(0,1)$ 衰减旧状态，再在衰减后的状态上执行同一条 Delta correction：

$$ \widetilde{\mathbf S}_{t-1}=\alpha_t\mathbf S_{t-1},\qquad \widehat{\mathbf v}_t=\widetilde{\mathbf S}_{t-1}^\top\mathbf k_t, $$

$$ \mathbf S_t=\widetilde{\mathbf S}_{t-1}+\beta_t\mathbf k_t(\mathbf v_t-\widehat{\mathbf v}_t)^\top. $$

合并后是

$$ \mathbf S_t=\alpha_t(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\top)\mathbf S_{t-1}+\beta_t\mathbf k_t\mathbf v_t^\top. $$

$\alpha_t$ 回答“这个 head 的全部历史保留多少”，$\beta_t$ 回答“当前 key 方向改写多少”。若 $\|\mathbf k_t\|_2=1$，所有与 $\mathbf k_t$ 正交的方向乘 $\alpha_t$，当前 key 方向额外再乘 $1-\beta_t$；这给出全局遗忘和定向擦除的严格分工。

为什么先使用标量而不是更细的向量 gate？表达力上，标量当然更弱；工程上，它与任何矩阵可交换：

$$ \prod_{r=1}^{C}\alpha_r(\mathbf I-\beta_r\mathbf k_r\mathbf k_r^\top)=\left(\prod_{r=1}^{C}\alpha_r\right)\prod_{r=1}^{C}(\mathbf I-\beta_r\mathbf k_r\mathbf k_r^\top). $$

累计 decay 可以从 WY/UT 的 rank-one 乘积中单独提出，所以 Parallel DeltaNet 的 chunkwise 结构几乎可以原样保留。若一开始就改成任意对角矩阵，衰减与 rank-one edit 通常不交换，原并行推导会被打破；KDA 后来要解决的正是这个困难。

作者用 S-NIAH 的三种失败模式说明两种操作为何互补：

| 方法 | 简单长期保存 S1@8K | 强干扰 S2@4K | 复杂 value S3@2K |
|---|---:|---:|---:|
| DeltaNet | **98.8** | 18.6 | 47.0 |
| Mamba2 | 30.4 | 56.2 | 47.6 |
| Gated DeltaNet | 91.8 | **92.2** | **84.2** |

作者指出，简单长期保存时不主动遗忘的 DeltaNet 最好；强干扰时它因缺少清理而发生碰撞；复杂 value 又要求基于旧读出误差的精确写入，因此只有 decay 的 Mamba2 不够。GDN 的平均表现更强，但 S1 从 98.8 降到 91.8 也说明 gate 若判断错误，会损坏本可保留的记忆。1.3B/100B-token 实验支持纯 GDN 整体优于 Mamba2 与 DeltaNet、训练吞吐与 DeltaNet 接近；真实检索上纯 GDN 仍落后 attention/hybrid，说明它改善了有限状态的管理，却没有消除容量上限。

GDN 接下来的瓶颈恰好来自这个硬件友好的 scalar gate：一个 head 内不同 key 通道只能以同一速度衰减，无法让一部分通道长期保存、另一部分通道快速更新。KDA 将从这里把“全局遗忘”细化为“逐通道遗忘”。


## KDA

Gated DeltaNet 已经能决定一个 head 何时整体遗忘，但一个标量 $\alpha_t$ 迫使该 head 的所有 key 通道同寿命。可以把它理解为一排记忆槽只有一个总闸门：开小了所有槽都忘得快，开大了所有槽都记得久。Kimi Delta Attention（KDA）的核心主张是：**把长期保留率细化到 key/state 的每个通道，同时把转移限制在可高效并行的 diagonal-plus-rank-one 结构中。**

令 $\boldsymbol\alpha_t\in(0,1]^{d_k}$，并定义 key 轴上的对角衰减矩阵

$$ \mathbf D_t=\operatorname{Diag}(\boldsymbol\alpha_t)\in\mathbb R^{d_k\times d_k}. $$

KDA 先逐通道衰减旧状态，再执行 Delta correction：

$$ \widetilde{\mathbf S}_{t-1}=\mathbf D_t\mathbf S_{t-1},\qquad \widehat{\mathbf v}_t=\widetilde{\mathbf S}_{t-1}^\top\mathbf k_t, $$

$$ \mathbf S_t=\widetilde{\mathbf S}_{t-1}+\beta_t\mathbf k_t(\mathbf v_t-\widehat{\mathbf v}_t)^\top. $$

合并后为

$$ \mathbf S_t=(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\top)\mathbf D_t\mathbf S_{t-1}+\beta_t\mathbf k_t\mathbf v_t^\top. $$

最容易误解的地方有两个。第一，变成向量的是长期 decay $\boldsymbol\alpha_t$，主动 edit 的强度 $\beta_t$ 仍是标量。第二，$\mathbf D_t$ 左乘状态，所以它控制 key/state 行，而不是给 value 的每一维加 write gate。若令 $\boldsymbol\alpha_t=\alpha_t\mathbf 1$，KDA 就退化为 Gated DeltaNet。

为什么不能简单把 GDN 的 scalar $\alpha_t$ 换成向量后结束？因为一般有

$$ \mathbf D_t\mathbf k_t\mathbf k_t^\top\ne\mathbf k_t\mathbf k_t^\top\mathbf D_t. $$

scalar decay 能从 rank-one 转移积中提出，diagonal decay 却会与每次擦除纠缠。KDA 的关键算法工作不是“多预测一个向量”，而是找到一个变量替换，把累计逐通道衰减吸收到 key 两侧，使递推重新回到 Parallel DeltaNet 能处理的 generalized rank-one 形式。

在一个 chunk 内定义累计保留率

$$ \boldsymbol\gamma_r=\boldsymbol\alpha_r\odot\boldsymbol\alpha_{r-1}\odot\cdots\odot\boldsymbol\alpha_1, $$

并令

$$ \mathbf S_r=\operatorname{Diag}(\boldsymbol\gamma_r)\widehat{\mathbf S}_r. $$

因为 $\operatorname{Diag}(\boldsymbol\gamma_r)=\mathbf D_r\operatorname{Diag}(\boldsymbol\gamma_{r-1})$，代回 KDA recurrence 后左乘 $\operatorname{Diag}(\boldsymbol\gamma_r)^{-1}$。再定义

$$ \mathbf a_r=\mathbf k_r/\boldsymbol\gamma_r,\qquad \mathbf b_r=\boldsymbol\gamma_r\odot\mathbf k_r, $$

其中乘除均逐元素进行，便得到

$$ \widehat{\mathbf S}_r=(\mathbf I-\beta_r\mathbf a_r\mathbf b_r^\top)\widehat{\mathbf S}_{r-1}+\beta_r\mathbf a_r\mathbf v_r^\top. $$

这一步负责什么？$\mathbf a_r$ 是移除累计 decay 后的写入地址，$\mathbf b_r$ 是带累计 decay 的擦除读方向；两者不再相同，但状态转移仍是“单位阵减 rank-one”。因此 Parallel DeltaNet 的核心思路可以复用：用块内因果内积 $\mathbf b_r^\top\mathbf a_i$ 构造严格下三角系统，求出 pseudo-values 和擦除系数，再在 chunk 边界恢复 $\operatorname{Diag}(\boldsymbol\gamma_C)$。这不是近似，而是同一递推的精确重参数化。

KDA 也可以被看成受约束的 DPLR（Diagonal-Plus-Low-Rank）转移：$\mathbf D_t$ 给各通道不同寿命，rank-one 项负责当前 key 的定向 edit。它没有采用完全自由的两个低秩因子，因为一般 DPLR 虽然更灵活，却需要更多带累计 decay 的相关矩阵、secondary chunking 和矩阵乘法。作者报告 KDA kernel 在 64K 上相对其一般 DPLR 实现约为 $1.98\times$；这里是 kernel 对比，不是完整模型的同等倍数。

累计归一化也带来明确代价：当某个 $\gamma_{r,j}$ 很小时，$\mathbf k_r/\boldsymbol\gamma_r$ 可能在低精度中上溢。受约束 DPLR 减少了需要稳定化的计算路径，却没有消除除法；实际实现仍需更高精度累计或二级分块等稳定化手段。

完整 Kimi Linear 并不是纯 KDA，而是 3 个 KDA 层配 1 个全局 MLA 层。作者明确指出，固定状态仍难以无损完成任意精确长程检索，因此让 KDA 低成本压缩大部分历史，让 MLA 周期性保留 token 级全局访问。混合比例消融中，纯 MLA、1:1、3:1、7:1、15:1 的验证 PPL 分别为 5.77、5.66、5.65、5.70、5.82；3:1 是该实验配方下的最佳折中，不是理论常数。速度数字也要区分：KDA kernel、单请求 decode、以及利用更低 KV cache 扩大 batch 后的系统吞吐不是同一口径。

KDA 解决了“不同 key 通道应该忘得多快”，但一次主动 Delta edit 内部仍只有一个 $\beta_t$：旧关联擦多少与新 value 写多少被绑定在同一个标量上。Gated DeltaNet-2 要拆开的正是这两个决定。


## Gated DeltaNet-2

KDA 的逐通道 gate 控制的是长期衰减，并没有细化当前这一次主动编辑。用同一个 $\beta_t$ 同时控制 erase 与 write，隐含了两个约束：若只想删除旧关联却不想写入全部新 value，做不到；若只想写入 value 的部分通道，也必须接受同强度的旧值擦除。Gated DeltaNet-2（GDN2）的核心主张是：**长期 decay、读取并擦除旧关联、写入新 value 是三个不同问题，应该由不同轴上的 gate 控制。**

保留 KDA 的逐 key 通道 decay

$$ \mathbf D_t=\operatorname{Diag}(\boldsymbol\alpha_t), $$

再定义

$$ \mathbf e_t=\mathbf b_t\odot\mathbf k_t\in\mathbb R^{d_k},\qquad \mathbf z_t=\mathbf w_t\odot\mathbf v_t\in\mathbb R^{d_v}. $$

$\mathbf b_t\in[0,1]^{d_k}$ 是 key/erase 侧 gate，决定形成待擦旧读出时哪些 key 通道参与；$\mathbf w_t\in[0,1]^{d_v}$ 是 value/write 侧 gate，决定新 value 的哪些通道成为写入目标。完整更新分四步：

$$ \overline{\mathbf S}_t=\mathbf D_t\mathbf S_{t-1},\qquad \mathbf r_t=\overline{\mathbf S}_t^\top\mathbf e_t, $$

$$ \boldsymbol\delta_t=\mathbf z_t-\mathbf r_t,\qquad \mathbf S_t=\overline{\mathbf S}_t+\mathbf k_t\boldsymbol\delta_t^\top. $$

自然语言就是：先让旧状态按 key 通道长期衰减；再用 gated erase key 读取“这次准备替换的旧内容”；用 gated 新 value 减去旧读出得到 residual；最后仍沿当前 $\mathbf k_t$ 指定的地址写回。合并后为

$$ \mathbf S_t=(\mathbf I-\mathbf k_t\mathbf e_t^\top)\mathbf D_t\mathbf S_{t-1}+\mathbf k_t\mathbf z_t^\top. $$

这里最容易误解的是 $\mathbf b_t$ 的作用。它不会直接把状态的某几行清零，而是改变旧读出 $\mathbf r_t$ 如何从 key 轴聚合；真正的状态改变量仍是 rank-one 外积 $\mathbf k_t(\mathbf z_t-\mathbf r_t)^\top$。写入地址仍由 $\mathbf k_t$ 决定，GDN2 增加的是 erase/read 侧和 value/write 侧的通道选择自由度，并没有把 rank-one edit 变成任意矩阵更新。

用一个二维例子检查各轴。设衰减后的状态为 $\overline{\mathbf S}_t=\begin{bmatrix}2&0\\0&1\end{bmatrix}$，$\mathbf k_t=[1,0]^\top$，$\mathbf b_t=[0.5,1]^\top$，$\mathbf v_t=[4,6]^\top$，$\mathbf w_t=[1,0.25]^\top$。于是

$$ \mathbf e_t=\begin{bmatrix}0.5\\0\end{bmatrix},\qquad \mathbf z_t=\begin{bmatrix}4\\1.5\end{bmatrix},\qquad \mathbf r_t=\overline{\mathbf S}_t^\top\mathbf e_t=\begin{bmatrix}1\\0\end{bmatrix}. $$

residual 为 $\boldsymbol\delta_t=[3,1.5]^\top$，所以

$$ \mathbf S_t=\begin{bmatrix}2&0\\0&1\end{bmatrix}+\begin{bmatrix}1\\0\end{bmatrix}\begin{bmatrix}3&1.5\end{bmatrix}=\begin{bmatrix}5&1.5\\0&1\end{bmatrix}. $$

$\mathbf b_t$ 决定从 key 轴读取多少旧内容，$\mathbf w_t$ 决定 value 轴上哪些新目标进入 residual，$\mathbf k_t$ 决定最终把 residual 写到哪个地址。三个量的作用轴因此不能互换。

退化关系揭示了这条发展路线的包含关系。若

$$ \mathbf b_t=\beta_t\mathbf 1_{d_k},\qquad \mathbf w_t=\beta_t\mathbf 1_{d_v}, $$

则 $\mathbf e_t=\beta_t\mathbf k_t$、$\mathbf z_t=\beta_t\mathbf v_t$，GDN2 退化为 KDA；若再令 $\boldsymbol\alpha_t=\alpha_t\mathbf 1$，就退化为 Gated DeltaNet；再令 $\alpha_t=1$，得到 DeltaNet。这证明表达空间严格包含，至于训练能否有效利用新增自由度，仍需实验判断。

为什么增加两个向量 gate 后还可以并行？对 chunk 内累计 decay 做与 KDA 相同的归一化，定义 $\overline{\mathbf k}_r=\mathbf k_r/\boldsymbol\gamma_r$、$\overline{\mathbf e}_r=\boldsymbol\gamma_r\odot\mathbf e_r$，递推会变成

$$ \widehat{\mathbf S}_r=(\mathbf I-\overline{\mathbf k}_r\overline{\mathbf e}_r^\top)\widehat{\mathbf S}_{r-1}+\overline{\mathbf k}_r\mathbf z_r^\top. $$

它仍是 generalized rank-one recurrence，所以 Parallel DeltaNet 的“因果内积→单位下三角系统→pseudo-value→chunk 更新”主链可以复用；区别只是干扰系数从 $\mathbf k_r^\top\mathbf k_i$ 变为 $\overline{\mathbf e}_r^\top\overline{\mathbf k}_i$，并让写入侧与擦除侧共享同一个下三角系统。本文不再重复一遍 WY/UT 细节，因为计算理由与 Parallel DeltaNet 相同。

作者的消融支持两个 gate 都有作用：

| 变体 | Wiki PPL ↓ | S3@2K ↑ | Multi-key@4K ↑ | 真实检索平均 ↑ |
|---|---:|---:|---:|---:|
| 仅 $\mathbf w$ 逐通道，$\mathbf b$ 标量 | 16.55 | 71.4 | 30.6 | 28.92 |
| 仅 $\mathbf b$ 逐通道，$\mathbf w$ 标量 | 16.12 | 84.6 | 35.2 | 29.51 |
| **$\mathbf b,\mathbf w$ 都逐通道** | **15.90** | **89.8** | **37.8** | **29.88** |

从实验可以推断，key/erase 侧的选择性是主要增益，value/write gate 提供进一步改善；论文没有证明某个具体通道固定负责某类语义。表达力也并非免费：单 H100、1.3B hybrid 训练中，GDN2 相比 KDA 在 2K 慢约 4.5%，16K 慢约 6.2%，原因是向量 gate 位于通道求和内部，需要 gate-aware backward 与 fused kernel。论文目前的证据仍局限在 1.3B/100B、最长 8K 合成上下文，没有证明固定状态能够无损承载任意长历史，也没有给出百万上下文服务延迟的终局结论。


## 结语

现在可以直接回答开头的问题。Linear attention 用固定大小状态替代随序列增长的 KV cache，把历史从“逐 token 显式存储”改成“有限矩阵中的有损压缩”。因此这条研究路线真正追问的不是怎样再写一个线性公式，而是怎样让有限状态可编辑、可遗忘，而且能在 GPU 上高效执行。

DeltaNet 先解决纯累加无法覆盖旧答案的问题：读取当前 key 的旧值，只写 residual。Parallel DeltaNet 没有改变这条记忆规则，而是通过展开仿射递推、匹配紧凑 WY 系数、再把因果依赖堆成单位下三角系统，使同一语义可以 chunkwise 训练。Gated DeltaNet 增加可交换的 scalar decay，把全局清理与定向覆盖结合；KDA 将长期 decay 细化到 key 通道，并用累计 decay 归一化恢复 generalized rank-one 结构；GDN2 最后把主动 edit 内部的 erase/read 与 value/write 拆成两个轴上的 gate。

因此主线可以压缩成

$$ \text{纯累加}\rightarrow\text{误差覆盖}\rightarrow\text{分块并行}\rightarrow\text{全局遗忘}\rightarrow\text{逐通道遗忘}\rightarrow\text{擦除与写入解耦}. $$

作者明确建立得最稳的是这些 recurrence 的代数关系、Parallel DeltaNet 的精确 chunkwise 等价，以及特定硬件配置下的 kernel 与训练吞吐。实验支持更细粒度控制在强干扰、重复覆盖任务中更有效利用有限容量，但不能推出容量限制已经消失。Kimi Linear 仍保留四分之一全局 MLA，GDN/GDN2 的强配置也常与滑动窗口 attention 混合，说明当前务实的结论仍是分工：线性状态负责低成本压缩和大部分时序混合，attention 负责需要 token 级精确访问的部分。


## 参考资料

- [Linear Transformers Are Secretly Fast Weight Programmers（ICML 2021）](https://proceedings.mlr.press/v139/schlag21a.html)
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d13a3eae72366e61dfdc7eea82eeb685-Abstract-Conference.html)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule（ICLR 2025）](https://openreview.net/forum?id=r8H7xhYPwz)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692v2)
- [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/abs/2605.22791v1)
