# DSA 之后：Indexer 优化的两条路线


## 稀疏注意力之后，为什么还有一个二次复杂度？

我最初看 DSA 时有一个很自然的困惑：假设模型面对长度为 128K 的上下文，最终只让每个 query 读取其中 2K 个 token，它似乎已经跳过了绝大部分历史，为什么长上下文下还会出现新的性能瓶颈？顺着执行流程往前看一步，答案就很直接了——模型必须先知道应该读取哪 2K 个 token，而“找出这 2K 个 token”本身也需要计算。

DeepSeek Sparse Attention（DSA）的答案是在主注意力之外增加一个轻量 indexer。它先让当前 query 与所有历史 key 计算相关性，从中选出 top-$k$，然后才让真正的 Sparse MLA 读取这些 token。

对位置 $t$ 的 query 和历史位置 $s$，DSA indexer 的分数可以概括为：

$$ I_{t,s}=\sum_{j=1}^{H_I}w^I_{t,j}\operatorname{ReLU}\!\left(q^I_{t,j}\cdot k^I_s\right) $$

其中 $H_I$ 是 indexer head 数，$q^I_{t,j}$ 和 $k^I_s$ 是低维 indexing query/key，$w^I_{t,j}$ 决定各个 head 对当前 query 的贡献。

这比直接运行完整 attention 便宜得多，但它仍然包含一个无法绕开的动作：**每个 query 先扫描所有历史 token。** 如果序列长度为 $L$，主注意力已经从 $O(L^2)$ 降到 $O(Lk)$，indexer 却仍会产生 $O(L^2)$ 次 query-key score。随着 Sparse MLA 越来越快，indexer 反而会从一个“小配件”变成新的主耗时。这也构成了我理解 DSA 后续工作的起点：

> 如果 indexer 的职责是帮助 attention 少看一些 token，那么 indexer 自己能不能也少看一些？

我把目前的工作大致整理成两条路线。第一条先接受 DSA indexer 的基本形态，再观察它算出来的 score 在哪些维度上存在冗余；第二条则退后一步，重新思考候选地址究竟应该怎样产生：

1. **从 index score pattern 出发。** 先观察完整 index score 或 top-$k$ 结果在哪些维度存在冗余，再让原有 indexer 少算、少选或复用已有结果。
2. **从 indexer architecture 出发。** 不再假定“低维 query 与全部 key 做一次平坦扫描”是唯一形式，而是重新设计候选生成、训练监督和系统执行路径。


## 从 score pattern 中寻找冗余

先把一次 indexer 的工作粗略写成：

$$ W_{\mathrm{index}}\propto U_L\times H_I\times Q_I\times N_I $$

- $U_L$：真正独立产生索引的层数；
- $H_I$：参与精确打分的 indexer heads；
- $Q_I$：需要分别发起检索的 queries；
- $N_I$：每次检索需要扫描的历史 keys。

传统 DSA 在这些维度上几乎都是“全量”：每层重新计算、所有 heads 都参与、每个 query 独立检索、每次都扫描完整历史。后续工作并没有立刻推翻整个 indexer，而是先回到实际路由图中寻找冗余：相邻 query 是否在反复选择同一批 token，相邻 layer 是否也在重复近似的检索结果？下面两张图给出了一个直观切片。

<figure style="margin: 1.75rem auto 2rem; width: 100%;">
  <div style="display: flex; gap: 14px; align-items: flex-start; justify-content: center; flex-wrap: wrap;">
    <div style="flex: 1 1 360px; min-width: 0;">
      <img src="./images/route-layer-23.png" alt="Layer 23 的 DSA 风格 indexer 路由热图" style="display: block; width: 100%; height: auto; border: 1px solid rgba(127, 127, 127, 0.24); border-radius: 8px;" />
      <div style="margin-top: 0.4rem; text-align: center; color: #737373; font-family: 'Noto Serif SC', 'Source Han Serif SC', 'Songti SC', SimSun, serif; font-size: 0.84rem; line-height: 1.5;">(a) Layer 23</div>
    </div>
    <div style="flex: 1 1 360px; min-width: 0;">
      <img src="./images/route-layer-18.png" alt="Layer 18 的 DSA 风格 indexer 路由热图" style="display: block; width: 100%; height: auto; border: 1px solid rgba(127, 127, 127, 0.24); border-radius: 8px;" />
      <div style="margin-top: 0.4rem; text-align: center; color: #737373; font-family: 'Noto Serif SC', 'Source Han Serif SC', 'Songti SC', SimSun, serif; font-size: 0.84rem; line-height: 1.5;">(b) Layer 18</div>
    </div>
  </div>
  <figcaption style="margin: 0.8rem auto 0; max-width: 94%; color: #666; text-align: left; font-family: 'Noto Serif SC', 'Source Han Serif SC', 'Songti SC', SimSun, serif; font-size: 0.9rem; line-height: 1.75; letter-spacing: 0.01em;">
    <strong style="font-weight: 600; color: #4f4f4f;">图 1｜DSA 风格 indexer 的路由模式。</strong>在 dense Llama-3.2-3B 中接入 DSA 风格 indexer，先使用约 1B tokens warm up indexer，再使用约 16B tokens 继续训练。左、右分别展示 Layer 23 与 Layer 18 的路由结果：红色表示固定保留的局部窗口或 attention sink；黄色表示 indexer 预测与 oracle 重合；青色表示仅被 indexer 选中，即误选；绿色表示仅被 oracle 选中，即漏选；蓝色表示未路由区域。
  </figcaption>
</figure>

这两层的具体误选与漏选位置并不完全相同，但整体结构非常接近：相邻 queries 往往沿着连续区域选择相似的 token，不同 layers 也保留了大体一致的路由骨架。这个现象还不能直接证明哪些计算一定可以删除，却给出了后续工作的共同出发点——既然 index pattern 在 token、head、layer 与 query 等维度上都可能重复，就可以逐一追问：哪些分数必须精确计算，哪些搜索可以被粗筛、共享或复用？

### Token axis：并非所有 token 都值得精确打分

[HISA：Efficient Hierarchical Indexing for Fine-Grained Sparse Attention](https://arxiv.org/abs/2603.28458) 的观察是，重要 token 往往不是完全无结构地散落在整个历史中。相邻 token 具有一定的局部一致性，因此可以先在较粗的 block 粒度判断哪些区域可能重要，再进入少量入选 blocks 内做原始 token-level DSA scoring。

![HISA 的块级粗筛与 token 级精排](./images/hisa.png)

这张图可以从上往下读：完整 indexing keys 先被 pooling 成 $L/B$ 个 block representatives，query 只选 top-$m$ blocks；这些块随后被展开，第二次打分才选出最终 top-$k$ tokens。蓝色结果仍是 token 索引，因此 HISA 改变的是候选生成过程，没有改变 Sparse MLA 的输入接口。

它把一次平坦搜索：

$$ N\text{ 个 token 的精确扫描} $$

改成：

$$ \frac{N}{B}\text{ 个 block 的粗筛}+C\text{ 个候选 token 的精筛},\qquad C\ll N $$

可以把它理解为在图书馆找一句话：原始 indexer 会逐页检查整座图书馆；HISA 先根据每个书架的摘要排除大部分书架，再逐页检查剩余部分。这里最关键的技巧不是泛泛的“改成 block attention”，而是**让 block mean 只承担召回候选的责任，最终排序仍回到原始 DSA token score**。具体地说，HISA 先把 indexing keys 按固定大小 $B$ 分块并求均值，用所有 indexer heads 对 $L/B$ 个 block representatives 打分；选出 $m$ 个 blocks 后，再展开其中最多 $mB$ 个原始 token，并使用与 DSA 完全相同的 score 做 token-level top-$k$。因此它最终交给 Sparse MLA 的仍是细粒度 token indices，而不是整块 mask。

这个技巧的好处是粗筛可以很便宜，精筛又不会浪费整个 block 内的预算；论文实现还强制保留首块与最后一个合法块，用确定性规则保护 attention sink、局部上下文和 packed-sequence 边界。它的不可逆风险也很清楚：若 block mean 把一个“块内只有单个强信号”的区域平均掉，第二阶段就再也看不到那个 token。HISA 把 $N_I$ 从完整前缀压到候选集，但它节省的每一次精确打分，都以第一级 block recall 为前提。

### Head axis：并非所有 indexer heads 都同样重要

DSA 使用多个 indexer heads，是因为不同低维子空间可能捕捉不同的相关性模式。但“拥有很多 heads”不等于“每个 query 都必须调用所有 heads”。[MISA：Mixture of Indexer Sparse Attention](https://arxiv.org/abs/2605.07363) 的核心观察是，对一个具体 query，真正决定 top-$k$ 的往往只是少数 heads。它先用 block-level key 摘要估计每个 head 的贡献：

![MISA 在 DSA Indexer 之前增加 head router](./images/misa.png)

图中的绿色路径是原 DSA indexer，橙色 Router 是 MISA 新增的轻量分支。Router 同时读取各个 indexer queries、对应 gates 与 pooled indexing keys，输出 active-head mask；后面的 Top-$k$ Selector 仍接收 token-level index scores。也就是说，MISA 没有把多头 indexer 换成单头，而是让多头计算变成 query-dependent conditional computation。

$$ E_{t,j}=\frac{1}{M}\sum_b\left|w^I_{t,j}\operatorname{ReLU}\!\left(q^I_{t,j}\cdot\widetilde k^I_b\right)\right| $$

这里最关键的技巧是：**head router 不能只看 gate $w_{t,j}$ 或 query norm，而必须让每个 head 先与历史的 block summaries 发生一次便宜交互。** 只看 query 只能说明某个 head 自身“声音大不大”，无法判断当前前缀里是否真的存在它擅长匹配的内容；$E_{t,j}$ 同时看 query、gate 和历史摘要，才是在估计这个 head 对本次检索是否有用。选出 top-$h$ 后，只有这些 heads 扫描完整 token 序列。更保守的 MISA$^\dagger$ 则先让少数 heads 召回较大的 $k'$ 候选集，再用全部 heads 在候选内恢复完整 DSA 排序。

它与 HISA 的区别非常清楚：

- HISA 保留全部 heads，但减少每个 head 精扫的 token 数；
- MISA 保留完整 token 前缀，但减少参与精扫的 heads 数。

因此 MISA 主要降低 $H_I$，而不是 $N_I$。以论文中的典型配置为例，DeepSeek-V3.2 可从 64 个 indexer heads 降到 8 个，但这 8 个 active heads 仍然需要读完整 key sequence；如果系统瓶颈来自 KV/index-key 带宽而非 head 算术，kernel FLOPs 的下降就未必能等比例转成端到端收益。

### Layer axis：相邻层不必每次重新找地址

[IndexCache：Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201) 观察到，相邻层产生的 top-$k$ token 集合通常有较高重合。于是它把层分成两类：少数 Full layers 正常计算 index，后续 Shared layers 跳过 indexer，直接复用最近一个前置 Full layer 的 token positions。这里复用的只是“去哪里读”，不是“读到之后给多少权重”；Shared layer 仍然使用自己的 query、key 和 value，在复用的支持集上重新计算本层 attention。若模型原本有 $L$ 层都运行 indexer，而只有 $M$ 个 anchor layers 真正生成 route，那么 $U_L$ 就从 $L$ 降到 $M$。

![IndexCache 的跨层 top-k 重合与共享分组](./images/indexcache.png)

IndexCache 原文没有单独画一张 F/S 执行框图，这张热图是理解其结构最关键的图。横纵轴都是层，颜色表示两层 top-$k$ 集合的 overlap，红框则是 loss-based greedy search 得到的共享组。值得注意的是，红框没有机械地沿最亮的局部区域分段：这正好说明高 overlap 只是复用的动机，最终哪些层成为 Full anchor 仍由端到端 loss 决定。

我认为 IndexCache 最值得讲清楚的技巧不是 reuse 本身，而是**作者如何决定哪些层可以被删掉**。一个看起来顺手的方案，是先计算相邻层 index、attention output 或 score vector 的 cosine similarity，再把最相似的层合并；论文确实尝试过相似度代理，但最终 training-free 方案使用的是逐层试删的 greedy loss search。它从全 Full 模式开始，每一步把每个候选层暂时改成 Shared，分别在固定校准集上跑完整模型，然后真正删除使验证损失最小的那个层：

$$ \ell^*=\arg\min_{\ell\in\mathcal R}\operatorname{EvalLoss}\!\left(M,\mathcal D,\mathbf c\vert_{c_\ell\rightarrow S}\right) $$

这里 $M$ 是冻结模型，$\mathcal D$ 是校准数据，$\mathcal R$ 是仍可删除的层。删除一层后，下一轮必须在新的共享结构上重新评估其他候选，因为多个删除动作之间会相互影响。用一个四层例子理解：若试删第 2、3、4 层后的损失分别为 1.03、1.01、1.05，第一步删第 3 层；第二步不能沿用旧分数，而要在“第 3 层已经共享”的模型上重新测试第 2、4 层。

为什么不用 cosine similarity？文中的解释是，它只衡量方向相似，对幅值和少量高分 token 不敏感；而 top-$k$ index 恰恰由 token-level score 的相对幅值和排序边界决定。两个向量可以具有很高 cosine similarity，却在第 $k$ 名附近交换几个幅值接近的 token；这些 token 又可能对后续任务非常关键。更根本地说，局部 index 相似并不能告诉我们误差经过剩余 Transformer 后会放大还是抵消。论文中相似度代理甚至弱于均匀共享，而 loss search 虽然昂贵，却直接测量最终模型分布是否被破坏。training-aware 版本则走另一条路：用一个 Full indexer 同时蒸馏其后多个层的 attention teachers，让均匀共享在训练后变得可用。

### Query axis：相邻 queries 不必分别扫描同一前缀

[PIVOT：Efficient Query-Group Indexing for Token-Level Sparse Attention](https://arxiv.org/abs/2607.24593) 把视线转向此前较少被处理的 query axis。它发现相邻 queries 的 top-$k$ 高度重叠：在论文的观测中，相邻 query 的共享比例约为 0.8–0.9；即使把四个 queries 放在一组，它们 top-$k$ 的并集通常也只比 $k$ 大一点，而远小于最坏情况 $4k$。于是 PIVOT 把一组 $g$ 个 queries 聚合成一个 proxy query，只做一次 full-prefix scan：

![PIVOT 用一次 proxy scan 服务一组 queries](./images/pivot.png)

图左的 DSA 为 $q_1$ 到 $q_4$ 各自执行一次 full scan；图右把 prefill 的相邻 queries 或同一次 MTP decode 的 queries 组成一组，只让 proxy query 扫描完整前缀。上半部分 Refine 对 proxy top-$c$ 逐 query 重排，下半部分 Reuse 直接共享 proxy top-$k$，两者的速度—精度差别在图中是一目了然的。

$$ \bar q_j=\operatorname{Mean}_{t\in G}q^I_{t,j} $$

随后有两种处理方式：

- PIVOT-Reuse：整组直接共享 proxy 的 top-$k$；
- PIVOT-Refine：proxy 先取 top-$c$ 候选，再让每个 query 使用原始 DSA score 在候选内重排。

PIVOT 最关键的技巧是**把“共享扫描”和“共享最终选择”分开**。论文的消融显示，mean pooling 明显优于直接拿组内第一个或最后一个 query 作为 proxy；均值更适合表达一组 query 的共同需求。PIVOT-Reuse 为最高速度直接共享 proxy top-$k$，而 PIVOT-Refine 默认先取约 $c=2k$ 的共享候选，再用每个 query 的原始多头 DSA indexer 在候选中重打分。这使 proxy 只负责高召回，query-specific score 负责最终排序，和 HISA 的 coarse-to-fine 精神相似，但共享发生在 query axis。于是组复杂度从 $O(gL)$ 变成 Reuse 的 $O(L)$，或 Refine 的 $O(L+gc)$。

一个极简例子是：四个 queries 分别需要 $\{1,4\}$、$\{1,5\}$、$\{1,4\}$、$\{1,6\}$。虽然它们一共提出了八次选择，实际并集只有四个 token；proxy 先召回这四个候选，Refine 再恢复每个 query 自己的两个结果，就没有必要把完整前缀扫描四次。PIVOT 的贡献因此不只是增加一条优化轴，它还揭示了一个更普遍的规律：**indexer 的输出不是四个轴彼此独立的随机张量，而是在 token、head、layer 和 query 上同时具有结构。**

### 

## 重新设计 Indexer Architecture

Pattern 路线问的是“原来的分数中，哪些可以不算”；Architecture 路线则进一步追问“我们为什么一定要用一个低维 query，平坦地扫描所有 key，再做 top-$k$”。一旦提出后一个问题，indexer 就不再只是 attention 前面的一个小 MLP，而可以被重新理解为一个检索系统：它需要定义候选表示、搜索算法、监督信号、复用范围和硬件执行方式。下面这些工作看起来差异很大，但都在修改这套检索系统的某个基础接口。

### Louver：把 Top-k 改写成范围查询

[Sparse Attention as a Range Searching Problem（Louver）](https://arxiv.org/abs/2605.06763) 是我在这组工作里看到的一个很不同的切入点：它不直接问“分数最高的 $k$ 个 key 是谁”，而是问：

![Louver 的阈值 oracle、几何索引、buffer 与稀疏 attention 数据流](./images/louver.png)

图中 query 一路进入 Louver Index，另一路让 Threshold Oracle 从样本 score 中估计 $\tau$；索引返回候选 $K^*,V^*$ 后再做 attention。新产生的 key 先进入 dense buffer，buffer 满后异步更新几何索引。这里值得关注的是，Louver 不只提出一个离线搜索算法，还补齐了自回归 KV cache 持续增长时的在线更新闭环。

$$ \langle q,k\rangle\ge\tau $$

的所有 keys 是谁。

它把 key 分成小簇，并为每个簇保存中心 $c$ 与半径 $\rho$。根据 Cauchy–Schwarz，不同 key 的最大可能得分受到下面的上界控制：

$$ \langle q,k\rangle\le \langle q,c\rangle+\rho\|q\|_2 $$

如果一个簇的上界都低于阈值 $\tau$，整个簇就可以被安全排除；只有剩余候选需要真实点积。最关键的技巧是，Louver 没有简单地给每个子空间拍脑袋分配一个 $\tau/S$，而是用 Threshold Algorithm 同步扫描多个子空间的簇上界。设第 $s$ 个子空间在当前扫描深度 $d$ 的下一簇上界为 $f_{s,\sigma_s(d)}$，那么所有尚未出现的 key 的全维内积都不会超过

$$ U(d)=\sum_s f_{s,\sigma_s(d)} $$

第一次出现 $U(d)<\tau$ 时就可以安全停止；此前出现过的簇成员再做完整精确点积，剔除 false positives。这个同步停止条件很重要，因为一个相关 key 可能在某个子空间贡献很低、在另一个子空间贡献很高，逐子空间独立剪枝容易产生 false negative。Louver 因而不是从某个 score 轴做复用，而是改变了搜索问题的数学接口：从固定预算 top-$k$ 变为带几何上界证书的 threshold range search。它的优点是过滤正确性可以相对于给定阈值得到保证；局限也恰好在阈值上——搜索器完整返回所有超过 $\tau$ 的 key，并不自动保证 $\tau$ 本身保留了足够的 attention mass。

### MiniMax Sparse Attention：让 Indexer 成为 GQA 的原生分支

[MiniMax Sparse Attention（MSA）](https://arxiv.org/abs/2606.13392) 没有直接沿用 DSA 的多头 token top-$k$ indexer，而是为每个 GQA group 配置一个低维 index query head，并在所有 groups 之间共享 index key。它先计算便宜的 token score，再对每个 128-token block 做 max pooling，最后选择 16 个 blocks；Main Branch 仍使用标准 GQA query，在选中的 2,048 个 token 上计算精确 softmax attention。这套设计中，indexer 的粒度由三个约束共同决定：

![MiniMax Sparse Attention 的 Index Branch 与 Main Branch](./images/minimax-sparse-attention.png)

图左清楚地分开了两条路径：绿色 Index Branch 用 $Q_{idx}K_{idx}^{\top}$、block max pooling 和 Top-$k$ 只产生 KV block 地址；蓝色 Main Branch 保留标准 $Q,K,V$，在选中 blocks 上执行精确 sparse softmax。图右则说明不同 GQA groups 可以选择不同的远程 blocks，而同组 query heads 共享一张访问 mask。

1. per-GQA-group 选择保留不同 KV groups 的检索差异；
2. block-level 输出换取连续 KV 访问和 GPU 友好性；
3. 低维、共享 index key 把全扫描的常数压到足够小。

这里真正困难、也最关键的技巧是**如何训练一个带不可导 top-$k$ 的纯 selector**。LM loss 可以更新被选中的 Main Branch，却不能直接告诉 indexer“哪个未选 block 本应排得更高”。MSA 先经历一段 full-attention warmup，让随机 indexer 在不破坏主干的情况下观察完整 teacher；随后用同一 GQA group 内多个 Main heads 的平均 attention distribution 作为 KL teacher，并对 hidden state 与 teacher 使用 stop-gradient，让辅助 KL 只更新 index projections。这样可以避免 backbone 为了迎合小 indexer 而主动把主 attention 变简单。论文消融中，不 detach 会出现 LM loss 停滞和梯度尖峰，说明这不是训练配方里的装饰项。

MSA 仍然包含二次的 Index Branch，因此不是从渐近复杂度上消灭 full scan；它靠低维、少 head、共享 index key 和规则 block access 把常数压低。它代表的是另一种 architecture 思路：**与其事后优化一个昂贵 token indexer，不如在预训练阶段就把 indexer 的接口、监督和 kernel 一起设计成硬件可执行的形式。**

### 

### CLSA：让共享路由成为模型结构的一部分

IndexCache 是在普通逐层 Transformer 上寻找哪些 index 可以复用；[You Only Index Once：Cross-Layer Sparse Attention with Shared Routing（CLSA）](https://arxiv.org/abs/2606.06467) 则走得更远。它建立在 YOCO 的共享 KV 架构上，让多个 cross-decoder layers 本来就读取同一份 KV cache，再为这些层只生成一次公共 token route。每个 cross layer 仍用自己的 query 重新计算被选 token 的 attention 权重，但“去哪里读”由一个公共 indexer 决定。

![CLSA 在共享 KV 架构上只生成一次 token route](./images/clsa.png)

图的下半部分是 self-decoder，它只生成一次 Full KV；左侧公共 Top-$k$ Indexer 也只运行一次，得到 Sparse KV 后供上半部分所有 cross-attention layers 复用。这里的 “Routing Once, Cache Once” 是成套约束：正因为 cross layers 读取同一份 KV，公共 positions 才具有清楚的共享对象。

它最关键的训练技巧是**不让公共 indexer 模仿任意一个单层，而是让它拟合所有 cross layers、所有 attention heads 的平均 dense-attention distribution**：

$$ \bar A=\frac{1}{L_cH_a}\sum_{l=1}^{L_c}\sum_{h=1}^{H_a}\operatorname{softmax}\!\left(Q^{(l,h)}K^{(h)\top}\right) $$

若两个层分别把主要概率放在 token 1 和 token 3，平均 teacher 会要求公共 route 同时覆盖二者，而不是偏向某一个 anchor layer。训练先冻结 backbone、只用 KL warm up indexer，再联合优化 LM loss 与蒸馏损失，避免随机共享 route 在一开始破坏整个 cross-decoder。这把 layer-axis reuse 从一个经验优化变成了 architecture invariant：不是观察相邻层看起来相似，于是尝试复用；而是先让这些层共享信息源，再训练一个代表共同需求的 route。收益更彻底，代价也更高——CLSA 不是现有 DSA 模型的 drop-in patch，而是需要共享 KV 的主干架构和大规模训练适配；平均 teacher 也可能稀释只对少数层至关重要的 token。

### SparDA：让 Indexer 提前一层工作

[SparDA：Sparse Decoupled Attention for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2606.04511) 处理的不是“索引结果是否重复”，而是“selection 为什么必须挡在 attention 前面”。它在第 $l$ 层增加 Forecast $F_l$，提前预测第 $l+1$ 层需要的 KV blocks：

![SparDA 将下一层选块与当前层执行解耦](./images/sparda.png)

图 (a) 的基线必须等当前层 $Q_l$ 出现后才能选块；图 (b) 中 $F_{l-1}$ 已经为第 $l$ 层准备 route；图 (c) 进一步把 $F_l$ 产生的下一层 route 与 CPU KV 预取放到当前层执行的旁路上。虚线跨层箭头正是 SparDA 的核心：Forecast 提前给地址，真实 query 仍负责 attention。

$$ \mathcal B_{l+1}=\mathcal B_{\mathrm{init}}\cup\mathcal B_{\mathrm{local}}\cup\operatorname{TopK}\!\left(F_l\widetilde K_{l+1}^{\top},k\right) $$

真正的第 $l+1$ 层 attention 仍由 $Q_{l+1}$ 完成。Forecast 只负责地址，因此系统可以在当前层计算期间提前从 CPU 取回下一层 KV。这里最关键的技巧是**Forecast 学的是下一层原始多头 query selector 的集合分布，而训练目标专门增加一个“其余 blocks”桶**：teacher top-$k$ 各自保留，其余所有概率质量合并成第 $k+1$ 项后再做 KL。若只监督 top-$k$ 内部排序，student 即使同时给大量错误 blocks 很高概率也不会被惩罚；“其余”桶让集合外 logits 也收到梯度，直接限制错误候选的总质量。

SparDA 重新设计的是 indexer 的时间位置：从“当前层先选、再搬、再算”，变成“上一层预测，选择与预取尽可能和当前计算重叠”。这也是为什么它的收益不能只用 FLOPs 衡量；减少 critical path 有时比减少总工作量更重要。不过它预测的是下一层访问地址，不是下一层 KV 内容，而且需要训练 Forecast 与专门的 CPU–GPU offload pipeline，不能当成 training-free 的调度技巧。

### 

## 参考工作

- DeepSeek-AI. DeepSeek Sparse Attention / DeepSeek-V3.2.
- HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention. https://arxiv.org/abs/2603.28458
- IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse. https://arxiv.org/abs/2603.12201
- MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference. https://arxiv.org/abs/2605.07363
- PIVOT: Efficient Query-Group Indexing for Token-Level Sparse Attention. https://arxiv.org/abs/2607.24593
- Sparse Attention as a Range Searching Problem: Towards an Inference-Efficient Index for KV Cache. https://arxiv.org/abs/2605.06763
- MiniMax Sparse Attention. https://arxiv.org/abs/2606.13392
- Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference. https://arxiv.org/abs/2602.07397
- You Only Index Once: Cross-Layer Sparse Attention with Shared Routing. https://arxiv.org/abs/2606.06467
- SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference. https://arxiv.org/abs/2606.04511

