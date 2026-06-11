# Hall's Theorem and Network Flow Running Time

这一讲接着 maximum bipartite matching 和 Ford-Fulkerson method，讨论：

- Hall's Marriage Theorem；
- 如何用 Max-Flow-Min-Cut Theorem 证明 Hall's Theorem；
- 为什么 regular bipartite graph 一定有 perfect matching；
- Edmonds-Karp Algorithm 的运行时间；
- Dinic's Algorithm 的 high-level idea；
- Hopcroft-Karp-Karzanov Algorithm for maximum bipartite matching。

## Hall's Marriage Theorem

考虑一个 bipartite graph：

$$
G=(A,B,E)
$$

其中 $A$ 和 $B$ 是左右两边的点集，边 $(a,b)\in E$ 表示 $a\in A$ 可以和 $b\in B$ 匹配。

对任意子集 $S\subseteq A$，定义它的 neighborhood：

$$
N(S)=\{b\in B:\exists a\in S\text{ such that }(a,b)\in E\}
$$

也就是说，$N(S)$ 是 $S$ 中所有点能连到的 $B$ 侧点的集合。

> **Hall's Marriage Theorem**：存在一个 matching 能够匹配 $A$ 中所有点，当且仅当对任意 $S\subseteq A$，都有
>
> $$
> |S|\le |N(S)|
> $$

这里“匹配 $A$ 中所有点”也可以说成：存在一个 size 为 $|A|$ 的 matching。

直观理解：

> 如果 $A$ 中某一批点 $S$ 只能连到更少的点 $N(S)$，即 $|S|>|N(S)|$，那么这批点不可能全部被匹配。

这就是 Hall condition。

## Proof: Necessity

先证明比较直观的方向：

$$
\text{存在 size }|A|\text{ 的 matching}
\Longrightarrow
\forall S\subseteq A,\ |S|\le |N(S)|
$$

假设存在一个 matching，可以匹配 $A$ 中所有点。

任取 $S\subseteq A$。因为 $S$ 中每个点都被匹配，而且 matching 中不同的点不能匹配到同一个点，所以 $S$ 中的 $|S|$ 个点必须匹配到 $B$ 侧 $|S|$ 个不同的点。

这些被匹配到的点一定都在 $N(S)$ 里面。因此：

$$
|N(S)|\ge |S|
$$

所以 Hall condition 必须成立。

换句话说，如果存在某个 $S$ 满足

$$
|S|>|N(S)|
$$

那么 $S$ 中的点比它们能选择的对象还多，不可能把 $S$ 全部匹配，更不可能把整个 $A$ 全部匹配。

## Proof: Sufficiency via Max-Flow-Min-Cut

接下来证明反方向：

$$
\forall S\subseteq A,\ |S|\le |N(S)|
\Longrightarrow
\text{存在 size }|A|\text{ 的 matching}
$$

这个方向更难。我们用 max-flow/min-cut 来证明。

### Flow Network Construction

把 bipartite graph 转化成 flow network：

1. 加 source $s$ 和 sink $t$；
2. 对每个 $a\in A$，加边

$$
s\to a
$$

capacity 为 $1$；

3. 对每条 bipartite edge $(a,b)\in E$，加边

$$
a\to b
$$

capacity 为 $\infty$；

4. 对每个 $b\in B$，加边

$$
b\to t
$$

capacity 为 $1$。

这里 $a\to b$ 的 capacity 用 $\infty$，是为了让 cut 不愿意切断原图中的匹配边。真正限制 matching 大小的是 $s\to a$ 和 $b\to t$ 这些 capacity 为 $1$ 的边。

由于所有实际限制 matching 的边 capacity 都是整数，integrality theorem 保证 maximum flow 可以取 integral flow。因此：

$$
\text{maximum matching size}
=
\text{maximum flow value}
$$

### Suppose No Perfect Matching Exists

现在假设 Hall condition 成立，但不存在匹配所有 $A$ 中点的 matching。

令 maximum matching 的大小为 $M$。那么：

$$
M<|A|
$$

对应的 maximum flow value 也是 $M$。

由 Max-Flow-Min-Cut Theorem，存在一个 minimum cut $(L,R)$，其 capacity 也是 $M$：

$$
c(L,R)=M<|A|
$$

我们要从这个 cut 推出一个违反 Hall condition 的集合。

### Structure of the Cut

把 $A$ 和 $B$ 按照 cut 分成四块：

$$
L_A=L\cap A,\quad L_B=L\cap B
$$

$$
R_A=R\cap A,\quad R_B=R\cap B
$$

因为 $s\in L$，$t\in R$，cut 的 capacity 来自三类边：

1. 从 $s$ 到 $R_A$ 的边，每条 capacity 为 $1$，贡献 $|R_A|$；
2. 从 $L_B$ 到 $t$ 的边，每条 capacity 为 $1$，贡献 $|L_B|$；
3. 从 $L_A$ 到 $R_B$ 的 bipartite edges，每条 capacity 为 $\infty$。

由于这个 minimum cut 的 capacity 是

$$
M<|A|<\infty
$$

所以它不可能切到任何 $\infty$ capacity 的边。也就是说，不存在边从 $L_A$ 指向 $R_B$。

因此：

$$
N(L_A)\subseteq L_B
$$

也就是说，$L_A$ 中所有点的邻居都只能落在 $L_B$ 里。

同时，这个 cut 的 capacity 是：

$$
c(L,R)=|R_A|+|L_B|
$$

又因为 $A$ 被分成 $L_A$ 和 $R_A$：

$$
|A|=|L_A|+|R_A|
$$

现在由 $c(L,R)=M<|A|$ 得到：

$$
|R_A|+|L_B|<|L_A|+|R_A|
$$

消去 $|R_A|$：

$$
|L_B|<|L_A|
$$

而前面已经知道：

$$
N(L_A)\subseteq L_B
$$

所以：

$$
|N(L_A)|\le |L_B|<|L_A|
$$

这就找到了一个集合 $L_A\subseteq A$，满足：

$$
|N(L_A)|<|L_A|
$$

这与 Hall condition 矛盾。

因此假设不成立。只要 Hall condition 成立，就一定存在一个 size 为 $|A|$ 的 matching。

Hall's Marriage Theorem 证毕。

## The Three Cut Cases

也可以把 minimum cut 分成三种情况来看，它们对应上面的统一证明。

### Case 1

如果：

$$
L=\{s\},\quad R=A\cup B\cup\{t\}
$$

那么 cut 会切断所有 $s\to a$ 的边，所以：

$$
c(L,R)=|A|
$$

但我们在反证中假设 minimum cut 的容量是 $M<|A|$，所以 Case 1 不可能发生。

### Case 2

如果：

$$
L=\{s\}\cup A\cup B,\quad R=\{t\}
$$

那么 cut 会切断所有 $b\to t$ 的边，所以：

$$
c(L,R)=|B|
$$

如果 minimum cut 的容量 $M<|A|$，那么会得到：

$$
|B|<|A|
$$

但 Hall condition 对 $S=A$ 也必须成立：

$$
|A|\le |N(A)|\le |B|
$$

矛盾。所以 Case 2 也不可能发生。

### Case 3

一般情况下，四块

$$
L_A,\ L_B,\ R_A,\ R_B
$$

都可能非空。

此时 minimum cut 的容量为：

$$
M=|L_B|+|R_A|
$$

又因为：

$$
|A|=|L_A|+|R_A|
$$

若 $M<|A|$，则：

$$
|L_B|+|R_A|<|L_A|+|R_A|
$$

所以：

$$
|L_B|<|L_A|
$$

此外，不能有边从 $L_A$ 到 $R_B$，否则 cut 会包含一条 capacity 为 $\infty$ 的边。因此：

$$
N(L_A)\subseteq L_B
$$

于是：

$$
|N(L_A)|\le |L_B|<|L_A|
$$

这违反 Hall condition。

三个 case 都不可能，所以反证完成。

## Application: Regular Bipartite Graph Has a Perfect Matching

一个经典应用是：

> Prove that a regular bipartite graph always has a perfect matching.

现在可以直接用 Hall's Marriage Theorem 证明。

设 $G=(A,B,E)$ 是一个 $d$-regular bipartite graph，并且 $d>0$。也就是说，每个顶点的 degree 都是 $d$。

这里要求 $d>0$ 是为了排除退化情况：如果 $d=0$ 且图中有点，那么没有任何边，自然不可能存在匹配所有点的 perfect matching。

先注意到：

$$
d|A|=|E|=d|B|
$$

因为 $d>0$，所以：

$$
|A|=|B|
$$

接下来验证 Hall condition。

任取 $S\subseteq A$。从 $S$ 发出的边一共有：

$$
d|S|
$$

这些边全部连到 $N(S)$ 中的点。

另一方面，$N(S)$ 中每个点的 degree 最多是 $d$，所以它们最多能接收：

$$
d|N(S)|
$$

条来自 $A$ 侧的边。

因此：

$$
d|S|\le d|N(S)|
$$

当 $d>0$ 时，两边除以 $d$：

$$
|S|\le |N(S)|
$$

这正是 Hall condition。

所以存在一个 matching 能够匹配 $A$ 中所有点。又因为 $|A|=|B|$，这个 matching 同时匹配了两侧所有点，因此是 perfect matching。

结论：

> Every nonzero regular bipartite graph has a perfect matching.

## Residual Network Recap

在分析运行时间之前，先回顾 residual network。

给定 flow network $G=(V,E)$、capacity $c$ 和当前 flow $f$，residual network 记作：

$$
G_f=(V,E_f)
$$

其中 residual edge 有两类。

第一类是 forward edge。如果原图中有边 $(u,v)\in E$，并且这条边还没有满：

$$
f(u,v)<c(u,v)
$$

那么 residual network 中有边 $(u,v)$，它的 residual capacity 是：

$$
c_f(u,v)=c(u,v)-f(u,v)
$$

第二类是 backward edge。如果原图中有边 $(v,u)\in E$，并且这条边上已经有正 flow：

$$
f(v,u)>0
$$

那么 residual network 中有反向边 $(u,v)$，它的 residual capacity 是：

$$
c_f(u,v)=f(v,u)
$$

forward edge 表示“还能继续往前送多少 flow”，backward edge 表示“最多可以撤回多少已经送过的 flow”。

## Edmonds-Karp Algorithm

Ford-Fulkerson 更准确地说是一个 method，因为它没有规定每次怎么选择 augmenting path。

不同的 path 选择方式，会导致不同的时间复杂度：

- 如果随意选择 augmenting path，在整数 capacity 下复杂度可以是

$$
O(|E|\cdot f_{\max})
$$

- Edmonds-Karp Algorithm 固定每次选择 residual network 中边数最少的 augmenting path，也就是用 BFS 找 augmenting path。

### Algorithm

输入 flow network $G=(V,E)$、source $s$、sink $t$ 和 capacity $c$。

1. 初始化 $f(e)=0$ for every $e\in E$。
2. 构造 residual network $G_f$。
3. 当 $G_f$ 中还存在 $s$-$t$ path：
   - 用 BFS 找一条边数最少的 $s$-$t$ path $p$；
   - 令

$$
b=\min_{e\in p} c_f(e)
$$

   - 沿着 $p$ push $b$ units of flow；
   - 更新 flow $f$ 和 residual network $G_f$。
4. 当不存在 augmenting path 时，返回 $f$。

Edmonds-Karp 和普通 Ford-Fulkerson 的唯一差别是：

> 每次一定用 BFS 找 shortest augmenting path。

这里的 shortest 指的是边数最少，不是 capacity 或 weighted distance 最小。

### Why BFS Helps

在每次迭代的 residual network $G_f$ 中，令

$$
dist_f(u)
$$

表示从 $s$ 到 $u$ 的最短边数距离。

BFS 的作用是维护这些 distance。关键观察是：

> 随着 Edmonds-Karp 的运行，对任意点 $u$，$dist_f(u)$ 不会下降。

为什么？

每次沿 shortest augmenting path 更新 flow 后，residual network 中可能新增一些反向边。假设 path 上有一条边 $(u,v)$，更新后新增反向边 $(v,u)$。

因为这条边在 BFS shortest path 上，所以更新前：

$$
dist_f(v)=dist_f(u)+1
$$

新增的反向边方向是从 $v$ 回到 $u$，也就是从较远层指向较近层：

$$
dist_f(u)=dist_f(v)-1
$$

这种边不会创造一条更短的从 $s$ 到某个点的路径。因此所有点的 BFS distance 都是 non-decreasing。

这个 monotonicity 是 Edmonds-Karp 复杂度分析的核心。

### Critical Edges

在某次迭代中，设当前 flow 为 $f_i$，BFS 找到的 augmenting path 为 $p$。

令 bottleneck 为：

$$
b=\min_{e\in p}c_{f_i}(e)
$$

如果 path 上某条边 $(u,v)$ 满足：

$$
c_{f_i}(u,v)=b
$$

那么称 $(u,v)$ 是这次迭代的 critical edge。

critical edge 会在这次 augmentation 之后消失，因为它的 residual capacity 变成 $0$。

每次迭代至少有一条 critical edge。否则 bottleneck 就不存在。

### How Often Can One Edge Be Critical?

一条边 $(u,v)$ 可能消失后，在未来又重新出现。但如果它要再次成为 critical edge，中间必须发生一件事：

> 它的反向边 $(v,u)$ 必须先被某次 augmenting path 使用。

第一次 $(u,v)$ 成为 critical edge 时，因为它在 BFS shortest path 上：

$$
dist_i(v)=dist_i(u)+1
$$

后来如果 $(u,v)$ 要重新出现，某次 path 必须使用反向边 $(v,u)$。在那一次中，反向边也位于 BFS shortest path 上，因此：

$$
dist_j(u)=dist_j(v)+1
$$

又因为 distance 不下降：

$$
dist_j(v)\ge dist_i(v)
$$

所以：

$$
dist_j(u)
=
dist_j(v)+1
\ge
dist_i(v)+1
=
dist_i(u)+2
$$

也就是说：

> 同一条边两次成为 critical edge 之间，某个端点的 distance 至少增加 $2$。

而 distance 的可能取值只有：

$$
0,1,2,\dots,|V|,\infty
$$

所以每条边最多成为 critical edge $O(|V|)$ 次。

### Running Time

总结：

- 每次迭代至少产生一条 critical edge；
- 每条边最多成为 critical edge $O(|V|)$ 次；
- residual network 中边数是 $O(|E|)$；
- 所以迭代次数最多是

$$
O(|V|\cdot |E|)
$$

每次迭代需要一次 BFS，并更新 path 上的 flow，时间为：

$$
O(|E|)
$$

因此 Edmonds-Karp 的总时间复杂度是：

$$
O(|V|\cdot |E|^2)
$$

这个 bound 和 capacity 的数值大小无关，因此避免了 $O(|E|\cdot f_{\max})$ 对数值大小的依赖。即使 capacity 是 irrational，Edmonds-Karp 也会在 polynomial number of augmentations 后停止。

## Dinic's Algorithm

Edmonds-Karp 每轮只 push 一条 shortest augmenting path。一个自然的想法是：

> 能不能一次性处理掉所有 shortest $s$-$t$ paths？

Dinic's Algorithm 就是沿着这个方向改进。

### Level Graph

给定 residual network $G_f$，先用 BFS 计算每个点到 $s$ 的距离：

$$
level(u)=dist_f(u)
$$

然后构造 level graph $G_L^f$：

- 保留所有从 level $i$ 指向 level $i+1$ 的 residual edges；
- 删除其他 residual edges。

也就是说，level graph 只保留可能出现在 shortest $s$-$t$ paths 上的边。

### Blocking Flow

在 level graph 中，我们希望找到一个 blocking flow。

blocking flow 的含义是：

> push 一批 flow，使得 level graph 中每一条 $s$-$t$ path 都至少包含一条 saturated edge。

换句话说，blocking flow 不一定是 level graph 上的 maximum flow，但它足够“堵住”当前所有 shortest paths。

### Dinic's Algorithm Overview

1. 初始化 $f=0$。
2. 构造 residual network $G_f$。
3. 当 $t$ 仍然可以从 $s$ 到达：
   - 用 BFS 构造 level graph $G_L^f$；
   - 在 $G_L^f$ 中找到一个 blocking flow；
   - 把 blocking flow 加到当前 flow $f$ 上；
   - 更新 residual network $G_f$。
4. 当 $t$ 不可达时，返回 $f$。

### Finding a Blocking Flow

可以用 DFS 在 level graph 中找 blocking flow：

1. 从 $s$ 出发做 DFS。
2. 如果 DFS 到达 $t$，就沿着这条 path push bottleneck flow，并删除 path 上变成 saturated 的 critical edges。
3. 如果 DFS 到达某个 dead-end vertex $v$，也就是 $v$ 没有 outgoing edge 可以继续走，那么删除进入 $v$ 的边，因为这些边不可能再帮助形成 $s$-$t$ path。
4. 重复直到 level graph 中没有 $s$-$t$ path。

粗略分析：

- 每次搜索后至少删除一条边；
- 总共最多删除 $O(|E|)$ 条边；
- 每次搜索路径长度最多 $O(|V|)$。

所以在一般图中，一个 blocking flow 可以在

$$
O(|V|\cdot |E|)
$$

时间内找到。

### Why the Number of Phases Is Small

Dinic 的一次 BFS + blocking flow 称为一个 phase。

关键性质：

> 每个 phase 结束后，$dist_f(t)$ 严格增加。

原因是：当前 level graph 中包含了所有 shortest $s$-$t$ paths。blocking flow 会让每条这样的 path 都包含 saturated edge，因此下一轮 residual network 中已经没有同样长度的 $s$-$t$ path。

新的 $s$-$t$ path 要么使用某条新出现的反向边，要么使用原来 residual network 中不在 level graph 里的边。无论哪种情况，它的长度都会比原来的 shortest path 更长。

因此：

$$
dist_{new}(t)>dist_{old}(t)
$$

而 $dist(t)$ 最多从 $0$ 增加到 $|V|$，之后变成 $\infty$。所以 phase 数量最多是：

$$
O(|V|)
$$

### Running Time of Dinic's Algorithm

每个 phase：

- BFS 构造 level graph：$O(|E|)$；
- 找 blocking flow：$O(|V|\cdot |E|)$。

所以每个 phase 是：

$$
O(|V|\cdot |E|)
$$

phase 数量是 $O(|V|)$，因此总时间复杂度是：

$$
O(|V|^2\cdot |E|)
$$

和 Edmonds-Karp 相比：

$$
O(|V|^2\cdot |E|)
\quad\text{vs.}\quad
O(|V|\cdot |E|^2)
$$

哪个更好取决于图的稠密程度，但 Dinic 的思想更强：每个 phase 同时处理多条 shortest paths。

## Hopcroft-Karp-Karzanov Algorithm

对 maximum bipartite matching，还有一个更快的经典算法：

> Hopcroft-Karp-Karzanov Algorithm 可以在
>
> $$
> O(|E|\sqrt{|V|})
> $$
>
> 时间内找到 maximum bipartite matching。

它可以看成 Dinic's Algorithm 在 bipartite matching flow network 上的特例。

### Flow Network for Bipartite Matching

给定 bipartite graph：

$$
G=(A,B,E)
$$

构造 flow network：

- $s\to a$ for every $a\in A$，capacity 为 $1$；
- $a\to b$ for every $(a,b)\in E$，capacity 为 $1$；
- $b\to t$ for every $b\in B$，capacity 为 $1$。

所有 capacity 都是 $1$，所以 Dinic 输出的 flow 是 integral flow，可以直接转回 matching。

### Why This Special Case Is Faster

在这个 unit-capacity bipartite network 中，blocking flow 可以更快地找到。

直观原因：

- 每条边的 capacity 都是 $1$；
- 找到一条 augmenting path 后，这条 path 上的边都会被 saturate；
- 在 level graph 中，每条边最多被 DFS 访问和删除常数次。

因此每个 phase 找 blocking flow 的时间可以做到：

$$
O(|E|)
$$

而不是一般 Dinic 中的 $O(|V|\cdot |E|)$。

更具体地说，在 level graph 中反复做 DFS：

- 如果找到一条 $s$-$t$ path，就把这条 path 对应的 matching 翻转，并删除 path 上已经不能再用的边；
- 如果 DFS 走到 dead end，就回退并删除刚刚证明无用的边；
- 每条边最多被访问和删除常数次。

所以一个 blocking flow 的计算是线性的。

接下来需要 bound phase 数量。

如果算法在 $\sqrt{|V|}$ 个 phase 内结束，那么已经完成。

否则，经过 $\sqrt{|V|}$ 个 phase 后，所有剩余 augmenting paths 的长度都至少是：

$$
\sqrt{|V|}
$$

在二分图匹配的 residual network 中，剩余还可以增加的 matching 可以分解成若干条 vertex-disjoint augmenting paths。每条 path 长度至少 $\sqrt{|V|}$，而总顶点数只有 $|V|$，所以这样的 path 数量最多是：

$$
O(\sqrt{|V|})
$$

也就是说，剩余还能增加的 matching size 至多是 $O(\sqrt{|V|})$。之后每个 phase 至少增加 $1$ 个 matching，因此再过 $O(\sqrt{|V|})$ 个 phase 一定结束。

所以 phase 总数是：

$$
O(\sqrt{|V|})
$$

每个 phase 用 $O(|E|)$ 时间，因此总时间复杂度是：

$$
O(|E|\sqrt{|V|})
$$

## Other Max-Flow Algorithms

除了 Edmonds-Karp 和 Dinic，还有很多更快或更适合特定场景的 max-flow 算法。

一些经典结果包括：

- Malhotra-Kumar-Maheshwari algorithm：$O(|V|^3)$；
- Dinic with dynamic trees：$O(|V|\cdot |E|\log |V|)$；
- Push-relabel algorithm：经典 bound 为 $O(|V|^2|E|)$，后来有多种改进；
- Orlin's algorithm：$O(|V|\cdot |E|)$；
- 近年的 interior-point-method-based algorithms 可以达到接近线性的复杂度。

这些算法的细节不在这里展开。对本课程这一部分，最重要的是理解三条主线：

1. Ford-Fulkerson 的复杂度依赖于怎么选 augmenting path；
2. Edmonds-Karp 用 BFS 让 augmenting path 的层数具有 monotonicity；
3. Dinic 用 level graph 和 blocking flow 一次处理多条 shortest paths。

## Summary

本讲的主线是：

- Hall's Marriage Theorem 可以用 Max-Flow-Min-Cut Theorem 证明；
- regular bipartite graph 的 perfect matching 是 Hall's Theorem 的直接应用；
- Edmonds-Karp 用 BFS 实现 Ford-Fulkerson，复杂度为

$$
O(|V|\cdot |E|^2)
$$

- Dinic 每个 phase 处理所有 shortest augmenting paths，复杂度为

$$
O(|V|^2\cdot |E|)
$$

- Hopcroft-Karp-Karzanov 利用二分图匹配的特殊结构，复杂度为

$$
O(|E|\sqrt{|V|})
$$
