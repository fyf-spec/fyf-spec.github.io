# Greedy

## 一、什么是 Greedy

**Greedy algorithm** 的核心思想是：每一步都选择当前看起来最好的选项，也就是局部最优选择。

它通常具有下面的形式：

1. 维护一个已经构造好的部分解。
2. 在所有可选操作中，选择当前代价最小、收益最大或最符合某个局部标准的操作。
3. 把这个选择加入部分解，并继续向前。

需要注意的是，**Greedy 本身不是正确性的证明**。一个算法看起来很自然地做了局部最优选择，并不代表它一定能得到全局最优解。对 Greedy 算法来说，最重要的部分通常不是算法描述，而是证明：

- 为什么当前这个局部选择一定是安全的？
- 为什么做完这个选择之后，问题仍然保留一个最优解？
- 为什么连续做这些选择，最后能得到全局最优？

常见证明方式：

- **Contradiction**：假设贪心失败，推出矛盾。
- **Exchange argument**：拿一个最优解来，如果它没有包含贪心选择，就用贪心选择替换其中某个部分，得到一个不差的最优解。
- **Invariant**：证明每一步之后，当前部分解仍然可以扩展成某个全局最优解。

图算法中也有很多 Greedy 思想：

- Dijkstra：每次取当前距离源点最近的未确定点。
- Prim：每次取跨出当前树的最小边。
- Kruskal：每次取当前最小且不会成环的边。

---

## 二、作业截止期调度 Homework Scheduling

### 2.1 Problem

- **Input**: $n$ 个作业，每个作业 $j$ 有：
  - processing time / size: $s_j$
  - deadline: $d_j$
- **Output**: 一个作业执行顺序，使得所有作业都尽量在自己的 deadline 前完成。

这里讨论的是一个可行性问题：

> 如果存在一种安排能让所有作业都按时完成，Greedy 能否找到这样的安排？

### 2.2 Greedy Rule

**Earliest Deadline First (EDF)**:

每次优先完成 deadline 最早的作业。

也就是将作业按照 $d_j$ 从小到大排序，然后依次执行。

### 2.3 Correctness Claim

**Claim**: 如果按照 deadline 从小到大的顺序执行仍然无法按时完成所有作业，那么不存在任何顺序可以按时完成所有作业。

### 2.4 Proof

将作业按照 deadline 排序：

$$
d_1 \le d_2 \le \cdots \le d_n
$$

假设 Greedy 第一次失败发生在作业 $i$。也就是说，前 $i-1$ 个作业都可以按时完成，但是第 $i$ 个作业完成时间超过了 $d_i$：

$$
\sum_{j=1}^{i} s_j > d_i
$$

由于排序后对所有 $j \le i$ 都有：

$$
d_j \le d_i
$$

所以前 $i$ 个作业都必须在时间 $d_i$ 之前完成。无论怎么重新排序，这 $i$ 个作业的总工作量仍然是：

$$
\sum_{j=1}^{i} s_j
$$

但这个总工作量已经超过了 $d_i$。因此，不存在任何 schedule 能让这 $i$ 个作业全部在 $d_i$ 前完成，也就不可能让所有作业都按时完成。

所以，如果 EDF 失败，则任意算法都失败。

### 2.5 Key Point

这个证明说明了一个典型 Greedy 正确性思路：

- Greedy 选择 deadline 最早的作业。
- 一旦 Greedy 在某个 deadline 处失败，就能找到一个不可突破的工作量下界。
- 这个下界与顺序无关，所以不是 Greedy 的问题，而是问题本身无解。

---

## 三、Minimum Spanning Tree

### 3.1 Spanning Tree

- **Input**: 一个连通无向图 $G=(V,E)$。
- **Output**: 一个 spanning tree，也就是一个边集 $T \subseteq E$，满足：
  - 包含所有顶点。
  - 图 $(V,T)$ 连通。
  - 不包含环。

如果 $|V|=n$，任意 spanning tree 都恰好有 $n-1$ 条边。

### 3.2 Minimum Spanning Tree

- **Input**: 一个连通无向图 $G=(V,E)$，每条边 $e$ 有权重 $w(e)$。
- **Output**: 一个 spanning tree $T$，使得总权重最小：

$$
\sum_{e \in T} w(e)
$$

MST 的应用场景：

- 用最小成本连接所有节点。
- 网络布线。
- 聚类和图结构压缩。

### 3.3 Negative Weight 是否影响 MST？

不影响。

最短路算法中，负权边会破坏 Dijkstra 的距离单调性。但 MST 不依赖“路径越走越长”这一性质。MST 只关心边的相对权重和生成树结构，即使有负权边，Prim 和 Kruskal 仍然可以正确工作。

---

## 四、Partial MST 与安全边

### 4.1 P-MST

为了证明 Prim 和 Kruskal，我们引入一个比“小 MST”更强的概念。

**Partial MST (P-MST)**:

一个边集 $T$ 是 P-MST，当且仅当存在某个完整的 MST $T^*$，使得：

$$
T \subseteq T^*
$$

也就是说，$T$ 不是随便一棵局部最小的树，而是某个全局最优 MST 的一部分。

### 4.2 Greedy Invariant

Prim 和 Kruskal 的共同证明框架：

> 每一步都维护当前边集 $T$ 是一个 P-MST。

初始时 $T=\varnothing$，显然是 P-MST。每次加入一条安全边 $e$，证明 $T \cup \{e\}$ 仍然是 P-MST。最后当 $T$ 有 $|V|-1$ 条边时，它就是完整的 MST。

### 4.3 Cut Property

设一个 cut 将顶点分成两部分 $(S,V-S)$。如果某条边 $e$ 是跨越这个 cut 的最小权重边，那么 $e$ 对 MST 是安全的。

证明思路是 exchange argument：

1. 设 $T^*$ 是一个 MST。
2. 如果 $e \in T^*$，则已经完成。
3. 如果 $e \notin T^*$，把 $e$ 加入 $T^*$，会形成一个环，因为对于$e = (u, v)$， 全局最小生成树中一定包含$u，v$两个点，一定存在一条从$u$到$v$的路径， 现在再加上 $e$， 则一定成环。
4. 又因为$u,v$分别在$S$和$V-S$中，这个环中一定存在另一条跨越同一个 cut 的边 $f$。
5. 因为 $e$ 是该 cut 上最小的边，所以：

$$
w(e) \le w(f)
$$

6. 用 $e$ 替换 $f$，得到：

$$
T' = T^* - \{f\} + \{e\}
$$

新的 $T'$ 仍然是一棵 spanning tree，且权重不大于 $T^*$。因此 $T'$ 也是 MST，并且包含 $e$。

---

## 五、Prim Algorithm

### 5.1 Intuition

Prim 的思想类似 Dijkstra 的 growing idea：

- 已经有一棵包含顶点集合 $S$ 的树。
- 每次从 $S$ 向外扩展一条最便宜的边。
- 把新的顶点加入树中。

Dijkstra 选择的是距离源点最近的点；Prim 选择的是连接当前树最便宜的边。

### 5.2 Algorithm

维护：

- $S$: 已经加入 MST 的顶点集合。
- $T$: 已经选中的边集合。
- $cost[v]$: 从 $S$ 到 $v$ 的最小边权。
- $pre[v]$: 使得 $cost[v]$ 最小的前驱点。

```text
Prim(G = (V, E)):
    choose an arbitrary start vertex s
    S = {s}
    T = empty set

    cost[s] = 0
    for each v != s:
        cost[v] = infinity

    for each edge (s, v):
        cost[v] = w(s, v)
        pre[v] = s

    while S != V:
        choose v not in S with minimum cost[v]
        S = S union {v}
        T = T union {(pre[v], v)}

        for each edge (v, u):
            if u not in S and w(v, u) < cost[u]:
                cost[u] = w(v, u)
                pre[u] = v

    return T
```

### 5.3 Correctness

当前 $T$ 是一个 P-MST。Prim 选择一条跨越 cut $(S,V-S)$ 的最小边：

$$
e=(a,v)
$$

其中 $a \in S$，$v \notin S$。

根据 cut property，跨越该 cut 的最小边是安全边。所以将 $e$ 加入 $T$ 后，$T \cup \{e\}$ 仍然可以扩展成一个完整 MST。

这说明 Prim 每一步都保持 P-MST invariant。最终 $T$ 包含 $|V|-1$ 条边，所以 $T$ 是 MST。

### 5.4 Running Time

Prim 的时间主要来自两类操作：

- 每条边可能触发一次 update。
- 每个顶点会被 pop-min 一次。

如果用 Fibonacci Heap：

$$
O(|E| + |V|\log |V|)
$$

如果用 binary heap，常见复杂度是：

$$
O(|E|\log |V|)
$$

如果用邻接矩阵并线性扫描最小 $cost[v]$：

$$
O(|V|^2)
$$

---

## 六、Kruskal Algorithm

### 6.1 Intuition

Kruskal 是另一种 Greedy：

- 不从某个点开始长出一棵树。
- 而是从全图最小的边开始选。
- 只要加入这条边不会形成环，就保留它。

最后得到的是一片森林逐渐合并成一棵生成树。

### 6.2 Algorithm

```text
Kruskal(G = (V, E)):
    T = empty set
    sort all edges E by nondecreasing weight

    for each edge (u, v) in sorted E:
        if adding (u, v) does not create a cycle:
            T = T union {(u, v)}

    return T
```

更实际的写法会用 Union-Find 来判断是否成环：

```text
Kruskal(G = (V, E)):
    T = empty set
    for each vertex v:
        MakeSet(v)

    sort all edges E by nondecreasing weight

    for each edge (u, v) in sorted E:
        if Find(u) != Find(v):
            T = T union {(u, v)}
            Union(u, v)

    return T
```

### 6.3 Why Find(u) == Find(v) means cycle?

在 Kruskal 当前选中的边集中，每个连通块是一棵树。

- 如果 $u$ 和 $v$ 已经在同一个连通块中，那么它们之间已经存在一条路径。
- 此时再加入边 $(u,v)$，就会让这条路径和新边形成一个 cycle。
- 如果 $u$ 和 $v$ 不在同一个连通块中，加入 $(u,v)$ 会把两棵树合并，不会成环。

### 6.4 Correctness

Kruskal 的正确性也可以用和 Prim 类似的 **P-MST invariant** 来证明。

**Invariant**: 在 Kruskal 的每一步之后，当前已经选出的边集 $T$ 都是一个 P-MST。也就是说，存在某个完整 MST $T^*$，满足：

$$
T \subseteq T^*
$$

初始时 $T=\varnothing$，显然成立。

现在假设当前 $T$ 已经是 P-MST，并且 Kruskal 下一步选择了当前最小的、连接两个不同连通块的边：

$$
e=(u,v)
$$

设 $C$ 是当前森林 $T$ 中包含 $u$ 的连通块。因为 $e$ 被 Kruskal 选中，所以 $u$ 和 $v$ 当前不在同一个连通块中，于是：

$$
u \in C,\quad v \in V-C
$$

因此 $e$ 跨越 cut $(C,V-C)$。

下面证明 $e$ 是这个 cut 上的最小边。任取一条也跨越 $(C,V-C)$ 的边 $g$。由于 $g$ 的两个端点分属当前 $T$ 的不同连通块，加入 $g$ 不会在 $T$ 中形成 cycle，所以 $g$ 也是 Kruskal 当前可以接受的边。如果存在 $w(g)<w(e)$，Kruskal 在按权重扫描时应该先选择 $g$，这与它选择 $e$ 矛盾。因此：

$$
w(e) \le w(g)
$$

所以 $e$ 是 cut $(C,V-C)$ 上的 light edge。

接下来用 exchange argument 证明加入 $e$ 之后仍然是 P-MST。

因为当前 $T$ 是 P-MST，所以存在一个 MST $T^*$ 包含 $T$。

- 如果 $e \in T^*$，那么 $T \cup \{e\} \subseteq T^*$，直接成立。
- 如果 $e \notin T^*$，把 $e$ 加入 $T^*$。由于 tree 中加入一条非树边一定形成唯一一个 cycle，这时 $T^*+\{e\}$ 中出现一个环。

这个环里一定存在另一条跨越同一个 cut $(C,V-C)$ 的边 $f$。原因是：$e$ 从 $C$ 走到 $V-C$，环要回到起点，必须再从 $V-C$ 跨回 $C$。并且 $f \notin T$，因为 $C$ 是当前 $T$ 的一个连通块，$T$ 中没有边从 $C$ 连到外面。

由于 $e$ 是这个 cut 上的最小边，所以：

$$
w(e) \le w(f)
$$

用 $e$ 替换 $f$：

$$
T' = T^* - \{f\} + \{e\}
$$

得到的新图 $T'$ 仍然是一棵 spanning tree，并且总权重不超过 $T^*$。由于 $T^*$ 已经是 MST，所以 $T'$ 也是 MST。

同时，因为 $f \notin T$，删除 $f$ 不会破坏当前已经选出的边集 $T$；再加入 $e$ 后有：

$$
T \cup \{e\} \subseteq T'
$$

所以 $T \cup \{e\}$ 仍然是 P-MST。

最终 Kruskal 选出 $|V|-1$ 条边，得到 MST。

### 6.5 Running Time

排序花费：

$$
O(|E|\log |E|)
$$

因为在简单图中 $|E| \le |V|^2$，所以：

$$
O(|E|\log |E|) = O(|E|\log |V|)
$$

每条边会做两次 `Find`，每次成功选边会做一次 `Union`。如果 Union-Find 足够快，整体瓶颈通常是排序。

---

## 七、Union-Find Set

### 7.1 Why Union-Find?

Kruskal 需要频繁回答：

> 两个顶点当前是否属于同一个连通块？

Union-Find 支持三个操作：

- `MakeSet(x)`: 创建只包含 $x$ 的集合。
- `Find(x)`: 返回 $x$ 所属集合的代表元。
- `Union(x, y)`: 合并 $x$ 和 $y$ 所在的集合。

在 Kruskal 中：

- 如果 `Find(u) == Find(v)`，加入 $(u,v)$ 会成环。
- 如果 `Find(u) != Find(v)`，可以选择 $(u,v)$，然后执行 `Union(u,v)`。

### 7.2 Tree Representation

每个集合用一棵树表示：

- 根节点是这个集合的 representative。
- 每个节点维护一个 parent 指针。
- `Find(x)` 沿着 parent 指针一直走到根。
- `Union(x,y)` 将一棵树的根接到另一棵树的根下面。

如果不加控制，树高可能达到 $O(n)$，那么 `Find` 最坏就是：

$$
O(n)
$$

---

## 八、Union by Rank

### 8.1 Idea

为了减少树高，我们希望：

> 把矮树合并到高树下面。

维护一个数组：

$$
rank[v]
$$

表示以 $v$ 为根的树的 rank。可以把 rank 理解为树高的上界。

Union 规则：

- 如果 $rank[u] > rank[v]$，把 $v$ 的根挂到 $u$ 的根下面。
- 如果 $rank[u] < rank[v]$，把 $u$ 的根挂到 $v$ 的根下面。
- 如果 $rank[u] = rank[v]$，任选一个作为新根，并让新根 rank 加一。

### 8.2 Rank Bound

要构造一棵 rank 为 $k$ 的树，至少需要两个 rank 为 $k-1$ 的树合并。因此至少需要：

$$
2^k
$$

个节点。

所以如果总共有 $n$ 个节点：

$$
2^k \le n
$$

于是最大 rank 满足：

$$
k \le \log n
$$

因此使用 union by rank 后：

- `Find`: $O(\log n)$
- `Union`: $O(1)$，不包含为了找到根而做的 `Find`

对于 Kruskal 来说，结合排序后仍然是：

$$
O(|E|\log |E|) = O(|E|\log |V|)
$$

---

## 九、Path Compression

### 9.1 Idea

Path Compression 的想法是：

> 既然 `Find(x)` 已经沿路找到了根，那么把这条路径上的所有节点都直接挂到根下面，方便以后查询。

```text
Find(x):
    if parent[x] != x:
        parent[x] = Find(parent[x])
    return parent[x]
```

这一步不会改变集合划分，只会降低未来 `Find` 的成本。

### 9.2 Rank after Path Compression

使用 path compression 后，rank 不再等于真实树高。

原因是：

- `Union` 时会更新 rank。
- `Find` 做路径压缩时会改变树高。
- 但路径压缩不会修改 rank。

因此：

$$
rank[v] \ge height(v)
$$

rank 仍然是高度的上界。

### 9.3 Important Properties

在 union by rank + path compression 下，仍然成立：

**Lemma 1**: 一个节点的 parent 的 rank 严格大于它自己的 rank。

原因：

- Union 时低 rank 挂到高 rank 下。
- rank 相同的两棵树合并时，新根 rank 加一。
- Path compression 只会让节点直接挂到更高 rank 的祖先下面。

**Lemma 2**: rank 为 $k$ 的根所在树，至少曾经需要 $2^k$ 个节点构造出来。

所以 exact rank 为 $k$ 的节点数最多是：

$$
\frac{n}{2^k}
$$

---

## 十、Path Compression 的均摊分析

课件中给出了一个 charging argument，证明一个较弱但很有用的结论：

> 任意 $m$ 次 `Find` 的总成本是

$$
O(m \log^* n + n \log^* n)
$$

当 $m$ 足够大时，可以看成每次 `Find` 的均摊成本为：

$$
O(\log^* n)
$$

这里 $\log^* n$ 表示 iterated logarithm，也就是不断取 $\log$ 直到小于等于 $1$ 所需的次数。它增长极慢。

### 10.1 Charging Cost to Vertices

一次 `Find` 的成本等于访问路径上的边数。把成本分成两部分：

- **Self payment**: 每次 `Find` 自己支付 $O(1)$。
- **Charged cost**: 路径上非根方向的成本 charge 给对应 child vertex。

所以 $m$ 次 `Find` 的总成本可以写成：

$$
O(m) + \sum_v C(v)
$$

其中 $C(v)$ 是顶点 $v$ 被 charge 的次数。

### 10.2 Group Vertices by Rank

按照 rank 分组：

- Group 1: rank $0$
- Group 2: rank $1$
- Group 3: rank $2$ 到 $4$
- Group 4: rank $5$ 到 $16$
- Group 5: rank $17$ 到 $65536$

也就是每一组的上界增长为：

$$
k_i = 2^{k_{i-1}}
$$

因为最大 rank 至多是 $\log n$，所以组数至多是：

$$
O(\log^* n)
$$

### 10.3 Across Group Charging

如果一次 charge 中，child 和 parent 属于不同 rank group，这叫 **Across Group Charging (AGC)**。

在一条 `Find` 路径上，rank 严格递增，所以 group 也最多增加 $O(\log^* n)$ 次。

因此每次 `Find` 的 AGC 至多：

$$
O(\log^* n)
$$

$m$ 次 `Find` 的 AGC 总成本是：

$$
O(m \log^* n)
$$

### 10.4 Same Group Charging

如果 child 和 parent 属于同一个 rank group，这叫 **Same Group Charging (SGC)**。

考虑一个 rank 落在区间：

$$
[k+1, 2^k]
$$

中的顶点 $v$。

每次 $v$ 被同组 charge 之后，path compression 会让 $v$ 的 parent 变成更高 rank 的节点。因为同组内 rank 最多只有从 $k+1$ 到 $2^k$ 这些值，所以单个顶点在这个 group 内最多被同组 charge：

$$
O(2^k)
$$

另一方面，rank 在 $[k+1,2^k]$ 这一组中的顶点总数至多：

$$
\frac{n}{2^{k+1}} + \frac{n}{2^{k+2}} + \cdots + \frac{n}{2^{2^k}}
\le \frac{n}{2^k}
$$

所以这一整组的 SGC 总成本至多：

$$
\frac{n}{2^k} \cdot O(2^k) = O(n)
$$

一共有 $O(\log^* n)$ 个 group，因此所有 SGC 的总成本是：

$$
O(n \log^* n)
$$

### 10.5 Total Cost

综合三部分：

- Self payment: $O(m)$
- AGC: $O(m \log^* n)$
- SGC: $O(n \log^* n)$

得到：

$$
O(m \log^* n + n \log^* n)
$$

这就是 path compression 的一个均摊分析版本。

---

## 十一、更强的结论

课件最后提到更强的经典结果。

如果同时使用：

- union by rank / union by size
- path compression / path splitting / path halving

那么对 $m \ge n$ 次 `Find` 和 $n-1$ 次 `Union`，总时间可以达到：

$$
O(m \alpha(m,n))
$$

其中 $\alpha$ 是 inverse Ackermann function，增长比 $\log^* n$ 还慢。在实际规模的数据中，几乎可以认为它是一个非常小的常数。

因此 Kruskal 使用高级 Union-Find 后，非排序部分几乎是线性的：

$$
O(|E|\alpha(|V|))
$$

但如果仍然需要比较排序，整体通常还是：

$$
O(|E|\log |E|)
$$

---

## 十二、MST 复杂度小结

| Algorithm | Data Structure | Time Complexity |
| --- | --- | --- |
| Prim | Adjacency matrix | $O(|V|^2)$ |
| Prim | Binary heap | $O(|E|\log |V|)$ |
| Prim | Fibonacci heap | $O(|E| + |V|\log |V|)$ |
| Kruskal | Sorting + Union-Find | $O(|E|\log |E|)$ |
| Kruskal UF part | Rank + Path Compression | $O(|E|\alpha(|V|))$ |

更高级的 MST 算法：

- Karger-Klein-Tarjan: randomized $O(m)$。
- Chazelle: deterministic $O(m\alpha(n))$。
- Pettie-Ramachandran: 达到最优比较次数意义下的复杂度。

其中 $m=|E|$。

---

## 十三、本讲重点

1. Greedy 是“每一步做局部最优选择”，但必须证明局部选择是安全的。
2. 作业调度中，按照 earliest deadline first 排序可以判断是否能全部按时完成。
3. MST 的核心证明工具是 P-MST invariant 和 cut/exchange argument。
4. Prim 每次选择当前树向外扩展的最小边。
5. Kruskal 每次选择当前最小且不会成环的边。
6. Kruskal 的 cycle checking 依赖 Union-Find。
7. Union by rank 将树高控制到 $O(\log n)$。
8. Path compression 进一步把 `Find` 的均摊复杂度降到近似常数级。
