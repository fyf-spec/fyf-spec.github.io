# Network Flow: Correctness and Applications

这一讲接着 Ford-Fulkerson Algorithm 之后，讨论：

- 时间复杂度；
- 正确性，也就是 Max-Flow-Min-Cut Theorem；
- 整数定理 Integrality Theorem；
- 两个典型 application。

## Time Complexity

Ford-Fulkerson 的每一轮做三件事：

1. 在 residual network $G_f$ 中找一条 $s$-$t$ augmenting path $P$；
2. 找到这条路径上的 bottleneck capacity；
3. 按照 bottleneck 更新 flow 和 residual network。

如果用 DFS/BFS 找路径，那么每一轮的时间复杂度是：

$$
O(|E|)
$$

因为找路径、扫描路径、更新 residual network 都可以在线性时间内完成。

如果所有 capacity 都是整数，每一次 augmentation 都会让 $v(f)$ 至少增加 $1$。所以循环次数最多是 $f_{\max}$ 次，其中 $f_{\max}$ 是最大流的值。

因此，在整数 capacity 的情况下：

$$
T=O(|E|\cdot f_{\max})
$$

这个复杂度依赖于 $f_{\max}$ 的数值大小，所以它不是 strongly polynomial 的。换句话说，如果 capacity 很大，Ford-Fulkerson 可能跑很多轮。

关键点：

> Ford-Fulkerson 更准确地说是一个 method，而不是一个固定 algorithm。  
> 不同的 augmenting path 选择方式，会得到不同的实现和复杂度。

例如：

- 任意选择 augmenting path：复杂度可以达到 $O(|E|\cdot f_{\max})$；
- 每次选 residual network 中边数最少的 augmenting path，也就是 Edmonds-Karp Algorithm：

$$
O(|V|\cdot |E|^2)
$$

## Correctness

Ford-Fulkerson 的正确性依赖于 Max-Flow-Min-Cut Theorem。

> **Max-Flow-Min-Cut Theorem**：最大流的值等于最小割的容量。

这件事可以用两个 lemma 证明：

1. 任意 cut 都给出了任意 flow value 的上界；
2. Ford-Fulkerson 结束时，可以找到一个刚好等于当前 flow value 的 cut。

两个 lemma 合起来，就说明 Ford-Fulkerson 找到的 flow 已经达到某个 cut 的上界，因此不可能再变大。

### Cut and Notation

一个 $s$-$t$ cut 是对点集 $V$ 的划分：

$$
(L,R)
$$

满足：

$$
s\in L,\quad t\in R,\quad L\cup R=V,\quad L\cap R=\emptyset
$$

cut 的 capacity 是所有从 $L$ 指向 $R$ 的边的容量之和：

$$
c(L,R)=\sum_{u\in L,\ v\in R,\ (u,v)\in E} c(u,v)
$$

直观理解：

> cut 把 source 和 sink 分开。  
> 如果要从 $s$ 往 $t$ 送 flow，就一定要穿过这个 cut。

对任意点集 $A\subseteq V$，记：

$$
f_{\text{out}}(A)
$$

为从 $A$ 流到 $V\setminus A$ 的总 flow，记：

$$
f_{\text{in}}(A)
$$

为从 $V\setminus A$ 流回 $A$ 的总 flow。

对于 cut $(L,R)$，也就是：

$$
f_{\text{out}}(L)
=
\sum_{u\in L,\ v\in R,\ (u,v)\in E} f(u,v)
$$

和

$$
f_{\text{in}}(L)
=
\sum_{u\in R,\ v\in L,\ (u,v)\in E} f(u,v)
$$

flow 的 value 定义为 source 流出的总量：

$$
v(f)=f_{\text{out}}(\{s\})
$$

这里使用标准约定：source 没有流入，即 $f_{\text{in}}(s)=0$。

### Generalized Flow Conservation

先证明一个 claim：

> 对任意 feasible flow $f$ 和任意 $s$-$t$ cut $(L,R)$，
>
> $$
> v(f)=f_{\text{out}}(L)-f_{\text{in}}(L)
> $$

也就是说，flow value 等于穿过 cut 的净流量。

证明时，把 $L$ 中所有点的“流出量减流入量”加起来：

$$
\sum_{u\in L}
\left(
f_{\text{out}}(u)-f_{\text{in}}(u)
\right)
$$

其中 $f_{\text{out}}(u)$ 表示从顶点 $u$ 流出的总量，$f_{\text{in}}(u)$ 表示流入顶点 $u$ 的总量。

一方面，由 flow conservation，对于每个中间点 $u\in V\setminus\{s,t\}$：

$$
f_{\text{out}}(u)=f_{\text{in}}(u)
$$

又因为 $s\in L$ 且 $t\notin L$，所以 $L$ 里面除了 $s$ 以外的点净流量都是 $0$。因此：

$$
\sum_{u\in L}
\left(
f_{\text{out}}(u)-f_{\text{in}}(u)
\right)
=
f_{\text{out}}(s)-f_{\text{in}}(s)
+\sum_{u\in L\setminus\{s\}}0
=
v(f)
$$

另一方面，再从边的角度看同一个求和式：

- 如果一条边 $(u,v)$ 满足 $u,v\in L$，那么它会在 $f_{\text{out}}(u)$ 中贡献 $+f(u,v)$，又在 $f_{\text{in}}(v)$ 中贡献 $-f(u,v)$，两项 cancel。
- 如果一条边 $(u,v)$ 满足 $u\in L,\ v\in R$，它只贡献 $+f(u,v)$。
- 如果一条边 $(u,v)$ 满足 $u\in R,\ v\in L$，它只贡献 $-f(u,v)$。

所以：

$$
\sum_{u\in L}
\left(
f_{\text{out}}(u)-f_{\text{in}}(u)
\right)
=
f_{\text{out}}(L)-f_{\text{in}}(L)
$$

把两边合起来，得到：

$$
v(f)=f_{\text{out}}(L)-f_{\text{in}}(L)
$$

### Lemma 1: Every Cut Is an Upper Bound

> **Lemma 1.** 对任意 feasible flow $f$ 和任意 $s$-$t$ cut $(L,R)$，都有
>
> $$
> v(f)\le c(L,R)
> $$

证明：

由上面的 claim，

$$
v(f)=f_{\text{out}}(L)-f_{\text{in}}(L)
$$

因为 flow 非负，所以 $f_{\text{in}}(L)\ge 0$，于是：

$$
v(f)\le f_{\text{out}}(L)
$$

而
$$
f_{\text{out}}(L)=\sum_{u\in L,\ v\in R,\ (u,v)\in E} f(u,v)
\le
\sum_{u\in L,\ v\in R,\ (u,v)\in E} c(u,v)=c(L,R)
$$

所以：

$$
v(f)\le c(L,R)
$$

这说明：

> 任意 cut 的 capacity 都是任意 flow value 的上界。

因此：

$$
\max_f v(f)\le \min_{(L,R)} c(L,R)
$$

### Lemma 2: Ford-Fulkerson Finds a Tight Cut

> **Lemma 2.** 设 $f$ 是 Ford-Fulkerson 算法输出的 flow。那么存在一个 cut $(L,R)$，使得
>
> $$
> v(f)=c(L,R)
> $$

证明：

Ford-Fulkerson 停止时，residual network $G_f$ 中已经不存在 $s$-$t$ augmenting path。

令 $L$ 为在 $G_f$ 中从 $s$ 可以到达的所有点：

$$
L=\{u\in V:\text{there is a path from }s\text{ to }u\text{ in }G_f\}
$$

令

$$
R=V\setminus L
$$

因为没有 $s$-$t$ augmenting path，所以 $t\notin L$。又因为 $s\in L$，所以 $(L,R)$ 是一个合法的 $s$-$t$ cut。

接下来证明两个 claim。

**Claim A.** $f_{\text{out}}(L)=c(L,R)$。

如果不成立，那么存在一条边 $(u,v)$，其中 $u\in L,\ v\in R$，并且

$$
f(u,v)<c(u,v)
$$

于是它的 residual capacity 为正：

$$
c_f(u,v)=c(u,v)-f(u,v)>0
$$

所以 residual network $G_f$ 中存在 forward edge $(u,v)$。由于 $u\in L$，$u$ 可以从 $s$ 到达；再沿着 $(u,v)$，$v$ 也可以从 $s$ 到达。这与 $v\in R$ 矛盾。

因此所有从 $L$ 指向 $R$ 的边都 saturated，即：

$$
f_{\text{out}}(L)
=
\sum_{u\in L,\ v\in R} f(u,v)
=
\sum_{u\in L,\ v\in R} c(u,v)
=
c(L,R)
$$

**Claim B.** $f_{\text{in}}(L)=0$。

如果不成立，那么存在一条边 $(v,u)$，其中 $v\in R,\ u\in L$，并且

$$
f(v,u)>0
$$

那么 residual network $G_f$ 中存在 backward edge $(u,v)$，其 residual capacity 为：

$$
c_f(u,v)=f(v,u)>0
$$

由于 $u\in L$，$u$ 可以从 $s$ 到达；再沿着 backward edge $(u,v)$，$v$ 也可以从 $s$ 到达。这与 $v\in R$ 矛盾。

因此所有从 $R$ 流回 $L$ 的 flow 都是 $0$，也就是：

$$
f_{\text{in}}(L)=0
$$

由 generalized flow conservation：

$$
v(f)=f_{\text{out}}(L)-f_{\text{in}}(L)
$$

再代入 Claim A 和 Claim B：

$$
v(f)=c(L,R)-0=c(L,R)
$$

Lemma 2 证毕。

### Putting the Two Lemmas Together

现在把两个 lemma 放在一起：

- Lemma 1 说明，对任意 flow $f'$ 和任意 cut $(L,R)$，都有

$$
v(f')\le c(L,R)
$$

- Lemma 2 说明，Ford-Fulkerson 输出的 flow $f$ 对某个 cut $(L,R)$ 满足

$$
v(f)=c(L,R)
$$

所以没有任何 feasible flow $f'$ 可以满足

$$
v(f')>v(f)
$$

因此 Ford-Fulkerson 输出的 $f$ 是 maximum flow。与此同时，Lemma 2 中构造出的 cut $(L,R)$ 的容量等于这个 maximum flow value，所以它也是 minimum cut。

于是：

$$
\max_f v(f)=\min_{(L,R)} c(L,R)
$$

这就是 Max-Flow-Min-Cut Theorem，也就是 Ford-Fulkerson 正确性的来源。

等价地，对一个 feasible flow $f$，下面三件事等价：

1. $f$ 是 maximum flow；
2. residual network $G_f$ 中不存在 $s$-$t$ augmenting path；
3. 存在一个 cut $(L,R)$，使得 $v(f)=c(L,R)$。

### How to Find a Minimum Cut

Max-Flow-Min-Cut Theorem 不只证明了 Ford-Fulkerson 的正确性，也给出了一个找 minimum cut 的方法。

给定一个带权有向图：

$$
G=(V,E,w)
$$

以及两个点 $s,t\in V$。如果想找 minimum $s$-$t$ cut，可以这样做：

1. 把每条边的 weight 当成 flow network 中的 capacity：

$$
c(u,v)=w(u,v)
\quad
\text{for every }(u,v)\in E
$$

2. 在这个 flow network 上运行 max-flow algorithm，得到一个 maximum flow $f$。

3. 根据这个 maximum flow 构造 residual network $G_f$。

4. 令 $L$ 为在 $G_f$ 中从 $s$ 可以到达的所有点：

$$
L=\{u\in V:\text{there is a path from }s\text{ to }u\text{ in }G_f\}
$$

5. 令

$$
R=V\setminus L
$$

6. 返回 cut $(L,R)$。

为什么这个 cut 一定是 minimum cut？

因为 max-flow 已经结束，所以 $G_f$ 中没有 $s$-$t$ augmenting path，因此 $t\notin L$，$(L,R)$ 是合法的 $s$-$t$ cut。

由 Lemma 2：

$$
v(f)=c(L,R)
$$

而 $f$ 是 maximum flow，所以 $v(f)$ 是最大流的值。根据 Max-Flow-Min-Cut Theorem，这个值等于 minimum cut 的容量。因此：

$$
c(L,R)=\min_{(A,B)}c(A,B)
$$

所以用 residual network 中从 $s$ 可达的点集 $L$ 构造出的 cut，就是一个 minimum cut。

## Integrality Theorem

整数定理是 max-flow 里非常重要的结论。

> **Integrality Theorem**：如果所有 capacity $c(e)$ 都是整数，那么一定存在一个 maximum flow $f$，使得每条边上的 flow $f(e)$ 都是整数。

更具体地说，如果 Ford-Fulkerson 从 zero flow 开始，并且所有 capacity 都是整数，那么它找到的每一个 intermediate flow 都是整数 flow。

证明思路：

1. 初始时，所有边 $f(e)=0$，显然是整数；
2. 如果当前 flow 是整数，那么 residual capacity 也是整数：

$$
c_f(u,v)=c(u,v)-f(u,v)
$$

或者：

$$
c_f(v,u)=f(u,v)
$$

3. augmenting path 上的 bottleneck

$$
b=\min_{e\in P}c_f(e)
$$

也是整数；

4. 每次更新都是加上或减去整数 $b$，所以更新后的 flow 仍然是整数。

因此，在整数 capacity 的情况下，Ford-Fulkerson 会保持整数性；结合正确性，它最终可以找到一个 integral maximum flow。

关键结论：

> 只要 capacity 是整数，最大流不仅 value 是整数，而且可以用每条边都是整数的 flow 达到。

这个结论在很多离散问题建模中非常有用，因为它允许我们把“选或者不选”“赢几场比赛”“匹配几个点”这类离散问题转化成 max-flow。

## Application 1: Baseball Elimination

问题：有若干支队伍，已知每支队伍已经赢了多少场，以及队伍之间还剩多少场比赛。问某支队伍 $D$ 是否还有机会成为冠军。

考虑下面这个例子：

| Team | Current Wins |
| --- | ---: |
| A | 40 |
| B | 38 |
| C | 37 |
| D | 29 |

假设 $D$ 接下来和其他队伍的比赛全部获胜，那么 $D$ 最终最多有：

$$
29+12=41
$$

所以其他队伍最多还能赢：

| Team | Max Additional Wins |
| --- | ---: |
| A | $41-40=1$ |
| B | $41-38=3$ |
| C | $41-37=4$ |

现在只需要安排 $A,B,C$ 之间剩下的比赛结果，保证没有人超过 $41$。

### Flow Construction

建立一个 flow network：

1. source $s$ 连接到每一个 game node。

   比如 $A$ 和 $B$ 之间还有 $r_{AB}$ 场比赛，就建立一个 node $g_{AB}$，并加边：

$$
s\to g_{AB}
$$

capacity 是：

$$
r_{AB}
$$

2. 每个 game node 连接到参与比赛的两支队伍。

$$
g_{AB}\to A,\quad g_{AB}\to B
$$

capacity 可以设成 $\infty$，或者足够大的数。

3. 每个 team node 连接到 sink $t$。

   对于队伍 $A$，capacity 是它还能额外赢的最多场数：

$$
A\to t:\quad 41-40=1
$$

类似地：

$$
B\to t:\quad 3
$$

$$
C\to t:\quad 4
$$

### Decision Rule

令：

$$
R=\sum_{\{i,j\}:i,j\ne D} r_{ij}
$$

也就是不包含 $D$ 的所有剩余比赛总数。

如果 max flow 的值等于 $R$，说明所有这些比赛都能被合法分配胜场，并且没有队伍超过 $D$ 的最高胜场数。

因此：

> $D$ 还有机会成为冠军  
> $\Longleftrightarrow$  
> 这个 flow network 的 maximum flow 能 saturate 所有从 source 出发的 game edges。

为什么需要整数定理？

因为比赛结果必须是整数场胜利。max-flow 的 integrality theorem 保证：如果 capacity 是整数，那么存在 integral max flow。这样每个 game node 流向某个 team 的整数 flow 就可以解释成“这几场比赛由这支队伍赢”。

## Application 2: Maximum Bipartite Matching

给定一个 bipartite graph：

$$
G=(A,B,E)
$$

其中边 $(a,b)\in E$ 表示 $a\in A$ 和 $b\in B$ 可以匹配。

目标是找到一个 maximum matching，也就是选择尽可能多的边，并且没有两条被选中的边共享同一个顶点。

### Flow Construction

把 bipartite matching 转化成 max-flow：

1. 加一个 source $s$ 和 sink $t$；
2. 对每个 $a\in A$，加边：

$$
s\to a
$$

capacity 为 $1$；

3. 对每条 bipartite edge $(a,b)\in E$，加边：

$$
a\to b
$$

capacity 为 $1$；

4. 对每个 $b\in B$，加边：

$$
b\to t
$$

capacity 为 $1$。

### Why It Works

由于 $s\to a$ 的 capacity 是 $1$，每个 $a$ 最多只能参与一个匹配。

由于 $b\to t$ 的 capacity 是 $1$，每个 $b$ 最多只能参与一个匹配。

如果某条中间边 $(a,b)$ 上有 flow：

$$
f(a,b)=1
$$

就表示在 matching 中选择边 $(a,b)$。

所以：

> integral flow 的 value = matching 的大小。

反过来，任何 matching 也可以构造出同样大小的 flow：

- $s\to a$ 流 $1$；
- $a\to b$ 流 $1$；
- $b\to t$ 流 $1$。

因此：

$$
\text{maximum bipartite matching size}
=
\text{maximum flow value}
$$

这里再次用到了 integrality theorem：capacity 全是整数，所以 maximum flow 可以取 integral flow，从而能直接对应到 matching。

关键结论：

> Maximum Bipartite Matching 可以作为 Maximum Flow 的特例来求解。

Dessert:

> A graph is regular if all the vertices have the same degree.
> A matching is perfect if all the vertices are matched.
> Prove that a regular bipartite graph always has a perfect
