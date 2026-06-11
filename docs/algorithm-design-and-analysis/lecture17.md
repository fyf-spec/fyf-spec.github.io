# Applications of LP Duality

这一讲继续 Linear Programming Duality，重点看两个经典应用：

- 用 Strong Duality Theorem 重新证明 Max-Flow-Min-Cut Theorem；
- 用 Strong Duality Theorem 证明 von Neumann's Minimax Theorem。

核心思想：

> 很多“组合优化中的等价定理”可以看成 LP strong duality 的具体表现。  
> 如果问题本身是离散的，还需要额外证明 LP optimum 可以取 integral solution。

## Strong Duality Recap

对 standard form primal：

$$
\begin{aligned}
\text{maximize}\quad & \mathbf c^\top \mathbf x\\
\text{subject to}\quad & A\mathbf x\le \mathbf b\\
& \mathbf x\ge \mathbf 0
\end{aligned}
$$

它的 dual 是：

$$
\begin{aligned}
\text{minimize}\quad & \mathbf b^\top \mathbf y\\
\text{subject to}\quad & A^\top \mathbf y\ge \mathbf c\\
& \mathbf y\ge \mathbf 0
\end{aligned}
$$

> **Strong Duality Theorem**：如果 primal 和 dual 都有 optimal solution，设它们分别为 $\mathbf x^*$ 和 $\mathbf y^*$，那么
>
> $$
> \mathbf c^\top \mathbf x^*
> =
> \mathbf b^\top \mathbf y^*
> $$

也就是说：

$$
OPT(\text{primal})=OPT(\text{dual})
$$

## Part I: Max-Flow-Min-Cut Revisited

之前我们已经用 augmenting path 证明过 Ford-Fulkerson 的正确性。这一节用 LP duality 重新证明：

> **Max-Flow-Min-Cut Theorem**：maximum flow 的值等于 minimum cut 的容量。

## Maximum Flow as LP

给定 directed graph：

$$
G=(V,E)
$$

source 为 $s$，sink 为 $t$，每条边 $(u,v)$ 的 capacity 是 $c_{uv}$。

令 $f_{uv}$ 表示边 $(u,v)$ 上的 flow。Maximum Flow 可以写成：

$$
\begin{aligned}
\text{maximize}\quad
& \sum_{u:(s,u)\in E} f_{su}\\
\text{subject to}\quad
& 0\le f_{uv}\le c_{uv} && \forall (u,v)\in E\\
& \sum_{v:(v,u)\in E} f_{vu}
=
\sum_{w:(u,w)\in E} f_{uw}
&& \forall u\in V\setminus\{s,t\}
\end{aligned}
$$

其中：

- objective 是从 source 流出的总流量；
- 第一组约束是 capacity constraint；
- 第二组约束是 flow conservation。

为了写 dual，把 equality constraint 看成等式约束，并给它一个 unrestricted dual variable。

## A Convenient Form

把 LP 写成：

$$
\begin{aligned}
\text{maximize}\quad
& \sum_{u:(s,u)\in E} f_{su}\\
\text{subject to}\quad
& f_{uv}\le c_{uv}
&& \forall (u,v)\in E
\quad \rightarrow y_{uv}\\
& \sum_{v:(v,u)\in E} f_{vu}
-
\sum_{w:(u,w)\in E} f_{uw}
=0
&& \forall u\in V\setminus\{s,t\}
\quad \rightarrow z_u\\
& f_{uv}\ge 0
&& \forall (u,v)\in E
\end{aligned}
$$

这里：

- $y_{uv}\ge 0$ 对应 capacity constraint；
- $z_u$ 对应 flow conservation，因为是 equality constraint，所以 $z_u$ 可以是任意实数。

## Dual Program

这个 LP 的 dual 可以写成：

$$
\begin{aligned}
\text{minimize}\quad
& \sum_{(u,v)\in E} c_{uv}y_{uv}\\
\text{subject to}\quad
& y_{su}+z_u\ge 1
&& \forall u:(s,u)\in E\\
& y_{vt}-z_v\ge 0
&& \forall v:(v,t)\in E\\
& y_{uv}-z_u+z_v\ge 0
&& \forall (u,v)\in E,\ u\ne s,\ v\ne t\\
& y_{uv}\ge 0
&& \forall (u,v)\in E
\end{aligned}
$$

直觉：

- $y_{uv}$ 描述 edge $(u,v)$ 是否被 cut；
- $z_u$ 描述 vertex $u$ 在 cut 的哪一侧。

如果把一个 cut $(L,R)$ 看成：

$$
s\in L,\quad t\in R
$$

那么可以理解为：

$$
z_u=
\begin{cases}
1 & u\in L\\
0 & u\in R
\end{cases}
$$

以及：

$$
y_{uv}=
\begin{cases}
1 & u\in L,\ v\in R\\
0 & \text{otherwise}
\end{cases}
$$

此时 objective：

$$
\sum_{(u,v)\in E}c_{uv}y_{uv}
$$

刚好就是 cut capacity。

## Why This Is Not Immediately Enough

Strong duality 直接给出：

$$
OPT(\text{Max-Flow LP})
=
OPT(\text{Dual LP})
$$

但是我们还想说：

$$
OPT(\text{Dual LP})=OPT(\text{Min-Cut})
$$

这里有一个问题：

> Dual LP 是 fractional 的；Min-Cut 是 discrete 的。  
> Feasible cuts 只是 feasible dual solutions 的一个子集。

也就是说，dual LP 可能存在 fractional solution，比所有 integral cut 都便宜。

所以还需要证明：

> 这个 dual LP 总存在一个 integral optimum。

这就需要 total unimodularity。

## Totally Unimodular Matrix

> **Definition**：一个矩阵 $A$ 是 **totally unimodular (TU)** 的，如果它的任意 square submatrix 的 determinant 都属于
>
> $$
> \{-1,0,1\}
> $$

核心定理：

> 如果 $A$ 是 totally unimodular，并且 $\mathbf b$ 是 integer vector，那么 polytope
>
> $$
> P=\{\mathbf x:A\mathbf x\le \mathbf b\}
> $$
>
> 的所有 vertices 都是 integral 的。

证明思路：

取任意 vertex $\mathbf v$。在 $\mathbb R^n$ 中，一个 vertex 由 $n$ 个 linearly independent tight constraints 决定，所以存在一个可逆的 square submatrix $A'$，使得：

$$
A'\mathbf v=\mathbf b'
$$

其中 $\mathbf b'$ 是 $\mathbf b$ 的一个 sub-vector。

由 Cramer's Rule：

$$
v_i=\frac{\det(A'_i)}{\det(A')}
$$

其中 $A'_i$ 是把 $A'$ 的第 $i$ 列替换为 $\mathbf b'$ 得到的矩阵。

因为 $A$ 是 TU：

$$
\det(A')\in\{-1,1\}
$$

又因为 $\mathbf b'$ 是整数向量，所以：

$$
\det(A'_i)\in\mathbb Z
$$

因此每个 $v_i$ 都是整数。

## Integrality Corollary for LP

由于 LP 的 optimum 可以在某个 vertex 上取得，所以有：

> 如果 constraint matrix 是 totally unimodular，并且右端向量是整数，那么 LP 存在 integral optimal solution。

对于 primal-dual pair：

$$
\begin{aligned}
\text{maximize}\quad & \mathbf c^\top\mathbf x\\
\text{subject to}\quad & A\mathbf x\le \mathbf b\\
& \mathbf x\ge 0
\end{aligned}
$$

和：

$$
\begin{aligned}
\text{minimize}\quad & \mathbf b^\top\mathbf y\\
\text{subject to}\quad & A^\top\mathbf y\ge \mathbf c\\
& \mathbf y\ge 0
\end{aligned}
$$

如果 $A$ 是 TU，那么 $A^\top$ 也是 TU。

因此：

- 如果 $\mathbf b$ 是整数，primal LP 有 integral optimum；
- 如果 $\mathbf c$ 是整数，dual LP 有 integral optimum。

## Total Unimodularity in the Max-Flow Dual

回到 max-flow dual：

$$
\begin{aligned}
\text{minimize}\quad
& \sum_{(u,v)\in E} c_{uv}y_{uv}\\
\text{subject to}\quad
& y_{su}+z_u\ge 1\\
& y_{vt}-z_v\ge 0\\
& y_{uv}-z_u+z_v\ge 0\\
& y_{uv}\ge 0
\end{aligned}
$$

约束矩阵可以看成两部分拼接：

$$
[Y\ Z]
$$

其中：

- $Y$ 是关于 $y_{uv}$ 的 identity matrix；
- $Z$ 是关于 $z_u$ 的 directed incidence matrix 的变体。

因为 identity matrix 的拼接不会破坏 total unimodularity，所以只需要证明 $Z$ 是 totally unimodular。

## Why the Incidence Matrix Is TU

$Z$ 的每一行对应一条边。对于边：

- $(s,u)$：这一行在 $u$ 的列上是 $1$；
- $(u,v)$：这一行在 $u$ 的列上是 $-1$，在 $v$ 的列上是 $1$；
- $(v,t)$：这一行在 $v$ 的列上是 $-1$。

也就是说，每一行最多有两个 non-zero entries，并且如果有两个，它们分别是 $-1$ 和 $1$。

证明 $Z$ 是 TU 可以用 induction：

1. $1\times 1$ submatrix 的元素只可能是 $-1,0,1$；
2. 假设所有 $k\times k$ submatrix 的 determinant 都是 $-1,0,1$；
3. 考虑任意 $(k+1)\times(k+1)$ submatrix $Z'$：

如果某一行全是 $0$，那么：

$$
\det(Z')=0
$$

如果某一行只有一个 non-zero entry，那么沿这一行展开，$\det(Z')$ 等于 $\pm 1$ 乘以一个 $k\times k$ submatrix 的 determinant，所以仍然是 $-1,0,1$。

如果每一行都有两个 non-zero entries，那么每一行都有一个 $-1$ 和一个 $1$。把所有列向量相加会得到 zero vector，因此列向量线性相关：

$$
\det(Z')=0
$$

所以 $Z$ 是 TU，进而整个 dual constraint matrix 也是 TU。

结论：

> max-flow dual LP 存在 integral optimal solution。

## From Integral Dual Solution to a Cut

现在已经知道：

$$
OPT(\text{Dual LP})=OPT(\text{Primal LP})=v(f_{\max})
$$

并且 dual optimum 可以由 integral solution $(y^*,z^*)$ 达到。

还需要把这个 integral dual solution 转成一个 actual cut。

定义：

$$
L=\{s\}\cup\{u\in V\setminus\{s,t\}:z_u^*>0\}
$$

以及：

$$
R=V\setminus L
$$

因为 $s\in L$ 且 $t\in R$，所以 $(L,R)$ 是一个合法的 $s$-$t$ cut。

考虑任意从 $L$ 指向 $R$ 的边 $(u,v)$。

由于 $z^*$ 是整数：

- 如果 $u\in L$，那么 $z_u^*\ge 1$，除非 $u=s$；
- 如果 $v\in R$，那么 $z_v^*\le 0$，除非 $v=t$。

根据 dual constraints，可以推出每条 crossing edge 都满足：

$$
y_{uv}^*\ge 1
$$

例如对于中间边 $(u,v)$：

$$
y_{uv}^*-z_u^*+z_v^*\ge 0
$$

所以：

$$
y_{uv}^*\ge z_u^*-z_v^*\ge 1
$$

因此 cut capacity 满足：

$$
c(L,R)
=
\sum_{(u,v)\in out(L)} c_{uv}
\le
\sum_{(u,v)\in out(L)} c_{uv}y_{uv}^*
\le
\sum_{(u,v)\in E} c_{uv}y_{uv}^*
=
OPT(\text{Dual LP})
$$

另一方面，每个 cut 都可以构造一个 feasible dual solution，其 objective value 等于 cut capacity。因此：

$$
OPT(\text{Dual LP})\le c(L,R)
$$

两边合起来：

$$
c(L,R)=OPT(\text{Dual LP})
$$

于是：

$$
\text{MinCut}
=
OPT(\text{Dual LP})
=
OPT(\text{Max-Flow LP})
=
\text{MaxFlow}
$$

这就用 LP strong duality 证明了 Max-Flow-Min-Cut Theorem。

## General Framework

用 LP duality 证明这类 theorem 的套路：

1. 写出 primal LP；
2. 写出 dual LP；
3. 解释 primal 和 dual 分别对应哪两个问题；
4. 如果其中一个问题是离散的，证明 LP optimum 可以取 integral solution；
5. 用 Strong Duality Theorem 得到两个 optimum 相等。

在本节里：

- primal LP 描述 maximum flow；
- dual LP 描述 fractional min-cut；
- total unimodularity 保证 dual 有 integral optimum；
- integral dual optimum 可以转成 actual cut；
- strong duality 给出 max-flow = min-cut。

## Revisiting Max-Flow Integrality

我们也可以用 total unimodularity 重新证明 max-flow integrality theorem。

> **Max-Flow Integrality Theorem**：如果所有 capacities 都是整数，那么存在一个 integral maximum flow。

原因：

- max-flow LP 的 constraint matrix 是 TU；
- capacity vector $\mathbf b$ 是整数；
- 因此 max-flow LP 存在 integral optimal solution。

这说明 integrality theorem 不只是 Ford-Fulkerson 的性质，它也来自 max-flow LP 本身的 total unimodularity。

## Exercise: Kőnig's Theorem

另一个可以用 LP duality 证明的经典定理是：

> **Kőnig's Theorem**：在 bipartite graph 中，
>
> $$
> \text{Maximum Matching}
> =
> \text{Minimum Vertex Cover}
> $$

证明思路和 max-flow-min-cut 类似：

- 把 maximum bipartite matching 写成 LP；
- 写出它的 dual，也就是 vertex cover LP；
- 利用 bipartite incidence matrix 的 total unimodularity，证明 LP optimum 可以取 integral solution；
- 用 strong duality 得到 maximum matching size 等于 minimum vertex cover size。

## Part II: von Neumann's Minimax Theorem

第二个 application 是 zero-sum game 中的 Minimax Theorem。

## Zero-Sum Game

有两个玩家：

- Player $A$；
- Player $B$。

Player $A$ 有动作集合：

$$
\{a_1,a_2,\ldots,a_m\}
$$

Player $B$ 有动作集合：

$$
\{b_1,b_2,\ldots,b_n\}
$$

当 $A$ 选择 $a_i$，$B$ 选择 $b_j$ 时，Player $A$ 得到 utility：

$$
G_{ij}
$$

Player $B$ 得到 utility：

$$
-G_{ij}
$$

因此这是 zero-sum game：

$$
u_A(a_i,b_j)+u_B(a_i,b_j)=0
$$

矩阵：

$$
G\in\mathbb R^{m\times n}
$$

叫 payoff matrix，其中 $G_{ij}$ 表示 $A$ 的收益，也就是 $B$ 的损失。

## Rock-Scissors-Paper Example

Rock-Scissors-Paper 的 payoff matrix 可以写成：

$$
G=
\begin{pmatrix}
0 & 1 & -1\\
-1 & 0 & 1\\
1 & -1 & 0
\end{pmatrix}
$$

行对应 Player $A$ 的动作：

$$
\text{Rock},\ \text{Scissors},\ \text{Paper}
$$

列对应 Player $B$ 的动作。

例如：

- $A$ 出 Rock，$B$ 出 Scissors，则 $A$ 赢，收益为 $1$；
- $A$ 出 Rock，$B$ 出 Paper，则 $A$ 输，收益为 $-1$。

## Strategy

一个 pure strategy 就是固定选择某个动作。

一个 mixed strategy 是动作上的 probability distribution。

Player $A$ 的 mixed strategy 记为：

$$
\mathbf x=(x_1,\ldots,x_m)
$$

满足：

$$
\sum_{i=1}^m x_i=1,\quad x_i\ge 0
$$

Player $B$ 的 mixed strategy 记为：

$$
\mathbf y=(y_1,\ldots,y_n)
$$

满足：

$$
\sum_{j=1}^n y_j=1,\quad y_j\ge 0
$$

如果某个 action 的概率是 $1$，就是 pure strategy；否则就是 mixed strategy。

## Expected Utility

当 Player $A$ 使用 strategy $\mathbf x$，Player $B$ 使用 strategy $\mathbf y$ 时，Player $A$ 的期望收益是：

$$
U_A(\mathbf x,\mathbf y)
=
\mathbf x^\top G\mathbf y
=
\sum_{i,j}G_{ij}x_iy_j
$$

Player $B$ 的期望收益是：

$$
U_B(\mathbf x,\mathbf y)
=
-\mathbf x^\top G\mathbf y
$$

因为是 zero-sum，$A$ 想 maximize $\mathbf x^\top G\mathbf y$，而 $B$ 想 minimize 它。

## Does Order Matter?

如果 $A$ 先选 strategy，$B$ 看到了以后选择 best response，那么 $A$ 会选择：

$$
\max_{\mathbf x}\min_{\mathbf y}\mathbf x^\top G\mathbf y
$$

如果 $B$ 先选 strategy，$A$ 看到了以后选择 best response，那么 $A$ 的收益是：

$$
\min_{\mathbf y}\max_{\mathbf x}\mathbf x^\top G\mathbf y
$$

一般总有：

$$
\max_{\mathbf x}\min_{\mathbf y}\mathbf x^\top G\mathbf y
\le
\min_{\mathbf y}\max_{\mathbf x}\mathbf x^\top G\mathbf y
$$

Minimax Theorem 说，在 zero-sum game 中，这两个值其实相等。

## Minimax Theorem

> **von Neumann's Minimax Theorem**：
>
> $$
> \max_{\mathbf x}\min_{\mathbf y}\mathbf x^\top G\mathbf y
> =
> \min_{\mathbf y}\max_{\mathbf x}\mathbf x^\top G\mathbf y
> $$

也就是说：

> 在 zero-sum game 中，谁先选择 mixed strategy 不影响最终的 game value。

Rock-Scissors-Paper 中，双方都使用：

$$
\left(\frac13,\frac13,\frac13\right)
$$

时，期望收益是 $0$。这就是这个游戏的 value。

## Pure Strategy Best Response

先证明一个简单 lemma。

> 固定 Player $A$ 的 strategy $\mathbf x$，Player $B$ 总存在一个 pure strategy best response。

证明：

如果 $B$ 使用 mixed strategy $\mathbf y$，那么 $B$ 想 maximize：

$$
-\mathbf x^\top G\mathbf y
$$

也就是：

$$
-\sum_{j=1}^n y_j\left(\sum_{i=1}^m G_{ij}x_i\right)
$$

这是关于 $\mathbf y$ 的线性函数。在线性函数上，simplex 上的 optimum 可以在 vertex 上取得。

而 probability simplex 的 vertices 正是 pure strategies。

所以 $B$ 的 best response 可以取 pure strategy。

同理，固定 $B$ 的 strategy，$A$ 也存在 pure strategy best response。

## LP for the Max-Min Side

由上面的 lemma：

$$
\max_{\mathbf x}\min_{\mathbf y}\mathbf x^\top G\mathbf y
=
\max_{\mathbf x}\min_{j=1,\ldots,n}\sum_iG_{ij}x_i
$$

令 $z$ 表示 Player $A$ 能保证的 utility。为了保证无论 $B$ 选择哪一列，$A$ 的收益都至少是 $z$，需要：

$$
\sum_iG_{ij}x_i\ge z
\quad
\forall j=1,\ldots,n
$$

于是 max-min 可以写成 LP：

$$
\begin{aligned}
\text{maximize}\quad & z\\
\text{subject to}\quad
& \sum_iG_{ij}x_i\ge z && \forall j=1,\ldots,n\\
& \sum_{i=1}^m x_i=1\\
& x_i\ge 0 && \forall i
\end{aligned}
$$

这里 $z$ 是 unrestricted variable，因为 game value 可以是负数、零或正数。

## Dual LP

把第一组 constraints 改写为：

$$
z-\sum_iG_{ij}x_i\le 0
$$

给每一列对应一个 dual variable $y_j\ge 0$，给 equality constraint $\sum_i x_i=1$ 对应一个 unrestricted dual variable $w$。

最终 dual 可以化简为：

$$
\begin{aligned}
\text{minimize}\quad & w\\
\text{subject to}\quad
& \sum_jG_{ij}y_j\le w && \forall i=1,\ldots,m\\
& \sum_{j=1}^n y_j=1\\
& y_j\ge 0 && \forall j
\end{aligned}
$$

这个 LP 的含义是：

Player $B$ 选择 mixed strategy $\mathbf y$，让 Player $A$ 对任意 pure strategy $i$ 的 payoff 都不超过 $w$，并且让这个 upper bound $w$ 尽可能小。

因此它正是：

$$
\min_{\mathbf y}\max_{i=1,\ldots,m}\sum_jG_{ij}y_j
$$

由 pure strategy best response lemma，这等于：

$$
\min_{\mathbf y}\max_{\mathbf x}\mathbf x^\top G\mathbf y
$$

## Strong Duality Implies Minimax

Primal LP 的 optimum 是：

$$
\max_{\mathbf x}\min_{\mathbf y}\mathbf x^\top G\mathbf y
$$

Dual LP 的 optimum 是：

$$
\min_{\mathbf y}\max_{\mathbf x}\mathbf x^\top G\mathbf y
$$

由 Strong Duality Theorem：

$$
OPT(\text{Primal})=OPT(\text{Dual})
$$

因此：

$$
\max_{\mathbf x}\min_{\mathbf y}\mathbf x^\top G\mathbf y
=
\min_{\mathbf y}\max_{\mathbf x}\mathbf x^\top G\mathbf y
$$

这就证明了 von Neumann's Minimax Theorem。

## Summary

本讲的核心结论：

- Strong Duality 可以用来证明很多“两个 optimization problems 的 optimum 相等”的定理；
- Max-Flow-Min-Cut 可以看成 max-flow LP 和 min-cut dual LP 的 strong duality；
- 因为 min-cut 是 discrete problem，所以还需要 total unimodularity 来保证 dual LP 有 integral optimum；
- totally unimodular matrix 加 integer right-hand side 会让 LP vertices 都是 integral；
- max-flow integrality theorem 也可以由 total unimodularity 推出；
- Kőnig's Theorem 也可以通过 LP duality 和 total unimodularity 证明；
- zero-sum game 的 Minimax Theorem 可以通过把 max-min 写成 LP、把 min-max 写成 dual LP 来证明。
