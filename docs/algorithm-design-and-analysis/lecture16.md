# Linear Programming, Duality, and LP Relaxation

这一讲介绍 Linear Programming，也就是线性规划。主线是：

- 什么是 LP，以及 LP 的几何直觉；
- Simplex Method 的 high-level idea；
- LP 的标准型和多项式时间可解性；
- LP Duality，包括 weak duality 和 strong duality；
- LP-Relaxation：用 LP 设计 approximation algorithm；
- Vertex Cover 和 online matching 中的 primal-dual analysis。

## Linear Program

一个 **linear program (LP)** 由两部分组成：

1. 一组线性的等式或不等式约束；
2. 一个需要 maximize 或 minimize 的线性目标函数。

标准的 maximization 形式可以写成：

$$
\begin{aligned}
\text{maximize}\quad & c_1x_1+c_2x_2+\cdots+c_nx_n \\
\text{subject to}\quad
& a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n\le b_1\\
& a_{21}x_1+a_{22}x_2+\cdots+a_{2n}x_n\le b_2\\
& \cdots\\
& a_{m1}x_1+a_{m2}x_2+\cdots+a_{mn}x_n\le b_m\\
& x_1,x_2,\ldots,x_n\ge 0
\end{aligned}
$$

用矩阵形式写就是：

$$
\begin{aligned}
\text{maximize}\quad & \mathbf c^\top \mathbf x\\
\text{subject to}\quad & A\mathbf x\le \mathbf b\\
& \mathbf x\ge \mathbf 0
\end{aligned}
$$

## Factory Example

假设一个工厂可以生产两种产品：

- sugar：每吨 profit 是 $1$；
- oil：每吨 profit 是 $6$。

资源限制：

- sugar 最多生产 $200$ 吨；
- oil 最多生产 $300$ 吨；
- 总重量最多 $400$ 吨。

令：

- $x_1$ 表示 sugar 的产量；
- $x_2$ 表示 oil 的产量。

那么 LP 是：

$$
\begin{aligned}
\text{maximize}\quad & x_1+6x_2\\
\text{subject to}\quad
& x_1\le 200\\
& x_2\le 300\\
& x_1+x_2\le 400\\
& x_1,x_2\ge 0
\end{aligned}
$$

### Feasible Region

所有满足约束的点 $(x_1,x_2)$ 构成 feasible region。

几何上，每个线性不等式都是一个 half-plane；所有 half-plane 的交集就是 feasible region。

在二维中，它是一个多边形；在高维中，它是一个 convex polyhedron。如果这个区域有界，也叫 polytope。

### Maximizing the Objective

目标函数：

$$
x_1+6x_2=c
$$

是一族平行直线。我们希望把这条直线往 objective value 增大的方向移动，直到它最后一次接触 feasible region。

在这个例子中，最优点是：

$$
(x_1,x_2)=(100,300)
$$

objective value 是：

$$
100+6\cdot 300=1900
$$

## Important Observations

对于 LP，有几个重要的几何事实。

### Optimum at a Vertex

如果 LP feasible、bounded，并且最优解存在，那么总存在一个最优解位于 feasible region 的某个 vertex 上。

直观理解：

> 线性目标函数是一族平行 hyperplanes。  
> 当 hyperplane 被推到最大值时，它最后接触 feasible region 的地方可以选在 vertex。

注意：最优解不一定唯一；如果一整条边都最优，那么这条边的两个端点也是最优解。

### Convexity

LP 的 feasible region 是 convex 的。

原因是：

- 每个线性约束对应一个 half-space；
- half-space 是 convex 的；
- convex set 的交集仍然是 convex 的。

### Local Maximum Is Global Maximum

由于 feasible region 是 convex 的，线性目标函数没有“坏的局部最优”。

关键结论：

> 对 LP 来说，只要一个 feasible point 在所有相邻方向上都不能改进 objective，它就是 global optimum。

这也是 Simplex Method 能工作的几何直觉。

## Simplex Method

Simplex Method 的 high-level idea：

1. 找一个起始 vertex；
2. 如果存在相邻 vertex 能提高 objective，就沿着 edge 走过去；
3. 重复这个过程；
4. 到达一个 local maximum 时停止。

由于 LP 的 local maximum 也是 global maximum，算法停止时得到最优解。

### What Is a Vertex?

在 $\mathbb R^n$ 中，一个 vertex 可以理解为 $n$ 个 linearly independent hyperplanes 的交点。

例如二维中，一个 vertex 是两条边界直线的交点；三维中，一个 vertex 是三个平面的交点。

### What Is an Edge?

在 $\mathbb R^n$ 中，一条 edge 可以理解为 $n-1$ 个 linearly independent hyperplanes 的交集。

Simplex 从一个 vertex 走到相邻 vertex，本质上是：

- 放松当前 tight 的某个约束；
- 让另一个约束变 tight；
- 解一个新的线性方程组，得到新的 vertex。

### Example Path

在 factory example 中，Simplex 可以从原点 $O$ 开始：

$$
O\to A\to B\to C
$$

每一步都移动到一个相邻 vertex，并且 objective value 变大。到达 $C=(100,300)$ 后，继续走向任意相邻 vertex 都会降低 objective，所以 $C$ 是局部最优，也就是全局最优。

### Missing Details

真实的 Simplex Method 还需要处理很多细节：

- 如何找到一个起始 vertex；
- 如何选择 pivot rule，也就是选哪个相邻 vertex；
- degenerate vertex：超过 $n$ 个 constraints 同时 tight；
- feasible region unbounded；
- LP infeasible；
- 数值误差和实现问题。

实际编程时，一般不手写 LP solver，而是使用成熟工具：

- GLPK；
- Gurobi；
- CPLEX；
- OR-Tools 等。

### Time Complexity of Simplex

Simplex Method 在实践中非常快，也是最常用的 LP 算法之一。

但是 worst-case running time 是 exponential 的。

如果有 $m$ 个 constraints 和 $n$ 个 variables，vertex 数量最多可以达到：

$$
\binom{m}{n}
$$

不同 pivot rule 可能产生非常不同的表现。很多自然的策略，例如每次选择 objective 增加最多的邻居，也仍然可能在 worst case 下很慢。

一个重要结果是 Spielman 和 Teng 的 smoothed analysis：

> 如果对 constraints 加入轻微随机噪声，Simplex 的 average behavior 是 polynomial 的。

## Polynomial-Time Algorithms for LP

虽然 Simplex worst-case exponential，但 LP 本身是 polynomial-time solvable 的。

经典的 polynomial-time LP algorithms 包括：

- Ellipsoid Method；
- Interior Point Method。

关键结论：

> Linear Programming 可以在多项式时间内求解。  
> 如果一个问题可以被 formulation 成 LP，那么它就是 polynomial-time solvable。

## Standard Form LP

本讲使用的 standard form 是：

$$
\begin{aligned}
\text{maximize}\quad & \mathbf c^\top \mathbf x\\
\text{subject to}\quad & A\mathbf x\le \mathbf b\\
& \mathbf x\ge \mathbf 0
\end{aligned}
$$

也就是说：

- objective 是 maximization；
- constraints 都是 $\le$；
- variables 都是 non-negative。

很多其它形式都可以转成 standard form。

### Minimization to Maximization

$$
\min c_1x_1+\cdots+c_nx_n
\Longleftrightarrow
\max -c_1x_1-\cdots-c_nx_n
$$

### Greater-Than Constraints

$$
a_1x_1+\cdots+a_nx_n\ge b
\Longleftrightarrow
-a_1x_1-\cdots-a_nx_n\le -b
$$

### Equality Constraints

等式可以拆成两个不等式：

$$
a_1x_1+\cdots+a_nx_n=b
\Longleftrightarrow
\begin{cases}
a_1x_1+\cdots+a_nx_n\le b\\
a_1x_1+\cdots+a_nx_n\ge b
\end{cases}
$$

也可以通过 slack variable 把不等式转成等式：

$$
a_1x_1+\cdots+a_nx_n\le b
\Longleftrightarrow
a_1x_1+\cdots+a_nx_n+s=b,\quad s\ge 0
$$

### Variables with Unrestricted Sign

如果变量 $x$ 没有符号限制，可以引入：

$$
x=x^+-x^-
$$

其中：

$$
x^+,x^-\ge 0
$$

这样就把 unrestricted variable 转成了两个 non-negative variables。

## Maximum Flow as LP

Maximum Flow 也可以 formulation 成 LP。

给定 directed graph $G=(V,E)$，source $s$，sink $t$，capacity $c(u,v)$。

令 $f_{uv}$ 表示边 $(u,v)$ 上的 flow。

LP formulation：

$$
\begin{aligned}
\text{maximize}\quad
& \sum_{u:(s,u)\in E} f_{su}\\
\text{subject to}\quad
& 0\le f_{uv}\le c(u,v) && \forall (u,v)\in E\\
& \sum_{v:(v,u)\in E} f_{vu}
=
\sum_{w:(u,w)\in E} f_{uw}
&& \forall u\in V\setminus\{s,t\}
\end{aligned}
$$

这正是 maximum flow 的定义：

- capacity constraint；
- flow conservation；
- maximize total flow leaving source。

一个有趣的观点：

> Ford-Fulkerson Method 可以看作 Simplex Method 在 max-flow LP 上的特殊实现。

## LP Duality

LP duality 的核心问题是：

> 如果 primal LP 是在找最优解，那么 dual LP 是在找什么？

直观地说：

- primal 在寻找一个 feasible solution，让 objective 尽量大；
- dual 在寻找一个 certificate，证明 primal 的 objective 不可能超过某个上界；
- strong duality 说明：最好的上界刚好等于 primal optimum。

## Motivation for Dual

回到 factory example：

$$
\begin{aligned}
\text{maximize}\quad & x_1+6x_2\\
\text{subject to}\quad
& x_1\le 200 \quad (i)\\
& x_2\le 300 \quad (ii)\\
& x_1+x_2\le 400 \quad (iii)\\
& x_1,x_2\ge 0
\end{aligned}
$$

我们已经知道最优值是 $1900$。如何不用几何图像证明？

可以把约束线性组合起来。

例如，把 $(i)$ 加上 $6$ 倍 $(ii)$：

$$
x_1+6x_2\le 200+6\cdot 300=2000
$$

这说明 objective 不可能超过 $2000$，但这个 bound 不够紧。

更好的组合是：

- $5$ 倍 $(ii)$；
- 加上 $(iii)$。

得到：

$$
5x_2+(x_1+x_2)\le 5\cdot 300+400
$$

也就是：

$$
x_1+6x_2\le 1900
$$

而我们已经有 feasible solution $(100,300)$ 达到 $1900$，所以它就是最优解。

关键想法：

> dual variables 就是在给 primal constraints 分配权重，用这些 constraints 组合出 objective 的上界。

## Deriving a Dual Program

考虑三变量例子：

$$
\begin{aligned}
\text{maximize}\quad & x_1+6x_2+13x_3\\
\text{subject to}\quad
& x_1\le 200 \quad (i)\\
& x_2\le 300 \quad (ii)\\
& x_1+x_2+x_3\le 400 \quad (iii)\\
& x_2+3x_3\le 600 \quad (iv)\\
& x_1,x_2,x_3\ge 0
\end{aligned}
$$

给四个 constraints 分别乘上非负权重：

$$
y_1,y_2,y_3,y_4\ge 0
$$

把它们加起来，右边变成：

$$
200y_1+300y_2+400y_3+600y_4
$$

左边中 $x_1,x_2,x_3$ 的系数分别是：

$$
y_1+y_3
$$

$$
y_2+y_3+y_4
$$

$$
y_3+3y_4
$$

为了让这个线性组合成为 objective 的 upper bound，需要：

$$
y_1+y_3\ge 1
$$

$$
y_2+y_3+y_4\ge 6
$$

$$
y_3+3y_4\ge 13
$$

然后我们希望 upper bound 尽可能小，所以 dual LP 是：

$$
\begin{aligned}
\text{minimize}\quad
& 200y_1+300y_2+400y_3+600y_4\\
\text{subject to}\quad
& y_1+y_3\ge 1\\
& y_2+y_3+y_4\ge 6\\
& y_3+3y_4\ge 13\\
& y_1,y_2,y_3,y_4\ge 0
\end{aligned}
$$

这里，原问题叫 **primal program**，新构造的问题叫 **dual program**。

## General Dual Program

对于 standard form primal：

$$
\begin{aligned}
\text{maximize}\quad & \mathbf c^\top\mathbf x\\
\text{subject to}\quad & A\mathbf x\le \mathbf b\\
& \mathbf x\ge \mathbf 0
\end{aligned}
$$

它的 dual 是：

$$
\begin{aligned}
\text{minimize}\quad & \mathbf b^\top\mathbf y\\
\text{subject to}\quad & \mathbf y^\top A\ge \mathbf c^\top\\
& \mathbf y\ge \mathbf 0
\end{aligned}
$$

也可以写成：

$$
\begin{aligned}
\text{minimize}\quad & \mathbf b^\top\mathbf y\\
\text{subject to}\quad & A^\top\mathbf y\ge \mathbf c\\
& \mathbf y\ge \mathbf 0
\end{aligned}
$$

对应关系：

- primal 是 max，dual 是 min；
- primal 的每个 constraint 对应一个 dual variable；
- primal 的每个 variable 对应一个 dual constraint；
- matrix 从 $A$ 变成 $A^\top$。

## Weak Duality Theorem

> **Weak Duality Theorem**：  
> 如果 $\hat{\mathbf x}$ 是 primal feasible solution，$\hat{\mathbf y}$ 是 dual feasible solution，那么
>
> $$
> \mathbf c^\top \hat{\mathbf x}\le \mathbf b^\top \hat{\mathbf y}
> $$

证明非常短：

因为 primal feasible：

$$
A\hat{\mathbf x}\le \mathbf b
$$

因为 $\hat{\mathbf y}\ge 0$，所以：

$$
\hat{\mathbf y}^\top A\hat{\mathbf x}\le \hat{\mathbf y}^\top \mathbf b
$$

因为 dual feasible：

$$
\hat{\mathbf y}^\top A\ge \mathbf c^\top
$$

又因为 $\hat{\mathbf x}\ge 0$，所以：

$$
\mathbf c^\top \hat{\mathbf x}\le \hat{\mathbf y}^\top A\hat{\mathbf x}
$$

合起来：

$$
\mathbf c^\top \hat{\mathbf x}
\le
\hat{\mathbf y}^\top A\hat{\mathbf x}
\le
\hat{\mathbf y}^\top \mathbf b
=
\mathbf b^\top \hat{\mathbf y}
$$

关键结论：

> 任意 dual feasible solution 都给出 primal optimum 的一个 upper bound。

## Strong Duality Theorem

> **Strong Duality Theorem**：  
> 如果 primal 和 dual 都有 optimal solution，设它们分别为 $\mathbf x^*$ 和 $\mathbf y^*$，那么
>
> $$
> \mathbf c^\top\mathbf x^*
> =
> \mathbf b^\top\mathbf y^*
> $$

也就是说，weak duality 中的 gap 在最优解处一定 closed。

直观理解：

> primal 在找最好的 feasible solution；  
> dual 在找最紧的 upper bound；  
> strong duality 说明这两个值相等。

Strong duality 有很多重要应用：

- Max-Flow-Min-Cut Theorem；
- Minimax Theorem；
- Kőnig-Egerváry Theorem；
- approximation algorithms 中的 dual fitting；
- primal-dual schema；
- resource allocation 的经济解释。

## Economic Interpretation of Duality

可以把 duality 理解成 resource pricing。

假设有三种资源：

- $R_1$ 需要 $5$ 单位；
- $R_2$ 需要 $5$ 单位；
- $R_3$ 需要 $3$ 单位。

我们要给每种资源定价：

$$
p_1,p_2,p_3
$$

目标是 maximize revenue：

$$
5p_1+5p_2+3p_3
$$

但是资源不是直接卖，而是通过产品组合体现价格：

- Product 1 使用 $2R_1+3R_2$，市场价格是 $10$；
- Product 2 使用 $R_1+R_2+2R_3$，市场价格是 $5$。

为了避免“资源成本超过产品价格”，需要：

$$
2p_1+3p_2\le 10
$$

$$
p_1+p_2+2p_3\le 5
$$

所以 primal 是：

$$
\begin{aligned}
\text{maximize}\quad & 5p_1+5p_2+3p_3\\
\text{subject to}\quad
& 2p_1+3p_2\le 10\\
& p_1+p_2+2p_3\le 5\\
& p_1,p_2,p_3\ge 0
\end{aligned}
$$

dual 是：

$$
\begin{aligned}
\text{minimize}\quad & 10y_1+5y_2\\
\text{subject to}\quad
& 2y_1+y_2\ge 5\\
& 3y_1+y_2\ge 5\\
& 2y_2\ge 3\\
& y_1,y_2\ge 0
\end{aligned}
$$

解释：

- primal：资源拥有者如何给资源定价，使总收入最大；
- dual：购买者如何购买产品组合，满足资源需求并使成本最小。

Weak duality：

> 因为每个产品的市场价格至少覆盖资源价格，所以购买者的成本一定不小于资源拥有者的收入。

Strong duality：

> 最优资源定价收入 = 最优产品购买成本。

## Proof Idea of Strong Duality

Slides 中用 Farkas Lemma 证明 strong duality。这里记录核心思路。

### Farkas Lemma

> **Farkas Lemma**：  
> 对矩阵 $A\in\mathbb R^{m\times n}$ 和向量 $\mathbf b\in\mathbb R^m$，下面两个命题恰好有一个成立：
>
> 1. 存在 $\mathbf x\in\mathbb R^n$，满足 $\mathbf x\ge 0$ 且
>
> $$
> A\mathbf x=\mathbf b
> $$
>
> 2. 存在 $\mathbf y\in\mathbb R^m$，满足
>
> $$
> A^\top \mathbf y\ge 0
> $$
>
> 且
>
> $$
> \mathbf b^\top \mathbf y<0
> $$

几何理解：

> $A\mathbf x$ with $\mathbf x\ge 0$ 形成由 $A$ 的列向量生成的 cone。  
> Farkas Lemma 说：要么 $\mathbf b$ 在这个 cone 里，要么存在一个 hyperplane 把 $\mathbf b$ 和这个 cone 分开。

### Corollary

一个常用推论是：

> 对矩阵 $A$ 和向量 $\mathbf b$，下面两个命题恰好有一个成立：
>
> 1. 存在 $\mathbf x\ge 0$，使得
>
> $$
> A\mathbf x\ge \mathbf b
> $$
>
> 2. 存在 $\mathbf y\le 0$，使得
>
> $$
> A^\top\mathbf y\ge 0
> $$
>
> 且
>
> $$
> \mathbf b^\top\mathbf y<0
> $$

这个推论可以通过给 $A$ 拼上 $-I$ 得到：

$$
A'=[A\ -I]
$$

然后对 $A'$ 应用 Farkas Lemma。

### Strong Duality Proof Sketch

已知 weak duality：

$$
\mathbf c^\top\mathbf x\le \mathbf b^\top\mathbf y^*
$$

对所有 primal feasible $\mathbf x$ 成立。

假设 strong duality 不成立，即 primal 最优值严格小于 dual 最优值：

$$
\mathbf c^\top\mathbf x<\mathbf b^\top\mathbf y^*
$$

那么不存在 $\mathbf x\ge 0$ 同时满足：

$$
A\mathbf x\le \mathbf b
$$

和：

$$
\mathbf c^\top\mathbf x\ge \mathbf b^\top\mathbf y^*
$$

把它改写成一个 $A'\mathbf x\ge \mathbf b'$ 的不可行系统，然后用 Farkas Lemma 的 corollary，会推出存在一个 dual feasible solution 比 $\mathbf y^*$ 更好，或者推出 primal feasible region 为空。

这两种情况都矛盾。因此 strong duality 成立。

这里最重要的不是记住代数细节，而是理解：

> strong duality 本质上来自 separating hyperplane 的思想：  
> 如果最优值之间有 gap，就能构造出一个更好的 dual certificate，从而矛盾。

## Integer Program

如果在线性规划中要求变量必须是整数，就得到 **integer program (IP)** 或 **integer linear program (ILP)**。

标准形式：

$$
\begin{aligned}
\text{maximize}\quad & \mathbf c^\top\mathbf x\\
\text{subject to}\quad & A\mathbf x\le \mathbf b\\
& \mathbf x\ge \mathbf 0\\
& \mathbf x\in \mathbb Z^n
\end{aligned}
$$

很多 combinatorial optimization problems 都可以写成 IP。

但是：

> Integer Programming 是 NP-complete 的。  
> 即使是 $x_i\in\{0,1\}$ 的 0-1 IP，也已经很难。

## LP Relaxation

LP-relaxation 的基本思路：

1. 先把问题写成 integer program；
2. 放松整数约束，例如把

$$
x_i\in\{0,1\}
$$

放松成：

$$
0\le x_i\le 1
$$

3. 解这个 LP，得到 fractional solution；
4. 把 fractional solution rounding 成 integral solution；
5. 证明 rounding 后仍然 feasible，并且 objective 不会变差太多。

关键点：

> LP 比 IP 容易，因为 LP 多项式时间可解；  
> 但 LP 的 feasible region 更大，所以 LP optimum 往往只是 IP optimum 的 optimistic bound。

对于 minimization problem：

$$
OPT(LP)\le OPT(IP)
$$

对于 maximization problem：

$$
OPT(LP)\ge OPT(IP)
$$

## LP-Relaxation Example: Vertex Cover

给定无向图 $G=(V,E)$。

一个 vertex cover 是点集 $S\subseteq V$，满足每条边至少有一个 endpoint 在 $S$ 中。

Minimum Vertex Cover 问题：

> 找到 size 最小的 vertex cover。

### Integer Program

对每个 vertex $v\in V$，定义变量：

$$
x_v=
\begin{cases}
1 & v\text{ is selected}\\
0 & v\text{ is not selected}
\end{cases}
$$

IP formulation：

$$
\begin{aligned}
\text{minimize}\quad & \sum_{v\in V}x_v\\
\text{subject to}\quad
& x_u+x_v\ge 1 && \forall (u,v)\in E\\
& x_v\in\{0,1\} && \forall v\in V
\end{aligned}
$$

约束 $x_u+x_v\ge 1$ 的意思是：每条边 $(u,v)$ 至少选一个端点。

### LP Relaxation

把 $x_v\in\{0,1\}$ 放松成 $0\le x_v\le 1$：

$$
\begin{aligned}
\text{minimize}\quad & \sum_{v\in V}x_v\\
\text{subject to}\quad
& x_u+x_v\ge 1 && \forall (u,v)\in E\\
& 0\le x_v\le 1 && \forall v\in V
\end{aligned}
$$

因为 LP 的 feasible region 包含 IP 的 feasible region，所以：

$$
OPT(LP)\le OPT(IP)
$$

## A 2-Approximation for Vertex Cover

算法：

1. 解 LP relaxation，得到最优解 $x_v^*$；
2. 返回：

$$
S=\{v\in V:x_v^*\ge \frac12\}
$$

### Correctness

需要证明 $S$ 是 vertex cover。

考虑任意边 $(u,v)\in E$。由于 LP feasible：

$$
x_u^*+x_v^*\ge 1
$$

所以 $x_u^*$ 和 $x_v^*$ 中至少有一个不小于 $\frac12$。

因此 $u$ 或 $v$ 至少有一个被加入 $S$，这条边被 cover。

因为任意边都成立，所以 $S$ 是 vertex cover。

### Approximation Ratio

要证明：

$$
|S|\le 2\cdot OPT(IP)
$$

由于：

$$
OPT(LP)\le OPT(IP)
$$

只需要证明：

$$
|S|\le 2\cdot OPT(LP)
$$

对 LP 最优解：

$$
OPT(LP)=\sum_{v\in V}x_v^*
$$

而所有被选入 $S$ 的点都满足 $x_v^*\ge \frac12$，所以：

$$
OPT(LP)
=
\sum_{v\in V}x_v^*
\ge
\sum_{v\in S}x_v^*
\ge
\frac12 |S|
$$

因此：

$$
|S|\le 2\cdot OPT(LP)\le 2\cdot OPT(IP)
$$

关键结论：

> LP rounding 给出了 Minimum Vertex Cover 的 $2$-approximation。

补充：如果 Unique Games Conjecture 为真，Vertex Cover 不能被近似到小于 $2$ 的因子。

## Dual of Vertex Cover LP

为了写 dual，通常把 vertex cover LP 中的上界 $x_v\le 1$ 省略，因为在 minimization optimum 中不会需要 $x_v>1$。

Primal：

$$
\begin{aligned}
\text{minimize}\quad & \sum_{v\in V}x_v\\
\text{subject to}\quad
& x_u+x_v\ge 1 && \forall (u,v)\in E\\
& x_v\ge 0 && \forall v\in V
\end{aligned}
$$

Dual：

$$
\begin{aligned}
\text{maximize}\quad & \sum_{e\in E}y_e\\
\text{subject to}\quad
& \sum_{e\ni v} y_e\le 1 && \forall v\in V\\
& y_e\ge 0 && \forall e\in E
\end{aligned}
$$

这个 dual 可以理解成 fractional matching：

- 每条边 $e$ 有一个权重 $y_e$；
- 每个点 incident 的边权重总和最多是 $1$。

如果再要求 $y_e\in\{0,1\}$，就是 matching 的 integer program。

## Primal-Dual Analysis for Greedy Matching

考虑一个简单 greedy matching algorithm：

- 按某个顺序处理右侧点 $B$；
- 每次把当前点匹配给任意一个尚未匹配的 neighbor；
- 如果没有 unmatched neighbor，就跳过。

这个 greedy 结果是一个 maximal matching。它不一定 maximum，但可以证明是 $2$-approximation，也就是：

$$
ALG\ge \frac12 OPT
$$

### Gain Sharing

对 greedy 找到的每条 matched edge $(u,v)$，把 $1$ 的 gain 平均分给两个端点：

$$
x_u\leftarrow x_u+\frac12,\quad x_v\leftarrow x_v+\frac12
$$

因此：

$$
\sum_{v\in V}x_v=ALG
$$

因为 matching 中每个点最多出现一次，所以每个 $x_v$ 只可能是 $0$ 或 $\frac12$。

### Almost Feasible Vertex Cover

由于 greedy matching 是 maximal matching，任意一条边 $(u,v)$ 至少有一个 endpoint 已经被匹配。

否则，如果 $u$ 和 $v$ 都 unmatched，那么 greedy 结束后还可以把这条边加入 matching，矛盾。

因此对于任意边 $(u,v)$：

$$
x_u+x_v\ge \frac12
$$

所以 $2x$ 是 vertex cover LP 的 feasible solution：

$$
2x_u+2x_v\ge 1
$$

于是：

$$
2ALG
=
\sum_{v\in V}2x_v
\ge
OPT(\text{Vertex Cover LP})
$$

由 weak duality，任意 matching 的大小都不超过任意 vertex cover 的大小，特别是：

$$
OPT(\text{Matching})\le OPT(\text{Vertex Cover IP})
$$

更直接地，$2x$ 是一个 fractional vertex cover，所以它给出了 matching optimum 的 upper bound：

$$
OPT\le 2ALG
$$

因此：

$$
ALG\ge \frac12 OPT
$$

关键结论：

> greedy maximal matching 是 maximum matching 的 $1/2$-approximation；  
> 换成 minimization 的说法，它对应一个 $2$-approximation 的 bound。

## Ranking Algorithm for Online Bipartite Matching

Ranking Algorithm 是 Karp, Vazirani, and Vazirani 在 1990 年提出的 online matching algorithm。

问题设置：

- 左侧点 $A$ 一开始已知；
- 右侧点 $B$ 按 online order 一个个到来；
- 当一个 $b\in B$ 到来时，必须立即决定是否把它匹配给某个 unmatched neighbor；
- 决定之后不能修改。

Greedy 可能只得到 $1/2$ competitive ratio。

Ranking 的做法：

1. 对所有 $a\in A$ 随机生成一个 rank：

$$
r_a\in [0,1)
$$

2. 当 $b\in B$ 到来时，把它匹配给所有 unmatched neighbors 中 rank 最小的那个。

关键结论：

> Ranking Algorithm 是 $\left(1-\frac1e\right)$-competitive。

也就是说：

$$
\mathbb E[ALG]\ge \left(1-\frac1e\right)OPT
$$

### Primal-Dual View

Ranking 的经典分析比较复杂，但可以用 primal-dual gain sharing 简化。

当 algorithm 匹配边 $(u,v)$ 时，设 $v$ 的 rank 是 $r_v$。定义 gain sharing function：

$$
g(r)=e^{r-1}
$$

把这条 matched edge 的 $1$ 单位 gain 分给两端：

$$
y_v=g(r_v)
$$

$$
y_u=1-g(r_v)
$$

总 gain 仍然是：

$$
y_u+y_v=1
$$

所以：

$$
\sum_z y_z=ALG
$$

目标是证明对任意 edge $(u,v)$：

$$
\mathbb E[y_u+y_v]\ge 1-\frac1e
$$

如果成立，那么把期望 gain 放大：

$$
\frac{e}{e-1}\mathbb E[y_z]
$$

就得到一个 feasible fractional vertex cover。于是：

$$
\frac{e}{e-1}\mathbb E[ALG]\ge OPT
$$

等价于：

$$
\mathbb E[ALG]\ge \left(1-\frac1e\right)OPT
$$

### Gain Lower Bound

固定一条边 $(u,v)$，并固定除了 $v$ 以外所有左侧点的 rank。

考虑在删除 $v$ 后，$u$ 的匹配情况。

Case 1：$u$ unmatched。

把 $v$ 加回来后，无论 $r_v$ 是多少，$v$ 都会被某个点匹配。因此：

$$
\mathbb E[y_u+y_v]
\ge
\int_0^1 g(r)\,dr
$$

因为 $g(r)=e^{r-1}$：

$$
\int_0^1 e^{r-1}\,dr
=
1-\frac1e
$$

Case 2：$u$ 原本匹配到某个点 $v'$，其 rank 为：

$$
r_{v'}=\theta
$$

如果把 $v$ 加回来，并且 $r_v<\theta$，那么 $v$ 足够好，会造成一串 replacement chain。无论链如何传播，最终可以保证：

$$
y_v\ge g(r_v)\quad \text{for } r_v<\theta
$$

并且：

$$
y_u\ge 1-g(\theta)
$$

所以：

$$
\mathbb E[y_u+y_v]
\ge
\int_0^\theta g(r)\,dr+1-g(\theta)
$$

代入 $g(r)=e^{r-1}$：

$$
\int_0^\theta e^{r-1}\,dr+1-e^{\theta-1}
=
\left(e^{\theta-1}-\frac1e\right)+1-e^{\theta-1}
=
1-\frac1e
$$

因此两种情况都得到：

$$
\mathbb E[y_u+y_v]\ge 1-\frac1e
$$

最终：

$$
\mathbb E[ALG]\ge \left(1-\frac1e\right)OPT
$$

## Summary

本讲的核心结论：

- LP 是线性 objective 加线性 constraints 的优化问题；
- bounded feasible LP 的最优解可以在 vertex 上取得；
- Simplex Method 在实践中很快，但 worst case 是 exponential；
- LP 可以在 polynomial time 内求解；
- standard form 是

$$
\max \mathbf c^\top\mathbf x
\quad
\text{s.t. }
A\mathbf x\le \mathbf b,\ \mathbf x\ge 0
$$

- primal-dual pair 满足 weak duality：

$$
\mathbf c^\top\mathbf x\le \mathbf b^\top\mathbf y
$$

- strong duality 说明最优 primal value 等于最优 dual value；
- LP-relaxation 可以把 NP-hard 的 IP 放松成 LP，再通过 rounding 得到 approximation algorithm；
- Minimum Vertex Cover 的 LP rounding 给出 $2$-approximation；
- primal-dual analysis 可以用 dual certificate 解释 matching 和 online matching 的 approximation/competitive ratio。
