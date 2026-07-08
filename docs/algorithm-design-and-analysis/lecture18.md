# P, NP, NP-Completeness, and Reductions

这一讲进入计算复杂性理论的核心主题：P、NP、NP-hard、NP-complete，以及 polynomial-time reduction。

主线是：

- 先把问题限制为 decision problem；
- 用 Turing Machine 给“多项式时间算法”一个形式化模型；
- 定义复杂度类 P 和 NP；
- 用 Karp reduction 比较问题难度；
- 证明 SAT、3SAT、Independent Set、Vertex Cover 等问题之间的关系；
- 理解 Cook-Levin Theorem 和 NP-completeness 的意义。

## Famous Hard Problems

很多组合优化问题看起来都很自然，但目前没有已知的多项式时间算法。课件先列出几个典型 NP-hard / NP-complete 问题。

### SAT

SAT 是 Boolean Satisfiability Problem。

一个 Boolean formula 由变量、括号和逻辑运算构成：

- AND：$\land$
- OR：$\lor$
- NOT：$\neg$

一个公式是 **CNF** (conjunctive normal form)，如果它是若干个 clause 的 AND；每个 clause 是若干 literal 的 OR。

literal 指：

- 变量 $x_i$；
- 或变量的否定 $\neg x_i$。

例如：

$$
(x_1\lor x_3\lor \neg x_4)
\land
(x_2\lor \neg x_3)
\land
(\neg x_1\lor \neg x_2)
$$

是 CNF formula。

**SAT Problem**：

> 给定一个 CNF formula $\phi$，判断是否存在一组变量赋值，使得 $\phi$ 为 true。

### Vertex Cover

给定无向图：

$$
G=(V,E)
$$

一个点集 $S\subseteq V$ 是 vertex cover，如果每条边都至少有一个端点在 $S$ 中。

**Vertex Cover Problem**：

> 给定无向图 $G=(V,E)$ 和整数 $k$，判断 $G$ 是否存在大小为 $k$ 的 vertex cover。

更常见的 decision 版本也会写成“大小至多为 $k$”。课件里写成 size $k$，在很多归约中通过微调参数可以互相转换。

### Independent Set

给定无向图 $G=(V,E)$，一个点集 $S\subseteq V$ 是 independent set，如果 $S$ 中任意两个点之间都没有边。

**Independent Set Problem**：

> 给定无向图 $G=(V,E)$ 和整数 $k$，判断 $G$ 是否存在大小为 $k$ 的 independent set。

### Subset Sum

**Subset Sum Problem**：

> 给定整数集合 $S=\{a_1,\ldots,a_n\}$ 和目标值 $k$，判断是否存在子集 $T\subseteq S$，使得
>
> $$
> \sum_{a_i\in T}a_i=k
> $$

例如：

$$
S=\{1,1,6,13,27\},\quad k=21
$$

是 yes instance，因为：

$$
1+1+6+13=21
$$

但同一个 $S$ 对 $k=22$ 是 no instance。

### Hamiltonian Path

给定无向图 $G=(V,E)$，Hamiltonian path 是一条经过每个顶点恰好一次的路径。

**Hamiltonian Path Problem**：

> 给定无向图 $G=(V,E)$，判断图中是否存在 Hamiltonian path。

## Decision Problems

这一讲只讨论两类事情：

1. 输出 yes/no 的 decision problem；
2. polynomial time 和非 polynomial time 的区分。

不再关心 $O(n)$ 和 $O(n^2)$ 的细微差别，只关心是否能在 $n^{O(1)}$ 时间内解决。

### Formal Definition

一个 decision problem 是函数：

$$
f:\Sigma^*\to\{0,1\}
$$

其中：

- $\Sigma$ 是 alphabet，比如 $\Sigma=\{0,1\}$；
- $\Sigma^n$ 是长度为 $n$ 的所有字符串；
- $\Sigma^*=\bigcup_{n=0}^{\infty}\Sigma^n$ 是所有有限长度字符串；
- $x\in\Sigma^*$ 是一个 instance。

如果：

$$
f(x)=1
$$

则 $x$ 是 yes instance。

如果：

$$
f(x)=0
$$

则 $x$ 是 no instance。无效编码也可以统一看作 no instance。

例如 Vertex Cover 中，一个字符串 $x$ 可以编码一张图 $G$ 和整数 $k$：

- 如果 $G$ 有大小为 $k$ 的 vertex cover，则 $f(x)=1$；
- 否则 $f(x)=0$。

## Turing Machine

为了形式化“算法”和“多项式时间”，课件使用 Turing Machine。

在这门课里，可以把 Turing Machine 理解成一种理想化的计算机程序。它和普通程序在多项式时间计算能力上等价：

> 能被普通算法在 polynomial time 内计算的东西，也能被 Turing Machine 在 polynomial time 内计算。

### Basic Components

一个 Turing Machine 包含：

| 组成 | 含义 |
| --- | --- |
| tape | 无限长纸带，由许多 cell 组成，每个 cell 存一个 alphabet |
| head | 指向当前读写位置，可以左移或右移 |
| states | 有限个状态，表示机器当前处在哪一步 |
| transition function | 根据当前状态和读到的符号，决定写什么、去哪个状态、向左还是向右移动 |

形式上可写成：

$$
(Q,\Sigma,\delta)
$$

其中：

$$
\delta:Q\times\Sigma\to Q\times\Sigma\times\{L,R\}
$$

### Start and Halt

Turing Machine 从特殊状态 $q_{start}$ 开始运行，输入字符串放在 tape 上，head 指向第一个 cell。

它有两个特殊 halting states：

- $q_{acc}$：accept；
- $q_{rej}$：reject。

当机器进入 halting state 时停止。

### Polynomial Time TM

一个 Turing Machine $\mathcal A$ 是 polynomial time TM，如果存在某个多项式 $p$，使得对任意输入 $x$，$\mathcal A$ 都会在：

$$
p(|x|)
$$

步以内停机。

这里 $|x|$ 是输入字符串长度。

## The Complexity Class P

**P** 是 polynomial-time decidable problems 的集合。

形式化定义：

> 一个 decision problem $f:\Sigma^*\to\{0,1\}$ 属于 P，如果存在 polynomial time Turing Machine $\mathcal A$，使得：
>
> - 当 $f(x)=1$ 时，$\mathcal A$ accepts $x$；
> - 当 $f(x)=0$ 时，$\mathcal A$ rejects $x$。

也就是说，P 中的问题是我们认为“efficiently solvable”的问题。

### Examples in P

#### PATH

**PATH**：

> 给定图 $G=(V,E)$ 和两个点 $s,t\in V$，判断是否存在从 $s$ 到 $t$ 的 path。

算法：从 $s$ 开始跑 BFS 或 DFS。如果访问到 $t$，accept；否则 reject。

所以：

$$
PATH\in P
$$

#### k-FLOW

**k-FLOW**：

> 给定有向图 $G=(V,E)$、源点 $s$、汇点 $t$、容量函数 $c:E\to\mathbb R_+$ 和目标值 $k$，判断是否存在 value 至少为 $k$ 的 flow。

可以用 Edmonds-Karp、Dinic 等最大流算法求最大流，再和 $k$ 比较。

所以：

$$
k\text{-FLOW}\in P
$$

#### PRIME

**PRIME**：

> 给定一个以二进制编码的正整数 $k$，判断 $k$ 是否为素数。

Agrawal, Kayal and Saxena 在 2004 年给出 polynomial time primality test。

所以：

$$
PRIME\in P
$$

## Searching Problems and Verification

很多算法课中的问题本质上是 searching problem：

- 给定 instance $x$；
- 有很多可能的 solution $y$；
- 如果存在一个正确的 $y$，答案就是 yes；
- 如果所有 $y$ 都不正确，答案就是 no。

这类问题通常有两个任务：

| 任务 | 含义 |
| --- | --- |
| search | 找到一个正确解 |
| verify | 检查给定解是否正确 |

有些问题 search 容易，比如 shortest path、matching；有些问题 search 看起来很难，比如 Hamiltonian Path、Vertex Cover。

NP 试图刻画的是：

> 虽然不一定能快速找到解，但如果别人给你一个候选解，你能快速检查它是否正确。

## The Complexity Class NP

NP 可以从 verifier 的角度定义。

形式化定义：

> 一个 decision problem $f:\Sigma^*\to\{0,1\}$ 属于 NP，如果存在 polynomial time verifier $\mathcal A$，使得：
>
> 1. 如果 $x$ 是 yes instance，则存在 polynomial size 的字符串 $y$，使得 $\mathcal A$ accepts $(x,y)$；
> 2. 如果 $x$ 是 no instance，则对所有 polynomial size 的字符串 $y$，$\mathcal A$ 都 rejects $(x,y)$。

这里的 $y$ 叫做 **certificate**。

直观理解：

- yes instance 必须有一个短证据；
- 证据给出来后，可以快速验证；
- no instance 不能有任何假证据骗过 verifier。

SAT、Vertex Cover、Independent Set、Subset Sum、Hamiltonian Path 都在 NP。

### Verifier for Vertex Cover

对于 Vertex Cover：

- instance $x=(G,k)$；
- certificate $y$ 是一个点集 $S$。

verifier 做两件事：

1. 检查 $|S|=k$；
2. 检查每条边是否至少有一个端点在 $S$ 中。

这显然可以在 polynomial time 内完成。

因此：

$$
VertexCover\in NP
$$

### P Is Contained in NP

结论：

$$
P\subseteq NP
$$

关键证明思路：

NP 只要求 **yes instance 有一个短证据，并且这个证据可以在 polynomial time 内验证**。

但如果一个问题已经在 P 中，那说明它有 polynomial time algorithm 可以直接判断 yes/no。
也就是说，它甚至不需要 certificate。

因此对任意 $f\in P$：

- 如果 $x$ 是 yes instance，就取空证书 $y=\emptyset$，然后直接运行原来的 polynomial time algorithm，它会 accept；
- 如果 $x$ 是 no instance，不管给什么 certificate，原来的算法都会 reject。

所以 P 中的每个问题也满足 NP 的 verifier 定义：

$$
f\in NP
$$

因此：

$$
P\subseteq NP
$$

## P vs NP

核心开放问题：

$$
P\stackrel{?}{=}NP
$$

换句话说：

> 如果一个答案可以快速验证，那么这个答案是否也可以快速找到？

目前大多数研究者相信：

$$
P\ne NP
$$

直觉例子：

- 做一道难题可能很难；
- 但检查别人给出的完整解答是否正确，可能容易得多。

如果 $P=NP$，那么所有“容易验证”的搜索问题也都能高效求解，这会带来非常强的后果。

## Polynomial-Time Reduction

为了比较 NP 中不同问题的难度，需要 reduction。

核心思想：

> 如果我们能把问题 $f$ 的任意 instance 快速转换成问题 $g$ 的 instance，并且答案保持一致，那么只要会解 $g$，就会解 $f$。

这时说：

$$
f\le_K g
$$

读作：$f$ Karp reduces to $g$。

也可以理解为：

> $g$ 至少和 $f$ 一样难。

### Formal Definition

decision problem $f$ Karp reduce to decision problem $g$，如果存在 polynomial time Turing Machine $\mathcal A$，使得：

- 当输入 $x$ 是 $f$ 的 yes instance 时，$\mathcal A(x)$ 是 $g$ 的 yes instance；
- 当输入 $x$ 是 $f$ 的 no instance 时，$\mathcal A(x)$ 是 $g$ 的 no instance。

等价地：

$$
f(x)=g(\mathcal A(x))
$$

并且 $\mathcal A$ 的运行时间是 polynomial。

注意：

- yes 必须变 yes；
- no 必须变 no；
- yes 变 no、no 变 yes 不是这里的 Karp reduction。

### Reduction Gives Algorithms

如果：

$$
f\le_K g
$$

且：

$$
g\in P
$$

那么：

$$
f\in P
$$

算法就是：

1. 把 $f$ 的输入 $x$ 转成 $g$ 的输入 $\mathcal A(x)$；
2. 用 $g$ 的 polynomial time solver 求答案；
3. 输出同样的 yes/no。

### Transitivity

Reduction 具有传递性：

如果：

$$
f\le_K g
$$

并且：

$$
g\le_K h
$$

那么：

$$
f\le_K h
$$

原因是两个 polynomial time 转换可以串起来，polynomial 的复合仍然是 polynomial。

## SAT Reduces to 3SAT

3SAT 是 SAT 的限制版本。

一个 CNF formula 是 **3-CNF**，如果每个 clause 至多包含 3 个 literals。

**3SAT**：

> 给定一个 3-CNF formula，判断是否存在变量赋值使公式为 true。

显然：

$$
3SAT\le_K SAT
$$

因为 3-CNF 也是 CNF。

更重要的是反方向：

$$
SAT\le_K 3SAT
$$

这说明 SAT 和 3SAT 在多项式时间意义下“同样难”。

### Idea

给定任意 CNF formula $\phi$，构造一个 3-CNF formula $\phi'$，使得：

$$
\phi \text{ satisfiable}
\iff
\phi' \text{ satisfiable}
$$

并且构造过程是 polynomial time。

关键是把过长的 clause 拆成多个长度至多为 3 的 clause。

### Breaking a Long Clause

例如四个 literal 的 clause：

$$
x_1\lor x_2\lor \neg x_3\lor \neg x_4
$$

可以引入新变量 $y_1$，改写为：

$$
(x_1\lor x_2\lor y_1)
\land
(\neg y_1\lor \neg x_3\lor \neg x_4)
$$

这个改写保持 satisfiability：

- 如果原 clause 中前半部分有 true，可以令 $y_1=false$；
- 如果后半部分有 true，可以令 $y_1=true$；
- 如果原 clause 全 false，则两个新 clause 无法同时为 true。

更一般地，对一个长度为 $k$ 的 clause：

$$
\ell_1\lor\ell_2\lor\cdots\lor\ell_k
$$

可以引入新变量 $y_1,\ldots,y_{k-3}$，改写为：

$$
(\ell_1\lor \ell_2\lor y_1)
\land
(\neg y_1\lor \ell_3\lor y_2)
\land
\cdots
\land
(\neg y_{k-3}\lor \ell_{k-1}\lor \ell_k)
$$

如果某个 $\ell_i$ 为 true，可以适当设置 $y$ 变量让所有新 clause 为 true。

如果所有 $\ell_i$ 都为 false，则为了让第一个 clause 为 true 必须令 $y_1=true$，然后为了让第二个 clause 为 true 必须令 $y_2=true$，一路传下去，最后一个 clause 会失败。

所以：

$$
SAT\le_K 3SAT
$$

## 3SAT Reduces to Independent Set

接下来证明：

$$
3SAT\le_K IndependentSet
$$

### Construction

给定一个 3SAT instance：

$$
\phi=C_1\land C_2\land\cdots\land C_m
$$

每个 clause 有 3 个 literals。

构造图 $G=(V,E)$：

1. 对每个 clause $C_i$，建立一个 triangle；
2. triangle 的三个点分别表示这个 clause 中的三个 literals；
3. 同一个 triangle 内三个点两两相连；
4. 如果两个点代表互相矛盾的 literals，例如 $x$ 和 $\neg x$，就在它们之间连边；
5. 设：

$$
k=m
$$

也就是 clause 的数量。

### Why Yes Implies Yes

如果 $\phi$ satisfiable，那么每个 clause 至少有一个 literal 为 true。

从每个 triangle 中选一个值为 true 的 literal 对应的点，得到集合 $S$。

因为：

- 每个 triangle 只选一个点，所以不会选到同一个 clause 内的相邻点；
- 一个 truth assignment 不可能同时让 $x$ 和 $\neg x$ 为 true，所以不会选到互相矛盾的两个点；

因此 $S$ 是 independent set。

而且每个 clause 选一个点，所以：

$$
|S|=m=k
$$

所以 Independent Set instance 是 yes。

### Why Yes in Independent Set Implies Yes in 3SAT

反过来，如果构造出的图 $G$ 有大小为 $k=m$ 的 independent set $S$。

因为每个 triangle 内部是三角形，任意两个点相邻，所以一个 independent set 在每个 triangle 中最多选一个点。

但 $|S|=m$，一共有 $m$ 个 triangles，因此 $S$ 必须从每个 triangle 中恰好选一个点。

根据被选中的 literal 设置变量为 true。由于 independent set 不能同时包含 $x$ 和 $\neg x$，这个赋值不会自相矛盾。

未被赋值的变量可以任意赋值。

这样每个 clause 至少有一个被选中的 literal 为 true，因此 $\phi$ satisfiable。

所以：

$$
3SAT\le_K IndependentSet
$$

## NP-hard and NP-complete

现在可以定义“最难的 NP 问题”。

### NP-hard

一个 decision problem $f$ 是 **NP-hard**，如果对任意 $g\in NP$，都有：

$$
g\le_K f
$$

也就是说，NP 中所有问题都可以 polynomial-time reduce 到 $f$。

直观理解：

> 如果你会高效解决 $f$，你就会高效解决 NP 中所有问题。

### NP-complete

一个 decision problem $f$ 是 **NP-complete**，如果：

1. $f\in NP$；
2. $f$ 是 NP-hard。

也就是：

$$
NP\text{-complete}=NP\text{-hard}+in\ NP
$$

NP-complete 问题是 NP 里面最难的一批问题。

