## 背包问题 Knapsnap Problem

- **Input**: $n$ items with cost $c_i$ and value $v_i$ and a capacity $W$.
- **Output**: A subset of items such that $\sum c_i \le W$ and $\sum v_i$ is maximized.

| Item | Value | Cost |
| --- | ---: | ---: |
| iPhone | 8888 | 8888 |
| Algorithm Book | 10000 | 500 |
| Laptop | 8888 | 8500 |
| Hermes | 90000 | 100000 |

### A greedy approach
Select the item from larger *value-cost ratio*(最高性价比)

When the greedy approach fails?

Because the item and value are discrete, we can't divide the item. How to spend the capacity $W$ is a **NP-HARD** problem(namely how can we find a subset of $c_i$ that their sum exactly equals $W$)

### DP approach 
**Subproblem**: $f[i,w]$ is the maximum value we can get by using the first $i$ items and with total budget $w$.

**Transition Formula**

- What we always do before:
  Define the state clearly.
- $f[i,w]$: the maximum value we can get by using the first $i$ items, and with $w$ budget.

- Two options for item $i$:
  - **Buy it**: We can use at most $w-c_i$ budget before item $i$.
  - **Not buy it**: We can use at most $w$ budget before item $i$.

- Therefore,

$$
f[i,w] = \max\{f[i-1,w], f[i-1,w-c_i] + v_i\}
$$

when $w \ge c_i$.

If $w < c_i$, then

$$
f[i,w] = f[i-1,w]
$$

### Check the topological order

- $f[i,w]$ only depends on states in row $i-1$.
- So we can fill the DP table row by row, from small $i$ to large $i$.
- Inside each row, enumerate budget $w=0,1,2,\dots,W$.
- The final answer is $f[n,W]$.

| $f[i,w]$ | 0 | 1 | 2 | 3 | 4 | 5 | $\cdots$ | $W$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 |  |  |  |  |  |  |  |
| 2 | 0 |  |  |  |  |  |  |  |
| 3 | 0 |  |  |  |  |  |  |  |
| $\cdots$ | 0 |  |  |  |  |  |  |  |
| $n$ | 0 |  |  |  |  |  |  | $f[n,W]$ |

- Time complexity: $O(nW) \cdot O(1) = O(nW)$

### $O(nW)$ is NOT polynomial!
- Input Size: the unit of bits to represent the input.
- $W = 2^N$ by using $N$ bits. $O(w)$ is $O(2^N)$.
- So the complexity is $O(n2^N)$, which is exponential.

### Some approximal ideas
Some times $O(nW)$ is unbearable, can we use some approximal approach to quickly get answer near **OPT**?

1. Round cost $c_i$

2. Round value $v_i$
Let $V = max v_i$, then OPT <= nV.

Let $A[i, v]$ be the **minimum cost**  we spend to get value $v$ using the first $i$ items.

**Trasition Formula**:
$A[i+1, v] = min(A[i,v], A[i,v-v_i]+c_i)$

| Ori Value| Approxi value | cost
| --- | --- | --- |
| 86526 | 86000 -> 86 | 102435 |
| 25473 | 25000 -> 25 | 123543 |
| 87654 | 87000 -> 87 | 234525 |
| 65476 | 65000 -> 65 | 123543 |
| 25477 | 25000 -> 25 | 345756 |

### Divide value by $K$

为了让 value-based DP 的状态数变小，可以把每个价值按同一个尺度 $K$ 缩放：

$$
v_i' = \left\lfloor \frac{v_i}{K} \right\rfloor
$$

然后在缩放后的价值上做 DP。令 $V=\max_i v_i$，则任意物品的缩放后价值最多是 $\frac{V}{K}$，所有物品的缩放后总价值最多是：

$$
\sum_i v_i' \le \frac{nV}{K}
$$

所以如果继续使用

$$
A[i, v] = \text{minimum cost to get scaled value } v \text{ using first } i \text{ items}
$$

状态数为 $O(n \cdot \frac{nV}{K})$，每个状态 $O(1)$ 转移：

$$
\text{Time Complexity} = O\left(\frac{n^2V}{K}\right)
$$

空间如果只保留上一行，可以降到：

$$
O\left(\frac{nV}{K}\right)
$$

### Precision analysis

因为向下取整，每个物品的真实价值和缩放后价值满足：

$$
K v_i' \le v_i < K(v_i' + 1)
$$

也就是：

$$
v_i - K \le K v_i' \le v_i
$$

对任意一个包含不超过 $n$ 个物品的方案 $S$：

$$
\sum_{i\in S} K v_i' \ge \sum_{i\in S} v_i - nK
$$

设 $S^*$ 是原问题最优解，$S_A$ 是缩放后 DP 找到的解。由于 $S_A$ 在缩放价值上最优：

$$
\sum_{i\in S_A} v_i' \ge \sum_{i\in S^*} v_i'
$$

因此原始价值的下界为：

$$
value(S_A)
\ge K\sum_{i\in S_A} v_i'
\ge K\sum_{i\in S^*} v_i'
\ge OPT - nK
$$

结论：除以 $K$ 后，算法最多损失 $nK$ 的 additive error。

如果希望得到 $(1-\epsilon)$ 近似，可以取：

$$
K = \frac{\epsilon V}{n}
$$

在常见假设 $OPT \ge V$ 下（例如先去掉 $c_i>W$ 的不可行物品，并把 $V$ 理解为可行单物品的最大价值）：

$$
value(S_A) \ge OPT - \epsilon V \ge (1-\epsilon)OPT
$$

此时时间复杂度变为：

$$
O\left(\frac{n^2V}{\epsilon V/n}\right) = O\left(\frac{n^3}{\epsilon}\right)
$$

## Largest Number in k Consecutive Numbers
- **Input**: A sequence of n numbers 
- **Output**: Find the largest number in k consecutive numbers

1. 朴素的做法 :$O(nk)$

2. 使用堆维护: $O(nlogk)$

3. 动态规划
维护一个PLL列表，从前k个数开始，按降序排列，此先后移动一个元素，pop out最前面的，pop in新加入的元素。
新加入的元素在加入时，踢出比它小的数。

初步来看，每次最坏踢出k个数，总时间复杂度仍然是$O(nk)$?

均摊分析：当一个数($a_i$)踢出k个数，之后新加入的元素就不需要再与被踢出的数比较了，只需要与$a_i$比较。

从Charge的角度来看，每个数只会被pop in一次，比较并pop out一次，因此均摊时间复杂度为$O(n)$.

## Optimized longest increasing subsequence

在 lecture10 里，LIS 的基础 DP 是：

$$
lis[i] = 1 + \max_{\substack{j<i \\ a_j<a_i}} lis[j]
$$

它每次要枚举所有 $j<i$，所以复杂度是 $O(n^2)$。优化的关键是：我们不需要保留所有前缀，只需要保留未来还有可能成为最优前缀的那些状态。

### Potential Prefixes

定义：

$$
sm[len] = \text{the smallest ending number of an increasing subsequence with length } len
$$

也就是说，`sm[len]` 不是某一个固定位置的 LIS，而是所有长度为 `len` 的递增子序列中，最小的结尾值。

为什么取最小结尾值？因为在相同长度下，结尾越小，未来越容易接上新的数。例如长度都是 3，结尾为 5 的序列比结尾为 13 的序列更有潜力继续扩展。

初始化：

$$
sm[0] = -\infty,\quad sm[len] = +\infty \ (len \ge 1)
$$

### Updating `sm[len]`

处理新元素 $a_i$ 时，找到最大的 `len` 使得：

$$
sm[len] < a_i
$$

那么 $a_i$ 可以接在某个长度为 `len` 的递增子序列后面，形成长度为 `len+1` 的递增子序列：

$$
sm[len+1] = \min(sm[len+1], a_i)
$$

这正对应图里的两个 case：

- 如果 $a_i > sm[len]$，它可以 create a longer LIS。
- 如果 $a_i \le sm[len]$，它不能创建更长的 LIS，但可能把某个 `sm[len]` 更新得更小，让之后的数更容易接上。

`sm` 数组本身是递增的：

$$
sm[0] < sm[1] < sm[2] < \cdots
$$

所以可以用 binary search 找到更新位置。

### Binary Search version

等价写法：在当前 `sm[1..L]` 中找到第一个满足 `sm[pos] >= a_i` 的位置 `pos`，然后令：

$$
sm[pos] = a_i
$$

如果找不到这样的 `pos`，说明 $a_i$ 比所有当前 potential prefixes 的结尾都大，于是可以扩展 LIS 长度：

$$
L = L + 1,\quad sm[L] = a_i
$$

```text
// Input: sequence a[1..n]
// Output: length of LIS
Function Fast_LIS(a):
    L = 0
    sm[0] = -infinity
    
    for i from 1 to n:
        // find the first pos in [1, L] with sm[pos] >= a[i]
        pos = LowerBound(sm[1..L], a[i])
        
        if pos does not exist:
            L = L + 1
            sm[L] = a[i]
        else:
            sm[pos] = a[i]
            
    return L
```

这里使用 `LowerBound(sm, a[i])` 是为了处理 **strictly increasing**。如果遇到相同元素，新的元素会替换同一长度的结尾，而不会错误地把长度加一。

### Complexity

每个 $a_i$ 只做一次二分查找和一次更新：

$$
O(\log L) \le O(\log n)
$$

总时间复杂度：

$$
O(n\log n)
$$

空间复杂度：

$$
O(n)
$$
