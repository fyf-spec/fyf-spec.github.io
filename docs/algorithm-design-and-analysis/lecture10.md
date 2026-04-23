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


