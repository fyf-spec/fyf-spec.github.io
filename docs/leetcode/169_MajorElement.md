# [169. 多数元素](https://leetcode.cn/problems/majority-element/)

给定一个大小为 `n` 的数组 `nums` ，返回其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

 

**示例 1：**

```
输入：nums = [3,2,3]
输出：3
```

**示例 2：**

```
输入：nums = [2,2,1,1,1,2,2]
输出：2
```

**提示：**

- `n == nums.length`
- `1 <= n <= 5 * 104`
- `-109 <= nums[i] <= 109`
- 输入保证数组中一定有一个多数元素。
 

**进阶：**尝试设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题


# 题解
### 朴素哈希
遍历一遍整个列表，生成哈希表
```
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        unordered_map<int, int> counts;
        int majority=0, max_cnt=0;
        for(int num: nums){
            counts[num]++;
            if(counts[num] > max_cnt){
                majority = num;
                max_cnt = counts[num];
            }
        }
        return majority;
    }
};

```
哈希算法的时间复杂度和空间复杂度都为 $O(n)$

### Boyer-Moore投票法
有没有时间复杂度为$O(n)$，空间复杂度为$O(1)$的算法？

这道题定义众数是超过 `⌊ n/2 ⌋` 的数，这意味着其比剩下所有数个数的和都多，考虑到以下简单事实：如果一个数组有大于一半的数相同，那么任意删去两个不同的数字，新数组还是会有相同的性质。如果我们把众数记为 +1，把其他数记为 −1，将它们全部加起来，显然和大于 0，从结果本身我们可以看出众数比其他数多。

#### 过程
我们维护一个候选众数 `candidate` 和它出现的次数 `count`。初始时 `candidate` 可以为任意值，`count` 为 0；

我们遍历数组 nums 中的所有元素，对于每个元素 x，在判断 x 之前，如果 count 的值为 0，我们先将 x 的值赋予` candidate`，随后我们判断 x：

- 如果 x 与 candidate 相等，那么计数器 count 的值增加 1；

- 如果 x 与 candidate 不等，那么计数器 count 的值减少 1。

在遍历完成后，candidate 即为整个数组的众数。


