
# [189. 轮转数组](https://leetcode.cn/problems/rotate-array/)

给定一个整数数组 `nums`，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。

 

**示例 1:**

```
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]
```

**示例 2:**

```
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释: 
向右轮转 1 步: [99,-1,-100,3]
向右轮转 2 步: [3,99,-1,-100]
```

 

**提示：**

- `1 <= nums.length <= 105`
- `-231 <= nums[i] <= 231 - 1`
- `0 <= k <= 105`

 

**进阶：**

- 尽可能想出更多的解决方案，至少有 **三种** 不同的方法可以解决这个问题。
- 你可以使用空间复杂度为 `O(1)` 的 **原地** 算法解决这个问题吗？


### 朴素做法
另开一个数组，存储 `k % nums.size()` 个数的元素，移动完原数组元素之后，将k个数移动到数组的前面。

时间复杂度 $O(n)$，空间复杂度 $O(n)$

### 环状替换
朴素做法中，我们如果直接将一个元素放到它最后的位置上，那么那个位置上原来的元素就会被覆盖，那么如果我们不覆盖最终位置上的数，而是使用`temp`存储，从此位置开始继续往下遍历呢？这样空间开销只有一个`temp`，时间复杂度$O(n)$，空间复杂度$O(1)$。

- 另：这里是不是可以说借用了一点online的思维呢？

由于最终回到了起点，故该过程恰好走了整数数量的圈，不妨设为 `a` 圈；再设该过程总共遍历了 `b` 个元素。因此，我们有 `an=bk`，即 `an` 一定为 `n`,`k` 的公倍数。又因为我们在第一次回到起点时就结束，因此 `a` 要尽可能小，故 `an` 就是 `n`,`k` 的最小公倍数 `lcm(n,k)`，因此 `b` 就为 `lcm(n,k)/k`

这说明单次遍历会访问到 `lcm(n,k) / k`个元素，我们只需要计算一共需要多少次遍历即可，即 `n / (lcm(n,k) / k) = n * k / lcm(n,k) = gcd(n,k)` 次。

```
class Solution{
public:
    void rotate(vector<int>& nums, int k){
        int n = nums.size();
        k %= n;
        int count = 0;
        for(int start=0; count<n; start++){
            int current = start;
            int prev = nums[start];
            do{
                int next = (current + k) % n;
                int temp = nums[next];
                nums[next] = prev;
                prev = temp;
                current = next;
                count++;
            }while(start != current);
        }
    }
};
```


### 翻转法
我们同图例来形象地理解这个问题：

nums = "----->-->"; k =3
result = "-->----->";

reverse "----->-->" we can get "<--<-----"
reverse "<--" we can get "--><-----"
reverse "<-----" we can get "-->----->"

对于每次翻转，我们需要遍历翻转数组中所有元素，时间复杂度为$O(n)$，一共需要翻转3次，因此时间复杂度为$O(n)$，空间复杂度为$O(1)$。

```
class Solution{
public:
    void rotate(vector<int>& nums, int k){
        int n = nums.size();
        k %= n;
        reverse(nums.begin(), nums.end());
        reverse(nums.begin(), nums.begin() + k);
        reverse(nums.begin() + k, nums.end());
    }
};
```



