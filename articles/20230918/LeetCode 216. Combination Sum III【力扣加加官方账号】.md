
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 题目描述
Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

**Example:** 

```python
Input: k = 3, n = 7
Output: [[1,2,4]]

Explanation:
1 + 2 + 4 = 7
Therefore, return [1,2,4].
```

```python
Input: k = 3, n = 9
Output: [[1,2,6], [1,3,5], [2,3,4]]

Explanation:
1 + 2 + 6 = 9
1 + 3 + 5 = 9
2 + 3 + 4 = 9
Thus, the output is [[1,2,6],[1,3,5],[2,3,4]].
```

Note:

1 <= k <= 9
1 <= n <= 30
You may assume that repeat numbers are allowed in the input list.

## 2.背景介绍
在这个问题中，我们需要找到所有可能的组合，将给定的数组元素之和等于给定的值n。但是，限定了输入数组只能使用从1到9中的数字，而且每个组合的数字也必须是唯一的。因此，我们需要用一个递归函数来解决这个问题。

## 3.基本概念和术语说明

- `k`表示待选个数（也就是数组元素的个数）；
- `n`表示目标值；
- `nums`表示候选集，即允许使用的数字集合；
- `cur`表示当前正在求解的组合列表；
- `temp`表示临时变量存储的组合列表。

## 4.核心算法原理和具体操作步骤

### **1、确定递归函数参数：**

考虑两种情况：

1. 如果剩余可选数字个数小于`k`，则停止递归；
2. 如果剩余可选数字个数大于等于`k`，则继续递归。

参数设置：

- `start`: 表示起始索引位置，默认为0；
- `path`: 当前路径（用于记录当前递归方向），默认为空列表[];

### **2、确定递归终止条件：**

当`path`长度等于`k`，且`sum(path)`等于`n`，则表示找到了一个组合，可以把其加入结果列表；或者当`path`长度大于`k`，或`sum(path)`大于`n`，则表示无需再往下搜索，停止递归。

### **3、选择递归路径：**

首先判断索引是否越界；然后循环遍历每一个可用的数字，并递归进入；如果已经达到了`k`个数字，就停止递归，因为此时的递归方向是固定的。

### **4、处理递归返回值：**

对于每一次递归，都会得到一个子组合列表；最后我们需要合并所有的子组合列表，才能得到最终结果。

### **5、实现代码如下：**

```python
def combination_sum3(k: int, n: int) -> List[List[int]]:
    nums = range(1, 10) # candidates

    def backtrack(start=0, path=[], temp=[]):
        if len(path) == k and sum(path) == n:
            res.append(path[:])
            return

        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue

            if sum(path) + nums[i] > n or (len(path) >= k and sum(path) + nums[i] < n):
                break

            path.append(nums[i])
            backtrack(i+1, path, temp)
            del path[-1]

    res = []
    backtrack()
    return res
```

**分析：**

该题使用回溯法求解，时间复杂度O(C(N,K))，其中N为数组元素总个数，K为数组元素的最大个数。回溯法的基本过程是：
- 从初始状态出发，一步步探索所有可能的路径；
- 当发现一条路不通的时候，就退回到上一步，进行回溯，寻找另一条路；
- 重复以上过程，直到找到所有路径或者已走过尽头。

回溯法的特点是：它会穷举所有的可能性，但由于搜索空间的限制，通常很快就会发现一些没用的分支；另一方面，它采用自顶向下的策略，减少了内存占用；并且，它不需要像动态规划那样多次计算，通过记忆化递归，可以进一步减少计算量。