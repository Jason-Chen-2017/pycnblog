
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在经典动态规划问题中，有一个叫做“最长递增子序列(LIS)”的问题，它的目标就是找出给定数组中的一个最长的单调递增子序列。而其关键点是寻找一种方法能够在线性时间内计算出解。

另外一个经典动态规划问题则是“最长公共子串（Longest Common Subsequence，LCST）”。它的目标是在两个字符串之间找到最长的公共子序列，同时要求子序列是严格单调上升或下降的。

这两个问题都是很多高频出现在面试和笔试题目中，也是不容错过的问题。

“LeetCode 918: Maximum Sum Circular Subarray”，顾名思义，是个和最长递增子序列一样的动态规划问题，只不过此时要求得的是一个圆形数组中的最大的连续子序列和，而不是单独的数组的最长递增子序列。

为了解决这个问题，我们可以考虑如何去理解什么是“最长连续子序列和”。一个连续子序列和就是所有元素都相邻且按照顺序排列的连续子数组的和。例如，[1,-2,3,10,-4,7,2,-5]中的最大连续子序列和为3+10+7=17。

然而对于一个圆形数组来说，最大连续子序列和可能出现在头尾不同的地方。举例如下：

[-2, 1,-3, 4,-1, 2, 1,-5, 4]

其中最大连续子序列和为4+(-1)+1+(-5)=2。从数组的末端往前看，子序列[-5, 4,-1, 2, 1]即使比从数组开头向后看的子序列[-2, 1,-3, 4]要长，但是从圆心移动到其他位置也不会改变其长度。

因此，对于一个圆形数组，我们需要找到一种方式来判断哪些子序列才是可行的，并且最大化它们的和。


# 2.基本概念术语说明
## (1).动态规划
动态规划（Dynamic Programming）是指利用已经确定的结果，通过推导关系来建立状态转移方程，从而解决复杂问题的方法论。其方法是把复杂问题分成规模较小、重叠子问题多的子问题，然后逐步地递归求解这些子问题，最后得到整个问题的解。由于子问题的重复性，如果保存之前的解，就可以避免重新解。在许多应用中，动态规划被广泛使用。动态规划具有以下几个特点：

1. 最优子结构：一个问题的最优解所包含的子问题的解也是最优的。
2. 无后效性：一旦某个状态的计算过程确定，就不再受该状态以后的影响，也就是说一旦某个状态被计算出来，它的值将不会再发生变化。
3. 子问题重叠性：子问题在不同阶段的求解是重复的。动态规划设计的目的就是减少重复计算，使之只执行一次。

## （2）循环
一个序列循环的定义是：一个序列中任意一个元素都可以作为起始元素来生成相同的序列。比如，数字1、2、3、4的循环是12341234……，一个整数的循环可以是：任何一个整数加上1之后得到的整数的循环。一个圆环是指一个首尾相接的线段，它既可以表现为一个回圈，又可以表现为一个环状曲线。例如，圆环上的点的顺序总是以相同的方式分布。

## （3）最大连续子序列和
给定一个序列A=(a1,a2,...,an)，那么最大连续子序列和(LCS-sum)定义为：
```
    LCS-sum = max{ LCS-sum(i-2,j), LCS-sum(i-1,j-1), a_i + b_j }
```
其中`b_j`表示仅包含第一个序列中的第`j`个数的序列。

当序列B=(b1,b2,...,bn)和C=(c1,c2,...,cm)分别代表另一个序列，最大循环连续子序列和(LCCS-sum)定义为：
```
    LCCS-sum = min{ max{ LCS-sum(k,n), LCS-sum((k+m)%n, k%n) }, i < j <= n}
```
其中`m`，`n`分别代表第二个序列的长度和第三个序列的长度，`(k+m)`取模`n`，表示将第二个序列翻转得到的新序列的起始索引。

## （4）空间优化
对于一个序列A，它的最大连续子序列和可以通过如下三个变量来计算：
```
    dp[i][0],dp[i][1]: 表示在A[:i+1]的情况下，包含当前元素的最大连续子序列和，不包含当前元素的最大连续子序列和；
    dp[i][2],dp[i][3]: 表示在A[:i+1]的情况下，包含当前元素的最小连续子序列和，不包含当前元素的最小连续子序列和；
```
那么问题的关键就是如何维护这些变量，使得每次更新的时候，只需要关注上一次更新得到的结果即可。在本文中，将会使用滚动数组技巧对空间进行优化。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## （1）算法描述
首先定义两个函数：

- `func maxSumSubarray(arr):` 返回以arr[i]结尾的最大连续子序列和

    - 初始化：
        ```
            curMax = arr[0];
            preMax = arr[0];
            
            res = [preMax,curMax]; //记录着以arr[i]结尾的最大连续子序列和以及不包含arr[i]的最大连续子序列和
        ```
        
    - 循环：遍历数组的每一个元素，计算以该元素结尾的最大连续子序列和并比较是否大于等于之前存储的结果
        
        - 如果当前元素加上之前包含当前元素的最大连续子序列和大于之前不包含当前元素的最大连续子序列和，那么说明保留当前元素意义不大，直接用之前不包含当前元素的最大连续子序列和更新结果
            
        - 如果当前元素加上之前包含当前元素的最大连续子序列和小于之前不包含当前元素的最大连续子序列和，那么说明增加当前元素的贡献最大，将当前元素和之前包含当前元素的最大连续子序列和更新结果

        - 将之前的结果更新为现在的结果
        
- `func maxSumCircularSubarray(arr):` 返回以arr[0]结尾的最大循环连续子序列和
    
    - 初始化：
        ```
            lenArr = len(arr);
            
            if lenArr == 1:
                return arr[0];
            
            #先算出以arr[0]结尾的最大连续子序列和
            tempRes = maxSumSubarray([0]+arr[:-1]);
            tmpMax = max(tempRes[0],tempRes[1]);
            preMin = min(tempRes[0],tempRes[1]);
            curMax = sum(arr);
            
            #继续计算以其他元素结尾的最大连续子序列和
            for i in range(1,lenArr-1):
                nextTempRes = maxSumSubarray([0]+arr[:i]+[j]+arr[i:] for j in set(arr)-set(arr[i:]) )
                nextMax = max(nextTempRes[0],nextTempRes[1]);
                
                if nextMax > curMax or nextMax >= tmpMax:
                    continue;
                
                if abs(tmpMax-preMin)*2<abs(nextMax-curMax):
                    break;
                    
                curMax = nextMax;
                
                if curMax > tmpMax:
                    preMin = tmpMax;
                    tmpMax = curMax;
            
            return curMax;
        ```
        
    - 循环：对每个元素j，以j结尾的最大循环连续子序列和等于以j结尾的最大连续子序列和和由[0...j]和[j...i,0...m-i-1]（其中m为数组长度）组成的组合的最大值；
        
        - 筛选条件是若某种组合产生的子序列和比之前的子序列要长，或者组合的子序列和等于之前的子序列，则跳过该组合；
        
        - 此处的组合包括：
            - 以j结尾的最大连续子序列和，和以j及arr[i-2]至arr[i]-1的最大连续子序列和，以及一个仅包含arr[i]的子序列
            - 以j结尾的最大连续子序列和，和以j及arr[(i+1)%n]至arr[i]+arr[(i+1)%n]-(arr[i+1])的最大连续子序列和，以及一个仅包含arr[i]的子序列
        
        - 如果满足筛选条件，则对其中的两种情况分别求出其最大值，选出其中的更大值作为新的候选值；
            
        - 对于每种情况，若新的候选值大于之前的最大值，或新的候选值和之前的最大值差值比之前的差值小，则更新最大值和最小值；
        
    - 返回结果

## （2）实现
Python语言实现如下：

```python
def maxSumSubarray(arr):
    """
    Return the maximum subarray sum ending with each element of arr.
    """
    curMax = arr[0];
    preMax = arr[0];
    res = [preMax,curMax];
    
    for num in arr[1:]:
        curMax = max(num, num+curMax);
        preMax = max(preMax, curMax);
        res.append(curMax);
    
    return res
    
def maxSumCircularSubarray(arr):
    """
    Returns the maximum circular subarray sum.
    """
    lenArr = len(arr);
    
    if lenArr == 1:
        return arr[0];
    
    # Calculate the maximum subarray sum ending with arr[0].
    tempRes = maxSumSubarray([0]+arr[:-1]);
    tmpMax = max(tempRes[0],tempRes[1]);
    preMin = min(tempRes[0],tempRes[1]);
    curMax = sum(arr);
    
    # Iterate over all elements and calculate their contribution to 
    # the maximum looped contiguous subarray sum.
    for i in range(1,lenArr-1):
        nextTempRes = maxSumSubarray([0]+arr[:i]+[j]+arr[i:] for j in set(arr)-set(arr[i:]) )
        nextMax = max(max(nextTempRes)[0],max(nextTempRes)[1]);
        
        if nextMax > curMax or nextMax >= tmpMax:
            continue;
        
        if abs(tmpMax-preMin)*2<abs(nextMax-curMax):
            break;
            
        curMax = nextMax;
        
        if curMax > tmpMax:
            preMin = tmpMax;
            tmpMax = curMax;
    
    return curMax;
```