
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 一、问题描述：
给定一个整数数组arr，其中元素的值域在[0, n)，n是arr的长度。arr中可能存在以下两种类型的数字：

1.正整数num（0< num <= n）
2.负整数num（-n < num <= 0）

对于arr中的任意位置i，如果num等于arr[i]，那么我们称该位置为num所在位置；否则称其为跳点。跳转规则如下：
当且仅当arr[i+num]存在时，才能从位置i跳到位置i+num，即下一次能到的位置只能是本次下跳的目标。

假设当前位置i处于跳点跳跃范围，即[left, right]，右边界right初始值为n，则我们可以通过两个循环分别求出所有左侧的最大收益和所有右侧的最小收益，然后把两者相乘得到每一对相邻的最大收益，并取最大值。因此，我们的目标是找到最大的收益。

## 二、DP-Jump Game I 和 DP-Jump Game II的定义
### （1）DP-Jump Game I:
  * Input: arr=[2,4,-7,3,-9,8], n=9
  
  * Output: 29
  

  此图展示了计算过程中涉及到的变量以及对应关系。

### （2）DP-Jump Game II:
  * Input: arr = [-1,2,-3,4,5,-6,7,8,-9], n=9
  
  * Output: 32
  
  
  此图展示了计算过程中涉及到的变量以及对应关系。
  
  > DP-Jump Game I 和 DP-Jump Game II 是同一种类型的问题，只是输出不同。不同之处在于，对于第一个问题，我们只需要关心每一对相邻的最大收益，而对于第二个问题，我们还需要考虑子区间最大收益的问题。因此，对于第二个问题，我们可以构造dp数组，使得dp[i][j]表示从第i个元素到第j个元素的区间中，子区间最大的收益。

# 2.基本概念和术语
## 1.动态规划(Dynamic Programming)
在计算机科学里，动态规划是指，一个问题的最优解依赖于其他相关子问题的最优解，递归地定义这些子问题的解，然后根据子问题的解，构造总问题的解的方法。动态规划背后的基本思想就是通过组合子问题的解来构造原问题的解，通过这种方法，很多复杂的问题都可以在时间复杂度为$O(n^2)$或更低的时间内得到解决。

动态规划的一些要素包括：
* 最优化原理：在动态规划中，希望找寻一个全局最优解，也就是说，希望通过选择，在已知的情况下，能够确定某种动作或者事件的最好结果。
* 子问题重叠性：动态规划将原问题拆分成多个小问题，并且每一个小问题与整个问题之间只关联一次。
* 概括子结构：子问题的最优解被保存，使得无需重新计算相同的子问题。

## 2.空间优化技巧——滚动数组
为了降低空间复杂度，通常会采用滚动数组的方法。这里所谓的滚动数组，是指用一系列数组的形式，逐渐向前移动，从而节省空间。比如，用四个数组，前三个数组存放上一个迭代时的值，后一个数组存放本次迭代时的值。这样，如果某个数组不再需要，就直接抛弃它。由于后一个数组只保留一个元素，所以空间复杂度降低到了$O(1)$。滚动数组可以有效减少空间占用，但也增加了计算量，因为需要进行多次数组的交换。

## 3.方向性优化——从左往右
我们可以通过从左往右的方向来构造最优解。原因在于，对于每一个位置i，它右边的所有位置都是它的跳转位置，它们的跳转概率是一样的，我们只需考虑最优的那一个即可。此外，对于任意一个位置i，只有它的右边才会影响它的最大收益，所以右边的递推关系可以从右往左进行处理，从而避免重复计算相同的子问题。

## 4.状态定义
对于DP-Jump Game I 和 DP-Jump Game II，我们只需要定义两种状态：
1. pos：表示当前的位置。
2. jump_pos：表示当前位置左边的所有位置。

因此，可以用数组dp\[jump_pos\] 来表示从某一个jump\_pos（指向jump_pos之前的位置）之后的所有跳转。其中，dp\[jump_pos\]\[pos\] 表示从jump\_pos对应的位置的left到当前位置的right范围内，以pos作为结尾的最大收益。

# 3.核心算法和具体实现
## DP-Jump Game I
```python
def dp_jumpgameI(arr):
    """
    :param arr: List[int]
    :return: int
    """
    n = len(arr)
    if not arr or max(abs(x) for x in arr)<1:
        return sum(max(0, arr[i]) for i in range(n))
    
    dp = [[0]*n for _ in range(n)] # 初始化dp
    res = float('-inf')

    left_max = [float('-inf')] * n # 从左往右更新
    for i in range(n):
        left_max[i] = max(left_max[i-1], arr[i])

    right_min = [float('inf')] * n # 从右往左更新
    for j in range(n)[::-1]:
        right_min[j] = min(right_min[j+1], abs(arr[j]))
        
    for jump_pos in range(n):
        for pos in range(jump_pos, n):
            if arr[pos] >= 0 and (not jump_pos or dp[jump_pos-1][pos-1]):
                next_pos = pos + arr[pos]
                if next_pos == n:
                    continue
                
                cur_res = left_max[next_pos] + right_min[pos]
                if cur_res > res:
                    res = cur_res
            
            else: 
                if pos-arr[pos]<0: # 超出了边界
                    break
                    
                for k in range(pos, -1, -1):
                    if arr[k]>=0:
                        next_pos = k + arr[k]

                        cur_res = dp[jump_pos-1][k-1]+left_max[next_pos]+right_min[pos]
                        
                        if cur_res>res:
                            res = cur_res
                        
        for i in range(jump_pos+1, n):
            dp[jump_pos][i]=max([dp[jump_pos][j]+right_min[i]-right_min[j] for j in range(i)])
            
    return res
```
## DP-Jump Game II
```python
def dp_jumpgameII(arr):
    """
    :param arr: List[int]
    :return: int
    """
    n = len(arr)
    if not arr or max(abs(x) for x in arr)<1:
        return sum(max(0, arr[i]) for i in range(n))
    
    dp = [[0]*n for _ in range(n)] 
    for i in range(n):
        dp[i][i] = max(-arr[i], arr[i])
        
    res = dp[-1][0]
    
    def get_next_range(start, end):
        temp_max_profit = float('-inf')
        next_range = []
        for j in range(end-1, start-1, -1):
            if arr[j]<=0:
                continue
                
            profit = arr[j]*(j-start)+dp[start][j]
            if profit>temp_max_profit:
                temp_max_profit = profit
                next_range = [(start, j), (j, end)]
                
        return next_range
                
    while True:
        changed = False
        
        for start in range(n):
            for end in range(start+2, n+1):
                new_ranges = get_next_range(start, end)
                if not new_ranges:
                    continue
                    
                old_val = dp[new_ranges[0][0]][new_ranges[0][1]]
                
                profit = max(sum([-arr[i] for i in range(start, end)]),
                             sum([arr[i] for i in range(start, end)]))
                                
                for r in new_ranges:
                    dp[r[0]][r[1]] = profit
                    res = max(res, profit*(r[1]-r[0])+old_val)
                    
                    if dp[r[0]][r[1]]!=old_val: 
                        changed = True
                    
        if not changed:
            break
        
    
    return res
```
# 4.分析和可视化
在分析和可视化阶段，我们可以画出每个状态的转移图，以及从最初输入到最终输出的状态图。
## DP-Jump Game I


首先，我们可以发现，在初始化阶段，我们已经完成了从左往右的最大值的更新。而接下来的转移计算，我们是在依据上一个位置的值，并从当前位置往后遍历，选出所有可以达到的位置。例如，对于位置0，可以到达位置1，2，3，4，5，6，7。对于位置1，可以到达位置0，3，4，5，6，7。对于位置3，可以到达位置1，4，5，6，7。等等。因此，我们在计算dp[i][j]的时候，要考虑到所有jump_pos<=i，j的情况。

最后，我们发现，对于任何一对位置[i][j]来说，如果j不是跳转位置，那么就不需要再继续往后走，只要往前走就可以。因此，我们可以将非跳转位置后面的所有值直接设置为0，从而节约内存。

## DP-Jump Game II


与DP-Jump Game I类似，首先，我们已经完成了从左往右的最大值的更新。但是，对于DP-Jump Game II，每个状态都由多个范围组成，因此，我们要先处理这个范围。另外，由于跳转并不唯一，因此，我们不能简单地从当前位置往回跳到起始位置，而应该选择所有可能的跳过的位置，并分别计算收益，取最大值。

最后，我们发现，对于任意一对位置[i][j]来说，如果j不是跳转位置，那么就不需要再继续往后走，只要往前走就可以。因此，我们可以将非跳转位置后面的所有值直接设置为0，从而节约内存。