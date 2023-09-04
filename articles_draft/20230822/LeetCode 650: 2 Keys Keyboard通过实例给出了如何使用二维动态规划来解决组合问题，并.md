
作者：禅与计算机程序设计艺术                    

# 1.简介
  

关于这个问题，网上也有很多帖子，其核心算法就是一个二维动态规划问题——“650: 2 Keys Keyboard”，在力扣上已经有人做过详细的代码实现，比如这题解法。

## LeetCode 650: 2 Keys Keyboard（最长递增子序列）
给定一个整数数组 nums ，你需要找到一个长度为 k 的子序列使得从左到右每个元素都恰好出现一次，并且顺序不同于输入数组中的对应元素。

示例 1：
```
输入：nums = [1,2,3], k = 3
输出：[1,3,2]
```
提示：
1 <= nums.length <= 10^4
-109 <= nums[i] <= 109
nums 中的所有整数 互不相同 。
1 <= k <= length(nums) + 1

## 2.背景介绍
有两个按键可以让用户输入单个字符，但是为了防止错误输入，要求每个字符只能按一次。假设只有两种按键 A 和 B ，其中 A 按一次消耗的时间比 B 更短。现在有一个字符串 s ，它由小写字母组成，表示需要被输入的目标字符串。问怎么才能用这两个按键让用户输入字符串 s ，且输入的时间最短？

## 3.基本概念术语说明

### 1. 二维动态规划
所谓“二维动态规划”是指状态的定义包含两个维度，即时间和空间两个维度。而本题就是典型的“二维动态规划”。 

### 2. 状态定义
对于二维动态规划来说，首先要定义状态，而在本题中，状态又分为两个维度：行坐标 i 和列坐标 j ，这里 i 表示目标字符串的当前位置下标，j 表示目标字符串中剩余未输入的字符个数。因此，状态定义为 dp[i][j] ，代表当目标字符串的当前位置下标为 i 时，已知剩余未输入的字符个数为 j ，能否将目标字符串输入完成。

### 3. 转移方程
由于每一个位置上都可以选择 A 或 B 来进行输入，那么就需要确定转移方程。如果已知目标字符串当前位置 i ，剩余未输入字符个数 j ，我们考虑以下两种情况：

1. 第一种情况是，A 按键被选中，则此时 DP 状态的变化如下图所示：

   
   即：当前下标 i 不变， j - 1（因为此时 A 按键被选中），目标字符串之前的所有状态值均加 1 （因为 A 按键消耗一次时间）。
   
2. 第二种情况是，B 按键被选中，则此时 DP 状态的变化如下图所示：

   
   即：当前下标 i + 1（因为此时 B 按键被选中）， j 保持不变，目标字符串之前的所有状态值均减 1 （因为 B 按键消耗一次时间）。

综上，根据上面两种情况，可以总结出 DP 转移方程：

    if (s[i] == 'a') {
        dp[i+1][j-1] |= dp[i][j-1]; // 如果 s[i] 是 a 则可选 A 或 B，记录下所有方案中的最大值
        for (int k = 0; k < n; ++k)
            dp[i+1][j] &= dp[k][j]; // 把之前的方案向后传播，剔除不能前进的状态
    } else {
        dp[i][j] &=!dp[i][j-1]; // 如果 s[i] 是 b 则只能选 B，记录下所有方案中最小值
        for (int k = 0; k < n; ++k)
            dp[i][j-1] |= dp[k][j-1]; // 把之前的方案向后传播，扩充新的方案
    }

- `&=` 操作符的含义是对 `dp` 数组中对应位置的元素执行 AND 运算，即 `dp[i][j]` 为 true 时，才表示 `dp[i][j]` 及其之前的所有元素均为真；
- `|` 操作符的含义是对 `dp` 数组中对应位置的元素执行 OR 运算，即 `dp[i][j]` 为 false 时，才表示 `dp[i][j]` 及其之前的所有元素均为假。

### 4. 初始化
初始化 `dp` 数组的第一行和第一列的值，根据上面的分析，先考虑若第一个字符是一个 B 字符的话，它的下一步只能是 B 字符。因此，直接把 dp[0][0..n-1] 初始化为true即可。

## 4.具体代码实例和解释说明

### 4.1 Python 代码
```python
class Solution:
    def longestIncreasingPath(self, matrix):
        m, n = len(matrix), len(matrix[0])
        dirs = [(0,-1),(0,1),(-1,0),(1,0)] # 上、下、左、右四个方向

        dp = [[False]*n for _ in range(m)]
        
        def dfs(x, y, step=0):
            dp[x][y] = True
            maxlen = 1
            
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                
                if 0<=nx<m and 0<=ny<n and not dp[nx][ny]:
                    nextstep = dfs(nx, ny, step+1)
                    
                    if nextstep > maxlen or (nextstep == maxlen and ((matrix[x][y]>matrix[nx][ny])!= (dx==0))):
                        maxlen = nextstep
            
            return maxlen
            
        result = []
        
        for i in range(m):
            for j in range(n):
                result += [dfs(i, j)],
        
        return max(result)
    
```
函数 `longestIncreasingPath` 的参数矩阵 `matrix` 是一个列表，内部的元素是一维数组，数组内的元素表示每个格子上的数字。函数返回的是能够填满整个矩阵的数字，所以函数主体其实还是 DFS，只是针对不同类型的数值采用不同的 DFS 规则。我们这里只提取核心函数 `dfs`，看一下具体过程。

```python
def dfs(x, y, step=0):
    dp[x][y] = True
    maxlen = 1
    
    for dx, dy in dirs:
        nx, ny = x+dx, y+dy
        
        if 0<=nx<m and 0<=ny<n and not dp[nx][ny]:
            nextstep = dfs(nx, ny, step+1)
            
            if nextstep > maxlen or (nextstep == maxlen and ((matrix[x][y]>matrix[nx][ny])!= (dx==0))):
                maxlen = nextstep
    
    return maxlen
```

函数 `dfs` 的参数 `(x, y)` 表示矩阵坐标 `(x, y)` 当前所在位置， `step` 表示当前到达该位置经历的步数。函数首先标记 `(x, y)` 位置为已访问，然后记录 `maxlen` 的初始值为 1。接着，遍历四个方向，依次寻找可以到达的位置 `(nx, ny)` 。如果 `(nx, ny)` 没有访问过 (`not dp[nx][ny]`) ，就进入另一个 DFS 函数 `dfs(nx, ny, step+1)` 。

如果 `(nx, ny)` 的 `maxlen` 比当前值大或者相等且向下或向右移动 (`((matrix[x][y]>matrix[nx][ny])!= (dx==0))`) ，则更新 `maxlen`。最后返回 `maxlen`。

再回到主函数 `longestIncreasingPath` ，遍历整个矩阵，对于每个没有访问过的坐标 `(i, j)` 调用 `dfs(i, j)` ，并将返回结果存入 `result` 列表中。最后返回列表中的最大值。

### 4.2 C++ 代码
```cpp
const int MAXN = 1e5 + 5;
bool vis[MAXN][2], mp[MAXN][2];

void precompute() {
    memset(vis, false, sizeof(vis));
    memset(mp, false, sizeof(mp));
}

inline bool check(int u, char c) {
    return mp[u][ord(c)-'a'];
}

int solve(char* str, int l) {
    precompute();
    mp[l-1][str[l-1]-'a'] = true;
    int ans = 1;

    for (int i = l-2; i >= 0; --i) {
        char& prev = str[i];
        char cur = *str++;
        int up = ord(prev) - ord('a'), down = ord(cur) - ord('a');
        int pos = up ^ down;
        int old = mp[pos][up];
        mp[pos][down] = true;
        ans = max(ans, old + 1);
    }

    for (int i = l-2; i >= 0; --i) {
        char& prev = str[i];
        char cur = *--str;
        int up = ord(prev) - ord('a'), down = ord(cur) - ord('a');
        int pos = up ^ down;
        int old = mp[pos][down];
        mp[pos][up] = true;
        ans = max(ans, old + 1);
    }

    return ans;
}
```
首先定义变量 `vis` ，用于记录是否访问过的节点，定义变量 `mp` ，用于记录每个节点的子节点是否存在。函数 `precompute()` 初始化这些变量。

函数 `check` 检查节点 `u` 是否存在子节点 `c` 。

函数 `solve` 使用双指针方法求解，对每个位置 `i` ，维护两个子节点 `p1` 和 `p2` ，其中 `p1` 为 `i` 位置前的节点，`p2` 为 `i` 位置后的节点。如果两者都是同一个字符，则只需要检查一半的情况。否则，先确定它们的位置 `pos` ，然后尝试通过某些转换得到子节点。首先检查 `pos` 上不存在子节点，尝试添加 `p2` 上的节点；接着检查 `pos` 上不存在父节点，尝试添加 `p1` 上的节点。

函数返回 `ans` ，即从左到右和从右到左最长可行路径长度之和。