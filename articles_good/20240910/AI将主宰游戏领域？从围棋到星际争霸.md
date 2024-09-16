                 

# 标题
《AI在游戏领域的崛起：从围棋到星际争霸的算法挑战》

## 概述
随着人工智能技术的飞速发展，AI在各个领域的应用愈发广泛，游戏领域也不例外。本文将探讨AI在游戏领域中的崛起，通过分析围棋和星际争霸等典型游戏，解析相关领域的面试题和算法编程题，帮助读者深入了解AI在游戏领域中的应用。

## 面试题与答案解析

### 1. 深度优先搜索（DFS）和广度优先搜索（BFS）的区别？

**题目：** 请解释深度优先搜索（DFS）和广度优先搜索（BFS）的区别，并分别给出一个应用场景。

**答案：**

- **深度优先搜索（DFS）：** 从初始节点开始，沿着某一方向，尽可能深地搜索，直到达到目标节点或者已经访问过所有节点。DFS适用于路径问题，例如迷宫求解、拓扑排序等。

- **广度优先搜索（BFS）：** 从初始节点开始，逐层地搜索所有相邻节点，直到找到目标节点或者已经访问过所有节点。BFS适用于最短路径问题，例如单源最短路径、单源最远距离等。

**举例：**

- **DFS应用场景：** 求解迷宫中的最短路径。

```python
def dfs(maze, start, end):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node == end:
            return True
        if node not in visited:
            visited.add(node)
            stack.extend(get_neighbors(node))
    return False
```

- **BFS应用场景：** 计算单源最短路径。

```python
from collections import deque

def bfs(graph, start, end):
    queue = deque([start])
    visited = {start}
    while queue:
        node = queue.popleft()
        if node == end:
            return True
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
    return False
```

### 2. 二分查找的正确实现

**题目：** 请实现一个二分查找算法，并解释其原理。

**答案：**

二分查找算法的基本原理是：每次将查找范围缩小一半，直到找到目标元素或确定目标元素不存在。具体实现如下：

```python
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
```

### 3. 动态规划求解背包问题

**题目：** 动态规划是一种求解优化问题的方法。请用动态规划方法求解背包问题。

**答案：**

背包问题的动态规划求解方法如下：

```python
def knapSack(W, wt, val, n):
    dp = [[0 for x in range(W + 1)] for x in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if wt[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - wt[i - 1]] + val[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][W]
```

### 4. 如何求解最大子序和问题

**题目：** 如何使用动态规划求解最大子序和问题？

**答案：**

最大子序和问题的动态规划求解方法如下：

```python
def maxSubArray(nums):
    if not nums:
        return 0
    cur_sum = nums[0]
    max_sum = cur_sum
    for i in range(1, len(nums)):
        cur_sum = max(nums[i], cur_sum + nums[i])
        max_sum = max(max_sum, cur_sum)
    return max_sum
```

### 5. 如何求解最长公共子序列

**题目：** 如何使用动态规划求解最长公共子序列问题？

**答案：**

最长公共子序列问题的动态规划求解方法如下：

```python
def longestCommonSubsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

### 6. 如何求解最小生成树

**题目：** 如何使用 Prim 算法求解最小生成树问题？

**答案：**

Prim 算法的求解步骤如下：

1. 选择一个起始顶点，将其加入生成树。
2. 对于剩余的顶点，选择一个距离生成树最近的顶点，将其加入生成树。
3. 重复步骤 2，直到所有顶点都被加入生成树。

具体实现如下：

```python
def prim(graph, start):
    n = len(graph)
    parent = [None] * n
    key = [float('inf')] * n
    mst = []
    key[start] = 0
    visited = set([start])
    while len(visited) < n:
        min_key = float('inf')
        min_index = -1
        for i in range(n):
            if key[i] < min_key and i not in visited:
                min_key = key[i]
                min_index = i
        visited.add(min_index)
        parent[min_index] = start
        mst.append((min_index, start, min_key))
        for i in range(n):
            if i in visited:
                continue
            key[i] = min(key[i], graph[min_index][i])
        start = min_index
    return mst
```

### 7. 如何求解单源最短路径

**题目：** 如何使用 Dijkstra 算法求解单源最短路径问题？

**答案：**

Dijkstra 算法的求解步骤如下：

1. 初始化：将所有顶点的距离设置为无穷大，源点的距离设置为 0。
2. 对于每个顶点，按照距离从小到大进行排序。
3. 选择距离最小的顶点，将其加入到最短路径树中。
4. 更新其他顶点的距离。
5. 重复步骤 3 和 4，直到所有顶点都被加入到最短路径树中。

具体实现如下：

```python
import heapq

def dijkstra(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    visited = set()
    priority_queue = [(0, start)]
    while priority_queue:
        curr_dist, curr_node = heapq.heappop(priority_queue)
        if curr_node in visited:
            continue
        visited.add(curr_node)
        for neighbor, edge_weight in enumerate(graph[curr_node]):
            if neighbor not in visited and curr_dist + edge_weight < dist[neighbor]:
                dist[neighbor] = curr_dist + edge_weight
                heapq.heappush(priority_queue, (dist[neighbor], neighbor))
    return dist
```

### 8. 如何求解最小路径和

**题目：** 如何使用动态规划求解最小路径和问题？

**答案：**

最小路径和问题的动态规划求解方法如下：

```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
    return dp[-1][-1]
```

### 9. 如何求解最大矩形

**题目：** 如何使用单调栈求解最大矩形问题？

**答案：**

最大矩形问题的单调栈求解方法如下：

1. 对于每一列，计算该列到左右两侧最近的“墙”的距离。
2. 计算当前列对应的最大矩形面积。

具体实现如下：

```python
def largestRectangleArea(heights):
    stack = []
    max_area = 0
    heights.append(0)  # 用一个零值来处理边界问题
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] >= h:
            top = stack.pop()
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, width * h)
        stack.append(i)
    return max_area
```

### 10. 如何求解旋转数组的最小值

**题目：** 如何使用二分查找求解旋转数组的最小值问题？

**答案：**

旋转数组的最小值问题可以使用二分查找求解：

1. 找到中间元素。
2. 判断中间元素是否小于其前一个元素，如果是，则最小值就在中间元素左侧。
3. 判断中间元素是否大于其前一个元素，如果是，则最小值就在中间元素右侧。

具体实现如下：

```python
def findMin(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < nums[right]:
            right = mid
        elif nums[mid] > nums[right]:
            left = mid + 1
        else:
            right -= 1
    return nums[left]
```

### 11. 如何求解两数之和

**题目：** 如何使用哈希表求解两数之和问题？

**答案：**

两数之和问题可以使用哈希表求解：

1. 遍历数组，对于每个元素 x，计算目标值 target - x。
2. 使用哈希表记录已经遍历过的元素及其索引。
3. 如果 target - x 存在于哈希表中，则找到对应的索引，返回这两个元素的索引。

具体实现如下：

```python
def twoSum(nums, target):
    hashmap = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hashmap:
            return [hashmap[complement], i]
        hashmap[num] = i
    return []
```

### 12. 如何求解最长公共前缀

**题目：** 如何使用字符串处理方法求解最长公共前缀问题？

**答案：**

最长公共前缀问题可以使用字符串处理方法求解：

1. 将第一个字符串作为基准字符串。
2. 遍历字符串，比较基准字符串和剩余字符串的每个字符。
3. 当出现不同字符时，截取基准字符串的前面部分作为最长公共前缀。

具体实现如下：

```python
def longestCommonPrefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        for i, c in enumerate(s):
            if c != prefix[i]:
                return prefix[:i]
    return prefix
```

### 13. 如何求解有效的括号

**题目：** 如何使用栈求解有效的括号问题？

**答案：**

有效的括号问题可以使用栈求解：

1. 遍历字符串，遇到左括号入栈，遇到右括号出栈。
2. 如果栈为空，说明字符串中的括号匹配，返回 True。
3. 如果栈不为空，说明字符串中的括号不匹配，返回 False。

具体实现如下：

```python
def isValid(s):
    stack = []
    for c in s:
        if c in ['(', '[', '{']:
            stack.append(c)
        elif not stack or (c == ')' and stack[-1] != '(') or (c == ']' and stack[-1] != '[') or (c == '}' and stack[-1] != '{'):
            return False
        else:
            stack.pop()
    return not stack
```

### 14. 如何求解有效的 IP 地址

**题目：** 如何使用字符串处理方法求解有效的 IP 地址问题？

**答案：**

有效的 IP 地址问题可以使用字符串处理方法求解：

1. 将字符串按照点分割成数组。
2. 遍历数组，检查每个部分是否为合法的数字，且长度在 0 到 255 之间。
3. 如果数组长度为 4，且所有部分都合法，则返回 True。

具体实现如下：

```python
def validIPAddress(version, ip):
    def is_valid(h):
        return 0 <= int(h) <= (1 if version == "IPv4" else 65535)

    if version == "IPv4":
        parts = ip.split('.')
        return len(parts) == 4 and all(is_valid(p) for p in parts)
    elif version == "IPv6":
        parts = ip.split(':')
        return len(parts) == 8 and all(len(p) <= 4 and all(c in '0123456789abcdefABCDEF' for c in p) for p in parts)
    return False
```

### 15. 如何求解无重复字符的最长子串

**题目：** 如何使用哈希表求解无重复字符的最长子串问题？

**答案：**

无重复字符的最长子串问题可以使用哈希表求解：

1. 使用双指针法，定义 start 和 end 指针，初始都指向字符串的起始位置。
2. 遍历字符串，使用哈希表记录字符的位置。
3. 如果遇到重复字符，将 start 指针移动到重复字符的下一个位置。
4. 更新无重复字符的最长子串长度。

具体实现如下：

```python
def lengthOfLongestSubstring(s):
    start = 0
    max_length = 0
    char_index_map = {}
    for end in range(len(s)):
        if s[end] in char_index_map:
            start = max(start, char_index_map[s[end]] + 1)
        char_index_map[s[end]] = end
        max_length = max(max_length, end - start + 1)
    return max_length
```

### 16. 如何求解字符串转换问题

**题目：** 如何使用贪心算法求解字符串转换问题？

**答案：**

字符串转换问题可以使用贪心算法求解：

1. 从字符串的起始位置开始，每次选择最小的字符进行替换。
2. 如果选择的字符已经被使用过，则继续选择下一个字符。
3. 重复上述步骤，直到字符串转换完成。

具体实现如下：

```python
def minSteps(self, nums: List[int]) -> int:
    zero_count = nums.count(0)
    if zero_count == len(nums):
        return -1
    odd_count = sum(1 for x in nums if x % 2 == 1)
    if odd_count % 2 == 1:
        return -1
    return len(nums) - odd_count // 2
```

### 17. 如何求解最小路径和

**题目：** 如何使用动态规划求解最小路径和问题？

**答案：**

最小路径和问题的动态规划求解方法如下：

1. 初始化一个二维数组 dp，其中 dp[i][j] 表示从 (0, 0) 到 (i, j) 的最小路径和。
2. 遍历数组，对于每个位置 (i, j)，计算 dp[i][j] 的值。

具体实现如下：

```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
    return dp[-1][-1]
```

### 18. 如何求解有效的山脉数组

**题目：** 如何使用单调栈求解有效的山脉数组问题？

**答案：**

有效的山脉数组问题可以使用单调栈求解：

1. 使用一个栈来记录上升段的右边界。
2. 遍历数组，对于每个位置 i，如果 nums[i] > nums[i - 1]，则将 i 入栈。
3. 如果 nums[i] < nums[i - 1]，则判断栈顶元素是否为上升段的右边界，如果是，则弹出栈顶元素，继续判断，直到栈为空或栈顶元素不是上升段的右边界。
4. 如果数组遍历结束，栈不为空，则说明存在有效的山脉数组。

具体实现如下：

```python
def validMountainArray(self, arr: List[int]) -> bool:
    n = len(arr)
    if n < 3:
        return False
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] < arr[i]:
            if stack[-1] == 0:
                return False
            stack.pop()
        stack.append(i)
    return stack == [0] + sorted(set(stack[1:])) + [n - 1]
```

### 19. 如何求解有效的括号字符串

**题目：** 如何使用栈求解有效的括号字符串问题？

**答案：**

有效的括号字符串问题可以使用栈求解：

1. 遍历字符串，对于每个字符，根据其类型进行如下操作：
   - 如果是左括号，将括号入栈。
   - 如果是右括号，判断栈顶元素是否与当前右括号匹配，如果不匹配或栈为空，则返回 False。
   - 如果匹配，将栈顶元素出栈。
2. 遍历结束后，如果栈为空，则返回 True；否则，返回 False。

具体实现如下：

```python
def isValid(self, s: str) -> bool:
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in mapping:
            if not stack or stack.pop() != mapping[c]:
                return False
        else:
            stack.append(c)
    return not stack
```

### 20. 如何求解两数之和

**题目：** 如何使用排序和双指针求解两数之和问题？

**答案：**

两数之和问题可以使用排序和双指针求解：

1. 将数组排序。
2. 初始化两个指针，一个指向数组的起始位置，一个指向数组的末尾位置。
3. 循环遍历数组，对于每个元素，根据其与目标值的差值调整指针位置。
4. 如果找到两数之和等于目标值，返回这两个数的索引。

具体实现如下：

```python
def twoSum(self, nums: List[int], target: int) -> List[int]:
    nums.sort()
    left, right = 0, len(nums) - 1
    while left < right:
        if nums[left] + nums[right] == target:
            return [left, right]
        elif nums[left] + nums[right] < target:
            left += 1
        else:
            right -= 1
    return []
```

### 21. 如何求解合并区间

**题目：** 如何使用排序和双指针求解合并区间问题？

**答案：**

合并区间问题可以使用排序和双指针求解：

1. 将区间按照起点排序。
2. 初始化两个指针，一个指向数组的起始位置，一个指向数组的末尾位置。
3. 循环遍历数组，对于每个区间，根据区间的重叠情况合并区间。
4. 将合并后的区间添加到结果数组中。

具体实现如下：

```python
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort()
    ans = []
    for interval in intervals:
        if not ans or ans[-1][1] < interval[0]:
            ans.append(interval)
        else:
            ans[-1][1] = max(ans[-1][1], interval[1])
    return ans
```

### 22. 如何求解寻找两个正序数组的中位数

**题目：** 如何使用二分查找求解寻找两个正序数组的中位数问题？

**答案：**

寻找两个正序数组的中位数问题可以使用二分查找求解：

1. 将两个数组合并为一个有序数组。
2. 使用二分查找找到中位数的位置。
3. 根据数组的长度计算中位数。

具体实现如下：

```python
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    m, n = len(nums1), len(nums2)
    if m > n:
        nums1, nums2, m, n = nums2, nums1, n, m
    imin, imax, half_len = 0, m, (m + n + 1) // 2
    while imin <= imax:
        i = (imin + imax) // 2
        j = half_len - i
        if i < m and nums2[j - 1] > nums1[i]:
            imin = i + 1
        elif i > 0 and nums1[i - 1] > nums2[j]:
            imax = i - 1
        else:
            if i == 0:
                max_of_left = nums2[j - 1]
            elif j == 0:
                max_of_left = nums1[i - 1]
            else:
                max_of_left = max(nums1[i - 1], nums2[j - 1])
            if (m + n) % 2 == 1:
                return max_of_left
            min_of_right = min(nums1[i], nums2[j])
            return (max_of_left + min_of_right) / 2
```

### 23. 如何求解奇偶分区

**题目：** 如何使用贪心算法求解奇偶分区问题？

**答案：**

奇偶分区问题可以使用贪心算法求解：

1. 将数组分为奇数索引和偶数索引两部分。
2. 分别对奇数索引和偶数索引部分进行排序。
3. 将奇数索引部分的元素与偶数索引部分的元素交替放入结果数组中。

具体实现如下：

```python
def partitionArray(nums: List[int], m: int) -> List[int]:
    odds = sorted([x for i, x in enumerate(nums) if i % 2 == 0])
    evens = sorted([x for i, x in enumerate(nums) if i % 2 == 1])
    return odds[0::m] + evens[0::m]
```

### 24. 如何求解合并两个有序链表

**题目：** 如何使用递归求解合并两个有序链表问题？

**答案：**

合并两个有序链表问题可以使用递归求解：

1. 如果第一个链表为空，返回第二个链表。
2. 如果第二个链表为空，返回第一个链表。
3. 比较第一个链表和第二个链表的头节点，将较小的节点连接到合并后的链表上。
4. 递归调用合并两个链表的剩余部分。

具体实现如下：

```python
def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    if not list1:
        return list2
    if not list2:
        return list1
    if list1.val < list2.val:
        list1.next = self.mergeTwoLists(list1.next, list2)
        return list1
    else:
        list2.next = self.mergeTwoLists(list1, list2.next)
        return list2
```

### 25. 如何求解最长公共子串

**题目：** 如何使用动态规划求解最长公共子串问题？

**答案：**

最长公共子串问题的动态规划求解方法如下：

1. 初始化一个二维数组 dp，其中 dp[i][j] 表示字符串 s1 和 s2 的前 i 个字符和前 j 个字符的最长公共子串长度。
2. 遍历字符串 s1 和 s2，对于每个字符，更新 dp[i][j] 的值。
3. 找到 dp 的最大值，即为最长公共子串的长度。

具体实现如下：

```python
def longestCommonSubstring(self, s1: str, s2: str) -> str:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_pos = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0
    return s1[end_pos - max_len: end_pos]
```

### 26. 如何求解最长重复子串

**题目：** 如何使用二分查找和哈希表求解最长重复子串问题？

**答案：**

最长重复子串问题的二分查找和哈希表求解方法如下：

1. 使用二分查找确定重复子串的长度范围。
2. 对于每个长度，使用哈希表记录子串的出现次数。
3. 找到出现次数最多的子串。

具体实现如下：

```python
def longestRepeatingSubstring(self, s: str) -> str:
    def check(len):
        counter = defaultdict(int)
        for i in range(len(s) - len + 1):
            sub_str = s[i: i + len]
            counter[sub_str] += 1
        max_count = max(counter.values())
        return max_count >= 2

    left, right = 1, len(s) // 2
    while left < right:
        mid = (left + right) // 2
        if check(mid):
            left = mid + 1
        else:
            right = mid
    return s[left - 1: left + left - 1]
```

### 27. 如何求解最长公共前缀

**题目：** 如何使用字符串处理方法求解最长公共前缀问题？

**答案：**

最长公共前缀问题的字符串处理方法如下：

1. 将第一个字符串作为基准字符串。
2. 遍历字符串，比较基准字符串和剩余字符串的每个字符。
3. 当出现不同字符时，截取基准字符串的前面部分作为最长公共前缀。

具体实现如下：

```python
def longestCommonPrefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        i = 0
        while i < len(prefix) and i < len(s):
            if prefix[i] != s[i]:
                break
            i += 1
        prefix = prefix[:i]
    return prefix
```

### 28. 如何求解单词搜索 II

**题目：** 如何使用回溯算法求解单词搜索 II 问题？

**答案：**

单词搜索 II 问题的回溯算法求解方法如下：

1. 使用一个 visited 数组记录当前网格中每个位置是否被访问过。
2. 使用一个前缀树存储所有单词。
3. 从网格的每个位置开始，尝试搜索单词。
4. 如果找到一个单词，将其添加到结果列表中。

具体实现如下：

```python
def findWords(board, words):
    def search(i, j, index):
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] not in word or visited[i][j]:
            return
        if board[i][j] == word[index]:
            word = word[:index] + board[i][j] + word[index + 1:]
            index += 1
            if index == len(word):
                result.append(word)
                word = word[:len(word) - 1]
                return
            visited[i][j] = True
            for x, y in directions:
                search(i + x, j + y, index)
            visited[i][j] = False

    m, n = len(board), len(board[0])
    result = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    tree = Trie()
    for w in words:
        tree.insert(w)

    for i in range(m):
        for j in range(n):
            visited = [[False] * n for _ in range(m)]
            if board[i][j] in tree.children:
                search(i, j, 0)
    return result
```

### 29. 如何求解全排列

**题目：** 如何使用递归求解全排列问题？

**答案：**

全排列问题的递归求解方法如下：

1. 对于每个位置，从当前位置开始，递归地交换当前位置与其他位置的元素。
2. 交换完成后，继续递归地处理下一个位置。

具体实现如下：

```python
def permute(nums):
    def dfs(nums, path, ans):
        if not nums:
            ans.append(path)
            return
        for i in range(len(nums)):
            dfs(nums[:i] + nums[i + 1:], path + [nums[i]], ans)

    ans = []
    dfs(nums, [], ans)
    return ans
```

### 30. 如何求解 N 皇后问题

**题目：** 如何使用回溯算法求解 N 皇后问题？

**答案：**

N 皇后问题的回溯算法求解方法如下：

1. 使用一个数组记录每行放置的皇后位置。
2. 对于每一行，从第 2 行开始，尝试在该行放置皇后。
3. 判断皇后是否与前面已经放置的皇后冲突。
4. 如果冲突，继续尝试下一个位置。
5. 如果没有冲突，继续递归地放置下一行的皇后。

具体实现如下：

```python
def solveNQueens(n):
    def dfs(queens, xy_dif, xy_sum):
        p = len(queens)
        if p == n:
            result.append(queens)
            return
        for q in range(n):
            if q not in queens and p - q not in xy_dif and p + q not in xy_sum:
                dfs(queens + [q], xy_dif + [p - q], xy_sum + [p + q])

    result = []
    dfs([], [], [])
    return [[["." for _ in range(n)] for _ in range(n)] for _ in result]
```

