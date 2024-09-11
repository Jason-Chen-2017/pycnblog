                 

### 1. 编程题：快速排序（Quick Sort）

**题目描述：** 实现快速排序算法，对数组进行排序。

**代码要求：** 完成以下函数，返回排序后的数组：

```python
def quick_sort(arr):
    # 你的代码实现
    return arr
```

**答案解析：**

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# 测试代码
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = quick_sort(arr)
print(sorted_arr)  # 输出 [1, 1, 2, 3, 6, 8, 10]
```

**解析：** 快速排序是一种分治算法，通过选择一个基准元素（pivot），将数组分为小于基准的左子数组、等于基准的中位数组和大于基准的右子数组。递归地对左子数组和右子数组进行排序，最后将排序后的左子数组、中位数组和右子数组拼接起来。

### 2. 面试题：二叉树的层序遍历

**题目描述：** 实现二叉树的层序遍历，输出每一层的元素。

**代码要求：** 完成以下函数，输入一棵二叉树，返回层序遍历的结果：

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order(root):
    # 你的代码实现
    return []

# 测试代码
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

result = level_order(root)
print(result)  # 输出 [[1], [2, 3], [4, 5]]
```

**答案解析：**

```python
def level_order(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    
    return result
```

**解析：** 层序遍历使用广度优先搜索（BFS）算法。通过队列实现，每次从队列中取出一个节点，将其值加入当前层，并将它的左右子节点加入队列。依次类推，直到队列为空。

### 3. 编程题：最长公共前缀

**题目描述：** 编写一个函数来查找字符串数组中的最长公共前缀。

**代码要求：** 完成以下函数，返回最长公共前缀：

```python
def longest_common_prefix(strs):
    # 你的代码实现
    return ""
```

**答案解析：**

```python
def longest_common_prefix(strs):
    if not strs:
        return ""
    
    prefix = ""
    for i in range(len(min(strs, key=len))):
        char = strs[0][i]
        for s in strs:
            if s[i] != char:
                return prefix
        prefix += char
    
    return prefix

# 测试代码
strs = ["flower","flow","flight"]
result = longest_common_prefix(strs)
print(result)  # 输出 "fl"
```

**解析：** 最长公共前缀问题可以通过逐个比较字符串的前缀来求解。从第一个字符串开始，逐个字符与后续字符串进行比较，直到找到不同的字符。这种方法的时间复杂度为 O(S)，其中 S 是所有字符串的总长度。

### 4. 编程题：合并两个有序链表

**题目描述：** 合并两个已排序的链表并返回合并后的链表。

**代码要求：** 完成以下函数，返回合并后的链表：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_sorted_lists(l1, l2):
    # 你的代码实现
    return None
```

**答案解析：**

```python
def merge_sorted_lists(l1, l2):
    dummy = ListNode()
    current = dummy
    
    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 or l2
    return dummy.next

# 测试代码
l1 = ListNode(1, ListNode(3, ListNode(5)))
l2 = ListNode(2, ListNode(4, ListNode(6)))

merged = merge_sorted_lists(l1, l2)
while merged:
    print(merged.val, end=" ")
    merged = merged.next
# 输出 1 2 3 4 5 6
```

**解析：** 合并两个有序链表可以通过迭代的方式实现。创建一个哑节点（dummy），然后比较两个链表的当前节点值，将较小值链接到哑节点的下一个节点，并移动较小值的链表指针。重复此过程，直到某个链表到达末尾，然后将剩余的链表链接到结果链表的末尾。

### 5. 编程题：两数相加

**题目描述：** 给定两个非空链表表示两个非负整数，每位数字都按照逆序的方式存储在链表中，编写一个函数来计算这两个数字的和，并以链表形式返回结果。

**代码要求：** 完成以下函数，返回链表形式的和：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def add_two_numbers(l1, l2):
    # 你的代码实现
    return None
```

**答案解析：**

```python
def add_two_numbers(l1, l2):
    dummy = ListNode()
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = (l1.val if l1 else 0)
        val2 = (l2.val if l2 else 0)
        total = val1 + val2 + carry
        carry = total // 10
        current.next = ListNode(total % 10)
        current = current.next
        
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    
    return dummy.next

# 测试代码
l1 = ListNode(2, ListNode(4, ListNode(3)))
l2 = ListNode(5, ListNode(6, ListNode(4)))

result = add_two_numbers(l1, l2)
while result:
    print(result.val, end=" ")
    result = result.next
# 输出 7 0 8
```

**解析：** 两数相加问题可以通过模拟手工加法过程来解决。使用一个哑节点（dummy），然后遍历两个链表，对每个节点进行相加，并处理进位（carry）。将结果存储在新的链表中，并返回。

### 6. 面试题：二叉树的深度优先遍历

**题目描述：** 实现二叉树的深度优先遍历，返回遍历的结果。

**代码要求：** 完成以下函数，输入一棵二叉树，返回深度优先遍历的结果：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def dfs(root):
    # 你的代码实现
    return []
```

**答案解析：**

```python
def dfs(root):
    if not root:
        return []
    
    stack = [root]
    result = []
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result

# 测试代码
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

result = dfs(root)
print(result)  # 输出 [1, 2, 4, 5, 3]
```

**解析：** 深度优先遍历（DFS）可以使用栈实现。首先将根节点入栈，然后依次弹出栈顶节点，将其值加入结果列表，并访问其左右子节点。重复此过程，直到栈为空。

### 7. 编程题：最长公共子序列

**题目描述：** 给定两个字符串，找出它们的最长公共子序列。

**代码要求：** 完成以下函数，返回最长公共子序列：

```python
def longest_common_subsequence(s1, s2):
    # 你的代码实现
    return ""
```

**答案解析：**

```python
def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            result.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(result[::-1])

# 测试代码
s1 = "abcde"
s2 = "ace"
result = longest_common_subsequence(s1, s2)
print(result)  # 输出 "ace"
```

**解析：** 最长公共子序列（LCS）问题可以使用动态规划（DP）解决。创建一个二维数组 dp，其中 dp[i][j] 表示 s1 的前 i 个字符与 s2 的前 j 个字符的最长公共子序列长度。遍历字符串，更新 dp 数组，并在最后根据 dp 数组还原最长公共子序列。

### 8. 编程题：股票买卖的最佳时机

**题目描述：** 给定一个整数数组 prices，其中 prices[i] 是第 i 天股票的价格。如果当日买入股票，次日卖出，则可以获取 prices[i] - prices[i+1] 的利润。返回可以获得的最大利润。注意不能同时参与多笔交易。

**代码要求：** 完成以下函数，返回最大利润：

```python
def max_profit(prices):
    # 你的代码实现
    return 0
```

**答案解析：**

```python
def max_profit(prices):
    if not prices:
        return 0
    
    max_profit = 0
    for i in range(1, len(prices)):
        max_profit += max(0, prices[i] - prices[i - 1])
    
    return max_profit

# 测试代码
prices = [7, 1, 5, 3, 6, 4]
result = max_profit(prices)
print(result)  # 输出 5
```

**解析：** 股票买卖的最佳时机问题可以通过一次遍历求解。遍历价格数组，计算当日与次日价格之差，并累加到最大利润变量中。如果价格下降，则不累加，继续下一轮计算。

### 9. 面试题：环形数组的最小整数

**题目描述：** 给定一个整数数组 arr，其中 1 <= arr.length <= 104，arr 可能包含重复的数值。设计函数 searchMin，在数组中查找一个旋转数字的最小值。例如，给定数组 [4, 5, 6, 7, 0, 1, 2]，最小值是 0。

**代码要求：** 完成以下函数，返回旋转数组的最小值：

```python
def search_min(arr):
    # 你的代码实现
    return 0
```

**答案解析：**

```python
def search_min(arr):
    n = len(arr)
    low, high = 0, n - 1
    
    while low < high:
        mid = (low + high) // 2
        if arr[mid] > arr[high]:
            low = mid + 1
        else:
            high = mid
    
    return arr[low]

# 测试代码
arr = [4, 5, 6, 7, 0, 1, 2]
result = search_min(arr)
print(result)  # 输出 0
```

**解析：** 环形数组的最小整数问题可以通过二分查找解决。通过比较中间元素和最右边的元素，确定最小值在数组的哪一侧，然后缩小搜索范围。

### 10. 编程题：两数之和

**题目描述：** 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那两个整数，并返回他们的数组下标。

**代码要求：** 完成以下函数，返回两个数的下标：

```python
def two_sum(nums, target):
    # 你的代码实现
    return []
```

**答案解析：**

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []

# 测试代码
nums = [2, 7, 11, 15]
target = 9
result = two_sum(nums, target)
print(result)  # 输出 [0, 1]
```

**解析：** 两数之和问题可以通过哈希表实现。遍历数组，对于每个元素，计算其与目标值的差值，并检查该差值是否已存在于哈希表中。如果存在，则返回差值的下标和当前元素的下标。

### 11. 编程题：寻找两个正序数组的中位数

**题目描述：** 给定两个已排序的整数数组 nums1 和 nums2，请找到这两个数组的中位数。中位数是有序数组中间位置的元素，如果数组长度是偶数，则中位数是中间两个元素的平均值。

**代码要求：** 完成以下函数，返回中位数：

```python
def find_median_sorted_arrays(nums1, nums2):
    # 你的代码实现
    return 0
```

**答案解析：**

```python
def find_median_sorted_arrays(nums1, nums2):
    m, n = len(nums1), len(nums2)
    if m > n:
        nums1, nums2 = nums2, nums1
        m, n = n, m
    
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
    
    return 0

# 测试代码
nums1 = [1, 3]
nums2 = [2]
result = find_median_sorted_arrays(nums1, nums2)
print(result)  # 输出 2
```

**解析：** 寻找两个正序数组的中位数问题可以通过二分查找实现。将两个数组合并，然后计算中位数。通过比较中间元素，不断缩小查找范围，直到找到中位数。

### 12. 编程题：最小路径和

**题目描述：** 给定一个包含非负整数的 m x n 网格 grid ，找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**代码要求：** 完成以下函数，返回最小路径和：

```python
def min_path_sum(grid):
    # 你的代码实现
    return 0
```

**答案解析：**

```python
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i - 1][j - 1]
    
    return dp[m][n]

# 测试代码
grid = [
    [1, 3, 1],
    [1, 5, 1],
    [4, 2, 1]
]
result = min_path_sum(grid)
print(result)  # 输出 7
```

**解析：** 最小路径和问题可以通过动态规划（DP）解决。创建一个二维数组 dp，其中 dp[i][j] 表示从左上角到 (i, j) 的最小路径和。遍历网格，更新 dp 数组，并返回 dp[m][n]。

### 13. 编程题：最小栈

**题目描述：** 设计一个支持 push，pop，top 操作的栈，同时实现一个获取栈的最小元素的 min 函数。

**代码要求：** 完成以下函数，实现栈和获取最小元素的接口：

```python
class MinStack:
    def __init__(self):
        # 初始化代码

    def push(self, x):
        # 实现push代码

    def pop(self):
        # 实现pop代码

    def top(self):
        # 实现top代码

    def getMin(self):
        # 实现getMin代码
```

**答案解析：**

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, x):
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        if self.stack:
            x = self.stack.pop()
            if x == self.min_stack[-1]:
                self.min_stack.pop()
            return x

    def top(self):
        return self.stack[-1] if self.stack else None

    def getMin(self):
        return self.min_stack[-1] if self.min_stack else None

# 测试代码
minStack = MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
print(minStack.getMin())  # 输出 -3
minStack.pop()
print(minStack.top())    # 输出 0
print(minStack.getMin())  # 输出 -2
```

**解析：** 最小栈问题可以通过维护一个辅助栈来实现。主栈用于存储所有元素，辅助栈用于存储当前栈中的最小元素。每次 push 和 pop 操作时，都要更新辅助栈。top 和 getMin 操作可以直接从辅助栈获取。

### 14. 编程题：移除重复元素

**题目描述：** 给定一个排序数组，移除数组中的重复元素，返回移除后数组的新长度。

**代码要求：** 完成以下函数，返回新数组的长度：

```python
def remove_duplicates(nums):
    # 你的代码实现
    return 0
```

**答案解析：**

```python
def remove_duplicates(nums):
    if not nums:
        return 0
    
    slow, fast = 0, 1
    while fast < len(nums):
        if nums[slow] != nums[fast]:
            slow += 1
            nums[slow] = nums[fast]
        fast += 1
    
    return slow + 1

# 测试代码
nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
result = remove_duplicates(nums)
print(result)  # 输出 7，原数组变为 [0, 0, 1, 1, 2, 2, 3, 3, 4]
```

**解析：** 移除重复元素问题可以通过双指针实现。定义 slow 和 fast 两个指针，fast 指向当前需要比较的元素，slow 指向当前新数组的最后一个非重复元素的位置。当 nums[fast] 与 nums[slow] 不相等时，将 nums[fast] 移动到 slow 的下一个位置，并更新 slow。最后，返回 slow + 1，即新数组的长度。

### 15. 编程题：合并区间

**题目描述：** 给定一组区间，将重叠的区间合并为一个区间。

**代码要求：** 完成以下函数，返回合并后的区间列表：

```python
def merge_intervals(intervals):
    # 你的代码实现
    return []
```

**答案解析：**

```python
def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    result = [intervals[0]]
    
    for interval in intervals[1:]:
        if result[-1][1] >= interval[0]:
            result[-1][1] = max(result[-1][1], interval[1])
        else:
            result.append(interval)
    
    return result

# 测试代码
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
result = merge_intervals(intervals)
print(result)  # 输出 [[1, 6], [8, 10], [15, 18]]
```

**解析：** 合并区间问题可以通过排序和遍历实现。首先对区间列表按起始值排序，然后遍历区间列表，如果当前区间与上一个区间有重叠，则合并它们；否则，将当前区间添加到结果列表中。

### 16. 编程题：合并两个有序链表

**题目描述：** 给定两个有序链表，将它们合并为一个有序链表。

**代码要求：** 完成以下函数，返回合并后的链表：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_sorted_lists(l1, l2):
    # 你的代码实现
    return None
```

**答案解析：**

```python
def merge_sorted_lists(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    
    if l1.val < l2.val:
        result = l1
        result.next = merge_sorted_lists(l1.next, l2)
    else:
        result = l2
        result.next = merge_sorted_lists(l1, l2.next)
    
    return result

# 测试代码
l1 = ListNode(1, ListNode(3, ListNode(5)))
l2 = ListNode(2, ListNode(4, ListNode(6)))

merged = merge_sorted_lists(l1, l2)
while merged:
    print(merged.val, end=" ")
    merged = merged.next
# 输出 1 2 3 4 5 6
```

**解析：** 合并两个有序链表可以通过递归实现。比较两个链表的头节点，将较小值链接到结果链表，并递归地合并剩余部分。如果其中一个链表到达末尾，则将另一个链表链接到结果链表。

### 17. 编程题：两数相加

**题目描述：** 给定两个非空链表表示两个非负整数，每位数字都按照逆序的方式存储在链表中，编写一个函数来计算这两个数字的和，并以链表形式返回结果。

**代码要求：** 完成以下函数，返回链表形式的和：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def add_two_numbers(l1, l2):
    # 你的代码实现
    return None
```

**答案解析：**

```python
def add_two_numbers(l1, l2):
    dummy = ListNode()
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = (l1.val if l1 else 0)
        val2 = (l2.val if l2 else 0)
        total = val1 + val2 + carry
        carry = total // 10
        current.next = ListNode(total % 10)
        current = current.next
        
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    
    return dummy.next

# 测试代码
l1 = ListNode(2, ListNode(4, ListNode(3)))
l2 = ListNode(5, ListNode(6, ListNode(4)))

result = add_two_numbers(l1, l2)
while result:
    print(result.val, end=" ")
    result = result.next
# 输出 7 0 8
```

**解析：** 两数相加问题可以通过模拟手工加法过程来解决。使用一个哑节点（dummy），然后遍历两个链表，对每个节点进行相加，并处理进位（carry）。将结果存储在新的链表中，并返回。

### 18. 编程题：有效括号

**题目描述：** 给定一个包含大括号{[()]}的字符串，判断字符串是否有效。

**代码要求：** 完成以下函数，返回字符串是否有效的布尔值：

```python
def is_valid(s):
    # 你的代码实现
    return False
```

**答案解析：**

```python
def is_valid(s):
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    
    return not stack

# 测试代码
s = "{[()]}()"
print(is_valid(s))  # 输出 True
s = "{[()]"
print(is_valid(s))  # 输出 False
```

**解析：** 有效括号问题可以通过栈实现。遍历字符串，对于左括号，将其压入栈中；对于右括号，将其与栈顶元素匹配，如果匹配失败，则返回 False。遍历完成后，如果栈为空，则字符串有效。

### 19. 编程题：二进制表示中1的个数

**题目描述：** 编写一个函数，输入一个无符号整数，返回其二进制表达式中 1 的个数。

**代码要求：** 完成以下函数，返回 1 的个数：

```python
def count_bits(n):
    # 你的代码实现
    return 0
```

**答案解析：**

```python
def count_bits(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count

# 测试代码
n = 0b10110  # 输出 2
print(count_bits(n))
n = 0b11111  # 输出 5
print(count_bits(n))
```

**解析：** 二进制表示中 1 的个数问题可以通过位操作实现。每次将 n 与 n - 1 进行与运算，消除 n 最右边的 1，并累加消除的次数。

### 20. 编程题：反转整数

**题目描述：** 编写一个函数，输入一个 32 位有符号整数，输出其二进制表示中 1 的个数。

**代码要求：** 完成以下函数，返回反转后的整数：

```python
def reverse_integer(x):
    # 你的代码实现
    return 0
```

**答案解析：**

```python
def reverse_integer(x):
    MAX_INT = 2**31 - 1
    MIN_INT = -2**31
    
    reversed_num = 0
    while x:
        pop_bit = x % 10
        x //= 10
        if reversed_num > MAX_INT // 10 or (reversed_num == MAX_INT // 10 and pop_bit > MAX_INT % 10):
            return 0
        if reversed_num < MIN_INT // 10 or (reversed_num == MIN_INT // 10 and pop_bit < MIN_INT % 10):
            return 0
        reversed_num = reversed_num * 10 + pop_bit
    
    return reversed_num

# 测试代码
x = 123
print(reverse_integer(x))  # 输出 321
x = -123
print(reverse_integer(x))  # 输出 -321
x = 120
print(reverse_integer(x))  # 输出 21
```

**解析：** 反转整数问题可以通过模拟实现。每次从整数中取出个位数，并将其插入到结果整数的最左侧。在插入之前，需要检查结果整数是否超出 32 位整数的范围。

### 21. 编程题：字符串匹配（KMP 算法）

**题目描述：** 实现字符串匹配算法（KMP 算法），找出字符串 s 中的一个子串 t 的首次匹配位置。

**代码要求：** 完成以下函数，返回子串 t 在字符串 s 中的起始索引：

```python
def kmp(s, t):
    # 你的代码实现
    return -1
```

**答案解析：**

```python
def kmp(s, t):
    def build_lps(t):
        lps = [0] * len(t)
        length = 0
        i = 1
        while i < len(t):
            if t[i] == t[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps
    
    lps = build_lps(t)
    i = j = 0
    while i < len(s):
        if t[j] == s[i]:
            i += 1
            j += 1
        if j == len(t):
            return i - j
        elif i < len(s) and t[j] != s[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return -1

# 测试代码
s = "ABABDABACD"
t = "ABABCABAB"
print(kmp(s, t))  # 输出 2
s = "BBC ABCDAB ABCDAB ABCDE"
t = "ABCDAB"
print(kmp(s, t))  # 输出 7
```

**解析：** KMP 算法是一种高效字符串匹配算法。首先构建一个最长公共前后缀（LPS）数组，然后使用 LPS 数组和主字符串 s 与模式串 t 进行匹配。在匹配过程中，如果当前字符不匹配，可以通过 LPS 数组快速跳到下一个可能的匹配位置。

### 22. 编程题：两数相加

**题目描述：** 给定两个非空链表表示两个非负整数，每位数字都按照逆序的方式存储在链表中，编写一个函数来计算这两个数字的和，并以链表形式返回结果。

**代码要求：** 完成以下函数，返回链表形式的和：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def add_two_numbers(l1, l2):
    # 你的代码实现
    return None
```

**答案解析：**

```python
def add_two_numbers(l1, l2):
    dummy = ListNode()
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = (l1.val if l1 else 0)
        val2 = (l2.val if l2 else 0)
        total = val1 + val2 + carry
        carry = total // 10
        current.next = ListNode(total % 10)
        current = current.next
        
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    
    return dummy.next

# 测试代码
l1 = ListNode(2, ListNode(4, ListNode(3)))
l2 = ListNode(5, ListNode(6, ListNode(4)))

result = add_two_numbers(l1, l2)
while result:
    print(result.val, end=" ")
    result = result.next
# 输出 7 0 8
```

**解析：** 两数相加问题可以通过模拟手工加法过程来解决。使用一个哑节点（dummy），然后遍历两个链表，对每个节点进行相加，并处理进位（carry）。将结果存储在新的链表中，并返回。

### 23. 编程题：最长公共子串

**题目描述：** 给定两个字符串，找出它们的公共最长子串。

**代码要求：** 完成以下函数，返回最长公共子串：

```python
def longest_common_substring(s1, s2):
    # 你的代码实现
    return ""
```

**答案解析：**

```python
def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    result = ""
    max_len = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    result = s1[i - max_len: i]
    
    return result

# 测试代码
s1 = "abcdaf"
s2 = "bcdoze"
print(longest_common_substring(s1, s2))  # 输出 "bcd"
s1 = "apple"
s2 = "apricot"
print(longest_common_substring(s1, s2))  # 输出 "apple"
```

**解析：** 最长公共子串问题可以通过动态规划（DP）实现。创建一个二维数组 dp，其中 dp[i][j] 表示 s1 的前 i 个字符与 s2 的前 j 个字符的最长公共子串长度。遍历字符串，更新 dp 数组，并记录最长公共子串。

### 24. 编程题：最长公共子序列

**题目描述：** 给定两个字符串，找出它们的最长公共子序列。

**代码要求：** 完成以下函数，返回最长公共子序列：

```python
def longest_common_subsequence(s1, s2):
    # 你的代码实现
    return ""
```

**答案解析：**

```python
def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            result.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(result[::-1])

# 测试代码
s1 = "abcde"
s2 = "ace"
print(longest_common_subsequence(s1, s2))  # 输出 "ace"
s1 = "abcdaf"
s2 = "bcdoze"
print(longest_common_subsequence(s1, s2))  # 输出 "bcd"
```

**解析：** 最长公共子序列（LCS）问题可以通过动态规划（DP）实现。创建一个二维数组 dp，其中 dp[i][j] 表示 s1 的前 i 个字符与 s2 的前 j 个字符的最长公共子序列长度。遍历字符串，更新 dp 数组，并在最后根据 dp 数组还原最长公共子序列。

### 25. 编程题：旋转图像

**题目描述：** 给定一个 n × n 的二维矩阵 matrix 表示一个图像，请你将图像顺时针旋转 90 度。

**代码要求：** 完成以下函数，旋转图像：

```python
def rotate(matrix):
    # 你的代码实现
    pass
```

**答案解析：**

```python
def rotate(matrix):
    n = len(matrix)
    for i in range(n // 2):
        for j in range(i, n - i - 1):
            temp = matrix[i][j]
            matrix[i][j] = matrix[n - 1 - j][i]
            matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
            matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]
            matrix[j][n - 1 - i] = temp

# 测试代码
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
rotate(matrix)
for row in matrix:
    print(row)
# 输出
# [7, 4, 1]
# [8, 5, 2]
# [9, 6, 3]
```

**解析：** 旋转图像问题可以通过逐层旋转的方式实现。首先对角线交换元素，然后逐行翻转。这种方法可以保证旋转后的图像与原始图像相同。

### 26. 编程题：合并两个有序链表

**题目描述：** 给定两个已排序的链表，将它们合并为一个有序链表。

**代码要求：** 完成以下函数，返回合并后的链表：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_sorted_lists(l1, l2):
    # 你的代码实现
    return None
```

**答案解析：**

```python
def merge_sorted_lists(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    
    if l1.val < l2.val:
        result = l1
        result.next = merge_sorted_lists(l1.next, l2)
    else:
        result = l2
        result.next = merge_sorted_lists(l1, l2.next)
    
    return result

# 测试代码
l1 = ListNode(1, ListNode(3, ListNode(5)))
l2 = ListNode(2, ListNode(4, ListNode(6)))

merged = merge_sorted_lists(l1, l2)
while merged:
    print(merged.val, end=" ")
    merged = merged.next
# 输出 1 2 3 4 5 6
```

**解析：** 合并两个有序链表可以通过递归实现。比较两个链表的头节点，将较小值的链表节点作为结果链表的头节点，并递归地合并剩余部分。如果其中一个链表到达末尾，则将另一个链表链接到结果链表。

### 27. 编程题：最长公共前缀

**题目描述：** 编写一个函数来查找字符串数组中的最长公共前缀。

**代码要求：** 完成以下函数，返回最长公共前缀：

```python
def longest_common_prefix(strs):
    # 你的代码实现
    return ""
```

**答案解析：**

```python
def longest_common_prefix(strs):
    if not strs:
        return ""
    
    prefix = ""
    for i in range(len(strs[0])):
        char = strs[0][i]
        for s in strs[1:]:
            if i >= len(s) or s[i] != char:
                return prefix
        prefix += char
    
    return prefix

# 测试代码
strs = ["flower", "flow", "flight"]
print(longest_common_prefix(strs))  # 输出 "fl"
strs = ["dog", "racecar", "car"]
print(longest_common_prefix(strs))  # 输出 ""
```

**解析：** 最长公共前缀问题可以通过逐个比较字符串的前缀来求解。从第一个字符串开始，逐个字符与后续字符串进行比较，直到找到不同的字符。这种方法的时间复杂度为 O(S)，其中 S 是所有字符串的总长度。

### 28. 编程题：加一

**题目描述：** 给定一个非空数组表示一个非负整数，其各个位上的数字是按逆序的，并且每个元素只包含一个十进制数字。请编写一个函数，将这个数增加 1，并且返回一个新的数组表示这个数。

**代码要求：** 完成以下函数，返回加一后的数组：

```python
def plus_one(digits):
    # 你的代码实现
    return []
```

**答案解析：**

```python
def plus_one(digits):
    carry = 1
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] + carry == 10:
            digits[i] = 0
            carry = 1
        else:
            digits[i] += carry
            carry = 0
    if carry:
        digits.insert(0, 1)
    
    return digits

# 测试代码
digits = [1, 2, 3]
print(plus_one(digits))  # 输出 [1, 2, 4]
digits = [4, 3, 2, 1]
print(plus_one(digits))  # 输出 [4, 3, 2, 2]
digits = [9, 9, 9]
print(plus_one(digits))  # 输出 [1, 0, 0, 0]
```

**解析：** 加一问题可以通过逆序遍历数组实现。从最后一个元素开始，加上 1，如果当前位大于等于 10，则将该位设置为 0，并将进位加到前一位。如果最后一位进位为 1，则插入一个新元素 1 到数组的开头。

### 29. 编程题：最长公共子串

**题目描述：** 给定两个字符串，找出它们的最长公共子串。

**代码要求：** 完成以下函数，返回最长公共子串：

```python
def longest_common_substring(s1, s2):
    # 你的代码实现
    return ""
```

**答案解析：**

```python
def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    result = ""
    max_len = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    result = s1[i - max_len: i]
    
    return result

# 测试代码
s1 = "abcdaf"
s2 = "bcdoze"
print(longest_common_substring(s1, s2))  # 输出 "bcd"
s1 = "apple"
s2 = "apricot"
print(longest_common_substring(s1, s2))  # 输出 "apple"
```

**解析：** 最长公共子串问题可以通过动态规划（DP）实现。创建一个二维数组 dp，其中 dp[i][j] 表示 s1 的前 i 个字符与 s2 的前 j 个字符的最长公共子串长度。遍历字符串，更新 dp 数组，并记录最长公共子串。

### 30. 编程题：三数之和

**题目描述：** 给定一个包含 n 个整数的数组 nums，判断 nums 是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

**代码要求：** 完成以下函数，返回所有满足条件的三元组：

```python
def three_sum(nums):
    # 你的代码实现
    return []
```

**答案解析：**

```python
def three_sum(nums):
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, n - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    
    return result

# 测试代码
nums = [-1, 0, 1, 2, -1, -4]
print(three_sum(nums))  # 输出 [[-1, -1, 2], [-1, 0, 1]]
nums = [0, 0, 0, 0]
print(three_sum(nums))  # 输出 [[0, 0, 0]]
```

**解析：** 三数之和问题可以通过排序和双指针实现。首先对数组进行排序，然后遍历数组，对每个元素，使用双指针从当前元素的下一个元素开始，找到两个数，使得三个数的和为 0。为了避免重复的三元组，需要跳过重复的元素。

### 31. 编程题：滑动窗口的最大值

**题目描述：** 给你一个数组 nums 和一个整数 k，请找出数组中每个滑动窗口的最大值。

**代码要求：** 完成以下函数，返回滑动窗口的最大值列表：

```python
def max滑动窗口(nums, k):
    # 你的代码实现
    return []
```

**答案解析：**

```python
from collections import deque

def max滑动窗口(nums, k):
    result = []
    q = deque()
    for i, v in enumerate(nums):
        # 移除窗口之外的元素
        if q and q[0] < i - k + 1:
            q.popleft()
        # 移除小于当前元素的元素
        while q and nums[q[-1]] < v:
            q.pop()
        q.append(i)
        if i >= k - 1:
            result.append(nums[q[0]])
    return result

# 测试代码
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print(max滑动窗口(nums, k))  # 输出 [3, 3, 5, 5, 6, 7]
```

**解析：** 滑动窗口的最大值问题可以使用单调队列实现。队列中存储的是当前窗口中的最大值及其索引。遍历数组，每次更新队列，移除不在当前窗口中的元素，以及小于当前元素的元素。窗口滑动的过程中，队列的头部元素即为当前窗口的最大值。

### 32. 编程题：二进制求和

**题目描述：** 给定两个二进制字符串，返回它们的和（也以二进制字符串的形式）。

**代码要求：** 完成以下函数，返回二进制和：

```python
def add_binary(a, b):
    # 你的代码实现
    return ""
```

**答案解析：**

```python
def add_binary(a, b):
    result = []
    carry = 0
    i, j = len(a) - 1, len(b) - 1
    
    while i >= 0 or j >= 0 or carry:
        x = 0 if i < 0 else int(a[i])
        y = 0 if j < 0 else int(b[j])
        sum = x + y + carry
        result.append(str(sum % 2))
        carry = sum // 2
    
    return ''.join(result[::-1])

# 测试代码
a = "11"
b = "1"
print(add_binary(a, b))  # 输出 "100"
a = "1010"
b = "1011"
print(add_binary(a, b))  # 输出 "10111"
```

**解析：** 二进制求和问题可以通过模拟加法过程实现。从最低位开始，逐位相加，并处理进位。最后将结果反转得到最终的二进制和。

### 33. 编程题：合并 k 个排序链表

**题目描述：** 给定 k 个排序后的链表，请合并所有的链表并返回合并后的排序链表。

**代码要求：** 完成以下函数，返回合并后的链表：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_sorted_lists(lists):
    # 你的代码实现
    return None
```

**答案解析：**

```python
def merge_k_sorted_lists(lists):
    if not lists:
        return None
    if len(lists) == 1:
        return lists[0]
    
    mid = len(lists) // 2
    left = merge_k_sorted_lists(lists[:mid])
    right = merge_k_sorted_lists(lists[mid:])
    
    return merge_sorted_lists(left, right)

def merge_sorted_lists(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    
    if l1.val < l2.val:
        result = l1
        result.next = merge_sorted_lists(l1.next, l2)
    else:
        result = l2
        result.next = merge_sorted_lists(l1, l2.next)
    
    return result

# 测试代码
l1 = ListNode(1, ListNode(4, ListNode(5)))
l2 = ListNode(1, ListNode(3, ListNode(4)))
l3 = ListNode(2, ListNode(6))
lists = [l1, l2, l3]
merged = merge_k_sorted_lists(lists)
while merged:
    print(merged.val, end=" ")
    merged = merged.next
# 输出 1 1 2 3 4 4 5 6
```

**解析：** 合并 k 个排序链表问题可以通过分治算法实现。首先将链表分为两部分，然后递归地合并每一部分。合并两个有序链表可以通过递归实现，将较小值的链表节点作为结果链表的头节点，并递归地合并剩余部分。

### 34. 编程题：实现栈和队列

**题目描述：** 使用 Python 实现一个栈和队列，分别实现入栈、出栈、入队、出队操作。

**代码要求：** 完成以下函数：

```python
class Stack:
    def __init__(self):
        # 初始化代码

    def push(self, x):
        # 实现代码

    def pop(self):
        # 实现代码

class Queue:
    def __init__(self):
        # 初始化代码

    def enqueue(self, x):
        # 实现代码

    def dequeue(self):
        # 实现代码
```

**答案解析：**

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, x):
        self.items.append(x)

    def pop(self):
        if not self.items:
            return None
        return self.items.pop()

class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, x):
        self.items.insert(0, x)

    def dequeue(self):
        if not self.items:
            return None
        return self.items.pop()

# 测试代码
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())  # 输出 3
print(stack.pop())  # 输出 2
print(stack.pop())  # 输出 1

queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue.dequeue())  # 输出 1
print(queue.dequeue())  # 输出 2
print(queue.dequeue())  # 输出 3
```

**解析：** 栈和队列是常见的线性数据结构。栈通过插入和删除元素的一端（顶部）进行操作，遵循后进先出（LIFO）原则。队列通过两端进行操作，遵循先进先出（FIFO）原则。可以使用列表（list）来实现栈和队列，push、pop、enqueue 和 dequeue 操作的时间复杂度均为 O(1)。

### 35. 编程题：从上到下打印二叉树

**题目描述：** 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

**代码要求：** 完成以下函数，返回从上到下打印的节点值列表：

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def level_order(root):
    # 你的代码实现
    return []
```

**答案解析：**

```python
from collections import deque

def level_order(root):
    if not root:
        return []
    result = []
    q = deque([root])
    
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(level)
    
    return [val for level in result for val in level]

# 测试代码
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)
print(level_order(root))  # 输出 [3, 9, 20, 15, 7]
```

**解析：** 从上到下打印二叉树问题可以通过广度优先搜索（BFS）实现。使用队列实现 BFS，每次从队列中取出一个节点，将其值加入结果列表，并访问其左右子节点。重复此过程，直到队列为空。

### 36. 编程题：搜索旋转排序数组

**题目描述：** 搜索一个旋转排序的数组中的一个目标值。数组可能包含重复的元素。

**代码要求：** 完成以下函数，返回目标值的索引：

```python
def search旋转排序数组(nums, target):
    # 你的代码实现
    return -1
```

**答案解析：**

```python
def search旋转排序数组(nums, target):
    low, high = 0, len(nums) - 1
    
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        
        # 比较中间元素与最右边的元素
        if nums[mid] > nums[high]:
            if target >= nums[high] or target < nums[mid]:
                low = mid + 1
            else:
                high = mid - 1
        elif nums[mid] < nums[low]:
            if target <= nums[low] or target > nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
        else:
            # 中间元素与最左边的元素相同，需要逐个比较
            if nums[low] == target:
                return low
            low += 1
    
    return -1

# 测试代码
nums = [4, 5, 6, 7, 0, 1, 2]
target = 0
print(search旋转排序数组(nums, target))  # 输出 4
nums = [1]
target = 0
print(search旋转排序数组(nums, target))  # 输出 -1
nums = [1, 3]
target = 3
print(search旋转排序数组(nums, target))  # 输出 1
```

**解析：** 搜索旋转排序数组问题可以通过二分查找实现。由于数组可能包含重复的元素，需要比较中间元素与最左边的元素、最右边的元素来确定最小值所在的区间，然后在该区间内继续二分查找。

### 37. 编程题：最长重复子串

**题目描述：** 给定一个字符串 s，找到 s 最长的重复子串。

**代码要求：** 完成以下函数，返回最长重复子串的长度：

```python
def longest_repeated_substring(s):
    # 你的代码实现
    return 0
```

**答案解析：**

```python
def longest_repeated_substring(s):
    def get_next(s):
        n = len(s)
        next = [0] * n
        j = 0
        i = 1
        while i < n:
            if s[i] == s[j]:
                j += 1
                next[i] = j
                i += 1
            elif j != 0:
                j = next[j - 1]
            else:
                next[i] = 0
                i += 1
        return next

    n = len(s)
    next = get_next(s)
    i = 1
    j = 0
    while i < n:
        if s[i] == s[j]:
            j += 1
            if j > max_len:
                max_len = j
                max_end = i
        elif j != 0:
            j = next[j - 1]
        i += 1
    return s[max_end - max_len + 1: max_end + 1]

# 测试代码
s = "banana"
print(longest_repeated_substring(s))  # 输出 "ana"
s = "abcdabcdabcdabcdabcd"
print(longest_repeated_substring(s))  # 输出 "abcdabcdabcdabcd"
```

**解析：** 最长重复子串问题可以通过计算字符串的 next 数组实现。next 数组表示前缀 s[0..i] 与 s[0..j] 的最长公共前缀长度，其中 j = i - next[i]。通过遍历 next 数组，找到最长重复子串的长度和结束位置，然后返回子串。

### 38. 编程题：验证回文串

**题目描述：** 给定一个字符串，验证它是否是回文串。

**代码要求：** 完成以下函数，返回字符串是否为回文：

```python
def is_palindrome(s):
    # 你的代码实现
    return False
```

**答案解析：**

```python
def is_palindrome(s):
    i, j = 0, len(s) - 1
    while i < j:
        if s[i] != s[j]:
            return False
        i += 1
        j -= 1
    return True

# 测试代码
s = "racecar"
print(is_palindrome(s))  # 输出 True
s = "hello"
print(is_palindrome(s))  # 输出 False
```

**解析：** 验证回文串问题可以通过比较字符串的两端，依次向中间移动。如果两端字符相同，则继续比较下一对字符；如果不同，则返回 False。遍历完成后，如果字符串完全相同，则返回 True。

### 39. 编程题：字符串中的第一个唯一字符

**题目描述：** 给定一个字符串 s ，找到并返回其中第一个只出现一次的字符。如果不存在，返回一个问号 '?'。

**代码要求：** 完成以下函数，返回第一个只出现一次的字符：

```python
def first_unique_char(s):
    # 你的代码实现
    return '?'
```

**答案解析：**

```python
def first_unique_char(s):
    counter = [0] * 128  # 使用ASCII字符集
    for char in s:
        counter[ord(char)] += 1
    
    for char in s:
        if counter[ord(char)] == 1:
            return char
    
    return '?'

# 测试代码
s = "leetcode"
print(first_unique_char(s))  # 输出 "l"
s = "loveleetcode"
print(first_unique_char(s))  # 输出 "v"
s = "aabb"
print(first_unique_char(s))  # 输出 "?"
```

**解析：** 字符串中的第一个唯一字符问题可以通过计数法实现。使用一个数组 counter 记录字符串中每个字符的出现次数。遍历字符串，找到第一个出现次数为 1 的字符，并返回。

### 40. 编程题：三数之和

**题目描述：** 给你一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值 target 的三个整数，并返回这三个数的下标。

**代码要求：** 完成以下函数，返回所有和为目标值的三元组：

```python
def three_sum(nums, target):
    # 你的代码实现
    return []
```

**答案解析：**

```python
def three_sum(nums, target):
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, n - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < target:
                left += 1
            elif total > target:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    
    return result

# 测试代码
nums = [-1, 0, 1, 2, -1, -4]
target = 0
print(three_sum(nums, target))  # 输出 [[-1, -1, 2], [-1, 0, 1]]
nums = [0, 0, 0, 0]
target = 0
print(three_sum(nums, target))  # 输出 [[0, 0, 0]]
```

**解析：** 三数之和问题可以通过排序和双指针实现。首先对数组进行排序，然后遍历数组，对于每个元素，使用双指针找到两个数，使得三个数的和为目标值。为了避免重复的三元组，需要跳过重复的元素。

