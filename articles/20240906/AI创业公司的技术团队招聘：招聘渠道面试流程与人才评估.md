                 



# 输出结果
--------------------------------------------------------

### 1. 判断二进制数中 1 的个数

**题目：** 编写一个函数，计算二进制数 `n` 中 1 的个数。

**答案：**

```python
def hammingWeight(n: int) -> int:
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count
```

**解析：** 使用位运算，将 `n` 与 1 进行按位与操作，判断最后一位是否为 1。然后将 `n` 右移一位，继续判断下一位，直到 `n` 为 0 为止。

**进阶：** 使用 Python 的内置函数 `bin()` 和 `count()`，也可以实现：

```python
def hammingWeight(n: int) -> int:
    return bin(n).count('1')
```

### 2. 反转整数

**题目：** 编写一个函数，将 32 位有符号整数 `x` 反转。

**答案：**

```python
def reverse(x: int) -> int:
    res = 0
    while x:
        res = res * 10 + x % 10
        x //= 10
    return res
```

**解析：** 使用循环，不断将 `x` 的个位数添加到结果 `res` 的十位数位置，同时将 `x` 除以 10，去除个位数。

**进阶：** 使用位运算，将 `x` 的每一位分离出来，然后拼接到结果中：

```python
def reverse(x: int) -> int:
    res = 0
    n = 1 << 31  # 极限值
    while x != n:
        res = res * 10 + x % 10
        x //= 10
    return res
```

### 3. 合并两个有序链表

**题目：** 将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

**答案：**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        curr = dummy
        p, q = list1, list2
        while p and q:
            if p.val < q.val:
                curr.next = p
                p = p.next
            else:
                curr.next = q
                q = q.next
            curr = curr.next
        curr.next = p if p else q
        return dummy.next
```

**解析：** 创建一个虚拟头节点，然后遍历两个链表，将较小的节点链接到新链表中，直到其中一个链表结束。接着将剩下的链表链接到新链表的末尾。

### 4. 有效的括号

**题目：** 给定一个字符串 `s` ，请判断它是否是有效的括号字符串。

**答案：**

```python
from collections import deque

class Solution:
    def isValid(self, s: str) -> bool:
        q = deque()
        for c in s:
            if c in '([{':
                q.append(c)
            elif not q or (c == ')' and q[-1] != '(' or c == ']' and q[-1] != '[' or c == '}' and q[-1] != '{'):
                return False
            q.pop()
        return not q
```

**解析：** 使用栈（deque）来实现，遍历字符串，遇到左括号入栈，遇到右括号出栈，并检查出栈元素是否匹配。最后检查栈是否为空。

### 5. 最长公共前缀

**题目：** 编写一个函数来查找字符串数组中的最长公共前缀。

**答案：**

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        prefix = ""
        for c in strs[0]:
            for s in strs[1:]:
                if c not in s:
                    return prefix
            prefix += c
        return prefix
```

**解析：** 遍历第一个字符串，同时遍历其他字符串，当当前字符不在其他字符串中出现时，返回当前前缀。

### 6. 罗马数字转整数

**题目：** 罗马数字包含以下七种字符：I，V，X，L，C，D 和 M。

**答案：**

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        prev, ans = 0, 0
        for c in s:
            curr = roman[c]
            ans += curr if curr > prev else -curr
            prev = curr
        return ans
```

**解析：** 遍历字符串，将当前字符转换为整数，并与前一个字符进行比较，根据大小关系进行累加或减去当前字符的值。

### 7. 回文数

**题目：** 判断一个整数是否是回文数。

**答案：**

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        revertedNumber, x = 0, x
        while x > revertedNumber:
            revertedNumber, x = revertedNumber * 10 + x % 10, x // 10
        return x == revertedNumber or x == revertedNumber // 10
```

**解析：** 将整数反转，然后与原整数进行比较。注意处理 `0` 和奇数位的情况。

### 8. 二进制中 1 的个数

**题目：** 编写一个函数，计算一个无符号整数二进制表示中 1 的个数。

**答案：**

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        ans = 0
        while n:
            ans += n & 1
            n >>= 1
        return ans
```

**解析：** 使用位运算，将 `n` 与 1 进行按位与操作，判断最后一位是否为 1。然后将 `n` 右移一位，继续判断下一位，直到 `n` 为 0 为止。

### 9. 旋转数组

**题目：** 给定一个数组，将数组中的元素向右移动 `k` 个位置。

**答案：**

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        k %= len(nums)
        nums[:k], nums[k:] = nums[-k:], nums[:-k]
```

**解析：** 使用三个步骤：1. 逆序数组的前 `k` 个元素；2. 逆序数组的剩余元素；3. 逆序整个数组。

### 10. 相加链表

**题目：** 两个非负链表表示的非负整数相加，返回表示和的链表。

**答案：**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        curr = dummy
        carry = 0
        while l1 or l2 or carry:
            val1 = (l1.val if l1 else 0)
            val2 = (l2.val if l2 else 0)
            total = val1 + val2 + carry
            carry = total // 10
            curr.next = ListNode(total % 10)
            curr = curr.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        return dummy.next
```

**解析：** 使用虚拟头节点，遍历两个链表，计算每一位的和，如果有进位，则将进位传递给下一位。最后返回虚拟头节点的下一个节点。

### 11. 二叉树的层序遍历

**题目：** 给定一个二叉树，返回其节点值的层序遍历。

**答案：**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        q = deque([root])
        ans = []
        while q:
            ans.append([node.val for node in q])
            q = deque([child for node in q for child in (node.left, node.right) if child])
        return ans
```

**解析：** 使用广度优先搜索（BFS），队列（deque）存储节点。遍历每一层，将当前层的节点值添加到答案数组中，并将下一层的节点添加到队列中。

### 12. 有效的括号

**题目：** 给定一个只包括 `'('`，`)`，`'{'`，`'}'`，`'['`，`']'` 的字符串，判断字符串是否有效。

**答案：**

```python
from collections import deque

class Solution:
    def isValid(self, s: str) -> bool:
        q = deque()
        pairs = {'(': ')', '{': '}', '[': ']'}
        for c in s:
            if c in pairs:
                q.append(c)
            elif not q or pairs[q.pop()] != c:
                return False
        return not q
```

**解析：** 使用栈（deque）实现，遍历字符串，遇到左括号入栈，遇到右括号出栈，并检查是否匹配。

### 13. 盛最多水的容器

**题目：** 给定一个数组 `height` 表示容器的高度，求容器的最大容积。

**答案：**

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        ans, left, right = 0, 0, len(height) - 1
        while left < right:
            ans = max(ans, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return ans
```

**解析：** 双指针法，左右指针分别指向数组的两个端点，不断更新最大容积。

### 14. 合并两个有序链表

**题目：** 将两个有序链表合并为一个新的有序链表并返回。

**答案：**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1 or not list2:
            return list1 or list2
        if list1.val < list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2
```

**解析：** 递归合并两个链表，比较当前节点的值，将较小的节点链接到下一个节点。

### 15. 最长公共子序列

**题目：** 给定两个字符串，找出它们的最长公共子序列。

**答案：**

```python
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            result.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return result[::-1]
```

**解析：** 动态规划，创建一个二维数组 `dp`，其中 `dp[i][j]` 表示 `text1` 的前 `i` 个字符和 `text2` 的前 `j` 个字符的最长公共子序列的长度。遍历 `dp` 数组，找到最长公共子序列。

### 16. 两数相加

**题目：** 给出两个 非空 的链表用来表示两个非负的整数，每个节点包含一个数字。将这两个数相加返回一个新的链表。

**答案：**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        curr = dummy
        carry = 0
        while l1 or l2 or carry:
            val1 = (l1.val if l1 else 0)
            val2 = (l2.val if l2 else 0)
            total = val1 + val2 + carry
            carry = total // 10
            curr.next = ListNode(total % 10)
            curr = curr.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        return dummy.next
```

**解析：** 使用虚拟头节点，遍历两个链表，计算每一位的和，如果有进位，则将进位传递给下一位。最后返回虚拟头节点的下一个节点。

### 17. 三数之和

**题目：** 给你一个整数数组 `nums` ，判断是否存在三个数 `nums[i]`，`nums[j]` 和 `nums[k]` 使得 `i != j`，`i != k`，`j != k` 且 `nums[i] + nums[j] + nums[k] == 0` 。请

**答案：**

```python
def threeSum(nums):
    nums.sort()
    ans = []
    n = len(nums)
    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        j, k = i + 1, n - 1
        while j < k:
            total = nums[i] + nums[j] + nums[k]
            if total < 0:
                j += 1
            elif total > 0:
                k -= 1
            else:
                ans.append([nums[i], nums[j], nums[k]])
                while j < k and nums[j] == nums[j + 1]:
                    j += 1
                while j < k and nums[k] == nums[k - 1]:
                    k -= 1
                j += 1
                k -= 1
    return ans
```

**解析：** 首先对数组进行排序，然后遍历数组，使用双指针法查找两个数，使得它们的和与当前遍历的数相加等于 0。注意去除重复元素。

### 18. 盛最多水的容器

**题目：** 给定一个二

