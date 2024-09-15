                 

### PDCA实践：持续改进的法宝

#### 1. 什么是PDCA循环？

PDCA循环，即计划（Plan）、执行（Do）、检查（Check）和行动（Act）循环，是一种广泛应用于质量管理和其他管理活动中的循环改进模型。它旨在通过不断地计划、执行、检查和调整，实现持续改进。

#### 2. PDCA循环的基本步骤

**计划（Plan）：**
- 确定目标和问题
- 制定详细的计划
- 制定具体的行动方案

**执行（Do）：**
- 实施行动计划
- 按计划执行

**检查（Check）：**
- 对实施结果进行评估
- 检查目标和计划之间的差距

**行动（Act）：**
- 对成功经验进行总结和标准化
- 对问题进行纠正和改进
- 制定新的计划，开始新一轮的PDCA循环

#### 3. PDCA循环在面试和算法编程中的应用

**面试中的应用：**

在面试中，PDCA循环可以帮助你系统地准备和改进面试技能：

**Plan：**
- 确定面试的目标和要解决的问题
- 制定学习计划，包括复习知识点、做练习题、模拟面试等

**Do：**
- 按照计划进行学习和练习
- 实际参加面试，实践所学知识

**Check：**
- 评估面试的结果，包括哪些部分做得好，哪些部分需要改进
- 收集反馈，分析失败的原因

**Act：**
- 总结成功经验，将其标准化
- 针对失败的地方进行改进，制定新的学习计划

**算法编程中的应用：**

在算法编程中，PDCA循环可以帮助你优化代码和算法：

**Plan：**
- 确定算法的目标和要解决的问题
- 设计算法的框架和基本思路

**Do：**
- 编写代码，实现算法
- 运行代码，测试其性能和正确性

**Check：**
- 分析代码的执行效率，找出瓶颈
- 检查代码的逻辑，确保其正确性

**Act：**
- 对代码进行优化，提高其性能
- 重新测试代码，确保优化后的代码仍然正确

#### 4. 典型问题/面试题库

以下是国内头部一线大厂典型的高频面试题和算法编程题，以及对应的满分答案解析：

### 1. 阿里巴巴面试题：最大子序和

**题目描述：**
给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个数）。

**示例：**
```
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

**答案：**
```go
func maxSubArray(nums []int) int {
    maxSum := nums[0]
    currSum := nums[0]
    for i := 1; i < len(nums); i++ {
        currSum = max(nums[i], currSum+nums[i])
        maxSum = max(maxSum, currSum)
    }
    return maxSum
}

func max(x, y int) int {
    if x > y {
        return x
    }
    return y
}
```

**解析：** 这个算法使用了动态规划的思想，通过 `currSum` 记录当前子序列的和，如果当前元素加上 `currSum` 得到的和小于当前元素本身，说明之前的子序列和已经为负，因此需要重新开始计算。通过 `maxSum` 记录到目前为止找到的最大子序列和。

### 2. 百度面试题：合并两个有序链表

**题目描述：**
将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

**示例：**
```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**答案：**
```go
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
    if l1 == nil {
        return l2
    }
    if l2 == nil {
        return l1
    }
    if l1.Val < l2.Val {
        l1.Next = mergeTwoLists(l1.Next, l2)
        return l1
    }
    l2.Next = mergeTwoLists(l1, l2.Next)
    return l2
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 这个函数采用了递归的方法来合并两个链表。每次比较两个链表当前节点的值，选择较小的值并将其链接到结果链表中，然后递归地对下一个节点进行处理。

### 3. 腾讯面试题：最长公共子序列

**题目描述：**
给定两个字符串 text1 和 text2，返回他们的最长公共子序列的长度。

**示例：**
```
输入: text1 = "abcde", text2 = "ace" 
输出: 3 
解释: 最长公共子序列是 "ace"，它们同时出现在 text1 和 text2 中。
```

**答案：**
```go
func longestCommonSubsequence(text1 string, text2 string) int {
    m, n := len(text1), len(text2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if text1[i-1] == text2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }
    return dp[m][n]
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**解析：** 使用动态规划方法求解最长公共子序列问题。创建一个二维数组 `dp`，其中 `dp[i][j]` 表示 `text1` 的前 `i` 个字符和 `text2` 的前 `j` 个字符的最长公共子序列的长度。最后返回 `dp[m][n]`。

### 4. 字节跳动面试题：二进制求和

**题目描述：**
编写一个函数，实现二进制数求和。

**示例：**
```
输入：a = "11", b = "1"
输出："100"
```

**答案：**
```go
func addBinary(a string, b string) string {
    i, j := len(a)-1, len(b)-1
    carry := 0
    res := []byte{}
    for i >= 0 || j >= 0 || carry > 0 {
        x := 0
        if i >= 0 {
            x += int(a[i] - '0')
            i--
        }
        y := 0
        if j >= 0 {
            y += int(b[j] - '0')
            j--
        }
        sum := x + y + carry
        carry = sum / 2
        res = append(res, byte((sum % 2) + '0'))
    }
    reverse(res)
    return string(res)
}

func reverse(s []byte) {
    n := len(s)
    for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
        s[i], s[j] = s[j], s[i]
    }
}
```

**解析：** 从两个二进制数的最低位开始相加，加上之前的进位。如果相加的结果大于等于 2，则需要进位。最后将得到的结果进行反转，得到最终的二进制和。

### 5. 拼多多面试题：合并区间

**题目描述：**
以数组 intervals 表示若干个区间的集合，其中 intervals[i] = [starti, endi] 。你需要合并所有重叠的区间，并返回一个不重叠的区间数组。

**示例：**
```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
```

**答案：**
```go
func merge(intervals [][]int) [][]int {
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    var res [][]int
    for _, interval := range intervals {
        if len(res) == 0 || res[len(res)-1][1] < interval[0] {
            res = append(res, interval)
        } else {
            res[len(res)-1][1] = max(res[len(res)-1][1], interval[1])
        }
    }
    return res
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**解析：** 首先对区间数组按照起始点排序。然后遍历排序后的区间数组，如果当前区间与上一区间不重叠，则直接将当前区间添加到结果数组中。如果当前区间与上一区间重叠，则合并这两个区间。

### 6. 京东面试题：矩阵中的路径

**题目描述：**
给定一个包含字母的矩阵 board 和一个字符串 word ，找出是否可以通过从左上角到右下角的一条路径形成给定的字符串 word 。路径需要满足以下要求：
- 每一步只能向下或者向右移动。
- 不能重复访问同一位置。

**示例：**
```
输入：board = [
  ['A', 'B', 'C', 'E'],
  ['S', 'F', 'C', 'S'],
  ['A', 'D', 'E', 'E']
], word = "ABCCED"
输出：true
```

**答案：**
```go
func exist(board [][]byte, word string) bool {
    m, n := len(board), len(board[0])
    vis := make([][]bool, m)
    for i := range vis {
        vis[i] = make([]bool, n)
    }
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if dfs(board, word, 0, i, j, vis) {
                return true
            }
        }
    }
    return false
}

func dfs(board [][]byte, word string, k int, i, j int, vis [][]bool) bool {
    if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) || vis[i][j] || board[i][j] != word[k] {
        return false
    }
    if k == len(word)-1 {
        return true
    }
    vis[i][j] = true
    if dfs(board, word, k+1, i+1, j, vis) || dfs(board, word, k+1, i, j+1, vis) {
        return true
    }
    vis[i][j] = false
    return false
}
```

**解析：** 使用深度优先搜索（DFS）来查找路径。定义一个辅助函数 `dfs`，从矩阵的每个位置开始，向下或向右搜索是否有路径到达字符串的最后一个字符。

### 7. 美团面试题：有效的括号

**题目描述：**
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断是否有效。

**示例：**
```
输入: "()[]{}"
输出: true
```

**答案：**
```go
func isValid(s string) bool {
    stack := []rune{}
    m := map[rune]rune{
        ')': '(',
        ']': '[',
        '}': '{',
    }
    for _, v := range s {
        if m[v] == 0 {
            stack = append(stack, v)
        } else if len(stack) == 0 || stack[len(stack)-1] != m[v] {
            return false
        } else {
            stack = stack[:len(stack)-1]
        }
    }
    return len(stack) == 0
}
```

**解析：** 使用栈来模拟括号的匹配过程。遇到左括号时，将其压入栈中；遇到右括号时，检查栈顶元素是否与其匹配。如果不匹配或栈为空，则字符串无效。

### 8. 快手面试题：二分查找

**题目描述：**
实现一个二分查找算法，用于在一个有序数组中查找某个元素。

**示例：**
```
输入：nums = [-1, 0, 3, 5, 9, 12], target = 9
输出：4
```

**答案：**
```go
func search(nums []int, target int) int {
    left, right := 0, len(nums)-1
    for left <= right {
        mid := (left + right) / 2
        if nums[mid] == target {
            return mid
        } else if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    return -1
}
```

**解析：** 二分查找的基本实现。通过不断缩小区间，直到找到目标元素或确定其不存在。

### 9. 滴滴面试题：最长公共前缀

**题目描述：**
编写一个函数来查找字符串数组中的最长公共前缀。

**示例：**
```
输入：strs = ["flower","flow","flight"]
输出："fl"
```

**答案：**
```go
func longestCommonPrefix(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    prefix := strs[0]
    for _, s := range strs {
        for i := 0; i < len(prefix) && i < len(s); i++ {
            if prefix[i] != s[i] {
                prefix = prefix[:i]
                break
            }
        }
    }
    return prefix
}
```

**解析：** 从第一个字符串开始，逐步减少公共前缀的长度，直到找到一个公共前缀。

### 10. 小红书面试题：有效的数字

**题目描述：**
编写一个函数，确定输入字符串（字符串代表了数字）是否合法。

**示例：**
```
输入："0"
输出：true

输入："   0.1 2"
输出：false
```

**答案：**
```go
func isNumber(s string) bool {
    s = strings.TrimSpace(s)
    var dot, exp, num int
    var lastIsDigit, lastIsSign bool
    for i, c := range s {
        if c == '.' {
            if dot != 0 || exp != 0 || !lastIsDigit {
                return false
            }
            dot = 1
        } else if c == 'e' {
            if exp != 0 || dot == 0 || !lastIsDigit {
                return false
            }
            exp = 1
            lastIsDigit = false
        } else if c == '+' || c == '-' {
            if i == 0 || s[i-1] == 'e' || s[i-1] == '.' {
                return false
            }
            if i == len(s)-1 {
                return false
            }
            if !isdigit(s[i+1]) {
                return false
            }
            lastIsSign = true
        } else if isdigit(c) {
            if dot != 0 && i == len(s)-1 {
                return false
            }
            num++
            lastIsDigit = true
        } else {
            return false
        }
    }
    return num > 0
}

func isdigit(c rune) bool {
    return c >= '0' && c <= '9'
}
```

**解析：** 通过遍历字符串，判断字符是否为数字、小数点、指数符号等，确保字符串表示的数字是有效的。

### 11. 蚂蚁面试题：寻找两个正序数组的中位数

**题目描述：**
给定两个大小为 m 和 n 的正序数组 nums1 和 nums2，请你找出并返回这两个正序数组的中位数。

**示例：**
```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2

输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
```

**答案：**
```go
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
    m, n := len(nums1), len(nums2)
    if m > n {
        nums1, nums2 = nums2, nums1
        m, n = n, m
    }
    imin, imax, halfLen := 0, m, (m+n+1)/2
    for imin <= imax {
        i := (imin + imax) / 2
        j := halfLen - i
        if i < m && nums2[j-1] > nums1[i] {
            // i 太小，需要增加
            imin = i + 1
        } else if i > 0 && nums1[i-1] > nums2[j] {
            // i 太大，需要减少
            imax = i - 1
        } else {
            // 找到中位数
            if i == 0 {
                maxLeft := nums2[j-1]
            } else if j == 0 {
                maxLeft := nums1[i-1]
            } else {
                maxLeft := max(nums1[i-1], nums2[j-1])
            }
            if (m+n)%2 == 1 {
                return float64(maxLeft)
            }
            minRight := 0
            if i < m {
                minRight = nums1[i]
            }
            if j < n {
                minRight = min(minRight, nums2[j])
            }
            return float64(maxLeft+minRight)/2
        }
    }
    return 0
}
```

**解析：** 使用二分查找的方法，在两个数组中寻找中位数。通过不断缩小范围，找到两个数组合并后的中位数。

### 12. 阿里巴巴面试题：最长公共子串

**题目描述：**
给定两个字符串 `s1` 和 `s2`，找到它们最长的公共子串。

**示例：**
```
输入：s1 = "abc", s2 = "abcde"
输出："abc"
```

**答案：**
```go
func longestCommonSubstr(s1 string, s2 string) string {
    m, n := len(s1), len(s2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
        dp[i][0] = 0
    }
    maxLen, mx := 0, 0
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if s1[i-1] == s2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > maxLen {
                    maxLen = dp[i][j]
                    mx = i
                }
            }
        }
    }
    return s1[mx-maxLen : mx]
}
```

**解析：** 使用动态规划的方法，构建一个二维数组 `dp`，其中 `dp[i][j]` 表示 `s1` 的前 `i` 个字符和 `s2` 的前 `j` 个字符的最长公共子串的长度。遍历数组，更新最大长度和结束位置。

### 13. 百度面试题：重建二叉树

**题目描述：**
输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果是完整的，并且在二叉树中不存在重复的元素。

**示例：**
```
输入：
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]

输出：[3,9,20,15,7]
```

**答案：**
```go
func buildTree(preorder []int, inorder []int) *TreeNode {
    if len(preorder) == 0 {
        return nil
    }
    rootVal := preorder[0]
    root := &TreeNode{Val: rootVal}
    rootIndex := 0
    for i := 0; i < len(inorder); i++ {
        if inorder[i] == rootVal {
            rootIndex = i
            break
        }
    }
    root.Left = buildTree(preorder[1:1+rootIndex], inorder[:rootIndex])
    root.Right = buildTree(preorder[1+rootIndex:], inorder[rootIndex+1:])
    return root
}

type TreeNode struct {
    Val int
    Left *TreeNode
    Right *TreeNode
}
```

**解析：** 通过前序遍历的第一个元素作为根节点，从中序遍历中找到根节点的位置，然后递归构建左子树和右子树。

### 14. 腾讯面试题：滑动窗口的最大值

**题目描述：**
给定一个数组 `nums` 和一个整数 `k`，找到 `nums` 中的滑动窗口中的最大值。

**示例：**
```
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
```

**答案：**
```go
func maxSlidingWindow(nums []int, k int) []int {
    var res []int
    q := &list.List{}
    for _, num := range nums {
        for q.Len() > 0 && q.Front().Value.(int) <= num {
            q.Remove(q.Front())
        }
        q.PushFront(num)
        if q.Len() > k {
            q.Remove(q.Back())
        }
    }
    for q.Len() > 0 {
        res = append(res, q.Back().Value.(int))
        q.Remove(q.Back())
    }
    return res
}
```

**解析：** 使用双端队列（deque）来维护当前窗口中的最大值。当新元素大于队列末尾的元素时，移除队列末尾的元素。队列长度超过 `k` 时，移除队列末尾的元素。

### 15. 字节跳动面试题：两数相加

**题目描述：**
给定两个非空链表表示的两个非负整数，链表中的每个节点仅包含一个数字。将这两个数相加并返回一个新的链表。

**示例：**
```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,7]
```

**答案：**
```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    prev := dummy
    carry := 0
    for l1 != nil || l2 != nil || carry != 0 {
        x := 0
        if l1 != nil {
            x += l1.Val
            l1 = l1.Next
        }
        y := 0
        if l2 != nil {
            y += l2.Val
            l2 = l2.Next
        }
        sum := x + y + carry
        carry = sum / 10
        prev.Next = &ListNode{Val: sum % 10}
        prev = prev.Next
    }
    return dummy.Next
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 通过遍历两个链表，将对应的节点值相加，并加上之前的进位。如果和大于等于 10，则记录进位。最后将结果链表返回。

### 16. 京东面试题：最小栈

**题目描述：**
设计一个支持 push ，pop ，top 操作的栈。

```
push(x) -- 将元素 x 推到栈顶。
pop() -- 删除栈顶元素。
top() -- 获取栈顶元素。
empty() -- 返回栈是否为空。
```

**示例：**
```
push(1)
push(2)
top() --> 返回 2.
pop()
top() --> 返回 1.
```

**答案：**
```go
type MinStack struct {
    s   []int
    min []int
}

func Constructor() MinStack {
    return MinStack{[]int{}, []int{int(^uint(0) >> 1)}}
}

func (this *MinStack) Push(x int) {
    this.s = append(this.s, x)
    this.min = append(this.min, min(x, this.min[len(this.min)-1]))
}

func (this *MinStack) Pop() {
    this.min = this.min[:len(this.min)-1]
    this.s = this.s[:len(this.s)-1]
}

func (this *MinStack) Top() int {
    return this.s[len(this.s)-1]
}

func (this *MinStack) Empty() bool {
    return len(this.s) == 0
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

**解析：** 使用两个栈来维护数据栈和最小值栈。每次 `push` 操作时，将新元素和当前最小值加入最小值栈。每次 `pop` 操作时，同时从两个栈中弹出元素。

### 17. 美团面试题：无重复字符的最长子串

**题目描述：**
给定一个字符串 `s` ，找出其中不含有重复字符的最长子串 `T` 的长度。

**示例：**
```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**答案：**
```go
func lengthOfLongestSubstring(s string) int {
    n := len(s)
    ans := 0
    mp := map[rune]int{}
    i := 0
    for j := 0; j < n; j++ {
        if v, ok := mp[s[j]]; ok {
            i = max(i, v+1)
        }
        ans = max(ans, j-i+1)
        mp[s[j]] = j
    }
    return ans
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**解析：** 使用双指针滑动窗口的方法。当遇到重复字符时，移动左边界到重复字符的后一个位置。

### 18. 滴滴面试题：合并两个有序链表

**题目描述：**
将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

**示例：**
```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**答案：**
```go
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
    if l1 == nil {
        return l2
    }
    if l2 == nil {
        return l1
    }
    if l1.Val < l2.Val {
        l1.Next = mergeTwoLists(l1.Next, l2)
        return l1
    }
    l2.Next = mergeTwoLists(l1, l2.Next)
    return l2
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 递归合并两个有序链表。每次比较两个链表当前节点的值，选择较小的值并将其链接到结果链表中，然后递归地对下一个节点进行处理。

### 19. 小红书面试题：合并K个排序链表

**题目描述：**
合并K个已排序的链表，返回合并后的排序链表。请分析和描述算法的复杂度。

**示例：**
```
输入：
[
  [1,4,5],
  [1,3,4],
  [2,6]
]
输出：[1,1,2,3,4,4,5,6]
```

**答案：**
```go
func mergeKLists(lists []*ListNode) *ListNode {
    if len(lists) == 0 {
        return nil
    }
    for len(lists) > 1 {
        t := []*ListNode{}
        for i := 0; i < len(lists); i += 2 {
            if i+1 < len(lists) {
                lists[i], lists[i+1] = mergeTwoLists(lists[i], lists[i+1])
            }
            t = append(t, lists[i]...)
        }
        lists = t
    }
    return lists[0]
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 使用分治的思想，每次合并相邻的链表，直到最后合并成一个链表。时间复杂度为 `O(Nlogk)`，其中 `N` 是链表中的总节点数，`k` 是链表的个数。

### 20. 蚂蚁面试题：求最长公共前缀

**题目描述：**
编写一个函数来查找字符串数组中的最长公共前缀。

**示例：**
```
输入：strs = ["flower","flow","flight"]
输出："fl"
```

**答案：**
```go
func longestCommonPrefix(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    ans := strs[0]
    for i := 1; i < len(strs); i++ {
        for j := 0; j < len(ans) && j < len(strs[i]); j++ {
            if ans[j] != strs[i][j] {
                ans = ans[:j]
                break
            }
        }
    }
    return ans
}
```

**解析：** 遍历字符串数组，从第一个字符串开始，逐个比较后面的字符串，找出公共前缀。

### 21. 阿里巴巴面试题：设计一个LRU缓存

**题目描述：**
实现一个LRU（Least Recently Used）缓存，它应该支持以下操作：get 和 put。

```
get(key) - 如果密钥存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。
put(key, value) - 如果密钥不存在，则写入其数据值。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值。
```

**示例：**
```
LRUCache cache = new LRUCache( 2 );
cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // 返回 1
cache.put(3, 3);    // 容量满，该键值对被替换
cache.get(2);       // 返回 -1 (未找到)
cache.put(4, 4);    // 容量满，该键值对被替换
cache.get(1);       // 返回 -1 (未找到)
cache.get(3);       // 返回 3
cache.get(4);       // 返回 4
```

**答案：**
```go
type LRUCache struct {
    m map[int]int
    k []int
    capacity int
}

func Constructor(capacity int) LRUCache {
    return LRUCache{
        m:       map[int]int{},
        k:       make([]int, capacity),
        capacity: capacity,
    }
}

func (this *LRUCache) Get(key int) int {
    if v, ok := this.m[key]; ok {
        idx := -1
        for i, k := range this.k {
            if k == key {
                idx = i
                break
            }
        }
        if idx != -1 {
            this.k = append(this.k[:idx], this.k[idx+1:]...)
            this.k = append(this.k[:1], this.k[1:]...)
        }
        return v
    }
    return -1
}

func (this *LRUCache) Put(key int, value int) {
    if v, ok := this.m[key]; ok {
        this.m[key] = value
        idx := -1
        for i, k := range this.k {
            if k == key {
                idx = i
                break
            }
        }
        if idx != -1 {
            this.k = append(this.k[:idx], this.k[idx+1:]...)
            this.k = append(this.k[:1], this.k[1:]...)
        }
    } else {
        if len(this.m) == this.capacity {
            delete(this.m, this.k[this.capacity-1])
            this.k = this.k[:this.capacity-1]
        }
        this.m[key] = value
        this.k = append(this.k, key)
    }
}
```

**解析：** 使用一个哈希表和一个双向链表来实现 LRU 缓存。哈希表用于快速查找和更新节点，双向链表用于维护节点的顺序。

### 22. 百度面试题：两数相加

**题目描述：**
给你两个 非空 的链表来表示两个非负的整数。每个链表中的节点都包含一个单一 的数字。将这两个数相加返回它所表示的数字。

**示例：**
```
输入：l1 = [7,2,4,3], l2 = [5,6,4]
输出：[7,8,0,7]
```

**答案：**
```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    cur := dummy
    carry := 0
    for l1 != nil || l2 != nil || carry > 0 {
        val1 := 0
        if l1 != nil {
            val1 = l1.Val
            l1 = l1.Next
        }
        val2 := 0
        if l2 != nil {
            val2 = l2.Val
            l2 = l2.Next
        }
        sum := val1 + val2 + carry
        carry = sum / 10
        cur.Next = &ListNode{Val: sum % 10}
        cur = cur.Next
    }
    return dummy.Next
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 将两个链表相加，处理进位，构建结果链表。

### 23. 腾讯面试题：寻找两个正序数组的中位数

**题目描述：**
给定两个大小为 m 和 n 的正序数组 nums1 和 nums2，请你找出并返回这两个正序数组的中位数。

**示例：**
```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2

输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
```

**答案：**
```go
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
    m, n := len(nums1), len(nums2)
    if m > n {
        nums1, nums2 = nums2, nums1
        m, n = n, m
    }
    imin, imax, halfLen := 0, m, (m+n+1)/2
    for imin <= imax {
        i := (imin + imax) / 2
        j := halfLen - i
        if i < m && nums2[j-1] > nums1[i] {
            imin = i + 1
        } else if i > 0 && nums1[i-1] > nums2[j] {
            imax = i - 1
        } else {
            if i == 0 {
                maxLeft := nums2[j-1]
            } else if j == 0 {
                maxLeft := nums1[i-1]
            } else {
                maxLeft := max(nums1[i-1], nums2[j-1])
            }
            if (m+n)%2 == 1 {
                return float64(maxLeft)
            }
            minRight := 0
            if i < m {
                minRight = nums1[i]
            }
            if j < n {
                minRight = min(minRight, nums2[j])
            }
            return float64(maxLeft+minRight)/2
        }
    }
    return 0
}
```

**解析：** 使用二分查找的方法，在两个数组中寻找中位数。通过不断缩小范围，找到两个数组合并后的中位数。

### 24. 字节跳动面试题：二进制求和

**题目描述：**
编写一个函数，实现二进制数求和。

**示例：**
```
输入：a = "11", b = "1"
输出："100"
```

**答案：**
```go
func addBinary(a string, b string) string {
    i, j := len(a)-1, len(b)-1
    carry := 0
    res := []byte{}
    for i >= 0 || j >= 0 || carry > 0 {
        x := 0
        if i >= 0 {
            x += int(a[i] - '0')
            i--
        }
        y := 0
        if j >= 0 {
            y += int(b[j] - '0')
            j--
        }
        sum := x + y + carry
        carry = sum / 2
        res = append(res, byte((sum % 2) + '0'))
    }
    reverse(res)
    return string(res)
}

func reverse(s []byte) {
    n := len(s)
    for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
        s[i], s[j] = s[j], s[i]
    }
}
```

**解析：** 从两个二进制数的最低位开始相加，加上之前的进位。如果相加的结果大于等于 2，则需要进位。最后将得到的结果进行反转，得到最终的二进制和。

### 25. 拼多多面试题：合并区间

**题目描述：**
以数组 intervals 表示若干个区间的集合，其中 intervals[i] = [starti, endi] 。你需要合并所有重叠的区间，并返回一个不重叠的区间数组。

**示例：**
```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
```

**答案：**
```go
func merge(intervals [][]int) [][]int {
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    var res [][]int
    for _, interval := range intervals {
        if len(res) == 0 || res[len(res)-1][1] < interval[0] {
            res = append(res, interval)
        } else {
            res[len(res)-1][1] = max(res[len(res)-1][1], interval[1])
        }
    }
    return res
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**解析：** 首先对区间数组按照起始点排序。然后遍历排序后的区间数组，如果当前区间与上一区间不重叠，则直接将当前区间添加到结果数组中。如果当前区间与上一区间重叠，则合并这两个区间。

### 26. 京东面试题：旋转图像

**题目描述：**
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

**示例：**
```
给出一个 5x5 的矩阵：
[
  [1,2,3],
  [4,5,6],
  [7,8,9],
  [10,11,12],
  [13,14,15]
]
原地旋转输入矩阵，使其变为：
[
  [13, 9, 5, 1],
  [14,10,6,2],
  [15,11,7,3],
  [ 8,12, 4, 6],
  [ 7, 6, 5, 4]
]
```

**答案：**
```go
func rotate(matrix [][]int) {
    n := len(matrix)
    // 旋转外层循环
    for i := 0; i < n/2; i++ {
        for j := i; j < n-i-1; j++ {
            temp := matrix[i][j]
            matrix[i][j] = matrix[n-1-j][i]
            matrix[n-1-j][i] = matrix[n-1-i][n-1-j]
            matrix[n-1-i][n-1-j] = matrix[j][n-1-i]
            matrix[j][n-1-i] = temp
        }
    }
}
```

**解析：** 先旋转矩阵的外围循环，然后依次旋转内层循环，直到全部旋转完毕。

### 27. 美团面试题：单调栈

**题目描述：**
用单调栈解决以下问题：
- 给定一个数组 `prices`，其中 `prices[i]` 是第 `i` 天的股票价格。
- 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

**示例：**
```
输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3 。
```

**答案：**
```go
func maxProfit(prices []int) int {
    stk := []int{}
    ans := 0
    for _, price := range prices {
        for len(stk) > 0 && stk[len(stk)-1] > price {
            stk = stk[:len(stk)-1]
            ans += stk[len(stk)-1] - price
        }
        stk = append(stk, price)
    }
    return ans
}
```

**解析：** 使用单调栈，当栈顶元素大于当前元素时，弹出栈顶元素，计算利润。否则，将当前元素入栈。

### 28. 快手面试题：计算器

**题目描述：**
实现一个简单的计算器，可以处理加、减、乘、除四种运算。

**示例：**
```
输入：(10 + 5) * 2 / (5 + 2)
输出：8
```

**答案：**
```go
func calculate(s string) float64 {
    nums := []float64{}
    ops := []rune{}
    i := 0
    n := len(s)
    for i < n {
        if s[i] == ' ' {
            i++
            continue
        }
        if isDigit(s[i]) {
            num := 0.0
            for i < n && isDigit(s[i]) {
                num = num*10 + float64(s[i]-'0')
                i++
            }
            nums = append(nums, num)
        } else {
            ops = append(ops, s[i])
            i++
        }
    }
    ans, _ := eval(nums, ops)
    return ans
}

func eval(nums []float64, ops []rune) (float64, bool) {
    var stk []float64
    for i, num := range nums {
        stk = append(stk, num)
        for len(stk) >= 2 && len(ops) > 0 && prec(ops[len(ops)-1]) >= prec(ops[len(ops)-2]) {
            right, pop := stk[len(stk)-1], stk[len(stk)-2]
            stk = stk[:len(stk)-2]
            op := ops[len(ops)-1]
            ops = ops[:len(ops)-1]
            left := stk[len(stk)-1]
            stk = append(stk, evalOp(left, right, op))
        }
    }
    if len(ops) > 0 {
        return 0, false
    }
    return stk[0], true
}

func prec(op rune) int {
    switch op {
    case '+', '-':
        return 1
    case '*', '/':
        return 2
    }
    return -1
}

func isDigit(ch rune) bool {
    return '0' <= ch && ch <= '9'
}

func evalOp(left, right float64, op rune) float64 {
    switch op {
    case '+':
        return left + right
    case '-':
        return left - right
    case '*':
        return left * right
    case '/':
        if right == 0 {
            return 0
        }
        return left / right
    }
    return 0
}
```

**解析：** 使用逆波兰表达式（后缀表达式）求解。首先将表达式转换为逆波兰表达式，然后使用栈进行计算。

### 29. 滴滴面试题：爬楼梯

**题目描述：**
假设你正在爬楼梯。需要 n 阶台阶才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**示例：**
```
输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
```

**答案：**
```go
func climbStairs(n int) int {
    if n <= 2 {
        return n
    }
    a, b := 1, 2
    for i := 3; i <= n; i++ {
        c := a + b
        a = b
        b = c
    }
    return b
}
```

**解析：** 使用动态规划的方法。设 `a` 和 `b` 分别表示前两个台阶的方法数，每次更新 `b` 为 `a + b`。

### 30. 小红书面试题：删除链表的倒数第 N 个节点

**题目描述：**
给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头节点。

**示例：**
```
给定一个链表: 1->2->3->4->5, 和 n = 2.
当删除了倒数第二个节点后，链表变为 1->2->3->5.
```

**答案：**
```go
func removeNthFromEnd(head *ListNode, n int) *ListNode {
    dummy := &ListNode{Val: 0, Next: head}
    fast, slow := dummy, dummy
    for i := 0; i < n; i++ {
        fast = fast.Next
    }
    for fast != nil {
        fast = fast.Next
        slow = slow.Next
    }
    slow.Next = slow.Next.Next
    return dummy.Next
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 使用快慢指针的方法。快指针先走 `n` 步，然后快慢指针同时前进，当快指针到达末尾时，慢指针正好位于倒数第 `n` 个节点的前一个节点。

### 31. 蚂蚁面试题：两数相加

**题目描述：**
给定两个非空链表表示的两个非负的整数，链表中的每个节点仅包含一个数字。将这两个数相加并返回它所表示的数字。

**示例：**
```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,7]
```

**答案：**
```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    cur := dummy
    carry := 0
    for l1 != nil || l2 != nil || carry > 0 {
        val1 := 0
        if l1 != nil {
            val1 = l1.Val
            l1 = l1.Next
        }
        val2 := 0
        if l2 != nil {
            val2 = l2.Val
            l2 = l2.Next
        }
        sum := val1 + val2 + carry
        carry = sum / 10
        cur.Next = &ListNode{Val: sum % 10}
        cur = cur.Next
    }
    return dummy.Next
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 将两个链表相加，处理进位，构建结果链表。

### 32. 阿里巴巴面试题：搜索旋转排序数组

**题目描述：**
假设按照升序排序的数组在预先未知的某个点上进行了旋转。

例如，数组 `[0,1,2,4,5,6,7]` 可能变为 `[4,5,6,7,0,1,2]`。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 `-1`。

你可以假设数组中不存在重复的元素。

你的算法应该具有 O(log n) 时间复杂度。

**示例：**
```
输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4

输入: nums = [4,5,6,7,0,1,2], target = 3
输出: -1
```

**答案：**
```go
func search(nums []int, target int) int {
    l, r := 0, len(nums)-1
    for l <= r {
        mid := (l + r) / 2
        if nums[mid] == target {
            return mid
        }
        if nums[l] <= nums[mid] {
            if target >= nums[l] && target < nums[mid] {
                r = mid - 1
            } else {
                l = mid + 1
            }
        } else {
            if target > nums[r] || target < nums[l] {
                r = mid - 1
            } else {
                l = mid + 1
            }
        }
    }
    return -1
}
```

**解析：** 使用二分查找。当数组被旋转时，可以判断左半部分是否有序，然后根据目标值位于左半部分还是右半部分来更新左右边界。

### 33. 百度面试题：两数相加

**题目描述：**
给你两个 非空 的链表来表示两个非负的整数。每个链表中的节点都包含一个单一 的数字。将这两个数相加返回它所表示的数字。

**示例：**
```
输入：l1 = [7,2,4,3], l2 = [5,6,4]
输出：[7,8,0,7]
```

**答案：**
```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    cur := dummy
    carry := 0
    for l1 != nil || l2 != nil || carry > 0 {
        val1 := 0
        if l1 != nil {
            val1 = l1.Val
            l1 = l1.Next
        }
        val2 := 0
        if l2 != nil {
            val2 = l2.Val
            l2 = l2.Next
        }
        sum := val1 + val2 + carry
        carry = sum / 10
        cur.Next = &ListNode{Val: sum % 10}
        cur = cur.Next
    }
    return dummy.Next
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 将两个链表相加，处理进位，构建结果链表。

### 34. 腾讯面试题：删除链表的倒数第 N 个节点

**题目描述：**
给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头节点。

**示例：**
```
给定一个链表: 1->2->3->4->5, 和 n = 2.
当删除了倒数第二个节点后，链表变为 1->2->3->5.
```

**答案：**
```go
func removeNthFromEnd(head *ListNode, n int) *ListNode {
    dummy := &ListNode{0, head}
    fast, slow := dummy, dummy
    for i := 0; i < n; i++ {
        fast = fast.Next
    }
    for fast != nil {
        fast = fast.Next
        slow = slow.Next
    }
    slow.Next = slow.Next.Next
    return dummy.Next
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 使用快慢指针的方法。快指针先走 `n` 步，然后快慢指针同时前进，当快指针到达末尾时，慢指针正好位于倒数第 `n` 个节点的前一个节点。

### 35. 字节跳动面试题：无重复字符的最长子串

**题目描述：**
给定一个字符串 s ，找出其中不含有重复字符的最长子串的长度。

**示例：**
```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**答案：**
```go
func lengthOfLongestSubstring(s string) int {
    i, j := 0, 0
    cnt := 0
    for i < len(s) {
        for j < len(s) && s[j] != s[i] {
            cnt = max(cnt, j-i+1)
            j++
        }
        i++
        j++
    }
    return cnt
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**解析：** 使用滑动窗口的方法。当窗口内的字符串不包含重复字符时，更新最大长度。

### 36. 拼多多面试题：合并两个有序链表

**题目描述：**
将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

**示例：**
```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**答案：**
```go
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
    if l1 == nil {
        return l2
    }
    if l2 == nil {
        return l1
    }
    if l1.Val < l2.Val {
        l1.Next = mergeTwoLists(l1.Next, l2)
        return l1
    }
    l2.Next = mergeTwoLists(l1, l2.Next)
    return l2
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 递归合并两个有序链表。每次比较两个链表当前节点的值，选择较小的值并将其链接到结果链表中，然后递归地对下一个节点进行处理。

### 37. 京东面试题：爬楼梯

**题目描述：**
假设你正在爬楼梯。需要 n 阶台阶才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**示例：**
```
输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
```

**答案：**
```go
func climbStairs(n int) int {
    if n <= 2 {
        return n
    }
    a, b := 1, 2
    for i := 3; i <= n; i++ {
        c := a + b
        a = b
        b = c
    }
    return b
}
```

**解析：** 使用动态规划的方法。设 `a` 和 `b` 分别表示前两个台阶的方法数，每次更新 `b` 为 `a + b`。

### 38. 美团面试题：两数相加

**题目描述：**
给你两个非空链表表示的两个非负的整数，链表中的每个节点仅包含一个数字。将这两个数相加并返回它所表示的数字。

**示例：**
```
输入：l1 = [7,2,4,3], l2 = [5,6,4]
输出：[7,8,0,7]
```

**答案：**
```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    cur := dummy
    carry := 0
    for l1 != nil || l2 != nil || carry > 0 {
        val1 := 0
        if l1 != nil {
            val1 = l1.Val
            l1 = l1.Next
        }
        val2 := 0
        if l2 != nil {
            val2 = l2.Val
            l2 = l2.Next
        }
        sum := val1 + val2 + carry
        carry = sum / 10
        cur.Next = &ListNode{Val: sum % 10}
        cur = cur.Next
    }
    return dummy.Next
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 将两个链表相加，处理进位，构建结果链表。

### 39. 滴滴面试题：合并两个有序链表

**题目描述：**
将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

**示例：**
```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**答案：**
```go
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
    if l1 == nil {
        return l2
    }
    if l2 == nil {
        return l1
    }
    if l1.Val < l2.Val {
        l1.Next = mergeTwoLists(l1.Next, l2)
        return l1
    }
    l2.Next = mergeTwoLists(l1, l2.Next)
    return l2
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 递归合并两个有序链表。每次比较两个链表当前节点的值，选择较小的值并将其链接到结果链表中，然后递归地对下一个节点进行处理。

### 40. 小红书面试题：两数相加

**题目描述：**
给定两个非空链表表示的两个非负的整数，链表中的每个节点仅包含一个数字。将这两个数相加并返回它所表示的数字。

**示例：**
```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,7]
```

**答案：**
```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    cur := dummy
    carry := 0
    for l1 != nil || l2 != nil || carry > 0 {
        val1 := 0
        if l1 != nil {
            val1 = l1.Val
            l1 = l1.Next
        }
        val2 := 0
        if l2 != nil {
            val2 = l2.Val
            l2 = l2.Next
        }
        sum := val1 + val2 + carry
        carry = sum / 10
        cur.Next = &ListNode{Val: sum % 10}
        cur = cur.Next
    }
    return dummy.Next
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 将两个链表相加，处理进位，构建结果链表。

### 41. 蚂蚁面试题：两数相加

**题目描述：**
给定两个非空链表表示的两个非负的整数，链表中的每个节点仅包含一个数字。将这两个数相加并返回它所表示的数字。

**示例：**
```
输入：l1 = [7,2,4,3], l2 = [5,6,4]
输出：[7,8,0,7]
```

**答案：**
```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    cur := dummy
    carry := 0
    for l1 != nil || l2 != nil || carry > 0 {
        val1 := 0
        if l1 != nil {
            val1 = l1.Val
            l1 = l1.Next
        }
        val2 := 0
        if l2 != nil {
            val2 = l2.Val
            l2 = l2.Next
        }
        sum := val1 + val2 + carry
        carry = sum / 10
        cur.Next = &ListNode{Val: sum % 10}
        cur = cur.Next
    }
    return dummy.Next
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 将两个链表相加，处理进位，构建结果链表。

### 42. 阿里巴巴面试题：寻找两个正序数组的中位数

**题目描述：**
给定两个大小为 m 和 n 的正序数组 nums1 和 nums2，请你找出并返回这两个正序数组的中位数。

**示例：**
```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：因为 nums1 中有 1 个元素，nums2 中有 1 个元素，所以它们中位数的
```

**答案：**
```go
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
    m, n := len(nums1), len(nums2)
    if m > n {
        nums1, nums2 = nums2, nums1
        m, n = n, m
    }
    imin, imax, halfLen := 0, m, (m+n+1)/2
    for imin <= imax {
        i := (imin + imax) / 2
        j := halfLen - i
        if i < m && nums2[j-1] > nums1[i] {
            imin = i + 1
        } else if i > 0 && nums1[i-1] > nums2[j] {
            imax = i - 1
        } else {
            if i == 0 {
                maxLeft := nums2[j-1]
            } else if j == 0 {
                maxLeft := nums1[i-1]
            } else {
                maxLeft := max(nums1[i-1], nums2[j-1])
            }
            if (m+n)%2 == 1 {
                return float64(maxLeft)
            }
            minRight := 0
            if i < m {
                minRight = nums1[i]
            }
            if j < n {
                minRight = min(minRight, nums2[j])
            }
            return float64(maxLeft+minRight)/2
        }
    }
    return 0
}
```

**解析：** 使用二分查找的方法，在两个数组中寻找中位数。通过不断缩小范围，找到两个数组合并后的中位数。

### 43. 百度面试题：删除链表的倒数第 N 个节点

**题目描述：**
给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头节点。

**示例：**
```
给定一个链表: 1->2->3->4->5, 和 n = 2.
当删除了倒数第二个节点后，链表变为 1->2->3->5.
```

**答案：**
```go
func removeNthFromEnd(head *ListNode, n int) *ListNode {
    dummy := &ListNode{0, head}
    fast, slow := dummy, dummy
    for i := 0; i < n; i++ {
        fast = fast.Next
    }
    for fast != nil {
        fast = fast.Next
        slow = slow.Next
    }
    slow.Next = slow.Next.Next
    return dummy.Next
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 使用快慢指针的方法。快指针先走 `n` 步，然后快慢指针同时前进，当快指针到达末尾时，慢指针正好位于倒数第 `n` 个节点的前一个节点。

### 44. 腾讯面试题：最长公共子序列

**题目描述：**
给定两个字符串 text1 和 text2，返回它们的最大公共子序列的长度。

**示例：**
```
输入: text1 = "abcde", text2 = "ace" 
输出: 3 
解释: 最长公共子序列是 "ace"，它们同时出现在 text1 和 text2 中。
```

**答案：**
```go
func longestCommonSubsequence(text1 string, text2 string) int {
    m, n := len(text1), len(text2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if text1[i-1] == text2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }
    return dp[m][n]
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**解析：** 使用动态规划的方法求解最长公共子序列问题。创建一个二维数组 `dp`，其中 `dp[i][j]` 表示 `text1` 的前 `i` 个字符和 `text2` 的前 `j` 个字符的最长公共子序列的长度。最后返回 `dp[m][n]`。

### 45. 字节跳动面试题：无重复字符的最长子串

**题目描述：**
给定一个字符串 s ，找出其中不含有重复字符的最长子串的长度。

**示例：**
```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**答案：**
```go
func lengthOfLongestSubstring(s string) int {
    i, j := 0, 0
    cnt := 0
    for i < len(s) {
        for j < len(s) && s[j] != s[i] {
            cnt = max(cnt, j-i+1)
            j++
        }
        i++
        j++
    }
    return cnt
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**解析：** 使用滑动窗口的方法。当窗口内的字符串不包含重复字符时，更新最大长度。

### 46. 拼多多面试题：合并区间

**题目描述：**
以数组 intervals 表示若干个区间的集合，其中 intervals[i] = [starti, endi] 。你需要合并所有重叠的区间，并返回一个不重叠的区间数组。

**示例：**
```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
```

**答案：**
```go
func merge(intervals [][]int) [][]int {
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    var res [][]int
    for _, interval := range intervals {
        if len(res) == 0 || res[len(res)-1][1] < interval[0] {
            res = append(res, interval)
        } else {
            res[len(res)-1][1] = max(res[len(res)-1][1], interval[1])
        }
    }
    return res
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**解析：** 首先对区间数组按照起始点排序。然后遍历排序后的区间数组，如果当前区间与上一区间不重叠，则直接将当前区间添加到结果数组中。如果当前区间与上一区间重叠，则合并这两个区间。

### 47. 京东面试题：寻找两个正序数组的中位数

**题目描述：**
给定两个大小为 m 和 n 的正序数组 nums1 和 nums2，请你找出并返回这两个正序数组的中位数。

**示例：**
```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：因为 nums1 中有 1 个元素，nums2 中有 1 个元素，所以它们中位数的
```

**答案：**
```go
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
    m, n := len(nums1), len(nums2)
    if m > n {
        nums1, nums2 = nums2, nums1
        m, n = n, m
    }
    imin, imax, halfLen := 0, m, (m+n+1)/2
    for imin <= imax {
        i := (imin + imax) / 2
        j := halfLen - i
        if i < m && nums2[j-1] > nums1[i] {
            imin = i + 1
        } else if i > 0 && nums1[i-1] > nums2[j] {
            imax = i - 1
        } else {
            if i == 0 {
                maxLeft := nums2[j-1]
            } else if j == 0 {
                maxLeft := nums1[i-1]
            } else {
                maxLeft := max(nums1[i-1], nums2[j-1])
            }
            if (m+n)%2 == 1 {
                return float64(maxLeft)
            }
            minRight := 0
            if i < m {
                minRight = nums1[i]
            }
            if j < n {
                minRight = min(minRight, nums2[j])
            }
            return float64(maxLeft+minRight)/2
        }
    }
    return 0
}
```

**解析：** 使用二分查找的方法，在两个数组中寻找中位数。通过不断缩小范围，找到两个数组合并后的中位数。

### 48. 美团面试题：两数相加

**题目描述：**
给定两个非空链表表示的两个非负的整数，链表中的每个节点仅包含一个数字。将这两个数相加并返回它所表示的数字。

**示例：**
```
输入：l1 = [7,2,4,3], l2 = [5,6,4]
输出：[7,8,0,7]
```

**答案：**
```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    cur := dummy
    carry := 0
    for l1 != nil || l2 != nil || carry > 0 {
        val1 := 0
        if l1 != nil {
            val1 = l1.Val
            l1 = l1.Next
        }
        val2 := 0
        if l2 != nil {
            val2 = l2.Val
            l2 = l2.Next
        }
        sum := val1 + val2 + carry
        carry = sum / 10
        cur.Next = &ListNode{Val: sum % 10}
        cur = cur.Next
    }
    return dummy.Next
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 将两个链表相加，处理进位，构建结果链表。

### 49. 滴滴面试题：合并两个有序链表

**题目描述：**
将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

**示例：**
```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**答案：**
```go
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
    if l1 == nil {
        return l2
    }
    if l2 == nil {
        return l1
    }
    if l1.Val < l2.Val {
        l1.Next = mergeTwoLists(l1.Next, l2)
        return l1
    }
    l2.Next = mergeTwoLists(l1, l2.Next)
    return l2
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 递归合并两个有序链表。每次比较两个链表当前节点的值，选择较小的值并将其链接到结果链表中，然后递归地对下一个节点进行处理。

### 50. 小红书面试题：两数相加

**题目描述：**
给定两个非空链表表示的两个非负的整数，链表中的每个节点仅包含一个数字。将这两个数相加并返回它所表示的数字。

**示例：**
```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,7]
```

**答案：**
```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    cur := dummy
    carry := 0
    for l1 != nil || l2 != nil || carry > 0 {
        val1 := 0
        if l1 != nil {
            val1 = l1.Val
            l1 = l1.Next
        }
        val2 := 0
        if l2 != nil {
            val2 = l2.Val
            l2 = l2.Next
        }
        sum := val1 + val2 + carry
        carry = sum / 10
        cur.Next = &ListNode{Val: sum % 10}
        cur = cur.Next
    }
    return dummy.Next
}

type ListNode struct {
    Val int
    Next *ListNode
}
```

**解析：** 将两个链表相加，处理进位，构建结果链表。

