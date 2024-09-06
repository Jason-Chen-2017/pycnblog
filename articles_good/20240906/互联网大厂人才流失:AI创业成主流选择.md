                 

### 1. 算法面试题：最长公共子序列 (LCS)

**题目描述：** 给定两个字符串 `str1` 和 `str2`，找出它们的最长公共子序列。最长公共子序列（Longest Common Subsequence，LCS）是在两个序列中同时出现的最长子序列。

**输入：**  
- 字符串 `str1`：`"AGGTAB"`  
- 字符串 `str2`：`"GXTXAYB"`

**输出：**  
- 最长公共子序列："GTAB"

**算法思路：**  
使用动态规划算法。创建一个二维数组 `dp`，其中 `dp[i][j]` 表示 `str1` 的前 `i` 个字符和 `str2` 的前 `j` 个字符的最长公共子序列长度。

**解题步骤：**  
1. 初始化 `dp` 数组，其中 `dp[0][j]` 和 `dp[i][0]` 都为 0。
2. 遍历字符串 `str1` 和 `str2` 的每个字符，更新 `dp` 数组。
3. 如果 `str1[i-1] == str2[j-1]`，则 `dp[i][j] = dp[i-1][j-1] + 1`。
4. 如果 `str1[i-1] != str2[j-1]`，则 `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`。

**代码示例：**

```go
func longestCommonSubsequence(str1, str2 string) string {
    m, n := len(str1), len(str2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }

    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if str1[i-1] == str2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }

    // 回溯获取最长公共子序列
    var lcs []byte
    i, j := m, n
    for i > 0 && j > 0 {
        if str1[i-1] == str2[j-1] {
            lcs = append(lcs, str1[i-1])
            i--
            j--
        } else if dp[i-1][j] > dp[i][j-1] {
            i--
        } else {
            j--
        }
    }
    // 翻转结果
    for i, j := 0, len(lcs)/2; i < j; i, j = i+1, j-1 {
        lcs[i], lcs[j] = lcs[j], lcs[i]
    }
    return string(lcs)
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**答案解析：**  
- 动态规划的核心思想是将复杂问题分解为更小的子问题，并利用子问题的解来求解原问题。
- 在这道题目中，我们使用一个二维数组 `dp` 来存储子问题的解，其中 `dp[i][j]` 表示 `str1` 的前 `i` 个字符和 `str2` 的前 `j` 个字符的最长公共子序列长度。
- 通过遍历字符串 `str1` 和 `str2` 的每个字符，我们可以计算 `dp` 数组中每个元素的值，并最终得到最长公共子序列的长度。
- 回溯法用于从 `dp` 数组中获取最长公共子序列的具体字符序列。

### 2. 数据结构与算法面试题：图遍历（深度优先搜索）

**题目描述：** 实现一个函数，用于对图进行深度优先搜索（DFS）。图可以通过邻接表或邻接矩阵表示。

**输入：**    
- 图的表示方式（邻接表或邻接矩阵）
- 起始节点

**输出：**    
- 深度优先搜索的节点序列

**算法思路：**    
- 使用递归或栈实现深度优先搜索。
- 对于当前节点，首先将其标记为已访问，然后递归或迭代地访问所有未访问的邻接节点。

**代码示例（邻接表）：**

```go
// 邻接表表示图
type Graph struct {
    V   int
    adj [][]int
}

// 初始化图
func NewGraph(V int) *Graph {
    g := &Graph{V: V}
    g.adj = make([][]int, V)
    for i := 0; i < V; i++ {
        g.adj[i] = []int{}
    }
    return g
}

// 添加边
func (g *Graph) AddEdge(v, w int) {
    g.adj[v] = append(g.adj[v], w)
    g.adj[w] = append(g.adj[w], v)
}

// 深度优先搜索
func (g *Graph) DFS(v int) []int {
    visited := make([]bool, g.V)
    var dfsVisit func(int)
    dfsVisit = func(v int) {
        visited[v] = true
        println(v)
        for _, w := range g.adj[v] {
            if !visited[w] {
                dfsVisit(w)
            }
        }
    }
    dfsVisit(v)
    return nil
}
```

**答案解析：**  
- 图的深度优先搜索是一种遍历图的方法，它可以从任意节点开始，递归或迭代地访问所有与该节点相邻的未访问节点。
- 在这个示例中，我们使用邻接表表示图，并通过递归方式实现深度优先搜索。
- 对于每个节点，我们首先将其标记为已访问，然后递归地访问所有未访问的邻接节点。
- 深度优先搜索的输出是节点的访问顺序。

### 3. 算法面试题：计算字符串中最大子串的和

**题目描述：** 给定一个字符串 `s`，计算其中最大子串的和。子串定义为字符串中连续的字符序列。

**输入：**      
- 字符串 `s`："abcd"

**输出：**      
- 最大子串和：4（"d"）

**算法思路：**      
- 使用前缀和数组，计算字符串中每个位置的前缀和。
- 遍历字符串，对于每个位置 `i`，找到包含 `i` 的最大前缀和和最小前缀和，计算它们的差值，更新最大子串和。

**代码示例：**

```go
func maxSubstrSum(s string) int {
    n := len(s)
    if n == 0 {
        return 0
    }
    maxSum := -1 << 31
    preSum := make([]int, n+1)
    preSum[0] = 0
    for i := 1; i <= n; i++ {
        preSum[i] = preSum[i-1] + int(s[i-1]-'0')
    }
    for i := 1; i <= n; i++ {
        maxSum = max(maxSum, preSum[i])
        for j := i - 1; j >= 0; j-- {
            maxSum = max(maxSum, preSum[i]-preSum[j])
        }
    }
    return maxSum
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**答案解析：**  
- 通过计算前缀和数组，我们可以快速找到包含某个位置 `i` 的最大子串和。
- 在这个示例中，我们首先计算每个位置的前缀和，然后遍历每个位置 `i`，找到包含 `i` 的最大前缀和和最小前缀和，计算它们的差值，更新最大子串和。
- 时间复杂度为 `O(n^2)`，可以通过使用双指针或差分数组优化到 `O(n)`。

### 4. 数据结构与算法面试题：树的遍历（前序、中序、后序）

**题目描述：** 实现一个二叉树的遍历函数，包括前序遍历、中序遍历和后序遍历。

**输入：**    
- 二叉树的根节点

**输出：**    
- 遍历结果（前序、中序、后序）

**算法思路：**    
- 使用递归或栈实现树的遍历。
- 对于前序遍历，首先访问根节点，然后递归地访问左子树和右子树。
- 对于中序遍历，首先递归地访问左子树，然后访问根节点，最后递归地访问右子树。
- 对于后序遍历，首先递归地访问左子树，然后递归地访问右子树，最后访问根节点。

**代码示例：**

```go
type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

func (t *TreeNode) PreOrderTraversal() {
    if t != nil {
        println(t.Val)
        if t.Left != nil {
            t.Left.PreOrderTraversal()
        }
        if t.Right != nil {
            t.Right.PreOrderTraversal()
        }
    }
}

func (t *TreeNode) InOrderTraversal() {
    if t != nil {
        if t.Left != nil {
            t.Left.InOrderTraversal()
        }
        println(t.Val)
        if t.Right != nil {
            t.Right.InOrderTraversal()
        }
    }
}

func (t *TreeNode) PostOrderTraversal() {
    if t != nil {
        if t.Left != nil {
            t.Left.PostOrderTraversal()
        }
        if t.Right != nil {
            t.Right.PostOrderTraversal()
        }
        println(t.Val)
    }
}
```

**答案解析：**    
- 树的遍历是算法和数据结构中常见的基础操作，用于访问树的所有节点。
- 前序遍历首先访问根节点，然后递归地访问左子树和右子树。
- 中序遍历首先递归地访问左子树，然后访问根节点，最后递归地访问右子树。
- 后序遍历首先递归地访问左子树，然后递归地访问右子树，最后访问根节点。
- 递归实现简单直观，但需要注意栈溢出问题。

### 5. 算法面试题：找出字符串中的最长公共前缀

**题目描述：** 给定一个字符串数组 `strs`，找出其中最长的公共前缀。

**输入：**      
- 字符串数组 `strs`：`["flower", "flow", "flight"]`

**输出：**      
- 最长公共前缀："fl"

**算法思路：**      
- 比较字符串数组中的第一个字符串与其他字符串的前缀，找到它们的最长公共前缀。

**代码示例：**

```go
func longestCommonPrefix(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    prefix := strs[0]
    for _, str := range strs[1:] {
        for i, n := 0, len(prefix); i < n; i++ {
            if i >= len(str) || str[i] != prefix[i] {
                prefix = prefix[:i]
                break
            }
        }
    }
    return prefix
}
```

**答案解析：**      
- 通过比较字符串数组中的第一个字符串与其他字符串的前缀，我们可以找到它们的最长公共前缀。
- 时间复杂度为 `O(n * m)`，其中 `n` 是字符串数组的长度，`m` 是最长公共前缀的长度。

### 6. 数据结构与算法面试题：哈希表实现

**题目描述：** 使用哈希表实现一个字典（HashMap）。

**输入：**        
- 键值对数组：`[["apple", 1], ["banana", 2], ["cherry", 3]]`

**输出：**        
- 字典：`{"apple": 1, "banana": 2, "cherry": 3}`

**算法思路：**        
- 使用哈希函数将键映射到哈希表中。
- 对于每个键值对，计算键的哈希值，并将其存储在哈希表中。
- 如果哈希表中已存在相同哈希值的键，则需要解决哈希冲突。

**代码示例：**

```go
type HashMap struct {
    Buckets []*Bucket
}

type Bucket struct {
    Key   string
    Value int
    Next  *Bucket
}

func NewHashMap() *HashMap {
    m := &HashMap{}
    m.Buckets = make([]*Bucket, 16)
    return m
}

func (m *HashMap) Hash(key string) int {
    hash := 0
    for _, v := range key {
        hash = hash*31 + int(v)
    }
    return hash % 16
}

func (m *HashMap) Put(key string, value int) {
    index := m.Hash(key)
    bucket := m.Buckets[index]
    for bucket != nil && bucket.Key != key {
        bucket = bucket.Next
    }
    if bucket == nil {
        bucket = &Bucket{Key: key, Value: value}
        bucket.Next = m.Buckets[index]
        m.Buckets[index] = bucket
    } else {
        bucket.Value = value
    }
}

func (m *HashMap) Get(key string) (int, bool) {
    index := m.Hash(key)
    bucket := m.Buckets[index]
    for bucket != nil {
        if bucket.Key == key {
            return bucket.Value, true
        }
        bucket = bucket.Next
    }
    return 0, false
}
```

**答案解析：**        
- 哈希表通过哈希函数将键映射到哈希表中，以实现快速查找、插入和删除操作。
- 在这个示例中，我们使用数组作为哈希表的底层存储结构，每个数组元素指向一个链表，用于解决哈希冲突。
- 哈希表的性能依赖于哈希函数的质量和数组的大小。如果哈希函数分布均匀，哈希表可以达到近似常数时间（O(1)）的查找、插入和删除操作。

### 7. 算法面试题：两数之和

**题目描述：** 给定一个整数数组 `nums` 和一个目标值 `target`，找出数组中两数之和等于 `target` 的两个数，并返回它们的下标。

**输入：**          
- 整数数组 `nums`：`[2, 7, 11, 15]`
- 目标值 `target`：`9`

**输出：**          
- 下标：`[0, 1]`（因为 `nums[0] + nums[1] == 9`）

**算法思路：**          
- 使用哈希表存储数组中每个元素及其下标。
- 遍历数组，对于每个元素 `x`，计算 `target - x`，并检查哈希表中是否存在 `target - x`。

**代码示例：**

```go
func twoSum(nums []int, target int) []int {
    m := make(map[int]int)
    for i, num := range nums {
        m[num] = i
    }
    for i, num := range nums {
        complement := target - num
        if j, exists := m[complement]; exists && j != i {
            return []int{i, j}
        }
    }
    return nil
}
```

**答案解析：**          
- 通过使用哈希表，我们可以快速查找数组中是否存在与当前元素相加等于目标值的元素。
- 时间复杂度为 `O(n)`，空间复杂度为 `O(n)`。

### 8. 数据结构与算法面试题：字符串匹配（KMP 算法）

**题目描述：** 给定两个字符串 `text` 和 `pattern`，实现字符串匹配算法，找出 `text` 中所有与 `pattern` 匹配的子串。

**输入：**            
- 字符串 `text`："abcxabcdxyz"
- 字符串 `pattern`："abc"

**输出：**            
- 匹配的子串下标：`[0, 4, 10]`

**算法思路：**            
- KMP 算法通过构建一个前缀函数（部分匹配表）来优化模式匹配过程。
- 在匹配过程中，当出现不匹配时，使用前缀函数来确定下一次匹配的起点。

**代码示例：**

```go
func kmpSearch(text, pattern string) []int {
    n, m := len(text), len(pattern)
    lps := make([]int, m)
    j := 0

    buildLPS(pattern, m, lps)

    i := 0
    result := []int{}
    for i < n {
        if pattern[j] == text[i] {
            i++
            j++
        }
        if j == m {
            result = append(result, i-j)
            j = lps[j-1]
        } else if i < n && pattern[j] != text[i] {
            if j != 0 {
                j = lps[j-1]
            } else {
                i++
            }
        }
    }
    return result
}

func buildLPS(pattern string, m int, lps []int) {
    length := 0
    lps[0] = 0
    i := 1
    for i < m {
        if pattern[i] == pattern[length] {
            length++
            lps[i] = length
            i++
        } else {
            if length != 0 {
                length = lps[length-1]
            } else {
                lps[i] = 0
                i++
            }
        }
    }
}
```

**答案解析：**            
- KMP 算法通过构建前缀函数来避免不必要的比较，从而提高字符串匹配的效率。
- 在匹配过程中，当出现不匹配时，使用前缀函数来确定下一次匹配的起点，避免了从模式的首个字符开始重新匹配。
- 时间复杂度为 `O(n + m)`，空间复杂度为 `O(m)`。

### 9. 算法面试题：最长递增子序列

**题目描述：** 给定一个整数数组 `nums`，找出最长递增子序列的长度。

**输入：**            
- 整数数组 `nums`：`[10, 9, 2, 5, 3, 7, 101, 18]`

**输出：**            
- 最长递增子序列长度：4（子序列为 `[2, 3, 7, 101]`）

**算法思路：**            
- 动态规划算法。使用一个数组 `dp` 记录以每个元素为结尾的最长递增子序列长度。
- 遍历数组，对于每个元素 `nums[i]`，遍历所有之前的元素 `nums[j]`（`j < i`），如果 `nums[i] > nums[j]`，则更新 `dp[i]` 为 `dp[j] + 1`。
- 最长递增子序列长度为 `dp` 数组中的最大值。

**代码示例：**

```go
func lengthOfLIS(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    dp := make([]int, len(nums))
    dp[0] = 1
    maxLen := 1
    for i := 1; i < len(nums); i++ {
        for j := 0; j < i; j++ {
            if nums[i] > nums[j] {
                dp[i] = max(dp[i], dp[j]+1)
            }
        }
        maxLen = max(maxLen, dp[i])
    }
    return maxLen
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**答案解析：**            
- 动态规划算法通过构建一个辅助数组 `dp`，记录以每个元素为结尾的最长递增子序列长度，从而求得整个数组的最大递增子序列长度。
- 时间复杂度为 `O(n^2)`，可以通过使用二分搜索优化到 `O(n log n)`。

### 10. 数据结构与算法面试题：二叉搜索树（BST）

**题目描述：** 实现 二叉搜索树（BST）的基本操作，包括插入、删除、查找和遍历。

**输入：**            
- 操作序列：`["insert", "delete", "find"]`
- 值序列：`[20, 10, 30, 15, 25]`

**输出：**            
- 操作结果：`[20, 10, 15, 25, 30]`（删除后的树）

**算法思路：**            
- 插入操作：找到合适的位置，创建新的节点并插入。
- 删除操作：找到要删除的节点，处理删除节点的情况（叶节点、单分支节点、双分支节点）。
- 查找操作：递归地查找节点，找到目标节点返回。
- 遍历操作：前序、中序、后序遍历。

**代码示例：**

```go
type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

func (t *TreeNode) Insert(val int) *TreeNode {
    if t == nil {
        return &TreeNode{Val: val}
    }
    if val < t.Val {
        t.Left = t.Left.Insert(val)
    } else if val > t.Val {
        t.Right = t.Right.Insert(val)
    }
    return t
}

func (t *TreeNode) Delete(val int) *TreeNode {
    if t == nil {
        return nil
    }
    if val < t.Val {
        t.Left = t.Left.Delete(val)
    } else if val > t.Val {
        t.Right = t.Right.Delete(val)
    } else {
        if t.Left == nil && t.Right == nil {
            t = nil
        } else if t.Left == nil {
            t = t.Right
        } else if t.Right == nil {
            t = t.Left
        } else {
            minNode := t.Right.Min()
            t.Val = minNode.Val
            t.Right = t.Right.Delete(minNode.Val)
        }
    }
    return t
}

func (t *TreeNode) Find(val int) *TreeNode {
    if t == nil {
        return nil
    }
    if val == t.Val {
        return t
    } else if val < t.Val {
        return t.Left.Find(val)
    } else {
        return t.Right.Find(val)
    }
}

func (t *TreeNode) PreOrderTraversal() {
    if t != nil {
        println(t.Val)
        t.Left.PreOrderTraversal()
        t.Right.PreOrderTraversal()
    }
}

func (t *TreeNode) InOrderTraversal() {
    if t != nil {
        t.Left.InOrderTraversal()
        println(t.Val)
        t.Right.InOrderTraversal()
    }
}

func (t *TreeNode) PostOrderTraversal() {
    if t != nil {
        t.Left.PostOrderTraversal()
        t.Right.PostOrderTraversal()
        println(t.Val)
    }
}

func (t *TreeNode) Min() *TreeNode {
    if t == nil || t.Left == nil {
        return t
    }
    return t.Left.Min()
}
```

**答案解析：**            
- 二叉搜索树是一种特殊的树结构，其中每个节点的左子树中的所有节点的值都小于该节点的值，右子树中的所有节点的值都大于该节点的值。
- 插入操作：找到合适的位置，创建新的节点并插入。
- 删除操作：找到要删除的节点，根据节点类型（叶节点、单分支节点、双分支节点）处理删除。
- 查找操作：递归地查找节点，找到目标节点返回。
- 遍历操作：前序、中序、后序遍历。

### 11. 算法面试题：快速排序

**题目描述：** 实现快速排序算法，对整数数组进行排序。

**输入：**              
- 整数数组：`[3, 1, 4, 1, 5, 9]`

**输出：**              
- 排序后的数组：`[1, 1, 3, 4, 5, 9]`

**算法思路：**              
- 快速排序是一种分治排序算法，通过递归地将数组划分为较小的子数组，然后对每个子数组进行排序。
- 选择一个基准元素，将数组划分为两部分：一部分小于基准元素，另一部分大于基准元素。
- 递归地对小于和大于基准元素的子数组进行快速排序。

**代码示例：**

```go
func quicksort(arr []int, low, high int) {
    if low < high {
        pi := partition(arr, low, high)
        quicksort(arr, low, pi-1)
        quicksort(arr, pi+1, high)
    }
}

func partition(arr []int, low, high int) int {
    pivot := arr[high]
    i := low
    for j := low; j < high; j++ {
        if arr[j] < pivot {
            arr[i], arr[j] = arr[j], arr[i]
            i++
        }
    }
    arr[i], arr[high] = arr[high], arr[i]
    return i
}
```

**答案解析：**              
- 快速排序的核心思想是通过递归地将数组划分为较小的子数组，然后对每个子数组进行排序。
- 选择一个基准元素，将数组划分为两部分：一部分小于基准元素，另一部分大于基准元素。
- 递归地对小于和大于基准元素的子数组进行快速排序。
- 时间复杂度为 `O(n log n)`，但平均情况下优于其他常见的排序算法。

### 12. 算法面试题：二分查找

**题目描述：** 给定一个有序整数数组 `nums` 和一个目标值 `target`，在数组中查找 `target` 的位置。

**输入：**                  
- 有序整数数组 `nums`：`[-1, 0, 3, 5, 9, 12]`
- 目标值 `target`：`9`

**输出：**                  
- 目标值的位置：`4`（因为 `nums[4] == 9`）

**算法思路：**                  
- 使用二分查找算法，通过递归或迭代地缩小查找范围。
- 每次比较中间元素，如果中间元素等于目标值，返回位置；如果中间元素大于目标值，递归或迭代地搜索左半部分；如果中间元素小于目标值，递归或迭代地搜索右半部分。

**代码示例：**

```go
func search(nums []int, target int) int {
    left, right := 0, len(nums)-1
    for left <= right {
        mid := (left + right) / 2
        if nums[mid] == target {
            return mid
        } else if nums[mid] > target {
            right = mid - 1
        } else {
            left = mid + 1
        }
    }
    return -1
}
```

**答案解析：**                  
- 二分查找算法通过递归或迭代地缩小查找范围，可以快速地在有序数组中查找目标值。
- 时间复杂度为 `O(log n)`，是查找算法中的高效方法。

### 13. 数据结构与算法面试题：设计一个 LRU 缓存

**题目描述：** 设计一个 LRU（Least Recently Used）缓存，支持 `get` 和 `put` 操作。

**输入：**                  
- 操作序列：`["LRUCache", "put", "put", "get", "put", "get"]`
- 值序列：`[[2], [2, 1], [2, 3], [2], [4, 1], [2]`

**输出：**                  
- 操作结果：`[null, null, null, 1, null, 1]`

**算法思路：**                  
- 使用哈希表和双向链表实现 LRU 缓存。
- 哈希表用于快速查找缓存项。
- 双向链表用于记录缓存项的顺序，最近访问的项位于链表头部。

**代码示例：**

```go
type DLinkedNode struct {
    Key  int
    Val  int
    Prev *DLinkedNode
    Next *DLinkedNode
}

type LRUCache struct {
    size     int
    capacity int
    key2Node map[int]*DLinkedNode
    head     *DLinkedNode
    tail     *DLinkedNode
}

func Constructor(capacity int) LRUCache {
    lru := LRUCache{capacity: capacity, size: 0, key2Node: make(map[int]*DLinkedNode)}
    lru.head = &DLinkedNode{}
    lru.tail = &DLinkedNode{}
    lru.head.Next = lru.tail
    lru.tail.Prev = lru.head
    return lru
}

func (lru *LRUCache) Get(key int) int {
    if node, exists := lru.key2Node[key]; exists {
        lru.moveToHead(node)
        return node.Val
    }
    return -1
}

func (lru *LRUCache) Put(key int, value int) {
    if node, exists := lru.key2Node[key]; exists {
        node.Val = value
        lru.moveToHead(node)
    } else {
        newNode := &DLinkedNode{Key: key, Val: value}
        lru.key2Node[key] = newNode
        lru.addToHead(newNode)
        if lru.size > lru.capacity {
            lru.removeFromTail()
        }
        lru.size++
    }
}

func (lru *LRUCache) moveToHead(node *DLinkedNode) {
    lru.removeFromNode(node)
    lru.addToHead(node)
}

func (lru *LRUCache) removeFromNode(node *DLinkedNode) {
    node.Prev.Next = node.Next
    node.Next.Prev = node.Prev
}

func (lru *LRUCache) addToHead(node *DLinkedNode) {
    node.Next = lru.head.Next
    node.Prev = lru.head
    lru.head.Next.Prev = node
    lru.head.Next = node
}

func (lru *LRUCache) removeFromTail() {
    node := lru.tail.Prev
    lru.removeFromNode(node)
    delete(lru.key2Node, node.Key)
}
```

**答案解析：**                  
- LRU 缓存通过哈希表和双向链表实现，哈希表用于快速查找缓存项，双向链表用于维护缓存项的顺序。
- 当缓存命中时，将缓存项移动到双向链表头部。
- 当缓存未命中且缓存容量达到上限时，移除最久未使用（双向链表尾部）的缓存项。
- 时间复杂度为 `O(1)`。

### 14. 算法面试题：设计一个单调队列

**题目描述：** 设计一个单调队列，支持以下操作：  
- `addlast(int val)`：在队列尾部添加元素 `val`。
- `addfirst(int val)`：在队列头部添加元素 `val`。
- `popfirst()`：移除队列头部的元素。
- `poplast()`：移除队列尾部的元素。
- `max()`：返回队列中最大元素。

**输入：**      
- 操作序列：`["MonotonicQueue", "addlast", "addlast", "max", "addfirst", "popfirst", "max", "poplast", "max"]`
- 值序列：`[5, 1, 5, 5, 2, 2, 1, 5]`

**输出：**      
- 操作结果：`[null, null, 5, null, null, 5, 5, 5, 1]`

**算法思路：**      
- 使用两个栈实现单调队列。
- 对于 `addlast` 和 `addfirst` 操作，将元素添加到栈中。
- 对于 `popfirst` 和 `poplast` 操作，分别从栈顶和栈底弹出元素。
- 对于 `max()` 操作，返回栈顶元素，确保栈保持单调递增。

**代码示例：**

```go
type MonotonicQueue struct {
    stack []*Node
    queue []int
}

type Node struct {
    Val  int
    Next *Node
}

func Constructor() MonotonicQueue {
    q := MonotonicQueue{}
    q.stack = make([]*Node, 0)
    return q
}

func (q *MonotonicQueue) addlast(val int) {
    node := &Node{Val: val}
    node.Next = q.stack[0]
    q.stack[0] = node
    q.queue = append(q.queue, val)
}

func (q *MonotonicQueue) addfirst(val int) {
    node := &Node{Val: val}
    q.stack = append([]*Node{node}, q.stack...)
    q.queue = append([]int{val}, q.queue...)
}

func (q *MonotonicQueue) popfirst() {
    q.stack[0] = q.stack[0].Next
    q.queue = q.queue[1:]
}

func (q *MonotonicQueue) poplast() {
    last := q.stack[len(q.stack)-1]
    q.stack = q.stack[:len(q.stack)-1]
    q.queue = q.queue[:len(q.queue)-1]
}

func (q *MonotonicQueue) max() int {
    return q.queue[0]
}
```

**答案解析：**      
- 使用两个栈实现单调队列，一个栈用于维护单调递增的队列，另一个栈用于维护单调递减的队列。
- 对于 `addlast` 和 `addfirst` 操作，将元素添加到栈中，并更新队列。
- 对于 `popfirst` 和 `poplast` 操作，分别从栈顶和栈底弹出元素，并更新队列。
- 对于 `max()` 操作，返回队列头部元素。

### 15. 算法面试题：设计一个最小栈

**题目描述：** 设计一个最小栈，支持以下操作：  
- `push(int x)`：将元素 `x` 入栈。
- `pop()`：移除栈顶元素。
- `top()`：获取栈顶元素。
- `getMin()`：获取当前栈中的最小元素。

**输入：**        
- 操作序列：`["MinStack", "push", "push", "push", "getMin", "pop", "top", "getMin"]`
- 值序列：`[[], [1], [2], [3], [], [], [], []]`

**输出：**        
- 操作结果：`[null, null, null, null, 1, null, 2, 1]`

**算法思路：**        
- 使用一个辅助栈，记录当前栈中每个元素对应的最小值。
- 在 `push` 操作中，如果栈为空或新元素小于当前最小值，将新元素作为最小值入栈。
- 在 `pop` 操作中，如果弹出的是当前最小值，更新辅助栈的最小值。
- 在 `top` 操作中，返回栈顶元素。
- 在 `getMin()` 操作中，返回辅助栈的栈顶元素。

**代码示例：**

```go
type MinStack struct {
    s      []int
    minStack []int
}

func Constructor() MinStack {
    ms := MinStack{}
    ms.minStack = []int{-1<<31 - 1}
    return ms
}

func (ms *MinStack) Push(x int) {
    ms.s = append(ms.s, x)
    ms.minStack = append(ms.minStack, min(x, ms.minStack[len(ms.minStack)-1]))
}

func (ms *MinStack) Pop() {
    ms.s = ms.s[:len(ms.s)-1]
    ms.minStack = ms.minStack[:len(ms.minStack)-1]
}

func (ms *MinStack) Top() int {
    return ms.s[len(ms.s)-1]
}

func (ms *MinStack) GetMin() int {
    return ms.minStack[len(ms.minStack)-1]
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

**答案解析：**        
- 最小栈通过使用一个辅助栈来记录当前栈中每个元素对应的最小值。
- 在 `push` 操作中，如果新元素小于当前最小值，更新最小值。
- 在 `pop` 操作中，如果弹出的是当前最小值，更新辅助栈的最小值。
- 在 `top` 和 `getMin()` 操作中，直接访问栈顶和辅助栈的栈顶元素。

### 16. 算法面试题：设计一个缓存

**题目描述：** 设计一个缓存，支持以下操作：  
- `get(int key)`：如果键存在于缓存中，则返回其值（总是正数），否则返回 -1。
- `put(int key, int value)`：如果键已存在，则更新其值；否则插入键值对。

**输入：**            
- 操作序列：`["LRUCache", "get", "put", "get", "put", "get"]`
- 值序列：`[[2], [2], [2, 6], [1], [1, 5], [1]]`

**输出：**            
- 操作结果：`[null, -1, null, 6, null, 5]`

**算法思路：**            
- 使用哈希表和双向链表实现 LRU 缓存。
- 哈希表用于快速查找缓存项。
- 双向链表用于维护缓存项的顺序，最近访问的项位于链表头部。

**代码示例：**

```go
type DLinkedNode struct {
    Key  int
    Val  int
    Prev *DLinkedNode
    Next *DLinkedNode
}

type LRUCache struct {
    size     int
    capacity int
    key2Node map[int]*DLinkedNode
    head     *DLinkedNode
    tail     *DLinkedNode
}

func Constructor(capacity int) LRUCache {
    lru := LRUCache{capacity: capacity, size: 0, key2Node: make(map[int]*DLinkedNode)}
    lru.head = &DLinkedNode{}
    lru.tail = &DLinkedNode{}
    lru.head.Next = lru.tail
    lru.tail.Prev = lru.head
    return lru
}

func (lru *LRUCache) Get(key int) int {
    if node, exists := lru.key2Node[key]; exists {
        lru.moveToHead(node)
        return node.Val
    }
    return -1
}

func (lru *LRUCache) Put(key int, value int) {
    if node, exists := lru.key2Node[key]; exists {
        node.Val = value
        lru.moveToHead(node)
    } else {
        newNode := &DLinkedNode{Key: key, Val: value}
        lru.key2Node[key] = newNode
        lru.addToHead(newNode)
        if lru.size > lru.capacity {
            lru.removeFromTail()
        }
        lru.size++
    }
}

func (lru *LRUCache) moveToHead(node *DLinkedNode) {
    lru.removeFromNode(node)
    lru.addToHead(node)
}

func (lru *LRUCache) removeFromNode(node *DLinkedNode) {
    node.Prev.Next = node.Next
    node.Next.Prev = node.Prev
}

func (lru *LRUCache) addToHead(node *DLinkedNode) {
    node.Next = lru.head.Next
    node.Prev = lru.head
    lru.head.Next.Prev = node
    lru.head.Next = node
}

func (lru *LRUCache) removeFromTail() {
    node := lru.tail.Prev
    lru.removeFromNode(node)
    delete(lru.key2Node, node.Key)
}
```

**答案解析：**            
- LRU 缓存通过哈希表和双向链表实现，哈希表用于快速查找缓存项，双向链表用于维护缓存项的顺序。
- 当缓存命中时，将缓存项移动到双向链表头部。
- 当缓存未命中且缓存容量达到上限时，移除最久未使用（双向链表尾部）的缓存项。
- 时间复杂度为 `O(1)`。

### 17. 算法面试题：设计一个堆

**题目描述：** 设计一个最大堆，支持以下操作：  
- `push(int val)`：将元素 `val` 插入堆中。
- `pop()`：移除并返回堆顶元素。
- `peek()`：返回堆顶元素，但不移除它。

**输入：**      
- 操作序列：`["MaxHeap", "push", "push", "push", "peek", "pop", "peek"]`
- 值序列：`[[], [3], [2], [1], [], [], []]`

**输出：**      
- 操作结果：`[null, null, null, null, 3, null, 2]`

**算法思路：**      
- 使用数组实现堆，通过父子节点索引的关系进行操作。
- 在 `push` 操作中，将新元素添加到数组的末尾，然后进行上滤操作，使堆保持最大堆性质。
- 在 `pop` 操作中，将堆顶元素与数组末尾元素交换，然后移除末尾元素，并进行下滤操作，使堆保持最大堆性质。
- 在 `peek` 操作中，直接返回堆顶元素。

**代码示例：**

```go
type MaxHeap struct {
    arr []int
}

func Constructor() MaxHeap {
    heap := MaxHeap{}
    heap.arr = []int{-1<<31 - 1} // 存储哨兵，便于计算父子节点索引
    return heap
}

func (heap *MaxHeap) push(val int) {
    heap.arr = append(heap.arr, val)
    parent := (len(heap.arr) - 2) / 2
    for heap.arr[len(heap.arr)-1] > heap.arr[parent] {
        heap.arr[parent], heap.arr[len(heap.arr)-1] = heap.arr[len(heap.arr)-1], heap.arr[parent]
        parent = (parent - 1) / 2
    }
}

func (heap *MaxHeap) pop() int {
    val := heap.arr[1]
    heap.arr[1] = heap.arr[len(heap.arr)-1]
    heap.arr = heap.arr[:len(heap.arr)-1]
    if len(heap.arr) > 1 {
        maxIndex := 1
        left := maxIndex*2 + 1
        right := maxIndex*2 + 2
        if left < len(heap.arr) && heap.arr[left] > heap.arr[maxIndex] {
            maxIndex = left
        }
        if right < len(heap.arr) && heap.arr[right] > heap.arr[maxIndex] {
            maxIndex = right
        }
        heap.arr[maxIndex], heap.arr[len(heap.arr)-1] = heap.arr[len(heap.arr)-1], heap.arr[maxIndex]
        heap.arr = heap.arr[:len(heap.arr)-1]
    }
    return val
}

func (heap *MaxHeap) peek() int {
    return heap.arr[1]
}
```

**答案解析：**      
- 最大堆通过数组实现，使用数组长度来计算父子节点索引。
- 在 `push` 操作中，将新元素添加到数组的末尾，然后进行上滤操作，确保堆顶元素为最大值。
- 在 `pop` 操作中，将堆顶元素与数组末尾元素交换，然后移除末尾元素，并进行下滤操作，确保堆顶元素为最大值。
- 在 `peek` 操作中，直接返回堆顶元素。

### 18. 算法面试题：设计一个优先队列

**题目描述：** 设计一个优先队列，支持以下操作：  
- `enqueue(int val)`：将元素 `val` 入队。
- `dequeue()`：移除并返回优先队列中的最小元素。
- `peek()`：返回优先队列中的最小元素，但不移除它。

**输入：**          
- 操作序列：`["PriorityQueue", "enqueue", "enqueue", "dequeue", "dequeue", "dequeue"]`
- 值序列：`[[], [3], [1], [], [], []]`

**输出：**          
- 操作结果：`[null, null, null, 1, null, 3]`

**算法思路：**          
- 使用堆实现优先队列，确保最小元素总是位于堆顶。
- 在 `enqueue` 操作中，将新元素插入堆中。
- 在 `dequeue` 操作中，移除堆顶元素。
- 在 `peek` 操作中，返回堆顶元素。

**代码示例：**

```go
type PriorityQueue struct {
    heap []int
}

func Constructor() PriorityQueue {
    pq := PriorityQueue{}
    pq.heap = []int{-1<<31 - 1} // 存储哨兵，便于计算堆索引
    return pq
}

func (pq *PriorityQueue) enqueue(val int) {
    pq.heap = append(pq.heap, val)
    index := len(pq.heap) - 1
    parent := (index - 1) / 2
    for pq.heap[index] < pq.heap[parent] {
        pq.heap[parent], pq.heap[index] = pq.heap[index], pq.heap[parent]
        index = parent
        parent = (parent - 1) / 2
    }
}

func (pq *PriorityQueue) dequeue() int {
    if len(pq.heap) == 1 {
        return pq.heap[1]
    }
    val := pq.heap[1]
    pq.heap[1] = pq.heap[len(pq.heap)-1]
    pq.heap = pq.heap[:len(pq.heap)-1]
    index := 0
    left := index*2 + 1
    right := index*2 + 2
    for left < len(pq.heap) && pq.heap[left] < pq.heap[index] {
        pq.heap[left], pq.heap[index] = pq.heap[index], pq.heap[left]
        index = left
        left = index*2 + 1
    }
    for right < len(pq.heap) && pq.heap[right] < pq.heap[index] {
        pq.heap[right], pq.heap[index] = pq.heap[index], pq.heap[right]
        index = right
        right = index*2 + 2
    }
    return val
}

func (pq *PriorityQueue) peek() int {
    return pq.heap[1]
}
```

**答案解析：**          
- 使用堆实现优先队列，确保最小元素总是位于堆顶。
- 在 `enqueue` 操作中，将新元素插入堆中，然后进行上滤操作，确保堆保持最小堆性质。
- 在 `dequeue` 操作中，移除堆顶元素，然后进行下滤操作，确保堆保持最小堆性质。
- 在 `peek` 操作中，直接返回堆顶元素。

### 19. 算法面试题：设计一个最小时间间隔

**题目描述：** 设计一个函数，计算给定数组中任意两个元素的最小时间间隔。时间间隔定义为两个元素的时间戳之差。

**输入：**            
- 时间戳数组：`[9, 9, 9]`
- 查询数组：`[9, 10, 19]`

**输出：**            
- 最小时间间隔：`0`

**算法思路：**            
- 使用二分查找算法，在时间戳数组中查找每个查询元素的时间戳，然后计算它们之间的最小时间间隔。

**代码示例：**

```go
func findMinTimeInterval(scores []int, queries []int) int {
    sort.Ints(scores)
    ans := 1 << 31 - 1
    for _, query := range queries {
        left, right := 0, len(scores)
        for left < right {
            mid := (left + right) / 2
            if scores[mid] >= query {
                right = mid
            } else {
                left = mid + 1
            }
        }
        if left > 0 {
            ans = min(ans, scores[left-1]-query)
        }
    }
    return ans
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

**答案解析：**            
- 通过将时间戳数组排序，我们可以使用二分查找算法快速找到每个查询元素的时间戳。
- 计算查询元素与时间戳数组中前一个元素之间的最小时间间隔，更新最小时间间隔。

### 20. 算法面试题：设计一个线段树

**题目描述：** 设计一个线段树，支持以下操作：  
- `update(l, r, v)`：将区间 `[l, r]` 的所有元素增加 `v`。
- `query(l, r)`：返回区间 `[l, r]` 的元素总和。

**输入：**              
- 操作序列：`["SegTree", "update", "update", "query"]`
- 值序列：`[[3], [1, 2, 3], [1, 3, 1], [1, 3]]`

**输出：**              
- 操作结果：`[null, null, null, 4]`

**算法思路：**              
- 使用线段树实现，每个节点代表一个区间的和。
- 在 `update` 操作中，递归地将更新值应用到对应的节点。
- 在 `query` 操作中，递归地计算区间的和。

**代码示例：**

```go
type Node struct {
    l, r int
    sum  int
}

type SegTree struct {
    root *Node
}

func Constructor(nums []int) SegTree {
    tree := SegTree{}
    tree.root = buildTree(nums, 0, len(nums)-1)
    return tree
}

func buildTree(nums []int, l, r int) *Node {
    if l == r {
        return &Node{l: l, r: r, sum: nums[l]}
    }
    mid := (l + r) / 2
    left := buildTree(nums, l, mid)
    right := buildTree(nums, mid+1, r)
    node := &Node{l: l, r: r, sum: left.sum + right.sum}
    node.left = left
    node.right = right
    return node
}

func (tree *SegTree) update(l, r, v int) {
    tree.root = updateTree(tree.root, l, r, v, 0, len(tree.root.vals)-1)
}

func updateTree(node *Node, l, r, v int, l2, r2 int) *Node {
    if node.l == l2 && node.r == r2 {
        node.sum += (r2 - l2 + 1) * v
        return node
    }
    mid := (node.l + node.r) / 2
    if l <= mid {
        node.left = updateTree(node.left, l, r, v, l2, mid)
    }
    if r > mid {
        node.right = updateTree(node.right, l, r, v, mid+1, r2)
    }
    node.sum = node.left.sum + node.right.sum
    return node
}

func (tree *SegTree) query(l, r int) int {
    return queryTree(tree.root, l, r, 0, len(tree.root.vals)-1)
}

func queryTree(node *Node, l, r int, l2, r2 int) int {
    if node.l == l2 && node.r == r2 {
        return node.sum
    }
    mid := (node.l + node.r) / 2
    if l <= mid {
        leftSum := queryTree(node.left, l, r, l2, mid)
    }
    if r > mid {
        rightSum := queryTree(node.right, l, r, mid+1, r2)
    }
    return leftSum + rightSum
}
```

**答案解析：**              
- 线段树通过递归地将区间分割成更小的区间，并计算每个区间的和。
- 在 `update` 操作中，递归地将更新值应用到对应的节点。
- 在 `query` 操作中，递归地计算区间的和。

### 21. 算法面试题：设计一个双向链表

**题目描述：** 设计一个双向链表，支持以下操作：    
- `append(int val)`：将元素 `val` 作为尾节点添加到链表中。
- `addbefore(int val, int key)`：在具有 `key` 的节点之前添加元素 `val`。
- `addafter(int val, int key)`：在具有 `key` 的节点之后添加元素 `val`。
- `delete(int key)`：删除具有 `key` 的节点。

**输入：**              
- 操作序列：`["DoublyLinkedList", "append", "addbefore", "addafter", "delete"]`
- 值序列：`[[], [1], [2, 3], [2, 4], [2]]`

**输出：**              
- 操作结果：`[null, null, null, null, null]`

**算法思路：**              
- 使用一个结构体表示节点，包含前驱和后继节点指针。
- 在 `append` 操作中，创建新节点并将其作为尾节点添加到链表中。
- 在 `addbefore` 和 `addafter` 操作中，创建新节点并将其插入到指定节点之前或之后。
- 在 `delete` 操作中，删除指定节点，并更新前驱和后继节点的指针。

**代码示例：**

```go
type ListNode struct {
    Val  int
    Prev *ListNode
    Next *ListNode
}

type DoublyLinkedList struct {
    Head *ListNode
    Tail *ListNode
}

func Constructor() DoublyLinkedList {
    dll := DoublyLinkedList{}
    dll.Head = &ListNode{Val: -1}
    dll.Tail = &ListNode{Val: -1}
    dll.Head.Next = dll.Tail
    dll.Tail.Prev = dll.Head
    return dll
}

func (dll *DoublyLinkedList) Append(val int) {
    node := &ListNode{Val: val}
    node.Prev = dll.Tail.Prev
    dll.Tail.Prev.Next = node
    node.Next = dll.Tail
    dll.Tail.Prev = node
}

func (dll *DoublyLinkedList) AddBefore(val, key int) {
    node := &ListNode{Val: val}
    curr := dll.Search(key)
    node.Prev = curr.Prev
    node.Next = curr
    curr.Prev.Next = node
    curr.Prev = node
}

func (dll *DoublyLinkedList) AddAfter(val, key int) {
    node := &ListNode{Val: val}
    curr := dll.Search(key)
    node.Prev = curr
    node.Next = curr.Next
    curr.Next.Prev = node
    curr.Next = node
}

func (dll *DoublyLinkedList) Delete(key int) {
    curr := dll.Search(key)
    curr.Prev.Next = curr.Next
    curr.Next.Prev = curr.Prev
}

func (dll *DoublyLinkedList) Search(key int) *ListNode {
    curr := dll.Head.Next
    for curr != dll.Tail {
        if curr.Val == key {
            return curr
        }
        curr = curr.Next
    }
    return nil
}
```

**答案解析：**              
- 双向链表通过前驱和后继节点指针实现，允许在任意位置添加或删除节点。
- 在 `append` 操作中，将新节点作为尾节点添加到链表中。
- 在 `addbefore` 和 `addafter` 操作中，创建新节点并插入到指定节点之前或之后。
- 在 `delete` 操作中，删除指定节点，并更新前驱和后继节点的指针。

### 22. 算法面试题：设计一个栈

**题目描述：** 设计一个栈，支持以下操作：    
- `push(int val)`：将元素 `val` 入栈。
- `pop()`：移除栈顶元素。
- `top()`：获取栈顶元素。
- `empty()`：判断栈是否为空。

**输入：**                  
- 操作序列：`["Stack", "push", "push", "pop", "pop", "top", "empty"]`
- 值序列：`[[], [1], [2], [], [], [], []]`

**输出：**                  
- 操作结果：`[null, null, null, null, null, 1, false]`

**算法思路：**                  
- 使用数组或链表实现栈，通过指针或索引管理栈顶。
- 在 `push` 操作中，将新元素添加到栈顶。
- 在 `pop` 操作中，移除栈顶元素。
- 在 `top` 操作中，返回栈顶元素。
- 在 `empty` 操作中，判断栈是否为空。

**代码示例：**

```go
type Stack struct {
    stack []int
}

func Constructor() Stack {
    return Stack{}
}

func (s *Stack) Push(val int) {
    s.stack = append(s.stack, val)
}

func (s *Stack) Pop() {
    if len(s.stack) == 0 {
        return
    }
    s.stack = s.stack[:len(s.stack)-1]
}

func (s *Stack) Top() int {
    if len(s.stack) == 0 {
        return -1
    }
    return s.stack[len(s.stack)-1]
}

func (s *Stack) Empty() bool {
    return len(s.stack) == 0
}
```

**答案解析：**                  
- 使用数组实现栈，通过数组长度管理栈顶。
- 在 `push` 操作中，将新元素添加到数组末尾。
- 在 `pop` 操作中，移除数组末尾元素。
- 在 `top` 操作中，返回数组末尾元素。
- 在 `empty` 操作中，判断数组长度是否为 0。

### 23. 算法面试题：设计一个队列

**题目描述：** 设计一个队列，支持以下操作：    
- `enqueue(int val)`：将元素 `val` 入队。
- `dequeue()`：移除并返回队首元素。
- `front()`：返回队首元素，但不移除它。

**输入：**                  
- 操作序列：`["Queue", "enqueue", "enqueue", "dequeue", "dequeue", "front"]`
- 值序列：`[[], [1], [2], [], [], []]`

**输出：**                  
- 操作结果：`[null, null, null, 1, null, 2]`

**算法思路：**                  
- 使用两个栈实现队列，一个用于入队，另一个用于出队。
- 在 `enqueue` 操作中，将元素入队栈。
- 在 `dequeue` 操作中，如果出队栈为空，将入队栈的元素依次弹出并入队到出队栈。
- 在 `front` 操作中，返回出队栈的栈顶元素。

**代码示例：**

```go
type MyQueue struct {
    inStack []int
    outStack []int
}

func Constructor() MyQueue {
    return MyQueue{}
}

func (q *MyQueue) Enqueue(val int) {
    q.inStack = append(q.inStack, val)
}

func (q *MyQueue) Dequeue() int {
    if len(q.outStack) == 0 {
        for len(q.inStack) > 0 {
            q.outStack = append(q.outStack, q.inStack[len(q.inStack)-1])
            q.inStack = q.inStack[:len(q.inStack)-1]
        }
    }
    if len(q.outStack) == 0 {
        return -1
    }
    return q.outStack[len(q.outStack)-1]
}

func (q *MyQueue) Front() int {
    if len(q.outStack) == 0 {
        for len(q.inStack) > 0 {
            q.outStack = append(q.outStack, q.inStack[len(q.inStack)-1])
            q.inStack = q.inStack[:len(q.inStack)-1]
        }
    }
    if len(q.outStack) == 0 {
        return -1
    }
    return q.outStack[len(q.outStack)-1]
}
```

**答案解析：**                  
- 使用两个栈实现队列，入队栈用于添加元素，出队栈用于移除元素。
- 在 `enqueue` 操作中，将元素入队到入队栈。
- 在 `dequeue` 和 `front` 操作中，如果出队栈为空，将入队栈的元素依次弹出并入队到出队栈。

### 24. 算法面试题：设计一个单链表

**题目描述：** 设计一个单链表，支持以下操作：      
- `append(int val)`：将元素 `val` 作为尾节点添加到链表中。
- `addbefore(int val, int key)`：在具有 `key` 的节点之前添加元素 `val`。
- `addafter(int val, int key)`：在具有 `key` 的节点之后添加元素 `val`。
- `delete(int key)`：删除具有 `key` 的节点。

**输入：**                  
- 操作序列：`["SinglyLinkedList", "append", "addbefore", "addafter", "delete"]`
- 值序列：`[[], [1], [2, 3], [2, 4], [2]]`

**输出：**                  
- 操作结果：`[null, null, null, null, null]`

**算法思路：**                  
- 使用结构体表示节点，包含指向下一个节点的指针。
- 在 `append` 操作中，创建新节点并将其作为尾节点添加到链表中。
- 在 `addbefore` 和 `addafter` 操作中，创建新节点并插入到指定节点之前或之后。
- 在 `delete` 操作中，删除指定节点，并更新前一个节点的指针。

**代码示例：**

```go
type ListNode struct {
    Val  int
    Next *ListNode
}

type SinglyLinkedList struct {
    Head *ListNode
}

func Constructor() SinglyLinkedList {
    return SinglyLinkedList{}
}

func (sll *SinglyLinkedList) Append(val int) {
    node := &ListNode{Val: val}
    if sll.Head == nil {
        sll.Head = node
    } else {
        curr := sll.Head
        for curr.Next != nil {
            curr = curr.Next
        }
        curr.Next = node
    }
}

func (sll *SinglyLinkedList) AddBefore(val, key int) {
    node := &ListNode{Val: val}
    curr := sll.Search(key)
    node.Next = curr
    if curr == sll.Head {
        sll.Head = node
    } else {
        curr.Prev.Next = node
    }
    node.Prev = curr.Prev
}

func (sll *SinglyLinkedList) AddAfter(val, key int) {
    node := &ListNode{Val: val}
    curr := sll.Search(key)
    node.Next = curr.Next
    curr.Next = node
    node.Prev = curr
}

func (sll *SinglyLinkedList) Delete(key int) {
    curr := sll.Search(key)
    if curr != nil {
        if curr == sll.Head {
            sll.Head = curr.Next
        } else {
            curr.Prev.Next = curr.Next
        }
        if curr.Next != nil {
            curr.Next.Prev = curr.Prev
        }
    }
}

func (sll *SinglyLinkedList) Search(key int) *ListNode {
    curr := sll.Head
    for curr != nil && curr.Val != key {
        curr = curr.Next
    }
    return curr
}
```

**答案解析：**                  
- 单链表通过节点指针实现，每个节点包含数据和一个指向下一个节点的指针。
- 在 `append` 操作中，创建新节点并将其作为尾节点添加到链表中。
- 在 `addbefore` 和 `addafter` 操作中，创建新节点并插入到指定节点之前或之后。
- 在 `delete` 操作中，删除指定节点，并更新前一个节点的指针。

### 25. 算法面试题：设计一个有限长度的队列

**题目描述：** 设计一个有限长度的队列，支持以下操作：    
- `enqueue(int val)`：将元素 `val` 入队。
- `dequeue()`：移除并返回队首元素。
- `size()`：返回队列的长度。

**输入：**                  
- 操作序列：`["LimitedQueue", "enqueue", "enqueue", "enqueue", "dequeue", "dequeue", "size"]`
- 值序列：`[[], [1], [2], [3], [], [], []]`

**输出：**                  
- 操作结果：`[null, null, null, null, 2, 1, 2]`

**算法思路：**                  
- 使用数组实现有限长度队列，通过数组长度和指针管理队列。
- 在 `enqueue` 操作中，将新元素添加到队尾，如果队列已满，覆盖队首元素。
- 在 `dequeue` 操作中，移除队首元素。
- 在 `size` 操作中，返回队列长度。

**代码示例：**

```go
type LimitedQueue struct {
    arr   []int
    size  int
    front int
}

func Constructor(k int) LimitedQueue {
    return LimitedQueue{arr: make([]int, k), size: 0, front: 0}
}

func (lq *LimitedQueue) Enqueue(val int) {
    if lq.size < len(lq.arr) {
        lq.arr[(lq.front+lq.size)%len(lq.arr)] = val
        lq.size++
    } else {
        lq.arr[lq.front] = val
        lq.front = (lq.front + 1) % len(lq.arr)
    }
}

func (lq *LimitedQueue) Dequeue() int {
    if lq.size == 0 {
        return -1
    }
    val := lq.arr[lq.front]
    lq.arr[lq.front] = 0
    lq.front = (lq.front + 1) % len(lq.arr)
    lq.size--
    return val
}

func (lq *LimitedQueue) Size() int {
    return lq.size
}
```

**答案解析：**                  
- 使用数组实现有限长度队列，通过数组长度和指针管理队列。
- 在 `enqueue` 操作中，如果队列未满，将新元素添加到队尾；如果队列已满，覆盖队首元素。
- 在 `dequeue` 操作中，移除队首元素。
- 在 `size` 操作中，返回队列长度。

### 26. 算法面试题：设计一个排序链表

**题目描述：** 设计一个排序链表，支持以下操作：    
- `insert(int val)`：将元素 `val` 插入到链表中，保持链表有序。
- `delete(int val)`：删除具有 `val` 的节点。

**输入：**                  
- 操作序列：`["SortedLinkedList", "insert", "insert", "delete", "delete"]`
- 值序列：`[[], [3], [2], [], [4]]`

**输出：**                  
- 操作结果：`[null, null, null, null, null]`

**算法思路：**                  
- 使用链表实现排序链表，在插入和删除操作中保持链表有序。
- 在 `insert` 操作中，找到合适的位置插入新节点。
- 在 `delete` 操作中，找到目标节点并删除。

**代码示例：**

```go
type ListNode struct {
    Val  int
    Next *ListNode
}

type SortedLinkedList struct {
    Head *ListNode
}

func Constructor() SortedLinkedList {
    return SortedLinkedList{}
}

func (sll *SortedLinkedList) Insert(val int) {
    node := &ListNode{Val: val}
    if sll.Head == nil || sll.Head.Val >= val {
        node.Next = sll.Head
        sll.Head = node
    } else {
        curr := sll.Head
        for curr.Next != nil && curr.Next.Val < val {
            curr = curr.Next
        }
        node.Next = curr.Next
        curr.Next = node
    }
}

func (sll *SortedLinkedList) Delete(val int) {
    if sll.Head == nil {
        return
    }
    if sll.Head.Val == val {
        sll.Head = sll.Head.Next
        return
    }
    curr := sll.Head
    for curr.Next != nil && curr.Next.Val != val {
        curr = curr.Next
    }
    if curr.Next != nil {
        curr.Next = curr.Next.Next
    }
}
```

**答案解析：**                  
- 使用链表实现排序链表，在插入和删除操作中保持链表有序。
- 在 `insert` 操作中，找到合适的位置插入新节点。
- 在 `delete` 操作中，找到目标节点并删除。

### 27. 算法面试题：设计一个有序链表

**题目描述：** 设计一个有序链表，支持以下操作：    
- `add(int val)`：将元素 `val` 添加到链表中，保持链表有序。
- `find(int target)`：查找具有 `target` 的节点，返回其索引。

**输入：**                  
- 操作序列：`["OrderedLinkedList", "add", "add", "add", "find", "find"]`
- 值序列：`[[], [3], [1], [], [1], [3]]`

**输出：**                  
- 操作结果：`[null, null, null, null, 2, 0]`

**算法思路：**                  
- 使用链表实现有序链表，在添加操作中保持链表有序。
- 在 `add` 操作中，找到合适的位置插入新节点。
- 在 `find` 操作中，遍历链表查找目标节点，返回其索引。

**代码示例：**

```go
type ListNode struct {
    Val  int
    Next *ListNode
}

type OrderedLinkedList struct {
    Head *ListNode
}

func Constructor() *OrderedLinkedList {
    return &OrderedLinkedList{Head: nil}
}

func (oll *OrderedLinkedList) Add(val int) {
    node := &ListNode{Val: val}
    if oll.Head == nil || oll.Head.Val > val {
        node.Next = oll.Head
        oll.Head = node
    } else {
        curr := oll.Head
        for curr.Next != nil && curr.Next.Val <= val {
            curr = curr.Next
        }
        node.Next = curr.Next
        curr.Next = node
    }
}

func (oll *OrderedLinkedList) Find(target int) int {
    curr := oll.Head
    index := 0
    for curr != nil && curr.Val < target {
        curr = curr.Next
        index++
    }
    if curr != nil && curr.Val == target {
        return index
    }
    return -1
}
```

**答案解析：**                  
- 使用链表实现有序链表，在添加操作中保持链表有序。
- 在 `add` 操作中，找到合适的位置插入新节点。
- 在 `find` 操作中，遍历链表查找目标节点，返回其索引。

### 28. 算法面试题：设计一个双向循环链表

**题目描述：** 设计一个双向循环链表，支持以下操作：    
- `append(int val)`：将元素 `val` 作为尾节点添加到链表中。
- `addbefore(int val, int key)`：在具有 `key` 的节点之前添加元素 `val`。
- `addafter(int val, int key)`：在具有 `key` 的节点之后添加元素 `val`。
- `delete(int key)`：删除具有 `key` 的节点。

**输入：**                  
- 操作序列：`["DoublyCircularLinkedList", "append", "addbefore", "addafter", "delete"]`
- 值序列：`[[], [1], [2, 3], [2, 4], [2]]`

**输出：**                  
- 操作结果：`[null, null, null, null, null]`

**算法思路：**                  
- 使用结构体表示节点，包含前驱和后继节点指针，实现双向循环链表。
- 在 `append` 操作中，创建新节点并将其作为尾节点添加到链表中。
- 在 `addbefore` 和 `addafter` 操作中，创建新节点并插入到指定节点之前或之后。
- 在 `delete` 操作中，删除指定节点，并更新前驱和后继节点的指针。

**代码示例：**

```go
type DNode struct {
    Val  int
    Prev *DNode
    Next *DNode
}

type DoublyCircularLinkedList struct {
    Head *DNode
    Tail *DNode
}

func Constructor() *DoublyCircularLinkedList {
    dll := &DoublyCircularLinkedList{}
    dll.Head = &DNode{Val: -1}
    dll.Tail = &DNode{Val: -1}
    dll.Head.Next = dll.Tail
    dll.Tail.Prev = dll.Head
    return dll
}

func (dll *DoublyCircularLinkedList) Append(val int) {
    node := &DNode{Val: val}
    node.Prev = dll.Tail.Prev
    node.Next = dll.Tail
    dll.Tail.Prev.Next = node
    dll.Tail.Prev = node
}

func (dll *DoublyCircularLinkedList) AddBefore(val, key int) {
    node := &DNode{Val: val}
    curr := dll.Search(key)
    node.Prev = curr.Prev
    node.Next = curr
    curr.Prev.Next = node
    node.Prev = curr.Prev
}

func (dll *DoublyCircularLinkedList) AddAfter(val, key int) {
    node := &DNode{Val: val}
    curr := dll.Search(key)
    node.Prev = curr
    node.Next = curr.Next
    curr.Next.Prev = node
    curr.Next = node
}

func (dll *DoublyCircularLinkedList) Delete(key int) {
    curr := dll.Search(key)
    if curr != nil {
        curr.Prev.Next = curr.Next
        curr.Next.Prev = curr.Prev
    }
}

func (dll *DoublyCircularLinkedList) Search(key int) *DNode {
    curr := dll.Head
    for curr != dll.Tail && curr.Val != key {
        curr = curr.Next
    }
    return curr
}
```

**答案解析：**                  
- 使用结构体表示节点，包含前驱和后继节点指针，实现双向循环链表。
- 在 `append` 操作中，创建新节点并将其作为尾节点添加到链表中。
- 在 `addbefore` 和 `addafter` 操作中，创建新节点并插入到指定节点之前或之后。
- 在 `delete` 操作中，删除指定节点，并更新前驱和后继节点的指针。

### 29. 算法面试题：设计一个优先级队列

**题目描述：** 设计一个优先级队列，支持以下操作：    
- `enqueue(int val)`：将元素 `val` 入队，按照优先级排序。
- `dequeue()`：移除并返回优先级最高的元素。

**输入：**                  
- 操作序列：`["PriorityQueue", "enqueue", "enqueue", "dequeue", "dequeue"]`
- 值序列：`[[], [2], [3], [], []]`

**输出：**                  
- 操作结果：`[null, null, null, 2, 3]`

**算法思路：**                  
- 使用堆实现优先级队列，确保最高优先级元素总是位于堆顶。
- 在 `enqueue` 操作中，将新元素插入堆中。
- 在 `dequeue` 操作中，移除堆顶元素。

**代码示例：**

```go
type PriorityQueue struct {
    heap []int
}

func Constructor() *PriorityQueue {
    pq := &PriorityQueue{}
    pq.heap = []int{-1 << 31 - 1} // 存储哨兵，便于计算堆索引
    return pq
}

func (pq *PriorityQueue) Enqueue(val int) {
    pq.heap = append(pq.heap, val)
    index := len(pq.heap) - 1
    parent := (index - 1) / 2
    for index > 0 && pq.heap[index] < pq.heap[parent] {
        pq.heap[parent], pq.heap[index] = pq.heap[index], pq.heap[parent]
        index = parent
        parent = (parent - 1) / 2
    }
}

func (pq *PriorityQueue) Dequeue() int {
    val := pq.heap[1]
    pq.heap[1] = pq.heap[len(pq.heap)-1]
    pq.heap = pq.heap[:len(pq.heap)-1]
    if len(pq.heap) > 1 {
        maxIndex := 1
        left := maxIndex * 2 + 1
        right := maxIndex * 2 + 2
        for left < len(pq.heap) && pq.heap[left] > pq.heap[maxIndex] {
            pq.heap[left], pq.heap[maxIndex] = pq.heap[maxIndex], pq.heap[left]
            maxIndex = left
            left = maxIndex * 2 + 1
        }
        for right < len(pq.heap) && pq.heap[right] > pq.heap[maxIndex] {
            pq.heap[right], pq.heap[maxIndex] = pq.heap[maxIndex], pq.heap[right]
            maxIndex = right
            right = maxIndex * 2 + 2
        }
    }
    return val
}
```

**答案解析：**                  
- 使用堆实现优先级队列，确保最高优先级元素总是位于堆顶。
- 在 `enqueue` 操作中，将新元素插入堆中，然后进行上滤操作，确保堆保持最大堆性质。
- 在 `dequeue` 操作中，移除堆顶元素，然后进行下滤操作，确保堆保持最大堆性质。

### 30. 算法面试题：设计一个事件队列

**题目描述：** 设计一个事件队列，支持以下操作：      
- `enqueue(int timestamp, int val)`：将事件 `(timestamp, val)` 入队，按照时间戳排序。
- `dequeue()`：移除并返回时间戳最小的元素。

**输入：**                  
- 操作序列：`["EventQueue", "enqueue", "enqueue", "enqueue", "dequeue", "dequeue"]`
- 值序列：`[[], [1, 1], [2, 2], [3, 3], [], []]`

**输出：**                  
- 操作结果：`[null, null, null, null, 1, 2]`

**算法思路：**                  
- 使用堆实现事件队列，确保时间戳最小的事件总是位于堆顶。
- 在 `enqueue` 操作中，将新事件插入堆中。
- 在 `dequeue` 操作中，移除堆顶事件。

**代码示例：**

```go
type Event struct {
    Timestamp int
    Val       int
}

type PriorityQueue struct {
    heap []*Event
}

func Constructor() *PriorityQueue {
    pq := &PriorityQueue{}
    pq.heap = []*Event{{Timestamp: -1 << 31 - 1}} // 存储哨兵，便于计算堆索引
    return pq
}

func (pq *PriorityQueue) Enqueue(timestamp, val int) {
    event := &Event{Timestamp: timestamp, Val: val}
    pq.heap = append(pq.heap, event)
    index := len(pq.heap) - 1
    parent := (index - 1) / 2
    for index > 0 && pq.heap[index].Timestamp < pq.heap[parent].Timestamp {
        pq.heap[parent], pq.heap[index] = pq.heap[index], pq.heap[parent]
        index = parent
        parent = (parent - 1) / 2
    }
}

func (pq *PriorityQueue) Dequeue() *Event {
    val := pq.heap[1]
    pq.heap[1] = pq.heap[len(pq.heap)-1]
    pq.heap = pq.heap[:len(pq.heap)-1]
    if len(pq.heap) > 1 {
        minIndex := 1
        left := minIndex * 2 + 1
        right := minIndex * 2 + 2
        for left < len(pq.heap) && pq.heap[left].Timestamp < pq.heap[minIndex].Timestamp {
            pq.heap[left], pq.heap[minIndex] = pq.heap[minIndex], pq.heap[left]
            minIndex = left
            left = minIndex * 2 + 1
        }
        for right < len(pq.heap) && pq.heap[right].Timestamp < pq.heap[minIndex].Timestamp {
            pq.heap[right], pq.heap[minIndex] = pq.heap[minIndex], pq.heap[right]
            minIndex = right
            right = minIndex * 2 + 2
        }
    }
    return val
}
```

**答案解析：**                  
- 使用堆实现事件队列，确保时间戳最小的事件总是位于堆顶。
- 在 `enqueue` 操作中，将新事件插入堆中，然后进行上滤操作，确保堆保持最小堆性质。
- 在 `dequeue` 操作中，移除堆顶事件，然后进行下滤操作，确保堆保持最小堆性质。

