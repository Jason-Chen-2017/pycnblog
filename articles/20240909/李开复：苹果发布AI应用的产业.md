                 

## 国内头部一线大厂高频面试题库及解答

### 1. 数据结构与算法

#### 1.1 单链表反转

**题目：** 实现一个函数，反转单链表。

**答案：**

```go
func reverseList(head *ListNode) *ListNode {
    var prev *ListNode = nil
    cur := head
    for cur != nil {
        nextTemp := cur.Next
        cur.Next = prev
        prev = cur
        cur = nextTemp
    }
    return prev
}
```

**解析：** 使用头插法，将原链表反转。

### 2. 并发编程

#### 2.1 利用 WaitGroup 实现并发任务控制

**题目：** 使用 `sync.WaitGroup` 实现 10 个 goroutine 同时执行任务，并在主线程等待所有任务完成。

**答案：**

```go
var wg sync.WaitGroup
wg.Add(10)  // 预期会启动 10 个 goroutine

for i := 0; i < 10; i++ {
    go func() {
        // 执行任务
        time.Sleep(time.Millisecond * 100)
        fmt.Println("goroutine", i, "完成")
        wg.Done()
    }()
}

wg.Wait()  // 等待所有 goroutine 完成
fmt.Println("所有任务完成")
```

**解析：** `wg.Add(10)` 设置等待的 goroutine 数量。每个 goroutine 执行任务后调用 `wg.Done()`，主线程通过 `wg.Wait()` 等待所有任务完成。

### 3. 设计模式

#### 3.1 策略模式实现不同排序算法

**题目：** 使用策略模式实现冒泡排序、选择排序和插入排序。

**答案：**

```go
type Sorter interface {
    Sort(arr []int)
}

type BubbleSort struct{}
func (bs BubbleSort) Sort(arr []int) {
    n := len(arr)
    for i := 0; i < n; i++ {
        for j := 0; j < n-i-1; j++ {
            if arr[j] > arr[j+1] {
                arr[j], arr[j+1] = arr[j+1], arr[j]
            }
        }
    }
}

type SelectionSort struct{}
func (ss SelectionSort) Sort(arr []int) {
    n := len(arr)
    for i := 0; i < n; i++ {
        minIndex := i
        for j := i+1; j < n; j++ {
            if arr[j] < arr[minIndex] {
                minIndex = j
            }
        }
        arr[i], arr[minIndex] = arr[minIndex], arr[i]
    }
}

type InsertionSort struct{}
func (is InsertionSort) Sort(arr []int) {
    n := len(arr)
    for i := 1; i < n; i++ {
        key := arr[i]
        j := i - 1
        for j >= 0 && arr[j] > key {
            arr[j+1] = arr[j]
            j--
        }
        arr[j+1] = key
    }
}

func main() {
    arr := []int{5, 2, 9, 1, 5, 6}
    sorter := BubbleSort{}
    sorter.Sort(arr)
    fmt.Println("BubbleSort:", arr)

    sorter = SelectionSort{}
    sorter.Sort(arr)
    fmt.Println("SelectionSort:", arr)

    sorter = InsertionSort{}
    sorter.Sort(arr)
    fmt.Println("InsertionSort:", arr)
}
```

**解析：** 定义一个 `Sorter` 接口，实现不同排序算法的结构体。主函数中使用不同的排序算法对象来对数组进行排序。

### 4. 网络编程

#### 4.1 TCP 网络通信

**题目：** 使用 Go 语言实现一个简单的 TCP 服务器和客户端。

**答案：**

**服务器端：**

```go
package main

import (
    "fmt"
    "net"
)

func main() {
    ln, err := net.Listen("tcp", ":8080")
    if err != nil {
        panic(err)
    }
    defer ln.Close()

    fmt.Println("Server is listening on port 8080...")
    for {
        conn, err := ln.Accept()
        if err != nil {
            panic(err)
        }
        go handleRequest(conn)
    }
}

func handleRequest(conn net.Conn) {
    buffer := make([]byte, 1024)
    n, err := conn.Read(buffer)
    if err != nil {
        panic(err)
    }
    msg := string(buffer[:n])
    fmt.Println("Received message:", msg)

    _, err = conn.Write([]byte("Hello from server!"))
    if err != nil {
        panic(err)
    }
    conn.Close()
}
```

**客户端：**

```go
package main

import (
    "fmt"
    "net"
)

func main() {
    conn, err := net.Dial("tcp", "localhost:8080")
    if err != nil {
        panic(err)
    }
    defer conn.Close()

    msg := "Hello from client!"
    _, err = conn.Write([]byte(msg))
    if err != nil {
        panic(err)
    }
    fmt.Println("Message sent:", msg)

    buffer := make([]byte, 1024)
    n, err := conn.Read(buffer)
    if err != nil {
        panic(err)
    }
    response := string(buffer[:n])
    fmt.Println("Response from server:", response)
}
```

**解析：** 服务器端在端口 8080 监听 TCP 连接，客户端连接服务器并发送消息，服务器端接收消息并回送响应。

### 5. 测试

#### 5.1 单元测试

**题目：** 使用 Go 语言实现一个简单的函数，并编写单元测试。

**答案：**

**main.go：**

```go
package main

func sum(a, b int) int {
    return a + b
}
```

**sum_test.go：**

```go
package main

import "testing"

func TestSum(t *testing.T) {
    tests := []struct {
        a int
        b int
        want int
    }{
        {1, 2, 3},
        {5, 6, 11},
        {-1, -2, -3},
    }

    for _, tt := range tests {
        t.Run(fmt.Sprintf("%d + %d", tt.a, tt.b), func(t *testing.T) {
            got := sum(tt.a, tt.b)
            if got != tt.want {
                t.Errorf("sum(%d, %d) = %d; want %d", tt.a, tt.b, got, tt.want)
            }
        })
    }
}
```

**解析：** 使用 `testing` 包编写单元测试，执行 `go test` 命令运行测试。

### 6. 其他

#### 6.1 JSON 解析

**题目：** 使用 Go 语言实现一个 JSON 解析器，解析 JSON 字符串并获取数据。

**答案：**

```go
package main

import (
    "encoding/json"
    "fmt"
)

type Person struct {
    Name    string `json:"name"`
    Age     int    `json:"age"`
    Jobs    []string `json:"jobs"`
}

func main() {
    jsonStr := `{"name": "John", "age": 30, "jobs": ["developer", "teacher"]}`
    var p Person
    err := json.Unmarshal([]byte(jsonStr), &p)
    if err != nil {
        panic(err)
    }
    fmt.Printf("%+v\n", p)
}
```

**解析：** 使用 `encoding/json` 包的 `Unmarshal` 函数解析 JSON 字符串，并获取结构体数据。

### 7. 面试题解析

#### 7.1 阿里巴巴面试题：数组中出现次数超过一半的数字

**题目：** 找出数组中次数超过一半的数字，时间复杂度为 O(n)，空间复杂度为 O(1)。

**答案：**

```go
func majorityElement(nums []int) int {
    candidate := nums[0]
    count := 1
    for i := 1; i < len(nums); i++ {
        if count == 0 {
            candidate = nums[i]
            count = 1
        } else if nums[i] == candidate {
            count++
        } else {
            count--
        }
    }
    return candidate
}
```

**解析：** 使用投票算法，时间复杂度为 O(n)，空间复杂度为 O(1)。

#### 7.2 腾讯面试题：最长公共子序列

**题目：** 给定两个字符串，求它们的最长公共子序列。

**答案：**

```go
func longestCommonSubsequence(s1 string, s2 string) int {
    dp := make([][]int, len(s1)+1)
    for i := range dp {
        dp[i] = make([]int, len(s2)+1)
    }
    for i := 1; i <= len(s1); i++ {
        for j := 1; j <= len(s2); j++ {
            if s1[i-1] == s2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }
    return dp[len(s1)][len(s2)]
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**解析：** 使用动态规划求解最长公共子序列，时间复杂度为 O(m*n)，空间复杂度为 O(m*n)。

### 8. 算法编程题库

#### 8.1 字符串匹配算法

**题目：** 实现一个字符串匹配算法，给定一个文本字符串和一个模式字符串，找出文本字符串中模式字符串的所有出现位置。

**答案：**

```go
func search(s, p string) []int {
    n, m := len(s), len(p)
    if m == 0 {
        return []int{0}
    }
    next := make([]int, m)
    j := -1
    for i := 1; i < m; i++ {
        for j >= 0 && p[i] != p[j+1] {
            j = next[j]
        }
        if p[i] == p[j+1] {
            j++
        }
        next[i] = j
    }
    i, j = 0, 0
    ans := []int{}
    for i < n {
        for j >= 0 && s[i] != p[j] {
            j = next[j]
        }
        if s[i] == p[j] {
            i++
            j++
        }
        if j == m {
            ans = append(ans, i-m)
            j = next[j]
        }
    }
    return ans
}
```

**解析：** 使用 KMP 算法进行字符串匹配，时间复杂度为 O(n+m)。

#### 8.2 二分查找

**题目：** 给定一个排序后的数组，找到目标值的位置。

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

**解析：** 使用二分查找算法，时间复杂度为 O(logn)。

#### 8.3 动态规划

**题目：** 给定一个数组，找出最长递增子序列的长度。

**答案：**

```go
func lengthOfLIS(nums []int) int {
    n := len(nums)
    dp := make([]int, n)
    for i := 0; i < n; i++ {
        dp[i] = 1
        for j := 0; j < i; j++ {
            if nums[i] > nums[j] {
                dp[i] = max(dp[i], dp[j]+1)
            }
        }
    }
    ans := 0
    for i := 0; i < n; i++ {
        ans = max(ans, dp[i])
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

**解析：** 使用动态规划求解最长递增子序列的长度，时间复杂度为 O(n^2)。

