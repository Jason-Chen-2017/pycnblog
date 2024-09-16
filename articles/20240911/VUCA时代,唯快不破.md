                 

### 1. 快速排序算法

**题目：** 实现快速排序算法，并给出其时间复杂度和空间复杂度。

**答案：**

```go
package main

import (
	"fmt"
)

func quickSort(arr []int) []int {
	if len(arr) <= 1 {
		return arr
	}

	left := make([]int, 0)
	right := make([]int, 0)

	pivot := arr[0]
	for _, v := range arr[1:] {
		if v < pivot {
			left = append(left, v)
		} else {
			right = append(right, v)
		}
	}

	return append(quickSort(left), append([]int{pivot}, quickSort(right)...)...)
}

func main() {
	arr := []int{5, 2, 9, 1, 5, 6}
	fmt.Println("原数组：", arr)
	fmt.Println("排序后：", quickSort(arr))
}
```

**解析：**

快速排序算法的基本思想是通过一趟排序将待排序的记录分割成独立的两部分，其中一部分记录的关键字均比另一部分的关键字小，则可分别对这两部分记录继续进行排序，以达到整个序列有序。这个过程类似递归。

时间复杂度：

* 最差情况：O(n^2)
* 平均情况：O(n log n)

空间复杂度：O(log n)

### 2. 逆波兰表达式求值

**题目：** 实现一个函数，用于计算逆波兰表达式（RPN）的值。例如，表达式 `[2, 1, +, 3, *]` 应该返回 `6`。

**答案：**

```go
package main

import (
	"fmt"
)

func evalRPN(tokens []string) int {
	stack := []int{}

	for _, token := range tokens {
		if token != "+" && token != "-" && token != "*" && token != "/" {
			stack = append(stack, atoi(token))
			continue
		}

		b := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		a := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if token == "+" {
			stack = append(stack, a+b)
		} else if token == "-" {
			stack = append(stack, a-b)
		} else if token == "*" {
			stack = append(stack, a*b)
		} else if token == "/" {
			stack = append(stack, a/b)
		}
	}

	return stack[0]
}

func atoi(token string) int {
	n := 0
	for _, c := range token {
		n = n*10 + int(c-'0')
	}
	return n
}

func main() {
	tokens := []string{"2", "1", "+", "3", "*"}
	fmt.Println(evalRPN(tokens))
}
```

**解析：**

逆波兰表达式是一种后缀表示方式，通过栈实现。从左到右遍历表达式，遇到数字直接入栈，遇到运算符时，弹出栈顶两个元素进行计算，并将结果入栈。

时间复杂度：O(n)

空间复杂度：O(n)

### 3. 二分查找

**题目：** 实现一个函数，用于在排序数组中查找一个给定目标值的索引。如果目标值存在返回其索引，否则返回 -1。

**答案：**

```go
package main

import (
	"fmt"
)

func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)/2
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

func main() {
	nums := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	fmt.Println(search(nums, 5))
}
```

**解析：**

二分查找的基本思想是不断将查找范围缩小一半。每次判断中间元素是否为目标值，如果是，直接返回；如果不是，将查找范围缩小到左侧或右侧。

时间复杂度：O(log n)

空间复杂度：O(1)

### 4. 合并两个有序链表

**题目：** 将两个升序链表合并为一个新的升序链表并返回。如果链表长度相等，则将 A 的节点 排在 B 的节点之前。

**答案：**

```go
package main

import (
	"fmt"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	if list1 == nil {
		return list2
	}
	if list2 == nil {
		return list1
	}

	if list1.Val < list2.Val {
		list1.Next = mergeTwoLists(list1.Next, list2)
		return list1
	} else {
		list2.Next = mergeTwoLists(list1, list2.Next)
		return list2
	}
}

func main() {
	list1 := &ListNode{Val: 1, Next: &ListNode{Val: 2, Next: &ListNode{Val: 4}}}
	list2 := &ListNode{Val: 1, Next: &ListNode{Val: 3, Next: &ListNode{Val: 4}}}
	fmt.Println(mergeTwoLists(list1, list2))
}
```

**解析：**

递归地将两个有序链表合并为一个有序链表。每次比较两个链表的头节点，将较小值放入新链表中，并递归处理剩余部分。

时间复杂度：O(n+m)

空间复杂度：O(n+m)

### 5. 最长公共前缀

**题目：** 编写一个函数来查找字符串数组中的最长公共前缀。

**答案：**

```go
package main

import (
	"fmt"
)

func longestCommonPrefix(strs []string) string {
	if len(strs) == 0 {
		return ""
	}

	for i, v := range strs[0] {
		for _, s := range strs[1:] {
			if i >= len(s) || s[i] != v {
				return strs[0][:i]
			}
		}
	}

	return strs[0]
}

func main() {
	strs := []string{"flower", "flow", "flight"}
	fmt.Println(longestCommonPrefix(strs))
}
```

**解析：**

遍历第一个字符串，同时遍历其他字符串，直到找到不匹配的字符。返回匹配的部分。

时间复杂度：O(m*n)，其中 m 是字符串的平均长度，n 是字符串的数量。

空间复杂度：O(1)

### 6. 盲数游戏中的密码

**题目：** 设计一个密码生成器，使得每次生成的密码都是随机的，但满足以下条件：密码长度为 4 位，每位数字都是 1 到 6 之间的整数，且密码中不能出现连续重复的数字。

**答案：**

```go
package main

import (
	"fmt"
	"math/rand"
	"time"
)

func generatePassword() string {
	rand.Seed(time.Now().UnixNano())
	password := make([]byte, 4)

	for i := 0; i < 4; i++ {
		for {
			password[i] = byte(rand.Intn(6) + 1)
			if i == 0 || password[i] != password[i-1] {
				break
			}
		}
	}

	return string(password)
}

func main() {
	fmt.Println(generatePassword())
}
```

**解析：**

每次生成密码时，随机生成一个 1 到 6 之间的整数，并判断是否与前一个数字相同。如果相同，则继续生成，直到生成不重复的数字。

时间复杂度：O(1)

空间复杂度：O(1)

### 7. 计算器

**题目：** 实现一个简单的计算器，支持加、减、乘、除四种运算。

**答案：**

```go
package main

import (
	"fmt"
)

func calculate(expression string) float64 {
	var stack []float64
	var num float64
	for _, c := range expression {
		if c >= '0' && c <= '9' {
			num = num*10 + float64(int(c)-'0')
		} else {
			if c == '+' {
				stack = append(stack, num)
				num = 0
			} else if c == '-' {
				stack = append(stack, -num)
				num = 0
			} else if c == '*' {
				stack[len(stack)-2] *= num
			} else if c == '/' {
				stack[len(stack)-2] /= num
			}
			num = 0
		}
	}

	return stack[0]
}

func main() {
	fmt.Println(calculate("2+3*4-5/2"))
}
```

**解析：**

使用栈实现计算器。遍历表达式，根据当前字符进行相应的操作，并将结果入栈。最后返回栈顶元素作为结果。

时间复杂度：O(n)

空间复杂度：O(n)

### 8. 最大子序和

**题目：** 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个数），返回其最大和。

**答案：**

```go
package main

import (
	"fmt"
)

func maxSubArray(nums []int) int {
	maxSum := nums[0]
	currentSum := nums[0]

	for i := 1; i < len(nums); i++ {
		currentSum = max(nums[i], currentSum+nums[i])
		maxSum = max(maxSum, currentSum)
	}

	return maxSum
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	nums := []int{-2, 1, -3, 4, -1, 2, 1, -5, 4}
	fmt.Println(maxSubArray(nums))
}
```

**解析：**

使用动态规划求解。遍历数组，计算每个元素对应的最大子序和，并将最大值更新到全局变量中。

时间复杂度：O(n)

空间复杂度：O(1)

### 9. 电话号码的字母组合

**题目：** 给定一个仅包含数字 2-9 的字符串，返回所有可能用字母组成的单词。

**答案：**

```go
package main

import (
	"fmt"
)

var letterMap = map[rune][]string{
	'2': {"a", "b", "c"},
	'3': {"d", "e", "f"},
	'4': {"g", "h", "i"},
	'5': {"j", "k", "l"},
	'6': {"m", "n", "o"},
	'7': {"p", "q", "r", "s"},
	'8': {"t", "u", "v"},
	'9': {"w", "x", "y", "z"},
}

func letterCombinations(digits string) []string {
	if digits == "" {
		return []string{}
	}

	var ans []string
	for _, v := range letterMap[rune(digits[0])] {
		ans = append(ans, v)
	}

	if len(digits) == 1 {
		return ans
	}

	var nextAns []string
	for _, v := range ans {
		for _, w := range letterMap[rune(digits[1])] {
			nextAns = append(nextAns, v+w)
		}
	}

	return letterCombinations(nextAns)
}

func main() {
	digits := "23"
	fmt.Println(letterCombinations(digits))
}
```

**解析：**

使用递归和回溯求解。首先获取第一个数字对应的字母列表，然后递归处理剩余的数字。

时间复杂度：O(3^n)，其中 n 是数字的个数。

空间复杂度：O(n)

### 10. 判断子序列

**题目：** 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。

**答案：**

```go
package main

import (
	"fmt"
)

func isSubsequence(s string, t string) bool {
	i, j := 0, 0
	for i < len(s) && j < len(t) {
		if s[i] == t[j] {
			i++
		}
		j++
	}

	return i == len(s)
}

func main() {
	s := "abc"
	t := "ahbgdc"
	fmt.Println(isSubsequence(s, t))
}
```

**解析：**

使用两个指针分别遍历 s 和 t，当 s 的当前字符与 t 的当前字符相等时，移动 s 的指针；否则，移动 t 的指针。如果 s 的指针移动到末尾，则 s 是 t 的子序列。

时间复杂度：O(n+m)

空间复杂度：O(1)

### 11. 合并两个有序链表

**题目：** 将两个升序链表合并为一个新的升序链表并返回。如果链表长度相等，则将 A 的节点排在 B 的节点之前。

**答案：**

```go
package main

import (
	"fmt"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	if list1 == nil {
		return list2
	}
	if list2 == nil {
		return list1
	}

	if list1.Val < list2.Val {
		list1.Next = mergeTwoLists(list1.Next, list2)
		return list1
	} else {
		list2.Next = mergeTwoLists(list1, list2.Next)
		return list2
	}
}

func main() {
	list1 := &ListNode{Val: 1, Next: &ListNode{Val: 2, Next: &ListNode{Val: 4}}}
	list2 := &ListNode{Val: 1, Next: &ListNode{Val: 3, Next: &ListNode{Val: 4}}}
	fmt.Println(mergeTwoLists(list1, list2))
}
```

**解析：**

递归地将两个有序链表合并为一个有序链表。每次比较两个链表的头节点，将较小值放入新链表中，并递归处理剩余部分。

时间复杂度：O(n+m)

空间复杂度：O(n+m)

### 12. 最长公共前缀

**题目：** 编写一个函数来查找字符串数组中的最长公共前缀。

**答案：**

```go
package main

import (
	"fmt"
)

func longestCommonPrefix(strs []string) string {
	if len(strs) == 0 {
		return ""
	}

	for i, v := range strs[0] {
		for _, s := range strs[1:] {
			if i >= len(s) || s[i] != v {
				return strs[0][:i]
			}
		}
	}

	return strs[0]
}

func main() {
	strs := []string{"flower", "flow", "flight"}
	fmt.Println(longestCommonPrefix(strs))
}
```

**解析：**

遍历第一个字符串，同时遍历其他字符串，直到找到不匹配的字符。返回匹配的部分。

时间复杂度：O(m*n)，其中 m 是字符串的平均长度，n 是字符串的数量。

空间复杂度：O(1)

### 13. 有效的括号

**题目：** 给定一个包含大写和小写字母的字符串 s ，请判断是否能通过添加括号的方法使得输入字符串变为一个有效格式。

**答案：**

```go
package main

import (
	"fmt"
)

func isValid(s string) bool {
	stack := []rune{}
	for _, c := range s {
		if c == '(' || c == '{' || c == '[' {
			stack = append(stack, c)
		} else {
			if len(stack) == 0 {
				return false
			}
			top := stack[len(stack)-1]
			if (c == ')' && top != '(') || (c == '}' && top != '{') || (c == ']' && top != '[') {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}

	return len(stack) == 0
}

func main() {
	s := "()[]{}"
	fmt.Println(isValid(s))
}
```

**解析：**

使用栈实现。遍历字符串，如果遇到左括号，将左括号入栈；如果遇到右括号，判断栈顶元素是否匹配，不匹配则返回 false。遍历结束后，栈为空则表示字符串有效。

时间复杂度：O(n)

空间复杂度：O(n)

### 14. 计数质数

**题目：** 给定一个整数 n ，返回所有小于等于 n 的质数的数目。

**答案：**

```go
package main

import (
	"fmt"
)

func countPrimes(n int) int {
	if n <= 2 {
		return 0
	}

	cnt := 0
	isPrime := make([]bool, n)
	for i := 2; i < n; i++ {
		if isPrime[i] {
			cnt++
			for j := i * i; j < n; j += i {
				isPrime[j] = false
			}
		}
	}

	return cnt
}

func main() {
	n := 10
	fmt.Println(countPrimes(n))
}
```

**解析：**

使用埃拉托斯特尼筛法。初始化一个布尔数组，表示每个数字是否为质数。从 2 开始，遍历到 n，对于每个质数，将其倍数标记为非质数。最后返回质数的数量。

时间复杂度：O(n log log n)

空间复杂度：O(n)

### 15. 两数相加

**题目：** 给出两个 非空 的链表用来表示两个非负的整数，其中，它们各自的位数是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

**答案：**

```go
package main

import (
	"fmt"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	current := dummy
	carry := 0

	for l1 != nil || l2 != nil || carry != 0 {
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
		sum = sum % 10

		current.Next = &ListNode{Val: sum}
		current = current.Next
	}

	return dummy.Next
}

func main() {
	l1 := &ListNode{Val: 2, Next: &ListNode{Val: 4, Next: &ListNode{Val: 3}}}
	l2 := &ListNode{Val: 5, Next: &ListNode{Val: 6, Next: &ListNode{Val: 4}}}
	fmt.Println(addTwoNumbers(l1, l2))
}
```

**解析：**

使用链表存储两个数，从最低位开始相加，如果有进位则加上进位。遍历两个链表的所有节点，直到最后一个节点。

时间复杂度：O(max(m, n))，其中 m 和 n 分别是两个链表的长度。

空间复杂度：O(1)

### 16. 两数之和 II - 输入有序数组

**题目：** 给定一个已按照升序排列 的有序数组 numbers ，请你从数组中找出两个数满足相加之和等于目标数 target 。

**答案：**

```go
package main

import (
	"fmt"
)

func twoSum(numbers []int, target int) []int {
	left, right := 0, len(numbers)-1

	for left < right {
		sum := numbers[left] + numbers[right]
		if sum == target {
			return []int{left + 1, right + 1}
		} else if sum < target {
			left++
		} else {
			right--
		}
	}

	return []int{-1, -1}
}

func main() {
	numbers := []int{2, 7, 11, 15}
	target := 9
	fmt.Println(twoSum(numbers, target))
}
```

**解析：**

使用双指针法。初始时，左指针指向数组的第一个元素，右指针指向数组的最后一个元素。每次循环中，计算两个指针指向的元素之和，根据和与目标值的关系来移动指针。

时间复杂度：O(n)

空间复杂度：O(1)

### 17. 合并两个有序链表

**题目：** 将两个升序链表合并为一个新的升序链表并返回。如果链表长度相等，则将 A 的节点排在 B 的节点之前。

**答案：**

```go
package main

import (
	"fmt"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	if list1 == nil {
		return list2
	}
	if list2 == nil {
		return list1
	}

	if list1.Val < list2.Val {
		list1.Next = mergeTwoLists(list1.Next, list2)
		return list1
	} else {
		list2.Next = mergeTwoLists(list1, list2.Next)
		return list2
	}
}

func main() {
	list1 := &ListNode{Val: 1, Next: &ListNode{Val: 2, Next: &ListNode{Val: 4}}}
	list2 := &ListNode{Val: 1, Next: &ListNode{Val: 3, Next: &ListNode{Val: 4}}}
	fmt.Println(mergeTwoLists(list1, list2))
}
```

**解析：**

递归地将两个有序链表合并为一个有序链表。每次比较两个链表的头节点，将较小值放入新链表中，并递归处理剩余部分。

时间复杂度：O(n+m)

空间复杂度：O(n+m)

### 18. 有效括号字符串

**题目：** 给你一个下标从 0 开始的字符串 s ，该字符串只包含 '(' 和 ')'。如果 s 是一个 有效括号字符串 ，返回 true ；否则，返回 false 。

**答案：**

```go
package main

import (
	"fmt"
)

func isValid(s string) bool {
	stack := []rune{}
	for _, c := range s {
		if c == '(' {
			stack = append(stack, c)
		} else {
			if len(stack) == 0 {
				return false
			}
			top := stack[len(stack)-1]
			if top != '(' {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}

	return len(stack) == 0
}

func main() {
	s := "()()"
	fmt.Println(isValid(s))
}
```

**解析：**

使用栈实现。遍历字符串，遇到左括号入栈，遇到右括号出栈，判断是否匹配。如果栈为空，表示字符串有效。

时间复杂度：O(n)

空间复杂度：O(n)

### 19. 有效的括号字符串

**题目：** 给你一个只包含 '(' 和 ')' 的字符串 s ，判断它是否有效。

**答案：**

```go
package main

import (
	"fmt"
)

func isValid(s string) bool {
	stack := []rune{}
	for _, c := range s {
		if c == '(' {
			stack = append(stack, c)
		} else {
			if len(stack) == 0 {
				return false
			}
			top := stack[len(stack)-1]
			if top != '(' {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}

	return len(stack) == 0
}

func main() {
	s := "()()"
	fmt.Println(isValid(s))
}
```

**解析：**

使用栈实现。遍历字符串，遇到左括号入栈，遇到右括号出栈，判断是否匹配。如果栈为空，表示字符串有效。

时间复杂度：O(n)

空间复杂度：O(n)

### 20. 两数相加

**题目：** 给出两个 非空 的链表用来表示两个非负的整数，其中，它们各自的位数是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

**答案：**

```go
package main

import (
	"fmt"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	current := dummy
	carry := 0

	for l1 != nil || l2 != nil || carry != 0 {
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
		sum = sum % 10

		current.Next = &ListNode{Val: sum}
		current = current.Next
	}

	return dummy.Next
}

func main() {
	l1 := &ListNode{Val: 2, Next: &ListNode{Val: 4, Next: &ListNode{Val: 3}}}
	l2 := &ListNode{Val: 5, Next: &ListNode{Val: 6, Next: &ListNode{Val: 4}}}
	fmt.Println(addTwoNumbers(l1, l2))
}
```

**解析：**

使用链表存储两个数，从最低位开始相加，如果有进位则加上进位。遍历两个链表的所有节点，直到最后一个节点。

时间复杂度：O(max(m, n))，其中 m 和 n 分别是两个链表的长度。

空间复杂度：O(1)

### 21. 判断子序列

**题目：** 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。

**答案：**

```go
package main

import (
	"fmt"
)

func isSubsequence(s string, t string) bool {
	i, j := 0, 0
	for i < len(s) && j < len(t) {
		if s[i] == t[j] {
			i++
		}
		j++
	}

	return i == len(s)
}

func main() {
	s := "abc"
	t := "ahbgdc"
	fmt.Println(isSubsequence(s, t))
}
```

**解析：**

使用两个指针分别遍历 s 和 t，当 s 的当前字符与 t 的当前字符相等时，移动 s 的指针；否则，移动 t 的指针。如果 s 的指针移动到末尾，则 s 是 t 的子序列。

时间复杂度：O(n+m)

空间复杂度：O(1)

### 22. 最小差值 I

**题目：** 给你一个整数数组 arr ，请你找出最小差值，这个差值是数组中任意两个不相邻元素的绝对差值。

**答案：**

```go
package main

import (
	"fmt"
)

func minimumDifference(arr []int) int {
	n := len(arr)
	res := int(1e9)

	for i := 0; i < n; i++ {
		for j := i + 2; j < n; j++ {
			res = min(res, abs(arr[i]-arr[j]))
		}
	}

	return res
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func main() {
	arr := []int{4, 2, 1, 4}
	fmt.Println(minimumDifference(arr))
}
```

**解析：**

遍历数组，对相邻的元素进行计算，更新最小差值。

时间复杂度：O(n^2)

空间复杂度：O(1)

### 23. 最长连续序列

**题目：** 给你一个整数数组 nums ，其中可能包含重复元素，请你返回最长连续序列的长度（最长连续序列是按升序排列的连续元素组成的子数组）。

**答案：**

```go
package main

import (
	"fmt"
)

func longestConsecutive(nums []int) int {
	if len(nums) == 0 {
		return 0
	}

	sets := make(map[int]bool)
	for _, v := range nums {
		sets[v] = true
	}

	res := 1
	start := 0
	for i := range nums {
		if !sets[i] {
			continue
		}

		len := 1
		for j := i + 1; ; j++ {
			if !sets[j] {
				break
			}
			len++
		}

		if len > res {
			res = len
			start = i
		}
	}

	return res
}

func main() {
	nums := []int{100, 4, 200, 1, 3, 2}
	fmt.Println(longestConsecutive(nums))
}
```

**解析：**

使用哈希表记录数组中每个元素是否存在。遍历数组，对于每个元素，判断是否存在一个连续序列，更新最长连续序列的长度。

时间复杂度：O(n)

空间复杂度：O(n)

### 24. 逆元

**题目：** 给你两个正整数 n 和 k，找出所有小于或等于 n 的 k 位的正整数中，按位与（&）操作的结果为 0 的数字的个数。

**答案：**

```go
package main

import (
	"fmt"
)

func countBits(n int, k int) int {
	if k == 0 {
		return n + 1
	}

	if n < k {
		return 0
	}

	mask := 1 << k
	res := 0

	for i := 0; ; i++ {
		if n-i < k {
			break
		}

		mask &= ((1 << (k + 1)) - 1)
		mask = mask << (i + 1)

		res += n / mask
	}

	return res
}

func main() {
	n := 10
	k := 2
	fmt.Println(countBits(n, k))
}
```

**解析：**

使用位运算。对于每个 k 位的二进制数，计算其按位与的结果为 0 的数字个数。通过移位和减法操作，计算所有可能的 k 位数。

时间复杂度：O(k)

空间复杂度：O(1)

### 25. 前 K 个高频元素

**题目：** 给定一个整数数组 nums 和一个整数 k ，请你找出数组中第 k 个高频元素的频率，并按 升序 排列返回前 k 个频率并按升序排列后的结果。

**答案：**

```go
package main

import (
	"fmt"
)

func kthFrequent(nums []int, k int) []int {
	frequencyMap := make(map[int]int)
	for _, v := range nums {
		frequencyMap[v]++
	}

	sortedFrequency := make([][2]int, 0, len(frequencyMap))
	for num, frequency := range frequencyMap {
		sortedFrequency = append(sortedFrequency, [2]int{frequency, num})
	}

	sort.Slice(sortedFrequency, func(i, j int) bool {
		return sortedFrequency[i][0] < sortedFrequency[j][0]
	})

	res := make([]int, 0, k)
	for i := 0; i < k; i++ {
		res = append(res, sortedFrequency[i][1])
	}

	return res
}

func main() {
	nums := []int{1, 1, 1, 2, 2, 3}
	k := 2
	fmt.Println(kthFrequent(nums, k))
}
```

**解析：**

使用哈希表记录每个数字的频率，并将频率和数字组成一个二元数组。对二元数组进行排序，取出前 k 个频率对应的数字。

时间复杂度：O(nlogn)

空间复杂度：O(n)

### 26. 数组中的逆序对

**题目：** 在数组中的两个数字，如果前面数字大于后面的数字，则这两个数字组成一个逆序对。例如，数组[2, 3, 5, 4, 7]中存在5个逆序对：(2, 4)、(2, 5)、(3, 4)、(3, 5) 和 (5, 7)。

**答案：**

```go
package main

import (
	"fmt"
)

func reversePairs(nums []int) int {
	n := len(nums)
	if n < 2 {
		return 0
	}

	cnt := 0

	var merge func(l, r int)
	merge = func(l, r int) {
		m := l + (r-l)/2
		merge(l, m)
		merge(m+1, r)
		t := 0
		i, j := l, m+1
		for i <= m && j <= r {
			if nums[i] <= nums[j] {
				nums[t], t, i = nums[i], t+1, i+1
			} else {
				nums[t], t, j = nums[j], t+1, j+1
				cnt += m - i + 1
			}
		}
		for i <= m {
			nums[t], t, i = nums[i], t+1, i+1
		}
		for j <= r {
			nums[t], t, j = nums[j], t+1, j+1
		}
	}

	merge(0, n-1)
	return cnt
}

func main() {
	nums := []int{7, 5, 6, 4}
	fmt.Println(reversePairs(nums))
}
```

**解析：**

使用归并排序。在合并过程中，当 nums[i] > nums[j] 时，说明 [i, j] 区间内有 j-i 个逆序对。

时间复杂度：O(nlogn)

空间复杂度：O(n)

### 27. 环形链表

**题目：** 给定一个链表，判断链表中是否有环。

**答案：**

```go
package main

import (
	"fmt"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func hasCycle(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}
	return false
}

func main() {
	node1 := &ListNode{Val: 3}
	node2 := &ListNode{Val: 2}
	node3 := &ListNode{Val: 0}
	node4 := &ListNode{Val: -4}
	node1.Next = &ListNode{Val: 4, Next: node2}
	node2.Next = &ListNode{Val: 6, Next: node3}
	node3.Next = &ListNode{Val: 1, Next: node4}
	node4.Next = node2
	fmt.Println(hasCycle(node1))
}
```

**解析：**

使用快慢指针。快指针每次移动两个节点，慢指针每次移动一个节点。如果快指针追上慢指针，则链表中存在环。

时间复杂度：O(n)

空间复杂度：O(1)

### 28. 设计循环缓冲队列

**题目：** 设计一个循环缓冲队列，支持以下操作：enQueue、deQueue 和 peek。请实现 MyCircularQueue 类：

* MyCircularQueue(int k) ：初始化循环缓冲队列，设置位置总数为 k 。
* boolean enQueue(int value) ：向循环缓冲队列中插入一个元素。如果成功，返回 true 。如果队列已满，则返回 false 。
* boolean deQueue() ：从循环缓冲队列中删除一个元素。如果成功，返回 true 。否则，返回 false 。
* int peek() ：获取循环缓冲队列中的元素。如果队列不为空，返回前一个元素。否则，返回 -1 。

**答案：**

```go
package main

import (
	"fmt"
)

type MyCircularQueue struct {
	k       int
	cap     int
	queue   []int
	front   int
	rear    int
}

func Constructor(k int) MyCircularQueue {
	return MyCircularQueue{k: k, cap: k, queue: make([]int, k), front: 0, rear: 0}
}

func (this *MyCircularQueue) enQueue(value int) bool {
	if (this.rear+1)%this.cap == this.front {
		return false
	}

	this.queue[this.rear] = value
	this.rear = (this.rear + 1) % this.cap
	return true
}

func (this *MyCircularQueue) deQueue() bool {
	if this.front == this.rear {
		return false
	}

	this.front = (this.front + 1) % this.cap
	return true
}

func (this *MyCircularQueue) peek() int {
	if this.front == this.rear {
		return -1
	}

	return this.queue[this.front]
}

func main() {
	circularQueue := Constructor(3)
	fmt.Println(circularQueue.enQueue(1)) // true
	fmt.Println(circularQueue.enQueue(2)) // true
	fmt.Println(circularQueue.enQueue(3)) // true
	fmt.Println(circularQueue.enQueue(4)) // false
	fmt.Println(circularQueue.peek())    // 1
	fmt.Println(circularQueue.deQueue()) // true
	fmt.Println(circularQueue.peek())    // 2
}
```

**解析：**

使用数组实现循环缓冲队列。rear 表示尾部，front 表示头部。当队列满时，rear 和 front 相等；当队列为空时，rear 和 front 也相等。

时间复杂度：O(1)

空间复杂度：O(k)

### 29. 设计前缀树

**题目：** 设计前缀树（Trie）并实现以下功能：

* `Trie()` 初始化前缀树对象。
* `insert(word)` 插入字符串 word 到前缀树中。
* `search(word)` 检查是否在前缀树中存在字符串 word 。
* `startsWith(prefix)` 检查是否存在以字符串 prefix 开头的前缀树节点。

**答案：**

```go
package main

import (
	"fmt"
)

type Trie struct {
	children [26]*Trie
	isEnd    bool
}

/** Initialize your data structure here. */
func Constructor() Trie {
	return Trie{}
}

func (this *Trie) Insert(word string) {
	node := &this
	for _, c := range word {
		if node.children[c-'a'] == nil {
			node.children[c-'a'] = &Trie{}
		}
		node = node.children[c-'a']
	}
	node.isEnd = true
}

func (this *Trie) Search(word string) bool {
	node := &this
	for _, c := range word {
		if node.children[c-'a'] == nil {
			return false
		}
		node = node.children[c-'a']
	}
	return node.isEnd
}

func (this *Trie) StartsWith(prefix string) bool {
	node := &this
	for _, c := range prefix {
		if node.children[c-'a'] == nil {
			return false
		}
		node = node.children[c-'a']
	}
	return true
}

func main() {
 trie := Constructor()
	trie.Insert("apple")
	fmt.Println(trie.Search("apple"))    // true
	fmt.Println(trie.Search("app"))       // false
	fmt.Println(trie.StartsWith("app"))    // true
	trie.Insert("app")
	fmt.Println(trie.Search("app"))       // true
}
```

**解析：**

使用数组实现 Trie，数组下标为 0 到 25，对应字母 a 到 z。每个节点包含 26 个子节点和是否为单词结束标志。

时间复杂度：

* Insert：O(m)，其中 m 是单词长度。
* Search：O(m)，其中 m 是单词长度。
* StartsWith：O(m)，其中 m 是前缀长度。

空间复杂度：O(n)，其中 n 是单词数量。

### 30. 搜索旋转排序数组

**题目：** 给你一个长度为 n 的整数数组 nums ，其中 nums[0] = 0，nums[1] = 1 且对于每个 i > 1 ，nums[i] 的值与之前两个数的值都不同。这样的数组把数组分成两个递增子序列。

**答案：**

```go
package main

import (
	"fmt"
)

func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] >= nums[left] {
			if nums[left] <= target && target < nums[mid] {
				right = mid - 1
			} else {
				left = mid + 1
			}
		} else {
			if nums[right] > target && target >= nums[mid] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
	}

	return -1
}

func main() {
	nums := []int{4, 5, 6, 7, 0, 1, 2}
	target := 0
	fmt.Println(search(nums, target))
}
```

**解析：**

使用二分查找。对于旋转后的有序数组，先确定查找的区间，然后根据区间的最小值和最大值确定下一次查找的区间。时间复杂度：O(log n)。空间复杂度：O(1)。 <|im_end|>

