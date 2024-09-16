                 

### AI 2.0 时代的未来工作

#### 领域相关典型问题/面试题库

**1. AI 2.0 与传统 AI 的主要区别是什么？**

**答案：** AI 2.0，通常指的是第二代人工智能，与第一代人工智能（传统 AI）的主要区别在于：

- **自主性增强：** AI 2.0 具有更高的自主性，能够自主学习和决策，而传统 AI 需要明确的规则和指令。
- **数据驱动：** AI 2.0 更依赖于数据驱动的方法，能够从大量数据中自我学习和优化。
- **通用性和灵活性：** AI 2.0 更具通用性和灵活性，能够解决更广泛的问题，而传统 AI 更多地专注于特定领域的问题。
- **更强的交互能力：** AI 2.0 能够与人类更加自然地交互，而传统 AI 的交互能力相对有限。

**2. AI 2.0 如何影响未来的工作？**

**答案：** AI 2.0 将深刻影响未来的工作，具体体现在以下几个方面：

- **自动化替代：** 一些重复性高、规则明确的工作将被自动化取代，降低人力成本。
- **技能需求变化：** 未来对复杂问题解决能力、创造性思维、人际交往能力等的需求将增加。
- **工作模式转变：** 远程办公、弹性工作等新工作模式将成为主流，工作与生活的界限更加模糊。
- **新型职业兴起：** 数据科学家、AI 训练师、AI 风险管理专家等新兴职业将出现。

**3. AI 2.0 的主要技术挑战是什么？**

**答案：** AI 2.0 的主要技术挑战包括：

- **数据隐私和安全：** 如何在确保数据隐私和安全的前提下，充分利用数据来训练 AI 模型。
- **算法透明性和可解释性：** 如何提高算法的透明性和可解释性，使其结果可被用户理解。
- **计算资源消耗：** 如何在有限的计算资源下，高效地训练和部署复杂的 AI 模型。
- **算法偏见和公平性：** 如何避免算法在训练过程中产生偏见，确保其对不同群体的公平性。

**4. AI 2.0 如何促进社会进步？**

**答案：** AI 2.0 将在多个领域促进社会进步：

- **医疗健康：** AI 可以辅助医生进行疾病诊断、个性化治疗，提高医疗服务的效率和质量。
- **教育：** AI 可以个性化教学，帮助学生更高效地学习，同时为教师提供教学辅助。
- **环境保护：** AI 可以在环境监测、资源优化等方面发挥重要作用，助力可持续发展。
- **城市治理：** AI 可以在交通管理、公共安全等领域提供智能决策支持，提高城市治理水平。

**5. 如何确保 AI 2.0 的发展遵循伦理道德原则？**

**答案：** 确保 AI 2.0 发展遵循伦理道德原则的方法包括：

- **制定伦理规范：** 制定明确的 AI 伦理规范，确保 AI 系统在设计和应用过程中遵循。
- **透明度和可解释性：** 提高算法的透明度和可解释性，使其结果可被监督和审查。
- **法律监管：** 通过立法对 AI 应用进行监管，确保其不违反法律法规。
- **社会共识：** 加强社会各界的对话和合作，形成共同的伦理道德共识。

#### 算法编程题库及答案解析

**1. 请实现一个函数，判断一个字符串是否为回文。**

```go
func isPalindrome(s string) bool {
    for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
        if s[i] != s[j] {
            return false
        }
    }
    return true
}
```

**解析：** 该函数使用双指针法，一个指针从字符串的开头开始遍历，另一个指针从字符串的结尾开始遍历，比较两个指针指向的字符是否相同，直到两个指针相遇。如果过程中出现不相等的字符，函数返回 `false`；否则返回 `true`。

**2. 请实现一个函数，找出字符串中的第一个唯一字符。**

```go
func firstUniqChar(s string) int {
    cnt := [128]int{}
    for i := 0; i < len(s); i++ {
        cnt[s[i]]++
    }
    for i, v := range s {
        if v == ' ' {
            continue
        }
        if cnt[v] == 1 {
            return i
        }
    }
    return -1
}
```

**解析：** 该函数首先使用一个数组 `cnt` 统计字符串中每个字符的出现次数。然后遍历字符串，如果找到一个字符只出现一次，则返回其位置。如果遍历完整个字符串都没有找到，返回 `-1`。

**3. 请实现一个函数，将字符串中的空格替换为 `%20`。**

```go
func replaceSpaces(s string, n int) string {
    return strings.Replace(s, " ", "%20", -1)
}
```

**解析：** 该函数直接使用 `strings` 包的 `Replace` 函数将字符串中的空格替换为 `%20`。`-1` 参数表示替换所有匹配到的空格。

**4. 请实现一个函数，找出字符串中的最长公共前缀。**

```go
func longestCommonPrefix(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    prefix := strs[0]
    for i := 1; i < len(strs); i++ {
        for len(prefix) > 0 {
            if !strings.HasPrefix(strs[i], prefix) {
                prefix = prefix[:len(prefix)-1]
            } else {
                break
            }
        }
        if len(prefix) == 0 {
            return ""
        }
    }
    return prefix
}
```

**解析：** 该函数首先取第一个字符串作为初始的前缀。然后依次与后续的字符串比较，如果当前字符串不以当前前缀开头，则缩短前缀。这个过程一直持续到找到一个公共前缀或者所有字符串都没有公共前缀。

**5. 请实现一个函数，计算两个数的最大公约数。**

```go
func gcd(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}
```

**解析：** 该函数使用辗转相除法计算最大公约数。在每次迭代中，交换 `a` 和 `b` 的值，并将 `a` 替换为 `b`，将 `b` 替换为 `a%b`。这个过程一直持续到 `b` 为 0，此时 `a` 的值即为最大公约数。

**6. 请实现一个函数，判断一个整数是否是回文。**

```go
func isPalindrome(x int) bool {
    if x < 0 || (x != 0 && x%10 == 0) {
        return false
    }
    reversed := 0
    for x > reversed {
        reversed = reversed*10 + x%10
        x /= 10
    }
    return x == reversed || x == reversed/10
}
```

**解析：** 该函数首先排除负数和末尾为 0 的整数，因为这些整数不可能是回文。然后，它通过不断将 `x` 的个位数加到 `reversed` 上，并将 `x` 除以 10，将 `x` 逐位减少。当 `x` 小于或等于 `reversed` 时，比较 `x` 和 `reversed` 的值，如果相等，则 `x` 是回文。

**7. 请实现一个函数，计算两个数的最小公倍数。**

```go
func lcm(a, b int) int {
    return a / gcd(a, b) * b
}
```

**解析：** 该函数使用最大公约数（`gcd`）来计算最小公倍数（`lcm`）。最小公倍数等于两个数的乘积除以它们的最大公约数。

**8. 请实现一个函数，将一个字符串逆序。**

```go
func reverseString(s string) string {
    runes := []rune(s)
    n := len(runes)
    for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
        runes[i], runes[j] = runes[j], runes[i]
    }
    return string(runes)
}
```

**解析：** 该函数将字符串转换为 rune 切片，然后使用双指针交换法逆序 rune 切片，最后将 rune 切片转换回字符串。

**9. 请实现一个函数，找出字符串中的第一个重复字符。**

```go
func firstRecurringCharacter(s string) rune {
    seen := make(map[rune]bool)
    for _, v := range s {
        if seen[v] {
            return v
        }
        seen[v] = true
    }
    return ' '
}
```

**解析：** 该函数使用一个映射表 `seen` 来记录已见过的字符。遍历字符串时，如果字符已在 `seen` 中，则返回该字符。否则，将该字符添加到 `seen` 中。如果遍历完整个字符串都没有找到重复字符，则返回空格。

**10. 请实现一个函数，判断一个二进制字符串是否是有效的回文。**

```go
func isBinaryPalindrome(s string) bool {
    for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
        if s[i] != s[j] {
            return s[i] == '0' && s[j] == '1' || s[i] == '1' && s[j] == '0'
        }
    }
    return true
}
```

**解析：** 该函数与第 1 题类似，使用双指针法遍历字符串。如果遇到两个不同的字符，则判断它们是否一个是 `0` 另一个是 `1`，如果是，则继续遍历；否则返回 `false`。如果遍历完整个字符串没有遇到不同的字符，则返回 `true`。

**11. 请实现一个函数，计算字符串中单词的个数。**

```go
func countWords(s string) int {
    words := strings.Fields(s)
    return len(words)
}
```

**解析：** 该函数使用 `strings` 包的 `Fields` 函数将字符串按空格分割成单词，然后返回单词的个数。

**12. 请实现一个函数，找出字符串中的第一个唯一单词。**

```go
func firstUniqueWord(s string) string {
    cnt := [128]int{}
    words := strings.Fields(s)
    for _, v := range words {
        for _, c := range v {
            cnt[c]++
        }
    }
    for _, v := range words {
        for _, c := range v {
            if cnt[c] != 1 {
                break
            }
        }
        if cnt[c] == 1 {
            return v
        }
    }
    return ""
}
```

**解析：** 该函数首先统计字符串中每个字符的出现次数。然后遍历分割后的单词，如果单词中的每个字符都只出现一次，则返回该单词。如果遍历完所有单词都没有找到，则返回空字符串。

**13. 请实现一个函数，计算字符串中所有单词的总长度。**

```go
func sumLengthOfWords(s string) int {
    words := strings.Fields(s)
    sum := 0
    for _, v := range words {
        sum += len(v)
    }
    return sum
}
```

**解析：** 该函数与第 12 题类似，使用 `strings` 包的 `Fields` 函数将字符串按空格分割成单词，然后计算单词长度的总和。

**14. 请实现一个函数，检查字符串是否是数字。**

```go
func isNumber(s string) bool {
    _, err := strconv.ParseFloat(s, 64)
    return err == nil
}
```

**解析：** 该函数使用 `strconv` 包的 `ParseFloat` 函数尝试将字符串解析为浮点数。如果解析成功，说明字符串是一个数字，返回 `true`；否则返回 `false`。

**15. 请实现一个函数，找出字符串中的最长公共子串。**

```go
func longestCommonSubstring(s1, s2 string) string {
    m, n := len(s1), len(s2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    start, mx := 0, 0
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if s1[i-1] == s2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > mx {
                    mx = dp[i][j]
                    start = i - mx
                }
            } else {
                dp[i][j] = 0
            }
        }
    }
    return s1[start : start+mx]
}
```

**解析：** 该函数使用动态规划（DP）算法找到两个字符串的最长公共子串。`dp[i][j]` 表示 `s1` 的前 `i` 个字符和 `s2` 的前 `j` 个字符的最长公共子串的长度。在填充 DP 表时，如果当前字符匹配，则 `dp[i][j] = dp[i-1][j-1] + 1`；否则重置为 0。最后找到最长公共子串的起始位置和长度，并返回子串。

**16. 请实现一个函数，计算字符串中数字的和。**

```go
func sumNumbers(s string) int {
    result := 0
    sign := 1
    num := 0
    for _, c := range s {
        if c == '-' {
            sign = -sign
            num = 0
        } else if c >= '0' && c <= '9' {
            num = num*10 + int(c-'0')
        } else {
            result += sign * num
            num = 0
        }
    }
    result += sign * num
    return result
}
```

**解析：** 该函数遍历字符串，根据字符的值计算数字的和。如果遇到 `-`，则改变当前数字的符号；如果遇到数字字符，则将其累加到当前数字；如果遇到非数字字符，则将当前数字加到结果中，并重置当前数字。

**17. 请实现一个函数，找出字符串中的所有数字。**

```go
func extractNumbers(s string) []int {
    numbers := []int{}
    sign := 1
    num := 0
    for _, c := range s {
        if c == '-' {
            sign = -sign
            num = 0
        } else if c >= '0' && c <= '9' {
            num = num*10 + int(c-'0')
        } else {
            if num != 0 {
                numbers = append(numbers, sign*num)
                num = 0
            }
        }
    }
    if num != 0 {
        numbers = append(numbers, sign*num)
    }
    return numbers
}
```

**解析：** 该函数与第 16 题类似，遍历字符串并提取所有数字。如果遇到 `-`，则改变当前数字的符号；如果遇到数字字符，则将其累加到当前数字；如果遇到非数字字符，则将当前数字添加到结果列表中，并重置当前数字。遍历结束后，如果当前数字不为 0，也将其添加到结果列表中。

**18. 请实现一个函数，判断字符串是否是有效的括号序列。**

```go
func isValid(s string) bool {
    stack := []rune{}
    for _, c := range s {
        if c == '(' || c == '[' || c == '{' {
            stack = append(stack, c)
        } else if len(stack) == 0 || (c == ')' && stack[len(stack)-1] != '(') || (c == ']' && stack[len(stack)-1] != '[') || (c == '}' && stack[len(stack)-1] != '{') {
            return false
        } else {
            stack = stack[:len(stack)-1]
        }
    }
    return len(stack) == 0
}
```

**解析：** 该函数使用栈（`stack`）来检查字符串是否是有效的括号序列。遍历字符串时，如果遇到左括号，将其压入栈；如果遇到右括号，则检查栈顶元素是否匹配，如果匹配，则弹出栈顶元素；如果不匹配或栈为空，则返回 `false`。遍历结束后，如果栈为空，则返回 `true`。

**19. 请实现一个函数，计算字符串的哈希值。**

```go
func hash(s string) uint32 {
    hash := uint32(2166136261)
    for _, c := range s {
        hash = hash*65599 + uint32(c)
    }
    return hash
}
```

**解析：** 该函数使用简单的哈希算法计算字符串的哈希值。哈希值 `hash` 初始化为一个固定的常数，然后遍历字符串中的每个字符，使用乘法和加法运算更新哈希值。最后返回计算得到的哈希值。

**20. 请实现一个函数，找出字符串中的最长重复子串。**

```go
func longestRepeatingSubstring(s string) string {
    n := len(s)
    lps := make([]int, n)
    j := 0
    for i := 1; i < n; i++ {
        if s[i] == s[j] {
            j++
            lps[i] = j
        } else {
            if j != 0 {
                j = lps[j-1]
                i--
            }
        }
    }
    maxLen, endIndex := 0, 0
    for i := 1; i < n; i++ {
        if lps[i] > maxLen {
            maxLen = lps[i]
            endIndex = i
        }
    }
    return s[endIndex : endIndex+maxLen]
}
```

**解析：** 该函数使用最长公共前缀（LPS）算法找到字符串中的最长重复子串。`lps[i]` 表示字符串的前 `i` 个字符的最长公共前缀的长度。在遍历过程中，如果当前字符与前一个字符相同，则增加 `lps` 的值；否则，根据 `lps` 的前一个值调整当前 `lps` 的值，并可能回溯。最后找到最长重复子串的起始位置和长度，并返回子串。

#### 源代码实例

以下是一系列 Golang 源代码实例，用于实现上述算法编程题。

**1. 判断字符串是否为回文。**

```go
package main

import (
    "fmt"
)

func isPalindrome(s string) bool {
    for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
        if s[i] != s[j] {
            return false
        }
    }
    return true
}

func main() {
    s := "level"
    if isPalindrome(s) {
        fmt.Println(s, "是回文")
    } else {
        fmt.Println(s, "不是回文")
    }
}
```

**2. 找出字符串中的第一个唯一字符。**

```go
package main

import (
    "fmt"
    "strings"
)

func firstUniqChar(s string) int {
    cnt := [128]int{}
    for i := 0; i < len(s); i++ {
        cnt[s[i]]++
    }
    for i, v := range s {
        if v == ' ' {
            continue
        }
        if cnt[v] == 1 {
            return i
        }
    }
    return -1
}

func main() {
    s := "loveleetcode"
    idx := firstUniqChar(s)
    if idx != -1 {
        fmt.Println("第一个唯一字符的索引是：", idx)
    } else {
        fmt.Println("没有唯一字符")
    }
}
```

**3. 将字符串中的空格替换为 `%20`。**

```go
package main

import (
    "fmt"
    "strings"
)

func replaceSpaces(s string, n int) string {
    return strings.Replace(s, " ", "%20", -1)
}

func main() {
    s := "Hello World!"
    n := len(s)
    result := replaceSpaces(s, n)
    fmt.Println("替换后的字符串：", result)
}
```

**4. 找出字符串中的最长公共前缀。**

```go
package main

import (
    "fmt"
    "strings"
)

func longestCommonPrefix(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    prefix := strs[0]
    for i := 1; i < len(strs); i++ {
        for len(prefix) > 0 {
            if !strings.HasPrefix(strs[i], prefix) {
                prefix = prefix[:len(prefix)-1]
            } else {
                break
            }
        }
        if len(prefix) == 0 {
            return ""
        }
    }
    return prefix
}

func main() {
    strs := []string{"flower", "flow", "flight"}
    result := longestCommonPrefix(strs)
    fmt.Println("最长公共前缀：", result)
}
```

**5. 计算两个数的最大公约数。**

```go
package main

import (
    "fmt"
)

func gcd(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

func main() {
    a := 24
    b := 18
    result := gcd(a, b)
    fmt.Println("最大公约数：", result)
}
```

**6. 计算两个数的最小公倍数。**

```go
package main

import (
    "fmt"
)

func lcm(a, b int) int {
    return a / gcd(a, b) * b
}

func main() {
    a := 15
    b := 20
    result := lcm(a, b)
    fmt.Println("最小公倍数：", result)
}
```

**7. 将一个字符串逆序。**

```go
package main

import (
    "fmt"
)

func reverseString(s string) string {
    runes := []rune(s)
    n := len(runes)
    for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
        runes[i], runes[j] = runes[j], runes[i]
    }
    return string(runes)
}

func main() {
    s := "hello"
    result := reverseString(s)
    fmt.Println("逆序后的字符串：", result)
}
```

**8. 找出字符串中的第一个重复字符。**

```go
package main

import (
    "fmt"
)

func firstRecurringCharacter(s string) rune {
    seen := make(map[rune]bool)
    for _, v := range s {
        if seen[v] {
            return v
        }
        seen[v] = true
    }
    return ' '
}

func main() {
    s := "swiss"
    result := firstRecurringCharacter(s)
    fmt.Println("第一个重复字符：", string(result))
}
```

**9. 判断字符串是否是有效的回文。**

```go
package main

import (
    "fmt"
)

func isBinaryPalindrome(s string) bool {
    for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
        if s[i] != s[j] {
            return s[i] == '0' && s[j] == '1' || s[i] == '1' && s[j] == '0'
        }
    }
    return true
}

func main() {
    s := "10101"
    if isBinaryPalindrome(s) {
        fmt.Println(s, "是有效的回文")
    } else {
        fmt.Println(s, "不是有效的回文")
    }
}
```

**10. 计算字符串中单词的个数。**

```go
package main

import (
    "fmt"
    "strings"
)

func countWords(s string) int {
    words := strings.Fields(s)
    return len(words)
}

func main() {
    s := "Hello, world! This is a test."
    result := countWords(s)
    fmt.Println("单词个数：", result)
}
```

**11. 找出字符串中的第一个唯一单词。**

```go
package main

import (
    "fmt"
    "strings"
)

func firstUniqueWord(s string) string {
    cnt := [128]int{}
    words := strings.Fields(s)
    for _, v := range words {
        for _, c := range v {
            cnt[c]++
        }
    }
    for _, v := range words {
        for _, c := range v {
            if cnt[c] != 1 {
                break
            }
        }
        if cnt[c] == 1 {
            return v
        }
    }
    return ""
}

func main() {
    s := "loveleetcode"
    result := firstUniqueWord(s)
    if result != "" {
        fmt.Println("第一个唯一单词：", result)
    } else {
        fmt.Println("没有唯一单词")
    }
}
```

**12. 计算字符串中单词的总长度。**

```go
package main

import (
    "fmt"
    "strings"
)

func sumLengthOfWords(s string) int {
    words := strings.Fields(s)
    sum := 0
    for _, v := range words {
        sum += len(v)
    }
    return sum
}

func main() {
    s := "Hello, world! This is a test."
    result := sumLengthOfWords(s)
    fmt.Println("单词总长度：", result)
}
```

**13. 检查字符串是否是数字。**

```go
package main

import (
    "fmt"
    "strconv"
)

func isNumber(s string) bool {
    _, err := strconv.ParseFloat(s, 64)
    return err == nil
}

func main() {
    s := "123.456"
    if isNumber(s) {
        fmt.Println(s, "是数字")
    } else {
        fmt.Println(s, "不是数字")
    }
}
```

**14. 找出字符串中的最长公共子串。**

```go
package main

import (
    "fmt"
    "strings"
)

func longestCommonSubstring(s1, s2 string) string {
    m, n := len(s1), len(s2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    start, mx := 0, 0
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if s1[i-1] == s2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > mx {
                    mx = dp[i][j]
                    start = i - mx
                }
            } else {
                dp[i][j] = 0
            }
        }
    }
    return s1[start : start+mx]
}

func main() {
    s1 := "abcde"
    s2 := "acedb"
    result := longestCommonSubstring(s1, s2)
    fmt.Println("最长公共子串：", result)
}
```

**15. 计算字符串中数字的和。**

```go
package main

import (
    "fmt"
    "strconv"
)

func sumNumbers(s string) int {
    result := 0
    sign := 1
    num := 0
    for _, c := range s {
        if c == '-' {
            sign = -sign
            num = 0
        } else if c >= '0' && c <= '9' {
            num = num*10 + int(c-'0')
        } else {
            result += sign * num
            num = 0
        }
    }
    result += sign * num
    return result
}

func main() {
    s := "3-1-9+5+4-1"
    result := sumNumbers(s)
    fmt.Println("数字的和：", result)
}
```

**16. 提取字符串中的所有数字。**

```go
package main

import (
    "fmt"
)

func extractNumbers(s string) []int {
    numbers := []int{}
    sign := 1
    num := 0
    for _, c := range s {
        if c == '-' {
            sign = -sign
            num = 0
        } else if c >= '0' && c <= '9' {
            num = num*10 + int(c-'0')
        } else {
            if num != 0 {
                numbers = append(numbers, sign*num)
                num = 0
            }
        }
    }
    if num != 0 {
        numbers = append(numbers, sign*num)
    }
    return numbers
}

func main() {
    s := "Hello 123 World - 456"
    result := extractNumbers(s)
    fmt.Println("提取的数字：", result)
}
```

**17. 判断字符串是否是有效的括号序列。**

```go
package main

import (
    "fmt"
)

func isValid(s string) bool {
    stack := []rune{}
    for _, c := range s {
        if c == '(' || c == '[' || c == '{' {
            stack = append(stack, c)
        } else if len(stack) == 0 || (c == ')' && stack[len(stack)-1] != '(') || (c == ']' && stack[len(stack)-1] != '[') || (c == '}' && stack[len(stack)-1] != '{') {
            return false
        } else {
            stack = stack[:len(stack)-1]
        }
    }
    return len(stack) == 0
}

func main() {
    s := "({})"
    if isValid(s) {
        fmt.Println(s, "是有效的括号序列")
    } else {
        fmt.Println(s, "不是有效的括号序列")
    }
}
```

**18. 计算字符串的哈希值。**

```go
package main

import (
    "fmt"
)

func hash(s string) uint32 {
    hash := uint32(2166136261)
    for _, c := range s {
        hash = hash*65599 + uint32(c)
    }
    return hash
}

func main() {
    s := "hello"
    result := hash(s)
    fmt.Println("哈希值：", result)
}
```

**19. 找出字符串中的最长重复子串。**

```go
package main

import (
    "fmt"
)

func longestRepeatingSubstring(s string) string {
    n := len(s)
    lps := make([]int, n)
    j := 0
    for i := 1; i < n; i++ {
        if s[i] == s[j] {
            j++
            lps[i] = j
        } else {
            if j != 0 {
                j = lps[j-1]
                i--
            }
        }
    }
    maxLen, endIndex := 0, 0
    for i := 1; i < n; i++ {
        if lps[i] > maxLen {
            maxLen = lps[i]
            endIndex = i
        }
    }
    return s[endIndex : endIndex+maxLen]
}

func main() {
    s := "abcabcabc"
    result := longestRepeatingSubstring(s)
    fmt.Println("最长重复子串：", result)
}
```

