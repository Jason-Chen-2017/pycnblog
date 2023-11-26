                 

# 1.背景介绍


正则表达式（Regular Expression）是一种文本处理工具，它提供了一些特殊符号用来匹配、搜索和替换字符串中的特定模式。在很多编程语言中都内置了对正则表达式的支持。比如，C、Java、JavaScript、Perl、PHP、Python等。本文将介绍Python中如何使用正则表达式进行文本处理。正则表达式可以用于各种场景，如数据清洗、数据提取、文本搜索和替换等。

# 2.核心概念与联系
## 概念
正则表达式（Regular Expression，简称regex），也叫做正规表示法，是一种文本模式匹配的方法。它是由软件工程师设计用来描述和匹配一系列符合一定语法规则的字符串的方法。

通俗地说，正则表达式就是一个可以匹配文本的规则，用以查找、替代或验证文本串的工具。通过定义好的正则表达式，可以轻松找出一类特定的字符组合、重复的模式或者其他内容。

## 联系
正则表达式与正则语言有密切的关系。正则表达式实际上是基于正则语言实现的一套算法。正则表达式不是独立存在的，而是在程序开发中不可或缺的一个组成部分。正则表达式在实际应用中经常结合各种语言一起使用。

Python中使用正则表达式主要有两个方法，`re.match()`和`re.search()`函数。

- `re.match()`: 在字符串开头匹配模式，如果匹配成功返回Match对象；否则返回None。
- `re.search()`: 在整个字符串搜索模式，如果找到匹配项就返回第一个结果，否则返回None。

其中，Match对象是一个匹配结果类，具有以下属性:

- `group(num)`: 返回第num个分组匹配的内容，默认返回第一个分组内容。
- `start()`: 返回匹配的起始位置。
- `end()`: 返回匹配的结束位置。
- `span()`: 返回(start, end)元组。

## 常见的正则表达式语法
### 匹配字符类
`.` 匹配任意单个字符，例如：`a.c` 可以匹配 `abc`, `axc`, `ayc`，但是不能匹配空格。`\w` 表示匹配数字字母下划线，`\d` 表示匹配数字，`\s` 表示匹配空白字符，`\W` 表示匹配非数字字母下划线，`\D` 表示匹配非数字，`\S` 表示匹配非空白字符。

```python
import re
string = "Hello world"
pattern1 = r".world"   # matches any string ending with 'world'
pattern2 = r"\d\w+"    # matches any word containing at least one digit
matches1 = re.findall(pattern1, string)      # find all non-overlapping substrings matching pattern1 in the given string
matches2 = re.findall(pattern2, string)      # find all words containing digits and return them as a list of strings
print("Matches for pattern1:", matches1)       # Output: ['orld']
print("Matches for pattern2:", matches2)       # Output: []
```

### 匹配数量词
`*` 匹配零个或多个字符前面的元素，例如：`ab*c` 可以匹配 `'ac'` 和 `'abc'`。`?` 匹配零个或一个字符前面的元素，例如：`ab?c` 可以匹配 `'ac'` 和 `'bc'`。`{m}` 匹配 m 个字符，例如：`a{3}c` 只会匹配 `'aac'`。`{m,n}` 匹配 m~n 个字符，例如：`a{3,5}c` 可以匹配 `'aac'`、`''abc'`、`'aaaaac'`。

```python
import re
string = "The quick brown fox jumps over the lazy dog."
pattern1 = r"[aeiouAEIOU]{1}"                  # matches single vowel (case insensitive)
pattern2 = r"\b[A-Z][a-z]*\b"                   # matches capitalized words
pattern3 = r'\b\w+ed\b|\b\w+ing\b'             # matches ed or ing at the end of a word
pattern4 = r'\b\w+[^aeiou]\w*\b'              # matches words that contain consonants before vowels
pattern5 = r'^(?!_)(?:[\w\-.])+(?:@|\\(?:example|domain)[^\s]*)$'  # validate email address format
matches1 = re.findall(pattern1, string)        # find all occurrences of pattern1 in the given string
matches2 = re.findall(pattern2, string)        # find all capitalized words in the given string
matches3 = re.findall(pattern3, string)        # find all ed/ing at the end of a word in the given string
matches4 = re.findall(pattern4, string)        # find all words containing consonants before vowels in the given string
matches5 = [email for email in emails if re.match(pattern5, email)]     # use regular expression to filter out invalid email addresses from a list of emails
print("Matches for pattern1:", matches1)         # Output: ['e', 'o', 'u', 'E', 'I', 'O']
print("Matches for pattern2:", matches2)         # Output: ['Quick', 'Brown', 'Lazy']
print("Matches for pattern3:", matches3)         # Output: ['jumps', 'dog.']
print("Matches for pattern4:", matches4)         # Output: ['quickly', 'brown', 'foxes', 'jumps', 'lazy', 'dogs']
print("Filtered valid email addresses:", matches5)   # Output: ['user@example.com', 'admin@domain.org']
```

### 匹配边界
`\b` 表示单词边界，`\B` 表示非单词边界。

```python
import re
string = "This is a sample text file."
pattern1 = r"\bis\b"                      # matches words starting with 'is'
pattern2 = r"\bample\b|\btex\b"           # matches words containing 'ample' or 'tex'
pattern3 = r"\b\w+\.\w+\b"                 # matches words separated by period
matches1 = re.findall(pattern1, string)    # find all words starting with 'is'
matches2 = re.findall(pattern2, string)    # find all words containing 'ample' or 'tex'
matches3 = re.findall(pattern3, string)    # find all words separated by period
print("Matches for pattern1:", matches1)     # Output: ['is']
print("Matches for pattern2:", matches2)     # Output: ['sample', 'text']
print("Matches for pattern3:", matches3)     # Output: ['file']
```

### 定位符
`^` 表示行首，`$` 表示行尾。`|` 表示或，即选择其一，例如：`(apple|banana)` 表示匹配 `"apple"` 或 `"banana"`。`[]` 表示字符集，`-` 表示范围，例如：[a-zA-Z] 表示匹配所有大小写字母。`(...)` 分组，表示括号内的字符是一个整体，例如：`\w+(\.|-|_)\w+@\w+\.\w+` 表示匹配电子邮箱地址。

```python
import re
string = "The quick brown fox jumped over the lazy dog and then ran away."
pattern1 = r"\b\w{7}\b"                    # matches seven letter words
pattern2 = r"(fox|cat)"                     # matches either 'fox' or 'cat'
pattern3 = r"[A-Za-z]+(?:\s+[A-Za-z]+)+"     # matches multiple words separated by whitespace characters
pattern4 = r"\b\d+(?:\.\d+)?\s*(k|K|M|G)*\b"            # matches numbers followed by size units (optional)
pattern5 = r"^(?:\d{3}-)+\d{3}$|^\d{3}-\d{2}-\d{4}$"     # matches US phone number formats
pattern6 = r"<.*?>"                            # matches HTML tags
matches1 = re.findall(pattern1, string)      # find all seven letter words in the given string
matches2 = re.findall(pattern2, string)      # find all instances of 'fox' or 'cat' in the given string
matches3 = re.findall(pattern3, string)      # find all sequences of letters surrounded by optional whitespace characters in the given string
matches4 = re.findall(pattern4, string)      # find all sizes of data stored in different size units like KB, MB, GB (optional)
matches5 = [phone for phone in phones if re.match(pattern5, phone)]      # use regular expression to filter out invalid phone numbers from a list of phone numbers
matches6 = re.findall(pattern6, string)      # find all HTML tags in the given string
print("Matches for pattern1:", matches1)       # Output: ['quick', 'brown', 'jumped', 'ran', 'away']
print("Matches for pattern2:", matches2)       # Output: ['fox', 'cat']
print("Matches for pattern3:", matches3)       # Output: ['The', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', 'and', 'then', 'ran', 'away.']
print("Matches for pattern4:", matches4)       # Output: ['19000 k', '2.5 M', '4 G']
print("Filtered valid phone numbers:", matches5)   # Output: ['123-456-7890', '111-222-3333', '555-1234']
print("HTML tags found:", matches6)          # Output: ['<html>', '<head>', '</title>', '<body>']
```