
作者：禅与计算机程序设计艺术                    

# 1.简介
  

正则表达式(Regular Expression)是一个用于匹配字符串模式的工具，它的功能非常强大且灵活，可以用来查找、替换和修改文本中的字符。正则表达式被广泛应用于数据处理、文本分析、文本清洗等领域，为这些领域提供了极其便利的方法。本文旨在提供对正则表达式的简单而全面的介绍，通过一些实例，帮助读者了解如何使用正则表达式进行文本处理。
# 2.基本概念和术语
## 2.1 正则表达式是什么？
正则表达式（regular expression）是一个描述一个字符串的序列的形式，它由普通文字和特殊符号组成。它是一个强大的工具，用于高效的文本匹配、搜索并替换文本内容。正则表达式主要用来验证、解析复杂的文本，提取出有效信息。在应用中，正则表达式通常与正则表达式引擎(regular expression engine)一起工作，它负责对正则表达式进行编译和执行。

正则表达式分为两大类：

1. **基础正则表达式** - 包括最基本的元字符、字母类、数字类等；
2. **组合式正则表达式** - 通过对基础正则表达式的组合和运算，构造出更加复杂的正则表达式。例如，或运算，反向引用，递归等。

## 2.2 基本元字符
基础元字符是构成正则表达式的基本单元。每一个基础元字符都有一个特定的含义。下表给出了一些常用的基础元字符：

| 字符 | 描述                                                         |
| ---- | ------------------------------------------------------------ |
|.    | 表示任意单个字符                                             |
| \d   | 表示任意十进制 digit                                         |
| \D   | 表示任意非十进制 digit                                       |
| \s   | 表示任意空白字符，包括空格、制表符、换行符                       |
| \S   | 表示任意非空白字符                                           |
| \w   | 表示任意字母、数字或者下划线                                  |
| \W   | 表示任意非字母、数字和下划线的字符                             |
| ^    | 在多行模式下表示行首，在单行模式下表示字符串开头               |
| $    | 在多行模式下表示行尾，在单行模式下表示字符串结尾               |
| []   | 表示范围                                                     |
| *    | 零次或多次匹配前面的字符                                     |
| +    | 一次或多次匹配前面的字符                                     |
|?    | 零次或一次匹配前面的字符                                     |
| {n}  | n 次匹配前面的字符                                            |
| {m,n}| m-n 次匹配前面的字符                                          |
| ()   | 分组                                                         |
| \|   | 或运算                                                       |
| (...)| 匹配括号中的字符，作为独立的一组                              |
| (?=...) | 正向肯定界定符，如果...后面出现匹配的位置则成功                  |
| (?!...) | 负向否定界定符，如果...后面没有出现匹配的位置则成功                 |
| (?<=...)| 正向前瞻界定符，如果...前面出现匹配的位置则成功                    |
| (?<!...)| 负向前瞻否定界定符，如果...前面没有出现匹配的位置则成功               |

## 2.3 常用正则表达式命令
### 2.3.1 替换命令 r"pattern"
将正则表达式 pattern 替换为 repl ，其中 repl 可以是一个字符串或函数。

```python
import re

text = "the quick brown fox jumps over the lazy dog."

new_text = re.sub("o", "X", text)

print(new_text) # thX quicX brwn fXX Xmps vr thX lzy Xg Xb
```

在上例中，re.sub() 函数会搜索 text 中的所有 o 字符，然后将它们替换为 X 。 repl 参数也可以是一个函数，这个函数的参数就是每一个匹配到的子串，可以返回一个新的值代替当前的值。 

```python
def replace_caps(match):
    if match.group().isupper():
        return match.group().lower()
    else:
        return match.group().upper()

text = "The Quick Brown Fox Jumps Over The Lazy Dog!"

new_text = re.sub(r"[A-Z]+", replace_caps, text)

print(new_text) # tHE QUIC bROWN FOX JUMP svR TH LZY DOG!
```

在上例中，[A-Z] 代表所有大写字母，replace_caps() 函数接受一个 Match 对象，调用 group() 方法获得匹配到的子串。如果子串全都是大写字母，则转换为小写字母；否则，转化为大写字母。

### 2.3.2 查找命令 re.search() 和 findall()
re.search() 返回一个匹配对象，findall() 返回一个列表，包含所有的匹配结果。 

```python
text = """<NAME>: 800-555-1234
John Smith: 900-555-5678"""

phone_number = re.findall(r"\d{3}-\d{3}-\d{4}", text)

for num in phone_number:
    print(num)

output:
800-555-1234
900-555-5678
```

在上例中，\d{3}-\d{3}-\d{4} 是一个基本正则表达式，匹配电话号码的格式。findall() 函数会找到所有匹配的子串并返回一个列表。

### 2.3.3 忽略大小写 re.IGNORECASE
默认情况下，re模块不区分大小写。我们可以通过设置 re.IGNORECASE 属性让匹配变得不区分大小写。

```python
text = "The Quick Brown Fox Jumps Over The Lazy Dog!"

new_text = re.sub("[a-z]", "X", text, flags=re.IGNORECASE)

print(new_text) # TXX QUIXXX BXXXX FXXXXX JMPXS VRX TH XXZY DGXX!
```

在上例中，设置 re.IGNORECASE 属性之后，"[a-z]" 匹配所有小写字母。

### 2.3.4 多行模式 re.DOTALL
在默认的单行模式下，"." 匹配除 "\n" 以外的所有字符。设置 re.DOTALL 属性后，"." 将匹配所有字符，包括 "\n" 。

```python
text = "the first line.\nThe second line."

new_text = re.sub("\.", ".", text, flags=re.DOTALL)

print(new_text) # the first line..The second line.
```

在上例中，设置 re.DOTALL 属性后，"." 会匹配所有字符，包括 "." 和 "\n" 。因此，第二行被省略掉了。

### 2.3.5 贪婪匹配和惰性匹配
Python 的正则表达式引擎默认采用的是贪婪匹配方式，也就是匹配尽可能多的字符。比如，对于匹配数字的正则表达式 `\d+`，它将匹配尽可能多的数字，而不是只匹配一个。

有时，我们需要让正则表达式采用惰性匹配的方式，也就是只匹配满足条件的最少数量的字符。比如，对于匹配数字的正则表达式 `\d+?`，它只匹配一个数字，然后停止匹配。这种类型的正则表达式在匹配嵌套结构、循环结构等场景下非常有用。

为了实现惰性匹配，我们可以在正则表达式的末尾添加问号 `?` 来表示匹配的可选性。

```python
text = "<div><span>Hello</span></div>"

new_text = re.sub("<.*?>", "", text)

print(new_text) # Hello
```

在上例中，"<.*?>" 匹配所有的 HTML 标签，但是问号 `?` 表示匹配的可选性。因此，最终只保留了文本 "Hello" 。

```python
text = "<div>\n<p>Paragraph 1.</p>\n<ul>\n<li>Item A</li>\n<li>Item B</li>\n</ul>\n</div>"

new_text = re.sub("</?[^>]*>", "", text)

print(new_text) # Paragraph 1 Item A Item B 
```

在上例中，"</?[^>]*>" 匹配所有的 XML/HTML 标签，但是其中又包含了一个非贪婪匹配符号 `*` ，它匹配所有字符直到遇到下一个 > 符号结束标签。因此，最终只保留了文本 "Paragraph 1 Item A Item B" 。