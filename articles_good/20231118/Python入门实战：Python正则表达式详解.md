                 

# 1.背景介绍


正则表达式（Regular Expression）是一个特殊字符串用于匹配文本或者查找模式，它属于编程领域的一类技术。它的作用主要是在文本处理中用来快速进行批量修改、搜索、替换等操作。许多高级语言都提供了对正则表达式的支持，Python也不例外。本文将通过实战案例，全面掌握Python中的正则表达式语法及各种应用场景。文章涉及的内容包括：
# （1）正则表达式概述
# （2）基本知识
# （3）模式语法
# （4）匹配方法
# （5）字符集
# （6）量词
# （7）预定义模式
# （8）贪婪与非贪婪匹配
# （9）子组与零宽断言
# （10）反斜杠转义
# （11）Python标准库中的re模块
# （12）案例解析
其中，前七章主要介绍了一些基础概念和语法结构，后三章则是一些进阶应用案例。如果你对正则表达式已经有了一定的了解，那么可以直接跳过前七章的内容。对于这么一个庞大的技术体系，如何用最简洁的代码实现复杂功能呢？下面让我们开始学习Python正则表达式吧！


# 2.核心概念与联系
## 2.1 正则表达式概述
正则表达式（Regular Expression）是一种文本处理工具，它可以帮助用户方便的进行字符串匹配、编辑、搜索以及替换操作。在Python中，正则表达式也称作regex或re。正则表达式可以用来匹配文本，也可以用来验证输入字段是否符合指定的格式。正则表达式通常都是用一种灵活、便捷且强大的模式语言编写的，但是需要仔细阅读并理解其规则才能更好地使用它们。正则表达式具有高度的灵活性，可以匹配几乎所有的数据类型。此外，正则表达式还有一些高级功能，如查找并替换、捕获分组、数据提取、正则表达式速查表等，这些都超出了本文范围。但总之，只要掌握了正则表达式的基本知识和用法，就可以解决很多实际的问题。

## 2.2 基本知识
### 2.2.1 模式和匹配
模式(Pattern)：正则表达式的核心功能就是从大量文本当中匹配特定的模式。模式有两种表达方式，第一种是普通模式，即没有任何特殊符号的模式；第二种是元字符模式，它由一些元字符组成，可以用来指定各种匹配规则。常用的元字符如下：
- `.`：匹配任意单个字符，除了换行符。
- `\d`：匹配任意数字。
- `\D`：匹配任意非数字字符。
- `\s`：匹配任意空白字符，包括空格、制表符、换页符等。
- `\S`：匹配任意非空白字符。
- `\w`：匹配任意单词字符，即字母、数字和下划线。
- `\W`：匹配任意非单词字符。
- `[]`：匹配括号内的任何字符。
- `|`：或运算符，表示两个或多个匹配项。
- `*`：匹配零个或多个前面的模式。
- `+`：匹配一个或多个前面的模式。
- `{n}`：匹配n个前面的模式。
- `{m,n}`：匹配m到n个前面的模式。
- `?`：匹配零个或一个前面的模式。
- `()`：创建并返回一个组。
- `^`：匹配行首。
- `$`：匹配行尾。
- `\b`：匹配单词边界。
- `\B`：匹配非单词边界。

如果想要匹配元字符本身，则可以使用`\`对其进行转义，比如`\*`匹配星号，`\+`匹配加号。

匹配(Match)：正则表达式用来描述待匹配的模式，当搜索文本或文档时，正则表达式引擎会尝试用这个模式与目标文本进行匹配。匹配成功时，正则表达式引擎返回的是一个Match对象，该对象存储了匹配的文本信息。

例子：
```python
import re

text = "hello world"
pattern = r"\b\w+\b"   # pattern匹配单词边界开头的一个或多个单词结尾的边界

match_object = re.search(pattern, text)    # 使用search()方法在text中搜索pattern模式
if match_object:
    print("Match found:", match_object.group())     # 获取匹配到的文本信息
else:
    print("No Match found.")
```
输出结果：
```python
Match found: hello
```

### 2.2.2 替换操作
替换操作是指用另一个字符串替换掉指定模式的操作。在Python中，使用`sub()`方法完成字符串替换。该方法的第一个参数是搜索模式，第二个参数是替换字符串，第三个参数是被替换的字符串。

例子：
```python
import re

text = "The quick brown fox jumps over the lazy dog."
pattern = r"\b[a-z]+\b"
replacement = "[CENSORED]"

new_text = re.sub(pattern, replacement, text)
print(new_text)      # [CENSORED] [CENSORED] [CENSORED] [CENSORED].
```

### 2.2.3 分组
正则表达式可以创建并返回一个组。组可以提取匹配到的文本，并在其他地方重新引用。例如，下面是创建了一个名为"numbers"的组：
```python
import re

text = "the price of apple is $10 and the weight of orange is 50kg."
pattern = r"(apple|orange)\s(\d{2})([\.\d]*)kg"

match_object = re.search(pattern, text)
if match_object:
    fruit = match_object.group(1)        # 获取组1匹配到的文本信息，即果树名称
    quantity = int(match_object.group(2))  # 将组2匹配到的文本信息转换为整数，即数量
    unit = match_object.group(3)           # 获取组3匹配到的文本信息，即单位

    if quantity > 100 or (quantity == 100 and unit!= ""):
        print(fruit, quantity, unit + " are too expensive!")
    else:
        print(fruit, quantity, unit + " are affordable prices.")
else:
    print("No Match found.")
```

### 2.2.4 定位点(Anchor)
定位点用来控制正则表达式搜索文本时的起始位置和结束位置。其中，`^`表示行首，`$`表示行尾，`\A`表示整个文本的开始，`\Z`表示整个文本的结束，`\b`表示单词边界，`\B`表示非单词边界。另外，可以在正则表达式中使用圆括号对定位点进行命名，这样就可以方便地引用。

例子：
```python
import re

text = """This is a sample sentence with some numbers
12345 and letters like A B C D E F G H I J K L M N O P Q R S T U V W X Y Z."""
pattern = r"\b(?P<num>\d+)\b"          # 创建一个组"num"，用来匹配数字

matches = re.finditer(pattern, text)   # 使用finditer()方法搜索所有匹配项

for match in matches:                  # 遍历所有匹配项
    num = int(match.groupdict()["num"])   # 获取组"num"匹配到的文本信息并转换为整数
    print("Number", num, "found at position", match.start(), "-", match.end()-1)
```

### 2.2.5 原生字符串(Raw String)
原始字符串（Raw String）用`r`标识，它使得字符串中的每个反斜杠(`\`)都被视为普通字符而不是转义字符。因此，如果模式中包含这种转义序列，则需要使用原始字符串。由于原始字符串的存在，所以正则表达式可以跨越多行，并且可以防止在字符串中嵌入`\\`。

例子：
```python
import re

text = """This is a \
multiline string."""
pattern = r"This.*string"

print(re.findall(pattern, text)[0])       # This is a multiline string.
```

## 2.3 模式语法
### 2.3.1 单字符元字符
单字符元字符是指只包含一个字符的元字符，包括`.`、`\d`、`[\d]`、`[^x]`等。常用单字符元字符的含义如下：

`.`：匹配除换行符之外的任意单个字符。
`\d`：匹配数字，相当于`[0-9]`。
`\D`：匹配非数字字符，相当于`[^0-9]`。
`\s`：匹配任意空白字符，包括空格、制表符、换页符等。
`\S`：匹配任意非空白字符。
`\w`：匹配任意单词字符，即字母、数字和下划线。
`\W`：匹配任意非单词字符。
`[]`：匹配括号内的任何字符，如`[abcde]`可以匹配`a`, `b`, `c`, `d`, 或 `e`。

### 2.3.2 组合模式
组合模式是指由简单模式组合而成的模式。常用组合模式的形式如下：

- `.`：匹配任意单个字符，除了换行符。
- `\d`：匹配任意数字。
- `\D`：匹配任意非数字字符。
- `\s`：匹配任意空白字符，包括空格、制表符、换页符等。
- `\S`：匹配任意非空白字符。
- `\w`：匹配任意单词字符，即字母、数字和下划线。
- `\W`：匹配任意非单词字符。
- `[]`：匹配括号内的任何字符，如`[abcde]`可以匹配`a`, `b`, `c`, `d`, 或 `e`。
- `|`：或运算符，表示两个或多个匹配项，如`[abc]|[def]`可以匹配`a`, `b`, `c`，或`d`, `e`, `f`。
- `*`：匹配零个或多个前面的模式，如`\d*`:可以匹配空字符串、连续数字、或其它字符。
- `+`：匹配一个或多个前面的模式，如`\d+`:只能匹配至少一个连续数字。
- `{n}`：匹配n个前面的模式，如`\d{5}`:只能匹配5个连续数字。
- `{m,n}`：匹配m到n个前面的模式，如`\d{3,5}`:只能匹配3到5个连续数字。
- `?`：匹配零个或一个前面的模式，如`\w?`:可以匹配单个字母、或空字符串。
- `()`：创建并返回一个组，括号内的模式可以作为整体参与匹配，并可提取匹配到的文本。
- `^`：匹配行首，如`^\d`:只能匹配文本开头的数字。
- `$`：匹配行尾，如`\d$`:只能匹配文本结尾的数字。
- `\b`：匹配单词边界，如`\be`只能匹配以`e`结尾的单词。
- `\B`：匹配非单词边界，如`\Be`可以匹配以`e`开头的非单词字符串。
- `(?:)`：匹配括号内的模式，不创建组，但不影响匹配。
- `\num`或`\g<name>`：引用之前的或命名的组，如`\1`:引用第一个组的匹配结果。

### 2.3.3 限定符
限定符用来控制前面模式的匹配次数。限定符包括`?`、`*`、`+`、`{m,n}`，分别表示懒惰匹配、零次或多次匹配、一次或多次匹配、固定次数匹配。若不出现限定符，则默认是贪婪匹配，即尽可能匹配更多的字符。若出现了`*`/`+`/`?`限定符，则一般都用括号将模式包起来，否则容易造成误解。

例子：
```python
import re

text = "aaa11bbb22ccc33ddd"
pattern = r"\b\w{3}\d{3}(?=\w)"         # (?=pattern) positive lookahead assertion
matches = re.findall(pattern, text)
print(matches)                           # ['aaa', 'ccc'] 

pattern = r"\b\w{3}(\d{3})?\b"          # \d{3}? optional group
matches = re.findall(pattern, text)
print(matches)                           # ['aaa', None, 'ccc', 'ddd'] 
```

## 2.4 匹配方法
### 2.4.1 search()方法
`search()`方法是最基本的方法，它接受两个参数：模式字符串和待匹配的字符串，搜索字符串中的第一个匹配项。如果找到匹配项，则返回一个Match对象，否则返回None。

例子：
```python
import re

text = "hello world"
pattern = r"\bworld\b"

match_object = re.search(pattern, text)
if match_object:
    print("Match found:", match_object.group())
else:
    print("No Match found.")
```
输出结果：
```python
Match found: world
```

### 2.4.2 findall()方法
`findall()`方法搜索字符串中的所有匹配项，并返回一个列表。

例子：
```python
import re

text = "The quick brown fox jumps over the lazy dog."
pattern = r"[aeiouAEIOU]"

matches = re.findall(pattern, text)
print(matches)                   # ['t', 'o', 'i', 'e', 'u', 'l', 'o', 't', 'h', 'r', 'y', 'd']
```

### 2.4.3 sub()方法
`sub()`方法替换字符串中的匹配项。

例子：
```python
import re

text = "The quick brown fox jumps over the lazy dog."
pattern = r"[aeiouAEIOU]"
replacement = "*"

new_text = re.sub(pattern, replacement, text)
print(new_text)                 # Th*ck brwn fx jmps v*s th lzy dg.
```

### 2.4.4 split()方法
`split()`方法按照模式将字符串分割成列表。

例子：
```python
import re

text = "this is a test"
pattern = r"\s+"

words = re.split(pattern, text)
print(words)                    # ['this', 'is', 'a', 'test']
```