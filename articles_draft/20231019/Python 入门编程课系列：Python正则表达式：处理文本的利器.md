
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


正则表达式（Regular Expression）简称 regex 是用于匹配字符串中符合某种模式(规则)的一类特殊符号。由于在不同的编程语言中实现的方式不同，因此在 Python 中应用正则表达式主要依赖于 re 模块。Python 的正则表达式语法参考 https://docs.python.org/zh-cn/3/library/re.html。

一般来说，正则表达式可以用来做以下几件事情：

1、检索匹配文本中的指定模式；
2、替换或删除文本中的特定模式；
3、验证文本是否满足指定的模式。

本文将向大家介绍 Python 中最常用的两种正则表达式模块——re 和 re2，以及如何进行高级功能扩展。


# 2.核心概念与联系
## 2.1 正则表达式概述
正则表达式通常被用来匹配、搜索和替换那些符合某些模式（规则）的字符串。正则表达式作为一种特殊的字符序列，定义了一个用来描述或者匹配一个或者多个符合某个句法规则的字符串。正则表达式提供了对文本分析、数据清洗、文本分割及处理的强大能力。

### 2.1.1 元字符（Metacharacters）
元字符就是指那些有特殊含义的字符。下面列举了一些常见的元字符：

| 字符 | 描述       | 示例                  |
| ---- | ---------- | --------------------- |
|.    | 匹配任意字符 | 'a.c' 匹配 'abc', 'axc'等   |
| \w   | 匹配单词字符 | '\w' 匹配所有英文字母、数字和下划线 |
| \W   | 匹配非单词字符 | '\W' 匹配除字母、数字、下划线之外的所有字符 |
| \s   | 匹配空白字符 | '\s' 匹配所有的空白字符（空格、制表符、换行符等） |
| \S   | 匹配非空白字符 | '\S' 匹配除空白字符以外的所有字符 |
| [abc] | 匹配方括号内的任何一个字符 | '[abc]' 匹配 'a' 或 'b' 或 'c' 中的任意一个字符 |
| [^abc] | 匹配不在方括号内的任何字符 | '[^abc]' 匹配除了 'a'、'b'、'c' 以外的所有字符 |
| *   | 匹配前面的字符零次或多次 | 'ab*' 匹配 'a' 后跟零个或多个 'b' |
| +   | 匹配前面的字符一次或多次 | 'ab+' 匹配 'a' 后跟至少一个 'b' |
|?   | 匹配前面的字符零次或一次 | 'ab?' 匹配 'a' 后跟零个或一个 'b' |
| {n}  | 匹配前面的字符恰好 n 次 | 'ab{2}' 匹配 'a' 后跟两个 'b' |
| {m,n}  | 匹配前面的字符 m~n 次 | 'ab{2,3}' 匹配 'a' 后跟两个到三个 'b' |
| ( )  | 分组操作符 | '(a\|b)' 可以匹配 'a' 或 'b' |
| ^   | 表示行首     | '^hello' 只匹配以 hello 为开头的行 |
| $   | 表示行尾     | 'world$' 只匹配以 world 为结尾的行 |

上表中，^ 和 $ 符号用来限制匹配的字符串的起始位置和结束位置。

### 2.1.2 可选标志
可选标志表示对正则表达式的一些特定功能进行打开或关闭。比如忽略大小写、多行匹配等。可选标志如下：

| 字符 | 描述           | 示例                                |
| ---- | -------------- | ----------------------------------- |
| i    | 不区分大小写模式 | 'AbC' 能匹配 'abc'                   |
| m    | 多行匹配模式     | '^hello' 能够匹配第一行的 hello      |
| s    | 包括换行符     | '.' 能够匹配所有的字符，包括换行符 |

比如，'pattern' 这个模式如果用 re.IGNORECASE 来修饰，则能匹配到 'Pattern' 或 'PATTERN'。如果用 re.MULTILINE 来修饰，则能匹配到每一行中的模式。

### 2.1.3 模式修正
模式修正指的是对正则表达式添加额外的结构和语义信息。如下表所示：

| 字符 | 描述         | 示例                          |
| ---- | ------------ | ----------------------------- |
| (?iLmsux)  | 在模式内部使用这些修饰符都可以开启相应的功能。比如，(?i) 表示开启不区分大小写模式。 | (?im)^\d+(\.\d+)?$ 匹配带小数点的数字 |
| (?P<name>pattern)  | 用名称来捕获子模式的匹配结果，可以在后续的匹配中引用该子模式的匹配结果。 | (?P<int>\d+)\\.(?P<frac>\d+) 匹配整数部分和小数部分并分别捕获 |
| (?:...)  | 这是一个非捕获括号，它将模式的内容视为非独立的组，不会对其结果进行计数。 | ((?:Monday\s)|(?:Tuesday\s)|(?:Wednesday\s)|(?:Thursday\s)|(?:Friday\s)) matches "Monday Tuesday Wednesday Thursday Friday" but not the date string itself |
| (?#...)  | 添加注释，注释的内容不会影响匹配结果。 | 

比如，'\d+(\.\d+)?' 这个模式可以匹配到带小数点的数字。'(?P<name>pattern)' 将匹配到的内容保存到命名的变量里面。'(?:...)' 这个语法告诉匹配引擎不要对括号里的内容进行计数。

### 2.1.4 预编译
re 模块支持编译后的正则表达式对象，可以使用 compile 方法进行预编译。通过预编译的正则表达式对象可以重复使用相同的模式，节省时间。

``` python
import re

pat = re.compile('\d+')
result = pat.findall('one two 3 four five six seven')
print(result) # ['3', '4', '5', '6', '7']
```

## 2.2 re 模块
Python 的 re 模块实现了 Perl 风格的正则表达式，提供了大量函数用于处理正则表达式。

### 2.2.1 findall()
findall() 函数用于从字符串中找到正则表达式所匹配的所有子串，并返回列表。

``` python
import re

string = 'The quick brown fox jumps over the lazy dog.'
pattern = r'\b[a-z]+\b'

matches = re.findall(pattern, string)
for match in matches:
    print(match)
```

输出结果：

```
quick
brown
fox
jumps
lazy
dog
```

这里的 pattern 是小写字母和下划线组合的一个单词。findall() 函数会查找 string 中的每个单词，并返回一个列表。

### 2.2.2 search()
search() 函数用于查找字符串中第一个匹配正则表达式的地方。

``` python
import re

string = 'The quick brown fox jumps over the lazy dog.'
pattern = r'\b[a-z]+\b'

match = re.search(pattern, string)
if match:
    print("Match found at index %d" % match.start())
    print(match.group())
else:
    print("No match found")
```

输出结果：

```
Match found at index 4
over
```

这里的 pattern 同样是小写字母和下划线组合的一个单词。search() 函数会查找 string 中的第一个单词，并返回一个 Match 对象，其中包含匹配的信息。

### 2.2.3 sub()
sub() 函数用于替换字符串中的匹配项。

``` python
import re

string = 'The quick brown fox jumps over the lazy dog.'
pattern = r'(the)\s([a-z]*)'
repl = r'\1 \2'

new_str = re.sub(pattern, repl, string, flags=re.IGNORECASE)
print(new_str)
```

输出结果：

```
The quick brown fox jumps over the lazy dog.
```

这里的 pattern 查找字符串中的'the'后面跟着零个或多个小写字母或下划线的组合。替换字符串中的'the'替换成'\1 \2'，将所有匹配的单词之间加上空格。re.IGNORECASE 表示忽略大小写模式。