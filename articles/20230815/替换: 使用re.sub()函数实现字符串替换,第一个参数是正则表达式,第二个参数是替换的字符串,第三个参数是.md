
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在正则表达式中有一个重要的替换命令就是`re.sub()`函数。这个函数的作用就是通过正则表达式匹配到的字符串进行替换。它的原型如下：
```python
re.sub(pattern, repl, string, count=0, flags=0)
```
其中，`pattern` 是正则表达式，用于匹配文本中的子串；`repl` 是替换文本或函数，它可以是一个字符串也可以是一个函数；`string` 是要被处理的原始字符串；`count` 是可选参数，指定最多替换的次数，默认值为零表示所有匹配都将被替换；`flags` 为可选参数，它可以指定模式匹配的选项，比如忽略大小写、多行模式等。

本文主要就这个函数的用法做一个简单的介绍和讲解。关于正则表达式的详细语法讲解，可以参考我的另一篇文章《Python中的正则表达式实战指南》。

# 2.1 场景举例
## 2.1.1 简单替换
假如我们有一个文件名为"hello world.txt",希望把其中的空格去掉，只保留文件名。那么可以这样做：
```python
import re
filename = "hello world.txt"
new_name = re.sub(" ", "", filename)
print(new_name) # helloworld.txt
```
## 2.1.2 复杂替换
假设我们有一个日志文件，里面记录了很多错误信息。我们想把一些敏感词汇替换成**口令**。我们可以先定义好字典：
```python
sensitive_dict = {"username": "mypassword"}
```
然后再读取日志文件，并对里面的敏感词汇进行替换：
```python
with open('error.log', 'r') as f:
    log_content = f.read()
    
for key in sensitive_dict:
    pattern = r'\b' + key + r'\b' # 加\b防止替换不准确
    new_value = sensitive_dict[key] * len(re.findall(pattern, log_content)) # 替换多个相同字符
    
    log_content = re.sub(pattern, new_value, log_content)
    
print(log_content)
```
输出结果类似：
```
2021-07-01 12:00:00 ERROR username not found
2021-07-01 12:00:00 WARNING user password error
2021-07-01 12:00:00 INFO login failed for the third time!
```
这里用到了 `\b` 来精确匹配单词边界，因为上述例子中，关键字 `"user"` 和 `"not"` 在原文中都是连续出现的，所以需要精确匹配。另外，还用到了 `len()` 函数计算替换后的字符个数，使得替换的结果与匹配到的敏感词数量一致。

# 2.2 特殊字符及转义
## 2.2.1 点号（.）
`.` 可以匹配任意单个字符，但是使用 `\.` 可以匹配 `.` 本身：
```python
>>> import re
>>> s = "abc.def"
>>> re.match(".", s).group()
'a'
>>> re.match("\.", s).group()
'.'
```
## 2.2.2 反斜杠（\）
`\` 可以用来转义某些特殊字符，包括 `. ^ $ * +? { } [ ] \ | ( )`。例如，`\*` 可以匹配星号：
```python
>>> s = "foo*bar"
>>> re.match("\\*", s).group()
'*bar'
```
## 2.2.3 限定符（* +? { }）
限定符可以让正则表达式匹配的数据数量更少或者更多。比如，`+` 可以匹配一个或多个数据：
```python
>>> s = "aaabbbccc"
>>> re.findall("[a-z]+", s)
['aaa', 'bbb', 'ccc']
```
而 `{n}` 可以匹配 `n` 个数据：
```python
>>> s = "hello world"
>>> re.search("^h{3}", s).group()
'hel'
```
注意：在 Python 的 `re` 模块中，默认情况下，`^` 和 `$` 锚定整个字符串的开头和结尾，而不是单独匹配每一行的开头和结尾。如果想要使用这些锚点匹配每一行的开头和结尾，可以设置 `re.MULTILINE` 标志。