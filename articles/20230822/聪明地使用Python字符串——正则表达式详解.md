
作者：禅与计算机程序设计艺术                    

# 1.简介
  

正则表达式（regular expression）是一种用于匹配字符串的模式的特殊字符序列。它提供了对文本进行复杂搜索、替换的强大功能。它的语法灵活易用，经过几十年的发展已成为电脑领域中最常用的字符串处理工具之一。本文将主要介绍Python中的正则表达式，并结合实际应用场景，分享其使用技巧和注意事项。

# 2.为什么要用正则表达式？
使用正则表达式可以对文本进行快速高效地处理，如检索、过滤、替换等。下面就举几个实际场景来说明为什么需要使用正则表达式。

1. 检索或过滤特定信息
在爬虫、数据分析等领域，需要从大量文本中筛选出所需的信息，正则表达式就可以帮助我们非常方便地完成这一工作。比如，我们想从HTML文档中提取特定网页的内容时，可以使用正则表达式查找某个标签的开始位置和结束位置，然后根据这些位置从原始HTML文本中提取出对应的内容。

2. 数据清洗
由于各种原因，很多文本文件中会存在不规范、错误的数据，正则表达式就可以帮助我们快速有效地清理这些数据。例如，一些网站发布的新闻稿往往有脏数据，包括广告、特殊符号等，这些数据可以通过正则表达式过滤掉，避免影响后续分析结果。

3. 数据格式转换
有时候，我们需要把非结构化的文本数据转成结构化的格式，如XML、JSON等。而转换过程中需要进行各种规则的验证和清洗，正则表达式就显得尤为重要。

4. 模板匹配和替换
模板匹配和替换是许多文本编辑器的基础功能。当用户保存一个文档的时候，编辑器通常会自动帮我们匹配相关模板，并按照模板格式生成新的文档。正则表达式也可以用于模板匹配，但它比传统的模板匹配更加灵活、精准。

5. 更多……
正则表达式的使用范围不仅限于以上五个例子，它还可以用于其他各类场景，比如身份证校验、分词、字符串拆分、日志解析等等。

# 3. Python中的正则表达式
Python中，re模块提供对正则表达式的支持。re模块包含四个函数：`match()`、`search()`、`findall()` 和 `sub()` 。前三个函数分别用于匹配字符串的开头、中间或者末尾，查找是否有匹配的字符串；最后一个函数用于替换字符串中匹配到的子串。下面分别介绍这四个函数。


## re.match() 函数

`re.match(pattern, string)` 尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回None。 

```python
import re
  
line = "Cats are smarter than dogs"
  
matchObj = re.match(r'(.*)are (.*?).*', line, re.M|re.I)
  
if matchObj:
    print ("matchObj.group():", matchObj.group())
else:
    print ("No match!!") 
```
输出：

```bash
matchObj.group(): Cats are smarter than dogs
```
第一个参数 pattern 是匹配的正则表达式，第二个参数 string 是待匹配的字符串。第三个可选参数 re.M 表示多行模式，也就是说，^和$分别匹配字符串开始和结束的位置，而. 则匹配除了换行符 \n 以外的所有字符。re.I 表示忽略大小写。

## re.search() 函数

`re.search(pattern, string)` 搜索字符串string中的所有位置，找到第一个匹配的地方，并返回一个Match对象，如果没有找到匹配的对象，则返回 None。 

```python
import re
  
text = '''Hello world! this is a test text for matching regex in python.'''
  
searchObj = re.search(r'\b\w{6}\b', text)
  
if searchObj:
    print("searchObj.group():", searchObj.group())
else:
    print("Nothing found!!")  
```
输出：

```bash
searchObj.group(): Hello
```

searchObj.group() 方法返回被匹配的字符串。

## re.findall() 函数

`re.findall(pattern, string)` 在字符串中找到所有的匹配正则表达式的子串，并返回列表，如果没有找到匹配的对象，则返回空列表。 

```python
import re
  
text = 'The quick brown fox jumps over the lazy dog.'
  
listOfWords = re.findall(r'\w+', text)
  
print('List of words:', listOfWords)  
```
输出：

```bash
List of words: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
```

## re.sub() 函数

`re.sub(pattern, repl, string, count=0)` 把字符串中所有的匹配正则表达式的子串都替换成repl指定的字符串，如果指定了count，则替换不超过这个次数。 

```python
import re
  
text = 'The quick brown fox jumps over the lazy dog.'
  
newText = re.sub(r'\b\w{3}\b', r'', text)
  
print('New Text:', newText)  
    
newText = re.sub(r'\b\w{3}\b', r'\g<0> XYZ', text)
  
print('New Text with replacement:', newText)  
```
输出：

```bash
New Text: The  quick brown fox jumps over the lazy dog.
New Text with replacement: The XYZ quick brown XYZ fox jumps over the lazy XYZ.
```