
作者：禅与计算机程序设计艺术                    

# 1.简介
         

在Python中进行字符串匹配时，最常用的函数莫过于`re`(regular expression)模块中的`findall()`方法了。本文将详细介绍该方法的用法、原理及应用场景。
## 什么是正则表达式？
正则表达式（Regular Expression）是一种文本模式匹配的工具，它可以用来方便地检查一个字符串是否与某种模式匹配。它是一个强大的文本搜索和替换工具，是高级编程语言中字符串处理的一项重要功能。它的语法灵活多变，涵盖了方方面面，因此几乎可以匹配任何需要的模式。相比其他匹配方式，如精确匹配、大小写不敏感等，正则表达式更加灵活、全面，并适用于各种各样的场景。
## 如何使用正则表达式？
在Python中，正则表达式提供了三个方法用于处理字符串匹配：

1.`re.search()`: 查找字符串的起始位置，如果找到了一个匹配的子串就返回一个Match对象，否则返回None；

2.`re.match()`: 只从字符串的开头开始匹配，匹配成功后立即结束查找，返回一个Match对象，否则返回None；

3.`re.findall()`: 在整个字符串中找到所有匹配的子串，并以列表形式返回；

这里重点介绍一下`re.findall()`方法。
### `re.findall(pattern, string[, flags])`
该方法在字符串中搜索所有的匹配正则表达式的字符串。它返回一个列表，包括所有匹配到的子串。
```python
import re
string = "The quick brown fox jumps over the lazy dog."
result = re.findall("the", string)
print(result)   # ['the']
```
上例中，正则表达式是"the"，匹配到了字符串中首次出现的"the"。因此输出结果是['the']。

如果你想同时匹配到多个子串，只需在正则表达式中加入更多的词汇就可以了。比如说：
```python
import re
string = "The quick brown fox jumps over the lazy dog and eats a fat rat on the mat."
result = re.findall("the|fox|rat", string)
print(result)    # ['the', 'fox', 'rat']
```
这里的正则表达式匹配到了"the"、"fox"和"rat"这几个关键字，并将它们都作为单独的元素添加到了列表里。

## 使用案例
### 检测用户名是否合规
用户注册系统一般会要求输入用户名，用户名必须满足特定规则。比如说，用户名只能由字母数字下划线组成，并且长度在6-16之间。检测用户名是否符合规则可以使用正则表达式。
```python
import re
username = "abc_123_"
if not re.match("^[a-zA-Z0-9_]{6,16}$", username):
print("Invalid username!")
else:
print("Valid username.")
```
上面这个例子中，正则表达式是"^[a-zA-Z0-9_]{6,16}$"，它匹配了用户名必须包含至少6个字符、不能超过16个字符、只能包含字母数字或下划线。如果用户名不符合规则，那么程序就会打印"Invalid username!"，否则打印"Valid username."。

### 网页文本分析
你经常会遇到要分析网页上的文字信息，而这些文字信息往往会包含一些特定的关键词或者相关数据。通过正则表达式可以快速定位到想要的数据。比如说，假设你需要分析新浪首页上的股票价格数据，可以这样做：
```python
import requests
from bs4 import BeautifulSoup
import re
url = "http://finance.sina.com.cn/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
text = soup.get_text()
stock_prices = re.findall("\d+\.\d+", text)
for price in stock_prices:
print(price)
```
首先，程序向新浪财经首页发送请求，获取页面源码。然后解析HTML代码，获得网页中的全部文本。最后，使用正则表达式查找出所有匹配的数字，并输出到控制台。得到的股票价格结果如下所示：
```
7015.13
7039.62
7009.85
7012.02
...
```