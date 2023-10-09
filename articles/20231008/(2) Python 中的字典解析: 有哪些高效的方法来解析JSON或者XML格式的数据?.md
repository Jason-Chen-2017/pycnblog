
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


在Python中，能够直接处理复杂的数据结构的模块主要有三种：列表、元组和字典。字典是最常用的一种数据类型，因为它可以让我们通过键值对的方式存储和访问任意数量的信息。而有时候我们的数据往往存在于一些外部的格式文件，比如json或者xml等。如何从这些外部数据源中提取出字典数据并处理其中的信息呢？接下来，我们将了解两种不同场景下的字典解析方法。
首先，我们来看一下什么是JSON（JavaScript Object Notation）和XML（Extensible Markup Language），它们都是用于存储和传输数据的格式。但是JSON和XML不完全相同，所以我们需要清楚它们的区别。
## JSON（JavaScript Object Notation）
JSON是JavaScript的一个子集，用于描述对象的属性和值，基于ECMAScript标准。它是轻量级的数据交换格式，易于人阅读和编写，同时也易于机器解析和生成。例如：
```
{
  "name": "John Smith",
  "age": 30,
  "city": "New York"
}
```
在上面的示例中，`name`，`age`和`city`称之为键，对应的值则是字符串“John Smith”、数字30和字符串“New York”。
JSON的语法很简单，包括对象{}、数组[]、字符串""、数值0-9、布尔值true/false、null等。当我们用Python处理JSON时，只需要导入json模块就可以了。下面是一个简单的例子：
```
import json

data = '{"name":"John Smith","age":30,"city":"New York"}' #假设这是从网络或文件读取到的JSON数据
parsed_data = json.loads(data)    #将JSON格式转换为Python对象
print(type(parsed_data))         #打印数据类型
print(parsed_data["name"])       #打印名字
```
上面这段代码会输出：<class 'dict'> <NAME>，即JSON数据被成功解析并转换为字典形式。
## XML（Extensible Markup Language）
XML（Extensible Markup Language）由W3C组织推荐的一种数据编码语言，用于标记电子文档中各个元素的内容及其相关属性。它的目的是定义一套简单的、较为抽象的、用来编码各种结构化文档的规则。以下是一个示例XML文档：
```
<library>
    <book category="history">
        <title lang="en">The Hitchhiker's Guide to the Galaxy</title>
        <author><NAME></author>
        <year>1979</year>
    </book>
    <book category="fiction">
        <title lang="en">To Kill a Mockingbird</title>
        <author>Harper Lee</author>
        <year>1960</year>
    </book>
</library>
```
如上所示，XML中的每个元素都有一个开头标签和一个闭合标签，中间可容纳文本或其他元素。这里`<library>`是根元素，里面嵌套着两个`<book>`元素，分别代表两本图书。每本书都有三个属性，`<category>`、`lang`和`<year>`，分别代表图书类别、语言和出版年份。
相比JSON来说，XML更适合用于表示复杂的、层次化的数据结构，而且更适合处理数据间的关系。但是，XML的语法比较复杂，如果需要手动编写XML，那么编写起来就比较麻烦。因此，Python提供了第三方库ElementTree来简化XML的解析工作。下面的代码展示了一个例子：
```
from xml.etree import ElementTree as ET

tree = ET.parse('books.xml')      #解析XML文档
root = tree.getroot()             #获取根节点
for book in root.findall("book"):   #遍历所有图书
    title = book.find("title").text   #获取图书名
    author = book.find("author").text #获取作者名
    year = book.find("year").text     #获取出版年份
    print("{} by {} ({})".format(title, author, year))  #打印信息
```
这段代码会输出：
```
The Hitchhiker's Guide to the Galaxy by Douglas Adams (1979)
To Kill a Mockingbird by Harper Lee (1960)
```