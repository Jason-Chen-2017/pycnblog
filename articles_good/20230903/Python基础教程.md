
作者：禅与计算机程序设计艺术                    

# 1.简介
  


Python 是一种易于学习的高级编程语言，具有强大的功能和丰富的库支持。它广泛用于数据科学、Web开发、自动化运维、机器学习、云计算、移动应用等领域。Python 的语法简单灵活，独特的设计理念、优美的编码风格和广泛的第三方库使其成为一种非常适合解决各类问题的语言。

本文将介绍 Python 基础知识、语法特性及数据结构等。希望能帮助读者了解 Python 的基础知识、技巧、用法，进而提升自己的编程能力。

## 2.安装配置

安装配置Python环境主要包括三个步骤：

1. 安装Python

2. 配置Python环境变量
安装完毕后，需要把 Python 添加到系统环境变量中，这样才能方便地运行 Python 命令行或编写脚本调用。具体方法如下：

+ 查找Python安装目录
打开命令提示符（cmd），输入以下命令查看Python安装目录：

    ```
    where python
    ```
例如我的电脑上的安装路径为：

```
C:\Users\Administrator\AppData\Local\Programs\Python\Python39-32
```

+ 将Python安装目录添加到系统环境变量PATH中
双击“计算机”-->“属性”-->“高级系统设置”-->“环境变量”-->“PATH”-->编辑。添加以下内容：

```
C:\Users\Administrator\AppData\Local\Programs\Python\Python39-32;C:\Users\Administrator\AppData\Local\Programs\Python\Python39-32\Scripts
```

注意：如果遇到权限问题，请修改文件/文件夹的访问权限或者重新启动计算机。

3. 测试Python是否成功安装
打开命令提示符（cmd）或终端（Terminal）输入以下命令测试是否安装成功：

```
python --version
```

如出现类似如下信息则表示安装成功：

```
Python 3.9.5
```

如果出现错误信息，请仔细检查之前的所有操作是否都正确执行。

## 3.第一个Python程序

编写第一个Python程序主要包括四个步骤：

1. 创建一个新文件
使用记事本（notepad）创建名为hello.py的文件，并在编辑器中写入以下内容：

```python
print("Hello World!")
```

2. 使用IDLE或其他Python IDE编辑该文件
IDLE是一个集成开发环境（Integrated Development Environment，IDE）软件，可以用来编写、运行和调试Python程序。推荐使用IDLE作为Python程序的编辑工具，它的图形界面直观易懂，功能强大。当然，您也可以选择其他Python IDE或集成开发环境。

打开IDLE，点击菜单栏中的文件-打开，找到并打开刚才创建好的hello.py文件。

3. 在IDLE窗口中运行程序
在IDLE窗口中选中该文件，然后按F5键或点击工具栏的运行按钮（▶️按钮）即可运行该程序，控制台输出“Hello World!”。

4. 修改程序
如果要修改程序，只需再次编辑hello.py文件的内容并保存，IDLE会自动检测到文件的变化并运行程序。

至此，您已经掌握了最基本的Python程序编写和运行方法。欢迎继续阅读下面的内容。

## 4.Python基本语法

Python 是一种动态的、面向对象、解释型的编程语言。它具有简洁、紧凑的语法和结构，能够有效地处理大量的数据。

### 4.1 注释

Python 中单行注释以“#”开头，多行注释可以用三引号"""括起来。

```python
# This is a single line comment.
'''This is a 
multiline comment.'''
```

### 4.2 保留字

下表列出了 Python 中的保留字（关键字）。这些关键字不能用作标识符名称，因为它们已经被 Python 用作内部命令或语法结构。
|and	|as |assert	|break	|class|
|---|---|---|---|---|
|continue	|def	|del	|elif	|else|
|except	|exec	|finally	|for	|from|
|global	|if	|import	|in	|is|
|lambda	|nonlocal	|not	|or	|pass|
|raise	|return	|try	|while	|with|
|yield|

一般来说，在命名变量时尽量避免使用关键字，因为容易造成误解。

### 4.3 数据类型

Python 支持的基础数据类型有：整数、长整型、浮点数、复数、布尔值、字符串、列表、元组、集合、字典。其中，整数和长整型可以使用十进制或八进制表示，也可以使用二进制表示（前缀 0b 或 0B 表示）。浮点数可以使用小数点表示，也可以使用指数表达式表示；复数由实部和虚部构成，可用 complex() 函数构造。布尔值只有两种值 True 和 False；字符串是字符序列，可用单引号 '' 或双引号 "" 括起来的任意文本；列表是可以变更的元素序列，可用 [] 括起来的零个或多个元素，可通过索引获取对应位置的值，列表支持切片操作；元组是不可变元素序列，可用 () 括起来的零个或多个元素，元组也是不可变的，无法修改元素的值；集合是一个无序不重复元素集，可用 set() 函数或 {} 括起来的零个或多个元素创建；字典是键值对的无序映射表，可用 {} 括起来的零个或多个键值对表示，每个键必须唯一，值可以取任何数据类型。

除基础数据类型外，还有 None（空值）、自定义数据类型、函数、模块、类等。

### 4.4 操作符

Python 有丰富的运算符用于各种数据类型的运算。下表列出了 Python 中的运算符及其优先级（从高到低）：

|运算符|描述|
|---|---|
|**|指数|
|~ + - abs|一元算术运算符|
|* / // % divmod|乘、除、取整除和余数|
|+ -|加、减|
|< <= > >=|比较运算符|
|= += -= *= /= **=|赋值运算符|
|,|用于连接或分隔表达式|
|in|成员运算符|
|not in|非成员运算符|
|is|身份运算符|
|is not|非身份运算符|
|( )|用于改变运算顺序|
|[ ]|用于访问元素|
|.|用于访问对象的属性或方法|
|and|短路逻辑 AND|
|or|短路逻辑 OR|
|not|逻辑 NOT|

另外，Python 提供了条件语句 if...else、循环语句 for 和 while、跳转语句 break、continue、pass 和返回语句 return。

### 4.5 输入输出

Python 提供了 input() 函数用于接受用户输入，它返回一个字符串。print() 函数用于打印输出结果，它可以接受多个参数，并以空格分隔输出。

### 4.6 字符串

字符串是 Python 中最常用的基本数据类型。你可以定义一个字符串，比如 "Hello, world!" ，也可以使用 "" 或 '' 括起来的任何文本。字符串是不可变的，因此不能修改它的内容，只能重新分配给另一个变量。

字符串可以使用索引和切片操作访问它的元素。索引以 0 开始，-1 为最后一个元素，步长默认为 1 。

```python
s = 'Hello, world!'
print(s[0])   # H
print(s[-1])  #!
print(s[:5])  # Hello
print(s[::2]) # Hloowrd
```

Python 中提供的内置函数 len() 返回字符串的长度，upper() 返回大写的字符串，lower() 返回小写的字符串，startswith() 判断字符串是否以某个子串开头，endswith() 判断字符串是否以某个子串结尾。

```python
s = 'Hello, world!'
print(len(s))       # 13
print(s.upper())    # HELLO, WORLD!
print(s.lower())    # hello, world!
print(s.startswith('H'))     # True
print(s.startswith('W'))     # False
print(s.endswith('world!'))  # True
print(s.endswith('python'))  # False
```

字符串可以使用 join() 方法进行拼接：

```python
words = ['apple', 'banana', 'orange']
result = ', '.join(words)
print(result)      # apple, banana, orange
```

还可以使用 split() 方法分割字符串：

```python
s = 'hello world'
words = s.split()
print(words)          # ['hello', 'world']
```

### 4.7 列表

列表是 Python 中一种有序的集合数据类型，可以存储不同类型的数据。列表是可以变更的元素序列，可以用 [ ] 括起来的逗号分隔的元素来初始化列表。列表支持切片操作。

```python
numbers = [1, 2, 3]
letters = ['a', 'b', 'c']
mixed_list = [1, 'two', True]
nested_list = [[1, 2], [3, 4]]

# 通过索引访问元素
print(numbers[0])        # 1
print(letters[2])        # c

# 对列表进行切片
print(numbers[:2])       # [1, 2]
print(letters[::-1])     # ['c', 'b', 'a']

# 获取元素个数
print(len(numbers))      # 3
```

列表可以使用 append() 方法追加元素，insert() 方法插入元素：

```python
numbers = [1, 2, 3]
numbers.append(4)         # [1, 2, 3, 4]
numbers.insert(1, 1.5)    # [1, 1.5, 2, 3, 4]
```

列表可以使用 remove() 方法删除元素，pop() 方法删除指定位置的元素：

```python
numbers = [1, 2, 1.5, 3, 4]
numbers.remove(2)             # [1, 1.5, 3, 4]
numbers.pop(-1)               # 4
print(numbers.index(1.5))     # 1
```

列表可以使用 sort() 方法排序：

```python
numbers = [4, 2, 3, 1, 1.5]
numbers.sort()              # [1, 1.5, 2, 3, 4]
```

### 4.8 元组

元组与列表类似，不同之处在于元组是不可变的，即一旦创建就不能更改。元组通常用 ( ) 来表示。

```python
coordinates = (3, 4)
x, y = coordinates
print(x, y)     # 3 4
```

### 4.9 集合

集合是一个无序不重复元素集，可以通过 set() 函数或 { } 来创建。集合支持以下操作：add()、clear()、copy()、difference()、intersection()、union()、update()。

```python
fruits = {'apple', 'banana', 'orange'}
vegetables = {'carrot', 'broccoli','spinach'}

# 添加元素
fruits.add('grape')
print(fruits)           # {'orange', 'banana', 'apple', 'grape'}

# 删除元素
fruits.discard('orange')
print(fruits)           # {'banana', 'apple', 'grape'}

# 清空集合
fruits.clear()
print(fruits)           # set()

# 更新集合
fruits.update(vegetables)
print(fruits)           # {'carrot', 'broccoli','spinach', 'banana', 'apple'}

# 两个集合的差集
print(fruits.difference(vegetables))   # {'banana', 'apple'}

# 两个集合的交集
print(fruits.intersection(vegetables))   # {'carrot', 'broccoli','spinach'}

# 两个集合的并集
print(fruits.union(vegetables))   # {'carrot', 'broccoli','spinach', 'banana', 'apple'}
```

### 4.10 字典

字典是 Python 中另一种常用的数据类型，它是键-值对（key-value pair）的集合，字典用 { } 来表示。字典的每一项由 key-value 对组成，由冒号 : 分隔。字典中所有的 key 必须是唯一的。

```python
person = {
    'name': 'Alice',
    'age': 25,
    'city': 'Beijing'
}

print(person['name'])     # Alice
print(person.keys())      # dict_keys(['name', 'age', 'city'])
print(person.values())    # dict_values(['Alice', 25, 'Beijing'])
```

字典可以使用 get() 方法获取元素值，pop() 方法删除元素：

```python
person = {
    'name': 'Alice',
    'age': 25,
    'city': 'Beijing'
}

print(person.get('name'))    # Alice
person.pop('city')            # Beijing
```

字典也支持 items() 方法获取所有键值对：

```python
person = {
    'name': 'Alice',
    'age': 25,
    'city': 'Beijing'
}

print(person.items())    # dict_items([('name', 'Alice'), ('age', 25), ('city', 'Beijing')])
```