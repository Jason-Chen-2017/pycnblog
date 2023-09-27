
作者：禅与计算机程序设计艺术                    

# 1.简介
  

“TypeError: 'NoneType' object is not iterable” 是Python中经常出现的一种错误信息，一般发生在调用函数或者方法时传入的参数不符合要求导致，例如字符串类型要求传入列表等容器类型参数。这种错误很难定位问题所在，本文将从背景知识、概念、术语、算法原理及操作步骤等方面，详细剖析这个错误及其原因，并给出具体方案解决方法。文章末尾还会附上一些常见的问题和解答，对大家的学习和提高帮助非常大！

# 2.背景介绍
## 2.1 什么是Python？
Python是一种开源的、跨平台的、可用的、动态类型的脚本语言。它被设计用于编写系统软件、Web应用程序、科学计算、网络爬虫、机器学习等各类程序。它的语法简单易懂，允许用户创建功能强大的程序。许多流行的第三方库都基于Python开发。

## 2.2 为什么要学习Python？
Python拥有丰富的应用领域，如：

- Web开发（Django、Flask）
- 数据分析和处理（Numpy、Scipy、Pandas）
- 机器学习（TensorFlow、Keras、Scikit-learn）
- 图形绘制和数据可视化（Matplotlib、Seaborn、Plotly）
- 网络爬虫（BeautifulSoup、Requests）
- 桌面GUI编程（Tkinter、PyQt）
- 游戏编程（Pygame）
- 命令行接口工具（Click）
-...

## 2.3 Python安装配置
### 2.3.1 安装Python
### 2.3.2 配置环境变量
如果安装好了Python后，在命令行中运行python，如果没有报错则表示安装成功。此时可以使用命令 `pip install` 来安装一些第三方模块。为了更方便地管理这些模块，可以设置环境变量。
打开注册表编辑器(Win + R -> regedit)，找到路径 HKEY_CURRENT_USER\Environment ，双击编辑，添加新的系统变量 PYTHONHOME，值为 Python 的安装目录，在PATH环境变量末尾追加 `%PYTHONHOME%\Scripts` 。这时重新打开命令行，输入 `pip install XXXXX`，就可以安装指定模块了。

至此，Python环境就配置完毕了。接下来就可以开始我们的第一次程序编写之旅了！

# 3.基本概念术语说明
## 3.1 变量和数据类型
计算机程序中的变量是程序运行过程中变化的数据值。变量的值可以随着程序的执行而改变，因此可以在程序的不同部分之间传递变量的值。在Python中，变量的名称通常用小写字母加下划线命名，且只能包含字母数字或下划线字符。变量在赋值时必须赋予一个有效的值，不能赋值空值或其他类似None这样的特殊值。

Python支持以下几种数据类型：

- Numbers（数字）
    - int（整数）
    - float（浮点数）
    - complex（复数）
- String（字符串）
- List（列表）
- Tuple（元组）
- Set（集合）
- Dictionary（字典）

## 3.2 流程控制语句
流程控制语句是指按照特定顺序执行指令的语句。在Python中，包括以下几类流程控制语句：

- if/elif/else（条件判断语句）
- for循环（遍历序列）
- while循环（重复执行语句块）
- break（跳出当前循环）
- continue（跳过当前次迭代）

## 3.3 函数和模块
函数是程序执行的基本单位，每个函数都有自己独立的作用范围和局部变量空间。函数通过返回值的方式输出结果，如果函数执行过程引起了异常，也可以捕获异常并进行相应处理。函数通过定义函数的名称、形式参数、返回值和实现体来创建。

模块是程序的基本构成单元，是封装数据的集合，可以通过导入模块的方式使用其中的函数和变量。模块可以理解为一个文件，里面包含多个函数、类、全局变量等。

# 4.核心算法原理及操作步骤
## 4.1 list()函数和tuple()函数
list()函数用来将元组、字符串转换成列表；tuple()函数用来将列表、字符串转换成元组。

``` python
>>> tuple([1, 2, 3])   # 将列表转换成元组
(1, 2, 3)

>>> tuple('hello')    # 将字符串转换成元组
('h', 'e', 'l', 'l', 'o')

>>> list(('a', 'b'))   # 将元组转换成列表
['a', 'b']

>>> list("world")     # 将字符串转换成列表
['w', 'o', 'r', 'l', 'd']
```

当将一个可迭代对象（如列表、元组、字符串）转换成列表或元组时，原对象的元素会被拷贝到新对象中，不会影响原对象。

## 4.2 range()函数
range()函数用来生成一系列连续整数，可用来构造for循环的迭代对象。range()函数的一般形式如下：

``` python
range(stop)        # 生成从0开始到stop-1的整数序列
range(start, stop[, step])   # 从start开始，每step个取一个整数，生成直到stop但不包括stop的整数序列
```

## 4.3 iter()函数
iter()函数用来获取一个可迭代对象（如列表、元组、字符串）的迭代器对象。

``` python
>>> it = iter(['apple', 'banana', 'orange'])
>>> next(it)          # 返回第一个元素
'apple'
>>> next(it)          # 返回第二个元素
'banana'
>>> next(it)          # 返回第三个元素
'orange'
>>> next(it)          # 抛出StopIteration异常，因为迭代结束
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

## 4.4 next()函数
next()函数用来获取迭代器的下一个元素。调用该函数前，必须先使用iter()函数获得迭代器对象。

``` python
>>> my_list = [1, 2, 3]
>>> it = iter(my_list)
>>> print(next(it))
1
>>> print(next(it))
2
>>> print(next(it))
3
>>> print(next(it))       # 如果迭代已经结束，抛出StopIteration异常
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

## 4.5 TypeError: 'NoneType' object is not iterable
TypeError: 'NoneType' object is not iterable

这个错误通常是由于某个函数（方法）期望某个参数能够被迭代，但是却收到了一个空值或None作为参数。以下是一个例子：

``` python
def say_hi():
    names = None      # 指定names为None
    for name in names:
        print('Hello', name)
    
say_hi()               # 会抛出TypeError
```

虽然`names=None`指定了一个空列表，但是这里仍然会报一个类型错误。这是为什么呢？实际上，函数`say_hi()`并不知道应该怎么去迭代`None`值。

解决的方法就是确保所接收到的参数不是`None`。如果确实期待得到一个空值的列表，那么最简单的做法是创建一个空列表并使用默认参数给函数指定初始值。

``` python
def say_hi(names=[]):         # 创建一个空列表作为默认参数
    for name in names:
        print('Hello', name)
        
say_hi()                     # 输出：[]
```