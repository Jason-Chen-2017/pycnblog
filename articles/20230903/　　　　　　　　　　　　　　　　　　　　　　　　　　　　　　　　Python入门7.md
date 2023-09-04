
作者：禅与计算机程序设计艺术                    

# 1.简介
  

​    Python 是一种高级编程语言，可以用来进行Web开发、机器学习、数据分析、金融建模等。无论是在数据科学领域还是在游戏制作领域，Python都有着广泛的应用。由于其简单易学、代码可读性强、功能强大、社区活跃等特点，使得它被越来越多的人青睐。本文将通过一个实际例子，让大家快速入门并掌握Python的基本语法。希望本文能够帮助读者快速上手Python，理解Python的特点，并且有所收获！
# 2.Python 2 VS Python 3
​    Python 2 和 Python 3 有什么区别呢？相对于 Python 2 来说，Python 3 主要改进了以下几个方面：

1.语法更加清晰、易读；

2.内置函数和库更新；

3.支持多线程和异步编程；

4.改善的 Unicode 支持；

5.字符串处理的速度更快；

6.支持第三方模块管理工具 pip 。

综上所述，推荐新手阅读《 Learning Python 5th Edition》或《 Expert Python Programming》，了解 Python 的最新特性。
# 3.安装配置Python
​    在 Windows 下，可以通过下载并安装 Anaconda 包（适用于 Python 2 和 3）来安装 Python 环境。Anaconda 是一个开源的数据科学包，包括了最常用的数学、统计、科学计算、机器学习、图像处理等常用软件包。Anaconda 安装后，直接双击运行 Anaconda Prompt 命令行窗口即可进入 Python 交互界面。

    在 Linux 或 macOS 上，也可以通过包管理器如 apt-get、yum 来安装 Python。如果系统已安装 Python 环境，则可以使用该环境下的 pip 命令来安装第三方模块。
# 4.Python 语法基础
## 4.1 基本语句
​    Python 使用缩进（indentation）来组织代码块，每个语句占据一行，最后一行不要忘记添加分号。例如：

```python
a = 1 + 2 * 3 / (4 - 5) # python 中使用反斜杠 \ 表示续行符，这种做法不推荐
if a > 0:
    print('positive')
else:
    print('non-positive')
for i in range(10):
    if i % 2 == 0:
        continue   # 如果条件为 True，则继续执行下一次循环
    print(i)
```

## 4.2 数据类型
​    Python 的数据类型有以下几种：

1.Number（数字）——整数、浮点数、复数
2.String（字符串）——字符型、字节型
3.List（列表）——同质列表
4.Tuple（元组）——不可变列表
5.Dictionary（字典）——键值对映射表
6.Set（集合）——无序且元素唯一的序列

### 4.2.1 数字类型
#### int（整数）
​    可以表示任意大小的整数，没有大小限制。示例如下：

```python
x = 123456789
y = -321
z = 0b10101010  #二进制形式
u = 0o777     #八进制形式
v = 0xff      #十六进制形式
print(type(x), type(y))
```

输出结果为：

```python
<class 'int'> <class 'int'>
```

#### float（浮点数）
​    浮点数就是带小数点的数字，也叫做小数或者实数。示例如下：

```python
x = 3.14
y = -.5
print(type(x), type(y))
```

输出结果为：

```python
<class 'float'> <class 'float'>
```

#### complex（复数）
​    复数由实部和虚部构成，由`j or J`表示。示例如下：

```python
x = 3+5j
y = -2j
print(type(x), type(y))
```

输出结果为：

```python
<class 'complex'> <class 'complex'>
```

### 4.2.2 字符串类型
​    Python 中的字符串用单引号 `'`、`"`、`'''`、`"""` 分割，三种类型的字符串可以自由混用。

```python
s1 = 'hello world'
s2 = "I'm OK"
s3 = '''Python is powerful.'''; print(s3); s4 = """Let's coding!"""; print(s4)
```

输出结果为：

```python
Python is powerful.
Let's coding!
```

### 4.2.3 列表类型
​    List（列表）是 Python 中最常用的数据结构之一。它是一种类似数组的有序集合，可以存储多个数据项。它的每个元素可以是任意数据类型，而且可以动态调整大小。

```python
list1 = [1, 'apple', 3.14, ['orange']]
list2 = list(range(10)); print(list2[:]); del list2[-1]; print(list2[:-1]) 
```

输出结果为：

```python
[0, 1, 2, 3, 4, 5, 6, 7, 8]
[0, 1, 2, 3, 4, 5, 6, 7]
```

### 4.2.4 元组类型
​    Tuple（元组）也是 Python 中重要的数据结构。它和 List 非常相似，但它是不可修改的列表，也就是说不能新增或删除元素。

```python
tuple1 = ('apple', 3.14, False)
tuple2 = tuple(['orange'])
print((tuple1, tuple2))
```

输出结果为：

```python
(('apple', 3.14, False), ('orange'))
```

### 4.2.5 字典类型
​    Dictionary（字典）是 Python 中另一种非常常用的数据结构。它是一个无序的键值对集合，其中键是唯一标识符。字典中的值可以是任意数据类型。

```python
dict1 = {'name': 'Alice', 'age': 20}
dict2 = dict([(1, 'apple'), (2, 'banana')])
print(dict1['name'], dict2[2])
```

输出结果为：

```python
Alice banana
```

### 4.2.6 集合类型
​    Set（集合）是一个无序且元素唯一的序列。

```python
set1 = {1, 2, 3}; set2 = set([4, 5, 6]); print(set1 | set2)
```

输出结果为：

```python
{1, 2, 3, 4, 5, 6}
```

## 4.3 变量作用域
​    Python 的变量作用域分为全局作用域和局部作用域两种。变量在定义时，默认就处于局部作用域中，而在第一次赋值之后，变量会转移到内存中。如果不使用全局声明，则只能在当前函数或模块内访问到这个变量。

```python
def test():
    x = 10; y = 20
    def inner_func():
        nonlocal x; x += 1  # 修改外部函数的变量值
        global y            # 此处允许修改全局变量的值
        y *= 2             # 修改全局变量的值
    return inner_func

f = test()
f(); print(x); print(y)
```

输出结果为：

```python
11
40
```

## 4.4 函数
​    Python 既支持全局函数（global function），又支持嵌套函数（nested function）。

### 4.4.1 全局函数
​    全局函数不需要显式地定义函数体，只需要提供函数名和参数，然后就可以调用。全局函数可以跨文件访问，并且可以在任何位置调用。

```python
def my_func(param):
    print("This is the message from the global function:", param)
    
my_func("Hello World!")
```

输出结果为：

```python
This is the message from the global function: Hello World!
```

### 4.4.2 嵌套函数
​    嵌套函数是指定义在其他函数里面的函数，只有调用函数才能执行内部函数的代码。

```python
def outer_func():
    num = 10
    
    def nested_func():
        nonlocal num
        num += 1
        
        print("num inside nested_func:", num)
        
    nested_func()
    print("num outside nested_func:", num)
    
outer_func()
```

输出结果为：

```python
num inside nested_func: 11
num outside nested_func: 11
```

## 4.5 模块导入
​    Python 提供了很多内置模块，比如 os、sys、math、json、csv、time、datetime 等。这些模块可以帮助完成常见任务，免去了重复编写代码的烦恼。为了方便管理项目中使用的模块，Python 提供了一个模块管理工具，即包（Package）管理工具 pip 。

pip 安装命令如下：

```shell
$ pip install requests
```

安装成功后，就可以引入模块来使用了。

```python
import math  
print(math.sqrt(25))

from datetime import date
today = date.today()
print(today)
```

输出结果为：

```python
5.0
(year=2019, month=7, day=17)
```