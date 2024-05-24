
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 什么是Python？
Python 是一种高级编程语言，它被设计用于可读性、易用性和可扩展性。它的语法类似于 Perl、Ruby 或 Java，并具有丰富的数据结构和控制结构。Python 是一种面向对象的语言，支持多种编程范式，包括命令式、函数式和面向对象。

## 1.2 为什么要学习Python？
在数据科学、机器学习、Web开发、系统管理员和移动应用开发等领域都可以使用 Python。以下是一些主要原因：

1. Python 有着广泛的库和框架。这些库可以帮助您解决实际问题，而无需编写复杂的代码。
2. Python 是开源的。这意味着您可以免费获取源代码并修改它。
3. Python 支持多平台。它可以在 Linux、Windows 和 macOS 上运行。
4. Python 有着快速的执行速度。它可以轻松处理大型数据集。
5. Python 具有简单易用的语法。它使编程变得更加容易，并简化了编码过程。

## 2.安装与配置环境
### 2.1 安装 Python
Mac OS X 上安装 Python 可以通过 Homebrew 来完成。如果没有安装过 Homebrew ，请先参考下面的链接安装：https://brew.sh/index_zh-cn。

打开终端输入以下命令进行安装：

```bash
brew install python
```

安装成功后，在命令行输入 `python` 命令测试是否安装成功。出现如下图所示信息则代表安装成功。


如果出现以上提示，说明 Python 已安装成功。如果只想看 Python 的版本号，可以输入 `python --version`。

### 2.2 配置环境变量
默认情况下，Homebrew 会将 Python 安装到 `/usr/local/` 下，因此需要将该目录添加到环境变量中才能正常使用。

首先，打开 `~/.zshrc` 文件（如果使用的是 bash 就打开 `~/.bashrc`），并添加以下内容：

```bash
export PATH="/usr/local/bin:$PATH"
```

保存文件后，重新加载环境变量：

```bash
source ~/.zshrc   # 如果使用的是 zsh，则用 ~/.zprofile 替换掉上面两行
```

之后就可以在任意目录下使用 `python` 命令运行 Python 程序。

### 2.3 IDE选择
推荐使用 PyCharm IDE，它是一个功能全面的 Python IDE。这里不做过多介绍，详细使用方法请参考官方文档 https://www.jetbrains.com/help/pycharm/installation-guide.html。

## 3.Python基础知识
### 3.1 数据类型
Python 中有七个标准的数据类型：

1. Number (数字)
    - int (整型)
    - float (浮点型)
    - complex (复数)
2. String (字符串)
3. List (列表)
4. Tuple (元组)
5. Set (集合)
6. Dictionary (字典)
7. Boolean (布尔值)

其中，除去 Number 和 String 以外，其余均属于不可变类型（如 list）。String 可用单引号或双引号表示。

**Number:**
示例代码：

```python
num = 10    # 整数
print(type(num))    # <class 'int'>
 
num = 3.14     # 浮点数
print(type(num))   # <class 'float'>
 
num = 2 + 3j      # 复数
print(num.real)    # 2.0
print(num.imag)    # 3.0
```

**String:**
示例代码：

```python
str = "Hello World!"
print(len(str))       # 12
print(str[::-1])      #!dlroW olleH
```

**List:**
示例代码：

```python
list = [1, "apple", True]
print(len(list))          # 3
list.append("orange")
print(list)               # [1, 'apple', True, 'orange']
```

**Tuple:**
示例代码：

```python
tuple = ("apple", 1, False)
print(len(tuple))         # 3
try:
    tuple[2] = True        # TypeError: 'tuple' object does not support item assignment
except Exception as e:
    print(e)                # 不支持修改元素，TypeError: 'tuple' object does not support item assignment
    
new_tuple = (True,)         # 需要在末尾添加逗号
print(type(new_tuple))     # <class 'tuple'>
```

**Set:**
示例代码：

```python
set = {1, 2, 3}
print(type(set))           # <class'set'>

set.add(4)                 # 添加元素
print(set)                 #{1, 2, 3, 4}

set.remove(3)              # 删除元素
print(set)                 #{1, 2, 4}
```

**Dictionary:**
示例代码：

```python
dict = {"name": "John Doe", "age": 30, "city": "New York"}
print(dict["name"])                   # John Doe
print(dict.get("email", "Not Found")) # Not Found

dict["country"] = "USA"
print(dict)                           # {'name': 'John Doe', 'age': 30, 'city': 'New York', 'country': 'USA'}
del dict["age"]                       # 删除键值对
print(dict)                           # {'name': 'John Doe', 'city': 'New York', 'country': 'USA'}
```

**Boolean:**
示例代码：

```python
true_bool = True
false_bool = False
print(type(true_bool))     # <class 'bool'>
```

### 3.2 条件语句
Python 中的条件语句包括 `if`、`elif`、`else`，分别对应 if-then-else 逻辑，以及多个条件匹配时的 fallback 操作。

示例代码：

```python
number = 10
 
if number % 2 == 0:
    print(number, "is even.")
elif number % 2!= 0 and number >= 0:
    print(number, "is positive odd number.")
elif number <= 0:
    print(number, "is negative odd number.")
else:
    print(number, "is an odd number.") 
```

**输出结果：**
```
10 is even.
```

注意：Python 中的空格不能省略。

### 3.3 循环语句
Python 中的循环语句分为两种，即 `for` 和 `while` 循环，二者各有特色。

#### for 循环
`for` 循环是最基本的循环，用来遍历序列中的每个元素。示例代码：

```python
fruits = ["apple", "banana", "orange"]
 
for fruit in fruits:
    print(fruit)
```

**输出结果：**
```
apple
banana
orange
```

#### while 循环
`while` 循环用来基于某些条件进行循环，直至条件满足为止。示例代码：

```python
i = 0
while i < 5:
    print(i)
    i += 1
```

**输出结果：**
```
0
1
2
3
4
```

**break 语句:**
当 `while` 循环的条件变为 `False` 时，便会跳出循环体，并执行后续代码。示例代码：

```python
i = 0
while True:
    print(i)
    i += 1
    
    if i > 5:
        break
```

**输出结果：**
```
0
1
2
3
4
5
```

**continue 语句:**
当 `while` 循环的当前迭代被跳过时，便会直接进入下一次迭代。示例代码：

```python
i = 0
while i < 5:
    i += 1
    
    if i == 3:
        continue
        
    print(i)
```

**输出结果：**
```
1
2
4
5
```

### 3.4 函数定义
Python 中的函数使用 `def` 关键字进行定义，并接受若干参数。示例代码：

```python
def add(x, y):
    return x + y
 
result = add(3, 4)
print(result)   # Output: 7
```

### 3.5 模块导入
模块允许在一个 `.py` 文件内导入另一个 `.py` 文件中的函数或类，从而实现代码重用。示例代码：

```python
import moduleExample
 
moduleExample.hello()    # Output: Hello from example.py!
```

### 3.6 文件操作
文件操作可以进行文件的读取、写入及追加，还可以通过文件操作实现数据交换。示例代码：

```python
fileObject = open('example.txt','r')

lines = fileObject.readlines()   # 读取全部内容

for line in lines:
    print(line)                  # 输出每一行的内容

fileObject.close()                # 关闭文件
```

## 4.Python 进阶
### 4.1 Lambda 函数
Lambda 函数是一个匿名函数，是一段小函数，只能有一个表达式，并且没有自己的名称。它的格式如下：

```python
lambda arg1, arg2,... : expression
```

示例代码：

```python
double = lambda x: x * 2
triple = lambda x: x * 3
quadruple = lambda x: double(x) + triple(x)
print(quadruple(2))    # Output: 10
```

### 4.2 List Comprehensions
列表推导式是一种方便创建新的列表的方式，它提供了一种简洁的方法来创建列表。它通过对已有列表进行筛选和映射，并生成新列表。语法如下：

```python
new_list = [expression for item in iterable if condition]
```

示例代码：

```python
numbers = range(1, 6)
squares = [n ** 2 for n in numbers if n % 2 == 0]
print(squares)   # Output: [4, 16]
```

### 4.3 Generator Expressions
生成器表达式也称为推导式，它是一种特殊的迭代器，返回一个 generator 对象而不是完整的列表。语法如下：

```python
generator_object = (expression for item in iterable if condition)
```

示例代码：

```python
squared_nums = (n ** 2 for n in range(1, 6))
print(next(squared_nums))    # Output: 1
print(next(squared_nums))    # Output: 4
print([next(squared_nums), next(squared_nums)])    # Output: [9, 16]
```

### 4.4 Iterators vs. Generators
迭代器是一个可以顺序访问某个特定序列元素的对象，例如列表；而生成器是一种惰性迭代器，它的元素只能通过一次迭代才产生，例如生成器表达式或者 map() 函数。通常情况下，如果函数调用生成了一个列表，那么它一般返回完整的列表；但对于生成器表达式来说，只有迭代生成器的时候才会真正计算，才会产生元素，从而节约内存空间。