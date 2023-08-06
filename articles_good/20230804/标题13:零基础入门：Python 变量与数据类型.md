
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         Python 是一种高级编程语言，其优点主要有以下几点：
         
         1.简单易学：Python 的语法和简单特性使它很容易学习和上手。不用担心晦涩难懂的各种语言规则或特殊语法，通过一些简单的规则即可轻松实现代码功能。
         2.丰富的库支持：Python 有大量丰富的第三方库支持，能够满足不同领域的需求。例如科学计算、图像处理、Web开发等。
         3.海量资源：Python 有海量的开源资源和教程，从最初的学习知识到各个方向的深入应用都有大量的资源可以学习。
         
         本文旨在帮助初学者快速上手 Python ，从变量、数据类型开始，逐步深入至函数、模块、对象、异常处理、类等高级知识，希望达到事半功倍的效果。
         # 2.基本概念术语说明
         ## 2.1 数据类型
         
         在计算机编程中，数据类型是指存储在内存中的数据的形式、大小、表示方法和操作方式。Python 中共有七种内置的数据类型：
         
         ### （1）数值型(Number)
         - int (整数): 可以为正、负、0，如 10, -3, 0
         - float (浮点数): 用于表示小数，如 3.14, -9.8
         - complex (复数): 表示复数，如 1+2j, -4-7j
         
         ### （2）字符串型（String）
         - str (字符串): 用单引号' 或双引号" 括起来的任意文本，如 'hello', "world", "I'm OK!"
         
         ### （3）布尔型（Boolean）
         - bool (布尔值): True 和 False 的值，如 True, False
         
         ### （4）列表型（List）
         - list (列表): 元素可变的有序集合，用 [ ] 来表示，元素之间用, 分隔，如 [1, 'apple', True]
         
         ### （5）元组型（Tuple）
         - tuple (元组): 元素不可变的有序集合，用 ( ) 来表示，元素之间用, 分隔，如 (1, 'apple'), ('cat', 'dog')
         
         ### （6）字典型（Dictionary）
         - dict (字典): 由键-值对组成的无序集合，用 { } 来表示，键值对之间用 : 分隔，如 {'name': 'Alice', 'age': 20}
         
         ### （7）集合型（Set）
         - set (集合): 元素唯一且无序，用 { } 来表示，元素之间用, 分隔，如 {1, 2, 3}, {'apple', 'banana'}
         
         ## 2.2 变量
         
         变量是给数据命名的符号，用来存储并管理数据。在 Python 中，可以使用等号 "=" 为变量赋值，也可以使用空格分隔多个变量。
         
         ```python
         a = b = c = 1   # 将 1 赋值给多个变量 a、b、c
         
         x, y, z = 1, 2, 3    # 使用空格分隔多个变量
         
         name = 'Alice'      # 创建一个变量名为 name 的变量
         age = 20            # 创建一个变量名为 age 的变量
         ```
         
         注意：变量名必须遵循标识符命名规范，并且不能用关键字作为名字，比如 print, def 等。
         ## 2.3 注释
         
         在编写程序时，通常需要添加注释来描述代码，方便其他程序员理解你的想法。在 Python 中，有三种类型的注释：
         
         ### （1）单行注释
         
         以 "#" 开头的直到该行结束的所有文字都是注释，会被 Python 解释器忽略。如：
         
         ```python
         # This is a single line comment.
         ```
         
         ### （2）多行注释
         
         也叫块注释，使用三个单引号或者三个双引号括起来的文字，当作一个整体注释，不会影响程序的执行。如：
         
         ```python
         '''This is the first line of a multi-line
         
         comment'''
         ```

         ```python
         """This is the second line of a multi-line
          
         comment"""
         ```
         
         ### （3）文档注释
         
         即为源文件开头的一段特殊注释，可以提供对整个文件的信息。如：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A simple program to add two numbers."""

__author__ = "Alice"
__email__ = "<EMAIL>"
__version__ = "v1.0"
```

除此之外，还有一种特殊注释叫做 shebang（井号），用来告诉系统如何运行这个脚本。Shebang 一般放在第一行，放在 Unix 和 Linux 操作系统中，如 #!/bin/sh。而 Windows 命令提示符只能识别.bat 文件，所以没有办法指定命令运行脚本。

## 2.4 操作符

运算符是一种特殊符号，用于对数值进行计算。Python 支持很多运算符，包括：

### （1）算术运算符

| 运算符 | 描述     | 例子          |
| ------ | -------- | ------------- |
| +      | 加       | `x + y`       |
| -      | 减       | `x - y`       |
| *      | 乘       | `x * y`       |
| /      | 除       | `x / y`       |
| **     | 乘方     | `x ** y`      |
| %      | 求余     | `x % y`       |

### （2）比较运算符

| 运算符 | 描述                         | 例子           |
| ------ | ---------------------------- | -------------- |
| ==     | 等于                         | `x == y`       |
|!=     | 不等于                       | `x!= y`       |
| >      | 大于                         | `x > y`        |
| <      | 小于                         | `x < y`        |
| >=     | 大于等于                     | `x >= y`       |
| <=     | 小于等于                     | `x <= y`       |

### （3）逻辑运算符

| 运算符 | 描述                  | 例子               |
| ------ | --------------------- | ------------------ |
| and    | 逻辑 AND              | `True and False`   |
| or     | 逻辑 OR               | `True or False`    |
| not    | 逻辑 NOT              | `not True`         |

## 2.5 流程控制语句

流程控制语句用于基于某些条件执行某段代码。Python 提供了 if...else、while、for、break、continue、pass 等语句。

### （1）if...else 语句

if...else 语句是判断条件是否成立，如果成立则执行一系列的代码，否则跳过这些代码。

```python
num = input("请输入一个数字：")

if num.isdigit():
    num = int(num)
    if num % 2 == 0:
        print(f"{num} 是偶数")
    else:
        print(f"{num} 是奇数")
else:
    print("输入错误！")
```

### （2）while 循环

while 循环用于重复执行一系列代码，只要条件表达式为真，就一直执行代码块。

```python
count = 1
total = 0

while count <= 10:
    total += count
    count += 1
    
print(f"和为{total}")
```

### （3）for 循环

for 循环用于遍历序列（如字符串、列表、元组、集合）或可迭代对象（如 range 对象）。

```python
fruits = ['apple', 'banana', 'orange']

for fruit in fruits:
    print(fruit)
```

### （4）break 语句

break 语句用于提前退出循环。

```python
for i in range(1, 10):
    for j in range(1, i+1):
        if i*i == j*j + i:
            break
    else:
        continue
    
    print(f"1*{i} + {i}^2 = {i*(i+1)*(2*i+1)//6}")
    print(f"已完成第 {(i*(i+1)*2//6)} 个组合。")

    if i*(i+1)*(2*i+1)//6 == 20:
        break
        
    if i == 8:
        exit()
```

### （5）continue 语句

continue 语句用于跳过当前循环的剩余代码，直接开始下一次循环。

```python
sum = 0
n = 1

while n <= 100:
    if n % 2 == 0:
        sum += n
    n += 1

print(sum)
```

### （6）pass 语句

pass 语句什么都不做，一般用于占据一个位置。

```python
if num == "":
    pass   # 当输入为空时，执行 pass 语句
else:
   ...    # 如果输入非空，继续执行后续语句
```

## 2.6 函数

函数是可重用的代码块，可以将相同或相似任务的操作封装起来，避免代码重复，提升代码的可读性和效率。在 Python 中，函数就是一个独立的实体，可以通过 def 关键字定义。

```python
def greet(name):
    print(f"Hello, {name}!")
    
    
greet('Alice')
greet('Bob')
```

函数参数既可以按顺序传入，也可以按名称传入，还可以设置默认值，这样可以在调用函数时省略一些参数的值。

```python
def pow(base=2, exp=2):
    result = base ** exp
    return result


result1 = pow(exp=3)  # 设置了默认值，exp=2 时，base=2
result2 = pow(3, 2)   # 参数按顺序传入

print(result1)
print(result2)
```

函数还可以返回多个值，这时候需要使用星号 (*) 将结果打包。

```python
def fibonacci(n):
    prev_num = 0
    current_num = 1
    
    for _ in range(n):
        yield prev_num
        
        temp = current_num
        current_num += prev_num
        prev_num = temp
        
        
fib = fibonacci(10)   # 生成器对象
print([num for num in fib])   # 打印前10个数列

```

## 2.7 模块

模块是一个包含着相关函数和变量的文件，可以被别的程序引用。Python 中的模块就是一个.py 文件，可以在文件开头加上“#!/usr/bin/env python”的 shebang 或者“from module import function”导入某个模块中的函数。

创建模块非常简单，只需创建一个.py 文件，然后在其中定义相关函数和变量就可以了。