                 

# 1.背景介绍


首先，我们应该清楚地了解什么是Python编程。它是一个高级语言，用于进行人工智能、机器学习、数据分析等领域的开发。其语法类似于Java、C++，具有简单易学的特点，适合初学者入门。本系列教程将带您快速上手Python编程。

那么，什么是条件语句和循环结构呢？

条件语句是用来实现对某些特定条件下的执行流程的控制。在Python中，包括if-else语句、多分支if-elif-else语句、for循环、while循环等。其中，最常用的就是if-else语句和多分支if-elif-else语句。它们可以让程序根据不同的条件执行不同的动作。

而循环结构则是用来重复执行某段代码的语句。循环结构一般包括for和while循环。两者的区别主要在于，for循环更加强大灵活，可以使用索引访问元素；而while循环较为简单直接，但只能在循环条件为真时才会继续循环。

在学习条件语句和循环结构之前，先来看一些基础知识。如果您已经熟悉了这些内容，那么可以跳过这一部分。

## 1.1 基础知识
### 1.1.1 数据类型
Python的数据类型包括以下几种：

1. 数字(Number)
    - int (整型): 可以表示整数值，如 7 或 -9
    - float (浮点型): 可以表示小数，如 3.14或 -4.2
    - complex (复数型): 可以表示虚数及其对应的实数，如 1 + 2j 或 3 - 4j
    
2. 字符串(String)
    - 用单引号'或双引号"括起来的文本，例如 'hello world', "I'm a string."
    

3. 布尔值(Boolean)
    - True 和 False 两个值的逻辑关系
    

4. 列表(List)
    - 有序集合，元素可修改
    

5. 元组(Tuple)
    - 有序集合，元素不可修改
    

6. 字典(Dictionary)
    - 无序集合，键值对存储
    

7. 集合(Set)
    - 无序集合，元素不可重复

### 1.1.2 变量
在计算机中，变量是用于存储数据的一个容器。我们可以在程序运行期间给变量赋值、读取它的值。在Python中，变量名遵循标识符的命名规则，且不能为空白字符（即不能用空格，tab等），不可以用关键字，也不能与已有的变量名相同。

举例如下:

```python
name = "John" # 声明一个名字叫做 John 的变量
age = 25      # 声明一个年龄为 25 的变量
height = 1.75 # 声明一个身高为 1.75m 的变量
```

也可以把变量的值赋给另一个变量：

```python
a = 10          # 给变量a赋值10
b = a           # 把变量a的值赋给变量b
c = b + 2       # 对变量b进行计算得到新的值并保存到变量c
print(c)        # 输出变量c的值，结果为12
```

### 1.1.3 运算符
在Python中，共有六个运算符：

1. 算术运算符 (+,-,*,/)
2. 赋值运算符 (=)
3. 比较运算符 (>, <, ==,!=, >=, <=)
4. 逻辑运算符 (and, or, not)
5. 身份运算符 (is, is not)
6. 位运算符 (~, &, |, ^, <<, >>)

以上运算符的优先级和结合性，请参考相关文档。

## 1.2 条件语句
条件语句是Python中的一种结构，用于选择不同分支执行不同的操作。它的基本语法形式如下：

```python
if condition1:
    statement1    
    
elif condition2:
    statement2
    
...
    
else: 
    default_statement   
```

当满足`condition1`时，执行`statement1`，否则如果满足`condition2`，执行`statement2`，依此类推，直到某个`condition`满足或者没有更多的`condition`。如果所有`condition`都不满足，则执行默认的`default_statement`。

比如：

```python
num = input("Enter a number:")   # 获取用户输入的数字
if num % 2 == 0:                 # 如果数字是偶数
    print(num,"is even")         # 打印信息
else:                            # 如果数字是奇数
    print(num,"is odd")          # 打印信息
```

可以看到，通过条件语句，程序可以根据用户输入的数字判断它是偶数还是奇数，并分别输出相应的信息。这样就可以根据输入的数字做出不同的响应。

还可以加入多个条件，如：

```python
num = input("Enter a number:") 
if num > 0:                    # 如果数字大于0
    if num % 2 == 0:            # 如果数字是偶数
        print(num,"is positive and even") # 打印信息
    else:                       # 如果数字是奇数
        print(num,"is positive and odd")  # 打印信息
elif num < 0:                  # 如果数字小于0
    if num % 2 == 0:            # 如果数字是偶数
        print(num,"is negative and even") # 打印信息
    else:                       # 如果数字是奇数
        print(num,"is negative and odd")  # 打印信息
else:                          # 如果数字等于0
    print(num,"is zero")        # 打印信息
```

通过判断数字的正负和奇偶性，程序可以输出各种类型的消息。这个例子演示了如何利用条件语句的嵌套特性。

## 1.3 循环结构
循环结构是Python中的另一种结构，它提供了多种方式来重复执行一段代码。它的基本语法形式如下：

```python
for variable in sequence:
    statements
    
while condition:
    statements
```

对于for循环，`variable`代表序列中的每个元素，`sequence`代表需要遍历的序列。`statements`代表要执行的代码块。如果序列为空，则不会执行任何操作。示例如下：

```python
words = ["apple", "banana", "cherry"]  
for x in words:            
    print(x)                
```

输出结果为：

```
apple
banana
cherry
```

对于while循环，`condition`代表循环条件，只有满足该条件时，才会重复执行代码块。示例如下：

```python
i = 1
while i < 6:             
    print(i)              
    i += 1               
```

输出结果为：

```
1
2
3
4
5
```

需要注意的是，由于Python是动态语言，因此for循环的变量并不是一个固定的局部变量，而是一个迭代器对象。所以我们不能在循环体内对其进行重新赋值。但是可以通过其他方式完成相应的操作。