                 

# 1.背景介绍


Python中元组（tuple）是一个不可变序列，类似于列表但元素不能修改，创建元组可以使用小括号 () 或逗号隔开的一系列值来表示，如 (1, "hello", True) 。元组可以作为函数参数传递、存储多个相关的值、定义函数返回值等。元组可以用作字典键或集合成员。

本文将向读者介绍Python中的元组及其特性。

# 2.核心概念与联系
## 2.1元组概述

元组是一种不可变的序列数据类型，它由若干个元素组成，这些元素在括号中通过逗号分隔。元组中的每个元素都有自己的位置编号，下标从0开始，可以通过索引访问元组中的元素。

元组最重要的特征就是它们是不可变的。这意味着元组一旦被创建，它的元素就不能改变。创建元组时，如果需要改变元组中的某些元素，则只能创建一个新的元组，然后重新赋值给变量。元组中的元素可以是不同类型的对象，也可以是另一个元组。

元组的创建方式如下：

```python
t = ("apple", "banana", "orange")   # 创建字符串元组
t = tuple("hello world")           # 通过序列转换创建元组
t = ()                             # 创建空元组
```

## 2.2元组相关操作

### 2.2.1 访问元组元素

元组是按顺序排列的一组元素，可以通过下标访问其中的元素。下标从0开始，最大值为元组长度减1。以下示例展示了访问元组元素的两种方式：

```python
fruits = ("apple", "banana", "orange")    # 定义元组

print(fruits[0])     # 获取第一个元素 apple
print(fruits[-1])    # 获取最后一个元素 orange
print(fruits[1:])    # 获取第二个至最后一个元素 (banana, orange)
```

### 2.2.2 修改元组元素

由于元组是不可变的，因此不能修改元组中的元素。但是可以重新赋值给变量。如下示例：

```python
fruits = ("apple", "banana", "orange")    # 定义元组

fruits_copy = fruits                     # 将元组复制到新变量
fruits_copy[0] = "pear"                   # 修改第一个元素
print(fruits_copy)                        # ('pear', 'banana', 'orange')
```

或者可以先创建新的元组，再进行替换：

```python
fruits = ("apple", "banana", "orange")    # 定义元组

new_fruits = (fruits[0], "grapefruit", fruits[2])      # 创建新元组
print(new_fruits)                                   # ('apple', 'grapefruit', 'orange')

fruits = new_fruits                                 # 替换旧元组
print(fruits)                                        # ('apple', 'grapefruit', 'orange')
```

### 2.2.3 拆包元组

当有多个值要同时赋值给多个变量时，可以采用拆包的方式。元组可以作为一个整体来接收，然后把整个元组“拆”开赋值给对应变量。例如，要把元组 `t` 的前两个元素分别赋予变量 `x` 和 `y`，可按照如下方式实现：

```python
t = (1, 2, 3)                 # 定义元组
x, y, z = t                  # 用元组赋值给三个变量
print(x, y, z)                # 输出结果：1 2 3
```

## 2.3元组相关函数

### 2.3.1 len() 函数

len() 函数用于获取元组的长度，其返回值是一个整数，表示元组中元素的数量。例如：

```python
fruits = ("apple", "banana", "orange")
num_elements = len(fruits)
print(num_elements)        # 输出结果：3
```

### 2.3.2 min(), max() 函数

min() 函数用于获取元组中的最小值，max() 函数用于获取元组中的最大值。这两个函数只适用于数字类型元素的元组。例如：

```python
numbers = (1, 2, 3, 4, 5)
smallest = min(numbers)
largest = max(numbers)
print(smallest, largest)   # 输出结果：1 5
```

### 2.3.3 sum() 函数

sum() 函数用于计算元组中所有元素的总和。如果元组为空，则会报错。例如：

```python
numbers = (1, 2, 3, 4, 5)
total = sum(numbers)
print(total)               # 输出结果：15
```

### 2.3.4 sorted() 函数

sorted() 函数用于对元组进行排序并返回一个新的元组。排序规则遵循小到大排列，且对于非数字类型元素，按照ASCII码大小比较。例如：

```python
fruits = ("apple", "banana", "orange")
sorted_fruits = sorted(fruits)
print(sorted_fruits)       # 输出结果：('apple', 'banana', 'orange')
```