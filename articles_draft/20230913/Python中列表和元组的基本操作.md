
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Python语言具有强大的列表、元组、字典等数据结构，能够轻松实现复杂的数据处理任务。本文将介绍两种最常用的数据类型——列表（list）和元组（tuple），并对它们进行一些基本操作。
## 为什么需要列表和元组？

在实际编程中，经常会遇到要存储多个元素的场景，如字符串、数字等。而这些元素可能是动态生成的，因此不能事先确定大小，只能通过元素的数量和下标来索引访问元素。这就需要一种数据结构来存储这些元素。列表和元组就是这样一种数据结构。列表可以存储任意多的元素，支持动态添加、删除元素；而元组则是不可变的序列，一次性创建完毕后其元素值就固定了，不能修改。如果需要修改某个元组中的元素，可以先转换成列表再进行修改。

## 列表（List）

列表（List）是Python中的一个内置数据结构。它类似于数组，可以保存多个元素。列表可以使用方括号 [] 来定义，里面可以存放不同类型的对象，甚至可以包含列表或者其他可迭代对象。也可以把列表看作是一个线性集合，其中每个元素都有一个唯一的位置或索引，可以通过索引直接访问列表中的元素。

```python
my_list = ['apple', 'banana', 'orange']
print(my_list) # Output: ['apple', 'banana', 'orange']
```

### 创建列表

#### 使用[]语法创建空列表

创建一个空列表，只需用[]符号即可，但元素个数默认为零。

```python
empty_list = []
print(len(empty_list)) # Output: 0
```

#### 使用list()函数创建列表

如果已经存在一系列的值，可以使用list()函数将其转换为列表。该函数可以接收可迭代对象作为参数，将其转换为列表。

```python
numbers = [1, 2, 3]
fruits = list('hello')
mixed_list = list([1,'a',[2]])
print(numbers)    # Output: [1, 2, 3]
print(fruits)     # Output: ['h', 'e', 'l', 'l', 'o']
print(mixed_list) # Output: [1, 'a', [2]]
```

### 访问列表元素

列表中的每一个元素都有一个唯一的编号，称之为下标（Index）。列表的第一个元素的下标是0，第二个元素的下标是1，依此类推。我们可以通过下标的方式来访问列表中的元素。列表支持负数索引，也就是从右边开始计数。

```python
my_list = ['apple', 'banana', 'orange']
print(my_list[0])   # Output: apple
print(my_list[-1])  # Output: orange
```

列表索引值越界时，会引发IndexError异常。

```python
my_list = ['apple', 'banana', 'orange']
print(my_list[3]) # Output: IndexError: list index out of range
```

### 修改列表元素

列表中的元素可以通过赋值给下标的方式进行修改。

```python
my_list = ['apple', 'banana', 'orange']
my_list[1] = 'grape'
print(my_list) # Output: ['apple', 'grape', 'orange']
```

当然，如果下标越界或者不是整数，那么也会引发异常。

```python
my_list = ['apple', 'banana', 'orange']
my_list['name'] = 'John Doe' # TypeError: list indices must be integers or slices, not str
my_list[2] = ('pear', 'peach') # TypeError: 'tuple' object does not support item assignment
```

### 添加元素

列表中的元素可以在末尾追加、插入新的元素。

#### 在末尾追加元素

使用append()方法可以将元素添加到列表的末尾。

```python
my_list = ['apple', 'banana', 'orange']
my_list.append('watermelon')
print(my_list) # Output: ['apple', 'banana', 'orange', 'watermelon']
```

#### 在指定位置插入元素

使用insert()方法可以将元素插入到指定的位置。

```python
my_list = ['apple', 'banana', 'orange']
my_list.insert(1, 'grape')
print(my_list) # Output: ['apple', 'grape', 'banana', 'orange']
```

当插入位置超出列表长度范围时，不会引发异常，而是自动调整插入位置使得索引合法。

```python
my_list = ['apple', 'banana', 'orange']
my_list.insert(-7,'mango')
print(my_list) # Output: ['mango', 'apple', 'banana', 'orange']
```

### 删除元素

列表中的元素可以通过del语句来删除，或者pop()方法来删除末尾或者指定位置的元素。

#### del语句删除元素

del语句可以在列表中删除元素。

```python
my_list = ['apple', 'banana', 'orange']
del my_list[1]
print(my_list) # Output: ['apple', 'orange']
```

del语句也可以删除整个列表。

```python
my_list = ['apple', 'banana', 'orange']
del my_list
print(my_list) # NameError: name'my_list' is not defined
```

#### pop()方法删除元素

pop()方法可以删除末尾或者指定位置的元素。

```python
my_list = ['apple', 'banana', 'orange']
element = my_list.pop(1)
print(element) # Output: banana
print(my_list) # Output: ['apple', 'orange']
```

pop()方法可以返回被删除的元素值。如果没有指定位置，则默认删除末尾元素。

```python
my_list = ['apple', 'banana', 'orange']
last_elem = my_list.pop()
print(last_elem) # Output: orange
print(my_list)   # Output: ['apple', 'banana']
```