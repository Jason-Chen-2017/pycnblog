
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Python中的字典（Dictionary）是一种非常灵活的数据结构，它可以存储任意类型对象，并且键值对之间的关系可以用各种方式表示，并且不需要按照特定的顺序存储。字典在 Python 中被称为映射、关联数组或哈希表。字典提供了一些操作字典的函数，比如添加、删除、修改元素等。以下将通过几个例子带领大家了解如何使用字典。

# 2.背景介绍
字典是一个无序的键值对集合，其中每个键都是唯一的，值可以没有限制地取任何对象。字典具有以下特征：

1. 无序性：字典中的元素是无序的。
2. 可变性：字典中的元素可以增加、删除或者改变。
3. 索引性：可以通过键访问字典中的元素。
4. 重复性：可以出现相同的键，但是值不允许重复。

本文将会通过几个代码实例阐述字典的创建、添加、删除、修改、查找等常用操作。

# 3.基本概念术语说明

## 3.1 字典的定义及初始化方法

字典是一种无序的键-值对容器，它的元素之间没有顺序可言。字典在创建时，可以指定键和对应的值；也可以使用字典推导式快速初始化字典。键必须不可变对象，如字符串、数字或元组，不能为 None。

```python
# 创建字典
dict1 = {'name': 'John', 'age': 29}
print(dict1) # Output: {'name': 'John', 'age': 29}

# 使用字典推导式快速初始化字典
dict2 = {num: num*num for num in range(1, 11)}
print(dict2) # Output: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81, 10: 100}
```

## 3.2 字典的基本操作

### 添加元素

向字典中添加元素的方法是使用 `dict[key] = value` 这种赋值语句，这里的 key 是字典中所存对象的标识符，value 是对应的对象。

```python
# 添加元素
dict1['city'] = 'New York'
print(dict1) # Output: {'name': 'John', 'age': 29, 'city': 'New York'}

# 当键已经存在时，如果值为新值，则覆盖原有的值；否则什么都不做。
dict1['age'] = 30
print(dict1) # Output: {'name': 'John', 'age': 30, 'city': 'New York'}

# 不建议用 list 作为字典的键！
list_key = [1, 2, 3]
dict1[list_key] = "This is not a good idea!"
print(dict1) # Output: {'name': 'John', 'age': 30, 'city': 'New York', '[1, 2, 3]': 'This is not a good idea!'}
```

### 删除元素

从字典中删除元素可以使用关键字 del 和 pop() 方法。del 是直接删除字典中指定的键值对，pop() 可以同时删除字典中的元素并返回相应的值。

```python
# 删除元素
del dict1['age']
print(dict1) # Output: {'name': 'John', 'city': 'New York'}
print('age' in dict1) # Output: False

# 从字典中随机删除一个元素
dict2 = {i : chr(i+97) for i in range(10)}
random_key = random.choice(list(dict2))
removed_value = dict2.pop(random_key)
print("Removed element:", removed_value)
print("Dictionary after deletion:", dict2)
```

### 修改元素

使用赋值语句即可修改字典中的元素。

```python
# 修改元素
dict1['city'] = 'London'
print(dict1) # Output: {'name': 'John', 'city': 'London'}
```

### 查找元素

字典提供了两种方式查找元素：通过 key 获取 value 或通过 value 获取 key。

```python
# 通过 key 获取 value
if 'age' in dict1:
    print(dict1['age']) # Output: 30
    
# 通过 value 获取 key
for key, val in dict1.items():
    if val == 'London':
        print(key) # Output: city
```

## 4.具体代码实例和解释说明

以下两个代码实例展示了字典的创建、添加、删除、修改、查找等基本操作。

### 创建字典

创建一个空字典，然后添加键值对，或者使用字典推导式快速初始化字典。

```python
# 创建一个空字典
my_dict = {}

# 添加键值对
my_dict['apple'] = 5
my_dict['banana'] = 10
print(my_dict)  # Output: {'apple': 5, 'banana': 10}

# 使用字典推导式快速初始化字典
squares = {x: x ** 2 for x in range(1, 6)}
print(squares)  # Output: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
```

### 添加元素

向字典中添加元素的方式有多种，可以直接赋值或者通过方法进行添加。

```python
# 用赋值语句添加元素
my_dict['orange'] = 7
print(my_dict)  # Output: {'apple': 5, 'banana': 10, 'orange': 7}

# 通过方法添加元素
my_dict.update({'peach': 8})
print(my_dict)  # Output: {'apple': 5, 'banana': 10, 'orange': 7, 'peach': 8}
```

### 删除元素

从字典中删除元素的方式有三种：直接删除、pop() 方法、popitem() 方法。

```python
# 用 del 操作符直接删除元素
del my_dict['banana']
print(my_dict)  # Output: {'apple': 5, 'orange': 7, 'peach': 8}

# 用 pop() 方法删除元素
val = my_dict.pop('orange')
print(val)     # Output: 7
print(my_dict)  # Output: {'apple': 5, 'peach': 8}

# 用 popitem() 方法随机删除元素
val = my_dict.popitem()
print(val)     # Output: ('peach', 8) or any other pair of the dictionary
print(my_dict)  # Output: {'apple': 5} or any remaining elements in the original dictionary
```

### 修改元素

使用赋值语句即可修改字典中的元素。

```python
# 用赋值语句修改元素
my_dict['apple'] = 6
print(my_dict)   # Output: {'apple': 6, 'orange': 7, 'peach': 8}
```

### 查找元素

字典提供了两种方式查找元素：通过 key 获取 value 或通过 value 获取 key。

```python
# 通过 key 获取 value
if 'apple' in my_dict:
    print(my_dict['apple'])    # Output: 6

# 通过 value 获取 key
for k, v in my_dict.items():
    if v == 7:
        print(k)                # Output: orange
```

### 深拷贝与浅拷贝

字典的深拷贝和浅拷贝可以分别用 copy() 和 deepcopy() 函数实现。

```python
import copy

# 浅拷贝
original_dict = {'a': 1, 'b': [{'c': 2}]}
shallow_copy_dict = original_dict.copy()
original_dict['b'][0]['c'] = 3      # 此处修改原字典 b 的元素，不会影响 shallow_copy_dict 中的元素
print(original_dict)              # Output: {'a': 1, 'b': [{'c': 3}]}
print(shallow_copy_dict)          # Output: {'a': 1, 'b': [{'c': 3}]}

# 深拷贝
original_dict = {'a': 1, 'b': [{'c': 2}]}
deep_copy_dict = copy.deepcopy(original_dict)
original_dict['b'][0]['c'] = 3      # 此处修改原字典 b 的元素，不会影响 deep_copy_dict 中的元素
print(original_dict)              # Output: {'a': 1, 'b': [{'c': 3}]}
print(deep_copy_dict)             # Output: {'a': 1, 'b': [{'c': 2}]}
```

### 排序

用 sorted() 函数对字典排序，输出列表形式的键值对序列。

```python
my_dict = {'apple': 5, 'banana': 10, 'orange': 7, 'peach': 8}
sorted_pairs = sorted(my_dict.items())
print(sorted_pairs)   # Output: [('apple', 5), ('banana', 10), ('orange', 7), ('peach', 8)]
```