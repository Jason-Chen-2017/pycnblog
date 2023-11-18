                 

# 1.背景介绍


## 什么是字典？
字典（Dictionary）是一种无序的数据结构，它以键值对的方式存储数据。其中，键（key）与值（value）是成对出现的，键是不可改变的，也就是说，字典中的每一个键只能对应唯一的值。字典具有以下几个特点：

1、键-值对
2、动态性
3、高效性

## 什么是集合？
集合（Set）是由零个或多个元素组成的无序集合。集合中不允许重复元素，每个集合都有一个独特的名称或者编号。集合具有以下几个特点：

1、元素唯一且无序
2、集合内运算的效率很高
3、可变性

## 为什么要用字典和集合？
字典和集合在实际开发中经常作为数据结构出现，并用来实现各种功能，例如：

1、存放用户信息、商品信息等
2、进行数据库查询、数据筛选
3、缓存管理
4、数据分类、统计分析

因此，掌握字典和集合，对于实际项目开发和解决问题是至关重要的。本文旨在提供给初级开发人员学习和实践字典和集合所需的基础知识和技能。


# 2.核心概念与联系
## 1.字典的基本概念
字典（Dictionary）是一个无序的容器，其中的元素是通过键值对（Key-Value Pairs）来存放的。每个键值对对应着唯一的键和值。

## 2.字典的创建方法
创建一个空字典可以使用下面的方式：

```python
my_dict = {} # 创建一个空字典
```

或者也可以通过字典构造函数 `dict()` 来创建：

```python
my_dict = dict() # 创建一个空字典
```

可以通过以下两种方式来添加键值对到字典中：

1. 直接赋值：

```python
my_dict['name'] = 'John' # 添加新键值对
```

2. 使用 `update()` 方法：

```python
my_dict.update({'age': 27}) # 添加新键值对
```

## 3.字典的访问、删除和遍历方法
字典的访问，删除，以及遍历可以有多种方法。以下将介绍一些常用的访问、删除和遍历的方法：

### 3.1 通过索引访问字典元素

通过索引 `[]` 的方式访问字典元素。但需要注意的是，字典的索引是按照 key 排序后的顺序，所以可能不是按顺序排列的原始顺序。

```python
my_dict = {'apple': 1, 'banana': 2, 'orange': 3}

print(my_dict['apple']) # Output: 1
```

### 3.2 获取字典长度

获取字典的长度，可以使用 `len()` 函数。

```python
my_dict = {'apple': 1, 'banana': 2, 'orange': 3}

print(len(my_dict)) # Output: 3
```

### 3.3 删除字典元素

可以通过以下三种方式删除字典元素：

1. 通过 `del` 语句：

```python
my_dict = {'apple': 1, 'banana': 2, 'orange': 3}

del my_dict['apple']
```

2. 使用 `pop()` 方法：

```python
my_dict = {'apple': 1, 'banana': 2, 'orange': 3}

my_dict.pop('banana')
```

3. 修改字典之后，直接重新赋值即可：

```python
my_dict = {'apple': 1, 'banana': 2, 'orange': 3}

my_dict = {k: v for k, v in my_dict.items() if k!= 'apple'}
```

### 3.4 遍历字典元素

遍历字典元素有三种方法：

1. 使用 `keys()` 和 `values()` 方法：

```python
my_dict = {'apple': 1, 'banana': 2, 'orange': 3}

for key in my_dict.keys():
    print(key)
    
for value in my_dict.values():
    print(value)
```

2. 使用 `items()` 方法：

```python
my_dict = {'apple': 1, 'banana': 2, 'orange': 3}

for key, value in my_dict.items():
    print("The key is %s and the value is %d" % (key, value))
```

3. 使用 `in` 操作符：

```python
my_dict = {'apple': 1, 'banana': 2, 'orange': 3}

if 'apple' in my_dict:
    print(my_dict['apple'])
else:
    print('Not Found')
```

## 4.集合的基本概念
集合（set）是一个无序的元素集，其中的元素不能重复。集合有以下两个特点：

1. 不允许重复元素：集合中不会出现相同元素
2. 集合运算效率高：对集合进行的运算通常比列表快很多

## 5.集合的创建方法
创建一个空集合可以使用下面的方式：

```python
my_set = set() # 创建一个空集合
```

或者也可以通过集合构造函数 `set()` 来创建：

```python
my_set = set([1, 2, 3]) # 创建一个集合，元素为1，2，3
```

## 6.集合的操作方法
集合支持以下几种操作：

1. 添加元素：

```python
my_set = set()
my_set.add(1)    # 添加元素1
```

2. 删除元素：

```python
my_set = set([1, 2, 3])
my_set.remove(2) # 删除元素2
```

3. 查找元素是否存在：

```python
my_set = set([1, 2, 3])

if 2 in my_set:
    print('Found')
else:
    print('Not Found')
```

4. 求交集、并集、差集：

```python
a = set(['apple', 'banana', 'cherry'])
b = set(['google','microsoft', 'apple'])

print(a & b)     # Output: {'apple'}
print(a | b)     # Output: {'banana','microsoft', 'cherry', 'apple'}
print(a - b)     # Output: {'banana', 'cherry'}
```

5. 对集合进行排序：

```python
my_set = set([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
sorted_list = sorted(list(my_set))   # 将集合转换为列表并排序

print(sorted_list)                 # Output: [1, 2, 3, 4, 5, 6, 9]
```