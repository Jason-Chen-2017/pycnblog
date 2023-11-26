                 

# 1.背景介绍


列表(List)是一个Python中重要的数据结构之一。它可以存储多个元素、包括数字、字符串、布尔值等类型，并且可以通过索引来访问其中的元素。列表提供了一种灵活的方式来存储、组织数据。本文将对Python列表进行介绍、学习使用列表的方法。
# 2.核心概念与联系
## 2.1.什么是列表？
列表是Python中提供的一种数据类型，用于存储一系列元素。每一个列表都由一个有序的元素组成，每个元素用括号[]中的逗号分隔开，并放在一对方括号之间。如下面的例子所示：

```
fruits = ['apple', 'banana', 'orange']
numbers = [1, 2, 3, 4]
names = ['Alice', 'Bob', 'Charlie']
booleans = [True, False]
```

以上三个列表分别保存了水果、数字、名字、布尔值的集合。在Python中，列表的应用非常广泛，例如列表可用来保存任意数量的元素，可以作为函数的参数传入或者从函数中返回。另外，列表也可以嵌套，即一个列表可以包含其他的列表或元组。

## 2.2.列表的索引、切片及迭代器
### 2.2.1.索引（Index）
索引用来访问列表中某一个位置的元素。通过给定一个索引，可以从列表中得到相应的元素。索引值以0开始计算，所以第0个元素的索引值为0，第1个元素的索引值为1，依次类推。下图展示了一个列表的索引示意图：


如上图所示，假设列表`[a, b, c]`的索引值为0，则其第一个元素`a`的索引值为0；索引值为1时，第二个元素`b`的索引值为1，以此类推。

### 2.2.2.切片（Slice）
切片是指从列表中获取子序列。切片操作通常包括两个参数——起始索引和结束索引，中间有一个冒号`:`。如果没有指定起始索引，默认从头开始；如果没有指定结束索引，默认到末尾结束。下图展示了列表的切片示意图：


如上图所示，假设列表`[a, b, c, d, e]`的切片，那么取出索引值为1至3的元素为`b`、`c`、`d`，而`e`不属于这个范围。再假设有列表`[p, q, r, s, t]`的切片，那么取出索引值为1至3的元素为`q`、`r`、`s`，而`t`不属于这个范围。

### 2.2.3.迭代器（Iterator）
迭代器是一个对象，它能顺序地访问列表中的元素，一般通过一个for循环来实现。下面的代码展示了如何利用for循环遍历一个列表：

```python
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    print(num * num)
```

输出结果为：

```
1
4
9
16
25
```

这里，for语句会自动生成一个迭代器，并调用它的__next__()方法来获取列表中的元素，直到所有元素被访问完毕。

## 2.3.列表的基本操作
### 2.3.1.创建列表
创建一个空列表，直接使用`[]`。

```python
empty_list = []
print(type(empty_list)) # <class 'list'>
```

创建一个非空列表，可以使用list()函数。

```python
non_empty_list = list(['apple', 'banana', 'orange'])
print(type(non_empty_list)) # <class 'list'>
```

还可以在创建列表时通过把不同的值用逗号隔开来初始化列表。

```python
my_list = ['hello', 123, True, None]
print(my_list) # Output: ['hello', 123, True, None]
```

### 2.3.2.添加元素
向列表中添加元素可以使用列表的append()方法。该方法接收一个参数，表示要添加的元素，并将该元素添加到列表的结尾。

```python
my_list = [1, 2, 3]
my_list.append(4)
print(my_list) # Output: [1, 2, 3, 4]
```

也可以使用extend()方法将多个元素一次性追加到列表的结尾。

```python
my_list = [1, 2, 3]
my_list.extend([4, 5])
print(my_list) # Output: [1, 2, 3, 4, 5]
```

### 2.3.3.删除元素
删除列表中某个元素有两种方式，第一种是使用remove()方法，该方法接收一个参数，表示要删除的元素，然后查找这个元素所在的位置，并将其替换成None。

```python
my_list = [1, 2, 3, 4, 5]
my_list.remove(3)
print(my_list) # Output: [1, 2, 4, 5]
```

第二种方式是直接根据索引值来删除元素。

```python
my_list = [1, 2, 3, 4, 5]
del my_list[2]
print(my_list) # Output: [1, 2, 4, 5]
```

### 2.3.4.更新元素
更新列表中的元素有两种方式。第一种是直接修改某个元素的值。

```python
my_list = [1, 2, 3, 4, 5]
my_list[2] = 10
print(my_list) # Output: [1, 2, 10, 4, 5]
```

另一种是使用pop()方法来更新列表中的元素。pop()方法默认删除并返回列表中的最后一个元素，但是也接受一个参数，表示要删除的元素的索引值。

```python
my_list = [1, 2, 3, 4, 5]
new_element = my_list.pop(2)
print("The new element is:", new_element)   # Output: The new element is: 3
print("After popping the element, the list becomes:", my_list)    # Output: After popping the element, the list becomes: [1, 2, 4, 5]
```