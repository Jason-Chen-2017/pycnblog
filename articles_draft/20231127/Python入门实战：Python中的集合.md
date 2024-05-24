                 

# 1.背景介绍


Python 是一种高级的、面向对象的编程语言，它内置了一系列的数据结构，包括列表、元组、字典等，还提供了一些集合类（set）、序列（sequence）、迭代器（iterator）等数据类型。集合类和序列都可以用来存储数据，但两者在使用上有一些不同点。对于列表和元组来说，它们的元素都是按顺序存储的，并通过索引访问；而集合和序列则不一定有序，而且可以重复，因此可以更灵活地存储数据。

本文将会给读者介绍Python中最重要的两个集合类——集合和字典。本文不会涉及很多复杂的数据结构，如队列、栈、树等。文章的最后会结合代码实例和例子进一步完善这个集合类的使用方法，以及集合和字典的区别。

# 2.核心概念与联系
## 2.1 集合（set）
集合是一个无序且不可变的容器，其元素不能重复。集合通常被用来存储元素，但是可以用于任何需要用到集合数据的地方。和列表、元组类似，集合也可以用方括号([])或者大括号({})创建。

集合的两种创建方式如下：

1. 使用set()函数创建：

   ```python
   my_set = set([1, 2, 3]) # 创建集合
   print(my_set) # {1, 2, 3}
   ```

2. 使用大括号{}创建：

   ```python
   my_set = {1, 2, 3} # 创建集合
   print(type(my_set)) # <class'set'>
   print(my_set) # {1, 2, 3}
   ```

和列表一样，集合也支持下标操作和切片操作。

```python
>>> s = {1, 2, 3}
>>> s[0]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError:'set' object is not subscriptable
>>> for i in s:
    print(i)
    1
    2
    3
>>> s[-1]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: -1
>>> s[1:2]
{2}
```

不过，由于集合不能进行索引操作，所以不能使用`s[i]`的方式获取集合中的元素。

集合的基本操作包含添加、删除和查找元素，这些操作分别对应于`add()`、`remove()`、`pop()`、`clear()`等方法。

```python
>>> s = {1, 2, 3}
>>> s.add(4)   # 添加元素
>>> s          # {1, 2, 3, 4}
>>> s.remove(2)    # 删除元素
>>> s              # {1, 3, 4}
>>> s.pop()        # 默认删除第一个元素
1
>>> s              # {3, 4}
>>> s.clear()      # 清空集合
>>> s              # set()
```

集合的基本操作还有求交集、并集、差集，可以通过`intersection()`、`union()`、`difference()`等方法实现。

```python
>>> a = {1, 2, 3}
>>> b = {2, 3, 4}
>>> a.intersection(b)       # 求交集，返回一个新的集合 {2, 3}
>>> a & b                   # {2, 3}
>>> a | b                   # {1, 2, 3, 4}
>>> a.union(b)              
>>> a ^ b                   # 求异或，返回一个新的集合 {1, 4}
>>> a.symmetric_difference(b)   # {1, 4}
>>> a -= b                    # 将b中所有元素从a中移除
>>> a                        # {1}
>>> b -= a                    # {4}
>>> b                        # {3}
```

还可以使用`update()`方法将多个集合合并为一个新的集合。

```python
>>> c = {4, 5, 6}
>>> d = {'x', 'y', 'z'}
>>> e = {}
>>> e.update(c)             # 更新e，使得e={4, 5, 6}
>>> e                      # {4, 5, 6}
>>> e.update(['x', 'y'])     # 可以传入列表或元组
>>> e                      # {4, 5, 6, 'x', 'y'}
>>> e.update({'m': 7})       # 可以传入字典
>>> e                      # {4, 5, 6, 'x', 'y','m'}
>>> f = {9}                  # 创建新集合f
>>> e.update(f)             # 更新e，使得e={4, 5, 6, 'x', 'y','m', 9}
>>> e                      # {4, 5, 6, 'x', 'y','m', 9}
```

上面介绍了集合的一些基本操作。关于集合更多的方法，可以参考Python官方文档。

## 2.2 字典（dictionary）
字典（dict），顾名思义，就是键值对的“字典”。字典存储的是无序的键值对集合，字典的每个键值对用冒号分割，每个键值对之间用逗号隔开，整个字典放在花括号({})中。

字典的键可以是数字、字符串、元组等任意不可变类型的值。但键值可以是任何可变类型的值，包括列表、字典、集合等。

字典的基本操作包括添加、修改、删除键值对，以及根据键查找值。

```python
>>> d = {'name': 'Alice', 'age': 20, 'city': ['Beijing', 'Shanghai']}
>>> d['gender'] = 'female'    # 添加键值对
>>> d                       # {'name': 'Alice', 'age': 20, 'city': ['Beijing', 'Shanghai'], 'gender': 'female'}
>>> d['city'][0] = 'Tianjin'    # 修改字典中的元素
>>> d                           # {'name': 'Alice', 'age': 20, 'city': ['Tianjin', 'Shanghai'], 'gender': 'female'}
>>> del d['age']                 # 删除键值对
>>> d                            # {'name': 'Alice', 'city': ['Tianjin', 'Shanghai'], 'gender': 'female'}
>>> value = d['city']            # 根据键查找值
>>> value                         # ['Tianjin', 'Shanghai']
```

字典还支持字典的拼接操作，即把两个字典组合成为一个大的字典。

```python
>>> d1 = {'name': 'Bob', 'height': 170}
>>> d2 = {'age': 30, 'job': 'Engineer'}
>>> d3 = d1.copy()                # 拷贝d1作为新的字典d3
>>> d3.update(d2)                 # 把d2的内容合并到d3
>>> d3                             # {'name': 'Bob', 'height': 170, 'age': 30, 'job': 'Engineer'}
```

字典可以嵌套，即键也可以是另一个字典。字典的长度由键值对的个数决定，而不是像列表那样只能靠元素个数确定。另外，字典同样可以作为字典的键。