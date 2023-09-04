
作者：禅与计算机程序设计艺术                    

# 1.简介
  

itertools是一个内置的Python模块，它提供了很多用于操作迭代对象的函数，包括迭代器（Iterator）、生成器（Generator），等等。在平时开发中，如果需要对序列进行一些高级的操作，比如组合、排序、筛选、转换，itertools提供的便捷函数会非常方便，本文将介绍一些最常用的函数。
# 2.基本概念
## 2.1 Iterator（迭代器）
迭代器（Iterator）是一个可以记住遍历的位置的对象。

一个Iterator对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。

Python的Iterator对象表示的是一个数据流，Sequence，List，Tuple等这样的可迭代对象。

可以用`iter()`函数或带`__iter__()`方法的对象创建Iterator对象。

例如，可以使用range()函数创建一个整数序列，然后用iter()函数获取它的Iterator对象：

```python
>>> it = iter(range(5))
>>> print(it)
<range_iterator object at 0x7f95fd9d8c30>
```

上面这个迭代器对象从0开始计数并迭代到4结束。

要获得Iterator对象中的下一个元素，可以调用`next()`方法，或者使用`__next__()`方法。

```python
>>> next(it)
0
>>> next(it)
1
>>> next(it)
2
>>> next(it)
3
>>> next(it)
4
>>> next(it)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

当所有元素都已经被迭代完，抛出`StopIteration`错误。

可以使用`for...in`循环来迭代Iterator对象，但是这种方法不是真正的迭代器模式，而只是把每个元素看做是序列中的一个元素。

```python
for x in range(5):
    print(x)
    
# Output: 
# 0
# 1
# 2
# 3
# 4
```

上面的例子也可以用Iterator的方式实现：

```python
it = iter(range(5))
while True:
    try:
        x = next(it)
        print(x)
    except StopIteration:
        break
        
# Output: 
# 0
# 1
# 2
# 3
# 4
```

可以看到，这两个实现方式完全相同。

## 2.2 Generator（生成器）
生成器（Generator）是一个返回迭代器的函数，只要把生成器的星号`*`放在函数名的前面，该函数就变成了一个生成器。

和普通函数不同，生成器是一个返回迭代器的协程函数，只能用于迭代操作，更简单点理解就是：

```python
def generator():
    yield 1
    yield 2
    return 'Done'
    
gen = generator()
print(type(gen)) # Output: <class 'generator'>

for i in gen:
    print(i)
    
# Output: 
# 1
# 2
```

这里我们定义了一个简单的生成器函数`generator`，其中有两个`yield`语句，每次执行`next()`方法，就会产出两次值。但是第三个`return`语句则是终止生成器的运行，之后再调用`next()`方法也会引发异常。

## 2.3 itertools模块
itertools模块包含了非常多的函数用于操作迭代器，包括：

- `count(start=0, step=1)`：无限序列，从`start`开始每次增加`step`。
- `cycle(iterable)`：无限重复序列，重复`iterable`序列无穷次。
- `repeat(object, times=None)`：重复`object`若干次，如果`times`为空，无限重复。
- `chain(*iterables)`：将多个迭代器链接起来，形成一个更大的迭代器。
- `groupby(iterable[, keyfunc])`: 根据某个键函数对迭代器进行分组，返回一个迭代器，每个元素是一个元组`(key, group)`，表示`key`对应的`group`。
- `ifilter(function or None, iterable)`：过滤掉`iterable`序列中不满足条件的元素，并返回一个新的迭代器。
- `ifilterfalse(function or None, iterable)`：过滤掉`iterable`序列中满足条件的元素，并返回一个新的迭代器。
- `islice(iterable, start, stop[, step])`：从迭代器中切片出指定范围的元素，并返回一个新的迭代器。
- `izip(\*iterables)`：返回一个迭代器，其内容是由对应位置的元素组成的元组。

这些函数都很容易理解，感兴趣的读者可以自己试试。

## 2.4 itertools示例
以下用几个例子来展示一下itertools模块的一些实用功能。

### 2.4.1 count()函数
count()函数创建一个无限序列，也就是说，这个序列中的元素是不断增长的，永远不会停止。

语法：

```python
count(start=0, step=1)
```

参数说明：

- `start`：起始值，默认为0。
- `step`：步长，默认为1。

例如：

```python
import itertools

c = itertools.count(1, 0.2) # 从1开始，每次递增0.2
for i in c:
    if i > 5: 
        break
    print(i)

# Output:
# 1
# 1.2
# 1.4
# 1.6
# 1.8
# 2
# 2.2
# 2.4
# 2.6
# 2.8
#...
```

### 2.4.2 cycle()函数
cycle()函数创建一个无限重复序列，也就是说，这个序列中的元素会无限重复，每次到结尾又重新开始。

语法：

```python
cycle(iterable)
```

参数说明：

- `iterable`：一个迭代器对象。

例如：

```python
import itertools

a = [1, 2, 3]
b = itertools.cycle(a)

for i in b:
    print(i)
    
    if i == 3: # 当第四个元素出现时退出循环
        break

# Output:
# 1
# 2
# 3
# 1
# 2
# 3
# 1
# 2
# 3
#...
```

### 2.4.3 repeat()函数
repeat()函数用来创建重复元素的序列。

语法：

```python
repeat(object, times=None)
```

参数说明：

- `object`：需要重复的值。
- `times`：需要重复的次数，如果为空，则无限重复。

例如：

```python
import itertools

r = itertools.repeat('hello', 3)

for s in r:
    print(s)
    
# Output:
# hello
# hello
# hello
```

### 2.4.4 chain()函数
chain()函数将多个迭代器链接起来，形成一个更大的迭代器。

语法：

```python
chain(*iterables)
```

参数说明：

- `*iterables`：多个迭代器。

例如：

```python
import itertools

a = [1, 2, 3]
b = ['x', 'y']
c = itertools.chain(a, b)

for x in c:
    print(x)

# Output:
# 1
# 2
# 3
# x
# y
```

### 2.4.5 groupby()函数
groupby()函数根据某个键函数对迭代器进行分组，返回一个迭代器，每个元素是一个元组`(key, group)`，表示`key`对应的`group`。

语法：

```python
groupby(iterable[, keyfunc])
```

参数说明：

- `iterable`：一个迭代器。
- `keyfunc`：一个函数，接受一个元素作为输入，并返回一个用于比较的键值。如果省略，则默认使用元素本身作为键值。

例如：

```python
import itertools

fruits = ['apple', 'banana', 'orange', 'pear', 'peach', 'pineapple', 'grape']

for fruit, group in itertools.groupby(sorted(fruits), lambda x: len(x)):
    print(fruit, list(group))
    
   # Output: 
   # 3 ['pear']
   # 5 ['banana', 'grape', 'orange', 'peach']
   # 6 ['apple', 'pineapple']
```

上面的代码首先将列表按长度排序，然后使用groupby()函数，按照每种长度的元素生成一个组，最后输出组中元素及其数量。

### 2.4.6 ifilter()函数
ifilter()函数过滤掉`iterable`序列中不满足条件的元素，并返回一个新的迭代器。

语法：

```python
ifilter(function or None, iterable)
```

参数说明：

- `function or None`：函数对象，用于测试元素是否满足某些条件。如果该值为None，则测试元素是否为真值。
- `iterable`：一个迭代器。

例如：

```python
import itertools

# 计算偶数的平方
squares = []
for n in filter(lambda x: x % 2 == 0, range(10)):
    squares.append(n**2)
    
print(squares)

# Output:
# [0, 4, 16, 36, 64]

# 如果不使用filter()函数，可以用如下方式实现：
squares = []
for n in range(10):
    if n % 2 == 0:
        squares.append(n ** 2)
print(squares)

# Output:
# [0, 4, 16, 36, 64]
```

以上两种方式结果都是一样的，但是使用filter()函数相对于使用循环来判断条件和添加元素更加简洁、易于阅读。

### 2.4.7 islice()函数
islice()函数从迭代器中切片出指定范围的元素，并返回一个新的迭代器。

语法：

```python
islice(iterable, start, stop[, step])
```

参数说明：

- `iterable`：一个迭代器。
- `start`：起始索引。
- `stop`：结束索引（不含）。
- `step`：步长。

例如：

```python
import itertools

fruits = ['apple', 'banana', 'orange', 'pear', 'peach', 'pineapple', 'grape']

sliced_fruits = itertools.islice(fruits, 3)

for fruit in sliced_fruits:
    print(fruit)
    
# Output:
# apple
# banana
# orange
```

上面的代码将列表`fruits`切片成由三个元素组成的子列表，并将其打印出来。

### 2.4.8 izip()函数
izip()函数返回一个迭代器，其内容是由对应位置的元素组成的元组。

语法：

```python
izip(*iterables)
```

参数说明：

- `*iterables`：多个迭代器。

例如：

```python
import itertools

colors = ['red', 'green', 'blue']
sizes = ['S', 'M', 'L']

for color, size in itertools.izip(colors, sizes):
    print('{} {}'.format(color, size))

# Output:
# red S
# green M
# blue L
```

这段代码用zip()函数来合并两个列表，但是由于zip()函数要求参数长度一致，所以有时候可能需要先对参数进行处理才能正确地合并。

但使用izip()函数不需要处理参数的长度，因为izip()函数自动匹配参数个数。