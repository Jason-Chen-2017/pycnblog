
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


在 Python 中，对象的迭代有两种形式：一种是基于 iterable 对象（如 list、tuple 和字符串），另一种是基于 iterator 对象（比如 generator）。iterable 是可以被用于 for... in 循环或者其他可迭代协议的对象；iterator 是可以被用来调用 next() 函数获取下一个元素的对象。

列表、元组和字符串都是 iterable 对象，但是它们都没有自己定义自己的迭代器。Python 使用生成器函数创建 generator，这个函数返回一个可以迭代的对象。generator 的好处之一就是，它可以在每次需要的时候计算出下一个值，避免不必要的计算或占用过多的内存。另外，通过 yield 语句返回的值会在第一次调用 next() 函数时生成，然后保存在 generator 对象中，直到最后一个 yield 返回时结束。因此，虽然 generator 有自己独特的迭代方式，但它们也是 iterable 对象。

因此，使用 __iter__() 方法可以让自定义类变成可迭代对象，并返回一个 iterator 对象。这个方法必须返回一个实现了 __next__() 方法的对象。__next__() 方法应该返回序列中的下一个元素，并在所有元素都已经被迭代完毕时抛出 StopIteration 异常。

如下所示，最简单的实现 __iter__() 方法的方法是在类的头部定义一个列表作为数据存储容器，然后利用 enumerate() 函数将其转换为 iterator 对象：
```python
class MyList:
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(enumerate(self.items))
    
    def __getitem__(self, index):
        return self.items[index]
```

当然，也可以定义一个私有的变量 `_data` 来保存数据的引用，然后直接返回此变量作为 iterator 对象：
```python
class MyList:
    def __init__(self, data):
        self._data = data
        
    def __iter__(self):
        return self._data.__iter__()
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        return self._data[index]
```

两者的区别是，前者更加简洁易读，而后者允许外部代码修改数据。

# 2.核心概念与联系
## 2.1.迭代器模式
迭代器模式是一个设计模式，它提供了一种顺序访问集合对象元素的方式。这种模式特别适合于那些含有大量元素的数据结构，例如列表、链表等。

迭代器模式的主要角色如下：

1. Iterator（抽象迭代器）角色：这是一种接口，提供两个方法： hasNext() 方法检查是否还有下一个元素，getNext() 方法返回下一个元素。

2. ConcreteIterator（具体迭代器）角色：这是迭代器的实现，负责维护当前遍历位置，并且向外提供相应的元素。

3. Aggregate（抽象聚合类）角色：这是一种抽象类或接口，定义了创建迭代器的方法 createIterator() 。

4. ConcreteAggregate（具体聚合类）角色：这是聚合类的实现，负责实现 createIterator() 方法，该方法创建一个具体的迭代器对象。

图 1：迭代器模式结构

## 2.2.`__iter__()` 方法
Python 中的每一个类都可以有 `__iter__()` 方法，该方法返回一个 iterator 对象。Python 中的 iterator 对象表示的是一个惰性序列，只有在需要的时候才会计算它的元素。

对于可迭代对象（如列表、元组、字符串等），`__iter__()` 方法默认情况下返回自己本身。因此，对于这些可迭代对象，只要用户自己没有重载 `__iter__()` 方法，就可以像迭代器一样正常地使用。

对于不可迭代对象（如字典、整数等），`__iter__()` 方法会抛出 TypeError 异常。要使得一个对象成为可迭代对象，必须显式地定义 `__iter__()` 方法。

除了 `__iter__()` 方法之外，还可以使用其他方法（如 `__next__()` 方法）来实现迭代器。但是一般建议优先使用 `__iter__()` 方法，因为它更加高效，而且不会损失任何功能。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## 3.1.迭代器
迭代器是 Python 中用来表示可遍历元素的对象。迭代器协议规定了如何从集合中获取元素，并且只能通过接口来访问元素。

迭代器的特点是：

- 可以按需计算，无需事先知道集合的长度，更节省空间；
- 支持多次遍历同一集合，可以方便对集合进行切片、排序等操作；
- 提供统一的、标准的访问机制，可以屏蔽底层集合的具体实现；
- 在不同编程语言之间移植性较好，因为只要求实现 iterator 接口即可；

Python 中的 iterator 对象可以通过内置的 `iter()` 函数来创建。例如：

```python
for i in range(3):
    print(i)
print('--------')
it = iter([1, 2, 3])
while True:
    try:
        x = next(it)
        print(x)
    except StopIteration:
        break
```

输出结果：

```
0
1
2
--------
1
2
3
```

可以看出，`range()` 函数创建了一个可迭代对象，而 `[1, 2, 3]` 创建了一个迭代器对象。

通过迭代器，我们可以方便地对集合中的元素进行遍历。而对可迭代对象的遍历实际上也是对迭代器的遍历。

迭代器协议包括两个方法：`__iter__()` 方法返回一个 iterator 对象，`__next__()` 方法返回集合中的下一个元素，当集合中元素耗尽时，抛出 StopIteration 异常。

## 3.2.生成器表达式
生成器表达式 (Generator expression) 类似于列表推导式 (list comprehension)，但是生成器表达式返回的是一个生成器对象，而不是列表。这样，我们就能将生成器对象赋值给变量，并逐个取出元素，从而节约内存。

语法格式如下：

```python
genexp = (expression for item in iterable if condition)
```

其中，`expression` 表示将 `item` 转换为另一种类型的值的表达式；`iterable` 表示一个可迭代对象，如列表、元组、字符串等；`condition` 可选，表示对 `item` 进行条件筛选的表达式。

例如：

```python
numbers = [1, 2, 3, 4, 5]
squared_even_nums = (num ** 2 for num in numbers if num % 2 == 0)
print(type(squared_even_nums))   # <class 'generator'>
for num in squared_even_nums:
    print(num)                    # Output: 4 16
```

这里，我们用一个生成器表达式 `(num ** 2 for num in numbers if num % 2 == 0)` 来创建了一个生成器对象，并赋值给变量 `squared_even_nums`。

当我们对这个生成器对象调用 `next()` 方法时，它会自动执行表达式，并返回第一个满足条件的 `num`，也就是 `num=2`，再次调用 `next()` 方法时，它会重复执行表达式直到找到下一个满足条件的 `num`，即 `num=4`。当 `squared_even_nums` 生成的所有元素都被取光时，会抛出 `StopIteration` 异常。

生成器表达式可以替代列表推导式，效率更高，并且减少了内存开销。