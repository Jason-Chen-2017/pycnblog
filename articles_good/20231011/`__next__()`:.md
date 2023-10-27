
作者：禅与计算机程序设计艺术                    

# 1.背景介绍

：`__next__(self)`方法在迭代器协议中，被用来获取下一个元素，这一点很重要，因为它决定了迭代器能否遍历所有的元素。但实际上，很多人不清楚这个方法到底是怎么工作的。另外，如果需要实现自己的迭代器，还得要对这一方法有足够的了解和掌握。因此，本文从Python的内置迭代器及其迭代行为出发，剖析该方法的作用，然后再以自定义迭代器为例，给出它的定义及工作原理。 

迭代器是一个可遍历对象，它可以顺序访问集合中的元素，并且一次只访问一个元素，直到结束。常见的迭代器包括列表、元组、字符串等。在Python中，可以使用`iter()`函数将任意序列或集合转换成迭代器。

举个例子：

```python
>>> list_iter = iter([1, 2, 3])   # 创建列表迭代器
>>> tuple_iter = iter((4, 5, 6)) # 创建元组迭代器
>>> string_iter = iter('hello')  # 创建字符串迭代器
>>> for i in list_iter:          # 使用for循环遍历迭代器
	print(i)                    # 每次迭代都会输出下一个元素
	                               
	                               
	                               
	                               
	
```

每当执行完`for`循环后，迭代器会自动释放资源，不可继续使用。

# 2.核心概念与联系

1. 可迭代对象（Iterable）：能够返回一个迭代器的对象叫做可迭代对象。在Python中，如果一个对象实现了__iter__()方法，那么它就是可迭代对象。比如list，tuple，set等都是可迭代对象。
2. 迭代器（Iterator）：实现了__iter__()方法和__next__()方法的对象，称之为迭代器。当调用iter()函数时，返回的结果就叫做迭代器。
3. 生成器表达式：通过使用`yield`关键字创建的可迭代对象叫做生成器表达式。
4. 生成器函数：使用`yield`关键字定义的函数叫做生成器函数。
5. __next__() 方法：通过 next() 函数来获取下一个元素。可迭代对象的 `__next__()` 方法，应该返回下一个可迭代对象的元素。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解

当使用 `iter()` 函数创建一个迭代器时，会先尝试判断这个对象是否实现了 `__iter__()` 和 `__next__()` 方法。

- 如果实现了 `__iter__()` 方法，那么 `__iter__()` 方法必须返回一个新的迭代器对象，并设置好状态信息，方便 `__next__()` 方法的获取元素；

- 如果实现了 `__next__()` 方法，那么 `__next__()` 方法必须返回下一个元素，如果没有更多的元素，则抛出 `StopIteration` 异常。

# 4.具体代码实例和详细解释说明

## 4.1 迭代器协议的实现

先定义一个样例类 `MyIter`, 如下：

```python
class MyIter:
    def __init__(self):
        self.index = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < 10:
            num = self.index ** 2
            self.index += 1
            return num
        else:
            raise StopIteration
```

- 在 `__init__()` 方法中，初始化了一个变量 index，用于记录当前元素位置。

- 在 `__iter__()` 方法中，直接返回自身对象，表明自己是一个可迭代对象。

- 在 `__next__()` 方法中，首先判断当前元素位置是否小于10，如果是的话，计算当前元素平方值并返回，然后将当前元素位置加1。如果超过10，则抛出 StopIteration 异常。

测试一下：

```python
my_iter = MyIter()
for item in my_iter:
    print(item)
    
# Output: 
# 0
# 1
# 4
# 9
# 16
#...
# 81
```

通过 `__iter__()` 和 `__next__()` 方法，实现了一个最简单的迭代器。

## 4.2 Python内置迭代器的分析

Python 中的内置迭代器都继承于 `collections.abc.Iterator` 基类，而 `collections.abc` 是 Python 的标准库中提供抽象基类的模块。所以我们可以通过查看源码来学习各个迭代器的特点。

### 4.2.1 range()函数

range()函数返回一个迭代器，可以按照指定步长生成一系列的整数。

源代码如下：

```python
def range(*args, **kwargs):
    """
    Returns an iterator that generates a sequence of numbers, 
    starting from start (inclusive), and ending at stop - step (exclusive).
    If stop is not given, it defaults to start + 0.
    """
    if len(args) == 1:
        stop = args[0]
        start = 0
        step = 1
    elif len(args) == 2:
        start, stop = args
        step = 1
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise TypeError("range() requires 1-3 arguments")

    if'step' in kwargs or'start' in kwargs or'stop' in kwargs:
        msg = ("range() got some positional-only arguments "
               "passed as keyword arguments.")
        raise TypeError(msg)

    i = start
    nxt = i + step
    while nxt < stop:
        yield i
        i = nxt
        nxt = i + step
    if nxt == stop or step > 0:
        yield i
```

可以看到，`range()` 函数采用不同的参数形式，返回不同的迭代器类型。如果只有一个参数，表示迭代次数，步长默认为1，起始值为0。如果有两个参数，表示起始值和终止值，步长默认为1。如果有三个参数，表示起始值，终止值，步长。

通过阅读源码，我们发现，`range()` 函数内部采用一个 while 循环来生成迭代值，每次产生一个值之后，会更新当前值的索引，并紧接着产生下一个值。这样做的目的就是为了节省内存，避免生成过多无用的数字。

在 `__iter__()` 方法中，`range()` 函数返回 `self`，表示自己是一个可迭代对象。在 `__next__()` 方法中，每次循环都会调用 `_get_next()` 函数来获取下一个值，`_get_next()` 函数会根据当前索引值和步长计算下一个值。如果当前值比终止值小或者步长为负数，就会抛出 StopIteration 异常。

```python
def _get_next(self):
    i = self._index
    self._index = i + self._step
    try:
        value = self._func(i)
    except ValueError:
        self._index -= self._step
        raise StopIteration
    return value
```

`_get_next()` 函数主要逻辑如下：

- 根据索引值和步长计算下一个值；
- 如果计算结果超出范围，抛出 StopIteration 异常；
- 返回计算结果。

### 4.2.2 reversed()函数

reversed()函数返回一个反转后的迭代器。

源代码如下：

```python
@classmethod
def reverse(cls, data):
    """Create a new iterator that yields the elements of iterable in reverse order."""
    if isinstance(data, Sequence):
        getter = lambda x, y: operator.getitem(x, slice(-y, None))[::-1]
    elif isinstance(data, Iterator):
        getter = lambda x, y: itertools.islice(x, (-y)-1, None)[::-1]
    else:
        raise TypeError("Expected str, bytes, os.PathLike, Sequence, or Iterator, "
                        f"got {type(data).__name__}")
    return cls(_gen_getter(getter, data))
```

可以看到，`reversed()` 函数是一个类方法，通过传入可迭代对象或序列，返回反转后的迭代器。

在 `__iter__()` 方法中，`reversed()` 函数返回 `self`，表示自己是一个可迭代对象。在 `__next__()` 方法中，`reversed()` 函数会调用子迭代器的 `__next__()` 方法来获取下一个值，并把值进行反转处理。如果已经到达子迭代器末尾，则抛出 StopIteration 异常。

```python
def __next__(self):
    try:
        result = self._it.__next__()
    except StopIteration:
        if self._reversed:
            raise StopIteration
        result = next(self._it)
    if self._reverse_state:
        self._reverse_state = False
    else:
        self._reverse_state = True
    return result
```

`reversed()` 函数的 `__next__()` 方法会先调用子迭代器的 `__next__()` 方法来获取下一个值，并通过一个布尔变量 `_reverse_state` 来记录当前值是否已经反转过。如果当前值是反转过的，就会重新生成一个反转过的值，并跳过重复的值。

### 4.2.3 enumerate()函数

enumerate()函数接收一个可迭代对象，并返回一个迭代器，其中包含每个元素及其对应的索引值。

源代码如下：

```python
def enumerate(iterable, start=0):
    """Return an iterator that produces (index, element) tuples for items in iterable"""
    return zip(count(start), iterable)
```

可以看到，`enumerate()` 函数的参数 `start` 表示起始索引值，默认值为0。

在 `__iter__()` 方法中，`enumerate()` 函数返回 `self`，表示自己是一个可迭代对象。在 `__next__()` 方法中，`enumerate()` 函数调用 count() 函数生成一个计数器，并获取迭代器中对应的值，组合成元组 `(index, element)` ，返回给用户。如果迭代器已经到达末尾，则抛出 StopIteration 异常。

```python
def __next__(self):
    while True:
        self._current += 1
        if self._iterator is None:
            break
        try:
            elem = next(self._iterator)
            return (self._current, elem)
        except StopIteration:
            continue
```

`enumerate()` 函数的 `__next__()` 方法会在每次迭代的时候，先用 count() 函数创建一个计数器，并获得下一个可迭代对象的值。如果遇到空迭代器，则抛出 StopIteration 异常。否则，返回 `(index, element)` 元组。

### 4.2.4 map()函数

map()函数接收一个可迭代对象作为函数参数，并返回一个迭代器，其中包含应用该函数到每一个元素后的结果。

源代码如下：

```python
def map(function, *iterables):
    """Apply function to every item of one or more iterators"""
    if not all(isinstance(it, Iterable) for it in iterables):
        raise TypeError("All arguments must be iterable.")
    return itertools.starmap(function, zip_longest(*iterables))
```

可以看到，`map()` 函数接收一个函数和多个可迭代对象作为参数。

在 `__iter__()` 方法中，`map()` 函数返回 `self`，表示自己是一个可迭代对象。在 `__next__()` 方法中，`map()` 函数使用 `zip_longest()` 函数来合并多个迭代器，并使用 itertools 模块中的 starmap() 函数对合并后的结果进行映射运算。

```python
def __next__(self):
    while True:
        values = [None]*len(self._iters)
        for i, it in enumerate(self._iters):
            try:
                values[i] = next(it)
            except StopIteration:
                pass
        if any(values):
            return self._fun(*values)
        raise StopIteration
```

`map()` 函数的 `__next__()` 方法会首先初始化 `values` 列表，长度为可迭代对象个数。然后对每个迭代器，尝试调用 `next()` 函数来获取下一个值，并存储到 `values` 列表中。如果某个迭代器已经到达末尾，则用 `None` 代替相应的值。最后，判断 `values` 中是否存在 `None` 值，如果有，则代表所有迭代器均已到达末尾，会抛出 StopIteration 异常。如果所有迭代器都已生成值，则调用 `self._fun()` 函数来对这些值进行映射运算，并返回结果。

### 4.2.5 filter()函数

filter()函数接收一个可迭代对象，和一个函数，返回一个迭代器，其中包含对输入可迭代对象经过过滤后的元素。

源代码如下：

```python
def filterfalse(predicate, iterable):
    """Filter out falsey elements from iterable based on predicate"""
    if not callable(predicate):
        raise TypeError("predicate must be a callable")
    return itertools.compress(iterable, map(not_, itertools.repeat(predicate)))
```

可以看到，`filterfalse()` 函数接收一个函数和可迭代对象作为参数，并返回一个迭代器。

在 `__iter__()` 方法中，`filterfalse()` 函数返回 `self`，表示自己是一个可迭代对象。在 `__next__()` 方法中，`filterfalse()` 函数使用 `map()` 函数和 `itertools.repeat()` 函数来对输入可迭代对象进行过滤。

```python
def __next__(self):
    while True:
        for elem in self._it:
            if not self._pred(elem):
                return elem
        raise StopIteration
```

`filterfalse()` 函数的 `__next__()` 方法会尝试从 `self._it` 获取下一个值，并对这个值进行判定，如果值是假的，则会返回这个假值。如果所有值均假，则会抛出 StopIteration 异常。