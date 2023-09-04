
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Python中的列表（list）、元组（tuple）、集合（set）、字典（dict）等数据类型都支持通过索引（index）访问元素值，比如列表的索引访问语法 list[i] 。那么如果要实现类似列表的随机访问，那就需要自定义一个类，让它支持随机访问功能，即可以通过下标（index）直接获取对应的值。本文介绍一种通过__getitem__()方法实现随机访问的类的设计。

# 2.相关概念
Python中类（class）的定义语法如下：
```python
class ClassName:
    class_suite
```
其中，ClassName表示类的名称，class_suite表示类的属性和方法。在这里，我们只需关注__getitem__()方法即可，它用于从对象中根据索引（index）获取值。该方法应该返回相应的元素值。

Python中对象的创建语法如下：
```python
obj = ClassName(args)
```
其中，obj是一个类的实例，args是该类的构造参数，可以不提供。

# 3.__getitem__()方法概述
Python中的内置函数len()用于计算序列的长度，而序列包括列表、字符串、元组、字典等。例如，当调用 len([1, 2, 3]) 时，返回的是 3；调用 len("hello") 时，返回的是 5。

同样地，类也支持len()函数，但对于非数字类型的对象（比如字符串），len()返回的是字符串的字符个数。为了统一序列的处理方式，Python引入了抽象基类collections.abc中的Sequence ABC，它规定了序列的接口协议。

Sequence ABC定义了两个方法：__getitem__()和__len__()。前者用于获取序列的元素，后者用于获取序列的长度。因此，如果需要实现可随机访问的数据结构，只需定义__getitem__()方法即可。由于此方法在Python中的作用非常广泛，所以通常会用索引（index）作为参数名，如__getitem__(self, index)。

# 4.__getitem__()方法实现
下面我们来实现一个随机访问的类的例子。

假设有一个包含若干整数的数组arr，现在想实现一个Array类，它可以使用负数索引访问其中的元素，即-n表示倒数第n个元素。该类的构造方法如下：
```python
class Array:

    def __init__(self, arr):
        self._data = arr
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self._data[index % len(self)]
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            # handle negative indices
            start = (start + len(self)) % len(self)
            stop = (stop + len(self)) % len(self)
            return [self[i] for i in range(start, stop, step)]
        
        raise TypeError('Invalid argument type')
        
a = Array([1, 2, 3, 4, 5])
print(a[-2:])   # output: [4, 5]
```

这个例子中，Array类继承自object类，并实现了自己的构造方法__init__()和自己的__getitem__()方法。__init__()方法用于初始化数组内容，而__getitem__()方法则是获取数组元素值的入口函数。

__getitem__()方法接受一个参数index，它可以是一个整数或者切片对象。如果index是一个整数，则代表了具体某个索引位置的元素，则该方法需要将index与数组长度进行取余操作，以防止越界访问。

如果index是一个切片对象，则代表了多个索引位置的元素，则该方法需要将切片参数转换成标准索引序列，然后再依次对每个索引位置访问数组元素值。

通过判断index的参数类型，该方法可以很容易地处理不同类型的索引请求。