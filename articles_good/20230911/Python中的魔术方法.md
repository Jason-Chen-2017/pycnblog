
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概念
在Python中，一个对象可以具有一些特殊的功能或者属性。这些特殊的功能可以通过魔术方法实现。所谓魔术方法就是在Python中定义的方法名以`__`开头和结尾，并且会自动地被调用。这些方法不属于类本身的任何一个成员函数，但它们实际上可以被认为是类的方法。
## 魔术方法列表
魔术方法如下：

1. `__init__(self[, args...])`:类的构造函数，在创建类的实例时被调用，用于初始化该实例的状态。比如，在`class MyClass:`中定义了`def __init__(self):`，当实例化`MyClass()`时，就会调用`__init__()`方法。
2. `__del__(self)`:类的析构函数，在销毁类的实例时被调用，通常用来做一些资源清理工作。比如，在`class MyClass:`中定义了`def __del__(self):`，当实例化`MyClass()`后，如果没有显式删除该实例，那么`__del__()`方法就被执行。
3. `__str__(self)`:类的转换函数，返回一个字符串表示当前对象的内容。
4. `__repr__(self)`:类的“调试”函数，返回一个更加详细的字符串表示当前对象的内容。
5. `__call__(self[, args...])`:类的调用函数，允许类的实例像函数一样被调用。
6. `__len__(self)`:类的长度获取函数，返回当前对象可迭代对象的长度。
7. `__getitem__(self, key)`:类的索引访问函数，通过key获得对应的值。比如，`list[i]`这种形式。
8. `__setitem__(self, key, value)`:类的索引赋值函数，通过key设置对应的值。比如，`list[i] = x`。
9. `__delitem__(self, key)`:类的索引删除函数，通过key删除对应的元素。比如，`del list[i]`。
10. `__iter__(self)`:类的迭代器生成函数，返回一个迭代器对象。当一个类的实例被用于for循环遍历时，就会调用这个方法。
11. `__next__(self)`:类的迭代器函数，从当前位置开始，顺序返回下一个值。
12. `__add__(self, other)`:类的加法运算符重载，当两个类的实例进行加法运算时，就会调用这个方法。
13. `__sub__(self, other)`:类的减法运算符重载，同上。
14. `__mul__(self, other)`:类的乘法运算符重载，同上。
15. `__div__(self, other)`:类的除法运算符重载，同上。
16. `__mod__(self, other)`:类的取模运算符重载，同上。
17. `__pow__(self, other[, modulo])`:类的幂运算符重载，同上。
18. `__lshift__(self, other)`:类的左移运算符重载，同上。
19. `__rshift__(self, other)`:类的右移运算符重载，同上。
20. `__and__(self, other)`:类的按位与运算符重载，同上。
21. `__xor__(self, other)`:类的按位异或运算符重载，同上。
22. `__or__(self, other)`:类的按位或运算符重载，同上。
23. `__neg__(self)`:类的取负运算符重载，同上。
24. `__pos__(self)`:类的取正运算符重载，同上。
25. `__invert__(self)`:类的取反运算符重载，同上。
26. `__eq__(self, other)`:类的等于比较运算符重载，同上。
27. `__ne__(self, other)`:类的不等于比较运算符重载，同上。
28. `__lt__(self, other)`:类的小于比较运算符重载，同上。
29. `__gt__(self, other)`:类的大于比较运算符重载，同上。
30. `__le__(self, other)`:类的小于等于比较运算符重载，同上。
31. `__ge__(self, other)`:类的大于等于比较运算符重载，同上。
32. `__getattr__(self, name)`:类的属性获取函数，获取不存在的属性时，就会调用这个方法。
33. `__setattr__(self, name, value)`:类的属性赋值函数，给不存在的属性赋值时，就会调用这个方法。
34. `__getattribute__(self, name)`:类的属性获取、获取、赋值时都要调用的函数，可以控制属性访问权限等。
35. `__enter__(self)`:类的进入上下文管理器时被调用，一般用在with语句中。
36. `__exit__(self, exc_type, exc_val, traceback)`:类的退出上下文管理器时被调用，一般用在with语句中。

除了以上介绍的几个魔术方法，还有很多其他的方法，比如`__name__`, `classmethod()`, `@staticmethod()`, `@property()`等，详情请看官方文档：https://docs.python.org/zh-cn/3/reference/datamodel.html#special-method-names 。

# 2.介绍
在日常开发过程中，我们可能经常碰到需要自定义一些功能，但是因为语言本身的限制导致无法实现，只能通过一些奇技淫巧的方式才能实现。比如说，通过元类来控制类的创建过程；通过类的继承关系来扩展类的功能；通过特殊方法来实现一些高级功能。那么，什么是魔术方法呢？魔术方法又如何工作呢？

# 3.魔术方法工作原理
## 初始化
首先，让我们看一下 `__init__()` 方法，这是类构造函数，是在创建类的实例时被调用的。
``` python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p1 = Person('Alice', 20)
print(p1.name) # Alice
print(p1.age)   # 20
```
上面是一个简单的例子，Person 是类的名称，`__init__()` 是构造函数，接收两个参数：`name` 和 `age`，并将其作为实例变量保存在 `self` 中。

当我们创建了一个 Person 的实例，例如 `p1 = Person('Alice', 20)` ，Python 会自动调用 `Person.__init__(p1, 'Alice', 20)` 来完成对 `p1` 的初始化。

## 删除
接着，让我们看一下 `__del__()` 方法，这是类的析构函数，可以在销毁类的实例时被调用，通常用来做一些资源清理工作。

比如，下面的例子展示了如何在类销毁时释放一些资源：
``` python
import time

class Resource:
    def __init__(self, id):
        self.id = id
    
    def __del__(self):
        print("Release resource", self.id)
        
res = Resource(123)
time.sleep(1)
del res  # 此处调用 __del__() 方法释放资源
```
输出结果：
```
Release resource 123
```

## 转换
有时候，我们想把一个类的实例转换成字符串，就可以用到 `__str__()` 方法。

比如，有一个类 `Student`，我们想打印出每个学生的信息，就可以定义 `__str__()` 方法。
``` python
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score
        
    def __str__(self):
        return "Name:%s Score:%d" % (self.name, self.score)
    
s1 = Student('Bob', 85)
print(s1)  # Name:Bob Score:85
```

## 调试
有的时候，我们想知道类的内部信息，就可以用到 `__repr__()` 方法。

比如，有一个类的实例，我们想获得他的内存地址，就可以定义 `__repr__()` 方法。
``` python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "<Point object at %d>" % id(self)

point = Point(1, 2)
print(point)      # <Point object at 0x10c8e31c0>
print(repr(point)) # <Point object at 0x10c8e31c0>
```

## 调用
有时候，我们希望类的实例能够像函数一样被调用，就可以用到 `__call__()` 方法。

比如，创建一个 `Multiply` 类，它接受多个数字，并返回所有数字的积：
``` python
class Multiply:
    def __init__(self, *args):
        self.numbers = args
        
    def __call__(self):
        result = 1
        for num in self.numbers:
            result *= num
        return result

multiply = Multiply(2, 3, 4)
result = multiply()    # 24
``` 

## 长度
有时候，我们希望获得类的实例的长度，就可以用到 `__len__()` 方法。

比如，有一个 `List` 类，我们想知道它的长度，就可以定义 `__len__()` 方法：
``` python
class List:
    def __init__(self, items=[]):
        self._items = items
        
    def append(self, item):
        self._items.append(item)
        
    def remove(self, item):
        self._items.remove(item)
        
    def clear(self):
        del self._items[:]
        
    def __len__(self):
        return len(self._items)

lst = List([1, 2, 3])
print(len(lst))        # 3
```

## 索引访问
有时候，我们希望能够通过索引访问类的实例的元素，就可以用到 `__getitem__()` 方法。

比如，有一个 `Dict` 类，我们想根据键获得值，就可以定义 `__getitem__()` 方法：
``` python
class Dict:
    def __init__(self, **kwargs):
        self._dict = kwargs
        
    def __getitem__(self, key):
        if key not in self._dict:
            raise KeyError(key)
        return self._dict[key]

d = Dict(a=1, b=2)
print(d['b'])     # 2
```

## 索引赋值
有时候，我们希望能够通过索引赋值类的实例的元素，就可以用到 `__setitem__()` 方法。

比如，有一个 `List` 类，我们想更新某个索引的值，就可以定义 `__setitem__()` 方法：
``` python
class List:
    def __init__(self, items=[]):
        self._items = items
        
    def __setitem__(self, index, value):
        self._items[index] = value

lst = List(['apple', 'banana', 'orange'])
lst[1] = 'peach'
print(lst[1])     # peach
```

## 索引删除
有时候，我们希望能够通过索引删除类的实例的元素，就可以用到 `__delitem__()` 方法。

比如，有一个 `List` 类，我们想删除某个元素，就可以定义 `__delitem__()` 方法：
``` python
class List:
    def __init__(self, items=[]):
        self._items = items
        
    def __delitem__(self, index):
        del self._items[index]
        
lst = List(['apple', 'banana', 'orange'])
del lst[1]        
print(lst)          # ['apple', 'orange']
```

## 迭代器生成
有时候，我们希望获得类的实例的迭代器，就可以用到 `__iter__()` 方法。

比如，有一个 `LinkedList` 类，我们想遍历里面的每一个节点，就可以定义 `__iter__()` 方法：
``` python
class Node:
    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next
        
class LinkedList:
    def __init__(self):
        self.head = None
        
    def add(self, data):
        node = Node(data)
        if self.head is None:
            self.head = node
        else:
            curr = self.head
            while curr.next is not None:
                curr = curr.next
            curr.next = node
                
    def __iter__(self):
        current = self.head
        while current is not None:
            yield current.data
            current = current.next
            
lst = LinkedList()
lst.add(1)
lst.add(2)
lst.add(3)
for i in lst:
    print(i)           # 1 2 3
```

## 迭代器
有时候，我们希望迭代类的实例，就可以用到 `__next__()` 方法。

比如，有一个 `FibonacciGenerator` 类，它返回斐波那契序列，就可以定义 `__next__()` 方法：
``` python
class FibonacciGenerator:
    def __init__(self):
        self.prev = 0
        self.curr = 1
        
    def __next__(self):
        prev, curr = self.curr, self.prev + self.curr
        self.prev, self.curr = curr, prev
        return prev
        
    def __iter__(self):
        return self
        
fg = FibonacciGenerator()
for n in fg:
    if n > 1000:
        break
    print(n)             # 0 1 1 2 3 5...
```

## 加法运算
有时候，我们希望自定义类的加法运算，就可以用到 `__add__()` 方法。

比如，有一个 `Vector` 类，我们希望能够进行向量相加，就可以定义 `__add__()` 方法：
``` python
class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2       # Vector(4, 6)
```

## 减法运算
有时候，我们希望自定义类的减法运算，就可以用到 `__sub__()` 方法。

比如，有一个 `Matrix` 类，我们希望能够进行矩阵相减，就可以定义 `__sub__()` 方法：
``` python
class Matrix:
    def __init__(self, rows):
        self.rows = rows
        
    def __sub__(self, other):
        m1 = []
        m2 = []
        for r1 in self.rows:
            for c1 in r1:
                m1.append(c1)
        for r2 in other.rows:
            for c2 in r2:
                m2.append(c2)
        
        s = [[m1[j*3+k]-m2[j*3+k] for k in range(3)] for j in range(3)]
        return Matrix(s)
        
m1 = Matrix([[1, 2, 3], [4, 5, 6]])
m2 = Matrix([[3, 2, 1], [6, 5, 4]])
m3 = m1 - m2            # Matrix([[[-2, 0, 2],[-2,-1,1]]])
```

## 乘法运算
有时候，我们希望自定义类的乘法运算，就可以用到 `__mul__()` 方法。

比如，有一个 `Polynomial` 类，我们希望能够进行多项式乘法，就可以定义 `__mul__()` 方法：
``` python
class Polynomial:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        
    def __mul__(self, other):
        p1 = reversed(self.coefficients)
        p2 = reversed(other.coefficients)
        product = [(a*b)%2 for a in p1 for b in p2]
        return Polynomial(product)
        
poly1 = Polynomial([1, 2, 3])
poly2 = Polynomial([4, 5, 6])
poly3 = poly1 * poly2              # Polynomial([4, 13, 28, 27])
```