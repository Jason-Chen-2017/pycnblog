                 

# 1.背景介绍


装饰器（Decorator）在Python中是一个非常强大的特性，它可以对函数、类或者方法进行额外的功能扩展。其主要作用包括实现面向切面的编程，将代码模块化，提高代码复用率，同时也增强了代码的可读性。本文将主要介绍Python中的装饰器机制。

装饰器是一种特殊的函数，它接受一个函数作为输入参数，返回一个新的函数。我们可以使用@符号来定义一个装饰器，并在其后面添加被装饰的函数名，如下所示：

```python
@decorator_name
def function():
    pass
```

例如，我们可以使用@property装饰器给属性添加get和set方法：

```python
class Person:

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError('Name must be a string')
        self._name = value
```

这里，Person类的name属性被@property装饰器修饰，生成了getter方法和setter方法。

除了常用的@property和@staticmethod装饰器之外，Python还提供了很多其他的装饰器用于扩展函数或类等。

# 2.核心概念与联系
## 2.1 什么是装饰器？
装饰器（Decorator）是指附加到另一个函数上的一个函数，目的是修改或增强该函数的行为。装饰器可以用来计时、监视函数的执行时间、记录日志、检查参数、统一异常处理等。以下是装饰器的一些典型应用场景：

1. 负责性能优化的@profile/@timeit/@cProfile
2. 用作缓存的@lru_cache/@cache
3. 为函数提供缓存支持的@cached_property/@lazy
4. 提供线程安全的@synchronized
5. 适配器模式的@functools.wraps
6. 为函数增加回调功能的@singledispatch/@singledispatchmethod/@multipledispatch

装饰器不仅仅是上述这些具体的装饰器，还有一些更通用的装饰器，比如@classmethod、@staticmethod等。

## 2.2 装饰器原理
装饰器最重要的一个作用就是修改被装饰的函数的功能或特性，这种修改是通过替换掉原函数对象来完成的，所以说装饰器其实就是一个函数，接受一个函数作为输入，返回一个修改后的函数，从而达到“修饰”的效果。如下图所示：


在调用被装饰函数时，首先会调用装饰器函数，再调用真正的被装饰函数，最后才是其余的代码。因此，装饰器的核心就是把原函数替换成自己的新函数，达到修改的目的。

我们可以通过inspect库获取被装饰函数的信息：

```python
import inspect

def decorator(func):
    print("Decorating:", func.__name__)

    def wrapper(*args, **kwargs):
        print("Running decorated function:", func.__name__)
        result = func(*args, **kwargs)
        return result

    return wrapper


@decorator
def example():
    """This is an example"""
    print("Hello world!")


print("Before decoration:")
example()
print("After decoration")
print("-" * 30)

print(inspect.getsource(example))
```

输出结果如下：

```python
Before decoration:
Decorating: example
Running decorated function: example
Hello world!
After decoration
------------------------------
def decorator(func):
    print("Decorating:", func.__name__)

    def wrapper(*args, **kwargs):
        print("Running decorated function:", func.__name__)
        result = func(*args, **kwargs)
        return result

    return wrapper
    
<BLANKLINE>

@decorator
def example():
    """This is an example"""
    print("Hello world!")
    

if __name__ == '__main__':
    example()