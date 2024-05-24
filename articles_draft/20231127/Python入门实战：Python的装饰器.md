                 

# 1.背景介绍


装饰器（Decorator）是Python中的一个重要特性，它可以用来修改已有函数的行为或提供额外功能。简单来说，装饰器就是一种特殊的函数，它接受被装饰的函数作为输入参数并返回一个新的函数。下面是一个简单的例子：

```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_whee():
    print("Whee!")

say_whee()
```

输出结果如下所示:

```
Something is happening before the function is called.
Whee!
Something is happening after the function is called.
```

上面的示例中，`say_whee()` 函数没有定义任何参数，而 `@my_decorator` 装饰器在 `say_whee()` 上方附着了一个装饰器函数，这个函数叫做 `wrapper`，它包裹了原始的 `say_whee()` 函数，并添加了两个打印语句。当执行 `say_whee()` 时，首先会调用 `wrapper()` 方法，然后才会进入到原始的 `say_whee()` 函数中。当退出 `say_whee()` 时，又会调用一次 `wrapper()` 方法，并且在退出前会输出另外的两句话。

显然，装饰器非常强大且灵活，可以用于许多场景，比如实现日志记录、监控性能、事务处理等。一般情况下，装饰器可以用以下方式编写：

```python
def decorator_name(function):
    # Do something here to the function
   ...
    return modified_function
```

其中，`decorator_name()` 是装饰器函数的名称，`function` 参数是要被修饰的函数。

# 2.核心概念与联系
## 2.1 Python中装饰器的位置
Python中有三种不同的位置可供装饰器使用：

1. 函数定义时
2. 函数调用时
3. 函数对象创建时

### 2.1.1 函数定义时
这种情况下，装饰器注解于函数定义的第一行，如下所示：

```python
@my_decorator
def greetings(name="world"):
    print("Hello " + name)
```

这种装饰器位置的限制性较小，适合单次装饰。但如果函数由多个装饰器包裹时，此时的装饰顺序取决于装饰器的定义顺序，即先应用最外层的装饰器，最后才能应用内层的装饰器。例如：

```python
@f
@g
def foo():
    pass
```

假设函数 `foo()` 有两个装饰器 `f()` 和 `g()`，那么，`foo()` 会先经过 `f()` 装饰器处理，再经过 `g()` 装饰器处理。也就是说，`f()` 的作用范围仅限于函数 `foo()` 中的第一个 `print()` 语句；而 `g()` 的作用范围则是整个函数体，包括其中的所有 `print()` 语句。

### 2.1.2 函数调用时
这种情况下，装饰器直接放在被装饰函数的括号后面，如下所示：

```python
greeting = my_decorator(greetings())
```

这种装饰器位置的限制性较大，适合单次装饰，但缺乏对函数对象的控制能力。因为此处的装饰器无法对函数的某些属性进行修改或控制。例如，如果你想动态设置某个函数的参数值，就无法使用这种装饰器位置的方式。因此，一般情况下，我们会选择使用函数定义时或者函数对象创建时两种方式。

### 2.1.3 函数对象创建时
这种情况下，装饰器通过调用 `functools.wraps()` 函数为被装饰函数增加一些元信息，并返回一个新的函数对象，如：

```python
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Some code before calling original function
        result = func(*args, **kwargs)
        # Some code after calling original function
        return result

    return wrapper

@my_decorator
def hello(name='World'):
    """Print a greeting"""
    print('Hello {}.'.format(name))

hello.__doc__   # Output: 'Print a greeting'
hello.__name__  # Output: 'wrapper'
```

这种装饰器位置的限制性不高，不建议使用。原因是由于该位置不受函数定义位置的影响，所以此时无法对函数的名字、文档字符串、参数列表等属性进行修改。如果需要这些属性的修改，可以使用其他两种装饰器位置。但是，该装饰器使用起来较为复杂，在实际应用中并不多见。

## 2.2 Python中装饰器的使用方式
装饰器既可以在函数定义时也可以在函数调用时使用，它也能以各种方式嵌套，形成复合装饰器。

### 2.2.1 函数定义时使用
函数定义时使用的装饰器会把装饰后的函数替换原来的函数对象，这样就不需要修改原始函数的代码。这种装饰器方法比较直观易懂，但是缺少灵活性。

例如：

```python
@my_decorator
def add(x, y):
    return x + y

print(add(1, 2))     # Output: Something is happening before the function is called.
                    #          Calculating...
                    #          3
                    #        Something is happening after the function is called.
```

### 2.2.2 函数调用时使用
函数调用时使用的装饰器不会修改原函数的源代码，而是返回一个新的函数对象，这个新函数对象在运行期间会自动调用原始函数。这种方法较为隐蔽，但灵活性较好。

例如：

```python
def log(func):
    import logging
    
    logger = logging.getLogger(__name__)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info('Calling %s', func.__name__)
        return func(*args, **kwargs)
    
    return wrapper
    
@log
def add(x, y):
    return x + y

print(add(1, 2))    # Output: Calling add
                   #         Hello world
```

上面这个例子中，`log()` 装饰器接收原始函数 `add()` ，并返回一个新的函数对象 `wrapper()` 。这个新函数在运行期间会自动调用原始函数，并输出一条日志消息。在这个例子里，日志级别设置为 INFO ，表示只记录 INFO 及以上级别的信息。你可以根据你的需要设置不同的日志级别。

### 2.2.3 混合使用装饰器方法
你还可以使用混合的方法，比如函数定义时加装饰器，函数调用时加装饰器，或者两者结合。例如：

```python
def trace(func):
    import timeit
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t1 = timeit.default_timer()
        result = func(*args, **kwargs)
        t2 = timeit.default_timer()
        print('{} took {:.3} seconds'.format(func.__name__, t2 - t1))
        
        return result
    
    return wrapper


@trace
def sleepy(seconds):
    time.sleep(seconds)


sleepy(2)      # Output: sleepy took 2.000 seconds
```

这个例子中，`trace()` 装饰器接收原始函数 `sleepy()` ，并返回一个新的函数对象 `wrapper()` 。这个新函数在运行期间会自动调用原始函数，并记录函数的执行时间。在这个例子里，只记录函数执行的时间，但你还可以记录更多的内容，比如函数调用的参数值。