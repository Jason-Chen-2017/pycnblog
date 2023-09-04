
作者：禅与计算机程序设计艺术                    

# 1.简介
  


“Error occurred:”这一句子本身表达了对某个错误信息的恐慌，因此需要打印出来看看具体原因。可不可以把打印语句放在try-except代码块中呢？这样就可以在程序出错时提醒用户查看错误信息。不过如果只是简单的print语句放在try-except代码块中会引起很多误解，比如说错误信息会被忽略掉而导致不易排查问题。因此要特别小心地去处理这个问题。

通过阅读本文，读者将会了解到以下知识点：

1.什么是Try-Except语法
2.Print语句的陷阱
3.如何正确的用法使用print函数输出错误信息
4.异常类型及其含义

# 2.Try-Except语法
```python
try:
    # try部份的代码
except ExceptionType as error_object:
    # except部份的代码，用于处理try部份代码中的特定类型的异常
finally:
    # finally部份的代码，无论是否发生异常都会执行
```
该语法由三部分组成，分别是try、except和finally。其中，try部分是必需的，表示的是程序可能出现的某些错误，如果try部分的代码发生了错误，则抛出指定的异常类型(ExceptionType)，并将具体的错误信息保存在error_object对象中。若没有发生错误，则直接跳过该部分。

except部分用于处理try部分代码中的某些特殊情况，当try部分代码发生指定类型的异常时，将自动运行except部分的代码，此处的ExceptionType可以是一个或多个具体的异常类。当遇到指定的异常类型，系统将捕获该异常，将具体的错误信息保存到error_object对象中，再运行相应的错误处理逻辑；若不处理该异常，系统将继续往上抛出，最终程序结束。

finally部分通常用来进行一些清理工作，包括关闭打开的文件等，无论异常是否发生都将执行该部分代码。

# 3.Print语句的陷阱

print语句是输出信息的基本命令，但当我们在try-except代码块中使用它的时候，可能会出现一些意想不到的问题。如下面的例子所示，当try部分出现了一个异常后，系统会自动跳转至except部分进行处理，但是由于print语句会强制终止当前线程的执行，所以错误信息不会被打印出来。

```python
def some_function():
    try:
        a = b + c   # 此处出现了一个NameError异常
    except NameError:
        print("Variable b is not defined")

some_function()     # 函数调用
```

解决这种问题的方法之一就是将print语句移动到except代码块外面，让它自然地跟随着except语句一起执行。另外一种方法是使用logging模块，它可以帮助我们记录日志，并且提供一些控制台输出的格式设置功能，还可以灵活地定制日志级别。

# 4.如何正确的用法使用print函数输出错误信息

正确的方式是将错误信息用print语句打印出来，即便是在try-except代码块中也是如此。正确处理方式如下：

```python
import traceback

try:
    # try部分的代码
except ExceptionType as error_object:
    traceback.print_exc()   # 使用traceback模块打印异常信息
finally:
    # finally部份的代码
```
traceback模块能够帮助我们获取具体的错误信息，并且按照格式显示，非常直观。

# 5.异常类型及其含义

一般来说，Python中的异常分为两类：

1. 内置异常（built-in exceptions）
2. 用户自定义异常（user-defined exceptions）

我们先来看一下内置异常。Python的所有内置异常都定义在内置模块exceptions中，详细列表如下：

- BaseException：所有异常类的基类。
- SystemExit：解释器请求退出。
- KeyboardInterrupt：用户中断执行（通常是输入^C）。
- GeneratorExit：生成器（generator）发生异常来通知退出。
- Exception：常规错误的基类。
- StopIteration：迭代器没有更多的值。
- ArithmeticError：所有的算术运算错误的基类。
- FloatingPointError：浮点计算错误。
- OverflowError：数值运算超出最大限制。
- ZeroDivisionError：除数为零。
- AssertionError：断言语句失败。
- AttributeError：试图访问一个对象的一个不存在的属性。
- BufferError：缓冲区相关的错误。
- EOFError：没有内建输入，到达EOF标记。
- ImportError：导入模块/对象失败。
- LookupError：查找元素错误。
- IndexError：序列中索引不存在。
- KeyError：字典中不存在该键。
- MemoryError：内存溢出错误。
- NameError：尝试访问一个还未初始化的变量。
- UnboundLocalError：访问未绑定局部变量。
- OSError：操作系统错误。
- ReferenceError：弱引用（Weak reference）试图访问已经垃圾回收了的对象。
- RuntimeError：一般的运行时错误。
- NotImplementedError：尚未实现的方法。
- SyntaxError：语法错误。
- IndentationError：缩进错误。
- TabError：Tab和空格混用。
- SystemError：一般的解释器系统错误。
- TypeError：传入对象的类型不匹配实参类型。
- ValueError：传入无效的参数。
- UnicodeError：Unicode相关的错误。
- Warning：警告的基类。
- UserWarning：用户代码生成的警告。
- DeprecationWarning：关于被弃用的特征的警告。
- FutureWarning：关于构造将来语义会有改变的警告。
- PendingDeprecationWarning：关于特性将会被废弃的警告。
- RuntimeWarning：可疑的运行时行为(runtime behavior)的警告。
- SyntaxWarning：可疑的语法的警告。
- ImportWarning：导入模块过程中可能出现的问题的警告。
- UnicodeWarning：可疑的Unicode编码的警告。
- BytesWarning：可疑的字节串的警告。

这些异常类型各有一个对应的英文单词，且每个异常的名字都具有特定的含义。比如，ZeroDivisionError就是指除数为零，这很容易理解。其他的异常类型比较复杂，这里就不一一列举了。