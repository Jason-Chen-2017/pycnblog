
作者：禅与计算机程序设计艺术                    

# 1.简介
  

except Exception as e:（Python中捕获异常的正确方式）是当我们在处理异常时需要遵循的一个最佳实践。使用except Exception as e可以方便地捕获所有类型的异常，并且不会错过任何重要信息。通过这种方式，我们可以在程序的其他部分对这些异常进行进一步分析、定位和解决。

使用except后面的Exception关键字声明要捕获的异常类型，如果没有指定具体的异常类型，则默认捕获所有的异常。as e用于将捕获到的异常对象赋值给一个变量e，这样就可以在except块中访问该对象的属性和方法了。

在处理异常时，我们需要注意以下几点：

1.捕获恰当的异常：只有真正需要处理的异常才应该被捕获，不能捕获那些无关紧要的异常或者处理不必要的事情。
2.准确描述上下文：捕获到异常时，需要描述清楚发生了什么，让别人能够明白我们所处的环境以及为什么会出错。
3.记录错误日志：即便只是简单的打印错误消息也比完全忽略掉错误要好很多。记录错误日志可以帮助我们更快地找到问题所在，并快速解决问题。
4.细心体贴：一些语法上的小差错或逻辑上的误判都可能导致程序运行出错。做好充分的测试，细心体贴才能避免出现意想不到的问题。
以上就是正确处理异常的基本套路。

# 2.基本概念术语说明
## 2.1 try-except
try-except是Python中用于处理异常的两个关键词。它允许我们按照指定的顺序去尝试执行某段代码，并在某个位置发生异常时捕获该异常。当异常发生时，控制权就会转移到except子句中，然后可以处理该异常，也可以继续抛出该异常，让其继续向外传播。

如下是一个例子：
```python
try:
    x = int(input("Please enter a number:"))
    y = 1/x
    print(y)
except ZeroDivisionError:
    print("You cannot divide by zero!")
except ValueError:
    print("Invalid input.")
else:
    print("The program completed successfully without error.")
finally:
    print("Goodbye!")
```
这个例子首先会尝试输入一个数字，然后计算它的倒数。如果用户输入了一个非法值，比如字符串或者空值，那么ValueError就会被捕获，而不会引起程序崩溃；而如果用户试图除以零，则会触发ZeroDivisionError。else子句只会被执行，如果没有异常发生，表示整个程序成功完成。最后，finally子句总会被执行，无论是否有异常发生。

除了捕获异常之外，try-except还可以使用raise语句手动抛出异常。此外，可以用sys模块中的exc_info函数获取当前线程的异常信息，从而能够进一步分析异常。

## 2.2 raise语句
raise语句用来手动抛出异常。如果在一个函数内部需要抛出一个自定义异常，可以通过raise语句抛出该异常。

例如：
```python
def myfunc():
    if something_bad_happened:
        raise MyCustomError("Something bad happened")
        
class MyCustomError(Exception):
    pass
```
上述例子定义了一个名为myfunc的函数，该函数模拟了一个发生异常的场景。如果something_bad_happened标志为True，那么myfunc就会抛出MyCustomError异常。

## 2.3 sys.exc_info()函数
sys.exc_info()函数可以获取当前线程的异常信息，包括异常类型、异常对象、 traceback 对象。traceback 对象包含异常发生时的调用栈信息。

例：
```python
import sys

try:
    1/0
except:
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals).strip()
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line, exc_obj))
```
这段代码在一个 try-except 结构中捕获了一个 ZeroDivisionError 。之后，使用 sys 模块中的 exc_info 函数来获取异常信息。得到的exc_type、exc_obj和traceback对象可以用来分析异常，比如输出详细的错误信息。