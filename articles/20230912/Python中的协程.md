
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
协程（coroutine）是一个子程序，或者一个函数，可以暂停执行、保存当前状态、接着从离开处继续运行。这种特性使得在单线程里实现多任务变得简单，因为每个任务都可被视为一个独立的协程。由于协程有很多优点，比如可以很方便地实现非阻塞IO、并发计算、分布式计算等功能，所以越来越多的语言都提供了对协程的支持。本文将介绍Python中的协程。
## Python协程的实现方式
Python中的协程的实现方式主要有两种，即生成器函数和异步I/O模型。下面分别介绍这两种方法。
### 生成器函数
生成器函数（generator function）是一种特殊的函数，它通过“yield”语句返回一个迭代器对象，这个迭代器会在每次调用next()时产生一个值。而调用方可以利用for循环或其他迭代器协议（如iter()）逐个获取这些值。这里有一个简单的示例：
```python
def simple_coroutine():
    i = 0
    while True:
        print('Yielding:', i)
        x = yield i
        if x is not None:
            i += x
        else:
            return


coro = simple_coroutine()
print(next(coro))   # Output: Yielding: 0
print(coro.send(2))    # Output: Yielding: 2
                    #          17
print(coro.send(10))   # Output: Yielding: 12
                    #          22
print(coro.close())     # Output: StopIteration (raised in generator code)
```
上面的例子中定义了一个名为simple_coroutine的生成器函数，该函数打印数字i的值，并且每隔一次迭代就中断等待，等待外界的代码发送消息。调用方可以使用for循环来逐个获取yield语句返回的值。当调用方的代码完成后，可以通过调用close()方法来通知生成器函数结束。
生成器函数相比一般函数最大的特点就是，它可以暂停并恢复执行。因此，对于需要长时间执行的任务来说，生成器函数是非常好的选择。另一方面，生成器函数也是一种更高级的编程抽象，它允许用户自定义协程的行为。
### asyncio模块
Python内置的asyncio模块提供了另外一种实现协程的方式——异步I/O模型。asyncio模块提供的协程是基于事件驱动的模型，在事件发生时，协程会自动切换到另一个待执行的协程，因此协程之间不会互相抢占CPU资源。其接口类似于线程，但由事件驱动模型驱动，可以同时处理多个任务。asyncio模型的协程也具有迭代器协议，可以被用于for循环。下面是一个示例：
```python
import asyncio

async def coro1():
    for i in range(10):
        await asyncio.sleep(1)
        print("coro1:", i)

async def coro2():
    for j in range(5):
        await asyncio.sleep(2)
        print("coro2:", j)

loop = asyncio.get_event_loop()
tasks = [loop.create_task(coro1()), loop.create_task(coro2())]
try:
    loop.run_until_complete(asyncio.wait(tasks))
finally:
    loop.close()
```
这个例子展示了如何创建两个异步协程，然后使用asyncio.wait()方法启动它们。asyncio.wait()会返回一个Future对象，代表这两个协程的所有工作已经完成。最后，我们调用loop.run_until_complete()来运行事件循环，直至所有的任务完成。