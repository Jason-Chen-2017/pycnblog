
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　Python作为一种高级编程语言，拥有广泛的应用领域，从数据科学到机器学习、web开发等众多领域都有着广泛的需求。Python中使用并行计算，可以充分利用多核CPU资源，实现更加高效率的运算。在Python3.4版本引入了asyncio模块，使得编写异步代码更加容易。本文将详细介绍Python中基于asyncio模块实现并行编程的基础知识。

　　# 2.Parallel programming with asyncio 
　　## 2.1 Introduction to Parallel Programming 
　　　　Parallel computing is a technique used for processing large data sets by dividing them into smaller sub-tasks and executing them concurrently on multiple processors or cores of the same computer. It has been an active research area in Computer Science for years, but it was not until recent times that parallel programming languages have become popular as they make writing high performance applications easier. Since then there are several different approaches to parallelization, ranging from multi-threading to distributed systems. In this article we will focus on parallel programming using Python's asyncio module. 

　　## 2.2 Asynchronous I/O (async / await) 
　　　　Asynchronous programming is a paradigm where tasks or operations can be executed independently without blocking each other. This means that while one task waits for another to complete, the current task can execute something else. The main idea behind asynchronous programming is that when dealing with slow I/O operations such as reading files, waiting for network requests, etc., our application does not freeze and becomes responsive during those periods. Another advantage of async IO is that it allows us to use non-blocking functions which don't block the event loop. This makes it possible to handle many simultaneous connections or events in a single thread. 

　　In Python, asyncio provides support for creating asynchronous programs. Asyncio is built upon the concept of generators and coroutine functions which allow us to write asynchronous code using synchronous syntax. Coroutines are a lightweight way to run CPU bound tasks asynchronously by suspending execution instead of switching threads. These coroutines can yield control between each other, allowing the scheduler to switch back to the main program at any time. 

　　AsyncIO modules are divided into three parts:

　　　　1. Event Loop - Responsible for scheduling and managing tasks and callbacks.

　　　　2. Coroutine objects - Act like normal function calls but pause their execution and return a future object.

　　　　3. Future objects - Represent results of coroutine computations that haven't yet completed. 
　　　　To create an asyncio program, you need to define your coroutine functions using the @asyncio.coroutine decorator. Here's an example: 

```python
import asyncio

@asyncio.coroutine
def my_coroutine():
    print('Hello')
    result = yield from some_other_coroutine()
    print(result)
    
loop = asyncio.get_event_loop()  
task = loop.create_task(my_coroutine())  
loop.run_until_complete(task)   
``` 

In this example, `some_other_coroutine()` is called within `my_coroutine()`. When the first line of `my_coroutine()` is reached, the coroutine returns immediately, letting the event loop know that further execution is needed. Then, the event loop schedules the execution of `my_coroutine()` along with its dependencies `some_other_coroutine()`. During this process, the event loop monitors all running tasks and switches between them as necessary based on the needs of the currently executing task. Once `my_coroutine()` completes, its future object is resolved and its result is printed out. After printing 'Hello', the script resumes the execution of the remaining statements after the loop terminates normally.  


The above example demonstrates how to call simple coroutine functions inside other coroutine functions using yield from statement. However, asyncio also supports more advanced features such as timeouts, cancellation, and exception handling. You should consult the official documentation for more details about these features.

