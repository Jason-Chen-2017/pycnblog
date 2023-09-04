
作者：禅与计算机程序设计艺术                    

# 1.简介
  

什么是yield？yield是一个Python关键字，它可以让一个生成器函数（Generator Function）直接产出一个值，并暂停它的执行，等待下一次调用继续运行。相比于普通函数的返回值，yield允许在函数执行过程中不断产生中间结果，并且最后一次产出的值可以被赋值给一个变量。

传统的基于迭代器的编程模型，比如Java中的Stream或Python中的generator，都无法实现实时的数据流处理，因此引入了异步非阻塞编程模型，通过消息驱动的方式将任务分派到多个线程或进程上执行。然而这种模型对于初级用户来说不够直观，使得调试和理解起来非常困难，需要对系统底层的原理有很好的理解才能编写出高效、可靠的程序。

为了帮助初级用户快速理解和使用yield关键字进行数据流处理，作者开发了一套Python工具包KerasFlow，包含实时机器学习、图像分类、文本情感分析等常用机器学习应用场景的示例程序。KerasFlow能够利用OpenCV获取摄像头视频流并进行简单的数据预处理，然后利用Keras深度学习框架训练卷积神经网络模型，实时识别摄像头前方物体的类别及位置。可以说，KerasFlow就是用最简单的形式展示了yield关键字的强大功能，并且结合了OpenCV、Tensorflow、Keras等现代开源技术，可以让初级用户快速理解并使用该语言特性。

为了更好地推广yield关键字，作者计划整合一些机器学习应用案例，并发布一系列入门教程、进阶课程和知识问答集锦，帮助更多的程序员快速入手yield关键字进行数据流处理，提升自身能力水平。这也是作者在知乎、CSDN等技术社区发布过相关文章的原因。

# 2.基本概念及术语
## 生成器函数（Generator Function）

生成器函数是一种特殊的函数，它使用了yield关键字而不是return关键字来返回一个生成器对象，这个生成器对象可以在for循环或者其他消费者中逐个产出元素。一般情况下，生成器函数会包含一个for循环，但是这不是必须的。如果函数没有yield语句，那么它也会变成一个正常函数，只不过它不会产出任何值，也不能作为生成器消费掉。

```python
def my_range():
    n = 0
    while True:
        yield n   # 每次调用my_range()函数时，都会产出下一个值
        n += 1
```

## 生成器表达式（Generator Expression）

生成器表达式类似于列表解析，但它返回的是一个生成器对象。生成器表达式可以把列表解析的[]改为()，从而创建出一个生成器函数：

```python
g = (x*x for x in range(10))    # 使用生成器表达式创建生成器
print(next(g))                 # 输出第一个值
print(next(g))                 # 输出第二个值
```

## 可迭代对象（Iterable Object）

一个可迭代对象（iterable object）是一个可以用于for...in循环的对象，通过__iter__()方法定义其迭代方式。除了list、tuple之外，所有的集合类型（set、frozenset、dict等）都属于可迭代对象。

```python
my_list = [1, 2, 3]       # 创建一个list
for i in my_list:
    print(i)              # 输出所有元素

my_str = "hello"          # 创建一个string
for c in my_str:
    print(c)              # 输出所有字符
```

## 生成器（Generator）

生成器是一个特殊的迭代器对象，它是可迭代对象的生成器函数。当生成器被创建时，它并不执行任何计算，而是在每次被请求提供数据的同时，保存当前的状态信息。当调用next()方法时，生成器会自动完成计算，并返回下一个可用值；如果生成器已经计算完毕，再调用next()就会抛出StopIteration异常。

```python
g = my_range()             # 创建生成器
print(next(g))             # 输出第1个值
print(next(g))             # 输出第2个值
```

## 消费者（Consumer）

一个消费者是指使用一个生成器的地方。对于生成器函数，一般来说都是由for循环来消费生成器，例如：

```python
def my_consumer():
    g = my_range()           # 创建生成器
    for value in g:          # 用for循环消费生成器
        print("Got", value)  # 对每个值做处理

my_consumer()               # 调用消费函数
```

对于生成器表达式，则可以直接作为可迭代对象来使用，也可以在with语句中使用，例如：

```python
g = (x*x for x in range(10))      # 创建生成器表达式
print(sum(g))                     # 输出求和结果

with open('file') as f:
    data = ''.join(line for line in f if 'hello' in line)
                                        # 在文件中查找字符串'hello'并读取内容
```