                 

# 1.背景介绍


迭代（Iteration）在计算机编程中经常被用来解决复杂的数据集合或循环问题。在 Python 中，提供了两种基本的迭代方式：一种是 for...in 结构，另一种是 while 循环。但一般来说，for 循环可以轻松地解决很多问题，所以很少需要用到 while 循环。

但是，还有一些场景下，while 循环更方便、更直接。比如，当要处理的集合元素个数不确定时，while 循环能够很好地配合条件表达式一起工作。比如，求一个整数 n 的阶乘，我们可以使用 while 循环来实现：

```python
n = int(input("请输入一个正整数: "))
fact = 1
i = 1
while i <= n:
    fact *= i
    i += 1
print(f"{n}! = {fact}")
```

这种方法虽然简单，但缺点也很明显：如果 n 比较大，计算量会非常大。因此，更多时候，我们会选择 for 循环，它能在处理固定数量的元素时更加高效、更直观。

还有其他的情况也可能用到 while 循环。比如，实现一个密码验证功能，用户输入密码后，服务器端接收到的是加密后的密码。如果要验证密码是否正确，则需要先解密再判断是否相同。那么，就可以使用 while 循环，对加密密码中的每一个字符进行解密和比较：

```python
encrypted_password = "ENCRYPTED-PASSWORD"
password = input("请输入密码: ")
index = 0
decrypted_password = ""
while index < len(password):
    decrypted_password += chr(ord(encrypted_password[index]) - ord('A') + ord(password[index]))
    index += 1
if decrypted_password == encrypted_password:
    print("密码正确")
else:
    print("密码错误")
```

这个例子里，while 循环用来对密码中的每一个字符进行解密，chr() 和 ord() 函数用来转换字符和 ASCII 码。实际上，这种加密和解密的方法并非加密技术本身，而是为了突出 while 循环的作用。

Python 中的另一种迭代机制就是生成器（Generator）。顾名思义，生成器不是真正的产生数据的值，而是在运行过程中逐步产生值的函数。它的特点就是一次只产出一个值，而且不需要占用大量内存空间。这是因为，每次 yield 时函数会暂停，等到下一次调用时才从停止位置继续执行。这样做的好处是节省了内存，提升了性能。

例如，我们可以定义一个无限序列的生成器：

```python
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1
```

这个函数创建了一个无限序列，通过循环不断增加 num 的值，并使用 yield 暂停并返回当前的 num 值。此外，注意到这里没有给 num 指定初始值，所以第一次调用 next(generator) 会抛 StopIteration 异常。如果要用 for...in 结构迭代无限序列，可以修改一下：

```python
g = infinite_sequence()
for _ in range(10):
    print(next(g))
```

这段代码首先定义了一个无限序列 g，然后用 for...in 来遍历前 10 个元素，每次 next(g) 返回对应的值。这里的 g 是生成器对象，而不是值。

# 2.核心概念与联系
## 2.1 什么是迭代？
在计算机编程中，“迭代”是一个概念。当需要重复执行某一操作时，迭代便派上用场。迭代通常分为两类：

1. 离散型迭代：比如，向量 (list, tuple) 或字符串都属于离散型迭代；
2. 有限型迭代：比如，迭代数字范围 (range) ，或者迭代容器内的所有元素都是可知的。

## 2.2 为何使用迭代？
迭代主要用于解决以下两个问题：

1. 避免代码重复：对于某些问题，比如遍历文件列表、读取网络数据、从数据库获取记录等，重复的代码编写使得代码库变得臃肿。而使用迭代可以简化代码，减少开发成本。
2. 提高代码效率：对于有限集合的遍历，迭代比普通的循环要快很多，因为它能跳过不必要的遍历。而且，它还能避免创建临时变量和对象，节约内存。

## 2.3 生成器与迭代器
在 Python 中，生成器（Generator）和迭代器（Iterator）是两个重要的概念。生成器和迭代器是两个不同的概念，但是它们之间又有着紧密的联系。

生成器和迭代器是相互关联的。生成器是一种特殊的函数，它使用关键字 `yield` 而不是 `return`，并且可以通过 `__next__()` 方法获取下一个值。迭代器则是一种协议，它规定应该如何访问某个对象中的元素，包括定义 `__iter__()` 方法让对象成为可迭代对象，定义 `__next__()` 方法让对象按照顺序返回下一个元素。

生成器是一种迭代器，它可以在内部保存状态，可以被迭代多次。对于耗时的运算，例如图像处理、网页爬取、数据库查询等，使用生成器能极大地提升效率。

生成器的典型应用如下：

- 在线统计日志：日志文件通常很大，如果一次性加载整个文件到内存的话可能会造成内存溢出。利用生成器的方式，可以边读边处理日志，只在需要时才将日志数据载入内存。
- 生产任意长度序列：比如，可以用生成器构造斐波那契数列。
- 数据流处理：如压缩文件或网络传输数据时，需要处理成批的数据，使用生成器可以降低内存占用。
- 模拟游戏行为：游戏引擎通常采用单线程模式，但通过协程（Coroutine）和生成器（Generator），可以支持多线程同时执行。

## 2.4 Python 迭代器协议
Python 中迭代器协议的描述如下：

- 对象实现 `__iter__()` 方法，该方法返回一个可迭代对象，即实现了 `__next__()` 方法的对象；
- 当一个可迭代对象使用 `next()` 方法获取下一个元素时，首先检查是否已经到了最后一个元素，若是，则抛出 `StopIteration` 异常；否则，返回该元素。

以字符串为例，`str.__iter__()` 方法返回一个迭代器，该迭代器维护当前位置，并实现 `__next__()` 方法，以字符形式返回当前位置字符。以下是示例代码：

```python
s = 'hello'
it = iter(s)   # 创建迭代器
print(type(it))    # <class'str_iterator'>
print(next(it))     # h
print(next(it))     # e
print(next(it))     # l
print(next(it))     # l
print(next(it))     # o
print(next(it))     # 抛出 StopIteration 异常
```

由于字符串是一个不可变序列，其长度已知，所以迭代器只有一个元素，即空字符串。当 `next()` 方法反复调用时，由于已经到达最后一个元素，所以抛出 `StopIteration` 异常。

## 2.5 何时应该使用生成器？
在 Python 中，建议优先使用生成器作为迭代器。原因如下：

1. 使用生成器可以提升性能，因为在创建迭代器时不会预留完整的内存空间；
2. 生成器具有更高的灵活性，可以在不同的情景下更换算法；
3. 生成器可以更容易控制数据流，如下载大文件时，可以限制缓冲区大小，防止内存占用过多；
4. 生成器更加易于测试和调试，因为它们会一步步生成数据。