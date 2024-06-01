
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Python 是一种非常受欢迎的编程语言，被誉为“胶水语言”（batteries included）或者“Bells and Whistles Language”（符合你的需求），并且在各种领域都得到广泛应用。Python 拥有丰富的内置数据类型、模块化编程特性、动态类型系统、高级语法、自动内存管理等特性，极大地提升了编程效率，也方便了程序的开发。它是一款开源项目，其源代码可以在 Python Software Foundation 的官方网站上免费获取。
近几年，Python 在机器学习、Web 开发、数据分析领域取得了非常好的成绩。因此，本课程的目标就是对 Python 有深入理解和精通之后，能够帮助你顺利地进行实际的项目开发。
本文从以下几个方面对 Python 的基本知识做一个简单的介绍：
- 1.什么是 Python？
- 2.Python 发展历史及特点
- 3.为什么要用 Python？
- 4.Python 的优缺点
- 5.Python 运行环境配置方法
- 6.如何安装第三方库
# 2.核心概念与联系
## 什么是 Python?
Python 是一种多种编程语言中的一种，它最初由Guido van Rossum于1991年圣诞节期间，为了打造一种可免费使用的解释型计算机程序语言而创建，具有简单易懂的语法和语义，是一个非常适合作为初学者的语言。Python 的设计理念强调代码的可读性、易学性和互动性，使得其成为一种多范式语言，支持多种编程范式，包括命令式、函数式、面向对象等。
## Python 发展历史及特点
Python 发展史始于1991年 Guido van Rossum 和他的同事林荫Enumerable发起编写的一个开放源码的 interpreted language，该语言第一版发布于1994年1月1日。Python 是一款面向对象的动态语言，它的设计具有简洁、高效、动态等特点。Python 的最新版本为 Python 3.7.1，2018年10月份正式宣布进入维护阶段，并将在 2020 年底正式退出维护状态。
### Python 2 vs Python 3
Python 2 是当前版本较早的发布版本，Python 3 是 Python 的最新版本，两者之间存在一些不同之处：
#### Syntax Differences
Python 2 使用 print "hello" 来输出字符串，Python 3 使用 print("hello") 。

Python 2 默认使用 ASCII 编码，Python 3 默认使用 Unicode 编码。

Python 2 的整数除法运算结果是浮点数，而 Python 3 中整数除法运算结果是整数。

Python 2 不允许用户自定义异常类，而 Python 3 可以通过继承 Exception 类来定义自定义异常类。

Python 2 没有“True”或“False”关键字，取而代之的是“1”或“0”。

Python 2 中的 True 和 False 为非标识符变量，不能当作函数参数传入。

Python 2 中没有实现生成器表达式（即yield）。

Python 2 只有一种整数类型 int，没有整型、短整型、长整型之分。

Python 2 没有列表推导式（list comprehension）。

Python 2 中的 unicode 字符串以 u 作为前缀表示，而 Python 3 中的 unicode 字符串则不再需要此前缀。

Python 2 中的 str 和 bytes 类型无法区分，它们都是 bytearray 的别名。

Python 2 中不支持切片赋值。

Python 2 中 range 函数参数必须小于等于 2^31 - 1，超过这个值会报错。

Python 2 中没有枚举类型（enum type）。

Python 2 中不支持同时迭代多个可迭代对象。

Python 2 中 list() 函数可以接收任意类型的可迭代对象，但返回值的类型只能是列表。

Python 2 中没有比较两个集合是否相等的操作。

Python 2 没有实现“神奇数”（__pow__()、__lshift__()、__rshift__() 方法）。

Python 2 支持以“**”运算符指定字典扩展语法，但这种语法并不是常用的功能。

Python 2 中的 raw_input() 函数用于输入一行文本，而 input() 函数用于输入多行文本。

Python 2 中的 reload() 函数作用仅限于导入模块时。

Python 2 中 xrange() 函数生成一个迭代器对象，而不是列表。

Python 2 的无参装饰器语法 @decorator，而 Python 3 的无参装饰器语法 @decorator()。

Python 2 没有定义 tuple 的键值对方式 {key: value}，而 Python 3 中允许这样定义。

Python 2 中 map() 函数的参数数量只能是一元，而 Python 3 中 map() 函数的参数数量可以是一到两个，并且可以接受多个可迭代对象。

Python 2 中没有实现 with 语句，可以使用上下文管理器（context manager）来实现类似效果。

Python 2 没有实现异步 I/O。

Python 2 中单引号和双引号可以自由嵌套。

Python 2 没有实现协程。

Python 2 没有将字节数组视为不可变序列。

Python 2 没有实现反射机制。

Python 2 保留了“from module import *”，但是会忽略掉 module.__all__ 变量，而 Python 3 会使用 module.__all__ 变量来决定哪些成员应该被导入。

Python 2 中的异常处理语句可能会捕获一些看似不会引发异常的错误，比如文件句柄过多、内存分配失败等。

Python 2 的 round() 函数默认返回 float 数据类型，而 Python 3 的 round() 函数默认返回 int 数据类型。

Python 2 中垃圾回收机制采用引用计数的方法，而 Python 3 中改为标记清除的方法。

Python 2 对 Unicode 和字符集有更严格的要求，而 Python 3 对 Unicode 和字符集的要求更宽松。

Python 2 没有实现列出模块中所有的属性和方法。

Python 2 没有提供删除元素的高效方法。

Python 2 的标准库中的 struct 模块只能处理有符号的 C 结构体，而不支持无符号的 C 结构体。

Python 2 不支持 PEP 3115（增强可调用对象协议）。

Python 2 不能在 Windows 下使用。

Python 2 中 len() 函数计算容器中的元素个数，而 Python 3 中 len() 函数计算容器占用的内存大小。

Python 2 的 range() 函数的默认值是半开半闭区间，而 Python 3 的 range() 函数的默认值是全闭区间。

Python 2 的 open() 函数只能打开二进制文件，而 Python 3 中的 open() 函数也可以打开文本文件。

Python 2 不允许定义指向自己内部的循环链表，而 Python 3 可以。

Python 2 的 sys.getsizeof() 函数只计算对象占用的内存空间，而 Python 3 的 sys.getsizeof() 函数还包括对象所引用的其他对象的内存空间。

Python 2 提供的 zlib、bz2、lzma 压缩库比 Python 3 提供的 gzip、zipfile 压缩库更先进。

Python 2 中的 __metaclass__ 属性必须直接放在 class 后面，而 Python 3 中可以放在类的任何位置。

Python 2 中不支持 PEP 3102（单分支条件表达式）。

Python 2 中没有实现 GeneratorExit 异常。

Python 2 的 raise 语句只接受异常类型和异常实例，而 Python 3 的 raise 语句可以接受异常实例。

Python 2 没有内置的 deque 数据类型。

Python 2 没有支持 yield from 语法。

Python 2 中没有实现 pickling 和 unpickling。

Python 2 的 basestring 类型可以引用 str 或 unicode 类型，而 Python 3 的 basestring 类型已消失。

Python 2 没有实现符号表（symbol table）。

Python 2 没有实现垃圾收集器（garbage collector）插件接口。

Python 2 没有实现线程局部存储（thread local storage）。

Python 2 中没有实现生成器重连（generator reconnecting）。

Python 2 中的 unicodedata 模块提供了额外的字符类型信息。

Python 2 没有提供线程同步机制。

Python 2 中的 httplib 模块和 urllib 模块只能处理 HTTP 请求，而 Python 3 中的 http.client 和 urllib.request 都可以处理 HTTP 和 FTP 请求。

Python 2 中的 cPickle 模块速度慢且占用内存，而 Python 3 中的 pickle 模块性能好且占用内存少。

Python 2 的 os.path 模块不能访问 UTF-8 文件路径，而 Python 3 的 os.path 模块可以访问 UTF-8 文件路径。

Python 2 中的 unittest 模块支持以两种不同的风格编写测试用例，一种是以 class TestClass(unittest.TestCase) 形式编写，另一种是以 def test_function() 形式编写。

Python 2 没有实现导入子包的特性。

Python 2 中的全局变量 __name__ 会指向 "__main__"，而 Python 3 中的 __name__ 会指向模块名称。

Python 2 中没有实现真正的尾递归优化。

Python 2 没有实现元类。

Python 2 的 XMLRPC 和 SOAP 客户端库需要手动解析 XML 响应，而 Python 3 提供的 xmlrpc.client 和 suds 库可以自动解析 XML 响应。

Python 2 中的 uuid 模块不遵循 RFC 4122 规范。

Python 2 中的 repr() 函数会打印出对象所有属性，而 Python 3 中的 repr() 函数只会打印出对象的某个特定属性。

Python 2 没有提供异常链（exception chaining）。

Python 2 没有实现基于协程的信号量（semaphore）和事件（event）。

Python 2 的 doctest 模块只能用于文档测试，而 Python 3 的 doctest 模块可以用于单元测试。

Python 2 的 itertools 模块中的 repeat() 函数一次性生成所有重复的值，而 Python 3 的 itertools 模块中的 repeat() 函数可以生成指定的重复次数的值。

Python 2 没有提供 PEP 479（StopIteration 异常应该带上参数）。

Python 2 没有实现 goto 语句。

Python 2 没有实现不定长参数。

Python 2 的编码风格指南有明显的差异。

Python 2 没有解决 UnicodeDecodeError 错误。

Python 2 没有实现与字符串相关的随机数生成器。

Python 2 没有实现多进程共享内存。

Python 2 中的 subprocess 模块依赖于 shell 命令，而 Python 3 的 subprocess 模块不需要依赖 shell 命令就可以执行子进程。

Python 2 没有实现多线程共享内存。

Python 2 的 logging 模块只能记录普通的日志消息，而 Python 3 的 logging 模块可以记录各种类型的日志消息。

Python 2 的 traceback 模块不能显示非 ASCII 字符。

Python 2 的 datetime 模块只提供日期和时间的数据类型，而 Python 3 的 datetime 模块除了提供日期和时间的数据类型外，还提供时间的各个维度的数据类型（如时间戳）。

Python 2 中的 heapq 模块提供了堆排序算法，而 Python 3 的 heapq 模块提供了优先队列数据类型。

Python 2 的 selectors 模块只能用于基于 Select 模型的 I/O 复用，而 Python 3 的 selectors 模块可以用于基于 Poll、Epoll 或 Kqueue 的 I/O 复用。

Python 2 的 buffer() 函数可以创建一个新的缓冲区对象，但没有办法获得缓冲区对应的原始内存地址。

Python 2 没有实现弱引用（weak reference）。

Python 2 没有实现超时功能。

Python 2 没有实现虚拟环境工具。

Python 2 没有实现脚本化部署工具。

Python 2 没有实现分布式任务调度框架。

Python 2 没有提供数据库连接池。

Python 2 的 unittest 模块并不能像其他的语言那样智能地确定失败的测试用例，而且对于复杂的测试用例可能难以定位失败的原因。

Python 2 的 sqlite3 模块不能加密数据库文件。

Python 2 没有实现支持 IPv6 的 socket 模块。

Python 2 没有提供对线程本地数据的支持。

Python 2 的 zipfile 模块只能处理 ZIP 文件，而 Python 3 的 zipfile 模块可以处理各种格式的压缩文件。

Python 2 的 hashlib 模块只能计算 MD5、SHA1 哈希值，而 Python 3 的 hashlib 模块可以计算任意哈希算法。

Python 2 的 smtplib 模块只能发送纯文本邮件，而 Python 3 的 smtplib 模块可以发送 HTML 邮件。

Python 2 的 Tkinter GUI 框架需要手工编写 Tcl/Tk 代码，而 Python 3 提供的 tkinter GUI 框架只需要简单地声明控件即可构建用户界面。

Python 2 的 Crypto 库在安全方面还不够成熟。

Python 2 的 Tkinter 框架的兼容性较差。

Python 2 没有实现 TLS/SSL 支持。

Python 2 的 PyPI 服务器不安全。

Python 2 没有实现分布式计算。

Python 2 没有实现内存共享。

Python 2 的 asyncio 模块需要手动实现事件循环，而 Python 3 中的 asyncio 模块已经实现了事件循环。

Python 2 没有实现标准文件锁。

Python 2 的 Twisted 框架目前还不稳定。

Python 2 的 curses 库有许多已知的问题。

Python 2 没有实现物理机资源隔离。

Python 2 的 pydoc 模块存在 bug。

Python 2 的 BaseHTTPServer 模块是单进程的服务器，而 Python 3 的 http.server 模块是一个多进程的 HTTP 服务。

Python 2 没有实现基于 XML 的配置文件。

Python 2 没有实现异步网络 I/O。