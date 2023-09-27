
作者：禅与计算机程序设计艺术                    

# 1.简介
  
：Netty是一个异步事件驱动的网络应用程序框架，它提供了对TCP/IP协议族中高级应用层（比如HTTP、FTP、WebSocket等）的支持。其设计目标就是提供一个易于理解、快速开发的网络应用框架。Netty是一个开源项目，托管在Github上，地址为：https://github.com/netty/netty 。

本书从三个角度对Netty进行全面的剖析：功能、实现原理和架构。通过对Netty各个模块的介绍，读者可以更加深入地理解Netty及其工作原理。

读完本书，读者将能够掌握：

1. Netty的功能特性；
2. 基于NIO和Epoll的事件驱动模型；
3. 流水线(Pipeline)架构模式；
4. Java NIO组件Selector、Channel、Buffer的作用和使用方法；
5. Netty的高性能实现机制（包括Reactor线程模型、内存池管理、零拷贝技术、优化参数设置等）。
6. 系统设计技巧（如微内核设计模式、RESTful API设计模式）。

本书可作为高级技术人员、Java开发人员或架构师的必备工具书，也是开源世界里最具参考价值的技术书籍之一。

# 2. 概念、术语、概念

## 2.1 Netty概述

### 2.1.1 异步非阻塞IO库

Netty是一个异步非阻塞I/O库。它提供了对TCP/IP协议族的高级设计和应用支持。它主要负责以下几方面：

1. 提供了一种异步事件驱动的网络应用程序框架，用来开发高性能、高并发的网络应用程序。
2. 提供了Java NIO组件中的SocketChannel、ServerSocketChannel、Selector、ByteBuffer等类和接口的实现，使得JDK的I/O编程接口符合Netty的要求。
3. 为开发人员提供了零拷贝（Zero-copy）技术。
4. 为开发人员提供了TCP/UDP套接字、文件、数据库连接等资源的管理能力。
5. 提供了针对大数据量实时应用的扩展性和高吞吐量性能。

总体来说，Netty是构建真正意义上的“异步I/O”程序的基础框架。它可以在多线程环境下，无需使用锁或竞争条件即可安全、高效地处理复杂的通信和I/O请求。Netty架构具有高度灵活性、可扩展性和可伸缩性，可以轻松应付各种网络传输场景下的需求。

### 2.1.2 Netty模块结构

Netty的模块化架构非常适合开发复杂、分布式、高性能的应用。Netty由以下几个子模块构成：


**Netty Core**：Netty核心模块，包括网络调用、编解码、缓冲区、事件和handler接口定义。该模块依赖第三方库如ReflectASM、Guava等。

**Netty Codec**：Netty编解码模块，包括对Protobuf、Thrift、JSON、XML等二进制和文本数据编解码器支持。

**Netty Transport**：Netty传输模块，包括通用TCP/UDP客户端、服务端，以及SSL/TLS、HTTP代理支持。

**Netty HTTP Client**：Netty的异步HTTP客户端模块，实现了HTTP/1.x、HTTP/2协议，支持同步、异步和事件驱动的API。

**Netty WebSocket**：Netty的WebSocket客户端和服务器模块。

**Netty AIO**：Netty的异步IO实现模块，可以使用AIO方式访问文件系统、网卡、键盘鼠标等底层资源，有效提升性能。

**Netty Examples**：Netty的示例模块，展示了如何利用Netty构建各种各样的应用，如路由、代理、负载均衡、文件传输、游戏服务器等。

**Netty Tools**：Netty的工具模块，包括调试工具、监控工具、安全工具等。

## 2.2 术语

### 2.2.1 Channel

在Netty中，Channel是对底层操作系统网络接口的抽象，包含用于发送和接收数据的双向缓冲区，负责缓冲字节流的数据，并在必要时执行特殊处理（如压缩、加密等）。Channel接口的设计目的如下：

1. 将底层实现解耦，屏蔽不同传输协议之间的差异。
2. 在一个Channel实例上可以注册多个事件监听器，通过它们可以获取到Channel的状态变化、数据读写完成等信息。
3. 支持异步非阻塞的IO操作，避免线程切换，提高吞吐量。

Channel接口由两个核心方法组成：read() 和 write()。当某个线程发起一个write操作时，Netty会立即返回，不等待实际写入完成，之后再通知用户写入结果。同样，当某个线程发起一个read操作时，Netty也不会等待实际读取到数据，而是在读取到可用字节后立即返回给用户。这种异步、非阻塞的设计方式能够有效地提高应用的并发处理能力。

### 2.2.2 Buffer

Buffer是Netty提供的字节容器，用于存储待发送或者已收到的字节流。除了字节数组外，还可以通过其它形式的容器，如堆缓冲区Heapbuffer、直接缓冲区Direct buffer等。这些容器都是抽象的，用户无法直接访问到它们。Netty提供了两种类型的缓冲区，如下图所示：


Heapbuffer是堆上直接分配的缓冲区，它的好处是占用的空间固定，效率较高，缺点是需要垃圾回收，容易造成内存碎片。因此，建议只在本地使用。

Directbuffer是由JVM虚拟机直接分配的内存，由操作系统来管理，它的好处是不需要额外的内存分配和回收操作，读写速度快，缺点是占用的空间不确定，可能溢出。因此，建议只在本地使用，尽量减少内存碎片产生。

Netty的Buffer使用起来相当方便，只要声明相关变量，然后把数据写入到Buffer里面，就可以传递到远端的Channel。如果需要读取数据，则从Channel读取到Buffer里面。通过不同的Buffer实现可以达到不同的效果。例如，堆Buffer可以实现缓存，而直接Buffer可以实现零拷贝。

### 2.2.3 Handler

Handler是Netty的核心组件之一，用于处理所有I/O事件。Netty的应用逻辑都需要通过Handler来实现。Handler是Netty提供的一个接口，其设计目的如下：

1. 将一个复杂的业务逻辑切割成多个独立的Handler，降低耦合度。
2. 通过灵活的Handler组合的方式，灵活地搭建应用的网络处理链路。
3. 支持通过统一的接口，同时将请求处理的工作委托给不同的Handler实现。

Netty提供四种类型的Handler：

1. 基于注解的Handler：这是一种配置简单，但功能却强大的Handler实现方式。通过注解，开发者可以很容易地创建自己的Handler，然后通过配置文件来启用它。

2. ChannelInboundHandler：这种Handler处理从远端Channel收到的入站数据。

3. ChannelOutboundHandler：这种Handler处理从远端Channel写入出站数据的过程。

4. ChannelDuplexHandler：这是一种特殊的Handler，既处理入站数据，又处理出站数据。

Handler的生命周期如下图所示：


每个Handler的实现者都可以继承对应的HandlerAdapter，从而获得默认的方法实现。但一般情况下，开发者自己编写Handler的实现类，然后通过HandlerChain调用他们。

Netty提供了许多基础的Handler，例如用于编解码、异常处理、统计、过滤器、日志等。开发者也可以根据自己的业务需求，自行扩展Handler。

### 2.2.4 EventLoop

EventLoop是Netty运行时实体，通常对应于一个线程。每个EventLoop都包含一个Selector，用于监听注册在其上的Channel，并且调用相应的处理Handler。当一个Channel有可读数据时，就触发读事件，进而调用对应的Handler的read方法来读取数据。同样，当一个Channel有可写数据时，就触发写事件，进而调用对应的Handler的write方法来写入数据。因此，整个系统可以充分利用多线程、事件驱动、非阻塞I/O的优点，最大限度地提高吞吐量和并发能力。

EventLoop有两个重要的方法：run() 和 start()。run() 方法是启动事件循环的入口，在当前线程中运行，直到EventLoop退出。start() 方法在新线程中启动EventLoop，并自动启动selector。

Netty的设计理念是将一个Channel和一个EventLoop绑定，因此，一个Channel的所有IO操作都应该由绑定的EventLoop来处理，不能跨越线程边界。因此，为了满足这一要求，Netty采用线程安全的设计，所有的操作都必须由同一个线程进行。也就是说，所有涉及到Selector的操作，只能在同一个线程中进行。