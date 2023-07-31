
作者：禅与计算机程序设计艺术                    

# 1.简介
         
Web Socket（通常缩写为WS）是HTML5一种协议，它实现了浏览器与服务器之间全双工(full-duplex)的实时通信（Real-time Communication）。WebSocket使得客户端和服务器之间的数据交换变得更加简单，允许服务端主动向客户端推送数据。在WebSocket API中，浏览器和服务器只需要完成一次握手，两者之后就直接可以互相发送和接收数据。

WebSocket 是HTML5 定义的一种网络通信协议，它是一种双向通信通道，建立在 TCP/IP 协议之上，目的是为了能更好地进行网页间的通信。WebSocket 在建立之后，服务器和客户端都能主动的将数据发送给对方。这种方式不同于 HTTP 请求-响应模式的通信方式，建立在真正的 TCP/IP 协议基础上的 WebSocket 提供了持久性的连接，能够更好地节省带宽及 server 的资源开销。另外，WebSocket 可用于游戏领域、IM 即时通讯场景、股票行情推送等多种应用场景。

本文主要从以下三个方面对WebSocket进行介绍:

1. WebSocket的基本概念和用法；

2. WebSocket的应用场景及优缺点分析;

3. WebSocket的安全性保障机制。



# 2.基本概念及术语说明
## 2.1 WebSocket的基本概念
WebSocket 是HTML5定义的一个基于TCP的协议。

它是一种双向通信通道，它的特点如下：

1. 建立在 TCP/IP 协议基础之上，提供可靠的数据传输能力。

2. 数据传输使用帧的形式进行。

3. 使用标准化的端口，不同的浏览器、操作系统可能采用不同的端口号，但都遵循ws（WebSocket协议）或wss（WebSocket Secure协议）。

4. 支持扩展，支持对数据的压缩、加密等功能的定制。

5. 可以发送文本、XML、JSON、二进制文件等任意格式的数据。

6. 没有同源限制。

## 2.2 WebSocket的相关术语
### 2.2.1 URI scheme
WebSocket协议是在HTTP协议上运行的一种新的协议。所以WebSocket协议通过一种新的URI scheme——ws://或wss://（表示WebSocket Secure协议）来区分与HTTP协议之间的区别。

WebSocket URI 的格式如下：
```javascript
ws://host[:port]/path[?query]
wss://host[:port]/path[?query]
```
其中，host和port指定了WebSocket服务器的地址及端口，path表示访问资源路径，query表示查询字符串参数。

注意：WebSocket只能使用http(s)协议，不支持其他类型的协议如ftp。

### 2.2.2  Frame
WebSocket协议中，每个数据帧都由两部分组成：

*   Header：2字节长，前4个bit为固定值（FIN=1，opcode）后4个bit为payload length的长度，剩余的12个bit为mask key或者是掩码值，根据这个mask key解密。其结构如下图所示：

   ![FrameHeader](./images/FrameHeader.jpg "FrameHeader")

*   Payload：负载，最大容量为64KB，根据数据类型不同，负载会有不同的格式。如文本类型的数据，负载就是UTF-8编码的文本，而图像类型的数据则是一个二进制的图片文件。

每一个数据帧都会被打包为一个完整的帧格式。对于WebSocket客户端来说，WebSocket收到的数据都是以frame的形式返回的。

### 2.2.3 Handshake

WebSocket协议的建立过程，要经过Client请求与Server回应两个阶段。该过程即为Handshake（握手）。Handshake是为了建立 WebSocket 信道，在此过程中，客户端和服务器都能确认对方的身份。

Handshake的过程包括如下四个步骤：

1.    Client向Server发起WebSocket请求，并包含Sec-WebSocket-Key字段。这个字段的值是随机生成的一个值，用于作为Sec-WebSocket-Accept的验证依据。

2.    Server接受WebSocket请求，并检查Sec-WebSocket-Key的值是否合法。如果合法，Server将返回一个响应消息。这个响应消息包括一个 Sec-WebSocket-Accept 的头部，其值为使用Sec-WebSocket-Key作为KEY计算出来的SHA-1哈希值。

3.    Client读取并验证Sec-WebSocket-Accept值，如果一致，则完成握手，开始WebSocket数据交互。否则，关闭WebSocket连接。

4.    Server确认握手成功。

### 2.2.4 Message Type

WebSocket协议中的数据分为两种类型：Text类型和Binary类型。

1. Text类型：Text类型的数据就是普通的UTF-8编码的文本。一般情况下，Text类型的数据仅被用作传输少量文本信息。

2. Binary类型：Binary类型的数据是一个二进制的数组，比如图像、视频等格式的文件。一般情况下，Binary类型的数据会被用作传输大量非UTF-8编码的文本或者数据。

