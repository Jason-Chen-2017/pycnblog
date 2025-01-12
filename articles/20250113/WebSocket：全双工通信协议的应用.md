                 

## WebSocket：全双工通信协议的应用

### 关键词：WebSocket、全双工通信、实时通信、消息推送、应用实战

> **摘要**：本文将深入探讨WebSocket这一全双工通信协议的应用。我们将首先介绍WebSocket的起源与发展，然后详细讲解其核心概念、工作原理、系统架构设计，并通过实际项目演示其应用。文章还将总结最佳实践并提供拓展阅读，以帮助读者全面了解和使用WebSocket。

## 目录大纲设计思路

为了设计出一本关于《WebSocket：全双工通信协议的应用》的详细且逻辑清晰的目录大纲，我们将遵循以下步骤：

1. **背景介绍与核心概念**：首先，我们需要介绍WebSocket协议的背景，包括其起源、发展历程以及在现代网络通信中的重要性。接着，明确核心概念，如全双工通信、消息推送、长轮询等，并解释这些概念在WebSocket中的应用。

2. **原理讲解与数学模型**：详细介绍WebSocket的通信原理，使用Mermaid流程图展示WebSocket连接、消息发送和接收的过程。同时，配合Python源代码示例，讲解WebSocket的API和编程技巧。

3. **系统分析与架构设计**：讨论WebSocket在不同应用场景下的系统架构设计，使用Mermaid类图和序列图展示系统功能设计、系统架构设计和系统接口设计。

4. **项目实战**：通过一个实际项目，演示如何安装WebSocket环境，实现核心功能，并进行代码解读与分析。

5. **最佳实践与拓展**：总结实战中的经验和技巧，给出最佳实践建议，并提供相关资源和拓展阅读。

6. **小结与注意事项**：总结全书内容，强调关键知识点，并提供使用WebSocket时需要注意的事项。

### 目录大纲

```markdown
# 《WebSocket：全双工通信协议的应用》目录大纲

## 第一部分：WebSocket基础

### 第1章 WebSocket简介

#### 1.1 WebSocket的起源与发展

##### 1.1.1 WebSocket协议的起源

##### 1.1.2 WebSocket协议的发展历程

##### 1.1.3 WebSocket在现代网络通信中的重要性

#### 1.2 WebSocket核心概念

##### 1.2.1 全双工通信

##### 1.2.2 消息推送

##### 1.2.3 长轮询与WebSocket的对比

#### 1.3 WebSocket工作原理

##### 1.3.1 WebSocket连接过程

##### 1.3.2 WebSocket消息发送与接收

##### 1.3.3 WebSocket与HTTP的关系

#### 1.4 WebSocket应用场景

##### 1.4.1 实时聊天应用

##### 1.4.2 在线游戏

##### 1.4.3 IoT设备通信

#### 1.5 本章小结

## 第二部分：WebSocket原理与实现

### 第2章 WebSocket原理详解

#### 2.1 WebSocket协议详解

##### 2.1.1 WebSocket协议规范

##### 2.1.2 WebSocket协议数据格式

##### 2.1.3 WebSocket协议扩展

#### 2.2 WebSocket编程模型

##### 2.2.1 WebSocket API

##### 2.2.2 WebSocket事件监听

##### 2.2.3 WebSocket错误处理

#### 2.3 WebSocket示例代码

##### 2.3.1 Python WebSocket客户端实现

##### 2.3.2 Python WebSocket服务端实现

##### 2.3.3 WebSocket连接管理

#### 2.4 WebSocket与数据库的集成

##### 2.4.1 WebSocket与关系型数据库

##### 2.4.2 WebSocket与NoSQL数据库

#### 2.5 本章小结

## 第三部分：WebSocket应用实战

### 第3章 WebSocket应用案例

#### 3.1 实时聊天系统设计

##### 3.1.1 系统需求分析

##### 3.1.2 系统架构设计

##### 3.1.3 系统功能实现

#### 3.2 在线游戏开发

##### 3.2.1 游戏场景设计

##### 3.2.2 游戏逻辑实现

##### 3.2.3 游戏性能优化

#### 3.3 IoT设备通信

##### 3.3.1 设备通信需求

##### 3.3.2 设备连接实现

##### 3.3.3 设备数据传输

#### 3.4 WebSocket在电商中的应用

##### 3.4.1 电商实时库存同步

##### 3.4.2 电商实时聊天支持

##### 3.4.3 电商用户行为分析

#### 3.5 本章小结

-------------------

## 附录

#### 附录A：技术术语解释

#### 附录B：Python WebSocket库使用示例

#### 附录C：WebSocket常见问题与解决方案

#### 附录D：相关资源与拓展阅读

#### 附录E：作者介绍
```

## 第一部分：WebSocket基础

### 第1章 WebSocket简介

#### 1.1 WebSocket的起源与发展

##### 1.1.1 WebSocket协议的起源

WebSocket协议是由Ian Davis于2007年首次提出的，目的是为了解决传统的HTTP通信模式在实时通信方面的不足。在此之前，Web应用中的实时通信主要依赖于轮询技术，即客户端定期向服务器发送请求以获取更新信息，这种方式效率低下且浪费资源。

Ian Davis认为，网络通信应该是一种双向、实时的通信，类似于电话通信。于是，他提出了WebSocket协议，旨在实现全双工通信，即客户端与服务器可以同时发送和接收消息。WebSocket的设计理念得到了业界的广泛认可，并在2011年被正式标准化为RFC 6455。

##### 1.1.2 WebSocket协议的发展历程

WebSocket协议的标准化过程经历了多个版本。从2007年最初的版本，到2008年的草案，再到2011年的正式发布，WebSocket协议逐渐成熟。自2011年以来，WebSocket协议一直处于活跃维护状态，定期发布更新和扩展。

随着时间的推移，WebSocket在Web应用中的使用越来越广泛。许多框架和库开始支持WebSocket，如Node.js、Python的web.py、Java的Spring等。WebSocket的应用场景也不断扩展，包括实时聊天、在线游戏、IoT设备通信、实时数据监控等。

##### 1.1.3 WebSocket在现代网络通信中的重要性

WebSocket的出现，极大地改变了Web通信的模式。传统的HTTP通信是一种单向、请求响应式的通信，而WebSocket则提供了一种双向、全双工的通信方式。这意味着客户端和服务器可以在任何时候发送消息，而不需要等待对方的请求。

WebSocket的重要性主要体现在以下几个方面：

1. **实时性**：WebSocket提供了真正的实时通信，可以立即发送和接收消息，大大提升了应用的用户体验。
2. **效率**：相比轮询技术，WebSocket减少了客户端和服务器之间的通信次数，节省了带宽和服务器资源。
3. **灵活性**：WebSocket支持自定义的协议和数据格式，使得开发者可以根据具体需求进行定制。
4. **广泛支持**：WebSocket已经成为Web标准的一部分，得到了各大浏览器和服务器框架的支持。

#### 1.2 WebSocket核心概念

##### 1.2.1 全双工通信

全双工通信指的是通信双方可以同时进行发送和接收数据，就像打电话一样，双方可以同时说话和听对方说话。在WebSocket中，客户端和服务器通过WebSocket连接实现全双工通信。

##### 1.2.2 消息推送

消息推送是一种由服务器主动向客户端发送消息的技术。在WebSocket中，服务器可以通过WebSocket连接直接向客户端发送消息，而不需要客户端轮询请求。

##### 1.2.3 长轮询与WebSocket的对比

长轮询是一种实时通信技术，客户端发送请求后，如果服务器没有数据，会保持连接打开，直到有新的数据可以发送。而WebSocket则是一种真正的全双工通信，客户端和服务器可以同时发送和接收消息。

相比长轮询，WebSocket具有以下优势：

1. **实时性**：WebSocket提供了真正的实时通信，消息可以立即发送和接收。
2. **效率**：WebSocket减少了客户端和服务器之间的通信次数，节省了带宽和服务器资源。
3. **灵活性**：WebSocket支持自定义的协议和数据格式，使得开发者可以根据具体需求进行定制。

##### 1.3 WebSocket工作原理

WebSocket协议的工作原理可以分为以下几个步骤：

1. **握手**：客户端向服务器发送一个特殊的HTTP请求，请求升级为WebSocket连接。
2. **建立连接**：服务器响应客户端的握手请求，如果握手成功，客户端和服务器之间建立WebSocket连接。
3. **消息发送与接收**：客户端和服务器通过WebSocket连接发送和接收消息。

##### 1.3.1 WebSocket连接过程

WebSocket连接过程可以分为以下几个步骤：

1. **客户端发送握手请求**：客户端向服务器发送一个HTTP请求，请求头部包含特定的Upgrade请求头，指定升级为WebSocket协议。
   ```python
   GET /chat HTTP/1.1
   Host: server.example.com
   Upgrade: websocket
   Connection: Upgrade
   Sec-WebSocket-Key: dGFiY2VyZWQ=
   Sec-WebSocket-Protocol: chat, superchat
   Sec-WebSocket-Version: 13
   ```

2. **服务器响应握手请求**：服务器接收客户端的握手请求后，发送一个HTTP响应，确认升级为WebSocket连接。
   ```python
   HTTP/1.1 101 Switching Protocols
   Upgrade: websocket
   Connection: Upgrade
   Sec-WebSocket-Accept: s3pPLMBiTxA9kQ8XY8eSDQ==*
   ```

3. **建立WebSocket连接**：如果握手成功，客户端和服务器之间建立WebSocket连接。

##### 1.3.2 WebSocket消息发送与接收

WebSocket连接建立后，客户端和服务器可以通过连接发送和接收消息。WebSocket消息分为文本消息和二进制消息两种类型。

1. **发送文本消息**：客户端可以通过`send()`方法发送文本消息，服务器接收到文本消息后可以通过`onmessage`事件处理。
   ```python
   # 客户端发送文本消息
   websocket.send("Hello, server!")

   # 服务器处理文本消息
   def on_message(self, message):
       print("Received message:", message)
   ```

2. **发送二进制消息**：客户端可以通过`send()`方法发送二进制消息，服务器接收到二进制消息后可以通过`onmessage`事件处理。
   ```python
   # 客户端发送二进制消息
   websocket.send(b'\x00\x01\x02\x03')

   # 服务器处理二进制消息
   def on_message(self, message):
       print("Received binary message:", message)
   ```

##### 1.3.3 WebSocket与HTTP的关系

WebSocket协议是基于HTTP协议的，但与HTTP协议有本质的区别。HTTP协议是一种单向、请求响应式的通信协议，而WebSocket协议则是一种双向、全双工的通信协议。

WebSocket协议在HTTP协议的基础上，引入了“握手”这一步骤，用于建立WebSocket连接。通过握手，客户端和服务器可以协商WebSocket协议的版本、数据格式等信息。

##### 1.4 WebSocket应用场景

WebSocket协议广泛应用于各种实时通信场景，以下是一些典型的应用场景：

1. **实时聊天**：WebSocket可以用于实现实时聊天功能，客户端和服务器可以实时发送和接收消息。
2. **在线游戏**：WebSocket可以用于实现在线游戏的双向通信，玩家之间可以实时互动。
3. **IoT设备通信**：WebSocket可以用于连接IoT设备，实时传输设备数据。
4. **实时数据监控**：WebSocket可以用于实时监控数据，如股市行情、气象数据等。

##### 1.5 本章小结

本章介绍了WebSocket协议的起源、发展历程以及其在现代网络通信中的重要性。我们还详细讲解了WebSocket的核心概念，如全双工通信、消息推送和长轮询的对比，以及WebSocket的工作原理。通过本章的学习，读者可以对WebSocket有基本的了解，为后续章节的深入学习打下基础。

## 第二部分：WebSocket原理与实现

### 第2章 WebSocket原理详解

#### 2.1 WebSocket协议详解

##### 2.1.1 WebSocket协议规范

WebSocket协议是基于RFC 6455规范的，该规范定义了WebSocket协议的通信格式和通信流程。WebSocket协议由两部分组成：握手协议和消息传输协议。

1. **握手协议**：握手协议用于客户端和服务器之间建立WebSocket连接。客户端通过发送HTTP请求，请求服务器升级协议为WebSocket协议。服务器响应请求，确认升级协议，并返回WebSocket连接的唯一标识。

2. **消息传输协议**：消息传输协议定义了WebSocket消息的格式和传输方式。WebSocket消息分为文本消息和二进制消息两种类型，每种类型都有不同的格式。

##### 2.1.2 WebSocket协议数据格式

WebSocket协议的数据格式可以分为以下几部分：

1. **帧结构**：WebSocket帧是数据传输的基本单位。一个WebSocket帧由帧头和数据体组成。帧头包含帧类型、帧长度、掩码等信息，数据体包含实际传输的数据。

2. **控制帧**：控制帧用于管理WebSocket连接，如连接建立、连接关闭、发送ping消息等。控制帧的类型由帧头中的opcode字段指定。

3. **文本帧**：文本帧用于传输文本消息。文本帧的数据体中包含UTF-8编码的文本数据。

4. **二进制帧**：二进制帧用于传输二进制消息。二进制帧的数据体中包含原始的二进制数据。

##### 2.1.3 WebSocket协议扩展

WebSocket协议支持扩展，扩展可以用于增加新的功能或优化性能。扩展由客户端和服务器在握手过程中协商。扩展的协商过程如下：

1. **客户端发送扩展列表**：在握手请求中，客户端发送一个Sec-WebSocket-Extensions头部，列出支持的扩展。

2. **服务器选择扩展**：服务器从客户端发送的扩展列表中选择一个扩展，并在握手响应中返回选中的扩展。

3. **扩展应用**：客户端和服务器使用选中的扩展进行通信。

#### 2.2 WebSocket编程模型

WebSocket编程模型主要包括API、事件监听和错误处理。

##### 2.2.1 WebSocket API

WebSocket API提供了创建WebSocket连接、发送消息、接收消息和关闭连接的方法。

1. **创建WebSocket连接**：使用WebSocket构造函数创建WebSocket对象，指定连接的URL和协议。
   ```python
   websocket = WebSocket('wss://example.com/socket')
   ```

2. **发送消息**：使用WebSocket对象的`send()`方法发送消息。
   ```python
   websocket.send('Hello, server!')
   ```

3. **接收消息**：使用WebSocket对象的`onmessage`事件处理接收到的消息。
   ```python
   def on_message(self, message):
       print('Received message:', message)
   ```

4. **关闭连接**：使用WebSocket对象的`close()`方法关闭连接。
   ```python
   websocket.close()
   ```

##### 2.2.2 WebSocket事件监听

WebSocket事件监听用于处理WebSocket连接的各种事件。

1. **连接打开**：当WebSocket连接成功建立时，触发`onopen`事件。
   ```python
   def on_open(self):
       print('Connected to WebSocket server')
   ```

2. **消息接收**：当WebSocket接收到消息时，触发`onmessage`事件。
   ```python
   def on_message(self, message):
       print('Received message:', message)
   ```

3. **连接关闭**：当WebSocket连接关闭时，触发`onclose`事件。
   ```python
   def on_close(self):
       print('Disconnected from WebSocket server')
   ```

4. **错误处理**：当WebSocket发生错误时，触发`onerror`事件。
   ```python
   def on_error(self, error):
       print('WebSocket error:', error)
   ```

##### 2.2.3 WebSocket错误处理

WebSocket错误处理包括异常处理和错误日志记录。

1. **异常处理**：使用try-except语句捕获WebSocket错误，并处理错误。
   ```python
   try:
       # WebSocket操作
   except Exception as e:
       print('WebSocket error:', e)
   ```

2. **错误日志记录**：将WebSocket错误记录到日志文件中，便于调试和分析。
   ```python
   with open('websocket_error.log', 'a') as f:
       f.write('WebSocket error: %s\n' % e)
   ```

#### 2.3 WebSocket示例代码

下面是一个简单的WebSocket客户端和服务器的示例代码，演示了如何使用WebSocket进行实时通信。

##### 2.3.1 Python WebSocket客户端实现

```python
import websocket
import threading

def on_open(ws):
    print('Connected to WebSocket server')
    ws.send('Hello, server!')

def on_message(ws, message):
    print('Received message:', message)
    ws.close()

def on_error(ws, error):
    print('WebSocket error:', error)

def on_close(ws):
    print('Disconnected from WebSocket server')

if __name__ == '__main__':
    websocket = websocket.WebSocketApp(
        'ws://localhost:8000/socket',
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    websocket.run_forever()
```

##### 2.3.2 Python WebSocket服务端实现

```python
import websocket
import threading

def on_open(ws):
    print('Connected to WebSocket client')
    ws.send('Hello, client!')

def on_message(ws, message):
    print('Received message:', message)
    ws.send('Received message: ' + message)

def on_error(ws, error):
    print('WebSocket error:', error)

def on_close(ws):
    print('Disconnected from WebSocket client')

if __name__ == '__main__':
    websocket = websocket.WebSocketServer()
    websocket.run_forever()
```

##### 2.3.3 WebSocket连接管理

WebSocket连接管理包括连接的建立、维护和关闭。

1. **连接建立**：客户端通过WebSocket构造函数创建WebSocket连接，并调用`run_forever()`方法启动连接。
2. **连接维护**：WebSocket连接在建立后，会保持活跃状态。如果连接断开，WebSocket会自动尝试重新连接。
3. **连接关闭**：使用`close()`方法关闭WebSocket连接。

#### 2.4 WebSocket与数据库的集成

WebSocket可以与数据库集成，用于实时数据传输和更新。

1. **WebSocket与关系型数据库**：可以使用WebSocket与关系型数据库（如MySQL、PostgreSQL）集成，实现实时数据同步。
2. **WebSocket与NoSQL数据库**：可以使用WebSocket与NoSQL数据库（如MongoDB、Redis）集成，实现实时数据传输和缓存更新。

#### 2.5 本章小结

本章详细介绍了WebSocket协议的规范、数据格式和编程模型。通过示例代码，我们展示了如何使用WebSocket进行实时通信。此外，本章还讨论了WebSocket与数据库的集成，为读者提供了更广泛的应用场景。

## 第三部分：WebSocket应用实战

### 第3章 WebSocket应用案例

#### 3.1 实时聊天系统设计

##### 3.1.1 系统需求分析

实时聊天系统需要实现以下功能：

1. 用户登录注册：用户可以通过注册账号或使用第三方登录方式登录系统。
2. 添加好友：用户可以添加其他用户为好友，并建立聊天关系。
3. 发送消息：用户可以向好友发送文本消息、图片消息、语音消息等。
4. 聊天记录：系统需要存储用户的聊天记录，并支持查看历史消息。
5. 系统通知：系统需要向用户发送系统通知，如好友请求、消息通知等。

##### 3.1.2 系统架构设计

实时聊天系统的架构设计包括前端、后端和数据库三部分。

1. **前端**：前端负责与用户交互，实现用户登录、注册、添加好友、发送消息等功能。前端使用Vue.js框架，实现页面布局和交互。
2. **后端**：后端负责处理业务逻辑，包括用户管理、消息发送和接收、聊天记录存储等。后端使用Spring Boot框架，实现RESTful API和WebSocket连接。
3. **数据库**：数据库用于存储用户信息、聊天记录等数据。数据库使用MySQL，实现数据持久化。

##### 3.1.3 系统功能实现

实时聊天系统的功能实现分为前端和后端两部分。

1. **前端**：前端实现用户登录、注册、添加好友、发送消息等功能。前端使用Vue.js框架，实现页面布局和交互。
2. **后端**：后端实现用户管理、消息发送和接收、聊天记录存储等功能。后端使用Spring Boot框架，实现RESTful API和WebSocket连接。

##### 3.2 在线游戏开发

##### 3.2.1 游戏场景设计

在线游戏需要实现以下功能：

1. 游戏登录：玩家可以通过注册账号或使用第三方登录方式登录游戏。
2. 创建房间：玩家可以创建游戏房间，并设置游戏规则和玩家数量。
3. 加入房间：玩家可以加入已创建的房间，与其他玩家一起游戏。
4. 游戏交互：玩家在游戏中进行实时交互，如移动、攻击、聊天等。
5. 游戏结束：游戏结束后，系统自动计算得分，并记录游戏结果。

##### 3.2.2 游戏逻辑实现

在线游戏的游戏逻辑实现包括客户端和服务器两部分。

1. **客户端**：客户端实现玩家的游戏操作，如移动、攻击、聊天等。客户端使用Unity游戏引擎，实现游戏场景和交互。
2. **服务器**：服务器实现游戏逻辑处理，如玩家位置更新、战斗计算、消息传递等。服务器使用WebSocket连接，实现实时通信。

##### 3.2.3 游戏性能优化

为了提高游戏性能，可以采取以下优化措施：

1. **数据压缩**：对游戏数据进行压缩，减少数据传输量。
2. **批量处理**：将多个操作合并为一个批量操作，减少服务器负载。
3. **异步处理**：使用异步编程，提高服务器处理效率。

##### 3.3 IoT设备通信

##### 3.3.1 设备通信需求

IoT设备通信需要实现以下功能：

1. 设备连接：设备可以通过WiFi、蓝牙等无线方式连接到服务器。
2. 数据传输：设备将采集到的数据实时传输到服务器，供用户查看和分析。
3. 远程控制：用户可以通过服务器远程控制设备，如启动、停止、调整参数等。

##### 3.3.2 设备连接实现

设备连接的实现可以分为以下几个步骤：

1. **设备认证**：设备连接服务器前，需要进行认证，确保设备合法。
2. **设备注册**：设备连接服务器后，需要注册设备信息，以便服务器管理。
3. **数据传输**：设备通过WebSocket连接实时传输数据到服务器，服务器存储和处理数据。

##### 3.3.3 设备数据传输

设备数据传输可以分为以下步骤：

1. **数据采集**：设备采集传感器数据，如温度、湿度、光照等。
2. **数据编码**：将采集到的数据编码为WebSocket消息格式。
3. **数据发送**：设备通过WebSocket连接发送数据到服务器。
4. **数据存储**：服务器接收到数据后，存储到数据库，供用户查询和分析。

##### 3.4 WebSocket在电商中的应用

##### 3.4.1 电商实时库存同步

电商实时库存同步需要实现以下功能：

1. 库存监控：实时监控商品库存，及时发现库存不足或过剩的情况。
2. 库存更新：当商品库存发生变化时，及时更新库存信息。
3. 库存预警：当库存达到预设阈值时，触发预警，提醒商家进行库存调整。

##### 3.4.2 电商实时聊天支持

电商实时聊天支持需要实现以下功能：

1. 用户登录：用户可以通过注册账号或使用第三方登录方式登录电商系统。
2. 添加好友：用户可以添加其他用户为好友，并建立聊天关系。
3. 发送消息：用户可以向好友发送文本消息、图片消息、语音消息等。
4. 聊天记录：系统需要存储用户的聊天记录，并支持查看历史消息。

##### 3.4.3 电商用户行为分析

电商用户行为分析需要实现以下功能：

1. 用户行为采集：采集用户在电商平台的浏览、购买、评价等行为数据。
2. 用户行为分析：对用户行为数据进行分析，提取用户特征和偏好。
3. 用户行为预测：根据用户行为数据预测用户的购买行为和偏好。

##### 3.5 本章小结

本章介绍了WebSocket在实时聊天系统、在线游戏、IoT设备通信和电商中的应用。通过实际案例，展示了如何使用WebSocket实现实时通信和数据处理。这些应用场景展示了WebSocket在提高系统性能和用户体验方面的优势。

## 附录

#### 附录A：技术术语解释

本附录将对本文中涉及的技术术语进行详细解释，帮助读者更好地理解相关概念。

1. **WebSocket**：WebSocket是一种全双工通信协议，允许服务器和客户端之间实时双向通信。
2. **全双工通信**：全双工通信是指通信双方可以同时发送和接收数据，类似于打电话。
3. **消息推送**：消息推送是指服务器主动向客户端发送消息的技术。
4. **长轮询**：长轮询是一种实时通信技术，客户端发送请求后，如果服务器没有数据，会保持连接打开，直到有新的数据可以发送。
5. **RESTful API**：RESTful API是一种基于HTTP协议的API设计风格，用于实现服务器和客户端之间的数据交互。
6. **IoT**：物联网（Internet of Things）是指将各种物品通过互联网连接起来，实现智能管理和控制。
7. **数据库**：数据库是一种用于存储和管理数据的系统，可以快速高效地查询、更新和删除数据。

#### 附录B：Python WebSocket库使用示例

以下是一个使用Python WebSocket库的简单示例，演示了如何创建WebSocket客户端和服务器，并进行消息传输。

**WebSocket客户端示例：**

```python
import websocket

def on_open(ws):
    ws.send("Hello, server!")

def on_message(ws, message):
    print("Received message:", message)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws):
    print("Disconnected from WebSocket server")

if __name__ == "__main__":
    websocket = websocket.WebSocketApp(
        "ws://localhost:8080/socket",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    websocket.run_forever()
```

**WebSocket服务器示例：**

```python
import websocket
import threading

def on_open(ws):
    print("Connected to WebSocket client")

def on_message(ws, message):
    print("Received message:", message)
    ws.send("Received message: " + message)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws):
    print("Disconnected from WebSocket client")

if __name__ == "__main__":
    websocket = websocket.WebSocketServer()
    websocket.run_forever()
```

#### 附录C：WebSocket常见问题与解决方案

以下是一些WebSocket常见的问

