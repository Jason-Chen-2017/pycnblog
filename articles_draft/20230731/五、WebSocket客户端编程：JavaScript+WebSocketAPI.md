
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 WebSocket（全称 Web Sockets）是一种协议，使得客户端和服务器之间可以建立持久连接。它属于应用层协议，工作在 TCP 之上，但它比 HTTP 更底层，并且被设计用来做实时通信，可以支持双向通信。其目的是替代轮询或者长轮询，来实现更高速的通信。WebSocket 是 HTML5 中的一个协议，它提供了一系列解决同源策略限制的方法。
          目前，很多网站都已经支持了 WebSocket，比如网页聊天室、实时游戏、股票行情显示等。WebSocket 在移动端的应用也越来越广泛。基于 JavaScript 的浏览器作为 WebSocket 的客户端开发语言，本文将从基础知识开始，教大家如何使用 WebSocket 来进行应用编程。
          # 2.基本概念术语说明
          ## 2.1 WebSocket 通信流程
          WebSocket 的通信流程分为三步：

          - 第一步：建立 WebSocket 连接
          - 第二步：通信
          - 第三步：关闭 WebSocket 连接

          ### 1) 建立 WebSocket 连接
          建立 WebSocket 连接需要经过以下几个步骤：

          1. 创建 WebSocket 对象；
          2. 设置 WebSocket URL 属性；
          3. 通过 send() 方法发送数据给服务端；
          4. 监听 onopen、onerror、onmessage 和 onclose 事件处理器；
          
          ```javascript
          // 创建 WebSocket 对象
          const ws = new WebSocket("ws://localhost:8080/chat");
      
          // 设置 WebSocket URL 属性
          ws.url = "ws://localhost:8080/chat";
      
          // 发送消息给服务端
          ws.send('Hello from client');
      
          // 监听 open 事件
          ws.addEventListener("open", () => {
            console.log(`已连接到 WebSocket 服务端.`);
          });
      
          // 监听 message 事件
          ws.addEventListener("message", (event) => {
            console.log(`收到消息：${event.data}`);
          });
      
          // 监听 error 事件
          ws.addEventListener("error", (error) => {
            console.log(error);
          });
      
          // 监听 close 事件
          ws.addEventListener("close", (event) => {
            console.log(`已断开与 WebSocket 服务端的连接.`);
          });
          ``` 

          ### 2) 通信
          通信阶段，客户端通过 WebSocket 对象向服务端发送数据。服务端接收到数据后，可以对消息作出相应的反馈处理。如下面的例子所示：

          ```javascript
          // 创建 WebSocket 对象
          const ws = new WebSocket("ws://localhost:8080/chat");
      
          // 发送消息给服务端
          ws.send('Hello from client');
      
          // 接收服务端消息
          ws.addEventListener("message", (event) => {
            console.log(`收到消息：${event.data}`);
            ws.send(`Server received your message: ${event.data}. Thank you!`);
          });
          ``` 
          上述代码表示，客户端通过 WebSocket 对象向服务端发送一条消息“Hello from client”，并在接收到服务端返回的信息后回复一句话“Thank you！”。

          ### 3) 关闭 WebSocket 连接
          当客户端不再需要连接 WebSocket 服务端时，可以调用 close() 方法关闭 WebSocket 连接。如下所示：

          ```javascript
          ws.close();
          ``` 
          ## 2.2 WebSocket 状态码与原因
          WebSocket 通信过程中，除了数据传输的消息外，还会有各种类型的状态码和原因信息。这些信息都是通过 onError、onMessage 和 onClose 等事件通知。当发生错误时，可以通过 err.code 属性获取状态码。而一般情况下，onClose 事件只用于通知通信终止，并不提供任何其他详细信息。

          | 状态码      | 原因                             | 描述                      |
          | ----------- | -------------------------------- | ------------------------- |
          | 1000        | 正常关闭                         |                           |
          | 1001        | 终止登录                         |                           |
          | 1002        | 没有关闭就挂掉                   |                           |
          | 1003        | 重传消息                         |                           |
          | 1007        | 无效帧                           |                           |
          | 1008        | 延迟关闭                         |                           |
          | 1009        | 缓冲区满                         |                           |
          | 1010        | 拒绝扩展名                       |                           |
          | 1011        | 服务端内部错误                   |                           |
          | 1012-2999   | 非标准状态码                     | 可自定义                  |
          | 3000-3999   | 暂时性错误，尝试重新连接         | 可自定义                  |
          | 4000-4999   | 连接错误，尝试重新连接           | 可自定义                  |
          | 5000-5999   | TLS 错误，尝试升级或降级协议     | 可自定义                  |
          | >= 6000     | 应用程序定义的状态码             | 大于等于 6000 时表示自定义 |

          ## 2.3 WebSocket 数据类型
          WebSocket 支持多种数据类型，包括文本数据（text）、二进制数据（binary）、JSON 对象（json）等。其中，文本数据的默认编码方式为 UTF-8。在 Node.js 中，可以通过 Buffer 对象来操作二进制数据。对于 JSON 对象，可以直接使用 JSON.stringify() 方法将其转换成字符串形式。

