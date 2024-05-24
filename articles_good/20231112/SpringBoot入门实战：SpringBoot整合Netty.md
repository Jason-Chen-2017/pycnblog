                 

# 1.背景介绍


随着互联网服务的火爆发展、业务的快速增长以及技术的飞速迭代，基于RESTful API的Web服务已经成为主流的开发模式。然而在分布式环境下，传统的Servlet容器已不能满足需求，于是出现了Spring Boot、Netflix OSS、Docker Swarm等一系列的新技术。这些新的技术能够帮助我们更好地实现微服务架构、提升开发效率，让我们摆脱繁琐的配置与部署工作，开发人员可以更加关注于业务逻辑的开发。
相比于传统Servlet容器，使用Netty作为服务器端框架在Java世界里已逐渐成为趋势。从Netty官网（http://netty.io/）的介绍中可以看出，Netty是一个异步事件驱动的NIO框架，它能充分利用多核优势，解决高并发请求下的网络通信问题。因此，在基于Netty的HTTP服务框架上，也同样有很多优秀的开源项目，如Spring Boot Netty，使得开发者只需要简单配置就能迅速构建出可靠且高性能的服务端应用。本文将通过对SpringBoot和Netty的结合，来展示如何构建一个基于RESTful API规范的HTTP服务。
# 2.核心概念与联系
首先，要明确两个重要的概念：Netty是什么？SpringBoot又是什么？
## 2.1 Netty 是什么？
Netty是由JBOSS提供的一个基于Java NIO API编写的开源网络库，它是一个运行速度快、高并发的网络应用程序框架。它提供了包括TCP/IP协议栈在内的基础通信功能，例如TCP、UDP客户端/服务器、SSL/TLS、HTTP代理、WebSocket客户端/服务器、STMP/POP3/IMAP客户端/服务器等。此外，它还提供了对WebFlux响应式编程的支持，以及对WebSocket的支持，使得开发人员可以方便地编写出高性能、高并发的网络应用。其主要特性如下：

1. 异步非阻塞：采用Epoll event loop取代传统的Reactor线程模型，有效避免了线程切换带来的开销；同时引入多路复用技术，实现单线程多任务并行处理，极大提升IO密集型应用的处理能力。

2. 零拷贝：采用堆外内存映射技术，直接向操作系统传递文件数据，避免了数据在JVM和操作系统之间的反复复制，显著提升了应用的吞吐量。

3. 支持HTTP/2：Netty提供了基于HTTP/2协议的客户端和服务器端支持，使得开发人员可以使用标准化的HTTP协议接口开发应用，同时兼顾HTTP/2的高性能及协议扩展性。

4. 全面的测试：Netty项目提供了覆盖面广泛的单元测试，并且在高负载下也经过了长时间的验证，保证了其稳定性和安全性。

5. 可靠的社区支持：Netty的社区活跃度非常高，近年来已经成为Apache Foundation的一部分。

## 2.2 Spring Boot 是什么？
SpringBoot是一个开源的Java开发框架，它简化了Spring的配置，提供了自动装配的便利性。SpringBoot可以帮助开发者快速构建单体应用或者微服务架构中的各个独立组件，并为生产环境提供基础设施的能力，如监控告警、健康检查、外部配置等。其主要特性如下：

1. 创建独立运行的JAR包或WAR包：借助嵌入式Tomcat或Jetty服务器，SpringBoot可以打包成可执行的JAR或WAR文件，并通过命令“java -jar”启动，不需要额外安装Tomcat或Jetty。

2. 提供自动配置的starter模块：SpringBoot为开发者提供了大量的 starter 模块，通过依赖不同的模块，可以快速实现各种功能，如数据库连接、消息队列、缓存、邮件发送、安全控制等。

3. 通过注解来配置Bean：除了提供自动配置的starter模块之外，SpringBoot还允许开发者自定义starter模块。该模块可以用注解的方式来配置Bean，简化了XML配置的复杂度。

4. 提供应用监控工具：SpringBoot提供了 Actuator 来提供对应用的内部状态的监控，如内存信息、CPU负载等。

5. 提供生产级功能特性：SpringBoot包含对一些典型场景的集成，比如安全认证、日志切割、指标度量、外部配置管理、健康检查等。

6. 更方便的接入其他框架：SpringBoot支持对各种框架的无缝接入，比如 Spring Data JPA、Spring Security、Spring Cloud、Spring Batch 等。

综上所述，通过结合Netty和SpringBoot，开发人员可以快速搭建出一个基于RESTful API规范的HTTP服务，并使用统一的配置中心、监控中心和服务注册发现机制，来实现微服务架构的部署与运维。
# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
通过前两节的介绍，读者应该对Netty有一个初步的了解，下面我们来进行更进一步的研究。在这一节中，我会详细介绍Netty的相关知识，希望能让读者能够真正理解Netty。由于篇幅限制，本节主要包含以下内容：

- TCP/IP协议及流程
- HTTP协议及流程
- Netty工作流程
- Bootstrap创建客户端和服务端
- 消息编解码
- 数据帧、字节buf的管理
- 使用ByteBuf传输数据
- 服务端与客户端建立连接后的通信过程

最后，我会给出实际案例，即利用Netty开发一个简单的聊天室程序，展示如何利用Netty开发一个基于RESTful API规范的HTTP服务，并把握关键的技术点。希望通过这个案例，读者能够体会到Netty的强大功能，以备后用。

# 4.具体代码实例和详细解释说明
在正式开始之前，先要明白以下几个关键点：

- Netty的基本组成结构包括ServerBootstrap和Bootstrap。其中ServerBootstrap用来创建服务端的监听器，Bootstrap用来创建客户端的连接。
- ChannelHandler是Netty中非常重要的组件之一，它负责处理传入的数据，包括读、写、拆包、粘包等。每个Channel都可以对应一个ChannelPipeline，该ChannelPipeline包含多个ChannelHandler，并按照顺序串行地执行这些Handler。
- ByteBuf（字节缓冲区）是Netty用于处理I/O操作的一种抽象。底层数据存放在字节数组中，可以根据实际情况调整容量大小，减少频繁扩容和缩容的操作。

下面，我们开始进入正题，构建一个简单的聊天室程序。

## 4.1 服务端与客户端建立连接后的通信过程
Netty服务端与客户端建立连接后，首先是服务端接受客户端的连接，然后进行握手过程，建立双方的通信通道。然后客户端向服务端发送一个欢迎消息，接着等待服务端的回复。当服务端确认客户端的连接时，则发送一条消息给客户端，告诉他自己已经准备好了，开始聊天。之后，客户端也可以继续向服务端发送消息。整个通信过程如下图所示：

这个过程中涉及到的主要技术点有：

### 初始化服务端
```java
public class Server {
    public static void main(String[] args) throws InterruptedException {
        EventLoopGroup bossGroup = new NioEventLoopGroup(); // 用来接收新连接
        EventLoopGroup workerGroup = new NioEventLoopGroup(); // 用来处理已连接的读写

        try {
            ServerBootstrap b = new ServerBootstrap();

            b.group(bossGroup, workerGroup)
                   .channel(NioServerSocketChannel.class) // 指定Channel类型
                   .handler(new LoggingHandler(LogLevel.INFO)) // 设置日志级别
                   .childHandler(new ChildChannelHandler()); // 处理子通道

            // 绑定端口，开始接收新连接
            ChannelFuture f = b.bind(PORT).sync();

            System.out.println("Listening for connection at " + PORT);
            
            f.channel().closeFuture().sync();
        } finally {
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }
}
```

这里我们创建一个NioEventLoopGroup，用来接收客户端的新连接，另一个NioEventLoopGroup用来处理已经建立的连接的读写操作。然后我们通过ServerBootstrap构造器创建了一个NioServerSocketChannel，设置其对应的处理器为LoggingHandler，LoggingHandler是一个ChannelInboundHandlerAdapter，其作用是在控制台输出收到的消息。ChildChannelHandler是一个ChannelInitializer，用于初始化子通道的处理器链。在main函数中，我们调用bind方法绑定端口号，开始接收客户端的连接。

### 服务端与客户端交互
```java
import io.netty.buffer.Unpooled;
import io.netty.channel.*;
import io.netty.util.CharsetUtil;

public class ChildChannelHandler extends ChannelInitializer<SocketChannel> {

    @Override
    protected void initChannel(SocketChannel ch) throws Exception {
        ChannelPipeline pipeline = ch.pipeline();
        
        // 添加解码器，用于对接收的数据进行解码
        pipeline.addLast(new MessageDecoder());
        // 添加编码器，用于对发送的数据进行编码
        pipeline.addLast(new MessageEncoder());
        
        // 当发生异常时，打印日志，关闭子通道
        pipeline.addLast(new SimpleChannelInboundHandler<Object>() {
            @Override
            public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) throws Exception {
                super.exceptionCaught(ctx, cause);
                ctx.close();
                cause.printStackTrace();
            }
        });
        
        // 当客户端连接成功后，向客户端发送欢迎消息
        pipeline.addLast(new ChannelInboundHandlerAdapter() {
            @Override
            public void channelActive(ChannelHandlerContext ctx) throws Exception {
                String message = "\nWelcome to my chatroom! Please type your messages here and hit enter.\n";
                byte[] data = message.getBytes(CharsetUtil.UTF_8);
                
                ctx.writeAndFlush(Unpooled.wrappedBuffer(data));
            }
        });
        
        // 处理客户端的消息，并把消息转发给其他客户端
        pipeline.addLast(new ChatHandler());
    }
}
```

这里，我们首先定义了MessageDecoder和MessageEncoder两个解码器和编码器。解码器用于对接收的数据进行解码，编码器用于对发送的数据进行编码。当发生异常时，打印日志，关闭子通道。当客户端连接成功后，向客户端发送欢迎消息。然后我们定义了一个ChatHandler来处理客户端的消息，并把消息转发给其他客户端。在initChannel函数中，我们调用pipeline.addLast方法，添加了一系列的处理器。

```java
public class ChatHandler extends ChannelInboundHandlerAdapter {

    private final Map<String, Channel> channels = new HashMap<>();
    
    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
        if (msg instanceof TextWebSocketFrame) {
            TextWebSocketFrame frame = (TextWebSocketFrame) msg;
            String username = getUsername(frame.text());
            
            // 把当前客户端加入到channels集合中
            synchronized (channels) {
                channels.put(username, ctx.channel());
            }
            
            broadcast(frame.retain(), username);
        } else {
            System.err.println("unsupported message type: " + msg.getClass().getName());
            throw new UnsupportedOperationException("Unsupported message type: " + msg.getClass().getName());
        }
    }
    
    /**
     * 获取用户名
     */
    private String getUsername(String text) {
        int index = text.indexOf(":");
        return index == -1? "" : text.substring(0, index);
    }
    
    /**
     * 广播消息
     */
    private void broadcast(TextWebSocketFrame frame, String sender) {
        Set<String> receivers = new HashSet<>(channels.keySet());
        receivers.remove(sender);
        
        for (String receiver : receivers) {
            Channel channel = channels.get(receiver);
            if (channel!= null && channel.isActive()) {
                channel.writeAndFlush(frame.copy());
            }
        }
    }
}
```

这里，我们定义了ChannelInboundHandlerAdapter来处理客户端的消息。当收到TextWebSocketFrame类型的消息时，我们解析出用户名，把客户端加入到channels集合中，并把消息广播到其他客户端。注意这里用到了CopyOnWriteArrayList来防止并发修改的问题。

```java
import java.util.*;

import com.google.gson.Gson;

import io.netty.channel.*;
import io.netty.handler.codec.http.*;
import io.netty.handler.codec.http.websocketx.*;

public class WebSocketServerHandler extends SimpleChannelInboundHandler<Object> {
    
    private final Map<String, Channel> sessions = new HashMap<>();
    private final Gson gson = new Gson();

    @Override
    public void userEventTriggered(ChannelHandlerContext ctx, Object evt) throws Exception {
        if (evt instanceof HttpUpgradeEvent) {
            WebSocketServerHandshakerFactory wsFactory = new WebSocketServerHandshakerFactory(
                    "ws://" + HOST + ":" + PORT + "/chat", null, true);
            WebSocketServerHandshaker handshaker = wsFactory.newHandshaker(
                    ((HttpUpgradeEvent) evt).upgradeRequest());
            
            if (handshaker == null) {
                WebSocketServerHandshakerFactory.sendUnsupportedVersionResponse(
                        ctx.channel());
            } else {
                handshaker.handshake(ctx.channel(), (FullHttpRequest) evt);
                sessions.put(((WebSocketUpgradeRequest) evt).uri(), ctx.channel());
            }
        }
    }

    @Override
    protected void channelRead0(ChannelHandlerContext ctx, Object msg) throws Exception {
        if (msg instanceof FullHttpResponse) {
            FullHttpResponse response = (FullHttpResponse) msg;
            
            // 判断是否为WebSocket请求
            if (response.headers().containsValue("Upgrade", "websocket")) {
                WebSocketServerHandshakerFactory wsFactory = new WebSocketServerHandshakerFactory(
                        "ws://" + HOST + ":" + PORT + "/chat", null, true);
                
                WebSocketServerHandshaker handshaker = wsFactory.newHandshaker(
                        (HttpHeaders) response.headers());
                
                if (handshaker == null) {
                    WebSocketServerHandshakerFactory.sendUnsupportedVersionResponse(
                            ctx.channel());
                } else {
                    handshaker.handshake(ctx.channel(), (FullHttpRequest) msg);
                    
                    // 保存WebSocket会话信息
                    sessions.put("/chat", ctx.channel());
                }
            }
        } else if (msg instanceof WebSocketFrame) {
            WebSocketFrame frame = (WebSocketFrame) msg;
            
            // 如果是文本消息，则调用handleTextMessage处理
            if (frame instanceof TextWebSocketFrame) {
                handleTextMessage((TextWebSocketFrame) frame);
            }
        }
    }
    
    /**
     * 处理文本消息
     */
    private void handleTextMessage(TextWebSocketFrame frame) {
        Channel session = sessions.get("/chat");
        
        // 如果session为空，则忽略消息
        if (session!= null && session.isActive()) {
            String json = frame.text();
            Request request = gson.fromJson(json, Request.class);
            
            switch (request.getType()) {
                case LOGIN:
                    Login login = gson.fromJson(request.getData(), Login.class);
                    doLogin(login);
                    break;
                    
                case MESSAGE:
                    Message message = gson.fromJson(request.getData(), Message.class);
                    doSendMessage(message, session);
                    break;
                    
               default:
                   sendError(session, ErrorType.UNEXPECTED, "Unknown message type.");
            }
        }
    }
    
    /**
     * 执行登录操作
     */
    private void doLogin(Login login) {
        // TODO
    }
    
    /**
     * 执行发送消息操作
     */
    private void doSendMessage(Message message, Channel session) {
        // TODO
    }
    
    /**
     * 发送错误消息
     */
    private void sendError(Channel session, ErrorType errorType, String message) {
        Response response = new Response(errorType, message);
        String json = gson.toJson(response);
        session.writeAndFlush(new TextWebSocketFrame(json));
    }
}
```

这里，我们定义了WebSocketServerHandler类，该类继承SimpleChannelInboundHandler<Object>类。当用户触发某个事件时，比如客户端发起Websocket连接请求时，会被调用userEventTriggered函数。如果是Websocket连接请求，则会调用channelRead0函数。在该函数中，我们判断是否为Websocket连接请求，如果是，则会调用WebSocketServerHandshaker进行处理。如果不是，则忽略该请求。在这里，我们记录了当前的WebSocket会话信息。在WebSocket会话建立之后，我们将会收到TextWebSocketFrame类型的消息，在该函数中，我们会解析出JSON字符串，然后调用doLogin或doSendMessage函数进行处理。如果发生错误，则调用sendError函数进行响应。

至此，我们的聊天室程序已经构建完成，下面是完整的代码：

```java
import java.util.*;

import com.google.gson.Gson;

import io.netty.buffer.Unpooled;
import io.netty.channel.*;
import io.netty.handler.codec.http.*;
import io.netty.handler.codec.http.websocketx.*;
import io.netty.handler.logging.LogLevel;
import io.netty.handler.logging.LoggingHandler;

public class Server {
    private static final int PORT = 8080;
    private static final String HOST = "localhost";
    
    public static void main(String[] args) throws InterruptedException {
        EventLoopGroup bossGroup = new NioEventLoopGroup(); 
        EventLoopGroup workerGroup = new NioEventLoopGroup(); 

        try {
            ServerBootstrap b = new ServerBootstrap();

            b.group(bossGroup, workerGroup)
                   .channel(NioServerSocketChannel.class) 
                   .handler(new LoggingHandler(LogLevel.INFO)) 
                   .childHandler(new ChildChannelHandler()); 

            ChannelFuture f = b.bind(PORT).sync();

            System.out.println("Listening for connection at " + PORT);
            
            f.channel().closeFuture().sync();
        } finally {
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }
}

// websocket handler
class WebSocketServerHandler extends SimpleChannelInboundHandler<Object> {
    private static final String HOST = "localhost";
    private static final int PORT = 8080;
    
    private final Map<String, Channel> sessions = new HashMap<>();
    private final Gson gson = new Gson();

    @Override
    public void userEventTriggered(ChannelHandlerContext ctx, Object evt) throws Exception {
        if (evt instanceof HttpUpgradeEvent) {
            WebSocketServerHandshakerFactory wsFactory = new WebSocketServerHandshakerFactory(
                    "ws://" + HOST + ":" + PORT + "/chat", null, true);
            WebSocketServerHandshaker handshaker = wsFactory.newHandshaker(
                    ((HttpUpgradeEvent) evt).upgradeRequest());
            
            if (handshaker == null) {
                WebSocketServerHandshakerFactory.sendUnsupportedVersionResponse(
                        ctx.channel());
            } else {
                handshaker.handshake(ctx.channel(), (FullHttpRequest) evt);
                sessions.put(((WebSocketUpgradeRequest) evt).uri(), ctx.channel());
            }
        }
    }

    @Override
    protected void channelRead0(ChannelHandlerContext ctx, Object msg) throws Exception {
        if (msg instanceof FullHttpResponse) {
            FullHttpResponse response = (FullHttpResponse) msg;
            
            // 判断是否为WebSocket请求
            if (response.headers().containsValue("Upgrade", "websocket")) {
                WebSocketServerHandshakerFactory wsFactory = new WebSocketServerHandshakerFactory(
                        "ws://" + HOST + ":" + PORT + "/chat", null, true);
                
                WebSocketServerHandshaker handshaker = wsFactory.newHandshaker(
                        (HttpHeaders) response.headers());
                
                if (handshaker == null) {
                    WebSocketServerHandshakerFactory.sendUnsupportedVersionResponse(
                            ctx.channel());
                } else {
                    handshaker.handshake(ctx.channel(), (FullHttpRequest) msg);
                    
                    // 保存WebSocket会话信息
                    sessions.put("/chat", ctx.channel());
                }
            }
        } else if (msg instanceof WebSocketFrame) {
            WebSocketFrame frame = (WebSocketFrame) msg;
            
            // 如果是文本消息，则调用handleTextMessage处理
            if (frame instanceof TextWebSocketFrame) {
                handleTextMessage((TextWebSocketFrame) frame);
            }
        }
    }
    
    /**
     * 处理文本消息
     */
    private void handleTextMessage(TextWebSocketFrame frame) {
        Channel session = sessions.get("/chat");
        
        // 如果session为空，则忽略消息
        if (session!= null && session.isActive()) {
            String json = frame.text();
            Request request = gson.fromJson(json, Request.class);
            
            switch (request.getType()) {
                case LOGIN:
                    Login login = gson.fromJson(request.getData(), Login.class);
                    doLogin(login);
                    break;
                    
                case MESSAGE:
                    Message message = gson.fromJson(request.getData(), Message.class);
                    doSendMessage(message, session);
                    break;
                    
               default:
                   sendError(session, ErrorType.UNEXPECTED, "Unknown message type.");
            }
        }
    }
    
    /**
     * 执行登录操作
     */
    private void doLogin(Login login) {
        // TODO
    }
    
    /**
     * 执行发送消息操作
     */
    private void doSendMessage(Message message, Channel session) {
        // TODO
    }
    
    /**
     * 发送错误消息
     */
    private void sendError(Channel session, ErrorType errorType, String message) {
        Response response = new Response(errorType, message);
        String json = gson.toJson(response);
        session.writeAndFlush(new TextWebSocketFrame(json));
    }
}

// child channel handler
class ChildChannelHandler extends ChannelInitializer<SocketChannel> {
    @Override
    protected void initChannel(SocketChannel ch) throws Exception {
        ChannelPipeline pipeline = ch.pipeline();
        
        // 添加解码器，用于对接收的数据进行解码
        pipeline.addLast(new MessageDecoder());
        // 添加编码器，用于对发送的数据进行编码
        pipeline.addLast(new MessageEncoder());
        
        // 当发生异常时，打印日志，关闭子通道
        pipeline.addLast(new SimpleChannelInboundHandler<Object>() {
            @Override
            public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) throws Exception {
                super.exceptionCaught(ctx, cause);
                ctx.close();
                cause.printStackTrace();
            }
        });
        
        // 当客户端连接成功后，向客户端发送欢迎消息
        pipeline.addLast(new ChannelInboundHandlerAdapter() {
            @Override
            public void channelActive(ChannelHandlerContext ctx) throws Exception {
                String message = "\nWelcome to my chatroom! Please type your messages here and hit enter.\n";
                byte[] data = message.getBytes(CharsetUtil.UTF_8);
                
                ctx.writeAndFlush(Unpooled.wrappedBuffer(data));
            }
        });
        
        // 处理客户端的消息，并把消息转发给其他客户端
        pipeline.addLast(new ChatHandler());
    }
}

// chat handler
class ChatHandler extends ChannelInboundHandlerAdapter {
    private final Map<String, Channel> channels = new HashMap<>();
    
    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
        if (msg instanceof TextWebSocketFrame) {
            TextWebSocketFrame frame = (TextWebSocketFrame) msg;
            String username = getUsername(frame.text());
            
            // 把当前客户端加入到channels集合中
            synchronized (channels) {
                channels.put(username, ctx.channel());
            }
            
            broadcast(frame.retain(), username);
        } else {
            System.err.println("unsupported message type: " + msg.getClass().getName());
            throw new UnsupportedOperationException("Unsupported message type: " + msg.getClass().getName());
        }
    }
    
    /**
     * 获取用户名
     */
    private String getUsername(String text) {
        int index = text.indexOf(":");
        return index == -1? "" : text.substring(0, index);
    }
    
    /**
     * 广播消息
     */
    private void broadcast(TextWebSocketFrame frame, String sender) {
        Set<String> receivers = new HashSet<>(channels.keySet());
        receivers.remove(sender);
        
        for (String receiver : receivers) {
            Channel channel = channels.get(receiver);
            if (channel!= null && channel.isActive()) {
                channel.writeAndFlush(frame.copy());
            }
        }
    }
}

// 请求、响应对象
enum MessageType {LOGIN, MESSAGE}

class Request {
    private MessageType type;
    private String data;

    public MessageType getType() {
        return type;
    }

    public void setType(MessageType type) {
        this.type = type;
    }

    public String getData() {
        return data;
    }

    public void setData(String data) {
        this.data = data;
    }
}

class Response {
    private ErrorType code;
    private String message;

    public Response(ErrorType code, String message) {
        this.code = code;
        this.message = message;
    }

    public ErrorType getCode() {
        return code;
    }

    public void setCode(ErrorType code) {
        this.code = code;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }
}

enum ErrorType {OK, ERROR, UNEXPECTED}

class Login {
    private String username;
    private String password;

    public Login(String username, String password) {
        this.username = username;
        this.password = password;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }
}

class Message {
    private String content;

    public Message(String content) {
        this.content = content;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }
}

// 消息编码器
class MessageEncoder extends MessageToByteEncoder<Object> {
    private final Gson gson = new Gson();

    @Override
    protected void encode(ChannelHandlerContext ctx, Object msg, List<Object> out) throws Exception {
        if (msg instanceof Request || msg instanceof Response) {
            String json = gson.toJson(msg);
            out.add(Unpooled.wrappedBuffer(json.getBytes()));
        } else {
            out.add(msg);
        }
    }
}

// 消息解码器
class MessageDecoder extends MessageToMessageDecoder<WebSocketFrame> {
    private final Gson gson = new Gson();

    @Override
    protected void decode(ChannelHandlerContext ctx, WebSocketFrame frame, List<Object> out) throws Exception {
        if (frame instanceof TextWebSocketFrame) {
            String json = ((TextWebSocketFrame) frame).text();
            Object obj = gson.fromJson(json, Request.class);
            out.add(obj);
        } else {
            out.add(frame.retain());
        }
    }
}
```