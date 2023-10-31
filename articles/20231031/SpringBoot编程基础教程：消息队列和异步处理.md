
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 消息队列简介
消息队列（Message Queue）是一个用于保存、转发和传递消息的应用程序组件，它是一个点对点（P2P）通信方式。在面向对象编程中，消息队列通常被实现为一个中间件服务，该服务提供可靠的消息传递。消息队列又称为中间件或消息系统。它具有以下几个主要作用：

1. 异步处理：消息队列提供了一种异步处理的方式。生产者发送消息之后，无需等待消费者响应就直接将其扔到消息队列里，生产者可以继续发送新的消息；而消费者则不必等待生产者发送新消息，只需要从消息队列里面获取消息并进行处理即可。这样就可以提高性能和吞吐量。

2. 解耦合：通过引入消息队列，生产者与消费者之间就不需要强耦合了。生产者无需知道消息在哪儿，只管往里面放就可以了；同样，消费者也无需知道消息从哪儿来，只管从里面拿就可以了。两边都需要依赖消息队列中间件才可以。

3. 流量削峰：消息队列能够平滑地处理消息流量。由于消息队列缓冲了消息，使得消费者不会因此陷入过载状态。即使因为某些原因导致消息积压，也可以通过设置消息超时时间或者消息数量阀值来控制消息积压情况。

4. 数据存储：通过消息队列把数据保存在队列里面，可以有效防止数据的丢失。如果消费者处理消息失败，消息仍然保留在队列里面，下次再启动消费者的时候可以再次读取。

5. 冗余机制：消息队列提供的冗余机制可以避免单点故障。如果某个消息队列服务器出现故障，其他队列服务器可以接管它的工作，保证整个系统的可用性。

## SpringBoot集成RabbitMQ
### 什么是RabbitMQ？
RabbitMQ 是开源的消息代理软件(MOM)，它用作应用间的消息通讯。消息队列的特点就是，发送方(生产者)和接收方(消费者)之间没有实时连接，生产者发布消息后立即返回，由 RabbitMQ 将消息持久化到磁盘上，并复制到所有队列中，然后才告诉消费者。

### RabbitMQ 安装与配置
#### 安装 RabbitMQ 服务端
- 在 CentOS/RHEL 操作系统上安装
```bash
sudo yum install rabbitmq-server -y
```
- 在 Ubuntu 操作系统上安装
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install erlang -y #下载 erlang 语言环境
wget https://dl.bintray.com/rabbitmq/all/rabbitmq-server/3.7.9/rabbitmq-server_3.7.9-1_all.deb
sudo dpkg -i rabbitmq-server_3.7.9-1_all.deb #安装 rabbitmq 服务端
sudo systemctl start rabbitmq-server #启动 rabbitmq 服务端
sudo systemctl enable rabbitmq-server #开机自启 rabbitmq 服务端
```
#### 配置 RabbitMQ 服务端
- 默认安装路径：`/var/lib/rabbitmq/`
- 修改配置文件：`vim /etc/rabbitmq/rabbitmq.conf`，默认端口为 `5672`，用户名密码默认为 `guest`/`guest`
- 创建管理插件用户：`rabbitmqctl add_user {username} {password}`
- 设置管理插件权限：`rabbitmqctl set_permissions {username} ".*" ".*" ".*"`
- 重启 RabbitMQ 服务端：`systemctl restart rabbitmq-server.service`。

#### 安装 RabbitMQ 客户端
- 在 CentOS/RHEL 操作系统上安装
```bash
sudo yum install rabbitmq-erlang-client -y
sudo yum install rabbitmq-tools -y
```
- 在 Ubuntu 操作系统上安装
```bash
sudo apt-get install rabbitmq-server -y
```
### SpringBoot 集成 RabbitMQ
#### 添加依赖
```xml
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-amqp</artifactId>
        </dependency>
```
#### 配置文件 application.properties
```yaml
spring:
  rabbitmq:
    host: ${host}
    port: 5672
    username: guest
    password: guest
    virtual-host: /
```
#### 使用 RabbitTemplate 发送消息
```java
@Autowired
private AmqpAdmin amqpAdmin;
@Autowired
private RabbitTemplate rabbitTemplate;
public void send() throws IOException{
    ConnectionFactory factory = new CachingConnectionFactory("localhost");
    //声明交换器类型及名称
    Channel channel = factory.createChannel();
    channel.exchangeDeclare("logs", BuiltinExchangeType.FANOUT);

    String message = "Hello World!";

    rabbitTemplate.convertAndSend("logs","",message);
    System.out.println("[x] Sent '" + message + "'");
}
```