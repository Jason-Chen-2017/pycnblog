
作者：禅与计算机程序设计艺术                    

# 1.简介
  

RabbitMQ 是一款开源的消息中间件，基于AMQP协议，由Pivotal公司开发，并于2007年推出，目前最新版本为v3.8.0。ActiveMQ 是一款 Apache 项目下的开源消息总线（MOSAIC），由JBoss提供支持。两者都有不少知名的企业用户，有些应用场景下可以选择其中之一。本文将对这两者进行比较分析，并给出它们的适用场景。
# 2.主要区别
## 2.1 安装配置方面
首先需要强调的是：RabbitMQ 和 ActiveMQ 可以单独部署，也可以部署在同一个服务器上共用资源。因此在安装配置方面没有太多差异。
- 配置文件方面：RabbitMQ 的配置文件默认为 /etc/rabbitmq/rabbitmq.config ，ActiveMQ 的配置文件一般位于 <activemq_home>/conf/activemq.xml 或 <activemq_home>/bin/activemq.profile 中。
- 服务启动方式方面：RabbitMQ 使用 systemd 管理服务，而 ActiveMQ 默认使用 bin/activemq 命令。
- 数据存储路径方面：RabbitMQ 默认数据存储路径为 /var/lib/rabbitmq ，而 ActiveMQ 的默认数据存储路径为 <activemq_home>/data/kahadb 。
- 安全性方面：RabbitMQ 有其自身的权限控制机制，而 ActiveMQ 没有相关配置。
- 监控指标方面：RabbitMQ 提供了系统监控插件，包括 Management 插件和 Queue Length Monitor 插件；而 ActiveMQ 没有相关插件。
- Web管理后台方面：RabbitMQ 提供基于浏览器的 Web 管理界面，但需要配置才能正常工作；而 ActiveMQ 提供基于 RESTful API 的管理接口，无需额外配置即可使用。
- 可用性方面：当 RabbitMQ 节点出现故障时，可以容忍该节点的消息丢失，而 ActiveMQ 只能接受写入或读取请求。
## 2.2 消息模型方面
RabbitMQ 支持三种消息模型：队列、主题和分发器。它支持多种类型的交换机：direct、topic、headers、fanout等。
- direct exchange：将消息路由到那些binding key与routing key完全匹配的queue。
- topic exchange：将消息路由到那些binding key与routing key匹配的routing pattern（路由模式）的queue。
- headers exchange：根据消息头中的属性匹配routing key，将消息路由到对应的queue中。
- fanout exchange：不管接收到的消息是什么类型，都将其广播到所有绑定到此exchange上的queue。
激活消息模型可以通过设置 exchange 的 type 属性来实现。
## 2.3 持久化方面
RabbitMQ 支持持久化，允许消息被存放到磁盘上，即使 RabbitMQ 服务重启或者意外崩溃也不会丢失消息。这种特性对于某些重要的业务消息非常有用，例如订单交易信息等。
而 ActiveMQ 不支持持久化，如果需要启用持久化功能，则需要配置 BrokerPlugin 来实现。
## 2.4 消费确认机制
RabbitMQ 支持自动确认消费，消费者完成任务后会发送 ACK 报文确认消息已被处理，从而确保消息被正确地消费。而 ActiveMQ 需要手动确认消费。
## 2.5 流量控制及配额限制
RabbitMQ 提供了消费者流量控制、发布者流量控制和配额限制等功能。
- 消费者流量控制：RabbitMQ 通过 prefetch 参数可以设置每个消费者最多可以获取多少条消息。
- 发布者流量控制：RabbitMQ 通过 QoS (Quality of Service) 来实现发布者流量控制。QoS 通过设置参数 max-length 和/或 global 参数来实现。
- 配额限制：RabbitMQ 通过 rabbitmqctl set_user_limits 设置用户配额，限制其可创建、使用等资源。
ActiveMQ 在授权方面也有相应的控制，但是更加复杂，通常需要修改 BrokerPlugin 或扩展源码才能满足要求。
## 2.6 高可用性
RabbitMQ 通过镜像集群的方式实现高可用性。两个节点之间通过网络互联，各个节点通过 Erlang 虚拟机维护一个心跳检测机制，互相通信保持同步。
而 ActiveMQ 则通过主备模式实现高可用性，主节点运行 BrokerPlugin，从节点作为 Backup Broker 对外提供服务。
## 2.7 协议支持方面
RabbitMQ 支持多种协议，包括 AMQP、STOMP、MQTT、Redis、 STOMP.over WebSockets、Stomp.plus 等。
而 ActiveMQ 支持 JMS、OpenWire、Stomp、HornetQ、WebSocket 等协议。
# 3. RabbitMQ 和 ActiveMQ 在使用过程中可能遇到的问题和解决方案
## 3.1 性能方面
由于各项特性的不同，RabbitMQ 和 ActiveMQ 在性能方面存在不同点，这里就讨论一下一些我所了解的。
### 3.1.1 RabbitMQ 的性能优势
- 发布订阅模型：RabbitMQ 中的交换机和队列采用发布订阅模型，因此消息在发布和订阅的时候只需要简单地将消息投递到指定的队列上，不需要重复的传递给订阅了多个队列的消费者。
- 轻量级协议：RabbitMQ 使用的 AMQP 协议是一个轻量级的面向消息的中间件标准。
- 异步处理机制：RabbitMQ 通过异步处理机制，使得生产者和消费者之间的通信变得迅速且稳定。RabbitMQ 可以采用“分组”机制将多条消息放在一起发送，还可以利用专门的插件来实现延迟、死信和持久化存储等功能。
- 工作队列：RabbitMQ 可以将消息分派到多个消费者，将负载均衡分担到多个消费者上。
- 集群能力：RabbitMQ 可以方便地部署在多台机器上形成集群，实现消息的最终一致性。
- 消息持久化：RabbitMQ 支持消息持久化，可以将消息保存在硬盘上，保证消息不会因为异常情况而丢失。
### 3.1.2 ActiveMQ 的性能优势
- Message Oriented Middleware（MOM）架构：ActiveMQ 是基于 MOM 架构的，其消息代理可以无缝地整合到应用程序框架中。
- 细粒度的角色划分：ActiveMQ 将角色分成四类，Broker（消息代理服务器）、Producer（消息生产者）、Consumer（消息消费者）、NetworkConnector（网络连接器）。
- 更灵活的消息路由机制：ActiveMQ 提供更灵活的消息路由机制，比如发送到特定目的地的消息，或者将消息广播至多个目标地址。
- 消息缓存机制：ActiveMQ 提供消息缓存机制，可以避免重复消费相同的消息，以提升消息消费性能。
- 分布式事务机制：ActiveMQ 提供分布式事务机制，允许将本地事务提交至远程消息代理服务器，以实现跨越不同 Broker 的事务一致性。
- 实时的消息消费：ActiveMQ 支持实时消息消费，可以在消息到达Broker之后立刻进行消费，而不是等待 Broker 轮询检查消息是否到达。
- 压缩机制：ActiveMQ 支持消息压缩，可以减小传输的数据量，节省网络带宽。
虽然各项性能优势不同，但是很多应用场景下 ActiveMQ 比较受欢迎，特别是在微服务架构中。
### 3.1.3 性能差异
最后需要注意的是，两者的性能差距并不一定是绝对的，ActiveMQ 在某些情况下表现会更好，这是因为其更为灵活的消息路由机制。另外，生产环境中不要选择不必要的特性，因为它们可能会增加额外的开销，影响性能。
## 3.2 集群方面
因为 RabbitMQ 和 ActiveMQ 使用不同的消息模型，所以它们的集群部署方式存在一些差异。
### 3.2.1 RabbitMQ 的集群方案
在 RabbitMQ 中，消息都通过交换机分发到队列，生产者和消费者通过 exchange 和 queue 进行通信。
要将 RabbitMQ 集群起来，首先要将各个节点进行集群部署，然后将 exchange 和 queue 分布到各个节点。
这种方案存在如下缺陷：
- 性能瓶颈：如果队列或交换机过多，将会导致各节点间的数据同步压力增大。
- 节点宕机风险：节点宕机后，可能造成数据丢失。
- 新节点加入时同步麻烦：当新节点加入集群后，其他节点需要同步消息。
- 运维成本高：如果要扩容集群，需要停止服务，修改配置，再重新启动服务。
- 管理复杂：集群环境下需要管理多个节点的配置。
### 3.2.2 ActiveMQ 的集群方案
在 ActiveMQ 中，Broker 根据其角色，可以分成四类：Broker（消息代理服务器）、Producer（消息生产者）、Consumer（消息消费者）、NetworkConnector（网络连接器）。
要将 ActiveMQ 集群起来，首先要将消息代理服务器以集群的方式部署在多台机器上，然后配置 network connector 将它们连接到同一个消息服务网格，这样就可以形成集群。
这种方案具有如下优点：
- 节点动态添加和删除：可以随时动态地增加或删除 Broker 节点，不影响消息的投递。
- 网络分区容错：在集群中，消息可以广播到所有 Broker 节点，因此即使网络出现分区，也可以确保消息的投递。
- 高可用性：如果某个 Broker 节点出现故障，其他节点仍然可以正常运行，消息也不会丢失。
- 管理简单：由于角色不同，Broker 节点可以简单处理，而其他节点则不必考虑，管理起来很方便。
ActiveMQ 集群配置要求比较高，有时候难以做到自动化，需要手工修改配置。
## 3.3 数据存储方面
因为 RabbitMQ 和 ActiveMQ 采用的存储方案不同，所以它们的数据存储位置也是不同的。
### 3.3.1 RabbitMQ 的数据存储位置
RabbitMQ 使用的存储方案为基于文件的存储方案，默认的文件存储路径为 /var/lib/rabbitmq。这个目录下有几个子目录：
-.erlang.cookie 文件保存着 erlang 的 cookie 值。
- rabbit 下的 subdirectories 表示队列和交换机的信息。
- messages 下的 subdirectories 表示消息的元数据。
- plugins 下的 subdirectories 表示插件的配置文件。
除了以上文件，RabbitMQ 还可以使用数据库来存储元数据信息。
### 3.3.2 ActiveMQ 的数据存储位置
ActiveMQ 默认的消息存储路径为 data/kahadb，它是一个文件系统的存储方案。
这个文件夹下有一个 kaha.log 文件用于记录消息，还有其他临时文件。
如果希望使用数据库来存储消息，则需要下载额外的 ActiveMQ 组件：JDBC 实现的 Store。这个组件可以将消息信息存储到关系型数据库中。