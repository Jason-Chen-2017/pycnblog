
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Apache Kafka 是一款开源分布式流处理平台，由LinkedIn开发并维护。它最初由Cloudera公司开源，之后成为Apache顶级项目，并于2011年7月进入 Apache 孵化器。2018年9月，Kafka 宣布进入 Apache 基金会孵化。
Kafka 是一个分布式、可扩展、高吞吐量、多分区、多副本、基于Zookeeper协调的消息系统。它可以应用在大数据实时分析、日志采集、即时消息传输、事件源ourcing等领域。主要特性如下：

1. 消息队列：Apache Kafka 是一个开源的分布式消息系统，是一个高吞吐量、低延迟的分布式发布订阅消息系统，具有高容错性和可靠性。

2. 可扩展性：Apache Kafka 集群可以水平扩展，单个集群支持百万级别的消息量和TB级的数据量。

3. 数据完整性：Apache Kafka 以分布式、高容错的方式存储数据，确保数据不丢失，也不会造成数据重复消费。

4. 分布式计算：Apache Kafka 支持水平可扩展的分布式计算，能够轻松应对数据量、并发性等多变的环境变化。

5. 消息顺序性：Apache Kafka 提供了一个按顺序生产和消费消息的机制，保证数据不会乱序。

6. 高性能：Apache Kafka 在很短的时间内就能够处理数十亿的消息，每秒钟处理几百万条消息。

7. 统一发布/订阅模型：Apache Kafka 为用户提供了统一的发布/订阅模型，可以轻松实现不同类型数据的广播或订阅。

8. 消息过滤与投递策略：Apache Kafka 提供了丰富的消息过滤和投递策略，用户可以灵活地控制消息的消费方式。

Kafka 可以作为一种新的分布式流处理引擎，被广泛应用于日志处理、事件溯源、游戏服务器实时状态跟踪、运营监控、在线推荐系统、物联网数据采集、应用数据分析等众多领域。

# 2.基本概念术语说明
## 2.1 Apache Kafka 基本术语
### Topic（主题）
Kafka 中一个非常重要的概念就是 topic（主题）。它类似于数据库中的表格或者消息队列中的交换机，用于分类消息。topic 的消息在逻辑上被分割成一个个的 partition（分区），partition 中的消息可以被多个消费者进行并行消费。每个主题可以配置多个分区，每个分区可以有零个或多个副本，每个副本在磁盘上保存相同的数据。
图1：Topic（主题）示意图

- Partition（分区）：每个 topic 可以包含多个 partition。分区的作用主要是通过并行消费提升消费效率。Partition 是物理上的概念，每个分区都有一个唯一的标识符，称为 Partition ID，其值从0开始。
- Leader：每个分区都会有一个 leader，负责储存所有的消息和复制它们到 follower 上面。只有 leader 会接受写入请求，其他副本（follower）则为只读副本。如果 leader 宕机，某个 follower 将自动选举出来作为新 leader。所有读写请求都需要通过 leader 来完成。
- Consumer Group（消费者组）：Consumer Group 是 Kafka 允许消费者订阅主题并且批量消费消息的一个概念。一个 Consumer Group 可以包含多个 consumer，且多个 consumer 可以属于同一个 Consumer Group。同一个 Consumer Group 下的所有 consumer 共享这个主题的所有分区。每个 consumer 持续消费该分区中的消息直到消费完毕。但是不能把两个 consumer 分配给不同的 Partition。

### Producer（生产者）
Producer 是指向 Kafka 主题发送消息的客户端程序。Producer 可以将消息发布到指定的 topic 和 partition 上。为了达到最佳性能，生产者通常采用异步发送模式，即直接将消息放入内部的 buffer 缓存中，而不需要等待缓冲区满后再发送。因此，生产者发送消息的过程并不是立刻得到反馈，而是积攒到一定数量后批量发送，以减少网络IO次数。

### Consumer（消费者）
Consumer 是指从 Kafka 主题接收消息的客户端程序。Consumer 通过指定要消费的 topic、partition、offset 及消费者组 ID 订阅主题，然后通过向 Kafka 服务器拉取数据包的方式获取消息。Consumer 可以通过两种方式读取消息：

1. 拉模式（Pull Mode）：Consumer 通过主动轮询服务器拉取消息。这种方式要求 Consumer 长时间地空闲，因为它不断地向服务器请求消息，直到消息到达。这种模式比较简单，但缺点是会存在服务器拉取压力。

2. 推模式（Push Mode）：Consumer 通过向 Kafka 服务器注册自己所关心的 Topic、Partition、Offset，当消息追加到这些位置后，Kafka 服务器会主动推送消息给 Consumer。这种方式不需要 Consumer 长期地轮询，可以节省网络资源。

### Broker（代理节点）
Broker 是 Kafka 集群中的服务节点。它负责维持整个 Kafka 集群中 topics、partitions、replicas 的状态信息，接收和处理客户端的 API 请求。

### Offset（偏移量）
Offset 是消费者用来追踪自身位置的标记。在每个消费者组里，每个消费者都有一个 offset 位移指针。消费者消费消息时，会首先记录自己消费到的 offset 值。当某些消息被重新消费时，消费者的 offset 会被更新。比如，消费者 A 消费了一些消息，第二次消费时，A 的 offset 就会被更新。

## 2.2 Apache Kafka 术语详解
图2：Apache Kafka 术语详解