
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 Apache Kafka 是一种高吞吐量、分布式、可靠的分布式发布订阅消息系统。它被设计为一个统一的消息平台，能够轻松实现海量数据实时传送。Kafka 的主要目标是提供一个高吞吐量、低延迟、可扩展性的平台，能够支持消费者数量动态增减。它最初起源于 LinkedIn 的内部消息系统，之后成为独立项目。2011年，Apache Kafka 正式成为 Apache 顶级项目，并且由多个公司贡献者作为开发人员进行维护。
          
          2019年3月1日，Apache 软件基金会宣布启动 Apache Kafka 框架基金会，负责维护该框架并致力于改善其开发体验和能力。基金会成立目的在于促进 Apache Kafka 社区的发展和繁荣，推动 Apache Kafka 在企业应用中的流行。此外，基金会将推出 Kafka Training Program，帮助企业用户掌握 Apache Kafka 技术并有效使用它。
          
          本文档适用于对Apache Kafka有一定了解的人群。如无相关经验，建议先阅读官方文档学习相关概念和基础知识。
         # 2.基本概念术语说明
         ## 2.1 Apache Kafka
         ### 2.1.1 Apache Kafka 定义
         Apache Kafka 是一种高吞吐量、分布式、可靠的分布式发布订阅消息系统，它可以处理持续的数据流，同时保证数据完整性。Apache Kafka 是 Apache 项目下的一个子项目。
          
          Apache Kafka 一词来自于 Java 消息服务(Java Message Service，JMS) 中用到的同名组件。它是一个开源项目，由LinkedIn 公司开发。Kafka 以发布/订阅模式提供实时的 messaging 服务，具有以下特征：
          
          1. 高吞吐量：Kafka 提供了通过高效率的磁盘访问提升大数据量处理能力，同时还支持实时计算能力，可以在毫秒级内消费大量数据。
          2. 分布式：Kafka 支持集群部署，允许多个服务器同时提供消息服务，因此可以在不同的机器上并行处理数据，提升整体处理能力。
          3. 可靠性：Kafka 通过分区（Partition）和副本（Replica）机制实现数据的冗余备份，并通过 Broker 服务器的多副本机制实现数据可用性。同时还提供数据传输的 CRC32 校验码机制，确保数据传输的可靠性。
          4. 良好的容错性：当发生服务器节点故障或网络分区时，Kafka 仍然可以保持正常工作，不会丢失任何数据。
          
          
         ### 2.1.2 Apache Zookeeper
         ### 2.1.3 Broker 服务器
         简而言之，就是 Kafka 集群中的一个服务器，专门用来存储和转发消息。每个 Broker 服务器都有一个唯一的 ID 标识符，可用于跟踪它所接收到的消息。Broker 服务器主要分为控制器（Controller）和普通节点两种类型。控制器是 Kafka 集群的中央管理节点，负责管理集群元数据，分配 topic 和 partition 到对应的 broker 上面。另外，Broker 可以设置策略，决定哪些消息可以进入队列。通常情况下，至少需要三个 Broker 来组成 Kafka 集群。
         
         ### 2.1.4 Topic
         每条发布到 Kafka 集群的消息都有一个类别，这个类别就是 Topic。Topic 在逻辑上是一个分类名，物理上对应的是一个文件夹，这个文件夹保存着发送到这个主题的消息。消息在被投递到 Consumer 时，可以根据创建 topic 时指定的 key 来决定消息的传递顺序。例如，可以将所有登录事件按 userID 分组，这样就可以让相同 userID 的消息尽可能集中地传给同一个 consumer。
         
         ### 2.1.5 Partition
         Topic 是物理上的一个目录，每个目录下有若干个分区，每个分区是一个有序的、不可变的消息序列。其中每条消息都会分配一个整数序号作为它的偏移量（offset），然后根据 offset 对消息进行排序。分区中的消息也是按照 ProduceRequest 请求的顺序追加到日志文件末尾。这里，一个 Partition 可以视作是一个“队列”，里面的消息都是按照 offset 有序的。
         
         ### 2.1.6 Producer
         即消息的发布者，就是向 Kafka 集群中写入消息的客户端应用程序。
         
         ### 2.1.7 Consumer
         即消息的消费者，就是从 Kafka 集群读取消息的客户端应用程序。
         
         ### 2.1.8 Offset
         表示消费者当前消费到了多少条消息。生产者和消费者都各自维护自己的 offset，表示自己消费了多少消息。由于 Kafka 只保留最近消费过的消息，所以当消费者意外失败后，它可以通过 offset 来重新消费之前没消费完的消息。
         
         ### 2.1.9 Group
         即消费者的组，一般一个 group 下可以包含多个 consumer，他们共同消费一个或多个 topic 中的消息。如果某个 consumer 出现问题，另一个 consumer 可以接管他的工作，继续消费 topic 中的消息。
         
         ## 2.2 安装与配置
         ### 2.2.1 安装
         #### 2.2.1.1 下载安装包
            https://www.apache.org/dyn/closer.cgi?path=/kafka/2.3.0/kafka_2.12-2.3.0.tgz&as_json=1
           curl -o kafka_2.12-2.3.0.tgz http://mirrors.tuna.tsinghua.edu.cn/apache/kafka/2.3.0/kafka_2.12-2.3.0.tgz
           tar xzf kafka_2.12-2.3.0.tgz
           cd kafka_2.12-2.3.0
         #### 2.2.1.2 配置环境变量
           vi ~/.bashrc
            export KAFKA_HOME=$PWD/kafka_2.12-2.3.0
            export PATH=$PATH:$KAFKA_HOME/bin
           source ~/.bashrc
         #### 2.2.1.3 设置启动脚本
           cp $KAFKA_HOME/config/server.properties $KAFKA_HOME/config/server.properties.bak
           vi $KAFKA_HOME/config/server.properties
           listeners=PLAINTEXT://:9092
           advertised.listeners=PLAINTEXT://10.0.0.1:9092
           log.dirs=/var/log/kafka-logs
        #### 2.2.1.4 启动集群
           nohup bin/kafka-server-start.sh config/server.properties > logs/server.log &
         #### 2.2.1.5 查看集群状态
           bin/kafka-topics.sh --list --bootstrap-server localhost:9092

         ### 2.2.2 配置
         #### 2.2.2.1 创建topic
            bin/kafka-topics.sh --create --zookeeper localhost:2181 \
                             --replication-factor 1 --partitions 1 \
                             --topic test
         #### 2.2.2.2 修改配置
            vi config/server.properties
            listeners=PLAINTEXT://10.0.0.1:9092,SSL://localhost:9093
            ssl.keystore.location=/tmp/kafka.keystore.jks
            ssl.keystore.password=<PASSWORD>
            ssl.key.password=<PASSWORD>
            security.inter.broker.protocol=SSL
            producer.security.protocol=SSL
            consumer.security.protocol=SSL
            min.insync.replicas=2
            delete.topic.enable=true
            
         ### 2.2.3 测试
         #### 2.2.3.1 测试Producer
            bin/kafka-console-producer.sh --broker-list localhost:9092 \
                                           --topic test \
                                           --producer-property security.protocol=SSL
    