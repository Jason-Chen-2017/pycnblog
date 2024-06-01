
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1. Kafka是一个开源分布式消息系统，它由LinkedIn公司开发并开源，是Apache软件基金会下的顶级项目。Kafka最初起源于一个分布式日志收集系统，后来被用于在微服务架构中作为异步通信工具，主要解决数据实时同步、削峰填谷、故障转移等问题。
         2. 在使用过程中，由于Kafka作为一个分布式系统，它自身也具有一定的复杂性，如集群规划、配置参数设置、存储管理、性能调优、安全认证等。同时，Kafka还支持多种编程语言，包括Java、Scala、Python、Go、C/C++、Ruby等，可以方便地集成到各种应用场景中。
         3. 本文将介绍Kafka集群运维实践的一些知识点，内容包括：Kafka集群安装部署；Kafka集群扩容缩容；Kafka集群分区分配策略；Kafka集群重平衡优化；Kafka集群监控告警方案；Kafka集群升级维护。文章涉及的主要知识点和技能要求如下所示：
         1. Linux操作系统基础
         2. Kafka相关组件（Broker、Zookeeper）、CLI命令
         3. Java、Python、Shell脚本编程能力
         4. Kafka集群运维相关经验、管理经验及心得 
         5. 有关软件工程和系统架构设计理念，具备较强的学习能力 
         最后，本文的目的是为了帮助读者了解到Kafka集群运维方面的知识和技能，并且能够提升自己对Kafka集群的理解、掌握、运用、管理。
         # 2.基本概念与术语说明
         ## 2.1 Apache Kafka概念与术语
         ### Apache Kafka的定义
         Apache Kafka(简称Kafka)是一个开源分布式流处理平台，由Linkedin公司开发并开源，是Apache软件基金会下开源项目，是一种高吞吐量的分布式发布-订阅消息系统，它最初起源于LinkedIn的生产消息流需求，2011年10月开源，由Apache软件基金会孵化器捐赠给了Apache基金会，目前是 Apache 软件基金会顶级项目之一。该项目以BSD许可协议开源，版本号从0.9.0.0开始，官网为https://kafka.apache.org/.
         
         ### Apache Kafka的基本概念
         - Broker: Kafka集群中的服务器节点，每个服务器都是一个broker。Kafka集群中负责数据的分发和存储。所有的写入请求都会先到达一个随机的broker，然后该broker把数据复制到其它broker。所有读取请求都会在broker之间进行负载均衡。
         - Topic: 消息的主题。每个topic可以看作是一个分类名称或一个频道。生产者向某个topic发送消息，消费者则从同一个或者多个topic订阅并消费这些消息。Topic由多个Partition组成，每个Partition是一个有序的队列。每个Partition可以存在于不同的服务器上，以实现扩展性。
         - Partition: 每个Topic包含一个或多个Partition。每条消息被发送到一个特定的Partition，这样可以保证该消息被不同消费者消费的顺序性。一个Partition就是一个提交日志，保存着要被发送到Broker的消息。
         - Producer: 消息的发布者。一个Producer通过Kafka提供的API往Kafka集群中写入消息。
         - Consumer: 消息的消费者。一个Consumer通过Kafka提供的API从Kafka集群中读取消息。
         - Message: 一段文本、图片、视频或者其他二进制字节流。
         - Offset: 消息在partition中的位置信息，用来标识消息的位置。
         - ZooKeeper: Kafka依赖ZooKeeper来存放集群元数据，比如各个Broker的地址信息、每个Partition的Leader选举情况等。
         - API: Kafka提供了多种客户端API，用户可以通过它们向Kafka集群发送消息或者接收消息。典型的客户端API有Java、Python、Scala、C/C++。
         
         ### Apache Kafka的术语
         - Leader: 每个Partition都有一个Leader。Leader是Partition内唯一的序列消息的生成者。
         - Follower: 每个Partition都有多个Follower，Follower是Leader的追随者。当Leader出现故障时，其余的Follower会自动接管Partition，继续提供服务。
         - ISR (In Sync Replica): 是指与Leader保持同步的非只读副本。
         - HW (High Watermark): 是指每个Partition内最后一条已提交消息的Offset值。
         - LEO (Log End Offset): 是指所有副本的HW取最大值的副本，也就是说，LSO是HW更新的一个障碍物，所以需要重新计算LSO。
         - EoS (End of Segment): 是指Segment文件中最后一个字节的偏移量。
         
         ## 2.2 Zookeeper相关术语
         ### 什么是Zookeeper？
         Zookeeper是一种集中管理分布式应用程序配置和命名的开放源码分布式协调服务，它为分布式应用提供了一致的配置中心和命名服务，并允许集群中的客户动态地发现彼此，并进行相互协作。Zookeeper的目标是在分布式环境中提供一种透明且高效的机制来维护和监视数据变化。
         
         ### Zookeeper的作用
         Zookeeper的作用主要如下：
         1. 配置管理：Zookeeper能够存储和管理各种配置信息，并让客户端在运行时能够实时的获取最新的配置信息，因此可以在不停止服务的情况下调整程序的行为。
         2. 集群管理：Zookeeper能够提供基于领导者选举的主备切换，使得集群中各个节点的数据始终保持同步。
         3. 命名服务：Zookeeper能够提供高性能的分布式协调服务，为分布式应用提供基于名称空间的结构。
         4. 分布式锁：Zookeeper提供了一个简单而独特的分布式锁实现方式。
         5. 集群管理：在实际的分布式环境中，经常需要对集群中的服务器做一些管理操作，譬如添加删除节点，增加新机器等。Zookeeper天生就具有集群管理功能，通过路径注册表，监听器和时间戳，能够很好的处理这些工作。
            
         ### Zookeeper术语
         - Client: Zookeeper客户端。
         - Server: Zookeeper服务器，集群由一组Server构成。
         - Node: Zookeeper中的树状结构中的一个结点。
         - Pseudo-Node: Zookeeper中的一些结点比较特殊，称为伪节点。
         - Watcher: 对Zookeeper树中的结点设置的监听器，一旦结点的内容或者子节点发生改变，立即通知客户端进行相应的处理。
         - Transaction: 数据更新事务，一次完整的业务逻辑操作，包括create，delete，set，get等。
         - Data Model: 树状数据模型，结点之间通过路径连接起来。
         - ACL: Access Control List，权限控制列表，控制对Zookeeper节点的访问权限。
         - Quorum: 当且仅当半数以上节点正常工作时，集群才处于正常状态，可用状态，否则称为失效状态。
           
         # 3.Kafka集群安装部署
         ## 3.1 安装准备
         ### 操作系统
         Linux操作系统建议选择CentOS或Ubuntu，由于Kafka依赖Java运行，因此操作系统安装好之后，需要安装Java环境。
         
         ### JDK安装
         根据操作系统版本，安装对应的JDK，这里以CentOS 7.5为例，下载JDK 1.8到/usr/local目录下：
         ```shell
         cd /usr/local
         wget --no-check-certificate --no-cookies --header "Cookie: oraclelicense=accept-securebackup-cookie" http://download.oracle.com/otn-pub/java/jdk/8u211-b12/965bab60dc17c9282f292a2d33e1ddd2/jdk-8u211-linux-x64.rpm
         yum localinstall jdk-8u211-linux-x64.rpm
         ```
         设置JAVA_HOME变量：
         ```shell
         vi /etc/profile
         export JAVA_HOME=/usr/local/jdk1.8.0_211
         export PATH=$PATH:$JAVA_HOME/bin
         source /etc/profile
         java -version # 测试是否安装成功
         ```
         ### 配置文件准备
         创建kafka安装目录并进入：
         ```shell
         mkdir kafka && cd kafka
         ```
         创建配置文件并编辑：
         ```shell
         cp $KAFKA_HOME/config/*.
         vim server.properties
         ```
         添加以下内容到配置文件中：
         ```conf
         listeners=PLAINTEXT://localhost:9092
         log.dirs=/data/kafka-logs
         num.partitions=3
         default.replication.factor=3
         min.insync.replicas=2
         delete.topic.enable=true
         ```
         `listeners`项指定了Kafka监听的端口号，`log.dirs`项指定了日志文件的存储目录，建议使用独立的文件系统，避免影响到操作系统的磁盘IO。
         `num.partitions`项指定了默认创建的Topic的分区数量，`default.replication.factor`项指定了默认的副本因子，`min.insync.replicas`项指定了最小同步副本数，`delete.topic.enable`项指定了是否允许删除Topic。
         如果有多个节点部署Kafka，还需要修改`zookeeper.connect`项指向Zookeeper集群的地址，例如：
         ```conf
         zookeeper.connect=zk1.example.com:2181,zk2.example.com:2181,zk3.example.com:2181
         ```
         ## 3.2 安装启动Kafka
         ### 使用yum安装
         推荐使用yum直接安装，首先需要导入RPM签名：
         ```shell
         rpm --import https://www.apache.org/dist/kafka/KEYS
         ```
         将下载到的文件复制到YUM的仓库目录：
         ```shell
         mv *.rpm /etc/yum.repos.d/
         ```
         更新YUM缓存：
         ```shell
         yum clean all && yum update -y
         ```
         安装Kafka：
         ```shell
         yum install -y kafka
         ```
         ### 使用二进制包安装
         从官网下载对应版本的压缩包，解压到任意目录即可，然后进入解压后的文件夹并执行启动脚本：
         ```shell
         tar xzf apache-kafka_2.12-2.3.0.tgz
         cd apache-kafka_2.12-2.3.0
         bin/zookeeper-server-start.sh config/zookeeper.properties &
         sleep 5
         bin/kafka-server-start.sh config/server.properties &
         ```
         上面脚本中，`config/zookeeper.properties`是Zookeeper的配置文件，`config/server.properties`是Kafka的配置文件。`&`符号表示后台运行。由于Zookeeper一般部署在单独的一台服务器上，因此在启动Kafka之前需要先启动Zookeeper。
         启动完成后可以使用`jps`命令查看进程状态：
         ```shell
         jps
         ```
         可以看到其中有KafkaMain和QuorumPeerMain两个进程，代表Kafka已经正常启动。
         ### 命令行管理工具
         Kafka提供了命令行管理工具，能够通过简单的命令来管理Kafka集群，包括查看集群信息、创建和删除Topic、查询和消费Topic消息等。
         下载安装命令行管理工具：
         ```shell
         curl -o kafka-cli.tgz http://mirror.bit.edu.cn/apache/kafka/2.3.0/kafka_2.12-2.3.0.tgz
         tar xf kafka-cli.tgz
         ln -s./kafka_2.12-2.3.0/bin/kafka-topics.sh ~/bin/kafka-topics
         ln -s./kafka_2.12-2.3.0/bin/kafka-console-consumer.sh ~/bin/kafka-console-consumer
         ln -s./kafka_2.12-2.3.0/bin/kafka-console-producer.sh ~/bin/kafka-console-producer
         ```
         以上命令会下载Kafka的压缩包并解压到当前目录，并创建三个链接文件，分别指向三个命令行工具的脚本文件。
         创建Topic：
         ```shell
         kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 3 --partitions 3 --topic test
         ```
         删除Topic：
         ```shell
         kafka-topics.sh --delete --zookeeper localhost:2181 --topic test
         ```
         查看Topic信息：
         ```shell
         kafka-topics.sh --list --zookeeper localhost:2181
         ```
         查看Topic详细信息：
         ```shell
         kafka-topics.sh --describe --zookeeper localhost:2181 --topic test
         ```
         生产和消费消息：
         ```shell
         kafka-console-producer.sh --broker-list localhost:9092 --topic test < /dev/urandom
         ```
         ```shell
         kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic test --from-beginning
         ```
         `--broker-list`选项指定Kafka集群的地址，`--bootstrap-server`选项也可以指定，但是最好不要混用这两个选项。`--from-beginning`选项指定消费最早消息。`-timeout`选项指定超时时间。
         注意：生产和消费消息的命令不能放在一起，因为生产和消费消息都是阻塞的命令，如果没有消费端启动，生产端还是会等待，导致生产端无限等待。
         # 4.Kafka集群扩容缩容
         ## 4.1 分区扩容
         ### 通过增加分区来扩容
         分区数只能增加，不能减少。扩容的过程主要分为两步：
         1. 添加新分区。
         2. 迁移数据。
         执行以下命令增加分区：
         ```shell
         kafka-topics.sh --alter --zookeeper localhost:2181 --topic my_topic --add-partitions 2
         ```
         参数说明：
          - `--alter`: 表示对Topic做变更，而不是新建。
          - `--zookeeper`: 指定Zookeeper集群地址。
          - `--topic`: 指定Topic名。
          - `--add-partitions`: 表示新增分区数。
         增加分区之后，需要手动的迁移数据到新分区中。
         
        ### 通过副本因子扩容
         默认情况下，Kafka会在集群中创建一个分区(Partition)，一个分区会有一个leader，其他副本是follower。因而，当broker宕机或磁盘损坏时，集群会自动的检测到这一事实，并重新分布分区，确保数据完整性。如果有多个副本，就会形成ISR(In-Sync Replicas集合)，只有ISR集合中的副本才能参加到leader选举。如果ISSR集合中的副本数小于最小同步副本数，则会触发LeaderAndIsrNotAvailable异常，需要人工介入处理。
         
         Kafka提供了配置参数来自动扩容，包括如下几个：
          - `num.partitions`: 初始分区数，默认为1。
          - `default.replication.factor`: 初始副本因子，默认为1。
          - `min.insync.replicas`: 最小同步副本数，默认为1。
          - `auto.expand.replicas`: 是否允许自动扩容，默认为false。
         但是默认情况下，Kafka不会自动扩容，需要人工的执行扩容操作。如果需要开启自动扩容，可以将`auto.expand.replicas`设置为true，并设置合适的`num.partitions`。Kafka会根据分区数和副本因子自动分配Partition。
         具体的扩容操作如下：
         1. 修改配置文件。
         2. 执行扩容命令。
         修改配置文件：
         ```conf
         auto.expand.replicas=true
         num.partitions=<new_partition_count>
         ```
         执行扩容命令：
         ```shell
         kafka-configs.sh --zookeeper localhost:2181 --entity-type topics --entity-name my_topic --alter --add-config max.message.bytes=<max_message_size> --add-config flush.messages=<flush_interval_messages> --add-config retention.bytes=<retention_size> --add-config segment.ms=<segment_size_in_milliseconds>
         ```
         参数说明：
          - `--zookeepr`: 指定Zookeeper集群地址。
          - `--entity-type`: 表示修改实体类型，这里修改的是Topic。
          - `--entity-name`: 指定Topic名。
          - `--alter`: 表示对实体做变更。
          - `--add-config`: 表示增加配置。
         执行完毕后，可以通过`kafka-topics.sh --describe`命令查看分区的分布情况。
       
       ## 4.2 副本扩容
        Kafka提供了几个配置参数来控制副本因子的数量：
         - `default.replication.factor`: 初始化的副本因子，默认为1。
         - `min.insync.replicas`: 最小同步副本数，默认为1。
         - `num.replica.fetchers`: 副本拉取线程数，默认为1。
        其中，`min.insync.replicas`设置的是ISR集合中的副本数，当ISR集合中的副本数小于最小同步副本数时，不会参与投票。`num.replica.fetchers`设置的是拉取线程数，该参数决定了Kafka集群可以同时处理的最大请求数。
        
        扩容副本因子的方式：
        1. 修改配置文件。
        2. 执行扩容命令。
        修改配置文件：
        ```conf
        default.replication.factor=<new_replication_factor>
        ```
        执行扩容命令：
        ```shell
        kafka-reassign-partitions.sh --zookeeper localhost:2181 --reassignment-json-file reassign.json --execute
        ```
        参数说明：
         - `--zookeeper`: 指定Zookeeper集群地址。
         - `--reassignment-json-file`: 指定扩容计划。
         - `--execute`: 表示执行扩容计划。
        执行完毕后，可以通过`kafka-topics.sh --describe`命令查看分区的分布情况。
       
       ## 4.3 Kafka集群缩容
        当Kafka集群不再需要某个topic的时候，可以通过缩容操作来释放资源。但Kafka不支持在线的缩容操作，只能先停止集群，清除数据，然后重启集群，释放资源。
        
        1. 停掉Kafka集群：
        ```shell
        pkill -9 java
        ```
        2. 清理数据：
        ```shell
        rm -rf /path/to/kafka-logs
        ```
        3. 重启集群：
        ```shell
        nohup kafka-server-start.sh /path/to/kafka/config/server.properties > logs/server.log 2>&1 &
        nohup kafka-run-class.sh kafka.admin.ReassignPartitionsTool --bootstrap-server localhost:9092 --generate --topics-to-move-json-file /tmp/topics-to-move.json --broker-list broker1,broker2,broker3 >> /var/log/kafka/reassign.log 2>&1 &
        ```
       此外，还有其他方法可以实现缩容操作，比如直接删除topic，关闭相应的partition等。不过这些方法需要先对集群进行滚动升级，可能会引起服务不可用，不建议使用。
       
      # 5.Kafka集群分区分配策略
      ## 5.1 为何需要分区分配策略
      Kafka集群中的分区对于消息的处理非常重要，尤其是在集群扩容和负载均衡等情况下。由于Kafka集群中的分区属于有序的队列，因此对于同一个主题的消息的顺序性也是至关重要的。
      但如果没有合适的分区分配策略，消息的顺序可能就会乱掉。因此，分区分配策略就是为了解决这个问题，即如何将消息平均的分布到Kafka集群中的多个分区中，让同一个主题的消息都落在固定的分区上。
      
      ## 5.2 Kafka的默认分区分配策略
      Kafka提供了两种分区分配策略，分别是`range`和`roundrobin`，前者按范围分区，后者轮询分区。下面通过具体例子来展示这两种策略。
      ### Range分区策略
      Range分区策略是默认的分区分配策略，将主题的消息分布到固定的分区上。假设主题名为my_topic，初始分区数为3，则分配的分区规则如下图所示：
      
      | Partitions|           Partition Ids            |               Brokers                |
      |:--------:|:----------------------------------:|:------------------------------------:|
      |    0~1   | 0, 1                               |    b1(p0), b2(p1), b3(p2), b4(p0)     |
      |    2~2   | 2                                  | b1(p2), b2(p1), b3(p2), b4(p0), b5(p0)|
      
      其中，Partition Id是分区在主题中的编号，Brokers是分区所在的Broker。
      
      以生产者客户端为例，发送消息时会选择一个分区作为key，通过key-hash的方式映射到固定的分区。如果没有key，则轮询选择分区。例如，当key为“alice”时，映射到的分区为“0”，“1”，“2”。
      
      ### RoundRobin分区策略
      RoundRobin分区策略是另一种分区分配策略，这种策略将主题的消息平均的分配到所有可用的分区上。假设主题名为my_topic，初始分区数为3，则分配的分区规则如下图所示：
      
      | Partitions|           Partition Ids            |               Brokers                |
      |:--------:|:----------------------------------:|:------------------------------------:|
      |    0~1   | 0, 1                               |    b1(p0), b2(p1), b3(p2), b4(p0)     |
      |    2~2   | 2                                  | b1(p2), b2(p1), b3(p2), b4(p0), b5(p0)|
      
      与Range分区策略不同的是，RoundRobin策略没有考虑每个分区的大小。也就是说，分区A可能比其他分区要多很多消息。
      
      以生产者客户端为例，消息会被平均的分配到所有可用的分区上。例如，当有三条消息需要发送时，分别选择分区0和1、分区2的任何一个。
      
    ## 5.3 自定义分区分配策略
    用户可以自定义分区分配策略，这需要编写Java代码。用户可以按照自己的业务需求来实现分区分配策略，如根据消息大小、消息时间戳等。
    
    下面以消息大小为例，演示如何自定义分区分配策略。
    1. 编写Java类。
    ```java
    public class MyCustomPartitioner implements Partitioner {
        @Override
        public int partition(String topic, Object key, byte[] keyBytes, Object value, byte[] valueBytes, Cluster cluster) {
            // 获取分区总数
            int numPartitions = cluster.partitionsForTopic(topic).size();
            
            // 获取消息大小
            int messageSize = valueBytes.length;
            
            // 按照消息大小分配分区
            return Math.abs(messageSize % numPartitions);
        }
    }
    ```
    
    2. 修改配置文件。
    ```conf
    producer.partitioner.class=your.package.MyCustomPartitioner
    ```
    
    3. 编译Java文件。
    ```shell
    javac your/package/MyCustomPartitioner.java
    ```
    
    4. 拷贝jar包到kafka目录下。
    ```shell
    cp your/package/*.jar /path/to/kafka/libs
    ```
    
 # 6.Kafka集群重平衡优化
   ## 6.1 问题背景
   随着时间的推移，Kafka集群会越来越多的分区，会越来越多的消费者消费这些分区。当集群的分区数大于消费者的个数时，集群的消费速率会降低，这就造成了消息积压的问题。
   
   当消费者的个数大于分区的个数时，Kafka集群无法为消费者分配分区，会出现如下的错误：
   ```text
   ERROR [ReplicaManager on broker 1]: Error creating consumer: Cannot assign partitions to this consumer since the subscription is inconsistent with the current assignment
   ```
   此外，在某些情况下，Kafka集群可能会重新分配分区，这就意味着消息的重新排序。
   
   ## 6.2 主动重平衡（Manual Rebalancing）
   对于主动重平衡，需要Kafka管理员手动调用命令，该命令会触发集群的重新分配。
   1. 停止消费者。
   2. 执行主动重平衡。
   3. 启动消费者。
   例如，若需对主题test进行重平衡，则执行如下命令：
   ```shell
   kafka-preferred-replica-election.sh --zookeeper localhost:2181 --path "/brokers/topics/test"
   ```
   
## 6.3 自动重平衡（Automatic Rebalancing）
Kafka提供了两种自动重平衡策略：`sticky.partitions`和`cooperative.intra.node.assignor`。

1. sticky.partitions: 该策略的思想是将特定分区固定到特定的消费者节点上。这种策略会为每个消费者分配固定的分区，确保不会发生分区的重新分配。但是这种策略不能够避免消费者之间的竞争，可能会造成某些消费者的流量过高。另外，由于固定的分区数量限制了集群的扩展性，因此建议配合`unclean.leader.election.enable`设置为false来禁止unclean leader选举。
   1. 修改配置文件。
    ```conf
    enable.auto.commit=true
    auto.commit.interval.ms=1000
    session.timeout.ms=15000
    group.id=<group_id>
    partition.assignment.strategy=org.apache.kafka.clients.consumer.StickyAssignor
    unclean.leader.election.enable=false
    ```
   2. 启动消费者。
   3. 检查消费者日志。
   
2. cooperative.intra.node.assignor: 该策略在分配分区时会尽量均匀地将分区分配给各个消费者节点，避免消费者之间的竞争，并且可以在消费者发生崩溃时继续分区的分配。
   1. 修改配置文件。
    ```conf
    enable.auto.commit=true
    auto.commit.interval.ms=1000
    session.timeout.ms=15000
    group.id=<group_id>
    partition.assignment.strategy=org.apache.kafka.clients.consumer.CooperativeStickyAssignor
    coop.enabled.state=True
    node.ids=<broker1>,<broker2>,...
    ```
   2. 启动消费者。
   3. 检查消费者日志。

# 7.Kafka集群监控告警方案
## 7.1 监控指标
### 7.1.1 JVM监控
- **堆内存**：HeapMemoryUsage
- **非堆内存**：NonHeapMemoryUsage
- **GC次数和频率**：GcInfo
- **线程池大小**：ThreadPoolInfo
### 7.1.2 客户端监控
- **连接数**：connectedClientCount
- **消息积压数**：incomingMsgCountLag
### 7.1.3 Broker监控
- **持久化消息量**：UnderReplicatedPartitions
- **分区数和大小**：TopicMetrics中的PartitionCount和MeanPartitionSize
- **网络带宽利用率**：NetworkInputRate和NetworkOutputRate
- **磁盘读写**：DiskUsage
- **生产者和消费者个数**：numProduceRequests和numConsumerFetchRequests
- **操作请求和响应时间**：RequestHandlerAvgIdlePercent和ResponseQueueTimeMs
## 7.2 Prometheus+Grafana监控方案
Prometheus是一个开源的监控系统和报警工具，主要用于监控集群的各种指标，包括CPU、内存、磁盘、网络、JVM等。Grafana是一个开源的可视化工具，用于对监控数据进行展示，并提供丰富的图表功能。

下面介绍使用Prometheus+Grafana对Kafka集群进行监控的方案。

### 7.2.1 安装Prometheus
#### 7.2.1.1 CentOS安装Prometheus
```shell
sudo yum install epel-release
sudo yum install prometheus
```
#### 7.2.1.2 Ubuntu安装Prometheus
```shell
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo 'deb https://mirrors.tuna.tsinghua.edu.cn/prometheus-release/debian/ $(lsb_release -cs) main' | sudo tee /etc/apt/sources.list.d/prometheus.list
sudo apt-get update
sudo apt-get install prometheus
```
### 7.2.2 安装Grafana
#### 7.2.2.1 CentOS安装Grafana
```shell
sudo yum install grafana
```
#### 7.2.2.2 Ubuntu安装Grafana
```shell
wget https://dl.grafana.com/oss/release/grafana_7.3.6_amd64.deb
sudo dpkg -i grafana_7.3.6_amd64.deb
```
### 7.2.3 配置Prometheus
#### 7.2.3.1 编辑prometheus.yml
```yaml
global:
  scrape_interval:     15s 
  evaluation_interval: 15s 

scrape_configs:
  - job_name:       'kafka' 
    static_configs: 
      - targets: ['localhost:9092']
```
#### 7.2.3.2 启动Prometheus
```shell
sudo systemctl start prometheus
```
### 7.2.4 配置Grafana
#### 7.2.4.1 打开浏览器输入http://<主机ip>:3000，创建新数据源，选择Prometheus，输入http://localhost:9090，选择保存并测试。