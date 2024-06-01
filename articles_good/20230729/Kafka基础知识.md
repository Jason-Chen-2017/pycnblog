
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　Apache Kafka是一个分布式流平台，主要应用于构建实时数据管道和流处理应用程序。Kafka是开源项目，由LinkedIn开源。它提供了高吞吐量、低延迟、可扩展性和容错能力。由于其内置了集群管理功能，因此可以方便地实现横向扩展。
          　　本文档从以下方面详细介绍Kafka：
           　　1）概述：包括Apache Kafka的定义、特性和优点；
            2）安装配置：介绍如何安装及配置Kafka，并介绍相关命令及工具；
            3）核心概念：介绍Kafka的核心概念，包括主题（Topic）、分区（Partition）、消息（Message）等；
            4）生产者和消费者：介绍如何通过生产者向Kafka主题发送消息，以及如何通过消费者订阅和消费Kafka主题中的消息；
            5）集群组成和管理：介绍Kafka集群的组成和管理方式，包括Broker、Controller、Zookeeper、Kafka-manager等组件；
            6）持久化机制：介绍Kafka中消息的持久化机制，包括磁盘存储和日志压缩两种方式；
            7）安全机制：介绍Kafka中的授权和加密配置，以及认证、SSL/TLS加密传输等安全措施；
            8）扩展性：介绍Kafka的性能调优、水平拓展和垂直拓展；
            9）其它高级特性：如消息顺序性保证、事务性消息支持、不同版本之间的兼容性等。
         　　对于Kafka初学者来说，这份文档将帮助他们快速入门，掌握Kafka的基本知识和用法，具有良好的学习效果。
        # 2.Kafka的概念
         ## 2.1 概念
         Apache Kafka 是一种高吞吐量、低延迟的数据传输系统。它的设计目标是作为一个分布式、可伸缩、 fault-tolerant 的 messaging system。它最初由 Linkedin 开发，并于 2011 年成为 Apache 顶级项目之一。

         Apache Kafka 是为了在分布式环境下提供实时的、可靠的、容错的消息传递服务而创建的。它允许多个发布者生产同样的消息，多个消费者消费这些消息。这种体系结构使得Kafka能够在大规模部署中提供低延迟和高吞吐量。

         Apache Kafka 使用 publish-subscribe 模型来组织数据。每个发布到 Kafka 的消息都被分配到特定的 topic。主题在逻辑上对消息进行分类，物理上划分为多个 partition。partition 是 Kafka 数据的物理隔离单位，每个 partition 可以被多台服务器上的多个消费者消费。同时，publishers 和 consumers 不需要知道其他消费者的存在，它们只需要向自己所关心的 topic 发布或订阅消息即可。

         通过这个模型，Kafka 提供了两个主要的优点：

         1. 容错性: Kafka 支持多副本机制，可以让数据更加可靠。当 broker 节点出现故障时，备份数据仍然可用。
         2. 可伸缩性: Kafka 在 producer 和 consumer 数量变化时可以线性增加。Kafka 的 partition 分布可以自动均匀，无需手动配置。
         ## 2.2 一些名词
         ### Topic
         每条发布到 Kafka 集群的消息都有一个类别，这就是 topic。一个topic可以有多个分区(partition)。topic的名字很重要，因为生产者和消费者都要指定 topic 才能消费该 topic 的消息。如果不存在则会被自动创建。

         ### Partition
         当producer把消息发送到kafka后，消息会被存储到一个或多个partition，每个partition是一个有序的队列。其中每一条消息都会被分配一个序号，用于排序。该序号由kafka生成，在整个kafka集群范围内唯一。partition里的数据可以分布到多个服务器上，以便kafka可以横向扩展。

         ### Consumer Group
         consumer group 是 kafka 中一个非常重要的概念，允许多个consumer共享一个topic的消息。这意味着你可以创建一个consumer group，它包含多个consumer，且每个consumer都消费相同的topic的消息。这样一来，一个topic的数据就被多个consumer共享，从而达到负载均衡的目的。每个consumer属于一个特定的 consumer group，同一个 consumer group 下的 consumer 消费的是同一个 topic 的不同分区(partition)的数据。

         每个consumer都有一个 ID，称为 member id。当新 consumer join 时，它会加入到 group 中，然后消费 topic 上剩余的消息。当一个 consumer crash 或网络中断导致连接断开时，它会重新加入 group，接着再次消费 topic 的消息。这也意味着，一个 consumer group 可以让某些 consumer 优先消费消息。例如，你可以设置优先级，让优先消费最新的消息的 consumer。

        ### ZooKeeper
        ZooKeeper 是 Apache Hadoop 中的子项目，是 Apache Kakfa 依赖的协调服务。它主要用来解决分布式环境中节点动态上下线的问题。

        每个 kafka 服务实例都会和 zookeeper 集群保持长连接。kafka 的很多功能都依赖于 zookeeper 来保存元数据信息，比如 topics、partitions 的路由信息等。另外，当 broker 节点发生故障时，zookeeper 会检测到该节点不可用，并通知相应的 kafka 服务实例。



        ## 3.Kafka核心概念
        本节将详细阐述Kafka的核心概念，包括主题（Topic）、分区（Partition）、消息（Message）等。
        ### 3.1 消息（Message）
        消息是指由主题传送到Kafka集群的记录。在Kafka中，消息有如下几个特征：
        - 消息是按照key-value形式存储的。key是可选字段，可以使消息更容易分割，value为消息的内容，通常是字节数组。
        - key不能重复，但可以设置为null。如果没有设置key，则默认使用partition和offset。
        - value可以重复，即使不设置key也可以。
        - 消息可以被订阅者选择过滤。
        - 消息可以设置过期时间，超过此时间之后消息将被删除。
        
        ### 3.2 分区（Partition）
        主题可以有多个分区。一个分区是一个有序的、不可变序列。每个分区都是有broker端本地磁盘存储，可以通过多副本机制提升数据冗余。一个分区只能属于一个主题，但可以被很多主题消费。
        
        ### 3.3 Broker
        一个Kafka集群由多个Broker组成。每个Broker是一个独立的进程，负责维护自身的数据和客户请求。Broker之间通过TCP协议通信。
        
        ### 3.4 Producers
        Producer是客户端应用，它负责产生(produce)消息并将其发送给Kafka集群。一个Producer可以向任意主题(topic)发送消息，并指定一个可选的分区键。如果不指定分区键，则Kafka会随机分配消息到不同的分区。
        
        ### 3.5 Consumers
        Consumer是另一个客户端应用，它负责消费(consume)主题(topic)中的消息。每个Consumer实例在启动时，都需要指定一个group_id，用于标识自己所属的Consumer Group。同一个Consumer Group下的所有Consumer实例都属于同一个消费者组，共同消费主题中的消息。
        
        ### 3.6 Replication
        一条消息可以复制到多个broker上，称为复制(replication)。为了提升消息可用性，我们可以在不同的机器上部署多个相同的broker，即副本(replica)。当一个broker失效时，另一个副本可以继续服务。副本数量越多，系统的可靠性越高。
        
        ## 4.Kafka安装配置
        本节将详细介绍Kafka的安装配置过程。

        ### 安装
        首先下载最新稳定版的Kafka压缩包，然后解压到某目录，假设为 /opt/kafka：

        ```bash
        $ wget http://mirrors.hust.edu.cn/apache/kafka/2.4.1/kafka_2.12-2.4.1.tgz
        $ tar xzf kafka_2.12-2.4.1.tgz -C /opt
        $ ln -s /opt/kafka_2.12-2.4.1 /opt/kafka
        ```

        将/opt/kafka/config目录下的server.properties文件复制一份，修改如下属性：

        ```ini
        listeners=PLAINTEXT://host.name:9092   //监听端口为9092
        log.dirs=/tmp/kafka-logs       //日志目录
        zookeeper.connect=localhost:2181      //zookeeper地址
        delete.topic.enable=true     //是否允许删除主题
       ```

        server.properties文件中的属性说明如下：

        | 属性名称            | 描述                             |
        | :------------- | :------------------------------ |
        | listeners           | 配置kafka使用的listener，包括`PLAINTEXT`(明文), `SSL`, `SASL_PLAINTEXT`, `SASL_SSL`四种类型，默认为`PLAINTEXT`，一般使用`PLAINTEXT`即可。 |
        | log.dirs            | 指定kafka的数据存放路径，目录可以有多个，用逗号隔开，若目录不存在，则创建之。 |
        | zookeeper.connect    | 指定zookeeper集群地址，主机名或IP地址和端口号用逗号隔开。 |
        | delete.topic.enable | 是否允许删除主题，默认为true。 |

        执行启动脚本，启动kafka：

        ```bash
        $ cd /opt/kafka
        $ bin/kafka-server-start.sh config/server.properties & 
        ```

        以上脚本会在后台运行kafka进程，日志保存在/var/log/kafka目录下。

        ### 创建主题
        Kafka不仅可以接收消息，还可以向其中写入消息。首先创建一个名为test的主题：

        ```bash
        $ bin/kafka-topics.sh --create --zookeeper localhost:2181 \
              --replication-factor 1 --partitions 1 --topic test
        Created topic "test".
        ```

        参数说明：
        - `--create`：表示创建一个新的主题。
        - `--zookeeper`：指定zookeeper地址。
        - `--replication-factor`：设置复制因子，即将数据复制几份。
        - `--partitions`：设置分区个数，即数据按照多少片存储。
        - `--topic`：设置主题名称。

        ### 查看主题列表
        命令：`$ bin/kafka-topics.sh --list --zookeeper localhost:2181`。

        如果正常执行，会返回当前zookeeper中所有的主题。

        ### 查看主题详情
        命令：`$ bin/kafka-topics.sh --describe --zookeeper localhost:2181 --topic test`。

        返回该主题的详细信息，包括分区个数、副本因子、ISR集合等。

        ### 删除主题
        命令：`$ bin/kafka-topics.sh --delete --zookeeper localhost:2181 --topic mytopic`。

        删除指定的主题，注意此操作无法撤回！

        ### 测试
        使用Kafka提供的测试工具kafkacat，向刚才创建的主题中写入一些消息，然后查看消息：

        ```bash
        $ echo 'hello' | kafkacat -b localhost:9092 -t test
        % Message delivered (topic=[0], offset=[0])
        ```

        此时，可以使用消费者消费该消息，命令如下：

        ```bash
        $ kafkacat -b localhost:9092 -t test -o beginning -c 1
        hello% Local: Broker port: 9092
        % Auto-selecting Consumer mode (use '-X list' to see options)
        hello
        ^CProcessed 1 messages in 1.4ms, 72 bytes
        ```

        `-b`: 指定Kafka集群的Broker地址，这里是`localhost:9092`。`-t`: 指定消费的主题名为`test`。`-o`: 设置读取消息的位置，这里是从头开始`-beginning`，`-p`: 设置读取消息的偏移量，这里没有设置`-e`: 设置命令结束的超时时间，这里没有设置`-c`: 设置消费线程的个数，这里是1。

        使用ctrl+c退出命令。

    ## 5.Producer和Consumer
    本节介绍Kafka中的生产者（Producer）和消费者（Consumer），以及它们的工作流程。

    ### Producer
    生产者是向Kafka集群发送消息的客户端应用。生产者以key-value对的形式发送消息到特定的主题。生产者可以选择发送到哪个分区，也可以不指定分区，Kafka会将消息随机分配到不同的分区。

    生产者发送消息的方法有两种：同步和异步。同步方法是在发送消息后等待broker确认消息已成功写入。异步方法是生产者把消息直接发送到对应的分区，并不需要等待broker的确认。

    ### Consumer
    消费者是从Kafka集群接收消息的客户端应用。消费者从主题订阅感兴趣的消息，消费者可以选择读取最近的消息、最早的消息、一个固定的offset或者消费者自己的进度（offset）。

    当消费者启动时，它会先确定自己属于哪个消费者组，并向zookeeper注册自己。当消费者读取了消息，它就会向zookeeper更新自己消费到的位置。当消费者宕机或者停止消费时，它会向zookeeper注销自己，从而其他消费者可以接替消费。

    ### 工作流程
    1. 生产者创建一个主题或者获得一个现有的主题。
    2. 生产者初始化一个生产者对象，指定生产者组，用于标记生产者身份。
    3. 生产者调用`send()`方法，向指定主题发送消息。`send()`方法会检查主题是否存在，如果不存在，会抛出异常。
    4. 生产者调用`flush()`方法，等待所有消息写入kafka。
    5. 消费者创建一个消费者对象，指定消费者组，用于标记消费者身份。
    6. 消费者调用`subscribe()`方法，订阅主题，可选择读取模式，包括latest（默认值）、earliest、beginning、end。
    7. 消费者调用`poll()`方法，获取一个待处理的消息，如果没有消息，则阻塞。
    8. 消费者处理消息，调用`commitSync()`方法提交位置，如果没有commit，则在下次启动时会重置。
    9. 如果消费者处理完毕，调用`close()`方法关闭。

    ### 示例代码
    以下是一个简单的Producer和Consumer的代码示例，演示了生产者和消费者的简单交互。

    #### Producer代码
    ```java
    import org.apache.kafka.clients.producer.*;
    
    public class SimpleProducer {
    
        private static final String TOPIC = "my-topic";
        private static final String BOOTSTRAP_SERVERS = "localhost:9092";
        
        public static void main(String[] args) throws Exception{
            
            Properties props = new Properties();
            props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, BOOTSTRAP_SERVERS);
            props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,
                    "org.apache.kafka.common.serialization.StringSerializer");
            props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,
                    "org.apache.kafka.common.serialization.StringSerializer");

            try (KafkaProducer<String, String> producer = new KafkaProducer<>(props)) {
                for (int i = 0; i < 10; i++) {
                    long time = System.currentTimeMillis();
                    String message = "Hello_" + time;

                    // Asynchronously send the message
                    producer.send(new ProducerRecord<>(TOPIC, message));
                    System.out.println("Sent message: (" + message + ")");
                    
                    Thread.sleep(1000);
                }
                
                producer.flush();
                
            } catch (Exception e) {
                e.printStackTrace();
            }
            
        }
        
    }
    ```

    **说明**：
    - `BootstrapServers`指定kafka集群的地址，这里是`localhost:9092`。
    - `KeySerializer`指定key序列化器，这里是`StringSerializer`。
    - `ValueSerializer`指定value序列化器，这里也是`StringSerializer`。
    - 在循环中，生成10个消息，并异步发送到主题。
    - 调用`flush()`方法等待所有消息写入kafka。

    #### Consumer代码
    ```java
    import org.apache.kafka.clients.consumer.*;
    import java.util.*;
    
    public class SimpleConsumer {
    
        private static final String GROUP_ID = "my-group";
        private static final String TOPIC = "my-topic";
        private static final String BOOTSTRAP_SERVERS = "localhost:9092";
        
        public static void main(String[] args) throws InterruptedException {
            Properties props = new Properties();
            props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, BOOTSTRAP_SERVERS);
            props.put(ConsumerConfig.GROUP_ID_CONFIG, GROUP_ID);
            props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, 
                    "org.apache.kafka.common.serialization.StringDeserializer");
            props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, 
                    "org.apache.kafka.common.serialization.StringDeserializer");

            try (KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props)) {

                // Subscribe the consumer to our topic
                List<String> topics = Collections.singletonList(TOPIC);
                consumer.subscribe(topics);

                while (true) {
                    // Poll for a message
                    ConsumerRecords<String, String> records = consumer.poll(100);

                    for (ConsumerRecord<String, String> record : records)
                        System.out.printf("Received message: (%d, %s, %d, %s)
",
                                record.partition(), record.key(), record.offset(), record.value());
                        
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
            
        }
        
    }
    ```

    **说明**：
    - `Group Id`指定消费者组，这里是`my-group`。
    - `KeyDeserializer`指定key反序列化器，这里是`StringDeserializer`。
    - `ValueDeserializer`指定value反序列化器，这里也是`StringDeserializer`。
    - 消费者订阅主题，读取最新的消息。
    - 循环调用`poll()`方法，每次最多返回100条消息。
    - 对返回的每条消息打印出主题分区、消息key、offset、消息value。

    ## 6.集群组成和管理
    本节将介绍Kafka集群的组成和管理方式。

    ### 组成
    Apache Kafka由一个或多个服务器组成，这些服务器被称为broker。每个broker运行一个Kafka进程，负责存储和转发消息。每个Kafka集群至少需要三个broker。

    有两种类型的broker：
    - 控制器（Controller）：Kafka集群中的一台或多台服务器，用于管理集群元数据和负责在broker间分配分区。
    - 代理（Broker）：除去控制器外的Kafka集群中的一台或多台服务器，用于存储和转发消息。

    ### 管理
    管理员可以通过命令行工具kafka-admin.sh和web UI控制集群。命令行工具包括创建、删除、描述、配置主题、管理分区、调整配置等命令。Web UI提供了一个图形界面，用户可以浏览集群中的主题、分区、消费者组等信息。

    ### Controller
    控制器是一个特殊的Kafka broker，负责管理集群元数据，分配分区以及各个代理的角色。控制器通过Zookeeper协调集群成员关系、分区的移动和失败检测。控制器在一个Kafka集群中只存在一台。

    控制器的主要职责有：
    - 跟踪集群成员关系。控制器周期性的向Zookeeper提交关于集群成员的信息。控制器在收到任何成员崩溃、分区重新分配、新增分区等通知时，会做出相应的动作。
    - 分配分区。控制器根据主题和群组的需要来决定在哪些代理上创建新的分区。
    - 监控代理的健康状态。控制器会定期向代理发送心跳信号，如果发现代理异常，控制器会触发重新分配分区或踢掉该代理。
    - 维持集群的高可用性。控制器通过选举产生新的控制器来实现高可用性。

    ### Broker
    代理是一个Kafka服务器，负责存储和转发消息。每个代理运行一个Kafka进程，可以存储多个分区，每个分区存储着特定主题的消息。

    代理的主要职责有：
    - 存储消息。代理接收来自生产者的消息，并将其保存在硬盘上。
    - 向消费者提供消息。当消费者向主题订阅时，代理向消费者提供消息。
    - 向控制器报告健康状况。每个代理定期向控制器发送心跳信号，表明自身的状态。

    ### 元数据
    Kafka集群中的所有数据都存储在topics和partitions中。topics和partitions的元数据存储在Zookeeper中。Zookeeper是Kafka集群的协调者，存储着Kafka集群的所有元数据。

    ### 副本
    每个分区都可以配置为多份副本，以提高数据可靠性。Kafka采用主从复制的方式，每个分区都有一个单独的主副本和零个或多个从副本。只有主副本接收生产者的写入请求，并且向消费者提供消息。

    主从复制允许Kafka集群具有高可用性。一旦某个代理出现故障，其它的代理会接管它，确保集群始终有且仅有一个可用的leader副本。

    副本的数量可以在创建主题时指定。创建主题时，可以指定每个分区的副本数量，但是不能超过总的broker数量。

    ## 7.持久化机制
    本节介绍Kafka中消息的持久化机制，包括磁盘存储和日志压缩两种方式。

    ### 磁盘存储
    Kafka将消息保存在磁盘上。在生产者端，生产者调用`send()`方法发送消息到指定分区，该方法返回一个Future对象，代表消息是否被成功写入。如果消息被成功写入，Future对象的get()方法会立即返回；否则，它会一直阻塞，直到消息写入完成。

    在消费者端，消费者调用`poll()`方法，获取一个待处理的消息，如果没有消息，则阻塞，直到获取到消息。当消费者处理完消息，调用`commitSync()`方法提交位置。提交位置后，消费者认为自己已经消费到了该消息。

    Kafka中消息的存储策略有两种：
    - 文件存储：Kafka中每个消息都保存在一个或多个文件中。文件的大小由参数`message.max.bytes`控制。
    - 索引存储：Kafka中每个分区都有一个索引文件，该文件存储着消息的偏移量、消息大小等元数据。该索引文件可以防止消息顺序错误。

    ### 日志压缩
    Kafka使用日志压缩功能来降低磁盘占用率。日志压缩可以将多个小的消息合并为更大的消息，以降低磁盘空间的占用。日志压缩是Kafka中可选的，通过`log.cleaner.enable`参数控制。

    日志压缩有两种策略：
    - 时间窗口：日志按时间窗口进行压缩，比如每天一份。这种压缩方式不会降低消息的吞吐量，但可以减少磁盘空间占用。
    - 大小限制：日志按大小限制进行压缩。当日志文件的大小超过`log.segment.bytes`时，Kafka会创建一个新的日志段，并压缩之前的日志。

    ### 批量消费
    Kafka的消费者可以批量消费消息。批量消费可以提高消费速度，因为消费者一次性读取多个消息，然后提交。批量消费通过参数`fetch.min.bytes`和`fetch.max.wait.ms`控制。

    参数`fetch.min.bytes`指定每次批量消费的最小字节数。`fetch.max.wait.ms`指定消费者最多等待的时间。如果当前批次的字节数小于`fetch.min.bytes`，则等待更多消息到达；如果当前批次的字节数大于等于`fetch.min.bytes`，则消费者立即返回。

    ### 消息顺序性
    Kafka的分区中消息的顺序性是通过分区编号和offset保证的。

    分区编号：消息首先被发送到哪个分区是由Kafka的分区控制器决定的。分区编号是一个整数，由0开始。当新分区创建后，分区控制器会为每个分区分配一个新的编号。

    Offset：每个消息都有一个唯一的offset。Offset是64位整型数字，在一个分区内唯一。

    消息的顺序性依赖于这两个因素：首先，Kafka保证一个分区内的消息顺序；其次，Kafka保证分区之间消息的顺序。由于Kafka的分区是有序的，所以消息被送往同一个分区中的消息一定是顺序送达的。同一个分区中的消息的顺序保证依赖于分区编号和offset。

    ## 8.安全机制
    本节介绍Kafka中的安全机制，包括授权和加密配置。

    ### 授权
    Kafka支持两种授权方式：
    - ACL：访问控制列表，包括读、写、创建、删除权限。ACL规则被保存到Zookeeper中。生产者和消费者都可以向Zookeeper申请权限。
    - SSL/TLS：Kafka支持SSL/TLS加密传输。生产者和消费者可以配置SSL证书，要求服务器验证客户端证书。

    推荐使用ACL授权。

    ### 加密
    Kafka支持两种加密方式：
    - PLAINTEXT：消息不经过加密。
    - SSL/TLS：消息经过SSL/TLS加密传输。生产者和消费者需要配置SSL证书。

    推荐使用SSL/TLS加密。

    ## 9.扩展性
    本节介绍Kafka的性能调优、水平拓展和垂直拓展。

    ### 性能调优
    性能调优主要通过三方面来进行：
    1. 选取合适的序列化方式：由于磁盘IO是影响性能的主要瓶颈，所以选择高效的序列化方式是提升性能的关键。
    2. 避免过多的网络IO：使用压缩方案来减少网络IO。
    3. 使用分区方案来提升并发处理能力：当有多块CPU或多核CPU时，可以将多个分区分配给不同的CPU来提升并发处理能力。

    ### 水平拓展
    水平拓展是指增加Kafka集群的节点数量。水平拓展涉及以下三个方面：
    - 添加更多的Kafka broker：通过增加Kafka集群中的broker节点来提升集群容量。
    - 为主题分配更多的分区：通过增加分区数量来提升消息处理的并发能力。
    - 为分区分配更多的副本：通过增加分区副本数量来提升消息可靠性。

    ### 垂直拓展
    垂直拓展是指提升Kafka集群的计算性能。垂直拓展涉及以下三个方面：
    - 升级硬件：通过升级硬件来提升Kafka集群的计算性能。
    - 升级软件：通过升级软件来提升Kafka集群的计算性能。
    - 使用云服务商：通过云服务商提供的Kafka服务来提升Kafka集群的计算性能。

    ## 10.其它高级特性
    本节介绍Kafka中的一些高级特性。

    ### 消息顺序性
    Kafka保证了分区内的消息顺序。为了保证分区之间消息的顺序，Kafka使用了基于flume和Bookkeeper的事务性消息支持。

    ### 事务性消息
    Kafka支持事务性消息，它允许用户发送和消费一系列消息，但这些消息要么都被写入，要么都不被写入。

    事务性消息提供了ACID（Atomicity，Consistency，Isolation，Durability）的属性。事务是一个一阶段提交协议，它包括准备、提交、中止和恢复阶段。

    ACID分别对应于以下四个特性：
    - Atomicity（原子性）：事务是一个不可分割的工作单元，事务中包括的诸操作要么都执行，要么都不执行。
    - Consistency（一致性）：在事务开始之前和事务结束以后，数据库的完整性约束必须得到满足。
    - Isolation（隔离性）：一个事务的执行不能被其他事务干扰。
    - Durability（持久性）：一个事务一旦提交，对数据库中的数据的改变应该永久保存。

    事务性消息有两种模式：
    - PRODUCER_ACK：生产者通过ack应答确认写入成功，这种情况下，如果有消息发送失败，生产者可以重试。
    - CONSUMER_ACK：消费者通过确认消费成功，Kafka会保证消费之前的消息全部都被消费成功。

    ### 消息丢弃
    Kafka提供消息丢弃策略，允许消费者指定消息丢弃的条件。

    用户可以指定消息最大大小和最大延迟时间。当消息达到最大大小时，消息会被丢弃；当消息在多久内没有被消费时，消息会被丢弃。

    ### 分布式事务
    Distributed Transactions in Apache Kafka is supported using two mechanisms: transactional producers and idempotent consumers.

    1. Transactional Producers: A transactional producer is an extension of the standard synchronous producer that provides support for transactions with exactly once delivery guarantee. The client application can use this feature by specifying a unique transactionalId during initialization. Upon successful initiation, all messages published under that transactionalId will be written atomically to all replicas before either committing or aborting the transaction. To ensure at-least-once delivery semantics, it is necessary to enable retries on the client side if the producer returns failures due to timeouts or other transient errors.

    2. Idempotent Consumers: An idempotent consumer is an extension of the basic Kafka consumer that enables users to specify what should happen when a consumer encounters duplicate messages. When enabled, duplicates are detected based on the combination of topic name, partition number, and message key. If a duplicate is encountered, one of three actions can be taken: discard the message, fail the consumer (default behavior), or invoke a user-defined callback function that handles the message according to specific business logic requirements. For example, a bank transfer processing application might use an idempotent consumer to handle duplicate transfers but rely on the ordering guarantees provided by Kafka's partitions to avoid double-spending issues.