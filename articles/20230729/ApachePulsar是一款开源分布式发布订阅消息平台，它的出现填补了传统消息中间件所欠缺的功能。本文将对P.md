
作者：禅与计算机程序设计艺术                    

# 1.简介
         
Apache Pulsar（下称Pulsar）是一个开源分布式发布-订阅消息平台，由 apache 下开源项目apache/incubator-pulsar 发起并贡献至社区，它最初起源于 Yahoo 的一个项目，曾经被用在多家公司的产品中，包括 Netflix，Yelp 和 Twitter。现在，Pulsar 已经成为 Apache 顶级项目之一。 Pulsar 最初是为了满足实时流数据处理和存储需求而开发的，它的灵活架构、高吞吐量、低延迟以及支持多种数据格式和协议使其在实时数据领域得到广泛应用。它最初目的是作为 Kafka 在消息存储和消费方面的替代品，但是随着时间的推移，Pulsar 逐渐演变成了一个独立的开源项目，并且开源社区不断加强它的开发。

自身定位于云原生分布式消息系统，能够支持多种工作负载类型，包括日志、数据分析、即时查询、事件通知、IoT 数据收集和实时计算。目前，Pulsar 有超过百万行代码，累计提交次数超过三千次，覆盖了多个子项目。截止到2021年1月，Pulsar Github 项目已经拥有 7 个主要 contributor 和近 1000+ commits。截止到2020年12月，Pulsar 在 Apache 基金会孵化器拥有 9个 Committer、7个 PPMC 成员、6个 Contributor 和 300多个 committers 及 contributors。

本文将详细阐述Pulsar的架构、功能、性能、安全性等方面。
# 2.基本概念术语说明
## 2.1 消息模型
首先，我们需要知道Pulsar的消息模型，Pulsar支持三种消息模型：
### 点对点(P2P)模型
这种模型类似于TCP协议中的Socket模型，消息只能单向传输到指定Topic的Partition中，当Consumer订阅Topic时，可以选择指定从哪个偏移位置开始接收消息。这种模式非常适合于写入少量数据的场景。

![point_to_point](https://img-blog.csdnimg.cn/20210617202414231.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoYW5sb24=,size_16,color_FFFFFF,t_70)

### PubSub模式
这种模型基于发布者和订阅者模式，生产者发送消息到指定的Topic上，所有订阅该Topic的消费者都可以接收到这些消息。这种模式对于写入大量数据的场景非常友好，但消费者的数量和速度取决于集群的容量和订阅的主题数量。因此，这种模式有可能造成消息堆积或者消费者不可用。PubSub 模型支持多个订阅者消费同一条消息。

![pub_sub](https://img-blog.csdnimg.cn/20210617202439358.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoYW5sb24=,size_16,color_FFFFFF,t_70)

### 分布式队列(Distributed Queues)
这是Pulsar新加入的一种消息模型，它可以在不同的Cluster之间进行消息传递。这种模型对比点对点和发布订阅模型更加复杂，但仍然可以提供很好的性能。分散式队列提供了额外的可靠性保障，可以使用多个独立的集群组成一个全局的消息网络，同时可以避免单点故障，增加鲁棒性。

![distributed_queue](https://img-blog.csdnimg.cn/2021061720250177.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoYW5sb24=,size_16,color_FFFFFF,t_70)


## 2.2 BookKeeper
Pulsar 底层依赖于 Apache BookKeeper 作为存储引擎。BookKeeper 提供了一系列用于实现分布式协调服务的组件，例如选举、配置管理、元数据存储等，这些组件可以被用于构建 Pulsar 上的很多分布式系统。我们先来看一下它是如何工作的。

**角色划分**：BookKeeper 被分为以下几个角色：
- **Ledgers**：这是实际存储消息的地方。它把数据按照固定的大小切割成小块，称为“Entries”，每条消息都对应有一个 Entry，这些 Entries 构成了 Ledger。Ledger 可以分布在不同的物理磁盘上，以提高性能。
- **Journal**：用于记录 Ledger 相关的所有修改信息，包括添加/删除 Entries，分配 Ledger，分裂等。每个 Journal 会被复制到多个节点上，以提高数据可用性。
- **Bookies**：Bookies 扮演着 Bookkeeper 服务的参与者角色，它们负责存储数据并执行读写请求。每个 Bookie 会负责多个 Ledgers 的数据，以提高整体性能。
- **Metadata Service**：用于维护集群相关的元数据，如列表、路由表等。元数据服务一般采用 ZooKeeper 来实现，但也可以用其他的方式实现。

**数据模型**：BookKeeper 按照如下方式组织数据：
- 每个消息都会被分配一个全局唯一的序号，这个序号就是 EntryID。
- Entry 只会被追加到最后一个 Ledger 中，直到达到最大限制，才会创建新的 Ledger。
- 当某个Ledger满了之后，BookKeeper会创建一个新的 Ledger，然后将老的Ledger数据迁移到新的 Ledger中。
- 如果某个节点宕机，则数据不会丢失，BookKeeper 会自动通过复制机制保证数据安全。

**存储结构**：BookKeeper 使用Hierarchical Namespace 的方式来组织数据，即树形结构。每一个命名空间可以包含若干个子命名空间或条目（条目的例子是消息）。BookKeeper 通过路径名来标识命名空间，每个条目都有一个编号（EntryID），这些编号用树状结构表示。

**写入流程**：当生产者要发送消息给 BookKeeper 时，它会首先在本地缓存数据，待一定量的数据被写入后，再异步地将这些数据批量地写入多个 Bookies 上。如果某个 Bookie 故障，BookKeeper 会自动感知，并将相应的消息重新复制到另一个 Bookie 上。当 Broker 需要读取消息时，它会连接多个 Bookies 并从不同的位置读取消息，以防止某些节点故障影响数据读取。

**读取流程**：Broker 发送读取请求给任意的一个 Bookie，然后它就会从该 Bookie 返回一个包含一些消息的消息集（MessageSet）。然后 Broker 就把这些消息合并成一个结果集（Resultset），最终返回给消费者。 Broker 可以一次从多个 Bookies 读取消息，以提高性能。

以上就是 BookKeeper 的基本工作原理。

## 2.3 运行环境搭建
安装运行 Pulsar 之前，需要搭建好运行环境，包括 Java 环境、Zookeeper 服务器、BookKeeper 服务器。其中 Java 环境需要选择 8 或以上版本。

### 安装 Java

推荐安装 Oracle JDK，可以从官方网站下载安装包：https://www.oracle.com/java/technologies/javase-downloads.html。

```bash
wget https://download.oracle.com/otn/java/jdk/11.0.13+8/f51449fcd52f4d52b93a989c5c56ed3c/jdk-11.0.13_linux-x64_bin.tar.gz -O jdk-11.0.13_linux-x64_bin.tar.gz
sudo mkdir /usr/lib/jvm && sudo tar xzf jdk-11.0.13_linux-x64_bin.tar.gz --directory /usr/lib/jvm
sudo update-alternatives --install "/usr/bin/java" "java" "/usr/lib/jvm/jdk-11.0.13/bin/java" 1
sudo update-alternatives --install "/usr/bin/javac" "javac" "/usr/lib/jvm/jdk-11.0.13/bin/javac" 1
sudo update-alternatives --install "/usr/bin/jar" "jar" "/usr/lib/jvm/jdk-11.0.13/bin/jar" 1
sudo update-alternatives --config java   # 配置默认 java 版本
sudo update-alternatives --config javac   # 配置默认 javac 版本
java -version    # 测试是否安装成功
```

### 安装 Zookeeper

Zookeeper 是一个分布式协调服务，可以用来维护分布式环境下的各种状态信息。Pulsar 使用 Zookeeper 来存储其元数据，包括 Broker 信息、Topic 配置等。

```bash
wget https://archive.apache.org/dist/zookeeper/zookeeper-3.7.0/apache-zookeeper-3.7.0-bin.tar.gz
tar zxvf apache-zookeeper-3.7.0-bin.tar.gz
cd zookeeper-3.7.0/conf
cp zoo_sample.cfg zoo.cfg    # 修改配置文件，设置数据存储目录 dataDir
cd..
./bin/zkServer.sh start     # 启动 Zookeeper
./bin/zkCli.sh      # 查看状态信息
```

### 安装 BookKeeper

BookKeeper 是一个高性能的存储服务，可以用来持久化发布/订阅消息。Pulsar 依赖 BookKeeper 来存储消息数据，以及作为 Pulsar 集群中的一个独立角色参与到数据流的传输中。

```bash
mkdir bookkeeper
cd bookkeeper
wget http://apache.mirrors.lucidnetworks.net/bookkeeper/stable/bookkeeper-server-4.14.1/bookkeeper-server-distribution-4.14.1-bin.tar.gz
tar xf bookkeeper-server-distribution-4.14.1-bin.tar.gz
rm *.tar.gz
mv apache-bookkeeper*/*.
rmdir apache-bookkeeper*
vi conf/bk_server.conf     # 修改配置文件
bin/bookkeeper shell metaformat       # 清空元数据，重新生成必要的文件夹
bin/bookkeeper localbookie 3181          # 启动一个本地 Bookie 实例
bin/bookkeeper shell bookies            # 查看 Bookie 状态
```

注意：每次重启 BookKeeper 集群，都需要执行 `metaformat` 命令，否则可能会导致数据损坏。另外，建议设置 Bookie 数量为奇数个，以保证数据均匀分布。

完成以上环境准备，即可开始部署 Pulsar 服务。

