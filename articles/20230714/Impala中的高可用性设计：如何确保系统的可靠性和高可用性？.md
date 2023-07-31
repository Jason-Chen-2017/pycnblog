
作者：禅与计算机程序设计艺术                    
                
                
随着大数据、云计算、容器化、微服务等新兴技术的快速发展，越来越多的企业把数据库从单机上迁移到分布式集群中进行运行。由于分布式系统环境复杂、系统规模庞大，因此对其系统可用性要求也越来越高。而Impala作为开源分布式查询引擎，正在成为众多大数据平台的重要组成部分。为了保证Impala在真实业务场景中的高可用性，需要做好以下几方面的工作。本文将详细阐述Impala高可用性的设计思路及原理，并通过具体案例介绍其部署方法及配置参数优化。

# 2.基本概念术语说明
## 2.1 Impala
Impala是一个开源的分布式SQL查询引擎，其支持结构化和半结构化数据，支持高性能查询，能够最大限度地利用底层存储资源，具有高容错、高可用性的特点。其系统架构如下图所示：

![impala](https://www.cloudera.com/sites/default/files/inline-images/Architecture_diagram_of_Apache_Impala_for_BigData.png)

## 2.2 HDFS
HDFS（Hadoop Distributed File System）是一个用于存储文件系统的开源框架。HDFS被设计用来处理海量的数据集，同时支持高吞吐量访问，适合于各种不同大小的文件，如：日志、数据等。它由一个NameNode和一个或多个DataNode组成，其中NameNode管理文件系统名称空间，记录每个文件的位置信息；而DataNode保存实际文件内容，提供文件I/O服务。HDFS是一个中心服务器，负责存储、调度数据块，然后向各个客户端提供访问接口。

## 2.3 Hive
Hive是一个基于Hadoop的一款数据仓库工具，可以将结构化的数据文件映射为一张表格，并提供 SQL 查询功能。用户可以通过Hive提供的命令行界面或者图形化界面查询数据，并且Hive 支持 HQL(Hive QL)。

## 2.4 Hive Metastore
Hive Metastore 是一个独立的元数据存储库，用于存储Hive表的信息。Metastore负责在 Hive 服务器启动时读取 HDFS 中 Hive 数据仓库相关的元数据信息，并维护 Hive 对象之间的关系。当执行建表语句或者其他的 DDL 操作时，Metastore 会相应更新元数据信息。

## 2.5 HBase
HBase 是一种可扩展的非关系型分布式数据库，基于 Hadoop 之上。它提供高可靠性、高性能、可伸缩性以及ACID事务特性。HBase 可以在 Hadoop 的框架下运行，并提供高效率的数据访问接口。

## 2.6 Zookeeper
Zookeeper 是一个开源的分布式协调服务，提供了统一的视图，使得分布式应用程序可以更容易地进行同步和通知。Zookeeper 可用于管理分布式环境中数据的一致性，它是一个树型结构，每一个节点都标识一个特定的服务器或服务。Zookeeper 负责维护节点之间的通信，包括数据的同步。

## 2.7 Coordinator Node
Coordinator Node 即协调器节点，用于协调查询请求。Coordinator Node 通过封装查询计划、执行查询任务的细节，使得不同的查询引擎可以使用同样的方式查询数据。Coordinator Node 通常会选择一个最近的节点作为自己的主节点，并向主节点发送查询请求。

## 2.8 State Store
State Store 是一个独立的服务，用于存储一些全局状态信息，比如：当前的分区信息、加载的元数据等。它通过一个单独的进程进行管理，独立于查询引擎，可以提升查询速度和避免冲突。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 3.1 Coordinator Node
### 3.1.1 分布式查询优化
Impala Coordinator Node接收到的查询请求，首先会经过分析预处理阶段，对查询语句进行解析和验证。经过解析，Coordinator Node可以判断出该查询语句涉及到的表、列信息。然后，它会尝试将查询的范围限制在本地存储的节点上，减少网络传输开销。此外，它还会根据查询语句的不同特征，选择合适的查询方式，如索引扫描、分区扫描、排序合并等。

### 3.1.2 执行顺序调整
Coordinator Node会根据查询执行效率、内存、磁盘利用率等因素，调整查询执行的顺序，以达到最优查询性能。主要包括两方面：

1. 动态查询规划：对于计算密集型的查询，Coordinator Node 会自动调整查询执行的优先级，使计算密集型的查询优先被执行。
2. 静态查询规划：对于内存压力大的查询，Coordinator Node 会自动增加内存分配以满足查询的内存需求。

### 3.1.3 查询结果缓存
Impala Coordinator Node除了缓存查询计划外，还可以缓存查询的执行结果。如果某个查询在近期有频繁的查询请求，那么就可以利用缓存加速查询执行。缓存中的结果可以直接返回给客户端，避免了重新计算。

### 3.1.4 错误重试机制
Impala Coordinator Node遇到错误后，它会自动尝试重试。它会先等待一段时间再重试，直至成功为止。在失败次数超过一定阈值之后，它会把失败的节点从集群中移除，防止出现风暴效应。

## 3.2 Executors
### 3.2.1 负载均衡
Impala Cluster中包含许多Execurot节点。当集群中有新节点加入或旧节点失去联系时，Impala 便会通过负载均衡算法，将查询请求转发给不同的节点。负载均衡算法的目标是避免所有节点负载不平衡。

### 3.2.2 查询处理流程
Impala Executor处理查询请求的流程比较简单，如下图所示：

![executor](https://www.cloudera.com/content/dam/blogs/wp-content/uploads/2017/07/query-processing-flowchart.jpg)

1. Coordinator Node收到查询请求，首先会解析查询语句得到要查询的表、列信息。
2. 根据表所在节点的信息，找到相应的数据节点。
3. 依次检查数据节点的负载情况，按照负载的多少进行排序。
4. 将查询请求发送到排名靠前的数据节点上。
5. 在数据节点上执行查询，获得结果。
6. Executor把结果汇总后，返回给Coordinator Node。
7. Coordinator Node将结果返回给客户端。

### 3.2.3 查询缓存
Executor缓存了部分查询结果。当数据发生变化时，缓存也会失效。另外，当某个查询的结果已经在缓存中时，可以直接返回缓存结果。这样可以加快查询响应时间。

### 3.2.4 执行结果压缩
Impala Executor在返回查询结果之前，会对结果进行压缩。虽然压缩过程消耗CPU资源，但压缩率通常比其他形式的序列化和反序列化方式更高。压缩后的结果在网络上传输时也会节省带宽。

### 3.2.5 表达式编译
Impala Executor在执行查询之前，会对查询语句中的表达式进行编译。编译的目的是减少重复计算，提高查询性能。

### 3.2.6 内存管理
Impala Executor会根据系统资源、节点负载、查询内存使用情况，动态调整查询的内存使用。

### 3.2.7 文件管理
Impala Executor会在磁盘上维护查询执行过程中使用的临时文件。这些文件不会很大，但它们占用的磁盘空间会逐渐增长。如果某个节点上的磁盘空间不足，则会触发清理过程，删除无用文件。

### 3.2.8 查询计划生成
Impala Executor在接收到查询请求后，会先生成查询计划。查询计划是指查询执行过程中需要执行的操作序列。

## 3.3 NameNode
### 3.3.1 NameNode的作用
NameNode的主要作用就是作为文件系统的命名者和资源管理者，它负责跟踪文件系统中文件的属性信息，比如存放在哪台机器的哪个目录。并监控集群中机器的健康状况，以保证整个文件系统的正常运行。

### 3.3.2 NameNode角色
NameNode角色在集群中扮演着重要的作用，它有三个主要职责：

1. 名字空间管理：管理文件系统的名称空间。
2. 锁管理：维护整个文件系统的共享资源。
3. 服务器协调：负责数据的复制、自动故障切换等操作。

### 3.3.3 NameNode状态存储
NameNode保存着整个文件系统的状态信息。包括文件和目录的元数据、访问权限、数据块信息、数据流入流出的数量、以及整个集群的活跃节点信息等。

### 3.3.4 Secondary NameNode
Secondary NameNode 是 NameNode 的热备份。它定期从主 NameNode 复制元数据信息和文件数据，以保证主 NameNode 的数据安全和高可用性。

### 3.3.5 NameNode数据块恢复
如果 NameNode 因为硬件损坏、软件错误或其它原因，导致数据的丢失，则可以通过 Secondary NameNode 获取丢失的数据。

## 3.4 DataNodes
### 3.4.1 DataNode的作用
DataNode 是 Hadoop 文件系统的主要工作节点。它主要负责存储文件系统的数据块，并向上游的 NameNode 报告数据块的块状况。

### 3.4.2 DataNode角色
DataNode角色主要负责：

1. 数据块存储：DataNode 在本地存储数据块，并定期将数据块报告给 NameNode。
2. 数据块检验：DataNode 检测数据块是否损坏，并将损坏的数据块报告给 NameNode。
3. 客户端服务：DataNode 提供文件系统的读写服务。

### 3.4.3 数据块校验
如果 DataNode 接收到某个数据块的报告，发现其中存在错误，则会自动修复数据块。如果某个数据块已经损坏太久，则会被标记为垃圾数据块，当 DataNode 要检索这个数据块的时候，就会跳过这个数据块。

### 3.4.4 数据块复制
DataNode 可以设置副本数，将数据块复制到其他的 DataNode 上。如果某个 DataNode 失效或宕机，则可通过副本数据块获取数据。

## 3.5 HDFS高可用性
### 3.5.1 配置
HDFS的高可用性配置十分复杂，目前主要有两个参数影响HA能力：

1. dfs.replication：副本数量，默认是3。
2. dfs.nameservices：名称服务，一个集群可以有多个名称服务。

### 3.5.2 架构
NameNode和DataNode的部署架构可以采用Active/Standby模式。架构图如下：

![hdfsha](https://www.cloudera.com/content/dam/blogs/wp-content/uploads/2017/07/hdfs-architecture-with-namenode-and-datanodes-in-active-standy-mode.png)

### 3.5.3 脑裂问题
由于各个NameNode的状态信息都是一样的，所以如果其中一个NameNode失效了，则会造成数据丢失。为了避免脑裂问题，需要配置HA模式，即两个NameNode必须同时工作。Active/Standby模式中，只有Active NameNode 可以参与数据的读写操作，Standby NameNode 只用来做容灾，不参与任何数据操作。当Active NameNode失效时，另一个NameNode会自动接管操作。

### 3.5.4 备份机制
为了保证数据的完整性，HDFS提供了数据备份功能，可以配置多个备份目录，数据的写入和读取都是在备份目录上进行。如果Primary NameNode失效，则会切换到Secondary NameNode继续提供服务。

## 3.6 YARN
Yarn 是 Hadoop 下的一个子项目，是一个通用集群资源管理系统。它提供了容错能力和系统弹性。Yarn提供了两种不同粒度的资源管理：ApplicationMaster 和 Container。

### 3.6.1 ApplicationMaster
ApplicationMaster 是一个特殊的节点，负责管理应用的生命周期。它的职责如下：

1. 请求资源：ApplicationMaster 会向 ResourceManager 申请资源，包括 CPU 和内存等。
2. 任务调度：ApplicationMaster 会根据集群的资源状态，调度任务在集群中的执行。
3. 任务监控：ApplicationMaster 监控任务的执行状态，包括进度、状态等。
4. 失效转移：如果 ApplicationMaster 失效，则会重启新的 ApplicationMaster。
5. 作业提交和完成：当 ApplicationMaster 完成任务时，它会汇报任务的完成情况，并释放资源。

### 3.6.2 Container
Container 是 Yarn 的基本调度单位，它封装了一个任务需要的所有组件，包括了任务的运行环境、任务的代码、数据、依赖库等。Yarn 会将资源按需分配给 Container，实现资源隔离和容错。

### 3.6.3 队列模型
Yarn 提供了队列模型，管理员可以创建不同队列，并指定不同队列的访问控制策略。队列模型提供多租户和弹性资源共享的能力。

### 3.6.4 容错机制
Yarn 提供了容错机制，如果某台机器宕机，Yarn 仍然可以将任务分配到其余机器上运行。同时，Yarn 还可以检测和识别应用的失效，并重新启动新的 ApplicationMaster 来接管任务。

## 3.7 Resource Manager HA
Resource Manager 是 Yarn 的中心节点，它管理整个集群的资源。ResourceManager 的角色包括：

1. 集群资源管理：ResourceManager 从所有 NodeManager 上获取集群的资源使用信息，分配给各个 ApplicationMaster。
2. 任务调度管理：ResourceManager 接受 Client 发来的资源申请，并调度任务在集群上运行。
3. 恢复管理：ResourceManager 监控 NodeManager 的心跳，感知 NodeManager 的崩溃和恢复。
4. 容错管理：ResourceManager 如果失效，则会选举出新的 Active ResourceManager。

### 3.7.1 配置
Yarn ResourceManager 的高可用性配置非常复杂，包含以下参数：

1. yarn.resourcemanager.ha.enabled：是否开启 ResourceManager HA，默认为 false。
2. yarn.resourcemanager.recovery.enabled：是否开启 RM 的故障恢复，默认为 true。
3. yarn.resourcemanager.zk-address：连接 ZooKeeper 集群的地址，用于RM的故障恢复。
4. yarn.resourcemanager.store.class：RM 的元数据的持久化方式，默认为 fs。
5. yarn.resourcemanager.hostname：RM 的主机名，如果没有配置则自动探测主机名。

### 3.7.2 故障切换
Yarn ResourceManager 的 HA 模式下，只有 Active RM 可以提供资源，而 Standby RM 一旦 Active RM 失效就自动切换为 Active RM 。当 Active RM 失效时，Standby RM 立马成为新的 Active RM ，开始提供资源服务。

### 3.7.3 高可用性
Yarn ResourceManager 的 HA 模式下，多个 RM 会运行在不同机器上，避免单点故障。并且，Yarn 会通过 ZooKeeper 实现 RM 间的状态同步，实现高可用性。

## 3.8 MapReduce
MapReduce 是 Hadoop 里的一个计算框架，可以用于大数据量的并行运算。它提供了 Map 和 Reduce 两个运算，并通过切分数据集来处理，来提高并行运算的效率。

### 3.8.1 任务类型
MapReduce 有四种不同的任务类型：

1. Job 任务：由 MapTask 和 ReduceTask 组成，一般由用户提交的。
2. TaskTracker 任务：负责执行 MapTask 和 ReduceTask。
3. Master 任务：管理整个集群，包括 Job 任务和 TaskTracker 任务。
4. CommonContainer 任务：启动在各个节点上执行的容器。

### 3.8.2 Job
Job 定义了 MapReduce 任务执行的细节。它包含输入输出路径、作业名称、作业Jar包、map函数、reduce函数、中间结果保存的路径、作业配置等。

### 3.8.3 MapTask
MapTask 是 MapReduce 任务的组成部分，它负责将输入的键值对转换成一系列中间键值对。其基本逻辑为：

1. 拆分输入文件，将文件读入内存或磁盘中。
2. 对内存或磁盘中的数据进行处理，得到中间键值对。
3. 对产生的中间键值对进行排序和分区。
4. 写入磁盘。

### 3.8.4 ReduceTask
ReduceTask 是 MapReduce 任务的组成部分，它负责从 map 任务产生的中间结果中聚合数据。其基本逻辑为：

1. 读取 map 任务产生的中间结果。
2. 进行合并、过滤和统计等操作。
3. 输出结果。

### 3.8.5 InputSplit
InputSplit 表示 MapTask 需要处理的数据集。它是 MapTask 在运行时，根据输入文件的大小、分布等，将文件切分为一系列小数据集，每个数据集作为一个 MapTask 的输入。

### 3.8.6 Partitioner
Partitioner 决定 Mapper 输出的 key 值对落入哪个 partition。它是 Mapper 的一个选项，默认使用 HashPartitioner，即用 hash 函数计算 key 的 hash code，然后取 hash code 的 modulus 作为 partition index。

### 3.8.7 Reducer个数
Reducer 的个数取决于数据的分布和内存大小。Reducer 个数越多，内存越大，但运行效率会降低。当 Reducer 个数过少，会导致数据倾斜，且产生的结果可能不精准。

### 3.8.8 本地模式
MapReduce 默认使用远程模式，但也可以配置为本地模式，也就是说在 MapReduce 集群中只配置一个节点即可运行。这种模式适用于调试或测试。

## 3.9 Tez
Tez 是 Apache 的开源的运行时引擎，它是 Hadoop MapReduce 的替代方案。它可以在 Hadoop 生态系统之上构建更复杂的应用程序，比如联邦学习、交互式查询等。

### 3.9.1 定位
Tez 定位为 Hadoop on Demand的一种新方式，即开发人员不需要编写复杂的 MapReduce 应用，只需要声明整个流程的逻辑，即可让 Tez 根据数据的存储、切片、压缩等情况，将应用的执行计划进行优化。

### 3.9.2 特性
Tez 有以下几个特性：

1. DAG（有向无环图）：Tez 使用有向无环图（DAG），将应用程序的计算流程表示出来。
2. 超融合：Tez 可以自动进行优化，将数据处理过程与应用的执行过程整合在一起。
3. 动态资源分配：Tez 可根据集群资源的利用率，动态调整任务的执行计划。
4. 数据局部性：Tez 可以充分利用数据局部性，仅处理必要的数据。

### 3.9.3 运行模式
Tez 有三种不同的运行模式：

1. 批处理模式：Tez 在批处理模式下，将作业的输入和输出写在本地磁盘中，并在本地磁盘上运行。
2. 联邦学习模式：Tez 在联邦学习模式下，支持多方数据协作的联邦学习应用，比如隐私数据分享。
3. 交互式查询模式：Tez 在交互式查询模式下，支持复杂的交互式查询，比如搜索推荐等。

## 3.10 Presto
Presto 是 Facebook 的开源分布式 SQL 查询引擎，具有高性能、高并发、易用性等特点。Presto 使用标准 SQL 语法来查询数据，但它可以在异构的分布式数据源之间进行查询。Presto 内置很多高级功能，如分区和表缓存，这使得它成为 Apache Spark 或 Apache Hive 的竞品。

### 3.10.1 架构
Presto 包括两个关键组件：coordinator 和 worker。前者负责查询计划的生成，以及查询的调度。后者负责数据拉取和查询执行，以及结果的聚合。Presto 的架构如下图所示：

![presto](https://cdn.mysql.com/kb/media/blog/images/tip-how-to-set-up-highly-available-presto-cluster-part-one/figure2.png)

### 3.10.2 服务发现
Presto 使用 zookeeper 服务发现机制来发现 worker 节点。如果 worker 节点挂掉，coordinator 会感知到，并停止对该节点的查询。

### 3.10.3 查询优化
Presto 使用自己的查询优化器来进行查询计划的生成。优化器会选择一个合适的查询计划，基于以下几方面：

1. 数据局部性：优化器可以考虑数据局部性，比如同一个物理节点的数据，或者同一个分区的数据。
2. 数据倾斜：优化器可以自动检测和解决数据倾斜的问题。
3. 并行性：优化器可以选择并行运行的 Map 任务个数。
4. 内存使用：优化器可以考虑节点的内存使用情况。

### 3.10.4 分区缓存
Presto 允许用户将数据分成多个分区，然后在每个节点上缓存分区的数据。这样的话，相同的查询不需要访问多个节点，可以大大提升查询的性能。

### 3.10.5 并行执行
Presto 可以利用所有 worker 节点进行并行查询执行。每个 worker 节点会并行执行一部分查询，而不是串行执行所有的查询。

### 3.10.6 安全性
Presto 自带安全性模块，可以支持用户认证、授权等功能。

### 3.10.7 插件机制
Presto 允许用户开发插件，可以自定义数据的输入、输出、聚合、查询计划生成等逻辑。

# 4.具体代码实例和解释说明
## 4.1 Impalad配置
```
[impala]
    # To use an Impala instance that is not started by the system startup scripts,
    # set this value to its host and port separated by a colon (e.g., 'localhost:21000').
    # Otherwise, leave it commented out or set it to null and let the coordinator node find and start one for you.
    impalad=null

    # The maximum number of times to retry failed queries before giving up. Set to zero to disable retries entirely.
    max_retry_on_rpc_errors=-1

    # Whether to attempt to restart the query if the client disappears without closing the connection. Only applies when using the HTTP protocol.
    request_pool_max_wait_millis=60000

    # The timeout in milliseconds for network operations such as fetching metadata from the catalog service or reading data over the network.
    rpc_timeout_ms=120000

    # The size of the result set cache per coordinator. This can be used to reduce memory usage in exchange for increased latency due to cache misses.
    resultset_cache_size=0

    # The minimum number of nodes required for executing queries. If less than this many nodes are available, then queries will wait until additional nodes become available.
    min_cluster_size=1

    # The maximum number of nodes allowed for executing queries. Once this limit is reached, no new queries may be scheduled.
    max_cluster_size=16

    # The percentage of hosts that must report success to consider a query successful. Default is 90% but can be adjusted downward based on cluster size.
    expected_cluster_size_percent=90

    # Controls whether Impala should run in debug mode. Enabling debug mode will log more detailed information about each query execution and may impact performance negatively.
    enable_debug_queries=false

    # Enables experimental features which have not been thoroughly tested or refined yet. Features include EXPLAIN PLAN, SHOW CREATE TABLE AS SELECT, and distributed joins.
    enable_experimental_features=false

    # The port used by the frontend webserver running with --webserver_port. Set to -1 to disable the webserver.
    webserver_port=25000

    # The IP address where the frontend webserver listens for requests. Defaults to localhost.
    webserver_addr=0.0.0.0

    # Maximum number of connections the frontend server will accept at any given time.
    webserver_max_connections=1000

    # Directory where all temporary files created by Impala will be stored. Must be accessible to both coordinator and workers. If unset, defaults to /tmp/$USER/impala-$PID/.
    temp_file_path=null

    # Comma-separated list of paths containing runtime libraries needed by Impala. These directories must contain only.so files. Libraries provided by distributions like Anaconda can be included here.
    extra_library_paths=/usr/lib/anaconda/lib:/usr/local/lib/impala

    # Configure the behavior of INSERT statements. Valid values are "insert_immediate" or "insert_as_select". "insert_immediate" means that rows are immediately visible after insert; "insert_as_select" means that data is first inserted into a staging table, and then a final operation selects from that table and inserts the results into the main table (this allows the source table to continue receiving updates while the insert is being processed). The default is "insert_immediate", but older versions of Impala may still default to "insert_as_select".
    insert_behavior=insert_immediate
    
    # Enable dynamic resource allocation for multi-tenant environments. When enabled, different users' queries may share resources based on their submitted workload. 
    enable_dynamic_resource_allocation=true
```
## 4.2 Hive Server2配置
```
[hive]
  metastore = hive

  # Host name and port of the Thrift metastore service.
  metastore_port = 9083

  # Host name of the HiveServer2 Interactive service. Leave blank to run in local mode.
  hiveserver2_interactive_host = 
  hiveserver2_interactive_port = 10500

  # The JAAS configuration file to authenticate against LDAP.
  ldap_jaas_config_file = 

  # Additional Java options to pass to HiveServer2 JVM during startup.
  java_options = 

  # A comma-separated list of user names who are authorized to access HS2 via HiveServer2 Interactive. Leave empty to allow anonymous access.
  hs2_allow_users = 

  # How long a session lasts in seconds. Set to negative numbers to indicate sessions that do not expire. By default, sessions never expire.
  idle_session_timeout = 3600

  # Whether to reuse threads across different clients or create a new thread for each incoming connection. Setting this to true improves concurrency performance but requires properly configuring the underlying OS.
  reuse_result_set_threads = false

  # Use SSL encryption for communication between HiveServer2 and clients. This uses the openssl library and stores keys inside $HADOOP_HOME/ssl-client directory.
  ssl_client_keystore_path = 
  ssl_client_keystore_password = 

  # Allow SSL connections only from clients with certificates signed by trusted Certificate Authorities.
  ssl_truststore_path = 
  ssl_truststore_password = 

  # Log level of HMS events logged through EventLogger interface. Can be DEBUG, INFO, WARN, ERROR, FATAL.
  event_logger_log_level = INFO

  # Type of password hashing algorithm to use. Options are MD5, SHA-1, SHA-256, SHA-384, SHA-512. If set to none, passwords will not be hashed.
  password_hashing_algorithm = 

  # The file path where trace logs are stored. If set to empty string, tracing will be disabled.
  audit_event_log_dir = 
```
## 4.3 hdfs-site.xml配置
```
<configuration>
  <property>
    <name>dfs.nameservices</name>
    <value>hdfs</value>
  </property>
  <property>
    <name>dfs.ha.automatic-failover.enabled</name>
    <value>true</value>
  </property>
  <property>
    <name>dfs.client.failover.proxy.provider.hdfs-ha</name>
    <value>org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider</value>
  </property>
  <property>
    <name>dfs.ha.namenodes.hdfs</name>
    <value>nn1,nn2</value>
  </property>
  <!-- Namenode specific properties -->
  <property>
    <name>dfs.namenode.http-address.hdfs.nn1</name>
    <value>hdfs://nn1.example.com:50070</value>
  </property>
  <property>
    <name>dfs.namenode.http-address.hdfs.nn2</name>
    <value>hdfs://nn2.example.com:50070</value>
  </property>
  <property>
    <name>dfs.namenode.https-address.hdfs.nn1</name>
    <value>hdfs://nn1.example.com:50470</value>
  </property>
  <property>
    <name>dfs.namenode.https-address.hdfs.nn2</name>
    <value>hdfs://nn2.example.com:50470</value>
  </property>
  <property>
    <name>dfs.namenode.shared.edits.dir</name>
    <value>qjournal://hdfs-ha-nn1.example.com:8485;hdfs-ha-nn2.example.com:8485/hdfs</value>
  </property>
  <property>
    <name>dfs.ha.fencing.methods</name>
    <value>sshfence</value>
  </property>
  <property>
    <name>dfs.ha.fencing.ssh.private-key-files</name>
    <value>/home/user/.ssh/id_rsa,/home/user/.ssh/another_key</value>
  </property>
  <!-- JournalNode specific properties -->
  <property>
    <name>dfs.journalnode.edits.dir</name>
    <value>file:///var/run/hadoop-hdfs/dfs/jn</value>
  </property>
  <!-- Configuration for automatic failover triggered by health check script. -->
  <property>
    <name>dfs.ha.healthmonitor.script.path</name>
    <value>/etc/hadoop/hdfs-healthcheck.sh</value>
  </property>
  <property>
    <name>dfs.ha.healthmonitor.interval-ms</name>
    <value>30000</value>
  </property>
  <property>
    <name>dfs.ha.zkfc.port</name>
    <value>8018</value>
  </property>
  <!-- Remote SSH commands used for fencing. -->
  <property>
    <name>dfs.ha.fencer.ssh.command.before</name>
    <value></value>
  </property>
  <property>
    <name>dfs.ha.fencer.ssh.command.after</name>
    <value></value>
  </property>
  <property>
    <name>dfs.ha.fencer.ssh.username</name>
    <value></value>
  </property>
  <property>
    <name>dfs.ha.fencer.ssh.connect-timeout.millis</name>
    <value>5000</value>
  </property>
  <property>
    <name>dfs.ha.fencer.ssh.read-timeout.millis</name>
    <value>20000</value>
  </property>
  <property>
    <name>dfs.ha.fencer.ssh.connection-retries</name>
    <value>3</value>
  </property>
  <property>
    <name>dfs.block.replicator.classname</name>
    <value>org.apache.hadoop.hdfs.server.namenode.BlockPlacementPolicyWithUpgradeDomain</value>
  </property>
  <property>
    <name>dfs.storage.policy.impl</name>
    <value>org.apache.hadoop.hdfs.server.namenode.DefaultStoragePolicy</value>
  </property>
  <property>
    <name>dfs.upgradeDomain.storage.policy.enable</name>
    <value>true</value>
  </property>
</configuration>
```

