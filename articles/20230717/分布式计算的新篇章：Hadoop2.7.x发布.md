
作者：禅与计算机程序设计艺术                    
                
                
Hadoop(高德纳尔·伯克利曼）是一个基于Java开发的一个框架，主要提供对大规模数据集的存储、处理和分析。作为一个框架，Hadoop提供了很多功能，包括MapReduce计算框架、HDFS文件系统、容错机制、可扩展性等。Hadoop当前最新版本为2.7.x，其中2.7.x在性能和稳定性方面都有重大提升，此次更新内容将会带来许多改进，这就是本文要阐述的内容。本文并不会详细介绍Hadoop的安装配置过程，如果你没有相关的基础知识，建议先学习相关知识。
# 2.基本概念术语说明
## 2.1 Hadoop生态体系
### 2.1.1 Hadoop简介
Apache Hadoop(TM)是一个开源的框架，它能够对大量的数据进行分布式处理，适用于离线批处理或者实时分析等场景。Hadoop框架由HDFS(Hadoop Distributed File System)、MapReduce计算框架、YARN资源管理器、Zookeeper协调服务以及其他组件组成。这些组件之间通过RPC远程调用协议进行通信。

![hadoop-components](https://www.ibm.com/developerworks/cn/opensource/os-cn-bigdata-hadoop/figure3.png "hadoop-components")

1. HDFS: Hadoop Distributed File System (HDFS)，是一个文件系统，提供高容错性、高可用性的数据存储。HDFS有着独特的设计目标：一次写入多次读取。HDFS采用主从架构，一台服务器作为NameNode节点，存储文件的元数据；其余的服务器作为DataNode节点，存储文件数据。数据块可以保存多个副本，能够容忍节点失效或网络分区。HDFS的优点包括高容错性、高吞吐量以及易于扩展。
2. MapReduce: MapReduce是一个编程模型和运行环境，用于编写并行程序，解决海量数据的并行运算问题。MapReduce把复杂的并行计算任务分解为独立的“映射”和“归约”阶段。对于每一个输入数据，它都会被映射到一系列的键值对中。然后，这些键值对会被传送到相同的“归约”函数，这个函数接收来自所有映射任务的所有键值对，并对它们进行汇总。这个过程会重复执行，直到所有的映射任务和归约任务完成。MapReduce通过分而治之的方式，解决了海量数据的并行处理问题。
3. YARN: Yet Another Resource Negotiator (YARN)，是一个基于 hadoop 的集群资源管理器。它允许用户提交作业并指定执行的应用程序类型。YARN 提供了一套自己的 API 和工具用来管理集群中的资源。
4. ZooKeeper: Apache ZooKeeper 是 Apache Hadoop 的依赖组件，负责管理 HDFS 的命名空间和一些元数据信息。它具备高可用性和强一致性特性，并提供了诸如同步、群组管理和 Master 选举等分布式应用支持。
5. Hue: Cloudera's Distribution of Hadoop for Big Data Analysis software package that includes a web interface for managing the Hadoop cluster and running jobs. It also provides an SQL editor that can be used to run Hive queries against data in HDFS.
6. Sqoop: Apache Sqoop is a tool designed for efficiently transferring data between relational databases and Hadoop. The goal of Sqoop is to move data from a structured database into Hadoop or vice versa by creating an efficient data import process. 

以上为Hadoop生态中各个子项目的介绍。

### 2.1.2 Hadoop架构图
Hadoop的架构图如下所示：

![hadoop-architecture](https://www.ibm.com/developerworks/cn/opensource/os-cn-bigdata-hadoop/figure4.jpg "hadoop-architecture")

Hadoop包含三层结构：

1. 数据层：HDFS（Hadoop Distributed File System），数据存储层。
2. 计算层：MapReduce，数据处理层。
3. 控制层：Yarn、Zookeeper，集群管理层。

其中，数据层由HDFS和本地磁盘组成；计算层由MapReduce和Yarn组成。由于HDFS具有高容错性、高可用性的特性，所以它被设计成一个单一节点的故障转移（failover）集群，能够处理成百上千的节点，同时保证服务的高可用性。而计算层的MapReduce并不是Hadoop的核心，不过它为HDFS和Yarn的工作奠定了坚实的基础。控制层则包含了两个重要的组件——Yarn和Zookeeper。Yarn负责集群资源管理，即资源的分配和调度；Zookeeper则是一个基于Paxos算法的分布式协调服务。两者共同组成了Hadoop的集群控制管理架构。

## 2.2 Hadoop基本概念及术语
### 2.2.1 文件系统
HDFS（Hadoop Distributed File System）是Hadoop提供的分布式文件系统。它提供高容错性、高可用性的数据存储。HDFS有着独特的设计目标：一次写入多次读取。HDFS采用主从架构，一台服务器作为NameNode节点，存储文件的元数据；其余的服务器作为DataNode节点，存储文件数据。数据块可以保存多个副本，能够容忍节点失效或网络分区。HDFS的优点包括高容错性、高吞吐量以及易于扩展。

HDFS的文件系统接口定义了对文件的读、写和删除等操作。客户端向NameNode请求元数据信息，NameNode返回给客户端。然后客户端向相应的DataNode请求访问文件的操作。如果DataNode出现故障，NameNode会感知到并重新调度数据块，确保数据安全。HDFS还提供了Hadoop命令行接口（Hadoop Command Line Interface, HCLI）, 通过它可以直接操作HDFS上的文件。

### 2.2.2 分布式计算
MapReduce是一个分布式计算框架。它利用HDFS为数据存储和分发，用并行化的Map（映射）和Reduce（归约）操作符来对大数据集进行处理。MapReduce程序通常以编程语言编写，并交给MapReduce框架运行。Map和Reduce操作符被设计成分布式的、可以并行运行的。

#### 2.2.2.1 Map
Map操作是指把输入数据集合中的每个元素映射为一组键值对。其中，键是中间结果的输出位置，值则是处理后的数据。Map操作由用户自定义的mapper类实现。Map输出的键值对会被分发到对应的reduce任务进行处理。

#### 2.2.2.2 Reduce
Reduce操作是指对map阶段产生的中间结果进行合并。它跟Map操作相反，接受键值对形式的输入，并输出累计结果。Reduce操作由用户自定义的reducer类实现。当所有map任务的输出均收集完成之后，reduce任务开始处理。

#### 2.2.2.3 MapReduce过程
MapReduce的过程如下所示：

1. 首先，输入数据会被分割成若干个数据片段（split）。
2. 每个数据片段会被发送到不同的机器上运行map任务。
3. map任务会对输入的数据片段进行切割、排序、过滤等操作，生成中间键值对。
4. 不同map任务的输出会被聚合（merge）成一个大的输出集合，该集合会被发送到Reduce所在的那台机器上。
5. 在reduce端，reduce任务会合并不同map的输出结果，并按照一定规则生成最终结果。

![mr-process](https://www.ibm.com/developerworks/cn/opensource/os-cn-bigdata-hadoop/figure5.png "mr-process")

### 2.2.3 资源管理
Yarn是Hadoop的资源管理模块，它负责集群资源的管理。它通过调度器（scheduler）对资源进行统一管理，分配给各个应用容器。Yarn分为ResourceManager和NodeManager两大模块。

1. ResourceManager：ResourceManager是Hadoop的中心管理者。它是Hadoop集群的资源管理者，负责整个集群资源的划分、调度和分配。ResourceManager主要负责两个方面的工作：（1）资源的分配；（2）任务的调度和监控。它维护着集群中所有节点的资源使用情况、任务队列，并且会根据集群的策略向各个应用提交资源申请。ResourceManager的主要职责是分配内存、CPU、磁盘、网络等物理资源给任务。
2. NodeManager：NodeManager是一个独立的 daemon进程，它监听 ResourceManager 向自己报告的变化，并根据 ResourceManager 的指令启动和停止 ApplicationMaster 。它还负责运行具体的任务，例如 MapTask 和 ReduceTask 。

![yarn-archi](https://www.ibm.com/developerworks/cn/opensource/os-cn-bigdata-hadoop/figure6.png "yarn-archi")

## 2.3 Hadoop框架概览
Hadoop框架的部署模型分为客户端服务器模式和单机模式。客户端服务器模式一般在生产环境下使用，其中，Client运行在客户端设备（比如，笔记本电脑）上，Server运行在Hadoop集群的某个节点上。单机模式是在开发测试环境下使用，整个Hadoop框架运行在同一台计算机上，且仅有一个节点。

在客户端服务器模式下，Client通过与Server之间的通信获取HDFS的数据和计算资源。Client向Namenode请求元数据信息，Namenode返回给Client。然后Client向Datanodes请求数据块，Datanodes返回给Client。Client需要对HDFS的各种数据块进行切分、合并等操作，并把操作结果上传到HDFS。

在单机模式下，Hadoop框架运行在单台计算机上。整个框架的角色由多个daemon进程组成，其中包括Namenode、Secondary Namenode、Datanode、JobTracker、TaskTracker、JournalNode等。每个daemon进程都有单独的配置文件，配置文件中都记录了该进程的运行端口号，主机名和日志文件路径等信息。

Hadoop框架整体的架构如下图所示：

![hadoop-framework](https://www.ibm.com/developerworks/cn/opensource/os-cn-bigdata-hadoop/figure7.jpg "hadoop-framework")

### 2.3.1 Hadoop MapReduce概览
Hadoop MapReduce框架是Hadoop提供的分布式计算框架。它利用HDFS为数据存储和分发，用并行化的Map（映射）和Reduce（归约）操作符来对大数据集进行处理。MapReduce程序通常以编程语言编写，并交给MapReduce框架运行。Map和Reduce操作符被设计成分布式的、可以并行运行的。

MapReduce框架具有以下几个主要特点：

1. 可靠性：Hadoop MapReduce具有高度可靠性，能够容错，即便有节点故障也不会影响系统的正常运行。
2. 有效性：Hadoop MapReduce通过分而治之的思想，有效地解决海量数据的并行处理问题。
3. 容错性：Hadoop MapReduce采用多副本机制，保证数据的可靠性。
4. 可扩展性：Hadoop MapReduce能够充分利用集群的资源，利用广播算法可以加快处理速度。

Hadoop MapReduce的执行流程如下图所示：

![mr-flowchart](https://www.ibm.com/developerworks/cn/opensource/os-cn-bigdata-hadoop/figure8.png "mr-flowchart")

1. 用户提交作业。首先，用户需要准备好作业的数据文件。然后，通过命令行工具提交作业。
2. JobTracker识别并分配任务。JobTracker会读取并解析作业描述文件，将作业分派给Datanode上的TaskTracker。
3. TaskTracker执行任务。TaskTracker会启动一个Java虚拟机，并将任务输入流和输出流通过系统管道连接起来。
4. TaskTracker从输入文件中读取输入数据，并处理数据。处理完成后，TaskTracker将输出数据写入系统管道。
5. 当所有的Map任务和Reduce任务完成后，任务结束。

### 2.3.2 Hadoop Streaming概览
Streaming是一种Hadoop的工具，它可以用来实现分布式的实时计算功能。用户可以使用Streaming在Hadoop集群中执行长时间运行的脚本。

Streaming程序的输入通常来源于外部数据源，如日志文件、消息队列、数据库等。Streaming会将数据流以一定间隔周期性地传输到Hadoop集群中，并执行MapReduce作业来处理数据。

Streaming的优势在于简单易用，不需考虑底层技术细节。Streaming的输入输出处理可以自由选择，而且可以通过MapReduce来完成。

### 2.3.3 Hadoop Yarn概览
Yarn是Hadoop提供的资源管理系统。它可以管理整个集群的资源，包括计算资源、存储资源、网络资源。Yarn可以动态分配和回收集群资源，让更多的任务可以运行在集群上。

Yarn的作用如下：

1. 资源共享：Yarn可以共享集群中的资源，包括计算资源、存储资源和网络资源。这可以极大地减少集群资源的浪费，同时也可以提升集群的利用率。
2. 动态资源分配：Yarn可以根据实际的工作负载状况，动态调整集群的资源使用比例。
3. 容错性：Yarn提供容错机制，使得集群中的节点出现故障时，仍然可以继续运行。

Yarn的运行机制如下图所示：

![yarn-archi](https://www.ibm.com/developerworks/cn/opensource/os-cn-bigdata-hadoop/figure9.png "yarn-archi")

Yarn主要包含ResourceManager和NodeManager两大模块。

1. ResourceManager：ResourceManager是Yarn的中心管理者。它是Yarn集群的资源管理者，负责整个集群资源的划分、调度和分配。ResourceManager主要负责两个方面的工作：（1）资源的分配；（2）任务的调度和监控。ResourceManager的主要职责是分配内存、CPU、磁盘、网络等物理资源给任务。
2. NodeManager：NodeManager是一个独立的 daemon进程，它监听 ResourceManager 向自己报告的变化，并根据 ResourceManager 的指令启动和停止 ApplicationMaster 。它还负责运行具体的任务，例如 MapTask 和 ReduceTask 。

Yarn的调度器是个非常重要的模块，它的主要职责就是资源分配和任务调度。当一个任务提交到Yarn时，Yarn的调度器就会根据集群的资源状况、队列状态、作业的优先级等因素，自动地选择一个最佳的节点来运行该任务。

Yarn使用一个全局的资源管理器（Global Resource Manager）来统一管理整个集群的资源。该资源管理器能访问各个节点的资源使用情况，并根据集群的负载情况实时调整集群资源的使用比例。

