
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Impala 是 Cloudera 提供的一款开源的分布式查询引擎，在 Hadoop 的 MapReduce 上进行了优化。Impala 通过合并多个计算节点（节点之间通过网络通信）的方式提高查询性能，并提供了快速而一致的数据读取能力。
Impala 的主要特性如下：
- 支持复杂的SQL语句；
- 采用多种编码格式，包括 Text、RCFile 和 SequenceFile；
- 在内存中缓存表的数据；
- 使用分布式计划生成器，自动生成执行计划；
- 数据压缩功能；
- SQL 支持 Hive Metastore 中存储的元数据信息；
- 可以与 Hive 或 Spark 一起使用；
- 支持各种文件系统（HDFS，S3等）。
本文将对 Impala 的整体架构进行介绍，相关概念及术语也会进行详细阐述。
# 2.基本概念
## 2.1 Impala基本概念
Impala是一个开源的分布式查询引擎，基于Apache Hadoop。它可以运行于Hadoop YARN集群之上。其分布式执行引擎由Impalad进程组成，每一个Impalad进程负责处理由多个Coordinator节点协调管理的查询任务。
### 2.1.1 Coordinator节点
Coordinator是Impala中的角色，它负责接收查询请求、解析SQL语句、生成查询计划、分配查询任务到各个Impalad节点，并对结果集进行汇总。它也可以管理查询计划的变化，如DDL语句、DML语句等。Coordinator是Impala中的单点故障，因此建议部署HA方案。
### 2.1.2 Impalad节点
Impalad是Impala的工作进程，它负责执行查询任务，从HDFS、本地磁盘或内存中读取数据，对查询请求进行实际的计算，并将结果集返回给客户端。每个Impalad都有自己独立的内存资源，并且可以通过配置项控制最大内存分配量。每个Impalad节点需要连接至一个Coordinator节点，其负责处理来自不同用户或应用的查询任务。
### 2.1.3 HDFS
HDFS（Hadoop Distributed File System）是Hadoop文件系统，用于存储海量数据，属于Apache顶级项目。Impala可以通过HDFS作为其数据源。它支持各种文件格式，包括Text、RCFile和SequenceFile。
### 2.1.4 DataStream API
DataStream API是Impala提供的一个Java编程接口，用于定义实时数据流，允许开发人员创建可重用的、高效的应用程序。通过该API可以轻松地开发面向实时数据流的应用程序，这些应用程序能够消费实时的输入数据并产生实时的输出数据。DataStream API可以用来实现对实时数据的复杂分析、实时报告等场景。
## 2.2 Impala核心组件
Impala的主要组件有：
- Catalog服务：它是Impala用来存放数据库、表及分区等元数据的服务。
- Statestore服务：它是一个高可用性服务，用来保存Impala的运行状态，包括查询计划、查询队列和数据库元数据等信息。
- Impalad节点：Impalad是Impala的工作进程，它负责执行查询任务，从HDFS、本地磁盘或内存中读取数据，对查询请求进行实际的计算，并将结果集返回给客户端。每个Impalad都有自己独立的内存资源，并且可以通过配置项控制最大内存分配量。每个Impalad节点需要连接至一个Coordinator节点，其负责处理来自不同用户或应用的查询任务。
- Coordinator节点：Coordinator是Impala中的角色，它负责接收查询请求、解析SQL语句、生成查询计划、分配查询任务到各个Impalad节点，并对结果集进行汇总。它也可以管理查询计划的变化，如DDL语句、DML语句等。Coordinator是Impala中的单点故障，因此建议部署HA方案。
- Beeswax：它是Impala的Web界面。
- Auxiliary 服务：它包含三个服务，包括LLAMA、CatalogServer和Statestore。LLAMA是一个键值存储，用以缓冲Impalad查询的中间结果，减少网络带宽压力。CatalogServer是Impala的元数据存储，用于保存数据库、表和分区信息。Statestore是Impala运行状态监控中心。
- Planner：它是一个查询优化模块，它根据SQL语句的语法和语义生成查询计划。
- Metrics 服务：它是一个性能指标收集模块，用于记录Impala服务器的运行指标，包括CPU利用率、内存使用情况、IO等待时间、网络流量等。
- Kudu：它是一个Apache基金会开发的开源分布式存储，可以用来存储Impala内部数据。Kudu可以替代Impala的其它数据存储方式，同时还可以帮助解决传统存储引擎遇到的诸如延迟、可扩展性不足、一致性问题等问题。
# 3.Impala概览
## 3.1 Impala架构
Impala有以下几个关键组成部分：
- Coordinator: 它是一个服务节点，负责处理客户端提交的查询请求。它主要做了以下几件事情：
  - 接受来自Beeswax或HiveServer2的查询请求；
  - 根据查询请求获取元数据信息；
  - 生成查询计划；
  - 分配查询任务到多个Impalad节点；
  - 返回最终结果；
  - 将结果持久化存储到HDFS。
  
- Impalad：Impalad是Impala的一个工作进程，它负责执行查询任务，从HDFS、本地磁盘或内存中读取数据，对查询请求进行实际的计算，并将结果集返回给客户端。每个Impalad都有自己独立的内存资源，并且可以通过配置项控制最大内存分配量。每个Impalad节点需要连接至一个Coordinator节点，其负责处理来自不同用户或应用的查询任务。

- Catalog：它是一个服务节点，用来存储Impala的元数据信息，包括数据库、表、分区、索引等。Impala使用元数据信息来判断查询请求是否合法，以及决定查询的物理执行计划。
  - Catalog是Impala的底层依赖，它是一个元数据存储系统，用于存储表和数据库的信息。
  - Catalog是单点故障，因此建议部署HA方案。

- Statestore：它是一个高可用性服务，用来保存Impala的运行状态，包括查询计划、查询队列和数据库元数据等信息。Statestore服务是一个zookeeper集群，其中包含三个节点。当某个节点失效时，另两个节点均可以接管它的工作。Statestore服务在Impala中扮演着重要角色，因为它存储了所有查询相关的元数据，用于维护Impala的运行状态，包括查询计划、查询队列等。

- Llama：Llama是一个HBase存储模块，用于存储Impala的临时数据。当Impala需要访问远程数据源（如HDFS）时，它先将数据加载到Llama中，然后再传输到Impalad节点进行计算。Llama使用了HBase作为底层依赖，可以有效地降低Impala查询性能瓶颈。

- Metrics 服务：它是一个性能指标收集模块，用于记录Impala服务器的运行指标，包括CPU利用率、内存使用情况、IO等待时间、网络流量等。Metrics 服务可以帮助管理员对Impala的性能和资源消耗进行管理和监控。

- Metastore：它是一个独立的服务，用于存储Hive中的元数据信息。Metastore的作用类似于Impala的Catalog，但是它专门用于Hive。

- DDL Server：它是一个独立的服务，用来管理Impala的元数据信息。它可以用于更新表结构或修改表属性。

- Storage Layer：它是一个与底层文件系统交互的库。它封装了底层文件的读写，通过它可以更加方便地读写各种文件类型（如HDFS，本地磁盘，S3），且无需关注底层文件系统的实现细节。

- Java/Python客户端：Impala提供了Java和Python两种客户端。Java客户端可以通过JDBC或ODBC接口与Impala进行交互；Python客户端可以连接到Impala，并使用IMPYLA库来进行SQL查询。


 ## 3.2 查询流程
 当用户发送一条查询请求到Impala的时候，它经过以下几个步骤：
 1. 用户提交查询请求；
 2. 解析器将SQL语句转换成抽象语法树AST；
 3. 词法分析器将SQL语句切分成Token序列；
 4. 语法分析器验证Token序列的语法正确性，生成解析树；
 5. 优化器根据统计信息和SQL语句的复杂性调整查询计划；
 6. 执行器按照查询计划的顺序执行查询任务；
 7. 最后结果集通过网络传输到客户端。