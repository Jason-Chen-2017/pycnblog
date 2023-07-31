
作者：禅与计算机程序设计艺术                    

# 1.简介
         
BigTable是一个可伸缩的分布式结构化存储数据库系统。其主要特点是高度可靠、高性能、可扩展性强、提供海量数据存储。目前在Google内部广泛应用于大规模集群环境的实时计算场景。由于其采用了稀疏数据结构，所以它可以有效地解决海量数据存储的问题。本篇文章将会对BigTable的整体架构进行详细阐述，并通过源码剖析的方式，带领读者实现自己的理解。
# 2.BigTable相关术语
## 2.1 数据模型
BigTable中的数据模型包括：
 - Table: BigTable中的数据按照行的形式存放在一个个的表中，每个表都有一个唯一的名称，用户可以通过表名和RowKey定位到具体的一行。
 - Row Key: 每一行的数据都是根据行的RowKey进行索引的。RowKey由用户指定，并且可以非常灵活，既可以是单个的字符串值也可以是多个列组合起来的复合索引键。
 - Column Family（列族）: 用户可以在每张表中定义一个或多个Column Family，每一列族下都包含若干列（Column），每个列都具有相同的前缀名称（即列族名称）。这样做可以把相关的列集中在一起，方便管理。
 - Column Qualifier（列限定符）: 一个列由两部分组成——列限定符和版本号。其中列限定符用于标识该列属于哪一列族，而版本号则用于对同一列的不同版本进行区分。
 - Timestamp: 每个Cell都有相应的时间戳，用于标记Cell数据的版本。

## 2.2 存储与计算分离
BigTable在存储和计算分离的同时，还提供了灾难恢复、数据复制、负载均衡等功能，使得它具备了高可用性。在某些情况下，用户也可通过控制数据的访问权限来保护数据安全。
BigTable通过分层的结构保证了数据的高可用性：
 - 物理层（硬件）: BigTable集群运行在多台物理服务器上，保证存储系统的高可用性。如果某个服务器出现故障，其他服务器仍然可以提供服务。
 - 分布式层（软件）: BigTable支持自动的副本机制，保证数据冗余。当集群中的机器发生故障时，BigTable可以自动切换到另一台机器上继续提供服务。
 - API层: 提供易用的API接口，用户可以使用它们操纵数据。

# 3.BigTable架构设计与实现
## 3.1 概览
![BigTable Architecture Overview](https://drek4537l1klr.cloudfront.net/bigtable-architecture-overview.png)
如图所示，BigTable是一种分布式的结构化存储数据库系统，由多个服务器构成。客户端向其中写入数据时，首先需要选择一个表，然后根据表的规则生成RowKey；BigTable先根据RowKey将数据分配给不同的Tablet Server处理，每个Tablet Server再分别将数据保存到不同的磁盘中。用户可以通过表名、RowKey或者Cell定位到具体的一行或一列的值。

每个Tablet Server上都维护着一个内存中的数据结构——MemStore。当用户读取数据时，首先从本地的MemStore读取，然后如果没有查找到目标数据，则查询对应的多个磁盘文件。为了减少磁盘I/O开销，BigTable采用了列族的机制，只将相同列族的列存在一起。所有的数据都是经过压缩的，只有被修改的部分才会被重新编码后存储。

数据写入完成后，数据会持久化到一系列的磁盘上。这些磁盘被划分为多个块，每个块内包含多个tablet。BigTable默认每一个tablet的大小为64M。当某个tablet被填满时，会触发将该tablet的数据分裂成两个新的tablet。此外，BigTable还维护着多个预写日志（WAL），记录着所有的写操作。

当用户需要读取数据时，BigTable会将请求发送给Master Server，Master Server会将请求路由到正确的Tablet Server。Master Server还会负责将数据合并成最终结果返回给用户。

## 3.2 Master Server
Master Server用于管理整个集群，并负责Tablet Server之间的调度。Master Server主要有以下职责：
 - 将数据分配给Tablet Server。
 - 检查Tablet Server是否正常运行。
 - 处理客户端请求。

Master Server使用Paxos协议选举出一个Leader。在任何时间点，只能有一个Leader存在。当Leader挂掉时，会重新选举出一个新的Leader。

## 3.3 Tablet Server
Tablet Server用于实际保存和检索数据的服务器。Tablet Server主要有以下职责：
 - 将数据存储在内存或磁盘上。
 - 执行数据查找、排序等操作。
 - 根据请求实时更新数据。

Tablet Server使用Bloom过滤器提升查询效率。另外，Tablet Server还会缓存最近访问的数据。Tablet Server使用顺序写、随机读的机制，以提升写入效率。

## 3.4 MemStore
MemStore是Tablet Server上的内存结构，用于缓存最近写入的数据。当Tablet Server启动时，都会清空MemStore。当Tablet Server收到写请求时，首先会写入MemStore，当达到一定数量后，会刷新到磁盘上的SSTable文件中。SSTable就是Sorted String Table的简称，用于存储数据。SSTable文件里包含了很多的键值对，每个键值对由ColumnFamily、Qualifier和Timestamp三部分组成。SSTable可以被压缩，方便快速读取。

MemStore采用了LRU策略，当容量达到限制后，会优先淘汰老旧的数据。

## 3.5 WAL
Write Ahead Log（WAL）是BigTable中用来记录写操作日志的文件。当Tablet Server接收到写请求时，会首先记录在WAL文件中，然后才会更新MemStore中的数据。如果Tablet Server意外崩溃，可以从WAL中重建MemStore。

WAL文件采用了追加方式写入，因此当多个Tablet Server同时写数据时，可能会产生竞争条件。不过，因为Tablet Server之间的数据完全相同，因此并不会造成冲突。

## 3.6 SSTable Splitting
BigTable采用Splitting（拆分）机制来解决tablet过大的问题。当某个tablet的大小超过64M时，会被切割成两个相邻的tablet。切割 tablet 的过程叫作 SSTable splitting。切割时，当前tablet 的右半部分会成为新的 tablet ，左半部分依然留在原来的tablet 上。

对于新生成的tablet，新的tablet server 会接管原先tablet 的工作，原先tablet 上的数据也会转移到新的tablet 上，使得tablet 拆分后的容量分布更加均匀。

# 4.BigTable源码分析
BigTable的代码是开源的，可通过github获得：https://github.com/apache/hbase 。下面就让我们一起探索BigTable的源代码吧！

