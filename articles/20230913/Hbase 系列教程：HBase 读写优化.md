
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是 HBase?
HBase 是 Apache 的开源 NoSQL 数据库，由 Apache Hadoop 框架内的 HDFS 提供支持。它是一个分布式、面向列的存储系统。通过 HBase 可以对大量结构化和半结构化的数据进行高效率地查询分析。同时，HBase 可以通过扩展多个 regionserver 来提供高可用性，在某些情况下甚至可以做到线性可扩展。HBase 为实时分析和数据仓库提供了强大的能力。
## 为何要用 HBase？
HBase 适用于需要高吞吐量、低延迟的分析场景。下面是一些典型应用场景：
- 时序数据分析：对日志、监控数据等按时间戳索引的数据进行快速分析，例如搜索最近十分钟发生的所有故障。
- 网页爬虫：海量网页数据的存储和检索。
- 数据收集：收集各种数据源（日志、跟踪信息、设备数据等）并存储在 HBase 中进行实时分析。
- 社交网络：对用户间的关系网络、社交图谱进行复杂分析。
- 实时数据处理：对实时流数据进行实时计算和分析。
- 数据仓库：将结构化和非结构化数据集成到一起，形成一个统一的数据仓库。
# 2.HBase 的基本概念
## Namespace 和 Table
HBase 中的 Namespace 和 Table 分别对应于关系型数据库中的数据库和表。Namespace 可以认为是在逻辑上划分的一组 Table，它们共享相同的物理 schema。而每个 Table 有自己的 row key 和 column family 。
- **Row Key**：每行都有一个唯一标识符，一般为字符串或整数类型。它主要用来进行范围扫描、排序等操作。
- **Column Family**：列簇是指一个具有共同前缀的集合。它允许开发人员对某一列族下所有列进行灵活的访问控制。
- **Version**：版本号用于实现多版本控制，即保存不同版本的值。当更新一条记录的时候，新值会覆盖旧值，默认保留最新的三个版本。
- **Timestamp**：时间戳用于区分不同版本的值。每个值都有对应的时间戳。
- **Cell**：单元格是指特定行列组合下的一个具体的值。
## RegionServer
RegionServer 是存储数据的主要组件。它负责管理 Region ，并且接收客户端请求，返回结果。它也负责分配 Region 到不同的服务器节点，确保数据均匀分布。RegionServer 通过从 Master 获取元数据信息，来确定自己所服务的 Region 。Master 是 HBase 集群中唯一的中心节点。
- **Master**：负责管理 Region 及其分布，并协调 RegionServer 之间的通信。
- **Client**：负责发送请求给任意 RegionServer ，并且获取响应结果。
- **Region**：HBase 会把表按照 rowkey 的范围分割成固定大小的 region，每一个 region 包含了一部分 rowkey 的数据，存储在对应的 regionserver 上。
# 3.HBase 读写优化
## 1.预热缓存：当表刚被打开或者空闲了一段时间，需要预先加载 HFile 文件到内存，这样客户端可以很快的访问这些数据。
可以通过 hbase shell 命令执行该操作：`scan 'table_name',{CACHE => IN_MEMORY}`。
注意：如果预热缓存之后，发现 HFile 文件占用的内存比预期还多，可以尝试手动调用 GC 清除掉这些数据。
```
hbase(main):012:0> scan 'user_info', { CACHE=>IN_MEMORY }  
ROW                          COLUMN+CELL                                                                                                                           
9999                         column=age:timestamp\x00, timestamp=1579140327455, value=\x00\x00\x00\x00\x00\x00\x00\x00                                           
1                           column=email:\x00, timestamp=1579140332797, value=test@qq.com  
2                           column=name:type,\x00, timestamp=1579140332781, value=UTF8Type           
...                                              ...                                                                                                     
```
## 2.Memstore 和 WAL
HBase 使用 MemStore 把数据写入内存，MemStore 中是即时生效的数据。当 MemStore 中的数据量达到一定阈值时，将数据刷新到磁盘上的 HFile 文件。HBase 使用 Write Ahead Log （WAL）保证数据安全。WAL 主要用来防止系统崩溃或者机器宕机导致数据丢失。WAL 在内存中，当 MemStore 刷新到磁盘后，WAL 中的数据也就被清空了。
## 3.压缩：为了减少磁盘空间的消耗，HBase 支持对数据块（HFile）进行压缩。压缩方式有很多种，例如 Gzip、Snappy、LZO 等。
可以使用 `alter ‘table_name’, { NAME=>'family_name', COMPRESSION => ‘algorithm’ }` 设置某个 ColumnFamily 的压缩方式。如：
```
alter 'user_info', { NAME=>'info', COMPRESSION => 'GZ' };
```
目前 HBase 官方推荐的压缩算法是 LZO，但由于 LZO 需要 Java Native Interface，可能无法直接在命令行或者脚本中配置。如果一定要使用 LZO，则需要安装 JNI 依赖包才能启用压缩功能。
## 4.BlockCache：HBase 默认开启 BlockCache，它的作用是加速随机读取的数据。BlockCache 会缓存热点数据到内存中，这样，当有热点数据需要访问时，可以直接从内存中读取，不用再到磁盘中读取，提升了查询效率。
使用 `set ‘hbase.regionserver.hfileblockcache.size’=0.4` 设置 BlockCache 占用内存的比例。
## 5.Bloom Filter：Bloom Filter 是一种数据结构，它可以判断元素是否在一个集合中。当我们需要检查一个元素是否存在于一个集合中时，如果这个元素不存在，那么 Bloom Filter 可以非常快速的告诉我们这个元素一定不存在；但是如果这个元素存在，那么 Bloom Filter 只能说可能存在，不能证明一定存在。因此，Bloom Filter 通常可以用来快速判断大型集合中的元素是否存在，且误判概率较低。
使用 `alter ‘table_name’, { NAME=>'family_name', BLOOMFILTERS => true }` 配置某个 ColumnFamily 用 Bloom Filter。如：
```
alter 'user_info', { NAME=>'info', BLOOMFILTERS => true };
```