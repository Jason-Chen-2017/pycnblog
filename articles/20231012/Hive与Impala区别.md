
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Hive是一种开源的分布式数据仓库系统，它能够将结构化的数据文件映射到一张表上，并提供HQL（Hive Query Language）查询语言用于对其进行分析处理。它最初由Facebook开发，2009年成为Apache基金会顶级项目。在最近几年，Hive开始蓬勃发展，已经成为当今最受欢迎的开源分布式数据仓库系统。

Impala是Cloudera公司基于开源的Hive项目实现的一款分布式数据仓库系统，其特点是其运行速度快、兼容性好。它支持SQL，具有高性能、易用、安全、可扩展等优点。

Hive与Impala都是基于HDFS(Hadoop Distributed File System)构建的分布式数据仓库系统。但是，Hive可以理解成是一个更加通用的解决方案，而Impala则更偏向于海量数据的高性能查询。由于它们共享了很多相同的代码，所以对于同样的应用场景，选择它们中的一个主要取决于业务需求和团队资源。

本文将对两者进行区分、功能特性、区别以及适用场景等方面作出阐述，希望能帮助读者对两者有一个整体的认识和理解。

# 2.核心概念与联系
## 2.1 概念介绍
### 2.1.1 Hadoop Ecosystem
Hadoop是Apache基金会开放源代码的“大数据”框架之一，具有高度的容错性、可靠性、伸缩性、易用性及可扩展性。Hadoop生态系统包括四个主要组件：Hadoop Common、HDFS、MapReduce、YARN。


- Hadoop Common: 提供了Hadoop最基础的编程接口、集群管理器、I/O处理、压缩、序列化、RPC等功能。
- HDFS (Hadoop Distributed File System): 是一种分布式文件系统，提供了高容错性、高吞吐率和可扩展性，可用于存储大规模数据集。
- MapReduce: 是一种编程模型和计算框架，用于编写快速、可靠且容错的分布式应用程序。
- YARN (Yet Another Resource Negotiator): 是Hadoop的资源调度器，负责集群中各节点上的资源分配和任务执行。

Hadoop Ecosystem常见的应用场景有：

- 数据采集：如日志收集、搜索索引、实时数据源、离线批量处理等。
- 数据分析：如联邦数据集市、数据挖掘、机器学习等。
- 数据湖：使用HDFS进行海量数据的存储、处理和分析。
- 大数据统计：Hadoop提供高性能的数据分析和处理能力，可用于实时分析、数据挖掘等应用场景。

### 2.1.2 Hive Architecture
Hive是基于Hadoop的一个数据仓库工具。它包含一个用于定义数据的抽象层——数据结构存储在Hadoop的一个文件系统上；并且还包含用于定义针对数据的操作的基于SQL的语言——HQL（Hive Query Language）。

Hive架构如下图所示：


1.元存储：元存储主要存储hive相关的信息，比如表结构、表的统计信息、存储位置、权限等。元数据保存在MySQL数据库或Derby数据库。
2.驱动程序：驱动程序负责接收用户提交的HQL语句并转换为MapReduce任务。
3.HiveServer2：HiveServer2是实际执行HQL语句的服务进程。它接收客户端提交的HQL请求，将其解析、优化、编译成MapReduce任务，并提交给执行引擎。
4.HIVE-METASTORE：hive-metastore是hive中用于存放元数据的组件，负责存储hive中表的结构信息、表的统计信息、字段注释信息等。
5.执行引擎：执行引擎用于执行MapReduce任务，负责输入输出数据的切分和排序、MapReduce任务的调度、任务的监控和日志记录等。
6.HDFS：hdfs是hadoop的分布式文件系统。用于存储hive的数据。

### 2.1.3 Impala Architecture
Impala是Cloudera公司基于Hadoop的Hive实现的一个开源分布式数据仓库系统。它与Hive一样使用HDFS作为存储，但不同的是它没有自己的元存储。它直接与HDFS交互，通过读取表元数据信息，来执行HQL查询。

Impala架构如下图所示：


1.Impalad进程：Impalad进程是一个Impala的服务进程，用于接收客户端提交的HQL请求，并将其编译为查询计划，并提交给Impala底层的执行引擎。
2.Coordinator进程：Coordinator进程是一个协调者进程，它负责生成查询计划，并把查询计划提交给各个Impalad进程。
3.执行引擎：执行引擎是Impala的核心模块，它接受Impalad进程的查询计划，并执行查询计划。
4.HDFS：Impala使用的就是HDFS。

## 2.2 功能特性
### 2.2.1 数据导入
Hive与Impala都可以通过导入外部数据文件的方式来增加数据，导入方式一般分为两种：

**内部表导入**：创建空表，然后通过LOAD命令来导入数据，内部表导入不需要指定列名。

```sql
CREATE TABLE table_name LIKE src_table; -- 通过 LIKE 操作符复制表结构，但不包括数据
LOAD DATA INPATH 'file:///path/to/data' INTO TABLE tablename; -- 从本地文件导入数据
```

**外部表导入**：创建一个外部表，然后插入数据，再用CREATE TABLE AS SELECT命令将外部表转换为内部表。

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS external_table (
  col1 STRING,
  col2 INT,
 ...
) STORED AS TEXTFILE LOCATION '/my/external/location'; -- 创建外部表

INSERT OVERWRITE DIRECTORY '/my/external/location/' 
  ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
  COLLECTION ITEMS TERMINATED BY ':' 
  MAP KEYS TERMINATED BY '-' 
  ENCLOSED BY '"' 
  INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat' 
  OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
SELECT * FROM internal_table WHERE condition = xxx; -- 插入数据

CREATE TABLE internal_table like external_table; -- 将外部表转换为内部表
INSERT INTO TABLE internal_table select * from external_table; -- 用内部表替代外部表
```

### 2.2.2 分区表
Hive与Impala都支持分区表。

**Hive分区表**：

```sql
CREATE TABLE my_partitioned_table (
    id int,
    name string
) PARTITIONED BY (year int, month int); 

ALTER TABLE my_partitioned_table ADD PARTITION (year=2021, month=10); 
```

**Impala分区表**：

```sql
CREATE TABLE impala_partitioned_table (
    id int,
    name string
) partitioned by (year int, month int) stored as parquet; 

alter table impala_partitioned_table add partition (year=2021, month=10);  
```

分区表能够提升查询效率，因为只需要扫描与查询条件匹配的分区即可。

### 2.2.3 查询优化
Hive与Impala都支持查询优化。

**Hive查询优化**：

Hive采用基于规则（rule based optimization，RBO）的查询优化器。它会自动检测查询的谓词、表之间的依赖关系、数据分布情况等，并根据不同的场景选择最优的查询计划。

**Impala查询优化**：

Impala采用基于成本（cost-based optimization，CBO）的查询优化器。它根据多种优化策略，如代价估计、查询路径选择、列裁剪、索引选择等，来产生有效的查询计划。

### 2.2.4 数据缓存
Hive与Impala都支持数据缓存。

**Hive数据缓存**：

Hive支持对表和分区数据进行缓存。当查询计划运行时，数据会被缓存到内存或者磁盘中。

**Impala数据缓存**：

Impala也支持对表和分区数据进行缓存。但是与Hive相比，它的缓存机制更为复杂。

## 2.3 区别
### 2.3.1 存储格式
Hive的默认存储格式为ORC格式，而Impala的默认存储格式为Parquet格式。

**ORC**: ORC（Optimized Row Columnar Format）是一种列式存储格式，利用率更高，压缩比更高，查询速度更快。Hive的ORC格式支持高压缩率、高查询性能。

**Parquet**: Parquet是一种列式存储格式，它同时支持高压缩率、高查询性能。Parquet可以存储复杂的数据类型，例如数组、嵌套类型、字典编码字符串、二进制数据等。

### 2.3.2 时间序列数据
Hive支持对时间序列型数据进行查询。而Impala仅支持非时间序列型数据。

### 2.3.3 支持复杂数据类型
Hive支持复杂的数据类型，包括数组、集合、地理位置、复合类型、结构类型等。而Impala仅支持基本数据类型，不支持复杂的数据类型。

### 2.3.4 延迟加载
Hive支持延迟加载。当查询访问的数据没有在内存缓存中时，才会从HDFS中加载数据。

Impala不支持延迟加载。

### 2.3.5 查询计划优化
Hive支持基于RBO的查询优化，Impala支持基于CBO的查询优化。

### 2.3.6 数据缓存
Hive支持对数据进行内存和磁盘缓存，而Impala仅支持对数据进行内存缓存。

## 2.4 适用场景
Hive适用于大数据分析和BI，适合ETL（extract transform load），由于它可以支持复杂的数据类型，适合复杂业务系统的数据分析。

Impala适用于OLAP（on line analytical processing）和数据仓库，支持复杂的数据类型，查询速度更快，适合OLTP（on line transaction processing）。