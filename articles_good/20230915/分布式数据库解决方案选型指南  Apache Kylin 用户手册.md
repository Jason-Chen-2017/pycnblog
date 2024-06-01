
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Apache Kylin是一个开源的分布式数据分析引擎（OLAP）系统。它的设计目标就是通过大规模并行处理、多维分析查询和快速响应时间，解决企业海量数据的复杂分析问题。目前，Kylin已在多个大型互联网公司落地，成为事实上的“大数据 OLAP 引擎”。它具备以下主要优点：

- 数据倾斜处理能力强：Kylin 可以有效处理数据倾斜问题，对热点数据进行采样聚合等方式，提升查询性能；
- 支持灵活的数据模型定义：Kylin 可以灵活的支持多种数据模型，包括星型模型、雪花模型、Lookup模式等，满足各种分析场景需求；
- 高度可扩展性：Kylin 的存储分片和计算分区机制能够灵活的支持海量数据量和多节点集群部署；
- 近乎无限的查询能力：Kylin 提供 RESTful API 和 JDBC/ODBC 两种接口，用户可以使用任意语言或工具快速接入；
- 大规模并行计算能力：Kylin 通过 MapReduce 和 Spark 两种计算引擎支持大规模并行计算，处理 PB 级以上的数据量；
- 自动生成SQL语句：Kylin 使用优化器自动生成 SQL 查询计划，用户可以省去大量SQL编写工作。
本文将从Apache Kylin用户手册的“快速入门”章节，介绍Apache Kylin相关概念及架构、安装配置、核心功能等方面的知识。
# 2.快速入门
## 2.1 概念和术语
### 2.1.1 Apache Kylin简介
Apache Kylin是开源的分布式数据分析引擎，其具备以下特征：

- **高效:** Kylin 使用 HLL算法实现超精确近似 distinct count、TOP N等复杂查询功能，大幅减少了磁盘 IO 压力，提升查询效率；
- **统一接口:** Kylin 提供了一套标准化的 RESTful API，支持多种客户端，如 Java、Python、R、Node.js 等；
- **数据驱动:** Kylin 在设计之初就把数据分析当作中心任务，根据用户输入的业务逻辑，智能生成最适合分析的数据模型；
- **容错性:** Kylin 使用 Hadoop 和 Zookeeper 技术栈提供高度容错能力，保证查询服务的高可用；
- **易用性:** Kylin 提供友好的用户界面，使得非技术人员也能轻松上手，并且支持丰富的查询语法；
- **扩展性:** Kylin 可伸缩性好，通过切分 Cube 和索引数据，集群中节点不限制，支持 PB 级别数据量的处理；
- **免费:** Apache Kylin 是完全免费的，且社区活跃，开发者团队一直在积极参与项目开发。

### 2.1.2 Cube概述
Cube 是 Apache Kylin 中重要的概念。它是用户使用 Apache Kylin 时所创建的分析模型，是Apache Kylin中的基本数据单位。一个Cube由以下三部分构成：

- **Dimensions:** Dimensions 表示维度字段，例如统计城市、统计日期等。
- **Measures:** Measures 表示指标字段，例如订单金额、点击次数等。
- **Aggregation Groupings:** Aggregation Groupings 表示聚合组，用于划分多个不同维度值组合的维度子集，如按月份划分。

每个Cube都有一个默认的时间粒度（比如按天、按周），而其他维度需要在该默认的时间粒度之外进行拆分。对于维度的数量没有硬性限制，但通常不超过十个。对于Measures，其数目只能有一个，且必须指定。 

除了数据模型本身，还可以通过 Star Schema 模式和 Lookup 模式扩展 Cube 的维度和度量。Star Schema 模式通过组合维度表的方式将维度表中的数据保存到离线层面，并通过关联维度表及事实表的方式来支持快速分析。Lookup 模式提供了额外的维度表，用于为度量提供辅助信息，如地理位置信息，而不实际存储在主表中。 

### 2.1.3 Rollup概述
Rollup 是一种Cube的预聚合策略。为了避免Cube在低基数维度上过于细化造成的资源消耗过多，可以考虑对Cube进行预聚合，即将某些维度做预先聚合，聚合之后的结果作为结果存储。Rollup可以有效的降低数据量和分析查询的响应时间。

Apache Kylin 提供了两种 Rollup 策略:




### 2.1.4 Instance概述
Instance 是一个运行环境，其中包含了一个或多个Kylin实例。每个Kylin实例具有独立的元数据存储库和计算资源池。Kylin实例之间共享元数据和计算资源，所以单个Kylin服务器无法满足高并发访问需求。

Apache Kylin 的安装包中包含了一个名为 kylin.sh 的脚本文件。该脚本负责管理Kylin实例，包括启动、停止、升级、回滚、查看日志等。实例在启动时，会自动连接ZooKeeper，获取集群信息，根据集群信息启动相应的计算资源。

## 2.2 安装配置
### 2.2.1 下载安装包
Apache Kylin提供两种安装包形式，分别为源代码包和压缩包。用户可以根据自己的网络环境选择安装包下载地址，从而完成安装。

| 版本 | 类型          | 安装包                                              |
| ---- | ------------- | --------------------------------------------------- |
| 3.x  | 源码包        | https://archive.apache.org/dist/kylin                  |
|      |               | /apache-kylin-3.1.1-src.tar.gz                      |
|      | 压缩包        | https://archive.apache.org/dist/kylin                  |
|      |               | /apache-kylin-3.1.1-bin-hadoop3.tgz                 |
|      |               | 或                                                   |
|      |               | apache-kylin-3.1.1-bin-hbase2.1-spark2.4.tgz         |
| 2.x  | 源码包        | https://archive.apache.org/dist/incubator                |
|      |               | /kylin-2.6.0-incubating-src.zip                     |
|      | 压缩包        | https://archive.apache.org/dist/incubator                |
|      |               | /kylin-2.6.0-incubating-bin-hbase1.x-hive1.x.tar.gz   |
|      |               | 或                                                   |
|      |               | kylin-2.6.0-incubating-bin-hbase2.x-hive2.x.tar.gz    |



### 2.2.2 配置Kylin
安装完成后，首先修改conf目录下kylin.properties配置文件，主要修改如下参数：
```bash
######################################################
# Cluster Configuration
######################################################
# The cluster name, default is "Cluster"
kylin.cluster.name=my_cluster
# Set the server mode to "all" if you want this instance to serve both query and ingestion requests
kylin.server.mode=query # or all
# Additional metadata url that will be loaded in the system. Multiple urls can be set separated by comma. For example, http://meta01:7070/kylin
kylin.metadata.url=http://meta01:7070/kylin
# Set hive variable here, so that spark jobs could use it to connect Hive
kylin.hive.variables=set mapreduce.job.queuename=root.queueName;

######################################################
# Query Configuration
######################################################
# The max number of concurrent threads executing queries on one node
kylin.query.concurrent-threads-per-query=10
# The threshold value for scanning segments during query execution. Any segment with scanned rows count above the threshold would not be executed
kylin.query.segment-scan-threshold = 100000000
# Enable pushdown filter optimization for ORC format files. Default false. This configuration should only enabled when there are filters pushed down to storage layer because Kylin merges data into a big table before scan using Spark which may impact performance. But it might also enable some additional optimizations like skipping useless column projection. Please note, disabling this option may have impact on query correctness especially when filtering on low cardinality dimensions.
kylin.optimize.pushdown.orc.filter=true
# Enable pushdown column projection for Parquet format files. Default true. When disabled, query engine skip unnecessary columns from Parquet file reducing disk io and memory footprint. It may slow down query latency slightly but improve query throughput significantly. Please note, disabling this option may cause incorrect results especially when selecting high cardinality dimensions due to missing dimension columns in row groups.
kylin.optimize.pushdown.parquet.column-projection=false

######################################################
# Storage Configuration
######################################################
# Local directory where Kylin stores its metadata and intermediate result
kylin.storage.url=/home/<username>/.kylin/
# HBase related settings. Kylin can leverage HBase as its Metadata store and Job Engine cache
kylin.hbase.zk-quorum=hbase_host1:port,hbase_host2:port,hbase_host3:port
kylin.hbase.hbase.client.retries.number=3
kylin.hbase.hbase.rpc.timeout=60000
kylin.hbase.hbase.client.pause=10000

######################################################
# Security Configuration
######################################################
# Define authentication type, supported values are "NONE", "SPNEGO" and "LDAP". If set to NONE, Kylin won't check any user identity. Otherwise, client needs to provide valid Kerberos ticket or LDAP username/password to access resources.
kylin.security.authentication=NONE
# If SPNEGO is used, specify the keytab file location (this file contains your login credentials)
kylin.security.spnego.keytab=/etc/security/keytabs/HTTP.keytab
# Specify the service principal of KDC (for SPENGO). For example HTTP/kylin.example.com@EXAMPLE.COM
kylin.security.spnego.principal=HTTP/hostname.domain.com@REALM.COM
# To enable LDAP authentication, configure these parameters accordingly
kylin.ldap.url=ldaps://ldap_host1:636
kylin.ldap.domain=dc=company,dc=com
kylin.ldap.base.dn=ou=users,dc=company,dc=com
kylin.ldap.user.search_pattern=(uid={0})
kylin.ldap.group.search_pattern=(memberUid={0})
kylin.ldap.manager.dn=cn=Manager,dc=company,dc=com
kylin.ldap.manager.password=secretpassword
# Kylin uses a separate set of properties for encryption keys etc., please refer to Encryption section for details
```

### 2.2.3 初始化Hive连接
Kylin需要配置Hadoop、Hive连接，才能访问Hive数据源，使用以下命令初始化：
```bash
./bin/kylin.sh org.apache.kylin.common.util.HadoopUtil createHadoopConf /path/to/conf/kylin.properties
```
此命令会在 conf/kylin下生成 hive-site.xml 文件，修改该文件时需要注意，否则会导致连接失败。

### 2.2.4 创建第一个Cube
登录Kylin UI页面，在左侧菜单栏依次选择“Configuration” -> “Cubes”，创建一个新的Cube。

- 名称：Cube名称，推荐与数据源表名相同，便于理解。
- Model Name：数据模型名称。
- Project：所属项目名称。
- Source Type：数据源类型，目前支持HIVE、KYLIN、GENERIC。HIVE表示来自Hive数据源，KYLIN表示来自另一个Kylin数据模型，GENERIC表示来自其他数据源。
- Source Filter Condition：过滤条件，Kylin利用该条件过滤出需转换的表格数据。
- Kylin Data Source：数据源名称。
- Kylin Database：Hive数据库名称。
- Kylin Table：Hive表格名称。
- Dimensions：Cube的维度设置，可以添加多个维度。
- Measures：Cube的度量设置，可以添加多个度量。
- Groups：Cube的聚合组设置，可以添加多个聚合组。
- Time Granularity：时间粒度，Cube中包含数据的最小粒度。

填写完信息后，点击右下角的“Save”按钮，即可创建新Cube。

## 2.3 核心功能介绍

### 2.3.1 查询
Apache Kylin支持SQL、JDBC和RESTFul API形式的查询接口。使用KAP演示版测试的Apache Kylin集群可以正常访问该站点。

登录Kylin UI页面，在搜索框输入要查询的SQL语句，然后点击“Run”按钮执行查询。如果查询语句中包含中文字符，可能出现乱码问题。


### 2.3.2 写入数据
Apache Kylin支持写入数据到Hive表。但是由于安全限制，建议仅允许内网写入，禁止外网写入。

登录Kylin UI页面，在“Modeling”菜单栏选择需要写入数据的Cube，然后点击右边的“Update Cube”按钮进入Cube编辑页面。切换到“Derive”标签页，找到需要写入数据的维度，勾选右侧的复选框，点击“Next Step”按钮进入设置Derived Columns设置页面。


在页面上，需要填写一下信息：

- Column Name：输出列名。
- Column Expression：列表达式，即将新加入的维度映射到列中。
- Inactive：是否生效，生效的Derived Column才会被保存到CUBE元数据中。
- Description：描述信息。

示例：

假设原始维度是“USERID”和“CREATED_DATE”，希望新加入两个维度“DAY”和“HOUR”，并将他们组装起来得到完整的时间戳。则填充的信息为：

- Column Name：`TIMESTAMP`
- Column Expression：`day(CREATED_DATE)*86400 + hour(CREATED_DATE)*3600`
- Inactive：未勾选
- Description：构造出来的时间戳。

确认无误后，点击右上角的“Apply”按钮保存，再次打开Cube编辑页面，点击左侧的“Jobs”按钮，选择“Build”菜单项，等待Cube构建完成。

### 2.3.3 设置缓存
Apache Kylin支持缓存机制，在查询时，首先会检查缓存中是否有所需的数据，若有则直接返回结果，否则会通过查询源数据库获取。对于大型数据集，可以有效减少查询响应时间，提升整体吞吐量。

Kylin在创建Cube时，会在HDFS上创建一个新的目录，用于存放该Cube的缓存数据。可以将该目录上传至HDFS，或者使用KAP演示版中的S3 Bucket上传。然后在“Configuration” -> “Cache Sources”中配置该目录。


配置完成后，点击右上角的“Save”按钮，即可应用缓存配置。

### 2.3.4 快速检索
Apache Kylin支持用户词典的快速检索。在“Configuration” -> “User Dictionary”中，可以增加自定义的关键词，并进行检索。
