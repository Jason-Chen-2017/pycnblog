
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Apache Hive是一个开源的分布式数据仓库软件，可以用来进行数据提取、转换、加载（ETL）、查询等功能。作为Hadoop生态系统的一员，Hive具有强大的分析能力、灵活的数据定义、数据处理、数据分析和可扩展性，是一个理想的企业级数据仓库解决方案。为了更高效地管理海量的数据，需要对Hive的配置和运行方式进行优化。本文将介绍Hive优化技术，包括Hive配置参数、分区设计、表扫描方式、Join操作优化、外部表存储优化等方面，并结合实际案例分析Hive集群的部署架构及集群规划。
# 2. 相关技术基础
## 2.1 Hadoop生态体系
- HDFS（Hadoop Distributed File System）：Hadoop分布式文件系统，是一个高度容错的存储系统，能够提供高吞吐量的数据访问。HDFS有助于在集群中存储和处理大型数据集，同时它也具备高容错性，能够保证数据的安全和完整性。
- YARN（Yet Another Resource Negotiator）：一个分布式资源管理框架，用于启动和监控MapReduce作业，并根据集群中的可用资源调度任务执行。它使得用户不需要了解底层集群如何运行，只需指定作业逻辑，就可以提交给YARN并让其自动处理。
- MapReduce：一个分布式计算模型，主要用于批量数据处理，将输入数据集分割成独立的“映射”任务，并把每一个映射任务的输出发送到相应的“归约”任务，最后得到整个数据集的一个汇总结果。
- Tez：一种基于Hadoop的计算框架，可以实现复杂的交互式查询，并通过减少数据移动和连接来提升性能。
- Zookeeper：一个开源的分布式协调服务，主要用于管理分布式环境中的数据节点。它帮助维护集群状态、广播消息、负载均衡、同步数据等。
## 2.2 Hive概述
### 2.2.1 Hive的特点
- Apache Hive 是基于 Hadoop 的数据仓库工具，提供类似SQL语言的查询功能。
- 使用户无需编写 MapReduce 代码即可完成复杂的数据分析工作。
- 能够通过 HQL（Hive Query Language）来描述数据仓库的结构及查询，并且支持使用 MapReduce 或 Tez 之类的计算框架来执行查询。
- 可以利用自己的 UDF（User Defined Functions，用户定义函数）来拓宽 SQL 语法的功能。
- 支持 ACID（Atomicity、Consistency、Isolation、Durability），确保数据一致性。
- 内置多种类型的数据源：结构化数据（如 CSV、JSON 和 XML）、半结构化数据（如日志文件和电子邮件）、列式存储格式（例如 ORC、Parquet）。
- 在 Apache Hadoop 之上实现了元数据和查询缓存，进一步提升性能。
### 2.2.2 Hive的优势
- 数据仓库的结构化查询语言：Hive 提供 SQL (Hive QL) 语言来对数据仓库进行结构化查询。
- 透明的计算引擎：Hive 隐藏了 MapReduce 细节，用户可以使用类似 SQL 的语句来查询数据仓库，而 Hive 会根据查询需求自动选择最适合的计算引擎进行计算。目前支持的计算引擎有 MapReduce 和 Tez。
- 用户自定义函数：Hive 提供了丰富的 UDF 函数库，用户可以自己编写 Java 函数或者 Python 函数，然后注册到 Hive 中使用。
- 面向对象的数据建模：Hive 提供面向对象的建模能力，可以方便地对数据进行分类、合并、过滤、聚合等操作。
- 易于使用和理解：Hive 的接口简单易用，学习曲线平缓，文档齐全，适合非专业人员也能轻松掌握。
- 内置丰富的函数：Hive 内置了一系列的函数，这些函数可以用于不同类型的数据处理，比如字符串处理、日期时间处理、统计分析、加密解密等。
### 2.2.3 Hive的局限性
- 不支持联接关联分析：Hive 对于复杂的关联分析不支持，只能使用笛卡尔积来实现关联分析。如果遇到大表之间的关联分析，会严重影响性能。
- 缺乏自动索引生成工具：Hive 没有自动索引生成工具，如果表的列很多，则需要手工创建索引。因此，Hive 对数据仓库的维度建模有很大的局限性。
- 查询优化器不够成熟：Hive 中的查询优化器还处于初级阶段，无法做到极致的优化。如果遇到复杂查询或大表的关联分析，可能会导致性能下降。
# 3. 性能调优方法论
## 3.1 配置参数调优
### 3.1.1 参数介绍
Hive的配置文件hive-site.xml位于$HIVE_HOME/conf目录下，包括以下参数：
```xml
  <property>
    <name>hive.auto.convert.join</name>
    <value>true|false</value>
    <description>
      Enables the automatic conversion of join operations into map joins where applicable for queries using advanced optimizations.
      When enabled, Hive will examine the size of tables being joined and determine if a map join is faster than an equivalent reduce join. If so, it will automatically convert the join to a map join for improved performance. The value should be set to true or false (default: true).
    </description>
  </property>

  <property>
    <name>hive.auto.convert.join.noconditionaltask</name>
    <value>true|false</value>
    <description>
      Determines whether conditional tasks are allowed in auto conversions from join to map join. A conditional task is any operation that evaluates a condition, such as filter or aggregation functions with conditions. If this property is not specified or its value is true, then any query that contains a join without a condition can potentially be converted to a map join based on statistics collected during query optimization. However, enabling this feature may lead to slower execution times due to additional shuffle operations. Setting this parameter to false prevents the inclusion of conditional tasks when converting joins to map joins. This value should be set to true or false (default: false).
    </description>
  </property>

  <property>
    <name>hive.cbo.enable</name>
    <value>true|false</value>
    <description>
      Enables CBO (Cost Based Optimizer), which optimizes the cost of running each statement by predicting how long individual jobs will take to complete and assigns costs to different operators in the plan based on their selectivity and relative computational resources needed to execute them. When enabled, CBO dynamically re-optimizes the query plan based on changing cluster resource utilization and workload patterns. The value should be set to true or false (default: true).
    </description>
  </property>

  <property>
    <name>hive.limit.optimize.enable</name>
    <value>true|false</value>
    <description>
      Enables pushdown of LIMIT clause into subqueries. Enabling this feature allows Hive to stop reading input data after the limit has been satisfied, reducing memory usage and improving query performance. This value should be set to true or false (default: true).
    </description>
  </property>

  <property>
    <name>hive.groupby.orderby.position.alias</name>
    <value>true|false</value>
    <description>
      Controls whether orderby expressions appearing inside a GROUP BY clause must refer to columns only via aliases, rather than direct column references. For example, given "SELECT t1.*, t2.* FROM table1 t1 JOIN table2 t2 ON t1.id = t2.table1_id ORDER BY t2.column1 DESC", if this option is disabled (the default behavior), the optimized physical plan would include both scans and sorts before joining the two tables, while if this option is enabled, the sort would be done after the join and use the full table name instead of an alias for table2's columns to allow for more efficient sorting within a single stage. Note that enabling this option may require changes to existing code that depends on implicit ordering of expressions within a SELECT list. The value should be set to true or false (default: false).
    </description>
  </property>

  <property>
    <name>hive.groupby.skewindata</name>
    <value>true|false</value>
    <description>
      Allows skewed data to be grouped together. By default, grouping keys with large differences between values may cause poor performance due to skewed data distribution. To prevent this situation, hive can group skewed data together at runtime. However, setting this flag to true causes some potential issues, including possible out-of-memory errors or incorrect results due to underestimated groups sizes. It also adds extra processing overhead to distribute small groups across multiple nodes. The value should be set to true or false (default: true).
    </description>
  </property>
  
  <property>
    <name>hive.map.aggr</name>
    <value>true|false</value>
    <description>
      Enables/disables the optimization that combines aggregations and column pruning into a single MapReduce job by applying filters directly to the Aggregation operator, thereby avoiding unnecessary shuffling and deserialization operations. This optimization applies only to vectorized ORC readers. The value should be set to true or false (default: true).
    </description>
  </property>

  <property>
    <name>hive.merge.tezfiles</name>
    <value>true|false</value>
    <description>
      Instructs Hive to merge tez container files spilled to disk during query execution. Merging reduces the number of files written to disk and improves overall system performance. The value should be set to true or false (default: true).
    </description>
  </property>

  <property>
    <name>hive.metastore.warehouse.dir</name>
    <value>/user/hive/warehouse</value>
    <description>
      Defines the warehouse directory used by the metastore when storing table metadata and partition information. Defaults to /user/hive/warehouse.
    </description>
  </property>

  <property>
    <name>hive.orcfile.encoding.strategy</name>
    <value>SPEED|COMPRESSION|SPEED_OR_COMPACT</value>
    <description>
      Specifies the encoding strategy to use for writing ORC files. SPEED means encode all columns using simple encoding, COMPRESSION means compress selected columns, and SPEED_OR_COMPACT means try both strategies depending on the file size. The value should be one of {SPEED,COMPRESSION,SPEED_OR_COMPACT} (default: COMPRESSION).
    </description>
  </property>

  <property>
    <name>hive.orc.splits.include.file.footer</name>
    <value>true|false</value>
    <description>
      Includes the file footer in every ORC split, which enables skipping over corrupted sections of the file. Enabling this feature requires additional I/O, but ensures data consistency and correctness. The value should be set to true or false (default: false).
    </description>
  </property>

  <property>
    <name>hive.querylog.location</name>
    <value>${system:java.io.tmpdir}/hive.log</value>
    <description>
      Defines the location of the Hive log file, which records various details about Hive queries. The value defaults to ${system:java.io.tmpdir}/hive.log.
    </description>
  </property>

  <property>
    <name>hive.security.authorization.enabled</name>
    <value>true|false</value>
    <description>
      Enables Hive security authorization support. With this feature enabled, users can be granted privileges to perform specific actions, such as creating databases or accessing specific tables. The value should be set to true or false (default: false).
    </description>
  </property>

  <property>
    <name>hive.server2.authentication.kerberos.keytab</name>
    <value>path/to/keytab</value>
    <description>
      Specifies the keytab file to be used when authenticating against Kerberos KDC. Used when HiveServer2 is configured for kerberos authentication mode. This value should point to a valid keytab file.
    </description>
  </property>

  <property>
    <name>hive.server2.authentication.kerberos.principal</name>
    <value>hive/_HOST@DOMAIN.COM</value>
    <description>
      Specifies the principal to be used when authenticating against Kerberos KDC. Used when HiveServer2 is configured for kerberos authentication mode. This value should match the service principal defined in the Hive server's krb5 configuration.
    </description>
  </property>

  <property>
    <name>hive.server2.authentication</name>
    <value>KERBEROS|NONE|NOSASL</value>
    <description>
      Sets the authentication type for clients connecting to HiveServer2. Used when HiveServer2 is configured for either no authentication, SASL (Simple Authentication and Security Layer) authentication, or kerberos authentication modes. Valid options are KERBEROS, NONE, NOSASL (default: NOSASL).
    </description>
  </property>

  <property>
    <name>hive.stats.autogather</name>
    <value>true|false</value>
    <description>
      Enables automatic gathering of statistics for newly created partitions and materialized views. Statistics autogathering works best for small clusters or environments with limited CPU or network bandwidth. The value should be set to true or false (default: false).
    </description>
  </property>

  <property>
    <name>hive.support.concurrency</name>
    <value>true|false</value>
    <description>
      Enables concurrency support in Hive DDL operations, allowing concurrent creation of objects like tables, partitions, and indexes. Disabling this feature makes these operations sequential and less efficient. The value should be set to true or false (default: true).
    </description>
  </property>

  <property>
    <name>hive.txn.manager</name>
    <value>org.apache.hadoop.hive.ql.lockmgr.DbTxnManager</value>
    <description>
      Configures the transaction manager class used by the Hive transactional locking framework. The default implementation uses database transactions to provide strong consistency guarantees. Other implementations could use other technologies like ZooKeeper or file locks to achieve similar functionality.
    </description>
  </property>
  
```
### 3.1.2 参数调优建议
#### 3.1.2.1 hive.auto.convert.join
该参数控制是否开启自动转换join操作到map join。当设置为true时，Hive会自动判断两个表进行关联的大小，然后决定采用哪种join策略。建议设置成true。
```xml
<property>
    <name>hive.auto.convert.join</name>
    <value>true</value>
</property>
```
#### 3.1.2.2 hive.auto.convert.join.noconditionaltask
该参数控制是否允许自动转换join操作到map join时的条件任务。若设置为true，则任何没有条件的join操作都可能被转换成map join，即使这种转换不一定更好。建议设置为false。
```xml
<property>
    <name>hive.auto.convert.join.noconditionaltask</name>
    <value>false</value>
</property>
```
#### 3.1.2.3 hive.cbo.enable
该参数控制是否开启基于代价的优化器（CBO）。建议设置为true。
```xml
<property>
    <name>hive.cbo.enable</name>
    <value>true</value>
</property>
```
#### 3.1.2.4 hive.limit.optimize.enable
该参数控制是否开启pushdown LIMIT子句优化。若设置为true，Hive会将LIMIT子句推送到子查询，以避免读取过多的输入数据，进而改善查询性能。建议设置为true。
```xml
<property>
    <name>hive.limit.optimize.enable</name>
    <value>true</value>
</property>
```
#### 3.1.2.5 hive.groupby.orderby.position.alias
该参数控制在GROUP BY子句中出现的ORDER BY表达式是否必须以别名的方式引用表列。例如，如果有一个表t1和另一个表t2，t2的列column1与t1的列id存在一对一关系，且要进行如下查询：
```sql
SELECT * FROM t1 INNER JOIN t2 ON t1.id=t2.table1_id ORDER BY t2.column1 DESC;
```
由于默认情况下该选项被禁止，因此优化计划中会包含两次扫描和一次排序操作，前者扫描了t1和t2，后者再次扫描了t2并按t2.column1进行排序。如果启用该选项，优化计划将只扫描t1，而将排序操作移至t1与t2的JOIN之后，这样的话就能以更有效率的方式进行排序操作。注意，开启该选项可能需要修改现有的代码，以依赖于隐式顺序来确定SELECT列表中的表达式。建议设置为false。
```xml
<property>
    <name>hive.groupby.orderby.position.alias</name>
    <value>false</value>
</property>
```
#### 3.1.2.6 hive.groupby.skewindata
该参数控制是否允许倾斜数据被聚合。在默认情况下，组键值差异较大的情况可能导致性能低下。为防止此类情况，Hive可以在运行时将倾斜数据组合在一起。但是，将该标志设置为true可能会导致一些潜在的问题，其中可能出现内存溢出错误，或者由于估计的组大小产生不正确的结果。另外，也会增加额外的处理开销，以便将小型组分散到多个节点上。建议设置为false。
```xml
<property>
    <name>hive.groupby.skewindata</name>
    <value>false</value>
</property>
```
#### 3.1.2.7 hive.map.aggr
该参数控制是否在ORC阅读器（vectorized ORC reader）上应用聚合和列剪枝优化。优化器将过滤直接应用于Aggregation操作符，从而避免不必要的混洗和反序列化操作。该优化仅适用于向量化ORC阅读器。建议设置为true。
```xml
<property>
    <name>hive.map.aggr</name>
    <value>true</value>
</property>
```
#### 3.1.2.8 hive.merge.tezfiles
该参数控制Tez容器文件在查询执行期间是否合并。合并减少了写入磁盘的文件数量，并提高了整体系统性能。建议设置为true。
```xml
<property>
    <name>hive.merge.tezfiles</name>
    <value>true</value>
</property>
```
#### 3.1.2.9 hive.metastore.warehouse.dir
该参数定义了元存储在存储表元数据和分区信息时使用的仓库目录。默认值为/user/hive/warehouse。
```xml
<property>
    <name>hive.metastore.warehouse.dir</name>
    <value>/user/hive/warehouse</value>
</property>
```
#### 3.1.2.10 hive.orcfile.encoding.strategy
该参数指定了用于写入ORC文件的编码策略。SPEED表示对所有列进行简单编码；COMPRESSION表示压缩选定的列；SPEED_OR_COMPACT表示尝试两种策略，依据文件大小而定。建议设置为COMPRESSION。
```xml
<property>
    <name>hive.orcfile.encoding.strategy</name>
    <value>COMPRESSION</value>
</property>
```
#### 3.1.2.11 hive.orc.splits.include.file.footer
该参数控制是否每个ORC切片都包含文件页脚。开启该特性会导致额外的I/O，但确保数据一致性和准确性。建议设置为false。
```xml
<property>
    <name>hive.orc.splits.include.file.footer</name>
    <value>false</value>
</property>
```
#### 3.1.2.12 hive.querylog.location
该参数定义了Hive日志文件的位置，记录了关于Hive查询的各种详细信息。默认值为${system:java.io.tmpdir}/hive.log。
```xml
<property>
    <name>hive.querylog.location</name>
    <value>${system:java.io.tmpdir}/hive.log</value>
</property>
```
#### 3.1.2.13 hive.security.authorization.enabled
该参数控制是否开启Hive安全授权支持。开启此功能后，用户可以被授予特定权限，如创建数据库或访问特定表。建议设置为false。
```xml
<property>
    <name>hive.security.authorization.enabled</name>
    <value>false</value>
</property>
```
#### 3.1.2.14 hive.server2.authentication.kerberos.keytab
该参数指定Kerberos KDC中用于身份认证的keytab文件路径。当HiveServer2配置为Kerberos身份验证模式时，使用该参数。建议设置该参数指向有效的keytab文件。
```xml
<property>
    <name>hive.server2.authentication.kerberos.keytab</name>
    <value>/etc/security/keytabs/hive.service.keytab</value>
</property>
```
#### 3.1.2.15 hive.server2.authentication.kerberos.principal
该参数指定用于Kerberos KDC身份验证的主体名称。当HiveServer2配置为Kerberos身份验证模式时，使用该参数。建议设置该参数与Hive服务器的krb5配置中定义的服务主体匹配。
```xml
<property>
    <name>hive.server2.authentication.kerberos.principal</name>
    <value>hive/_HOST@domain.com</value>
</property>
```
#### 3.1.2.16 hive.server2.authentication
该参数指定客户端连接到HiveServer2时的身份验证类型。当HiveServer2配置为无身份验证、SASL（简单认证和安全层）身份验证或Kerberos身份验证模式时，使用该参数。建议设置为Kerberos身份验证。
```xml
<property>
    <name>hive.server2.authentication</name>
    <value>KERBEROS</value>
</property>
```
#### 3.1.2.17 hive.stats.autogather
该参数控制是否自动收集新创建的分区和物化视图的统计信息。统计信息自动采集最适合小型集群或有限CPU或网络带宽的环境。建议设置为false。
```xml
<property>
    <name>hive.stats.autogather</name>
    <value>false</value>
</property>
```
#### 3.1.2.18 hive.support.concurrency
该参数控制是否在Hive DDL操作中支持并发。如果关闭此功能，DDL操作将变成串行且效率低下。建议设置为true。
```xml
<property>
    <name>hive.support.concurrency</name>
    <value>true</value>
</property>
```
#### 3.1.2.19 hive.txn.manager
该参数配置事务锁管理框架所使用的事务管理类。默认实现使用数据库事务提供强一致性保证。其他实现可能使用ZooKeeper或文件锁等技术达到同样效果。
```xml
<property>
    <name>hive.txn.manager</name>
    <value>org.apache.hadoop.hive.ql.lockmgr.DbTxnManager</value>
</property>
```
# 4. 分区设计方法论
## 4.1 目的
在Hive的存储机制中，表由多个数据文件组成，每个数据文件对应一部分数据，数据文件按照一定规则进行分区。数据文件的分区可以有效地让Hive快速定位、读写、合并这些数据。因此，在建表之前，务必充分考虑分区的设计，以提升查询效率。
## 4.2 原则
- 相同的数据不应该放在不同的分区里，否则会造成性能损失。
- 每个分区应该尽可能保持在一个节点上，以减少网络IO。
- 数据仓库的访问模式应该符合表的访问模式。
- 谨慎选择分区字段，不能太多也不能太少，否则会造成性能损失。
- 删除分区时要慎重，因为删除操作会导致元数据信息丢失，而且可能导致数据丢失。
## 4.3 准备工作
- 设置hive.enforce.bucketing=true，默认情况下，Hive不会自动创建分区，需要手动指定分区字段才能创建分区。
- 检查数据格式、大小。如果数据量较大，可以考虑压缩数据文件。
- 查看表的访问模式。如果表的访问模式频繁涉及到分区范围外的查询，那么可以将不需要的分区放入到外部表中。
- 如果表存在大的分区，可以考虑增大mapreduce.job.reduces的值，以加快查询速度。
## 4.4 步骤
1. 确定分区字段。分区字段应尽可能符合数据特性，不要太分散也不要太集中。如有必要，可以建立多级分区。
2. 根据业务要求对分区进行裁剪。对于查询来说，不需要的所有分区都可以进行裁剪，也可以减少查询的时间和资源开销。
3. 将不需要的分区放入到外部表中。对于那些查询不涉及到的分区，可以将它们放入到外部表中，这样可以加速查询。
4. 清理旧的分区。如果过期的数据已经不再需要，则可以删除对应的分区。删除分区需要慎重，因为删除操作会导致元数据信息丢失，而且可能导致数据丢失。
5. 创建分区表。创建分区表时，分区字段将作为表的分区字段，另外，建议设置numBuckets的数量，可以减少网络IO。
6. 测试查询。测试查询时，不要忘记在WHERE条件中加入分区字段，以限制查询范围。
# 5. 表扫描方式
Hive中的表扫描方式又称为HDFS扫描方式。Hive中的表扫描主要分为静态扫描和动态扫描。静态扫描就是每次执行查询时，都会先遍历所有的块，根据需要读取相应的块，直到读取完毕，然后再返回查询结果。动态扫描是在查询过程中，根据条件直接跳过不满足条件的块，同时在内存中缓存符合条件的块，因此查询速度更快。
## 5.1 静态扫描
静态扫描的实现方式为一个TableScanOperator，该操作符接受TableScanDesc，其描述了扫描方式、表、过滤条件等信息，并调用readNext()方法逐个读取表中块的数据。
```scala
class TableScanOperator extends Operator[TableScanDesc] implements DeserializerOperatorInterface {
  var scanner : Iterator[Object] = null // iterator over current row data or block meta

  override def process(parent: OpContext): Unit =???

  override def initialize(): Unit = {
    super.initialize()
    val tbl = desc.getReferencedTablename().toLowerCase()
    val partSpec = desc.getPartSpec()
    val bucketNum = desc.getBucketNum()

    assert!partSpec.isEmpty(), s"Static scan does not expect dynamic partitions"
    assert bucketNum == -1, s"Static scan does not handle buckets"
    
    ScannerCtxHolder.getCurrentScanner().setTable(tbl)

    val fsPath = new Path(desc.getInputFileFormatClassName())
    val sessionConf = opHandle.getConf()
    val maxRows = desc.getMaxRows()

    val conf = ShimLoader.getHadoopShims().getConfiguration(sessionConf)
    try {
      val inputStream = if (fsPath.toString().startsWith("viewfs://")) {
        FileSystem.get(URI.create(tbl), conf).open(fsPath)
      } else {
        FileSystem.get(fsPath.toUri(), conf).open(fsPath)
      }

      scanner = RowSchemaReader.readData(inputStream, tbl, getOpProperties(), maxRows)

      closeStreamAndLogExceptions(inputStream)

    } catch {
      case e: Exception => throw new HiveException("Unable to instantiate InputFormat object.", e)
    }
  }

  @throws[IOException]
  private def closeStreamAndLogExceptions(inputStream: InputStream): Unit = {
    try {
      inputStream.close()
    } catch {
      case e: IOException => LOG.error("Error closing stream")
    }
  }

  override protected def getNextRowInternal(): Object = {
    if (scanner!= null && scanner.hasNext) {
      return scanner.next()
    } else {
      throw new NoSuchElementException()
    }
  }

 ... // other methods inherited from Operator and DeserializerOperatorInterface  
}
```
## 5.2 动态扫描
Dynamic scans can vary widely, ranging from filtering blocks and caching qualifying blocks to generating a query plan that incorporates range partitioning techniques. However, the general concept remains the same: Dynamic scans keep track of qualifying blocks during query planning, and apply filters lazily at runtime. Here is one way to implement a generic range partition filter:
```scala
object RangeFilterPushDown {
  /**
   * Given a set of predicates, extract those that represent equality comparisons involving a range
   * partition column and rewrite them to compare against endpoints of the range instead. Returns a
   * pair consisting of the remaining predicate conjunction and a sequence of expression trees representing
   * endpoint comparisons. Each tree corresponds to a single interval of the range partition,
   * and satisfies the constraint that left <= x <= right for a particular element x. These constraints
   * ensure that we do not need to read unnecessarily many rows and satisfy arbitrary index conditions.
   */
  def extractRangePredicates(predInfo: PredicateInfo,
                             context: RuleContext,
                             parts: Seq[PartitionDesc],
                             aliases: Set[String]): Option[(PredicateList, Seq[ExpressionTree])] = {
    import org.apache.hadoop.hive.ql.plan.{ExprNodeColumnDesc, PartitionDesc, TableScanDesc}
    predInfo.getExpressions()
         .filter(_.isInstanceOf[FuncCall])
         .collectFirst({
            case FuncCall(
                Equal(Cast(_, _)),
                Literal(_)) =>
              () // TODO(kasiazevs): add handling for String literal types?
            case FuncCall(
                Equal(
                  Cast(_, CharType()),
                  Literal(_))) =>
              () // TODO(kasiazevs): add handling for char literal types?
            case func: FuncCall =>
              val colName = func.getChildren()(0).asInstanceOf[ExprNodeColumnDesc].getColumn()

              parts.find(_.getPartSpec().contains(colName)).flatMap { pDesc =>
                val expr = func.getChildren()(1).asInstanceOf[Literal]

                parseIntervalEndpoints(expr.getValueAs(StringType.INSTANCE)) match {
                  case Some((left, right)) =>
                    // Extract the relevant partition descriptor for the comparison function argument.
                    val argPDesc =
                      parts.find(_.getPartSpec().keys
                       .exists(k =>
                          k.equalsIgnoreCase(colName) ||
                            pDesc.getPartCols().exists(_.getName().equalsIgnoreCase(k))))

                    argPDesc match {
                      case None =>
                        None // Column cannot be found among partition descriptors, ignore this predicate

                      case Some(pDesc) =>
                        val typeName = pDesc.getPartColNames().headOption.flatMap(aliases.lift)
                        val dataType = typeName.flatMap(PrimitiveTypeInfo.getTypeInfoByName)

                        if (dataType.isEmpty) {
                          sys.error(s"Unknown data type $typeName for column $colName")
                        }

                        val lowerBound = ExpressionFactory.makeColumn(argPDesc.getTableName(),
                                                                         argPDesc.getPartSpec().head._2,
                                                                         dataType.get.getTypeName()).eval(null)

                        // Rewrite the original comparison predicate to compare against endpoints.
                        val intervals = buildIntervals(left, right, dataType.get)
                        val cmpExprs = intervals.zipWithIndex.flatMap { case ((lower, upper), i) =>
                          val geLower = ExpressionFactory.greaterThanOrEquals(func.getChildren()(1),
                                                                                Literal.create(lower, dataType.get))

                          val ltUpper = ExpressionFactory.lessThan(func.getChildren()(1),
                                                                    Literal.create(upper, dataType.get))

                          if (i > 0) {
                            List(and(geLower, ltUpper))
                          } else {
                            List(geLower, ltUpper)
                          }
                        }.toList

                        Some((context.parseConjunction(predInfo.getRemaining()),
                              cmpExprs))

                  case None =>
                    None // Invalid interval format, ignore this predicate
                }
              }

          }).flatten

  }

  def parseIntervalEndpoints(intervalStr: String): Option[(Any, Any)] = {
    val regex = """\[(.*?),(.*?)\]""".r
    intervalStr match {
      case regex(start, end) =>
        Some((start, end))

      case _ =>
        None
    }
  }

  /**
   * Build a sequence of contiguous intervals that cover a range [start, end]. Depending on the datatype
   * of the endpoints, the resulting intervals may have different lengths. Additionally, they may overlap.
   */
  def buildIntervals(start: Any, end: Any, dataType: PrimitiveTypeInfo): IndexedSeq[(Any, Any)] = {
    dataType match {
      case _: IntegerTypeInfo | _: DateTypeInfo =>
        val step = start.asInstanceOf[Int] + 1

        Vector.range(start.asInstanceOf[Int], end.asInstanceOf[Int] + step, step)
              .sliding(2)
              .flatMap {
                 case Seq(x, y) =>
                   Some(((x.asInstanceOf[Int], y.asInstanceOf[Int])))

                 case _ =>
                   None

               }.toIndexedSeq

      case _: TimestampTypeInfo =>
        val timestampClass = Class.forName("java.sql.Timestamp")
        val instantStart = timestampClass.cast(start).toInstant
        val instantEnd = timestampClass.cast(end).toInstant
        val durationStep = java.time.Duration.ofMillis(1)
        scala.compat.Platform.collectNonEmpty(instantStart.until(instantEnd, durationStep))
                                            .map{duration =>
                                               (timestampClass.newInstance().setTime(System.currentTimeMillis()-duration.toMillis())) ->
                                                 (timestampClass.newInstance().setTime(System.currentTimeMillis())))}.toIndexedSeq

      case _ =>
        Vector.empty
    }
  }

 ... // other helper methods
}
```