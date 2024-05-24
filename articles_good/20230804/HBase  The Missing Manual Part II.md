
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2012年底Apache项目发布了HBase，号称“没有中间件的NoSQL数据库”。如今HBase已经成为最流行的开源分布式NoSQL数据库之一。无论对于分布式数据存储、高吞吐量的数据分析处理、实时数据查询等场景都有着广泛应用。本文从HBase架构设计、基本概念、常用命令及配置等方面，逐步剖析HBase内部实现原理并给出代码实例。深入理解HBase对性能的影响和优化措施，并展望其在云计算、数据分析和高可用性方面的应用前景。
         # 2.HBase架构设计
         2.1 HBase架构概览
         HBase是一个分布式列族数据库，支持海量数据的存储和实时分析，它由HDFS作为底层存储，采用伙伴系统(即主/备集群模式)提供高可用服务。HBase分为客户端API、协调器（Region Server）、HDFS、Zookeeper三个主要组件。
         <center>HBase架构图</center>
         2.2 RegionServer
         RegionServer是HBase中用来存储数据的主要模块，负责维护HRegion的生命周期，每个RegionServer上可以启动多个Region。RegionServer基于Hadoop MapReduce框架运行MapReduce任务，利用HFile作为数据存储文件。RegionServer向协调器汇报自身状态信息，同时也从协调器获取元数据信息来维护自己所管理的Region的状态。
         一个RegionServer通常部署在物理机或虚拟机中，有两个工作线程：
        * WAL(Write Ahead Log)，预写日志，用于记录客户端的写入操作；
        * MemStore，内存中的预聚合缓存区，负责将收到的更新写入到磁盘上的HFile中。当MemStore达到一定大小或过期时间后，会被溢写到磁盘上的HFile中。
         除此之外，RegionServer还负责处理来自客户端的读写请求。
         各个RegionServer之间通过zookeeper进行协作，以确保高可用性。
         2.3 Master
         Master负责协调HBase的整体工作流程，包括Regions的分配、监控和故障转移。Master有以下职责：
        * 分配Regions；
        * 执行Region切分、合并操作；
        * 监控RegionServer的健康状态；
        * 对数据进行自动备份和恢复；
        * 处理客户端的请求，例如创建表、删除表等；
        * 提供各种API接口供外部程序调用。
         Master将管理信息存储在Zookeeper中。
         2.4 Client
         Client是访问HBase的入口点，一般指应用程序或者脚本。Client API一般通过Thrift或RESTful接口来访问。每个Client都需要连接至少一个RegionServer才能正常使用HBase。
         2.5 HDFS
         Hadoop Distributed File System (HDFS) 是HBase的底层存储模块，也是Hadoop生态系统中的重要组成部分。HDFS以容错性和高吞吐量著称，能够为HBase提供海量数据存储空间。HDFS的文件存储结构类似于多级目录树，每个节点代表一个块，数据以流式的方式写入到这些块中，并可被复制到不同服务器上。HDFS提供数据高可用性，可以通过Hadoop HA (High Availability，HA)模式部署多个NameNode以提升HBase的高可用性。
         通过HDFS，HBase可以支持海量数据的存储和实时分析，并且不依赖任何中心化服务。
         # 3.HBase概念和术语
         3.1 Namespace
         在HBase中，Namespace相当于关系型数据库中的数据库，不同Namespace下同名的表之间的关系互相独立。一个集群可以有多个Namespace。如果没有指定Namespace，则默认使用default Namespace。
         3.2 Table
         Table是在HBase中用来保存数据的逻辑结构，相当于关系型数据库中的表格。Table由两部分组成：Row Key和Column Family+Qualifier。其中，Row Key定义了每行数据的唯一标识符；Column Family定义了一系列相关列簇，所有列簇下的列共享相同的前缀；Qualifier用于限定某个Column的子集，主要用于排序和范围查询。
         3.3 Column Family
         列簇是一种逻辑概念，对应于关系型数据库中表的一个字段。列簇中的所有列共享相同的前缀。列簇通常按照业务逻辑对行进行分类，不同的列簇具有不同的访问控制权限。
         3.4 Row Key
         行键是每行数据的唯一标识符，相当于关系型数据库中的主键。Row Key可以选择单列或者多列作为主键，也可以自定义组合方式。
         3.5 Cell
         Cell是最小的存取单位，其值可以是字节数组、字符串、整数或者其它类型。Cell的标识符由Row Key、列簇、列限定符和时间戳组成。时间戳用于版本化和防止数据覆盖。
         3.6 TimeToLive(TTL)
         TTL可以用来设置数据过期的时间。设置之后，HBase会自动清除过期的数据。
         3.7 Versioning
         版本控制可以跟踪表中数据的历史变更，并保留每一个版本的快照。
         3.8 Multi-Tenancy
         多租户是一种软件架构设计方法，允许多个用户共用同一套HBase集群，但是隔离彼此的数据。这种架构方案适用于一些大型公司或政府部门，它们需要让不同业务部门共享数据而不干扰彼此的工作。
         3.9 Filter
         Filter可以帮助减少扫描的数据量。可以基于列值、Row Key、时间戳或者只读取最新版本数据等条件过滤数据。
         3.10 Scan
         Scanning是检索数据时的过程。Scanning可以从指定的起始Row Key开始，向前或向后遍历所有的Row Key直到结束位置。
         3.11 Increment/Append
         Increment是一种原子操作，可以为某一列增加指定的值；Append是一种原子操作，可以向某一列追加一个值。
         3.12 Secondary Index
         二级索引是指为表添加另外一张索引表，使得检索某种特定值的记录更快速。HBase支持本地Secondary Index和全局Secondary Index两种形式。
         3.13 Coprocessors
         coprocessor是一种编程模型，它可以在RegionServer上执行特定任务，以加速查询或执行复杂的功能。
         3.14 Regions
         Region是HBase的基本存储单元，它的大小由block size和region server个数决定。一个Region包含多个Store。Store又划分为多个CompactedFiles。
         3.15 Compacting 和 Merging
         当一个Region中的数据量太大时，会导致性能下降。为了解决这个问题，HBase引入了Compaction和Merging机制。Compaction是当对一个Region执行一次Minor Compact操作时，HBase把该Region的所有数据文件合并成一个小文件。Minor Compact操作不会影响正在写入的事务，不会阻塞其他写入操作。Major Compact操作会阻塞其他写入操作，释放掉compacted文件的空间。Merging是当对一个Region执行一次Major Compact操作时，HBase将两个相邻的Region合并成一个Region。Major Compact操作会影响正在写入的事务，可能会造成短暂的延迟。
         3.16 SSTable
         SSTable是HBase的持久化存储格式。它是一个列族的集合。它可以包含多个SSTable文件。在读写时，SSTable文件被顺序地读取。SSTable以Bloom Filter和索引来优化读操作。
         3.17 Catalog
         目录是HBase中元数据信息的存储模块，存储了表、Region及Store的元数据信息。它是HBase的核心组件，负责维护整个系统的元数据信息。
         3.18 Metrics
         Metrics是监控指标的集合。它包含HBase集群的整体状态信息，比如集群的负载、集群的可用空间等。
         3.19 Backup and Restore
         数据备份可以帮助用户进行灾难恢复，它可以创建一个副本，并将其拷贝到另一个位置，以防止数据丢失或损坏。数据恢复也可以在发生灾难时恢复HBase集群。Backup和Restore是HBase常用的管理工具。
         3.20 Load Balancing
         负载均衡是HBase集群的重要特性之一，它保证数据均匀分布在所有RegionServer上。
         3.21 CheckAndPut
         CheckAndPut操作是一种原子操作，首先通过rowkey和columnFamily定位指定的cell，然后与当前cell进行比较，如果一致，则执行put操作，否则什么都不做。
         3.22 Consistency
         HBase有两种一致性策略：
        * Strong Consistency：客户端在读取或修改数据时，总是读取到最近写入的数据。这是一种保守的一致性策略，由于可能存在网络分区或机器故障，因此读取到的数据可能是过期或陈旧的。
        * Eventual Consistency：客户端在读取或修改数据时，读取到的结果是上次更新后最近的一份数据。这是一种最终一致性策略，读取到的数据随着时间的推移逐渐变新。
         根据应用场景的要求，选择不同的一致性策略可以提高HBase的性能和可用性。
         # 4.HBase常用命令及配置
         4.1 Basic commands
         查看HBase版本：hbase version

         启动HBase进程：start-hbase.sh

         停止HBase进程：stop-hbase.sh

         重启HBase进程：restart-hbase.sh

         检查HBase进程是否启动成功：jps

         4.2 Create table command
         创建一个名为mytable的Table：hbase shell "create'mytable', {NAME => 'cf', VERSIONS=> 1}"

         4.3 List tables command
         显示HBase中的所有表：hbase shell "list"

         4.4 Describe a table command
         描述一个表mytable的详细信息：hbase shell "describe'mytable'"

         4.5 Delete a table command
         删除一个名为mytable的Table：hbase shell "disable'mytable'; drop'mytable'"

         4.6 Show regions in a table command
         显示一个表中包含的regions：hbase shell "show'mytable', regions"

         4.7 Enable/Disable a table command
         启用一个名为mytable的Table：hbase shell "enable'mytable'"

         禁用一个名为mytable的Table：hbase shell "disable'mytable'"

         4.8 Truncate a table command
         清空一个名为mytable的Table中的数据：hbase shell "truncate'mytable'"

         4.9 Add column command
         添加一列到一个表中：hbase shell "alter'mytable', NAME=>'new_col',TYPE=>'string'"

         4.10 Delete column command
         删除一个表中的一列：hbase shell "alter'mytable', NAME=>'old_col',DELETE COLUMN"

         4.11 Modify table command
         修改一个表的属性：hbase shell "alter'mytable',{NAME => 'COMPRESSION', VALUE => 'GZ'}"

         4.12 Insert data into the table command
         将数据插入一个表中：hbase shell "put'mytable','row1','cf:a','value1'"

         4.13 Get data from the table command
         从一个表中获取数据：hbase shell "get'mytable','row1'"

         获取指定列的数据：hbase shell "get'mytable','row1',{COLUMN=>'cf:b'}"

         4.14 Scan the table command
         扫描一个表中的数据：hbase shell "scan'mytable'"

         4.15 Perform a batch operation on the table command
         对一个表执行批量操作：hbase shell "batch '<command list>'"

         命令列表中每条命令的格式如下："<operation> '<tablename>',{<columns>,...}=>{'<operator>',<'operand'>},[<conditions>]"。例如：

        ```hbase
        put'mytable','row1','cf:c','value1',NOW(),v=5
        delete'mytable','row1','cf:d',v='value5'
        increment'mytable','row2','cf:e',1
        append'mytable','row1','cf:f','append value'
        checkandput'mytable','row1','cf:g','oldvalue','newvalue'
        scan'mytable',{COLUMNS=>['cf:a'],FILTER=>"ValueFilter(=,'binary:value')",LIMIT=>10}
        ```

         更多命令参考官方文档：https://hbase.apache.org/book.html#shell

         4.16 Configure ZooKeeper quorum
         配置Zookeeper的quorum参数：hbase-site.xml

         ```xml
         <property>
             <name>hbase.zookeeper.quorum</name>
             <value>zkserver1,zkserver2,zkserver3</value>
         </property>
         ```

         4.17 Configure file system for HBase root directory
         配置HBase的根目录：hbase-site.xml

         ```xml
         <property>
             <name>hbase.rootdir</name>
             <value>hdfs://namenode:8020/hbase</value>
         </property>
         ```

         4.18 Configure master principal and keytab location for security features
         配置主节点的kerberos认证信息：hbase-site.xml

         ```xml
         <property>
             <name>hbase.security.authentication</name>
             <value>kerberos</value>
         </property>
         <property>
             <name>hbase.master.kerberos.principal</name>
             <value>hbase/_HOST@EXAMPLE.COM</value>
         </property>
         <property>
             <name>hbase.regionserver.kerberos.principal</name>
             <value>hbase/_HOST@EXAMPLE.COM</value>
         </property>
         <property>
             <name>hbase.regionserver.keytab.file</name>
             <value>/etc/security/keytabs/hbase.service.keytab</value>
         </property>
         ```

         # 5.代码实例和具体应用案例
         5.1 Demo 1
         在HBase中创建一个名为user_info的表，表中包含三列族："basic","location"和"contact"，分别存储用户的基本信息、地理位置信息和联系方式，并分别设置压缩选项为NONE、SNAPPY、LZO。然后插入若干样例数据：

         ```hbase
         create 'user_info',{NAME=>'basic',VERSIONS=>1},{NAME=>'location',COMPRESSION=>'lzo',VERSIONS=>1},{NAME=>'contact',COMPRESSION=>'snappy',VERSIONS=>1};

         put 'user_info','user1','basic:name','Alice',NOW();
         put 'user_info','user1','basic:age','25',NOW();
         put 'user_info','user1','basic:gender','female',NOW();

         put 'user_info','user1','location:country','USA',NOW();
         put 'user_info','user1','location:state','California',NOW();
         put 'user_info','user1','location:city','San Francisco',NOW();

         put 'user_info','user1','contact:email','alice@example.com',NOW();
         put 'user_info','user1','contact:phone','123-456-7890',NOW();

         put 'user_info','user2','basic:name','Bob',NOW();
         put 'user_info','user2','basic:age','30',NOW();
         put 'user_info','user2','basic:gender','male',NOW();

         put 'user_info','user2','location:country','China',NOW();
         put 'user_info','user2','location:province','Guangdong',NOW();
         put 'user_info','user2','location:city','Shenzhen',NOW();

         put 'user_info','user2','contact:email','bob@example.com',NOW();
         put 'user_info','user2','contact:phone','987-654-3210',NOW();

         put 'user_info','user3','basic:name','Cathy',NOW();
         put 'user_info','user3','basic:age','28',NOW();
         put 'user_info','user3','basic:gender','female',NOW();

         put 'user_info','user3','location:country','Canada',NOW();
         put 'user_info','user3','location:province','Quebec',NOW();
         put 'user_info','user3','location:city','Montreal',NOW();

         put 'user_info','user3','contact:email','cathy@example.com',NOW();
         put 'user_info','user3','contact:phone','555-123-4567',NOW();
         ```

         5.2 Demo 2
         在HBase中创建一个名为event_logs的表，表中包含两个列族："events"和"meta"，分别存储事件信息和元数据信息。首先插入若干样例数据：

         ```hbase
         create 'event_logs',{NAME=>'events',COMPRESSION=>'none',VERSIONS=>1},{NAME=>'meta',COMPRESSION=>'snappy',VERSIONS=>1};

         put 'event_logs','20190801-001','events:id','event1',NOW();
         put 'event_logs','20190801-001','events:type','login',NOW();
         put 'event_logs','20190801-001','events:ipaddr','127.0.0.1',NOW();

         put 'event_logs','20190801-001','meta:client_version','1.0',NOW();
         put 'event_logs','20190801-001','meta:username','Alice Smith',NOW();
         put 'event_logs','20190801-001','meta:timestamp','August 1st, 2019 00:00:00 UTC',NOW();

         put 'event_logs','20190802-002','events:id','event2',NOW();
         put 'event_logs','20190802-002','events:type','logout',NOW();
         put 'event_logs','20190802-002','events:ipaddr','192.168.1.1',NOW();

         put 'event_logs','20190802-002','meta:client_version','1.0',NOW();
         put 'event_logs','20190802-002','meta:username','Bob Johnson',NOW();
         put 'event_logs','20190802-002','meta:timestamp','August 2nd, 2019 00:00:00 UTC',NOW();

         put 'event_logs','20190803-003','events:id','event3',NOW();
         put 'event_logs','20190803-003','events:type','upload',NOW();
         put 'event_logs','20190803-003','events:ipaddr','192.168.1.2',NOW();

         put 'event_logs','20190803-003','meta:client_version','1.0',NOW();
         put 'event_logs','20190803-003','meta:username','Catherine Brown',NOW();
         put 'event_logs','20190803-003','meta:timestamp','August 3rd, 2019 00:00:00 UTC',NOW();
         ```

         查询所有数据：

         ```hbase
         scan 'event_logs'
         ```

         返回的结果类似：

         ```
         ROW                                CELL
          20190801-001                  events:id               event1              timestamp=1564782400976,1
        ...
                                    meta:client_version    1.0                 timestamp=1564782400976,1
                                   meta:timestamp        August 1st, 2019 00:00:00 UTC    timestamp=1564782400976,1
                                  meta:username         Alice Smith          timestamp=1564782400976,1
                                          TIMESTAMP          1                     


         ROW                                CELL
          20190802-002                  events:id               event2              timestamp=1564782400976,1
        ...
                                    meta:client_version    1.0                 timestamp=1564782400976,1
                                   meta:timestamp        August 2nd, 2019 00:00:00 UTC    timestamp=1564782400976,1
                                  meta:username         Bob Johnson          timestamp=1564782400976,1
                                          TIMESTAMP          1                     


         ROW                                CELL
          20190803-003                  events:id               event3              timestamp=1564782400976,1
        ...
                                    meta:client_version    1.0                 timestamp=1564782400976,1
                                   meta:timestamp        August 3rd, 2019 00:00:00 UTC    timestamp=1564782400976,1
                                  meta:username         Catherine Brown      timestamp=1564782400976,1
                                          TIMESTAMP          1                     

         ```

         5.3 使用Java客户端访问HBase
         如果要使用Java语言开发应用，可以使用HBase的客户端库。HBase中提供了多种客户端API，包括Java API、Thrift API和RESTful API。这里以Java API为例，演示如何在Java环境中访问HBase。
         安装HBase客户端API依赖包：

         ```xml
         <dependency>
            <groupId>org.apache.hbase</groupId>
            <artifactId>hbase-client</artifactId>
            <version>${hbase.version}</version>
         </dependency>
         ```

         初始化一个Connection对象：

         ```java
         Configuration conf = HBaseConfiguration.create();
         Connection connection = ConnectionFactory.createConnection(conf);
         ```

         操作示例：

         ```java
         // create a new table with two column families named cf1 and cf2
         Admin admin = connection.getAdmin();
         try {
             HTableDescriptor desc = new HTableDescriptor("tableName");
             desc.addFamily(new HColumnDescriptor("cf1"));
             desc.addFamily(new HColumnDescriptor("cf2"));
             admin.createTable(desc);
         } finally {
             admin.close();
         }

         // write some data to the table using standard Put and Get operations
         Table table = connection.getTable(TableName.valueOf("tableName"));
         try {
             Put p = new Put(Bytes.toBytes("row1"));
             p.addColumn(Bytes.toBytes("cf1"), Bytes.toBytes("qualifier1"), Bytes.toBytes("value1"));
             table.put(p);

             Result result = table.get(new Get(Bytes.toBytes("row1")));
             String value = Bytes.toString(result.getValue(Bytes.toBytes("cf1"), Bytes.toBytes("qualifier1")));
             System.out.println(value);
         } finally {
             table.close();
         }
         ```

         # 6.HBase未来发展方向与挑战
         当前，HBase已经成为 Apache基金会顶级项目，有丰富的社区贡献者、活跃的开发者群体，以及用户的广泛关注。但是，HBase也存在很多潜在问题、不足和可改进地方，下面就对HBase未来的发展方向和挑战进行简单阐述。
         ## 6.1 性能优化
         ### 6.1.1 RegionServer因素
         在HBase中，RegionServer为其所在机器提供数据的存储空间，每个RegionServer节点可以承载多个Region。因此，HBase集群中各个节点的硬件配置非常重要。在RegionServer的性能优化方面，以下几个方面可以考虑：

         * CPU：RegionServer节点应尽可能地采用多核CPU架构，这样可以充分利用多核计算资源。
         * 内存：RegionServer节点应配置足够的内存，以便加载和处理数据。HBase集群中各个节点的内存配置差异应保持在一个较低水平。
         * 文件系统：SSD固态硬盘和高速网络硬盘相比，传统的机械硬盘的随机I/O速度要慢约四倍。因此，建议将文件系统安装在SSD上。
         * 网络带宽：HBase集群中各个节点的网络带宽需要足够，以便支持大规模的数据传输。
         * 集群规模：HBase集群越大，单个RegionServer节点的性能瓶颈就越容易暴露出来。因此，集群规模应该根据业务需求进行扩缩容。

         ### 6.1.2 JVM因素
         HBase是一个基于Java编写的分布式数据库，因此，JVM也成为HBase性能调优的一个关键因素。在JVM配置上，需要注意以下几点：

         * GC配置：由于垃圾收集器的原因，HBase经常需要配置较大的GC堆大小。因此，内存回收频率也需要相应调整。
         * Classpath配置：由于HBase需要加载大量的类，因此，JVM classpath配置很有必要。特别是，HBase依赖的第三方JAR包应精心配置。
         * Native libraries配置：如果使用Native Libraries，应将它们安装在HBase集群的同一个文件系统上，以避免访问冲突。

         ### 6.1.3 HDFS因素
         HBase依赖HDFS作为底层存储，HDFS的优化同样也会影响HBase的性能。HDFS的性能优化方面，以下几方面可以考虑：

         * 数据读写吞吐量：HDFS支持多种数据读写方式，例如数据管道（Pipelines），分块（Block），流式（Streaming）。HBase推荐使用流式写数据，以最大化读写效率。
         * 数据冗余度：HDFS支持多副本机制，既可以避免数据丢失，又可以提高数据可靠性。HBase也支持多副本机制。
         * 备份数量：HDFS支持的备份数量受限于可用空间，因此，如果备份数量过多，将会导致HDFS负载过重。HBase建议将副本数量设定为3。

         ## 6.2 可扩展性
         HBase集群的可扩展性直接影响其性能。在HBase中，可扩展性主要包括以下方面：

         * 数据容量：HBase集群可以横向扩展，通过添加RegionServer节点来增强集群容量。
         * 读写负载：HBase集群可以水平扩展，通过使用分片技术（Splitting）将单个表分布到多个RegionServer节点上。
         * 请求处理能力：HBase集群的读写处理能力也需要随着集群规模的扩大而增加。HBase支持多线程处理请求，因此，处理请求的节点数量也需要相应增多。
         * 并发用户数量：HBase集群的并发用户数量受限于集群内存、网络带宽和Region数量。当集群资源紧张时，可以考虑增加集群规模或减少并发用户数量。

         ## 6.3 高可用性
         HBase集群的高可用性依赖于HDFS和Zookeeper，因此，为了实现HBase集群的高可用性，需满足以下条件：

         * HDFS高可用性：HDFS支持多副本机制，可以通过配置不同的副本策略来实现HDFS的高可用性。
         * Zookeeper高可用性：Zookeeper是分布式协调服务，支持主备模式，可以通过配置多个Zookeeper节点来实现Zookeeper的高可用性。
         * HBase高可用性：HBase的高可用性依赖于HDFS和Zookeeper的高可用性。当HDFS出现故障时，HBase的可用性就会受到影响。可以通过添加Zookeeper节点来提高HBase的可用性。

         ## 6.4 大数据应用
         HBase有助于解决大数据应用的难题。在大数据应用中，HBase可以应用以下技术：

         * 批处理（Batch processing）：HBase支持批处理，即将一批数据批量导入到HBase，减少与HBase交互次数，加快处理速度。
         * 联接（Joins）：HBase支持联接操作，将多个表关联起来，以实现复杂的查询操作。
         * 汇总（Aggregations）：HBase支持聚合操作，对数据的统计和分析。
         * 实时分析（Real time analysis）：HBase支持实时数据分析，对实时变化的业务数据进行实时分析。

         ## 6.5 用户界面和监控
         HBase缺乏统一的用户界面和监控系统，这限制了HBase的应用场景和实践。因此，HBase需要进一步完善自己的用户界面和监控系统，包括以下方面：

         * 图形界面：目前，HBase的图形界面仅支持管理操作，不支持查询操作。因此，图形界面不利于支持业务人员快速了解数据情况，难以满足快速决策的需要。HBase需要支持图形界面，包括查询界面、可视化界面等。
         * 用户角色权限管理：HBase当前仅支持管理员角色权限管理，没有细粒度的用户角色管理。因此，HBase需要进一步完善用户角色权限管理，包括用户角色权限的细粒度划分、授权、审核等。
         * 监控告警：HBase当前仅支持系统级别的监控和告警，缺乏针对用户查询请求和数据操作的监控。HBase需要支持用户级别的监控和告警，包括用户查询请求和数据操作的监控、告警规则设置等。

         ## 6.6 安全性
         HBase作为一个开源的分布式数据库，安全性是其首要关注点。在HBase中，安全性主要包括以下方面：

         * 数据加密：HBase支持数据加密，以防止敏感数据泄漏。
         * 用户权限管理：HBase支持用户角色权限管理，但权限管理粒度仍然有待进一步细化。
         * 访问控制：HBase尚未提供访问控制机制，因此，需要结合其他组件一起使用，例如Kerberos。
         * 流程审计：HBase尚未提供流程审计机制，因此，需要结合其他组件一起使用，例如HDFS Audit Log。