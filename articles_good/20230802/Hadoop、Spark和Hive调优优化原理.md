
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1995年伊藤博文等人提出了MapReduce计算框架，将海量的数据分布式地处理在多台计算机上，并通过分割输入数据集并将其划分为多个任务来并行执行计算，最后合并结果得到整体输出。然而随着互联网和大数据的普及以及处理器性能的提升，当时的技术已经远远超过了当时能够想象的范围。这段时间MapReduce计算框架已经成为一个主流的开源计算框架，包括Hadoop、Pig、Hive、Mahout、Storm等。
         2010年Apache Spark横空出世，基于内存计算框架，是当前最火的大数据分析引擎之一，基于Scala语言实现，是Hadoop MapReduce的替代者。Spark支持Java、Python、R等多种编程语言，其快速的计算速度让其广受欢迎。
         2014年Facebook开发Hive，支持HQL(Hive Query Language)语法查询大数据，成为目前最流行的大数据分析工具。Hive自带数据倾斜解决方案、复杂SQL语句自动优化和分区表支持等功能均十分强大。
         2017年谷歌开发了Google Cloud Dataproc，为用户提供云端运行Hadoop、Spark和Hive集群的能力，解决由于Hadoop单点故障导致业务无法正常运行的问题。
         2018年，微软发布了Azure HDInsight，作为服务于企业的Hadoop、Spark、Hive集群管理平台，帮助客户轻松创建、删除和配置计算资源，提升大数据工作负载的效率。此外，还有更多的平台提供大数据服务，例如Cloudera、Databricks、Amazon EMR等。
         2020年，蚂蚁集团宣布开源了达摩院开源项目Druid，其是一个开源分布式时间序列数据库，能够满足海量数据、高速查询需求。该项目拥有来自世界各地的开发者提交的代码贡献，是一个拥有活跃社区和庞大的生态系统的优秀开源项目。
         2021年，阿里巴巴宣布开源了Nebula Graph，其是一个兼具图数据库和分布式键值存储的开源图数据库，能够为业务应用提供高效可靠的图数据库服务。与另一个名为Nebula的分布式数据库不同，Nebula Graph是一个分布式的面向属性的图数据库，在数据模型和接口设计上都与传统的关系型数据库有所不同。Nebula Graph的社区由来自阿里巴巴、腾讯、华为、京东、美团、360等大公司的工程师共同组建。
         2022年，百度发布了龙芯向量数据库Lindorm，是一个基于LSM树索引的存储引擎，能够有效地存储和处理海量的实时或离线日志数据。同时，百度也推出了可插拔的分析引擎，包括数据查询语言BaikalDB和超融合查询语言Fesql。
         本篇文章将从Hadoop、Spark和Hive三个框架的调优优化原理出发，详细阐述Hadoop、Spark和Hive各自的特点、优缺点、适用场景，以及它们的调优优化方法和具体操作步骤。同时，我们还会根据实际案例分析Hadoop、Spark和Hive的调优优化方法和注意事项，给出实践建议。
         
         # 2. Hadoop基本概念与特点
         ## 2.1 Hadoop概览
         Hadoop是由Apache基金会开源的一款开源框架，它是一个基于HDFS的分布式文件系统、MapReduce计算框架和HQL(Hive QL)查询语言的组合体，具有高容错性、高扩展性、可靠性和可靠性。
         Hadoop提供了一个完整的大数据分析流程，包括数据采集、存储、清洗、转换、分析、挖掘和存储。其中，MapReduce是Hadoop的计算引擎，用于对大规模数据进行并行计算。HQL则是一个声明式查询语言，用于对海量数据进行结构化查询、分析和挖掘。
         ### 2.1.1 Hadoop体系结构图
         Hadoop体系结构如图1所示：
         
            HDFS (Hadoop Distributed File System)
            DataNodes (存储节点)   
              ↓                 
            NameNode (命名节点) 
              ↓    
            JobTracker (作业跟踪器)   
            TaskTrackers (任务跟踪器)       
                           
            YARN (Yet Another Resource Negotiator)
               
                 
        * HDFS: Hadoop Distributed File System (HDFS)，它是一个分布式文件系统，提供高容错性、高吞吐量和可靠性。HDFS被设计成可以部署在廉价的商用服务器上，为hadoop集群提供了共享的文件存储和名字空间。HDFS有很好的扩展性、灵活性和容错性，并且能够提供高吞吐量，因此在大数据处理中得到广泛应用。   
        * DataNodes: HDFS中的存储节点，每个NameNode下会有一个或多个DataNodes。DataNode用来存放HDFS上的数据块，它存储着HDFS的文件数据，同时它还充当客户端和其他DataNode之间的通信代理，DataNode中的主要组件是datanode进程。  
        * NameNode: HDFS的主服务器，负责管理文件系统的名称空间和数据块映射，并接收客户端读写文件的请求。NameNode中的主要组件是namenode进程。   
        * JobTracker: Hadoop作业跟踪器，它是Hadoop MapReduce的主控节点。JobTracker跟踪整个MapReduce程序的进度，并协调tasktrackers来分配MapReduce任务。JobTracker中的主要组件是jobtracker进程。   
        * TaskTrackers: 这些节点为集群中的worker节点上执行map/reduce任务。每个任务跟踪器负责执行MapTask或者ReduceTask。tasktracker中的主要组件是tasktracker进程。   
        * YARN（Yet Another Resource Negotiator）： Hadoop Yet another resource Negotiator（YARN），它是一个新的资源管理框架，它提供了MapReduce的资源抽象，统一计算和存储资源，并简化了集群的操作。yarn是hadoop2.0引入的新版本的资源管理框架。yarn与hdfs、mapreduce、hbase等功能相辅相成，为hadoop提供更好的可扩展性和稳定性。   
       
        在hadoop体系结构图中，HDFS负责数据存储、调度、处理、容错；YARN负责资源管理、作业调度和资源调度；而JobTracker和TaskTrackers则用于计算资源的管理。这三者相互配合，共同完成Hadoop的功能。     
          
       ## 2.2 MapReduce基本概念与特点
       ### 2.2.1 MapReduce概述
       MapReduce是一种基于Hadoop框架的编程模型，它提供了一种用于大数据集合运算的编程范式。通过定义一系列的Map函数和Reduce函数，可以将海量的数据集进行切分、分片和排序，然后利用多台计算机集群并行计算。最终结果再合并产生完整的结果。
     
       ### 2.2.2 Map函数
       Map函数是对输入数据的处理过程，它的输入是一组键值对(key-value pair)，输出也是一组键值对，但是这种输出的形式取决于开发者编写的Reducer。一个典型的Map函数示例如下：
     
         map(k1, v1) -> list(k2, v2), list(k3, v3),...
         
     
       可以看到，Map函数接受一组键值对作为输入，返回一个列表。这个列表通常包含一组键值对，即键值对经过一些变换后形成的结果。比如，统计文本中单词出现的次数，就可以把文本看做输入，把单词和对应出现的次数看做输出，每一次出现就增加一条键值对到输出列表。这种方式可以把一个输入数据集按照一定的规则切分成若干个小数据集，然后分别对这些小数据集并行地进行处理。
     
       ### 2.2.3 Shuffle操作
       在Map阶段结束后，数据经过shuffle操作，这一步又称为“shuffle and sort”，其目的是将数据集随机划分成多份，在多台计算机上进行并行处理，最终得到完整的结果。
     
       ### 2.2.4 Reduce函数
       Reducer函数是对中间数据的处理过程，它的输入是一个key-value对列表，输出也是一个键值对，一般情况下，Reducer只接受一组相同key的键值对。一个典型的Reducer函数示例如下：
     
         reduce(k2, list(v2, v3,...)) -> k3, v4
      
     
       可以看到，Reducer函数接受一个key-value对列表作为输入，然后把它们合并成一个输出键值对。Reducer通常以key-value对的方式处理中间结果，处理完所有的中间结果之后，得到最终的结果。
     
       ### 2.2.5 数据类型
       Hadoop MapReduce支持几种不同的数据类型：
     
         * Key-Value Pairs: Hadoop MapReduce默认处理的是Key-Value Pairs，例如，我们可以把URL作为key，页面大小作为value，把文件路径作为key，文件内容作为value，甚至把网页文本作为value。
         * Text: Hadoop MapReduce还支持处理文本数据，可以把一段文本作为输入，每行文本作为value，把词汇表作为value，甚至可以把网页源码作为value。
         * Images: 由于图像数据通常非常大，Hadoop MapReduce支持对图像数据进行处理，把图像作为value，进行图像搜索、图像分类等任务。
         * Big Data: Hadoop MapReduce可以处理大数据，把数据先压缩、分区，然后在分区间进行分布式处理，最后再把结果汇总。
     
       ### 2.2.6 MapReduce框架的特点
       #### 2.2.6.1 分布式并行计算
       MapReduce是一种分布式并行计算的框架，它提供了一种简单且高效的方法来处理大量的数据集。MapReduce框架将海量的数据集切分成较小的数据集，分别运行在不同的机器上，并在这么多计算机上并行计算，最终将结果汇总得出完整的结果。
       #### 2.2.6.2 容错机制
       Hadoop MapReduce拥有容错机制，它能够自动检测并恢复错误，确保计算结果的准确性。在发生节点故障、网络连接失败等异常情况时，它能够自动重试，重新启动任务，并继续完成计算任务。
       #### 2.2.6.3 可扩展性
       Hadoop MapReduce框架是一个可扩展的系统，它可以在不影响已有数据的情况下，动态添加新节点来增加集群的计算能力。它可以根据需要调整集群的规模，以应对数据量的增长。
       #### 2.2.6.4 编程模型的便利性
       MapReduce是一种简单的编程模型，因为它只要求开发者编写Map和Reduce函数。它的编程接口比较简单，学习起来比较容易。另外，Hadoop MapReduce拥有丰富的库函数，使得开发者可以使用熟悉的编程语言，如Java、Python、C++等，快速编写MapReduce程序。
       #### 2.2.6.5 智能的数据分割
       MapReduce能够智能地将数据集切分成较小的数据集，然后在不同机器上并行处理。它能够识别热点数据，并优先分配计算资源。对于多次执行同样的计算任务，它会缓存中间结果，避免重复计算，节省时间。
   
       ## 2.3 Hive基本概念与特点
       ### 2.3.1 Hive概述
       Apache Hive是开源的分布式数据仓库基础设施，它是一个数据库构建工具，能够将结构化的数据文件映射为一张表格，并提供 SQL 查询功能。Hive 的目标是在 Hadoop 上构建一个开源数据仓库，可以类似 Presto、Impala 一样运行查询。 Hive 通过将数据加载到 Hadoop 文件系统 (HDFS) 中，将结构化的数据转化为关联的表，并提供类 SQL 的查询语言。
     
       ### 2.3.2 Hive架构
       Hive 有两个主要组件：Metastore 和 Execution Engine。
         
         Metastore：它是一个独立的元数据存储。它存储关于 Hive 对象（表、视图、存储过程等）的信息，并包含有关 Hive 数据的元数据。 
         Execution Engine：它是执行 SQL 查询的组件。它接受 SQL 命令，将其翻译成 MapReduce 或者 Tez 任务，并运行这些任务来处理数据。
     
       下图显示了 Hive 的架构：
       
           +-----+       +------------+       +--------------+       +-----------+
           | User| --Queries-> |   Parser   |--Parses--> | Optimizer  | -Translates to MapReduce jobs or Tez tasks-> |Execution Engine|
           +-----+       +------------+       +--------------+       +-----------+
                   ^                                    |
                   |                                    |
                 Physical                             Physical
                         Database                           Cluster
                          .                              .
                          .                              .
                          HDFS                             Yarn
                              .                             .
                             Data                         Docker Containers
                                   .                           .
                                  Tables                         Nodes
                                       .                      .
                                      Views                       
                                Metastore                   
                                           .
                                        Other Services
        
     
       可以看到，Hive 有两层架构：一层是 Hive 用户层，负责编写 SQL 查询。另一层是 Hive 执行层，负责把 SQL 查询翻译成执行任务。
      
       ### 2.3.3 Hive的特点
       #### 2.3.3.1 SQL语法兼容性
       Hive 支持类 SQL 的语法，包括 DDL（Data Definition Language）、DML（Data Manipulation Language）、DCL（Data Control Language）。因此，开发人员可以使用熟悉的 SQL 语句来访问 Hive 中的数据，并对 Hive 中的数据执行各种操作。
       #### 2.3.3.2 自动的数据分割
       Hive 将数据存储在 HDFS 中，并自动地对数据进行分区和切片。它能够识别热点数据，并优先分配计算资源。对于多次执行同样的查询任务，它会缓存中间结果，避免重复计算，节省时间。
       #### 2.3.3.3 支持复杂的事务处理
       Hive 支持复杂的事务处理，包括 ACID（Atomicity、Consistency、Isolation、Durability）特性。
       #### 2.3.3.4 高级查询优化
       Hive 提供了一些高级查询优化器，它能够生成针对特定查询的最佳执行计划。
       #### 2.3.3.5 统一的框架和生态系统
       Hive 使用 Hadoop 来统一其框架和生态系统，包括 Hadoop、Spark、Impala、Presto 等。
      
    # 3. Hadoop、Spark、Hive调优优化原理
       # 3.1 Hadoop调优原理
       ## 3.1.1 Hadoop的调优步骤
       1. 硬件配置：Hadoop的性能和资源消耗都依赖于底层机器的配置，配置好机器可以提升Hadoop集群的整体性能。
       2. 安装配置：安装好配置好的硬件之后，下一步就是配置Hadoop。首先检查是否正确安装了Hadoop。然后，确认设置了JAVA_HOME环境变量，并设置了HADOOP_CONF_DIR环境变量。
       3. 配置参数：修改配置文件hadoop-env.sh和hdfs-site.xml。hadoop-env.sh配置文件用于设置JVM堆栈大小和其它环境变量，hdfs-site.xml用于配置HDFS的相关参数，例如数据目录、副本数量等。
       4. 设置优化：为集群设置垃圾收集器，启用压缩，调整内存使用，设置队列容量，调整磁盘使用策略。
       5. 测试集群：测试集群的运行状态，比如查看进程，查看监控信息，查看日志等。
       6. 监控集群：监控集群的运行状态，比如查看CPU、内存、网络等使用情况，检查磁盘使用情况，查看磁盘IO压力等。
       7. 维护操作：进行Hadoop集群的维护操作，如备份数据，清除日志等。
       ## 3.1.2 Hadoop的参数设置
       ### 3.1.2.1 修改core-site.xml文件
       1. 配置HDFS的名称节点地址。
           <property>
             <name>fs.defaultFS</name>
             <value>hdfs://master:9000</value>
           </property>
       2. 配置文件系统的类型。
           <property>
             <name>fs.default.name</name>
             <value>hdfs://master:9000</value>
           </property>
       3. 配置HDFS的名称节点端口号。
           <property>
             <name>dfs.namenode.rpc-address</name>
             <value>master:9000</value>
           </property>
       4. 配置HDFS的客户端端口号。
           <property>
             <name>dfs.client.use.datanode.hostname</name>
             <value>true</value>
           </property>
       5. 配置HDFS访问的用户名。
           <property>
             <name>hadoop.tmp.dir</name>
             <value>/user/$USER</value>
           </property>
       6. 配置文件上传下载的缓冲区大小。
           <property>
             <name>io.file.buffer.size</name>
             <value>131072</value>
           </property>
       7. 配置安全模式。
           <property>
             <name>dfs.permissions</name>
             <value>false</value>
           </property>
       8. 配置HDFS的客户端身份认证。
           <property>
             <name>dfs.web.authentication.kerberos.principal</name>
             <value>hdfs/_HOST@EXAMPLE.COM</value>
           </property>
           <property>
             <name>dfs.web.authentication.kerberos.keytab</name>
             <value>/path/to/hdfs.keytab</value>
           </property>
       ### 3.1.2.2 修改hdfs-site.xml文件
       1. 配置副本数量。
           <property>
             <name>dfs.replication</name>
             <value>3</value>
           </property>
       2. 配置数据存储位置。
           <property>
             <name>dfs.data.dir</name>
             <value>file:///data/1,/data/2,/data/3</value>
           </property>
       3. 配置数据传输缓冲区大小。
           <property>
             <name>dfs.block.size</name>
             <value>134217728</value>
           </property>
       4. 配置HDFS校验和类型。
           <property>
             <name>dfs.checksum.type</name>
             <value>CRC32C</value>
           </property>
       5. 配置块复制因子。
           <property>
             <name>dfs.replication.max-streams</name>
             <value>1000</value>
           </property>
       6. 配置写数据的最小数量。
           <property>
             <name>dfs.write.packet.size</name>
             <value>13565952</value>
           </property>
       7. 配置最大写入线程数。
           <property>
             <name>dfs.client.write.threadpool.size</name>
             <value>10</value>
           </property>
       8. 配置数据块的大小。
           <property>
             <name>dfs.blocksize</name>
             <value>134217728</value>
           </property>
       9. 配置RPC通讯超时时间。
           <property>
             <name>ipc.server.read.timeout</name>
             <value>180000</value>
           </property>
       10. 配置Hadoop运行日志目录。
           <property>
             <name>hadoop.log.dir</name>
             <value>/var/log/hadoop/</value>
           </property>
       11. 配置Hadoop临时文件目录。
           <property>
             <name>hadoop.tmp.dir</name>
             <value>/var/run/hadoop/</value>
           </property>
       ### 3.1.2.3 修改mapred-site.xml文件
       1. 配置作业历史记录保存位置。
           <property>
             <name>mapreduce.jobhistory.location</name>
             <value>/var/log/hadoop-mapreduce/history</value>
           </property>
       2. 配置作业提交脚本目录。
           <property>
             <name>mapreduce.map.executable</name>
             <value>/usr/bin/hadoop</value>
           </property>
           <property>
             <name>mapreduce.reduce.executable</name>
             <value>/usr/bin/hadoop</value>
           </property>
       3. 配置作业运行最大内存。
           <property>
             <name>mapreduce.map.memory.mb</name>
             <value>1024</value>
           </property>
           <property>
             <name>mapreduce.reduce.memory.mb</name>
             <value>2048</value>
           </property>
       4. 配置作业运行的虚拟机的最大内存。
           <property>
             <name>mapreduce.map.java.opts</name>
             <value>-Xmx768m</value>
           </property>
           <property>
             <name>mapreduce.reduce.java.opts</name>
             <value>-Xmx1024m</value>
           </property>
       5. 配置MapReduce任务运行的线程池个数。
           <property>
             <name>mapreduce.map.parallelism</name>
             <value>20</value>
           </property>
           <property>
             <name>mapreduce.reduce.parallelism</name>
             <value>20</value>
           </property>
       ### 3.1.2.4 修改yarn-site.xml文件
       1. 配置ResourceManager的地址和端口。
           <property>
             <name>yarn.resourcemanager.hostname</name>
             <value>rm</value>
           </property>
           <property>
             <name>yarn.resourcemanager.webapp.address</name>
             <value>rm:8088</value>
           </property>
       2. 配置NodeManager的运行主机名。
           <property>
             <name>yarn.nodemanager.aux-services</name>
             <value>mapreduce_shuffle</value>
           </property>
           <property>
             <name>yarn.nodemanager.aux-services.mapreduce_shuffle.class</name>
             <value>org.apache.hadoop.mapred.ShuffleHandler</value>
           </property>
       3. 配置NodeManager的日志存放位置。
           <property>
             <name>yarn.nodemanager.local-dirs</name>
             <value>/data/yarn/local</value>
           </property>
           <property>
             <name>yarn.nodemanager.log-dirs</name>
             <value>/data/yarn/logs</value>
           </property>
       4. 配置ResourceManager的工作目录。
           <property>
             <name>yarn.nodemanager.remote-app-log-dir</name>
             <value>${yarn.log-aggregation-enable}</value>
           </property>
       5. 配置ResourceManager的Web UI地址。
           <property>
             <name>yarn.web-proxy.address</name>
             <value>localhost:8088</value>
           </property>
       6. 配置历史服务器Web UI地址。
           <property>
             <name>yarn.timeline-service.webapp.address</name>
             <value>localhost:10200</value>
           </property>
       7. 配置作业提交脚本目录。
           <property>
             <name>yarn.app.mapreduce.am.command-opts</name>
             <value>-Xmx1024m</value>
           </property>
       # 3.2 Spark调优原理
       ## 3.2.1 Spark的调优步骤
       1. 确定集群的资源（CPU核数、内存、磁盘）：Spark作业的执行一般会占用大量的资源，所以需要预留足够的资源以保证任务的顺利执行。
       2. 配置JVM参数：Spark JVM参数的设置会影响Spark作业的性能。Spark官网推荐的JVM参数可以通过启动命令加上SparkConf参数的方式进行设置。
       3. 配置Shuffle策略：Shuffle的作用是将任务的输出结果以更快的方式传递给聚合操作。Spark有四种Shuffle策略，默认的Shuffle策略为SortMergeSortStrategy，通过将结果进行合并、排序的方式实现数据的传递。
       4. 配置Storage级别：Spark默认使用MemoryStore作为内存存储器，但如果数据量过大，可能会导致OutOfMemoryError。可以通过Storage级别对Storage进行配置，将数据放在本地磁盘上进行缓存，减少内存压力。
       5. 配置任务持久化：Spark作业在运行期间，会产生多个任务，这些任务的输出结果会被缓存到内存、磁盘等存储设备上。当任务执行失败或被取消时，这些缓存结果就会丢失。通过任务持久化功能，可以将这些结果存入外部存储系统（例如HDFS、S3、HBase等），以防止数据丢失。
       6. 配置反压策略：Spark作业为了避免任务陷入无限等待，会采用反压策略。当某些任务的输出速度远大于其他任务时，Spark会将这些任务的运行时调度进度降低，降低整体性能。可以通过反压策略对反压策略进行配置，比如调整垃圾回收频率，调整RDD持久化配置等。
       7. 模拟并行环境：模拟真实的并行环境，运行一些Spark作业进行测试，观察任务的执行时间和资源消耗，判断Spark作业的调优是否合理。
       8. 调优实施：当实验得出结论时，通过调优参数重新启动Spark作业，验证调优效果。
       ## 3.2.2 Spark的作业调优
       1. 减少输入的数据量：Spark可以运行在超大数据量上，但数据的大小决定了运行速度，所以在数据量较大时，可以通过采样的方式降低数据量。
       2. 启用压缩：Spark的运行速度取决于磁盘I/O，对于大数据来说，可以考虑对数据进行压缩，减少磁盘I/O的开销。
       3. 用Columnar Format存储数据：Spark可以用Columnar Format存储数据，这种格式的存储方式可以减少内存的压力。
       4. 对数据进行分区：对数据进行分区可以将数据集划分为多个分区，这样每个分区只需要处理自己负责的部分数据，加快处理速度。
       5. 设置join的策略：join操作一般会产生大量的网络流量，可以通过优化join策略来减少网络流量。
       6. 设置并行度：并行度的设置会影响作业的执行时间，可以通过尝试不同的并行度，找到最合适的并行度。
       7. 调优实施：当实验得出结论时，通过调优参数重新启动Spark作业，验证调优效果。
       # 3.3 Hive调优原理
       ## 3.3.1 Hive的调优步骤
       1. 检查表的元数据质量：Hive元数据在执行查询时起到了至关重要的作用，如果表的元数据质量不好，可能导致查询执行效率低下。
       2. 设置配置参数：Hive的配置文件hive-site.xml中包含许多参数，可以通过设置不同的参数来优化Hive查询的性能。
       3. 设置分区：如果数据存在于多个文件中，设置分区可以加快查询速度。
       4. 选择索引列：对于大表查询，选择索引列可以加快查询速度。
       5. 设置表的压缩格式：选择相应的压缩格式可以减少磁盘I/O。
       6. 设置Join的策略：选择合适的join策略可以减少网络流量。
       7. 分析慢查询：如果发现某些查询的执行时间长，可以通过分析Hive日志来定位原因。
       8. 其他调优措施：通过检查系统日志、Spark和Yarn的日志、监控集群性能等，可以发现其它调优措施。
       ## 3.3.2 Hive的配置参数
       1. hive.tez.container.size：指定Tez容器的大小。
       2. tez.am.resource.memory.mb：指定ApplicationMaster的JVM堆栈大小。
       3. mapreduce.job.queuename：指定查询队列的名称。
       4. hive.exec.dynamic.partition.mode：指定Hive的动态分区模式。
       5. hive.tez.dynamic.partition.pruning：指定是否开启Hive的动态分区剪枝。
       6. hive.auto.convert.join.noconditionaltask.size：指定条件谓词过滤后的最大数据量。
       7. hive.input.format：指定Hive的输入格式。
       8. hive.merge.mapfiles：指定是否合并MapFile。
       9. hive.compute.query.using.stats：指定是否使用元数据来计算查询。
       10. hive.optimize.sortmergebucketmapjoin：指定是否使用合并排序的BucketMapJoin。
       11. hive.tez.cache.filesystem.metadata：指定是否缓存HDFS的文件元数据。
       12. hive.llap.io.enabled：指定LLAP的IO模式。
       13. hive.server2.tez.default.queues：指定默认的队列。