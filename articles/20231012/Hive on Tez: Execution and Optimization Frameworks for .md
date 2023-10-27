
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Apache Hive 是apache基金会下的一个开源项目。主要用于数据仓库(data warehouse)数据查询、分析处理、ETL等。其最主要的特性就是采用SQL语言，执行效率高，数据倾斜优化简单。但是随着Hadoop集群中数据的增长，单机无法支持大量数据分析，故而出现了分布式计算框架MapReduce。然而，MapReduce存在诸多问题，如数据局部性差、节点失败等，这使得大规模数据分析变得复杂和困难。于是，Hive出现了，它可以将HDFS上的文件映射到内存，在本地进行运算，解决数据局部性和数据倾斜的问题。

MapReduce确实很好地解决了数据分析中的一些问题，但当数据量很大时，由于HDFS的存在，仍然面临着很多问题，如冗余存储、调度效率低下、资源利用率低等。因此，Facebook、百度等公司开发了自己的分布式计算框架Tez，并基于此开发了Hive on Tez。在Hive on Tez上运行的Hive查询语句，实际上就是在Tez上运行的MapReduce任务。Hive on Tez通过改进了Hive查询计划生成器、编译器、执行引擎等模块，提升了查询性能。

本文从以下几个方面介绍Hive on Tez的架构、执行原理、优化方法及扩展功能：

1. Hive on Tez的架构
2. 执行原理及优化方法
3. 查询性能优化方法
4. 扩展功能与未来发展方向
# 2.核心概念与联系
## 2.1 Tez概述
Tez是一个由Facebook、百度等公司开发的通用、开源、分布式的可扩展的基于YARN之上的作业调度系统。Tez采用一种DAG（有向无环图）的工作流（workflow），即用户定义的数据处理作业流程，该流程描述了一系列的Map-Reduce作业，并为它们提供数据依赖关系、容错机制、资源管理策略、错误处理策略等。Tez根据DAG中各个作业的依赖关系，自动地将它们按照可靠、有效的方式运行，以实现高吞吐量、低延迟的数据分析。

Tez的关键特性包括：

1. Map-Reduce兼容性：Tez可以与MapReduce运行环境完全兼容，并能够运行现有的MapReduce作业，同时还可以使用用户自己定义的处理函数。Tez可以直接调用MapReduce组件来执行数据转换等，也可以使用自己的组件。
2. DAG作业：Tez采用DAG（有向无环图）来描述数据处理作业流程。每个节点表示一次Map或Reduce过程；DAG中的边表示作业之间的依赖关系。每个作业可以设置不同的资源消耗和优先级，Tez会合理地调度分配资源。
3. 数据本地化：Tez默认情况下，只需要必要的数据就能运行作业。Tez会根据作业使用的磁盘和网络带宽的大小，选择合适的数据存储位置，减少网络IO。同时，它也提供了控制数据访问模式的机制，让用户可以指定数据是随机访问还是顺序访问。
4. 弹性扩缩容：Tez允许动态调整作业的资源需求，以满足不断变化的工作负载需求。Tez还可以通过运行时间的统计信息，自动调整作业的分配资源量和减少作业之间的竞争。
5. 容错与恢复：Tez具有容错机制，可以自动地恢复因计算节点失败、网络分区等原因导致的任务失败。Tez还提供了失败任务重试、容错副本机制等机制，可以在计算节点故障时继续运行作业。
6. 压缩与索引：Tez支持对数据进行压缩与索引，可以显著降低数据的传输量。Tez还可以应用基于列的索引机制，快速定位需要的数据。
7. 安全性：Tez通过Kerberos认证和授权机制来保护数据隐私。而且，它还支持使用具有特定权限的用户，限制对某些数据集的访问权限。

## 2.2 Hive on Tez概述
Hive on Tez是Apache Hive的一项重要特性。它是Apache Hive在Tez之上运行的一个子系统，专门用于对大数据进行分析处理。用户在Hive SQL查询语句中只需添加关键字“SET hive.execution.engine=tez;”，就可以启用Hive on Tez运行环境。Hive on Tez的执行模型和Tez类似，都是采用DAG（有向无环图）来描述作业流程，并利用Task控制单元来协调和调度各种处理过程。

Hive on Tez的特点包括：

1. Hive与Tez的兼容性：Hive on Tez既可以与传统的Hive语法兼容，也可以结合Tez的一些独特特性。例如，Hive on Tez可以选择更优秀的执行算法，比如排序合并树（Sorted Merge Tree）。另外，Hive on Tez还可以结合更丰富的函数库，通过自定义的UDF（user defined function）接口实现更多的分析功能。
2. Hadoop YARN之上的抽象：Hive on Tez利用了Hadoop YARN作为底层的资源管理平台，可以充分利用YARN的资源管理和调度能力。通过配置，Hive on Tez可以支持运行多个作业并行执行，以提升系统整体的并发度。
3. 更灵活的调度策略：相比于MapReduce的固定调度策略，Hive on Tez可以支持更多灵活的调度策略。例如，可以对不同作业设置优先级、共享资源、抢占资源等。
4. 支持复杂的数据类型：除了Hive内置的基本数据类型外，Hive on Tez还可以支持复杂的数据类型，比如结构化数据类型（struct）、数组（array）、map（dictionary）等。
5. 可以在线查询：Hive on Tez支持在线查询，即可以在Hive表上执行UPDATE/DELETE/INSERT语句，而不需要离线表的克隆或全量导入操作。

## 2.3 Hive on Tez的架构
Hive on Tez的架构如下图所示。


1. Resource Manager：RM（Resource Manager）是一个hadoop yarn的组件，它负责资源的划分、调度、分配和追踪。RM接收客户端提交的任务请求，然后为任务分配节点资源。

2. Node Manager：NM（Node Manager）是yarn的一个守护进程，每台服务器部署一个。它负责执行Container，并且监控执行状态。

3. Application Master（AM）：当提交一个任务到yarn之后，RM会把任务分配给某个NM。AM负责启动DAG程序，初始化MR App并为MR App跟踪任务进度。

4. Task Tracker：TT（Task Tracker）是yarn的一个守护进程，它负责处理来自Application Master的任务请求。

5. Container：Container是yarn中最小的调度单位。它封装了执行作业所需的环境，包括环境变量、资源、命令等。

6. HDFS：Hive on Tez依赖HDFS作为其存储系统。

7. Local File System：当开启MR UDF功能时，Hive on Tez在Local File System中保存用户自定义函数相关文件。

8. Queue：队列可以用来对任务进行分类，便于管理和监控。

9. Core Container：Core Container是指一个任务的最小执行单元。

10. Query Plan Generator：QPG（Query Plan Generator）是Hive on Tez的查询计划生成器。它会根据查询语句解析出最优执行计划。

11. Optimizer：优化器是Hive on Tez的查询优化器。它会针对查询计划生成的查询计划进行优化。

12. Driver：驱动器（Driver）是一个java程序，它加载配置文件、连接数据库等。

13. User Application：用户应用程序是指提交到Hive on Tez的查询任务。

14. MR Runtime Library：MR运行库是由Hadoop MapReduce官方维护的库。

15. Output Committer：输出提交ter是提交执行结果的最后一道工序。