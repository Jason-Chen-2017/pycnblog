
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Presto 是 Facebook 提供的一个开源分布式数据仓库系统，它提供 SQL 兼容的查询接口并支持高效地执行高并发和复杂的联机分析查询。Presto 可以运行在各种规模的数据仓库上，包括亚秒级查询响应时间的百亿条记录。其架构也使得它能够跨越 Hadoop、Hive 和 Impala 等传统的商业数据仓库进行部署。Presto 的优点主要有以下几点：

1. 高性能: Presto 使用了基于内存计算的执行引擎来快速处理复杂的联机分析查询。这种高性能的表现是通过使用成本更低的内存计算技术实现的。

2. 可扩展性: 在生产环境中，Presto 可以运行在具有多种规格的集群中，从而提供弹性伸缩能力。而且它的设计也允许 Presto 集群中的不同节点之间共享数据，进而提升整体性能。

3. 多样化的数据源支持: Presto 支持来自 Hive、MySQL、PostgreSQL、Oracle、DB2、SQL Server 和 Apache Cassandra 等众多数据源。同时还可以访问外部 Hadoop 文件系统。

4. 标准的 SQL 语法: Presto 通过其标准 SQL 语法和 JDBC/ODBC 驱动支持 ODBC、JDBC、Avatica 和第三方工具的连接。

5. 易于部署和管理: Presto 可以部署在廉价的商用服务器上，而且易于管理和监控。可以通过统一的 Web 用户界面或命令行工具来完成任务。

6. 无缝集成: Presto 提供了一个统一的视图来查看整个数据仓库，并且不需要用户手动指定数据的物理位置。用户只需要向 Presto 查询所需的元数据即可。

7. 自动优化器: Presto 使用基于成本模型和统计信息的优化器自动生成查询计划。该优化器也可以优化数据倾斜和复杂的联机分析查询。

# 2. 概念及术语

## 2.1 PrestoDB 术语表

- Connector：一种与特定数据源的集成插件，可用于连接到各种类型的数据源。Presto 会根据 connector 配置文件自动发现所有可用的数据源。目前官方支持如下数据源：
  - Hive：开源的分布式数据仓库，提供丰富的分析功能。
  - MySQL：一种关系数据库管理系统，提供高性能的联机事务处理功能。
  - PostgreSQL：一个开源的对象关系数据库管理系统，提供强大的关系型数据库服务。
  - Oracle：一种分散的数据库管理系统，支持大型企业级应用。
  - Microsoft SQL Server：一款关系数据库管理系统，提供高性能的联机事务处理和分析功能。
  - Amazon Redshift：一种基于 Amazon 的分布式数据仓库，支持快速的查询处理。
  - Google BigQuery：一种云端数据仓库，提供高性能的联机分析查询功能。
  - Apache Kudu：一个开源的分布式存储和分析平台，提供实时 HTAP（Hybrid Transactional and Analytical Processing） 分析功能。
- Catalog：Presto 会将每个数据源都视作一个 catalog 中的 schema，可以通过 catalog 来控制对源数据源的访问权限和数据路由规则。例如，我们可以在其中配置白名单，限制某些用户只能访问指定的 schema 或表。
- Schema：Presto 的数据架构由一系列的表组成，这些表可以按照逻辑或物理的方式组织在一起。每个 table 可以拥有自己的列和行，还可以存储索引和分区。
- Table：一个逻辑上的集合，里面包含了一组相关的数据项。
- Column：在 table 中定义的字段，包含了名称、类型和其他相关属性。
- Row：在 table 中存储的一行数据，包含了一组列值。
- Partition：表按照一定方式被划分成多个子集，称为 partition。Partition 的作用主要是为了提高查询效率，并通过并行查询来提升整体查询性能。例如，在一个 table 中，我们按年份对数据进行 partition，这样就可以在一个节点内快速查询某一年的全部数据。
- Query：一种对数据源请求信息的请求语句，通常遵循 SQL 语言规范。查询一般分为两类：联机分析查询（OLAP）和数据定义语言（DDL）。
- Stage：Presto 对查询的执行过程进行划分，主要分为以下几个阶段：
  1. 分析阶段：Presto 会解析查询语句，检查语法、语义和查询模式是否正确。
  2. 优化阶段：优化器会分析查询计划，生成最优的查询计划。
  3. 分发阶段：Presto 根据查询计划把数据分配给各个工作节点。
  4. 执行阶段：各个工作节点根据查询计划执行查询。
  5. 输出阶段：结果数据会被发送回客户端。
- Node：Presto 的工作节点，可以是独立的机器或者容器。
- Worker：一种运行在 Presto 集群中的进程，负责接收、解析和执行查询。
- Coordinator：Presto 集群中运行着唯一的一个协调者节点。当用户发起一条查询请求时，首先经过协调者节点的路由和协助调度，然后再将请求分发到各个 worker 节点执行。
- Split：当查询涉及到的表太大无法全部加载入内存时，Presto 将数据切分成小块 split，并将其分布到各个工作节点上执行。
- Operator：在查询计划中，用来描述查询操作行为的图形表示。比如 ScanOperator 表示从数据源读取数据；AggregationOperator 表示执行聚合函数；FilterOperator 表示过滤满足条件的数据项。
- Plan：Presto 生成的查询计划就是一张 DAG 图，其中每个结点表示一个算子，边表示输入输出关系。
- Task：一个 Presto 工作节点实际执行查询时的最小单位，即将 split 分配给当前节点执行的最小单元。一个 task 可能包含多个 split。
- Memory Pool：Presto 会预先分配一定的内存空间作为本地 buffer pool。当数据被读入内存后，Presto 可以直接使用；如果不够，则将数据交给磁盘进行缓存，并使用 cache manager 从磁盘缓存中加载数据。
- Buffer Cache：Presto 会将缓存数据写入 buffer cache 中，以便下次查询时直接命中。buffer cache 有两种大小：小 buffer cache 和 large buffer cache。
- Execution Engine：执行引擎是一个组件，它主要负责执行 query plan，并返回查询结果。Presto 支持两种执行引擎：标量执行引擎和列存执行引擎。
- Slice Threads：一个线程池，用于运行并发查询。
- Resource Groups：资源组是 Presto 特有的资源隔离机制，它允许管理员定义多个资源组，并将查询请求分别放到不同的资源组中执行。
- Session Properties：Presto 会使用 session properties 来配置当前查询的运行参数，如开启/关闭动态过滤、启用/禁用 Codegen、设置查询并行度等。
- Dynamic Filtering：动态过滤是一种查询优化方法，它允许 Presto 只扫描那些符合 WHERE 子句过滤条件的分片数据，而不会扫描整个分片。只有在非聚合查询的情况下才会生效。
- Codegen：Codegen 是一种编译器技术，可以将查询的运算符转译成机器码，加速查询执行。
- Optimizer：优化器是 Presto 的重要组件之一，它负责生成查询计划并选择查询执行策略。
- Cost Based Optimizer：基于成本的优化器是一种优化器算法，它结合统计信息、代价模型和查询计划的特征等因素来生成查询计划。
- Filter Pushdown：过滤下推是一种查询优化技术，它允许 Presto 利用索引筛选掉不满足 WHERE 子句条件的分片数据，并将这些分片数据传递给执行引擎执行。
- Join Ordering：Join 排序是指按照关联顺序匹配数据项，避免出现笛卡尔积。Join 排序是在 Join 操作前进行的优化，目的是减少不必要的网络传输、缓冲以及聚合操作。
- Parallelism Estimator：并行度估计器是 Presto 的另一重要组件，它会根据查询计划的依赖关系来估计每个节点的资源使用情况，并设置好每个节点的并发度。
- Data Skew：数据倾斜是指数据分布不均匀的问题。数据倾斜可能发生在 join、group by 和 aggregation 时，可以将相同的数据分配到不同的节点导致效率下降。
- Broadcast Hash Join：广播哈希连接是一种优化策略，它可以在两个表之间做全外连接时通过将其中较小表的全部数据复制到所有节点，避免网络传输消耗。
- Dynamic Partition Elimination：动态分区剪枝是一种查询优化方法，它会在查询过程中剔除不必要的分区，降低查询时的资源消耗。

## 2.2 数据仓库建模

### 2.2.1 Star Schema与 Dimensionality Denormalization

Presto 不支持对维度建模，但提供了一种改进的方式——Dimensionality Denormalization。

Dimensionality Denormalization 是一种数据仓库设计模式，它将常用的维度放在相关表中，而不是将它们冗余地存放在每一笔事务中。举个例子，假设有一个订单表，包含订单号、日期、客户编号、产品编号等字段。我们可以将日期、客户编号、产品编号等维度信息放在维度表中，并通过关联键来引用订单表。这样，订单表就只需要存储主键标识和交易金额等基本信息，而不需要重复存储维度信息。由于维度表已经按照相关的键值进行了索引，因此可以有效地避免JOIN操作，显著提升查询性能。

采用 Dimensionality Denormalization 的另一个优势是可以避免数据重复，因为维度表通常比事实表要小很多，所以占用的磁盘空间也相对较小。

对于大规模数据集来说，Dimensionality Denormalization 的效果十分明显。但是，也存在一些缺陷。首先，需要引入额外的维度表，导致数据维护成本增大。其次，需要修改关联关系，增加了查询复杂度。最后，由于引入了新的数据结构，可能会导致查询的延迟增加。总之，Dimensionality Denormalization 需要根据实际业务需求进行权衡和取舍。