
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Presto是一个开源的分布式查询引擎，它采用Java开发，基于Apache开源协议，由Facebook和Twitter开发。主要用于对海量数据进行快速、高效、复杂的分析。可以理解成Hive的替代品，但功能更加强大。Presto支持SQL，JDBC/JDBS接口，以及可以集成到现有的BI工具中。目前Presto已经成为Apache基金会孵化器项目，并进入了Apache孵化器，开始积极探索阶段。下面将通过一个案例学习一下Presto的工作原理和特点。
# 2.背景介绍
一个典型的数据仓库系统包括以下几个重要环节：数据收集、清洗、转换、存储、检索和报告。当需要对多种异构的数据源（如数据库、文件等）进行复杂查询时，就需要引入ETL工具来进行统一的数据规范化处理，然后才能进行复杂查询分析。然而随着互联网数据的爆炸式增长、海量数据的呈现，数据处理变得越来越难以进行。人们希望能够更快、更精准地获取所需的信息，并且不需要关注底层数据源本身。这就是为什么我们需要一种分布式查询引擎。Presto是一种开源的分布式查询引擎，它提供快速且高效的计算能力。它的特点如下：

1. 对复杂查询进行优化，提升执行效率；
2. 支持丰富的数据源类型，包括关系型数据库、NoSQL数据库、HDFS等；
3. 提供SQL接口，方便与其他程序集成；
4. 可连接到现有的BI工具中，提供数据的即席查询和报告。

# 3.核心概念和术语说明
## 3.1 Presto架构
Presto由以下几个组件构成：
1. Coordinator服务：负责接收客户端请求，根据查询计划将查询任务分配给各个Worker节点；
2. Worker节点：负责实际执行查询任务；
3. Query Processors：负责解析SQL语句，生成查询计划，并分发执行任务到其他节点执行；
4. Memory Manager：管理内存资源；
5. Node Managers：管理节点的生命周期；
6. Splitters：负责将单个的输入数据文件切分为多个Split，并将Split数据缓存到内存或磁盘；
7. File System Connectors：支持多种文件系统类型，如HDFS、本地文件系统等。

## 3.2 分布式查询
Presto分布式查询的特点：
1. 数据按片进行分区，每个片由一个查询进程执行，同时避免数据倾斜问题；
2. 可以利用集群资源提供更好的查询性能；
3. 通过广播机制，减少网络传输消耗；
4. 引入查询缓存机制，减少查询响应时间。

## 3.3 查询计划
Presto的查询计划由以下几个部分组成：
1. SQL Parser：将SQL语句解析为抽象语法树AST；
2. Statement Analyzer：通过优化器将AST划分为逻辑执行计划，并插入物理执行计划生成器；
3. Plan Optimizer：进行逻辑和物理优化；
4. Physical Execution Planner：生成物理执行计划；
5. Task Executor：按照物理执行计划分派任务到Worker节点上执行。

## 3.4 并行查询
Presto支持多线程执行查询，并行查询可以有效提升查询速度。其过程如下：
1. 执行器：接收客户端发送的查询请求，并创建线程池执行查询计划；
2. 协调器（Coordinator）：接受查询请求，并将查询计划分派给集群中的工作节点，协调工作节点的查询任务；
3. 分片器（Spliter）：将数据切割为多个数据块，并缓存到内存或磁盘；
4. 编译器：将SQL语句转换为内部语言指令；
5. 执行器：根据执行计划在集群中的各个工作节点上执行查询；
6. 结果集：接收执行器的执行结果，返回给客户端。

# 4. Presto的安装部署
## 4.1 安装前准备
下载地址：https://prestodb.io/download.html ，选择合适版本进行下载。将下载的文件上传至目标服务器，解压后将bin文件夹配置到环境变量PATH中即可。
```
tar -zxvf presto-server-0.179-executable.tar.gz 
mv presto-server-0.179* /usr/local/presto/ # 将下载的文件移动到指定目录
echo "export PATH=/usr/local/presto/bin:$PATH" >> ~/.bashrc   # 在bashrc配置文件末尾添加环境变量路径信息
source ~/.bashrc    # 更新bashrc配置文件使之立即生效
```
## 4.2 配置Presto
### 4.2.1 配置文件
Presto的配置文件一般放在`$PRESTO_HOME/etc`目录下，包括`config.properties`，`jvm.config`，`log.properties`。其中`config.properties`是最重要的配置文件，里面包含绝大多数配置项。
```
# 日志相关配置项
http-server.http.port=8080  # HTTP端口号
http-server.https.enabled=false     # 是否开启HTTPS
http-server.https.port=8443        # HTTPS端口号
node-scheduler.include-coordinator=true    # 是否启用协调节点
node-scheduler.network-topology=flat      # 网络拓扑结构
query.max-memory=50GB                # 最大可用内存，超出该值则查询失败
discovery-server.enabled=true         # 是否启用DiscoveryServer
discovery.uri=http://localhost:8080   # DiscoveryServer的地址

# connector相关配置项
catalog.default=jmx    # 默认的connector类型为JMX

# catalog相关配置项
jmx.user=admin       # JMX用户名
jmx.password=admin123    # JMX密码
jmx.port=1099             # JMX服务监听端口
jmx.refresh-period=1m     # JMX元数据刷新间隔
jmx.max-history=1d        # JMX元数据历史最大记录天数
```

### 4.2.2 修改认证方式
默认情况下，Presto不启用认证功能，如果需要限制访问权限，可以在`config.properties`文件里设置相关参数。比如禁止匿名登录：
```
http-server.authentication.type=PASSWORD,LDAP
http-server.allow-anonymous-access=false
```
此外还可以选择其他认证方式，如Kerberos、JWT等。

### 4.2.3 添加connector
除了默认的`jmx`类型的connector之外，还有很多第三方的connector可供选择，比如hive、mysql、postgresql等。可以通过编辑配置文件`config.properties`来加载额外的connector。示例如下：
```
# hive相关配置项
connector.name=hive-hadoop2
hive.metastore.uri=thrift://localhost:9083
hive.metastore.username=test
hive.metastore.password=<PASSWORD>
```
注意，这里使用的connector名称应该与配置文件中的`catalog.default`项匹配。

### 4.2.4 启动服务
完成配置之后，就可以启动Presto服务了：
```
./bin/launcher start
```
启动成功后，可以查看日志文件`$PRESTO_HOME/var/log/server.log`确认是否正常运行。

# 5. Presto的基本操作
Presto提供了丰富的接口和命令行工具来进行数据分析。本节将介绍一些常用命令，以及如何在Hadoop生态圈内与Hive进行对比。

## 5.1 命令行工具
Presto提供了两个命令行工具，用来帮助用户连接到Presto集群，并执行各种命令。

### 5.1.1 CLI
CLI (Command Line Interface) 是Presto的一款命令行工具，可以通过直接在命令行窗口输入命令的方式连接到Presto服务器，执行命令，查看查询结果。安装包里已经包含了CLI工具，安装路径在`$PRESTO_HOME/bin/`。

#### 连接集群
首先要连接到Presto服务器，可以使用以下命令：
```
$./presto --server localhost:8080
```
这样就会打开一个交互式命令行，可以输入查询语句，或者直接退出。

#### 查看元数据
可以使用以下命令查看Presto中所有的表、视图、函数等信息：
```
SHOW CATALOGS;
SHOW SCHEMAS FROM jmx;
SHOW TABLES FROM jmx.current;
DESCRIBE jmx.current.threads;
```
这些命令都是列出所有可用的对象，也可以指定对象名字进行过滤，例如：
```
SELECT * FROM jmx.current.threads WHERE name LIKE '%CompactionExecutor%';
```
这个例子显示了所有CompactExecutor相关的线程信息。

#### 查询数据
可以使用SELECT命令从Presto服务器查询数据：
```
SELECT * FROM jmx.current.runtime MX WHERE type = 'GarbageCollector' AND name LIKE '%MarkSweep%';
```
这个例子查询了Garbage Collector相关的指标，只返回MarkSweep垃圾回收器的数据。

#### 插件
除了命令行工具外，Presto还提供了Web界面和java客户端两种插件，这两种插件需要单独安装。Web界面和Java客户端均可用来连接到Presto服务器，并执行各种命令。

### 5.1.2 JDBC driver
Presto提供了Java驱动程序（JDBC Driver），允许应用通过JDBC API与Presto服务器通信。可以通过以下依赖加入到项目中：
```
<dependency>
    <groupId>com.facebook.presto</groupId>
    <artifactId>presto-jdbc</artifactId>
    <version>${project.version}</version>
</dependency>
```
然后创建一个Connection，设置用户名和密码，并执行查询语句：
```
        Connection conn = DriverManager.getConnection(
                "jdbc:presto://localhost:8080",
                "test",
                "test");

        try (Statement stmt = conn.createStatement()) {
            ResultSet rs = stmt.executeQuery("SELECT COUNT(*) FROM jmx.current.runtime MX WHERE type = 'MemoryPool'";

            while (rs.next()) {
                long used = rs.getLong("used"); // 使用的内存字节数
                String poolName = rs.getString("name");

                // do something with the data...
            }
        } finally {
            conn.close();
        }
```
这里的API比较简单，涉及到的功能也很少，可以满足日常查询需求。

## 5.2 Hive与Presto对比
虽然两者都是非常优秀的开源分布式查询引擎，但是还是有些不同之处。下面我们来对比一下它们之间的一些特性。

### 5.2.1 计算模型
Hive是基于 MapReduce 的批处理模型，而Presto是基于 Distributed query engine 的流处理模型。这意味着Hive需要先将数据写入HDFS、然后运行MapReduce作业计算出结果，再读取出来。相反，Presto只需要把查询计划提交给集群上的多个worker节点，然后让它们一起执行查询计划，实时产生结果并返回。这样做可以实现更快的查询响应时间。

### 5.2.2 数据格式
Hive只支持Hive表存储格式，而Presto支持很多数据源，比如MySQL、PostgreSQL、MongoDB等。这种灵活性使得Presto可以分析更广泛的来源的数据。

### 5.2.3 元数据管理
Hive的元数据存储在一个独立的Hive Metastore数据库中，而Presto的所有元数据都保存在内存中，因此查询速度快。另外，Presto对元数据的变化作出反应的速度很快，因此无需等待Hive Metastore的更新。

### 5.2.4 用户权限控制
Hive可以通过Hive权限模型对用户进行细粒度控制，控制用户可以访问哪些数据库、表和字段。但是，Presto没有独立的权限模型，它使用基于角色的访问控制（Role Based Access Control，RBAC）。角色定义了一系列权限规则，用户可以被授予某个角色，然后根据角色的权限控制进行访问。

### 5.2.5 报告工具集成
Hive不提供集成的BI工具，只能通过将结果导出到文件，然后导入到BI工具中。Presto提供了多种查询结果输出格式，包括CSV、JSON、Excel、Avro等，并且可以集成到诸如Tableau、Microsoft Excel、QlikView等工具中。