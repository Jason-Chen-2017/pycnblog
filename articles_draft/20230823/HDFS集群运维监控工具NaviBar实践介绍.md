
作者：禅与计算机程序设计艺术                    

# 1.简介
  

HDFS（Hadoop Distributed File System）是一个开源的分布式文件系统，用于存储海量数据。由于其高容错性、高可靠性和海量数据的处理能力，HDFS被广泛应用在大数据分析、日志处理、大型数据集存储等场景中。因此，HDFS集群运维监控是一个非常重要的工作。如果HDFS集群出现故障或性能下降，那么相应的管理人员往往需要快速定位和解决问题。因此，集群运维监控工具是集群管理中的重要环节之一。最近，随着 Hadoop 发展的日益普及，HDFS 管理工具也逐渐成为 Hadoop 用户不可缺少的一部分。本文将介绍一个开源的 HDFS 集群监控工具 NaviBar 的实践。
# 2.核心概念术语
- NameNode（NN）：NameNode 是 HDFS 中的主节点，负责管理整个 HDFS 文件系统的名称空间 (namespace) 和客户端对文件的访问。它主要职责如下：
  - 管理 HDFS 文件系统的名字空间，包括目录树、块大小、副本策略等；
  - 将用户的文件请求翻译成对应的DataNode地址；
  - 处理客户端读、写、复制、删除等请求；
  - 支持 secondary namenode，充当热备份角色；
- DataNode（DN）：DataNode 是 HDFS 中每个工作节点，负责存储实际的数据块。它主要职责如下：
  - 把磁盘上的数据划分成多个数据块并向 NameNode 注册；
  - 执行数据块上传、下载、复制、删除等操作；
  - 接收来自其他DataNode的命令，执行数据块校验和检验等；
  - 定期报告自身状态给 NameNode；
- Block：HDFS 中的数据块由多个 DataNode 上的连续的物理位置组成，一般是 128MB 或 256MB。HDFS 可以支持多种类型的块，如一般的副本块，Raid块，SSD快照块等。
- Replica：HDFS 中的每个数据块可以有多个副本，默认情况下，HDFS 使用机架感知(rack-aware)的副本放置策略，即副本会尽可能地放在不同机器上。如果某个 DataNode 发生故障，HDFS 会自动把该副本迁移到另一个健康的DataNode上。另外，也可以通过设置副本策略，让 HDFS 根据业务情况自动生成副本数量。
- Client：HDFS 的客户端通常是指运行 MapReduce 程序的用户。Client 通过网络连接到 HDFS Namenode，然后向 Namenode 发送各种命令，例如打开、关闭、读取、写入文件、创建目录等。
- JournalNode（JN）：JournalNode 是一个特殊的NameNode，它主要用来记录HDFS文件系统元数据的修改。JournalNode 以事务日志的方式记录对文件系统元数据的更改，确保了元数据操作的原子性、持久性和一致性。
- Zookeeper：Zookeeper 是 Apache Hadoop 生态系统中的重要组件。它主要用来维护 HDFS 的元数据信息、配置信息、命名空间等。Zookeeper 提供高可用、易于使用的分布式协调服务。
- JMX：JMX （Java Management Extension，java管理扩展）是 Java 平台中用于监视和管理 Java 应用程序的框架。JMX 提供了一套丰富的 API，使得开发者能够轻松地获取 Java 应用程序的信息和控制其运行时行为。
- Grafana：Grafana 是一款开源的基于网页的系统监控和数据可视化工具。它提供强大的查询语言来绘制丰富的图表，并提供了直观的 UI ，使得 HDFS 集群管理员和运维人员可以直观地看到集群的运行状况。
# 3.NaviBar 功能概述
NaviBar 是 Hadoop 官方推出的面向 HDFS 集群监控的开源工具，具备以下特性：
- 功能丰富：NaviBar 提供了诸如系统健康度监控、硬件资源监控、集群容量规划、集群压力测试、垃圾回收情况、目录文件数量、集群整体吞吐量、磁盘利用率等多项功能。
- 可视化展示：NaviBar 在界面设计上采用了简洁、美观的风格，使得用户能够直观地查看各项指标的变化曲线。同时，它还提供了丰富的图表选项，满足用户对不同维度的监控需求。
- 全天候监控：NaviBar 具有优秀的响应速度，可以实时的监测集群的运行状态。它不仅适合实时查看，还可以帮助管理员预警集群的异常情况，及时进行及时的处理措施。
# 4.安装部署
NaviBar 安装部署比较简单，只需按照官网提示一步步进行即可。详细步骤如下：
- 安装依赖库
NaviBar 服务端基于 Python Flask 框架实现，所以首先要安装 Python 的相关环境。安装完成后，使用 pip 命令安装所需依赖包：
```bash
pip install flask requests requests_kerberos gevent psutil redis pymongo flask_socketio eventlet configparser navibarclient kafka-python kubernetes pyyaml elasticsearch prometheus_client kazoo pyhdfs mako bokeh matplotlib click numpy pandas scipy networkx seaborn beautifulsoup4 gunicorn ujson lxml psycopg2
```
- 配置 NaviBar 服务端
NaviBar 服务端配置文件名为 `navibar-web.ini`，配置文件所在路径为 `/etc/navibar`。主要配置项如下：
```
[server]
host = 0.0.0.0 # NaviBar Web 服务监听主机 IP
port = 8000 # NaviBar Web 服务监听端口
debug = False # 是否开启调试模式
testing = False # 是否开启测试模式
secretkey = <KEY> # secret key

[zookeeper]
hosts = localhost:2181 # zookeeper 地址
zkrootpath = /hadoop-ha # HDFS HA 模式 root path

[mongodb]
hosts = localhost:27017 # MongoDB 地址
db = hadoopdata # MongoDB 默认数据库名
collections = clusterdata,datanodes,namenodes,jmxcollectors,hachecker # MongoDB collections 名称

[kafka]
brokers = localhost:9092 # Kafka broker 地址
topic = hdfsmetrics # Kafka topic 名称
groupid = navibar # Kafka group id

[prometheus]
url = http://localhost:9090 # Prometheus 地址

[elasticsearch]
hosts = localhost:9200 # Elasticsearch 地址
indexname = navibarlogs # Elasticsearch index 名称

[redis]
host = localhost # Redis 地址
port = 6379 # Redis 端口
password = # Redis 密码，如果没有则为空

[kafkatopics]
topics =, # 自定义 kafka topics，多个用逗号隔开

[journalsnode]
hostlist = journalnode1:8485,journalnode2:8485 # jounral node host list

[configfiles]
namenode = hdfs-site.xml # Hadoop namenode 配置文件名
datanode = core-site.xml # Hadoop datanode 配置文件名
navibardatasource = data.csv # 数据源文件，存储各个 Hadoop 节点配置参数，一般为 CSV 文件
```
其中，`kafkatopics` 为可选配置项，默认值为空。如果需要在 NaviBar 上自定义显示的 kafka topics，则配置此项。
- 创建导航栏配置文件
导航栏配置文件名为 `navibar-sidebar.yml`，配置文件所在路径为 `~/.navibar`，主要配置项如下：
```
datasource: navibar-datasource # 数据源文件，一般为 CSV 文件
clustername: hadooptest # 集群名称
sortby: hostname # 排序依据字段，hostname 表示按主机名排序
storages: # 存储列表
    - name: HDFS
      paths: # 文件系统路径
        - "/"
      servers: # 服务器列表
        - nn1:namenode
        - dn1:datanode
        - dn2:datanode
```
其中，`datasource` 为必填配置项，表示 NaviBar 从何处加载数据。`clustername` 为可选配置项，默认为 `navibar`。`sortby` 则表示按什么字段进行排序，`storages` 表示 NaviBar 应显示的存储列表，包括文件系统路径、服务器列表等。
- 启动 NaviBar 服务端
启动 NaviBar 服务端命令为：
```bash
nohup python navibar/main.py &
```
- 浏览器访问 NaviBar
浏览器输入 `http://<NaviBar server ip>:8000/`，登录用户名密码均为 `admin`，进入导航栏页面，即可看到具体的监控结果。