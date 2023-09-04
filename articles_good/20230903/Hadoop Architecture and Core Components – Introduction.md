
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Hadoop是一个分布式数据处理系统，主要面向批处理和实时分析的数据集上进行计算任务。它由两个主要组件组成，分别是Hadoop Distributed File System（HDFS）和MapReduce。HDFS是一个存储海量文件数据的分布式文件系统，用于支持大规模并行计算；而MapReduce是一个编程模型和运行框架，用于对海量的数据进行分治处理，并最终输出结果。两者合起来可以实现存储和处理海量数据的能力。因此，Hadoop架构就是把HDFS和MapReduce整合到一起，形成一个完整的平台，提供高效、可靠、可扩展的计算能力。
本篇文章将从Hadoop架构以及HDFS、YARN等核心组件的功能、作用、原理、工作流程、用法等方面进行介绍，帮助读者更好地理解Hadoop及其组件的特性和运作原理。

# 2.基本概念术语说明
## 2.1 Hadoop相关术语

- **HDFS(Hadoop Distributed File System):** HDFS是一个分布式文件系统，用于存储和处理大数据量。HDFS通过将文件存储在不同的服务器上，提高了存储容量、处理能力和扩展性。HDFS中的每个节点都存储整个文件系统的一部分，并且能够同时服务多个客户端请求。HDFS的功能主要包括文件的存储、文件的切片、文件的复制、文件的权限管理、集群的容错恢复等。
- **MapReduce:** MapReduce是一个编程模型和运行框架，用于对海量的数据进行分治处理，并最终输出结果。MapReduce模型将任务分成多个阶段，包括map阶段和reduce阶段，并采用容错机制保证任务的正确执行。MapReduce工作流程包括：Map函数、Shuffle和Sort阶段、Reduce函数。
- **JobTracker/TaskTracker:** JobTracker和TaskTracker都是Hadoop的两个守护进程，负责资源调度和任务处理。JobTracker主要管理所有的任务，包括任务的提交、资源分配、任务执行状况监控等；而TaskTracker则负责单个任务的执行。它们之间的通信是基于RPC协议进行的。
- **NodeManager/ResourceManager:** NodeManager和ResourceManager都是Hadoop的两个守护进程，分别管理DataNode和NodeManager。NodeManager是一个运行在各个节点上的服务，用于接收来自JobTracker的资源请求，并将它们分配给相应的TaskTracker去执行任务。ResourceManager也是Hadoop的一个守护进程，主要用来管理集群中所有NodeManager的状态信息。 ResourceManager将集群资源划分为若干个队列，并按照一定策略将资源分配给对应的任务。
- **Zookeeper:** Zookeeper是一个分布式协调系统，用于维护集群中的所有结点的同步、统一和订阅配置等。Zookeeper是一个开放源码的分布式协调工具，它是一个针对分布式应用的高可用服务框架。Zookeeper非常适合于Hadoop生态圈，因为它既可以管理HDFS的元数据信息，也可以作为Hadoop集群的中心协调者。
- **Flume:** Flume是一个分布式、可靠、和高可用的海量日志采集、聚合和传输的系统。Flume以流式的方式来收集数据，并在HDFS、HBase或Kafka等中存储。Flume可以安全、可靠地传输大量的数据，因此被广泛地用于各种场景，如日志采集、数据聚合、数据传输等。
- **Hive:** Hive是基于Hadoop的SQL查询引擎，具有高并发查询、海量数据分析的能力。Hive可以使用类似SQL语句的语法来查询HDFS中的数据，并将查询结果存入HDFS或数据库中。
- **Sqoop:** Sqoop是一个开源的ETL工具，可以用于导出和导入关系型数据库和Hadoop集群之间的数据。
- **Hue:** Hue是一个开源的Web界面，可用于管理Hadoop集群和服务。它提供了对HDFS、YARN、Hive、Flume、Sqoop、HBase等多个组件的图形化管理。
- **Pig:** Pig是一个基于Hadoop的MapReduce编程语言，可以用来转换、过滤、聚合和分析数据。

## 2.2 Hadoop相关概念

- **集群：** Hadoop集群是一个多主机的独立计算机网络，通常由HDFS和MapReduce所依赖的NodeManager和ResourceManager等组件构成。它通常由一组备份机构部署来防止单点故障。
- **名称节点（NameNode）：** NameNode也称为Master，它是一个独立的HDFS守护进程，存储着文件系统的名字空间和权限信息，管理文件系统的命名空间，处理客户端的请求，比如打开、关闭、编辑文件和目录。NameNode在集群启动的时候就会启动，它同时也是一个超级用户接口，用户可以通过它来创建、删除或者复制文件。
- **数据节点（DataNode）：** DataNode也称为Slave，它是一个独立的HDFS守护进程，负责储存文件系统的数据块。每一个DataNode都和NameNode保持联系，报告自己所保存的数据块列表。DataNodes是Hadoop文件系统的核心。它们向NameNode汇报磁盘使用情况，汇报其上已经加载的文件块信息。
- **路径名（Pathname）：** 路径名是指文件系统中某个文件或目录的逻辑地址。路径名可以包含斜杠`“/”`，表示层次结构，如`“/home/user/documents/file.txt”`。
- **主节点（Primary Node）和辅助节点（Secondary Node）：** HDFS中的数据块副本被划分为主节点和辅助节点两种角色。主节点负责写数据，辅助节点负责读数据。主节点失败时，会自动选择另一个节点代替。
- **Block：** 数据块是HDFS中最小的读写单元。HDFS中的数据块大小默认为128M。
- **副本系数（Replication Factor）：** 副本系数是指文件在不同节点间的拷贝数量。副本系数越大，文件存储的冗余度就越高。HDFS默认副本系数为3。
- **客户端（Client）：** 客户端是访问Hadoop的文件系统的组件。Hadoop提供了命令行客户端和图形界面客户端。客户端通过RPC协议与NameNode交互，来获取文件系统的路由表，并通过DataNode读写数据。
- **namenode所知之处（Known as the Master）：** Namenode是Hadoop的中心枢纽，所有的文件元数据信息以及其他HDFS数据都会存储在Namenode中。因此，当NameNode发生故障时，系统仍然可以继续运行，因为有另外的Namenode替代它。
- **datanode所知之处（Known as Slaves）：** Datanode则是Hadoop数据存储的主要节点，负责存储HDFS数据块并响应客户端读写请求。Datanode只需知道其他DataNode的存在即可，不需要直接感知其他DataNode的位置信息。当Datanode的存储空间不足时，会向NameNode发送指令来增加数据块的副本。

# 3.HDFS核心组件

## 3.1 HDFS架构设计


1. **NameNode**: HDFS中的主节点，管理着整个文件系统的名称空间和数据块映射。NameNode还负责处理客户端的读写请求，并报告DataNode的存储状况。NameNode在HDFS的生命周期内只有一个，随集群一起启动，并一直运行直到整个HDFS的垃圾回收完成。
2. **Secondary NameNode**: 次要的NameNode，可以看做是NameNode的热备。它监听NameNode的心跳信号，并在需要时通过Secondary NameNode自动切换到NameNode。
3. **DataNode**: HDFS中的数据节点，实际上存储着HDFS的真正数据。DataNode会定期向NameNode汇报自己的存储状况，并向NameNode报告需要复制哪些数据。
4. **客户端**: HDFS客户端允许用户读取和写入HDFS中的文件。客户端连接到NameNode，并向NameNode请求文件系统的相关信息，如文件的位置。然后客户端通过DataNode直接读写文件。

## 3.2 HDFS存储体系


HDFS的存储体系由HDFS的三种主要角色组成：NameNode、DataNode和Client。

**1. NameNode (主节点)** 

NameNode是HDFS的主节点，负责管理文件系统的名称空间和数据块映射。NameNode主要的职责如下：

- 文件系统的名称空间管理：它记录了文件系统里的文件和目录的元数据信息。元数据信息包括文件名、文件属性、权限、访问时间、数据块信息等。
- 数据块管理：它根据配置文件中设置的副本系数，创建文件的数据块，并管理这些数据块的映射关系。
- 名字服务：它提供基于文件路径的命名空间。客户端向NameNode请求文件或目录的元数据信息时，NameNode返回所查对象的绝对路径名，使得客户端能够方便地寻址。
- 客户请求处理：它接收客户端的读写请求，并将请求转发给相应的DataNode处理。如果某个DataNode出现故障，NameNode会立即感知并将该DataNode上的数据块信息更新。

**2. DataNode (数据节点)** 

DataNode是HDFS中最基本的角色，负责储存HDFS数据的最底层。DataNode主要的职责如下：

- 储存数据：它向NameNode报告自己所持有的块列表，并通过网络接口接受来自其他DataNode的块数据。
- 数据块校验：它验证从客户端接收到的块数据的有效性。
- 失效检测：它检测数据块是否丢失或损坏，并将损坏的块重新复制。

**3. Client （客户端）** 

客户端是用户应用程序的接口，它与NameNode通过网络接口相连，并通过调用远程过程调用（RPC）方式向DataNode发起请求。客户端向NameNode发起文件的读写请求，NameNode在返回结果前，会检查目标文件的元数据信息和块映射关系。如果元数据信息或块映射关系出错，NameNode会向客户端返回错误消息，并提示客户端重试。

## 3.3 HDFS体系结构总结


HDFS体系结构可以清楚地看到，HDFS由NameNode和DataNode两个核心组件构成，其中NameNode负责管理文件系统的名称空间、数据块映射和客户请求处理，DataNode负责存储实际数据，而客户端与NameNode和DataNode直接交互。NameNode和DataNode之间通过远程过程调用（RPC）通信，向外提供读写文件的接口。此外，HDFS体系结构还有JournalNode和ZKFC（FailOver Controller）两个后台进程，分别用于实现数据备份和Failover机制。


# 4.HDFS 操作演示

## 4.1 配置hadoop环境

### 安装hadoop

下载hadoop安装包，解压至指定文件夹下

```bash
tar -zxvf hadoop-X.X.X.tar.gz -C /usr/local/
mv /usr/local/hadoop-X.X.X /usr/local/hadoop
cd /usr/local/hadoop/etc/hadoop/
```

修改配置文件

```properties
# core-site.xml
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000/</value> # 指定namenode的位置
  </property>
  <property>
    <name>hadoop.tmp.dir</name>
    <value>/usr/local/hadoop/tmp</value> # 指定临时文件的存放位置
  </property>
</configuration>

# hdfs-site.xml
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>2</value> # 指定数据块的副本数目
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>/usr/local/hadoop/nn</value> # namenode的元数据和镜像文件的存放位置
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>/usr/local/hadoop/dn</value> # datanode上数据块的存放位置
  </property>
</configuration>

# mapred-site.xml
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value> # 指定运行模式为YARN
  </property>
  <property>
    <name>mapreduce.jobhistory.address</name>
    <value>localhost:10020</value> # jobhistoryserver的地址
  </property>
  <property>
    <name>mapreduce.jobhistory.webapp.address</name>
    <value>localhost:19888</value> # jobhistoryserver的web页面地址
  </property>
</configuration>
```

启动hadoop

```bash
# 如果之前启动过，先停止
$ sbin/stop-dfs.sh
$ sbin/stop-yarn.sh

# 分别启动namenode、datanode、resourcemanager和nodemanager
$ sbin/start-dfs.sh
$ sbin/start-yarn.sh
```

查看端口是否开启

```bash
# jps 查看是否有namenode、datanode、resourcemanager、nodemanager、jobhistoryserver等进程正在运行
# netstat -antup | grep java 查看端口是否开启
```

## 4.2 创建HDFS文件系统

首先我们创建一个目录，作为我们的HDFS文件系统根目录。

```bash
mkdir -p /myhdfs/input
```

然后将上面新建的文件夹作为我们的hdfs文件系统的根目录。

```bash
bin/hdfs dfs -mkdir /myhdfs
bin/hdfs dfs -put mytest.txt /myhdfs/input
```

查看文件系统：

```bash
bin/hdfs dfs -ls /myhdfs
```

## 4.3 使用HDFS

### 通过命令上传下载文件

```bash
# 把本地文件上传到HDFS：
bin/hdfs dfs -put localfile /myhdfs/input/remotefile

# 从HDFS下载文件到本地：
bin/hdfs dfs -get /myhdfs/input/remotefile localfile
```

### 通过java代码上传下载文件

```java
// upload file to hdfs
FileSystem fs = FileSystem.get(URI.create("hdfs://localhost:9000"), conf); // 创建文件系统对象
FileStatus fstatus = fs.getFileStatus(new Path("/myhdfs/input")); // 获取目标文件状态
if (!fstatus.isDir()) {
    throw new IOException("Target is not a directory");
}
InputStream in = new FileInputStream(new File("/path/to/localfile")); // 输入流
OutputStream out = fs.create(new Path("/myhdfs/input/remotefile"), true); // 创建输出流
IOUtils.copyBytes(in, out, conf, false); // 将输入流写入输出流
out.close(); // 关闭输出流
in.close(); // 关闭输入流
fs.close(); // 关闭文件系统

// download file from hdfs
FileSystem fs = FileSystem.get(URI.create("hdfs://localhost:9000"), conf); // 创建文件系统对象
FileStatus[] files = fs.listStatus(new Path("/myhdfs/output")); // 获取目标文件列表
for (FileStatus file : files) {
    if (!file.isDir()) {
        InputStream in = fs.open(file.getPath()); // 打开文件
        OutputStream out = new FileOutputStream("/path/to/download/" + file.getPath().getName()); // 输出流
        IOUtils.copyBytes(in, out, conf, false); // 写入本地文件
        out.close(); // 关闭输出流
        in.close(); // 关闭输入流
    }
}
fs.close(); // 关闭文件系统
```

### 命令行管理

#### 查看当前文件系统状态

```bash
bin/hdfs dfsadmin -report
```

#### 检测磁盘使用率

```bash
bin/hdfs fsck /myhdfs
```

#### 删除文件

```bash
bin/hdfs dfs -rm /myhdfs/input/remotefile
```

#### 查看文件属性

```bash
bin/hdfs dfs -stat [filepath]
```

#### 查看当前目录

```bash
bin/hdfs dfs -cwd
```

#### 修改文件权限

```bash
bin/hdfs dfs -chmod [-R] <permission> <filepath>
```

#### 显示当前用户名

```bash
bin/hdfs whoami
```

#### 格式化HDFS

```bash
bin/hdfs namenode -format
```

#### 设置配额限制

```bash
bin/hdfs dfsadmin -setQuota <quota> <dirname>
```

#### 增加配额限制

```bash
bin/hdfs dfsadmin -clrQuota <dirname>
```

#### 查看配额限制

```bash
bin/hdfs dfsadmin -getSpaceQuota <dirname>
```

#### 设置配额警告阀值

```bash
bin/hdfs dfsadmin -setSpaceQuotaWarning <warningsize> <dirname>
```