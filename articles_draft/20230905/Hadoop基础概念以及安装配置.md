
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Apache Hadoop（下称Hadoop）是一个开源的分布式计算框架。它由Apache基金会开发并维护，并在Apache许可证下开源。它的设计目标是为了能够对大规模数据集进行高速计算，具有高容错性、可靠性、扩展性等特性。Hadoop生态系统包括Java、MapReduce、HDFS、YARN、Hive、Pig、Spark等组件。本文主要讨论Hadoop的一些基本概念及其安装配置方法。

# 2.什么是Hadoop？
Hadoop是一种分布式存储、并行计算和分布式文件系统，最初起源于Apache Nutch项目。它具有以下几个主要特性：

1. 高容错性（Fault-tolerant）：集群中的节点随时都可能出现故障，Hadoop通过自带的容错机制保证了数据的安全性。
2. 可靠性（Reliable）：Hadoop提供了数据冗余备份功能，确保数据不丢失。
3. 弹性可扩展性（Scalable）：集群可以根据需要自动增加或减少节点。
4. 分布式计算（Distributed computing）：Hadoop基于分布式计算模型，支持海量数据的处理。
5. HDFS（Hadoop Distributed File System）：Hadoop的文件系统，具有高容错性和可靠性。
6. MapReduce（Massive parallel processing）：一个编程模型，用于将海量的数据集切分成独立的片段，并并行处理。
7. YARN（Yet Another Resource Negotiator）：一种资源管理系统，负责分配集群中各个节点上的资源。
8. Hive（Hadoop SQL Query Engine）：一种SQL查询引擎，能够运行复杂的分析任务。
9. Pig（Pig Latin）：一种基于MapReduce的脚本语言，能够运行分布式数据处理任务。
10. Spark（Apache Spark）：一种快速、通用、易用且开源的大数据分析引擎。

# 3.Hadoop安装配置
## 3.1 安装准备
Hadoop安装依赖jdk，请下载java sdk并安装：https://www.oracle.com/java/technologies/javase-downloads.html ，注意选择合适的版本。

## 3.2 下载Hadoop安装包
Hadoop官网：http://hadoop.apache.org/

Hadoop下载页面：http://hadoop.apache.org/releases.html

选择最新版本进行下载：http://mirror.nbtelecom.com.br/apache/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz （镜像站点）

```shell
wget http://mirror.nbtelecom.com.br/apache/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz
```

## 3.3 配置环境变量
编辑`~/.bashrc`文件，添加如下配置：

```bash
export JAVA_HOME=/usr/local/jdk # java SDK路径
export PATH=$PATH:$JAVA_HOME/bin   # 添加到PATH环境变量中
```

使环境变量立即生效：

```bash
source ~/.bashrc
```

## 3.4 创建Hadoop目录结构
创建hadoop根目录：

```bash
mkdir /opt/hadoop
```

然后解压下载好的hadoop安装包至上述目录：

```bash
cd /opt/hadoop
tar -zxvf ~/下载/hadoop-3.3.1.tar.gz --strip-components=1
```

然后设置Hadoop配置文件的环境变量：

```bash
export HADOOP_HOME=/opt/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
```

创建配置文件目录：

```bash
mkdir $HADOOP_HOME/etc/hadoop
```

## 3.5 配置Hadoop参数
配置文件路径为`$HADOOP_HOME/etc/hadoop/`，主要配置项有：

1. core-site.xml：包含HDFS和YARN共同的基础配置。
2. hdfs-site.xml：HDFS的相关配置，如副本数量、块大小、地址信息等。
3. mapred-site.xml：MapReduce的相关配置，如TaskTracker和JobHistory服务器的个数、切分阈值、执行器内存限制等。
4. yarn-site.xml：YARN的相关配置，如RM、NM的个数、日志存放位置、队列的划分规则等。

这里只对core-site.xml进行简单介绍。

core-site.xml一般在$HADOOP_HOME/etc/hadoop目录下。

### 3.5.1 设置NameNode地址
由于我们的Hadoop集群只有一个NameNode节点，所以我们不需要配置这个文件。

```xml
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value> <!-- NameNode地址 -->
  </property>
 ...
</configuration>
```

### 3.5.2 设置HDFS存储目录
指定HDFS存储文件的位置，默认情况下，Hadoop会把数据放在HDFS的data/ directory中，如果需要修改这个配置，可以在该文件中设置：

```xml
<configuration>
  <property>
    <name>hadoop.tmp.dir</name>
    <value>/opt/hadoop/tmp</value> <!-- 指定临时文件目录 -->
  </property>
 ...
</configuration>
```

也可以使用默认配置：

```xml
<configuration>
  <property>
    <name>hadoop.tmp.dir</name>
    <value>${io.file.tmpdir}/hadoop-${user.name}</value> 
  </property>
 ...
</configuration>
```

### 3.5.3 设置本地磁盘映射目录
可以指定多个磁盘目录作为DataNode的本地磁盘空间。这样当某个节点出现故障时，HDFS会自动识别到它，而不会影响其他节点的数据。

```xml
<configuration>
  <property>
    <name>hadoop.tmp.dir</name>
    <value>/opt/hadoop/tmp</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file:///opt/hadoop/hdfs/data1,/opt/hadoop/hdfs/data2</value> <!-- 指定DataNode数据目录 -->
  </property>
 ...
</configuration>
```

### 3.5.4 设置Hadoop参数
除了上面三个主要配置之外，还有很多参数可以使用，比如：

1. dfs.replication：指定HDFS副本数量，默认为3。
2. dfs.blocksize：指定HDFS块的大小，默认为128M。
3. hadoop.log.dir：指定日志文件的存储目录。
4. mapreduce.framework.name：指定MapReduce框架，默认值为local，表示单机模式。
5. io.compression.codecs：压缩Codec参数。

## 3.6 启动Hadoop
启动命令如下：

```bash
sbin/start-all.sh    # 在后台启动所有进程
jps                 # 查看进程是否正常启动
```

如果出现“successful”字样，则表明所有进程都已正常启动。可以通过Web界面查看服务状态：http://localhost:50070 。