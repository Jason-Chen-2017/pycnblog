
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Hadoop是一个开源的分布式计算框架，可以对大型数据进行分布式处理，能够高效地存储海量数据并提供高计算能力。它主要由两个子系统构成：HDFS(Hadoop Distributed File System)文件系统和MapReduce计算引擎。本文将介绍如何在云服务器上安装Hadoop环境。
# 2.基本概念术语说明
## 2.1 HDFS概述
HDFS（Hadoop Distributed File System）是Apache基金会旗下的开源分布式文件系统。它是一个高容错性、高吞吐量的文件系统，适合存储海量的数据，同时也具有很强的扩展能力。HDFS通过它独特的设计可以支持大文件的存储，而不需要切分和拆分，因此提供了高容错性。HDFS采用主从结构，一个HDFS集群包括一个NameNode和多个DataNode。其中，NameNode负责维护文件系统的元数据，如目录树、块映射表等；DataNode则存储实际的数据块。HDFS可以提供高吞吐量的数据访问服务，因为它可以充分利用集群中的多台服务器之间网络带宽。目前，HDFS已被许多大型互联网公司和大数据平台所使用，例如YouTube、Facebook、Twitter、Netflix、Amazon等。
## 2.2 MapReduce概述
MapReduce是一个编程模型和运行框架，它用于大规模数据的并行运算。它最初由Google开发，用于自然语言搜索，之后逐渐扩展到其他领域，如图形处理、社交网络分析、机器学习、搜索推荐等。MapReduce程序由两部分组成：Mapper和Reducer。Mapper阶段接收输入数据并产生键值对；Reducer阶段根据键值对进行局部聚合，并将结果输出。该框架能够自动进行数据分片和数据排序，有效避免了数据集过于庞大的情况下内存耗尽的问题。由于MapReduce框架十分简单易用，因此在当今很多计算框架中都使用其作为底层引擎。
# 3. Hadoop集群搭建
## 3.1 Hadoop单机模式
Hadoop单机模式下只有NameNode和DataNode两个角色，NameNode管理整个文件的元数据，DataNode存储具体的数据块。这种模式虽然较为简单，但是对于测试或者学习hadoop的初学者来说比较方便。
### 3.1.1 安装准备工作
为了让hadoop运行起来，需要以下几个组件：

1. java环境，一般选择jdk1.8版本以上。
2. ssh客户端，用于远程登录集群。
3. Hadoop压缩包，下载地址：http://archive.apache.org/dist/hadoop/common/stable/ 。
4. Hadoop配置文件，一般配置在/etc/hadoop/目录下。
5. 文件系统，这里使用本地文件系统。

为了方便，这里假设所有hadoop相关的文件都存放在用户目录下，即$HOME路径下。

### 3.1.2 配置文件说明
首先是core-site.xml文件，这个文件里配置了hdfs的通用属性，比如namenode主机名、用户名等信息。
``` xml
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>file:///</value>
    </property>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>$HADOOP_HOME/temp</value>
    </property>
</configuration>
```

然后是mapred-site.xml文件，这个文件里配置了mrjob任务调度器的属性，比如任务的最大内存限制等。
``` xml
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<!-- Put site-specific property overrides in this file. -->

<!DOCTYPE configuration SYSTEM "configuration.dtd">
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>local</value>
    <!-- run the framework as a local command -->
  </property>
  <property>
    <name>mapred.jobtracker.maxtasks.per.job</name>
    <value>-1</value>
    <!-- set to unlimited for testing -->
  </property>
  <property>
    <name>mapreduce.task.timeout</name>
    <value>600000</value>
    <!-- allow tasks to take up to ten minutes -->
  </property>
  <property>
    <name>mapreduce.job.acl-modify-events</name>
    <value>*</value>
    <!-- permit all users to modify jobs -->
  </property>
  <property>
    <name>mapreduce.cluster.acls.enabled</name>
    <value>true</value>
    <!-- enable job access control list checking -->
  </property>
  <property>
    <name>mapreduce.jobhistory.address</name>
    <value></value>
    <!-- disable history server -->
  </property>
  <property>
    <name>mapreduce.jobhistory.webapp.address</name>
    <value></value>
    <!-- disable history server web UI -->
  </property>
</configuration>
```

最后是hdfs-site.xml文件，这个文件里配置了hdfs的详细参数，包括副本数量、块大小等。
``` xml
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<!-- Put site-specific property overrides in this file. -->

<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
    <!-- default block replication -->
  </property>
  <property>
    <name>dfs.blocksize</name>
    <value>134217728</value>
    <!-- default block size -->
  </property>
  <property>
    <name>dfs.permissions</name>
    <value>false</value>
    <!-- enable permissions -->
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>/data/hdfs/data</value>
    <!-- data directories on each DataNode -->
  </property>
</configuration>
```

### 3.1.3 NameNode启动命令
首先切换到$HADOOP_HOME/bin目录下执行如下命令启动NameNode:
``` shell
./start-dfs.sh
```

执行后控制台会打印出如下信息：
``` log
Starting namenodes on [localhost]
Starting datanodes
Starting secondary namenodes [0.0.0.0]
Starting journal nodes [0.0.0.0]
Starting ZKFC
```

### 3.1.4 DataNode启动命令
接着切换到$HADOOP_HOME/sbin目录下执行如下命令启动DataNode:
``` shell
./start-dfs.sh
```

执行后控制台会打印出如下信息：
``` log
Starting datanodes
```

### 3.1.5 浏览器访问NameNode
打开浏览器，输入http://localhost:9870，如果出现web页面，证明NameNode已经正常启动。


## 3.2 Hadoop伪分布式模式
伪分布式模式是一种小型的Hadoop集群，在单个服务器上运行HDFS和MapReduce，适合学习和开发。伪分布式模式下只有一个角色：Master，它是NameNode和JobTracker的集合。
### 3.2.1 安装准备工作
同上面的单机模式安装准备工作一样，需要java环境、ssh客户端、Hadoop压缩包、配置文件、文件系统。这里假设所有hadoop相关的文件都存放在用户目录下，即$HOME路径下。

另外还需修改配置文件，分别修改core-site.xml、hdfs-site.xml、mapred-site.xml。

### 3.2.2 Master节点启动命令
首先切换到$HADOOP_HOME/bin目录下执行如下命令启动Master:
``` shell
./start-all.sh
```

执行后控制台会打印出如下信息：
``` log
Starting namenodes on [localhost]
Starting zookeeper... binding to port 2181
Starting JobTracker... binding to port 0.0.0.0:8021
Starting TaskTracker
Starting Datanodes
Starting secondary namenodes [0.0.0.0]
Starting JournalNodes [0.0.0.0]
Starting NFS gateway
```

### 3.2.3 浏览器访问Web页面
打开浏览器，输入http://localhost:9870，如果出现web页面，证明Master节点已经正常启动。
