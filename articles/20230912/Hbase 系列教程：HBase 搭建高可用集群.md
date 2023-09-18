
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Apache HBase 是 Apache 基金会开源项目之一，是一个分布式 NoSQL 数据库。它是一个可扩展的、面向列的、存储在 Hadoop 文件系统（HDFS）上的结构化数据存储。它支持 Hadoop 的 MapReduce 和它的周边生态系统，并且可以通过 Thrift 或 RESTful API 来访问。HBase 是基于 Google Bigtable 设计的。本文将介绍如何安装配置并搭建一个可靠的、高可用性的 HBase 集群。

什么是 HBase？

HBase 是 Apache 基金会旗下的一个开源 NoSQL 数据库。它是一个可扩展的、面向列的、存储在 HDFS 上面的结构化数据存储。HBase 支持 Hadoop MapReduce 和它的周边生态系统，并且可以通过 Thrift 或 RESTful API 来访问。HBase 是 Google BigTable 的开源实现，被誉为 Hadoop 的 NoSQL 数据存储。

为什么要用 HBase？

HBase 可以用来存储和处理大量的数据。HBase 提供了一个易于管理的分布式数据库，它可以像关系型数据库一样灵活地存储结构化和非结构化数据。对于海量数据的分析查询，HBase 具有出色的性能。由于 HBase 使用 HDFS，所以数据也是安全的。而且，HBase 提供了强大的容错能力，它可以在发生节点失败时自动修复数据。HBase 还可以用于实时分析查询。

# 2.基础知识

## 2.1 HDFS

HDFS (Hadoop Distributed File System) 是 Hadoop 所使用的底层文件系统。HDFS 是 Hadoop 的核心组件之一，它提供分布式文件系统存储，能够存储超大文件。HDFS 将数据分成一个个大小相等的分块，每个分块可以复制到不同的节点上，从而保证数据冗余备份和高可用。HDFS 的优点是，方便数据的存储、读取和共享；缺点是不适合随机写入和快速查询。

## 2.2 Zookeeper

ZooKeeper 是 Apache 基金会的一个开源分布式协调服务，是一个高效且 reliable 的分布式协调工具。HBase 需要使用 Zookeeper 来进行协调管理和维护。Zookeeper 本身功能简单，但是组合起来才能形成完整的分布式协调服务。Zookeeper 通过主/从模式对服务器进行分组，组内有一个leader节点负责服务调度和协调，组间采用 Paxos 协议进行多数派投票。HBase 在设计时就考虑到了可用性、一致性、容错性、可靠性。

## 2.3 Hadoop 体系架构

Hadoop 的体系架构由四个主要模块组成：HDFS、YARN、MapReduce、Hive。其中，HDFS 是 Hadoop 中的核心组件，负责存储和处理大数据集，同时也支持 MapReduce 计算框架。YARN 是 Hadoop 中资源管理器，它根据 MapReduce 的任务需求动态分配各个节点的资源。MapReduce 是 Hadoop 中计算框架，它基于分布式运算框架，可以轻松处理大数据量。Hive 是 Hadoop 中 SQL 查询引擎，它可以用来对 Hadoop 上的数据进行复杂的查询。 

# 3.安装配置

## 3.1 安装 Hadoop

下载 Hadoop 2.7.2: http://hadoop.apache.org/releases.html#download

选择下载最新的 stable release version。解压后，把 hadoop-2.7.2 重命名为 hadoop。

```bash
tar -zxvf apache-hadoop-2.7.2.tar.gz
mv apache-hadoop-2.7.2 hadoop
cd ~/hadoop
```

编辑配置文件 `etc/hadoop/core-site.xml`，添加以下信息：

```xml
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>

  <property>
    <name>hadoop.tmp.dir</name>
    <value>/home/yourname/hadoop_tmp</value> <!-- specify a temporary directory -->
  </property>
</configuration>
```

编辑配置文件 `etc/hadoop/hdfs-site.xml`，添加以下信息：

```xml
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value> <!-- replication factor for the file system -->
  </property>

  <property>
    <name>dfs.namenode.name.dir</name>
    <value>file:///home/yourname/hadoop_data/namenode</value> <!-- location of namenode directory -->
  </property>

  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file:///home/yourname/hadoop_data/datanode</value> <!-- location of datanodes -->
  </property>

  <property>
    <name>dfs.permissions</name>
    <value>false</value>
  </property>
</configuration>
```

最后，编辑配置文件 `etc/hadoop/mapred-site.xml`，添加以下信息：

```xml
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
  
  <property>
    <name>mapreduce.jobhistory.address</name>
    <value>localhost:10020</value>
  </property>
  
  <property>
    <name>mapreduce.jobhistory.webapp.address</name>
    <value>localhost:19888</value>
  </property>
  
  <property>
    <name>yarn.app.mapreduce.am.command-opts</name>
    <value>-Xmx4096m</value>
  </property>
</configuration>
```

创建临时文件夹：

```bash
mkdir /home/yourname/hadoop_tmp
chown yourname:hadoop_users /home/yourname/hadoop_tmp
chmod g+wrx /home/yourname/hadoop_tmp
```

启动 HDFS：

```bash
sbin/start-dfs.sh
```

启动 Yarn：

```bash
sbin/start-yarn.sh
```

验证 HDFS 是否正常运行：

```bash
bin/hdfs dfsadmin -report
```

## 3.2 配置 SSH 免密钥登录

为了实现集群中不同机器之间无密码登录，需要先生成密钥对：

```bash
ssh-keygen -t rsa -P '' # 一路回车，不要输入 passphrase
```

然后将 public key 拷贝到所有服务器的 `~/.ssh/authorized_keys` 文件里：

```bash
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```

最后，测试免密钥登录是否成功：

```bash
ssh localhost
```

如果连接成功，说明 SSH 免密钥登录已经配置成功。

## 3.3 安装 HBase

下载最新版的 HBase 发行包：http://hbase.apache.org/downloads.html

下载对应的版本的二进制文件。

```bash
wget http://archive.apache.org/dist/hbase/stable/hbase-1.2.5-bin.tar.gz
tar xzf hbase-1.2.5-bin.tar.gz
mv hbase-1.2.5 hbase
cd ~/hbase
```

修改配置文件 `conf/hbase-env.sh`，增加以下环境变量：

```bash
export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_151
export HBASE_OPTS="-XX:+UseConcMarkSweepGC"
```

编辑配置文件 `conf/hbase-site.xml`，添加以下信息：

```xml
<configuration>
  <property>
    <name>hbase.rootdir</name>
    <value>file:///home/yourname/hbase_data</value>
  </property>

  <property>
    <name>hbase.cluster.distributed</name>
    <value>true</value>
  </property>

  <property>
    <name>hbase.zookeeper.quorum</name>
    <value>localhost</value>
  </property>

  <property>
    <name>hbase.zookeeper.property.clientPort</name>
    <value>2181</value>
  </property>

  <property>
    <name>hbase.security.authentication</name>
    <value>simple</value>
  </property>

  <property>
    <name>hbase.security.authorization</name>
    <value>true</value>
  </property>
</configuration>
```

创建文件夹：

```bash
sudo mkdir /home/yourname/hbase_data
sudo chown yourname:hbase /home/yourname/hbase_data
```

启动 HBase：

```bash
bin/start-hbase.sh
```

验证 HBase 服务是否正常运行：

```bash
jps
```

# 4.搭建 HBase 集群

## 4.1 分布式集群

HBase 可以通过集群方式来提升系统的可用性和性能。HBase 有两种部署方式，分别是 Standalone 模式和 Distributed 集群模式。

Standalone 模式：

当集群规模比较小，不需要高度的可靠性和可用性时，可以使用 Standalone 模式。这种模式下，HBase 会运行在单个节点上，所有数据都存放在内存中。该模式没有经过任何优化，仅作为开发调试和学习目的使用。

Distributed 集群模式：

当集群规模达到一定程度时，建议使用 Distributed 模式。这种模式下，HBase 会运行在多个节点上，这些节点可以分布在不同的主机或者机架上，数据由 Zookeeper 管理。HBase 的 Master 和 RegionServer 角色分担了整个集群的工作负载。Master 节点用于管理元数据，RegionServer 节点用于存储实际的数据。HBase 集群中的节点需要通过 Zookeeper 协同工作。

## 4.2 高可用性

HBase 的 Master 和 RegionServer 一般都有很好的 HA（High Availability）机制，包括 Active-Standby 和 Hot-Standby 模式。

Active-Standby 模式：

这种模式下，HBase 的 Master 节点只有一个，其他节点都是 Slave。当 Master 节点故障时，Slave 节点自动成为新的 Master。

Hot-Standby 模式：

这种模式下，HBase 的 Master 和 RegionServer 都可以有多个实例，其中 Master 实例个数大于等于 3。当某个 Master 节点故障时，另一个 Master 会立即接管集群工作，确保集群继续工作。RegionServer 实例个数一般也设置成大于等于 3 个，这样就可以提高 HBase 可靠性。

## 4.3 配置 HBase 集群

### 4.3.1 配置 core-site.xml

编辑配置文件 `conf/core-site.xml`，添加以下信息：

```xml
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://namenodehost:8020</value>
  </property>

  <property>
    <name>hadoop.tmp.dir</name>
    <value>/home/yourname/hadoop_tmp</value>
  </property>

  <property>
    <name>ha.zookeeper.quorum</name>
    <value>zk1:2181,zk2:2181,zk3:2181</value>
  </property>

  <property>
    <name>hbase.rootdir</name>
    <value>hdfs://namenodehost:8020/apps/hbase/data</value>
  </property>

  <property>
    <name>hbase.cluster.distributed</name>
    <value>true</value>
  </property>

  <property>
    <name>hbase.regionserver.handler.count</name>
    <value>30</value>
  </property>

  <property>
    <name>hbase.security.authentication</name>
    <value>kerberos</value>
  </property>

  <property>
    <name>hbase.security.authorization</name>
    <value>true</value>
  </property>

  <property>
    <name>hbase.superuser</name>
    <value>hbase</value>
  </property>

  <property>
    <name>hbase.coprocessor.master.classes</name>
    <value></value>
  </property>

  <property>
    <name>hbase.coprocessor.region.classes</name>
    <value></value>
  </property>

  <property>
    <name>hbase.rpc.engine</name>
    <value>org.apache.hadoop.ipc.PhoenixRpcEngine</value>
  </property>

  <property>
    <name>hbase.region.server.rpc.scheduler.factory.class</name>
    <value>org.apache.hadoop.hbase.ipc.PhoenixRpcSchedulerFactory</value>
  </property>
</configuration>
```

### 4.3.2 配置 hdfs-site.xml

编辑配置文件 `conf/hdfs-site.xml`，添加以下信息：

```xml
<configuration>
  <property>
    <name>dfs.nameservices</name>
    <value>mycluster</value>
  </property>

  <property>
    <name>dfs.ha.automatic-failover.enabled</name>
    <value>true</value>
  </property>

  <property>
    <name>dfs.journalnode.edits.dir</name>
    <value>qjournal://zk1:8485;zk2:8485;zk3:8485/mycluster</value>
  </property>

  <property>
    <name>dfs.namenode.rpc-address.mycluster.nn1</name>
    <value>namenodehost1:8020</value>
  </property>

  <property>
    <name>dfs.namenode.rpc-address.mycluster.nn2</name>
    <value>namenodehost2:8020</value>
  </property>

  <property>
    <name>dfs.client.failover.proxy.provider.mycluster</name>
    <value>org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider</value>
  </property>
</configuration>
```

### 4.3.3 配置 mapred-site.xml

编辑配置文件 `conf/mapred-site.xml`，添加以下信息：

```xml
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>

  <property>
    <name>yarn.resourcemanager.address</name>
    <value>rmhost1:8032</value>
  </property>

  <property>
    <name>yarn.resourcemanager.scheduler.address</name>
    <value>rmhost1:8030</value>
  </property>

  <property>
    <name>yarn.resourcemanager.resource-tracker.address</name>
    <value>rmhost1:8031</value>
  </property>

  <property>
    <name>yarn.resourcemanager.admin.address</name>
    <value>rmhost1:8033</value>
  </property>

  <property>
    <name>yarn.resourcemanager.hostname</name>
    <value>rmhost1</value>
  </property>

  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>

  <property>
    <name>yarn.nodemanager.vmem-check-enabled</name>
    <value>false</value>
  </property>

  <property>
    <name>yarn.scheduler.maximum-allocation-mb</name>
    <value>10240</value>
  </property>

  <property>
    <name>yarn.scheduler.minimum-allocation-mb</name>
    <value>256</value>
  </property>

  <property>
    <name>yarn.scheduler.increment-allocation-mb</name>
    <value>256</value>
  </property>

  <property>
    <name>mapreduce.jobhistory.address</name>
    <value>jhs1:10020</value>
  </property>

  <property>
    <name>mapreduce.jobhistory.webapp.address</name>
    <value>jhs1:19888</value>
  </property>

  <property>
    <name>yarn.log-aggregation-enable</name>
    <value>true</value>
  </property>
</configuration>
```

## 4.4 配置 Kerberos

如果 HBase 需要支持 Kerberos 认证，则需要在集群中安装并配置 Kerberos 相关组件。

首先，安装 Kerberos 服务：

```bash
yum install krb5-server krb5-libs krb5-auth-dialog
```

配置 KDC 服务：

```bash
vim /var/kerberos/krb5kdc/kdc.conf
[kdcdefaults]
# default realm
default_realm = EXAMPLE.COM

[realms]
	EXAMPLE.COM = {
		kdc = kerberos.example.com
        admin_server = kerberos.example.com
	}

[domain_realm]
	.example.com = EXAMPLE.COM
	example.com = EXAMPLE.COM


[dbmodules]
	oracle = {
		application_name = kdc
	}

[logging]
	 kdc = FILE:/var/log/krb5kdc.log
	default = FILE:/var/log/krb5lib.log
	kdc_conns = FILE:/var/log/kadmind.log


systemctl start krb5kdc
systemctl enable krb5kdc
```

配置 Kerberos 客户端：

```bash
vim /etc/krb5.conf 
[libdefaults]
	dns_lookup_realm = false
	dns_lookup_kdc = false
	ticket_lifetime = 24h
	forwardable = true

[realms]
	EXAMPLE.COM = {
		kdc = kerberos.example.com
	}

[domain_realm]
	.example.com = EXAMPLE.COM
	example.com = EXAMPLE.COM
```

创建管理员账号：

```bash
kadmin.local
addprinc root/admin@EXAMPLE.COM
exit
```

创建 HBase 服务账号：

```bash
kadmin.local
addprinc hbase/admin@EXAMPLE.COM
ktadd -k /etc/security/keytabs/hbase.service.keytab hbase/admin@EXAMPLE.COM
exit
```

将服务账号的 keytab 文件拷贝到所有的 HBase 节点的 `/etc/security/keytabs/` 目录下。

## 4.5 配置 RegionServers

编辑配置文件 `conf/regionservers`，添加以下信息：

```bash
rs1
rs2
rs3
```

## 4.6 检查配置

检查 HDFS 文件系统的空间是否充足，因为 HBase 集群的所有数据都存放在 HDFS 上。

确认 NameNode 进程正在运行：

```bash
jps | grep NameNode
```

确认 DataNode 进程正在运行：

```bash
jps | grep DataNode
```

确认 JournalNode 进程正在运行：

```bash
jps | grep JournalNode
```

确认 Zookeeper 进程正在运行：

```bash
jps | grep QuorumPeerMain
```

确认 ResourceManager 进程正在运行：

```bash
jps | grep ResourceManager
```

确认 NodeManager 进程正在运行：

```bash
jps | grep NodeManager
```

确认 HBase master 进程正在运行：

```bash
jps | grep HMaster
```

确认 HBase regionserver 进程正在运行：

```bash
jps | grep HRegionServer
```

确认 MapReduce HistoryServer 进程正在运行：

```bash
jps | grep JobHistoryServer
```

确认 JMXMP 监控代理进程正在运行：

```bash
jps | grep JMXMP
```

确认 HBase UI 可访问。

# 5.测试集群

## 5.1 创建表格

进入 HMaster 页面，创建一个名为 "test" 的表格。

```bash
http://namenodehost:16010/master-status
```

点击 "test" 右侧的 "Actions" -> "Create Table...":


设置如下参数：

- Row Key 编码：UTF-8
- Column Family：info
- Column Qualifier：name, age, salary
- Number of Regions：3
- Split Keys：留空

点击 "Save" 按钮即可。

## 5.2 往表格插入记录

进入 HMaster 页面，选中刚才创建的 "test" 表格，点击 "Actions" -> "Put..."：


输入一条记录：

- Row Key：zhangsan
- Cells：info:name=张三, info:age=25, info:salary=5000

点击 "Put" 按钮即可。

再次输入一条记录：

- Row Key：wangwu
- Cells：info:name=王五, info:age=30, info:salary=6000

点击 "Put" 按钮即可。

再次输入一条记录：

- Row Key：zhaoliu
- Cells：info:name=赵六, info:age=35, info:salary=7000

点击 "Put" 按钮即可。

## 5.3 查询记录

打开浏览器，输入以下 URL 查看刚才插入的记录：

```bash
http://namenodehost:16010/test/row?row=zhangsan&filter=All%20Columns
http://namenodehost:16010/test/row?row=wangwu&filter=All%20Columns
http://namenodehost:16010/test/row?row=zhaoliu&filter=All%20Columns
```

查看结果：
