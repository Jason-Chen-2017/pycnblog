
作者：禅与计算机程序设计艺术                    

# 1.简介
  

HDFS（Hadoop Distributed File System）是一个分布式文件系统，它是一个非常高效的存储数据的方式，适用于高吞吐量的数据分析任务。HDFS通过提供容错能力、scalability以及高可用性，能够快速地处理TB、PB级别的数据集。 Hadoop生态系统中最重要的组件之一就是HDFS。本文将介绍如何在ubuntu上安装并配置一个HDFS集群，并用Python进行一些简单的文件操作。文章假设读者对Linux环境及Python编程有基本了解。
# 2.相关术语及概念
## HDFS: Hadoop Distributed File System
HDFS是一个开源的分布式文件系统，由Apache基金会开发维护。HDFS被设计用来存储大规模的数据集，它支持数据冗余备份，能够自动读取数据，并保证数据安全和完整性。HDFS采用主/从结构，其中一个NameNode节点管理文件系统元数据，而其它DataNode节点存储实际的数据块。
## NameNode: 管理文件系统元数据的主节点
NameNode负责管理文件系统的命名空间（namespace），它记录了文件的位置信息，并且可以确定哪些DataNode存储属于同一个文件的数据块。NameNode除了管理文件系统的命名空间外，还管理着整个系统的块的信息和块位置信息。当客户端提交文件的请求时，NameNode会确认DataNode是否可用的情况下，才将文件存入相应的DataNode节点。
## DataNode: 实际保存文件的节点
DataNode是HDFS中的工作节点，它存储实际的数据块，并提供数据服务。DataNode接收来自客户机或者其他DataNode的写请求，将数据写入本地磁盘，并向NameNode报告已上传的块信息。
## Block: 数据块
HDFS上的数据都被分割成固定大小的Block，默认大小为64MB。一个Block是一个连续的字节序列，数据块存储在DataNode中。每个文件被划分成多个数据块，这些数据块按照其在文件中的偏移量存储。HDFS中允许多个文件共享相同的数据块，这样就减少了数据的重复利用。
## Secondary NameNode(s): 支持读操作的辅助节点
Secondary NameNode是为了确保HA（High Availability）架构中的NameNode故障切换的辅助角色。如果NameNode出现故障，则Secondary NameNode将接管HDFS集群的写操作，而不会影响读操作。Secondary NameNode也可能同时服务于读取操作。Secondary NameNode会定期将自己的状态快照（snapshot）发送给所有的NameNode，以确保它们之间的数据一致性。

# 3.部署过程
## 安装环境准备
首先，需要在机器上安装Java运行环境。Ubuntu默认已经安装OpenJDK-7或Oracle Java 8。确认java版本：
```
java -version
```
如果版本号小于1.8，则需要手动升级到最新版。

然后安装SSH Server，方便后面连接HDFS集群。
```
sudo apt-get install ssh
```
安装完毕后，启动SSH Server：
```
sudo service ssh start
```
设置SSH免密登录。为了使得所有主机都信任这台机器作为SSH Server，我们可以通过编辑/etc/ssh/sshd_config文件，找到这一行：
```
RSAAuthentication yes
```
修改为：
```
RSAAuthentication no
```
然后重启SSH Server：
```
sudo service ssh restart
```
新建用户。由于HDFS通常运行在多台服务器上，所以我们建议创建单独的hdfs用户组，再创建一个hdfs账户。
```
groupadd hdfs
useradd hdfs -g hdfs -d /home/hdfs -m
passwd hdfs # 设置密码
```
确保/home/hdfs目录属于hdfs用户：
```
chown hdfs:hdfs /home/hdfs
```
创建SSH Key。为了实现免密登录，需要建立SSH Key并分发到各个机器上。
```
su hdfs
cd ~
ssh-keygen -t rsa
cat.ssh/id_rsa.pub >>.ssh/authorized_keys
chmod 600.ssh/*
```
最后，关闭防火墙（可选）。为了避免集群之间通信受到干扰，建议关闭防火墙。
```
sudo ufw disable
```
## 配置Namenode
首先，切换到hdfs用户：
```
su hdfs
```
创建配置文件夹：
```
mkdir ~/hadoop
mkdir ~/hadoop/conf
```
下载解压安装包。从官网下载解压后的hadoop安装包：http://mirror.apache-kr.org/hadoop/common/stable/
```
tar xzf hadoop-X.Y.Z.tar.gz
mv hadoop-X.Y.Z ~/hadoop
```
拷贝配置文件。为了方便管理配置，将配置文件拷贝到~/hadoop/conf目录下：
```
cp ~/hadoop/etc/hadoop/*.xml ~/hadoop/conf/
```
编辑core-site.xml文件。打开文件：
```
vi ~/hadoop/conf/core-site.xml
```
添加以下内容：
```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://namenode:9000/</value> <!-- 这里需要修改成你的主机名 -->
    </property>

    <property>
        <name>ha.zookeeper.quorum</name>
        <value>zk1:2181,zk2:2181,zk3:2181</value> <!-- 此处填写ZooKeeper集群的主机名 -->
    </property>

    <property>
        <name>ha.zookeeper.parent-znode</name>
        <value>/hbase-unsecure</value> <!-- 随意指定，后面会讲到 -->
    </property>
</configuration>
```
注意，此处的主机名，只能填入主机名或IP地址，不能填入localhost或127.0.0.1，否则会导致无法解析域名。如，我的主机名为hadoop01，那么ha.zookeeper.quorum的值应该设置为：
```xml
<value>hadoop01:2181,hadoop02:2181,hadoop03:2181</value>
```
编辑hdfs-site.xml文件。打开文件：
```
vi ~/hadoop/conf/hdfs-site.xml
```
添加以下内容：
```xml
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>

    <property>
        <name>dfs.name.dir</name>
        <value>/home/hdfs/hadoop/namenode</value>
    </property>

    <property>
        <name>dfs.data.dir</name>
        <value>/home/hdfs/hadoop/datanode</value>
    </property>

    <property>
        <name>dfs.client.use.datanode.hostname</name>
        <value>true</value>
    </property>
</configuration>
```
最后，编辑slaves文件。此文件记录了所有DataNode的主机名或IP地址，每行一个。打开文件：
```
vi ~/hadoop/conf/slaves
```
添加以下内容：
```
datanode1
datanode2
datanode3
```
注意，主机名和IP地址都可以，但不要写localhost或127.0.0.1，否则会导致无法解析域名。保存并退出。
## 创建文件系统
在hdfs主节点（即NameNode所在机器）上，切换到hdfs用户：
```
su hdfs
```
进入hadoop目录：
```
cd ~/hadoop
```
格式化文件系统。此命令会删除之前格式化时产生的所有数据。因此，除非确定不需要保留任何数据，否则应谨慎执行此命令：
```
bin/hdfs namenode -format
```
启动NameNode进程。由于HDFS是一种高吞吐量的分布式文件系统，因此建议开启多个DataNodes。此命令在后台运行：
```
sbin/start-dfs.sh
```
查看集群状态。等待几秒钟，查看结果：
```
jps
```
如果看到如下输出，则表示NameNode进程正常运行：
```
1293 Namenode
1321 DataNode
```
## 启动Zookeeper
前面提到了需要配置core-site.xml文件，其中ha.zookeeper.quorum属性值需要填写ZooKeeper集群的主机名。这里假设我们有三台ZooKeeper服务器，分别是zk1、zk2、zk3。如果要启动三台ZooKeeper服务器，请确保三台服务器之间可以相互通信。
### 安装ZooKeeper
建议安装二进制包，因为源码安装容易出错。我们可以从官网下载ZooKeeper安装包：https://zookeeper.apache.org/releases.html
```
wget https://archive.apache.org/dist/zookeeper/stable/apache-zookeeper-3.4.14.tar.gz
tar xzf apache-zookeeper-3.4.14.tar.gz
mv apache-zookeeper-3.4.14 zookeeper-3.4.14
```
### 配置ZooKeeper
编辑zoo.cfg文件。打开文件：
```
vi zookeeper-3.4.14/conf/zoo.cfg
```
修改内容如下：
```
tickTime=2000
initLimit=10
syncLimit=5
dataDir=/var/lib/zookeeper
clientPort=2181
server.1=zk1:2888:3888
server.2=zk2:2888:3888
server.3=zk3:2888:3888
```
启动ZooKeeper服务器。在zookeeper-3.4.14目录下执行：
```
./bin/zkServer.sh start
```
查看日志。打开另一个终端窗口，切换到zookeeper-3.4.14目录，执行：
```
tail -f logs/zkServer.out
```
如果看到如下输出，则表示ZooKeeper服务器正常运行：
```
Starting ZooKeeper server
zookeeper version: 3.4.14-cdh5.13.3--1, built on 06/29/2018 15:50 GMT
Clients request URLs through ZooKeeper server
Openning socket connection to server zk1/192.168.0.20:2181. Will not attempt to authenticate using SASL (unknown error)
Opening socket connection to server zk2/192.168.0.21:2181. Will not attempt to authenticate using SASL (unknown error)
Openning socket connection to server zk3/192.168.0.22:2181. Will not attempt to authenticate using SASL (unknown error)
Created session with id 0x167c9a4c62e000b for client /192.168.0.18:56184 who are sending requests at 0x167c9a4cf3a0000
...
...
```
## 配置secondary namenode（可选）
为了确保HA架构中的NameNode故障切换，可以设置一个secondary namenode。secondary namenode只提供写操作，不提供读操作。
编辑hdfs-site.xml文件。打开文件：
```
vi ~/hadoop/conf/hdfs-site.xml
```
添加以下内容：
```xml
<property>
  <name>dfs.secondary.http.address</name>
  <value>hadoop02:50090</value>
</property>

<property>
  <name>dfs.namenode.checkpoint.dir</name>
  <value>/home/hdfs/hadoop/secondary-namenode/checkpoints</value>
</property>
```
在hadoop主节点上，编辑slaves文件，添加secondary namenode的主机名。例如：
```
vi ~/hadoop/conf/slaves
```
```
datanode1
datanode2
datanode3
hadoop02
```
启动secondary namenode。在secondary namenode的主机上，执行：
```
sbin/start-secondary.sh
```
停止NameNode。在hadoop主节点上，执行：
```
sbin/stop-dfs.sh
```
启动secondary namenode。在secondary namenode的主机上，执行：
```
sbin/start-dfs.sh
```
查看集群状态。等待几秒钟，查看结果：
```
jps
```
如果看到如下输出，则表示secondary namenode正常运行：
```
2069 DataNode
2115 NameNode
2120 Jps
2143 SecondaryNameNode
```
## 配置Web UI（可选）
HDFS支持通过Web页面查看系统状态。启用Web UI的方法很简单。编辑core-site.xml文件。打开文件：
```
vi ~/hadoop/conf/core-site.xml
```
添加以下内容：
```xml
<property>
  <name>webhdfs.enabled</name>
  <value>true</value>
</property>

<property>
  <name>hadoop.tmp.dir</name>
  <value>/home/hdfs/hadoop/temp</value>
</property>

<property>
  <name>dfs.permissions</name>
  <value>false</value>
</property>
```
启动Web UI。在hadoop主节点上，执行：
```
mr-jobhistory-daemon.sh start historyserver
```
查看Web UI。打开浏览器，输入http://hadoop01:50070 ，即可查看HDFS Web UI。

至此，HDFS集群已经搭建完成，可以使用Python操作HDFS文件系统。