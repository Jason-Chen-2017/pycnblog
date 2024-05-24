
作者：禅与计算机程序设计艺术                    

# 1.简介
  

HDFS（Hadoop Distributed File System）是Apache基金会开发的一款分布式文件系统，由Apache Hadoop项目在2006年贡献给开源社区。HDFS被设计用来存储超大文件的分布式存储，能够处理多用户并发访问、海量数据存储等特性。HDFS通过高度抽象的数据块（Block），它能够保证数据的冗余性和可靠性。HDFS集群通常由一个NameNode和多个DataNode组成。NameNode负责管理整个文件系统的名称空间，客户端请求首先要查询NameNode获取元数据信息，然后再向对应的DataNode读写数据。DataNode存储实际的数据块。NameNode和DataNode通过心跳检测机制保持正常运行。如果NameNode或DataNode发生故障，则自动切换到另一台机器进行数据读写。HDFS具有高容错性和高可用性，并且具备高吞吐率，适合于大规模数据集的存储。
# 2.集群架构
HDFS集群分为两个角色——NameNode和DataNode。NameNode维护着整个文件系统的命名空间以及文件块的位置信息，同时也负责处理客户端对文件的各种操作请求。数据节点（DataNode）则存储实际的文件数据块。集群中可以有多个NameNode，但是只能有一个NameNode处于激活状态，其他的NameNode都是standby状态，主要用于故障转移。客户端应用程序可以通过名称服务器（DNS）或者Hadoop的URI（Uniform Resource Identifier）向不同的NameNode提出文件系统请求。
NameNode在启动时，会选举产生一个活动的NameNode，其余的都是standby状态。当出现失效NameNode时，系统会通过一个预先设定的时间周期进行自动故障切换。NameNode会定时发送心跳消息给所有的DataNode，告知自己仍然存活。如果超过一定时间内没有接收到DataNode的心跳，则认为该DataNode已经失败。同时，NameNode会将那些发生故障的DataNode上的 blocks 数据拷贝到另一台正常的DataNode上。
DataNode除了保存实际的文件数据之外，还负责数据块的读写。数据块是一个固定大小的、可以包含多个副本的存储单位。HDFS中的数据块默认为128MB，可以根据需要进行修改。DataNode会周期性地检查其本地磁盘上的数据块是否需要复制到其他数据节点上。如果某个数据块的副本数量过少，则会向NameNode请求增加副本的操作。如果某个数据块的复制因子过低，则会向NameNode请求减少副本的操作。
# 3.高可用配置
为了保证HDFS集群的高可用，HDFS提供了两种架构模式，即主/从模式和联邦模式。其中，主/从模式即一个NameNode（active NameNode）和多个DataNode（standby DataNode）。另一种模式，联邦模式，则是一个NameNode（active NameNode）和多个SecondaryNameNode（standby SecondaryNameNode）以及多个DataNode。第二种模式中，每个SecondaryNameNode都连接到一个从属于该SecondaryNameNode的DataNode上。这种模式能够提供更强大的容错能力，同时使得HDFS集群具备高可用性。
在实际生产环境中，建议使用联邦模式，其架构如下图所示：
在联邦模式下，SecondaryNameNode可以充当热备份，如果主NameNode失效，则会自动切换到SecondaryNameNode，从而实现HDFS集群的高可用。另外，由于SecondaryNameNode可以连接不同的DataNode，因此可以有效应对某些特殊场景下的读写压力。总体来说，联邦模式比主/从模式提供更高的容错能力和可用性。
# 4.搭建HDFS集群
本文将以主/从模式为例，详细阐述如何搭建一个HDFS集群。
## （1）前置条件
### 硬件需求
HDFS集群至少需要3台服务器，每台服务器至少需要配置两块硬盘，分别作为NameNode和DataNode的存储目录。最好还要准备一台单独的监控主机用于系统的监控和报警。当然，生产环境中，还需要考虑网络带宽、存储IOPS、网络延迟、服务器故障恢复时间等因素，做好集群的规划、部署和运维工作。
### 操作系统要求
目前HDFS支持的操作系统包括Linux和Windows Server。对于高可用集群来说，建议使用Linux操作系统。
### Java版本要求
HDFS需要Java开发环境才能编译和运行。推荐版本为JDK1.8及以上。
## （2）下载安装
### 安装OpenJDK
下载OpenJDK安装包，解压缩到指定目录。
```shell
wget https://download.java.net/openjdk/jdk8u312/ri/openjdk-8u312-b07-linux-x64-17_jan_2022.tar.gz
tar -zxvf openjdk-8u312-b07-linux-x64-17_jan_2022.tar.gz -C /opt/
```
设置环境变量
```shell
vim ~/.bashrc
export JAVA_HOME=/opt/jdk1.8.0_312
export PATH=$JAVA_HOME/bin:$PATH
source ~/.bashrc
```
验证Java版本
```shell
java -version
```
### 安装Hadoop
下载Hadoop安装包，解压缩到指定目录。
```shell
wget http://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/stable/hadoop-3.3.0.tar.gz
tar -zxvf hadoop-3.3.0.tar.gz -C /opt/
```
设置环境变量
```shell
vim ~/.bashrc
export HADOOP_HOME=/opt/hadoop-3.3.0
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH
source ~/.bashrc
```
验证Hadoop是否安装成功
```shell
hdfs version
```
## （3）配置集群
### 配置core-site.xml
编辑配置文件`conf/core-site.xml`，加入以下内容：
```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://hadoop1:9000/</value>
    </property>

    <!-- 指定副本因子 -->
    <property>
        <name>dfs.replication</name>
        <value>3</value>
    </property>
    
    <!-- 配置namenode地址 -->
    <property>
        <name>fs.default.name</name>
        <value>hdfs://hadoop1:9000</value>
    </property>
    
</configuration>
```
- `fs.defaultFS`: 设置默认的文件系统。
- `dfs.replication`: 设置副本因子。
- `fs.default.name`: 设置namenode地址。

### 配置hdfs-site.xml
编辑配置文件`conf/hdfs-site.xml`，加入以下内容：
```xml
<configuration>
    <!-- 配置namenode的地址 -->
    <property>
        <name>dfs.namenode.http-address</name>
        <value>hadoop1:50070</value>
    </property>

    <!-- 配置namenode的文件系统元数据路径 -->
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>/data/nn</value>
    </property>

    <!-- 配置datanode数据块的本地缓存路径 -->
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/data/dn</value>
    </property>
    
    <!-- 配置 SecondaryNameNode 的地址 -->
    <property>
        <name>dfs.secondary.http.address</name>
        <value>hadoop2:50090</value>
    </property>
</configuration>
```
- `dfs.namenode.http-address`: 设置namenode的HTTP服务地址，供客户端访问。
- `dfs.namenode.name.dir`: 设置namenode的元数据存储目录，多个目录使用逗号分隔。
- `dfs.datanode.data.dir`: 设置datanode的数据块存储目录，多个目录使用逗号分隔。
- `dfs.secondary.http.address`: 设置SecondaryNameNode的HTTP服务地址，供客户端访问。

### 配置slaves文件
编辑`etc/hadoop/slaves`文件，加入所有DataNode的主机名或IP地址，例如：
```text
hadoop1
hadoop2
hadoop3
```

### 创建文件夹
在各个DataNode上创建数据块的本地缓存路径。
```shell
mkdir -p /data/dn
chown -R hdfs:hdfs /data/dn
```
在namenode节点创建一个文件夹作为元数据存储路径。
```shell
mkdir -p /data/nn
chown -R hdfs:hdfs /data/nn
```
## （4）启动集群
### 格式化namenode
格式化namenode，在任意一台NameNode执行以下命令：
```shell
hdfs namenode -format
```
### 启动NameNode
在NameNode主机上执行以下命令启动NameNode进程：
```shell
start-dfs.sh
```
启动完成后，可以使用浏览器打开`http://hadoop1:50070/`查看namenode的状态，如果页面显示集群状态正常，则表示NameNode启动成功。

### 添加DataNode
在所有DataNode主机上执行以下命令添加DataNode：
```shell
ssh hadoop1 "sudo mkdir /data/dn"
ssh hadoop1 "sudo chown hdfs:hdfs /data/dn"
start-dfs.sh datanode
```
执行完成后，可以在浏览器中刷新namenode的状态页面，看到DataNode已经添加进集群。

### 启动SecondaryNameNode
SecondaryNameNode可以在主/从模式和联邦模式下独立启动，也可以和NameNode一起启动。这里仅以独立模式启动SecondaryNameNode，在任何一台机器上执行以下命令：
```shell
hdfs secondarynamenode
```
启动完成后，可以使用浏览器打开`http://hadoop2:50090/`查看SecondaryNameNode的状态，如果页面显示SecondaryNameNode正在运行，则表示SecondaryNameNode启动成功。

### 测试集群
配置完成后，即可测试集群是否正常工作。首先在任意一台NameNode上创建一个文件夹，然后将一些数据放入文件夹中，并验证写入是否成功。然后在所有DataNode上查看生成的数据块，验证数据块是否存在，且副本数是否达标。最后关闭集群的所有进程，等待集群完全停止。