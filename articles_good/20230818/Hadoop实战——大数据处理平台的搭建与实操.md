
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Hadoop是一个开源的分布式计算框架，可以用于存储海量的数据并进行分布式计算，是一个非常有用的工具。一般情况下，当我们需要对大量的数据进行分析时，我们会选择一些大数据处理工具来实现数据的存储、计算和分析。例如，Hive、Spark等。但是，如果我们需要构建一个真正意义上的大数据处理平台，那就需要多个组件协同工作才能完成任务。那么，如何构建一个真正能够提供海量数据的处理能力的大数据处理平台呢？下面我们一起走进这个世界吧！

# 2.准备工作
- 硬件环境：
  - 一台服务器或者多台服务器（推荐3个节点）
  - 操作系统：CentOS7以上版本
- 软件环境：
  - Java：Sun JDK或OpenJDK
  - Hadoop：官网下载安装即可
  - Zookeeper：官网下载安装即可
  - Hive：官网下载安装即可
  - Spark：官网下载安装即可
  
# 3.集群规划
Hadoop集群可以分为三层架构，即：

1. 第一层是NameNode，主要管理文件系统元数据，它保存了整个分布式文件系统的文件树结构；
2. 第二层是DataNode，负责存储数据块，它保存了文件的内容；
3. 第三层是中心调度器，主要管理集群中各个节点的资源，如分配内存和CPU资源。


# 4.安装配置Hadoop
## 安装Java
查看是否已经安装JDK：

```shell
[root@node1 ~]# java -version
java version "1.8.0_221"
Java(TM) SE Runtime Environment (build 1.8.0_221-b11)
Java HotSpot(TM) 64-Bit Server VM (build 25.221-b11, mixed mode)
```

如果没有安装，则安装OpenJDK：

```shell
yum install -y java-1.8.0-openjdk* 
```

设置Java环境变量：

```shell
[root@node1 ~]# vim /etc/profile
JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.222.b10-1.el7_7.x86_64
CLASSPATH=.:$JAVA_HOME/jre/lib/$CLASSPATH
PATH=$JAVA_HOME/bin:$PATH
export JAVA_HOME CLASSPATH PATH

source /etc/profile
```

## 配置Hadoop
下载Hadoop安装包到/opt目录下，解压：

```shell
mkdir ~/downloads && cd ~/downloads
wget https://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-3.1.3/hadoop-3.1.3.tar.gz
tar xzf hadoop-3.1.3.tar.gz -C /opt/
mv /opt/hadoop-3.1.3 /opt/hadoop
```

创建配置文件模板：

```shell
cd /opt/hadoop/etc/hadoop
cp mapred-site.xml.template mapred-site.xml
cp yarn-site.xml.template yarn-site.xml
cp core-site.xml.template core-site.xml
cp hdfs-site.xml.template hdfs-site.xml
```

编辑core-site.xml文件：

```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://node1:9000</value>
    </property>

    <!--指定HDFS的位置-->
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/var/run/hadoop</value>
        <description>A base for other temporary directories.</description>
    </property>
    
    <!--修改namenode地址-->
    <property>
        <name>fs.default.name</name>
        <value>hdfs://node1:9000</value>
    </property>
    
</configuration>
```

编辑hdfs-site.xml文件：

```xml
<configuration>
    <property>
        <name>dfs.data.dir</name>
        <value>/var/lib/hadoop-3.1.3/hdfs/datanode</value>
    </property>
    <property>
        <name>dfs.name.dir</name>
        <value>/var/lib/hadoop-3.1.3/hdfs/namenode</value>
    </property>
    <property>
        <name>dfs.replication</name>
        <value>2</value>
    </property>
</configuration>
```

编辑yarn-site.xml文件：

```xml
<?xml version="1.0"?>
<!--
     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

         http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
-->


<!-- Put site-specific property overrides in this file. -->

<configuration>

  <property>
    <name>yarn.resourcemanager.address</name>
    <value>${node1}:8032</value>
  </property>
  
  <!--开启资源预留功能-->  
  <property>
      <name>yarn.resourcemanager.scheduler.class</name>
      <value>org.apache.hadoop.yarn.server.resourcemanager.scheduler.capacity.CapacityScheduler</value>
  </property>
  
  <!--设定最大资源限制-->
  <property>
      <name>yarn.scheduler.capacity.maximum-am-resource-percent</name>
      <value>1</value>
  </property>
  
  <!--将各个队列的资源配额设置为默认值-->  
  <property>
      <name>yarn.scheduler.capacity.root.default.user-limit-factor</name>
      <value>1</value>
  </property>
  
  <property>
      <name>yarn.scheduler.capacity.root.acl_submit_applications</name>
      <value>*</value>
  </property>
  
  <property>
      <name>yarn.scheduler.capacity.root.acl_administer_queue</name>
      <value>*</value>
  </property>
  

  <!-- 设置队列 -->
  <property>
      <name>yarn.scheduler.capacity.root.queues</name>
      <value>default</value>
  </property>

  <!-- 给默认队列设置资源配额 -->
  <property>
      <name>yarn.scheduler.capacity.root.default.capacity</name>
      <value>100</value>
  </property>

   <!-- 提交应用程序所需资源的最小比例 --> 
  <property>
      <name>yarn.scheduler.minimum-allocation-mb</name>
      <value>256</value>
  </property>

  <!-- 每个节点上可运行的作业数 -->
  <property>
      <name>yarn.nodemanager.resource.memory-mb</name>
      <value>4096</value>
  </property>

  <!-- 给每个节点上的nm进程分配的虚拟内存 -->
  <property>
      <name>yarn.nodemanager.resource.cpu-vcores</name>
      <value>4</value>
  </property>
</configuration>
```

编辑mapred-site.xml文件：

```xml
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
</configuration>
```

配置完毕后，把它们复制到各个节点的/opt/hadoop/etc/hadoop目录下：

```shell
scp * node{2..3}:/opt/hadoop/etc/hadoop/
```

## 分发文件系统
为了使得Hadoop集群中的所有节点都能够共享相同的文件系统，我们需要分发文件系统到每台机器上。

### 创建共享目录
```shell
mkdir -p /data/nn
mkdir -p /data/sdb1
```

### 创建命名空间
```shell
sudo mke2fs -F /dev/sdb1
sudo mkdir -p /mnt/mydisk
sudo mount /dev/sdb1 /mnt/mydisk
```

### 分发文件系统
```shell
for i in $(seq 1 3); do 
    ssh root@node$i'mkdir /data'
    scp -r /mnt/mydisk/* root@node$i:/data; 
done
```

### 修改名称空间
```shell
for i in $(seq 1 3); do 
    ssh root@node$i 'echo "/dev/sdb1        /data          ext4    defaults        0 0" >> /etc/fstab'; 
    ssh root@node$i'mount -a'; 
done
```

## 初始化NameNode
在NameNode所在的节点上执行以下命令初始化NameNode：

```shell
cd /opt/hadoop/bin
./hdfs namenode -format # 格式化NameNode
./hdfs namenode # 启动NameNode
```

## 配置DataNode
在DataNode所在的节点上执行以下命令配置DataNode：

```shell
cd /opt/hadoop/bin
./hdfs datanode # 启动DataNode
```

## 检查状态
```shell
jps # 查看进程是否正常启动
http://node1:50070 # NameNode UI界面
http://node1:8088 # ResourceManager UI界面
```