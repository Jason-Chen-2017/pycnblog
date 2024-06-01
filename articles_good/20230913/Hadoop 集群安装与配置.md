
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Hadoop 是一个开源的分布式计算框架。它是一个云计算平台基础组件之一，被多个知名公司广泛应用。包括百度、阿里巴巴、腾讯、京东、微软等互联网巨头在内的多家公司都使用 Hadoop 来存储海量数据，进行实时分析，处理海量数据流等。

本文将通过详细地阐述 Hadoop 的安装部署过程和相关配置参数，让读者可以快速、清晰地掌握 Hadoop 的安装部署方法，并最终对 Hadoop 的运行机制有一个全面的认识。文章中将重点关注 Hadoop 的安装部署，以及典型应用场景下的配置优化。

# 2.环境准备
## 2.1 操作系统版本及硬件配置要求
Hadoop 是基于 Java 的开发，因此首先需要确认电脑上是否已经安装了 Oracle JDK 或 OpenJDK 开发工具包。同时还要确定 Hadoop 安装包下载地址和安装路径。不同版本操作系统的兼容性要求也会影响到 Hadoop 的安装部署。

Hadoop 在部署前需要做好以下准备工作：

1. 准备好操作系统（推荐 CentOS 7）
2. 配置操作系统防火墙
3. 配置主机名解析（可选）
4. 配置 hosts 文件
5. 安装 SSH 服务（可选）
6. 配置SSH免密码登录（可选）
7. 配置 SSH Key免密登录（可选）
8. 安装必要的依赖库（如:sudo yum -y install openssh-server)
9. 计划安装的节点数量和内存大小
10. 分配合适的磁盘空间
11. 配置网络和 IP 地址

根据实际情况，选择合适的节点规模和资源配置。例如，假设我们计划安装 Hadoop 3.x 版本，并且节点规模为 5 个，每个节点的内存为 4G 左右，硬盘大小为 100G。那么我们至少需要准备如下资源：

5 x (4GB + 100GB) = 200GB + 200GB = 400GB SSD/SAS 固态硬盘 + 400GB 机械硬盘，或者 1TB SAS 固态硬盘。

而这个配置仅仅是一种建议值，具体多少需要根据安装的节点规模和机器性能而定。

## 2.2 安装工具介绍
Hadoop 有两种安装方式：一种是全量安装模式，即把所有 Hadoop 软件安装到同一个目录下；另一种是自定义安装模式，即自定义安装路径、文件名和版本号，然后只安装所需软件。

对于 Hadoop 全量安装模式，通常采用二进制包安装的方式，该包中包含所有 Hadoop 组件，因此安装起来比较简单。但是这种安装模式难以管理不同版本之间的冲突，不利于维护 Hadoop 集群。如果考虑到维护成本，则建议采用自定义安装模式。

目前比较知名的自定义安装包，主要有以下三种：

1. Ambari：基于 Cloudera Manager 的管理界面。
2. Bigtop：Hortonworks 开源社区的一套 Hadoop 发行版，里面包含了 HDFS、MapReduce、YARN、Hive、Spark 等组件。
3. CloudERA：RedHat 基于 Apache Ambari 的一款开源管理工具。

本文将以 Bigtop 为例，演示如何安装部署 Hadoop。Bigtop 的安装部署脚本主要由三部分组成：Puppet 模块、剧本和配置文件。其中 Puppet 模块负责自动化安装 Hadoop 各个组件，剧本负责根据用户提供的参数配置 Hadoop，配置文件则指定各个组件的属性值。

# 3.Bigtop 安装部署
## 3.1 下载安装包


点击下载后，开始下载，大概需要几分钟时间，请耐心等待。下载完成后，解压压缩包到任意目录。

## 3.2 安装配置
### 3.2.1 配置环境变量
编辑 /etc/profile 文件，加入以下内容：
```bash
export JAVA_HOME=/usr/java/jdk1.8.0_xxx # 填写自己的 java 路径
export PATH=$JAVA_HOME/bin:$PATH
export HADOOP_HOME=/opt/bigtop-1.3.0-incubating/hadoop # 指定 Hadoop 目录
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop # 指定 Hadoop 配置目录
export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop # 指定 yarn 配置目录
export PATH=$PATH:$HADOOP_HOME/bin # 添加 Hadoop bin 目录到 PATH
export PATH=$PATH:$HADOOP_HOME/sbin # 添加 Hadoop sbin 目录到 PATH
```

保存退出，执行 source /etc/profile 命令使得配置立即生效。

### 3.2.2 创建 Hadoop 用户
切换到 root 用户：su root 。执行以下命令创建 hadoop 用户：
```bash
useradd hadoop
passwd hadoop # 设置密码
```

### 3.2.3 分配权限
切换到 hadoop 用户：su hadoop ，并执行以下命令分配权限：
```bash
mkdir $HADOOP_HOME && chown -R hadoop:hadoop $HADOOP_HOME # 创建并分配 Hadoop 目录权限
mkdir /data1/namenode && mkdir /data1/datanode # 创建 HDFS 数据目录
chmod 755 /data1/namenode && chmod 755 /data1/datanode # 修改 HDFS 数据目录权限
chown -R hadoop:hadoop /data1/namenode && chown -R hadoop:hadoop /data1/datanode # 修改 HDFS 数据目录所有权
```

### 3.2.4 配置参数
修改 $HADOOP_HOME/etc/hadoop/core-site.xml 文件，添加以下内容：
```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>

    <!-- namenode HA -->
    <property>
        <name>ha.zookeeper.quorum</name>
        <value>zk1:2181,zk2:2181,zk3:2181</value>
    </property>
    <property>
        <name>dfs.nameservices</name>
        <value>mycluster</value>
    </property>
    <property>
        <name>dfs.ha.namenodes.mycluster</name>
        <value>nn1,nn2</value>
    </property>
    <property>
        <name>dfs.namenode.rpc-address.mycluster.nn1</name>
        <value>hdfs1:8020</value>
    </property>
    <property>
        <name>dfs.namenode.rpc-address.mycluster.nn2</name>
        <value>hdfs2:8020</value>
    </property>
    <property>
        <name>dfs.client.failover.proxy.provider.mycluster</name>
        <value>org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider</value>
    </property>
    <property>
        <name>dfs.ha.automatic-failover.enabled</name>
        <value>true</value>
    </property>

    <!-- datanode -->
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/data1/datanode</value>
    </property>
    <property>
        <name>dfs.replication</name>
        <value>2</value>
    </property>
    <property>
        <name>dfs.permissions</name>
        <value>false</value>
    </property>
    
    <!-- jmx监控 -->
    <property>
        <name>hadoop.metrics2.impl</name>
        <value>org.apache.hadoop.metrics2.impl.MetricsSystemImpl</value>
    </property>    
    <property>
        <name>hadoop.security.authentication</name>
        <value>simple</value>
    </property>
    
</configuration>
```

修改 $HADOOP_HOME/etc/hadoop/yarn-site.xml 文件，添加以下内容：
```xml
<configuration>
  <property>
      <name>yarn.resourcemanager.resource-tracker.address</name>
      <value>hdfs1:8025</value>
  </property>

  <property>
      <name>yarn.resourcemanager.scheduler.address</name>
      <value>hdfs1:8030</value>
  </property>

  <property>
      <name>yarn.resourcemanager.address</name>
      <value>hdfs1:8050</value>
  </property>
  
  <property>
      <name>yarn.resourcemanager.admin.address</name>
      <value>hdfs1:8141</value>
  </property>
  
  <property>
      <name>yarn.resourcemanager.hostname</name>
      <value>hdfs1</value>
  </property>

  <!-- nodemanagers配置 -->
  <property>
      <name>yarn.nodemanager.aux-services</name>
      <value>mapreduce_shuffle</value>
  </property>

  <!-- queue管理 -->
  <property>
      <name>yarn.scheduler.capacity.root.queues</name>
      <value>default</value>
  </property>

  <property>
      <name>yarn.scheduler.capacity.root.default.user-limit-factor</name>
      <value>1</value>
  </property>

  <property>
      <name>yarn.scheduler.capacity.root.default.state</name>
      <value>RUNNING</value>
  </property>

  <!-- 隔离队列 -->
  <property>
      <name>yarn.scheduler.capacity.root.acl_submit_applications</name>
      <value>*</value>
  </property>

  <property>
      <name>yarn.scheduler.capacity.root.capacity</name>
      <value>100</value>
  </property>

  <property>
      <name>yarn.scheduler.capacity.root.accessible-node-labels</name>
      <value></value>
  </property>

  <property>
      <name>yarn.scheduler.capacity.maximum-am-resource-percent</name>
      <value>1</value>
  </property>

  <!-- MapReduce配置 -->
  <property>
      <name>mapreduce.framework.name</name>
      <value>yarn</value>
  </property>

  <property>
      <name>yarn.app.mapreduce.am.command-opts</name>
      <value>-Xmx2048m</value>
  </property>

  <!-- AM启动命令模板 -->
  <property>
      <name>yarn.app.mapreduce.am.staging-dir</name>
      <value>/tmp</value>
  </property>

  <property>
      <name>yarn.log-aggregation-enable</name>
      <value>true</value>
  </property>
  
</configuration>
```

修改 $HADOOP_HOME/etc/hadoop/slaves 文件，添加 slave 主机名。

配置完成后，最后一步是启动 Hadoop：
```bash
$HADOOP_HOME/bin/start-all.sh
```

至此，Bigtop Hadoop 安装成功。