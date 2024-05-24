
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Apache Hadoop 是 Apache 基金会的一个开源分布式计算平台，可以实现对大数据集的存储、处理、分析和实时查询等功能。其主要由 HDFS（Hadoop Distributed File System）、MapReduce（基于 Hadoop 分布式计算框架）、YARN（Hadoop Yet Another Resource Negotiator）、Hive（基于 Hadoop 的数据仓库系统）等多个子项目组成。其中，HDFS 为海量数据的存储提供了底层支持，而 MapReduce 和 YARN 分别负责并行计算和资源管理；Hive 提供 SQL 查询接口，将结构化的数据映射到 Hadoop 文件系统上，并提供复杂的查询能力。

本文将从以下方面对 Hadoop 集群进行介绍和配置：

1.集群规划及硬件选择：了解 Hadoop 集群需要多少主节点和工作节点，以及每台服务器的硬件配置要求。
2.各个模块的安装与配置：包括 Java 安装、SSH 配置、Hadoop、Zookeeper、HBase、Storm、Spark 等模块的安装和配置。
3.集群运行环境测试：验证 Hadoop 集群的正确性和可用性。

# 2.集群规划及硬件选择
首先，确定 Hadoop 集群需要多少主节点和工作节点，以及每台服务器的硬件配置要求。如图所示：


如上图所示，Hadoop 集群由三种角色的节点组成：

1.NameNode（nn）：Hadoop 中的名字服务，负责维护文件系统命名空间、记录每个文件的块列表信息、以及执行文件系统操作请求。它一般在单个节点上运行，通常作为独立的服务器部署。

2.DataNodes（dn）：分布式文件存储，存储实际的数据块。DataNode 将数据切分成固定大小的 block，并定期向 NameNode 上报自己存储的文件信息，以便 NameNode 能够掌握整体的存储情况。通常一个集群中配置一个或多个 DataNode 来提高数据处理的吞吐率。

3.JobTracker（jt）：作业跟踪器，用于监控任务进度和状态，协调任务分配和执行。作业提交后，JobTracker 会根据计算资源的空闲情况分配任务给集群中的 TaskTracker。

一般来说，Hadoop 集群至少需要三个 NameNode 和两个以上 DataNode。Master 节点一般建议配置为物理机，因为它们不经过网络传输，所以性能比虚拟机更好；而 Core 节点则推荐配置为虚拟机，以便利用云计算的弹性伸缩能力。另外，集群中的服务器数量也应该控制在合理范围内，避免单点故障或集群失效。

Hadoop 集群的配置需求如下：

1.CPU：每个服务器至少需要 4 个 CPU core，并且 SSD 硬盘可以提升 IO 速度。
2.内存：由于 Hadoop 集群的计算密集型特性，因此内存要求比较高。推荐配置为 32GB 以上的内存，并且采用双核 E5 或 E6 CPU 即可达到较好的效果。
3.磁盘：数据和日志的存储都需要 SSD 硬盘。

# 3.各个模块的安装与配置
## 3.1 Java 安装
Hadoop 使用 Java 语言开发，需要先安装 Java Runtime Environment（JRE）。在 Linux 操作系统上安装 JRE 有两种方式：

1.手动安装：下载 JRE 压缩包，解压到指定目录，并设置 JAVA_HOME 变量。这种方式简单易懂，但是版本更新麻烦，且需要手动修改 JAVA_HOME 变量。

2.自动安装：可以使用软件包管理工具安装 JRE，比如 yum、apt-get 命令。例如 CentOS 下可以使用 yum 命令安装：

   ```
   sudo yum install java-1.8.0-openjdk-devel
   ```

   Ubuntu 下可以使用 apt-get 命令安装：

   ```
   sudo apt-get install default-jre
   ```

   或者直接安装 OpenJDK：

   ```
   sudo apt-get install openjdk-8-jdk
   ```

   此外，还可以在线安装 Oracle JDK 或其他版本的 JRE。

## 3.2 SSH 配置
Hadoop 使用 SSH 协议远程管理，需要在所有服务器之间建立信任关系。首先，所有节点需要安装 SSH 服务端和客户端软件。然后，需要配置 SSH 免密码登录。

### 在 CentOS 上配置 SSH
如果你的系统是 CentOS 发行版，可以按以下步骤完成 SSH 设置：

1.安装 openssh-server 和 openssh-clients 软件包：

   ```
   sudo yum install -y openssh-server openssh-clients
   ```

   如果没有找到相应的包，可能是因为该软件包已经被标记为 “obsoleted” ，可以使用以下命令代替：

   ```
   sudo yum install -y centos-release-scl epel-release
   sudo yum install -y rh-git29
   sudo scl enable rh-git29 bash
   # 重新登录 shell
   git clone https://github.com/git/git.git
   cd git
   make configure
  ./configure --prefix=/usr --with-openssl
   sudo make altinstall
   ```

   

2.启动 sshd 服务：

   ```
   sudo systemctl start sshd
   sudo systemctl status sshd   # 查看 sshd 服务状态
   ```

   通过上面命令可以看到 sshd 服务已启动，并且正在监听 22 端口（默认端口）。

3.配置防火墙：

   ```
   sudo firewall-cmd --permanent --zone=public --add-service=ssh
   sudo firewall-cmd --reload
   ```
   
   通过上面命令，允许防火墙开放 SSH 服务（默认端口 22），无需再手工添加。
   
4.配置 SSH 免密码登录：

   生成 SSH 密钥对：
   
   ```
   ssh-keygen -t rsa -P ''    # 不输入 passphrase
   cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   ```

   修改 `~/.ssh/config` 文件，加入以下内容：

   ```
   Host *
     PubkeyAuthentication yes     # 使用密钥进行身份认证
     PasswordAuthentication no     # 禁止 password 认证
   ```

   然后就可以使用 ssh 连接到服务器了：

   ```
   ssh hadoop@hadoop01
   ```

   用户名是 hadoop，密码为空。第一次连接时需要输入确认密钥，之后就可以使用私钥 ssh 互相免密登录了。

### 在 Ubuntu 上配置 SSH
如果你的系统是 Ubuntu 发行版，可以按以下步骤完成 SSH 设置：

1.安装 openssh-server 和 openssh-client 软件包：

   ```
   sudo apt update && sudo apt upgrade
   sudo apt install openssh-server openssh-client
   ```

2.启动 sshd 服务：

   ```
   sudo service ssh start
   sudo systemctl status sshd   # 查看 sshd 服务状态
   ```

3.配置防火墙：

   ```
   sudo ufw allow ssh
   ```

4.配置 SSH 免密码登录：

   生成 SSH 密钥对：

   ```
   mkdir ~/.ssh
   chmod 700 ~/.ssh
   ssh-keygen -t ed25519 -C ""      # 不输入 passphrase
   mv ~/.ssh/id_ed25519* ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   ```

   修改 `~/.ssh/config` 文件，加入以下内容：

   ```
   Host *
     PubkeyAuthentication yes        # 使用密钥进行身份认证
     PasswordAuthentication no        # 禁止 password 认证
   ```

   然后就可以使用 ssh 连接到服务器了：

   ```
   ssh hadoop@hadoop01
   ```

   用户名是 hadoop，密码为空。第一次连接时需要输入确认密钥，之后就可以使用私钥 ssh 互相免密登录了。

## 3.3 Hadoop 安装
Apache Hadoop 的各个模块都有自己的安装包，可以通过官网下载：https://hadoop.apache.org/releases.html。下载压缩包后，把压缩包里面的内容解压到指定位置，比如 /opt/hadoop-3.2.2/ 。配置 Hadoop 需要修改配置文件，配置文件一般放在 /etc/hadoop/ 文件夹下。配置文件以 xml 格式存储，通常包括 hdfs-site.xml、core-site.xml、mapred-site.xml 三个配置文件。

在所有配置文件里面，最重要的是 `core-site.xml`，这个文件用来配置 Hadoop 的一些基本参数，比如文件系统（HDFS）的地址、端口号、副本数量等。编辑 `core-site.xml` 文件，在 <configuration> 标签内部添加以下内容：

```
  <property>
      <name>fs.defaultFS</name>
      <value>hdfs://hadoop01:9000</value>
  </property>

  <property>
      <name>hadoop.tmp.dir</name>
      <value>/opt/hadoop-3.2.2/data/tmp</value>
  </property>
```

这里的 fs.defaultFS 配置项的值就是 HDFS 的名称节点的主机名和端口号，后续要使用 HDFS 时就需要用这个地址。hadoop.tmp.dir 配置项的值表示临时文件目录，建议设置为磁盘大的磁盘。

接着，我们配置 `hdfs-site.xml`。编辑 `hdfs-site.xml` 文件，在 <configuration> 标签内部添加以下内容：

```
  <property>
      <name>dfs.namenode.name.dir</name>
      <value>/opt/hadoop-3.2.2/data/name</value>
  </property>

  <property>
      <name>dfs.datanode.data.dir</name>
      <value>/opt/hadoop-3.2.2/data/data</value>
  </property>
```

这里的 dfs.namenode.name.dir 配置项的值是 HDFS 元数据存储目录，用于保存文件系统命名空间和块信息；dfs.datanode.data.dir 配置项的值是 HDFS 数据存储目录，用于存放实际的数据。

最后，我们配置 `mapred-site.xml`。编辑 `mapred-site.xml` 文件，在 <configuration> 标签内部添加以下内容：

```
  <property>
      <name>mapreduce.framework.name</name>
      <value>yarn</value>
  </property>
```

这里的 mapreduce.framework.name 配置项的值表示 Hadoop 运行框架，一般设置为 yarn。

以上，我们完成了 Hadoop 的基本配置，下面开始安装各个模块。

## 3.4 Zookeeper 安装
Zookeeper 是 Hadoop 的依赖组件之一，作用类似于 “领导者”。Zookeeper 可确保在分布式环境下多个节点的状态一致。在 Hadoop 中，Zookeeper 提供 HDFS 的高可用性，即任何时候只有一个 NameNode 是可用的。Zookeeper 可以通过 puppet 脚本自动部署，也可以手工安装。

安装 Zookeeper 有两种方式：

1.下载安装包并手动部署：先到官方网站下载最新版本的 Zookeeper 安装包，然后把压缩包解压到指定位置，比如 /opt/zookeeper-3.6.3/ 。进入 zookeeper-3.6.3/conf 目录，修改 myid 文件，改成当前机器的唯一标识符，例如：

   ```
   1
   ```

   把 conf 目录下的 zoo.cfg 拷贝到其它所有节点上。然后启动 Zookeeper 服务：

   ```
   sh bin/zkServer.sh start
   ```

2.使用 puppet 安装：可以使用 puppet 模块自动部署 Zookeeper。首先，把 zookeeper 类定义在 site.pp 文件中：

   ```
   class { 'zookeeper': }
   ```

   然后，把 Zookeeper 的 yum 源加到节点的 sources.list 中：

   ```
   vim /etc/yum.repos.d/epel.repo
   [epel]
   name=Extra Packages for Enterprise Linux $releasever - $basearch
  ...
   gpgcheck=1
   enabled=1
  ...
   url=http://download.fedoraproject.org/pub/epel/$releasever/Everything/$basearch
   ```

   刷新 yum 缓存：

   ```
   sudo yum clean all
   ```

   安装 puppet：

   ```
   sudo rpm -ivh https://yum.puppet.com/puppet6-release-el-7.noarch.rpm
   sudo yum install -y puppet-agent
   ```

   执行 puppet agent 编译资源：

   ```
   sudo puppet apply --verbose manifests.pp
   ```

   这将自动部署 Zookeeper 到所有节点上。

## 3.5 HDFS 安装
HDFS (Hadoop Distributed File System) 是 Hadoop 生态系统的基础。HDFS 是一个可靠、高容错和负载平衡的分布式文件系统，提供高吞吐量写入以及松耦合的架构。HDFS 支持POSIX 文件系统接口，并提供一个简单的元数据模型，使得文件和数据的位置透明化。HDFS 的安装过程非常简单，只需要把压缩包解压到指定位置，比如 /opt/hadoop-3.2.2/ 即可。配置 HDFS 主要修改配置文件，配置文件一般放在 /etc/hadoop/ 文件夹下。配置文件以 xml 格式存储，通常包括 hdfs-site.xml、core-site.xml、mapred-site.xml 三个配置文件。

在所有配置文件里面，最重要的是 `core-site.xml`，这个文件用来配置 Hadoop 的一些基本参数，比如文件系统（HDFS）的地址、端口号、副本数量等。编辑 `core-site.xml` 文件，在 <configuration> 标签内部添加以下内容：

```
  <property>
      <name>fs.defaultFS</name>
      <value>hdfs://hadoop01:9000</value>
  </property>

  <property>
      <name>hadoop.tmp.dir</name>
      <value>/opt/hadoop-3.2.2/data/tmp</value>
  </property>
```

这里的 fs.defaultFS 配置项的值就是 HDFS 的名称节点的主机名和端口号，后续要使用 HDFS 时就需要用这个地址。hadoop.tmp.dir 配置项的值表示临时文件目录，建议设置为磁盘大的磁盘。

接着，我们配置 `hdfs-site.xml`。编辑 `hdfs-site.xml` 文件，在 <configuration> 标签内部添加以下内容：

```
  <property>
      <name>dfs.namenode.name.dir</name>
      <value>/opt/hadoop-3.2.2/data/name</value>
  </property>

  <property>
      <name>dfs.datanode.data.dir</name>
      <value>/opt/hadoop-3.2.2/data/data</value>
  </property>
```

这里的 dfs.namenode.name.dir 配置项的值是 HDFS 元数据存储目录，用于保存文件系统命名空间和块信息；dfs.datanode.data.dir 配置项的值是 HDFS 数据存储目录，用于存放实际的数据。

最后，我们配置 `mapred-site.xml`。编辑 `mapred-site.xml` 文件，在 <configuration> 标签内部添加以下内容：

```
  <property>
      <name>mapreduce.framework.name</name>
      <value>yarn</value>
  </property>
```

这里的 mapreduce.framework.name 配置项的值表示 Hadoop 运行框架，一般设置为 yarn。

以上，我们完成了 HDFS 的基本配置，下面开始安装 NameNode 和 DataNode。

## 3.6 NameNode 安装
NameNode 是 Hadoop 文件系统的核心，所有的文件都会先存储到 NameNode 上，然后由它来决定数据是否最终会存入 DataNode 中。NameNode 负责管理 HDFS 文件系统的命名空间和数据块。NameNode 需要部署在独立的服务器上，建议采用物理机或虚拟机部署。

为了让 NameNode 正常工作，我们需要先把它初始化：

1.安装 Java：参考上面的 “Java 安装” 一节。

2.启动 NameNode：启动之前需要设置 JAVA_HOME 变量：

   ```
   export JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::")
   ```

3.格式化 HDFS：创建一个新目录 /opt/hadoop-3.2.2/data/name，并切换到该目录下，然后执行命令：

   ```
   bin/hdfs namenode -format
   ```

   注意：不要在生产环境使用 -format 命令！

4.启动 NameNode：切换到安装目录，然后执行命令：

   ```
   sbin/start-dfs.sh
   jps
   ```

   此时，NameNode 服务应该已经启动成功。查看进程列表，如果看到 “NameNode” 和 “QuorumPeerMain”，则 NameNode 已经正常启动。

5.停止 NameNode：切换到安装目录，然后执行命令：

   ```
   sbin/stop-dfs.sh
   ```

   此时，NameNode 服务应该已经停止。

## 3.7 DataNode 安装
DataNode 是 HDFS 的数据存储单元，负责存储真正的数据。一个集群中可以有多台 DataNode，但一般都只设置一台作为主节点，另一些备份节点，以便应对硬件故障。

配置 DataNode 需要修改配置文件，配置文件一般放在 /etc/hadoop/ 文件夹下。配置文件以 xml 格式存储，通常包括 hdfs-site.xml、core-site.xml、mapred-site.xml 三个配置文件。

在所有配置文件里面，最重要的是 `core-site.xml`，这个文件用来配置 Hadoop 的一些基本参数，比如文件系统（HDFS）的地址、端口号、副本数量等。编辑 `core-site.xml` 文件，在 <configuration> 标签内部添加以下内容：

```
  <property>
      <name>fs.defaultFS</name>
      <value>hdfs://hadoop01:9000</value>
  </property>

  <property>
      <name>hadoop.tmp.dir</name>
      <value>/opt/hadoop-3.2.2/data/tmp</value>
  </property>
```

这里的 fs.defaultFS 配置项的值就是 HDFS 的名称节点的主机名和端口号，后续要使用 HDFS 时就需要用这个地址。hadoop.tmp.dir 配置项的值表示临时文件目录，建议设置为磁盘大的磁盘。

接着，我们配置 `hdfs-site.xml`。编辑 `hdfs-site.xml` 文件，在 <configuration> 标签内部添加以下内容：

```
  <property>
      <name>dfs.namenode.name.dir</name>
      <value>/opt/hadoop-3.2.2/data/name</value>
  </property>

  <property>
      <name>dfs.datanode.data.dir</name>
      <value>/opt/hadoop-3.2.2/data/data</value>
  </property>
```

这里的 dfs.namenode.name.dir 配置项的值是 HDFS 元数据存储目录，用于保存文件系统命名空间和块信息；dfs.datanode.data.dir 配置项的值是 HDFS 数据存储目录，用于存放实际的数据。

最后，我们配置 `mapred-site.xml`。编辑 `mapred-site.xml` 文件，在 <configuration> 标签内部添加以下内容：

```
  <property>
      <name>mapreduce.framework.name</name>
      <value>yarn</value>
  </property>
```

这里的 mapreduce.framework.name 配置项的值表示 Hadoop 运行框架，一般设置为 yarn。

以上，我们完成了 DataNode 的基本配置，下面开始启动整个集群。

## 3.8 集群运行环境测试
通过上面的步骤，我们已经完成了 Hadoop 集群的安装。为了确保集群运行正常，我们可以做一下测试。

### 测试 NameNode
首先，我们尝试访问 NameNode Web UI：

```
http://hadoop01:50070
```

如果看到类似如下页面，则表明 NameNode Web UI 已经正常启动：


如果看到警告信息，可能是因为 HDFS 目录权限设置错误。此时，我们可以尝试执行以下命令：

```
sudo chown -R hduser:hadoop /opt/hadoop-3.2.2/data/
```

再次访问 NameNode Web UI，如果看到如下页面，则表明权限设置成功：


### 测试 DataNode
我们尝试在 Hadoop 集群上创建目录并上传文件：

```
$ hadoop fs -mkdir /test
$ hadoop fs -put test.txt /test
```

如果出现以下提示，则表明 DataNode 服务已经正常工作：

```
Putting file:///home/hadoop/test.txt to hdfs://hadoop01:9000//test
```

我们还可以尝试从 NameNode 获取文件：

```
$ hadoop fs -cat /test/test.txt
Hello World!
```

### 测试 JobTracker
最后，我们尝试运行 MapReduce 作业：

```
$ hadoop jar wordcount.jar org.apache.hadoop.examples.WordCount /input /output
```

如果出现以下提示，则表明 JobTracker 服务已经正常工作：

```
15/11/20 23:30:47 INFO client.RMProxy: Connecting to ResourceManager at hadoop01/192.168.1.101:8032
...
15/11/20 23:30:49 INFO input.FileInputFormat: Total input paths to process : 1
15/11/20 23:30:49 INFO mapreduce.JobSubmitter: number of splits:1
15/11/20 23:30:49 INFO Configuration.deprecation: session.id is deprecated. Instead, use dfs.metrics.session-id
...
15/11/20 23:30:49 INFO impl.YarnClientImpl: Submitted application application_1506687761733_0002
```