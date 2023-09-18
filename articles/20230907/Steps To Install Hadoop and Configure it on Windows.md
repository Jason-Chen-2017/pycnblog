
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Hadoop 是 Apache 基金会开发的一个开源分布式计算框架，它能够将海量数据集中到多个节点，并对其进行处理、分析、存储。然而，在实际生产环境中部署和管理 Hadoop 集群一般都比较复杂，因此，本文通过简单的几步安装配置 Hadoop 集群教程，帮助读者快速上手 Hadoop 在 Windows 平台上的部署及使用。

# 2.版本说明
本文基于以下两个版本的 Hadoop:
- Hadoop 2.9.2
- JDK 1.8

# 3.前提条件
- 安装 Oracle JDK (Java Development Kit)
    - 从官网下载安装包 Java SE Development Kit 8u202, 链接地址为 https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html 。
    - 将下载好的安装包解压至 C:\Program Files\Java 文件夹下，同时勾选 Add to PATH 选项。
- 配置 JAVA_HOME 环境变量
    - 在系统环境变量中找到 User Variables 中的 Path ，然后点击 Edit, 在弹出的编辑框内输入 `%JAVA_HOME%\bin` ，然后点击确定保存。

# 4.安装步骤
1. 配置好 Hadoop 的依赖库
   在安装之前需要先配置好 Hadoop 的依赖库。由于 Hadoop 需要运行在 Java 8 上，因此，首先需要确认系统已经安装了 Java 8 或更高的版本。

   ```bash
   # 查看是否安装了 Java 8 或以上版本
   java -version
   ```

2. 下载 Hadoop 安装包
   可以从官方网站 https://hadoop.apache.org/releases.html 下载最新版本的 Hadoop，本文基于 Hadoop 2.9.2 版本进行演示。下载链接为 http://mirror.cc.columbia.edu/pub/software/apache/hadoop/common/hadoop-2.9.2/hadoop-2.9.2.tar.gz 。

3. 解压 Hadoop 安装包
   将下载好的 Hadoop 安装包 hadoop-2.9.2.tar.gz 复制到任意目录，然后进入该目录执行以下命令解压：

   ```bash
   tar xzf hadoop-2.9.2.tar.gz 
   ```

4. 配置 Hadoop 安装路径
   默认情况下，Hadoop 会安装到 /usr/local/hadoop 下面。为了方便后续配置，可以修改 Hadoop 安装路径到 C:\Program Files\Hadoop 下。修改方式如下：

   ```bash
   mv /usr/local/hadoop $HADOOP_INSTALL_DIR    //移动之前的安装路径到新位置
   ln -s $HADOOP_INSTALL_DIR /usr/local/hadoop   //创建符号连接
   ```

5. 配置 Hadoop 环境变量
   修改 `core-site.xml` 和 `hdfs-site.xml` 文件，将相关配置加入到配置文件中。分别在 `$HADOOP_INSTALL_DIR\etc\hadoop\` 下找到这两个文件进行修改。

6. 配置 HDFS 文件系统
    执行以下命令初始化 HDFS 文件系统：

    ```bash
    hdfs namenode -format
    start-all.cmd      //启动 namenode 和 datanode 服务
    jps                //检查进程，确认 namenode 和 datanode 是否启动成功
    ```

    此时，HDFS 文件系统就准备好了，可以使用 Hadoop 来对其进行文件上传、下载、查询等操作。

# 5.总结
本文通过简单易懂的五步安装配置 Hadoop 集群的教程，帮助读者快速上手 Hadoop 在 Windows 平台上的部署及使用。此外，还包括了 Hadoop 的版本选择、依赖库的配置、安装路径的修改、环境变量的设置以及 HDFS 文件系统的初始化等关键步骤。希望这些内容能帮助读者快速地上手 Hadoop，并实现自己的目标。