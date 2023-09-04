
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
HDFS(Hadoop Distributed File System)是Apache Hadoop项目的一部分。HDFS是一个分布式文件系统，用于存储超大规模数据集。它具有高容错性、高可用性、可扩展性和海量数据访问吞吐量等特点。HDFS被设计用来处理海量的数据，包括来自Google，Facebook，LinkedIn，Netflix等大型网站和搜索引擎的原始日志、Web页面、视频或其他大型二进制文件等。HDFS既能够提供高性能的数据访问，也能够支撑实时数据分析，是 Hadoop MapReduce 和 Apache Spark 的基础设施。
## 架构
HDFS由NameNode和DataNodes组成，如下图所示：

1. NameNode: HDFS的主服务进程，负责管理文件系统的命名空间(namespace)。它是一个单独的进程，运行在集群中的任何一个节点上，整个HDFS只有一个NameNode，它记录了文件的元信息（如文件名、大小、权限）；同时它也是客户端访问的入口，用户需要先向NameNode请求相关的文件操作，然后再访问对应的DataNode获取数据。

2. DataNodes: 数据存放在一组称为DataNode的服务器上。它们一般部署在物理机或者虚拟机中，充当HDFS中存储数据的角色。每个DataNode都有自己独立的硬盘，并保持与NameNode之间的通信。NameNode通过心跳机制检查DataNode是否正常工作。如果某个DataNode长期不回应心跳信号，NameNode将认为该节点已经失败，并从该节点上读取的数据副本会重新复制到其他正常的DataNode上。

3. Secondary NameNode(Backup Node): 当NameNode发生故障时，可以启动Secondary NameNode来替代NameNode的功能。它与Active NameNode共享相同的磁盘资源和内存缓存，可以继续提供对外服务。但是由于与Primary NameNode共享存储资源，因此其所维护的信息可能出现延迟。另外，备份节点不会参与文件的读写操作，仅作为辅助。

## 文件系统操作
### 创建目录
```bash
$ hadoop fs -mkdir /user/root/data
```
### 删除目录及其内容
```bash
$ hadoop fs -rmr /user/root/data
```
### 查看当前工作路径
```bash
$ hadoop fs pwd
```
### 修改文件或目录名
```bash
$ hadoop fs -mv /user/root/oldname /user/root/newname
```
### 拷贝文件到本地目录
```bash
$ hadoop fs -cp file:///path/to/file hdfs:///path/to/dest
```
### 从HDFS下载文件到本地目录
```bash
$ hadoop fs -get /path/to/src localdir
```
### 将本地文件上传到HDFS
```bash
$ hadoop fs -put srcfile hdfs:///path/to/dest
```
### 查看文件详情
```bash
$ hadoop fs -stat info /path/to/file
```
其中`info`参数可以取值为`-help`, `-h`，显示帮助信息；`-blocks`，显示文件块信息；`-checksum`，显示校验和信息；`-length`，显示文件长度；`-owner`，显示文件所有者；`-group`，显示文件所属群组；`-perms`，显示文件权限；`-replication`，显示副本数量；`-type`，显示文件类型；`-accessTime`，显示最近一次访问时间；`-modificationTime`，显示最近一次修改时间；`-blockSize`，显示文件块大小。
### 列出文件列表
```bash
$ hadoop fs -ls /path/to/dir
```
### 创建符号链接
```bash
$ hadoop fs -ln -s source dest
```