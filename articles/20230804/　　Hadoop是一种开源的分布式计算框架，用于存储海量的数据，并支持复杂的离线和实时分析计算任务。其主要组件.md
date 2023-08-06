
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 HDFS(Hadoop Distributed File System)是一个分布式文件系统，它作为Hadoop的一个子项目被集成到计算集群之中，实现存储大数据。HDFS由两部分组成:一个master节点和多个slave节点。master负责管理整个文件系统的名字空间(namespace)，并对客户端提供读写数据服务；slave节点则存储真正的数据块。HDFS通过自动平衡集群中的DataNode的数量，来确保高可用性。Hadoop Mapreduce(MR)是一个编程模型，用于并行处理大数据集合的驱动器。MR可以充分利用多台服务器的资源来处理大型数据集，并且它还提供了丰富的数据处理功能，例如排序、过滤、JOIN和数据库查询。Hive是基于 Hadoop 的数据仓库工具。它提供一个SQL语法用来查询HDFS上的数据，并且能够将HDFS数据直接加载到数据库或HBase。Pig是基于MapReduce的高级语言，用于编写大规模数据处理作业。同时，还有其他一些工具如Tez、Mahout、Zookeeper、Flume、Sqoop等，它们可以帮助用户更加高效地进行数据处理。
         # 2.HDFS概述
        ##  2.1 HDFS架构
           HDFS (Hadoop Distributed File System)是Apache Hadoop项目的一部分。HDFS是一个高度容错性的、高吞吐量的分布式文件系统。HDFS采用主-备结构，即存在一个Namenode进程，用于元数据的维护和磁盘空间管理，一个或多个Datanode进程则作为数据节点，用于实际存放数据块。HDFS将文件系统划分成一个个的block，这些block会分布在多个datanodes上。客户端向namenode发送请求，获取文件的位置信息，然后直接与对应的datanode通信进行数据读写。HDFS的文件由块组成，块是HDFS的最小单位，通常大小为64MB~1GB。每个HDFS文件都有一个与之相关联的权限、属性、校验和等元数据。HDFS使用两个名词：数据块（Data Block）和数据节点（Data Node）。数据块是HDFS中最基本的存储单元，也是同一个文件的不同片段。数据节点是一个存储数据块的机器，一个HDFS集群可以包含多个数据节点。