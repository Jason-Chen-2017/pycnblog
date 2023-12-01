                 

# 1.背景介绍

Redis（Remote Dictionary Server）是一个开源的高性能键值存储系统，由 Salvatore Sanfilippo 于2009年创建。Redis 支持数据的持久化，可以将内存中的数据保存在磁盘中，重启的时候可以再次加载进行使用。Redis 不仅仅支持简单的键值对类型的数据，同时还提供列表、集合、有序集合及哈希等数据结构的存储。

Redis 和关系型数据库之间的一个主要区别是：Redis 是内存中的数据库，命令运行速度非常快；而关系型数据库则依赖于磁盘操作，因此运行速度较慢。

## 1.1 Redis 与其他 NoSQL 产品对比

NoSQL（Not only SQL）是一种不使用SQL语言访问的数据库系统。NoSQL产品分为四大类：键值对存储（Key-Value Stores）、文档型数据库（Document-Oriented Databases）、列式存储（Column-Family Stores）和图形型数据库（Graph Databases）。Redis属于键值对存储类别。其他知名的 NoSQL 产品有 MongoDB、Cassandra、HBase等。

|                         | Redis                      | MongoDB                    | Cassandra               | HBase                   |
|-------------------------|----------------------------|----------------------------|-------------------------|--------------------------|
| **核心特点**            | Key-Value Store            | Document Database         | Column-Family Store     | Wide-Column Store       |
| **语言**                | C, C++, Ruby, Python, Lua  | C#, Java, JavaScript, Ruby | Java, C#, Python        | Java                     |
| **WAN Replication**     | Primary-Replica           | Master-Slave              | Gossip Protocol        | HBase Master/Region     |
| **Consistency Model**   | Single Node                | Multi Node Cluster        | Multi Node Cluster     | Multi Node Cluster      |
| **Data Model**          || Document                 || Row Column Family       || Column Family          ||