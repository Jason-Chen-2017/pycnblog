
作者：禅与计算机程序设计艺术                    

# 1.简介
         
Apache Hadoop 是目前最流行的开源分布式计算框架之一，其被广泛应用于数据分析、实时查询、日志处理等领域。作为一款开源项目，Hadoop 有着强大的生态系统支持，包括 HDFS（Hadoop Distributed File System）、MapReduce（分布式计算框架）、Hive（数据仓库工具）、Pig（小型脚本语言）等多个组件。为了更好地使用 Hadoop 来提升效率和节省成本，大量的教程和学习材料随处可见。相比于传统的单体架构模式，Hadoop 在存储、计算和集群管理方面都进行了高度的优化，使得它能够在云端、本地以及海量数据的情况下实现高性能和扩展性。然而，对于初级用户来说，掌握 Hadoop 的基本知识并不能直接帮助他们解决复杂的问题，因此需要掌握 Hadoop 的一些特点及技巧来提升效率、节约成本、优化资源利用率。同时，还要学会用自助的方式发现 Hadoop 中的坑、漏洞、陷阱和优化方案，提升自己的编程水平。本书将提供有关 Hadoop 的核心概念、特性、原理和一些技巧，并分享基于 Hadoop 的实际案例，帮助读者快速上手、理解、使用 Hadoop 框架。
# 2.知识结构
本书共分八章：

⒈ Hadoop 基础知识

⒉ MapReduce

⒊ HDFS

⒋ Hive

⒌ Pig

⒍ Zookeeper

⒎ YARN

⒏ 云计算与 Hadoop 大数据框架对比
# 3.作者信息
李开复，清华大学计算机系博士研究生。曾就职于阿里巴巴集团、百度、英伟达。擅长大数据平台架构设计及研发工作。主攻数据存储和计算框架。主要研究方向为云计算、分布式计算、数据分析以及大数据存储等。期待通过分享自己的经验，帮助更多技术人员顺利入门和掌握 Hadoop。欢迎联系他：<EMAIL> 。
# 4.版权声明
本书采用“保持署名—非商用”创意共享4.0许可证。只要保持原作者署名和非商用，您可以自由地阅读、分享、修改本书。详细的法律条文请参阅本书的网站 http://hadoop-book.cn/。
# 5.目录
第1章 Hadoop 基础知识
1.1 Hadoop 简介
1.2 Hadoop 发展历史及其定位
1.3 Hadoop 生态系统及各个组件的功能介绍
1.4 Hadoop 集群架构与角色划分
1.5 Hadoop 文件系统 HDFS 的特点
1.6 Hadoop 安全机制与常见问题解答
1.7 Hadoop 的虚拟机调度器 Mesos 简介

第2章 MapReduce
2.1 MapReduce 作业流程
2.2 Map 函数与 Reduce 函数的介绍
2.3 MapReduce 编程模型
2.4 Hadoop Streaming API 的使用
2.5 MapReduce 优化技术
2.6 数据压缩与归档

第3章 HDFS
3.1 HDFS 架构与组成
3.2 HDFS 操作命令
3.3 HDFS 一致性机制与故障切换
3.4 HDFS 常见问题解答
3.5 HDFS 性能调优
3.6 HDFS 在线扩容

第4章 Hive
4.1 Hive 简介
4.2 Hive 数据模型
4.3 Hive 查询语言 DDL、DML 和 DQL 语法
4.4 Hive 优化原则与方法论
4.5 Hive 安装配置及使用
4.6 Hive 内置函数与 UDF 开发
4.7 Hive 表分区和索引
4.8 Hive 数据导入导出

第5章 Pig
5.1 Pig 简介
5.2 Pig 语言
5.3 Pig 命令
5.4 Pig 数据加载与分割
5.5 Pig 排序与抽样
5.6 Pig 连接 Join 运算符
5.7 Pig 聚合统计运算符
5.8 Pig Latin 和 Java API 对比
5.9 Pig 插件开发与使用
5.10 Pig 性能调优

第6章 Zookeeper
6.1 ZooKeeper 简介
6.2 ZooKeeper 安装与配置
6.3 ZooKeeper 集群部署
6.4 ZooKeeper 数据模型
6.5 ZooKeeper 客户端编程模型
6.6 ZooKeeper ACL 权限控制
6.7 ZooKeeper 分布式锁服务

第7章 YARN
7.1 YARN 概述
7.2 YARN 运行环境设置
7.3 YARN 资源管理器
7.4 YARN 节点管理器
7.5 YARN 服务间通信协议
7.6 YARN 计算框架
7.7 YARN 容错机制
7.8 YARN 队列和抢占式资源分配
7.9 YARN 的日志和调试

第8章 云计算与 Hadoop 大数据框架对比
8.1 Hadoop 优缺点
8.2 Hadoop 发展趋势
8.3 AWS EMR 简介
8.4 Google Cloud Dataproc 简介

