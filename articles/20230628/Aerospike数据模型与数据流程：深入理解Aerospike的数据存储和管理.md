
作者：禅与计算机程序设计艺术                    
                
                
《3. Aerospike 数据模型与数据流程：深入理解 Aerospike 的数据存储和管理》
===========

1. 引言
-------------

1.1. 背景介绍

随着云计算和大数据技术的迅猛发展，数据存储和管理成为了一个越来越重要的问题。数据库管理系统（DBMS）和文件系统已经无法满足越来越复杂的需求，新的数据存储和管理技术正在不断涌现。

1.2. 文章目的

本文旨在深入理解Aerospike的数据模型和数据流程，帮助读者了解Aerospike如何实现高效的数据存储和管理。

1.3. 目标受众

本文主要面向数据存储和管理行业的技术人员和架构师，以及对Aerospike感兴趣的读者。

2. 技术原理及概念
--------------------

2.1. 基本概念解释

Aerospike是一个开源的分布式NoSQL数据库系统，旨在提供一种高效、可扩展的数据存储和管理解决方案。Aerospike支持多种扩展功能，包括支持分片、数据模型、数据流程等。

2.2. 技术原理介绍:算法原理，操作步骤，数学公式等

Aerospike的核心技术是基于列族存储和数据分片的支持，利用缓存技术提高数据读写性能。Aerospike的算法原理包括数据分片、列族存储、压缩、查询优化等。

2.3. 相关技术比较

Aerospike与传统关系型数据库（如MySQL、Oracle等）以及NoSQL数据库（如Cassandra、Redis等）进行了比较，说明了Aerospike在数据存储和管理方面的优势。

3. 实现步骤与流程
---------------------

3.1. 准备工作：环境配置与依赖安装

首先，需要在操作系统上安装Aerospike。然后，需要安装相关依赖，包括Java、Hadoop、Zookeeper等。

3.2. 核心模块实现

Aerospike的核心模块包括数据存储和管理模块、查询优化模块、缓存模块、分片模块等。其中，数据存储和管理模块负责数据的读写操作，查询优化模块负责优化查询过程，缓存模块负责数据的缓存，分片模块负责数据的分片处理。

3.3. 集成与测试

将Aerospike与其他技术进行集成，如Redis、Hadoop等，测试其性能和稳定性。

4. 应用示例与代码实现讲解
------------------------

4.1. 应用场景介绍

本节将通过一个在线购物系统的例子来说明Aerospike如何实现数据存储和管理。

4.2. 应用实例分析

首先，需要准备数据，包括用户信息、商品信息、订单信息等。然后，设计数据存储和管理模块，包括数据表结构设计、索引设计、缓存设计等。最后，编写核心代码实现数据读写、查询优化等功能。

4.3. 核心代码实现

```java
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.SqlMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.SqlQueryStats;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.SqlStatistics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Index;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeNode;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.AerospikeStore;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.Table;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.KeyValue;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.TableMetrics;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.QueryCache;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteCompaction;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.WriteScheduled;
import org.apache.hadoop.conf.Aerospike.HadoopAerospikeDB.A

