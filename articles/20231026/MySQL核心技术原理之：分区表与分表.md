
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


随着互联网网站、移动应用、大数据等新兴技术的发展，网站或应用的数据量越来越大，单个数据库已经无法承载，需要对数据库进行水平拆分，将数据分布到不同的服务器上，而数据存储方式也逐渐变得复杂多样。为解决这个问题，MySQL提供了分区（Partition）功能，允许将数据分布到多个物理磁盘上，从而提升性能和可靠性。

分区通常用于如下场景：

1. 对查询速度要求高的数据，可以根据业务特点将热点数据放置在主库，冷数据放置在分区。通过分区，可以有效地避免整个库的瓶颈。
2. 数据集中存放在某一天，可以创建每天一个分区，方便归档和备份。
3. 想要进行表结构升级，不能直接修改原表，可以先新建同名表，再导入旧表数据，最后再删除旧表，这种方式可以保证数据的一致性。

虽然分区功能十分强大，但它也是一项复杂的技术，容易出错，因此，作为开发人员，需要了解其原理和使用方法，并根据自己的实际情况，灵活选择合适的方案。本文主要介绍MySQL中的分区表和分表。

# 2.核心概念与联系
## 分区表（Partitioned Table）

分区表由多个物理磁盘组成，每个分区对应一个独立的索引和数据文件。当插入、更新、删除数据时，会自动选择对应的分区。分区表可以通过增加分区数目或减少分区数目来动态调整，还能通过切割分区来增加磁盘利用率。

创建分区表的方法：
```sql
CREATE TABLE tablename (
   ...
) PARTITION BY {KEY|LINEAR|RANGE} COLUMNS(colname)(
        SUBPARTITION BY LINEAR COLUMNS(sub_colname)
);
```
其中{KEY|LINEAR|RANGE}指定分区列的类型，COLUMNS(colname)指定分区键；SUBPARTITION BY LINEAR COLUMNS(sub_colname)指定子分区的列名及策略。

- KEY：按分区键值范围分区，使得每个分区数据可以自行排序。一般情况下，推荐采用KEY分区。
- LINEAR：按照顺序分区，依次分配给分区。如果使用了子分区，则子分区依次分配给每个分区。
- RANGE：按照范围划分分区，分配范围内的数据到相应的分区。

## 分表（Sharded Tables）

分表（Sharding）指的是将同类数据均匀分布到多个数据库或表中。相比于全体数据，分表能够更加充分利用服务器资源，提升查询效率。

对于大型互联网网站，一般以域名为粒度，把同一域名下的所有数据都放在一个数据库中，这样做可以减少锁竞争。此外，有的公司可能面临数据库容量限制的问题，因此可以考虑将不同模块的数据分别存放在不同的数据库中。

创建分表的方法：
```sql
CREATE DATABASE dbname; -- 创建数据库
USE dbname;            -- 使用数据库

-- 创建分片表shard1
CREATE TABLE shard1 (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    city VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
PARTITION BY HASH(id) PARTS 4; 

-- 创建分片表shard2
CREATE TABLE shard2 (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    city VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
PARTITION BY HASH(id) PARTS 4; 

-- 创建主从复制规则
CREATE USER'repl'@'%' IDENTIFIED BY 'password';  
GRANT REPLICATION SLAVE ON *.* TO repl@'%';   

-- 配置mysql主从复制
MASTER:
CHANGE MASTER TO 
    MASTER_HOST='masterip',     # mysql master ip
    MASTER_PORT=3306,           # mysql port
    MASTER_USER='username',     # mysql username
    MASTER_PASSWORD='password', # mysql password
    MASTER_LOG_FILE='binlog.000001',    # binlog file
    MASTER_LOG_POS=154;         # binlog position
 
START SLAVE;   #启动从库
SHOW SLAVE STATUS\G;   #查看slave状态

SLAVE:
STOP SLAVE;    #停止从库
CHANGE MASTER TO 
    MASTER_HOST='',      # 将从库设置为空
    MASTER_PORT='';      # 将端口号设置为空
  
START SLAVE;    #重启从库
SHOW SLAVE STATUS\G;    #查看slave状态
```