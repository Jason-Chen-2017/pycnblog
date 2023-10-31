
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


MyISAM是MySQL默认使用的存储引擎。它是一个高性能的静态表存储引擎，它保存了表结构信息及数据索引，适合于执行大量的静态SELECT操作。但是其不支持事物（transaction）、外键约束（foreign key constraints）、FULLTEXT索引等特性。因此，对于需要使用这些特性的应用场景，就需要使用其他类型的存储引擎。

InnoDB是另一种功能更丰富的存储引擎，提供了对ACID事务的支持、外键约束、行级锁定、全文搜索能力等特性。从MySQL 5.5版本开始，InnoDB取代MyISAM成为默认的存储引擎。

在本次博客中，我们将详细介绍MyISAM存储引擎，以期帮助读者了解MyISAM存储引擎的内部工作原理和核心算法原理，并可以提高数据库管理和优化技巧。文章不会涉及InnoDB存储引擎的具体内容。如需进一步阅读，可参考官方文档。

# 2.核心概念与联系
## 2.1 MyISAM文件构成
首先，我们需要理解MyISAM文件的组成结构。这里，为了简化表格，仅显示相关字段信息：
|名称|类型|描述|
|---|---|---|
|.frm 文件 | 文件 | 表单文件，存储表定义；<br/>如 CREATE TABLE t1(id INT) ENGINE=MyISAM; 会在当前目录下生成t1.frm文件。|
|.MYD 数据文件 | 文件 | 数据文件，存储表数据记录；<br/>可以把数据文件理解为一张表的数据集合。|
|.MYI 索引文件 | 文件 | 索引文件，存储表索引；<br/>可以在CREATE INDEX或ALTER TABLE ADD INDEX命令创建索引时自动生成此文件。<br/><font color="red">注意：索引文件只用于快速查询，不用于排序。</font>|

## 2.2 MyISAM索引实现方式
MyISAM存储引擎采用B-Tree索引方法组织索引，该索引的基本单位为页（page），一个页中最多可以存储数据索引项。每个索引块的大小通常为16KB。

当向MyISAM插入一条新记录时，如果数据量超过某个阈值（默认64K），则会自动创建一个新的磁盘页。同时，也会在之前的页中创建数据索引项，但旧页不会删除直到所有引用索引项都被更新为指向新页，然后才删除掉旧页。也就是说，如果插入数据的过程需要分裂页，那么整个页的写入和页内数据的删除都是原子性的，不会发生页的混乱。

另外，MyISAM还会尝试保持页内的数据顺序不变，即使已经插入了新数据。这一点是通过在每个页开头设置一个指针来实现的。该指针指向第一个字节处空余空间的位置。

MyISAM索引方式类似于哈希表，不过数据不是保存在磁盘中，而是在内存中。在MyISAM存储引擎中，索引也是被存放在磁盘上的，但数据却是在内存中的。由于索引可能很大，所以可以使用更快的内存访问速度进行搜索。

## 2.3 MyISAM锁机制
MyISAM存储引擎支持两种类型的锁，第一种是表级别的共享锁（S lock）和排他锁（X lock）。

所谓表级别的共享锁就是允许多个用户同时读取同一表的某些记录，也就是说，除了用WHERE条件指定要锁定的记录以外，其他用户也能继续往表中插入、修改或删除记录，但前提是不能读取这个范围以外的其他记录。

所谓表级别的排他锁（又称为写锁）是禁止其他用户读取或修改表的任何记录，也就是说，除了用WHERE条件指定要锁定的记录以外，其他用户不能对表做任何插入、修改或删除操作。但是，表上仍然可以执行查询语句。

除此之外，MyISAM还支持行级锁（又称为next-key lock）,也就是记录锁，能锁定单条记录。在一个事务（transaction）中，如果要对某行数据进行读或写操作，必须先获得这个记录所在的页上的S锁或X锁，才能对其进行读或写，直到事务结束才释放锁。

为了实现行级锁，MyISAM存储引擎在索引中加入了间隙锁（gap lock）。间隙锁是一种特殊的锁，它锁住了一个范围，但是不包括记录本身。比如，如果在一个范围[A, B]上加X锁，并在B上加S锁，则表示“这个范围内的所有记录均不能进行任何更新操作”。这种锁策略能够有效防止幻读的出现。

# 3.核心算法原理和具体操作步骤
## 3.1 数据读取
1. 如果请求的页面不在缓冲池，则读取页对应的磁盘块到缓冲池中；
2. 从缓冲池中查找请求的数据；
3. 如果缓存中没有命中，则从磁盘中读取对应的数据页到缓冲池中；
4. 对数据页进行解析，找出相应的数据；
5. 将数据返回给客户端。

## 3.2 数据写入
1. 检查是否已经打开了事务，若没有则打开；
2. 在插入缓冲池中准备好数据记录的各种信息；
3. 判断插入的记录是否属于完整的页，若不是，则将剩下的记录添加到预写日志中；
4. 在事务结束时，将预写日志中的记录刷新到磁盘，并合并插入缓冲池中与磁盘中相同记录的冲突信息；
5. 更新内存的数据字典。

## 3.3 数据删除
与插入流程相似，只是在提交时将记录的墓碑标记为可删除，再由后台线程定时扫描标记过期的墓碑，真正将数据从磁盘上物理删除。

## 3.4 创建索引
根据索引列值的前缀构建索引树，并生成相应的索引文件，其中树的高度决定了索引的效率。

## 3.5 搜索索引
按照索引文件的内容进行搜索，依据索引值的精确匹配或范围匹配，找到检索的数据地址，从数据文件中读取数据并返回结果。

## 3.6 分裂页
当需要插入的数据页占用空间超过一定比例时，则自动分裂该页。分裂后的新页将留出一半可用空间供插入数据，原页的剩余数据则移到新页。同时，在原页中插入一条指向新页的指针，以便于快速定位。

# 4.具体代码实例和详细解释说明
```
# 创建表
mysql> create table myisam_test (
    -> id int primary key auto_increment,
    -> name varchar(20),
    -> age int);
 
# 插入数据
mysql> insert into myisam_test values (null,'Tom',25),(null,'Jack',30),(null,'Lucy',20);
 
# 查询数据
mysql> select * from myisam_test where name='Tom';
+----+-----+---+
| id | name|age|
+----+-----+---+
|  1 | Tom | 25|
+----+-----+---+
 
# 关闭myisam
mysql> show tables like'myisam%';
+-----------------+
| Tables_in_mysql |
+-----------------+
| myisam_test     |
+-----------------+
 
mysql> alter table myisam_test engine = InnoDB;
 
mysql> show tables like'myisam%';
+-----------------+
| Tables_in_mysql |
+-----------------+
| myisam_test     |
+-----------------+
 
# 查看数据页
mysql> use information_schema;
 
mysql> SELECT 
    TABLE_SCHEMA, 
    TABLE_NAME, 
    DATA_LENGTH/1024 AS DATA_LENGTH_MB, 
    INDEX_LENGTH/1024 AS INDEX_LENGTH_MB 
FROM 
    TABLES WHERE 
    TABLE_TYPE LIKE '%myisam%' AND 
    ENGINE LIKE '%myisam%' ORDER BY 
    DATA_LENGTH DESC;
 
+------------------+--------------------+-------------+----------------+
| TABLE_SCHEMA     | TABLE_NAME         | DATA_LENGTH | INDEX_LENGTH   |
+------------------+--------------------+-------------+----------------+
| mysql            | columns_priv       |         197 |             22 |
| mysql            | db                 |          69 |            116 |
| mysql            | event              |          32 |              0 |
| mysql            | func               |       11049 |         5680620 |
| mysql            | general_log        |           0 |              0 |
| mysql            | help_category      |           0 |              0 |
| mysql            | help_keyword       |           0 |              0 |
| mysql            | help_relation      |           0 |              0 |
| mysql            | help_topic         |           0 |              0 |
| mysql            | ndb_binlog_index   |         168 |              0 |
| mysql            | plugin             |           0 |              0 |
| mysql            | proc               |           0 |              0 |
| mysql            | procs_priv         |           0 |              0 |
| mysql            | proxies_priv       |           0 |              0 |
| mysql            | server_cost        |           0 |              0 |
| mysql            | servers            |          69 |             40 |
| mysql            | slave_master_info  |           0 |              0 |
| mysql            | slow_log           |           0 |              0 |
| mysql            | tables_priv        |           0 |              0 |
| mysql            | time_zone          |          69 |              0 |
| mysql            | time_zone_leap_second|          76 |              0 |
| mysql            | time_zone_name     |           0 |              0 |
| mysql            | time_zone_transition|          88 |              0 |
| mysql            | time_zone_transition_type|        248 |              0 |
| mysql            | user               |          72 |              0 |
| mysql            | global_grants      |           0 |              0 |
| mysql            | db                |          16 |              0 |
| test             | myisam_test        |          36 |              0 |
+------------------+--------------------+-------------+----------------+
 
# 使用explain查看查询计划
mysql> explain select * from myisam_test where name='Tom';
*************************** 1. row ***************************
           id: 1
  select_type: SIMPLE
        table: myisam_test
   partitions: NULL
         type: ALL
possible_keys: PRIMARY
          key: PRIMARY
      key_len: 8
          ref: NULL
         rows: 3
     filtered: 100.00
        Extra: Using index
*************************** 1 row in set (0.00 sec)
 
 
# 删除表
mysql> drop table myisam_test;
 
# 打开innodb
mysql> create table innodb_test (
    -> id int primary key auto_increment,
    -> name varchar(20),
    -> age int);
 
mysql> show tables like 'innodb%';
+-----------------+
| Tables_in_mysql |
+-----------------+
| innodb_test     |
+-----------------+
 
mysql> insert into innodb_test values (null,'Tom',25),(null,'Jack',30),(null,'Lucy',20);
 
# 修改表引擎
mysql> alter table innodb_test engine = myisam;
 
mysql> show tables like 'innodb%';
+-----------------+
| Tables_in_mysql |
+-----------------+
| innodb_test     |
+-----------------+
 
# 清除缓存
mysql> flush tables with read lock; -- 读取锁表
 
mysql> unlock tables; -- 释放锁表
 
# 查询数据
mysql> select * from innodb_test where name='Tom';
Empty set (0.00 sec)
 
# 使用explain查看查询计划
mysql> explain select * from innodb_test where name='Tom';
*************************** 1. row ***************************
           id: 1
  select_type: SIMPLE
        table: INNODB_SYS_TABLES
   partitions: NULL
         type: SYSTEM
        rrn: NULL
        fsn: NULL
       rows: NULL
filtered: 100.00
        Extra: NULL
*************************** 1 row in set (0.00 sec)
 
```