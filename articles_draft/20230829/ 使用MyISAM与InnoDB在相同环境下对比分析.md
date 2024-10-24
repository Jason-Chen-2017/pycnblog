
作者：禅与计算机程序设计艺术                    

# 1.简介
  

随着互联网快速发展，网站访问量的激增，网站数据库的处理量也日渐增加，因此对数据库的优化已经成为企业不可或缺的一项工作。在数据库的选择上，常用的有两种存储引擎MYISAM与InnoDB。这两种引擎各有优劣势，为了更好的选择，本文将探讨两者的区别以及相同环境下的性能对比分析。

作者：刘宁
编辑：李宏鹏、郭凌霄
2019年07月15日
# 二、引言
## 2.1 需求背景 
随着互联网网站的快速发展，数据库的容量越来越大，为了更好的提高系统的效率和稳定性，数据存储技术应运而生。数据库管理系统（DBMS）在选择存储引擎时通常会考虑三个方面：

1. 数据处理效率：MyISAM 是默认的存储引擎，它是非事务型存储引擎，其执行速度比 InnoDB 更快一些。不过 MyISAM 的索引文件和数据文件分离，导致数据不能放在内存中缓存。因此当数据较多时，在查询时速度可能会慢些；
2. 支持事物处理：支持事物处理的 InnoDB 提供了崩溃恢复能力，保证数据的一致性。事物提交之前的中间状态不会被破坏，可以进行事务回滚，确保数据的完整性和安全性；
3. 行级锁定机制：MyISAM 不支持行级锁定，只能对整张表加锁，这限制了并发处理的能力。InnoDB 支持行级锁定，可以有效支持多用户读写同一表的数据同时进行。但是 InnoDB 暂不支持全文搜索和空间索引等高级功能。

从以上三点可以看出，在相同的硬件资源下，如果需要高速读写的场景，那么推荐使用 InnoDB，否则可以使用 MyISAM 。

## 2.2 系统环境描述 
### 2.2.1 操作系统版本 
CentOS release 6.x
### 2.2.2 MySQL版本 
MySQL 5.5.60 (mysql-community-server-5.5.60-1.el6_9.x86_64)
### 2.2.3 CPU、RAM、磁盘信息 
4核CPU
16G RAM
SAS固态硬盘

# 三、基本概念术语说明
## 3.1 事务 
事务(transaction)是由多个数据库操作组成的一个逻辑过程，要么都成功，要么都失败。事务具有四个属性：原子性、一致性、隔离性、持久性。

- 原子性（atomicity）：一个事务是一个不可分割的工作单位，事务中包括的诸操作要么都做，要么都不做。
- 一致性（consistency）：数据库总是从一个一致性的状态转换到另一个一致性的状态。
- 隔离性（isolation）：多个事务并发执行时，一个事务的执行不应该影响其他事务的执行。
- 持久性（durability）：一个事务一旦提交，对数据库中的数据的改变就永久保存下来。

## 3.2 ACID特性
ACID 是指 Atomicity、Consistency、Isolation 和 Durability 的首字母缩写，分别表示事务的原子性、一致性、隔离性、持久性。事务必须满足这四个特性才能保持数据一致性，能够确保数据的完整性和可靠性。

1. Atomicity：一个事务是一个不可分割的工作单位，事务中包括的诸操作要么都做，要么都不做。事务的原子性确保动作要么完全地执行，要么完全地不执行。
2. Consistency：数据库总是从一个一致性的状态转换到另一个一致性的状态。一致性确保事务的执行前后数据库的完整性没有遗漏、无遗漏地执行完毕。
3. Isolation：多个事务并发执行时，一个事务的执行不应该影响其他事务的执行。隔离性保证每个事务的隔离性。即一个事务内部的操作及使用的数据对并发的其他事务是隔离的，并发执行的各个事务之间不能互相干扰。
4. Durability：一个事务一旦提交，对数据库中的数据的改变就永久保存下来。只要不是系统故障或者 power loss ， committed 的事务对数据库中数据的修改就是永久性的，即使数据库发生异常重启也不会丢失该事务的操作结果。

## 3.3 InnoDB 与 MyISAM
MyISAM 与 InnoDB 是两个非常知名的 MySQL 存储引擎。其中 InnoDB 是 MySQL 默认的事务性存储引擎，支持事物的 ACID 特性，通过 redo log 和 undo log 来实现事务的原子性、持久性和隔离性。除此之外，InnoDB 可以提供行级锁定，并且还提供了众多额外的功能，如：多版本并发控制（MVCC），聚集索引，索引合并等。

MyISAM 则是 MySQL 默认的非事务性存储引擎，它提供了一个高速的读写速度，适用于绝大多数的情况。但是它的索引方式是非聚集索引，在查询时需要对全表扫描，所以速度可能会慢一些。另外 MyISAM 只支持表级锁定，只能对单张表加锁，无法实现真正的并行处理。

一般情况下，InnoDB 会比 MyISAM 有更多的优点。但是对于要求事务 ACID 特性的应用来说，InnoDB 比 MyISAM 更加适合。由于 MyISAM 在某些情况下的性能问题，很多开发人员都会选择使用 InnoDB 替代 MyISAM 。

# 四、核心算法原理和具体操作步骤以及数学公式讲解
## 4.1 两种存储引擎的区别
InnoDB 存储引擎提供了对数据库ACID特性的支持。主要特点如下：

1. 支持外键约束
2. 使用聚集索引组织数据
3. 行级锁定
4. 提供了几种日志记录方式，以便于进行错误恢复
5. 支持事务的原子性、一致性和隔离性
6. 支持MVCC
7. 支持更多存储功能

InnoDB 除了具备 InnoDB 特有的特性之外，它还是基于 B+ 树索引结构。B+ 树索引的好处在于，索引的插入、删除、修改操作都是对数级别的复杂度，对随机查询的速度非常快。因此，InnoDB 会在自适应的场景下选择聚集索引、覆盖索引，或辅助索引。

MyISAM 存储引擎也提供支持事务的 ACID 特性，但它不支持外键、行级锁定、MVCC 和其他一些 InnoDB 支持的特性。它的设计目标就是为了取代老旧的 MyISAM，所以它的索引方式也不是那种太先进的 B+ 树索引结构。MyISAM 会根据主键排序顺序建立索引，索引检索可以直接定位到数据记录所在的物理地址。

## 4.2 对比分析

| 功能 | MyISAM | InnoDB |
| :-------------: |:-------------:| -----:|
| 事务支持 | 支持 | 支持 |
| 外键支持 | 不支持 | 支持 |
| 索引支持 | 仅支持主键索引 | 支持所有类型的索引，包括聚集索引、辅助索引、唯一索引和普通索引 |
| 行级锁定 | 不支持 | 支持 |
| MVCC | 不支持 | 支持 |
| 日志记录 | 简单 | 复杂 |

## 4.3 查询过程

```sql
SELECT * FROM table_name WHERE id = value;
```

如果是 MyISAM 引擎，则会按照 ID 字段的索引顺序查找对应的值。这种查找方法比较简单，而且速度很快，但如果 ID 值不存在，则必须进行全表扫描，这就会影响效率。如果表存在多个索引，MyISAM 无法选择最优的索引，因此选择索引的策略比较随意。

InnoDB 存储引擎则不同，它在内存中缓存了一部分数据和索引。首先检查是否有 ID=value 的索引块。如果找到，则直接从索引节点中取得 ID 值的位置，读取并返回对应的数据记录。如果没有找到，则遍历整张表，直到找到 ID 值为止。这种全表扫描的方法速度相对慢一些，但可以保证每次查找都是原子化的，其效果类似于 SQL Server 中的 HOT 点查。

InnoDB 在对数据更新时也采取快照隔离策略，也就是说，InnoDB 会通过 Undo Log 把数据变化前的旧值存入临时文件，这样，如果这时候再有其他线程尝试访问这个数据，就可以直接访问 Undo Log 中存储的历史数据，保证数据的一致性。

## 4.4 创建索引过程

创建索引是一件十分费时的工作，尤其是在数据量很大的情况下。因此，对于比较小的表，选择索引可能带来一定的收益。如果要创建索引，建议按照以下几个步骤：

1. 确定需要建索引的列：首先要明确索引的目的，比如，对某些字段进行排序、分组、查找等。然后，查看表中数据量大小，如果数据量太小，就没必要建索引，反而会降低查询效率。确定要建索引的列之后，可以开始逐一分析该列的索引是否合适。

2. 选择索引类型：最常见的索引类型就是主键索引、唯一索引和普通索引。主键索引是一个聚集索引，唯一索引保证唯一性，普通索引则可以支持搜索、排序和统计数据，一般在使用范围查询时才会选择普通索引。

3. 选择索引列长度：在 MySQL 中，可以指定索引列的最大长度。过长的索引列长度可能占用较多的存储空间，并且会消耗更多的时间和资源，所以要合理选择索引列的长度。一般情况下，字符类型列的长度不超过 255，数值类型列的长度不超过 18。

4. 分析索引影响：创建索引涉及到空间的分配以及数据页的维护，所以，索引越多，占用的存储空间也就越大。同时，MySQL 也需要花时间来维护索引。因此，务必慎重选择索引。索引的性能开销主要体现在两方面：一是查询时，需要扫描索引，二是每当数据变化时，索引也要动态维护。因此，应尽量减少索引数量，也就是说，不要创建冗余的索引。

5. 执行创建索引语句：创建索引语句的语法如下：

   ```sql
   CREATE INDEX index_name ON table_name (column_list);
   ```

    column_list 为需要创建索引的列，index_name 为索引名称，table_name 为所属表。