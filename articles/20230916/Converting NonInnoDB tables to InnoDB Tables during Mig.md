
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在MySQL数据库中，对某些类型的表（如MyISAM、MEMORY等）进行优化处理时需要将其转换成其他类型的表(如InnoDB)，这样才能利用到其特有的功能特性。因此，当用户从其他数据库迁移数据到MySQL时，可能遇到这种情况。

一般情况下，对于非InnoDB类型的表，系统会自动根据查询语句的条件和具体执行计划进行表优化。但是，由于转换过程耗费时间，可能会导致应用不可用。所以，提前规划好数据库的表结构及类型，可以有效避免出现性能下降或业务受影响的问题。本文将详细阐述如何利用数据库工具实现非InnoDB表的转换。

# 2.相关概念
## 2.1 MyISAM和InnoDB
MyISAM和InnoDB都是MySQL的存储引擎。MyISAM是一个非常古老的存储引擎，它的设计目标就是快速和高效的读写操作。它被最初用于较小型的数据库，主要包括静态表和一些临时表。

InnoDB存储引擎是另一个最新引擎，是由Oracle数据库的裔 AUTHOR DORMAN NUTTBASELINE开发，其特点是在Web应用方面有更好的性能。相比于MyISAM，InnoDB支持事务安全型表锁定(table-level locking)。

## 2.2 数据迁移
数据库迁移又称为异构数据库间的数据交换，目的是将一个数据库中的数据导入到另一个数据库系统中。数据迁移的方式通常有两种：

第一种方式：完全导入导出。在这种方式中，首先将源数据库中的所有数据导出到一个文件，然后再将这个文件导入到目标数据库中。

第二种方式：按需迁移。在这种方式中，目标数据库中存在一张“控制表”，其中的记录表示哪些表需要被迁移。当应用访问这些表中的数据时，如果检测到该表还不存在或者数据的版本不匹配，则可以启动数据迁移流程。

## 2.3 Online Schema Change
Online Schema Change (OSC) 是一种在线修改数据库模式的方法。通过使用基于复制日志的异步复制机制，OSC 可以将数据库的架构更改推送到整个集群中的所有服务器，而无需停机。在某些情况下，OSC 可帮助减少停机时间和风险。 

通过适时的使用 OSC，数据库的架构可逐渐演进，从而解决了传统的重启数据库后重新同步数据所带来的延迟问题。

# 3.核心算法原理和具体操作步骤
## 3.1 准备阶段
1. 确认上游MySQL版本号是否大于等于5.6.

2. 从上游服务器导出要迁移的表的数据。

## 3.2 转换过程
1. 创建一个空的InnoDB表。

   CREATE TABLE new_table LIKE old_table;
   ALTER TABLE new_table ENGINE=InnoDB;
   
2. 在新的InnoDB表中插入上游表的数据。

   INSERT INTO new_table SELECT * FROM old_table; 
   
3. 删除上游表。

   DROP TABLE old_table;
   
4. 为新表创建触发器和索引。

   
5. 修改应用程序配置，连接新的InnoDB表。

## 3.3 恢复阶段
1. 根据预留的时间窗口，确保数据迁移完成。

2. 检查新的InnoDB表的数据是否完整无缺。

3. 验证所有应用程序的查询都正常工作。

# 4.具体代码实例及解释说明
## 4.1 准备阶段
1. 使用mysqldump导出表数据。

   mysqldump -h host -u username -p password --databases dbname table > dumpfile.sql

2. 通过互联网上传导出的SQL文件到目标服务器。

## 4.2 转换过程
1. 在目标服务器上创建一个空的InnoDB表。

   CREATE TABLE new_table LIKE old_table;
   ALTER TABLE new_table ENGINE=InnoDB;
   
2. 使用mysql命令行客户端导入数据。

   mysql -h host -u username -p password < dumpfile.sql

3. 删除上游表。

   DROP TABLE old_table;
   
4. 为新表创建触发器和索引。

   CREATE TRIGGER trigger_name BEFORE UPDATE ON new_table FOR EACH ROW SET NEW.modified = NOW();
   CREATE INDEX index_name ON new_table(column);
   
5. 修改应用程序配置文件，连接新的InnoDB表。

   option1: 修改代码里面的链接地址。

   option2: 配置别名，方便之后切换。

## 4.3 恢复阶段
1. 根据预留的时间窗口，确保数据迁移完成。

2. 检查新的InnoDB表的数据是否完整无缺。

   SELECT COUNT(*) FROM new_table;

3. 验证所有应用程序的查询都正常工作。