
作者：禅与计算机程序设计艺术                    

# 1.简介
  

PostgreSQL（Postgres）是一个开源的对象关系型数据库管理系统（ORDBMS），开发者为麻省理工学院的Andrew W. Mellon。该数据库利用了先进的开放源代码理念、丰富的功能和强大的性能，被广泛应用于各种规模的网站和大数据应用中。为了方便国内开发者更好地了解PostgreSQL，本文档基于PostgreSQL 9.6版本进行翻译编写。

# 2.起源
PostgreSQL从1986年由加里·马拉顿（<NAME>）在芬兰的赫尔辛基大学攻读硕士时期开发而成。创始人的名字叫做查尔斯·贝宁（<NAME>）和约翰·彼得森（John Porter）。他们希望它能够成为类似Unix或Sybase的关系数据库管理系统。当时的计算机资源还相对较少，因此，他们开发了一个设计简单、运行速度快的数据库管理系统。后来的版本则逐渐添加新特性来满足更多用户的需求。2000年，发行版Linux支持了PostgreSQL作为其默认的数据库，Linux操作系统也成为了PostgreSQL的最佳使用平台。

2003年，加利福尼亚大学的Marc Fournier教授团队开发了一款名为PGfoundry的工具箱，为PostgreSQL提供了丰富的特性。同时，另一个创始人的名字叫做史蒂夫·伯纳斯·李（Steve Jobs）。他希望能够推出一款新的操作系统——Mac OS X，同时支持PostgreSQL。不过，由于工程师能力所限，这一项目并没有取得预期效果。

2006年，Facebook收购了Macy’s（马克韦森）公司，Macy’s将建立一套完整的业务线解决方案。其中包括其自有的“Social Graph”服务。这套服务使用PostgreSQL作为底层数据库，为用户提供丰富的社交网络功能。Facebook选择PostgreSQL主要还是因为其易用性，操作效率高，而且社区活跃。

2010年，红帽公司收购了雅虎研究院（Yahoo! Research Labs）的Neeva项目。Neeva项目由两个成员组成，分别是Brian Dean和David Boyton。Neeva项目致力于为包括雅虎在内的多个互联网企业提供搜索引擎服务。但是，由于Neeva内部的分歧，最终失败。

2012年，纽约时报（The New York Times）出版了一本名为《SQL: A Beginner's Guide to Data Manipulation》的书。这本书详细介绍了SQL语言。该书作者卡尔·弗莱明（<NAME>）教授使用MySQL作为示例，并介绍了如何使用SQL语句来操控数据库。

2012年9月，PostgreSQL项目发布了它的第一个正式版本。

2013年初，Red Hat收购了CitusData公司，推出了一个名为Citus的分布式数据库解决方案，为其客户提供无限可扩展的数据库。Red Hat选择PostgreSQL作为Citus集群的基础。

2013年7月，PostgreSQL进入了维护状态，并发布了它的第五个版本，即9.3版本。此外，2014年初发布的PostgreSQL 9.4版本中增加了大量新特性。

2015年5月，PostgreSQL 9.5版本正式发布。此版本新增了诸如逻辑复制、视图、存储过程等多项特性，并且优化了性能和稳定性。

2016年2月，Postgreslq 9.6版本正式发布。此版本对性能和安全性进行了优化，并新增了物化视图、多范围索引、执行计划统计信息以及系统监控指标等多项特性。

# 3.基本概念术语说明
## 3.1 Postgresql概述
PostgreSQL（简称Postgresql）是一个开源的对象关系数据库管理系统（Object-Relational Database Management System，ORDBMS），由加里·马拉顿（<NAME>）博士于2003年开始开发。目前最新版本为9.6，是业界非常热门的数据库之一，是不错的选择。

Postgresql采用客户端/服务器体系结构，允许用户通过客户端应用程序直接与数据库进行交互，也可以使用不同的编程接口与数据库通信。PostgreSQL支持丰富的数据类型，包括数值类型（如整数、浮点数、复数）、字符类型（如字符串、文本、字节数组）、日期时间类型（如日期、时间、时间戳）、布尔类型、枚举类型、JSON数据类型等。

Postgresql支持数据库事务处理，也就是说，在一次会话中，所有命令都视为事务的一部分，要么全成功，要么全失败。这可以确保数据库数据的一致性，保证数据库操作的正确性及一致性。

Postgresql支持丰富的SQL命令集，包括数据定义命令（CREATE、ALTER、DROP、TRUNCATE）、数据查询命令（SELECT、INSERT、UPDATE、DELETE）、事务控制命令（BEGIN TRANSACTION、COMMIT、ROLLBACK、SAVEPOINT、RELEASE SAVEPOINT、ROLLBACK TO SAVEPOINT）、系统管理命令（GRANT、REVOKE、COMMENT ON）、声明性语言（DML、DDL、DCL、TCL）等。

Postgresql还有许多优秀特性，如可靠性、ACID兼容性、高度可扩展性、原生支持JSON数据类型、高可用性、灾难恢复、备份策略等。

## 3.2 数据类型
PostgreSQL支持丰富的数据类型，包括数值类型、字符类型、日期时间类型、布尔类型、枚举类型、JSON数据类型等。以下对每个数据类型进行说明。
### 3.2.1 数值类型
PostgreSQL支持以下数值类型：
- smallint：短整型，取值范围-32768至+32767；
- integer：整型，取值范围-2147483648至+2147483647；
- bigint：长整型，取值范围-9223372036854775808至+9223372036854775807；
- real：单精度浮点数，取值范围大约是1.17549e-38到3.40282e+38；
- double precision：双精度浮点数，通常占用两倍空间；
- numeric(p,s)：定点数，可以实现任意精度的整数算术运算，其中p表示总共的有效数字位数，s表示小数点右边的位数，因此有效数字位数为p-s；
- decimal(p,s)：同上，只不过以字符串的形式存储。

### 3.2.2 字符类型
PostgreSQL支持以下字符类型：
- char(n)：固定长度字符串，存储n个字符，超出的部分自动截断；
- varchar(n)：变长字符串，存储最大为n个字符，超出的部分存放在磁盘上的一张表中；
- text：用于存储长文本数据，存储的是一段不可分割的文本。

### 3.2.3 日期时间类型
PostgreSQL支持以下日期时间类型：
- date：日期类型，表示年、月、日；
- time [（p])：时间类型，表示时、分、秒、微秒，可选的参数p指定时间精度；
- timestamp [（p])：时间戳类型，表示日期和时间，精度最多到纳秒级；
- interval [fields] [(\[precision\])]：时间间隔类型，表示时间之间的差异，字段可以是YEAR、MONTH、DAY、HOUR、MINUTE、SECOND、MILLISECOND。

### 3.2.4 布尔类型
PostgreSQL支持两种布尔类型：boolean、bool。前者用于存储TRUE和FALSE，后者用于存储1和0。

### 3.2.5 枚举类型
PostgreSQL支持ENUM类型，语法如下：
```
CREATE TYPE color AS ENUM ('red', 'green', 'blue');
```
该类型只能存储三种颜色的值：'red', 'green', 'blue'。其他的值不能插入到这个类型列中。

### 3.2.6 JSON数据类型
PostgreSQL 9.4版本引入了JSON数据类型。JSON数据类型可以用于存储和检索符合JSON标准的数据。它支持两种JSON值类型：
- json：存储JSON对象、数组和原始类型值；
- jsonb：存储JSONB对象、数组和原始类型值，并压缩储存，适合大型数据。

JSONB数据类型可以显著提升查询性能，尤其是在多列组合索引的情况下。JSONB数据类型在PostgreSQL中是不透明的，只有客户端代码才能解析和操作它。但是，客户端代码可以通过相应的库函数来解析JSONB数据类型。

## 3.3 SQL语法
PostgreSQL支持丰富的SQL命令集，包括数据定义命令（CREATE、ALTER、DROP、TRUNCATE）、数据查询命令（SELECT、INSERT、UPDATE、DELETE）、事务控制命令（BEGIN TRANSACTION、COMMIT、ROLLBACK、SAVEPOINT、RELEASE SAVEPOINT、ROLLBACK TO SAVEPOINT）、系统管理命令（GRANT、REVOKE、COMMENT ON）、声明性语言（DML、DDL、DCL、TCL）等。

### 3.3.1 DDL命令
PostgreSQL中的数据定义语言（Data Definition Language，DDL）命令用来创建、修改和删除数据库对象，包括表、视图、序列、索引、函数、存储过程等。

DDL命令一般用于定义数据库对象的结构和特征。例如：
```sql
-- 创建数据库mydb
CREATE DATABASE mydb;

-- 创建表mytable
CREATE TABLE mytable (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    description TEXT
);

-- 添加一列
ALTER TABLE mytable ADD COLUMN email VARCHAR(100);

-- 删除一列
ALTER TABLE mytable DROP COLUMN age RESTRICT;

-- 修改表名称
ALTER TABLE mytable RENAME TO myothertable;

-- 删除表
DROP TABLE mytable;
```

### 3.3.2 DML命令
PostgreSQL中的数据操纵语言（Data Manipulation Language，DML）命令用来读取、写入和更新数据库记录。

DML命令一般用于对数据库中的数据进行增、删、改操作。例如：
```sql
-- 插入一条记录
INSERT INTO mytable (name,description,email) VALUES ('Alice','This is Alice''s record.','alice@example.com');

-- 更新一条记录
UPDATE mytable SET email='bob@example.com' WHERE id=1;

-- 删除一条记录
DELETE FROM mytable WHERE id=1;
```

### 3.3.3 TCL命令
事务控制语言（Transaction Control Language，TCL）命令用来管理数据库事务。

TCL命令一般用于启动、回滚和提交事务。例如：
```sql
-- 开启事务
BEGIN;

-- 提交事务
COMMIT;

-- 回滚事务
ROLLBACK;
```

### 3.3.4 函数
PostgreSQL支持PL/pgSQL、SQL/PL和Java语言编写的函数。

PL/pgSQL是PostgreSQL中一种用来编写存储过程的程序语言。它结合了SQL语言的一些特点，并引入了一些特定于PL/pgSQL的语法。

SQL/PL用于编写具有特定功能的数据库对象，例如触发器、类型转换器和窗口函数。SQL/PL语言比较复杂，但提供了更大的灵活性和功能。

Java函数用于编写带有复杂算法的自定义函数。

### 3.3.5 模板
PostgreSQL支持模板，可以根据实际需要快速地创建对象。

模板可以把一些通用的属性和结构定义成模板，然后根据需要生成对应的对象。例如，可以定义一个名为"employee_t"的模板，用于创建员工相关的表，这样就可以根据需要快速地创建员工表。