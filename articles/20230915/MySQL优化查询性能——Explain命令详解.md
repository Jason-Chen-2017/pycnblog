
作者：禅与计算机程序设计艺术                    

# 1.简介
  

MySQL数据库是一个关系型数据库管理系统，其使用了一种叫做基于行的存储引擎来存储数据。由于设计目标就是处理大量数据的高速查询，所以在查询优化方面做了很多工作。如果不对查询进行优化，那么查询效率将会很低。因此，对于查询优化来说，一条好的SQL语句至关重要。本文将详细介绍Explain命令，并通过具体的例子来说明其使用方法。

## Explain命令是什么？
Explain命令可以分析SQL语句执行的详细信息，包括查询计划、索引信息、执行时间等。它可以帮助DBA找出最优的查询计划，避免不必要的昂贵资源消耗。Explain命令输出结果中的“id”字段表示每个SELECT子句或操作符的标识符号，从上到下按顺序依次递增。通过“type”字段的值，可以判断MySQL数据库引擎到底采用了哪种访问方法来处理表的数据。

## 使用Explain命令有什么好处？
1. 可以直观地看出MySQL数据库的执行计划，从而确定是否需要对查询进行优化；
2. 可以分析优化器选用的索引、条件过滤等信息，对查询进行进一步的分析；
3. 可以检测到查询优化器未考虑到的列访问情况，从而更好地理解索引选择和查询计划的机制；
4. 可以找出查询中存在的错误或者不合理的配置，从而降低查询效率；
5. 通过explain命令，可以让DBA跟踪慢查询，找出资源瓶颈所在；
6. 通过explain command，可以了解到锁的使用情况，发现并解决死锁问题。 

## Explain命令语法及用法
```sql
EXPLAIN statement; 
```
Explain命令用于分析SQL语句的执行过程，并返回一张数据表作为结果。其语法形式如下所示，statement表示要执行的SQL语句。

## 示例
### 场景一: 没有索引的情况下查询多表关联查询,explain语句分析查询计划
#### 数据准备
```sql
-- 创建表
CREATE TABLE user (
  id INT(11) PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(20),
  password CHAR(32),
  age INT(11) NOT NULL DEFAULT '0',
  sex ENUM('male','female') NOT NULL DEFAULT'male'
);

CREATE TABLE role (
  id INT(11) PRIMARY KEY AUTO_INCREMENT,
  rolename VARCHAR(20),
  description TEXT
);

CREATE TABLE access (
  id INT(11) PRIMARY KEY AUTO_INCREMENT,
  role_id INT(11),
  resource_url VARCHAR(255),
  action VARCHAR(20),
  CONSTRAINT fk_role FOREIGN KEY (role_id) REFERENCES role(id) ON DELETE CASCADE ON UPDATE NO ACTION
);

INSERT INTO `user` (`username`, `password`, `age`, `sex`) VALUES ('admin', MD5('123456'), 25,'male');
INSERT INTO `user` (`username`, `password`, `age`, `sex`) VALUES ('test', MD5('abcde'), 20, 'female');
INSERT INTO `user` (`username`, `password`, `age`, `sex`) VALUES ('user1', MD5('123456'), 25,'male');
INSERT INTO `user` (`username`, `password`, `age`, `sex`) VALUES ('user2', MD5('123456'), 30,'male');
INSERT INTO `user` (`username`, `password`, `age`, `sex`) VALUES ('user3', MD5('123456'), 35,'male');

INSERT INTO `role` (`rolename`, `description`) VALUES ('admin', 'administrator');
INSERT INTO `role` (`rolename`, `description`) VALUES ('tester', 'test');
INSERT INTO `role` (`rolename`, `description`) VALUES ('developer', 'develop');

INSERT INTO `access` (`role_id`, `resource_url`, `action`) VALUES (1, '/','read');
INSERT INTO `access` (`role_id`, `resource_url`, `action`) VALUES (2, '/users/create', 'write');
INSERT INTO `access` (`role_id`, `resource_url`, `action`) VALUES (2, '/users/*','read');
INSERT INTO `access` (`role_id`, `resource_url`, `action`) VALUES (3, '*', 'all');

```
#### 查询SQL语句
```sql
SELECT u.*, r.* FROM user AS u INNER JOIN role AS r ON u.id = r.id LEFT JOIN access AS a ON r.id = a.role_id WHERE u.username LIKE '%test%' AND EXISTS (SELECT * FROM access AS b WHERE u.id = b.role_id AND b.action IN ('all'));
```
该查询语句没有给任何索引，而且涉及三个表的关联查询，关联条件是WHERE子句中的u.id=r.id。因此，我们可以使用Explain命令查看查询计划。

#### 执行 explain 命令
```sql
EXPLAIN SELECT u.*, r.* FROM user AS u INNER JOIN role AS r ON u.id = r.id LEFT JOIN access AS a ON r.id = a.role_id WHERE u.username LIKE '%test%' AND EXISTS (SELECT * FROM access AS b WHERE u.id = b.role_id AND b.action IN ('all'));
```
执行explain后，得到以下结果。其中，各列的含义如下：

- id：SELECT的序列号，由上到下按顺序递增。
- select_type：SELECT类型，主要是用于区分普通查询、联合查询、子查询等。
- table：显示这一行的数据是关于哪个表的。
- type：数据访问类型，常用的类型有ALL、index、range等。
- possible_keys：指出MySQL可能选用的索引。
- key：实际使用的索引。
- key_len：索引长度。
- ref：表示上述表的连接匹配条件。
- rows：扫描的行数估计值。
- Extra：包含其他信息，如using filesort表示 MySQL 会额外再排序一次找到的匹配行。 

根据查询计划，我们可以知道，优化器认为第一个JOIN的关联列r.id可以加索引。并且没有任何筛选条件，因此可以直接利用全表扫描的方式来查询。也就是说，由于没有索引和关联列的限制，因此直接导致整个查询被优化器认为是一个全表查询。此时，我们应该注意到queries_examined和rows列的值非常大，其值为1或大。

为了提高查询效率，可以通过创建索引或优化查询方式进行优化。例如，给r.id增加索引，就可以大幅度提升查询速度：
```sql
ALTER TABLE role ADD INDEX idx_rid (id);
```
然后重新运行explain，可以看到查询计划发生了变化，query_executemnt的值明显减少。另外，也需要注意到在应用中应尽量避免使用模糊查询，否则索引失效的概率增加，查询效率降低。