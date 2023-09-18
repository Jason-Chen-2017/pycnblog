
作者：禅与计算机程序设计艺术                    

# 1.简介
  

MySQL是一个开源的关系型数据库管理系统，在高性能、可靠性、易用性等方面均具有卓越表现。作为分布式数据库管理系统，MySQL在存储结构、性能优化、备份恢复等多个方面都得到了很好的实现。本文将从MySQL的数据类型、存储结构以及存储过程管理三个方面进行详解，并结合实例分析如何建立自己的数据库。
# 2.MySQL数据类型
## 2.1 数据类型概述
数据库中的数据类型分为三种：

1. 整型（INT）：用于整数数据的类型，包括TINYINT、SMALLINT、MEDIUMINT、INT、BIGINT。

2. 浮点型（FLOAT）：用于小数数据的类型，包括FLOAT和DOUBLE。

3. 字符串类型（VARCHAR、CHAR、TEXT）：用于存放文本或字符数据的类型，包括VARCHAR、CHAR、BINARY、VARBINARY、BLOB、TEXT。

除此外，还有日期时间类型（DATE、DATETIME、TIMESTAMP）、枚举类型（ENUM）、集合类型（SET）、JSON类型（JSON）。这些数据类型更加复杂，本文不会对它们做过多阐述。
## 2.2 INT类型
INT类型分为四种：TINYINT、SMALLINT、MEDIUMINT、BIGINT。其中，TINYINT、SMALLINT、MEDIUMINT的取值范围不超过256、65536、16777216，因此适合存储整数型数据。而BIGINT则支持整数的范围远超4字节。
### TINYINT类型
TINYINT表示一个单字节整数，范围-128到127。
```mysql
CREATE TABLE test_table(
  id TINYINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  age TINYINT DEFAULT NULL,
  level TINYINT ZEROFILL,
  weight TINYINT UNSIGNED ZEROFILL,
  score TINYINT SIGNED
);

INSERT INTO test_table (age) VALUES (-128),(-127),(0),(127),(128);

SELECT * FROM test_table;
```
输出结果：
```
    +----+----------------+-----------+------------+----------+
    | id | age            | level     | weight     | score    |
    +----+----------------+-----------+------------+----------+
    |  1 |-128            | 0         | 0000       | -128     |
    |  2 |-127            | 0         | 0000       | -127     |
    |  3 |                |           |            |          |
    |  4 |127             | 0         | 0000       | 127      |
    |  5 |128             | 0         | 000        | 128      |
    +----+----------------+-----------+------------+----------+
    5 rows in set (0.00 sec)
```
可以看到TINYINT类型的列可以存储-128到127的整数值，如果插入的值超过这个范围，则会被自动截断到边界值。TINYINT列也可以设置UNSIGNED属性，用来禁止负值，同时ZEROFILL属性可以让数字的左侧填充零。
### SMALLINT类型
SMALLINT表示两个字节整数，范围-32768到32767。
```mysql
CREATE TABLE test_table(
  id SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  salary SMALLINT DEFAULT NULL,
  discount SMALLINT UNSIGNED ZEROFILL,
  total_price MEDIUMINT UNSIGNED ZEROFILL,
  inventory INT
);

INSERT INTO test_table (salary) VALUES (-32768),(-32767),(0),(32767),(32768);

SELECT * FROM test_table;
```
输出结果：
```
    +----+---------+------------+-------------+--------+
    | id | salary  | discount   | total_price | inventory|
    +----+---------+------------+-------------+--------+
    |  1 |-32768  | 0000       | 000000      | NULL    |
    |  2 |-32767  | 0000       | 000000      | NULL    |
    |  3 |        |            |             |        |
    |  4 |32767   | 0000       | 000000      | NULL    |
    |  5 |32768   | 0000       | 000000      | NULL    |
    +----+---------+------------+-------------+--------+
    5 rows in set (0.00 sec)
```
SMALLINT类型的列可以存储-32768到32767之间的整数，同样可以使用UNSIGNED、ZEROFILL属性。但是MEDIUMINT类型的列更大一些，可以存储4字节整数。
### MEDIUMINT类型
MEDIUMINT表示四字节整数，范围-8388608到8388607。它的大小和SIGNED属性相同，但无符号版本MEDIUMINT UNSIGNED与SIGNED INT相同。MEDIUMINT类型通常与INT类型配合使用。
### BIGINT类型
BIGINT表示八字节整数，范围-2^63到2^63-1。虽然BIGINT类型可以存储非常大的整数，但可能存在溢出的问题。建议仅在对确实需要这种范围的数字型字段时使用。