
作者：禅与计算机程序设计艺术                    

# 1.简介
  

目前市面上存在很多开源数据库系统，如MySQL、PostgreSQL等。对于某些业务场景，比如对中文数据的支持非常重要。因此，在决定选择MySQL作为应用数据存储的时候，需要考虑字符集的问题。

MySQL数据库默认使用的字符集是latin1。虽然它能够处理多种语言，但对于中国人来说，日文、韩文等非ASCII字符就无法正常显示，很难满足应用需求。另外，不同字符集之间转换也会带来性能损失。

所以，如果要支持更加丰富的语言及特殊字符，或者对搜索引擎优化（SEO）有较大的帮助，则需要选择更加健全的字符集配置。本文将介绍MySQL中常用的几种字符集及其优劣。

# 2.字符集介绍
## 2.1 ASCII字符集(ASCII charset)
最早的计算机系统采用的是ASCII码编码，它只使用7位二进制表示法，前128个字符是被保留的标准ASCII码。但随着计算机系统的发展，越来越多的人们迫切地需要各种各样的文字。为了解决这个问题，后来的计算机制造商又设计出了各种语言字符集，如汉字字符集、西里尔文字符集、日本字符集等。这些字符集一般都是基于标准ASCII码扩展得到的，但通常每个字符集都有自己独特的编码规则，使得它们不能互通。比如，中文字符集GBK的编码范围是8140-FEFE，而ASCII字符集只能编码到7F。这样，当一个系统同时兼容多个字符集时就会出现乱码的问题。

## 2.2 ISO/IEC 8859字符集系列
ISO/IEC 8859共收录了十五种语言所使用的字符集，分别对应Latin-1(第一种字符集)至Latin-9(第九种字符集)。它们均使用相同的编码机制，只是规定了不同的范围。比如，Latin-1的编码范围是00-FF，即前128个字符；Latin-9的编码范围是80-FF，即包括前128个字符以及其后面所有扩展字符。

这十五种字符集都属于ASCII字符集系列，但因为它们各自的字符集范围不同，导致它们之间的转换相当费时，甚至有的还会出现乱码。因此，在创建MySQL数据库时，我们应该尽量选用完整的ISO/IEC 8859字符集系列之一，而不是只选择其中一种字符集。

## 2.3 Unicode字符集
Unicode是由万国语言联盟提出的一种字符编码方案，是目前世界上最通用的字符集。它涵盖了当前和历史上所有的语言和文化。Unicode使用16位表示每个字符，可以表示超过三百万种字符。而目前主流的Web浏览器都已经开始支持Unicode字符集。

然而，MySQL数据库并不直接支持Unicode字符集。由于历史原因，MySQL的字符集有两种设定方式：

1. 设置数据库级别的默认字符集，影响所有表字段的字符集
2. 在创建或修改表时指定字段的字符集

但无论哪种设定方式，MySQL都会默认采用3字节UTF-8字符集。也就是说，如果要使用Unicode字符集，必须采用第2种方法设置。否则，所有字符串都将按照3字节UTF-8字符集进行存储。

## 2.4 其它字符集
还有一些其他的字符集，如KOI8-R、EUC-JP等。它们各有特色，但都不是完全通用的字符集。

# 3.MySQL数据库字符集配置
## 3.1 查看当前字符集
查看MySQL服务器当前的字符集配置，可以使用以下语句：
```mysql
SHOW VARIABLES LIKE 'character_set_%';
```
其中%表示匹配任何关键字，显示结果类似如下：
```mysql
+--------------------------+----------------------------+
| Variable_name            | Value                      |
+--------------------------+----------------------------+
| character_set_client     | utf8                       |
| character_set_connection | utf8                       |
| character_set_database   | utf8mb4                    |
| character_set_filesystem | binary                     |
| character_set_results    | utf8                       |
| character_set_server     | latin1                     |
+--------------------------+----------------------------+
6 rows in set (0.00 sec)
```
可以通过`character_set_server`项看到服务器的默认字符集，此处输出值为`latin1`。

## 3.2 修改默认字符集
修改MySQL服务器的默认字符集，可以使用以下SQL语句：
```mysql
ALTER DATABASE <数据库名> CHARACTER SET = <新的默认字符集>;
```
例如，修改默认字符集为`utf8mb4`，可以使用以下语句：
```mysql
ALTER DATABASE mydb DEFAULT CHARACTER SET utf8mb4;
```
这条语句不会立即生效，只有当新建或打开连接时才会生效。也可以修改配置文件`/etc/my.cnf`，添加以下行：
```
[mysqld]
default-character-set=utf8mb4
```
然后重启MySQL服务。

## 3.3 创建新表指定字符集
创建新表时可以指定字段的字符集，可以使用以下SQL语句：
```mysql
CREATE TABLE <表名> (<字段名> <数据类型> CHARACTER SET <字符集>) ENGINE=<存储引擎>;
```
例如，创建一个字符集为`utf8mb4`的表，可以使用以下语句：
```mysql
CREATE TABLE test (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci
);
```
这里指定了id字段的数据类型为INT，name字段的数据类型为VARCHAR(100)，字符集为`utf8mb4`。注意，字段的排序规则由collation决定。如果没有指定collation，那么将使用默认排序规则。

## 3.4 修改已有表指定字符集
修改已有表的字符集，可以使用以下SQL语句：
```mysql
ALTER TABLE <表名> CONVERT TO CHARACTER SET <新字符集>;
```
例如，将test表的字符集从`latin1`改为`utf8mb4`，可以使用以下语句：
```mysql
ALTER TABLE test CONVERT TO CHARACTER SET utf8mb4;
```
注意，CONVERT TO命令会先将原字符集的字符重新编码成新字符集的字符，然后再更新表结构。因此，这种方式可能比较耗时，并且会锁住表，直到操作完成。如果数据量很大，建议先使用`SELECT INTO OUTFILE`命令导出数据，然后导入到新表。