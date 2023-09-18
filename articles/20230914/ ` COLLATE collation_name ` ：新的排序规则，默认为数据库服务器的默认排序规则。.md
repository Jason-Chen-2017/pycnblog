
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 一、引言

自从上世纪90年代末期IBM推出了第一版关系型数据库系统，到如今主流的关系型数据库产品大多都已经历经了几十年的发展，并且很难避免会存在各式各样的排序规则。在查询的时候如果不加以控制或者按照默认的排序规则进行排序的话，则可能导致数据出现错误或查询效率低下等问题。

## 二、历史背景

早在1974年，美国国家标准局（National Institute of Standards and Technology，NIST）发布了“建议960”。其中规定了一种通用数据格式的分类标准，包括ASCII字符、EBCDIC字符、可变长度字符、整形数字、浮点数字、时间和日期、Unicode字符串。NIST认为Unicode字符序列应当优先于ASCII和EBCDIC字符序列作为通用数据格式。但是由于历史原因以及计算机硬件技术水平的限制，使得许多应用软件仍然依赖于传统的ASCII字符集。

后来，一些软件开发商为了照顾不同地区用户的需要，提供了两种解决方案。第一种方法是提供双重编码支持，即同时支持ASCII和EBCDIC两个字符集，因此用户可以根据自己的习惯选择合适的字符集进行编码；第二种方法是强制所有文本文件采用统一的字符集，这也是目前很多桌面应用程序所采用的方式。

基于这种历史背景，为了更好地处理跨平台的数据一致性问题，关系型数据库领域内引入了排序规则这一概念，它用于定义特定语言环境下的文本的比较和排序顺序。虽然排序规则可以影响数据库的效率和功能，但毕竟随着时间的推移，各个排序规则之间往往存在相互兼容的问题。

## 三、排序规则概述

排序规则（collation）是用来确定数据列值排序时的逻辑规则。它包括区域设置信息（如语言、国家/地区）、词序大小写比较的规则（如A为小写字母，a为大写字母）、宽字节对齐的规则、数值比较的规则等。每个数据库都有自己默认的排序规则，不同的数据库之间排序规则也可能不同。一般情况下，用户无法修改数据库的排序规则，除非使用特殊的SQL语句（如CREATE DATABASE、ALTER DATABASE SET DEFAULT COLLATION、CREATE TABLE或ALTER TABLE ADD CONSTRAINT）。当然，某些情况下也可能需要修改排序规则，例如搜索引擎索引构建、外部数据导入等场景。

在MySQL中，我们可以使用COLLATE子句来指定创建表或列时使用的排序规则。例如：

```sql
CREATE TABLE mytable (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL COLLATE utf8mb4_general_ci
);
```

上面的例子中，将name列的排序规则设置为utf8mb4_general_ci，表示该列的数据类型为VARCHAR，其中的排序规则为utf8mb4的通用排序规则。我们可以通过SHOW COLLATION命令查看当前MySQL服务器所支持的所有排序规则：

```sql
mysql> SHOW COLLATION;
+---------+-------------+------+-----+---------+-------+
| Collation | Character Set | Id   | Sortlen | Charset | Description |
+---------+-------------+------+-----+---------+-------+
| armscii8 | ascii        |   46 |    1 | binary  | Armenian (armscii-8) |
| big5_chinese_ci | big5 |   10 |    1 | binary | Big5 Traditional Chinese |
...
| utf8mb4_bin | utf8mb4      |  464 |    1 | binary  | Unicode (multibyte sequence encoding form) |
| utf8mb4_general_ci | utf8mb4      |  450 |    1 | binary  | General Unicode |
+---------+-------------+------+-----+---------+-------+
58 rows in set (0.00 sec)
```

这里给出一个常见的排序规则名称及其含义。其他类型的排序规则还包括：

* 版本化排序规则（Versioned Collations）：它主要用于处理有多个版本的数据。通常用于处理历史遗留数据。

* 字符类排序规则（Character Class Collations）：它们通过映射某些字符类到其他字符来实现排序，比如通过映射数字字符到字母字符实现大小写敏感的排序。

* ICU（International Components for Unicode）排序规则：它是在Unicode规范之上增加了语言环境信息，并支持多种语言之间的排序规则。

* 第三方排序规则（Third-Party Collations）：这些排序规则是在开源社区的帮助下建立的，并且没有被MySQL官方认可。

实际上，除了使用默认排序规则外，我们也可以创建自定义的排序规则。例如：

```sql
CREATE COLLATION mycol USING myrule FROM 'zh_CN' WITH PAD SPACE FIRST SECONDARY 'GBK';
```

上面的示例创建一个名为mycol的新排序规则，它由名为myrule的规则组成，这个规则又使用中文语言环境（'zh_CN'），并且启用了中文后空格（PAD SPACE FIRST）的前缀规则，以便忽略空白符号进行排序。此外，该规则的次要排序规则设定为GBK，也就是说，对于无法用中文排序的字符，将按GBK的排序规则进行排序。

## 四、MySQL排序规则实现机制

MySQL的排序规则是如何工作的呢？以下是简单的介绍：

1. 当MySQL服务启动时，它首先读取配置文件中的sort_buffer_size参数的值，然后分配相应的内存空间给排序缓冲区。

2. 当执行排序操作时，MySQL客户端向服务器发送一条SELECT或INSERT语句，要求返回结果集或者更新记录。

3. MySQL服务接收到请求之后，首先检查是否有相同的SELECT或INSERT语句正在运行。

4. 如果没有同样的语句正在运行，则MySQL服务为该语句分配一个线程。

5. 当线程被分配给某个客户端后，该线程会打开临时表，用来存储查询结果。

6. 在得到查询结果的同时，MySQL服务又会生成一个MERGE_SORT任务，把结果集中有关的数据插入到这个临时表中。

7. 此时，MySQL服务将分配一个额外的线程来执行MERGE_SORT任务。

8. MERGE_SORT任务首先对数据进行排序，然后将排序好的结果集合并到临时表中。

9. 当MERGE_SORT任务结束后，临时表中的数据就可以按照用户指定的排序规则返回给客户端了。

10. 返回结果的过程中，如果发现数据里面有“二进制”数据，那么MySQL就不会再进一步解析和排序了。而是直接按照原始字节码的方式进行输出。

11. 在整个过程中，只有少量的排序相关的数据在内存中参与交换，因此即使排序规则较复杂，也还是能在线上快速响应。