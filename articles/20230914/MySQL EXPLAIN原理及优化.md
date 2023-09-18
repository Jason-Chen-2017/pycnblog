
作者：禅与计算机程序设计艺术                    

# 1.简介
  

EXPLAIN（Explain Explains），中文可翻译为“说明”，这个命令是用于分析MySQL执行计划的一个工具。由于优化器会根据实际情况生成执行计划，所以开发人员在编写SQL查询语句时经常需要借助于explain命令进行优化查询性能方面的分析。本文从三个方面对explain命令进行了深入浅出的介绍：
- 执行计划的组成元素
- explain的语法和用法
- explain的输出结果
- 优化查询性能的方法论
通过阅读本文，可以了解到如何利用explain命令进行数据库查询性能分析，并知道常用的优化查询方案，以便提升数据库的处理性能、节省资源。
# 2.执行计划的组成元素
explain命令的执行计划输出由以下几个部分组成：

1. id：表示每个select或union等子句的序列号，不同子句的id值不同；

2. select_type：表示查询类型，如SIMPLE、PRIMARY、SUBQUERY等；

3. table：表示查询涉及的表名；

4. type：表示查询访问类型，如ALL、INDEX、RANGE等；

5. possible_keys：表示查询可能用到的索引；

6. key：表示真正使用的索引；

7. key_len：表示索引中被选取的列的长度；

8. ref：表示索引的引用情况；

9. rows：表示扫描行数，估算值；

10. Extra：表示额外信息，如Using filesort表示使用文件排序。

我们将结合下面几个例子进一步学习explain命令输出的结构：
```mysql
explain select * from t where a = 1 and b = 'abc' limit 1;
explain select * from t use index (idx) order by c desc limit 10;
explain select * from t force index(idx);
```
第一个例子中的SQL语句只有两个条件限制，没有任何索引用到了，因此它的查询类型为SIMPLE。对应的执行计划输出如下：
```mysql
id	select_type	table	partitions	type	possible_keys	key	key_len	ref	rows	filtered	Extra
1	SIMPLE	t	NULL	ALL	NULL	NULL	NULL	NULL	8208	8208	NULL
```
第二个例子中的SQL语句有索引，并且选择的是按照c倒序排序，因此它的查询类型为index。对应的执行计划输出如下：
```mysql
id	select_type	table	partitions	type	possible_keys	key	key_len	ref	rows	filtered	Extra
1	IndexScan	t	NULL	index	idx	idx	2	const,const	10000	10000	Using index
```
第三个例子中的SQL语句强制指定使用索引idx，因此它的查询类型为ref。对应的执行计划输出如下：
```mysql
id	select_type	table	partitions	type	possible_keys	key	key_len	ref	rows	filtered	Extra
1	Ref	t	NULL	range	idx	idx	2	eq_ref	1	1	Using index condition
```
这三种情况下，SELECT查询都是返回了所有满足条件的数据行，只是展示了不同的执行计划方式。

# 3.explain的语法和用法
explain 命令的一般语法格式如下：

```mysql
explain statement [into outfile]
```
其中，statement 为要执行的 SQL 语句，除此之外，还可以用关键字 INTO OUTFILE 指定 explain 的结果输出到一个文件里，方便查看和调试。比如：

```mysql
explain select * from t into outfile '/tmp/result.txt';
``` 

另外，除了用explain命令分析执行计划之外，还可以通过慢日志分析工具mysqldumpslow进行分析。mysqldumpslow用来分析mysql服务器的慢日志文件，将其解析为易于理解的形式。 mysqldumpslow的一般语法格式如下：

```mysql
mysqldumpslow [options] /path/to/slowlog
```
其中，slowlog 是 MySQL 服务器的慢日志文件路径，通常保存在 /var/lib/mysql 下。mysqldumpslow支持的 options 有：

1. --long-query-time=<N>: 只显示超过 N 毫秒的慢查询。默认值为 1000 （单位为毫秒）。例如：

   ```mysql
   mysqldumpslow --long-query-time=500 /var/lib/mysql/mysql-slow.log
   ```
   
   将只显示时间超过 500 毫秒的慢查询。
   
2. --file=/path/to/output: 将结果输出到指定的文件。例如：

   ```mysql
   mysqldumpslow -t 120 --file=/tmp/slowqueries.txt /var/lib/mysql/mysql-slow.log
   ```
   
   将结果输出到 /tmp/slowqueries.txt 文件。
   
3. --sort-by=<column name>: 根据指定的列排序输出结果。例如：

   ```mysql
   mysqldumpslow -t 120 --sort-by="timestamp" --file=/tmp/slowqueries.txt /var/lib/mysql/mysql-slow.log
   ```
   
   会按照 timestamp 字段的值排序输出结果。
   
4. --order-by={asc|desc}: 指定排序顺序。默认是按 asc 排序。
   
5. --count={number of queries}: 指定输出结果的数量。默认值为 10。
   
6. --version: 显示当前版本信息。
   
7. --help: 显示帮助信息。