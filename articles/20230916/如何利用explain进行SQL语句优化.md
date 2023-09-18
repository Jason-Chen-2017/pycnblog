
作者：禅与计算机程序设计艺术                    

# 1.简介
  

explain是一个非常重要的工具，用于分析执行sql查询语句并获得执行计划，提升数据库的查询效率。但是如何更好的利用explain，帮助用户发现优化点，提升数据库性能呢？本文将详细介绍 explain 的使用方法、原理及优化技巧。

## 1.背景介绍
explain 是一种用来分析 SQL 查询语句性能的方法。它可以提供关于 SQL 执行过程的详细信息，如表扫描顺序、索引选择等。通过 explain 可以查看到执行 SQL 语句的详细开销，包括各个阶段的耗时，索引使用情况，读写临时表的数量，每个节点处理的数据量等。从而为开发人员找到系统瓶颈所在、定位问题提供了依据。
同时，explain 提供了两种用法，即命令模式（command mode）和直接模式（direct mode）。在命令模式下，直接运行 explain 关键字，然后输入要分析的 SQL 语句即可。在直接模式下，通过指定参数 -analyze 来直接分析 SQL 语句，不再需要执行一条 select 语句后等待输出结果。

```
mysql> explain SELECT * FROM employees WHERE emp_no = 'E001';
+----+-------------+------------+------+---------------+---------+---------+-------+------+--------------------------+
| id | select_type | table      | type | possible_keys | key     | key_len | ref   | rows | Extra                    |
+----+-------------+------------+------+---------------+---------+---------+-------+------+--------------------------+
|  1 | SIMPLE      | employees  | ALL  | NULL          | NULL    | NULL    | const |    1 | Using where              |
+----+-------------+------------+------+---------------+---------+---------+-------+------+--------------------------+
1 row in set (0.00 sec)

mysql> explain extended SELECT * FROM employees WHERE emp_no = 'E001';
+------------------------------+------------+------+---------------+---------------+---------+---------+-------+------+---------------------------------+
| id                            | select_type | table | partitions    | group_by      | key     | key_len | ref   | rows | filtered                        |
+------------------------------+------------+------+---------------+---------------+---------+---------+-------+------+---------------------------------+
| 1                             | SIMPLE     | NULL |               | NULL          | NULL    | NULL    | NULL  | NULL | No tables used                   |
| 2                             | PRIMARY    | NULL | NULL          | emp_no        | emp_no  | 4       | const |    1 | eq(employees.emp_no, 'E001')     |
+------------------------------+------------+------+---------------+---------------+---------+---------+-------+------+---------------------------------+
2 rows in set (0.00 sec)
```

## 2.基本概念术语说明

### 2.1 概念

explain 是 SQL Server 中用于分析 SQL 查询语句性能的方法。它提供关于 SQL 执行过程的详细信息，如表扫描顺序、索引选择等。通过 explain 可查看到执行 SQL 语句的详细开销，包括各个阶段的耗时，索引使用情况，读写临时表的数量，每个节点处理的数据量等。explain 提供两种用法：命令模式和直接模式。

- 命令模式：该模式下，通过命令行的方式运行 explain ，输入 SQL 语句，返回执行计划信息。
- 直接模式：该模式下，explain 不输入 SQL 语句，而是在 SQL 语句上方加上参数 -analyze，服务器直接分析 SQL 语句并返回执行计划信息。

explain 会分析 SQL 语句的语法、逻辑结构、物理设计、查询处理和资源分配等多方面因素，给出一个详尽的执行计划。其中最重要的信息就是各个步骤花费的时间、使用的索引、涉及的表格等。



### 2.2 术语

1. **id**：每个SELECT都会生成一个唯一的ID，后续的性能调优中可以通过该ID来分析该查询的执行计划；

2. **select_type**：表示SELECT类型，可以是SIMPLE或PRIMARY，主要区分是普通的SELECT、联合查询、子查询等；

3. **table**：显示这一行的数据所来源于的表名；

4. **partitions**：若查询有分区表参与，此项显示分区名字；

5. **type**：表示MySQL的存储引擎类型，比如：ALL、InnoDB、MEMORY等；

6. **possible_keys**：可能应用到的索引；

7. **key**：实际使用的索引；

8. **key_len**：使用的索引长度；

9. **ref**：表示上述表的连接匹配条件；

10. **rows**：表示扫描的数据量；

11. **filtered**：表示数据经过过滤后的剩余比例；

12. **Extra**：显示额外信息，比如是否使用索引、Using temporary等；



## 3.核心算法原理及具体操作步骤

explain分析SQL的执行计划，可以对查询进行性能分析，确定其查询计划、消除性能瓶颈，优化查询速度。下面我们将对explain的工作原理进行简单介绍，并详细介绍其功能和使用方式。

Explain工作原理

explain 根据不同版本的 MySQL 有不同的工作原理，但总的来说，会通过计算各种统计信息，然后按照一定的规则来构造出一个执行方案，这个执行方案实际上是指导 MySQL 如何高效地执行相应的 SQL 请求的，并给出相应的建议。

它的工作流程如下：

1. 通过 ANALYZE TABLE 语句收集表级的统计信息，包括索引扫描的频次，数据页的访问次数，数据页的读取次数等等。
2. 生成 select_type 和 join_type 字段，描述的是查询类型和关联类型。
3. 生成 possible_keys 和 index fields，描述的是可能会用到的索引列。
4. 判断关联类型及索引的适用性，如果出现索引失效，则根据其他字段来确定是否应该增加索引。
5. 生成成本估算值，包括每条扫描的行数，IO消耗等等。
6. 如果 SQL 语句存在 LIMIT 或者 OFFSET 操作，则判断是否超出范围，如果超出范围，则提示需要改进。

### 3.1 Explain命令

Explain 命令通常可在 MySQL 命令窗口输入，也可以嵌入到 SQL 脚本中。语法如下：

```sql
EXPLAIN [EXTENDED] sql_statement;
```

- `extended` 参数用于打印更详细的执行计划，默认为不开启。

- `sql_statement` 为任意有效的 SQL 语句。

使用 EXPLAIN 时一般会结合具体的 SQL 查询语句一起使用，以便得到更有意义的执行计划。例如：

```sql
EXPLAIN SELECT * FROM mytable WHERE a=1 AND b=2 ORDER BY c DESC LIMIT 10;
```

除了直接分析 SELECT SQL 语句之外，还可以使用 EXPLAIN 的相关参数，实现诸如索引监视、慢日志记录、主从延迟检查等功能。

### 3.2 Explain参数

除了使用 EXPLAIN 命令分析 SQL 语句之外，还可以在配置文件 my.cnf 中设置相关参数，以控制 EXPLAIN 的行为。以下是一些常用的配置选项：

1. `slow_query_log`：启用慢日志功能，默认值为 OFF 。当设置为 ON 时，所有超过指定时间阈值的慢查询都被记录到慢日志文件中。

2. `long_query_time`：指定慢查询时间阈值，单位为秒，默认值为 10。

3. `profiling`：启用性能分析功能，默认值为 OFF 。当设置为 ON 时，对于 SQL 查询语句的执行时间超过 slow_query_log 指定阈值，系统会自动记录执行计划。

4. `performance_schema`：打开性能 Schema，默认值为 OFF 。

### 3.3 使用示例

#### 3.3.1 查看所有表的执行计划

```sql
-- 查看所有表的执行计划
EXPLAIN SELECT * FROM t1;
```

#### 3.3.2 查看某个表的执行计划

```sql
-- 查看 t1 表的执行计划
EXPLAIN SELECT * FROM t1;
```

#### 3.3.3 查看复杂查询的执行计划

```sql
-- 查看复杂查询的执行计划
EXPLAIN SELECT DISTINCT product_name, SUM(unit_price*quantity) AS total_cost 
FROM orders JOIN order_details USING (order_id) JOIN products USING (product_id) 
WHERE order_date BETWEEN '2018-01-01' AND '2018-12-31' GROUP BY product_name;
```

#### 3.3.4 查看分区表的执行计划

```sql
-- 创建测试表
CREATE TABLE test_partition (
  col1 INT NOT NULL AUTO_INCREMENT, 
  col2 VARCHAR(50), 
  PRIMARY KEY (col1)
) ENGINE=MyISAM PARTITION BY RANGE COLUMNS(col2) (PARTITION p0 VALUES LESS THAN ('2019-01-01'), PARTITION p1 VALUES LESS THAN ('2020-01-01'));

-- 查看分区表的执行计划
EXPLAIN SELECT * FROM test_partition;
```

#### 3.3.5 用 Explain 测试数据查询

```sql
-- 插入1000万数据
INSERT INTO employee (emp_no, birth_date, first_name, last_name, gender, hire_date) 
VALUES 
    (1,'1990-01-01','John', 'Doe', 'M', '2018-01-01'),
   ...,
    (1000000,'1990-01-01','John', 'Smith', 'M', '2018-01-01');
    
-- 先执行一下查询，生成执行计划
EXPLAIN SELECT emp_no, birth_date, first_name, last_name, gender, hire_date from employee limit 10; 

-- 执行 explain extended 指令，显示执行计划的详细信息
EXPLAIN EXTENDED SELECT emp_no, birth_date, first_name, last_name, gender, hire_date from employee limit 10;  
```

## 4.未来发展方向

explain的功能远不止于此，它提供了许多实用化的特性，如分析索引选择、查询规划、执行计划监控等。随着数据库的不断演进和发展，explain也正在不断完善，希望能够给大家带来更多有益的启发。