
作者：禅与计算机程序设计艺术                    

# 1.简介
  

# 2.Hive 的特性
## 2.1 SQL on Hadoop
Hive 提供了一种类似于关系型数据库的 SQL 查询方式，这种方式也被称作 SQL on Hadoop。它支持类似于创建表、插入数据等基础操作，而且在执行查询时，会自动优化查询计划，并充分利用集群资源加速查询速度。另外，Hive 通过自动生成代码的方式来优化查询效率，并能够把相同逻辑的查询任务交给 Hadoop 执行，无需编写复杂的 MapReduce 程序。而这些能力都使得 Hive 可以很好地解决大数据量下的分析问题。

## 2.2 高性能
Hive 支持许多内置函数，包括字符串函数、日期函数、聚合函数、统计函数等，能够让用户灵活地进行数据分析。由于 Hive 会根据查询条件生成不同的查询计划，因此在执行查询时，可以充分利用集群资源加速查询速度。另外，Hive 使用 MapReduce 来并行计算，并提供了压缩、分区、索引、自联结等优化措施，能够极大地提升查询速度。

## 2.3 数据倾斜
Hive 对数据的存储是采用分区（Partition）机制实现的。每一个表可以指定多个列作为 Partition Key ，用于划分数据集，确保同一个 Partition 中的数据不会放在不同节点上，从而解决数据倾斜的问题。另外，Hive 也提供了一个稀疏索引功能，能够快速定位包含某个值的行。

## 2.4 动态并行
Hive 可以在运行期间自动调整查询计划，并同时使用集群的多个节点。这种能力可以极大的提升查询性能。例如，Hive 可以根据当前的数据负载情况调整查询计划，同时利用多个节点共同处理查询请求。

## 2.5 ACID 事务
Hive 支持 ACID 事务，能够确保数据安全性。ACID 是指 Atomicity（原子性）、Consistency（一致性）、Isolation（隔离性）、Durability（持久性）。事务的四个属性分别保证事务的原子性、一致性、隔离性、持久性。在数据更新或删除前，Hive 会对相关表进行锁定，防止数据损坏。事务的提交和回滚也支持，从而保证数据的完整性。

## 2.6 BI 工具集成
Hive 可以轻松集成 BI 工具，包括 Apache Zeppelin、Tableau、Microsoft Excel等。这样就可以实现对 Hive 数据的直观可视化，从而更直观地理解数据。另外，Hive 还支持 Apache Spark，可以实现 HiveQL 代码的快速转换，并利用 Spark 引擎进行快速数据分析。

# 3.Hive SQL语法
## 3.1 DDL 操作
DDL(Data Definition Languages)，数据定义语言，用来定义数据库对象，如数据库，表，视图，触发器，存储过程等。Hive 支持以下的 DDL 操作：

1. CREATE DATABASE: 创建数据库，语法如下：
```sql
CREATE DATABASE database_name;
```
注意事项：Hive 不支持跨数据库事务操作。

2. CREATE TABLE：创建表，语法如下：
```sql
CREATE EXTERNAL TABLE table_name (
  column_name data_type,
 ...
) PARTITIONED BY (
  partition_column data_type,
 ...
);
```
示例：
```sql
CREATE EXTERNAL TABLE mytable (
  empid int,
  name string,
  salary double,
  dept string
) PARTITIONED BY (
  year int,
  month int
);
```
说明：
- `EXTERNAL`关键字：表示表为外部表。外部表是 Hive 在非 HDFS 文件系统中存储的表，Hive 只保存表结构信息，不实际保存数据。外部表可以使用 INSERT INTO 命令将数据导入表中。
- `PARTITIONED BY`关键字：定义表的分区。分区是将数据按照一定的规则分割成小块的过程，在查询时只扫描部分分区即可提高查询速度。这里定义的分区由年份和月份组成。

3. ALTER TABLE：修改表结构，语法如下：
```sql
ALTER TABLE table_name [SET SERDEPROPERTIES serde_properties]
    [[ADD|DROP] COLUMNS column_definition] 
    [COMMENT comment];
    
column_definition:
  column_name data_type [comment]
  
serde_properties:
  "property_name"="property_value",...
```
示例：
```sql
ALTER TABLE mytable SET SERDEPROPERTIES ("serialization.encoding" = "UTF-8");

ALTER TABLE mytable ADD COLUMNS (age int COMMENT 'age of employee');

ALTER TABLE mytable DROP COLUMN age;
```
说明：
- `SET SERDEPROPERTIES`：设置 SerDe(Serializer/Deserializer) 属性，用于序列化和反序列化数据。SerDe 即序列化器/反序列化器，负责将数据按特定格式编码，并将编码后的字节流读出或者写入文件系统中的某些位置。
- `ADD COLUMNS`/`DROP COLUMN`：添加或删除表字段。添加新字段后，Hive 将尝试根据现有的分区信息重新分配分区，以减少数据的移动。

4. DROP TABLE：删除表，语法如下：
```sql
DROP TABLE [IF EXISTS] table_name;
```
示例：
```sql
DROP TABLE IF EXISTS mytable;
```
说明：
- `IF EXISTS`：若表不存在，则忽略错误。

## 3.2 DML 操作
DML(Data Manipulation Languages)，数据操纵语言，用来操作数据库中的数据，包括增删改查。Hive 支持以下的 DML 操作：

1. SELECT：查询表数据，语法如下：
```sql
SELECT select_expr,...
FROM table_reference
[WHERE where_condition]
[GROUP BY grouping_columns]
[ORDER BY ordering_columns]
```
示例：
```sql
SELECT * FROM mytable WHERE name='Alice';

SELECT empid, sum(salary) AS total_salary 
FROM mytable GROUP BY empid;
```
- `select_expr`: 指定要查询的字段列表。
- `table_reference`: 指定要查询的表。
- `where_condition`: 过滤查询结果的条件。
- `grouping_columns`: 分组字段，用于对结果进行聚合。
- `ordering_columns`: 排序字段，用于对结果进行排序。

2. INSERT：插入数据，语法如下：
```sql
INSERT INTO table_name [(column_list)] 
  VALUES (expression_list);
```
示例：
```sql
INSERT INTO mytable (empid, name, salary, dept, year, month) VALUES (1,'Bob',75000,'IT', 2018, 01);
```
说明：
- `[(column_list)]`：指定要插入的字段列表。如果不指定，默认所有字段都插入。
- `(expression_list)`：指定每个字段的值。

3. DELETE：删除数据，语法如下：
```sql
DELETE FROM table_name [WHERE where_condition];
```
示例：
```sql
DELETE FROM mytable WHERE empid=1;
```
说明：
- `[WHERE where_condition]`：指定删除条件。

## 3.3 变量替换
Hive 支持动态替换，允许用户在运行时设置变量值，支持变量引用。用户可以在配置文件中配置变量，也可以在运行时设置变量值。如 ${myvar} 表示引用名为 myvar 的变量的值。下面给出几个典型场景的例子。
### 替换目录路径
假设有一个目录 /data/orders/2018/10，想要把所有数据导入到该目录下，可以通过下面的语句完成：
```sql
LOAD DATA INPATH '/user/data/orders' OVERWRITE INTO TABLE mytable PARTITION (year=2018,month=10);
```
这个语句会把 `/user/data/orders` 替换为 `/data/orders`，进而将数据导入到 `/data/orders/2018/10` 下。
### 替换文本字符串
假设有一个字符串 s='Hello World!',想要把所有出现 'World' 的地方替换为 'Mars'，可以通过下面的语句完成：
```sql
UPDATE mytable SET name = replace(name, 'World','Mars') WHERE name LIKE '%World%';
```
这个语句会在所有姓名中包含 'World' 的地方用 'Mars' 替换掉。