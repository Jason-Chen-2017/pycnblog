                 

# 1.背景介绍


## 为什么需要聚合函数？
在任何数据处理任务中，数据的统计、分析和处理都是不可或缺的一环。数据分析人员通过对数据进行统计、计算、分析等操作，才能更好地理解业务，从而做出明智的决策。SQL语言提供了丰富的数据统计和分析功能，包括聚合函数、分组函数、排序函数等。本文将带领读者学习SQL语言的基本知识以及如何利用这些功能进行数据的统计、分析、处理。
## 什么是聚合函数？
聚合函数（Aggregate Function）就是把多条记录映射到一个值上的函数。比如求总和、平均数、最大值、最小值、求和等。如果要处理的是表中的所有行记录，就可以使用聚合函数。常用的聚合函数包括COUNT()、SUM()、AVG()、MAX()、MIN()等。例如：SELECT SUM(salary) FROM employees; 查询employees表的所有员工薪水的总和。
## 分组查询
分组查询（Group Query）是指根据特定的条件，将多条记录按照一定规则进行分类，然后应用聚合函数进行汇总计算，得到每个分类下的汇总值。在分组查询中，我们使用GROUP BY子句指定要按照哪个字段进行分组，GROUP BY后面的语句表示分组的依据；HAVING子句用于筛选出符合特定条件的分组。例如：SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department HAVING avg_salary > 50000; 查询employees表中各部门的平均薪水，并且只显示平均薪水大于50000的部门信息。
## SQL语言简介
SQL，Structured Query Language，结构化查询语言，是关系型数据库管理系统（RDBMS）的标准语言。它用于存取、更新和管理关系数据库管理系统（RDBMS）中的数据。SQL语言包括DDL（Data Definition Language，数据定义语言）、DML（Data Manipulation Language，数据操纵语言）、DCL（Data Control Language，数据控制语言）三大部分。其中DDL用于定义数据库对象，如创建表、视图、索引等；DML用于操作表中的数据，如插入、删除、修改等；DCL用于控制数据库的访问权限，如用户授权等。
# 2.核心概念与联系
## COUNT()函数
COUNT()函数可以统计满足WHERE子句条件的记录的个数，或者统计表中的记录数量。语法如下所示：
```sql
SELECT COUNT(*) FROM table_name;        -- 统计表中记录的个数
SELECT COUNT(column_name) FROM table_name;   -- 统计指定列的非空记录个数
```
## SUM()函数
SUM()函数可以对指定列的值求和，其语法如下所示：
```sql
SELECT SUM(column_name) FROM table_name;    -- 对指定列的值求和
```
## AVG()函数
AVG()函数可以计算指定列值的平均值，其语法如下所示：
```sql
SELECT AVG(column_name) FROM table_name;    -- 计算指定列值的平均值
```
## MAX()函数
MAX()函数可以计算指定列的最大值，其语法如下所示：
```sql
SELECT MAX(column_name) FROM table_name;    -- 计算指定列的最大值
```
## MIN()函数
MIN()函数可以计算指定列的最小值，其语法如下所示：
```sql
SELECT MIN(column_name) FROM table_name;    -- 计算指定列的最小值
```
## GROUP BY和HAVING子句
GROUP BY子句用来对结果集进行分组，指定按哪些字段来进行分组。HAVING子句用来过滤分组后的结果集，只有满足条件的分组才会被输出。一般情况下，GROUP BY和HAVING都可以一起使用。
```sql
SELECT column1, aggregate_function(column2),...
    FROM table_name 
    WHERE condition 
    GROUP BY group_by_column 
    HAVING having_condition;
```
## DISTINCT关键字
DISTINCT关键字用来去除重复行，其语法如下所示：
```sql
SELECT DISTINCT column_name1, column_name2 FROM table_name;      -- 仅返回指定列的唯一值
```