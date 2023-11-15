                 

# 1.背景介绍


MySQL是目前最流行的开源关系型数据库管理系统之一。无论是在开发、测试还是部署应用时，都需要用到MySQL作为关系型数据库。本文将教你如何快速上手并掌握一些基础的SQL命令，使得你可以轻松地进行CRUD（创建、读取、更新、删除）操作，以及一些高级功能的实现。
# 2.核心概念与联系
## SQL语言概述
SQL（Structured Query Language，结构化查询语言），是一种通用的语言，用于管理关系数据库系统，由美国计算机科学家<NAME>于1986年提出。它包括数据定义语言DDL（Data Definition Language）、数据操纵语言DML（Data Manipulation Language）、事务控制语言TCL（Transaction Control Language）。
## SQL语句分类
SQL语句分为数据定义语言（Data Definition Language，DDL）、数据操纵语言（Data Manipulation Language，DML）、事务控制语言（Transaction Control Language，TCL）三种类型。
- 数据定义语言包括CREATE、ALTER、DROP等语句；
- 数据操纵语言包括INSERT、UPDATE、DELETE、SELECT等语句；
- 事务控制语言包括COMMIT、ROLLBACK、SAVEPOINT、SET TRANSACTION等语句。
## SQL常用指令
### SELECT 语句
- `SELECT` 语句用于从表中检索数据。
- 可以指定选择哪些列，可以使用表达式、聚集函数和计算字段。
- 可以对结果进行排序，过滤，限制数量等操作。
```sql
-- 从表 employees 中选取 id, name, salary 列，并按 salary 升序排列，输出前 5 条记录
SELECT id, name, salary
FROM employees
ORDER BY salary ASC
LIMIT 5;
```
### INSERT 语句
- `INSERT INTO` 语句用于向表中插入新的行。
- 可以一次插入多行记录。
```sql
-- 将一条新记录插入到 employees 表中
INSERT INTO employees (id, name, salary) VALUES (1001, 'John Doe', 5000);
```
### UPDATE 语句
- `UPDATE` 语句用于修改表中的现有记录。
- 可以指定修改哪些列的值，并设置条件。
```sql
-- 更新 employees 表中的 id 为 1001 的记录的 salary 值为 5500
UPDATE employees SET salary = 5500 WHERE id = 1001;
```
### DELETE 语句
- `DELETE FROM` 语句用于从表中删除记录。
- 可以指定删除哪些记录，并设置条件。
```sql
-- 删除 employees 表中的所有记录
DELETE FROM employees;
```
### CREATE TABLE 语句
- `CREATE TABLE` 语句用于创建新表。
- 需要提供表名及各列信息。
```sql
-- 创建一个名为 people 的表，包含姓名和年龄两个字段
CREATE TABLE IF NOT EXISTS people (
  name VARCHAR(50),
  age INT
);
```
### ALTER TABLE 语句
- `ALTER TABLE` 语句用于修改已存在的表。
- 可用于添加或删除表中的列，或改变数据类型或约束。
```sql
-- 在 people 表中添加一个手机号码字段
ALTER TABLE people ADD COLUMN phone_number VARCHAR(20);
```
### DROP TABLE 语句
- `DROP TABLE` 语句用于删除表。
- 慎用！
```sql
-- 删除 people 表
DROP TABLE people;
```
### CREATE INDEX 语句
- `CREATE INDEX` 语句用于在表或视图中创建索引。
- 通过索引可加快搜索速度。
```sql
-- 创建一个名为 employee_salary 的索引
CREATE INDEX employee_salary ON employees (salary DESC);
```
### DROP INDEX 语句
- `DROP INDEX` 语句用于删除索引。
- 如果不再需要索引，应当删除。
```sql
-- 删除 employee_salary 索引
DROP INDEX employee_salary;
```