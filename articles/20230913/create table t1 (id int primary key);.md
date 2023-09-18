
作者：禅与计算机程序设计艺术                    

# 1.简介
  

“create table”命令是一个非常基础但是经常被忽略的MySQL命令。它的作用就是在指定数据库中创建一个表格。本文将通过实例的方式对这个命令进行深入剖析，全面理解其功能、特点及应用场景。

# 2.核心概念
## 2.1 MySQL数据库
MySQL是一个开源关系型数据库管理系统（RDBMS），由瑞典MySQL AB公司开发并发布。MySQL是目前世界上最流行的数据库管理系统之一，广泛用于各种Web应用，例如WordPress、Joomla等。

2008年9月1日，MySQL AB公司宣布推出第一个企业版产品MySQL Community Edition，它是免费的、开放源代码的软件，而且提供了完全免费的分支版本——MariaDB。由于MariaDB与MySQL语法兼容，因此很多开发人员选择MariaDB作为主力数据库。

## 2.2 数据表
数据表是关系型数据库中的重要组成单位，它存储着实体和其属性的信息。每个数据库都至少包含一个数据表，可以根据需要创建更多的数据表。数据表通常包括以下几种结构：

1. 列：每张表都包含多列，每列都对应于一个特定的信息，如名字、地址、联系方式等。

2. 行：每张表都包含多行数据，每行对应于一个实体，比如学生、老师、部门等。

3. 主键：主键是一个特殊的列或组合列，主要用来标识表中的每行数据，确保数据的唯一性和完整性。

4. 外键：外键是一种约束，它保证两个表之间的参照完整性。

## 2.3 SQL语言
SQL（Structured Query Language）结构化查询语言，它是用于管理关系数据库的语言。SQL包括SELECT、INSERT、UPDATE、DELETE、CREATE TABLE等命令，并且可以通过它们实现对数据库的各种操作。

# 3.原理及操作步骤
## 3.1 创建表
要创建一个空表，可以使用如下SQL语句：

```sql
CREATE TABLE table_name (column1 datatype constraint, column2 datatype constraint,... );
```

其中，`table_name`是表名，可自定义；`datatype`是数据类型，如int、varchar、datetime、text等；`constraint`是约束条件，如not null、primary key、unique、default等。

举例：

```sql
CREATE TABLE myTable (
  id INT(11) PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(50),
  age INT(3),
  email VARCHAR(100),
  date_of_birth DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

上述例子创建了一个名为myTable的表格，包含四列：id、name、age、email。其中，id为主键，自动增长，date_of_birth列的值会随着时间自动更新；name、age、email均为非空字段，且邮箱格式符合规范。

## 3.2 插入数据
插入数据到表中，可以使用如下SQL语句：

```sql
INSERT INTO table_name (column1, column2,...) VALUES (value1, value2,...);
```

其中，`table_name`是表名；`(column1, column2,...)`表示待插入的列名，支持逗号分隔多个列；`(value1, value2,...)`表示待插入的各个值，也支持逗号分隔多个值。

举例：

```sql
INSERT INTO myTable (name, age, email) VALUES ('John', '25', 'john@example.com');
```

上述例子向myTable表格插入一条记录，姓名为John，年龄为25，邮箱为john@example.com。如果id列不为空，则插入成功后该列将自动递增。

## 3.3 查询数据
查询数据，可以使用如下SQL语句：

```sql
SELECT * FROM table_name WHERE condition;
```

其中，`* `代表选择所有列；`table_name`是表名；`condition`是筛选条件，可以是比较运算符、逻辑运算符、函数等。

举例：

```sql
SELECT * FROM myTable WHERE age > 25 AND email LIKE '%gmail%';
```

上述例子从myTable表格中查询年龄大于25岁且邮箱包含"gmail"的记录。

## 3.4 更新数据
更新数据，可以使用如下SQL语句：

```sql
UPDATE table_name SET column1 = new-value1, column2 = new-value2,... WHERE condition;
```

其中，`table_name`是表名；`column1`、`column2`、...是待更新的列名；`new-value1`、`new-value2`、...是新的值；`condition`是筛选条件。

举例：

```sql
UPDATE myTable SET age = 30 WHERE name = 'Mary';
```

上述例子修改myTable表格中姓名为Mary的年龄为30。

## 3.5 删除数据
删除数据，可以使用如下SQL语句：

```sql
DELETE FROM table_name WHERE condition;
```

其中，`table_name`是表名；`condition`是筛选条件。

举例：

```sql
DELETE FROM myTable WHERE name = 'Jack';
```

上述例子从myTable表格中删除姓名为Jack的记录。

# 4.代码示例及解释
下面展示一些实际代码示例，说明如何使用SQL语句创建、插入、查询、更新、删除数据表。

## 4.1 创建表
假设有一个数据库表格需求如下：

| Column Name | Data Type | Description                     | Constraints                      |
|-------------|-----------|---------------------------------|----------------------------------|
| id          | integer   | Unique identifier for the record | primary key autoincrement        |
| name        | varchar   | First and last name of the user | not null                         |
| age         | integer   | Age of the user                 | default 0                        |
| gender      | char      | Gender of the user              | values ('M','F')                 |
| email       | varchar   | Email address of the user       | unique                           |
| phone       | varchar   | Phone number of the user        |                                  |
| city        | varchar   | City where the user lives in    |                                  |

创建该表格的SQL语句如下：

```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name VARCHAR NOT NULL,
  age INTEGER DEFAULT 0,
  gender CHAR CHECK (gender IN('M','F')),
  email VARCHAR UNIQUE,
  phone VARCHAR,
  city VARCHAR
);
```

## 4.2 插入数据
假设需要插入以下用户数据：

| id | name     | age | gender | email             | phone           | city     |
|----|----------|-----|--------|-------------------|-----------------|----------|
| 1  | John Doe | 25  | M      | johndoe@gmail.com | 555-1234567890 | New York |
| 2  | Jane Smith| 30  | F      | janesmith@yahoo.co.uk | 555-0987654321 | Los Angeles |
| 3  | David Lee | 40  | M      | davidlee@hotmail.com |                | Chicago |

插入这些用户数据的SQL语句如下：

```sql
INSERT INTO users (id, name, age, gender, email, phone, city)
VALUES 
  (1, "John Doe", 25, 'M', 'johndoe@gmail.com', '555-1234567890', 'New York'),
  (2, "Jane Smith", 30, 'F', 'janesmith@yahoo.co.uk', '555-0987654321', 'Los Angeles'),
  (3, "David Lee", 40, 'M', 'davidlee@hotmail.com', '', 'Chicago');
```

注意：第二条记录没有给phone列赋值，此时phone列的值默认为NULL。第三条记录的city列的值为空字符串。

## 4.3 查询数据
假设需要查询名字为"John Doe"或者"Jane Smith"的用户信息。查询语句如下：

```sql
SELECT * FROM users 
WHERE name IN ('John Doe', 'Jane Smith');
```

得到结果如下：

| id | name     | age | gender | email               | phone           | city       |
|----|----------|-----|--------|---------------------|-----------------|------------|
| 1  | John Doe | 25  | M      | johndoe@gmail.com   | 555-1234567890 | New York   |
| 2  | Jane Smith| 30  | F      | janesmith@yahoo.co.uk | 555-0987654321 | Los Angeles |

## 4.4 更新数据
假设需要把名字为"John Doe"的用户的年龄设置为30。更新语句如下：

```sql
UPDATE users 
SET age=30 
WHERE name='John Doe';
```

## 4.5 删除数据
假设需要把所有年龄大于等于30岁的用户信息都删除掉。删除语句如下：

```sql
DELETE FROM users 
WHERE age>=30;
```