                 

# 1.背景介绍

在数据库系统中，表是数据的组织和存储的基本单位。MySQL是一种流行的关系型数据库管理系统，它使用表来存储和组织数据。在本教程中，我们将深入探讨如何创建和修改MySQL表。

## 1.1 MySQL简介
MySQL是一个开源的关系型数据库管理系统，由瑞典MySQL AB公司开发。它广泛应用于Web应用程序、企业应用程序和数据挖掘等领域。MySQL支持多种数据类型，如整数、浮点数、字符串、日期和时间等。它还支持事务、索引和存储过程等特性。

## 1.2 表的概念
在MySQL中，表是数据的组织和存储的基本单位。表由一组列组成，每个列表示一个数据的属性。表由一组行组成，每行表示一个数据的实例。表的结构由一个名为表定义（DDL）的数据结构描述。表定义包括表名、列名、数据类型、约束等信息。

## 1.3 表的创建
在MySQL中，可以使用CREATE TABLE语句创建表。CREATE TABLE语句的基本格式如下：

```sql
CREATE TABLE table_name (
    column1 data_type,
    column2 data_type,
    ...
    columnN data_type
);
```

在这个语句中，table_name是表的名称，column1、column2、...、columnN是表的列名，data_type是列的数据类型。

例如，创建一个名为employee的表，其中包含名字、年龄和薪资三个列，可以使用以下语句：

```sql
CREATE TABLE employee (
    name VARCHAR(255),
    age INT,
    salary DECIMAL(10,2)
);
```

在这个例子中，name列的数据类型为VARCHAR，表示字符串；age列的数据类型为INT，表示整数；salary列的数据类型为DECIMAL，表示小数。

## 1.4 表的修改
在MySQL中，可以使用ALTER TABLE语句修改表。ALTER TABLE语句的基本格式如下：

```sql
ALTER TABLE table_name
    ADD COLUMN column_name data_type,
    DROP COLUMN column_name,
    MODIFY COLUMN column_name data_type,
    CHANGE COLUMN old_column_name new_column_name data_type;
```

在这个语句中，table_name是表的名称，column_name是表的列名，data_type是列的数据类型。

例如，修改employee表，添加一个新的列job_title，表示职位，可以使用以下语句：

```sql
ALTER TABLE employee
    ADD COLUMN job_title VARCHAR(255);
```

在这个例子中，job_title列的数据类型为VARCHAR，表示字符串。

## 1.5 总结
在本教程中，我们介绍了MySQL中表的创建和修改的基本概念和操作方法。我们学习了如何使用CREATE TABLE语句创建表，以及如何使用ALTER TABLE语句修改表。在后续的教程中，我们将深入探讨MySQL的查询和管理操作。