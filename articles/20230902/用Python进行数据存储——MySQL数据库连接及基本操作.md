
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Python作为一个高级编程语言，早已成为大众日常开发中不可或缺的一部分。相比其他编程语言来说，Python在数据处理方面可以说是一个“无所不能”的工具。借助Python的强大功能，不仅可以轻松地进行数据分析、机器学习等各类计算任务，还可以通过编写简单的脚本实现数据的各种读写操作。因此，越来越多的人开始利用Python进行数据存储。其中，MySQL是最具代表性的关系型数据库管理系统，它已经成为当今最流行的开源数据库。那么，如何通过Python连接并对MySQL进行基本的增删改查操作呢？本文将给出相应的教程，希望能够帮助到大家。

2.安装MySQL服务器和Python数据库接口模块mysql-connector-python
首先需要安装MySQL服务器，推荐使用较新版本的MySQL（如5.7）服务器。如果没有安装过 MySQL 服务器，可参考我之前的文章《使用 Docker 安装 MySQL 5.7 服务器》进行安装。然后，需要安装 Python 数据库接口模块 mysql-connector-python。

mysql-connector-python 是用于访问 MySQL 服务器的 Python 模块，可根据 Python 的不同版本和平台进行安装。你可以到它的官方网站 https://dev.mysql.com/downloads/connector/python/ 下载最新版本的安装包，根据你的操作系统选择对应的安装文件进行安装即可。

3.连接MySQL数据库并创建表格
完成以上两步之后，就可以用 Python 连接至 MySQL 服务器并创建表格了。这里，我们将以一个简单的例子来演示如何连接并插入、查询和删除数据。

首先，引入 MySQLdb 和 mysql-connector-python 两个模块：
``` python
import mysql.connector as mariadb
from mysql.connector import errorcode
```
接着，创建一个 MariaDB 的连接对象，并指定连接信息，包括数据库名称、用户名和密码等。此处我们假设数据库名为 test，用户名为 root，密码为 password：
``` python
cnx = mariadb.connect(user='root', password='password',
                      host='localhost', database='test')
cursor = cnx.cursor() # 创建游标
```
然后，执行创建表格的 SQL 语句：
``` sql
CREATE TABLE mytable (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255),
  age INT,
  address CHAR(50)
);
```
接下来，我们可以向该表格插入一些数据：
``` python
sql = "INSERT INTO mytable (name, age, address) VALUES (%s, %s, %s)"
val = ("John", 30, "USA")
try:
    cursor.execute(sql, val)
    cnx.commit() # 提交事务
except mariadb.Error as err:
    print("An error occurred: {}".format(err))
finally:
    cursor.close()
    cnx.close()
```
最后，我们可以用 SELECT 语句从表格中读取数据：
``` python
sql = "SELECT * FROM mytable"
try:
    cursor.execute(sql)
    for row in cursor:
        print(row)
except mariadb.Error as err:
    print("An error occurred: {}".format(err))
finally:
    cursor.close()
    cnx.close()
```
当然，也可以用 DELETE 或 UPDATE 来修改或删除数据。另外，这里只是举了一个最简单的数据操作例子，实际上还有许多更复杂的用法。比如，你可以批量插入数据，设置索引等。为了便于理解，这里就不详细展示了。

4.扩展阅读
MySQL是一个功能丰富的关系型数据库管理系统，这里只涉及其最基础的操作，后续还会继续更新其它相关内容。关于 MySQL 更加深入的学习，建议参考以下资源：