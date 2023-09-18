
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在MySQL中，当用户创建数据库或者表时，可以指定字符集和排序规则，如果不指定，则默认使用数据库服务器的默认字符集和排序规则，通常情况下，我们使用的都是数据库服务器默认字符集和排序规则。但是有些时候，比如需要兼容一些旧的系统或旧的数据，或者为了满足特殊业务需求，我们可能需要修改字符集和排序规则，例如，修改数据库默认的字符集和排序规则。MySQL的全局变量character_set_server和collation_connection就用于设置这种情况下的字符集和排序规则。本文将详细描述character_set_server和collation_connection这两个全局变量的作用、配置方法和注意事项等。

# 2.基本概念术语说明
## 2.1 character_set_server参数
character_set_server参数用来定义MySQL服务器默认的字符集。它的值是一个字符串，表示了这个服务器的默认字符集。可以通过以下命令查看当前服务器的默认字符集：
```mysql> SHOW VARIABLES LIKE 'character\_set\_server';```
执行结果类似如下所示：
```+---------------+----------------------------+
| Variable_name | Value                      |
+---------------+----------------------------+
| character_set_server | utf8                       |
+---------------+----------------------------+
```
此处，默认的字符集就是utf8。

## 2.2 collation_connection参数
collation_connection参数用来定义MySQL服务器连接的默认排序规则。它的值是一个字符串，表示了这个服务器连接的默认排序规则。可以通过以下命令查看当前服务器连接的默认排序规则：
```mysql> SHOW VARIABLES LIKE 'collation\_connection';```
执行结果类似如下所示：
```+--------------+-------------------+
| Variable_name | Value             |
+--------------+-------------------+
| collation_connection | utf8_general_ci   |
+--------------+-------------------+
```
此处，默认的排序规则就是utf8_general_ci。

## 2.3 设置character_set_server和collation_connection
前面提到，当用户创建数据库或者表时，可以指定字符集和排序规则，如果不指定，则默认使用数据库服务器的默认字符集和排序规则。实际上，如果创建数据库或者表时都不指定字符集和排序规则，那么数据库默认的字符集和排序规则就会生效。但是，如果创建数据库或者表时，显式地指定了字符集和排序规则，那么就会覆盖掉数据库默认的字符集和排序规则。

因此，如果要更改服务器的默认字符集和排序规则，则可以使用character_set_server和collation_connection两个全局变量。character_set_server和collation_connection这两个参数可以在my.ini文件中或者用SET语句来设置，或者通过SHOW GLOBAL VARIABLES命令来查询。以下分别介绍三种方式来设置这两个参数。

### 在my.ini配置文件中设置character_set_server和collation_connection
一般来说，MySQL的配置文件是my.ini，位于安装目录下的conf文件夹中。打开该配置文件，找到[mysqld]段落，添加如下两行配置信息：
```
character-set-server=gbk
collation-connection=gbk_chinese_ci
```
其中，character-set-server和collation-connection参数后面的值可以根据自己的实际情况进行修改，也可以保留服务器默认字符集和排序规则。保存并关闭配置文件，重启MySQL服务器，即可完成设置。

### 通过SET语句设置character_set_server和collation_connection
设置方式如下：
```mysql> SET NAMES 'gbk' COLLATE 'gbk_chinese_ci';```
执行成功后，会话的默认字符集和排序规则变成了指定的gbk字符集和gbk_chinese_ci排序规则。

### 查询和设置全局变量
另外一种设置方式是通过SHOW GLOBAL VARIABLES 和SET GLOBAL语句来设置global变量。如要查询global变量的character_set_server和collation_connection的值，可运行以下命令：
```mysql> SHOW GLOBAL VARIABLES WHERE variable_name IN ('character_set_server','collation_connection');```
执行结果类似如下所示：
```
+--------------------------+----------------------------+
| Variable_name            | Value                      |
+--------------------------+----------------------------+
| character_set_server     | gbk                        |
| collation_connection     | gbk_chinese_ci             |
+--------------------------+----------------------------+
```
如果要修改global变量的character_set_server和collation_connection的值，可运行以下命令：
```mysql> SET GLOBAL character_set_server = 'gbk', COLLATION_CONNECTION = 'gbk_chinese_ci';```
执行成功后，会话的默认字符集和排序规则变成了指定的gbk字符集和gbk_chinese_ci排序规则。