
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


在现代计算机应用中，对于大型数据存储和处理而言，关系数据库管理系统（RDBMS）是最主流的解决方案。本文将从以下三个方面展开：

1. JDBC简介及数据库连接的配置
2. SQL语句执行的基本语法规则
3. 数据查询的基本方法和操作流程

# 2.核心概念与联系
## 2.1 JDBC简介
JDBC(Java Database Connectivity) 是一套用来在 Java 中访问数据库的 API。通过 JDBC，应用程序可以跟数据库建立连接、执行SQL语句、检索结果集并对数据库进行更新等。它的主要作用是屏蔽不同数据库厂商的差异性，使得 Java 开发者只需使用统一的接口就可以操作不同的数据库产品。目前已有的 JDBC 驱动包括：MySQL Connector/J、PostgreSQL JDBC Driver、SQL Server JDBC Driver 和 Oracle JDBC Driver。

## 2.2 配置数据库连接
首先需要定义好数据库的 URL、用户名和密码信息，例如：
```java
String url = "jdbc:mysql://localhost:3306/testdb";   //数据库地址
String user = "root";                                //用户名
String password = "<PASSWORD>";                             //密码
```
然后调用`Class.forName()`方法加载数据库驱动类：
```java
Class.forName("com.mysql.cj.jdbc.Driver");          //加载MySQL驱动
//Class.forName("org.postgresql.Driver");           //加载PostgreSQL驱动
//Class.forName("com.microsoft.sqlserver.jdbc.SQLServerDriver");   //加载SQLServer驱动
//Class.forName("oracle.jdbc.driver.OracleDriver");      //加载Oracle驱动
Connection conn = DriverManager.getConnection(url,user,password);     //创建数据库连接
```

## 2.3 执行SQL语句
### 2.3.1 创建表
创建一个名为employee的表，包含id、name、age、email字段：
```java
Statement stmt = conn.createStatement();    //创建Statement对象
stmt.executeUpdate("CREATE TABLE employee (id INT PRIMARY KEY AUTO_INCREMENT," +
                   " name VARCHAR(50), age INT, email VARCHAR(50))");        //执行SQL语句
stmt.close();                               //关闭Statement对象
conn.close();                               //关闭数据库连接
```
### 2.3.2 插入数据
向employee表插入一些数据：
```java
PreparedStatement pstmt = conn.prepareStatement("INSERT INTO employee (name, age, email)" +
                                                 " VALUES (?,?,?)");     //准备执行插入数据的PreparedStatement对象
pstmt.setString(1, "Tom");                                  //设置第一个参数的值（name字段）
pstmt.setInt(2, 30);                                       //设置第二个参数的值（age字段）
pstmt.setString(3, "tom@example.com");                      //设置第三个参数的值（email字段）
pstmt.executeUpdate();                                     //执行插入操作
pstmt.close();                                             //关闭PreparedStatement对象
conn.close();                                               //关闭数据库连接
```
### 2.3.3 查询数据
查询id值为1的数据：
```java
ResultSet rs = stmt.executeQuery("SELECT * FROM employee WHERE id=1");       //执行查询操作，返回ResultSet对象
while(rs.next()){                                                            //循环遍历ResultSet
  System.out.println("Name:" + rs.getString("name"));                          //打印出结果集中的每条记录中的name值
  System.out.println("Age:" + rs.getInt("age"));                              //打印出结果集中的每条记录中的age值
  System.out.println("Email:" + rs.getString("email"));                        //打印出结果集中的每条记录中的email值
}
rs.close();                                                                  //关闭ResultSet对象
stmt.close();                                                                //关闭Statement对象
conn.close();                                                                //关闭数据库连接
```
输出：
```
Name:Tom
Age:30
Email:tom@example.com
```
## 2.4 更新数据
更新id值为1的员工的年龄：
```java
Statement stmt = conn.createStatement();                   //创建Statement对象
int rows = stmt.executeUpdate("UPDATE employee SET age=31 WHERE id=1");             //执行SQL语句，返回影响行数
System.out.println("Update affected " + rows + " rows.");               //打印出受影响的行数
stmt.close();                                                  //关闭Statement对象
conn.close();                                                  //关闭数据库连接
```
## 2.5 删除数据
删除id值为1的员工数据：
```java
Statement stmt = conn.createStatement();                     //创建Statement对象
int rows = stmt.executeUpdate("DELETE FROM employee WHERE id=1");              //执行SQL语句，返回影响行数
System.out.println("Delete affected " + rows + " rows.");                //打印出受影响的行数
stmt.close();                                                   //关闭Statement对象
conn.close();                                                   //关闭数据库连接
```