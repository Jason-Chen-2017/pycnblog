
作者：禅与计算机程序设计艺术                    

# 1.简介
  

MyBatis 是一款优秀的持久层框架，它支持定制化SQL、存储过程以及高级映射。Mybatis-Generator 插件是MyBatis官方提供的逆向工程工具，通过定义数据库表结构或配置文件，自动生成符合mybatis规范的Dao接口及映射文件。插件可以帮你快速完成开发工作，节省了大量的时间，提升了开发效率。本文将详细介绍使用Mybatis Generator插件生成 MyBatis代码的方法。


# 2.准备工作
## 2.1 安装环境
1. JDK（1.7或以上版本）
2. MySQL 或 Oracle 或 SQL Server
3. Maven (下载并解压到指定目录)
4. Eclipse IDE（可选）

## 2.2 创建项目
首先创建一个maven工程，引入依赖。
```xml
    <dependencies>
        <!-- mybatis generator -->
        <dependency>
            <groupId>org.mybatis</groupId>
            <artifactId>mybatis-generator-core</artifactId>
            <version>1.3.5</version>
        </dependency>

        <!-- mysql驱动 -->
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
            <version>5.1.47</version>
        </dependency>

        <!-- junit -->
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.12</version>
            <scope>test</scope>
        </dependency>

    </dependencies>
```

## 2.3 配置数据库连接信息
在resources目录下创建config.xml配置文件。配置数据库连接信息如下：
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE configuration PUBLIC "-//mybatis.org//DTD Config 3.0//EN" "http://mybatis.org/dtd/mybatis-3-config.dtd">
<configuration>

  <environments default="development">

    <environment id="development">
      <transactionManager type="JDBC"/>
      <dataSource type="POOLED">
        <property name="driver" value="com.mysql.jdbc.Driver"/>
        <property name="url" value="jdbc:mysql://localhost:3306/your_database?useSSL=false&amp;useUnicode=true&amp;characterEncoding=utf8"/>
        <property name="username" value="root"/>
        <property name="password" value="<PASSWORD>"/>
      </dataSource>
    </environment>
  </environments>

  <mappers>
    <mapper resource="mapping/*.xml"/>
  </mappers>
  
</configuration>
```
其中注意修改数据库连接信息。

## 2.4 创建实体类
定义实体类，通常情况下实体类只需要包含字段名和类型即可，不需要包含getters/setters方法。例如：
```java
public class User {
    private int id;
    private String username;
    private Date createDate;
    
    // getters and setters...
}
``` 

## 2.5 生成配置文件
使用Mybatis Generator插件生成 MyBatis代码前，需要先创建生成配置文件。在项目资源目录中执行以下命令创建generatorConfig.xml文件：
```
mvn mybatis-generator:generate
```
然后根据提示输入以下选项：

1. [1] MyBatis Generator Configuration 
2. [2] Generator Configuration File  
3. Enter the location of your mybatis-generator.xml file.   
4. The directory where you want to generate your Java files.    （默认路径：./target/generated-sources/mybatis-generator）  
5. package name model （此处输入要生成Mapper文件的包名）  
6. target project (选择Eclipse Project 或 IDEA Project)  
7. The fully qualified classname of the root class for your SQL map package.(这里输入使用的SqlMapConfig)  
8. JDBC connection URL for the database.  （此处输入数据库连接信息）  
9. Database driver class name.   （此处输入数据库驱动名称）  
10. Database user name. （此处输入数据库用户名）  
11. Database password.     （此处输入数据库密码）  
12. Generate records with BLOBs (y|n): n （是否生成BLOB记录，一般不用）  
13. Output verbose messages while running. (y|n) : y （是否输出运行信息，建议开启）  
14. Specify any file path to a Velocity template file containing custom comment generation logic: n （是否自定义注释生成逻辑）  
15. Add comments to generated XML files (y|n) : n （是否添加注释到生成的XML文件）  
16. Use deprecated xml element names(deprecated means that it is planned to remove in future versions). (y|n) : n （是否使用过期的XML元素名称，一般选择否）  
17. Complete the plugin execution by entering Y or N: y （是否继续执行插件）  

## 2.6 修改生成器配置文件
打开generatorConfig.xml文件，修改自动生成的文件内容，比如：
```xml
<!-- targetRuntime 指定目标运行时环境为 MyBatis3 -->
<context id="DB2Tables" targetRuntime="MyBatis3">
    <!-- 配置数据库连接 -->
    <jdbcConnection driverClass="${jdbc.driver}"
                    uri="${jdbc.url}">
        <property name="userId" value="${jdbc.user}"/>
        <property name="password" value="${jdbc.password}"/>
    </jdbcConnection>
  
    <!-- 设置生成 mapper 所需的类型别名等信息-->
    <javaTypeResolver>
        <typeAlias alias="BigDecimal" type="java.math.BigDecimal"/>
        <typeAlias alias="Boolean" type="boolean"/>
        <typeAlias alias="Double" type="double"/>
        <typeAlias alias="Float" type="float"/>
        <typeAlias alias="Integer" type="int"/>
        <typeAlias alias="Long" type="long"/>
        <typeAlias alias="Short" type="short"/>
        <typeAlias alias="String" type="java.lang.String"/>
        <typeAlias alias="Object" type="java.lang.Object"/>
    </javaTypeResolver>
    <!-- 控制 javaModelGenerator 是否生成 toString() 方法 -->
    <javaModelGenerator targetPackage="com.xx.dao.entity"
                        enableToStringGeneration="true" >
        <property name="trimStrings" value="true"/>
    </javaModelGenerator>

    <!-- 控制 sqlMapGenerator 的行为，设置生成 SQL 语句的文件位置 -->
    <sqlMapGenerator targetPackage="com.xx.dao.mapper"
                     targetProject=${project}>
    </sqlMapGenerator>

    <!-- 控制生成的 DAO 接口的命名规则，如 xxxDAO, xxxService -->
    <javaClientGenerator type="DAO"
                         targetPackage="com.xx.dao.mapper"
                         targetProject=${project}>
         <property name="rootInterface" value="com.xx.base.BaseDao"/>
    </javaClientGenerator>
 
    <!-- 用于设定生成哪些表的 SQL 文件 -->
    <table tableName="users" domainObjectName="User"
           enableCountByExample="false" enableUpdateByExample="false"
           enableDeleteByExample="false" enableSelectByExample="false"
           selectByExampleQueryId="false">
    	<!-- 指定该表对应实体类的属性和列之间的对应关系 -->
        <columnOverride column="name" property="userName" />
        <columnOverride column="create_time" property="createTime" />
    </table>
    
    <!-- 除了上面的 table 配置项之外，还可以使用自定义的 tableConfiguration 来进一步配置需要生成的代码 -->
    <table tableName="xxxxx" remarks="用户相关表">
        <generatedKey identity="true" keyProperty="id" jdbcType="INTEGER" />
        <columnOverride column="name" property="userName" />
        <columnOverride column="email" property="emailAddress" />
    </table>
</context>
``` 

## 2.7 执行代码生成
执行以下命令：
```
mvn mybatis-generator:generate
```
代码生成完毕后，可以在指定的目录找到生成好的代码。可以直接使用，也可以继续调整生成的代码。