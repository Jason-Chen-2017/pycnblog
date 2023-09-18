
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 为什么要用 MyBatis Generator？
MyBatis 是一款优秀的 ORM 框架，它可以用于数据库持久层访问，但 MyBatis Generator 是 MyBatis 的一个扩展插件，它提供了一种简单的方式来快速生成 MyBatis 模板文件，帮助开发人员快速实现 CRUD 操作。通过 MyBatis Generator，用户只需编写简单的 XML 配置文件即可完成模型类的生成，并可生成 mapper.xml 文件和 DAO 接口及其映射文件。这样做可以降低开发难度、提升效率、保证数据一致性，减少代码维护成本。

## 安装 MyBatis Generator
MyBatis Generator 可直接从 Maven 中央仓库下载安装，在 pom.xml 文件中添加以下依赖：

```
<dependency>
  <groupId>org.mybatis.generator</groupId>
  <artifactId>mybatis-generator-core</artifactId>
  <version>1.4.0</version>
</dependency>
<!--mysql-connector-java dependency for using MySQL database-->
<dependency>
  <groupId>mysql</groupId>
  <artifactId>mysql-connector-java</artifactId>
  <scope>runtime</scope>
</dependency>
```

然后执行命令 `mvn clean install` 来安装 MyBatis Generator 。如果你的项目还没有 MyBatis 相关配置，请按照 MyBatis 官方文档进行设置。

## 配置 MyBatis Generator Plugin
MyBatis Generator 插件的使用主要通过在 pom.xml 文件中的 build 元素下定义 plugin ，如下所示：

```
<build>
    <!-- other configuration elements -->
    <plugins>
       ... 
        <!-- add mybatis-generator-maven-plugin here -->
        <plugin>
            <groupId>org.mybatis.generator</groupId>
            <artifactId>mybatis-generator-maven-plugin</artifactId>
            <version>1.4.0</version>
            <configuration>
                <verbose>true</verbose>
                <configfile>src/main/resources/generatorConfig.xml</configfile>
                <overwrite>true</overwrite>
            </configuration>
            <dependencies>
                <dependency>
                    <groupId>mysql</groupId>
                    <artifactId>mysql-connector-java</artifactId>
                    <version>8.0.17</version>
                </dependency>
            </dependencies>
        </plugin>
       ... 
    </plugins>
</build>
```

上面的配置表示将 MyBatis Generator 插件注册到 Maven 生命周期的构建过程中。其中 verbose 参数用来显示详细日志信息； configfile 指定 MyBatis Generator 插件的配置文件位置； overwrite 参数表示是否覆盖已存在的文件（即是否重新生成）。

## 配置 MyBatis Generator Config File
MyBatis Generator 使用 xml 配置文件作为输入，该文件指定数据库连接信息、需要生成哪些表以及表之间的关系等。生成的文件包括 Java Model、Mapper 和 SQLMap 文件。

下面是一个示例配置文件：

```
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE generatorConfiguration PUBLIC "-//mybatis.org//DTD MyBatis Generator Configuration 1.0//EN" "http://mybatis.org/dtd/mybatis-generator-config_1_0.dtd">
<generatorConfiguration>

    <!-- target project to run generators against -->
    <context id="DB2Tables" targetRuntime="MyBatis3">

        <!--jdbc connection information-->
        <jdbcConnection driverClass="${driver}" connectionURL="${url}" userId="${username}" password="${password}">
            <!-- any specific pool properties here -->
        </jdbcConnection>

        <!-- generated classes will be placed in this package -->
        <javaModelGenerator defaultTargetPackage="${package}.model">
            <property name="enableSubPackages" value="false"/>
            <!-- enable or disable annotations here based on your requirements -->
            <property name="trimStrings" value="true"/>
        </javaModelGenerator>

        <!-- sql map files will be created in this directory -->
        <sqlMapGenerator targetPackage="${package}.mapper" targetProject="${project.basedir}/src/main/resources/">
            <property name="enableSubPackages" value="false"/>
        </sqlMapGenerator>

        <!-- the resulting mapping file will be named ${jdbc.table}.xml and placed in the same directory as the Dao files -->
        <javaClientGenerator type="XMLMAPPER" targetPackage="${package}.dao" targetProject="${project.basedir}/src/main/java/">
            <property name="enableSubPackages" value="false"/>
        </javaClientGenerator>

        <!-- specify which tables to generate from, example uses two schemas, i.e., DBA and APP with different table names -->
        <table schema="DBA" tableName="${tableName}">
        	...
        </table>
        
        <table schema="APP" tableName="${tableName}">
        	...
        </table>

        <!-- do not include ALL synonyms but only those required by the selected tables (optional) -->
        <excludes>
            <exclude objectName="ALL_SYNONYMS"/>
        </excludes>

        <!-- override certain generator behaviors based on specified conditions (optional) -->
        <conditionalHandlers>
            <conditionalHandler columnTypeOverride="${columnType}">{test}</conditionalHandler>
        </conditionalHandlers>

        <!-- mapping warnings can be suppressed based on specified conditions (optional) -->
        <warnings>
            <warning key="1099" message="No TABLE_TYPE attribute found in result set."/>
            <warning message="Property.* of class.* is mapped multiple times in.*\.xml\. Mapping may be brittle if one overwrites another without a comment."/>
        </warnings>
        
    </context>

</generatorConfiguration>
```

上述配置文件共分为五个区域：

1. context：该区域定义了生成文件的输出目录、包名等属性。
2. jdbcConnection：该区域定义了 JDBC 数据库连接参数，例如驱动类、数据库 URL、用户名和密码。
3. javaModelGenerator：该区域定义了生成的 Model 文件输出路径、包名以及一些额外属性，如是否生成子目录。
4. sqlMapGenerator：该区域定义了生成的 SQL Map 文件输出路径、包名以及是否生成子目录。
5. javaClientGenerator：该区域定义了生成的 DAO 文件输出路径、包名以及是否生成子目录。

具体各项参数的含义及用法请参阅 MyBatis Generator 用户指南。