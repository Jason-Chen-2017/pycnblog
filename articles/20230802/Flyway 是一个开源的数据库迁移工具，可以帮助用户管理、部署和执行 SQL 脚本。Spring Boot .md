
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 ## 什么是 Flyway？
         Flyway 是一款开源的数据库版本管理工具，它允许你管理你的数据库变更。从而你可以轻松地将你的数据库迁移到最新版本，同时还能保留数据的完整性。你不用手动编写复杂的SQL脚本，只需要编写一个描述你想要迁移到的最新版本的SQL脚本即可。Flyway 可以通过连接不同的数据库系统实现不同数据库之间的迁移。它也支持很多种数据库，包括 MySQL、PostgreSQL、Oracle、SQL Server、DB2等。
          
          ## 为什么要使用 Flyway?
          使用 Flyway 有以下好处：
          
          - 自动化的数据库版本管理；
          - 降低数据库维护成本；
          - 加快项目上线时间；
          - 保证数据一致性。
          
          ### 自动化的数据库版本管理
          使用 Flyway ，你只需要关注你所要的数据库更新，而不是手动编写 SQL 脚本。Flyway 会自动检测你的数据库版本，并按照设定的顺序进行执行升级或降级的操作。你只需要编写最新的数据库结构或数据的更新，Flyway 将会自动处理剩下的工作，确保数据库的版本一直保持在最新状态。
          
          ### 降低数据库维护成本
          使用 Flyway ，你可以快速且正确地迁移你的数据库，并确保数据始终保持一致性。Flyway 能够识别已有的数据库表结构的改变，根据你提供的 SQL 脚本自动修改数据库结构。这样你就不需要手动修改数据库结构或运行繁琐的数据库脚本。而且，由于 Flyway 的自动化特性，你也无需担心数据库性能的影响。
          
          ### 加快项目上线时间
          Flyway 使得数据库变更过程自动化，项目上线速度明显加快。如果应用中的某个功能需要对数据库做出改变，只需要编写相应的 SQL 脚本，然后发布给其他开发人员就可以了。Flyway 会自动检测数据库的版本，并执行相应的升级或降级操作。其他开发人员只需要检查这个提交中是否包含相关的 SQL 脚本即可，不需要自己手动执行任何数据库操作。这大大提高了项目的敏捷性和效率。
          
          ### 保证数据一致性
          在分布式应用系统中，Flyway 可以保证数据库的一致性。Flyway 会监控你应用的数据库，并且在发生主备切换时，能够确保数据库的版本一致。这可以避免由于网络延迟或者其他原因造成的数据不一致问题。另外，Flyway 提供了一个回滚机制，你可以在发生错误时，很容易地回退到之前的版本。
          
          ## Spring Boot 中的 Flyway
          在 Spring Boot 中集成了 Flyway。你可以通过 spring-boot-starter-flyway 模块实现数据库版本管理。该模块提供了自动配置 Flyway 的能力，通过简单配置，你就可以使用 Flyway 来管理数据库变更。
          
          ### 配置 Flyway
          通过引入依赖 spring-boot-starter-flyway 和添加 flyway.xml 文件，你可以配置 Flyway。
          
          #### 添加依赖
          ```
          <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-flyway</artifactId>
          </dependency>
          ```
          
          #### 创建 flyway.xml
          在 resources/db 目录下创建一个名为 flyway.xml 的配置文件。
          
          ```
          <?xml version="1.0" encoding="UTF-8"?>
          <!DOCTYPE configure PUBLIC "-//mybatis.org//DTD Config 3.0//EN" "http://mybatis.org/dtd/mybatis-3-config.dtd">
          <!--
            | This is a sample Flyway configuration that demonstrates some of the available options.
            |
            | It is recommended to NOT include credentials in this file and instead provide them separately from
            | your application's properties files or as command line arguments when running migrations.
            | See https://flywaydb.org/documentation/commandline for more information on how to do this.
            -->
          <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
              xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
              
                  <datasources>
                      <datasource driver="${jdbc.driverClassName}" url="${jdbc.url}" user="${jdbc.username}" password="${<PASSWORD>}"/>
                  </datasources>
              
              <defaultSchemaStrategy>
                  <baseSchemaPattern></baseSchemaPattern>
                  <targetSchema>${flyway.schemas}</targetSchema>
              </defaultSchemaStrategy>
  
              <!-- Baselines can be used to specify a common starting point for each database schema. -->
              <!--<baselineVersion initialVersion="0.0.1" /><!-- Do not use in production! -->
              
              <!-- The table to store applied migration versions -->
              <table name="${flyway.table}">${flyway.placeholderReplacement}</table>
              
              <!-- You can customize which sql script types are executed by specifying their prefixes (e.g., V) here -->
              <!-- Note that the default value includes all known types such as.sql,.rsql and.java. -->
              <sqlScriptEncoding>UTF-8</sqlScriptEncoding>
              
              <!-- If set to true, only out-of-order migrations will be run. -->
              <outOfOrder>true</outOfOrder>
              
              <!-- Whether to automatically create the flyway_schema_history table if it does not exist. -->
              <createDatabaseSchemas>true</createDatabaseSchemas>
              
              <!-- Whether to group all pending migrations together in the same transaction. Set this to false if you experience issues with transactions in your database. -->
              <group>true</group>
              
              <!-- Whether to allow mixing transactional and non-transactional statements within the same migration. -->
              <mixed>false</mixed>
              
              <!-- Whether to validate migrations before running them. -->
              <validateOnMigrate>false</validateOnMigrate>
              
              <!-- Whether to clean up metadata tables during a clean. -->
              <cleanDisabled>false</cleanDisabled>
              
              <!-- Whether to disable streaming of change sets. When disabled, Flyway streams changes rather than loading them into memory.-->
              <stream>false</stream>
              
              <!-- Allows migrations to be run even when there are no changes to apply. -->
              <ignoreMissingMigrations>false</ignoreMissingMigrations>
              
              <!-- Allows applying migrations in multiple threads in parallel. -->
              <parallel>false</parallel>
              
              <!-- The maximum number of retries when attempting to connect to the database. -->
              <retryCount>0</retryCount>
              
              <!-- The interval between retries (in milliseconds). -->
              <retryInterval>1000</retryInterval>
              
              <!-- The username to use to connect to the database using JDBC. This property has lower priority than dataSource.user if both are set. -->
              <dataSource.user></dataSource.user>

              <!-- The password to use to connect to the database using JDBC. This property has lower priority than dataSource.password if both are set. -->
              <dataSource.password></dataSource.password>
              
          </configuration>
          ```
          
          ##### 参数说明
          - datasources：指定你的数据库连接信息。这里建议将数据库密码加密存储。
          - ${jdbc.driverClassName}：数据库驱动类名。例如：com.mysql.cj.jdbc.Driver
          - ${jdbc.url}：数据库 URL。例如：jdbc:mysql://localhost:3306/${databaseName}?useSSL=false&serverTimezone=UTC
          - ${jdbc.username}：数据库用户名。
          - ${jdbc.password}：数据库密码。
          - baseSchemaPattern：数据库模式名称（如果为空则匹配所有模式）。
          - targetSchema：目标模式名称。
          - placeholderReplacement：是否开启占位符替换。
          - flyway.table：数据库 flyway_schema_history 表名。
          - flyway.placeholderReplacement：是否开启占位符替换。默认为 true。
          - sqlScriptEncoding：SQL 脚本编码格式。默认 UTF-8 。
          - outOfOrder：是否允许跳过已经执行过的 Migration 。
          - createDatabaseSchemas：是否创建数据库模式（由 targetSchema 指定）。
          - mixed：是否允许混合事务和非事务语句。默认 false。
          - validateOnMigrate：是否验证 SQL 脚本。默认 false。
          - stream：是否启用流处理。默认 false。
          - ignoreMissingMigrations：是否忽略缺失的 Migration 。默认 false。
          - parallel：是否允许多线程。默认 false。
          - retryCount：连接失败重试次数。默认 0。
          - retryInterval：重试间隔。默认 1秒 。
          - dataSource.user：JDBC 用户名。此设置优先于 dataSource.user 设置。
          - dataSource.password：JDBC 密码。此设置优先于 dataSource.password 设置。
          
          
          ### 使用 Flyway
          1. 创建实体类并继承 JpaRepository：
        
            ```java
            @Entity
            public class User {
                private String id;
                
                // other fields
                
                @Id
                public String getId() {
                    return id;
                }

                public void setId(String id) {
                    this.id = id;
                }
                
            }
            
            public interface UserRepository extends JpaRepository<User, String> {}
            ```
            
          2. 修改 flyway.xml 中的数据库模式：
        
            ```xml
            <defaultSchemaStrategy>
                <baseSchemaPattern></baseSchemaPattern>
                <targetSchema>newSchema</targetSchema>
            </defaultSchemaStrategy>
            ```
            
          3. 执行数据库初始化操作：`./gradlew bootRun`，并在浏览器输入 `http://localhost:8080/actuator/flyway`。在 flyway 属性页中确认 newSchema 的状态为 `current`，并且 User 表已被创建。
           
          如果出现无法连接数据库的问题，请检查配置文件中的数据库连接参数是否正确。
          
          ## Flyway VS Liquibase
          Flyway 和 Liquibase 都是 Java 编程语言编写的开源数据库版本管理工具。但两者之间存在一些差异。Liquibase 更侧重于 XML 格式的配置文件，而 Flyway 是基于 Java 的配置方式。Liquibase 的优点是在 XML 配置文件中可以配置更多选项，如指定回滚策略、执行自定义的校验规则等。另一方面，Flyway 没有 XML 配置文件，所有的配置都通过 Java 代码来完成。对于小型的项目来说，Flyway 比较适合使用，但对于大型的分布式集群环境来说，Liquibase 可能会更适合一些。
          
          ## Flyway 的限制
          Flyway 最大的限制就是只支持单一数据库系统。因此，如果你需要管理多个数据库系统，比如关系型数据库和 NoSQL 数据库，Flyway 只能选择其中一个作为主要的版本控制数据库。对于分布式环境来说，Flyway 会遇到额外的限制。如果节点间的网络连接较差，Flyway 会导致大量的版本控制任务超时，进而导致整个流程的失败。另外，Flyway 不支持执行数据库脚本文件，只能执行 DDL 操作。虽然 Flyway 针对分布式集群环境设计了可靠的失败恢复机制，但仍然不能完全解决分布式环境中的同步问题。