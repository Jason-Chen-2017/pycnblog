
作者：禅与计算机程序设计艺术                    

# 1.简介
  

MyBatis 是一款优秀的持久层框架，它支持定制化 SQL、存储过程以及高级映射。 MyBatis 本身集成 Hibernate 对象关系映射框架而生，相对 Hibernate 来说更加简单易用，而且 MyBatis 集成了数据库连接池，所以性能上比 Hibernate 更好。同时 MyBatis 提供 XML 和注解两种形式的配置文件，更适合复杂项目的开发。本文基于 MyBatis 的版本 3.x ，首先介绍 MyBatis 的基本概念、背景及优点。然后详细介绍 MyBatis 的配置方法，包括数据库连接池、Mapper 文件编写、SQL 动态执行等。最后，我将根据 MyBatis 在实际项目中的应用场景，提出 MyBatis 使用建议并给出一些典型案例。希望读者通过阅读本文后，能够对 MyBatis 有个全面的认识，提升自己的技术水平。
# 2. MyBatis 的基本概念
## 2.1 Mybatis 是什么？
MyBatis 是一款优秀的持久层框架，它支持定制化 SQL、存储过程以及高级映射。 MyBatis 本身集成 Hibernate 对象关系映射框架而生，相对 Hibernate 来说更加简单易用，而且 MyBatis 集成了数据库连接池，所以性能上比 Hibernate 更好。
### 2.1.1 持久层（Persistence Layer）
持久层（英语：Persistence Layer）就是把程序的数据（如数据表记录或对象状态）保存到永久性存储器中，以便后续可以访问该数据，常用的数据库有 Oracle、MySQL、PostgreSQL、SQLite、SQL Server 等。而 MyBatis 通过定义简单的 xml 或注解的方式来管理这些关系数据库的查询和更新。通过mybatis框架，应用程序不需要直接处理 JDBC API，就可以执行简单的 CRUD 操作。因此， MyBatis 把数据访问层和业务逻辑层分离开来，在一定程度上提高了程序的可维护性和灵活性。
### 2.1.2 缓存（Cache）
缓存是计算机系统中重要的优化手段之一。当用户频繁访问同一个数据库时，为了减少响应时间，将访问过的结果（一般是从数据库读取出的）进行缓存，下次再相同请求时，直接从缓存中获取数据，避免重新访问数据库降低服务器负担。缓存对于提高网站的访问速度具有至关重要的作用。MyBatis 由于引入了本地缓存（Local Cache），使得每次查询都会先在缓存中查找，缓存命中率更高，效率也更高。
### 2.1.3 简化JDBC编程
JDBC(Java Database Connectivity) 是Java用来连接关系型数据库的一种API。如果要访问多个数据库系统的话，需要重复编写大量的代码。MyBatis 可以简化 JDBC 编程，将原有的 JDBC 代码移植到 MyBatis 上面去，只需简单地配置 mapper 文件即可完成数据库操作。
### 2.1.4 ORM (Object-Relational Mapping) 框架
ORM 框架将数据库表映射成为对象，提供简单的 API 来操作数据库。ORM 框架可以极大的方便程序员完成数据库操作。目前比较流行的两个开源 ORM 框架为 Hibernate 和 JPA 。Hibernate 是 Apache 基金会的一个子项目，主要是一个开放源代码的hibernate ORM框架。JPA （Java Persistence API）是 Sun Microsystems 推出的ORM规范。Hibernate 和 JPA 都是ORM框架，都可以跟Spring、Struts等web框架集成。
### 2.1.5 反向工程（Reverse Engineering）
反向工程（英语：Reverse Engineering）是指通过分析存储在计算机中的信息，重建、复制或者还原原始的设计或编码文档。按照这种方式，可以获取数据库结构、字段信息、索引、约束条件等。MyBatis 支持逆向工程，能够自动生成对应的 entity、xml 配置文件、mapper接口、DAO实现类。这样可以节省大量的重复工作。
### 2.1.6 分布式事务（Distributed Transaction）
分布式事务是指事务的参与者、支持事务的资源服务器以及Transaction Manager分别位于不同的分布式节点上。容错机制保证了事务最终的成功，使得分布式事务同单机事务一样，是一种非常有力的技术。
# 3. MyBatis 的优点
MyBatis 有很多优点，如下所示：
1. 使用简单的 XML 或注解来配置mybatis，将持久层代码从程序逻辑中分离出来，使得业务逻辑关注于具体的SQL语句，而不是纠结于各种技术细节。
2. SQL 只写一次，mybatis 会将 SQL 解析成可重用的 prepared statement，使得代码不用重复编写，提高了开发效率。
3. 单表查询只需要一条 sql 语句，复杂的 join 查询mybatis也能自动拼装sql，简化开发。
4. 可以使用绑定变量或映射参数，灵活调用 stored procedure。
5. 提供映射器接口，自定义返回类型和输入参数类型，实现 sql 和 java 对象的双向转换。
6. mybatis 预编译语句功能可以有效防止 SQL 注入攻击。
7. 执行数据库操作，返回的结果自动映射到 java 对象，支持多种数据库，如 MySQL、Oracle、SQLServer等。

# 4. MyBatis 配置
MyBatis 配置包含三个方面：数据库连接池配置、Mapper 配置、全局配置文件。

## 4.1 数据库连接池配置
MyBatis 使用数据库连接池来连接数据库，在 MyBatis 的配置文件中，可以使用 properties 标签配置数据库连接信息。示例如下：

```xml
<properties>
    <property name="driver" value="${jdbc.driver}" />
    <property name="url" value="${jdbc.url}" />
    <property name="username" value="${jdbc.username}" />
    <property name="password" value="${jdbc.password}" />
</properties>
```

其中 jdbc.* 表示系统属性，${} 表示引用系统属性的值。数据库连接池可以通过以下三种方式进行配置：

1. c3p0 数据连接池：c3p0 是 MyBatis 默认的数据库连接池，可以在 MyBatis 的 pom.xml 文件中通过 maven 依赖加入 c3p0 数据连接池的 jar 包，并在 mybatis 配置文件中加入如下内容：

   ```xml
   <typeAliases>
       <package name="com.mycompany.pojo"/>
   </typeAliases>
   
   <!-- DBCP 数据连接池配置 -->
   <!--<typeAliases>-->
      <!--<package name="com.mchange.v2.c3p0"/>-->
   <!--</typeAliases>-->
   
   <!-- 使用数据连接池，取消注释 -->
   <dataSource type="POOLED">
     <property name="driverClass" value="${driver}"/>
     <property name="jdbcUrl" value="${url}"/>
     <property name="username" value="${username}"/>
     <property name="password" value="${password}"/>
   </dataSource>
   
   <!-- 使用 C3P0 数据连接池，取消注释 -->
   <!-- 
   <dataSource type="POOLED">
      <property name="driverClass" value="${driver}"/>
      <property name="jdbcUrl" value="${url}"/>
      <property name="username" value="${username}"/>
      <property name="password" value="${password}"/>
      <property name="maxPoolSize" value="10"/> 
      <property name="minPoolSize" value="5"/>
      <property name="initialPoolSize" value="5"/>  
      <property name="checkoutTimeout" value="30000"/> 
      <property name="acquireIncrement" value="5"/> 
      <property name="maxIdleTime" value="600"/>      
      <property name="breakAfterAcquireFailure" value="true"/>
      <property name="testConnectionOnCheckin" value="true"/>   
      <property name="testConnectionOnCheckout" value="false"/>    
      <property name="numTestsPerEvictionRun" value="-1"/> 
      <property name="softMinEvictableIdleTimeMillis" value="10000"/> 
      <property name="timeBetweenEvictionRunsMillis" value="300000"/>     
      <property name="preferredTestQuery" value="SELECT 1"/>
   </dataSource>
   -->
   
   <!-- 初始化脚本和清理脚本，可选配置 -->
   <!--
   <script>
      <![CDATA[
          CREATE TABLE IF NOT EXISTS test_table (
              id INT PRIMARY KEY,
              field VARCHAR(50),
              create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          );
          
          TRUNCATE TABLE test_table;
      ]]>
  </script>
   -->
   
   <!-- 测试连接是否可用，可选配置 -->
   <!--
   <environments default="development">
      <environment id="development">
         <transactionManager type="JDBC"/> 
         < dataSource type="POOLED">
            <property name="driverClass" value="${driver}"/>
            <property name="jdbcUrl" value="${url}"/>
            <property name="username" value="${username}"/>
            <property name="password" value="${password}"/>            
            <property name="maxPoolSize" value="10"/> 
            <property name="minPoolSize" value="5"/>
            <property name="initialPoolSize" value="5"/>  
            <property name="checkoutTimeout" value="30000"/> 
            <property name="acquireIncrement" value="5"/> 
            <property name="maxIdleTime" value="600"/>      
            <property name="breakAfterAcquireFailure" value="true"/>
            <property name="testConnectionOnCheckin" value="true"/>   
            <property name="testConnectionOnCheckout" value="false"/>    
            <property name="numTestsPerEvictionRun" value="-1"/> 
            <property name="softMinEvictableIdleTimeMillis" value="10000"/> 
            <property name="timeBetweenEvictionRunsMillis" value="300000"/>     
            <property name="preferredTestQuery" value="SELECT 1"/>
         </dataSource>         
      </environment>        
   </environments>
   -->
   
   <!-- 支持可插拔的数据源插件，可选配置 -->
   <!--
   <plugins>
      <plugin interceptor="org.apache.ibatis.cache.plugins.FifoCachePlugin">
         <property name="size" value="1024"/>
      </plugin>
     ...
   </plugins>
   -->
   ```

2. DBCP 数据连接池：DBCP（DataBase Connection Pool）是另一种常用的数据库连接池。可以从 http://commons.apache.org/proper/commons-dbcp/download_dbcp.cgi 获取 DBCP 的 jar 包，并在 mybatis 配置文件中加入如下内容：

   ```xml
   <typeAliases>
       <package name="com.mycompany.pojo"/>
   </typeAliases>
   
   <!-- 使用数据连接池，取消注释 -->
   <dataSource type="UNPOOLED">
     <property name="driverClassName" value="${driver}"/>
     <property name="url" value="${url}"/>
     <property name="username" value="${username}"/>
     <property name="password" value="${password}"/>
     <property name="maximumActiveConnections" value="10"/>
   </dataSource>
   
   <!-- 初始化脚本和清理脚本，可选配置 -->
   <!--
   <script>
      <![CDATA[
          CREATE TABLE IF NOT EXISTS test_table (
              id INT PRIMARY KEY,
              field VARCHAR(50),
              create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          );
          
          TRUNCATE TABLE test_table;
      ]]>
  </script>
   -->
   
   <!-- 测试连接是否可用，可选配置 -->
   <!--
   <environments default="development">
      <environment id="development">
         <transactionManager type="RESOURCE_LOCAL"/> 
         <dataSource type="UNPOOLED">
            <property name="driverClassName" value="${driver}"/>
            <property name="url" value="${url}"/>
            <property name="username" value="${username}"/>
            <property name="password" value="${password}"/>
            <property name="maximumActiveConnections" value="10"/>
         </dataSource>           
      </environment>        
   </environments>
   -->
   
   <!-- 支持可插拔的数据源插件，可选配置 -->
   <!--
   <plugins>
      <plugin interceptor="org.apache.ibatis.cache.plugins.LRUCachePlugin">
         <property name="maxSize" value="512"/>
      </plugin>
     ...
   </plugins>
   -->
   ```

3. Druid 数据连接池：Druid 是阿里巴巴开源产品，集成了 c3p0、DBCP 等主流的数据库连接池功能。可以从 https://github.com/alibaba/druid/releases 获取 Druid 的 jar 包，并在 mybatis 配置文件中加入如下内容：

   ```xml
   <typeAliases>
       <package name="com.mycompany.pojo"/>
   </typeAliases>
   
   <!-- 使用数据连接池，取消注释 -->
   <dataSource type="POOLED">
      <property name="driverClassName" value="${driver}"/>
      <property name="url" value="${url}"/>
      <property name="username" value="${username}"/>
      <property name="password" value="${password}"/>
      <property name="filters" value="stat"/>
      <property name="maxActive" value="10"/>
      <property name="initialSize" value="5"/>
      <property name="maxWait" value="60000"/>
      <property name="timeBetweenEvictionRunsMillis" value="60000"/>
      <property name="minEvictableIdleTimeMillis" value="300000"/>
      <property name="validationQuery" value="select 'x'"/>
      <property name="testWhileIdle" value="true"/>
      <property name="testOnBorrow" value="false"/>
      <property name="testOnReturn" value="false"/>
      <property name="poolPreparedStatements" value="false"/>
      <property name="maxOpenPreparedStatements" value="-1"/>
   </dataSource>
   
   <!-- 初始化脚本和清理脚本，可选配置 -->
   <!--
   <script>
      <![CDATA[
          CREATE TABLE IF NOT EXISTS test_table (
              id INT PRIMARY KEY,
              field VARCHAR(50),
              create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          );
          
          TRUNCATE TABLE test_table;
      ]]>
  </script>
   -->
   
   <!-- 测试连接是否可用，可选配置 -->
   <!--
   <environments default="development">
      <environment id="development">
         <transactionManager type="JDBC"/> 
         <dataSource type="POOLED">
            <property name="driverClassName" value="${driver}"/>
            <property name="url" value="${url}"/>
            <property name="username" value="${username}"/>
            <property name="password" value="${password}"/>
            <property name="filters" value="stat"/>
            <property name="maxActive" value="10"/>
            <property name="initialSize" value="5"/>
            <property name="maxWait" value="60000"/>
            <property name="timeBetweenEvictionRunsMillis" value="60000"/>
            <property name="minEvictableIdleTimeMillis" value="300000"/>
            <property name="validationQuery" value="select 'x'"/>
            <property name="testWhileIdle" value="true"/>
            <property name="testOnBorrow" value="false"/>
            <property name="testOnReturn" value="false"/>
            <property name="poolPreparedStatements" value="false"/>
            <property name="maxOpenPreparedStatements" value="-1"/>
         </dataSource>         
      </environment>        
   </environments>
   -->
   
   <!-- 支持可插拔的数据源插件，可选配置 -->
   <!--
   <plugins>
      <plugin interceptor="com.alibaba.druid.support.spring.stat.DruidStatInterceptor">
         <property name="mergeSql" value="true"/>
      </plugin>
     ...
   </plugins>
   -->
   ```

以上提供了三种常用的数据库连接池的配置方法，各自适用于不同场景。

## 4.2 Mapper 配置
MyBatis 将 SQL 语句抽象成 XML 文件，并称之为 mapper 文件。每个 mapper 文件中包含了相似的 SQL 语句集合，并使用 `<statement>` 标签定义每条 SQL 语句，并为其指定唯一标识符 id。示例如下：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<!-- namespace: com.mycompany.dao.UserDao -->
<mapper namespace="com.mycompany.dao.UserDao">
  
  <!-- 定义查询所有用户的 SQL 语句 -->
  <select id="selectAllUsers" resultType="com.mycompany.pojo.User">
    SELECT * FROM users
  </select>

  <!-- 定义插入用户的 SQL 语句 -->
  <insert id="insertUser" parameterType="com.mycompany.pojo.User">
    INSERT INTO users (name, age, email) VALUES (#{name}, #{age}, #{email})
  </insert>

  <!-- 定义删除用户的 SQL 语句 -->
  <delete id="deleteUser" parameterType="int">
    DELETE FROM users WHERE id = #{id}
  </delete>

</mapper>
```

此处，namespace 指定了相应 DAO 接口的命名空间，即 `com.mycompany.dao.UserDao`，id 为 SQL 语句的唯一标识符。parameterType 和 resultType 为 MyBatis 在执行 SQL 时所需的参数类型和结果类型，分别对应于 SQL 中的 IN/OUT 参数和 SELECT 返回值。具体的属性名称要和 POJO 属性名称匹配，如 User 的 name 属性对应 "#{name}"。

## 4.3 全局配置文件
 MyBatis 全局配置文件包含了 MyBatis 运行环境相关的所有配置项。一般情况下， MyBatis 的全局配置文件为 `mybatis-config.xml` 文件，示例如下：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE configuration SYSTEM "http://mybatis.org/dtd/mybatis-3-config.dtd">
<configuration>

  <!-- 配置数据库连接信息 -->
  <properties resource="database.properties"/>

  <!-- 定义实体扫描路径，MyBatis 会自动扫描该路径下的 pojo 类 -->
  <typeAliases>
    <package name="com.mycompany.pojo"/>
  </typeAliases>

  <!-- 设置 mappers 路径，告诉 MyBatis 需要加载哪些 mapper 文件 -->
  <mappers>
    <mapper resource="com/mycompany/dao/UserDao.xml"/>
  </mappers>

  <!-- 设置数据库连接池信息，MyBatis 自动加载数据连接池 -->
  <settings>
    <setting name="logImpl" value="LOG4J"/>
    <setting name="cacheEnabled" value="true"/>
    <setting name="lazyLoadingEnabled" value="true"/>
    <setting name="aggressiveLazyLoading" value="true"/>
    <setting name="multipleResultSetsEnabled" value="true"/>
    <setting name="useColumnLabel" value="true"/>
    <setting name="autoMappingBehavior" value="PARTIAL"/>
    <setting name="defaultExecutorType" value="REUSE"/>
    <setting name="mapUnderscoreToCamelCase" value="true"/>
  </settings>

  <!-- 可选配置，设置 MyBatis 日志输出类型 -->
  <!--
  <loggers>
    <logger type="STDOUT"/>
  </loggers>
  -->

  <!-- 可选配置，设置 MyBatis 插件信息 -->
  <!--
  <plugins>
    <plugin interceptor="com.mycompany.MyBatisPlugin"/>
  </plugins>
  -->
  
</configuration>
```

此处，resource 指定了数据库连接信息的配置文件 database.properties；`<typeAliases>` 标签定义了 MyBatis 所需要扫描的实体类所在的包名；`<mappers>` 标签定义了 MyBatis 所使用的 mapper 文件所在的文件夹，MyBatis 会自动加载该文件夹中的所有 mapper 文件；`<settings>` 标签定义了 MyBatis 的一些运行环境相关的设置选项，比如日志输出类型、是否启用缓存、懒加载等；`<loggers>` 标签可选择性配置 MyBatis 的日志输出方式，比如输出到控制台或者日志文件；`<plugins>` 标签可选择性配置 MyBatis 的插件，例如自定义插件 MybatisPlugin。

# 5. MyBatis 实践
## 5.1 Mybatis 在 Spring Boot 项目中的使用
### 5.1.1 添加 MyBatis starter 依赖
Mybatis starter 是 spring boot 官方提供的 mybatis 模块的 starter。在pom.xml 文件中添加如下依赖：

```xml
<dependency>
    <groupId>org.mybatis.spring.boot</groupId>
    <artifactId>mybatis-spring-boot-starter</artifactId>
    <version>${mybatis-spring-boot.version}</version>
</dependency>
```

其中 `${mybatis-spring-boot.version}` 为 mybatis-spring-boot 的最新版本号。

### 5.1.2 创建 MyBatis 数据库表
创建数据库表 user ：

```sql
CREATE TABLE user (
  id int primary key auto_increment not null,
  name varchar(50) not null,
  age int not null,
  email varchar(100) not null
);
```

### 5.1.3 创建 POJO 类
```java
public class User {

    private Integer id;
    private String name;
    private Integer age;
    private String email;
    
    // getter and setter methods
}
```

### 5.1.4 创建 MyBatis mapper 文件
创建一个 `UserDao.xml` 文件，内容如下：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.demo.dao.UserDao">
 
  <!-- 根据 ID 查找用户 -->
  <select id="findById" parameterType="int" resultType="com.example.demo.model.User">
    select id, name, age, email from user where id=#{id};
  </select>

  <!-- 新增用户 -->
  <insert id="saveUser" parameterType="com.example.demo.model.User">
    insert into user(name, age, email) values(#{name}, #{age}, #{email});
  </insert>

  <!-- 删除用户 -->
  <delete id="removeById" parameterType="int">
    delete from user where id=#{id};
  </delete>

</mapper>
```

这里，`namespace` 指定了 `UserDao` 的命名空间。`parameterType` 和 `resultType` 为 MyBatis 在执行 SQL 时所需的参数类型和结果类型。具体的属性名称要和 POJO 属性名称匹配，如 User 的 name 属性对应 "#{name}"。

### 5.1.5 创建单元测试类
```java
@RunWith(SpringRunner.class)
@SpringBootTest
public class UserServiceTest {

    @Autowired
    private UserDao userDao;
    
    /**
     * 测试根据 ID 查找用户
     */
    @Test
    public void findById() {
        User user = userDao.findById(1);
        System.out.println(user);
    }
    
    /**
     * 测试新增用户
     */
    @Test
    public void saveUser() {
        User user = new User();
        user.setName("小明");
        user.setAge(20);
        user.setEmail("<EMAIL>");
        
        int rows = userDao.saveUser(user);
        Assert.assertEquals(rows, 1);
    }
    
    /**
     * 测试删除用户
     */
    @Test
    public void removeById() {
        int rows = userDao.removeById(1);
        Assert.assertEquals(rows, 1);
    }
    
}
```

注意 `@SpringBootTest` 注解，这是启动整个 Spring Boot 项目的注解。

### 5.1.6 启动 Spring Boot 项目
启动 Spring Boot 项目，并查看控制台输出结果。可以看到 MyBatis 执行了相应的 SQL 语句，并将查询到的用户信息打印到了控制台。

## 5.2 Mybatis 在 Spring 项目中的使用
### 5.2.1 创建 Spring 项目
创建一个 Spring 项目，在 pom.xml 文件中添加 MyBatis starter 依赖：

```xml
<dependencies>
    <dependency>
        <groupId>org.mybatis.spring.boot</groupId>
        <artifactId>mybatis-spring-boot-starter</artifactId>
        <version>${mybatis-spring-boot.version}</version>
    </dependency>
    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <scope>runtime</scope>
    </dependency>
    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
        <optional>true</optional>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
        <exclusions>
            <exclusion>
                <groupId>org.junit.vintage</groupId>
                <artifactId>junit-vintage-engine</artifactId>
            </exclusion>
        </exclusions>
    </dependency>
</dependencies>
```

其中 `${mybatis-spring-boot.version}` 为 mybatis-spring-boot 的最新版本号。

### 5.2.2 修改 application.yml 文件
修改 application.yml 文件，增加数据库连接信息：

```yaml
spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/test?serverTimezone=UTC&useSSL=false
    username: root
    password: root
```

### 5.2.3 创建 MyBatis 数据库表
创建数据库表 user : 

```sql
CREATE TABLE user (
  id int primary key auto_increment not null,
  name varchar(50) not null,
  age int not null,
  email varchar(100) not null
);
```

### 5.2.4 创建 POJO 类
```java
import lombok.*;

@Data
@AllArgsConstructor
@NoArgsConstructor
@ToString
public class User {

    private Integer id;
    private String name;
    private Integer age;
    private String email;

}
```

### 5.2.5 创建 MyBatis mapper 文件
创建一个 `UserDao.xml` 文件，内容如下：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.demo.dao.UserDao">
 
  <!-- 根据 ID 查找用户 -->
  <select id="findById" parameterType="int" resultType="com.example.demo.model.User">
    select id, name, age, email from user where id=#{id};
  </select>

  <!-- 新增用户 -->
  <insert id="saveUser" parameterType="com.example.demo.model.User">
    insert into user(name, age, email) values(#{name}, #{age}, #{email});
  </insert>

  <!-- 删除用户 -->
  <delete id="removeById" parameterType="int">
    delete from user where id=#{id};
  </delete>

</mapper>
```

这里，`namespace` 指定了 `UserDao` 的命名空间。`parameterType` 和 `resultType` 为 MyBatis 在执行 SQL 时所需的参数类型和结果类型。具体的属性名称要和 POJO 属性名称匹配，如 User 的 name 属性对应 "#{name}"。

### 5.2.6 创建 Spring 配置类
```java
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.context.annotation.Configuration;

@Configuration
@MapperScan({"com.example.demo.dao"})
public class ApplicationConfig {
    
}
```

`@MapperScan` 注解表示扫描指定路径下的 mapper 文件。

### 5.2.7 创建单元测试类
```java
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes={ApplicationConfig.class})
public class UserServiceTest {

    @Autowired
    private UserDao userDao;
    
    /**
     * 测试根据 ID 查找用户
     */
    @Test
    public void findById() {
        User user = userDao.findById(1);
        System.out.println(user);
    }
    
    /**
     * 测试新增用户
     */
    @Test
    public void saveUser() {
        User user = new User(null, "小明", 20, "<EMAIL>");
        int rows = userDao.saveUser(user);
        Assert.assertEquals(rows, 1);
    }
    
    /**
     * 测试删除用户
     */
    @Test
    public void removeById() {
        int rows = userDao.removeById(1);
        Assert.assertEquals(rows, 1);
    }
    
}
```

`@ContextConfiguration` 注解指定使用的 Spring 配置类，这里设置为 `ApplicationConfig`。

### 5.2.8 启动 Spring 项目
启动 Spring 项目，并查看控制台输出结果。可以看到 MyBatis 执行了相应的 SQL 语句，并将查询到的用户信息打印到了控制台。