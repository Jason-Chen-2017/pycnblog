
作者：禅与计算机程序设计艺术                    

# 1.简介
  

SpringMVC是一个基于Java的轻量级Web开发框架。它非常适用于构建Web应用。在Web应用中，一般会用到数据交互，如读取、修改、添加、删除等功能。常用的技术是RESTful API，而 MyBatis 是 Java 持久层框架，它可以将数据库中的记录转化成 Java 对象，并对对象进行crud操作。因此本文将详细介绍如何结合SpringMVC和MyBatis，完成CRUD（create、read、update、delete）操作。
# 2.SpringMVC架构图

如上图所示，SpringMVC的请求处理过程主要由以下几个阶段组成：

1.前端控制器DispatcherServlet：首先经过前端控制器DispatcherServlet的映射，将用户的请求分派给相应的HandlerMapping组件，找到相应的Controller对象；
2.解析器HandlerMapping：根据用户请求中的URL查找对应的Handler，并生成HandlerExecutionChain对象，该对象包含一个保存了handler对象的列表，按照order排序；
3.执行链HandlerExectionChain：依次遍历执行链中的handler，直至有一个处理完毕；
4.视图渲染ViewReslover：如果Handler返回的是 ModelAndView对象，则继续通过视图解析器进行视图解析；否则，继续寻找下一个Handler；
5.模型与视图：ModelAndView对象封装了需要传递给view的数据，view负责将model数据填充到页面展示。

# 3.集成Mybatis前提条件
使用Mybatis前提条件如下：

1.安装 MyBatis Generator 插件：Mybatis 提供了一个 MyBatis Generator 插件，该插件用来自动生成 mapper 和 model 的代码文件。下载地址为：http://www.mybatis.org/generator/running/index.html。

2.配置 MyBatis 配置文件：在 applicationContext.xml 文件中配置 MyBatis 数据源 DataSource ，SessionFactory, MapperLocations 。

```xml
<!-- 配置数据源 -->
<bean id="dataSource" class="com.mchange.v2.c3p0.ComboPooledDataSource">
  <property name="driverClass" value="${jdbc.driverClassName}"/>
  <property name="jdbcUrl" value="${jdbc.url}"/>
  <property name="user" value="${jdbc.username}"/>
  <property name="password" value="${jdbc.password}"/>
</bean>

<!-- 配置 SqlSessionFactory -->
<bean id="sqlSessionFactory" class="org.mybatis.spring.SqlSessionFactoryBean">
    <property name="dataSource" ref="dataSource"/>
    <!-- 指定 mybatis 配置文件位置 -->
    <property name="configLocation" value="classpath:mybatis-config.xml"/>
    <!-- 指定 mapper.xml 文件位置 -->
    <property name="mapperLocations" value="classpath*:mapper/*.xml"/>
</bean>

<!-- 配置 MyBatis 类型转换器 -->
<bean id="typeAliasRegistry" class="org.mybatis.spring.boot.autoconfigure.TypeAliasesAutoConfiguration$TypeAliasesRegistrationBean"></bean>

<!-- 将 SqlSessionTemplate 设置为 spring bean 以便于在各个 dao 中直接注入 -->
<bean id="sqlSessionTemplate" class="org.mybatis.spring.SqlSessionTemplate">
    <constructor-arg index="0" ref="sqlSessionFactory"></constructor-arg>
    <!-- 设置默认事物隔离级别 -->
    <property name="isolationLevelName" value="REPEATABLE_READ"></property>
</bean>
```

# 4.创建数据库表及实体类
这里不再赘述，假设已创建好数据库表（test_table），并编写了对应的实体类。如实体类 TestTable：

```java
@Data
public class TestTable implements Serializable {

    private static final long serialVersionUID = -727536704931890542L;
    
    @Id
    @GeneratedValue(strategy=GenerationType.IDENTITY)
    private Integer id;

    private String title;

    private Date createdDate;
}
```

# 5.编写DAO接口及Mapper XML
根据实际情况编写 DAO 接口及其对应的 Mapper XML 文件。如 DAO 接口 ITestTableDao :

```java
public interface ITestTableDao {

    List<TestTable> getAll();

    void add(TestTable testTable);

    void update(TestTable testTable);

    void deleteById(Integer id);

    TestTable getById(Integer id);
}
```

如 Mapper XML 文件 testTableDao.xml：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="dao.ITestTableDao">

    <resultMap type="domain.TestTable" id="BaseResultMap">
        <id property="id" column="ID" />
        <result property="title" column="TITLE" />
        <result property="createdDate" column="CREATEDDATE" />
    </resultMap>

    <select id="getAll" resultMap="BaseResultMap">
        SELECT * FROM TEST_TABLE ORDER BY ID ASC
    </select>

    <insert id="add" parameterType="domain.TestTable">
        INSERT INTO TEST_TABLE (TITLE, CREATEDDATE) VALUES (#{title}, NOW())
    </insert>

    <update id="update" parameterType="domain.TestTable">
        UPDATE TEST_TABLE SET TITLE=#{title}, CREATEDDATE=NOW() WHERE ID=#{id}
    </update>

    <delete id="deleteById" parameterType="int">
        DELETE FROM TEST_TABLE WHERE ID=#{id}
    </delete>

    <select id="getById" resultMap="BaseResultMap">
        SELECT * FROM TEST_TABLE WHERE ID=#{id}
    </select>
    
</mapper>
```

注意：此处省略了 MyBatis 的依赖导入， MyBatis 在启动时会自动扫描 mapper.xml 文件加载配置信息，如果依赖已经导入则可忽略此项配置。

# 6.编写Spring Bean配置文件
编写Spring Bean配置文件 applicationContext.xml 。如：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-3.0.xsd">

    <context:component-scan base-package="dao"/>

    <bean id="testTableDao" class="dao.TestTableDaoImpl">
        <property name="sessionFactory" ref="sqlSessionFactory"></property>
    </bean>

</beans>
```

# 7.编写 Controller 类
编写 Spring MVC Controller 类，如：

```java
@RestController
@RequestMapping("/testTables")
public class TestTableController {

    @Autowired
    private ITestTableDao testTableDao;

    /**
     * 获取所有测试数据
     */
    @GetMapping("")
    public Result<List<TestTable>> getAll(){

        return new Result<>(true,"查询成功",testTableDao.getAll());
    }

    /**
     * 添加测试数据
     */
    @PostMapping("")
    public Result<String> add(@RequestBody TestTable testTable){

        int count = testTableDao.add(testTable);

        if(count > 0){
            return new Result<>(true,"新增成功");
        }else{
            return new Result<>(false,"新增失败");
        }
    }

    /**
     * 更新测试数据
     */
    @PutMapping("{id}")
    public Result<String> update(@PathVariable("id") Integer id,@RequestBody TestTable testTable){

        int count = testTableDao.update(testTable);

        if(count > 0){
            return new Result<>(true,"更新成功");
        }else{
            return new Result<>(false,"更新失败");
        }
    }

    /**
     * 删除测试数据
     */
    @DeleteMapping("{id}")
    public Result<String> deleteById(@PathVariable("id") Integer id){

        int count = testTableDao.deleteById(id);

        if(count > 0){
            return new Result<>(true,"删除成功");
        }else{
            return new Result<>(false,"删除失败");
        }
    }

    /**
     * 根据 ID 查询测试数据
     */
    @GetMapping("{id}")
    public Result<TestTable> getById(@PathVariable("id") Integer id){

        TestTable testTable = testTableDao.getById(id);

        if(testTable!= null){
            return new Result<>(true,"查询成功",testTable);
        }else{
            return new Result<>(false,"查询失败");
        }
    }
}
```

# 8.测试 CRUD 操作
运行项目后，通过浏览器或者工具（如Postman）发送各种 HTTP 请求调用 RESTful API ，查看数据库是否生效，如：

获取所有测试数据：

GET http://localhost:8080/testTables

新增测试数据：

POST http://localhost:8080/testTables body:"{"title":"hello"}"

更新测试数据：

PUT http://localhost:8080/testTables/1 body:"{"title":"world"}"

删除测试数据：

DELETE http://localhost:8080/testTables/1

根据 ID 查询测试数据：

GET http://localhost:8080/testTables/1

等等。。。