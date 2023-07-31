
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　 MyBatis 是mybatis框架的简称，是一个优秀的持久层框架。它支持定制化sql、存储过程以及高级映射。 MyBatis 避免了几乎所有的JDBC代码和参数处理，使开发人员只需要关注业务逻辑即可快速完成任务。 Mybatis 可以通过简单的 XML 或注解来配置和映射原始类型、接口及 POJO（Plain Old Java Object，普通的 Java 对象）为数据库中的记录。 MyBatis 将 SQL 语句加载到内存中，通过绑定变量输入到 SQL 中并执行，最后将结果映射为自定义对象并返回。

         # 2.背景介绍
         ## 2.1 为什么要使用 MyBatis？
         　　2007年，Sun公司的JavaEE项目MyBatis（又译作mybatis，读音：[ma'a] [jeep]），最初被当做ORM（Object Relational Mapping，对象-关系映射）工具。为什么要重新造轮子呢？因为ORM虽然简单易用，但实际上存在很多不足之处，比如繁琐的SQL编写、对数据库性能的影响、缓存一致性问题等等。所以后来Sun公司在 MyBatis 的基础上，开发出 MyBatis-Spring、iBatis等更高级更强大的工具。而 MyBatis 社区也迅速发展起来，成为Java开发领域里的一个重要利器。

         ## 2.2 MyBatis 的优点
         1. 简单易用：由于 MyBatis 使用 XML 或注解来配置映射关系，所以配置简单，学习曲线平滑。即使没有多少 MyBatis 经验的同学也可以轻松上手。
         2. 可控性高： MyBatis 提供详细的日志系统，可以定位到底出了哪些问题。还提供方便的查询缓存机制，可以提升查询效率。
         3. SQL支持能力强： MyBatis 支持全面的动态 SQL 和SQL片段，支持对关系型数据源、海量数据进行分页、排序等操作。
         4. ORM特性： MyBatis 可以很好的与 Hibernate 框架等进行整合，提供ORM的特性。

         ## 2.3 MyBatis 的缺陷
         1. 存在反直觉的语法错误： MyBatis 在 XML 配置文件中，存在一种自我感觉良好但是并非编译时的错误检查机制。即便出现语法错误，MyBatis 也不会立刻抛出异常，而是在运行时抛出异常。导致定位问题困难，并且调试也比较麻烦。
         2. 无法直接获取查询结果对象： MyBatis 对复杂查询的结果集处理能力有限。只能获取一条或者多条记录，对于某些统计查询、聚合函数等功能无法直接返回统计值或者聚合后的结果。
         3. 查询缓存机制无效率：由于 MyBatis 中的缓存机制只能针对某一个mapper类进行缓存，并且该类中的SQL语句只能固定不变，不能够灵活地根据查询条件进行缓存。所以 MyBatis 中的查询缓存一般适用于单表查询，不适合进行跨表关联查询的情况。

        # 3.基本概念术语说明
        ## 3.1 Mapper
        mapper 是 MyBatis 组件中非常重要的一部分。它负责把用户的请求命令转换成具体的数据查询语句，然后再通过 JDBC 将数据从数据库取出。Mapper 本质上就是定义了一系列的 SQL 命令。我们可以通过 XML 文件或注解的方式来定义 MyBatis 的 mapper。

        ### 3.1.1 XML 文件方式
        ```xml
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
        
        <mapper namespace="com.github.houbb.mybatis.learn.mapper.UserMapper">
            <!-- 查询用户列表 -->
            <select id="listUsers" resultType="com.github.houbb.mybatis.learn.model.User">
                select * from users where username like #{username} or email like #{email}
            </select>
            
            <!-- 插入用户信息 -->
            <insert id="addUser" parameterType="com.github.houbb.mybatis.learn.model.User">
                insert into users (id, username, email) values(#{id}, #{username}, #{email})
            </insert>
        </mapper>
        ```

        ### 3.1.2 Annotaion 方式
        ```java
        package com.github.houbb.mybatis.learn.mapper;
    
        import com.github.houbb.mybatis.learn.model.User;
        import org.apache.ibatis.annotations.*;
        
        @Mapper
        public interface UserMapper {
        
            // 查询用户列表
            @Select("select * from users where username like #{username} or email like #{email}")
            List<User> listUsers(@Param("username") String username, @Param("email") String email);
    
            // 插入用户信息
            @Insert("insert into users (id, username, email) values(#{id}, #{username}, #{email})")
            void addUser(User user);
        }
        ```

    ## 3.2 ParameterType
    ParameterType 属性指定的是待传入的参数类型，也就是传递给 Mapper 方法的参数类型。如果不指定则 MyBatis 会默认匹配方法签名的参数类型。例如：

    ```java
    int deleteUserById(int userId);
    ```
    
    参数类型是 Integer，所以我们可以在 XML 中为此方法指定 `parameterType`，如下所示：
    
    ```xml
    <delete id="deleteUserById">
        delete from users where id = #{userId}
    </delete>
    ```

    ## 3.3 ResultType
    ResultType 属性指定的是查询结果的类型。在 MyBatis 中通常会配合 `<select>` 标签一起使用。其作用是将查询到的结果集封装成指定类型的 Java 对象。例如：
    
    ```xml
    <select id="getUserById" parameterType="int" resultType="com.github.houbb.mybatis.learn.model.User">
        select * from users where id = #{id}
    </select>
    ```
    
    指定了查询结果的类型为 `User`，查询的结果集会被自动映射为 `User` 对象。

    ## 3.4 Language driver
        Language Driver 是 MyBatis 的中间件模块，负责解析 XML 配置文件或注解，生成具体的 SQL 执行计划。Language Driver 接口中包含方法 accept() ，该方法返回 boolean 类型，判断是否接受对应的配置文件。accept() 返回 true 时，表示该配置文件可以被 Language Driver 处理；否则就会跳过该配置文件，继续遍历下一个配置文件。

        MyBatis 官方提供了两种语言驱动实现：
            1. XML Language Driver ：对应 XML 配置文件，继承于 BaseXMLConfigBuilder；
            2. Annotation Language Driver ：对应注解方式的 mapper，继承于 BaseAnnotationHandler 。

        如果希望自定义自己的语言驱动，那么就需要继承相应的基类，然后实现 accept() 方法和 createSqlSource() 方法。

        比如新建一个自定义语言驱动，命名为 MyCustomLanguageDriver，我们可以如下定义：

            class MyCustomLanguageDriver extends XMLLanguageDriver {
                private static final Set<Class<? extends TypeHandler>> DEFAULT_TYPE_HANDLER_SET;

                static {
                    try {
                        Class<?> clazz = Resources.classForName("org.apache.ibatis.type.UnknownTypeHandler");
                        DEFAULT_TYPE_HANDLER_SET = Collections.<Class<? extends TypeHandler>>singleton(clazz);
                    } catch (ClassNotFoundException e) {
                        throw new RuntimeException("Error creating default type handlers", e);
                    }
                }

                /**
                 * 判断是否接受指定的配置文件
                 */
                @Override
                public boolean accept(String fileName) {
                    return fileName!= null && fileName.endsWith(".mycustom");
                }

                /**
                 * 创建 SqlSource 对象
                 */
                @Override
                public SqlSource createSqlSource(Configuration configuration, XNode script) throws Exception {
                    CustomScriptBuilder customScriptBuilder = new CustomScriptBuilder();
                    Properties properties = loadCustomProperties();
                    customScriptBuilder.setProperties(properties);
                    return customScriptBuilder.parseScriptNode(script);
                }
                
                /**
                 * 加载自定义属性
                 */
                private Properties loadCustomProperties() {
                    Properties props = new Properties();
                    InputStream is = Resources.getResourceAsStream("config/mycustom.properties");
                    if (is == null) {
                        System.err.println("[WARN] Cannot find mycustom.properties file.");
                    } else {
                        try {
                            props.load(is);
                        } finally {
                            is.close();
                        }
                    }

                    return props;
                }
            }

            在配置文件中增加以下内容，指定用这个新的语言驱动：

                <settings>
                  <setting name="defaultScriptingLanguage" value="mycustom"/>
                </settings>

