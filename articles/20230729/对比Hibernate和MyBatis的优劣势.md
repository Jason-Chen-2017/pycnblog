
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　Hibernate 是 Java 语言中的一个ORM（Object Relational Mapping）框架，它的主要作用是在Java编程中将关系型数据库映射到对象模型上。Hibernate 使用一种名为 Hibernate Session 的对象来代表数据库会话，并通过 DAO（Data Access Object）模式与业务逻辑层进行交互。Hibernate 支持多种映射方式，包括基于 XML 文件、基于 annotations 的配置、基于自定义类的映射方式等。但是 Hibernate 有很多缺点，比如 ORM 框架过于庞大，学习成本高，性能不佳等。相比之下，MyBatis 是另一款著名的 ORM 框架，它的优点在于它是半自动化框架，不会对你的代码产生任何侵入性影响；同时 MyBatis 更加灵活，可以实现更复杂的映射关系。而 MyBatis 也有自己的一些缺点，比如 MyBatis 中缺乏事务管理功能，对于复杂查询操作可能会出现性能问题等。
         　　因此，从易用性和功能性角度出发，我们应该选择 MyBatis 来实现后台开发。但是，Hibernate 也可以用来实现后台开发，这是因为 MyBatis 可以很好地整合 Spring 框架，使得集成 Hibernate 时不需要修改代码；并且 MyBatis 在小数据量时，效率要优于 Hibernate；另外，Hibernate 提供了更丰富的功能支持，如缓存、动态加载等，这些都是 MyBatis 不具备的。总结来说， MyBatis 比 Hibernate 更适合小数据量场景下的快速开发，而 Hibernate 更适合大数据量或复杂场景下的项目部署。 
         　　本文作者：熊博士 
         # 2.Hibernate的基本概念和术语
         　　Hibernate就是一个Java框架，专门用于Java应用程序与关系数据库之间的数据持久化。这里需要重点介绍一下Hibernate的基本概念和术语，以便后面更好地理解Hibernate。
            　　1．Hibernate实体（Entity）：就是Java类与表之间的对应关系。在Hibernate中，每个Java类都是一个实体，每个实体对应的表中都有一个唯一标识符，即主键。 Hibernate实体是一个类，通常情况下，它们都继承Hibernate实体基类。实体基类定义了实体所需的基本属性，例如名称、创建日期、最后更新时间等。
            　　2．Hibernate映射文件（Mapping file）：映射文件就是一个XML文件，里面定义了实体和表之间的对应关系。映射文件可帮助Hibernate识别实体类，关联关系，主键生成策略等信息。映射文件的扩展名一般是*.hbm.xml。
            　　3．Hibernate配置文件（Configuration File）：Hibernate配置文件是Hibernate运行时的基础配置，一般存放在hibernate.cfg.xml文件中。该配置文件中设定了JDBC连接参数、Hibernate的设置参数等。
            　　4．Hibernate会话（Session）：当打开Hibernate会话时，Hibernate就会建立与数据库的连接，为后续查询做准备工作。 Hibernate会话是一个事务性上下文环境，它负责对持久化对象的CRUD操作，并维护其生命周期。 Hibernate 会话具有显式开启和关闭的过程。
            　　5．Hibernate Query Language (HQL)：Hibernate Query Language（HQL）是Hibernate的对象查询语言，可以用于执行各种复杂的查询，支持分页、排序等高级特性。 HQL类似SQL语句，但与实际SQL语句存在较大的差别。
            　　6．Hibernate O/R Mapping（ORM）：Hibernate O/R Mapping（ORM）是指Java对象与关系数据库之间的一种映射。ORM工具利用JavaBean或者POJO转换为关系数据库中的记录，然后提供简单的接口来访问数据。ORM框架可以自动处理诸如数据保存、检索、更改和删除等操作。
            　　7．Hibernate Metadata API：Hibernate Metadata API允许用户获取到当前Hibernate会话的元数据信息，如表结构，列数据类型，索引信息等。
            　　8．Hibernate事务（Transaction）：Hibernate事务处理提供了一致的事务控制机制，包括自动提交和手动回滚两种方式。Hibernate事务可以在应用服务器、EJB容器、Web框架等不同环境下实现，保证数据的一致性。
            　　9．Hibernate优化（Optimization）：Hibernate有多种优化方案，如延迟加载、批量加载、查询缓存等，可提升查询速度。
            　　10．Hibernate迁移（Migration）：Hibernate提供了一个叫做Liquibase的开源项目，可用于数据库结构的版本管理。Liquibase会跟踪数据库结构的变化并在必要的时候进行变更操作。
            　　11．Hibernate持久化框架（Persistence Framework）：Hibernate持久化框架由 Hibernate Entity Manager、Hibernate Caching、Hibernate Search、Hibernate Validator等构成。
            　　12．Hibernate框架配置（Framework Configuration）：Hibernate框架配置主要包括几个方面：配置Hibernate实体类；配置Hibernate映射文件；配置Hibernate数据库链接信息等。
            　　13．Hibernate依赖注入（Dependency Injection）：Hibernate依赖注入用于把资源的管理交给框架，而不是自己手动管理。它可以通过注解或XML配置文件的方式来完成配置。
            　　14．Hibernate对象状态管理（Object State Management）：Hibernate对象状态管理决定了Hibernate如何跟踪对象实例的状态，并提供相应的方法来检测状态是否发生改变。 Hibernate提供了两种状态检测策略：Lazy Loading和Dirty Checking。
            　　15．Hibernate代理（Proxy）：Hibernate代理是Hibernate用来隐藏底层持久化实现细节的一种技术。 Hibernate根据对象的状态创建对应的代理，并根据需要在幕后完成持久化操作。
            　　16．Hibernate集合映射（Collection Mapping）：Hibernate集合映射是Hibernate用于处理实体中包含其他实体的情况。 Hibernate支持以下几种集合映射方式：一对多、多对一、一对一、多对多。
            　　17．Hibernate查询语言（Query Language）：Hibernate查询语言（HQL）是Hibernate的对象查询语言，可以用于执行各种复杂的查询，支持分页、排序等高级特性。 HQL类似SQL语句，但与实际SQL语句存在较大的差别。
            　　18．Hibernate缓存（Caching）：Hibernate缓存是Hibernate用来减少对数据库的频繁访问的一种机制。它通过缓存已经加载过的实体对象，避免重复查询相同的数据。 Hibernate提供了三种缓存级别：NONE、SESSION和FULL。
         # 3.Hibernate映射文件的具体语法解析
         　　Hibernate的映射文件分为两种：XML映射文件和Annotation映射文件。其中XML映射文件就是传统意义上的映射文件，定义了实体类和表之间的映射关系；而Annotation映射文件则完全不依赖XML文件，而是采用注解的方式定义实体类和表的映射关系。
            　　1．XML映射文件
          　　Hibernate的XML映射文件以*.hbm.xml为扩展名，其根元素为hibernate-mapping。其主要元素如下：
            <hibernate-mapping>
                <!-- 配置实体 -->
                <class name="com.example.User">
                    <!-- 配置主键 -->
                    <id name="userId" column="user_id"/>
                    <!-- 配置属性 -->
                    <property name="username" type="string" column="username"/>
                    <property name="password" type="string" column="password"/>
                    <!-- 配置一对多关系 -->
                    <set name="orders">
                        <key column="order_id"/>
                        <one-to-many class="com.example.OrderItem">
                            <column name="item_id" unique="true"/>
                            <join table="order_items"/>
                            <lazy load="extra"/>
                        </one-to-many>
                    </set>
                    <!-- 配置多对一关系 -->
                    <many-to-one property="company" class="com.example.Company" lazy="false">
                        <column name="company_id" />
                    </many-to-one>
                    <!-- 配置多对多关系 -->
                    <bag name="tags">
                        <key column="tag_name"/>
                        <many-to-many inverse="false" lazy="extra">
                            <class name="com.example.Tag" column="tag_id"/>
                        </many-to-many>
                    </bag>
                </class>
                <!-- 配置外键约束 -->
                <collection-table name="employees" >
                    <join-columns>
                        <JoinColumn name="department_id" referencedColumnName="dept_id"/>
                    </join-columns>
                    <inverse-join-columns>
                        <JoinColumn name="employee_id" referencedColumnName="emp_id"/>
                    </inverse-join-columns>
                </collection-table>
                <class name="com.example.Employee">
                    <id name="employeeId" column="emp_id" generator="increment"></id>
                    <!-- 配置属性 -->
                    <property name="firstName" type="string" column="fname"></property>
                    <property name="lastName" type="string" column="lname"></property>
                    <!-- 配置一对多关系 -->
                    <set name="phoneNumbers">
                        <cache usage="nonstrict-read-write"/>
                        <key column="phone_type"/>
                        <one-to-many class="com.example.PhoneNumber">
                            <column name="number_id" unique="true"/>
                            <join table="employee_phones"/>
                            <cascade all="delete-orphan"/>
                            <fetch-mode select="eager"/>
                            <sort order="asc" />
                        </one-to-many>
                    </set>
                    <!-- 配置多对多关系 -->
                    <bag name="projects">
                        <key column="project_id"/>
                        <many-to-many inverse="false" fetch="select" lazy="extra">
                            <class name="com.example.Project" column="proj_id"/>
                        </many-to-many>
                    </bag>
                    <!-- 配置联合主键 -->
                    <composite-id>
                        <key-property name="departmentNumber" type="int" column="dept_num"/>
                        <key-property name="employeeNumber" type="long" column="emp_num"/>
                    </composite-id>
                </class>
            </hibernate-mapping>
         　　通过XML映射文件，我们可以轻松定义实体类和表之间的映射关系。如上述示例，我们定义了一个User实体类和多个相关实体类，包括OrderItem、Company、Tag和Phone。通过关系标签，我们可以定义实体间的各种联系，包括一对多、多对一、多对多、自引用等。除此之外，还有许多标签可用，比如property，cache，index，generator等，它们都可以为实体属性添加额外的描述。
            　　2．Annotation映射文件
          　　Hibernate Annotation映射文件也是以*.hbm.xml为扩展名，不过其格式与XML映射文件有很大不同。Annotation映射文件没有独立的hibernate-mapping标签，直接以@Entity和@Table注解定义实体类和表的映射关系。
            @Entity(name = "users")
            public class User {
                // 属性定义，包括主键，非主键，外键，引用外键等
                private Integer userId;
                private String username;
                private String password;
                
                // 一对多关系定义
                @OneToMany(mappedBy = "user", cascade = CascadeType.ALL)
                List<OrderItem> orders;
                
                // 多对一关系定义
                @ManyToOne(optional = false)
                Company company;
                
                // 多对多关系定义
                @ManyToMany(targetEntity=Tag.class)
                Set<Tag> tags;
                
                // getter setter方法省略
            }
            
            @MappedSuperclass
            public abstract class BaseDomain {
                // id属性定义，并加入标识生成策略
                @Id
                @GeneratedValue(strategy = GenerationType.AUTO)
                private Long domainId;
            }
            
            // Tag实体类，继承BaseDomain抽象类
            @Entity(name="tags")
            public class Tag extends BaseDomain{
                private String tagName;
            }

            // OrderItem实体类，定义一对多关系
            @Entity(name="order_items")
            public class OrderItem extends BaseDomain{
                private Integer itemId;
                private Double price;
                private int quantity;
                
                @OneToOne(cascade = CascadeType.ALL)
                @JoinColumn(name="order_id", nullable=false)
                private Order order;

                @ManyToOne(cascade = CascadeType.ALL)
                @JoinColumn(name="product_id", nullable=false)
                private Product product;
            
                // getter setter方法省略
            }
         　　通过Annotation映射文件，我们可以轻松定义实体类和表之间的映射关系。如上述示例，我们定义了一个User实体类和两个相关实体类：OrderItem和Tag，并分别定义了一对多、多对一、多对多的关系。其中多对多关系是通过中间表实现的，并使用@ManyToMany注解定义。同时，我们还定义了一个抽象类BaseDomain，所有实体类的id属性都继承于BaseDomain，并使用@Id和@GeneratedValue注解指定标识生成策略。
         　　Annotation映射文件虽然简单易读，但不如XML映射文件灵活，如果有特殊需求，还是推荐XML映射文件。不过，由于Annotation映射文件不依赖XML文件，所以仍然可以在任意Java环境下使用。
         # 4.MyBatis的具体语法解析
         　　MyBatis是一款半自动化的ORM框架。它的主要作用是对JDBC的操作进行封装，屏蔽掉JDBC底层复杂的API，方便开发者编写高质量的代码。MyBatis通过xml文件或注解的方式将pojo和sql语句映射起来，在启动时读取映射关系，之后只需要调用mapper接口即可调用数据库操作。下面我会详细介绍MyBatis的基本概念和术语，以及MyBatis的配置文件及具体语法。
            　　1．MyBatis实体（Entity）：就是Java类与表之间的对应关系。在MyBatis中，每个Java类都是一个实体，每个实体对应的表中都有一个唯一标识符，即主键。
            　　2．Mybatis映射文件（Mapper文件）：Mybatis映射文件就是一个xml文件，里面定义了实体和SQL语句的映射关系。Mapper文件可帮助Mybatis识别实体类，绑定SQL语句，处理查询结果等。Mapper文件的扩展名一般是*.xml。
            　　3．Mybatis配置文件（Config File）：Mybatis配置文件是Mybatis运行时的基础配置，一般存放在mybatis-config.xml文件中。该配置文件中设定了JDBC连接参数、Mybatis的设置参数等。
            　　4．Mybatis插件（Plugin）：Mybatis插件是Mybatis的拦截器，它可以拦截JDBC执行过程的请求，并根据需求作进一步的处理。Mybatis提供了很多内置的插件，比如分页插件、性能监控插件等。
            　　5．Mybatis mapper接口（Mapper Interface）：Mapper接口是一个Java接口，它声明了各个SQL语句的接口方法。每一个方法代表了数据库中的一条SELECT、INSERT、UPDATE、DELETE语句。
            　　6．Mybatis statement：statement是Mybatis运行时生成的代理对象，它包含映射语句的参数，执行SQL语句。
            　　7．Mybatis参数映射（Parameter Map）：参数映射是Mybatis处理传入的SQL参数的过程。Mybatis通过参数映射把输入的参数转化成PreparedStatement对象中的占位符。
            　　8．Mybatis结果映射（Result Map）：结果映射是Mybatis根据数据库返回的结果集，映射成Java对象实例的过程。
            　　9．Mybatis命名空间（Namespace）：命名空间是Mybatis用来定位SQL语句的前缀，可解决不同数据库之间的兼容问题。
            　　10．Mybatis动态SQL：Mybatis动态SQL是Mybatis提供的一种类似SQL语句的语法，可根据条件构建动态SQL语句。
            　　11．Mybatis缓存（Cache）：Mybatis缓存是Mybatis提供的一项查询优化机制，它可以缓冲一段时间内请求过的结果，避免反复查询相同的数据，以提升系统的性能。
            　　12．Mybatis二级缓存：Mybatis二级缓存是Mybatis提供的一项缓存机制，它是Hibernate缓存的一种替代方案。
            　　13．Mybatis代码生成器（Code Generator）：Mybatis代码生成器是Mybatis提供的一个独立的模块，它可以根据XML或注解映射文件生成Java实体类和Dao接口，以及对应xml文件。
         　　MyBatis的配置文件mybatis-config.xml的内容如下：
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE configuration PUBLIC "-//mybatis.org//DTD Config 3.0//EN"
            "http://mybatis.org/dtd/mybatis-3-config.dtd">
            <configuration>
              <environments default="development">
                <environment id="development">
                  <transactionManager type="JDBC"/>
                  <dataSource type="POOLED">
                    <property name="driver" value="${jdbc.driver}"/>
                    <property name="url" value="${jdbc.url}"/>
                    <property name="username" value="${jdbc.username}"/>
                    <property name="password" value="${jdbc.password}"/>
                  </dataSource>
                </environment>
              </environments>
              <mappers>
                <mapper resource="com/example/UserMapper.xml"/>
                <mapper resource="com/example/BlogMapper.xml"/>
              </mappers>
            </configuration>
         　　通过mybatis-config.xml，我们可以轻松定义Mybatis的环境参数，包括JDBC连接参数，Mybatis的全局参数等。myBatis-config.xml中还包括Mybatis插件列表，它可以拦截JDBC执行过程的请求，并根据需求作进一步的处理。Mybatis提供了很多内置的插件，比如分页插件、性能监控插件等。
         　　 MyBatis的XML映射文件主要由四个部分组成：
          　　# mapper标签
          　　# sql节点
          　　# resultMap节点
          　　# parameterMap节点
        （1）mapper标签
            　　mapper标签是Mybatis的根标签，标记着该XML文件是一个Mybatis的映射文件。每个xml文件只能有一个mapper标签。在mapper标签内部包含四个子标签：cache、parameterMap、resultMap和sql。
         　　（2）cache标签
            　　cache标签用于配置Mybatis的缓存。它有三个属性：flushInterval flushOnCommit size。flushInterval表示刷新间隔，单位是毫秒；flushOnCommit表示是否在提交事务时刷新缓存；size表示缓存大小，单位是字节。
         　　（3）parameterMap标签
            　　parameterMap标签用于配置输入参数的映射关系。它有三个属性：id、type、extend。id表示唯一标识符，type表示输入参数的类型，extend表示是否扩展父类。
         　　（4）resultMap标签
            　　resultMap标签用于配置输出结果的映射关系。它有五个属性：id、type、extends、autoMapping和constructor。id表示唯一标识符，type表示输出结果的类型，extends表示父类，autoMapping表示是否自动映射结果，constructor表示构造函数。
            　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　（5）sql节点
          　　sql节点用于配置数据库操作命令，它有四个属性：id、useGeneratedKeys、reultType、parameterType。id表示唯一标识符，useGeneratedKeys表示是否生成自增长的ID，resultType表示输出结果的类型，parameterType表示输入参数的类型。
         　　MyBatis的SQL语句主要由select、insert、update、delete关键字组成，分别用于查询、插入、修改、删除数据。SQL语句支持占位符，它可以将参数值插入到SQL语句的特定位置，也可以用于防止SQL注入攻击。
         　　# 5.Hibernate VS MyBatis的优劣势分析
         　　先抛开框架的名字，对两者进行比较。首先，Hibernate的优势是功能完整，提供了更丰富的功能支持，如缓存、动态加载等；而MyBatis的优势则在易用性和灵活性上。Hibernate可以整合Spring框架，使得集成Hibernate时不需要修改代码；MyBatis提供了一套简单的XML或注解来配置映射关系；MyBatis在小数据量时，效率要优于Hibernate。至于具体原因，笔者认为主要还是Hibernate被设计为ORM框架，并且提供了更多的功能，而MyBatis则倾向于只关注SQL语句的执行。除此之外，Hibernate有更好的性能，且对版本兼容性较好，因此是更好的选择。
        下面结合Hibernate与MyBatis的映射配置文件和查询操作进行比较。
            　　假设有两个实体类：User和OrderItem。它们之间的关系如下图所示。
            |-----------|---------|----------|--------------|---------------------|
            |    user   |     --  |          |              |        order_item   |
            |-------------|---------|----------|---------------|----------------------|
            | userId     | PK      | FK       | owner_id      | orderId             |
            | username   |         |          |               | itemName             |
            | password   |         |          |               | description          |
            | company_id |         | FK       |               | itemId               |
            | tag_names  |         |          |               | price                |
            | created_at |         |          |               | quantity             |
            | updated_at |         |          |               |                       |
            |---------------------------------------+-----------------------|
            |<----orm-----><---restful----->|<---------web------------>|
         　　Hibernate的映射配置文件：
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE hibernate-mapping SYSTEM 
                "http://www.hibernate.org/dtd/hibernate-mapping-3.0.dtd">
            <hibernate-mapping package="cn.itcast.entity">
              <class name="User" table="t_user">
                <id name="userId" column="user_id">
                  <generator class="identity"/>
                </id>
                <property name="username" column="username"/>
                <property name="password" column="password"/>
                <property name="createdTime" column="create_time"/>
                <property name="updatedTime" column="update_time"/>
              </class>
              <class name="OrderItem" table="t_order_item">
                <id name="itemId" column="item_id">
                  <generator class="identity"/>
                </id>
                <property name="itemName" column="item_name"/>
                <property name="description" column="description"/>
                <property name="price" column="price"/>
                <property name="quantity" column="quantity"/>
                <property name="orderId" column="order_id"/>
              </class>
              <collection-relations>
                <bag name="user.orderItems">
                  <key column="owner_id"/>
                  <one-to-many class="OrderItem" mapped-by="user"/>
                </bag>
              </collection-relations>
            </hibernate-mapping>
         　　通过Hibernate的映射配置文件，我们可以轻松定义实体类和表之间的映射关系，其中包括主键生成策略。这里我省略了相关的一对多关系定义。MyBatis的映射配置文件如下：
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" 
                    "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
            <mapper namespace="cn.itcast.mapper">
              <resultMap id="userResultMap" type="User">
                <id column="user_id" property="userId"/>
                <result column="username" property="username"/>
                <result column="password" property="password"/>
                <result column="create_time" property="createdTime"/>
                <result column="update_time" property="updatedTime"/>
              </resultMap>
              <resultMap id="orderItemResultMap" type="OrderItem">
                <id column="item_id" property="itemId"/>
                <result column="item_name" property="itemName"/>
                <result column="description" property="description"/>
                <result column="price" property="price"/>
                <result column="quantity" property="quantity"/>
                <result column="order_id" property="orderId"/>
              </resultMap>
              <sql id="userColumns">user_id,username,password,create_time,update_time</sql>
              <sql id="orderItemColumns">item_id,item_name,description,price,quantity,order_id</sql>
              <select id="getUserById" parameterType="int" resultMap="userResultMap">
                SELECT ${userColumns} FROM t_user WHERE user_id = #{userId};
              </select>
              <select id="getOrderByUserId" parameterType="int" resultType="OrderItem">
                SELECT ${orderItemColumns} FROM t_order_item WHERE owner_id = #{userId};
              </select>
              <insert id="saveUser" parameterType="User">
                INSERT INTO t_user(${userColumns}) VALUES(${userColumns});
              </insert>
              <insert id="saveOrderItem" parameterType="OrderItem">
                INSERT INTO t_order_item(${orderItemColumns}) VALUES(${orderItemColumns});
              </insert>
              <update id="updateUser" parameterType="User">
                UPDATE t_user SET (${userColumns}) = (#{userId},#{username},#{password},#{createdTime},#{updatedTime})WHERE user_id=#{userId} ;
              </update>
              <update id="updateOrderItem" parameterType="OrderItem">
                UPDATE t_order_item SET (${orderItemColumns}) = (#{itemId},#{itemName},#{description},#{price},#{quantity},#{orderId})WHERE item_id=#{itemId} ;
              </update>
              <delete id="removeUser" parameterType="int">
                DELETE FROM t_user WHERE user_id = #{userId};
              </delete>
              <delete id="removeItem" parameterType="int">
                DELETE FROM t_order_item WHERE item_id = #{itemId};
              </delete>
            </mapper>
         　　通过MyBatis的映射配置文件，我们可以轻松定义实体类和SQL语句的映射关系，其中包括输入参数的类型、输出结果的类型、动态SQL语句等。此外，MyBatis提供了一套简单的XML来配置映射关系，并且支持SQL脚本。
         　　下面看一下查询操作：
            　　假设我们需要根据user的ID查询User信息，Hibernate的代码如下：
            Session session = getSessionFactory().openSession();
            User user = new User();
            user.setUserId(1);
            Criteria criteria = session.createCriteria(User.class);
            criteria.add(Example.create(user));
            User result = (User)criteria.uniqueResult();
            System.out.println(result);
            session.close();
         　　通过Hibernate，我们可以根据ID查询用户的信息。但是，这种查询方式非常复杂，而且效率低下。
            　　假设我们需要根据userID查询订单信息，MyBatis的代码如下：
            SqlSession sqlSession = getSqlSessionFactory().openSession();
            UserMapper userMapper = sqlSession.getMapper(UserMapper.class);
            User result = userMapper.getUserById(1);
            System.out.println(result);
            sqlSession.close();
         　　通过MyBatis，我们可以使用Mapper接口直接调用SQL语句，查询用户信息。此外，MyBatis提供简单易用的查询方法，并且支持SQL脚本，使得代码书写更直观。
            　　综上所述，Hibernate和MyBatis之间的区别在于：Hibernate是全功能的ORM框架，而MyBatis只关注SQL语句的执行，以简化开发难度，同时又保持了灵活性。

