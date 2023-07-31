
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　 MyBatis 是一款优秀的持久层框架，它支持定制化SQL、存储过程以及高级映射。 MyBatis 可以直接封装ulates SQL 语句，或者通过 XML 标签来配置映射关系。 MyBatis 也是 JavaEE 中最流行的 ORM 框架之一。

　　MyBatis 的官方网站是 www.mybatis.org ，它的下载地址是 http://www.mybatis.org/mybatis-3/download.html 。本教程将从 MyBatis 的基本用法开始，逐步掌握 MyBatis 的核心技术和特性，并最终演示如何基于 MyBatis 开发完整项目。

　　# 2.基本概念术语说明
       ## 2.1. MyBatis 的特点
       　　　　MyBatis 是一款优秀的持久层框架，它支持定制化 SQL、存储过程以及高级映射。 MyBatis 可以直接封装ulates SQL 语句，或者通过 XML 标签来配置映射关系。 MyBatis 也是 JavaEE 中最流行的 ORM 框架之一。
        
          1）简化数据库访问：MyBatis 将所有的数据库操作都包装成简单的配置和接口，使得应用只需要关注业务逻辑，而不需要花费精力去处理诸如连接管理、事务管理等繁琐的低级细节。
           
          2）ORM（Object Relational Mapping，对象-关系映射），简化了数据持久化编程。由于 MyBatis 使用配置文件来描述查询结果和映射，所以使用者可以很方便地完成对数据的 CRUD 操作。
           
          3）灵活性： MyBatis 提供了丰富的自定义插件机制，用户可以通过实现接口来完成各种各样的功能扩展。
           
          4）提升测试效率： MyBatis 提供了一个简单易用的日志模块，方便开发人员定位错误和调试 MyBatis 应用。
          
          ## 2.2. Mybatis 的术语表

          | **概念**        | **解释**     | 
          |:-------------|:------------|
          |SQL Mapper      |    对象关系映射器(Object-Relational Mapping)的一种实现，用于实现面向对象的编程语言中的关系数据库(RDBMS)的持久化。SQL Mapper 把数据库中的表结构映射成为内存中的对象，并由此建立起一个面向对象的数据库系统。|
          | MyBatis       |  MyBatis 是一款优秀的开源持久层框架，它支持定制化 SQL、存储过程以及高级映射。 MyBatis 可以直接映射 SQL 语句，也可以通过 XML 配置文件来映射关系。 MyBatis 在 XML 配置文件中提供了 ORM 映射能力。   |          
          |SqlSessionFactoryBuilder |  MyBatis 中的 SqlSessionFactoryBuilder 是创建 SqlSession 的工厂类。     |                
          |SqlSessionFactory |  MyBatis 中的 SqlSessionFactory 是一个工厂类，用来创建 SqlSession 对象。     |  
          |SqlSession    |  MyBatis 中的 SqlSession 代表和数据库的一次会话，用 SqlSession 可以执行增删改查以及参数映射。     |               
          |Mapper        |  MyBatis 中的 Mapper 相当于 MyBatis 的 Dao，负责crud操作，SqlSession 通过 Mapper 来操作数据库。     |                  
          |CRUD          | create、read、update、delete 的缩写，分别表示创建、读取、修改、删除。|                   
          |Entity      |     表示业务对象，可以认为是数据库中的一条记录。|              
          |Repository    |    Repository 是一个接口，它定义了一组用于管理 Entity 的方法。Repository 的目的是为了提供一层抽象，屏蔽底层的数据访问技术（比如 Hibernate、JDBC）的复杂性。在实际开发中可以使用 Spring Data JPA 或 MyBatis Plus 来实现 Repository。   |             
          |Plugin        |    插件是 MyBatis 的拓展点，通过实现 Plugin 接口，可以增加 MyBatis 的功能。比如分页插件 PageHelper 可以非常容易地帮助我们完成分页操作。   |            
          |PageHelper    |    PageHelper 是 MyBatis 的分页插件，它可以在 SQL 执行前或执行后进行条件过滤和排序，从而实现分页功能。   |            
          
          
        ## 2.3. 设计模式

         ### 2.3.1. 代理模式（Proxy Pattern）

         　　代理模式是 Structural Patterns 中的一类设计模式，它为其他对象提供一种代理以控制对这个对象的访问。代理模式最主要的作用就是在不改变原始对象的前提下，提供额外的功能。代理模式涉及到以下角色：

          - Subject（主题）：在 ProxyPattern 中，Subject 是一个抽象角色，表示当前要代理的对象；
          - RealSubject（真实主题）：RealSubject 是被代理的对象，是 Proxy 和 Subject 的真正逻辑链接；
          - Proxy（代理）：Proxy 是 Subject 的一个装饰者，负责包装真实主题，并提供额外的功能；
          - Client（客户端）：Client 是通过 Subject 来间接调用 RealSubject 的功能。

         ### 2.3.2. 适配器模式（Adapter Pattern）

         　　适配器模式是 Structural Patterns 中的一类设计模式，它用来把一个类的接口转换成客户希望的另一个接口。这种类型的设计模式属于结构型模式，它包括两个或多个现有的类，其接口不兼容，因此需要使用一个转换器类来适配它们之间的接口。这样，原本由于接口不兼容而不能一起工作的那些类就可以一起工作。

          - Target（目标）：接口目标，定义客户所期待的接口；
          - Adaptee（适配者）：被适配的对象，它定义了一个已经存在的接口，但是接口与客户期待的接口不同；
          - Adapter（适配器）：Adapter 用于创建一个符合 Target 接口的对象，它是 Adaptee 和 Target 的组合，Adaptee 委托给 Adapter 完成适配任务；
          - Client（客户端）：客户通过 Target 来调用新接口，无需知道被适配对象的具体类型。

         ### 2.3.3. 模板方法模式（Template Method Pattern）

         　　模板方法模式是 Behavioral Patterns 中的一类设计模式，它定义一个操作中的算法骨架，而将一些步骤延迟到子类中。模板方法使得子类可以不改变一个算法的结构即可重定义该算法的某些特定步骤。

          - AbstractClass（抽象类）：它定义了一个算法的骨架，按顺序调用所有步骤；
          - ConcreteClass（具体类）：它实现父类的抽象方法，完成抽象类中的空缺方法，并可选择重写某些方法来实现其它的功能；
          - Client（客户端）：使用具体子类时，只需指定父类，便可完成整个算法流程。

         ### 2.3.4. 策略模式（Strategy Pattern）

         　　策略模式是 Behavioral Patterns 中的一类设计模式，它定义了算法家族，分别封装起来，让他们之间可以互相替换，此模式让算法的变化，不会影响到使用算法的客户代码。

          - Strategy（策略）：定义一个算法接口，封装某种行为，例如排序、查找和加密算法等；
          - Context（上下文）：持有一个 Strategy 的引用，在运行时根据不同的策略进行不同算法的调用；
          - Client（客户端）：决定使用哪个策略，并交给上下文去执行。

