
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　MyBatis 是一款优秀的持久层框架，它支持定制化 SQL、存储过程以及高级映射。 MyBatis 的一个重要特点就是将接口和 xml 文件分离，使得 Mybatis 具备良好的灵活性、可拓展性和移植性。
         　　MyBatis-Plus（简称 MP）是一个 MyBatis 的增强工具，在 MyBatis 的基础上只做增强不做改变，弥补了 MyBatis 在一些地方的不足。
         　　MyBatis-PageHelper 是 MyBatis 中非常受欢迎的分页插件，通过它可以很方便地对结果集进行分页。但是 MyBatis-PageHelper 有两个缺陷，一是对所有 select 方法都有效果，而不是针对单独的一个方法，另一个是无法指定数据库方言。
         　　
         　　而 MyBatis-Spring 和 MyBatis-Spring-Boot 则是基于 MyBatis 和 Spring 的快速开发框架，开发者可以无缝结合 MyBatis 使用 Spring 的各种特性，比如 Spring Boot 的自动配置等功能。
         　　MyBatis-Generator 可以根据表结构生成 MyBatis 配置文件，解决了繁琐的数据交互配置。
         　　MyBatis 通用 Mapper 框架 (Common Mapper Framework) 则是 MyBatis 官方提供的一套全面、规范、易用的 ORM 框架。
         # 2.背景介绍
         ## 2.1 为什么要使用 MyBatis ？
         　　 MyBatis 是 Java 世界中目前最流行的持久层框架，相比 Hibernate、JPA 等其它框架，MyBatis 更加简单、灵活，学习成本也更低。它具有如下主要优点：
         
         　　(1)SQL 独立性: MyBatis 把自己的 SQL 代码封装起来，从而使得开发人员不需要直接编写 SQL 语句，这样就可以提高开发效率；
         
         　　(2)XML 配置灵活性：由于 MyBatis 是完全基于 XML 的配置文件，因此 MyBatis 的 XML 配置文件相对于 Hibernate、JPA 来说，更加灵活，这在复杂业务中尤其显著；
         
         　　(3)对象关系映射： MyBatis 使用对象关系映射，使得数据库中的记录和现实世界中的对象相互对应，避免了繁琐的 SQL 操作，最终达到简洁高效的目的。
         ## 2.2 MyBatis 和 MyBatis-Plus 的区别
         ### 2.2.1 MyBatis 版本
         #### 2.2.1.1 MyBatis 
         　　MyBatis 是 MyBatis 团队开发的一款优秀的持久层框架。它的特色是简单易用、高性能、ORM 映射能力强、支持自定义 SQL。
         　　但是 MyBatis 只支持 MyBatis 3 版本，后续版本虽然新增了新的功能，但并没有给出一个稳定的版本号。而且 MyBatis 团队也是推出新版本的计划，但实际上很少更新维护 MyBatis 3 版本。
         #### 2.2.1.2 MyBatis-Plus 
         　　MyBatis-Plus （简称 MP）是 MyBatis 的增强工具，已经集成到了 MyBatis 之中。它提供了更多的 API 支持、便捷的 CRUD 操作、支持自定义 SQL 及更多。
         　　同时 MyBatis-Plus 提供了多种分页方式，相比较于 MyBatis 本身的分页插件，它更加完善，且支持多种数据库。另外 MyBatis-Plus 比 MyBatis 的其它分页插件更轻量，不会引入额外的 jar 包。
         　　并且 MyBatis-Plus 可以让用户灵活地配置全局的规则，比如驼峰转下划线、下划线转驼峰、自动填充、自动去除空值等功能，这些都是 MyBatis 没有内置的。
         　　相比 MyBatis 本身，MyBatis-Plus 有更大的生态圈，可以帮助用户解决各种不同场景下的需求，降低用户的学习成本。
         ### 2.2.2 框架的设计原则
         　　MyBatis-Plus 的设计原则是能够按照 MyBatis 本身的思想进行设计，既保持简单、易用性、可扩展性，又符合工程实践中应用的最佳实践。
         　　比如 MyBatis-Plus 对分页插件进行了改进，提供多种分页方式，还统一了返回值类型，更加适合工程实践中的使用。
         　　此外，MyBatis-Plus 还采用动态代理的方式，进一步提升性能，不过对于复杂查询场景可能还是存在一定的局限性。
         　　MyBatis-Plus 对 SQL 执行流程进行了一定的优化，增加了缓存机制，可以有效防止 SQL 注入攻击，并减少了重复执行的 SQL 查询，有效提升系统的性能。
         ### 2.2.3 用途上的差异
         　　MyBatis 是一个优秀的 ORM 框架，它可以完成 SQL 和 POJO 对象之间的转换工作。它可以通过简单的配置，将数据库中的数据映射到 Java 语言中对应的实体类中。
         　　MyBatis-Plus 是一个增强框架，它在 MyBatis 的基础上提供了更多的功能，如分页插件、自定义 SQL、自动填充、乐观锁等。不过 MyBatis-Plus 仅限于 MyBatis 作为持久层框架的使用场景。
         　　相比 MyBatis-Plus ， MyBatis 本身也可以实现类似的功能，但是 MyBatis 的配置文件过于繁琐，不易于管理。