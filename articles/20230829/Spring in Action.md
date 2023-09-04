
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Spring是一个开源框架，提供了现代化应用开发所需的一系列功能。其中的IoC（Inversion of Control）控制反转，面向切面编程（AOP）等理念在实际项目中大量使用，也成为企业级开发的必备技能。通过阅读Spring相关的书籍、学习教程、官方文档和相关博文，可以帮助读者快速上手Spring开发，成为一个具有“生动感知”能力的人。本文将从Spring框架整体架构及模块特性入手，介绍其基础概念、主要组件及应用场景，并结合Spring Boot快速入门，展示如何构建一个简单的RESTful服务。最后，还会对Spring未来的发展方向进行展望，介绍可能出现的挑战与突破点。

2.文章主题与范围
Spring是一款优秀的Java开发框架，由Pivotal公司创造并推广开来。本文的主题是Spring框架，希望通过研究Spring框架的设计理念、功能特性及应用场景，能够帮助读者在工作、学习中更好地掌握Spring的使用技巧。文章的范围包含以下六个部分：
- Spring 框架整体架构介绍；
- Spring Core 模块介绍；
- Spring Context 模块介绍；
- Spring AOP 模块介绍；
- Spring MVC 模块介绍；
- Spring Data Access 模块介绍；
- Spring Boot 的快速入门。

3.目录结构
```
Spring in Action/
  README.md                     //文章介绍
  part_I                        //第一部分介绍Spring框架
    introduction.md              //第一章 导读
    concept_and_terminology.md   //第二章 基本概念和术语
    architecture_and_modules.md  //第三章 Spring框架架构及模块介绍
  part_II                       //第二部分 Spring Core 模块
    core_introduction.md          //第四章 Spring Core 介绍
    bean_management.md           //第五章 Bean管理
    spring_configuration.md      //第六章 Spring配置
    ioc_container.md             //第七章 IoC容器
    resource_loading.md          //第八章 资源加载
    data_binding.md              //第九章 数据绑定
  part_III                      //第三部分 Spring Context 模块
    context_introduction.md       //第十章 Spring Context 介绍
    application_context.md       //第十一章 ApplicationContext 和 WebApplicationContext
    message_source.md            //第十二章 MessageSource
    internationalization.md      //第十三章 国际化支持
    validation.md                //第十四章 验证和数据绑定
  part_IV                       //第四部分 Spring AOP 模块
    aop_introduction.md           //第十五章 Spring AOP介绍
    aspects.md                   //第十六章 Aspects
    weaving.md                   //第十七章 Weaving
    pointcuts.md                 //第十八章 Pointcuts
  part_V                        //第五部分 Spring MVC 模块
    mvc_introduction.md           //第十九章 Spring MVC介绍
    model_view_controller.md     //第二十章 Model-View-Controller
    request_mapping.md           //第二十一章 请求映射
    views.md                     //第二十二章 Views
    converter_and_formatter.md    //第二十三章 Converter 和 Formatter
    filter_interceptor.md        //第二十四章 Filter 和 Interceptor
    exception_handling.md        //第二十五章 Exception Handling
    multipart_file_upload.md     //第二十六章 文件上传
  part_VI                       //第六部分 Spring Data Access 模块
    dao_introduction.md           //第二十七章 DAO 层介绍
    jpa.md                       //第二十八章 JPA 支持
    mongo_db.md                  //第二十九章 MongoDB 支持
    jdbc.md                      //第三十章 JDBC 支持
    hibernate.md                 //第三十一章 Hibernate 支持
    mybatis.md                   //第三十二章 MyBatis 支持
  part_VII                      //第七部分 Spring Boot 的快速入门
    quickstart.md                //第三十三章 Spring Boot 快速入门
    autoconfig_classes.md        //第三十四章 @Configuration 类自动化配置
    runners.md                   //第三十五章 SpringApplication Runners
    logging.md                   //第三十六章 Logging
    actuators.md                 //第三十七章 Actuators
    externalized_configuration.md//第三十八章 Externalized Configuration
```