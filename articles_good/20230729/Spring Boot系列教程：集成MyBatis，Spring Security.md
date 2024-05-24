
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         Spring Boot是一种新的开源Java开发框架，它极大地简化了基于Java的应用配置，自动装配，消息处理等流程，并提供了运行时的健康检查、外部化配置等功能，因此在实际项目开发中扮演着越来越重要的角色。本系列教程将详细介绍如何利用Spring Boot框架，整合Mybatis和Spring Security，实现对数据库的增删改查及安全管理。
         
         本篇教程主要内容如下：
         1. Spring Boot介绍
         2. MyBatis介绍
         3. Spring Security介绍
         4. 如何整合Mybatis和Spring Security
         5. 用例实战（MySQL数据库）
         6. 小结
         7. 参考文献
         
         # 2. Spring Boot介绍
         ## 2.1 Spring Boot概述
         
         Spring Boot是由Pivotal团队提供的全新框架，其设计目的是用来简化新Spring应用的初始搭建以及开发过程。该框架使用了特定的方式来进行配置，从而使开发人员不再需要定义样板化的XML文件。通过一些默认设置项和模板，开发者可以快速启动应用，并立即使用production-ready功能，如指标监控，健康检查，外部化配置等。此外，Spring Boot也对各种应用部署环境做了适配，可以打包为单个可执行jar文件，也可以作为普通的war文件部署到Servlet容器中运行。Spring Boot还支持响应式Web编程，并且开箱即用的starter依赖能够帮助开发者快速添加所需功能模块。所以，Spring Boot无疑是一个极具吸引力的框架。
         
         ## 2.2 Spring Boot特性
         
         ### 2.2.1 官宣的“快速”开发方式
         
         Spring Boot有助于快速开发新型的基于云的微服务架构中的不可或缺的一环。它为传统企业应用程序开发领域带来的长期增长提供了一种全新的方式，因为它简化了创建独立的、生产级的、基于Spring的应用程序的过程。通过减少配置时间和依赖项数量，Spring Boot使开发人员可以更快、更轻松地发布、交付和部署应用程序。
         
         通过其“约定优于配置”的理念，Spring Boot帮助开发者通过避免冗余配置来简化应用开发。只需简单的注解或者少量的代码，就可以启用某些默认设置或自动装配库。这样可以节省大量的时间用于编写业务逻辑代码。
         
         在经历过一个多月的开发后，Spring Boot已然成为开发者必备的工具。对于开发者来说，没有什么比开发速度更重要的了。Spring Boot已经得到了业界广泛认可，并在2014年度的Stack Overflow Developer Survey中排名第十，总共收集到了超过10万个问题的回答。其中包括很多关于Spring Boot的评价，反映出开发者对它的喜爱程度。
         
         ### 2.2.2 提供常用功能的模块化解决方案
         
         Spring Boot遵循SpringBoot，模块化开发和插件化扩展的理念，使用户只需要添加必要的依赖项即可开启相应的功能。它支持众多开发框架，包括Spring Framework，Hibernate，JPA，Thymeleaf，REST API，WebSocket，Testing等等。这些模块化开发特性使得Spring Boot非常易于学习和使用。
         
         使用Spring Boot可以快速实现各种常见的功能，例如：

             * 分页查询
             * 数据校验
             * RESTful接口
             * 文件上传下载
             * 消息队列
             * 缓存
             * 搜索引擎
             * 定时任务
             * 服务注册与发现
         
         更多的功能将随着社区的发展逐渐加入Spring Boot生态圈。
         
         ### 2.2.3 支持响应式Web编程
         
         Spring Boot支持响应式Web编程，允许用户选择不同的视图技术，如Thymeleaf、FreeMarker、Groovy Template等，从而构建出功能强大的响应式Web应用。同时，Spring Boot也提供了开箱即用的异步支持，开发者可以使用像Netty、Undertow这样的高性能异步框架，同时保持Servlet API兼容性。
         
         ### 2.2.4 提供可插拔的DevTools支持
         
         Spring Boot提供的DevTools支持可在开发阶段提供实时重载能力，并且可以提升开发者的工作效率。在开发过程中，DevTools会监听文件变化，自动编译、测试并部署应用。通过热加载机制，开发者可以在短时间内看到应用的最新修改效果，进一步提升开发者的工作效率。
         
         ### 2.2.5 提供完善的运行时健康检查
         
         Spring Boot自带的健康检查模块能检测到应用的各类健康状态，如内存溢出、磁盘占用、线程阻塞、上下游连接失败等，并且能够向监控系统报告故障信息。当出现异常情况时，Spring Boot会触发自动恢复机制，确保应用始终处于正常运转状态。
         
         ### 2.2.6 外部化配置
         
         Spring Boot提供了一个统一的外部化配置解决方案，使得开发者可以根据不同运行环境（本地开发、测试、预生产、生产）来指定配置文件，而不需要额外编写复杂的配置代码。Spring Boot提供的配置文件可以是YAML、Properties、INI等各种形式，甚至可以通过命令行参数或者运行时属性来覆盖特定的值。同时，Spring Boot还提供了一个实用的命令行选项解析器，开发者可以通过命令行参数配置应用程序，而不需要编写额外的代码。
         
         ### 2.2.7 有关Spring Boot的所有信息都可以通过Spring Initializr网站获得
         
         Spring Boot提供了一个便利的Web界面，让开发者可以直接从浏览器中启动生成Spring Boot工程的脚手架。除此之外，Spring Boot还提供了starter依赖管理器，可以让开发者快速引入相关的第三方库。除此之外，Spring Boot还有官方的IDE插件，方便开发者在IntelliJ IDEA或Eclipse中开发应用。Spring Boot的网站Spring.io上还有许多资源，例如Spring Boot Reference Guide、Spring Boot Docs、Spring Boot Blog等。
         
         # 3. MyBatis介绍
         
         MyBatis是一款优秀的持久层框架。它支持SQL映射、存储过程以及动态SQL语法的XML配置，并可以使用简单的Annotation或者XML的方式来灵活地将SQL语句映射成对应的POJO对象。 MyBatis 在 JDK 5.0、6.0 和 7.0 的版本中得到了广泛应用，之后也有一些企业应用，比如说京东金融、维基百科等。2011年 MyBatis 被捐赠给 Apache Software Foundation ，之后又收到 MyBatis 组的参与者（例如延平和王开忠等）。
         
         Mybatis和Spring无缝集成，Spring可以用Mybatis替代掉原有的Dao层，达到降低耦合度的作用；另外，Mybatis也支持自定义类型，比如通用的POJO类，在这种情况下，Spring Data JPA等ORM框架无法胜任。
         
         # 4. Spring Security介绍
         
         Spring Security是一个安全框架，它能够帮助开发者对应用中用户访问权限进行控制。通过集成Spring Security，开发者可以快速验证用户身份、授权访问，保护web应用免受攻击。Spring Security支持几种常用的安全模式，如表单登录、HTTP基本认证、记住我、无感知认证等。
         
         Spring Security从3.0版本开始，支持oauth2，为OAuth2.0协议的实现提供了便利。Spring Security OAuth项目为开发者提供了许多开箱即用的安全配置，包括客户端认证、授权服务器、令牌存储、资源服务器等。
         
         # 5. 如何整合Mybatis和Spring Security
         
         首先，把Spring Security和Mybatis导入项目中。然后，创建一个配置文件securityConfig.java，并配置Spring SecurityFilterChain，以及使用MybatisSecurityInterceptor配置Securiry拦截器，这个拦截器负责将Spring Security表达式注入到Mybatis语句中。
         
         配置完成后，使用@EnablemybatisSecurity注解激活Spring Secutity的相关配置，注意此注解要在Spring Security的过滤器链之前。
         
         # 6. 用例实战（MySQL数据库）
         
         在这里我们以最简单的增删改查为例，用Mybatis和Spring Security框架实现对MySQL数据库的增删改查及安全管理。
         
         **环境准备**
         
         1. 安装jdk1.8+
         2. 安装maven3.5+
         3. MySQL5.x(推荐5.7以上)
         
         **目录结构**
         
        ```text
        springboot-demo
            |—— pom.xml                //maven依赖管理
            |—— src/main/java
                |—— com
                    |—— wang
                        |—— demo
                            |—— App.java      //启动类
                            |—— controller
                                |—— LoginController.java   //登录控制器
                            |—— dao 
                                |—— UserMapper.java    //UserMapper接口
                            |—— entity
                                |—— User.java          //实体类
                            |—— service
                                |—— UserService.java  //UserService接口
                                |—— impl
                                    |—— UserServiceImpl.java   //UserServiceImpl实现类
                            |—— util
                                |—— MD5Util.java       //MD5加密工具类
            |—— src/main/resources
                |—— application.properties   //配置文件
                |—— mybatis-config.xml        //mybatis配置文件
                |—— log4j.properties         //日志配置文件
        ```
         
         **pom.xml**
         
        ```xml
        <project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                 xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
            <modelVersion>4.0.0</modelVersion>
    
            <groupId>com.wang.springboot.demo</groupId>
            <artifactId>springboot-demo</artifactId>
            <version>1.0-SNAPSHOT</version>
            <packaging>jar</packaging>
    
            <name>springboot-demo</name>
            <url>http://maven.apache.org</url>
    
            <parent>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-starter-parent</artifactId>
                <version>2.1.1.RELEASE</version>
                <relativePath/> <!-- lookup parent from repository -->
            </parent>
    
            <dependencies>
                <dependency>
                    <groupId>org.springframework.boot</groupId>
                    <artifactId>spring-boot-starter-web</artifactId>
                </dependency>
    
                <dependency>
                    <groupId>org.mybatis.spring.boot</groupId>
                    <artifactId>mybatis-spring-boot-starter</artifactId>
                    <version>2.0.1</version>
                </dependency>
                
                <dependency>
                    <groupId>mysql</groupId>
                    <artifactId>mysql-connector-java</artifactId>
                </dependency>

                <dependency>
                    <groupId>org.springframework.boot</groupId>
                    <artifactId>spring-boot-starter-security</artifactId>
                </dependency>
            </dependencies>
    
            <build>
                <plugins>
                    <plugin>
                        <groupId>org.springframework.boot</groupId>
                        <artifactId>spring-boot-maven-plugin</artifactId>
                    </plugin>
                </plugins>
            </build>
        
        </project>
        ```
         
         **application.properties**
         
        ```properties
        server.port=8081
        logging.level.root=info
        spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
        spring.datasource.url=jdbc:mysql://localhost:3306/db_name?useSSL=false&serverTimezone=GMT%2B8
        spring.datasource.username=your_username
        spring.datasource.password=<PASSWORD>_password
        security.basic.enabled=true
        security.user.name=admin
        security.user.password=123456
        ```
         
         **log4j.properties**
         
        ```properties
        ###################################################### 
        ## Default Logging Configuration File
        ###################################################### 
    
        ## Console Appender 
        log4j.appender.stdout=org.apache.log4j.ConsoleAppender
        log4j.appender.stdout.layout=org.apache.log4j.PatternLayout
        log4j.appender.stdout.layout.ConversionPattern=%d{yyyy-MM-dd HH:mm:ss} [%p] %C{1}.%M(%L) - %m%n
    
        ## Root Logger 
        log4j.rootLogger=INFO, stdout
        ```
         
         **App.java**
         
        ```java
        package com.wang.springboot.demo;

        import org.springframework.boot.SpringApplication;
        import org.springframework.boot.autoconfigure.SpringBootApplication;

        @SpringBootApplication
        public class App {
            public static void main(String[] args) throws Exception {
                SpringApplication.run(App.class,args);
            }
        }
        ```
         
         **LoginController.java**
         
        ```java
        package com.wang.springboot.demo.controller;

        import org.springframework.beans.factory.annotation.Autowired;
        import org.springframework.stereotype.Controller;
        import org.springframework.ui.ModelMap;
        import org.springframework.validation.BindingResult;
        import org.springframework.web.bind.annotation.*;
        import javax.servlet.http.HttpSession;
import javax.validation.Valid;

/**
 * Created by WangChen on 2020-04-10.
 */
@Controller
public class LoginController {

    private String username = "admin";
    private String password = "<PASSWORD>";

    @RequestMapping("/login")
    public String loginPage() {
        return "/WEB-INF/jsp/login.jsp";
    }

    /**
     * 用户登录
     */
    @RequestMapping(value = "/doLogin", method = RequestMethod.POST)
    public String doLogin(@Valid LoginForm form, BindingResult result, HttpSession session) {
        if (result.hasErrors()) {
            System.out.println("用户名或密码错误！");
            return "/WEB-INF/jsp/login.jsp";
        } else if (!this.checkPassword(form)) {
            System.out.println("用户名或密码错误！");
            return "/WEB-INF/jsp/login.jsp";
        } else {
            // 用户登录成功
            session.setAttribute("userId", this.username);
            return "redirect:/welcome";
        }
    }
    
    /**
     * 检查密码是否正确
     * @param form 登陆表单
     * @return 是否正确
     */
    private boolean checkPassword(LoginForm form) {
        return this.password.equals(form.getPassword());
    }
    
}

```

         
         **dao/UserMapper.java**
         
        ```java
        package com.wang.springboot.demo.dao;

import com.wang.springboot.demo.entity.User;
import org.apache.ibatis.annotations.Param;
import java.util.List;

public interface UserMapper {

    int addUser(User user);

    List<User> getAllUsers();

    User getUserById(@Param("id") Integer id);

    int updateUser(User user);

    int deleteUser(@Param("id") Integer id);

}

        ```

         **entity/User.java**
         
        ```java
        package com.wang.springboot.demo.entity;

public class User {

    private Integer id;
    private String name;
    private String pwd;

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getPwd() {
        return pwd;
    }

    public void setPwd(String pwd) {
        this.pwd = pwd;
    }

    @Override
    public String toString() {
        return "User{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", pwd='" + pwd + '\'' +
                '}';
    }
}

        ```

         **service/UserService.java**
         
        ```java
        package com.wang.springboot.demo.service;

import com.wang.springboot.demo.entity.User;

import java.util.List;

public interface UserService {

    int addUser(User user);

    List<User> getAllUsers();

    User getUserById(int userId);

    int updateUser(User user);

    int deleteUser(int userId);

}
        ```


         **service/impl/UserServiceImpl.java**
         
        ```java
        package com.wang.springboot.demo.service.impl;

import com.wang.springboot.demo.dao.UserMapper;
import com.wang.springboot.demo.entity.User;
import com.wang.springboot.demo.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class UserServiceImpl implements UserService {

    @Autowired
    private UserMapper userMapper;

    @Override
    public int addUser(User user) {
        return userMapper.addUser(user);
    }

    @Override
    public List<User> getAllUsers() {
        return userMapper.getAllUsers();
    }

    @Override
    public User getUserById(int userId) {
        return userMapper.getUserById(userId);
    }

    @Override
    public int updateUser(User user) {
        return userMapper.updateUser(user);
    }

    @Override
    public int deleteUser(int userId) {
        return userMapper.deleteUser(userId);
    }
}


        ```
         
         **util/MD5Util.java**
         
        ```java
        package com.wang.springboot.demo.util;

import java.security.MessageDigest;

/**
 * Created by WangChen on 2020-04-10.
 */
public class MD5Util {

    public static String md5Encode(String str){
        try {
            MessageDigest messageDigest=MessageDigest.getInstance("MD5");
            byte[] bytes=messageDigest.digest(str.getBytes());

            StringBuilder stringBuilder=new StringBuilder();
            for(byte b :bytes){
                int num=b&0xff;
                String hexString=Integer.toHexString(num);
                if(hexString.length()==1){
                    stringBuilder.append('0');
                }
                stringBuilder.append(hexString);
            }
            return stringBuilder.toString().toUpperCase();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}

        ```

         **mybatis-config.xml**
         
        ```xml
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE configuration PUBLIC "-//mybatis.org//DTD Config 3.0//EN" "http://mybatis.org/dtd/mybatis-3-config.dtd">
        <configuration>
          <settings>
            <setting name="lazyLoadingEnabled" value="true"/>
            <setting name="aggressiveLazyLoading" value="false"/>
            <setting name="multipleResultSetsEnabled" value="true"/>
            <setting name="useGeneratedKeys" value="true"/>
          </settings>
          <typeAliases>
            <typeAlias type="com.wang.springboot.demo.entity.User" alias="User"/>
          </typeAliases>
          <mappers>
            <mapper resource="mapper/UserMapper.xml"/>
          </mappers>
        </configuration>

        ```
         
         **src/main/webapp/WEB-INF/jsp/login.jsp**
         
        ```html
        <%@ page contentType="text/html;charset=UTF-8" language="java" %>
        <html>
        <head>
            <title>用户登录</title>
        </head>
        <body>
        <h1 align="center">欢迎使用</h1><br>
        <div style="width: 30%;margin: auto;">
            <form action="${pageContext.request.contextPath}/doLogin" method="post">
                <label>用户名：</label>&nbsp;<input type="text" name="username" required="required"><br><br>
                <label>密 码：</label>&nbsp;<input type="password" name="password" required="required"><br><br>
                <button type="submit">登录</button>
            </form>
        </div>
        </body>
        </html>

        ```
         
         **src/main/webapp/WEB-INF/jsp/welcome.jsp**
         
        ```html
        <%@ page contentType="text/html;charset=UTF-8" language="java" %>
        <html>
        <head>
            <title>欢迎页面</title>
        </head>
        <body>
        <h1 align="center">欢迎您，${sessionScope.userId}</h1><br>
        <div style="width: 30%;margin: auto;"><a href="${pageContext.request.contextPath}/logout">退出登录</a></div>
        </body>
        </html>
        ```
         
         **mapper/UserMapper.xml**
         
        ```xml
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
        <mapper namespace="com.wang.springboot.demo.dao.UserMapper">
          <insert id="addUser" parameterType="User">
            INSERT INTO users (name,pwd) VALUES (#{name}, #{pwd})
          </insert>
          <select id="getAllUsers" resultType="User">
            SELECT id,name,pwd FROM users
          </select>
          <select id="getUserById" parameterType="int" resultType="User">
            SELECT id,name,pwd FROM users WHERE id=#{id}
          </select>
          <update id="updateUser" parameterType="User">
            UPDATE users SET name=#{name},pwd=#{pwd} WHERE id=#{id}
          </update>
          <delete id="deleteUser" parameterType="int">
            DELETE FROM users WHERE id=#{id}
          </delete>
        </mapper>
        ```

         **src/test/java/com/wang/springboot/demo/AppTests.java**
         
        ```java
        package com.wang.springboot.demo;

        import com.wang.springboot.demo.dao.UserMapper;
        import com.wang.springboot.demo.entity.User;
        import org.junit.Test;
        import org.junit.runner.RunWith;
        import org.springframework.beans.factory.annotation.Autowired;
        import org.springframework.boot.test.context.SpringBootTest;
        import org.springframework.test.context.junit4.SpringRunner;

       @RunWith(SpringRunner.class)
       @SpringBootTest(classes={App.class})
       public class AppTests {
           @Autowired
           private UserMapper userMapper;

           @Test
           public void testAddUser(){
               User user = new User();
               user.setName("zhangsan");
               user.setPwd(MD5Util.md5Encode("123456"));
               int count = userMapper.addUser(user);
               assert count == 1;
           }

           @Test
           public void testGetAllUsers(){
               List<User> list = userMapper.getAllUsers();
               for(User u :list){
                   System.out.println(u);
               }
           }

           @Test
           public void testGetUserById(){
               User user = userMapper.getUserById(1);
               System.out.println(user);
           }

           @Test
           public void testUpdateUser(){
               User user = userMapper.getUserById(1);
               user.setName("lisi");
               int count = userMapper.updateUser(user);
               assert count == 1;
           }

           @Test
           public void testDeleteUser(){
               int count = userMapper.deleteUser(1);
               assert count == 1;
           }
       }
        ```

         
         执行单元测试：
         
        ```shell
        mvn clean install
        ```
         
         启动项目：
         
        ```shell
        mvn spring-boot:run
        ```
         
         浏览器打开：
         
         
         输入账号密码：
         
        admin 123456
         
         可以看到登录成功跳转到欢迎页面，如下图：
         
         
         点击“退出登录”，即可退出当前账户，重新登录。