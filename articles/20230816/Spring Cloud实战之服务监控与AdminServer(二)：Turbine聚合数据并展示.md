
作者：禅与计算机程序设计艺术                    

# 1.简介
  

　　Spring Cloud提供了完善的微服务治理功能，包括配置中心、注册中心、网关、熔断器等，而这些功能都是基于Spring Boot实现的，其中最重要的是监控系统，监控系统主要用来实时监控各个微服务节点的运行状态、JVM指标、线程池信息、请求响应时间、链路追踪日志、接口访问情况等。

　　Spring Cloud Admin是一个用于管理Spring Boot应用程序的可选组件，它对分布式系统进行健康检查，并提供一个图形化界面，让管理员可以直观的查看各个应用程序的健康状况。其架构如下：

　　 Spring Cloud Admin Server作为Spring Cloud配置中心、注册中心、网关、熔断器的统一入口，它提供监控和管理各种应用程序的能力，通过网页或者API的方式可以看到各个应用程序的健康状态，并且提供多维度的监控视图，方便运维人员快速定位故障点。另外，Admin Server还可以提供单点登录（SSO）功能，用户可以使用统一认证中心的账户进行登录，同时也支持LDAP或OAuth认证方式。

　　本文将结合实际案例，带领大家掌握如何利用Turbine和Hystrix Turbine Stream结合实现服务监控Dashboard，并根据Hystrix Turbine Stream提供的数据进行进一步分析和展示，从而帮助大家更加高效地管理和维护微服务系统。

 

# 2.环境准备

 　　本次实践基于Spring Cloud 2.2.1版本进行开发。
  
  - 基础依赖：
  
      ```xml
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-dependencies</artifactId>
            <version>${spring-cloud.version}</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>

        <!-- Hystrix -->
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-starter-netflix-hystrix</artifactId>
        </dependency>
        
        <!-- Netflix Eureka Client -->
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
        </dependency>

         <!-- Spring Boot Admin Client -->
         <dependency>
             <groupId>de.codecentric</groupId>
             <artifactId>spring-boot-admin-starter-client</artifactId>
         </dependency>

         <!-- Spring Web -->
         <dependency>
             <groupId>org.springframework.boot</groupId>
             <artifactId>spring-boot-starter-web</artifactId>
         </dependency>
         
         <!-- Actuator -->
         <dependency>
             <groupId>org.springframework.boot</groupId>
             <artifactId>spring-boot-starter-actuator</artifactId>
         </dependency>
      ```
  - 配置文件
  
  　　需要在配置文件中添加以下内容:
  
   - application.yml
   
       ```yaml
       server:
           port: ${port:8081}

       spring:
           application:
               name: spring-cloud-admin-server

           security:
               user:
                   name: admin
                   password: admin
                   
           boot:
               admin:
                   client:
                       url: http://localhost:8082 # Spring Boot Admin 服务地址
                       
       eureka:
           instance:
              hostname: localhost
           client:
              serviceUrl:
                  defaultZone: http://${eureka.instance.hostname}:${server.port}/eureka/   
       ```
   
   - bootstrap.yml
     
     ```yaml
     management:
         endpoints:
             web:
                 exposure:
                     include: "*"
                     
         endpoint:
             health:
                 show-details: ALWAYS
     
     turbine:
         app-config: foo,bar
         cluster-name-expression: new String("default")    
     
     hystrix:
         command:
             default:
                 execution:
                     isolation:
                         thread:
                             timeoutInMilliseconds: 60000     
     ```
 
 　　`application.yml`定义了Spring Boot Admin Server 的端口号为`8081`，Eureka的默认地址为`http://localhost:8082`，同时启用了所有端点暴露。
  
  　　`bootstrap.yml`配置了turbine聚合的应用列表，以及命令执行超时时间。
  
  - 模拟微服务实例
  
  　　为了演示Dashboard效果，我们模拟两个微服务实例，分别命名为`foo`和`bar`，配置如下：
  
   - foo:
      
      ```yaml
      server:
          port: ${port:8080}

      spring:
          application:
              name: foo

          boot:
              admin:
                  client:
                      enabled: true
                      url: http://localhost:${management.server.port}/admin

      eureka:
          client:
              service-url:
                  defaultZone: http://localhost:${server.port}/eureka/
      ```
 
   - bar:
      
      ```yaml
      server:
          port: ${port:8081}

      spring:
          application:
              name: bar

          boot:
              admin:
                  client:
                      enabled: true
                      url: http://localhost:${management.server.port}/admin

      eureka:
          client:
              service-url:
                  defaultZone: http://localhost:${server.port}/eureka/      
      ```

 　　为了验证监控Dashboard效果，我们先启动`Spring Boot Admin Server`，再启动`foo`和`bar`两个微服务实例。
 

# 3.集群内监控效果

 　　启动后，我们打开浏览器输入 `http://localhost:8081/` ，打开Spring Boot Admin Dashboard页面，可以看到如下图所示的监控信息。
  
  
 　　页面左上角显示当前集群总实例数以及活跃实例数，点击红色框中的`Unregister`按钮会销毁当前微服务实例。右侧通过微服务名，依次显示每个微服务的健康状态，以及最近一次心跳时间和平均响应时间。蓝色区域为`Turbine`相关统计信息，我们可以通过下拉菜单选择不同的集群名称，可以看到具体的节点以及响应时间等统计数据。
  
　　点击下方的`Application Metrics`标签，进入到微服务监控面板，可以看到`Hystrix Command Metric Streams`、`Hystrix ThreadPool Metrics Streams`、`Refresh Scope Metrics`等详细的监控数据。

　　如上图所示，我们可以看到`foo`服务的实例名称为`foo:8080`，其健康状态显示为`UP`，最近一次心跳时间为`less than a minute`，平均响应时间为`0ms`。但点击该实例名称时，无法跳转到详情页面，因为这是一个虚拟实例，由`Turbine`根据多个服务实例数据聚合生成，无法详细区分某个具体实例的信息。而`bar`服务的实例名称同样为`bar:8081`，健康状态显示为`UP`，最近一次心跳时间为`less than a minute`，平均响应时间也是`0ms`。


# 4.集群间监控效果

 　　为了更好的了解集群之间的服务调用关系，我们把`foo`和`bar`两个服务增加一个依赖，让它们都调用`hello`服务，修改配置文件如下：
  
   - foo:
    
      ```yaml
     ...
      dependencies:
          - hello

      hello:
          ribbon:
              listOfServers: "http://localhost:8082"  
      ```
   - bar:
   
      ```yaml
     ...
      dependencies:
          - hello

      hello:
          ribbon:
              listOfServers: "http://localhost:8082"     
      ```

   - HelloController.java:
   
      ```java
      package com.example;

      import org.springframework.beans.factory.annotation.Autowired;
      import org.springframework.web.bind.annotation.RequestMapping;
      import org.springframework.web.bind.annotation.RestController;
      import org.springframework.web.client.RestTemplate;

      @RestController
      public class HelloController {

          private RestTemplate restTemplate;

          @Autowired
          public HelloController(RestTemplate restTemplate) {
              this.restTemplate = restTemplate;
          }

          @RequestMapping("/hello")
          public String hello() {
              return restTemplate.getForObject("http://hello", String.class);
          }
      }
      ```
  
  　　再次启动项目，打开Spring Boot Admin Dashboard页面，就可以看到两者之间建立起来的依赖关系。点击`hello`服务实例名称，就可以跳转到详情页面，里面可以看到该服务的依赖关系信息，以及各项指标监控信息。

  
 　　如上图所示，`foo`服务的依赖关系中有一个`hello`服务；而`bar`服务的依赖关系中也有一个`hello`服务，但是它的状态为`OUT_OF_SERVICE`。这是由于在`bootstrap.yml`配置文件中，我们设置了`turbine.clusterNameExpression="new String('default')"`，表示只聚合`foo`和`bar`这两个微服务实例的数据，所以当`foo`和`bar`之间存在依赖关系时，依赖于其他服务的`hello`服务的实例就不会被监控到。如果想让依赖于其他服务的服务也被监控，则可以在`turbine.clusterNameExpression`表达式中加入相应的微服务名称。