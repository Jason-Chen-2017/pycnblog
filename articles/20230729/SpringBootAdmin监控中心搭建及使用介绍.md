
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　Spring Boot Admin（下文简称SBA）是一个开源的基于Spring Boot的服务监控中心，主要功能包括服务注册发现、服务健康状态检测、主动和被动通知、服务 metrics 数据收集、日志查看、JVM 信息查看等。该项目从Spring Boot基础设施出发，提供了一系列便利的功能组件，让开发者能够快速、方便地集成到自己的应用中。本文档将详细介绍如何利用SBA实现微服务系统的监控中心部署和使用，帮助读者搭建起专业级的微服务监控平台。
         # 2.基本概念
         ## 服务注册与发现（Service Registry and Discovery）
         　　服务发现，即根据服务名查找其对应的服务实例地址或负载均衡器地址。在分布式架构中，服务发现通常是最重要也是复杂的环节。而服务注册就是把提供服务的信息记录到一个中心数据库里，供其他客户端查询使用。这样可以解决服务调用时服务实例动态变化的问题。
        ![](https://img-blog.csdnimg.cn/20200709150127767.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNjg1MzQyNw==,size_16,color_FFFFFF,t_70)
         ### Eureka服务注册中心
         　　Eureka是Netflix开源的一款高可用的服务发现和注册中心。它具有以下几个特性：
         （1）服务注册与发现：Eureka Server作为服务注册表，接受服务节点的注册和注销请求；Client向Server端注册，定期上报心跳；Server根据心跳信息，管理各个节点之间的数据同步。因此，Eureka既可以作为服务注册中心，也可以作为服务消费者。
         （2）负载均衡：Eureka Client通过获取到的服务注册表数据，实现了客户端的负载均衡功能。当某台服务器宕机或者上下线时，Eureka会立刻更新服务注册表，避免了调用失败引起的连锁反应。
         （3）故障转移：由于Server和Client都有内置的容错机制，Eureka具备自我修复能力，不会对外界环境造成影响。
         （4）弹性伸缩：Eureka通过集群模式提升系统容量，允许每个Region节点数量的不同。同时，Eureka也支持客户端的区域感知，只访问当前可用区域的Eureka Server。
         　　下面展示的是Eureka服务注册中心的架构图：
         　　![](https://img-blog.csdnimg.cn/20200709150246467.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNjg1MzQyNw==,size_16,color_FFFFFF,t_70)
         ### Consul服务注册中心
         　　Consul是 HashiCorp公司推出的开源服务注册与发现工具。Consul采用gossip协议，无中心结构，因此它的可靠性高于其他任何服务发现方案。
         　　Consul拥有如下几个特点：
         　　1.服务发现：Consul可以通过DNS或HTTP的方式进行服务发现，同时也支持可选的基于客户端的负载均衡策略。
         　　2.健康检查：Consul可以对服务的健康情况进行实时的监测，如果出现故障，则会及时剔除出服务列表。
         　　3.键值存储：Consul提供了一个简单的键值存储，供用户存放任意数据。
         　　4.多数据中心：Consul支持多数据中心，并且支持跨WAN连接。
         　　下面展示的是Consul服务注册中心的架构图：
         　　![](https://img-blog.csdnimg.cn/20200709150351616.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNjg1MzQyNw==,size_16,color_FFFFFF,t_70)
         ### Zookeeper服务注册中心
         　　Apache ZooKeeper 是 Apache Hadoop 的子项目，它是一个分布式协调服务。它是一个开放源代码的分布式协调服务，它是一个分布式过程管理系统，它是一个为分布式应用提供一致性服务的 software framework。它是一个分布式dbms，能够非常容易的进行数据分片。Zookeeper 是 Java 语言实现的一个开源分布式协调服务，是一个分布式的配置文件/配置项管理工具。
         ### Nacos服务注册中心
         　　Nacos是阿里巴巴开源的更易于构建云原生应用的动态服务发现、配置管理和服务管理平台。它支持直连至物理机，虚拟机，容器，云主机，OpenStack，Kubernetes，Mesos等。相比于传统的服务发现框架，Nacos 提供了更多企业级特性，如服务降级，熔断，灰度发布，流量控制，数据分发，多数据中心。另外，它还支持一站式管理所有服务，统一管理数据。
         ## 角色
         　　Spring Boot Admin 有四个角色：
         　　1. SBA Server：Spring Boot Admin Server 是一个独立的应用，用于接收监控数据并显示给管理员。它与各种监控系统兼容，例如 Prometheus，Datadog，JMX等。
         　　2. SBA Client：Spring Boot Admin Client是一个轻量级Java库，用于向Spring Boot Admin Server发送监控数据。你可以用它来监控你的应用程序运行状况。
         　　3. SBA Gateway：Spring Boot Admin Gateway是一个HTTP代理，它使得客户端不需要直接与Spring Boot Admin Server通信。它可以防止潜在的安全漏洞，并允许使用标准端口进行通信。
         　　4. SBA Admin UI：Spring Boot Admin UI是一个基于Angular的Web界面，用于管理监控服务。你可以通过它添加、删除客户端，设置警报规则，管理通知渠道等。
         　　下图展示的是Spring Boot Admin 的架构示意图：
         　　![](https://img-blog.csdnimg.cn/20200709150501841.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNjg1MzQyNw==,size_16,color_FFFFFF,t_70)
       　　# 3.功能模块
         Spring Boot Admin (SBA) 提供了几十种监控指标。它们分为如下六类：
         1. 概览：总体概述，显示集群整体资源使用率。
         2. JVM Metrics：显示堆内存、线程池、类加载和垃圾回收等运行状态。
         3. Data Stores Stats：显示数据库连接池、Hibernate统计数据、缓存命中率等。
         4. Messaging：显示消息代理（比如Kafka或RabbitMQ）的相关统计信息。
         5. HTTP Endpoints：显示应用暴露的REST、WebSockets或other HTTP endpoints的相关统计信息。
         6. Circuit Breakers：显示应用的熔断器状态。
         
         # 4.安装配置
         1. 下载安装包：进入spring官网 https://start.spring.io/ 选择适合自己环境的jdk版本，maven版本等。然后引入spring-boot-starter-admin-server依赖：
         ```xml
            <dependency>
                <groupId>de.codecentric</groupId>
                <artifactId>spring-boot-admin-starter-server</artifactId>
            </dependency>
         ```
         上面就引入了sbaserver相关的依赖，spring-boot-starter-actuator依赖是用于提供健康检查，无论什么时候启动成功都会去访问/health接口返回true，我们这里不需要使用这个。
         2. 配置文件 application.properties 设置数据库url、端口、账号密码等信息：
         ```yaml
            spring:
              datasource:
                url: jdbc:mysql://localhost:3306/sbaserver
                username: root
                password: yourpassword
                driverClassName: com.mysql.jdbc.Driver
              jpa:
                database-platform: org.hibernate.dialect.MySQL5InnoDBDialect
                hibernate:
                  ddl-auto: update
                  naming-strategy: org.springframework.boot.orm.jpa.hibernate.SpringNamingStrategy
           management:
             endpoint:
               health:
                 show-details: always
             endpoints:
               web:
                 exposure:
                   include: '*'
         ```
         在这里设置了sbaserver数据库链接信息，jpa相关配置，management相关配置等。
         3. 添加健康检查接口：通过实现HealthIndicator接口，定制自己的健康检查逻辑。
         ```java
            @Component
            public class MyHealthIndicator implements HealthIndicator {
            
                private static final String HEALTH = "OK";
                
                // 判断自己是否正常工作，这里就简单返回 OK 字符串
                @Override
                public Health health() {
                    if(isWorking()){
                        return Health.up().withDetail("status", HEALTH).build();
                    } else{
                        return Health.down().withDetail("status", "NOT OK").build();
                    }
                }
                
                private boolean isWorking(){
                    try {
                        // 此处判断逻辑替换为自己的业务判断
                        Thread.sleep(100);
                        return true;
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                }
                
            }
         ```
         4. 启动SbaServer工程，注意此时不要启动Sbaclient和ui相关工程，因为他们还没有配置。打开浏览器，输入http://localhost:port 登录后台页面，初始用户名密码都是admin/admin123.
         # 5.客户端接入
         1. 配置pom.xml文件引入sbaclient依赖：
         ```xml
            <dependency>
                <groupId>de.codecentric</groupId>
                <artifactId>spring-boot-admin-starter-client</artifactId>
            </dependency>
         ```
         2. 创建sbaconfig.yml配置文件，配置应用信息：
         ```yaml
            server:
              port: ${random.value}
              address: localhost
            spring:
              application:
                name: sbaclient
              boot:
                admin:
                  client:
                    url: http://localhost:port
                    username: admin
                    password: <PASSWORD>
           logging:
             level:
               de.codecentric: trace
         ```
         3. 创建SBAClientApplication类，并启动：
         ```java
            package cn.monitor;
            
            import org.springframework.boot.CommandLineRunner;
            import org.springframework.boot.autoconfigure.SpringBootApplication;
            import org.springframework.boot.builder.SpringApplicationBuilder;
            import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
            import org.springframework.context.annotation.Configuration;
            
            /**
             * 启动类，引入@EnableDiscoveryClient注解使服务注册到eureka
             */
            @SpringBootApplication
            @EnableDiscoveryClient
            @Configuration
            public class SBAClientApplication implements CommandLineRunner {
            
                public static void main(String[] args) throws Exception {
                    new SpringApplicationBuilder(SBAClientApplication.class)
                           .web(false) // 不使用web环境
                           .run(args);
                }
                
                @Override
                public void run(String... strings) throws Exception {
                    System.out.println("启动成功");
                }
                
            }
         ```
         4. 浏览器输入 http://localhost:${your_sbaserver_port}/applications 可以看到已注册的客户端列表，点击其中一个客户端的名称进入详情页。详情页显示该客户端的一些信息，比如健康状态、应用名称、ip地址、端口号、启动时间、jvm信息等。可以点击“健康状态”旁边的按钮查看详细的健康信息。
         5. 通过 /env 和 /metrics 两个端点查看应用内部状态和外部状态，/env 获取到应用的环境变量，/metrics 返回prometheus格式的监控指标，sbaserver并不依赖prometheus或者任何监控系统，所以这些指标需要自己去解析并展示出来。
         6. 配置启动命令行参数：
         ```yaml
            server:
              port: ${random.value}
              address: localhost
            spring:
              application:
                name: sbaclient
              boot:
                admin:
                  client:
                    url: http://localhost:port
                    username: admin
                    password: <PASSWORD>
                    instance:
                      metadata:
                        user.name: sbauser # 可配置启动参数，将传入到AdminUI的元数据页面
                      ...
          ```
         7. 使用docker部署：首先编写Dockerfile，将sbaclient打包进镜像：
         ```dockerfile
            FROM openjdk:8-jre-alpine
            VOLUME /tmp
            ADD target/${project.artifactId}-${project.version}.jar app.jar
            ENTRYPOINT ["sh","-c","java $JAVA_OPTS -Djava.security.egd=file:/dev/./urandom -jar /app.jar"]
            EXPOSE 8080
            CMD []
         ```
         8. 将本地生成的tar文件上传到目标机器，执行 docker load -i image.tar 命令导入镜像，然后 docker run -it -p 8080:8080 --rm --name sbaclient [imageId] 运行镜像。然后登陆后台管理页面 http://localhost:8080 ，页面左侧会显示已经启动的sbaclient服务。

