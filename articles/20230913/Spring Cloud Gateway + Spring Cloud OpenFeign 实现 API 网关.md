
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在微服务架构下，API网关的作用主要是用来聚合、控制、编排各个微服务的请求。Spring Cloud Gateway是一个基于Spring Framework之上的API网关，它利用了Spring Boot Admin来提供服务发现，Zuul1作为服务网关可以很好的对接现有的系统。而OpenFeign是一个声明式Web服务客户端，使得调用远程服务变得更简单，通过注解或者配置的方式完成远程调用。本文将主要介绍如何结合两者构建一个简单的API网关。

# 2.技术栈

# 3.前置准备工作
1. 安装JDK 8或以上版本
2. 安装Maven 3.x或以上版本
3. 配置IDE环境（推荐 IntelliJ IDEA）
4. 创建maven项目并引入相关依赖，pom文件如下:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>demo</artifactId>
    <version>1.0-SNAPSHOT</version>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.1.7.RELEASE</version>
    </parent>

    <!-- spring cloud -->
    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>org.springframework.cloud</groupId>
                <artifactId>spring-cloud-dependencies</artifactId>
                <version>Greenwich.SR3</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>

    <dependencies>
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-starter-gateway</artifactId>
        </dependency>

        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-starter-openfeign</artifactId>
        </dependency>

        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
        </dependency>
        
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-webflux</artifactId>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-actuator</artifactId>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
            <optional>true</optional>
        </dependency>

        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <optional>true</optional>
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

5. 在application.properties中添加以下配置信息：
```yaml
server.port=8080
spring.application.name=gateway
management.endpoints.web.exposure.include=*

# eureka server config
eureka.client.serviceUrl.defaultZone=http://localhost:8761/eureka/

# open feign client config
feign.hystrix.enabled=true # 设置feign client开启熔断保护功能
feign.sentinel.enabled=false # 不使用Sentinel组件
```
6. 创建Eureka Server项目，引入依赖如下：
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-server</artifactId>
</dependency>
```

7. 在启动类上添加@EnableEurekaServer注解，启动Application主函数。