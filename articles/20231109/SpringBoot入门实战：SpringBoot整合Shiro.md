                 

# 1.背景介绍


## Shiro简介
Apache Shiro是一个强大的Java安全框架，提供了身份验证、授权、加密和会话管理功能。其核心设计目标就是简单易用，同时提供足够的灵活性来满足特定应用需求。通过使用 Shiro，可以轻松完成对用户进行认证、授权、密码加密等操作，还可以通过灵活的配置控制权限。Shiro的核心设计理念是解耦，即应用中的各种功能模块之间无需相互了解，而只需要遵循特定的API接口规范即可实现功能。因此，在实际项目中使用 Shiro 时，需要结合实际情况选择使用哪个功能模块及如何集成才能完整地实现应用安全要求。

通过以下两个步骤，可以完成Spring Boot + Shiro 的集成:

1. 添加pom依赖
```xml
        <dependency>
            <groupId>org.apache.shiro</groupId>
            <artifactId>shiro-spring-boot-starter</artifactId>
            <version>${shiro.version}</version>
        </dependency>
        
        <!-- web 支持 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- 数据源支持 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>

        <!-- lombok 支持 -->
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <optional>true</optional>
        </dependency>
```
2. 配置文件application.yml
```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/shiro?useUnicode=true&characterEncoding=utf8&serverTimezone=GMT%2B8
    username: root
    password: 123456

  jpa:
    properties:
      hibernate:
        dialect: org.hibernate.dialect.MySQL5Dialect
  
shiro:
  # session超时时间，单位：秒
  session-timeout: 300
  
  # cookie信息，用于session持久化，不需要可以注释掉
  rememberme:
    cookie-name: rememberMe
    cookie-domain: localhost
    cookie-path: /
    httponly: true
    max-age: 1000
    remember-me-parameter: rememberMe
    encryption-cipher-key-size: 128
    
logging:
  level:
    com.example: DEBUG
```
其中shiro.rememberme选项用于配置RememberMe服务，默认关闭。

## SpringBoot简介


## 为什么要学习Spring Boot+Shiro
Spring Boot 和 Shiro 可以帮助我们快速构建出具备安全防护功能的系统，基于 SpringBoot 能更好地利用微服务架构模式。

- 更高效的开发体验：Spring Boot 提供了快速配置项目的便利；Shiro 作为一个优秀的安全框架，可以极大地提升系统安全能力；
- 抽象屏蔽底层实现：SpringBoot 将复杂的配置项封装成了 starter 包，屏蔽了底层实现，开发者可以直接使用；
- 云原生时代到来：Kubernetes + Spring Boot + Shiro 携手构建云原生应用，更加贴近真实的生产环境。

通过上述介绍，我们可以看到 SpringBoot+Shiro 是一款非常流行的组合框架，学习它，可以加深我们对于 Spring Boot、微服务架构、以及安全攻防领域的理解。