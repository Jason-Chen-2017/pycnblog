
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         Spring Cloud 是一系列框架的有机集合。其中 Spring Cloud Netflix 为微服务架构提供了基础设施支持，包括 Eureka、Ribbon、Hystrix 和 Turbine。Spring Boot 是微服务开发的更高级的抽象。Spring Cloud Feign 是一个声明式 Web Service 客户端，它使得编写 Web Service Client 的时候只需要创建一个接口，通过注解的方式即可完成其实现，并使用 Spring 配置文件或者 Java API 来配置。通过这种方式就可以方便地调用远程服务，而不需要显式地创建WebService调用的代码。本文主要介绍如何使用 Spring Cloud Feign 进行服务调用。
         
         # 2.基本概念及术语说明
         
         ## 2.1 服务发现（Service Discovery）
         
         Spring Cloud Netflix Eureka 是 Spring Cloud 的服务发现组件之一，主要用于微服务架构中的服务注册与发现。在 Spring Cloud 中，应用可以向 Eureka 服务器注册自己的身份信息，并通过心跳维护自己的状态。其他应用可以通过访问 Eureka 服务器来获取注册表中的可用服务列表，然后再根据负载均衡策略选择合适的服务实例进行调用。此外，Eureka 提供了一套完整的 Restful API，允许外部系统查询或修改注册信息。
         
         ## 2.2 RESTful 标准
         
         超文本传输协议（HTTP）是互联网上最基本的通信协议。REST 则是 HTTP 的一个设计风格，用来构建基于资源的Web应用程序。RESTful 是指基于 HTTP、URL、XML、JSON等普遍采用的网络协议标准。
         
         ## 2.3 Feign
         
         Spring Cloud Feign 是 Spring Cloud 中的声明式 Web Service 客户端，它使得编写 Web Service Client 的时候只需要创建一个接口，通过注解的方式即可完成其实现，并使用 Spring 配置文件或者 Java API 来配置。Feign 的底层依赖是 Retrofit 2，它是一个用 Java 编写的类型安全的 HTTP 客户端。它可以使用注解来定义 HTTP 请求方法，并提供对应的 Java 方法，Feign 会生成针对该 HTTP 接口的 Retrofit 2 实现类。
         
         # 3.核心算法原理及详细说明
         
         Feign 使用了动态代理的机制来创建一个接口的代理对象。代理对象的方法调用会被发送到目标服务的实际地址（URL）。Feign 支持很多种类型的注解，比如 @RequestMapping、@GetMapping、@PostMapping 等等。这些注解可以帮助 Feign 根据不同的 HTTP 请求方法和请求参数构造不同的 URL。
         
         通过 Feign 创建的代理对象会被注入到 Spring Bean 中，因此可以在需要的时候通过 DI 来使用该对象。Feign 可以自动将响应体转换成指定的 Java 对象，还可以对异常做统一的处理。Feign 默认使用 okhttp 作为 HTTP 客户端库。
         
         当客户端调用一个 Feign 代理对象的方法时，Feign 会根据方法的参数列表、返回值、注解等等生成符合 RESTful 规范的 HTTP 请求，并通过 HttpClient 或 okHttp 等 HTTP 客户端发送请求到指定的服务端。Feign 将 HTTP 响应的内容解析为指定的数据类型，并包装成 ResponseEntity 对象返回给客户端。
         
         Feign 的自动配置功能能够根据 Spring Environment 来加载不同环境下的 Feign 客户端配置。当 spring-cloud-starter-netflix-eureka-client 和 spring-boot-starter-web 这两个模块都在 classpath 下时，Feign 客户端会默认开启。Feign 的启动过程主要涉及以下三个步骤：
         
         * 查找 META-INF/spring.factories 文件中 org.springframework.boot.autoconfigure.EnableAutoConfiguration 类的声明，找到后便开始分析该类的内容；
         * 从当前应用上下文中搜索所有BeanFactoryPostProcessor类型的Bean，并依次调用它们的postProcessBeanFactory()方法；
         * 查找当前 ApplicationContext 中是否存在一个名为 feignContext 的 BeanFactory，如果不存在，则新建一个，然后遍历所有的bean，查找名称中包含feign的bean，如果存在，则把该bean加入到feign的BeanFactory中。

Feign 的配置文件如下所示：
```yaml
feign:
  httpclient:
    enabled: false
  okhttp:
    enabled: true
  client:
    config:
      default:
        connectTimeout: ${feign.client.connect-timeout:1000}
        readTimeout: ${feign.client.read-timeout:5000}
        loggerLevel: basic
```
Feign 的日志级别由 `loggerLevel` 属性控制，默认值为 `none`。可选值为 `NONE`, `BASIC`, `HEADERS`, `FULL`，分别表示不输出日志、仅输出请求头和响应头、仅输出请求头、请求头和响应体。

## 3.1 Feign 工作流程图

下图展示了一个 Feign 调用链路的流程。

![Feign 工作流程](https://upload-images.jianshu.io/upload_images/7985497-aa9db0b0f41d5d1f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)

 如上图所示，Feign 的工作流程比较简单。首先，Feign 根据方法的元数据生成对应的 HTTP 请求。Feign 在运行时，会尝试从 Spring Context 获取 Client 相关的 Bean，如果没有，则按照默认配置创建一个新的 OkHttpClient 。然后，Feign 会用 Client 发起 HTTP 请求，得到响应。Feign 对响应进行解析，并根据其内容转换为指定的数据类型。

# 4.代码实例

下面用一个简单的示例来演示如何使用 Feign 调用远程服务。假设有一个服务端 API 暴露在某个 URL 上（http://localhost:8081），并且该服务接受 GET 请求并返回字符串。下面通过 Feign 调用这个服务，并打印返回结果。

## 4.1 服务端 API

### pom.xml

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-server</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-openfeign</artifactId>
</dependency>
```

### application.yml

```yaml
server:
  port: 8081

eureka:
  instance:
    hostname: localhost
  client:
    registerWithEureka: false
    fetchRegistry: false
    serviceUrl:
      defaultZone: http://${eureka.instance.hostname}:${server.port}/eureka/

feign:
  client:
    config:
      default:
        connectTimeout: 1000
        readTimeout: 5000
```

### HelloController.java

```java
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HelloController {

    @GetMapping("/hello")
    public String sayHello() {
        return "Hello, world!";
    }
}
```

## 4.2 客户端调用

### pom.xml

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>

<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-openfeign</artifactId>
</dependency>
```

### application.yml

```yaml
server:
  port: 8082
  
eureka:
  client:
    registry-fetch-interval-seconds: 5
    
feign:
  client:
    config:
      default:
        connectTimeout: 1000
        readTimeout: 5000
```

### HelloClient.java

```java
import feign.Feign;
import feign.Logger;
import feign.RequestLine;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public interface HelloClient {
    
    @RequestLine("GET /hello")
    String sayHello();
}

@Component
public class HelloServiceClient implements HelloClient{

    @Autowired
    private Feign.Builder builder;

    @Override
    public String sayHello(){
        HelloClient helloClient = this.builder
               .logLevel(Logger.Level.FULL) // show request & response log in console
               .target(HelloClient.class, "http://localhost:8081");

        return helloClient.sayHello();
    }
}
```

通过 `@RequestLine` 注解定义了一个接口，该接口定义了一个名为 `sayHello()` 的方法，用于从服务端获取数据。然后通过 `@Feign` 注解来声明一个接口的实现类 `HelloServiceClient`，该实现类中通过 `Feign.Builder` 生成一个 `HelloClient` 代理对象，该代理对象指向服务端 API 的 `/hello` 路径。最后，通过 `Feign` 提供的方法调用 `sayHello()` 方法从服务端获取数据。由于 `LogLevel.FULL` 的设置，在控制台上可以看到相应的请求和响应信息。

