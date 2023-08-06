
作者：禅与计算机程序设计艺术                    

# 1.简介
         
Reactor 是 Reactive Streams （RS）规范中的一部分，它是构建异步流处理应用的基础设施。在 Spring 生态系统中，Spring Framework 5 将对 RS 提供更好的支持。从 Spring Framework 5.0 开始，Spring WebFlux 模块中已经默认集成了 Reactor Core ，并且提供了响应式 Web 框架。本文将通过 Spring WebFlux 和 Reactor 对 Spring Boot 进行集成、配置及原理分析，详细阐述 Spring Boot 在 Reactor 中的应用及实现。

# 2.知识背景
## 2.1 Reactive Streams（RS）规范
Reactive Streams 是一个标准，定义了可观察序列的规范。其中定义了发布者、订阅者、发布数据、接收数据等概念。其主要目的是为了提供一种统一的方法，让发布者和订阅者之间可以异步通信。该规范是 Java 9 引入的模块化开发工具 JDK 9 用来做异步编程的一部分。

## 2.2 Reactor Core 库
Reactor Core 是一个用于实现 Reactive Streams 规范的一组 Java 类。它的功能包括创建发布者、订阅者、连接器、发布、订阅事件流、调度任务等。Reactor Core 是一个独立的模块，允许开发者选择自己喜欢的实现方式。Reactor Core 可以在多种环境下运行，比如 Servlet、Netty、Undertow、RxJava。

## 2.3 Spring WebFlux
Spring WebFlux 模块是 Spring Framework 5.0 版本的一部分。它是一个全新的非阻塞 web 框架，旨在取代 Spring MVC 来编写基于响应式流的 Web 应用。它利用了 Spring Framework 5.0 对 Reactive Stream 支持的改进，并结合了 Reactor Core 提供的异步非阻塞 IO 支持。

# 3.工程实践
首先需要创建 Spring Boot 项目，这里不再赘述。然后添加 spring-boot-starter-webflux依赖。在 application.properties 配置文件中设置如下属性值：
```
server.port=8080
spring.webflux.static-path-pattern=/resources/**
```
上面的配置让 Spring Boot 服务监听端口号为 8080，并配置静态资源访问路径为 /resources/** 。注意，这个配置只能匹配静态资源请求，不能匹配动态 Web 请求。接着创建一个RestController类，添加一个方法返回字符串"Hello World!"。
```java
@RestController
public class HelloWorldController {

    @GetMapping("/")
    public String hello() {
        return "Hello World!";
    }
    
}
```
启动 Spring Boot 项目，浏览器打开 http://localhost:8080/ 页面查看结果。如果出现"Hello World!"字样则证明服务正常启动。

## 3.1 添加 Reactor Netty 依赖
Reactor Netty 是 Spring Boot 默认使用的 Netty 服务器框架。因此，为了让我们的 Spring Boot 项目支持 Reactor Netty，只需添加如下 Maven 依赖：
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-reactor-netty</artifactId>
</dependency>
```

## 3.2 配置 Reactor Server HTTP Handler Adapter
默认情况下，Spring Boot 会注册一个 Reactor Server HTTP Handler Adapter。但是，因为没有配置 static-path-pattern 属性，所以会发生以下错误：
```log
Error starting ApplicationContext. To display the conditions report re-run your application with 'debug' enabled.
2019-07-12 23:28:21.786 ERROR [localhost-startStop-1] o.s.b.SpringApplication - Application run failed
org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'httpHandlerAdapter': Lookup method resolution failed; nested exception is java.lang.IllegalStateException: Failed to introspect Class [org.springframework.boot.autoconfigure.web.reactive.HttpHandlerAutoConfiguration$ReactorNettyHttpHandlerConfiguration] from ClassLoader [sun.misc.Launcher$AppClassLoader@18b4aac2]
...
Caused by: org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'webFluxWebServerFactory' defined in class path resource [org/springframework/boot/autoconfigure/web/reactive/ReactiveWebServerFactoryAutoConfiguration$DefaultReactiveWebServerFactoryCustomizerConfiguration.class]: Bean instantiation via factory method failed; nested exception is org.springframework.beans.BeanInstantiationException: Failed to instantiate [org.springframework.boot.web.embedded.netty.NettyWebServerFactory]: Factory method 'webFluxWebServerFactory' threw exception; nested exception is java.lang.IllegalArgumentException: Cannot start a reactor server on port x because it's already in use.
```
解决方案是在配置文件中添加如下属性值：
```
spring.main.web-application-type = reactive
spring.webflux.base-path=/api
spring.webflux.static-path-pattern=/resources/**
```
上面这两个属性值的含义分别是：

1. `spring.main.web-application-type` : 设置 Spring Boot 项目的 Web 类型为响应式 Web。

2. `spring.webflux.base-path` : 设置应用程序的基础路由前缀。默认情况下，应用程序根目录上下文映射到的控制器方法的 URL 为 `/`。设置此属性后，控制器方法的 URL 将以 `/api` 为前缀。例如，`/hello` 会变成 `/api/hello`。

3. `spring.webflux.static-path-pattern` : 设置静态资源请求的路径模式。默认情况下，Spring Boot 使用 Undertow 作为静态资源服务器。由于 Undertow 不支持对异步 IO 的支持，因此无法使用异步非阻塞的方式读取静态资源。所以，我们需要使用其他支持异步 IO 的 Web 服务器。本例使用的是 Netty 。我们可以指定请求静态资源的路径模式，让 Netty 只拦截这些请求。这样就不会尝试去查找不存在的文件，也不会影响 WebFlux 的正常工作。

## 3.3 创建异步 Rest API 方法
下面，我们可以创建异步 Rest API 方法了。我们修改 HelloWorldController 类的代码如下：
```java
@RestController
public class HelloWorldController {

    private final Executor executor = Executors.newCachedThreadPool();
    
    @GetMapping("/async")
    public Mono<String> asyncHello() {
        return Mono.fromSupplier(() -> "Async Hello World!").subscribeOn(Schedulers.fromExecutor(executor));
    }
    
}
```
在这里，我们创建了一个线程池执行异步任务。我们使用 subscribeOn 方法指定该任务应该被哪个 Scheduler 执行。Scheduler 是 Reactor Core 中的一类重要概念，它负责计划任务并管理执行它们。这里，我们调用 Schedulers.fromExecutor 方法创建一个新 Scheduler，传入刚才创建的线程池。

我们创建了一个名叫 asyncHello 的新方法，它返回一个 Mono 对象。Mono 是 Reactor Core 中的一个概念，代表单项数据流。这里，我们调用了 Mono.fromSupplier 方法来生成一个包含 String 数据的 Mono 对象。

最后，我们使用 subscribeOn 方法指定该 Mono 对象应该被由我们指定的线程池执行。

## 3.4 测试异步 Rest API 方法
启动 Spring Boot 项目，浏览器打开 http://localhost:8080/async 页面查看结果。如果出现"Async Hello World!"字样则证明服务正常启动且异步 Rest API 方法成功运行。