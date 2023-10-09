
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Feign 是 Spring Cloud 提供的一个声明式的 HTTP 客户端。它使得编写 web service 客户端变得更加容易，只需要创建一个接口并添加注解即可，通过 Feign 的处理，你的代码将会和远程服务对接成为调用本地方法一样简单。Spring Cloud 封装了 Ribbon 和 Eureka 来实现服务发现，通过负载均衡的方式调用服务。但是，Feign 也具有以下优点:

1. 支持Spring Cloud组件，如配置管理（Config）、服务注册中心（Eureka）、熔断机制（Hystrix），降低了开发难度；
2. 支持RESTful请求，可以灵活地映射HTTP请求到RPC方法上；
3. 通过注释简化了服务调用逻辑，例如服务名、超时时间、重试次数等；
4. 可插拔的注解驱动方式，方便集成其他框架；

虽然 Feign 可以用于简单的调用场景，但是在复杂的微服务架构下，它仍然是一个非常重要的工具。 

RestTemplate 是 Spring Framework 提供的用于访问 REST 服务的客户端模板类，它提供一个抽象层次的方法库，用来定义底层HTTP通信的通用行为。使用 RestTemplate 可以很好地访问远程RESTful服务，并且提供了多种丰富的方法，可以直接发送各种HTTP请求，包括GET、POST、PUT、DELETE等。

本文将通过对比两种技术的基本特性、使用场景、示例、优缺点进行阐述，以及如何选择适合自己业务需求的工具。

# 2.核心概念与联系
## 2.1 Feign
### 2.1.1 基本特性
Feign是一个声明式的Web服务客户端，它整合了Ribbon和Hystrix，简化了微服务之间的调用。

#### (1) Feign客户端注解

Feign允许使用注解来定义Feign客户端，通过该注解声明目标方法和调用参数，不需要指定实现类，Feign根据注解自动生成对应的HTTP客户端接口。

```java
@FeignClient(value = "service-name", fallback = ServiceClientFallbackImpl.class)
public interface ServiceClient {

    @RequestMapping(method = RequestMethod.GET, value = "/{path}")
    ResponseEntity<String> get(@PathVariable("path") String path);
    
}
```

Feign客户端注解@FeignClient中通过value属性指定目标服务名称，如果服务调用失败或者超时，则通过fallback属性指定的回调类ServiceFallbackImpl响应错误信息。

#### (2) 支持REST客户端

Feign内置的支持REST客户端，可通过注解@RequestMapping的方法指定HTTP请求方法类型、路径及请求头部等。

```java
@RequestLine("GET /services/{id}")
ResponseEntity<Object> findById(@Param("id") int id);
```

#### (3) 支持负载均衡

Feign默认情况下支持负载均衡，通过Ribbon模块与服务发现组件结合，可自动从服务注册中心获取服务列表，并基于轮询负载均衡策略调用服务端点。

#### (4) 容错机制

Feign内置了Hystrix容错机制，可通过注解开启熔断功能，并通过线程隔离、断路器等方式保护目标服务，防止出现级联故障导致整个系统不可用。

#### (5) 支持缓存

Feign允许设置缓存注解，当方法被调用时，Feign首先检查是否有对应的缓存数据，如果有则直接返回缓存数据，否则才会真正执行方法，并将结果写入缓存。

#### (6) 配置管理

Feign通过@EnableFeignClients注解激活Feign组件，并通过配置管理组件（比如Spring Cloud Config或Consul）动态配置Feign客户端的相关属性。

### 2.1.2 使用场景
Feign主要应用于Spring Cloud体系中的微服务架构中，在不引入独立的API Gateway组件的前提下，直接通过HTTP接口调用各个服务，因此其使用场景比较广泛。Feign可以作为RPC客户端，也可以作为HTTP客户端，比如做爬虫时的模拟浏览器访问网站。

Feign也可以作为消息总线客户端，通过注解指定订阅或发布事件，可用于事件驱动架构下的微服务间通信。

## 2.2 RestTemplate
### 2.2.1 基本特性
RestTemplate是一个轻量级的、面向JAVA的HTTP客户端，它提供了一系列的方法用来发送各种类型的HTTP请求。
#### (1) 模板方法模式

RestTemplate采用的是模板方法设计模式，即把一些通用的逻辑，比如发送HTTP请求、处理响应，通过一个公共的父类AbstractTemplate方法来完成。

#### (2) URI变量

可以使用URI模板，通过占位符{varName}来表示要替换的参数，然后在调用的时候传入相应的值。

#### (3) 编码器

支持众多的编解码器，包括JSON编码器、XML编码器、字符串编码器、表单编码器等，还可以通过定制Encoder和Decoder来实现自定义的编码解码过程。

#### (4) 支持文件上传/下载

可以使用MultipartFile对象来完成文件的上传和下载。

#### (5) 支持HTTP消息转换器

可通过MessageConverters对HTTP消息进行序列化和反序列化，目前已内置对JSON、XML、表单、字符串、流、JavaBean、文件上传的支持。

#### (6) 请求响应日志记录

可通过设置请求/响应过滤器，以输出请求响应的日志信息。

### 2.2.2 使用场景
RestTemplate的使用场景主要分为两类：短期的请求/响应调用、长期的异步非阻塞请求。
短期调用场景：比如与单个服务的交互，短时间内只调用一次接口。这种场景下，使用Feign或RestTemplate都能较好的满足需求。

长期异步非阻塞请求场景：比如多个服务间的交互，并发量大、耗时长。这种场景下，建议使用Reactor模式，即将Http请求封装为异步任务，交给Reactor线程池去执行，从而避免同步等待带来的性能瓶颈。