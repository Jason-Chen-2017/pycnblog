
作者：禅与计算机程序设计艺术                    

# 1.简介
  


## 什么是Ribbon？
Ribbon是一个基于Netflix公司开源的客户端负载均衡器，它是一个独立的模块，也可以集成在Spring Cloud项目中使用。Ribbon可以做到：

1. **Client-side load balancing**：Ribbon提供了客户端的负载均衡，即在使用Ribbon进行服务调用时，由Ribbon根据相应策略（如Round Robin、Random）将请求分布到不同的服务实例上；
2. **Failover**：当某台机器宕机或网络不通的时候，Ribbon会自动切换至其他可用的服务实例；
3. **Retry**：由于服务调用过程中可能存在各种异常情况，比如超时、连接失败等，Ribbon提供相应的重试机制，可以自动对某些特定的异常状况进行重试；
4. **Caching**：Ribbon支持缓存，在同一个JVM进程内可以缓存已经获取到的服务信息，避免多次访问服务注册中心；

本文主要介绍如何使用Ribbon作为客户端负载均衡组件来实现微服务的负载均衡，并讨论其工作原理。

## 基本用法及原理

### 引入依赖

在使用Ribbon之前，需要先在工程的pom文件中引入Ribbon相关的依赖：

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-ribbon</artifactId>
</dependency>
```

### 配置文件

Ribbon配置包括两部分：

- 服务发现：用于指定要访问的微服务地址，目前支持静态配置（application.yml/properties）和服务发现系统（Eureka）。
- 负载均衡策略：定义了使用哪种负载均衡策略，例如轮询（RoundRobinRule），随机（RandomRule），加权（WeightedResponseTimeRule）等。

在配置文件中，通过如下方式进行配置：

#### Static Configuration（静态配置）

这是最简单的方式，直接配置服务地址即可，如：

```yaml
server:
  port: 8081 # The port of the application server

eureka:
  client:
    serviceUrl:
      defaultZone: http://localhost:${port}/eureka/
```

#### Eureka Integration（Eureka集成）

这种模式下，只需要在配置文件中设置eureka的url即可，如：

```yaml
server:
  port: 8081 # The port of the application server

eureka:
  client:
    serviceUrl:
      defaultZone: http://localhost:${port}/eureka/
```

然后，在启动类上添加注解@EnableEurekaClient，表示该应用是一个Eureka客户端，同时还需将依赖加入POM文件。

### Feign集成

Feign是一个声明式Web Service客户端，它使得编写Web Service客户端更加容易。Feign可以整合Ribbon，从而让Ribbon来帮助Feign实现客户端负载均衡功能。

在pom文件中加入依赖：

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-openfeign</artifactId>
</dependency>
```

配置好eureka后，Feign可以像使用接口一样调用远程服务，Feign自动选择合适的Ribbon负载均衡策略，具体示例如下：

```java
@FeignClient(value = "service-provider", fallback = HelloServiceFallback.class)
public interface HelloServiceClient {

    @RequestMapping("/hello")
    String hello(@RequestParam("name") String name);

}

// 使用fallback方法处理服务不可达的情况
class HelloServiceFallback implements HelloServiceClient {

    public String hello(String name) {
        return "Hello," + name + ",sorry,error occured!";
    }

}
```

这样，通过Feign+Ribbon，就可以很方便地调用远程服务的方法。


## 源码解析

Ribbon源码位置：`spring-cloud-netflix-core\src\main\java\org\springframework\cloud\netflix\ribbon\RibbonLoadBalancerClient`。

### 工作流程

Ribbon的工作流程如下图所示：


- 当第一次请求某个微服务时，通过Ribbon发送请求，Ribbon会通过负载均衡算法选取其中一个微服务节点，并将请求发送给这个节点。
- 如果此节点出现错误或者超时，Ribbon会自动切换至另一个可用节点，继续执行请求过程。
- 在Ribbon的负载均衡算法中，一般采用“轮询”和“加权”两种算法。

### 流程细节

#### 自定义负载均衡规则

如果希望实现自己的负载均衡策略，可以在配置文件中通过`loadbalancer.rule.name`属性指定要使用的负载均衡策略。如，如果希望自己定义的负载均衡策略名叫MyOwnRule，则可以在配置文件中这样配置：

```yaml
spring:
  cloud:
    loadbalancer:
      rule:
        name: MyOwnRule
```

然后，在自定义的Rule类中实现`ServerListFilter`，并把自己的负载均衡逻辑写进去，如：

```java
import java.util.*;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Component;

import com.netflix.client.config.IClientConfig;
import com.netflix.loadbalancer.*;

@Component
@ConditionalOnProperty(name="spring.cloud.loadbalancer.rule.name", havingValue="MyOwnRule")
public class MyOwnRule extends AbstractServerListFilteringRule {

    private final ServerList serverList;

    @Autowired
    public MyOwnRule(IClientConfig config, IPing myPing, IRule myRule, Predicate predicate) {
        this.serverList = new ConfiguredServerList(config);
    }

    // 自定义负载均衡逻辑
    protected List<Server> filteredListOfServers(List<Server> servers) {
        // TODO implement your own logic here...
       ...

        return filteredServers;
    }

    @Override
    public void initWithNiwsConfig(IClientConfig iClientConfig) {
        super.initWithNiwsConfig(iClientConfig);
    }

}
```

#### 自定义Ribbon状态指标收集器

Ribbon默认会采集各个服务器（微服务实例）的一些状态信息，例如成功请求数、出错次数等。如果希望自定义自己的状态指标收集器，可以在配置文件中通过`ribbon.metrics.name`属性指定要使用的状态指标收集器。如，如果希望自己定义的状态指标收集器名叫MyMetricsCollector，则可以在配置文件中这样配置：

```yaml
spring:
  ribbon:
    metrics:
      name: MyMetricsCollector
```

然后，在自定义的MetricsCollector类中实现`IRibbonMetricsCollector`，并把自己的状态指标收集逻辑写进去，如：

```java
import javax.inject.Inject;

import com.netflix.client.ClientException;
import com.netflix.client.AbstractLoadBalancerAwareClient.LifecycleState;
import com.netflix.client.http.HttpRequest;
import com.netflix.client.http.HttpResponse;
import com.netflix.niws.client.http.RestClient;

public class MyMetricsCollector implements IRibbonMetricsCollector {

    private RestClient restClient;

    @Inject
    public MyMetricsCollector(RestClient restClient) {
        this.restClient = restClient;
    }

    @Override
    public void markSuccess() {
        LifecycleState lifecycleState = getCurrentRequest().getUriInfo().getProperty(
                AbstractLoadBalancerAwareClient.NF_LOADBALANCER_REQUEST_LCS_STATE);
        if (lifecycleState == null ||!lifecycleState.equals(LifecycleState.DISCONNECTED)) {
            HttpRequest httpRequest = getRequestForCurrentService();
            incrementCounters(httpRequest, true);
        }
    }

    @Override
    public void markFailure(Throwable e) {
        LifecycleState lifecycleState = getCurrentRequest().getUriInfo().getProperty(
                AbstractLoadBalancerAwareClient.NF_LOADBALANCER_REQUEST_LCS_STATE);
        if (lifecycleState == null ||!lifecycleState.equals(LifecycleState.DISCONNECTED)) {
            HttpRequest httpRequest = getRequestForCurrentService();
            incrementCounters(httpRequest, false);
        }
    }

    private void incrementCounters(HttpRequest request, boolean success) {
        try {
            HttpResponse response = restClient.executeWithLoadBalancer(request).toBlocking().single();
            if (success && response!= null && response.getStatusCode() >= 200 && response.getStatusCode() < 300) {
                addCount(AbstractLoadBalancerAwareClient.LB_TOTAL_SUCCESS_COUNTER_NAME,
                        AbstractLoadBalancerAwareClient.DEFAULT_COUNTER_INCREMENT);
                addCount(AbstractLoadBalancerAwareClient.SERVICE_CLIENT_TOTAL_SUCCESS_COUNTER_PREFIX + "_"
                                + extractServiceNameFromRequestURI(request),
                        AbstractLoadBalancerAwareClient.DEFAULT_COUNTER_INCREMENT);
            } else {
                addCount(AbstractLoadBalancerAwareClient.LB_TOTAL_FAILURE_COUNTER_NAME,
                        AbstractLoadBalancerAwareClient.DEFAULT_COUNTER_INCREMENT);
                addCount(AbstractLoadBalancerAwareClient.SERVICE_CLIENT_TOTAL_FAILURE_COUNTER_PREFIX + "_"
                                + extractServiceNameFromRequestURI(request),
                        AbstractLoadBalancerAwareClient.DEFAULT_COUNTER_INCREMENT);
            }
        } catch (Exception ex) {
            addCount(AbstractLoadBalancerAwareClient.LB_TOTAL_FAILURE_COUNTER_NAME,
                    AbstractLoadBalancerAwareClient.DEFAULT_COUNTER_INCREMENT);
            addCount(AbstractLoadBalancerAwareClient.SERVICE_CLIENT_TOTAL_FAILURE_COUNTER_PREFIX + "_"
                            + extractServiceNameFromRequestURI(request),
                    AbstractLoadBalancerAwareClient.DEFAULT_COUNTER_INCREMENT);
        }
    }

    private void addCount(String key, int delta) throws Exception {
        // TODO use your preferred data store to store counters
        System.out.println(key + ": " + delta);
    }

    private static HttpRequest getRequestForCurrentService() {
        return ThreadLocalContext.getCurrentRequest().getRequestConfig().getLoadBalancerKey().getLoadBalancer()
                                                                                                  .chooseServer();
    }

    private static String extractServiceNameFromRequestURI(HttpRequest request) {
        return request.getUri().getPath().split("/", 2)[1];
    }

}
```

注意，如果不想使用Ribbon默认的状态指标收集器，可以在配置文件中将其关闭：

```yaml
spring:
  ribbon:
    enabled: false
```