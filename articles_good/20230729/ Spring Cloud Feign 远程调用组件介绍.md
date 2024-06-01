
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　Feign是一个声明式WebService客户端，它使得编写web service客户端变得非常容易。Feign集成了Ribbon，利用Ribbon可以基于负载均衡策略配置RestTemplate。通过注解的方式来定义接口，就像定义本地接口一样，然后由Feign创建出实现Webservice接口的HTTP客户端。在使用Feign时，只需定义服务名和对应的请求方法即可，至于如何连接到Webservice服务器这一过程则交给了Ribbon来管理。Spring Cloud Feign的主要优点包括：
         1. 使用简单。只需要添加依赖spring-cloud-starter-feign，然后按照注解定义自己的API接口，就可以轻松地调用其他微服务提供的RESTful API。
         2. 支持负载均衡。可以基于Ribbon的负载均衡功能，实现微服务之间的负载均衡。
         3. 降低耦合度。在使用Feign时，消费者不再需要关注底层的HTTP通讯细节，而是直接调用接口，屏蔽了各种网络传输、序列化等复杂性。
         4. 有利于接口的版本控制。可以基于URI或者其他属性对服务进行版本化。
         # 2.基本概念及术语说明
         1. Ribbon：负载均衡器组件，基于JAVA开发，可用于云端计算资源动态管理和故障切换。主要功能是在应用启动时，从注册中心获取服务列表并使用轮询，并动态地将服务请求发送给提供相同服务的机器，提高系统可用性。
         2. Hystrix：容错管理组件，用于处理分布式系统的延迟和异常，防止级联故障，避免整体雪崩效应，提升系统鲁棒性。
         3. RestTemplate：微服务之间通信的一种模板类，使用统一接口访问不同微服务，支持同步、异步方式调用。
         4. OpenFeign：支持Java注解的声明式REST客户端，使用Feign替换掉之前的客户端调用方式。
         5. URL：统一资源定位符（Uniform Resource Locator）用来描述互联网上某个资源的位置。
         6. 服务发现：服务发现模式，通过注册中心发现服务，一般包括DNS、ZooKeeper或Consul。
         7. RESTful API：遵循HTTP协议，使用URL对服务资源进行访问的一种设计风格。
         8. 客户端：微服务的消费方。
         9. 服务注册中心：服务注册中心，用来存储微服务的地址信息，方便其他微服务进行服务调用。
         # 3. 核心算法原理及操作步骤及数学公式讲解
         1. Feign组件工作流程示意图:
        
         
         2. 基于Feign调用服务流程图:

         
         3. Feign调用服务流程说明:

         * 用户调用Feign接口，Feign根据@FeignClient注解的值去注册中心查找微服务实例；
         * 通过ribbon向找到的微服务实例发起http请求；
         * 接收微服务响应结果后，Feign返回结果给用户。

         4. Feign接口方法参数支持以下类型：
           
           ```java
           @RequestLine("GET /{path}")
           List<Object> query(@Param(value = "path") String path); //查询
           void save(@Body Object object); //新增
           void update(@Body Object object); //修改
           void deleteById(@Param(value = "id") Integer id); //删除
           ```

         5. Feign注解:
           
          @FeignClient注解：指定微服务名称。
          @GetMapping注解：映射Get请求。
          @PostMapping注解：映射Post请求。
          @PutMapping注解：映射Put请求。
          @DeleteMapping注解：映射Delete请求。
          @RequestLine注解：指定http请求方式及url。
          
          * RequestLine注解中的方法中可以写{variable}，该变量会被绑定路径中的值。如@RequestLine("GET /users/{userId}/addresses")可以写成@RequestLine("GET /users/1/addresses"）。
          * Param注解：当@RequestParam没有指定name属性时，可以使用Param注解指定绑定参数。
          
          参数格式举例：
          
          ```java
          public interface MyService {
              @RequestLine("POST /greeting")
              Greeting sayHello (@Param("name") String name);
          }
          ```
          
          ```xml
          <dependency>
              <groupId>io.github.openfeign</groupId>
              <artifactId>feign-core</artifactId>
              <version>${feign.version}</version>
          </dependency>
          <dependency>
              <groupId>io.github.openfeign</groupId>
              <artifactId>feign-jackson</artifactId>
              <version>${feign.version}</version>
          </dependency>
          <dependency>
              <groupId>io.github.openfeign</groupId>
              <artifactId>feign-okhttp</artifactId>
              <version>${feign.version}</version>
          </dependency>
          <dependency>
              <groupId>io.github.openfeign</groupId>
              <artifactId>feign-slf4j</artifactId>
              <version>${feign.version}</version>
          </dependency>
          ```
          
          * Body注解：当请求体类型不是json时，可以使用@Body注解绑定请求对象。
          * Header注解：可以自定义http header。
          
          请求头举例：
          
          ```java
          @Headers({"Content-Type: application/json",
                  "Authorization: Bearer {access_token}"})
          public interface MyService {
              @RequestLine("POST /greeting")
              Greeting sayHello (@Param("name") String name);
          }
          ```
          
          返回值：
          
          ```java
          class Greeting {
              private final String message;

              public Greeting(String message) {
                  this.message = message;
              }

              public String getMessage() {
                  return message;
              }
          }
          ```
          
          * ResponseStatus注解：当接口调用失败时，可以通过ResponseStatus注解返回错误信息。
          
          案例：
          
          ```java
          public interface MyService {
              @RequestLine("POST /payment/createOrder")
              @Headers({
                      "Accept: application/json",
                      "Content-Type: application/json"})
              OrderResponse createOrder(@Body CreateOrderRequest request);
              
              @ResponseStatus(HttpStatus.BAD_REQUEST)
              @ExceptionHandler(PaymentException.class)
              PaymentErrorResponse handleBadRequestExceptions(PaymentException e);
          }
          ```
          
          当接口调用失败时，系统会抛出PaymentException，由@ExceptionHandler注解处理，并返回错误信息。
          
          * QueryMap注解：当查询参数比较多且不能用对象封装时，可以使用QueryMap注解指定参数。
          
          参数格式举例：
          
          ```java
          public interface MyService {
              @RequestLine("GET /payments")
              List<Payment> searchPayments(@QueryMap Map<String, Object> parameters);
          }
          ```
          
          * Verb注解：当@RequestMapping的method属性存在多个请求方式时，可以使用Verb注解单独指定请求方式。
          
          ```java
          public interface MyService {
              @RequestLine("POST /greeting")
              @Headers({"Content-Type: application/json"})
              @Verb("PUT")
              Greeting modifyGreeting (@Param("name") String name);
          }
          ```
         # 4. 具体代码实例和解释说明
         1. 创建服务项目hello-provider：
            
            pom.xml文件
            
            ```xml
            <?xml version="1.0" encoding="UTF-8"?>
            <project xmlns="http://maven.apache.org/POM/4.0.0"
                     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                     xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
                <modelVersion>4.0.0</modelVersion>
    
                <parent>
                    <groupId>org.springframework.boot</groupId>
                    <artifactId>spring-boot-starter-parent</artifactId>
                    <version>2.1.4.RELEASE</version>
                    <relativePath/> <!-- lookup parent from repository -->
                </parent>
    
                <groupId>cn.shishuihao</groupId>
                <artifactId>hello-provider</artifactId>
                <version>0.0.1-SNAPSHOT</version>
                <packaging>jar</packaging>
    
                <name>hello-provider</name>
                <description>Demo project for Spring Boot</description>
    
                <properties>
                    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
                    <project.reporting.outputEncoding>UTF-8</project.reporting.outputEncoding>
                    <java.version>1.8</java.version>
                    <spring-cloud.version>Greenwich.SR1</spring-cloud.version>
                </properties>
    
                <dependencies>
                    <dependency>
                        <groupId>org.springframework.boot</groupId>
                        <artifactId>spring-boot-starter-web</artifactId>
                    </dependency>
    
                    <dependency>
                        <groupId>org.springframework.boot</groupId>
                        <artifactId>spring-boot-starter-actuator</artifactId>
                    </dependency>
                    
                    <dependency>
                        <groupId>org.springframework.cloud</groupId>
                        <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
                    </dependency>
    
                    <dependency>
                        <groupId>org.springframework.boot</groupId>
                        <artifactId>spring-boot-starter-test</artifactId>
                        <scope>test</scope>
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
            
            
            HelloController.java文件：
            
            ```java
            package cn.shishuihao.demo.springcloud.feign.provider.controller;
            
            import org.springframework.beans.factory.annotation.Autowired;
            import org.springframework.web.bind.annotation.*;
            
            /**
             * @author shishuihao
             */
            @RestController
            public class HelloController {
            
                @Autowired
                private HelloRemote helloRemote;
                
                @GetMapping("/sayHello/{name}")
                public String sayHello(@PathVariable String name){
                    return helloRemote.sayHelloFromProvider(name);
                }
            }
            ```
            
            
            Application.java文件：
            
            ```java
            package cn.shishuihao.demo.springcloud.feign.provider;
            
            import org.springframework.boot.SpringApplication;
            import org.springframework.boot.autoconfigure.SpringBootApplication;
            import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
            import org.springframework.cloud.openfeign.EnableFeignClients;
    
            /**
             * @author shishuihao
             */
            @SpringBootApplication
            @EnableDiscoveryClient
            @EnableFeignClients(basePackages = {"cn.shishuihao.demo.springcloud.feign.api"})
            public class ProviderApplication {
            
                public static void main(String[] args) {
                    SpringApplication.run(ProviderApplication.class,args);
                }
            }
            ```
            
            关于@EnableFeignClients注解，是为了使Feign能扫描到定义的远程接口包cn.shishuihao.demo.springcloud.feign.api下的接口。
            
            2. 创建服务项目hello-consumer：
            
            pom.xml文件：
            
            ```xml
            <?xml version="1.0" encoding="UTF-8"?>
            <project xmlns="http://maven.apache.org/POM/4.0.0"
                     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                     xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
                <modelVersion>4.0.0</modelVersion>
    
                <parent>
                    <groupId>org.springframework.boot</groupId>
                    <artifactId>spring-boot-starter-parent</artifactId>
                    <version>2.1.4.RELEASE</version>
                    <relativePath/> <!-- lookup parent from repository -->
                </parent>
    
                <groupId>cn.shishuihao</groupId>
                <artifactId>hello-consumer</artifactId>
                <version>0.0.1-SNAPSHOT</version>
                <packaging>jar</packaging>
    
                <name>hello-consumer</name>
                <description>Demo project for Spring Boot</description>
    
                <properties>
                    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
                    <project.reporting.outputEncoding>UTF-8</project.reporting.outputEncoding>
                    <java.version>1.8</java.version>
                    <spring-cloud.version>Greenwich.SR1</spring-cloud.version>
                </properties>
    
                <dependencies>
                    <dependency>
                        <groupId>org.springframework.boot</groupId>
                        <artifactId>spring-boot-starter-web</artifactId>
                    </dependency>
    
                    <dependency>
                        <groupId>org.springframework.boot</groupId>
                        <artifactId>spring-boot-starter-actuator</artifactId>
                    </dependency>
                    
                    <dependency>
                        <groupId>org.springframework.cloud</groupId>
                        <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
                    </dependency>
                    
                    <dependency>
                        <groupId>org.springframework.cloud</groupId>
                        <artifactId>spring-cloud-starter-openfeign</artifactId>
                    </dependency>
                    
                    <dependency>
                        <groupId>cn.shishuihao</groupId>
                        <artifactId>hello-provider</artifactId>
                        <version>0.0.1-SNAPSHOT</version>
                    </dependency>
    
                    <dependency>
                        <groupId>org.springframework.boot</groupId>
                        <artifactId>spring-boot-starter-test</artifactId>
                        <scope>test</scope>
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
            
            
            HelloRemote.java文件：
            
            ```java
            package cn.shishuihao.demo.springcloud.feign.api;
            
            import feign.Headers;
            import feign.Param;
            import feign.RequestLine;
            
            /**
             * @author shishuihao
             */
            @Headers("X-Auth-Token: OAUTH-TOKEN")
            public interface HelloRemote {

                @RequestLine("GET /sayHelloFromProvider?name={name}")
                String sayHelloFromProvider(@Param("name") String name);

            }
            ```
            
            
            HelloController.java文件：
            
            ```java
            package cn.shishuihao.demo.springcloud.feign.consumer.controller;
            
            import cn.shishuihao.demo.springcloud.feign.api.dto.CreateOrderRequest;
            import cn.shishuihao.demo.springcloud.feign.api.dto.OrderResponse;
            import cn.shishuihao.demo.springcloud.feign.api.exception.PaymentException;
            import cn.shishuihao.demo.springcloud.feign.api.response.PaymentErrorResponse;
            import cn.shishuihao.demo.springcloud.feign.consumer.config.FeignConfig;
            import cn.shishuihao.demo.springcloud.feign.consumer.utils.JsonUtils;
            import com.fasterxml.jackson.databind.ObjectMapper;
            import org.springframework.beans.factory.annotation.Autowired;
            import org.springframework.cloud.context.config.annotation.RefreshScope;
            import org.springframework.http.HttpStatus;
            import org.springframework.http.ResponseEntity;
            import org.springframework.web.bind.annotation.*;
            import org.springframework.web.client.HttpClientErrorException;
            import org.springframework.web.client.HttpServerErrorException;
            import org.springframework.web.client.RestClientException;
            
            import java.util.HashMap;
            import java.util.Map;
            
            /**
             * @author shishuihao
             */
            @RestController
            @RefreshScope
            public class HelloController {
            
                private static final ObjectMapper OBJECT_MAPPER = JsonUtils.getObjectMapper();
            
                @Autowired
                private HelloRemote helloRemote;
            
                @GetMapping("/order")
                public ResponseEntity order(){
                    try{
                        CreateOrderRequest request=new CreateOrderRequest();
                        request.setUserId("1");
                        request.setOrderId("1");
                        request.setItemId("1");
                        OrderResponse response=helloRemote.createOrder(request);
                        if (response==null){
                            throw new HttpServerErrorException(HttpStatus.INTERNAL_SERVER_ERROR,"no response");
                        }else{
                            return ResponseEntity.ok(response);
                        }
                    }catch (HttpClientErrorException ex){
                        Map<String,Object> body=OBJECT_MAPPER.readValue(ex.getResponseBodyAsString(), HashMap.class);
                        String errorMsg=(String)body.get("error");
                        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(new PaymentErrorResponse(errorMsg));
                    }catch (RestClientException ex){
                        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
                    }
                }
            }
            ```
            
            FeignConfig.java文件：
            
            ```java
            package cn.shishuihao.demo.springcloud.feign.consumer.config;
            
            import feign.Logger;
            import feign.codec.Decoder;
            import feign.codec.Encoder;
            import org.springframework.beans.factory.ObjectFactory;
            import org.springframework.beans.factory.annotation.Value;
            import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
            import org.springframework.boot.autoconfigure.http.HttpMessageConverters;
            import org.springframework.cloud.commons.httpclient.ApacheHttpClientConnectionManagerFactory;
            import org.springframework.cloud.openfeign.support.ResponseEntityDecoder;
            import org.springframework.cloud.openfeign.support.SpringDecoder;
            import org.springframework.cloud.openfeign.support.SpringEncoder;
            import org.springframework.context.annotation.Bean;
            import org.springframework.context.annotation.Configuration;
            import org.springframework.http.converter.HttpMessageConverter;
            import org.springframework.http.converter.json.Jackson2ObjectMapperBuilder;
            
            import javax.annotation.Resource;
            import java.util.ArrayList;
            import java.util.List;
            
            /**
             * @author shishuihao
             */
            @Configuration
            public class FeignConfig {
            
                @Resource
                private ApacheHttpClientConnectionManagerFactory connectionManagerFactory;
            
                @Value("${feign.logger.level:#{null}}")
                Logger.Level logLevel;
            
                @Bean
                public Encoder encoder(){
                    return new SpringEncoder(getMessageConverters());
                }
            
                @Bean
                public Decoder decoder(){
                    return new ResponseEntityDecoder(new SpringDecoder(getMessageConverters()));
                }
            
                protected List<HttpMessageConverter<?>> getMessageConverters() {
                    Jackson2ObjectMapperBuilder builder = new Jackson2ObjectMapperBuilder();
                    builder.failOnUnknownProperties(false);
                    return new ArrayList<>(builder.build().getConverters());
                }
            
                @Bean
                @ConditionalOnMissingBean
                public Logger.Level loggerLevel() {
                    return logLevel!= null? logLevel : Logger.Level.NONE;
                }
            
                @Bean
                public ObjectFactory<HttpMessageConverters> messageConverters() {
                    final List<HttpMessageConverter<?>> converters = getMessageConverters();
                    return () -> new HttpMessageConverters(converters);
                }
            }
            ```
            
            
            在配置文件application.yml中添加如下配置：
            
            ```yaml
            server:
              port: 8081
            spring:
              application:
                name: hello-consumer
            eureka:
              client:
                serviceUrl:
                  defaultZone: http://localhost:${server.port}/eureka/
            feign:
              hystrix:
                enabled: true   # 是否开启熔断机制
              okhttp:
                enabled: false  # 默认为true，是否使用OkHttp作为底层HTTP客户端
              httpclient:
                connectTimeout: ${timeout:10000}    # 设置连接超时时间，单位ms
                readTimeout: ${timeout:60000}        # 设置读取超时时间，单位ms
                loggerLevel: basic                  # 设置日志级别
              client:
                config:
                  default:                      # 默认配置
                    connectTimeout: ${timeout:10000}      # 设置连接超时时间，单位ms
                    readTimeout: ${timeout:60000}          # 设置读取超时时间，单位ms
                  testClient:                    # 指定客户端配置
                    loggerLevel: headers            # 设置日志级别
              compression:                        # 压缩配置
                request:
                  enabled: true                   # 是否开启请求压缩
                response:
                  enabled: true                   # 是否开启响应压缩
              sofa:                                # sofa-rpc配置
                registry:                         # 注册中心配置
                  address: tcp://127.0.0.1:9603     # 连接地址
                consumer:                          # 消费者配置
                  timeout: 3000                     # 请求超时时间，单位ms
            ```
            
            
            最后启动服务项目hello-provider和hello-consumer，测试Feign远程调用。