
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         在微服务架构模式下，由于各个微服务之间耦合度低、部署独立性强、独立运行等特点，导致各个微服务的状态同步变得困难，需要借助消息总线实现状态的同步。Spring Cloud提供的消息总线中间件包括Spring Cloud Stream和Spring Cloud Bus。Spring Cloud Stream是用于构建事件驱动微服务应用的统一消息流（messaging middleware），可以轻松将应用程序中的数据流动变成可靠的消息，并通过消息代理（message broker）传输到另一个应用程序中去。而Spring Cloud Bus则是一个用于实现分布式系统间通信的消息总线工具，基于Spring Boot Admin作为网页管理界面，它能够监控集群中所有服务节点，展示服务健康状况。这篇文章主要介绍Spring Cloud Bus消息总线中间件的用法和实际案例。
         
         # 2.基本概念术语说明
         
         ## 2.1 Spring Cloud Stream
         
         Spring Cloud Stream是用于构建事件驱动微服务应用的统一消息流（messaging middleware）。它提供了一套简单易用的API来发送和接收各种不同消息类型的事件。Spring Cloud Stream的主要优势如下：
         
         * 为微服务应用的开发者提供了声明式编程模型
         * 支持多种消息中间件，如Kafka，RabbitMQ等
         * 内置了很多实用的消息组件，如持久化，反序列化，编解码器等模块
         * 提供高度可扩展的弹性伸缩能力
         
         ### 2.1.1 流和绑定器
         
         Spring Cloud Stream支持两种主要的消息传递模型——流和绑定器。顾名思义，流就是一系列的消息在连接之间的传递过程，而绑定器则是一种绑定两个流或者通道的方法。典型的流包括有关某个主题的所有消息，而绑定器则通过一组路由规则将消息从一个流或通道转发到另一个流或通道。流可以由多个应用程序组成，每个应用程序都可以向同一个主题发送消息。
         
        ![streams-and-bindings](https://www.springcloudstream.io/img/spring_cloud_stream_architecture.png)
         
         上图展示了Spring Cloud Stream的流和绑定器架构。流可以是任何类型的消息，例如订单信息，事件日志，传感器读ings等等；绑定器则定义了如何在不同的源和目标之间传递消息。通过绑定器，可以将多个应用程序的输出路由到单个共享的消息代理（如Apache Kafka）上，或者在同一个应用程序的多个实例之间进行负载均衡。
         
         ### 2.1.2 消息驱动函数即服务（Message Driven Function as a Service， MaaS）
         
         当微服务架构逐渐成为主流时，企业的开发团队也希望能够充分利用微服务的好处，即简单、灵活、可扩展性强等等。MaaS就是这样一种新的编程模型，它允许开发人员编写函数，这些函数可以订阅某些消息主题并处理消息。然后，这些函数被编排到一个容器中，让它们能够像服务一样按需伸缩。当微服务的数量增长到一定规模后，MaaS也可以有效地管理整个系统。
         
         MaaS通常采用无服务器架构，其中平台即服务（PaaS）提供基础设施，而消息代理则承担着消息传递的角色。这使得微服务和消息传递机制非常集成，并且可以更好地协调工作流程。另外，由于消息驱动的特性，微服务也可以将自己的功能扩展到不同的消息代理之上。
         
         ### 2.1.3 消息消费者
         
         消费者（consumer）是指在消息队列（message queue）中读取并处理消息的实体。消费者可以是应用程序，如消息消费者，也可以是后台进程，如定时任务调度程序。消费者监听着特定的主题（topic），只要有新消息到达，它就会按照一定的协议对其进行处理。一般情况下，消息消费者都会反复尝试读取消息直到成功为止，确保消息被完整地处理。
         
         消息消费者可以采用两种不同的方式来处理消息：批处理（batch processing）和流处理（streaming processing）。对于批处理，消费者一次性读取多个消息并对其进行处理。对于流处理，消费者只会等待接收新消息，并立即处理消息。
         
         ## 2.2 Spring Cloud Bus
         
         Spring Cloud Bus是Spring Cloud的一个子项目，它是一个用于实现分布式系统间通信的消息总线工具。Spring Cloud Bus通过AMQP协议与消息代理建立通信，提供自动注册、发现及消息传递功能。Spring Cloud Bus可以帮助你实现分布式系统的最终一致性。本质上来说，Spring Cloud Bus是一个轻量级的分布式事件总线，它可以用来广播状态变化，通知其它系统有哪些服务发生了变化，让分布式微服务架构中的各个微服务间能够互相通信。
         
        ![bus-messaging](https://docs.spring.io/spring-cloud-static/docs/current/reference/htmlsingle/images/scb-bus-messaging.jpg)
         
         Spring Cloud Bus由三个主要的部分构成：发布端（producer），订阅端（subscriber）和消息代理（message broker）。发布端和订阅端分别位于不同的微服务或应用程序中，它们订阅或发布消息到一个指定的总线上。消息代理负责存储消息，并将消息发送给订阅端。当订阅端启动或恢复时，它可以订阅消息总线上的指定主题。Spring Cloud Bus还可以根据需要设置消息过滤器，以便只传递感兴趣的消息。
         
         # 3.核心算法原理和具体操作步骤以及数学公式讲解
         
         本章节没有相关的内容，所以暂时略过。
         
         # 4.具体代码实例和解释说明
         
        ## 4.1 工程配置说明

         1. 添加依赖：
             ```xml
                <dependency>
                    <groupId>org.springframework.boot</groupId>
                    <artifactId>spring-boot-starter-webflux</artifactId>
                </dependency>

                <!--引入bus依赖-->
                <dependency>
                    <groupId>org.springframework.cloud</groupId>
                    <artifactId>spring-cloud-bus</artifactId>
                    <version>${spring-cloud.version}</version>
                </dependency>
            ```

         2. 配置文件 application.yml 文件:
              ```yaml
                  spring:
                      cloud:
                          bus:
                              trace:
                                  enabled: true #开启追踪日志
                              refresh:
                                enabled: false #关闭刷新
​
                  server:
                      port: 9090 #web服务端口
              ```

        ## 4.2 服务消费方

          1. 创建消费端模块 service-order-client
          2. pom.xml 文件依赖配置

              ```xml
                   <dependencies>
                       <dependency>
                           <groupId>org.springframework.boot</groupId>
                           <artifactId>spring-boot-starter-webflux</artifactId>
                       </dependency>

                       <dependency>
                           <groupId>org.springframework.boot</groupId>
                           <artifactId>spring-boot-starter-actuator</artifactId>
                       </dependency>

                       <!--bus依赖-->
                       <dependency>
                           <groupId>org.springframework.cloud</groupId>
                           <artifactId>spring-cloud-starter-bus-amqp</artifactId>
                       </dependency>

                       <dependency>
                           <groupId>org.springframework.boot</groupId>
                           <artifactId>spring-boot-starter-test</artifactId>
                           <scope>test</scope>
                       </dependency>
                   </dependencies>
               </project>
           ```

          3. yml配置文件

               ```yaml
                   management:
                     endpoints:
                        web:
                          exposure:
                            include: 'health,bus-refresh'
                   logging:
                     level:
                       org.springframework.cloud.bus: debug
               ```

          4. 添加 controller
          
              ```java
                  @GetMapping("/hello")
                  public Mono<String> hello() {
                      return Mono.just("Hello World!");
                  }
              ```

        ## 4.3 服务提供方

          1. 创建服务提供方模块 service-order-provider
          2. pom.xml文件依赖配置

            ```xml
                 <dependencies>
                     <dependency>
                         <groupId>org.springframework.boot</groupId>
                         <artifactId>spring-boot-starter-webflux</artifactId>
                     </dependency>

                     <dependency>
                         <groupId>org.springframework.boot</groupId>
                         <artifactId>spring-boot-starter-actuator</artifactId>
                     </dependency>

                     <!--bus依赖-->
                     <dependency>
                         <groupId>org.springframework.cloud</groupId>
                         <artifactId>spring-cloud-starter-bus-amqp</artifactId>
                     </dependency>

                     <dependency>
                         <groupId>org.springframework.boot</groupId>
                         <artifactId>spring-boot-starter-test</artifactId>
                         <scope>test</scope>
                     </dependency>
                 </dependencies>
             </project>
         ```

        3. yml配置文件

             ```yaml
                 server:
                   port: 7070
                   servlet:
                     context-path: /serviceOrderProvider

                 management:
                   endpoints:
                     web:
                       exposure:
                         include: 'health,bus-refresh'
                   endpoint:
                     health:
                       show-details: always

                   bus:
                     id: order-provider #服务名称
                     serviceId: order-provider #暴露服务id
                     # destination: default #指定队列名称
                 logging:
                   level:
                     root: INFO
                     com.example: DEBUG
             ```

        4. controller

            ```java
                @RestController
                public class OrderController {

                    private final Logger logger = LoggerFactory.getLogger(this.getClass());

                    /**
                     * 通过actuator的endpoint服务刷新bus
                     */
                    @RefreshScope //添加注解
                    @GetMapping("/refreshBus")
                    public String refreshBus(){

                        return "refresh";
                    }


                    /**
                     * 模拟业务逻辑，实现订单生成
                     * @return 返回订单号
                     */
                    @GetMapping("/createOrder/{orderId}")
                    public Mono<String> createOrder(@PathVariable("orderId") Integer orderId){

                        return Mono.fromSupplier(() -> {

                            try{
                                TimeUnit.SECONDS.sleep(2);

                                if (orderId % 2 == 0){
                                    throw new Exception();
                                }else{
                                    System.out.println("创建订单成功！"+orderId);
                                    return orderId.toString();
                                }
                            }catch (Exception e){
                                e.printStackTrace();
                                System.err.println("创建订单失败！");
                                return "";
                            }
                        });
                    }
                }
            ```

        5. 配置bootstrap.yml

              ```yaml
                  spring:
                    application:
                      name: ${vcap.application.name:${spring.config.name:${random.value}}}
                    cloud:
                      bus:
                        enabled: true
                        fail-fast: true
                        auto-startup: true
                        index: ${random.int} #随机分配角色
                        discovery:
                          register: true
                          locator:
                            enabled: true
                        mapping:
                          order-provider/**: #匹配消息总线路径
                            destination: order-queue #指定队列名称
                            group: orderGroup #指定分组名称
                            content-type: text/plain #消息类型
                      stream:
                        bindings:
                          input:
                            binder: rabbit #指定消息代理
                          output:
                            binder: rabbit #指定消息代理
                        default-binder: rabbit #默认消息代理
                  rabbitmq:
                    host: localhost #rabbitmq地址
                    username: guest #用户名
                    password: guest #密码
                    virtual-host: / #虚拟主机名
                  vcap:
                    application:
                      name: test-order-provider
              ```

        6. 配置Application类

              ```java
                  @SpringBootApplication
                  @EnableDiscoveryClient //使能服务发现
                  public class OrderProviderApplication implements CommandLineRunner {

                      private static final Logger LOGGER = LoggerFactory.getLogger(OrderProviderApplication.class);

                      public static void main(String[] args) {
                          ConfigurableApplicationContext run = SpringApplication.run(OrderProviderApplication.class, args);

                          RabbitHealthIndicator rabbitHealthIndicator = run.getBean(RabbitHealthIndicator.class);
                          while (!rabbitHealthIndicator.isAvailable()) {
                              LOGGER.info("wait for rabbit available... ");
                              Thread.sleep(1000);
                          }
                          LOGGER.info("rabbit available!");
                      }

                      /**
                       * 初始化方法
                       * @param args 参数列表
                       */
                      @Override
                      public void run(String... args) throws Exception {

                      }
                  }
              ```

     

    ## 4.4 测试

        1. 服务启动
        2. 请求测试，正常响应

        3. 停止服务
        4. 请求测试，请求超时，日志报错：

           ```
               ERROR o.s.c.a.AnnotationConfigApplicationContext - Application run failed
                                           java.lang.IllegalStateException: Failed to load ApplicationContext
                   at org.springframework.context.annotation.AnnotationConfigApplicationContext.onRefresh(AnnotationConfigApplicationContext.java:157)
                   at org.springframework.context.support.AbstractApplicationContext.refresh(AbstractApplicationContext.java:543)
                   at org.springframework.boot.web.reactive.context.ReactiveWebServerApplicationContext.refresh(ReactiveWebServerApplicationContext.java:66)
                   at org.springframework.boot.SpringApplication.refresh(SpringApplication.java:747)
                   at org.springframework.boot.SpringApplication.refreshContext(SpringApplication.java:397)
                   at org.springframework.boot.SpringApplication.run(SpringApplication.java:315)
                   at org.springframework.boot.SpringApplication.run(SpringApplication.java:1226)
                   at org.springframework.boot.SpringApplication.run(SpringApplication.java:1215)
                   at com.example.OrderProviderApplication.main(OrderProviderApplication.java:15)
               Caused by: org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'integrationFlowRegistrationCustomizer' defined in class path resource [org/springframework/cloud/stream/config/IntegrationAutoConfiguration$IntegrationFlowBindingRegistrar.class]: Bean instantiation via factory method failed; nested exception is org.springframework.beans.BeanInstantiationException: Failed to instantiate [org.springframework.cloud.stream.config.BindingServiceProperties$IntegrationFlowRegistrationCustomizer]: Factory method 'integrationFlowRegistrationCustomizer' threw exception; nested exception is org.springframework.amqp.AmqpIOException: java.net.ConnectException: Connection refused (Connection refused)
                   at org.springframework.beans.factory.support.ConstructorResolver.instantiate(ConstructorResolver.java:658)
                   at org.springframework.beans.factory.support.ConstructorResolver.instantiateUsingFactoryMethod(ConstructorResolver.java:486)
                   at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.instantiateUsingFactoryMethod(AbstractAutowireCapableBeanFactory.java:1334)
                   at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.createBeanInstance(AbstractAutowireCapableBeanFactory.java:1177)
                   at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.doCreateBean(AbstractAutowireCapableBeanFactory.java:564)
                   at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.createBean(AbstractAutowireCapableBeanFactory.java:524)
                   at org.springframework.beans.factory.support.AbstractBeanFactory.lambda$doGetBean$0(AbstractBeanFactory.java:335)
                   at org.springframework.beans.factory.support.DefaultSingletonBeanRegistry.getSingleton(DefaultSingletonBeanRegistry.java:234)
                   at org.springframework.beans.factory.support.AbstractBeanFactory.doGetBean(AbstractBeanFactory.java:333)
                   at org.springframework.beans.factory.support.AbstractBeanFactory.getBean(AbstractBeanFactory.java:208)
                   at org.springframework.beans.factory.support.DefaultListableBeanFactory.preInstantiateSingletons(DefaultListableBeanFactory.java:944)
                   at org.springframework.context.support.AbstractApplicationContext.finishBeanFactoryInitialization(AbstractApplicationContext.java:918)
                   at org.springframework.context.support.AbstractApplicationContext.refresh(AbstractApplicationContext.java:583)
                   at org.springframework.boot.web.reactive.context.ReactiveWebServerApplicationContext.refresh(ReactiveWebServerApplicationContext.java:66)
                   at org.springframework.boot.SpringApplication.refresh(SpringApplication.java:747)
                   at org.springframework.boot.SpringApplication.refreshContext(SpringApplication.java:397)
                   at org.springframework.boot.SpringApplication.run(SpringApplication.java:315)
                   at org.springframework.boot.SpringApplication.run(SpringApplication.java:1226)
                   at org.springframework.boot.SpringApplication.run(SpringApplication.java:1215)
                   at com.example.OrderProviderApplication.main(OrderProviderApplication.java:15)
               Caused by: org.springframework.beans.BeanInstantiationException: Failed to instantiate [org.springframework.cloud.stream.config.BindingServiceProperties$IntegrationFlowRegistrationCustomizer]: Factory method 'integrationFlowRegistrationCustomizer' threw exception; nested exception is org.springframework.amqp.AmqpIOException: java.net.ConnectException: Connection refused (Connection refused)
                   at org.springframework.beans.factory.support.SimpleInstantiationStrategy.instantiate(SimpleInstantiationStrategy.java:185)
                   at org.springframework.beans.factory.support.ConstructorResolver.instantiate(ConstructorResolver.java:653)
                  ... 24 common frames omitted
               Caused by: org.springframework.amqp.AmqpIOException: java.net.ConnectException: Connection refused (Connection refused)
                   at org.springframework.amqp.rabbit.connection.ConnectionFactoryUtils.getConnection(ConnectionFactoryUtils.java:100)
                   at org.springframework.amqp.rabbit.core.RabbitTemplate.doExecute(RabbitTemplate.java:224)
                   at org.springframework.amqp.rabbit.core.RabbitTemplate.execute(RabbitTemplate.java:214)
                   at org.springframework.amqp.rabbit.core.RabbitAdmin.declareExchange(RabbitAdmin.java:76)
                   at org.springframework.cloud.stream.binder.rabbit.properties.RabbitBinderConfigurationProperties.<init>(RabbitBinderConfigurationProperties.java:72)
                   at org.springframework.cloud.stream.binder.rabbit.config.RabbitStreamConfiguration.rabbitBinderConfigurationProperties(RabbitStreamConfiguration.java:67)
                   at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
                   at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
                   at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
                   at java.base/java.lang.reflect.Method.invoke(Method.java:566)
                   at org.springframework.beans.factory.support.SimpleInstantiationStrategy.instantiate(SimpleInstantiationStrategy.java:154)
                  ... 25 common frames omitted
               Caused by: java.net.ConnectException: Connection refused (Connection refused)
                   at java.base/sun.nio.ch.SocketChannelImpl.checkConnect(Native Method)
                   at java.base/sun.nio.ch.SocketChannelImpl.finishConnect(SocketChannelImpl.java:779)
                   at java.base/sun.nio.ch.SocketAdaptor.connect(SocketAdaptor.java:148)
                   at org.apache.activemq.transport.tcp.TcpTransport.doStart(TcpTransport.java:379)
                   at org.apache.activemq.util.ServiceSupport.start(ServiceSupport.java:55)
                   at org.apache.activemq.transport.tcp.TcpTransport.start(TcpTransport.java:366)
                   at org.apache.activemq.transport.TransportFilter.start(TransportFilter.java:64)
                   at org.apache.activemq.client.ActiveMQConnectionFactory.createActiveMQConnection(ActiveMQConnectionFactory.java:348)
                   at org.springframework.amqp.rabbit.connection.CachingConnectionFactory.createConnection(CachingConnectionFactory.java:248)
                   at org.springframework.amqp.rabbit.connection.CachingConnectionFactory.createBareChannel(CachingConnectionFactory.java:512)
                   at org.springframework.amqp.rabbit.connection.CachingConnectionFactory.getChannel(CachingConnectionFactory.java:481)
                   at org.springframework.amqp.rabbit.connection.CachingConnectionFactory.access$1300(CachingConnectionFactory.java:102)
                   at org.springframework.amqp.rabbit.connection.CachingConnectionFactory$ChannelFailoverCallback.createChannel(CachingConnectionFactory.java:1221)
                   at org.springframework.amqp.rabbit.listener.ConditionalRejectingErrorHandler.handleConsumerBatch(ConditionalRejectingErrorHandler.java:151)
                   at org.springframework.amqp.rabbit.listener.SimpleMessageListenerContainer.invokeListener(SimpleMessageListenerContainer.java:1344)
                   at org.springframework.amqp.rabbit.listener.SimpleMessageListenerContainer.doConsumeFromQueue(SimpleMessageListenerContainer.java:1266)
                   at org.springframework.amqp.rabbit.listener.SimpleMessageListenerContainer.receiveAndExecute(SimpleMessageListenerContainer.java:1250)
                   at org.springframework.amqp.rabbit.listener.SimpleMessageListenerContainer.access$1600(SimpleMessageListenerContainer.java:120)
                   at org.springframework.amqp.rabbit.listener.SimpleMessageListenerContainer$AsyncMessageProcessingConsumer.handleDelivery(SimpleMessageListenerContainer.java:1495)
                   at com.rabbitmq.client.impl.ConsumerDispatcher$5.run(ConsumerDispatcher.java:149)
                   at com.rabbitmq.client.impl.ConsumerWorkService$WorkPoolRunnable.run(ConsumerWorkService.java:104)
                   at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)
                   at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628)
                   at java.base/java.lang.Thread.run(Thread.java:834)

           ```

           

4. 查看日志

   通过日志，可以看到连接异常，然后重连。

