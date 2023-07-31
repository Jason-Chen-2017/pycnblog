
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2021年7月，Spring官方宣布了 Spring Boot 的正式版本发布，这是目前最主流的 Java Web开发框架。它提供了非常方便的集成环境配置、自动化配置、starter依赖管理、健康检查等功能。基于这些优秀特性，越来越多的人开始使用 Spring Boot 来进行企业级开发。
         
         Spring Boot 作为新时代 Java Web开发领域的先驱，在国内外都受到广泛关注，也被用于各种企业级项目的开发。其中集成消息队列的功能虽然有限，但依然可以提供高性能的分布式应用服务。本文将会以 Spring Boot 和 RabbitMQ 为主要示例，介绍如何通过 Spring Boot 提供的 starter 模块快速集成 RabbitMQ 消息队列。

         随着互联网业务的发展，海量的数据需要实时的处理和分析，而消息队列则是一种有效的分布式系统架构。RabbitMQ 是目前最流行的开源消息队列之一，它支持多种消息模型，包括点对点（P2P）、发布/订阅（Pub/Sub）、请求/响应（RPC）等。 RabbitMQ 的安装部署、配置和使用方法较为复杂，本文只涉及基本的集成方案，更多高级用法可参考官方文档。

         # 2.基本概念术语说明
         ## 2.1 Spring Boot
         Spring Boot 是由 Pivotal 团队提供的全新 Spring 框架，其目的是用来简化新 Spring 应用程序的初始搭建以及开发过程。该框架使用特定的方式来进行配置，从而使开发人员不再需要定义样板化的 XML 文件。简单来说，Spring Boot 将所有 Spring 框架基础设施的配置和自动化封装起来，让程序员关注于自己的核心业务逻辑即可。

         ## 2.2 Spring Cloud
         Spring Cloud 是一系列框架的有序集合。它利用 Spring Boot 的开发便利性巧妙地简化了分布式系统基础设施的开发，如配置中心、服务发现、断路器、路由、微代理、事件总线、全局锁、决策竞选、分布式消息传递、调度系统等组件，都可以直接运行于 Spring Boot 应用中，从而为用户提供了一系列完整的分布式系统工具。

         ## 2.3 Spring AMQP
         Spring AMQP 是一个基于 Spring 的 AMQP 操作库，可以帮助 Spring 应用与 Java 客户端之间进行松耦合的 AMQP 通信。RabbitMQ、Apache Qpid 等消息中间件可以作为 Spring AMQP 的消息代理实现。

         ## 2.4 RabbitMQ
         RabbitMQ 是一款开源的消息队列服务器。它遵循 AMQP (Advanced Message Queuing Protocol) 协议，是一个面向消息的中间件。RabbitMQ 可以作为 Spring Boot 的消息代理实现，也可以独立于 Spring Boot 以外运行。

         ## 2.5 Spring Boot Starter
         Spring Boot Starter 是 Spring Boot 的一个重要特性。它是一个包含了一组开箱即用的依赖项的包。Spring Boot Starter 本身不包含可执行的代码，它只是提供配置文件，让你可以快速添加所需的依赖项。例如，如果你想使用 Spring Data JPA 来访问数据库，只需要添加 spring-boot-starter-data-jpa starter 依赖即可。

         ## 2.6 RabbitMQ Broker
         RabbitMQ Broker 是消息队列服务器中的一个进程。它维护着队列、交换机、绑定关系以及其他内部组件的状态信息。每个 RabbitMQ 服务端至少运行一个 Broker。通常情况下，一个 RabbitMQ Cluster 中会有多个 Broker，实现了集群模式。

         ## 2.7 Queue、Exchange、Binding、Routing Key
         在 RabbitMQ 中，有四个基本概念，分别是：Queue、Exchange、Binding、Routing Key。

         1. Queue：消息队列，存储消息直到发送者确认接收成功。
         2. Exchange：交换机，用于转发消息或者根据某种规则分派消息到指定的队列。
         3. Binding：绑定，用于把交换机和队列进行关联。
         4. Routing Key：路由键，用于指定消息应该投递给哪个队列。

         # 3.核心算法原理和具体操作步骤以及数学公式讲解
         通过上述的基本概念和术语，下面我们开始讲解 RabbitMQ 中的一些核心概念以及相关的操作。首先是概括性的 RabbitMQ 术语介绍。
         
         1. Connection: 连接，建立网络连接的通道，包含 socket 描述符、SSL 安全套接层等信息，所有链接的信道共享同一个 TCP 连接。

         2. Channel: 信道，网络传输通道，一个连接里可以创建多个信道，每个信道代表一条独立的双向数据流，可以承载不同类型的消息，目前支持发布/订阅、请求/回复、主题等不同类型信道。

         3. Virtual host：虚拟主机，虚拟的 RabbitMQ 服务器资源池，每个 vhost 下可以创建多个 exchange、queue 和 binding，提供更好的隔离性并允许复用相同的 queue 名称。

         4. User：用户，用于认证身份信息，拥有管理员或普通用户角色，可设置权限控制。

         5. Permissions：权限，用于设定用户对 virtual host 下的 exchange、queue 或 binding 的操作权限。

         6. Exchange：交换机，负责转发消息，根据 key 投递消息到对应的队列，可以实现 direct、fanout、topic、headers 几种类型，选择不同的类型可以获得不同的性能。

         7. Binding：绑定，在交换机和队列进行关联，确保交换机根据 key 将消息投递到正确的队列。

         8. Queue：队列，存储消息直到被消费者取走，消息持久化存储或进入死信队列。

         9. Message routing：消息路由，将消息从生产者投递到相应的队列，根据路由规则确定最终目的地。

         然后，我们将详细的讲解一下 Spring Boot 和 RabbitMQ 的集成过程。

         1. 安装 RabbitMQ

            ```
            sudo apt update && sudo apt install rabbitmq-server
            ```

         2. 配置 Spring Boot

            添加以下依赖：
            
            ```xml
            <dependency>
              <groupId>org.springframework.boot</groupId>
              <artifactId>spring-boot-starter-web</artifactId>
            </dependency>
            <dependency>
              <groupId>org.springframework.amqp</groupId>
              <artifactId>spring-rabbit</artifactId>
            </dependency>
            ```

            application.properties 配置如下：

            ```properties
            spring.rabbitmq.host=localhost
            spring.rabbitmq.port=5672
            spring.rabbitmq.username=guest
            spring.rabbitmq.password=<PASSWORD>
            ```

         3. 创建 RabbitMQ Message Producer

            ```java
            @Autowired
            private AmqpTemplate amqpTemplate;
            
            public void send(String message) {
                log.info("Sending message='{}'", message);
                amqpTemplate.convertAndSend("myQueue", message);
            }
            ```

            上面的代码使用 @Autowired 注解注入了一个名叫 "amqpTemplate" 的 AmqpTemplate 对象，然后调用它的 convertAndSend() 方法来发送消息。参数 "myQueue" 表示消息应该投递到的队列名称。

         4. 创建 RabbitMQ Message Consumer

            ```java
            @Component
            public class MyMessageListener {
                
                @RabbitListener(queues = "${my.queue}")
                public void processMessage(String message) throws Exception {
                    log.info("Received message='{}'", message);
                }
                
            }
            ```

            上面的代码声明了一个 RabbitMQ Message Listener ，监听名为 my.queue 的队列。当有新的消息到达这个队列时，MyMessageListener 会收到消息并打印日志。

            需要注意的是，这里有一个坑，就是队列名称需要以 ${} 的形式配置，这样 Spring 才能动态地解析变量值。

         5. 测试

            使用以上代码编写的一个测试类：

            ```java
            @SpringBootTest
            public class ApplicationTests {
            
                @Autowired
                private MyMessageProducer producer;
            
            
                @Test
                public void testProduceAndConsume() throws InterruptedException {
                
                    String message = "Hello World!";
                    
                    // Send a message to the queue and receive it from another thread
                    new Thread(() -> {
                        try {
                            producer.send(message);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }).start();
                    
                    TimeUnit.SECONDS.sleep(2);
                    
                }
            }
            ```

            此测试类启动了一个 Spring Boot 应用上下文，并注入了一个 MyMessageProducer 实例。然后创建一个线程来模拟发送消息到队列，同时等待几秒钟查看消费者是否收到了消息。如果没收到，就会抛出异常。运行此测试类应该可以看到输出了 "Received message='{}'" 的日志。

            至此，RabbitMQ 集成到 Spring Boot 完成。

         # 4.具体代码实例和解释说明
         除了上述的基础概念和术语介绍，本文还会通过具体的代码实例和解释说明来进一步阐明 Spring Boot 和 RabbitMQ 的集成过程。
         ## 4.1 Spring Boot with RabbitMQ Starter Example
         ### 4.1.1 Project Structure
         After downloading and extracting the project zip file, you will see following directory structure. This example contains two modules 'producer' and 'consumer'. The 'producer' module is responsible for sending messages into RabbitMQ while the 'consumer' module is responsible for consuming them.
         ```
        .
        ├── README.md
        ├── pom.xml
        └── src
            ├── main
            │   ├── java
            │   │   └── com
            │   │       └── javatmp
            │   │           ├── consumer
            │   │           │   ├── MessageListenerConfig.java
            │   │           │   ├── RabbitMQConstants.java
            │   │           │   └── RabbitMQConsumerApplication.java
            │   │           ├── model
            │   │           │   ├── Greeting.java
            │   │           │   └── HelloMessage.java
            │   │           ├── producer
            │   │           │   ├── MessageProducerConfig.java
            │   │           │   ├── RabbitMQConstants.java
            │   │           │   └── RabbitMQProducerApplication.java
            │   │           └── util
            │   │               ├── CustomErrorController.java
            │   │               └── ResourceNotFoundException.java
            │   └── resources
            │       ├── application.properties
            │       ├── data.sql
            │       └── logback.xml
            └── test
                └── java
                    └── com
                        └── javatmp
                            └── TestSuite.java
         ```
         
         We have created three separate packages under `src/main/java` folder each containing configuration files and controllers for `producer`, `consumer` and their common models(`Greeting.java`). Additionally, we have added some utility classes such as `CustomErrorController` which handles any exceptions thrown during request processing by returning appropriate HTTP error codes, `ResourceNotFoundException`. And finally, there are integration tests in `TestSuite.java`.
         ### 4.1.2 Dependency Management
         Firstly, we need to add dependency management section in our parent POM file. Add this snippet inside `<dependencyManagement>` tag.
         ```xml
         <!-- Spring Boot Parent -->
         <parent>
           <groupId>org.springframework.boot</groupId>
           <artifactId>spring-boot-starter-parent</artifactId>
           <version>2.2.4.RELEASE</version>
           <relativePath/> 
         </parent>
         ```
         Here, we have specified the latest stable version of `spring-boot-starter-parent` which includes all required dependencies except `spring-cloud-*` starters.
         
         Next step would be adding the RabbitMQ dependency. Add these snippets below `<!-- Spring Boot Parent -->`:
         ```xml
         <!-- RabbitMQ -->
         <dependency>
           <groupId>com.rabbitmq</groupId>
           <artifactId>amqp-client</artifactId>
           <version>5.7.2</version>
         </dependency>
         <dependency>
           <groupId>org.springframework.boot</groupId>
           <artifactId>spring-boot-starter-actuator</artifactId>
         </dependency>
         ```
         In above code, we have included `amqp-client` library that provides access to RabbitMQ's native protocol, and also added `spring-boot-starter-actuator` dependency which allows us to monitor RabbitMQ metrics using actuators endpoint `/actuator/rabbitmq`.
         
         Now let's move on to `pom.xml` of `producer` and `consumer` subprojects where we can include only necessary dependencies.
         
         #### Module : `producer`
         In order to use RabbitMQ in `producer` module, we should include following dependency:
         ```xml
         <dependency>
           <groupId>org.springframework.boot</groupId>
           <artifactId>spring-boot-starter-amqp</artifactId>
         </dependency>
         ```
         Here, we have included `spring-boot-starter-amqp` which adds support for RabbitMQ messaging.
         
         Also, we can define custom properties in `application.properties` like port number, username and password etc.:
         ```properties
         server.port=8081
         spring.rabbitmq.host=localhost
         spring.rabbitmq.port=5672
         spring.rabbitmq.username=guest
         spring.rabbitmq.password=<PASSWORD>
         spring.rabbitmq.virtual-host=/
         spring.rabbitmq.publisher-confirms=true
         spring.rabbitmq.publisher-returns=true
         ```
         These properties help configure connection parameters to connect to RabbitMQ broker instance.
         
         Finally, create a package named `config` inside `producer` package and create a new class `MessageProducerConfig` extending `AmqpConfigurer` interface. This class helps initialize the RabbitMQ components before application starts. Here's an implementation of `MessageProducerConfig`:
         ```java
         import org.springframework.amqp.core.*;
         import org.springframework.context.annotation.Bean;
         import org.springframework.context.annotation.Configuration;
         import org.springframework.util.ErrorHandler;
         import org.springframework.util.backoff.FixedBackOff;
         
         /**
          * Configures RabbitMQ related beans
          */
         @Configuration
         public class MessageProducerConfig implements AmqpConfigurer {
         
             /**
              * Declare RabbitMQ queues, exchanges, bindings
              * Create topic exchange if not present
              * Set up publisher confirms
              */
             @Override
             public void configureRabbitMQ(final RabbitMQListenerContainerFactory containerFactory,
                                          final ConnectionFactory connectionFactory) {
                 
                 // Define error handler for publishing failures
                 ErrorHandler errorHandler = new CustomErrorHandler();
                 
                 // Create queue and set prefetch count
                 Queue queue = new AnonymousQueue();
                 queue.setArguments(Collections.singletonMap("x-max-priority", 10));
                 
                 DirectExchange exchange = new DirectExchange(RabbitMQConstants.EXCHANGE_NAME);
                 
                 Binding binding = new Binding(queue.getName(),
                                             Binding.DestinationType.QUEUE,
                                             RabbitMQConstants.EXCHANGE_NAME,
                                             RabbitMQConstants.ROUTING_KEY, null);
                 
                 containerFactory.setQosEnabled(false);
                 containerFactory.setConcurrentConsumers(1);
                 containerFactory.setMaxConcurrentConsumers(1);
                 containerFactory.setDefaultRequeueRejected(false);
                 containerFactory.getRabbitAdmin().declareQueue(queue);
                 containerFactory.getRabbitAdmin().declareExchange(exchange);
                 containerFactory.getRabbitAdmin().declareBinding(binding);
                 containerFactory.setErrorHandler(errorHandler);
                 containerFactory.setRetryPolicy(new SimpleRetryPolicy());
                 containerFactory.setConnectionLostBackoff(new FixedBackOff(1000, 30000));
                 
             }
         }
         ```
         As per the name suggests, this class configures RabbitMQ specific beans for message producers including declaring queues, exchanges and bindings. It sets up basic error handling strategies when publishing fails and retries sending messages in case of connectivity issues. You may customize the implementation according to your needs.
          
          #### Module : `consumer`
          In order to consume messages from RabbitMQ in `consumer` module, we should include the following dependency:
          ```xml
          <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-amqp</artifactId>
          </dependency>
          ```
          Here again, we have included `spring-boot-starter-amqp` which enables communication with RabbitMQ over its client libraries.
          
          Again, we can define custom properties in `application.properties` to establish connections between RabbitMQ instances:
          ```properties
          server.port=8082
          spring.rabbitmq.host=localhost
          spring.rabbitmq.port=5672
          spring.rabbitmq.username=guest
          spring.rabbitmq.password=<PASSWORD>
          spring.rabbitmq.virtual-host=/
          ```
          Once defined, create a package named `config` inside `consumer` package and create a new class `MessageListenerConfig` extending `AmqpConfigurer` interface. This class initializes the RabbitMQ components before starting the application context. Here's an implementation of `MessageListenerConfig`:
          ```java
          import org.springframework.amqp.core.*;
          import org.springframework.context.annotation.Bean;
          import org.springframework.context.annotation.Configuration;
          import org.springframework.util.ErrorHandler;
          import org.springframework.util.backoff.FixedBackOff;
          
          
          /**
           * Configures RabbitMQ related beans
           */
          @Configuration
          public class MessageListenerConfig implements AmqpConfigurer {
              
              /**
               * Declare RabbitMQ queues, exchanges, bindings
               * Create topic exchange if not present
               * Set up listener container factory with retry policy
               */
              @Override
              public void configureRabbitMQ(final RabbitMQListenerContainerFactory containerFactory,
                                           final ConnectionFactory connectionFactory) {
                  
                  // Define error handler for consuming failures
                  ErrorHandler errorHandler = new CustomErrorHandler();
                  
                  // Create durable queue and bind it to default exchange
                  Queue queue = new Queue(RabbitMQConstants.QUEUE_NAME);
                  Binding binding = new Binding(queue.getName(),
                                              Binding.DestinationType.QUEUE,
                                              RabbitMQConstants.EXCHANGE_NAME,
                                              RabbitMQConstants.ROUTING_KEY, null);
                  containerFactory.setQosEnabled(false);
                  containerFactory.setConcurrentConsumers(1);
                  containerFactory.setMaxConcurrentConsumers(1);
                  containerFactory.setDefaultRequeueRejected(false);
                  containerFactory.getRabbitAdmin().declareQueue(queue);
                  containerFactory.getRabbitAdmin().declareBinding(binding);
                  containerFactory.setErrorHandler(errorHandler);
                  containerFactory.setRetryPolicy(new SimpleRetryPolicy());
                  containerFactory.setConnectionLostBackoff(new FixedBackOff(1000, 30000));
                  
              }
          }
          ```
          In the above implementation, we have declared one durable queue and bound it to default exchange along with other standard configurations. Note that here we have used predefined constants `RabbitMQConstants` instead of hardcoding the values in annotations or configuration.
          
          #### Common Models
          Both the `producer` and `consumer` modules share the same `model` package where we can define domain objects shared across both applications. For simplicity purpose, let's define a simple `HelloMessage` object that wraps a greeting message and timestamp fields. Here's the code:
          ```java
          import lombok.Data;
          
          /**
           * Defines a simple hello message object with string payload and timestamp field
           */
          @Data
          public class HelloMessage {
              
              private String message;
              private long timestamp;
              
          }
          ```
          And now the `Greeting` model defines different types of greetings and includes references to corresponding images stored locally or in remote storage service providers such as AWS S3, Google Storage, etc. Here's the complete code:
          ```java
          import lombok.Getter;
          
          /**
           * Defines possible types of greetings and corresponding image URLs
           */
          public enum Greeting {
              
              HI("hi.png"),
              HELLO("hello.png");
              
              private String imageUrl;
              
              Greeting(String imageUrl) {
                  this.imageUrl = imageUrl;
              }
              
              /**
               * Returns URL of the corresponding image resource
               * @return image URL
               */
              public String getImageUrl() {
                  return getClass().getResource("/images/" + imageUrl).toExternalForm();
              }
              
              /**
               * Get random greeting type
               * @return randomly chosen {@link Greeting} value
               */
              public static Greeting getRandomGreeting() {
                  int index = new Random().nextInt(values().length);
                  return values()[index];
              }
          }
          ```
          #### Integration Tests
          Let's write some integration tests that demonstrate how to produce and consume messages using RabbitMQ within the `TestSuite.java` file located at `test/java/com/javatmp/TestSuite.java` as follows:
          ```java
          import com.google.common.collect.Lists;
          import com.javatmp.consumer.RabbitMQConsumerApplication;
          import com.javatmp.consumer.controller.GreetingController;
          import com.javatmp.consumer.controller.HelloMessageController;
          import com.javatmp.producer.RabbitMQProducerApplication;
          import com.javatmp.producer.domain.Greeting;
          import com.javatmp.producer.domain.HelloMessage;
          import com.javatmp.producer.service.HelloMessageService;
          import org.junit.jupiter.api.*;
          import org.slf4j.Logger;
          import org.slf4j.LoggerFactory;
          import org.springframework.beans.factory.annotation.Autowired;
          import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
          import org.springframework.boot.test.context.SpringBootTest;
          import org.springframework.http.MediaType;
          import org.springframework.test.web.servlet.MockMvc;
          import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
          import org.springframework.transaction.annotation.Transactional;
          import org.testcontainers.shaded.org.apache.commons.io.IOUtils;
          
          import javax.persistence.EntityManager;
          import javax.persistence.PersistenceContext;
          import java.nio.charset.StandardCharsets;
          import java.time.LocalDateTime;
          import java.util.List;
          import java.util.Random;
          
          import static org.awaitility.Awaitility.given;
          import static org.hamcrest.CoreMatchers.containsString;
          import static org.junit.Assert.assertEquals;
          import static org.junit.Assert.assertTrue;
          import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;
          import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
          
          /**
           * Integration tests demonstrating RabbitMQ integration between Spring Boot applications
           */
          @SpringBootTest(classes = {RabbitMQProducerApplication.class})
          @Transactional
          @AutoConfigureMockMvc
          public class TestSuite extends AbstractTestContainersTestBase {
              
              Logger logger = LoggerFactory.getLogger(getClass());
              
              @Autowired
              MockMvc mockMvc;
              
              @Autowired
              EntityManager entityManager;
              
              @Autowired
              HelloMessageService helloMessageService;
              
              @Autowired
              GreetingController greetingController;
              
              @Autowired
              HelloMessageController helloMessageController;
              
              @BeforeAll
              public static void setupDockerComposeFile() {
                  System.setProperty("spring.datasource.url", "jdbc:tc:postgresql://localhost:5432/postgres?TC_TMPFS=/testtmpfs:rw");
                  startDockerCompose();
              }
              
              /**
               * Creates database schema
               */
              @BeforeEach
              public void initDatabase() throws Exception {
                  executeSqlScript("/db/init-database.sql");
              }
              
              /**
               * Produce and consume messages
               */
              @Test
              public void testProduceAndConsumeMessages() throws Exception {
                  
                  List<Integer> ids = Lists.newArrayList();
                  
                  given().when().post("/greeting")
                         .then()
                         .statusCode(201)
                         .extract()
                         .jsonPath("$._links.self.href").isNotEmpty();
                  
                  for (int i = 0; i < 10; i++) {
                      HelloMessage message = new HelloMessage("Hello World!", LocalDateTime.now().toInstant().toEpochMilli());
                      
                      given().contentType(MediaType.APPLICATION_JSON)
                             .content(asJsonString(message))
                             .when()
                             .post("/messages")
                             .thenReturn(null);
                      
                      ids.add(i);
                  }
                  
                  awaitUntilMessagesAreConsumed();
                  
                  assertEquals(ids.size(), entityManager.createQuery("SELECT COUNT(*) FROM HelloMessage h WHERE h.id IN (:ids)", Long.class)
                                                             .setParameter("ids", ids)
                                                             .getSingleResult().intValue());
              }
              
              /**
               * Consumes messages asynchronously until they are consumed
               */
              private void awaitUntilMessagesAreConsumed() {
                  given().ignoreException(IllegalArgumentException.class)
                         .await()
                         .atMost(30)
                         .until(() -> helloMessageController.getMessageCount() == 0);
              }
              
              /**
               * Converts object to JSON string
               * @param obj Object to serialize
               * @return JSON representation of the object
               * @throws Exception If serialization fails
               */
              private String asJsonString(Object obj) throws Exception {
                  ObjectMapper mapper = new ObjectMapper();
                  return mapper.writeValueAsString(obj);
              }
              
              /**
               * Reads content of a SQL script file and executes it against configured JDBC datasource
               * @param filePath Path to SQL script file relative to root of classpath
               */
              protected void executeSqlScript(String filePath) {
                  DataSource dataSource = applicationContext.getBean(DataSource.class);
                  
                  Resource resource = new ClassPathResource(filePath);
                  
                  String sql = IOUtils.toString(resource.getInputStream(), StandardCharsets.UTF_8);
                  
                  jdbcTemplate.execute(sql);
              }
              
              /**
               * Verifies response status code and body matches expected values
               * @param mvc Mvc instance
               * @param uri URI to query
               * @param expectedStatus Expected HTTP status code
               * @param expectedBody Expected response body
               * @throws Exception When response does not match expectations
               */
              private void verifyResponse(MockMvc mvc, String uri, int expectedStatus, String... expectedBody) throws Exception {
                  mvc.perform(MockMvcRequestBuilders.get(uri))
                         .andDo(print())
                         .andExpect(status().is(expectedStatus))
                         .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
                         .andExpect((expectedBody.length > 0? content().string(containsString(expectedBody[0])) : null));
              }
              
              /**
               * Generates list of valid payloads for testing POST /greeting requests
               * @return Payloads for testing POST /greeting
               */
              private List<Greeting> generateValidGreetingPayloads() {
                  List<Greeting> payloads = Lists.newArrayList();
                  payloads.add(new Greeting(Greeting.HI, "Hi!"));
                  payloads.add(new Greeting(Greeting.HELLO, "Hello."));
                  return payloads;
              }
              
              /**
               * Generates list of invalid payloads for testing POST /greeting requests
               * @return Payloads for testing POST /greeting
               */
              private List<Greeting> generateInvalidGreetingPayloads() {
                  List<Greeting> payloads = Lists.newArrayList();
                  payloads.add(new Greeting(null, null));
                  payloads.add(new Greeting("", ""));
                  payloads.add(new Greeting(null, "Hi!"));
                  payloads.add(new Greeting(Greeting.HI, null));
                  payloads.add(new Greeting("", "Hello."));
                  return payloads;
              }
              
              /**
               * Generates list of valid payloads for testing POST /messages requests
               * @return Payloads for testing POST /messages
               */
              private List<HelloMessage> generateValidMessagePayloads() {
                  List<HelloMessage> payloads = Lists.newArrayList();
                  payloads.add(new HelloMessage("Hello World!", LocalDateTime.now().toInstant().toEpochMilli()));
                  payloads.add(new HelloMessage("Goodbye.", LocalDateTime.now().plusHours(1).toInstant().toEpochMilli()));
                  return payloads;
              }
              
              /**
               * Generates list of invalid payloads for testing POST /messages requests
               * @return Payloads for testing POST /messages
               */
              private List<HelloMessage> generateInvalidMessagePayloads() {
                  List<HelloMessage> payloads = Lists.newArrayList();
                  payloads.add(new HelloMessage(null, null));
                  payloads.add(new HelloMessage("", LocalDateTime.now().toInstant().toEpochMilli()));
                  payloads.add(new HelloMessage(null, LocalDateTime.now().plusHours(1).toInstant().toEpochMilli()));
                  return payloads;
              }
              
              /**
               * Validates GET /greetings/{type}/image returns an HTTP 200 OK with image data for supported types
               * @throws Exception Thrown when validation fails
               */
              @Test
              public void testGetImageForSupportedTypes() throws Exception {
                  Greeting[] types = Greeting.values();
                  
                  for (Greeting type : types) {
                      byte[] imgBytes = getBinaryContentFromUri("/greetings/" + type.name() + "/image");
                      assertTrue(imgBytes!= null && imgBytes.length > 0);
                  }
              }
              
              /**
               * Validates GET /greetings/{type}/image returns HTTP 404 NOT FOUND for unsupported types
               * @throws Exception Thrown when validation fails
               */
              @Test
              public void testGetImageForUnsupportedTypes() throws Exception {
                  verifyResponse(mockMvc, "/greetings/invalidType/image", 404);
              }
              
              /**
               * Sends valid payloads for testing POST /greeting requests and validates responses
               * @throws Exception Thrown when validation fails
               */
              @Test
              public void testPostValidGreetingPayloads() throws Exception {
                  List<Greeting> payloads = generateValidGreetingPayloads();
                  
                  for (Greeting payload : payloads) {
                      verifyResponse(mockMvc, "/greeting", 201, "\"" + payload.name() + "\"");
                      given().when().get("/greeting")
                             .then()
                             .statusCode(200)
                             .body("_embedded.greetings.[*].type", hasItem(payload.name()))
                             .body("_embedded.greetings.[*].text", hasItem(payload.getText()))
                             .body("_embedded.greetings.[*]._links.self.href", hasItem("/greetings/" + payload.name()));
                  }
              }
              
              /**
               * Sends invalid payloads for testing POST /greeting requests and validates responses
               * @throws Exception Thrown when validation fails
               */
              @Test
              public void testPostInvalidGreetingPayloads() throws Exception {
                  List<Greeting> payloads = generateInvalidGreetingPayloads();
                  
                  for (Greeting payload : payloads) {
                      verifyResponse(mockMvc, "/greeting", 400, "{\"timestamp\":\"\",\"path\":\"/greeting\",\"message\":\"Please provide a valid text\"}");
                  }
              }
              
              /**
               * Sends valid payloads for testing POST /messages requests and validates responses
               * @throws Exception Thrown when validation fails
               */
              @Test
              public void testPostValidMessagePayloads() throws Exception {
                  List<HelloMessage> payloads = generateValidMessagePayloads();
                  
                  for (HelloMessage payload : payloads) {
                      verifyResponse(mockMvc, "/messages", 201, "[{\"id\":\\d+,\"message\":\"" + payload.getMessage() + "\",\"timestamp\":" + payload.getTimestamp() + "}]");
                      given().when().get("/messages")
                             .then()
                             .statusCode(200)
                             .body("[*].message", hasItems(payload.getMessage()))
                             .body("[*].timestamp", hasItems(String.valueOf(payload.getTimestamp())));
                  }
              }
              
              /**
               * Sends invalid payloads for testing POST /messages requests and validates responses
               * @throws Exception Thrown when validation fails
               */
              @Test
              public void testPostInvalidMessagePayloads() throws Exception {
                  List<HelloMessage> payloads = generateInvalidMessagePayloads();
                  
                  for (HelloMessage payload : payloads) {
                      verifyResponse(mockMvc, "/messages", 400, "{\"timestamp\":\"\",\"path\":\"/messages\",\"message\":\"Validation failed\",\"details\":[\"object不能为空.\"]}","[\"Validation failed\"]");
                  }
              }
          }
          ```
          In the above code, we have written various integration tests that perform various operations such as producing and consuming messages via RabbitMQ, generating and validating endpoints responses, uploading and retrieving images from external services, etc. Some important points to note about the above test suite are:

          1. `@SpringBootTest(classes = {RabbitMQProducerApplication.class})` annotation ensures that only `RabbitMQProducerApplication` gets started for running integration tests for faster execution time. Similarly, we can specify `RabbitMQConsumerApplication.class` for integration tests involving consumption of messages.
          2. `@Transactional` annotation ensures that database transactions are rolled back after each test method to ensure consistent state among tests. By default, JUnit runs methods in parallel threads and sharing a single transaction manager can cause concurrency issues.
          3. `@AutoConfigureMockMvc` annotation enables auto-configuration of `MockMvc` bean for easy creation of REST API clients.
          4. All test methods annotated with `@Test` are executed in alphabetical order based on method names by default. We could explicitly declare execution orders by defining integers after `@Test` annotations.
          5. Each test method creates a fresh Spring Bean Factory by invoking constructor of `AbstractTestContainersTestBase` subclass. This ensures that tests don't affect each other due to shared state.

     
      

