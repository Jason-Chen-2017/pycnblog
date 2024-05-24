
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 在Spring框架中, Spring Integration提供了对消息传递的支持,它可以非常方便地集成各种消息中间件,如activemq、Rabbitmq等。本文将结合Spring Boot框架,利用Spring Integration实现在Spring Boot应用中集成ActiveMQ作为消息中间件。
          ## 主要内容
          1.项目搭建
          2.Maven依赖导入
          3.配置文件配置
          4.定义bean
          5.生产者发送消息
          6.消费者接收消息
          7.测试验证
          
          ### 一.项目搭建
          本例使用Spring Boot框架构建一个简单的消息生产者消费者模型。
          * 创建maven项目:File->New->Project->maven->ArtifactId:"springboot-activemq"
          * 添加pom依赖
            ```xml
            <dependency>
              <groupId>org.springframework.boot</groupId>
              <artifactId>spring-boot-starter-web</artifactId>
            </dependency>
            
            <dependency>
              <groupId>org.springframework.integration</groupId>
              <artifactId>spring-integration-core</artifactId>
            </dependency>
            
            <!-- Spring Boot整合ActiveMQ -->
            <dependency>
              <groupId>org.springframework.boot</groupId>
              <artifactId>spring-boot-starter-activemq</artifactId>
            </dependency>
            
            <!-- 使用H2数据库存储消息 -->
            <dependency>
              <groupId>com.h2database</groupId>
              <artifactId>h2</artifactId>
              <scope>runtime</scope>
            </dependency>
            ```
            
          ### 二.Maven依赖导入
          上面完成了项目的初始设置,这里需要引入activemq作为消息中间件,并声明依赖关系。其中Spring Boot整合ActiveMQ是通过spring-boot-starter-activemq自动引入所需依赖的。
          ### 三.配置文件配置
          在resources文件夹下新建application.yml文件,然后添加以下配置信息:
          ```yaml
          spring:
            activemq:
              broker-url: tcp://localhost:61616
              user: admin
              password: admin
          ```
          配置了ActiveMQ服务器地址、用户名和密码。
          ### 四.定义bean
          在配置文件类上注解@Configuration,然后创建两个bean:ActivemqConnectionFactory和JmsTemplate。
          ActivemqConnectionFactory用于连接到ActiveMQ服务器,JmsTemplate用于向队列、主题或订阅发送消息。
          
          下面的代码展示了如何定义两个bean：
          ```java
          import org.apache.activemq.ActiveMQConnectionFactory;
          import org.springframework.context.annotation.Bean;
          import org.springframework.context.annotation.Configuration;
          import org.springframework.jms.connection.CachingConnectionFactory;
          import org.springframework.jms.core.JmsTemplate;
  
          @Configuration
          public class MyConfig {
  
            // ActiveMQ connection factory bean
            @Bean
            public ActiveMQConnectionFactory activeMQConnectionFactory() {
                return new ActiveMQConnectionFactory("tcp://localhost:61616");
            }
  
            // Caching connection factory bean
            @Bean
            public CachingConnectionFactory cachingConnectionFactory(ActiveMQConnectionFactory connectionFactory) {
                CachingConnectionFactory ccf = new CachingConnectionFactory();
                ccf.setTargetConnectionFactory(connectionFactory);
                return ccf;
            }
  
            // JMS template bean
            @Bean
            public JmsTemplate jmsTemplate(CachingConnectionFactory cachingConnectionFactory) {
                JmsTemplate jmsTemplate = new JmsTemplate();
                jmsTemplate.setConnectionFactory(cachingConnectionFactory);
                return jmsTemplate;
            }
          }
          ```
          ### 五.生产者发送消息
          在控制器类上添加方法用于向队列发送消息。
          ```java
          import org.springframework.beans.factory.annotation.Autowired;
          import org.springframework.web.bind.annotation.GetMapping;
          import org.springframework.web.bind.annotation.RestController;
  
          @RestController
          public class MessageController {
  
            private static final String QUEUE_NAME = "myQueue";
  
            @Autowired
            private JmsTemplate jmsTemplate;
  
            @GetMapping("/send")
            public void sendMessage() throws Exception{
                jmsTemplate.convertAndSend(QUEUE_NAME, "Hello from Spring Integration!");
            }
          }
          ```
          在/send请求时会调用JmsTemplate的convertAndSend方法向名为"myQueue"的队列发送消息。
          ### 六.消费者接收消息
          在启动类上添加注解@EnableIntegrationPatterns(IntegrationAutoConfiguration.class)，该注解会开启spring integration的功能。
          在配置文件application.yml中加入activemq配置。
          ```yaml
          spring:
            jpa:
              database: h2
            datasource:
              url: jdbc:h2:mem:testdb
              driverClassName: org.h2.Driver
              username: sa
              password: 
              platform: h2
            activemq:
              broker-url: tcp://localhost:61616
              user: admin
              password: admin
          ```
          编写一个消息处理器类接收队列中的消息并打印出来。
          ```java
          package com.welink.learn.springbootactivemq;
  
          import org.springframework.beans.factory.annotation.Autowired;
          import org.springframework.messaging.support.GenericMessage;
          import org.springframework.stereotype.Component;
          import org.springframework.integration.annotation.ServiceActivator;
          import org.springframework.integration.handler.MessageHandler;
          import org.springframework.integration.config.EnableIntegration;
          import org.springframework.integration.annotation.*;
          import org.springframework.integration.channel.DirectChannel;
  
          /**
           * @Author weWelink
           */
          @EnableIntegration
          @Component
          public class MessageConsumer {
  
            @ServiceActivator(inputChannel="myInputChannel")
            public void handle(String message){
                System.out.println("Received message : " + message);
            }
  
            @Bean
            public DirectChannel myInputChannel(){
                return new DirectChannel();
            }
  
          }
          ```
          在handle方法中打印接收到的消息，并注入一个DirectChannel作为通道注入到启动类。
          在启动类上添加注解@MessagingGateway,指定defaultRequestChannel。
          ```java
          package com.welink.learn.springbootactivemq;
  
          import org.springframework.integration.annotation.*;
          import org.springframework.messaging.handler.annotation.*;
          import org.springframework.messaging.MessagingException;
          import org.springframework.stereotype.*;
  
          /**
           * @Author weWelink
           */
          @MessagingGateway(defaultRequestChannel="myOutputChannel")
          public interface MessageGateway {
  
            @Gateway(requestChannel= "myInputChannel")
            String sendMessage(@Payload String payload);
  
          }
          ```
          在接口上添加一个方法sendMessage,通过设置defaultRequestChannel绑定到名称为"myOutputChannel"的通道。
          在启动类上再次添加注解@EnableIntegrationPatterns,同样的启动类上也需要添加如下注解：
          ```java
          package com.welink.learn.springbootactivemq;
  
          import org.springframework.boot.CommandLineRunner;
          import org.springframework.boot.SpringApplication;
          import org.springframework.boot.autoconfigure.SpringBootApplication;
          import org.springframework.context.ApplicationContext;
          import org.springframework.messaging.MessageChannel;
          import org.springframework.messaging.support.GenericMessage;
          import javax.annotation.Resource;
  
          /**
           * @Author weWelink
           */
          @SpringBootApplication
          @EnableIntegration
          public class Application implements CommandLineRunner {
  
            public static void main(String[] args) {
                SpringApplication.run(Application.class, args);
            }
  
            @Resource(name="messageGateway")
            private MessageGateway gateway;
  
            @Override
            public void run(String... args) throws Exception {
                try {
                    String result = this.gateway.sendMessage("Hello from Gateway!");
                    System.out.println(result);
                } catch (MessagingException e) {
                    e.printStackTrace();
                }
            }
  
          }
          ```
          此处的注释即为将消息发送到队列,调用sendMessage方法，方法中指定发送的内容，并指定目标通道为"myInputChannel",当消息被路由到通道后就会触发服务activator的handle方法。
          ### 七.测试验证
          在浏览器访问http://localhost:8080/send,可以在控制台看到已收到消息。