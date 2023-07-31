
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2017年5月，Spring官方宣布支持Java开发人员通过云平台部署和管理微服务应用。Spring Cloud是一个基于Spring Boot实现的轻量级PaaS(Platform as a Service)框架，它整合了微服务架构模式及一些列开源组件帮助开发者快速构建分布式系统。Spring Cloud Data Flow 是一个用于开发和运行可缩able、可靠的事件驱动型微服务数据流的轻量级Serverless微服务引擎，它可以很好地与Spring Boot和Spring Cloud生态系统集成。在这一系列博文中，我将带领大家逐步深入学习并掌握Spring Cloud Data Flow 的使用方法和具体原理。为了能够更好的理解本文内容，建议大家具有扎实的编程基础和微服务架构知识。如果需要，可以在文章最后提供相关资源链接和参考书籍。
          本篇文章中主要围绕Spring Cloud Data Flow 在微服务流式处理方面的功能进行详细介绍，包括:
           1. Spring Cloud Data Flow 的架构设计；
           2. Spring Cloud Data Flow 的安装配置及其操作指南；
           3. Spring Cloud Data Flow 的应用示例及其操作指南；
           4. Spring Cloud Stream 和 Spring Cloud Task 的基本用法介绍；
           5. Spring Cloud Streams 流定义及其应用；
           6. Spring Cloud Task 流任务创建及其操作指南；
           7. Spring Cloud Data Flow 消息路由及其自定义配置方法；
           8. 数据持久化配置及其操作指南；
          通过阅读本篇文章，读者可以对Spring Cloud Data Flow 有全面的了解和理解，进而方便其正确运用到实际项目中的微服务流式处理之中。
         # 2.Spring Cloud Data Flow 的架构设计
         ## 2.1 架构概览
         Spring Cloud Data Flow 是微服务流式处理框架，它提供了一种声明式的方式来定义、部署和管理数据流。Spring Cloud Data Flow 由两大部分组成，分别是SCDF服务器端和客户端组件。如下图所示。
       ![](https://github.com/Jitesh291/SpringCloudDataFlowReferenceGuide/blob/main/images/architecture_overview.png?raw=true)

         SCDF的服务端组件包括:

            1. Resource Management (RM): SCDF 服务端的资源管理器，用于存储应用程序元数据，注册微服务（tasks），监控各项服务状态，提供统一的API接口，方便客户端和其他组件调用。
            2. Skipper Server: Skipper是Kubernetes上的服务器LESS PaaS系统，用来部署、管理和编排基于微服务的应用程序。Skipper由多个控制器模块组成，包括Controller，Scheduler，Monitoring等。
            3. Config Server: Spring Cloud Config是一个云原生配置管理工具，提供配置服务器和客户端实现配置中心。Config Server是专门用于存储微服务配置信息的数据库，该数据库由一系列的配置文件组成，配置文件可以是本地文件或者远程HTTP/HTTPS地址。
            4. DataFlow Server: Spring Cloud Dataflow是一个用于开发和运行可缩able、可靠的事件驱动型微服务数据流的微服务引擎，由CF-Server，SCDF-Server，Skipper-Server三个组件构成。其中，CF-Server是Sprin Cloud Foundry的一个组件，负责将应用程序打包为droplet和Docker镜像，并在Cloud Foundry平台上运行。SCDF-Server则是SCDF的服务端组件，主要用于处理流式数据，它提供了声明式方式来定义数据流，并最终转换为可执行的任务。Skipper-Server则是Apache Airflow的开源替代品，也是SCDF依赖的中间件组件。

         SCDF的客户端组件包括:
             1. Skipper Client: Skipper客户端是一个命令行工具，可以用来启动、停止、调试、监控应用程序。它还可以查看运行中的应用程序的日志，并可以使用类似kubectl命令的形式管理集群。
             2. Spring Cloud Data Flow Shell: Spring Cloud Data Flow Shell 是 Spring Cloud Data Flow 服务端和客户端之间的桥梁，可以用来与服务端进行交互，如创建应用程序、启动/停止任务等。
             3. Spring Cloud Data Flow UI: Spring Cloud Data Flow UI 是 Spring Cloud Data Flow 的Web界面，提供了一个用户友好的界面让用户管理所有微服务应用，以及编排流式任务。

         ## 2.2 微服务间通信
         Spring Cloud Data Flow 可以利用 Apache Kafka 或 RabbitMQ 等消息代理来处理微服务间的通信。Kafka和RabbitMQ都是开源的消息代理软件，它们可以用来作为微服务间的传输层。Kafka可以处理大量的实时数据，但会产生很多开销。RabbitMQ可以提供低延迟、高吞吐量的特性，但是需要安装额外的组件和服务。在选择传输层之前，需要根据业务需求和性能考虑。

         当消息从生产者发送到Spring Cloud Data Flow 时，它首先经过分区和分发的过程。然后，根据预先定义的路由规则，消息会被路由到相应的消费者。每个消费者可以独立的接收和处理消息。

         # 3.Spring Cloud Data Flow 安装配置及其操作指南
         ## 3.1 安装准备
         ### 3.1.1 配置环境变量
         ```bash
           export PATH=$PATH:/usr/local/bin
           source ~/.bashrc
         ```
         ### 3.1.2 下载最新版 Spring Cloud Data Flow 客户端
         访问[Spring Cloud Data Flow官网](https://dataflow.spring.io/)，找到最新的版本号，下载对应的客户端压缩包，解压至本地目录。例如：
         ```bash
           wget https://repo.spring.io/release/org/springframework/cloud/spring-cloud-dataflow-server-kubernetes/2.8.0/spring-cloud-dataflow-server-kubernetes-2.8.0.zip
           unzip spring-cloud-dataflow-server-kubernetes-2.8.0.zip
         ```
         此处我选择的是 Spring Cloud Data Flow for Kubernetes，因为我的 Kubernetes 集群已经准备就绪，不用再创建一个新的集群了。

         如果你的 Kubernetes 集群是通过 `k3s` 安装的，那么你可能需要下载一个不同的版本的 Spring Cloud Data Flow for Kubernetes。你可以参考 [Spring Cloud Data Flow 文档](https://docs.spring.io/spring-cloud-dataflow/docs/current/reference/htmlsingle/#getting-started-kubernetes)，获取不同版本的 Spring Cloud Data Flow for Kubernetes。
         ### 3.1.3 安装 kubectl 命令行工具
         使用 `apt` 或 `yum` 命令安装 `kubectl`，例如：
         ```bash
           sudo apt update && sudo apt install -y curl
           curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl"
           chmod +x./kubectl
           sudo mv./kubectl /usr/local/bin/kubectl
         ```
         ## 3.2 安装前检查
         使用以下命令检查是否满足所有必要条件：
         ```bash
           kubectl cluster-info            # 查看集群信息
           helm version                    # 查看 Helm 版本
           docker ps                       # 查看 Docker 进程是否正常运行
           docker info                     # 查看 Docker 信息
           minikube status                 # 检查 Minikube 集群状态
         ```
         上述命令的输出都没有任何异常，表示环境已经准备就绪。
         ## 3.3 创建命名空间
         为 SCDF 创建命名空间：
         ```bash
           kubectl create namespace scdf
           kubectl config set-context --current --namespace=scdf
         ```
         ## 3.4 安装 Helm Chart
         使用 Helm 将 SCDF 服务器端组件安装到 Kubernetes 中：
         ```bash
           helm repo add bitnami https://charts.bitnami.com/bitnami
           helm repo update

           # 根据具体的 Kubernetes 集群类型，选择不同的 Helm Chart 版本
           helm install scdf bitnami/spring-cloud-dataflow \
               --version <VERSION>
           # 例如：helm install scdf bitnami/spring-cloud-dataflow --version 2.8.0
         ```
         ## 3.5 配置数据源
         SCDF 需要连接到外部的数据源才能保存和查询任务元数据，比如说保存批处理任务的历史记录，以及持久化存储。默认情况下，SCDF 会尝试连接到 MySQL 数据库。如果你没有任何现有的数据库，你可以在 SCDF Pod 中启动临时数据库：
         ```bash
           kubectl run mariadb --image=mariadb:latest --env="MYSQL_ROOT_PASSWORD=<PASSWORD>" --port=3306 --command -- sleep infinity
         ```
         这里我们将 `mariadb` 部署到 Kubernetes 中的 `default` 命名空间，设置 root 用户的密码为 `<PASSWORD>`。等待几秒钟之后，你可以看到新创建的 Pod。
         ```bash
           kubectl get pods | grep mariadb   # 获取 mariadb pod 的名称
           kubectl exec -it $MARIA_DB_POD_NAME -- mysql -uroot -p$MARIA_DB_ROOT_PASSWORD    # 进入数据库命令行界面
         ```
         在命令行中，创建 `task` 表：
         ```sql
           CREATE DATABASE IF NOT EXISTS task;
           USE task;
           CREATE TABLE IF NOT EXISTS tasks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                name VARCHAR(255),
                description TEXT,
                definition BLOB NULL,
                deploymentId VARCHAR(255)
           );
         ```
         在 `/etc/scdf-config/application.yml` 文件中配置 SCDF 使用刚才创建的数据库：
         ```yaml
           spring:
             datasource:
               url: jdbc:mysql://localhost:3306/task
               username: root
               password: <PASSWORD>
               driverClassName: com.mysql.cj.jdbc.Driver
           app:
             stream:
               platform:
                 kubernetes:
                   accounts:
                     default:
                       limits:
                         memory: 1Gi
       ```
         修改完成后，重启 SCDF Pod：
         ```bash
           kubectl rollout restart deployment/scdf
         ```
         # 4.Spring Cloud Data Flow 的应用示例及其操作指南
         ## 4.1 简单 Hello World 流程
         这个例子展示了如何创建、部署、启动和删除一个简单的“Hello World”数据流。
         1. 创建任务定义：在 Spring Cloud Data Flow UI 上点击“Task Definition”菜单，创建一个名为 “hello” 的新任务。把下面的 YAML 文本粘贴到编辑框里，然后点击保存按钮。
         ```yaml
           apiVersion: dataflow.spring.io/v1beta2
           kind: Task
           metadata:
             name: hello
           spec:
             steps:
               - name: write-output
                 sink: log
               - name: say-hello
                 processor: org.springframework.cloud.dataflow.sample.hello.HelloProcessor
         ```
         2. 编译和部署任务：在 UI 右侧工具栏点击“Compile”按钮，将 YAML 文本编译成 Spring Cloud Data Flow 描述符。然后点击“Deploy”按钮部署应用。
         3. 启动任务：点击左侧导航栏中的“Task”，找到刚才部署的“hello”任务，然后点击“Launch”按钮启动任务。
         4. 查看输出：当任务启动成功后，点击“Logs”选项卡，就可以看到应用的输出。

         5. 删除任务：点击任务列表中的“Actions”，然后选择“Destroy Task”。确认要删除任务。

         6. （可选）删除部署：如果要删除应用，请在 UI 右侧工具栏点击“Undeploy”按钮。
         ## 4.2 HTTP 与 REST API 示例
         下面这个例子演示了如何使用 Spring Cloud Data Flow 来对 HTTP 请求进行过滤，然后调用另一个 HTTP API，并将响应结果发送到 kafka 主题。
         1. 创建任务定义：打开 Spring Cloud Data Flow UI，点击“Task Definition”菜单，创建一个名为 “http” 的新任务。把下面的 YAML 文本粘贴到编辑框里，然后点击保存按钮。
         ```yaml
           apiVersion: dataflow.spring.io/v1beta2
           kind: Task
           metadata:
             name: http
           spec:
             steps:
               - name: filter-requests
                 filter: org.springframework.cloud.dataflow.sample.filterprocessor.RequestFilterProcessor
                 destination: outputTopic
               - name: call-rest-api
                 task-app-name: rest-client
                 uri: http://web-api:8080/${vcap.application.instance_id}
               - name: send-to-kafka
                 sink: kafka:outputTopic
           deployer:
             resources:
               requests:
                 cpu: 100m
                 memory: 128Mi
               limits:
                 cpu: 200m
                 memory: 256Mi
         ```
         2. 编译和部署任务：点击页面右上角的“Compile”按钮，将 YAML 文本编译成 Spring Cloud Data Flow 描述符。然后点击页面左下角的“Create”按钮启动应用部署流程。

         3. 启动任务：点击页面左侧的“Task”菜单，找到刚才部署的“http”任务，然后点击页面右上角的“Launch”按钮启动任务。

         4. 查看输出：当任务启动成功后，切换到“Log”选项卡，可以看到应用的输出。在这个例子中，应该可以看到处理完请求后的响应结果。

         5. 查看任务元数据：点击页面左侧的“Task”菜单，然后单击任一任务，就可以看到任务元数据的详细信息，包括输入、输出，任务部署、启动时间等。

         6. （可选）删除部署：如果要删除应用，请点击页面右上角的“Undeploy”按钮。
         # 5.Spring Cloud Stream 和 Spring Cloud Task 的基本用法介绍
         Spring Cloud Stream 提供了消息通道能力，使得应用可以相互独立，异步地发送和接收消息。Spring Cloud Task 提供了批处理能力，使得应用可以自动执行长时间运行的任务。下面我们简单介绍一下他们的一些基本用法。
         ## 5.1 Spring Cloud Stream
         ### 5.1.1 起步
         1. 添加依赖：在 pom.xml 文件中添加 Spring Cloud Stream 依赖：
         ```xml
           <dependency>
               <groupId>org.springframework.boot</groupId>
               <artifactId>spring-boot-starter-actuator</artifactId>
           </dependency>
           <dependency>
               <groupId>org.springframework.boot</groupId>
               <artifactId>spring-boot-starter-webflux</artifactId>
           </dependency>
           <dependency>
               <groupId>org.springframework.cloud</groupId>
               <artifactId>spring-cloud-stream</artifactId>
           </dependency>
           <dependency>
               <groupId>org.springframework.cloud</groupId>
               <artifactId>spring-cloud-stream-binder-kafka</artifactId>
           </dependency>
           <dependency>
               <groupId>org.springframework.cloud</groupId>
               <artifactId>spring-cloud-stream-binder-rabbit</artifactId>
           </dependency>
         ```
         2. 配置文件：创建配置文件 `application.yml`，并添加以下配置：
         ```yaml
           server:
             port: 8081
           spring:
             application:
               name: message-producer
             cloud:
               stream:
                 bindings:
                   output:
                     destination: exampleOutputTopic
           logging:
             level:
               root: INFO
         ```
         ### 5.1.2 使用 Sink 和 Source
         Sink 和 Source 都是 Spring Cloud Stream 的重要概念。Sink 用于接收消息，Source 用于向外部系统发送消息。下面我们使用 Sink 和 Source 来编写一个简单的消息发布者和订阅者。
         1. 编写发布者类：编写一个控制器类，使用 @RestController 注解，并添加一个消息发布方法：
         ```java
           import org.springframework.beans.factory.annotation.Autowired;
           import org.springframework.messaging.support.MessageBuilder;
           import org.springframework.web.bind.annotation.*;
           import reactor.core.publisher.Flux;
           import reactor.core.publisher.Mono;
    
           @RestController
           public class MessageProducer {
    
               private final IntegrationFlow integrationFlow;
    
               @Autowired
               public MessageProducer(IntegrationFlow integrationFlow) {
                   this.integrationFlow = integrationFlow;
               }
    
               @GetMapping("/messages")
               public Flux<String> sendMessage() {
                   return Flux
                          .range(1, 10)
                          .map(i -> String.format("This is the %dth message", i))
                          .transform(message -> integrationFlow
                                  .handle(Flux.just(MessageBuilder
                                          .withPayload(message)
                                          .build())))
                          .thenMany(Flux.just("Messages sent"));
               }
           }
         ```
         这个控制器类接受一个 GET 请求，并向绑定了 `output` 名称的消息通道发送 10 个消息。
         2. 编写订阅者类：编写一个控制器类，使用 @RestController 注解，并添加一个消息订阅方法：
         ```java
           import org.springframework.beans.factory.annotation.Autowired;
           import org.springframework.cloud.stream.annotation.EnableBinding;
           import org.springframework.cloud.stream.annotation.StreamListener;
           import org.springframework.messaging.Message;
           import org.springframework.stereotype.Service;
    
           @Service
           @EnableBinding(value = MySink.class)
           public class MessageConsumer {
    
               @StreamListener(MySink.INPUT)
               public void receiveMessage(Message<String> message) throws Exception {
                   System.out.println(message.getPayload());
               }
           }
         ```
         这个控制器类定义了一个监听器，在收到一条消息时打印出消息的内容。
         3. 测试发布者与订阅者：测试这个消息发布者和订阅者的集成情况。运行应用，发送 10 次 GET 请求到 "/messages" URL，然后观察订阅者控制台的输出。
         ## 5.2 Spring Cloud Task
         ### 5.2.1 起步
         1. 添加依赖：在 pom.xml 文件中添加 Spring Cloud Task 依赖：
         ```xml
           <dependency>
               <groupId>org.springframework.boot</groupId>
               <artifactId>spring-boot-starter-batch</artifactId>
           </dependency>
           <dependency>
               <groupId>org.springframework.cloud</groupId>
               <artifactId>spring-cloud-starter-task</artifactId>
           </dependency>
         ```
         2. 配置文件：创建配置文件 `application.yml`，并添加以下配置：
         ```yaml
           server:
             port: ${PORT:8082}
           spring:
             application:
               name: batch-job
           logging:
             level:
               root: INFO
         ```
         3. 编写批处理作业：编写一个 BatchConfigurer 类，指定任务数据源：
         ```java
           package com.example.demo;
    
           import javax.sql.DataSource;
    
           import org.springframework.batch.core.Job;
           import org.springframework.batch.core.Step;
           import org.springframework.batch.core.configuration.annotation.EnableBatchProcessing;
           import org.springframework.batch.core.configuration.annotation.JobBuilderFactory;
           import org.springframework.batch.core.configuration.annotation.StepBuilderFactory;
           import org.springframework.batch.core.launch.support.RunJobLauncher;
           import org.springframework.batch.core.repository.JobRepository;
           import org.springframework.batch.item.ItemWriter;
           import org.springframework.beans.factory.annotation.Autowired;
           import org.springframework.beans.factory.annotation.Qualifier;
           import org.springframework.context.annotation.Bean;
           import org.springframework.context.annotation.Configuration;
           import org.springframework.jdbc.datasource.embedded.EmbeddedDatabaseBuilder;
           import org.springframework.jdbc.datasource.embedded.EmbeddedDatabaseType;
    
           @Configuration
           @EnableBatchProcessing
           public class DemoBatchConfiguration {
    
               @Autowired
               private JobBuilderFactory jobBuilderFactory;
    
               @Autowired
               private StepBuilderFactory stepBuilderFactory;
    
               @Bean
               public DataSource dataSource() {
                   EmbeddedDatabaseBuilder builder = new EmbeddedDatabaseBuilder();
                   return builder.setType(EmbeddedDatabaseType.H2).build();
               }
    
               @Bean
               public JobRepository jobRepository() throws Exception {
                   DataSource dataSource = dataSource();
                   return new JdbcJobRepository(dataSource);
               }
    
               @Bean
               public RunJobLauncher jobLauncher() {
                   return new RunJobLauncher();
               }
    
               @Bean
               public ItemWriter itemWriter() {
                   // TODO: implement an actual writer to persist items
                   return list -> {};
               }
    
               @Bean
               public Job sampleJob(@Qualifier("step1") Step step1) {
                   return jobBuilderFactory.get("sampleJob").start(step1).build();
               }
    
               @Bean
               protected Step step1(ItemWriter itemWriter) {
                   return stepBuilderFactory.get("step1").writer(itemWriter).build();
               }
           }
         ```
         这个批处理配置类生成了一个内存数据库作为任务数据源。
         4. 执行批处理作业：编写一个控制器类，使用 @RestController 注解，并添加一个批处理方法：
         ```java
           import java.util.Date;
    
           import org.springframework.batch.core.JobParameter;
           import org.springframework.batch.core.JobParameters;
           import org.springframework.batch.core.launch.JobLauncher;
           import org.springframework.beans.factory.annotation.Autowired;
           import org.springframework.web.bind.annotation.*;
    
           @RestController
           public class BatchDemoController {
    
               @Autowired
               private JobLauncher jobLauncher;
    
               @PostMapping("/run-job")
               public String runSampleJob() throws Exception {
                   Date now = new Date();
                   JobParameters params = new JobParametersBuilder().addDate("jobExecTime", now).toJobParameters();
                   jobLauncher.run(new SampleJob(), params);
                   return "Sample job executed at " + now.toString();
               }
           }
         ```
         这个控制器类接收 POST 请求，并触发一个批处理作业。
         5. 启动应用：启动应用，并访问 "/run-job" URL。
         # 6.Spring Cloud Streams 流定义及其应用
         Spring Cloud Streams 提供了声明式流式处理模型，允许开发者定义数据流管道。下面我们用两个简单实例来介绍 Spring Cloud Streams 流定义。
         ## 6.1 Word Count Processor
         ### 6.1.1 流定义
         我们可以创建一个名为 `wordCountProcessor` 的流定义，其中包含一个名为 `filterProcessor` 的 `Processor` 应用组件，用于接收和过滤输入文本，以及一个名为 `countProcessor` 的 `Processor` 应用组件，用于统计输入文本中每个词出现的次数。下面的 YAML 代码定义了这个流定义：
         ```yaml
           spring:
             cloud:
               stream:
                 bindings:
                   input:
                     group: wordGroup
                   output:
                     producer:
                       useNativeEncoding: false
                 function:
                   definition: >
                      def filterProcessor = { it.split(' ') }.collect { word ->
                          if (!word.isEmpty()) {
                              emit(word.toLowerCase(), null)
                          }
                      }
                      def countProcessor = { tuple ->
                          new Tuple2<>(tuple._1, tuple._2.count())
                      }
                      consume(input)
                       .filter(filterProcessor)
                       .groupByKey()
                       .flatMapValues(countProcessor)
                       .log()
                       .sendTo(output)
           streams:
             wordCountProcessor:
               definition: >
                  -- definition goes here...
         ```
         这里，我们使用 `spring.cloud.stream.bindings` 配置了输入和输出通道。`group` 属性指定了同一批次数据分片的处理逻辑。我们使用 `useNativeEncoding` 属性关闭消息的序列化。然后，我们使用 `function.definition` 属性来定义处理逻辑。`consume()` 方法捕获输入通道数据，`filter()` 方法对数据进行过滤，`groupByKey()` 方法对数据进行分组，`flatMapValues()` 方法对每个组内数据进行映射，并通过 `emit()` 方法发送到输出通道。`log()` 方法输出日志信息。
         ### 6.1.2 流应用
         我们可以创建一个 Spring Boot 应用，作为流式应用的消费者。这个应用通过 `@EnableBinding` 注解启用绑定的管道，并通过 `@StreamListener` 注解定义监听器。下面是一个典型的 Spring Boot 应用：
         ```java
           package com.example.demo;
    
           import org.springframework.boot.CommandLineRunner;
           import org.springframework.boot.SpringApplication;
           import org.springframework.boot.autoconfigure.SpringBootApplication;
           import org.springframework.cloud.stream.annotation.EnableBinding;
           import org.springframework.cloud.stream.annotation.Input;
           import org.springframework.cloud.stream.annotation.StreamListener;
           import org.springframework.cloud.stream.messaging.Processor;
           import org.springframework.messaging.handler.annotation.SendTo;
    
           @SpringBootApplication
           @EnableBinding(Processor.class)
           public class WordCountProcessorApplication implements CommandLineRunner {
    
               public static void main(String[] args) {
                   SpringApplication.run(WordCountProcessorApplication.class, args);
               }
    
               @StreamListener(target = Processor.INPUT)
               @SendTo(Processor.OUTPUT)
               public Object process(String input) {
                   String[] words = input.split("\\W+");
                   int[] counts = new int[words.length];
                   for (int i = 0; i < words.length; i++) {
                       String word = words[i].toLowerCase();
                       boolean found = false;
                       for (int j = 0; j < i; j++) {
                           if (word.equals(words[j])) {
                               counts[j]++;
                               found = true;
                               break;
                           }
                       }
                       if (!found) {
                           counts[i] = 1;
                       }
                   }
                   StringBuilder sb = new StringBuilder();
                   for (int i = 0; i < words.length; i++) {
                       sb.append(words[i]).append(": ").append(counts[i]);
                       if (i!= words.length - 1) {
                           sb.append(", ");
                       }
                   }
                   return sb.toString();
               }
    
               @Override
               public void run(String... args) throws Exception {}
    
           }
         ```
         这里，我们定义了一个名为 `process` 的 `StreamListener`，用于接收输入消息，并对数据进行处理。处理逻辑就是统计输入文本中每个词出现的次数。我们通过 `StringBuilder` 拼接了处理结果。

