
作者：禅与计算机程序设计艺术                    

# 1.简介
         

随着微服务架构的流行，服务依赖越来越复杂，使得开发、测试、运维、部署等环节变得更加繁琐和耗时。Spring Cloud生态系统通过Netflix公司开源的Spring Cloud微服务框架解决了这一难题。

作为微服务架构中的子系统之一，Eureka是一个非常重要的组件。它是一个基于RESTful API的服务注册中心，它可以让微服务应用相互发现，并且可以通过API接口提供完整的服务治理功能，包括服务注册、服务查询、服务健康检测、主动通知和集群管理等。同时，它还提供了UI界面，能够直观地展示服务信息。

本文将会介绍如何安装并启动Eureka AdminServer，并且通过Web UI对服务进行查看和管理。

# 2.基本概念术语
- Eureka Server: 是Netflix公司开源的一款java实现的服务注册中心。其主要职责就是维护服务实例的注册表，并且向其他服务发送心跳保持可用性。

- Client: 是向Eureka Server注册并订阅服务的微服务客户端。每个Client都有自己的IP地址和端口号，当某个Client不可用或需要更新时，Eureka Server能快速通知其他服务。

- Admin Server: 是Netflix公司开源的一款基于Spring Boot的服务管理工具。它是一个独立的服务器端程序，与Eureka Server紧密结合，可用于服务的注册、状态检查、元数据管理等。

- UI: 用户界面，是指用户通过浏览器访问的网站页面。

# 3.核心算法原理和具体操作步骤
## 安装Eureka Server
Eureka Server作为微服务架构中的服务注册中心，可以单独运行或者集成到Spring Cloud Config中一起使用。由于Eureka采用Java语言编写，所以首先需要在本地环境安装JDK。

### 在本地环境安装JDK
由于电脑配置差异，这里仅给出Windows平台上JDK安装的步骤：

1. 到Oracle官网下载JDK，选择合适版本（jdk-8uXXX-windows-i586.exe）；
2. 将下载好的压缩包放到任意目录下，如D盘；
3. 右击该压缩包，选择【7-zip】-> 【Extract Here】->选择目标文件夹；
4. 打开目标文件夹，进入“bin”目录，找到“javac.exe”文件并双击运行；
5. 如果出现提示框询问是否添加环境变量，点击“是”，然后根据提示框设置环境变量。

### 创建Maven项目

创建一个maven项目，在pom.xml中添加以下依赖：

``` xml
<dependency>
<groupId>org.springframework.boot</groupId>
<artifactId>spring-boot-starter-web</artifactId>
</dependency>

<!-- Spring Eureka -->
<dependency>
<groupId>org.springframework.cloud</groupId>
<artifactId>spring-cloud-starter-netflix-eureka-server</artifactId>
</dependency>

<!-- Netflix Eureka Dashboard -->
<dependency>
<groupId>org.springframework.cloud</groupId>
<artifactId>spring-cloud-netflix-eureka-dashboard</artifactId>
</dependency>
```

为了方便管理Eureka Server的配置文件，建议将配置文件统一放在resources目录下的eureka/目录下。

在启动类中添加@EnableEurekaServer注解，如下所示：

``` java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.netflix.eureka.server.EnableEurekaServer;

@SpringBootApplication
@EnableEurekaServer
public class ServiceRegistryApp {

public static void main(String[] args) {
SpringApplication.run(ServiceRegistryApp.class, args);
}
}
```

### 配置Eureka Server

修改application.properties文件，添加以下配置：

``` properties
spring.application.name=eureka-server # 服务名称

server.port=8761 # 监听端口

eureka.instance.hostname=${spring.cloud.client.ip-address} # 指定IP地址，默认为主机名
eureka.client.registerWithEureka=false # 不向Eureka Server注册自己
eureka.client.fetchRegistry=false # 不拉取其他服务的信息
eureka.client.serviceUrl.defaultZone=http://${eureka.instance.hostname}:${server.port}/eureka/
```

其中eureka.instance.hostname属性指定Eureka实例的主机名，eureka.client.serviceUrl.defaultZone属性指定注册到当前Eureka Server上的其他服务的URL。如果希望多个Eureka Server之间可以进行数据同步，可以在此配置上游Eureka Server的连接地址。

### 启动Eureka Server

配置完成后，即可启动Eureka Server。在IDEA中，直接运行Application类即可。命令行窗口切换至项目根目录，输入以下命令启动服务：

``` shell
mvn spring-boot:run
```

正常情况下，Eureka Server会在控制台打印出“Started Eureka in X seconds”字样表示成功启动。

## 安装Eureka AdminServer

Eureka AdminServer是一个基于Spring Boot开发的独立的服务管理工具，它与Eureka Server紧密结合，可用于服务的注册、状态检查、元数据管理等。由于AdminServer本身也是个服务，所以它的安装过程也比较简单。

### 添加依赖

由于AdminServer也是一个Spring Boot工程，因此除了要添加Eureka AdminServer相关依赖外，还需添加配置中心的依赖，因为AdminServer需要读取各个微服务的元数据。

``` xml
<!-- Spring Cloud Config Client -->
<dependency>
<groupId>org.springframework.cloud</groupId>
<artifactId>spring-cloud-config-client</artifactId>
</dependency>

<!-- Spring Cloud Admin Server -->
<dependency>
<groupId>org.springframework.cloud</groupId>
<artifactId>spring-cloud-starter-netflix-eureka-server</artifactId>
</dependency>

<!-- Spring Cloud Sleuth -->
<dependency>
<groupId>org.springframework.cloud</groupId>
<artifactId>spring-cloud-starter-sleuth</artifactId>
</dependency>

<!-- Spring Boot Actuator -->
<dependency>
<groupId>org.springframework.boot</groupId>
<artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

### 修改配置文件

一般来说，AdminServer的配置比较简单，不需要太多的自定义配置项。只需要在bootstrap.yml中设置spring.application.name属性，指定AdminServer的服务名即可。

``` yml
spring:
application:
name: eureka-admin

---
spring:
profiles: development
cloud:
config:
uri: http://localhost:8888

---
spring:
profiles: production
cloud:
config:
uri: ${CONFIG_SERVER_URI}

management:
endpoints:
web:
exposure:
include: '*'

eureka:
client:
service-url:
defaultZone: http://${EUREKA_HOST:localhost}:${EUREKA_PORT:8761}/eureka/
```

这里使用两个profile，分别对应开发环境和生产环境的配置，development代表开发环境，production代表生产环境。开发环境通过HTTP的方式连接配置中心，而生产环境则通过配置中心中的服务发现机制连接配置中心。

另外，在management.endpoints.web.exposure.include配置项中开启所有端点，以便能够通过HTTP方式访问AdminServer的各种管理接口。

最后，配置Eureka Server的连接地址，以便AdminServer能够连接到Eureka Server。

### 启动AdminServer

配置完成后，即可启动AdminServer。在IDEA中，直接运行EurekaAdminApplication类即可。命令行窗口切换至项目根目录，输入以下命令启动服务：

``` shell
mvn spring-boot:run -Deureka.env=${spring.profiles.active} 
```

其中${spring.profiles.active}是激活的profile，即要启动哪个环境的服务，比如development或production。正常情况下，AdminServer会在控制台打印出“Started Application in XXXX ms”字样表示成功启动。

### 访问AdminServer

默认情况下，AdminServer的管理接口都是以HTTP暴露的，可以通过如下方式访问：

``` shell
http://localhost:9000
```

其中localhost是AdminServer所在机器的IP地址，通常是127.0.0.1。9000是AdminServer的默认端口号。

登录成功之后，就可以看到AdminServer的管理页面。如下图所示：


这个页面展示了Eureka Server中的服务列表，以及这些服务的健康状况、状态、统计信息等。并且提供了丰富的管理操作，比如服务上下线、查看详细信息、重新启动服务等。

注意：在实际使用过程中，一定要确保不要将Eureka Server本身作为AdminServer的客户端，否则会导致AdminServer不断重连Eureka Server，导致CPU占用过高甚至引起服务崩溃。