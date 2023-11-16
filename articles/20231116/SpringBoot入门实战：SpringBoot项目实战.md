                 

# 1.背景介绍


> 在我看来，“快速启动、小型内存占用、无JVM外围依赖”是Java虚拟机（JVM）一个独具魅力的特性，也是Springboot的最大卖点。因此掌握Springboot非常有必要，对其理解也至关重要。本文将带领大家进行从基础到实践的全面认识Springboot。


# 2.核心概念与联系
## 2.1 SpringBoot概述
### 2.1.1 为什么要学习Springboot？
- Spring Boot让我们更容易开发单个、微服务或基于云的应用程序，通过开箱即用的设置可以加快应用的开发速度；
- 有助于提升Java开发人员的技能水平，减少编码工作量，改善软件质量；
- 降低部署与维护的难度，简化了开发流程。

### 2.1.2 SpringBoot架构图
#### 2.1.2.1 SpringBoot的优点
- 独立运行，内嵌servlet容器，减少部署依赖项；
- 提供自动配置功能，可以快速启动开发环境，简化开发过程；
- 完美支持Restful API开发；
- 支持模板引擎Thymeleaf等视图技术，构建出色的前后端分离系统；
- 提供DevTools热部署能力，使得开发者不必频繁重启应用即可看到更新效果。

#### 2.1.2.2 SpringBoot的缺点
- 没有完全解决企业级开发问题，比如分布式事务、缓存、消息队列、搜索引擎、安全等；
- 没有提供企业级框架，只能在特定场景下使用。

### 2.1.3 SpringBoot核心组件
- Spring Boot Starter: 一系列预配置好的starter，方便快速使用某个功能；
- Spring Boot Auto Configuration: Spring Boot 根据你引入的jar包自动配置对应的bean；
- Spring Application Context(ApplicationContext): 由BeanFactory和BeanDefinitionLoader两部分组成，负责读取配置文件，创建对象并管理各个 Bean 的生命周期；
- Spring Beans Container: BeanFactory 是 Bean 实际载体，里面包含着 Bean 对象和所有 Bean 之间的关系信息；
- WebApplicationContext: 用于开发 web 相关的 Spring ApplicationContext，可以用于接收外部请求；
- EmbeddedServletContainer: Spring Boot 内嵌了 Tomcat、Jetty、Undertow等servlet容器，用于提供 HTTP 服务。

## 2.2 SpringBoot入门
### 2.2.1 Spring Boot基本知识
#### 2.2.1.1 创建项目
在Intellij IDEA中选择新建项目，填写项目信息，勾选上Spring Initializr支持，点击next继续往下走，如图所示：
然后选择Spring Boot版本号、项目结构、Spring Boot的依赖，最后点击finish完成项目的创建。
#### 2.2.1.2 目录结构
创建成功之后，可以看到如下目录结构：
其中：
- pom.xml: Maven的依赖管理文件，主要作用是在编译、测试、打包时根据pom文件中的依赖信息来管理工程的依赖库。
- src/main/java: 存放的是主要的代码资源，一般包括java类、resources等。
- src/main/resources: 主要用来存储配置文件，包括application.properties、log4j.xml等。
- src/test/java: 单元测试源码，如果需要编写单元测试的话，可以把测试源码放在这个目录下。

#### 2.2.1.3 Hello World示例
创建一个HelloController类，代码如下：
```java
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HelloController {

    @RequestMapping("/")
    public String hello() {
        return "Hello,World!";
    }
}
```
然后配置SpringMVC，修改配置文件`src/main/resources/application.properties`，增加如下代码：
```yaml
spring.mvc.view.prefix=/WEB-INF/jsp/
spring.mvc.view.suffix=.jsp
```
再创建JSP页面`src/main/webapp/WEB-INF/jsp/hello.jsp`，添加内容：
```html
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title>Title</title>
</head>
<body>
Hello, World!
</body>
</html>
```
运行Application主类，访问http://localhost:8080查看结果：