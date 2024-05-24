                 

# 1.背景介绍

SpringBoot中的配置管理
======================

作者：禅与计算机程序设计艺术

## 背景介绍

随着微服务架构的普及，Spring Boot wurde zu einer sehr beliebten Wahl für die Entwicklung von Unternehmensanwendungen. Es vereinfacht die Erstellung von Spring-basierten Anwendungen durch die Bereitstellung einer Vielzahl von starter-Paketen und Konfigurationsautomatisierungsfunktionen. In diesem Artikel werden wir uns auf die Konfigurationsverwaltung in Spring Boot konzentrieren und wie sie dazu beiträgt, die Komplexität der Anwendungskonfiguration zu reduzieren.

## 核心概念与关系

### 1.1. Spring Boot 自动配置

Spring Boot 利用 Spring Fraamework 的 IoC / DI 功能，通过 Conditional 注解实现条件配置，根据类、属性上的 @Conditional 注解来判断该配置类是否需要加载，从而实现对外部配置文件的依赖。当然，在 Spring Boot 2.x 版本中，已经默认支持了 Spring 5 中对 Conditional OnXXXXXXX 注解的扩展，大大简化了自动配置的开发。

### 1.2. 外部化配置

Spring Boot 默认支持多种外部化配置方式，包括 properties 文件、YAML 文件、 environment variables 和 command line arguments。这些配置会被 Spring Boot 按照优先级从高到低排序，如果存在相同属性的配置，则以优先级较高的为准。

#### 1.2.1. application.properties 与 application.yml

application.properties 和 application.yml 是 Spring Boot 默认支持的两种外部化配置文件，它们的区别在于文件格式和语法。application.properties 采用 key-value 形式，key 和 value 之间用等号连接；而 application.yml 采用 YAML 格式，更符合人类可读性。

#### 1.2.2. environment variables 和 command line arguments

environment variables 和 command line arguments 是另外两种外部化配置方式，它们的优先级比 properties 和 yml 文件要高。在使用这两种方式时，需要将环境变量名或命令行参数转换成驼峰式命名，并加 prefix `SPRING_` 或 `spring.`。

### 1.3. Profile

Profile 是 Spring Boot 中的一种多环境配置机制，允许您在不同环境下使用不同的配置。在 Spring Boot 中，可以通过 `spring.profiles.active` 属性或 `-Dspring.profiles.active` 命令行参数来激活 Profile。

#### 1.3.1. 激活 Profile

激活 Profile 的方式有两种：

* 通过 `spring.profiles.active` 属性激活 Profile：在 application.properties 或 application.yml 文件中添加 `spring.profiles.active=dev,test` 表示激活 dev 和 test Profile。
* 通过 `-Dspring.profiles.active` 命令行参数激活 Profile：在启动 Spring Boot 应用时添加 `-Dspring.profiles.active=dev,test` 表示激活 dev 和 test Profile。

#### 1.3.2. Profile 配置文件

当激活某个 Profile 时，Spring Boot 会加载 profile 配置文件，profile 配置文件名称格式为 `application-{profile}.properties` 或 `application-{profile}.yml`。例如，激活 dev Profile 时，Spring Boot 会加载 `application-dev.properties` 或 `application-dev.yml` 文件。

#### 1.3.3. 共享配置

在 Spring Boot 中，Profile 也可以 sharing configurations between profiles。例如，有一个 config 模块，其中包含了所有环境共享的配置，那么可以在 config 模块中创建一个 `application.properties` 或 `application.yml` 文件，然后在其他环境下创建对应的 profile 配置文件，例如 `application-dev.properties` 或 `application-test.yml`。这样，config 模块中的共享配置就可以在所有环境下使用。

## 核心算法原理和具体操作步骤以及数学模型公式详细讲解

在本节中，我们将详细介绍 Spring Boot 中的配置管理算法原理，包括如何实现条件配置、外部化配置、Profile 以及它们之间的关系。

### 3.1. 条件配置

Spring Boot 利用 Spring Framework 的 IoC / DI 功能，通过 Conditional 注解实现条件配置。在 Spring Boot 中，Condition 接口是所有条件判断的基础接口，Condition 接口定义了 matches 方法，matches 方法用于判断该条件是否满足。

#### 3.1.1. Condition 接口

Condition 接口定义如下：
```java
public interface Condition {
   boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata);
}
```
matches 方法接收两个参数：

* ConditionContext：该参数提供了当前 Condition 的上下文信息，包括 Environment、ResourceLoader 和 BeanDefinitionRegistry。
* AnnotatedTypeMetadata：该参数提供了当前 Condition 的元数据信息，包括注解信息。

#### 3.1.2. @Conditional 注解

@Conditional 注解用于指定 Condition 的条件判断规则。@Conditional 注解可以用在 Configuration 类、Bean 类或方法上。当 @Conditional 注解用在 Configuration 类上时，表示整个 Configuration 类的所有 Bean 都需要满足 Condition 的条件判断规则；当 @Conditional 注解用在 Bean 类或方法上时，表示该 Bean 或方法的生命周期需要满足 Condition 的条件判断规则。

#### 3.1.3. ConditionalOnClass 与 ConditionalOnMissingClass

ConditionalOnClass 和 ConditionalOnMissingClass 是 Spring Boot 中常用的两个 Condition 实现类，分别用于判断 Class 是否存在和不存在。

##### 3.1.3.1. ConditionalOnClass

ConditionalOnClass 实现了 Condition 接口，在 matches 方法中通过 ClassUtils.isPresent 方法判断 Class 是否存在，如果 Class 存在，则返回 true，否则返回 false。

##### 3.1.3.2. ConditionalOnMissingClass

ConditionalOnMissingClass 实现了 Condition 接口，在 matches 方法中通过 ClassUtils.isPresent 方法判断 Class 是否不存在，如果 Class 不存在，则返回 true，否则返回 false。

#### 3.1.4. ConditionalOnProperty

ConditionalOnProperty 是另一个 Spring Boot 中常用的 Condition 实现类，用于判断属性是否存在或具有某个值。

##### 3.1.4.1. ConditionalOnProperty 构造函数

ConditionalOnProperty 实现了 Condition 接口，在构造函数中接收 name、havingValue 和 matchIfMissing 三个参数，分别用于指定属性名称、属性值和是否匹配缺失属性。

##### 3.1.4.2. ConditionalOnProperty matches 方法

ConditionalOnProperty 重写 matches 方法，在 matches 方法中通过 Environment 获取属性值，并根据 havingValue 和 matchIfMissing 参数进行判断。

#### 3.1.5. ConditionalOnExpression

ConditionalOnExpression 是另一个 Spring Boot 中常用的 Condition 实现类，用于判断 SpEL 表达式是否成立。

##### 3.1.5.1. ConditionalOnExpression 构造函数

ConditionalOnExpression 实现了 Condition 接口，在构造函数中接收 expression 参数，用于指定 SpEL 表达式。

##### 3.1.5.2. ConditionalOnExpression matches 方法

ConditionalOnExpression 重写 matches 方法，在 matches 方法中通过 SpelExpressionParser 解析 SpEL 表达式，并根据 evaluate 方法的返回值进行判断。

#### 3.1.6. ConditionalOnJava

ConditionalOnJava 是另一个 Spring Boot 中常用的 Condition 实现类，用于判断 Java 版本是否符合条件。

##### 3.1.6.1. ConditionalOnJava 构造函数

ConditionalOnJava 实现了 Condition 接口，在构造函数中接收 version 参数，用于指定 Java 版本。

##### 3.1.6.2. ConditionalOnJava matches 方法

ConditionalOnJava 重写 matches 方法，在 matches 方法中通过 SystemUtils.getJavaVersion 方法获取 Java 版本，并根据 version 参数进行判断。

### 3.2. 外部化配置

Spring Boot 支持多种外部化配置方式，包括 properties 文件、YAML 文件、environment variables 和 command line arguments。这些配置会被 Spring Boot 按照优先级从高到低排序，如果存在相同属性的配置，则以优先级较高的为准。

#### 3.2.1. properties 文件

properties 文件是 Spring Boot 默认支持的一种外部化配置方式，采用 key-value 形式。properties 文件可以 being placed in the root of the classpath or in a subdirectory under the classpath。Spring Boot 支持多个 properties 文件，并按照名称的前缀进行分类。

##### 3.2.1.1. application.properties

application.properties 是 Spring Boot 默认加载的 properties 文件，位于 src/main/resources 目录下。application.properties 文件中的属性会被 Spring Boot 自动注入 ApplicationContext 中，可以通过 @Value 注解获取。

##### 3.2.1.2. 其他 properties 文件

除 application.properties 之外，Spring Boot 还支持其他 properties 文件，例如 myapp.properties 或 myapp-dev.properties。这些 properties 文件需要满足以下条件：

* 文件名必须以 myapp 开头，后面可以跟随任意非特殊字符。
* 如果存在多个 myapp 开头的 properties 文件，则会按照名称的长度从小到大排序。
* 如果存在多个 myapp 开头的 properties 文件，且名称长度相等，则会按照字母表排序。

#### 3.2.2. YAML 文件

YAML 文件是另一种 Spring Boot 默认支持的外部化配置方式，更符合人类可读性。YAML 文件格式如下：
```yaml
server:
  port: 8080
  servlet:
   context-path: /myapp
spring:
  profiles:
   active: dev
```
YAML 文件中的属性也会被 Spring Boot 自动注入 ApplicationContext 中，可以通过 @Value 注解获取。

#### 3.2.3. environment variables

environment variables 是另一种 Spring Boot 支持的外部化配置方式，优先级比 properties 和 yml 文件要高。在使用 environment variables 时，需要将环境变量名转换成驼峰式命名，并加 prefix `SPRING_` 或 `spring.`。

##### 3.2.3.1. SPRING_

当环境变量名以 `SPRING_` 为前缀时，Spring Boot 会将环境变量名转换成驼峰式命名，并注入 ApplicationContext 中。例如，设置环境变量 `SPRING_SERVLET_CONTEXT_PATH=/myapp`，则会注入 `servlet.contextPath` 属性。

##### 3.2.3.2. spring.

当环境变量名以 `spring.` 为前缀时，Spring Boot 会直接注入 ApplicationContext 中。例如，设置环境变量 `spring.servlets.context-path=/myapp`，则会直接注入 `servlet.contextPath` 属性。

#### 3.2.4. command line arguments

command line arguments 是另一种 Spring Boot 支持的外部化配置方式，优先级比 properties 和 yml 文件要高。在使用 command line arguments 时，需要将参数名转换成驼峰式命名，并加 prefix `--spring.`。

##### 3.2.4.1. --spring.

当 command line arguments 名以 `--spring.` 为前缀时，Spring Boot 会直接注入 ApplicationContext 中。例如，启动 Spring Boot 应用时添加 `--spring.servlet.context-path=/myapp`，则会直接注入 `servlet.contextPath` 属性。

### 3.3. Profile

Profile 是 Spring Boot 中的一种多环境配置机制，允许您在不同环境下使用不同的配置。在 Spring Boot 中，可以通过 `spring.profiles.active` 属性或 `-Dspring.profiles.active` 命令行参数来激活 Profile。

#### 3.3.1. 激活 Profile

激活 Profile 的方式有两种：

* 通过 `spring.profiles.active` 属性激活 Profile：在 application.properties 或 application.yml 文件中添加 `spring.profiles.active=dev,test` 表示激活 dev 和 test Profile。
* 通过 `-Dspring.profiles.active` 命令行参数激活 Profile：在启动 Spring Boot 应用时添加 `-Dspring.profiles.active=dev,test` 表示激活 dev 和 test Profile。

#### 3.3.2. Profile 配置文件

当激活某个 Profile 时，Spring Boot 会加载 profile 配置文件，profile 配置文件名称格式为 `application-{profile}.properties` 或 `application-{profile}.yml`。例如，激活 dev Profile 时，Spring Boot 会加载 `application-dev.properties` 或 `application-dev.yml` 文件。

#### 3.3.3. 共享配置

在 Spring Boot 中，Profile 也可以 sharing configurations between profiles。例如，有一个 config 模块，其中包含了所有环境共享的配置，那么可以在 config 模块中创建一个 `application.properties` 或 `application.yml` 文件，然后在其他环境下创建对应的 profile 配置文件，例如 `application-dev.properties` 或 `application-test.yml`。这样，config 模块中的共享配置就可以在所有环境下使用。

## 具体最佳实践：代码实例和详细解释说明

在本节中，我们将介绍如何在 Spring Boot 中实现条件配置、外部化配置和 Profile。

### 4.1. 条件配置

在本节中，我们将演示如何在 Spring Boot 中实现条件配置。

#### 4.1.1. ConditionalOnClass 与 ConditionalOnMissingClass

ConditionalOnClass 和 ConditionalOnMissingClass 是 Spring Boot 中常用的两个 Condition 实现类，分别用于判断 Class 是否存在和不存在。

##### 4.1.1.1. ConditionalOnClass 实例

首先，创建一个 MyService 接口，定义一个 hello 方法：
```java
public interface MyService {
   void hello();
}
```
然后，创建一个 MyServiceImpl1 类，实现 MyService 接口：
```java
@Component
public class MyServiceImpl1 implements MyService {
   @Override
   public void hello() {
       System.out.println("Hello from MyServiceImpl1");
   }
}
```
接下来，创建一个 MyServiceImpl2 类，也实现 MyService 接口：
```java
@Component
public class MyServiceImpl2 implements MyService {
   @Override
   public void hello() {
       System.out.println("Hello from MyServiceImpl2");
   }
}
```
最后，创建一个 Config 类，利用 ConditionalOnClass 和 ConditionalOnMissingClass 实现条件配置：
```java
@Configuration
public class Config {

   @Bean
   @ConditionalOnClass(MyServiceImpl1.class)
   public MyService myService1() {
       return new MyServiceImpl1();
   }

   @Bean
   @ConditionalOnMissingClass(MyServiceImpl1.class)
   public MyService myService2() {
       return new MyServiceImpl2();
   }
}
```
在 Config 类中，通过 ConditionalOnClass 注解指定如果 MyServiceImpl1 类存在，则创建 MyServiceImpl1  bean；通过 ConditionalOnMissingClass 注解指定如果 MyServiceImpl1 类不存在，则创建 MyServiceImpl2 bean。

##### 4.1.1.2. ConditionalOnClass 测试

在测试类中，通过 @Autowired 注入 MyService bean：
```java
@RunWith(SpringRunner.class)
@SpringBootTest
public class ConditionalOnClassTest {

   @Autowired
   private MyService myService;

   @Test
   public void testMyService() {
       myService.hello();
   }
}
```
当 MyServiceImpl1 类存在时，测试结果为：
```csharp
Hello from MyServiceImpl1
```
当 MyServiceImpl1 类不存在时，测试结果为：
```csharp
Hello from MyServiceImpl2
```
#### 4.1.2. ConditionalOnProperty

ConditionalOnProperty 是另一个 Spring Boot 中常用的 Condition 实现类，用于判断属性是否存在或具有某个值。

##### 4.1.2.1. ConditionalOnProperty 实例

首先，创建一个 Config 类，通过 ConditionalOnProperty 注解实现条件配置：
```java
@Configuration
public class Config {

   @Bean
   @ConditionalOnProperty(name = "myapp.enabled", havingValue = "true")
   public MyService myService1() {
       return new MyServiceImpl1();
   }

   @Bean
   @ConditionalOnProperty(name = "myapp.enabled", matchIfMissing = true)
   public MyService myService2() {
       return new MyServiceImpl2();
   }
}
```
在 Config 类中，通过 ConditionalOnProperty 注解指定如果属性 myapp.enabled 存在且等于 true，则创建 MyServiceImpl1 bean；通过 ConditionalOnProperty 注解指定如果属性 myapp.enabled 不存在或等于 false，则创建 MyServiceImpl2 bean。

##### 4.1.2.2. ConditionalOnProperty 测试

在测试类中，通过 @Autowired 注入 MyService bean：
```java
@RunWith(SpringRunner.class)
@SpringBootTest
public class ConditionalOnPropertyTest {

   @Autowired
   private MyService myService;

   @Test
   public void testMyService() {
       myService.hello();
   }
}
```
当属性 myapp.enabled 存在且等于 true 时，测试结果为：
```csharp
Hello from MyServiceImpl1
```
当属性 myapp.enabled 不存在或等于 false 时，测试结果为：
```csharp
Hello from MyServiceImpl2
```
### 4.2. 外部化配置

在本节中，我们将演示如何在 Spring Boot 中实现外部化配置。

#### 4.2.1. properties 文件

properties 文件是 Spring Boot 默认支持的一种外部化配置方式，采用 key-value 形式。properties 文件可以 being placed in the root of the classpath or in a subdirectory under the classpath。Spring Boot 支持多个 properties 文件，并按照名称的前缀进行分类。

##### 4.2.1.1. application.properties

application.properties 是 Spring Boot 默认加载的 properties 文件，位于 src/main/resources 目录下。application.properties 文件中的属性会被 Spring Boot 自动注入 ApplicationContext 中，可以通过 @Value 注解获取。

##### 4.2.1.2. application.properties 测试

首先，在 application.properties 文件中添加以下内容：
```csharp
myapp.title=My App
myapp.description=This is my app
myapp.version=1.0.0
```
然后，在测试类中，通过 @Value 注入属性：
```java
@RunWith(SpringRunner.class)
@SpringBootTest
public class ExternalizedPropertiesTest {

   @Value("${myapp.title}")
   private String title;

   @Value("${myapp.description}")
   private String description;

   @Value("${myapp.version}")
   private String version;

   @Test
   public void testExternalizedProperties() {
       System.out.println("Title: " + title);
       System.out.println("Description: " + description);
       System.out.println("Version: " + version);
   }
}
```
测试结果为：
```makefile
Title: My App
Description: This is my app
Version: 1.0.0
```
#### 4.2.2. YAML 文件

YAML 文件是另一种 Spring Boot 默认支持的外部化配置方式，更符合人类可读性。YAML 文件格式如下：
```yaml
server:
  port: 8080
  servlet:
   context-path: /myapp
spring:
  profiles:
   active: dev
myapp:
  title: My App
  description: This is my app
  version: 1.0.0
```
YAML 文件中的属性也会被 Spring Boot 自动注入 ApplicationContext 中，可以通过 @Value 注解获取。

##### 4.2.2.1. YAML 文件测试

首先，在 application.yml 文件中添加以下内容：
```yaml
myapp:
  title: My App
  description: This is my app
  version: 1.0.0
```
然后，在测试类中，通过 @Value 注入属性：
```java
@RunWith(SpringRunner.class)
@SpringBootTest
@ActiveProfiles("dev")
public class ExternalizedYamlTest {

   @Value("${myapp.title}")
   private String title;

   @Value("${myapp.description}")
   private String description;

   @Value("${myapp.version}")
   private String version;

   @Test
   public void testExternalizedYaml() {
       System.out.println("Title: " + title);
       System.out.println("Description: " + description);
       System.out.println("Version: " + version);
   }
}
```
测试结果为：
```makefile
Title: My App
Description: This is my app
Version: 1.0.0
```
#### 4.2.3. environment variables

environment variables 是另一种 Spring Boot 支持的外部化配置方式，优先级比 properties 和 yml 文件要高。在使用 environment variables 时，需要将环境变量名转换成驼峰式命名，并加 prefix `SPRING_` 或 `spring.`。

##### 4.2.3.1. SPRING\_ 实例

首先，在操作系统中设置环境变量 `SPRING_MYAPP_TITLE=My App`。

然后，在测试类中，通过 @Value 注入属性：
```java
@RunWith(SpringRunner.class)
@SpringBootTest
public class ExternalizedEnvironmentVariablesTest {

   @Value("${myapp.title}")
   private String title;

   @Test
   public void testExternalizedEnvironmentVariables() {
       System.out.println("Title: " + title);
   }
}
```
测试结果为：
```makefile
Title: My App
```
##### 4.2.3.2. spring.\_ 实例

首先，在操作系统中设置环境变量 `spring.myapp.title=My App`。

然后，在测试类中，通过 @Value 注入属性：
```java
@RunWith(SpringRunner.class)
@SpringBootTest
public class ExternalizedEnvironmentVariablesTest {

   @Value("${myapp.title}")
   private String title;

   @Test
   public void testExternalizedEnvironmentVariables() {
       System.out.println("Title: " + title);
   }
}
```
测试结果为：
```makefile
Title: My App
```
### 4.3. Profile

Profile 是 Spring Boot 中的一种多环境配置机制，允许您在不同环境下使用不同的配置。在 Spring Boot 中，可以通过 `spring.profiles.active` 属性或 `-Dspring.profiles.active` 命令行参数来激活 Profile。

#### 4.3.1. 激活 Profile

激活 Profile 的方式有两种：

* 通过 `spring.profiles.active` 属性激活 Profile：在 application.properties 或 application.yml 文件中添加 `spring.profiles.active=dev,test` 表示激活 dev 和 test Profile。
* 通过 `-Dspring.profiles.active` 命令行参数激活 Profile：在启动 Spring Boot 应用时添加 `-Dspring.profiles.active=dev,test` 表示激活 dev 和 test Profile。

#### 4.3.2. Profile 配置文件

当激活某个 Profile 时，Spring Boot 会加载 profile 配置文件，profile 配置文件名称格式为 `application-{profile}.properties` 或 `application-{profile}.yml`。例如，激活 dev Profile 时，Spring Boot 会加载 `application-dev.properties` 或 `application-dev.yml` 文件。

##### 4.3.2.1. Profile 实例

首先，创建一个 config 模块，在 config 模块中创建一个 application.properties 文件，并添加以下内容：
```csharp
myapp.title=Config Module
myapp.description=This is the Config Module
myapp.version=1.0.0
```
然后，在 config 模块中创建一个 application-dev.properties 文件，并添加以下内容：
```csharp
myapp.title=Dev Environment
myapp.description=This is the Dev Environment
myapp.version=1.0.0-SNAPSHOT
```
接着，在 config 模块中创建一个 application-test.properties 文件，并添加以下内容：
```csharp
myapp.title=Test Environment
myapp.description=This is the Test Environment
myapp.version=1.0.0-RC
```
最后，在 main 模块中，引入 config 模块，并在 application.properties 文件中添加以下内容：
```bash
spring.profiles.include=dev,test
```
这样，当启动 main 模块时，会自动激活 dev 和 test Profile，并加载 config 模块中对应的 profile 配置文件。

##### 4.3.2.2. Profile 测试

在测试类中，通过 @Value 注入属性：
```java
@RunWith(SpringRunner.class)
@SpringBootTest
public class ProfileTest {

   @Value("${myapp.title}")
   private String title;

   @Value("${myapp.description}")
   private String description;

   @Value("${myapp.version}")
   private String version;

   @Test
   public void testProfile() {
       System.out.println("Title: " + title);
       System.out.println("Description: " + description);
       System.out.println("Version: " + version);
   }
}
```
测试结果为：
```makefile
Title: Dev Environment
Description: This is the Dev Environment
Version: 1.0.0-SNAPSHOT
```
## 实际应用场景

在本节中，我们将介绍如何在实际应用场景中利用 Spring Boot 的配置管理功能。

### 5.1. 分布式追踪

分布式追踪是微服务架构中的一项关键技术，它允许我们跟踪请求在各个服务之间的传递情况，从而更好地排查问题。Spring Boot 支持多种分布式追踪系统，包括 OpenTracing、Zipkin 和 Jaeger。

#### 5.1.1. OpenTracing

OpenTracing 是一个开放的分布式追踪标准，定义了一个 API 和一组协议，允许我们在不同的语言和框架之间使用相同的分布式追踪系统。Spring Boot 支持 OpenTracing 标准，可以通过 `spring-cloud-sleuth-opentracing` 依赖来启用 OpenTracing 支持。

##### 5.1.1.1. OpenTracing 配置

首先，在 application.yml 文件中添加以下内容：
```yaml
spring:
  sleuth:
   opentracing:
     enabled: true
     provider: jaeger
     endpoint: localhost:6831
```
这里，我们启用 OpenTracing 支持，选择 Jaeger 作为追踪提供商，指定 Jaeger 的端点为 localhost:6831。

##### 5.1.1.2. OpenTracing 测试

在测试类中，通过 @Autowired 注入 Tracer 对象：
```java
@RunWith(SpringRunner.class)
@SpringBootTest
public class OpenTracingTest {

   @Autowired
   private Tracer tracer;

   @Test
   public void testOpenTracing() {
       Span span = tracer.buildSpan("test").start();
       try (Scope scope = tracer.activateSpan(span)) {
           // ... do some work ...
       } finally {
           span.finish();
       }
   }
}
```
当测试运行时，Jaeger 会捕获 traces，并显示在 UI 中。

#### 5.1.2. Zipkin

Zipkin 是一个分布式追踪系统，它允许我们收集请求的 traces，并在 UI 中可视化 trace 信息。Spring Boot 支持 Zipkin 追踪系统，