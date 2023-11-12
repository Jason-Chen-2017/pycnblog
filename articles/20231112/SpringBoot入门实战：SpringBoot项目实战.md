                 

# 1.背景介绍


## 概念
Spring Boot是一个新的微服务开发框架，它使得开发单个、微服务或云应用更加容易。该框架使用了特定的方式来进行配置，通过一系列自动化配置项，帮助我们快速启动一个可运行的Spring应用程序。
## SpringBoot特点
- **无XML配置** : Spring Boot基于Spring Framework，在一定程度上简化了配置，允许用户使用简单、少量代码完成Spring Bean定义。
- **内嵌Servlet容器** : Spring Boot依赖于Tomcat或Jetty等轻量级的嵌入式Servlet容器，不需要部署 WAR 文件到外部容器。
- **提供 starter POMs** : Spring Boot 提供了一系列的 Starter POMs，方便开发者获取所需功能模块，如数据访问（jpa）、安全性（security）、缓存（redis/caffeine）、消息代理（amqp）、web开发（Thymeleaf/FreeMarker）等。
- **自动装配能力** : Spring Boot 使用 spring-boot-autoconfigure 模块对 Spring Bean 的自动配置。
- **命令行界面** : Spring Boot 为 spring 命令提供了强大的 shell 支持，包括运行、调试及监控 Spring Boot 应用。
- **无代码生成和 XML 配置** : 在 Spring Boot 中，可以使用 Java Config 或注解的方式来实现 bean 注入，避免了配置繁琐的 XML 配置文件。
- **测试支持** : Spring Boot 提供了 spring-boot-starter-test 模块和 JUnit、TestNG、Spock 测试框架支持，可以用于编写单元测试和集成测试用例。
## SpringBoot优点
- **创建独立运行的 jar 文件** : Spring Boot 可以创建一个独立运行的 Jar 文件，里面已经包含了所有运行环境需要的类库，不再需要像传统的 Spring 项目那样依赖 Tomcat 或 Jetty。
- **内置 Tomcat 服务器** : Spring Boot 默认使用内置 Tomcat 来作为服务端，也无需额外安装 Tomcat 到系统中。
- **无需 XML 配置** : Spring Boot 无需 xml 配置文件，只需要简单配置 java 配置即可，适合小型项目的快速开发。
- **热加载重启** : Spring Boot 开发过程中，修改代码可以立即生效，无需重新启动服务器，大大提升开发速度。
- **内置数据库支持** : Spring Boot 支持多种关系型数据库，比如 MySQL 和 PostgreSQL，并且提供了对 Hibernate、MyBatis 和 JPA 的支持。
- **支持多种开发语言** : Spring Boot 支持 Java、Groovy、Kotlin、Scala 等主流编程语言，可以快速构建各种 Web 应用。
- **提供插件扩展机制** : Spring Boot 通过 SPI (Service Provider Interface) 机制提供插件扩展能力，可以通过实现对应的接口快速构建功能。
- **可选注解配置** : Spring Boot 有很多可选注解配置，能够减少配置复杂度，并使得代码风格与官方推荐风格保持一致。
- **自动化配置** : Spring Boot 会自动根据你添加的jar依赖以及配置信息，帮你进行相应配置。
- **完善的 metrics 统计工具** : Spring Boot 提供了完善的 metrics 统计工具，能够很好的展示你的应用的各项指标。
- **高度可定制化** : Spring Boot 提供非常灵活的扩展机制，你可以自定义默认配置或者扩展已有的功能。
# 2.核心概念与联系
## 2.1 Bean
Bean 是 Spring 的核心组件之一，它代表着 Spring IOC 容器管理的一个对象，其持有一个或多个属性，这些属性值可能来源于配置文件中的设置或其他 Beans 中的计算结果。一般情况下，Bean 都是由 BeanFactoryPostProcessor 来进行配置的。在 Spring Boot 中，自动配置会扫描我们引入的依赖包，判断是否存在特定注解（例如 @Configuration、@Component），如果存在则会尝试进行自动配置。
### 2.1.1 Bean 作用域
Bean 的作用域决定了 Bean 在 Spring IoC 容器中的生命周期，Spring 支持以下五种作用域：
- Singleton （单例模式）: 唯一Bean实例，全局 accessible。
- Prototype （原型模式）: 每次请求都会产生一个新的Bean实例。
- Request （请求范围）：在一次HTTP请求中，Bean实例被缓存，不同的线程可以共享相同的Bean实例。
- Session （会话范围）：在一次 HTTP 会话中，Bean 实例被缓存，不同的线程可以共享相同的 Bean 实例。
- Grace Period （宽限期作用域）：类似于原型作用域，但是在一段时间内会从容器中删除这些Bean实例，主要用于临时场景。
### 2.1.2 Bean 命名
在 Spring 中，Bean 的名称应该是唯一的。但是在 Spring Boot 中，如果没有特别指定，Bean 的名字是用全类名来表示的。例如，下面两个 Bean 实际上是同一个 Bean：
```java
public class MyBean {
    //...
}
```
```xml
<bean id="myBean" class="com.example.demo.MyBean"/>
```
不过，建议给 Bean 指定一个明确的名称，这样在我们后面配置的时候就不会出错：
```xml
<bean id="myBean" class="com.example.demo.MyBean">
  <property name="name" value="John Doe"/>
</bean>
```
### 2.1.3 Bean 生命周期
在 Spring 中，Bean 的生命周期分为以下几步：

1. 创建 Bean；
2. 设置 Bean 属性；
3. 初始化 Bean；
4. 使用 Bean；
5. 销毁 Bean。

Bean 的生命周期在 Spring 中有两种不同的方法来实现：一种是继承接口 InitializingBean ，另一种是实现接口 DisposableBean 。另外，在 JavaConfig 中还能通过 @PostConstruct 和 @PreDestroy 来控制 Bean 的生命周期。在 Spring Boot 中，我们也可以在 Bean 上添加注解来控制 Bean 的生命周期，例如：

- @PostConstruct : 表示在初始化 Bean 之后执行的方法，通常用于一些资源初始化工作，例如打开文件等。
- @PreDestroy : 表示在销毁 Bean 之前执行的方法，通常用于一些资源清理工作，例如关闭连接池等。
- @Value : 用在字段上，用于将属性的值绑定到字段中，例如：`@Value("${server.port}") int port;`。

下面是在 Spring Boot 中定义 Bean 时，生命周期的顺序：

1. 执行 Bean 构造器；
2. 执行 @Autowired、@Inject 注入；
3. 执行 @PostConstruct；
4. 执行 afterPropertiesSet();
5. 使用 Bean；
6. 执行 @PreDestory；
7. 执行 destroy() 方法；

### 2.1.4 配置 Bean
#### 2.1.4.1 配置元数据
在 Spring 中，我们可以通过 XML 文件或者注解来定义 Bean，但是在 Spring Boot 中，我们往往使用 Java Config。因为 XML 很难阅读和维护，而且配置起来比较麻烦。在 Java Config 中，我们可以通过 `@Configuration`、`@ComponentScan`、`@ImportResource`、`@Bean` 等注解来定义 Bean，而这些注解最终都映射到 XML 文件上的配置元数据中。
#### 2.1.4.2 属性占位符 ${…}
在 Java Config 中，我们可以使用 `${...}` 来引用配置文件中的变量。例如：
```java
@Configuration
@PropertySource("classpath:/application.properties")
public class AppConfig {

    private final String property;
    
    public AppConfig(@Value("${app.property}") String property) {
        this.property = property;
    }
    
}
```
这里假设配置文件 `classpath:/application.properties` 中有一项键值对 `"app.property": "someValue"`，那么当我们 new 一个 `AppConfig()` 对象时，它的 `this.property` 的值为 `"someValue"`。
#### 2.1.4.3 自动配置
在 Spring Boot 中，我们可以通过 starter（起步依赖）来导入相关的依赖和配置，Spring Boot 将自动配置相关的 Bean。Starter 的意义在于提供开箱即用的依赖集合，让开发人员无需关注繁杂的配置，即可快速地使用某些功能。例如，要使用 Spring Data JPA，我们只需引入 starter-data-jpa 就可以了，不需要再去配置繁杂的 JDBC 连接池。
#### 2.1.4.4 根据名称查找 Bean
在 Spring 中，我们可以通过 `ApplicationContext` 的 `getBean(String)` 方法来根据 Bean 的名称来查找 Bean。但在 Spring Boot 中，我们也可以直接在 @Configuration 类中用 @Bean 的方法来声明 Bean，然后通过注解或通过 ApplicationContext 获取 Bean。例如：
```java
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.*;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.orm.jpa.JpaVendorAdapter;
import org.springframework.orm.jpa.LocalContainerEntityManagerFactoryBean;
import org.springframework.orm.jpa.vendor.HibernateJpaVendorAdapter;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;

import javax.persistence.EntityManagerFactory;
import javax.sql.DataSource;
import java.util.HashMap;
import java.util.Map;

@Configuration
@EnableTransactionManagement
public class DatabaseConfig {

  @Primary
  @Bean(name = "dataSource", initMethod = "initialize", destroyMethod = "close")
  public DataSource dataSource() throws Exception {
      DriverManagerDataSource dataSource = new DriverManagerDataSource();
      dataSource.setDriverClassName("org.h2.Driver");
      dataSource.setUrl("jdbc:h2:mem:testdb");
      dataSource.setUsername("sa");
      dataSource.setPassword("");
      return dataSource;
  }
  
  @Primary
  @Bean(name = "entityManagerFactory")
  public LocalContainerEntityManagerFactoryBean entityManagerFactory(
      EntityManagerFactoryBuilder builder,
      @Qualifier("dataSource") DataSource dataSource,
      @Qualifier("jpaVendorAdapter") JpaVendorAdapter jpaVendorAdapter) {
      
      Map<String, Object> properties = new HashMap<>();
      properties.put("hibernate.hbm2ddl.auto", "create-drop");
      properties.put("hibernate.dialect", "org.hibernate.dialect.H2Dialect");

      LocalContainerEntityManagerFactoryBean factoryBean =
          builder
             .dataSource(dataSource)
             .packages("com.example.demo.entity")
             .properties(properties)
             .jtaDataSource(null)
             .persistenceUnitName("")
             .build();

      factoryBean.setJpaVendorAdapter(jpaVendorAdapter);

      return factoryBean;
  }
  
  @Primary
  @Bean(name = "jpaVendorAdapter")
  public JpaVendorAdapter jpaVendorAdapter() {
      return new HibernateJpaVendorAdapter();
  }

  @Primary
  @Bean(name = "transactionManager")
  public PlatformTransactionManager transactionManager(
      @Qualifier("entityManagerFactory") EntityManagerFactory entityManagerFactory) {
      return new JpaTransactionManager(entityManagerFactory);
  }
  
}
```
在上面的例子中，我们声明了一个名为 `DatabaseConfig` 的 @Configuration 类，其中包含了三个 Bean。第一个 Bean 是名为 `dataSource` 的 `DataSource`，它通过 H2 内存数据库作为底层存储。第二个 Bean 是名为 `entityManagerFactory` 的 `LocalContainerEntityManagerFactoryBean`，它通过 Hibernate 来管理实体的生命周期。第三个 Bean 是名为 `jpaVendorAdapter` 的 `JpaVendorAdapter`，它描述了 Hibernate 所使用的 JPA 实现。第四个 Bean 是名为 `transactionManager` 的 `PlatformTransactionManager`，它用来管理事务。