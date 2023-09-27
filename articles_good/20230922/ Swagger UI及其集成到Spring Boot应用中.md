
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Swagger 是一款开源、功能丰富的 API 概述文档工具。它能够帮助我们快速、清晰地定义、结构化和展示我们的 API，同时也支持多种开发语言，如 Java、JavaScript、Python等。在 Spring Boot 中，使用 Swagger 可以极大方便 API 的测试、调试以及提供给其他团队进行交流学习。本文将会用简单的方式对 Spring Boot 和 Swagger 有个整体的认识，并通过一个实际案例——集成 Spring Boot 项目中 Swagger UI 来展现它的优点。
# 2.术语定义
## 2.1 Swagger
Swagger (发音同“栅栏”，如 Swagger UI) 是一款开源的 API 接口文档生成工具，是一个规范和定义如何描述、发布、消费 RESTful Web 服务的轻量级的方法。它提供了基于 OpenAPI（开放式接口通信标准）的Restful风格的API接口文档，使得服务的消费者可以直观地浏览各个服务 endpoints、methods 和 parameters，从而更好地理解、使用、交流和描述 APIs 。其主要特性包括：
* 提供完整的 API 描述，包括可视化接口定义图表；
* 支持多种开发语言，如 Java、Javascript、PHP、Python等；
* 可生成客户端 SDK，减少重复开发工作；
* 支持 Restful API 参数验证、请求示例和响应示例自动生成；
* 提供插件扩展能力，可用于实现特定的功能；
* 支持 OAuth 2.0 授权和身份验证机制。

## 2.2 OpenAPI
OpenAPI （开放式接口说明） 是一种用来描述 API 的数据交换格式，它是 Swagger 提出的一种方案。它是基于 JSON 或 YAML 格式的文本文件，里面包含了关于服务器的基本信息、paths(端点)、operations(方法)及 parameters(参数)。通过这些文件，可以通过不同编程语言或者框架生成对应的 API Client SDK，实现服务的调用。目前，Swagger 2.0、OpenAPI 3.0版本都已经推出。

## 2.3 Spring Boot
Spring Boot 是 Spring 家族中的一个新开源项目，它全面兼容 Spring 生态系统，是构建生产级 Spring 应用程序的最佳选择。它通过 starter 模块简化了配置，并且内置 Tomcat/Jetty web 容器，因此开发人员不需要自己搭建这些容器。Spring Boot 不仅适用于微服务架构，也可以应用于传统的 Servlet 环境。

## 2.4 Spring MVC
Spring MVC 是 Spring 的一个模块，它负责处理模型-视图-控制器（Model-View-Controller，MVC）架构中的用户请求，提供RESTful风格的 URL。它是一个轻量级的Web框架，Spring MVC 中的控制器（Controller）负责处理用户请求，Spring MVC 中的视图解析器（ViewResolver）负责渲染视图，Spring MVC 的入口类为 DispatcherServlet。

# 3.集成 Swagger UI 到 Spring Boot 工程中
## 3.1 添加依赖
首先，需要在 pom.xml 文件中添加 Swagger UI 相关依赖。以下只列举核心依赖，其它按需引入即可：
``` xml
    <dependencies>
        <!-- Spring Boot -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- Swagger UI -->
        <dependency>
            <groupId>io.springfox</groupId>
            <artifactId>springfox-swagger2</artifactId>
            <version>${swagger.version}</version>
        </dependency>
        <dependency>
            <groupId>io.springfox</groupId>
            <artifactId>springfox-swagger-ui</artifactId>
            <version>${swagger.version}</version>
        </dependency>

        <!-- Swagger 配置-->
        <dependency>
            <groupId>io.github.jhipster</groupId>
            <artifactId>jhipster-framework</artifactId>
            <exclusions>
                <exclusion>
                    <groupId>com.fasterxml.jackson.dataformat</groupId>
                    <artifactId>jackson-datatype-hibernate5</artifactId>
                </exclusion>
            </exclusions>
        </dependency>
    </dependencies>

    <properties>
        <!-- Swagger 版本号 -->
        <swagger.version>2.9.2</swagger.version>
    </properties>
``` 

其中 io.springfox 为 Swagger 依赖库的 groupId，swagger.version 为 swagger 版本号。由于 swagger 默认不支持 Hibernate 实体类生成 JSON Schema，所以为了支持，这里排除了 jackson-datatype-hibernate5 依赖，否则会报错。

## 3.2 创建 SwaggerConfig 配置类
创建类 SwaggerConfig，并继承 WebMvcConfigurerAdapter。该类提供了定制 Spring MVC 的方法，包括 addResourceHandlers()、addViewControllers()、configurePathMatch() 等。其中，addResourceHandlers() 方法用于设置静态资源路径，addViewControllers() 方法用于设置视图控制器，通常情况下不用修改，但对于 swagger ui 需要设置为 swagger-ui.html 路径。configurePathMatch() 方法用于匹配 url 请求路径。

``` java
@Configuration
public class SwaggerConfig extends WebMvcConfigurerAdapter {
    
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        // 设置静态资源路径
        registry.addResourceHandler("/**")
               .addResourceLocations("classpath:/META-INF/resources/")
               .addResourceLocations("classpath:/resources/")
               .addResourceLocations("classpath:/static/")
               .addResourceLocations("classpath:/public/");
        
        // 设置视图控制器，将 /swagger-ui.html 映射到默认的 index.html，
        // 以便浏览器在刷新时显示 swagger ui
        registry.addViewController("/").setViewName("forward:/swagger-ui.html");
        registry.addViewController("/swagger-ui.html").setViewName("forward:index.html");
        registry.addViewController("/v2/api-docs").setViewName("forward:/swagger-ui/index.html");
        registry.addViewController("/webjars/**").setViewName("forward:/swagger-ui/bower_components/{path}");
    }
}
```

## 3.3 修改启动类
若 Spring Boot 使用的是 Maven 作为构建工具，则需要在启动类上添加注解 EnableSwagger2，以启用 Swagger UI：
``` java
@EnableSwagger2
@SpringBootApplication
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}
```

若 Spring Boot 使用的是 Gradle 作为构建工具，则需要在 build.gradle 文件中添加如下配置：
``` groovy
apply plugin: 'java'
apply plugin: 'org.springframework.boot'
apply plugin: 'io.spring.dependency-management'

group = 'com.example'
version = '0.0.1-SNAPSHOT'
sourceCompatibility = 1.8

repositories {
    mavenCentral()
}

ext {
    set('mapstruct.version', "1.3.0.Final")
}

configurations {
    developmentOnly
    runtimeClasspath {
        extendsFrom developmentOnly
    }
}

dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-actuator'
    implementation 'org.springframework.boot:spring-boot-starter-security'
    implementation 'org.springframework.boot:spring-boot-starter-web'

    compileOnly 'org.projectlombok:lombok'
    annotationProcessor 'org.projectlombok:lombok'
    testCompileOnly 'org.projectlombok:lombok'
    testAnnotationProcessor 'org.projectlombok:lombok'

    implementation 'io.micrometer:micrometer-registry-prometheus'
    implementation 'io.prometheus:simpleclient_hotspot'
    implementation 'org.springframework.cloud:spring-cloud-starter-consul-discovery'
    implementation 'org.springframework.cloud:spring-cloud-config-server'
    implementation 'org.springframework.retry:spring-retry'
    implementation 'org.springframework.session:spring-session-core'
    implementation 'org.springframework.boot:spring-boot-starter-oauth2-client'
    implementation 'org.springframework.cloud:spring-cloud-starter-sleuth'

    implementation 'io.springfox:springfox-swagger2'
    implementation 'io.springfox:springfox-swagger-ui'

    testImplementation 'org.springframework.boot:spring-boot-starter-test'
    testImplementation 'org.springframework.security:spring-security-test'
    testImplementation 'io.rest-assured:rest-assured'
    testImplementation 'io.jsonwebtoken:jjwt'
    testImplementation 'org.junit.platform:junit-platform-console'
    testImplementation group: 'junit', name: 'junit', version: '4.12'
    testImplementation group: 'org.mockito', name:'mockito-core', version: '3.1.0'
}

dependencyManagement {
    imports {
        mavenBom "org.springframework.boot:spring-boot-dependencies:${springBootVersion}"
        mavenBom "org.springframework.cloud:spring-cloud-dependencies:${springCloudVersion}"
        mavenBom "io.zipkin.brave:brave-bom:${zipkinBraveVersion}"
    }
}

test {
    useJUnitPlatform()
    maxHeapSize = "4g"
    minHeapSize = "2g"
}

task wrapper(type: Wrapper) {
    gradleVersion = '5.5.1'
}

// enable for swagger
compileJava {
    dependsOn generateSwaggerDocumentation
}

tasks.named("classes") {
    dependsOn generateSwaggerDocumentation
}

generateSwaggerDocumentation.outputs.upToDateWhen { false }
```

修改完毕后，重新运行程序，然后访问 http://localhost:8080/swagger-ui.html ，就可以看到 Swagger UI 的主界面。