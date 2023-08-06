
作者：禅与计算机程序设计艺术                    

# 1.简介
         
12. Spring MVC实战：入门篇（一）概述

         概述
         什么是Spring MVC？Spring MVC是一个基于Java的模型-视图-控制器(MVC) Web框架，它提供一个用于开发灵活、松耦合的web应用程序的MVC设计模式。本文将带领读者一起学习Spring MVC框架的基础知识、配置项、用法及其实现原理。

         本系列文章共分成两部分，第一部分将带领读者了解Spring MVC相关概念和知识，包括Spring MVC的发展史、设计理念、运行流程等；第二部分则将帮助读者熟悉Spring MVC的用法、配置项、编程接口、扩展点、框架组件等知识。

         12. Spring MVC实战：入门篇（二）Spring MVC架构
         Spring MVC架构
         1. DispatcherServlet
            - 可以看作是前端控制器，用来分派请求到相应的Controller。
         2. HandlerMapping
            - 根据用户请求获取相应的Handler对象。处理器映射器从配置文件中读取一组HandlerMapping对象并按顺序执行它们来查找是否存在能够处理该请求的Handler对象。
         3. HandlerAdapter
            - 将HttpServletRequest和HttpServletResponse封装成适配器方法参数，通过反射调用Handler的方法处理请求，并返回一个 ModelAndView 对象。
            - 如果Handler不能直接处理请求，则抛出异常。
         4. ViewResolver
            - 从逻辑视图名称解析真实的视图资源。
            - 使用ViewResolvers和Views可以动态切换视图，提高程序的可维护性。
         5. LocaleResolver
            - 确定用户请求的Locale，根据不同的策略进行设置。
            - 可用于多语言支持。
         6. MultipartResolver
            - 对文件上传进行管理，处理请求中的multipart/form-data。
         7. FlashMapManager
            - 管理FlashMap。





         # 2.基本概念术语说明
         2.Spring Bean容器
         2.1.BeanFactory
         2.2.ApplicationContext
         2.3.WebApplicationContext
         2.4.Bean
         2.5.依赖注入
         2.6.Bean生命周期
         2.7.PostProcessor
         2.8.AOP
         2.9.AspectJ
         2.10.注解
         2.11.xml
         2.12.Properties文件
         2.13.JavaBeans
         2.14.XML配置
         2.15.组件扫描
         2.16.自动装配
         2.17.工厂Bean
         2.18.Lazy-Initialization
         2.19.Lazy-Loading
          
         


         # 3.核心算法原理和具体操作步骤以及数学公式讲解
         3.Spring MVC用法
         3.1.创建项目
         3.2.配置maven
         3.3.创建web.xml文件
         3.4.配置springmvc.xml文件
         3.5.编写Controller类
         3.6.编写视图页面
         3.7.添加jsp视图解析器
         3.8.配置AnnotationConfigServletWebServerApplicationContext
         3.9.配置WebMvcConfigurerAdapter
         3.10.配置静态资源访问
         3.11.添加过滤器
         3.12.文件上传
         3.13.异常处理
         3.14.国际化
         3.15.表单验证
         3.16.Cookie值绑定
         3.17.重定向与转发
         3.18.JSON输出
         3.19.AJAX异步请求
          
         


         # 4.具体代码实例和解释说明
         4.Spring MVC配置
         4.1.配置springmvc.xml文件
          <beans xmlns="http://www.springframework.org/schema/beans"
               xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">

               <!-- 配置整体处理器映射器 -->
               <bean class="org.springframework.web.servlet.handler.SimpleUrlHandlerMapping">
                   <property name="mappings">
                       <props>
                           <prop key="/hello">helloController</prop>
                           <prop key="/users/{id}/**">userController</prop>
                       </props>
                   </property>
               </bean>
               
               <!-- 配置视图解析器 -->
               <bean class="org.springframework.web.servlet.view.InternalResourceViewResolver">
                   <property name="prefix" value="/WEB-INF/views/"/>
                   <property name="suffix" value=".jsp"/>
               </bean>

           </beans>
          
          4.2.配置AnnotationConfigServletWebServerApplicationContext
          @Configuration
          public class ApplicationContextConfig implements WebMvcConfigurer {
              //...
          }
          
          @ComponentScan("cn.itcast.controller")
          public class RootConfig {}
          
          @ComponentScan("cn.itcast.service")
          @ComponentScan("cn.itcast.dao")
          @PropertySource({"classpath:application.properties"})
          public class AppConfig extends WebMvcConfigurerAdapter {
              //...
          }
          
          @EnableAutoConfiguration
          @SpringBootApplication
          public class DemoApplication {
              //...
          }
          
          // application.properties
          server.port=8080
          
          // web.xml
          <context-param>
              <param-name>contextClass</param-name>
              <param-value>
                  org.springframework.boot.web.servlet.ServletContextInitializer
                    adapterClassName = "cn.itcast.config.ApplicationContextConfig"
              </param-value>
          </context-param>
          
          <listener>
              <listener-class>
                  org.springframework.boot.web.servlet.context.WebServerStartListener
              </listener-class>
          </listener>
          
          <servlet>
              <servlet-name>dispatcherServlet</servlet-name>
              <servlet-class>
                  org.springframework.web.servlet.DispatcherServlet
              </servlet-class>
              <init-param>
                  <param-name>contextConfigLocation</param-name>
                  <param-value>/WEB-INF/spring/springmvc.xml</param-value>
              </init-param>
              <load-on-startup>1</load-on-startup>
          </servlet>
          
          <servlet-mapping>
              <servlet-name>dispatcherServlet</servlet-name>
              <url-pattern>/</url-pattern>
          </servlet-mapping>
          
          
          # 5.未来发展趋势与挑战
          # 6.附录常见问题与解答