
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　API网关是一个API统一的入口，作为各个业务系统和服务之间的代理服务，承接并保护访问应用或服务的所有外部请求，它可以处理服务发现、权限控制、流量控制、熔断降级等一系列功能。本文将从API网关的原理和工作流程出发，结合实际案例和开源组件，详细阐述如何实现一个高可用的API网关。
         　　
         # 2.微服务架构及API网关的原理
         　　微服务架构(Microservices Architecture)是一种分布式计算模型，其定义是通过一组松耦合的服务的方式来构建一个复杂的软件系统。每个服务运行在独立的进程中，并通过轻量级通讯机制进行通信。这种架构模式的主要好处是每个服务都可以独立部署、扩展，能够更好地适应业务的发展。同时，微服务架构也带来了许多新的挑战，如服务间的通信、数据一致性、服务的可用性、服务监控等。因此，API网关就是微服务架构的一个重要组成部分，用来集成各种服务，提供统一的服务接口，屏蔽内部服务的差异性。
          
         　　API网关的原理比较简单，即用统一的网关接收客户端的请求，再根据服务注册中心或配置中心的数据路由到不同的后端服务节点，完成对客户端请求的转发和处理。它主要包括如下功能模块：

           - 服务发现（Service Discovery）：即网关根据配置文件或其他方式获取服务信息，动态地把请求转发给对应的服务节点；
           
           - 请求路由（Request Routing）：路由规则的制定，决定哪些请求由网关负责处理，哪些请求直接发送到后台服务；
           
           - 鉴权与授权（Authentication & Authorization）：对用户身份进行验证，确保只有合法的用户才能访问服务；
           
           - 流量控制（Traffic Control）：控制访问频率和速率，防止恶意攻击或占用过多资源导致服务雪崩；
           
           - 熔断降级（Circuit Breaker）：自动识别服务异常并进入熔断状态，保障服务稳定性；
           
           - 服务容错（Resiliency）：采用集群容错方案，减少单点故障的影响；
           
           - API聚合（API Aggregation）：将多个服务的API整合为一个，降低客户端的开发难度；
           
           - 日志审计（Logging and Auditing）：记录所有的请求和响应信息，用于分析和跟踪问题。
           
           下图展示了一个典型的微服务架构下API网关的工作流程：
           
           
         
         # 3.API网关的功能模块介绍
         　　上面提到的API网关主要有五大功能模块，下面介绍它们的详细功能。
         
         1.服务发现（Service Discovery）：
         API网关必须要具有良好的服务发现能力，否则无法正常工作。微服务架构下，服务通常会部署在很多地方，为了保证服务的可用性和分发均衡，需要一个服务注册中心，而API网关需要使用这个服务注册中心获取服务列表，然后向这些服务节点进行请求。
         通过服务发现，API网关可以获得所有服务节点的地址信息，并且可以实时更新地址信息。当某个服务节点发生故障切换或者新增时，API网关立刻得到通知，然后重新路由请求到其他节点。
         
         2.请求路由（Request Routing）：
         根据路由规则，API网关根据客户端的请求选择对应的后端服务节点进行处理。路由规则包括基于路径、域名、URI、参数、IP等等。当客户端发起请求时，API网关根据请求的特征匹配相应的路由规则，并将请求转发到相应的服务节点。如果路由不到指定的服务节点，则返回错误信息。
         请求路由可以解决请求调度的瓶颈问题，提升API网关的吞吐量和性能。例如，对于新闻客户端来说，可能只需要访问新闻服务，而不必担心社交、搜索等相关服务的压力。
         
         3.鉴权与授权（Authentication & Authorization）：
         在微服务架构下，服务之间一般采用Token认证和授权机制，而API网关也要具有相同的特性。通过解析Token获取用户的身份信息，并判断是否有访问指定服务的权限。如果没有权限，则拒绝该请求。鉴权与授权可以有效避免未授权的用户访问敏感数据，保护数据安全。
         
         4.流量控制（Traffic Control）：
         当API网关接收到大量的请求时，可能会成为整个系统的性能瓶颈。因此，需要通过流量控制限制用户的访问速度。流量控制可以设定每秒最大访问次数，或者平均每分钟的访问次数。这样可以防止短时间内过于密集的访问导致服务器资源被消耗殆尽。另外，也可以设置超时时间，使得请求失败或响应超时。
         
         5.熔断降级（Circuit Breaker）：
         熔断降级是指当检测到某项服务出现异常时，立刻切断该服务的调用，并返回默认值或自定义提示信息，从而使得调用方知道当前服务不可用，并快速失败而不是长时间等待。这是为了避免因为某些慢甚至阻塞的服务导致整个系统瘫痪。当服务恢复时，可以通过流量控制策略快速恢复。
         
         6.服务容错（Resiliency）：
         微服务架构下，服务节点之间存在相互依赖关系，有时会出现单点故障或网络波动引起的问题。服务容错机制需要能够自动检测故障，并通过负载均衡和自动重试来保证服务的可用性。
         
         7.API聚合（API Aggregation）：
         在微服务架构下，不同服务暴露自己的API，客户端需要知道所有服务的接口才能调用。API聚合可以将多个服务的API合并为一个，简化客户端开发难度，提升效率。同时，也方便统一管理和监控。
         
         8.日志审计（Logging and Auditing）：
         有些公司为了方便管理和维护系统，会将所有的请求和响应日志记录下来。API网关也需要具备类似的功能，这样就可以记录每个请求的细节，以便后期进行问题排查。
         除了这些功能外，还可以通过上面的功能模块组合起来实现一个完整的API网关。
          
         
         # 4.项目背景介绍
         为了实践微服务架构下的API网关，我们需要一个简单的项目作为测试平台，需求如下：
         
         1.提供一个网页，用于注册用户。包括用户名、密码、邮箱等信息。
         
         2.提供一个网页，用户登录。输入用户名和密码即可登录成功。
         
         3.提供一个查询用户信息的RESTful API。该API只能允许已登录用户访问。
         
         4.提供一个提交订单的RESTful API。该API只能允许已登录用户访问。
         
         5.要求网页和API支持跨域访问。
          
         # 5.项目实现过程
         1.创建maven项目，引入以下依赖：

       ```xml
       <dependency>
           <groupId>org.springframework.boot</groupId>
           <artifactId>spring-boot-starter-web</artifactId>
       </dependency>
       
       <dependency>
           <groupId>org.springframework.cloud</groupId>
           <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
       </dependency>
       
       <dependency>
           <groupId>org.springframework.security</groupId>
           <artifactId>spring-security-web</artifactId>
       </dependency>
       
       <!-- 使用redis做session存储 -->
       <dependency>
           <groupId>org.springframework.boot</groupId>
           <artifactId>spring-boot-starter-data-redis</artifactId>
       </dependency>
       <!-- 配置redis连接 -->
       <dependency>
           <groupId>org.springframework.session</groupId>
           <artifactId>spring-session-data-redis</artifactId>
       </dependency>
       <!-- redis配置 -->
       <dependency>
           <groupId>redis.clients</groupId>
           <artifactId>jedis</artifactId>
       </dependency>
       ```
       
     2.编写配置文件application.properties，添加如下内容：

```yaml
server:
  port: ${port:8080}
  
# Eureka注册中心配置
eureka:
  client:
    serviceUrl:
      defaultZone: http://${hostName}:8761/eureka/
    
# 配置数据库链接
spring:
  datasource:
    driver-class-name: com.mysql.jdbc.Driver
    url: jdbc:mysql://localhost:3306/api_gateway?useUnicode=true&characterEncoding=utf-8&useSSL=false
    username: root
    password: 
  jpa:
    hibernate:
      ddl-auto: update   # 自动生成表结构
```

     3.编写实体类User，用于存储用户信息。

```java
import javax.persistence.*;

@Entity
public class User {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    private String userName;

    private String passWord;

    private String email;
    
    //省略getters和setters方法...

}
```

     4.编写Controller层，分别编写用户注册和登录Controller。

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@Controller
public class UserController {
    
    @Autowired
    private BCryptPasswordEncoder encoder;
    
    /**
     * 用户注册页面
     */
    @RequestMapping("/register")
    public String register() {
        return "user/register";
    }
    
    /**
     * 用户注册逻辑
     */
    @PostMapping("/register")
    public String doRegister(@RequestParam("userName") String userName,
                             @RequestParam("password") String password,
                             @RequestParam("email") String email, Model model) throws Exception {
        
        if (checkUserNameExists(userName)) {
            model.addAttribute("errorMessage", "用户名已存在");
            return "user/register";
        }
        
        Map<String, Object> map = new HashMap<>();
        try {
            
            // 对密码进行加密
            password = encoder.encode(password);
            
            // 保存用户信息到数据库
            User user = new User();
            user.setUserName(userName);
            user.setPassWord(password);
            user.setEmail(email);
            userService.save(user);
            
        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }

        map.put("successMessage", "注册成功！欢迎您：" + userName);
        return "redirect:/login";
        
    }
    
    /**
     * 判断用户名是否已经存在
     */
    private boolean checkUserNameExists(String userName) {
        User user = userService.findByUserName(userName);
        if (user!= null) {
            return true;
        } else {
            return false;
        }
    }
    
    /**
     * 用户登录页面
     */
    @RequestMapping("/login")
    public String loginPage() {
        return "user/login";
    }
    
    /**
     * 用户登录逻辑
     */
    @PostMapping("/login")
    public String doLogin(@RequestParam("userName") String userName,
                          @RequestParam("password") String password,
                          Model model) throws Exception {
        Map<String, Object> map = new HashMap<>();
        try {
            
            // 查询用户信息
            User user = userService.findByUserNameAndPassWord(userName, password);

            // 生成Session
            request.getSession().setAttribute("userId", user.getId());
            request.getSession().setMaxInactiveInterval(60*60*24);    // Session两小时后失效
            
        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }
        map.put("successMessage", "登录成功！欢迎您：" + userName);
        return "redirect:/";
    }
}
```

     5.编写UserService，用于操作用户信息。

```java
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface UserService extends JpaRepository<User, Long> {
    
    /**
     * 根据用户名查找用户信息
     */
    User findByUserName(String userName);
    
    /**
     * 根据用户名和密码查找用户信息
     */
    User findByUserNameAndPassWord(String userName, String passWord);
    
    List<User> findAllByOrderByUserNameAsc();
}
```

     6.编写OrderService，用于存放订单信息。

```java
import org.springframework.stereotype.Service;

@Service
public class OrderService {
    
    //... 此处省略Order相关的代码...

}
```

     7.编写API接口定义文件，分别定义查询用户信息和提交订单的API。

```java
package cn.com.vipkid.api.controller;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

@RestController
@Api(tags="用户接口")
public class UserController {
    
    @Autowired
    private OrderService orderService;
    
    /**
     * 获取所有用户列表
     * @return
     */
    @GetMapping("/users")
    @PreAuthorize("hasAuthority('ROLE_ADMIN') or hasRole('USER')")     // 指定角色访问此接口
    @ApiOperation(value="获取所有用户列表")
    public List<User> getAllUsers() {
        List<User> users = userService.findAllByOrderByUserNameAsc();
        return users;
    }
    
    /**
     * 获取登录用户信息
     */
    @GetMapping("/user/{id}")
    @PreAuthorize("hasAuthority('ROLE_ADMIN') or hasRole('USER')")      // 指定角色访问此接口
    @ApiOperation(value="获取登录用户信息")
    public User getUserInfoById(@PathVariable("id") long userId) {
        User user = userService.findById(userId).orElseThrow(() -> new IllegalArgumentException("用户不存在"));
        return user;
    }
    
    /**
     * 提交订单
     */
    @PostMapping("/order")
    @PreAuthorize("hasAuthority('ROLE_ADMIN') or hasRole('USER')")        // 指定角色访问此接口
    @ApiOperation(value="提交订单")
    public void submitOrder(@RequestBody Map<String, Object> params) {
        Date date = new Date();
        long orderId = System.currentTimeMillis();
        for (Object obj : params.values()) {
            if (!(obj instanceof ArrayList)) {
                continue;
            }
            List list = (List) obj;
            orderService.createOrder(orderId, (int) list.get(0), (String) list.get(1),
                    (double) list.get(2), (long) list.get(3), date);
        }
    }
    
    /**
     * 更新用户信息
     */
    @PutMapping("/user/{id}")
    @PreAuthorize("hasAuthority('ROLE_ADMIN')")           // 指定角色访问此接口
    @ApiOperation(value="更新用户信息")
    public User updateUserInfo(@PathVariable("id") long userId,
                               @RequestBody User user) {
        return userService.save(user);
    }
    
}
```

     8.在resources目录下创建templates文件夹，编写HTML模板文件。

```html
<!DOCTYPE html>
<html lang="en" xmlns:th="http://www.thymeleaf.org">
<head>
    <meta charset="UTF-8">
    <title>用户注册</title>
</head>
<body>
    <div th:if="${param.errorMessage}">
        <span style="color:red;" th:text="${param.errorMessage}"></span>
    </div>
    <form action="/register" method="post">
        <label for="username">用户名：</label><input type="text" name="userName"><br/>
        <label for="password">密码：</label><input type="password" name="password"><br/>
        <label for="email">邮箱：</label><input type="text" name="email"><br/>
        <button type="submit">提交</button>
    </form>
</body>
</html>


<!DOCTYPE html>
<html lang="en" xmlns:th="http://www.thymeleaf.org">
<head>
    <meta charset="UTF-8">
    <title>用户登录</title>
</head>
<body>
    <div th:if="${param.errorMessage}">
        <span style="color:red;" th:text="${param.errorMessage}"></span>
    </div>
    <form action="/login" method="post">
        <label for="username">用户名：</label><input type="text" name="userName"><br/>
        <label for="password">密码：</label><input type="password" name="password"><br/>
        <button type="submit">登录</button>
    </form>
</body>
</html>
```

     9.启动项目，访问注册页面http://localhost:8080/register ，输入用户名、密码、邮箱等信息，点击“提交”按钮进行注册，将跳转到登录页面http://localhost:8080/login ，输入用户名和密码，点击“登录”按钮进行登录，登陆成功。
     10.查看控制台输出日志，确认注册成功、登录成功等日志。
     11.访问接口文档页面http://localhost:8080/swagger-ui.html ，查看API接口定义。
     12.尝试修改用户信息、提交订单、删除用户等操作，确认API接口的正确性。
     
     13.为了改善API网关的性能，可以使用一些开源组件来优化API网关。
     
     14.首先安装Redis，Redis是一款开源的高性能键值存储，可以用作API网关的Session缓存和限流组件。
     
     15.pom.xml增加redis依赖：

```xml
<!-- 使用redis做session存储 -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>
<!-- 配置redis连接 -->
<dependency>
    <groupId>org.springframework.session</groupId>
    <artifactId>spring-session-data-redis</artifactId>
</dependency>
<!-- redis配置 -->
<dependency>
    <groupId>redis.clients</groupId>
    <artifactId>jedis</artifactId>
</dependency>
```

     16.编写application.yml，配置Redis连接信息：

```yaml
spring:
  session:
    store-type: redis

  redis:
    host: localhost
    port: 6379
    timeout: 10000ms
    lettuce:
      pool:
        max-active: 10
        max-idle: 10
        min-idle: 5
        max-wait: 10000ms
    database: 0
  
  datasource:
    driver-class-name: com.mysql.jdbc.Driver
    url: jdbc:mysql://localhost:3306/api_gateway?useUnicode=true&characterEncoding=utf-8&useSSL=false
    username: root
    password: 
```

     17.启动项目，访问接口文档页面，刷新页面会看到Redis连接成功的日志。
     18.为了实现限流功能，可以使用开源组件Sentinel。
     
     19.Sentinel是阿里巴巴开源的Java中间件产品，旨在解决微服务架构中的流量控制、熔断降级、服务隔离等问题，并提供了丰富的实施指南，包括最佳实践和经验。
     
     20.首先下载Sentinel的最新版本，然后安装Redis插件：

https://github.com/alibaba/Sentinel/releases

http://redisson.io/download.html

     21.然后在pom.xml中添加以下依赖：

```xml
<!-- sentinel -->
<dependency>
    <groupId>com.alibaba.csp</groupId>
    <artifactId>sentinel-core</artifactId>
    <version>${sentinel.version}</version>
</dependency>
<dependency>
    <groupId>com.alibaba.csp</groupId>
    <artifactId>sentinel-transport-simple-http</artifactId>
    <version>${sentinel.version}</version>
</dependency>
<dependency>
    <groupId>com.alibaba.csp</groupId>
    <artifactId>sentinel-datasource-redis</artifactId>
    <version>${sentinel.version}</version>
</dependency>
<dependency>
    <groupId>org.apache.commons</groupId>
    <artifactId>commons-pool2</artifactId>
    <version>${common-pool2.version}</version>
</dependency>
```

     22.在application.yml中添加Sentinel配置：

```yaml
## Sentinel ##
spring:
  cloud:
    sentinel:
      transport:
        dashboard: 127.0.0.1:8080       # Sentinel Dashboard地址（默认端口8080）
        port: 8719                        # Sentinel监听的端口号（默认为8719）
      filter:
        enabled: true                     # 是否开启Filter（默认为true）
        url-patterns: /api/*              # Filter监控的URL pattern（默认无，即所有请求都监控）
        exclude-urls: /*/*.js,/*/*.css,/actuator/**   # Filter排除的URL pattern （默认空，即不排除任何请求）
        fill-terror-code: true            # 是否填充BlockException的错误码（默认为true）
      datasource:
        ds:
          nacos:                          # Nacos作为Sentinel的DataSource
            server-addr: 127.0.0.1:8848   # Nacos Server地址
            data-id: spring-cloud-demo
            group-id: DEFAULT_GROUP
            rule-type: flow                 # 流控规则类型（flow 或 degrade）
```

     23.在配置文件中配置Redis信息：

```yaml
spring:
  cloud:
    sentinel:
      datasource:
        ds:
          nacos:
            namespace: csp                             # 命名空间
            cluster-name: default                       # Redis集群名称
            redis-type: standalone                      # Redis类型（standalone或cluster）
            connection-timeout-in-milliseconds: 10000   # 连接超时时间
            idle-timeout-in-seconds: 30                  # 空闲连接超时时间
            command-timeout-in-milliseconds: 10000       # 命令超时时间
            max-total: 10                               # Redis连接池最大连接数（默认值为8）
            max-idle: 5                                 # Redis连接池最大空闲连接数
            min-idle: 1                                  # Redis连接池最小空闲连接数
            max-wait-millis: 10000                       # 从连接池中获取连接的最大等待时间
            nodes:                                      
              - 127.0.0.1:6379                          # Redis节点地址（可以是主节点或从节点）
              - 127.0.0.1:6380
```

     24.在Controller类中，加入@SentinelResource注解，用于标识访问限流资源。

```java
package cn.com.vipkid.api.controller;

import com.alibaba.csp.sentinel.annotation.SentinelResource;
import com.alibaba.csp.sentinel.slots.block.BlockException;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.concurrent.TimeUnit;

@RestController
@Api(tags="用户接口")
public class UserController {

    private static final Logger LOGGER = LoggerFactory.getLogger(UserController.class);

    @Autowired
    private OrderService orderService;

    /**
     * 模拟耗时操作
     */
    @GetMapping("/time")
    @SentinelResource(value = "time", blockHandlerClass = CommonFallback.class, fallback = "${sentinel.fallback}")
    public String time() {
        int count = 5;
        while (count > 0) {
            TimeUnit.SECONDS.sleep(1);
            LOGGER.info("Time is running...");
            count--;
        }
        return "Time is over!";
    }

    /**
     * 获取所有用户列表
     */
    @GetMapping("/users")
    @PreAuthorize("hasAuthority('ROLE_ADMIN') or hasRole('USER')")
    @ApiOperation(value="获取所有用户列表")
    public List<User> getAllUsers() {
        List<User> users = userService.findAllByOrderByUserNameAsc();
        return users;
    }

    /**
     * 获取登录用户信息
     */
    @GetMapping("/user/{id}")
    @PreAuthorize("hasAuthority('ROLE_ADMIN') or hasRole('USER')")
    @ApiOperation(value="获取登录用户信息")
    public User getUserInfoById(@PathVariable("id") long userId) {
        User user = userService.findById(userId).orElseThrow(() -> new IllegalArgumentException("用户不存在"));
        return user;
    }

    /**
     * 提交订单
     */
    @PostMapping("/order")
    @PreAuthorize("hasAuthority('ROLE_ADMIN') or hasRole('USER')")
    @ApiOperation(value="提交订单")
    public void submitOrder(@RequestBody Map<String, Object> params) {
        Date date = new Date();
        long orderId = System.currentTimeMillis();
        for (Object obj : params.values()) {
            if (!(obj instanceof ArrayList)) {
                continue;
            }
            List list = (List) obj;
            orderService.createOrder(orderId, (int) list.get(0), (String) list.get(1),
                    (double) list.get(2), (long) list.get(3), date);
        }
    }

    /**
     * 更新用户信息
     */
    @PutMapping("/user/{id}")
    @PreAuthorize("hasAuthority('ROLE_ADMIN')")
    @ApiOperation(value="更新用户信息")
    public User updateUserInfo(@PathVariable("id") long userId,
                                @RequestBody User user) {
        return userService.save(user);
    }

}
```

     25.编写CommonFallback类，用于处理限流后的回调操作。

```java
package cn.com.vipkid.api.controller;

import com.alibaba.csp.sentinel.slots.block.BlockException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class CommonFallback {

    private static final Logger LOGGER = LoggerFactory.getLogger(CommonFallback.class);

    public String handleException(Long id, BlockException ex) {
        LOGGER.error("触发限流规则:{}",ex.getClass());
        return "触发限流规则，请稍后再试";
    }

}
```

     26.在配置文件中加入Sentinel配置：

```yaml
## Sentinel ##
spring:
  cloud:
    sentinel:
      transport:
        dashboard: 127.0.0.1:8080             # Sentinel Dashboard地址（默认端口8080）
        port: 8719                              # Sentinel监听的端口号（默认为8719）
      filter:
        enabled: true                           # 是否开启Filter（默认为true）
        url-patterns: /api/*                    # Filter监控的URL pattern（默认无，即所有请求都监控）
        exclude-urls: /*/*.js,/*/*.css,/actuator/**   # Filter排除的URL pattern （默认空，即不排除任何请求）
        fill-terror-code: true                  # 是否填充BlockException的错误码（默认为true）
      datasource:
        ds:
          nacos:                                # Nacos作为Sentinel的DataSource
            server-addr: 127.0.0.1:8848          # Nacos Server地址
            data-id: spring-cloud-demo
            group-id: DEFAULT_GROUP
            rule-type: flow                      # 流控规则类型（flow 或 degrade）
            namespace: csp                       # 命名空间
            cluster-name: default                # Redis集群名称
            redis-type: standalone               # Redis类型（standalone或cluster）
            connection-timeout-in-milliseconds: 10000   # 连接超时时间
            idle-timeout-in-seconds: 30                  # 空闲连接超时时间
            command-timeout-in-milliseconds: 10000       # 命令超时时间
            max-total: 10                            # Redis连接池最大连接数（默认值为8）
            max-idle: 5                              # Redis连接池最大空闲连接数
            min-idle: 1                             # Redis连接池最小空闲连接数
            max-wait-millis: 10000                    # 从连接池中获取连接的最大等待时间
            nodes:                                   
              - 127.0.0.1:6379                         # Redis节点地址（可以是主节点或从节点）
              - 127.0.0.1:6380
```

     27.在bootstrap.yml中配置feign相关参数：

```yaml
feign:
  hystrix:
    enabled: true                   # 是否启用 Feign 的 Hystrix 协同 circuit breaker 支持
  compression:
    response:
      enabled: true                 # 开启GZIP压缩传输响应数据
  httpclient:
    enabled: false                  # Feign 默认支持 SpringMvc 的 RestTemplate, 不用再额外配置HttpClient，故设置为false
```

     28.测试限流功能。打开一个新的命令行窗口，执行以下命令：

```shell
watch curl --connect-timeout 5 'http://localhost:8080/time'  
```

此时会触发限流规则，浏览器访问http://localhost:8080/time 不会立即响应，直到超时。当触发限流规则时，将打印日志：

```
[Sentinel Starter] Handle Request blocked by Sentinel: FlowException
```

打开第二个命令行窗口，执行：

```shell
for ((i=0;i<=10;i++));do curl 'http://localhost:8080/time';done
```

此时两个窗口都将触发限流规则，且都会返回触发限流规则的默认响应。可以打开sentinel dashboard查看详细信息：http://localhost:8080/#/dashboard/product-blocklist。