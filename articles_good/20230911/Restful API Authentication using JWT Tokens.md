
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是JWT？
JSON Web Token（JWT）是一个开放标准（RFC7519），它定义了一种紧凑且自包含的方式，用于在两个通信应用程序之间安全地传输信息。该信息可以被验证并且信任，因为JWT被签名。JWTs可以使用秘钥签名或密钥加密生成。

JWT由三部分组成：头部、载荷和签名。头部通常包含两种元素：声明和对称加密算法。载荷包含了有效载荷数据。最后，签名是通过秘钥签名或者加密得到的。

## 为什么要使用JWT？
使用JWT进行API认证可以提供以下优点：

1.无状态性：JWT没有服务器数据库依赖，在服务端不需要管理会话，实现了无状态化；
2.跨域请求：JWT可以在不同域名、不同端口，甚至不同协议的服务之间传递，并保持会话；
3.节省带宽：JWT可以通过HTTP头信息传递，并对请求和响应数据进行压缩，使得体积更小，减少网络流量消耗；
4.易于理解：JWT格式的文本较短，易于理解；
5.方便实施：JWT可以直接用于客户端验证，不需要像传统方案那样需要再次请求服务器验证；
6.多终端支持：JWT可用于各种平台、语言，包括移动端APP、网页应用、桌面应用、微服务等。

## RESTful API的认证方式
RESTful API的认证方式主要分为两类：基于cookie的认证和基于token的认证。

### 基于cookie的认证
基于cookie的认证最简单也最容易实现的方法，只需在服务端设置一个session cookie即可，然后客户端每次向服务器发送请求都会带上这个cookie。服务端根据cookie中的信息判断用户身份是否合法。这种方法的问题主要是：

1.用户登出后cookie失效，无法保证用户隐私的保护；
2.服务端需要持久化存储session，增加了复杂性；
3.所有用户共享一个cookie，存在安全风险。

### 基于token的认证
基于token的认证又叫“令牌”（Token）。它不依靠cookie来做认证，而是在每次请求时都传递一个token给服务器，服务器通过token识别用户身份。一般来说，token有以下特点：

1.无状态：服务器不保存任何关于用户的信息；
2.易于扩展：只需要修改服务器的密钥，就可以更新、续约token；
3.易于保管：Token 可以存储在浏览器中、Cookie 中、localStorage 或 sessionStorage 中，或者直接返回在响应中；
4.可移植性：由于采用了标准协议，因此 token 可在各种环境中通用。

# 2.JWT工作流程及原理
## JWT工作流程

1.客户端向服务器申请登录：客户端向服务器发送用户名密码，服务器验证成功后颁发JWT token。
2.客户端携带JWT token向服务器请求资源：客户端携带JWT token请求需要授权的接口，服务端解析JWT token获取用户信息。
3.客户端定时刷新JWT token：JWT token过期时间默认为1天，客户端需要定时刷新token，否则会出现用户退出不掉线的情况。

## JWT数据结构

JWT数据结构如上图所示，由header，payload，signature三个部分组成。其中header和payload都是json格式的数据，signature则是对header和payload加密后的结果。header包含了jwt的类型、加密算法以及token过期时间。payload存放实际需要传输的用户信息，比如用户id，用户名，过期时间等。当用户需要访问受保护的资源时，客户端携带上JWT，然后通过Authorization请求头发送给服务器。服务器通过检验JWT的合法性，如果合法，就向用户提供请求的资源，否则拒绝请求。

## JWT加密过程
客户端向服务器发送请求，服务器接收到请求后，首先生成一个随机字符串作为密钥，然后使用该密钥对header和payload进行加密。加密后的结果就是signature。

然后将header和payload，以及signature一起放在一起，成为JWT。

当客户端需要请求受保护的资源时，先从本地的storage（比如localStorage）获取当前已有的token，然后将其携带到下一次的请求头Authorization中。

服务器收到请求后，首先检查JWT是否有效，如果有效，就允许用户访问对应的资源，否则就拒绝访问。

# 3.JWT实现方案
## Spring Security + JWT 的实现
Spring Security 提供了一个名为`spring-security-oauth2-autoconfigure`，它能够帮助开发者快速配置基于OAuth2协议的资源服务器。该模块能够完成一些基础功能，比如校验token，生成token等。因此，我们只需要导入该模块，然后通过几个注解来开启jwt相关的配置即可。

```xml
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-security</artifactId>
    </dependency>

    <!-- JWT -->
    <dependency>
        <groupId>io.jsonwebtoken</groupId>
        <artifactId>jjwt</artifactId>
        <version>${jjwt.version}</version>
    </dependency>

    <dependency>
        <groupId>org.springframework.security</groupId>
        <artifactId>spring-security-config</artifactId>
    </dependency>

    <dependency>
        <groupId>org.springframework.security</groupId>
        <artifactId>spring-security-web</artifactId>
    </dependency>

    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-oauth2-resource-server</artifactId>
    </dependency>
```

### 配置文件
新建一个配置文件 `application.yml`，配置如下：

```yaml
server:
  port: 8080
  
# jwt
jwt:
  header: Authorization
  # expire time in seconds
  expiration: 604800
  secret: mysecret
  base64-secret: false
  # pub key location when base64-secret is true and public-key is null
  # if not specified, will try to resolve the path of 'classpath:/public_key.txt' by default
  public-key: classpath:my-rsa-public.pem
  issuer: authserver
    
logging:
  level:
    root: INFO
    org.springframework.security: DEBUG

# security config
spring:
  security:
    oauth2:
      resourceserver:
        jwt:
          jwk-set-uri: http://localhost:8080/.well-known/jwks.json
          
management:
  endpoints:
    web:
      exposure:
        include: "*"
```

这里我们使用 `jjwt` 来生成和解析 JWT token。配置项的意义如下：

- `header`: 指定 HTTP 请求 Header 中的字段名称，用来存放 JWT。默认值为 "Authorization"。例如：`Authorization: Bearer <TOKEN>`。
- `expiration`: 设置 JWT 超时时间（单位秒），默认值是 7 天 (`604800`)。
- `secret`: 用作 JWT 签名的密钥。建议不要设置成明文，而应该设置为一个足够复杂的随机字符串。
- `base64-secret`: 是否将 `secret` 选项编码为 Base64。如果设置为 `true`，则 `secret` 将被视为 base64 编码后的公钥或私钥。如果设置为 `false`，则 `secret` 将被直接用作 HMAC 哈希函数的密钥。
- `public-key`: 如果 `base64-secret` 是 `true`，并且没有设置该选项，则尝试从指定的文件加载 RSA 公钥。默认为 `null`。
- `issuer`: JWT 发行者（iss）标识，用来标识发行此 JWT 的系统。可以随便填写，但通常情况下应该设置成唯一的值。

### 控制器
我们编写一个测试控制器来演示如何使用 JWT 生成和校验 token。

```java
import io.jsonwebtoken.*;
import org.springframework.web.bind.annotation.*;

@RestController
public class TestController {
    
    @GetMapping("/login")
    public String login() throws UnsupportedEncodingException {
        String payload = "{\"username\":\"test\",\"authorities\":[\"USER\"]}";
        byte[] secretBytes = "mysecret".getBytes("UTF-8");
        
        // generate jwt
        String compactJws = Jwts.builder().setHeaderParam("typ", "JWT").setSubject("test")
               .claim("roles", "admin").claim("authorities", "USER")
               .signWith(SignatureAlgorithm.HS512, secretBytes).compact();

        return compactJws;
    }
    
    @PostMapping("/getuser")
    public Object getUser(@RequestHeader("Authorization") String authorization) throws Exception {
        // parse jwt
        Claims claims = Jwts.parser().setSigningKey("mysecret").parseClaimsJws(authorization.replace("Bearer ", ""))
               .getBody();
        
        System.out.println("subject:" + claims.getSubject());
        System.out.println("roles:" + claims.get("roles"));
        System.out.println("authorities:" + claims.get("authorities"));
        
        return claims;
    }
}
```

`/login` 方法用于生成 JWT token。我们将登录用户信息编码为 JWT 数据，并签名为 HMAC SHA-512 签名。该方法返回生成的 JWT token。

`/getUser` 方法用于校验 JWT token，并获取登录用户信息。我们使用 `Jwts.parser()` 解析 JWT，设置签名密钥为 `"mysecret"`，然后调用 `.parseClaimsJws()` 方法来获取包含登录用户信息的 Claims 对象。该对象提供了很多方法来读取各个 Claim 值。

### 测试
启动项目，在浏览器里打开 `/swagger-ui.html?configUrl=/v3/api-docs/swagger-config#/test-controller`，然后点击 authorize 按钮，填入如下信息：

- **Type**：`bearer`
- **Token**：`eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlVzZXIiLCJhdXRob3JpdGllcyI6WyJDTEFORUQiXX0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c`

点击Authorize按钮，如果授权成功的话，页面上会显示 JWT token 信息。

我们还可以通过 curl 命令测试，如下：

```bash
curl -H "Content-Type: application/json" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlVzZXIiLCJhdXRob3JpdGllcyI6WyJDTEFORUQiXX0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"\
     -X POST \
     http://localhost:8080/getuser | json_pp
{
   "sub" : "1234567890",
   "authorities" : [
      "ROLE_USER",
      "SCOPE_read",
      "SCOPE_write"
   ],
   "jti" : "a7ebdbb7-e6d4-4fc7-bf6b-ffba5c0d4f2a",
   "exp" : 1546300800,
   "roles" : [
      "admin"
   ]
}
```