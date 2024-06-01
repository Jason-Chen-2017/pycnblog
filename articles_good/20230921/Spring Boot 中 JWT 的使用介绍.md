
作者：禅与计算机程序设计艺术                    

# 1.简介
  

JWT（JSON Web Token）是一个基于 JSON 对象传输的、用于身份认证和信息交换的一种令牌规范。它允许在需要的时候通过共享密钥进行加密签名验证。JSON Web Tokens 有三种主要类型：
1. 短期 JWT - 有效期较短，一般几分钟到几小时。
2. 普通 JWT - 有效期一般较长，甚至几个月都可能有效。
3. 长期 JWT - 在某些情况下，用户可能会使用超过一年的时间。

JWT 可以在单点登录（Single Sign On）、API 授权（Authorization）、信息交换等方面提供巨大的便利。Spring Security 对 JWT 提供了支持，本文将介绍 Spring Boot 中的 JWT 使用方法。
# 2.核心概念
## 2.1 JSON Web Token (JWT)
JSON Web Token 是一种声明性的安全令牌，由三部分组成：头部（Header），载荷（Payload），签名（Signature）。为了防止篡改，签名计算值需要加入密钥。头部和载荷都是经过 Base64Url 编码之后的字符串，无可非议。载荷可以自定义添加一些必要的信息，如用户 ID、用户名、有效期等。当载荷中的信息发生变化时，需要重新生成新的签名值才能生效。如下图所示：


通常，我们会把 JWT 放在 HTTP 请求头或者 URL 参数中，例如：Authorization: Bearer xxx.xxxxx.xxx。

### 2.1.1 Header

头部承载两类信息：
- 声明类型（typ）：此处的值固定为 "JWT"；
- 加密方式（alg）：加密使用的算法，目前最多的就是 HMAC SHA256。

```json
{
  "typ": "JWT",
  "alg": "HS256"
}
```

### 2.1.2 Payload

载荷承载具体的业务数据，比如用户标识符、过期时间、权限列表等。载荷也要用 Base64Url 编码。载荷中也可以放一些系统保留字段，例如 iss（issuer）表示发行人、sub（subject）表示主题、aud（audience）表示接收方。

```json
{
  "iss": "authserver.com",
  "exp": 1440000000,
  "usr_id": "123456",
  "role": [
    {
      "id": "admin",
      "name": "管理员"
    },
    {
      "id": "user",
      "name": "普通用户"
    }
  ]
}
```

### 2.1.3 Signature

签名对头部、载荷进行加密，并且加入了密钥。签名可以保证数据的完整性、发送者身份认证。签名由两个部分组成：
1. 需要加密的数据串（header + payload），不包括签名；
2. 密钥。

对这两个串进行加密得到签名值。如果传输过程中被篡改，接收方可以通过密钥验证签名是否正确，但无法直接知道是哪里出错。

## 2.2 OAuth2.0

OAuth2.0 是一种认证协议，它的设计目标是让不同的应用之间安全地共享资源。OAuth2.0 通过四个角色分别代表资源所有者、客户端、授权服务器、资源服务器。OAuth2.0 描述了一套完整的授权机制，实现不同客户端之间的相互认证和授权。流程如下图所示：


1. 用户访问客户端，客户端要求访问自己的资源，并向用户显示授权页面，用户同意后，浏览器会自动跳转到授权服务器进行授权确认，然后得到一个授权码；
2. 客户端再请求资源服务器的 access token，并附带上刚才获取的授权码，资源服务器验证授权码，如果验证通过，就返回 access token 给客户端；
3. 客户端可以使用 access token 访问受保护的资源，资源服务器验证 access token，如果验证通过，就返回受保护资源给客户端。

OAuth2.0 既适用于 API 接口授权，也适用于 Web 应用授权，其中 Resource Owner 为用户，Client 为访问资源的应用，Resource Server 为服务端提供资源的地方，Authorization Server 为第三方认证服务器。

## 2.3 JSON Web Key Sets (JWKS)

JSON Web Key Sets 简称 JWKS，是一个用来存放公开密钥的标准文件。它其实是由一组 JSON 数据构成，里面包含了一系列用来验证 JWT 签名的公钥。实际上，当我们用一个 JWT 来签名，那么我们只需要把它附带的签名值和对应的公钥做一个映射关系即可，因为公钥已经在 JWKS 文件中声明过了。如下图所示：


当我们想拿到某个 JWT 的公钥时，首先去找这个公钥在 JWKS 文件中的索引位置，然后从相应的位置找到这个公钥，就可以直接用来验证签名。

# 3. Spring Boot 中 JWT 的使用方法

## 3.1 添加依赖

我们需要在项目中引入 jwt-starter 和 spring-security-oauth2-starter 依赖。如下所示：

```xml
<dependency>
   <groupId>io.jsonwebtoken</groupId>
   <artifactId>jjwt-api</artifactId>
   <version>0.11.2</version>
</dependency>
<dependency>
   <groupId>io.jsonwebtoken</groupId>
   <artifactId>jjwt-impl</artifactId>
   <version>0.11.2</version>
   <scope>runtime</scope>
</dependency>
<!-- https://mvnrepository.com/artifact/org.springframework.boot/spring-boot-starter-security -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>
<!-- https://mvnrepository.com/artifact/org.springframework.boot/spring-boot-starter-oauth2-client -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-oauth2-client</artifactId>
</dependency>
```

## 3.2 配置

Spring Boot 中 JWT 使用配置如下：

```yaml
# application.yml
spring:
  security:
    oauth2:
      resourceserver:
        jwt:
          jwk-set-uri: http://localhost:8080/.well-known/jwks.json # 设置公钥地址
```

上面的例子中，我们设置了公钥地址，该地址应该返回公钥集合的 JSON 数据，这里我们假设该地址为 `http://localhost:8080/.well-known/jwks.json`。

## 3.3 生成 JWT

我们先来创建一个 Controller，用于生成 JWT。首先，导入以下包：

```java
import io.jsonwebtoken.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
```

然后，定义一个路径 `/generateToken`，用于生成 JWT。这里我只是简单地生成了一个空白的 JWT，你可以根据需要自己添加信息：

```java
@Controller
public class JwtController {

    private static final Logger logger = LoggerFactory.getLogger(JwtController.class);
    
    @Value("${jwt.secret}") // 从配置文件中读取 JWT 的密钥
    private String secretKey;

    /**
     * 获取当前用户信息
     */
    public UserDetails getCurrentUser() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        return (UserDetails) authentication.getPrincipal();
    }

    /**
     * 生成 JWT
     * @return JWT 字符串
     */
    @RequestMapping("/generateToken")
    @ResponseBody
    public String generateToken(@RequestParam("username") String username, Model model) throws UnsupportedEncodingException {

        // 生成 JWT
        long nowMillis = System.currentTimeMillis();
        Date now = new Date(nowMillis);
        
        // 如果想自定义 JWT 的 claims，可以参考官方文档
        Claims claims = Jwts.claims().setSubject(username).setIssuedAt(now)
               .setExpiration(new Date(nowMillis + 1000*60)); // 1 分钟有效期
                
        String jwt = Jwts.builder().setHeaderParam("typ", "JWT").setHeaderParam("alg", "HS256")
                       .addClaims(claims)
                       .signWith(SignatureAlgorithm.HS256, secretKey)
                       .compact();
        
		// 返回 JWT 字符串
        return jwt;
        
    }
    
}
```

以上代码首先读取了 JWT 的密钥，其次，生成了一个 Claims 对象，里面包含了一个 username 属性和三个默认属性：iss（发行人）、iat（颁发时间）、exp（过期时间），还可以自行添加更多属性。接着调用 `Jwts` 的 builder 方法，设置了 JWT 的头部参数“typ”和“alg”，然后调用 `signWith()` 方法指定了签名使用的算法 HS256，最后调用 `compact()` 方法生成了最终的 JWT 字符串。

## 3.4 校验 JWT

校验 JWT 时，我们需要把 JWT 放置到 Authorization header 中。因此，我们可以在需要校验 JWT 的接口上加上注解 `@PreAuthorize("hasAuthority('SCOPE_access')")`，这样只有拥有 `SCOPE_access` 权限的用户才能够访问该接口。

然后，我们编写一个 `@RestController` 控制器用于校验 JWT。首先，导入以下包：

```java
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import javax.servlet.http.HttpServletRequest;
import javax.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.config.annotation.method.configuration.EnableGlobalMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.web.authentication.AbstractAuthenticationProcessingFilter;
import org.springframework.security.web.authentication.AuthenticationFailureHandler;
import org.springframework.security.web.authentication.AuthenticationSuccessHandler;
import org.springframework.security.web.authentication.RememberMeServices;
import org.springframework.security.web.csrf.CookieCsrfTokenRepository;
import org.springframework.web.bind.annotation.*;
import com.nimbusds.jose.*;
import com.nimbusds.jose.crypto.MACVerifier;
```

然后，定义一个路径 `/checkToken`，用于校验 JWT。这里我只是简单地校验了 JWT 是否合法，并打印了一些信息：

```java
@RestController
public class CheckTokenController {

    private static final Logger logger = LoggerFactory.getLogger(CheckTokenController.class);
    
    @Value("${jwt.secret}") // 从配置文件中读取 JWT 的密钥
    private String secretKey;

    @Autowired
    @Qualifier("jwtUserDetailsService") // 注入自定义的 user details service
    private UserDetailsService userDetailsService;
    
    /**
     * 检查 JWT
     * @param request 请求对象
     * @return 用户名
     */
    @GetMapping("/checkToken")
    public String checkToken(HttpServletRequest request) throws Exception {

        String authorization = request.getHeader("Authorization");

        if (authorization!= null &&!"".equals(authorization)) {
            try {
                // 解析 JWT
                SignedJWT signedJWT = VerifierUtils.parseUnsecured(authorization.replace("Bearer ", ""), MACVerifier.class);

                // 根据 JWT 的 payload 中含有的 subject，查询数据库
                UserDetails user = this.userDetailsService.loadUserByUsername(signedJWT.getSubject());

                // 设置权限
                GrantedAuthority authority = new SimpleGrantedAuthority(user.getUsername());
                
                // 设置用户登录状态
                UsernamePasswordAuthenticationToken auth = 
                        new UsernamePasswordAuthenticationToken(user, null, authority);
                auth.setDetails(user);
                SecurityContextHolder.getContext().setAuthentication(auth);
                
                // 输出信息
                logger.info("Current user login successfully.");
                
                return user.getUsername();

            } catch (JWSException e) {
                // JWT 解析失败
                throw new Exception("Invalid JWT token");
            }

        } else {
            // 缺少 JWT token
            throw new Exception("No JWT token in the request header");
        }

    }
    
}
```

以上代码首先从请求头中取出 JWT 字符串，然后调用 `VerifierUtils.parseUnsecured()` 方法解析 JWT。因为我们采用的是 HMAC SHA256 算法签名的 JWT，所以我们用 `MACVerifier` 作为解码器。如果解析成功，我们根据 JWT 的 payload 中含有的 subject，调用自定义的 `UserDetailsService` 查找用户信息。然后，我们设置了权限为当前用户的用户名，并设置用户登录状态。输出信息提示用户登录成功。

# 4. 总结

本文介绍了 Spring Boot 中的 JWT 的使用方法，并给出了具体的代码示例。Spring Boot 在集成 JWT 时可以极大的方便开发人员。但是，建议不要在生产环境中使用盐值（salt）、静态密钥（static key）的方式存储密钥。正确的做法应该是利用密钥管理工具（Key Management Tool）生成并管理密钥。除此之外，Spring Security 对 JWT 的支持更为全面，具体的授权功能可以使用 Spring Security 来完成。