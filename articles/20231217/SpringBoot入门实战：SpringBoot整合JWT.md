                 

# 1.背景介绍

Spring Boot 是一个用于构建新型 Spring 应用的优秀启动器。它的目标是提供一种简单的方法，以便将 Spring 应用程序从 Traditional Spring 项目到只包含驱动程序的 JAR 文件进行转换。Spring Boot 提供了一种简单的配置，使得开发人员可以快速地开始编写代码，而无需关心配置。

在这篇文章中，我们将讨论如何将 Spring Boot 与 JWT（JSON Web Token）整合在一起。首先，我们将介绍 JWT 的核心概念和联系，然后详细介绍 JWT 的算法原理和具体操作步骤，接着提供一个具体的代码实例，最后讨论 JWT 的未来发展趋势和挑战。

# 2.核心概念与联系

## 2.1 JWT 简介

JWT（JSON Web Token）是一种基于 JSON 的开放标准（RFC 7519）用于传递声明的方式。它的目的是在客户端和服务器之间传递安全的、自签名的数据。JWT 通常用于身份验证和授权，可以在 Web 应用程序、移动应用程序和 IoT 设备中使用。

## 2.2 JWT 的组成部分

JWT 由三个部分组成：

1. 头部（Header）：包含算法和编码方式。
2. 有效载荷（Payload）：包含一组声明，可以包含用户信息、权限等。
3. 签名（Signature）：用于验证数据的完整性和来源。

## 2.3 JWT 与 Spring Boot 的整合

Spring Boot 提供了对 JWT 的支持，可以通过 Spring Security 的 JWT 过滤器来实现 JWT 的身份验证和授权。此外，Spring Boot 还提供了一些 JWT 的辅助类，如 JWTParser 和 JWTBuilder，可以简化 JWT 的编码和解码过程。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解

## 3.1 JWT 的算法原理

JWT 使用了一些标准的算法来进行签名，这些算法包括 HMAC 和 RSA。HMAC 是一种基于共享密钥的消息认证码（MAC）算法，而 RSA 是一种基于公钥和私钥的加密算法。

JWT 的签名过程如下：

1. 首先，将头部和有效载荷通过 URL 编码后拼接成一个字符串。
2. 然后，使用选定的签名算法（如 HMAC 或 RSA）对拼接后的字符串进行签名。
3. 最后，将签名字符串与原始字符串拼接成最终的 JWT。

## 3.2 JWT 的具体操作步骤

### 3.2.1 生成 JWT

要生成 JWT，可以使用 Spring Boot 提供的 JWTBuilder 类。以下是一个简单的示例：

```java
import io.jsonwebtoken.*;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UsernameNotFoundException;

import java.util.Date;

public class JWTUtil {

    private final String SECRET_KEY = "your_secret_key";

    public String generateToken(UserDetails userDetails) {
        return Jwts.builder()
                .setSubject(userDetails.getUsername())
                .claim("authorities", userDetails.getAuthorities())
                .setIssuedAt(new Date())
                .setExpiration(new Date(System.currentTimeMillis() + 60 * 1000 * 60))
                .signWith(SignatureAlgorithm.HS512, SECRET_KEY)
                .compact();
    }
}
```

### 3.2.2 验证 JWT

要验证 JWT，可以使用 Spring Boot 提供的 JWTParser 类。以下是一个简单的示例：

```java
import io.jsonwebtoken.*;

public class JWTUtil {

    public boolean validateToken(String token) {
        try {
            Jwts.parser().setSigningKey("your_secret_key").parseClaimsJws(token);
            return true;
        } catch (MalformedTokenException e) {
            throw new IllegalArgumentException("Invalid JWT token");
        } catch (ExpiredJwtException e) {
            throw new IllegalArgumentException("Expired JWT token");
        } catch (UnsupportedJwtException e) {
            throw new IllegalArgumentException("Unsupported JWT token");
        } catch (SignatureException e) {
            throw new IllegalArgumentException("Invalid JWT signature");
        }
    }
}
```

### 3.2.3 从 JWT 中获取用户信息

要从 JWT 中获取用户信息，可以使用 Spring Security 的 JWT 过滤器。以下是一个简单的示例：

```java
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UsernameNotFoundException;

public class JWTUserDetailsService implements UserDetailsService {

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        // TODO: 从数据库或其他源中获取用户信息
        return new User(username, "", true, true, true, true, new ArrayList<>());
    }
}
```

# 4.具体代码实例和详细解释说明

在这个部分，我们将提供一个具体的代码实例，展示如何将 Spring Boot 与 JWT 整合在一起。

## 4.1 项目结构

```
spring-boot-jwt
├── src
│   ├── main
│   │   ├── java
│   │   │   ├── com
│   │   │   │   ├── example
│   │   │   │   │   ├── SpringBootJwtApplication.java
│   │   │   │   │   ├── config
│   │   │   │   │   │   ├── JwtConfig.java
│   │   │   │   │   ├── controller
│   │   │   │   │   │   ├── AuthController.java
│   │   │   │   │   ├── service
│   │   │   │   │   │   ├── AuthService.java
│   │   │   │   │   ├── utils
│   │   │   │   │   │   ├── JWTUtil.java
│   │   │   │   │   ├── security
│   │   │   │   │   │   ├── JWTUserDetailsService.java
│   │   │   │   │   └── filter
│   │   │   │   │       ├── JWTFilter.java
│   │   │   │   └── application.properties
│   │   └── resources
│   └── test
│       └── java
│           └── com
│               └── example
│                   └── SpringBootJwtApplicationTests.java
└── pom.xml
```

## 4.2 项目配置

在 `application.properties` 文件中，我们需要配置 Spring Security 的相关设置：

```properties
spring.security.user.name=user
spring.security.user.password=password
spring.security.user.roles=USER

security.basic.enabled=false
security.jwt.secret=your_secret_key
security.jwt.token.header=Authorization
security.jwt.token.prefix=Bearer
security.jwt.ignore.urls=/api/auth/login
```

## 4.3 项目实现

### 4.3.1 创建 JWT 工具类

在 `utils` 包中创建 `JWTUtil` 类，实现 JWT 的生成和验证：

```java
import io.jsonwebtoken.*;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Component;

import java.util.Date;
import java.util.HashMap;
import java.util.Map;

@Component
public class JWTUtil {

    private final String SECRET_KEY = "your_secret_key";

    public String generateToken(UserDetails userDetails) {
        return Jwts.builder()
                .setSubject(userDetails.getUsername())
                .claim("authorities", userDetails.getAuthorities())
                .setIssuedAt(new Date())
                .setExpiration(new Date(System.currentTimeMillis() + 60 * 1000 * 60))
                .signWith(SignatureAlgorithm.HS512, SECRET_KEY)
                .compact();
    }

    public boolean validateToken(String token) {
        try {
            Jwts.parser().setSigningKey(SECRET_KEY).parseClaimsJws(token);
            return true;
        } catch (MalformedTokenException e) {
            throw new IllegalArgumentException("Invalid JWT token");
        } catch (ExpiredJwtException e) {
            throw new IllegalArgumentException("Expired JWT token");
        } catch (UnsupportedJwtException e) {
            throw new IllegalArgumentException("Unsupported JWT token");
        } catch (SignatureException e) {
            throw new IllegalArgumentException("Invalid JWT signature");
        }
    }
}
```

### 4.3.2 创建 JWT 用户详细信息服务

在 `security` 包中创建 `JWTUserDetailsService` 类，实现用户详细信息服务：

```java
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

@Service
public class JWTUserDetailsService implements UserDetailsService {

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        // TODO: 从数据库或其他源中获取用户信息
        return new User(username, "", true, true, true, true, new ArrayList<>());
    }
}
```

### 4.3.3 创建 JWT 过滤器

在 `filter` 包中创建 `JWTFilter` 类，实现 JWT 过滤器：

```java
import io.jsonwebtoken.*;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.authorities.AuthorityUtils;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

@Component
public class JWTFilter extends OncePerRequestFilter {

    private final JWTUtil jwtUtil;
    private final JWTUserDetailsService jwtUserDetailsService;

    public JWTFilter(JWTUtil jwtUtil, JWTUserDetailsService jwtUserDetailsService) {
        this.jwtUtil = jwtUtil;
        this.jwtUserDetailsService = jwtUserDetailsService;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        final String requestTokenHeader = request.getHeader("Authorization");

        String username = null;
        String jwtToken = null;
        if (requestTokenHeader != null && requestTokenHeader.startsWith("Bearer ")) {
            jwtToken = requestTokenHeader.substring(7);
            try {
                username = jwtUtil.validateToken(jwtToken);
            } catch (IllegalArgumentException e) {
                response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Unauthorized request");
                return;
            }
        }

        if (username != null && SecurityContextHolder.getContext().getAuthentication() == null) {
            UserDetails userDetails = this.jwtUserDetailsService.loadUserByUsername(username);
            UsernamePasswordAuthenticationToken usernamePasswordAuthenticationToken =
                    new UsernamePasswordAuthenticationToken(userDetails, null, userDetails.getAuthorities());
            usernamePasswordAuthenticationToken.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
            SecurityContextHolder.getContext().setAuthentication(usernamePasswordAuthenticationToken);
        }

        filterChain.doFilter(request, response);
    }
}
```

### 4.3.4 创建认证控制器

在 `controller` 包中创建 `AuthController` 类，实现认证控制器：

```java
import io.jsonwebtoken.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    @Autowired
    private AuthenticationManager authenticationManager;

    @Autowired
    private JWTUtil jwtUtil;

    @Autowired
    private UserDetailsService userDetailsService;

    @PostMapping("/login")
    public ResponseEntity<?> authenticate(@RequestBody AuthRequest authRequest) {
        try {
            final UserDetails userDetails = this.userDetailsService
                    .loadUserByUsername(authRequest.getUsername());
            final UsernamePasswordAuthenticationToken usernamePasswordToken =
                    new UsernamePasswordAuthenticationToken(userDetails, null, userDetails.getAuthorities());
            this.authenticationManager.authenticate(usernamePasswordToken);
            final String jwtToken = this.jwtUtil.generateToken(userDetails);
            return ResponseEntity.ok().body(new AuthResponse(jwtToken));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(e.getMessage());
        }
    }

    @GetMapping("/logout")
    public ResponseEntity<?> logout() {
        // TODO: 实现用户退出
        return ResponseEntity.ok().build();
    }
}
```

### 4.3.5 创建认证请求和响应实体

在 `model` 包中创建 `AuthRequest` 和 `AuthResponse` 类，用于表示认证请求和响应：

```java
public class AuthRequest {

    private String username;
    private String password;

    // Getters and Setters
}

public class AuthResponse {

    private String token;

    // Getters and Setters
}
```

### 4.3.6 配置 Spring Security

在 `security` 包中创建 `WebSecurityConfig` 类，配置 Spring Security：

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.web.builders.HttpSecurityBuilder;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

@Configuration
@EnableWebSecurity
public class WebSecurityConfig extends WebSecurityConfigurerAdapter {

    @Autowired
    private JWTFilter jwtFilter;

    @Autowired
    private JWTUserDetailsService jwtUserDetailsService;

    @Override
    protected void configure(HttpSecurityBuilder http) throws Exception {
        http
                .csrf().disable()
                .authorizeRequests()
                .antMatchers("/api/auth/login").permitAll()
                .anyRequest().authenticated()
                .and()
                .addFilterBefore(jwtFilter, SecurityFilterChain.class);
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Autowired
    public void configureGlobal(AuthenticationManagerBuilder auth) throws Exception {
        auth.userDetailsService(jwtUserDetailsService).passwordEncoder(passwordEncoder());
    }
}
```

### 4.3.7 配置 Spring Boot

在 `application.properties` 文件中，配置 Spring Boot：

```properties
spring.application.name=spring-boot-jwt

server.port=8080

spring.datasource.url=jdbc:mysql://localhost:3306/spring_boot_jwt?useSSL=false
spring.datasource.username=root
spring.datasource.password=your_password

spring.jpa.hibernate.ddl-auto=update

spring.security.user.name=user
spring.security.user.password=password
spring.security.user.roles=USER

security.basic.enabled=false
security.jwt.secret=your_secret_key
security.jwt.token.header=Authorization
security.jwt.token.prefix=Bearer
security.jwt.ignore.urls=/api/auth/login
```

# 5.附录

## 5.1 关于 JWT 的未来发展

JWT 已经广泛应用于各种场景，如身份验证、授权、跨域通信等。未来，JWT 可能会继续发展，以适应新的需求和技术。例如，可能会出现更安全的签名算法，以解决现有算法的漏洞。此外，JWT 可能会被应用于更多的场景，如区块链技术、物联网等。

## 5.2 JWT 的潜在漏洞和安全风险

虽然 JWT 是一种强大的身份验证和授权机制，但它也存在一些潜在的漏洞和安全风险。以下是一些常见的问题：

1. 过期和未签名的 JWT：如果 JWT 未被正确签名或已过期，可能会导致安全问题。因此，在验证 JWT 时，需要检查其签名和有效期。
2. 重放攻击：如果 JWT 被窃取，攻击者可能会重复使用它，以获得无授权的访问。为了防止这种情况，需要确保 JWT 在每次请求中都是唯一的，并在不合法使用时立即失效。
3. 密钥泄露：如果 JWT 的密钥被泄露，攻击者可能会使用它来伪造有效的 JWT。因此，密钥需要保护得当，并定期更新。
4. 跨站请求伪造（CSRF）：虽然 JWT 本身不会导致 CSRF，但在某些情况下，可能会受到影响。为了防止 CSRF，需要在服务器端添加 CSRF 保护机制。

为了减少这些风险，需要采取一些措施，例如使用 HTTPS 传输 JWT、定期更新密钥、验证 JWT 的签名和有效期等。

# 6.参考文献

[1] JWT (JSON Web Token) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://tools.ietf.org/html/rfc7519

[2] JWT (JSON Web Token) - JWT.io。(n.d.). Retrieved from https://jwt.io/introduction/

[3] Spring Security。(n.d.). Retrieved from https://spring.io/projects/spring-security

[4] Spring Boot。(n.d.). Retrieved from https://spring.io/projects/spring-boot

[5] JSON Web Token (JWT) - Wikipedia。(n.d.). Retrieved from https://en.wikipedia.org/wiki/JSON_Web_Token

[6] RFC 7519 - JSON Web Token (JWT)。(n.d.). Retrieved from https://datatracker.ietf.org/doc/html/rfc7519

[7] JWT - Java Library for JSON Web Tokens。(n.d.). Retrieved from https://github.com/jwtk/jjwt

[8] Spring Security JWT。(n.d.). Retrieved from https://github.com/spring-projects/spring-security-oauth-jwt

[9] Spring Boot JWT Example。(n.d.). Retrieved from https://github.com/spring-projects/spring-boot/tree/main/spring-boot-samples/spring-boot-sample-jwt

[10] JWT - JSON Web Token - Auth0。(n.d.). Retrieved from https://auth0.com/learn/json-web-tokens/

[11] JWT - JSON Web Token - JWT.io。(n.d.). Retrieved from https://jwt.io/introduction/

[12] JWT - JSON Web Token - Baeldung。(n.d.). Retrieved from https://www.baeldung.com/json-web-tokens-jwt-tutorial-java-spring-security

[13] JWT - JSON Web Token - Medium。(n.d.). Retrieved from https://medium.com/@johnsonraw/a-guide-to-understanding-json-web-tokens-jwt-part-1-7f94d3f0f9a1

[14] JWT - JSON Web Token - DZone。(n.d.). Retrieved from https://dzone.com/articles/spring-boot-jwt-authentication-tutorial

[15] JWT - JSON Web Token - GeeksforGeeks。(n.d.). Retrieved from https://www.geeksforgeeks.org/json-web-token-jwt/

[16] JWT - JSON Web Token - Stack Overflow。(n.d.). Retrieved from https://stackoverflow.com/questions/tagged/jwt

[17] JWT - JSON Web Token - GitHub。(n.d.). Retrieved from https://github.com/jwt-token

[18] JWT - JSON Web Token - AuthGuide。(n.d.). Retrieved from https://authguide.com/json-web-tokens/

[19] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[20] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[21] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[22] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[23] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[24] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[25] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[26] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[27] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[28] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[29] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[30] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[31] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[32] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[33] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[34] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[35] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[36] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[37] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[38] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[39] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[40] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[41] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[42] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[43] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[44] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[45] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[46] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[47] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[48] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[49] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[50] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[51] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet Engineering Task Force)。(n.d.). Retrieved from https://jwt.io/introduction/

[52] JWT - JSON Web Token - JWT.io - JSON Web Token (JWT) - IETF (Internet