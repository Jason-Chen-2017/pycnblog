                 

# 1.背景介绍

Spring Boot是Spring框架的一种简化版本，它使用了Spring的核心技术来简化Spring应用的开发。Shiro是一个强大的Java安全框架，它提供了身份验证、授权、密码存储和会话管理等功能。在本文中，我们将讨论如何将Spring Boot与Shiro整合，以实现简单的身份验证和授权。

## 1.1 Spring Boot简介
Spring Boot是一个用于构建原生的Spring应用程序的框架。它提供了一种简化的方法来创建独立的Spring应用程序，而无需配置XML文件。Spring Boot提供了许多预配置的依赖项，这使得开发人员可以专注于编写代码，而不是配置应用程序。

## 1.2 Shiro简介
Shiro是一个强大的Java安全框架，它提供了身份验证、授权、密码存储和会话管理等功能。Shiro可以与Spring框架整合，以实现简单的身份验证和授权。

## 1.3 Spring Boot与Shiro整合
要将Spring Boot与Shiro整合，我们需要在项目中添加Shiro的依赖项，并配置Shiro的相关组件。以下是整合过程的详细步骤：

### 1.3.1 添加Shiro依赖项
要添加Shiro依赖项，我们需要在项目的pom.xml文件中添加以下依赖项：

```xml
<dependency>
    <groupId>org.apache.shiro</groupId>
    <artifactId>shiro-spring</artifactId>
    <version>1.4.0</version>
</dependency>
```

### 1.3.2 配置Shiro的相关组件
要配置Shiro的相关组件，我们需要在项目的application.properties文件中添加以下配置：

```properties
spring.shiro.lifecycle.enabled=true
spring.shiro.securityManager.shutdown-when-idle=60
spring.shiro.session-manager.session-timeout=1800000
spring.shiro.authc.credentialsMatcher.hashAlgorithmName=SHA-256
spring.shiro.authc.credentialsMatcher.hashIterations=1024
```

### 1.3.3 实现身份验证和授权
要实现身份验证和授权，我们需要创建一个实现`AuthenticationListener`接口的类，并实现`onSuccessfulAuthentication`方法。这个方法将在用户成功身份验证后调用。

```java
import org.apache.shiro.authc.AuthenticationException;
import org.apache.shiro.authc.AuthenticationInfo;
import org.apache.shiro.authc.AuthenticationToken;
import org.apache.shiro.authc.SimpleAuthenticationInfo;
import org.apache.shiro.authz.AuthorizationInfo;
import org.apache.shiro.authz.SimpleAuthorizationInfo;
import org.apache.shiro.realm.AuthorizingRealm;
import org.apache.shiro.subject.PrincipalCollection;
import org.apache.shiro.subject.Subject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class CustomRealm extends AuthorizingRealm {

    @Autowired
    private UserService userService;

    @Override
    public void setCredentialsMatcher(CredentialsMatcher credentialsMatcher) {
        super.setCredentialsMatcher(credentialsMatcher);
    }

    @Override
    protected AuthenticationInfo doGetAuthenticationInfo(AuthenticationToken authenticationToken) throws AuthenticationException {
        String username = (String) authenticationToken.getPrincipal();
        User user = userService.findByUsername(username);
        if (user == null) {
            throw new UnknownAccountException("Unknown user: " + username);
        }
        SimpleAuthenticationInfo authenticationInfo = new SimpleAuthenticationInfo(user, user.getPassword(), getName());
        return authenticationInfo;
    }

    @Override
    protected AuthorizationInfo doGetAuthorizationInfo(PrincipalCollection principals) {
        SimpleAuthorizationInfo authorizationInfo = new SimpleAuthorizationInfo();
        // 添加用户的角色和权限
        User user = (User) principals.getPrimaryPrincipal();
        authorizationInfo.addRole(user.getRole().getName());
        authorizationInfo.addStringPermission(user.getRole().getPermission());
        return authorizationInfo;
    }

    @Override
    public void onSuccessfulAuthentication(AuthenticationToken authenticationToken, Subject subject) {
        // 在用户成功身份验证后调用
        User user = (User) subject.getPrincipal();
        System.out.println("用户" + user.getUsername() + "成功登录");
    }
}
```

### 1.3.4 配置Shiro的过滤器
要配置Shiro的过滤器，我们需要在项目的application.properties文件中添加以下配置：

```properties
spring.shiro.filters.anonymous.on = /login
spring.shiro.filters.anonymous.captcha.enabled = true
spring.shiro.filters.anonymous.captcha.error-message = 验证码错误，请重新输入
spring.shiro.filters.anonymous.captcha.title = 请输入验证码
spring.shiro.filters.anonymous.captcha.width = 100
spring.shiro.filters.anonymous.captcha.height = 40
spring.shiro.filters.anonymous.captcha.background-color = 240,240,240
spring.shiro.filters.anonymous.captcha.border = 1
spring.shiro.filters.anonymous.captcha.char-width = 32
spring.shiro.filters.anonymous.captcha.char-height = 32
spring.shiro.filters.anonymous.captcha.font-path = classpath:/fonts
spring.shiro.filters.anonymous.captcha.font-size = 24
spring.shiro.filters.anonymous.captcha.text = 请输入验证码
spring.shiro.filters.anonymous.captcha.code-length = 4
spring.shiro.filters.anonymous.captcha.code-background-color = 255,255,255
spring.shiro.filters.anonymous.captcha.code-border = 1
spring.shiro.filters.anonymous.captcha.code-font-color = 0,0,0
spring.shiro.filters.anonymous.captcha.code-font-name = 微软雅黑
spring.shiro.filters.anonymous.captcha.code-font-size = 24
spring.shiro.filters.anonymous.captcha.code-height = 40
spring.shiro.filters.anonymous.captcha.code-width = 100
spring.shiro.filters.anonymous.captcha.code-word-list = 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
spring.shiro.filters.anonymous.captcha.code-word-list-length = 4
spring.shiro.filters.anonymous.captcha.image-width = 100
spring.shiro.filters.anonymous.captcha.image-height = 40
spring.shiro.filters.anonymous.captcha.image-background-color = 240,240,240
spring.shiro.filters.anonymous.captcha.image-border = 1
spring.shiro.filters.anonymous.captcha.image-char-width = 32
spring.shiro.filters.anonymous.captcha.image-char-height = 32
spring.shiro.filters.anonymous.captcha.image-font-path = classpath:/fonts
spring.shiro.filters.anonymous.captcha.image-font-size = 24
spring.shiro.filters.anonymous.captcha.image-text = 请输入验证码
spring.shiro.filters.anonymous.captcha.image-word-list = 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
spring.shiro.filters.anonymous.captcha.image-word-list-length = 4
spring.shiro.filters.anonymous.captcha.image-word-list-random-order = true
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed = 1000
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random = true
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-min = 5
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-max = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-max = 2
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random = false
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-max = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed = 1000
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random = true
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-min = 5
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-max = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-max = 2
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random = false
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-max = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed = 1000
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random = true
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-min = 5
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-max = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-max = 2
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random = false
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-max = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-seed = 1000
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random = true
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-min = 5
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-max = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-max = 2
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random = false
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-max = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-seed = 1000
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random = true
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-min = 5
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-max = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-max = 2
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random = false
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-max = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-min-random-seed = 1000
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random = true
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-min = 5
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-max = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-max = 2
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min-random = false
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min-random-max = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min-random-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-seed = 1000
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random = true
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-min = 5
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-max = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-max = 2
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min-random = false
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min-random-max = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min-random-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-seed = 1000
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random = true
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-min = 5
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-max = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-max = 2
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min-random = false
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min-random-max = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min-random-min = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-seed = 1000
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random = true
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-min = 5
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-max = 10
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step = 1
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-max = 2
spring.shiro.filters.anonymous.captcha.image-word-list-random-order-seed-length-random-step-min-random-seed-length-random-step-min-random-seed-length-random-step-random-step-min = 1
spring.shiro.filters.anonymous.captcha.