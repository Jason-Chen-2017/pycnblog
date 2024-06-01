
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Apache Shiro是一个非常著名的Java安全框架。它可以帮助你解决用户权限管理、Web应用授权、REST API授权等方面的问题。由于其功能强大而广泛的使用范围，使得它在Java开发者中的知名度不亚于Spring Security。在本文中，我们将深入到Apache Shiro的源码中，探讨其原理及其对安全性的影响。

# 2.认证过程
在Apache Shiro中，身份验证（Authentication）涉及到两步：认证（Authentication）和授权（Authorization）。在认证过程中，Shiro会检查用户提交的用户名密码是否正确；如果正确，则返回一个Subject对象表示用户的身份。在授权过程中，Shiro会查询当前Subject拥有的角色和权限，并决定是否允许访问特定资源。

对于身份验证，Apache Shiro提供了多种方式进行配置：可以用配置文件指定Realm，也可以通过编码的方式指定Realm，还可以通过其他插件如JAAS或LDAP等进行集成。通过Realm，Shiro可以获取用户相关信息，包括用户的身份（Username）、密码（Password）、角色（Roles）、权限（Permissions）等。

当用户提交用户名和密码后，Shiro首先会调用相应的Realm进行身份验证。Realm主要负责从存储介质（比如数据库或者缓存）加载用户信息，并对用户提交的信息进行比对校验。如果用户名和密码校验成功，则会创建一个Subject对象，代表该用户的身份。

对于授权，Shiro通过Subject对象的isPermitted()方法进行判断，该方法接受一个字符串参数，表示需要访问的资源所对应的权限名称。比如，某个用户需要查看某些敏感数据的权限，那么他需要包含该权限才能正常访问这些数据。Apache Shiro通过配置或注解的方式，把权限映射到具体的资源上。Apache Shiro提供了各种内置的权限（比如“admin”、“create-user”），可以通过创建自定义权限进行扩展。

# 3.清除角色
除了身份验证和授权之外，Apache Shiro还提供了另一种机制——角色清除（Role Elimination）。Apache Shiro提供了RoleElevator接口用来清除subject对象的角色，其中包括：

public interface RoleElevator {

    void removeRoles(Collection<String> roles);
    
    void removeAllRoles();
    
} 

一般情况下，开发人员只需实现自己的RoleElevator接口，并注入到Shiro的SecurityManager里面即可。但是，Shiro也提供了一些内置的RoleElevator实现类：

1. LifecycleAwareRoleElevator：这个RoleElevator在Subject被销毁时自动移除所有角色；
2. DefaultRoleElevator：这是最常用的RoleElevator实现类，默认不会移除任何角色，所以开发人员可以根据需求进行定制化改造。

# 4.实践示例
假设我们要实现一个基于Apache Shiro的登录模块。首先，我们需要定义一个Realm：

```java
import org.apache.shiro.authc.*;
import org.apache.shiro.authz.AuthorizationInfo;
import org.apache.shiro.realm.AuthorizingRealm;
import org.apache.shiro.subject.PrincipalCollection;

public class MyRealm extends AuthorizingRealm {
    public String getName() {
        return "myRealm";
    }
 
    protected AuthorizationInfo doGetAuthorizationInfo(PrincipalCollection principals) {
        // TODO Auto-generated method stub
        return null;
    }
 
    protected AuthenticationInfo doGetAuthenticationInfo(AuthenticationToken token) throws AuthenticationException {
        // 模拟从数据库读取用户信息
        if ("zhangsan".equals(((UsernamePasswordToken)token).getUsername())
                && "<PASSWORD>".equals(((UsernamePasswordToken)token).getPassword())) {
            SimpleAuthenticationInfo info = new SimpleAuthenticationInfo("zhangsan", "password123", this.getName());
            return info;
        } else {
            throw new UnknownAccountException();//没找到帐号
        }
    }
 
}
```

然后，我们需要实现一个LoginFilter：

```java
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
 
import org.apache.shiro.SecurityUtils;
import org.apache.shiro.authc.AuthenticationException;
import org.apache.shiro.authc.AuthenticationToken;
import org.apache.shiro.authc.IncorrectCredentialsException;
import org.apache.shiro.authc.UnknownAccountException;
import org.apache.shiro.authc.UsernamePasswordToken;
import org.apache.shiro.web.filter.authc.FormAuthenticationFilter;
 
public class LoginFilter extends FormAuthenticationFilter {
     
    @Override
    protected boolean onAccessDenied(ServletRequest request, ServletResponse response) throws Exception {
         
        if (isLoginRequest(request, response)) {//判断是否是登录请求，不是的话直接返回401
            if (!"post".equalsIgnoreCase(((HttpServletRequest) request).getMethod())) {//请求方式不是POST
                saveRequestAndRedirectToLogin(request, response);//保存Request并重定向至登录页面
            }
            return false;//继续拦截器链
        }
 
        return true;//继续拦截器链
    }
 
    /**
     * 如果登录失败了，转到login页面重新登录
     */
    @Override
    protected boolean onLoginFailure(AuthenticationToken token, AuthenticationException e, HttpServletRequest request, HttpServletResponse response) {
        request.setAttribute("shiroLoginFailure", e);//存放异常对象到request里
        return super.onLoginFailure(token, e, request, response);//转到登录页面重新登录
    }
 
    /**
     * 当登录成功后跳转页面
     */
    @Override
    protected boolean onLoginSuccess(AuthenticationToken token, Subject subject, InetAddress address,
                                     Session session, HttpServletRequest request, HttpServletResponse response) throws Exception {
         
        if (isContinueChainBeforeSuccessfulAuthentication()) {
            //先将之前登录失败的异常从request中移除，因为登录成功了
            request.removeAttribute("shiroLoginFailure");//移除登录失败的异常对象
            return executeLogin(request, response);//执行登录
        }
 
        issueSuccessRedirect(request, response);
        
        return false;
    }
     
}
```

最后，我们需要配置ShiroFilter，并注入MyRealm和LoginFilter：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-4.0.xsd">
 
  <!-- 引入shiro自带的过滤器 -->
  <bean id="securityManager" class="org.apache.shiro.web.mgt.DefaultWebSecurityManager">
 
    <!-- 配置Realm -->
    <property name="realms">
      <list>
        <bean class="com.example.demo.MyRealm"/>
      </list>
    </property>
 
    <!-- 配置URL权限控制 -->
    <property name="rememberMeManager">
      <bean class="org.apache.shiro.web.mgt.CookieRememberMeManager">
        <property name="cookie" value="${cookie}"/>
      </bean>
    </property>
 
    <!-- 全局过滤器配置-->
    <property name="globalFilters">
      <list>
        <!-- 登录过滤器-->
        <bean class="com.example.demo.LoginFilter"/>
      </list>
    </property>
 
  </bean>
  
  <!-- shiro filter chain定义，从固定位置加载，防止在xml中硬编码此项导致的版本冲突 -->
  <bean id="shiroFilter" class="org.apache.shiro.web.filter.mgt.DefaultFilterChainDefinitionSource">
    <property name="definitions">
      <value>/static/** = anon</value>
      <value>/* = authc</value>
    </property>
  </bean>
  
</beans>
```

这样，一个简单的基于Apache Shiro的登录模块就搭建完成了，现在测试一下：

1. 在浏览器中输入http://localhost:8080/hello，由于没有登录，系统会跳转至登录页面：

 
2. 输入正确的用户名密码（zhangsan / password123），点击登录：

   
   可以看到，登录成功！由于我们配置了无限制访问，所以这个页面可以正常显示。

3. 浏览器刷新页面：

   
   可以看到，Shiro已经将我的身份认证记录到了session中，避免了重复登录。

4. 尝试进入http://localhost:8080/logout，退出当前登录的账户：

   
   可以看到，已经成功退出登录，再次访问http://localhost:8080/hello将会被重定向至登录页面。

5. 对已有账号进行修改密码测试：

    修改密码：

   ```java
   @RequestMapping("/modify_pwd")
   public String modifyPwd(Model model){
      model.addAttribute("success","true");
      return "/login";
   }
   ```
  
   
    修改后再登录：
    