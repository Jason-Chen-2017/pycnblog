
作者：禅与计算机程序设计艺术                    

# 1.简介
  

`Spring Security`是一个基于`Java`的开源框架，它主要提供认证和授权的功能，允许用户访问受保护的资源。为了实现业务需求，开发者可以自定义实现`AuthenticationManager`、`UserDetailsService`、`UserDetail`等接口，并在配置中引用它们。这套体系虽然灵活、可扩展性强，但也存在一些局限性，如对于复杂业务场景的支持不够好，依赖了数据库存储密码的方式，安全性较弱等。因此，在某些业务场景下，可以使用自定义实现的`UserDetailsService`，即实现该接口，将用户信息从外部系统（如LDAP）同步到本地数据库，实现统一认证和授权管理，避免复杂的集成逻辑。这种方式可以有效降低对外部系统的依赖，提高系统的稳定性和安全性。本文将详细介绍如何实现自定义`UserDetailsService`，并提供一个简单的示例。

# 2.基本概念及术语
## 2.1 Spring Security
Spring Security是一个基于`Java`的开源框架，其核心组件包括：

1. Authentication: 用户身份验证。
2. Authorization: 用户权限控制。
3. Cryptography: 加密工具。
4. ACL: 对系统资源进行访问控制。
5. Session Management: 会话管理。
6. Web Security: `Web`应用安全控制。
7. Portlet Security: `Portlet`应用安全控制。
8. JavaEE Security: `Java EE`应用安全控制。
9. OpenID/OAuth: 集成第三方登录。

其中，Authentication、Authorization和Cryptography属于核心模块；ACL和Session Management属于集成模块；Web Security、Portlet Security和JavaEE Security属于支持模块；OpenID/OAuth属于第三方登录模块。

## 2.2 UserDetailsService
`UserDetailsService`是一个接口，用于加载用户详情信息，其中最重要的方法是`loadUserByUsername()`，它返回了一个实现了`UserDetails`接口的对象，该对象包含了用户名、密码、角色、权限等用户信息。以下是该接口的定义：

```java
public interface UserDetailsService {
    /**
     * Locates the user based on the username. In the actual implementation, the search may possibly be case
     * insensitive, or case sensitive depending on how the
     */
    UserDetails loadUserByUsername(String username) throws UsernameNotFoundException;

    //...
}
```

该接口继承自`UserDetailsChecker`，用于检查用户属性，如过期、锁定等状态。如果没有找到用户，则抛出`UsernameNotFoundException`。

## 2.3 UserDetails
`UserDetails`是一个接口，包含了用户相关的信息，如下所示：

```java
public interface UserDetails extends Serializable {
    Collection<? extends GrantedAuthority> getAuthorities();
    
    String getPassword();
    
    String getUsername();
    
    boolean isAccountNonExpired();
    
    boolean isAccountNonLocked();
    
    boolean isCredentialsNonExpired();
    
    boolean isEnabled();
    
}
```

其中，`getAuthorities()`方法返回的是用户拥有的权限列表；`getPassword()`方法返回的是用户的密码；`getUsername()`方法返回的是用户的名称；`isAccountNonExpired()`, `isAccountNonLocked()`, `isCredentialsNonExpired()`, `isEnabled()`方法用于获取用户账号的状态。

## 2.4 GrantedAuthority
`GrantedAuthority`是一个接口，表示授予给用户的一项权限或特权，一般由字符串标识符表示，如"ROLE_ADMIN"、"IS_AUTHENTICATED_ANONYMOUSLY"等。

# 3.实现UserDetailsService

## 3.1 引入jar包
首先，引入Spring Security和LDAP jar包。如果只用到`InMemoryDaoImpl`，不需要引入其他jar包。

```xml
        <dependency>
            <groupId>org.springframework.security</groupId>
            <artifactId>spring-security-config</artifactId>
            <version>${spring-security.version}</version>
        </dependency>

        <dependency>
            <groupId>org.springframework.security</groupId>
            <artifactId>spring-security-core</artifactId>
            <version>${spring-security.version}</version>
        </dependency>
        
        <!-- LDAP support -->
        <dependency>
            <groupId>org.springframework.security</groupId>
            <artifactId>spring-security-ldap</artifactId>
            <version>${spring-security.version}</version>
        </dependency>
```

## 3.2 配置文件
然后，按照官方文档，在`applicationContext.xml`文件中配置认证信息。这里配置`UserDetailsService`接口的实现类，并设置参数，包括：

1. `authentication-provider`: 指定`UserDetailsService`接口的实现类，比如：`com.example.MyUserDetailsServiceImpl`。
2. `contextSource`: 设置LDAP服务器地址、端口号、用户名、密码以及搜索基准DN等参数。
3. `userDnPatterns`: 指定用户的搜索模式，这里指定了用用户名搜索，比如："uid={0}"。
4. `groupSearchBase`: 设置组搜索基准DN。

```xml
<!-- 配置LDAP登录 -->
<bean id="ldapAuthProvider" 
    class="org.springframework.security.ldap.authentication.LdapAuthenticationProvider">
    <!-- 使用自定义的UserDetailsService -->
    <property name="userDetailsContextMapper" ref="myUserDetailsContextMapper"/>
    <property name="authoritiesPopulator" 
        ref="myLdapGrantedAuthoritiesPopulator"/>
    <!-- 配置LDAP服务器 -->
    <property name="contextSource">
        <bean class="org.springframework.security.ldap.DefaultSpringSecurityContextSource">
            <constructor-arg value="${ldap.url}"/>
            <constructor-arg value="${ldap.port}"/>
            <property name="managerDn" value="${ldap.username}"/>
            <property name="managerPassword" value="${ldap.password}"/>
        </bean>
    </property>
    <!-- 配置搜索模式 -->
    <property name="userDnPatterns">
        <list>
            <value>"uid={0}"</value>
        </list>
    </property>
    <!-- 设置组搜索 -->
    <property name="groupSearchBase">${ldap.groupsearchbase}</property>
    <property name="groupRoleAttributes">
        <map key-type="java.lang.String">
            <entry key="memberUid">
                <list>
                    <value>"{0}"</value>
                </list>
            </entry>
        </map>
    </property>
    <property name="groupSearchFilter">"(&amp;(objectClass=groupOfUniqueNames)(uniqueMember={0}))"</property>
</bean>

<bean id="myLdapGrantedAuthoritiesPopulator" 
       class="org.springframework.security.ldap.populators.DefaultLdapAuthoritiesPopulator">
   <!--...省略配置信息... -->
</bean>

<bean id="myUserDetailsContextMapper" class="com.example.MyUserDetailsContextMapper"/>

<security:custom-filter position="FORM_LOGIN_FILTER" type="org.springframework.security.web.authentication.www.BasicAuthenticationFilter" />

<!-- 配置身份验证管理器 -->
<security:authentication-manager alias="authenticationManager">
    <security:authentication-provider ref="ldapAuthProvider"/>
    <!-- 如果需要支持本地登录，则添加LocalDaoAuthenticationProvider -->
    <!--<security:authentication-provider ref="localAuthProvider"/>-->
</security:authentication-manager>
```

## 3.3 创建自定义UserDetailsService接口的实现类
创建一个实现了`UserDetailsService`接口的类，重写`loadUserByUsername()`方法。此处仅展示最简单的`loadUserByUsername()`方法，实际情况可能要更复杂。

```java
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

@Service("userDetailsService")
public class MyUserDetailsServiceImpl implements UserDetailsService {

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        if ("admin".equals(username)) {
            return new User(username, "123456", true, true, true, true,
                SimpleGrantedAuthority.create("USER"),
                SimpleGrantedAuthority.create("ADMIN"));
        } else {
            throw new UsernameNotFoundException("用户不存在");
        }
    }
}
```

## 3.4 创建自定义UserDetailsContextMapper
当系统调用`LdapAuthenticationProvider`进行LDAP认证时，会执行`contextMapper`方法，将LDAP的用户数据映射到`UserDetails`接口。创建自定义`UserDetailsContextMapper`接口的实现类，覆写`mapUserFromContext()`方法。此处仅展示最简单的`mapUserFromContext()`方法，实际情况可能要更复杂。

```java
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import javax.naming.NamingException;

import org.springframework.ldap.core.DirContextOperations;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.ldap.userdetails.LdapUserDetails;
import org.springframework.security.ldap.userdetails.LdapUserDetailsContextMapper;

/**
 * 将LDAP中的用户数据转换为Spring Security的UserDetails接口
 * 
 * @author Administrator
 */
public class MyUserDetailsContextMapper implements LdapUserDetailsContextMapper {

    private static final List<GrantedAuthority> DEFAULT_AUTHORITIES = new ArrayList<>();

    static {
        DEFAULT_AUTHORITIES.add(new SimpleGrantedAuthority("USER"));
    }

    @Override
    public UserDetails mapUserFromContext(DirContextOperations ctx, String username,
            Collection<? extends GrantedAuthority> authorities) {
        try {
            return new LdapUserDetails(ctx.getAttributes(), username, DEFAULT_AUTHORITIES);
        } catch (NamingException e) {
            throw new IllegalStateException("无法获取LDAP用户数据：" + e.getMessage());
        }
    }

    @Override
    public void mapUserToContext(UserDetails user, DirContextOperations ctx) {
        // do nothing here
    }

}
```