                 

# 1.背景介绍


## 什么是Spring Security？
　　Apache Shiro是一个基于Java的权限管理框架，它提供了一整套完整的安全机制。但是，作为一个老牌的Java框架，Shiro存在一些局限性，比如性能上的不足、功能的单一化、架构设计上过于复杂等。因此，很多人选择了Spring Security作为更加轻量级的替代方案。Spring Security是Spring Framework的一部分，它是一个针对 Java enterprise edition (Java EE)应用的安全框架。它提供一系列的安全控制功能，包括身份认证（Authentication）、授权（Authorization）和加密（Encryption）。
　　Spring Security支持多种场景下的安全需求，如：web环境下对Http请求进行安全保护；API接口服务端对RESTful接口请求进行安全保护；单点登录SSO等。同时，它也提供了丰富的扩展机制，允许开发者自定义安全控制流程及实现自己的安全需求。
　　本文将结合Spring Boot和Spring Security来探讨其安全特性。Spring Boot是为了简化Spring应用的开发，通过自动配置的方式，可以快速地创建独立运行的基于Spring的应用。本文将围绕Spring Boot和Spring Security展开讨论。
## Spring Security的角色
　　在理解Spring Security之前，需要了解一下Spring Security的角色。在Spring Security中，有以下4个主要角色：

　　1. Authentication Manager: 对用户进行身份验证并生成相应的Authentication对象。通常会委托给不同的AuthenticationProvider。
　　2. AccessDecisionManager: 根据SecurityContextHolder的内容决定是否允许访问某个资源。通常会委托给AccessDecisionVoter。
　　3. SecurityContextHolder: 在整个请求过程中保存当前用户的身份信息。
　　4. Filter Chain: 请求处理链，负责调用相应的过滤器进行请求预处理或后处理。

　　这些角色之间的关系如下图所示：

## Spring Security安全上下文
　　Spring Security从外部输入的请求到内部代码执行都要经过一系列的安全过滤器。Spring Security将外部输入的所有请求都封装成AuthenticationToken对象，再交由AuthenticationManager进行身份认证。如果认证成功，则把Authentication对象放入SecurityContextHolder。每个线程都有自己对应的SecurityContextHolder，保存着当前线程的身份验证状态信息。

　　当线程受到请求时，Spring Security会根据配置好的FilterChain来决定如何处理该请求。Spring Security默认的FilterChain包含三个过滤器：SecurityContextPersistenceFilter、ConcurrentSessionFilter和UsernamePasswordAuthenticationFilter。前两个过滤器用于保证安全上下文的持久化和并发控制，而后者负责处理用户名密码相关的认证请求。

　　当认证成功之后，SecurityContextHolder中的身份认证信息就会被传递给应用代码。接下来，应用代码就可以直接使用SecurityContextHolder来获取身份信息，以及根据特定策略来做访问控制。

## Spring Security架构模式
　　Spring Security的架构模式遵循MVC模式，如下图所示：

Spring Security的工作流程大致分为以下几个阶段：

　　　　1. 用户向客户端发送HTTP请求。

　　　　2. 请求被Spring Security拦截器拦截。

　　　　3. 拦截器尝试获取SecurityContext。如果不存在SecurityContext，那么就创建一个空的SecurityContext。

　　　　4. 如果SecurityContext为空，那么就创建一个匿名的Authentication。

　　　　5. 将HTTP请求包装成HttpServletRequestWrapper。

　　　　6. 将HttpServletRequestWrapper传递给AuthenticationManager进行身份认证。

　　　　7. 如果认证成功，就向SecurityContextHolder设置Authentication。

　　　　8. 从SecurityContextHolder中获取Authentication。

　　　　9. 创建FilterInvocation。

　　　　10. 将FilterInvocation传递给AccessDecisionManager进行访问控制。

　　　　11. 如果访问被允许，FilterChain才会继续调用其他过滤器或处理请求。否则，将产生一个AccessDeniedException异常。