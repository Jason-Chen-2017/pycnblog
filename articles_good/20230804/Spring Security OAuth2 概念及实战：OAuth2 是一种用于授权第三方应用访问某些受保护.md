
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         OAuth（开放授权）是一个开放标准，它允许用户提供一个标识符让应用得以安全地请求特定的服务（如照片，视频或联系人数据）。该标准由 IETF 管理，并得到了行业的广泛支持。OAuth 允许用户直接授权给第三方应用其需要的权限，而无需将用户名和密码等敏感信息暴露给这些应用。
         
         OAuth2 是 OAuth 的最新版本，具有更好的安全性、更强的可伸缩性和易用性。它允许第三方应用获得仅限特定资源的访问权，并且不会向资源所有者授予超过所需的权限。
         
         本文会介绍 Spring Security 在使用 OAuth2 时，一般涉及到的一些术语，并结合具体的代码案例，介绍如何通过 Spring Security 来对接 OAuth2 。
         # 2.基本概念术语说明

         ## OAuth2 协议流程


         上图为 OAuth2 协议的一个典型流程图。客户端（Client）需要通过认证服务器申请一个令牌（Token），然后通过资源服务器（Resource Server）获取自己需要的资源。其中，认证服务器就是颁发令牌的服务器，通常部署在互联网的某个位置。资源服务器则是托管需要保护资源的服务器，也是可以访问受保护资源的服务器。

         1. 用户同意授权。

           用户向 Client 注册并登录成功后，将被要求进行授权。例如，当 Client 需要访问自己的 Twitter 账户时，需要给予授权，确认是否允许 Client 获取其帐号内的私密信息。

         2. Client 请求授权码。

           当用户同意授权后，Client 会得到一个授权码，表示用户同意授权，然后生成一个令牌。授权码只能使用一次，授权码泄露可能导致令牌泄露。

         3. Client 向认证服务器请求令牌。

            Client 将授权码发送给认证服务器，请求获取令牌。认证服务器验证授权码，如果合法，就会颁发一个令牌，此令牌用于代表 Client 的身份。同时，还会向 Client 返回其他相关信息，如刷新令牌（refresh token）等。

         4. Client 使用令牌访问资源。

           Client 将令牌发送给资源服务器，请求访问受保护的资源。资源服务器验证令牌，如果合法，就返回资源给 Client。

         5. Client 更新令牌。

           如果当前令牌即将过期，Client 可以使用刷新令牌来获取新的令牌。

         6. 终止授权。

           用户决定不再同意 Client 申请的权限，或者出于其他原因，要终止授权关系，Client 可以向认证服务器发送一个请求，撤销之前颁发的令牌。

         ## OAuth2 角色

         ### 资源所有者

         资源所有者，也称为授权服务器，就是颁发令牌的服务器。这个服务器负责保管客户机凭据（client credentials）和受保护资源，以及确定哪个客户端可以访问这些资源。客户端必须向资源所有者注册，才能取得凭据。

         1. 注册客户端（Client Registration）

            资源所有者在向客户端提供服务之前，首先需要向认证服务器注册客户端。注册过程主要包括以下几个步骤：

            1. 提供必要的信息，比如：客户端 ID 和客户端密钥；
            2. 指定客户端类型；
            3. 设置重定向 URI；
            4. 选择许可范围（Scopes）。

         2. 授予客户端权限（Authorization Grant）

             注册完成后，客户端可以请求授予权限。不同的 grant type 有不同的作用，比如授权码 grant （授权码模式）和隐式 grant（隐式模式）。

             授权码 grant 模式：

             1. 用户同意授权；
             2. 客户端发起请求，获取授权码；
             3. 资源所有者向认证服务器提交授权码；
             4. 认证服务器确认授权码有效，颁发令牌。

             隐式 grant 模式：

             1. 用户同意授权；
             2. 客户端向资源所有者请求授权，带上用户名和密码或其他认证方式（如设备指纹、扫码或 FaceID 等）。
             3. 资源所有者确认授权，颁发令牌。

             注意：OAuth2 定义了四种类型的 grant ，但实际上只需要采用两种：授权码 grant 和隐式 grant 。

         3. 配置受保护资源（Protected Resource）

             资源所有者根据需求配置受保护资源，指定每个资源应该具备哪些权限。

         4. 检查访问权限（Access Token Introspection）

             认证服务器可以检查令牌的有效性，以确定用户是否被授权访问受保护资源。

         ### 客户端（Client）

         客户端是需要访问受保护资源的应用程序。它需要向资源所有者索取授权，才能获取到令牌。

         1. 请求授权码（Authorization Code Request）

             客户端首先需要向资源所有者发起授权请求，请求获得授权码，向用户确认授权。

         2. 获取访问令牌（Access Token Delivery）

             资源所有者向客户端颁发访问令牌，表示该客户端已被授权访问资源。客户端可以使用访问令牌访问受保护资源。

         3. 使用访问令牌访问资源（Resource Access Using an Access Token）

             客户端使用访问令牌访问受保护资源，请求资源的过程是透明的，用户无需额外的操作。

         ### 用户代理（User Agent）

         客户端可以理解为用户代理（user agent），也就是最终使用资源的应用程序。虽然 OAuth2 只是一种协议，但是具体的实现方案往往会依赖于具体的用户代理。目前主流的用户代理有浏览器、手机 App 和命令行工具。

         ## OAuth2 实现

         ### 认证服务器

         我们可以使用 Spring Boot + Spring Security + OAuth2 来实现一个简单的认证服务器。

         Spring Boot 是基于 Spring Framework 的一个轻量级框架，可以快速启动 Spring 应用。Spring Security 是 Spring 框架中的一个安全模块，用于集成 Spring Webmvc 或 Spring WebFlux 等 Web 框架，提供一系列的安全相关的功能，包括身份验证、授权、加密传输等。OAuth2 是微服务架构中的一种授权机制，用来保护系统中不同服务之间的安全通信。

         Maven 依赖如下：

         ```xml
         <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-security</artifactId>
        </dependency>
         <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
         <dependency>
            <groupId>org.springframework.security.oauth</groupId>
            <artifactId>spring-security-oauth2</artifactId>
            <version>${spring-security-oauth2.version}</version>
        </dependency>
         <!-- Other dependencies -->
         ```

         创建一个 SpringBootApplication 类，添加如下配置：

         ```java
         @Configuration
         @EnableWebSecurity // Enable security auto-configuration
         public class AuthorizationServerConfig extends WebSecurityConfigurerAdapter {

             @Autowired
             private UserDetailsService userDetailsService;

             /**
              * This method will be called when the application starts up to initialize the authorization server with necessary details and configurations.
              */
             @Override
             protected void configure(HttpSecurity http) throws Exception {
                 // Allow anonymous access to the endpoints
                 http.authorizeRequests().antMatchers("/oauth/**").permitAll();

                 // Set a session timeout of one hour for all requests except /oauth/**
                 http
                    .sessionManagement()
                    .sessionCreationPolicy(SessionCreationPolicy.IF_REQUIRED)
                    .and()
                    .requestCache().disable() // Disable default request cache as it may cause problems with authorization code flow
                    .csrf().disable(); // Disable CSRF protection due to the JWT tokens being added on authentication
             }

             /**
              * Configures the client details service which fetches the list of authorized clients from database or any other configuration source.
              */
             @Bean
             public ClientDetailsService clientDetailsService() throws IOException {
                 List<ClientDetails> clientDetailsList = new ArrayList<>();
                 InputStream inputStream = getClass().getResourceAsStream("/clients.yml");
                 Yaml yaml = new Yaml();
                 Map<String, Object> map = yaml.load(inputStream);
                 if (map!= null &&!map.isEmpty()) {
                     for (Object obj : map.values()) {
                         String clientId = ((Map<String, String>) obj).get("clientId");
                         String clientSecret = ((Map<String, String>) obj).get("clientSecret");
                         List<String> scope = Arrays.asList(((Map<String, String>) obj).get("scope").split(","));
                         Set<String> authorizedGrantTypes = Collections
                            .singleton(((Map<String, String>) obj).get("authorizedGrantTypes"));
                         int accessTokenValiditySeconds = Integer
                           .parseInt((String) ((Map<String, String>) obj).get("accessTokenValiditySeconds"));
                         int refreshTokenValiditySeconds = Integer
                           .parseInt((String) ((Map<String, String>) obj).get("refreshTokenValiditySeconds"));
                         List<String> resourceIds = Arrays.asList(((Map<String, String>) obj).get("resourceIds").split(","));
                         boolean autoApprove = Boolean.parseBoolean((String) ((Map<String, String>) obj).get("autoApprove"));
                         clientDetailsList.add(new ClientDetailsImpl(clientId, clientSecret, authorizedGrantTypes,
                                 resourceIds, scope, accessTokenValiditySeconds, refreshTokenValiditySeconds,
                                 autoApprove));
                     }
                 }
                 return new InMemoryClientDetailsService(clientDetailsList);
             }

             /**
              * Sets the password encoder used to encrypt passwords while saving them in the database. SHA-256 is used by default.
              */
             @Bean
             public PasswordEncoder passwordEncoder() {
                 return NoOpPasswordEncoder.getInstance(); // Not recommended for production use
             }

             /**
              * Configures the user detail service which loads users based on their usernames from the database. It also maps each user with its corresponding authorities.
              */
             @Bean
             public DaoAuthenticationProvider daoAuthenticationProvider() {
                 DaoAuthenticationProvider provider = new DaoAuthenticationProvider();
                 provider.setUserDetailsService(this.userDetailsService());
                 provider.setPasswordEncoder(passwordEncoder());
                 return provider;
             }

             /**
              * Specifies the custom error page controller.
              */
             @ControllerAdvice
             public static class GlobalExceptionHandler {
                 @ExceptionHandler(Exception.class)
                 public ResponseEntity handleException(Exception e) {
                     log.error("", e);

                     String errorMessage = "Internal server error";
                     if (e instanceof BadCredentialsException || e instanceof AuthenticationException) {
                         errorMessage = "Invalid username or password.";
                     } else if (e instanceof AccessDeniedException) {
                         errorMessage = "You do not have permission to perform this action.";
                     } else if (e instanceof UnsupportedMediaTypeException) {
                         errorMessage = "Unsupported media type. Please check your content type headers.";
                     }

                     ErrorResponse response = new ErrorResponse("SERVER_ERROR", errorMessage);
                     return ResponseEntity
                            .status(HttpStatus.INTERNAL_SERVER_ERROR)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(response);
                 }
             }
         }
         ```

         配置文件 clients.yml 中包含客户端的详细信息，包括客户端 ID、客户端密钥、访问权限范围、授权类型、令牌有效时间、刷新令牌有效时间、是否自动批准等属性。这里使用的是内存存储，实际生产环境建议使用数据库存储。

         **Password Encoder**

         默认情况下，Spring Security 不启用密码编码器。由于 OAuth2 中的密码不是保存到数据库中，所以不需要设置密码编码器。如果要在 OAuth2 服务端存储密码，需要自己实现密码编码器，推荐的做法是直接不进行编码，这会造成弱密码容易被猜测，建议在客户端加密密码后再提交到服务端。

         **User Detail Service**

         用户详情服务用于从数据库加载用户信息，并将其映射到权限模型中。

         为了简单起见，这里使用了一个 DaoAuthenticationProvider ，它利用 Spring Security 的 UsernamePasswordAuthenticationFilter 对用户名和密码进行验证。DaoAuthenticationProvider 要求配置一个 UserDetailsService ，用来查询用户信息，这里使用了一个自定义的 UserDetailsServiceImpl ，它从内存读取配置文件中的用户信息。如果你想把用户信息存储到数据库，可以在这里使用 JDBCUserDetailsManager 来替换 UserDetailsServiceImpl 。

         **Error Page Controller**

         通过注解 @ControllerAdvice ，我们可以捕获所有的异常，并生成对应的错误响应。

         根据不同的异常类型生成不同的错误响应，并返回对应的 HTTP 状态码和 JSON 数据。

         ### 资源服务器

         资源服务器也可以使用 Spring Security + OAuth2 来保护受保护资源。

         添加如下依赖：

         ```xml
         <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-security</artifactId>
        </dependency>
         <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
         <dependency>
            <groupId>org.springframework.security.oauth</groupId>
            <artifactId>spring-security-oauth2</artifactId>
            <version>${spring-security-oauth2.version}</version>
        </dependency>
         <!-- For testing purposes only -->
         <dependency>
            <groupId>org.springframework.security</groupId>
            <artifactId>spring-security-test</artifactId>
            <scope>test</scope>
        </dependency>
         ```

         资源服务器的配置比较简单，主要就是需要配置资源 ID 和配置受保护的资源，并使用 OAuth2AuthenticationEntryPoint 来处理 OAuth2 异常。

         ```java
         @Configuration
         @EnableWebSecurity // Enable security auto-configuration
         public class ResourceServerConfig extends WebSecurityConfigurerAdapter {

             /**
              * This method will be called when the application starts up to initialize the authorization server with necessary details and configurations.
              */
             @Override
             protected void configure(HttpSecurity http) throws Exception {
                 http
                    .exceptionHandling()
                    .authenticationEntryPoint(new OAuth2AuthenticationEntryPoint()) // Handle OAuth2 exceptions
                    .and()
                    .authorizeRequests()
                    .anyRequest().authenticated()
                    .and()
                    .oauth2ResourceServer()
                    .jwt(); // Configure JWT access token validation
             }

             /**
              * Enables CORS support for all endpoints using the following filter.
              */
             @Bean
             public CorsFilter corsFilter() {
                 UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
                 CorsConfiguration config = new CorsConfiguration();
                 config.setAllowCredentials(true);
                 config.addAllowedOrigin("*");
                 config.addAllowedHeader("*");
                 config.addAllowedMethod("*");
                 source.registerCorsConfiguration("/**", config);
                 return new CorsFilter(source);
             }

             /**
              * Configures the client registration repository containing the registered clients' configuration details like client id, secret, authorized scopes etc.
              */
             @Bean
             public ClientRegistrationRepository clientRegistrationRepository() {
                 ClientRegistration registration = ClientRegistration
                       .withRegistrationId("github")
                       .authorizationUri("https://github.com/login/oauth/authorize")
                       .tokenUri("https://github.com/login/oauth/access_token")
                       .clientName("GitHub")
                       .clientId("yourClientId")
                       .clientSecret("yourClientSecret")
                       .redirectUri("{baseUrl}/login/oauth2/code/{registrationId}")
                       .scope("read:user")
                       .build();
                 return new InMemoryClientRegistrationRepository(Collections.singletonList(registration));
             }

         }
         ```

         这里配置了一个 OAuth2AuthenticationEntryPoint ，用来处理 OAuth2 异常，并将它们转换为 HTTP 响应。

         为了测试方便，这里引入了 spring-security-test ，它提供了 MockMvc 对象，用于模拟 RESTful API 测试。

         资源服务器使用 JWT 来校验访问令牌，并提供相关接口以便客户端访问受保护资源。

         ```java
         @RestController
         public class ProtectedController {

             @Value("${app.secret}")
             private String appSecret;

             /**
              * Returns a protected resource. Only accessible to authenticated users with proper permissions.
              */
             @GetMapping("/api/protected")
             public ResponseEntity<?> getProtectedResource(@AuthenticationPrincipal Jwt jwt) {
                 if (!"test".equals(jwt.getClaimAsString("scope"))) {
                     throw new InsufficientScopeException("Not enough scope!");
                 }
                 return ResponseEntity.ok().build();
             }

             /**
              * Generates an access token for the given client credentials. Can be useful during development or for generating access tokens without redirecting through the authorization endpoint.
              */
             @PostMapping("/api/generate-token")
             public ResponseEntity<Jwt> generateToken(@RequestBody GenerateTokenRequest request) {
                 ClientDetails client = getClientByClientId(request.getClientId());
                 if ("my-trusted-client".equals(request.getClientId())) {
                     Claims claims = Jwts.claims().setSubject(request.getUsername());
                     claims.put("scope", "read:messages");
                     Date expirationTime = new Date(System.currentTimeMillis() + 60*1000); // 1 minute expiry time
                     String accessToken = Jwts.builder().claim(Claims.SUBJECT, request.getUsername()).claim("scope", "read:messages").setExpiration(expirationTime).signWith(SignatureAlgorithm.HS256, appSecret).compact();
                     return ResponseEntity.ok(new Jwt(accessToken, null, expirationTime));
                 } else {
                     throw new InvalidClientException("Unknown or unauthorized client.");
                 }
             }

             /**
              * Retrieves the client details based on the provided client Id. The implementation could vary depending on where the client information is stored.
              */
             private ClientDetails getClientByClientId(String clientId) {
                 try {
                     ClientRegistration clientRegistration = clientRegistrationRepository().findByRegistrationId("github").orElseThrow(() -> new NoSuchClientException("Could not find client with id " + clientId));
                     if (!clientRegistration.getClientId().equals(clientId)) {
                         throw new InvalidClientException("Given client does not match expected value.");
                     }
                     return InMemoryClientDetailsService.clientDetails().get(clientId);
                 } catch (IOException e) {
                     throw new RuntimeException(e);
                 }
             }

         }
         ```

         生成访问令牌的方法接受一个 GenerateTokenRequest 对象，包含客户端 ID 和用户名。对于受信任的客户端，我们生成含有读消息权限的 JWT 访问令牌；对于其他客户端，抛出 InvalidClientException 异常。

         资源服务器还提供了一个 /api/protected 接口，只有授权的用户才可以访问，其作用是返回受保护资源。

         此外，资源服务器还实现了跨域资源共享（Cross-Origin Resource Sharing，CORS），这样就可以允许前端 JavaScript 应用运行在不同的域名下。可以通过配置 CorsConfigurationSource 来实现。

         ### 客户端

         客户端使用 Spring Security + OAuth2 来访问资源。

         客户端项目中添加如下依赖：

         ```xml
         <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-security</artifactId>
        </dependency>
         <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
         <dependency>
            <groupId>org.springframework.security.oauth</groupId>
            <artifactId>spring-security-oauth2-client</artifactId>
            <version>${spring-security-oauth2.version}</version>
        </dependency>
         <dependency>
            <groupId>org.springframework.security.oauth</groupId>
            <artifactId>spring-security-oauth2-jose</artifactId>
            <version>${spring-security-oauth2.version}</version>
        </dependency>
         <dependency>
            <groupId>org.springframework.security</groupId>
            <artifactId>spring-security-crypto</artifactId>
        </dependency>
         <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-thymeleaf</artifactId>
        </dependency>
         ```

         客户端配置如下：

         ```java
         @Configuration
         @EnableWebSecurity // Enable security auto-configuration
         public class ClientAppConfig extends WebSecurityConfigurerAdapter {

             /**
              * Adds the oauth2Login() entry point which allows users to authenticate via third party providers such as Google, Facebook, GitHub etc.
              */
             @Override
             protected void configure(HttpSecurity http) throws Exception {
                 http
                    .authorizeRequests()
                    .anyRequest().authenticated()
                    .and()
                    .oauth2Login()
                    .loginPage("/login")
                    .defaultSuccessUrl("/")
                    .failureHandler(new OAuth2AuthenticationFailureHandler());
             }

             /**
              * Customizes the logout behavior to send the user back to our homepage after logging out.
              */
             @Override
             public void configure(WebSecurity web) throws Exception {
                 super.configure(web);
                 web.ignoring().antMatchers("/logout");
             }

             /**
              * Configures the OAuth2 Client properties including the client ID, client secret, and the authorization URI.
              */
             @Bean
             public OAuth2ClientProperties oAuth2ClientProperties() {
                 OAuth2ClientProperties props = new OAuth2ClientProperties();
                 props.getProviders().put("github", createGithubClientProperties());
                 return props;
             }

             /**
              * Creates the Github specific client properties. These are obtained upon registering a new Github OAuth Application.
              */
             private ClientRegistration createGithubClientProperties() {
                 ClientRegistration registration = new ClientRegistration();
                 registration.setRegistrationId("github");
                 registration.setClientId("yourClientId");
                 registration.setClientSecret("yourClientSecret");
                 registration.setRedirectUri("{baseUrl}");
                 registration.setClientName("GitHub");
                 registration.setAuthorizationUri("https://github.com/login/oauth/authorize");
                 registration.setTokenUri("https://github.com/login/oauth/access_token");
                 registration.getUserAuthoritiesConverter().setType(LinkedHashMap.class);
                 HashMap<String, String> scopes = new LinkedHashMap<>();
                 scopes.put("read:user", "Read user information.");
                 registration.setScope(scopes);
                 return registration;
             }

             /**
              * Initializes the OAuth2ClientContext so that we can store the access token in the session between requests.
              */
             @Bean
             public OAuth2ClientContextFactory oAuth2ClientContextFactory() {
                 return new DefaultOAuth2ClientContextFactory();
             }

             /**
              * Configures the OAuth2UserService which populates the current principal with the user details returned by the user info endpoint. We need to provide the appropriate implementation to fetch the user details from the appropriate provider.
              */
             @Bean
             public OAuth2UserService<OidcUserRequest, OidcUser> oidcUserService() {
                 return new CustomOidcUserService();
             }

         }
         ```

         配置文件 oauth2-client.properties 文件中包含客户端的配置信息，包括客户端 ID、客户端密钥、授权 URI、TOKEN URI 和重定向 URI 。这里使用的是内存存储，实际生产环境建议使用数据库存储。

         配置文件里还有一个 createGithubClientProperties 方法，用于创建 Github 客户端的配置信息。

         通过调用 oAuth2ClientContextFactory Bean ，初始化 OAuth2ClientContext ，以便在多个请求间持久化 access token 。

         OAuth2UserService 用于从 OpenID Connect Provider (OP) 获取用户信息，并将其映射到 Principal 对象。我们需要提供相应的实现，来从 Github OP 获取用户信息。

         接下来，我们编写一个控制器，用于处理 OAuth2 流程，包括登录、登出、授权码、访问令牌等。

         ```java
         @Controller
         public class LoginController {

             private final OAuth2UserService<OidcUserRequest, OidcUser> userService;

             private final OAuth2RestOperations restTemplate;

             @Autowired
             public LoginController(OAuth2UserService<OidcUserRequest, OidcUser> userService,
                                   OAuth2RestOperations restTemplate) {
                 this.userService = userService;
                 this.restTemplate = restTemplate;
             }

             /**
              * Displays the login form.
              */
             @GetMapping("/login")
             public String showLoginForm(@RequestParam(value="error", required=false) String error,
                                        Model model) {
                 model.addAttribute("error", error);
                 return "login";
             }

             /**
              * Processes the login form submission.
              */
             @PostMapping("/login")
             public ResponseEntity<?> handleLoginSubmit(@RequestParam("provider") String provider,
                                                      RedirectAttributes attributes) {
                 Authentication managerAuth = SecurityContextHolder.getContext().getAuthentication();
                 String callbackUrl = UriComponentsBuilder.fromHttpUrl(ServletUriComponentsBuilder.fromCurrentRequest().toUriString())
                                                        .queryParam("code", "{code}")
                                                        .buildAndExpand()
                                                        .encode()
                                                        .toUriString();
                 AuthorizationRequest authorizationRequest = AuthorizationRequest
                       .builder("github")
                       .state(new RandomValueStateGenerator().generateState())
                       .redirectUri(callbackUrl)
                       .build();
                 OAuth2AuthenticationToken authResult = authorize(provider, authorizationRequest, managerAuth);
                 addAuthentication(authResult);
                 return ResponseEntity.ok("Login success! You are now authenticated.")
                                    .header(HttpHeaders.LOCATION, "/");
             }

             /**
              * Performs the OAuth2 authorization flow.
              */
             private OAuth2AuthenticationToken authorize(String provider,
                                                        AuthorizationRequest authorizationRequest,
                                                        Authentication managerAuth) {
                 String baseUrl = ServletUriComponentsBuilder.fromCurrentContextPath().build().toString();
                 OAuth2AccessToken existingToken = getExistingToken(managerAuth);
                 if (existingToken == null || isExpired(existingToken)) {
                     LOGGER.info("Fetching fresh access token from provider {}", provider);
                     OAuth2AuthorizeRequest redirectRequest = OAuth2AuthorizeRequest
                                        .withConnection(createOAuth2ConnectionFactory(provider)).build(authorizationRequest);
                     HttpHeaders headers = new HttpHeaders();
                     headers.addAll(redirectRequest.getHeaders());
                     String url = redirectRequest.getRedirectUri();
                     RestTemplate restTemplate = new RestTemplate();
                     ResponseEntity<Void> response = restTemplate.exchange(url, HttpMethod.GET, new HttpEntity<>(headers), Void.class);
                     String location = response.getHeaders().getLocation().toString();
                     OAuth2AccessToken accessToken = extractAccessTokenFromQueryParameters(location);
                     return buildAuthenticationResult(provider, accessToken, managerAuth);
                 } else {
                     LOGGER.debug("Reusing existing access token {} granted at {}",
                                  existingToken.getTokenValue(), Instant.ofEpochSecond(existingToken.getIssuedAt()));
                     return buildAuthenticationResult(provider, existingToken, managerAuth);
                 }
             }

             /**
              * Extracts the access token from query parameters included in the redirection URL.
              */
             private OAuth2AccessToken extractAccessTokenFromQueryParameters(String location) {
                 String[] fragments = location.split("#|\\?");
                 StringBuilder sb = new StringBuilder(fragments[0]);
                 for (int i = 1; i < fragments.length; i++) {
                     String fragment = fragments[i];
                     if (fragment.startsWith("access_token=") || fragment.startsWith("&access_token=")) {
                         String[] params = fragment.split("&");
                         for (String param : params) {
                             if (param.startsWith("access_token=")) {
                                 sb.append("#").append(param);
                             } else {
                                 sb.append("&").append(param);
                             }
                         }
                     } else {
                         sb.append("#").append(fragment);
                     }
                 }
                 MultiValueMap<String, String> queryParams = QueryStringDecoder.decode(sb.toString());
                 String accessTokenValue = queryParams.getFirst("access_token");
                 return new OAuth2AccessToken(OAuth2AccessToken.TokenType.BEARER,
                                              accessTokenValue,
                                              "",
                                              null,
                                              null,
                                              0L,
                                              LocalDateTime.now().plusDays(1).atZone(ZoneId.systemDefault()).toInstant().getEpochSecond());
             }

             /**
              * Authenticates the user using the provided access token.
              */
             private OAuth2AuthenticationToken buildAuthenticationResult(String provider,
                                                                        OAuth2AccessToken accessToken,
                                                                        Authentication managerAuth) {
                 String subject = restTemplate.getForObject("https://api.github.com/user", String.class);
                 Collection<? extends GrantedAuthority> authorities = AuthorityUtils.commaSeparatedStringToAuthorityList("ROLE_USER");
                 Jwt jwt = new Jwt(accessToken.getTokenValue(), null, accessToken.getExpiresAt().toEpochMilli(),
                                 Collections.singletonMap(authorities.iterator().next().getAuthority(), true));
                 OidcUser user = userService.loadUser(OidcUserRequest.withOidcUser(new DefaultOidcUser(authorities, jwt)));
                 user.setAccessToken(accessToken);
                 user.setIdToken(null);
                 return new OAuth2AuthenticationToken(user, authorities, provider, accessToken.getRefreshToken());
             }

             /**
              * Obtains the access token from the authentication object.
              */
             private OAuth2AccessToken getExistingToken(Authentication managerAuth) {
                 if (managerAuth!= null && managerAuth.isAuthenticated()) {
                     String tokenValue = managerAuth.getName();
                     OAuth2AccessToken accessToken = new OAuth2AccessToken(OAuth2AccessToken.TokenType.BEARER,
                                                                          tokenValue,
                                                                          "",
                                                                          null,
                                                                          null,
                                                                          0L,
                                                                          System.currentTimeMillis() + 1000*60*60);
                     return accessToken;
                 } else {
                     return null;
                 }
             }

             /**
              * Checks whether the access token has expired.
              */
             private boolean isExpired(OAuth2AccessToken accessToken) {
                 return Instant.ofEpochSecond(accessToken.getExpiresAt()).isBefore(Instant.now());
             }

             /**
              * Adds the newly created authentication object to the security context holder.
              */
             private void addAuthentication(OAuth2AuthenticationToken result) {
                 SecurityContextHolder.getContext().setAuthentication(result);
             }

             /**
              * Creates the OAuth2 connection factory used to perform the authorization flow with the specified provider.
              */
             private OAuth2ConnectionFactory<OidcUserRequest> createOAuth2ConnectionFactory(String provider) {
                 ClientRegistration clientRegistration = oAuth2ClientProperties().getProviders().get(provider);
                 OAuth2AuthorizedClientService authorizedClientService = new InMemoryOAuth2AuthorizedClientService(oAuth2ClientContextFactory());
                 OAuth2AuthorizationRequestResolver resolver = new DefaultOAuth2AuthorizationRequestResolver(clientRegistrationRepository(), authorizedClientService);
                 OAuth2AuthorizedClientManager authorizedClientManager = new OAuth2AuthorizedClientManager(oAuth2ClientProperties(), clientRegistrationRepository(), authorizedClientService);
                 authorizedClientManager.setResolver(resolver);
                 OAuth2AuthorizedClientProvider authorizedClientProvider = new OAuth2AuthorizedClientProviderBuilder()
                        .authorizationCode().refreshToken().build();
                 authorizedClientManager.setAuthorizedClientProvider(authorizedClientProvider);
                 OAuth2ConnectionFactory<OidcUserRequest> connectionFactory = new OAuth2ConnectionFactory<>(clientRegistration,
                                                                                                        authorizedClientManager,
                                                                                                        restTemplate);
                 return connectionFactory;
             }

         }
         ```

         这里配置了三个控制器方法：

         1. showLoginForm - 显示登录表单；
         2. handleLoginSubmit - 处理登录表单提交；
         3. authorize - 执行 OAuth2 授权流程。

         在执行登录流程时，先生成一个随机的 state 值，并构造一个 AuthorizationRequest 对象，其中包含回调地址、state 参数等参数。

         然后使用 OAuth2ConnectionFactory 创建连接对象，并调用它的 authorize 方法，传入 AuthorizationRequest 对象作为参数。

         授权过程中，首先判断是否存在现有的 access token ，如果不存在或已经过期，则从 Github OP 获取新 access token。

         如果已存在 access token ，且没有过期，则跳过授权阶段，直接构建一个新的 OAuth2AuthenticationToken 对象。

         最后，将结果加入到 SecurityContextHolder ，并重定向到首页。

         当用户点击注销按钮时，直接清除 SecurityContextHolder 即可。