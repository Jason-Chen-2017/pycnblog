
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　Spring Boot 是由 Pivotal 开源的 Java 框架，旨在简化 Spring 的开发复杂度并加快应用程序的启动时间。它支持自动配置、特征注解、 starter 模块、内嵌服务器、外部配置等特性。本文将从 Spring Boot 的启动流程出发，带领读者了解 Spring Boot 的内部机制以及快速入门 Spring Boot 。
         # 2.Spring Boot 核心概念术语
         　　首先，我们需要搞清楚 Spring Boot 中几个重要的术语概念，比如：
         　　- Spring Context：Spring 上下文，它是 Spring 框架的基础，包括BeanFactory、ApplicationContext 等主要接口和实现类。它负责实例化、定位、配置及管理应用中的对象。
         　　- Bean：Bean 是 Spring 中的核心组件之一，是 Spring 容器中最基础、最简单的对象。Bean 的作用范围一般是全局的，可以被所有地方使用，其生命周期由 Spring 容器进行管理。
         　　- Configuration：Configuration 可以简单理解为 Bean 的配置文件，用于声明 Bean 对象。我们可以通过 @Configuration、@ComponentScan 和 @Import 注解定义配置类。
         　　- Auto Configuration：Auto Configuration 功能可以帮助我们自动地导入依赖项。Auto Configuration 通过一些规则（如特定注解）查找候选的配置类，然后通过 SpringFactoriesLoader 把它们导入到 Spring 容器中。
         　　- Starter：Starter 是 Spring Boot 中的一个重要概念，用于聚合多个依赖。例如，Spring Boot Starter Web 提供了构建 RESTful API 或 WEB 应用所需的所有依赖。
         　　- Initializr：Initializr 可以帮助我们快速创建基于 Spring Boot 的项目。
         　　总结来说，Spring Boot 在设计上具有以下几个特点：
         　　- Spring Boot 是一个轻量级框架，只提供应用框架和基本配置，并非一个完整的企业级解决方案。
         　　- Spring Boot 以约定优于配置的方式来简化 Spring 配置。
         　　- Spring Boot 有丰富的 Starter，可快速集成各种第三方库。
         　　- Spring Boot 支持多环境配置。
         # 3.Spring Boot 启动流程图
         　　Spring Boot 的启动流程分为以下几个阶段：
         　　![](https://img-blog.csdnimg.cn/20190606223718590.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjQyOTcxOA==,size_16,color_FFFFFF,t_70)
         # 4.Spring Boot 启动流程详解
         　　1. Spring Boot 执行静态初始化
           ```java
            /**
             * The entry point of the application.
             */
            public static void main(String[] args) {
                try {
                    // Prepare context
                    ConfigurableApplicationContext context =
                            new ClassPathXmlApplicationContext("applicationContext.xml");

                    // Run the application
                    MyApplication app = context.getBean(MyApplication.class);
                    app.run();

                    // Close the context
                    context.close();

                } catch (BeansException | IOException e) {
                    log.error("Failed to start application", e);
                    System.exit(-1);
                }

            }
           ```
           　　在 `main()` 方法中，Spring 创建了一个 `ClassPathXmlApplicationContext` 来加载 XML 文件中的 Bean 配置。`ConfigurableApplicationContext` 接口继承自 `ApplicationContext` 接口，提供了关闭上下文的方法。此时，Spring 只做了简单的校验，准备好了上下文，但具体的 Bean 初始化操作还没有进行。

         　　2. Spring 初始化 Bean Factory
           ```java
            /**
             * Create and initialize a {@link BeanFactory}.
             */
            protected DefaultListableBeanFactory createBeanFactory() {
                return new DefaultListableBeanFactory();
            }
           ```
           　　Spring 会创建一个 `DefaultListableBeanFactory`，用来保存 Bean 的注册信息。每个 Bean 都有一个唯一 ID，当请求某个 Bean 时，根据该 ID 查找对应的 Bean 对象。

         　　3. 扫描 Bean
           ```java
            /**
             * Scan for candidate components.
             */
            private Set<BeanDefinitionHolder> scan(String[] basePackages) {
                List<BeanDefinitionReaderUtils.BeanDefinitionResource> beanDefinitions =
                        new ArrayList<>();
                int count = loadBeanDefinitions(beanDefinitions, basePackages);
                if (count == 0) {
                    log.warn("No spring.factories entry found in META-INF/spring.factories. This may cause an error.");
                } else if (count > 1) {
                    throw new IllegalStateException("Only one configuration allowed per library, but " + count + " were found in META-INF/spring.factories: " + Arrays.toString(basePackages));
                }
                processBeanDefinitions(beanDefinitions);
                
                Set<BeanDefinitionHolder> holders = new LinkedHashSet<>(this.registry.getBeanDefinitionHolders());
                this.registry.clear();
                return holders;
            }
           ```
           　　Spring 会扫描所有的 Bean 配置文件（XML 和 YAML），并解析其中的 Bean 定义。其中，XML 配置文件会使用 `XmlBeanDefinitionReader` 解析，而 YAML 配置文件则会用 `YamlBeanDefinitionReader`。这里会收集符合条件的 Bean 定义，并保存到 `BeanDefinitionResource` 对象中。

         　　4. 注册 Bean Definition
           ```java
            /**
             * Register each given bean definition with the registry by name and aliases.
             */
            private void registerBeanDefinitions(Map<String, Object> config) {
                String[] names = config.keySet().toArray(new String[config.size()]);
                for (String name : names) {
                    if (!this.registry.containsBeanDefinition(name)) {
                        BeanDefinitionHolder holder = parseBeanDefinitionElement(name, config.get(name));
                        if (holder!= null) {
                            BeanDefinitionReaderUtils.registerBeanDefinition(holder, this.registry);
                        }
                    }
                }
            }
           ```
           　　遍历所有 Bean 的名称，如果该名称尚不存在于 Bean 注册表中，则调用 `parseBeanDefinitionElement()` 方法解析 Bean 定义元素，并注册到 Bean 注册表中。

         　　5. 根据 Bean Factory 生成 Bean Instance
           ```java
            /**
             * Instantiate a single bean instance for the given bean name.
             */
            @Nullable
            protected Object createBean(String beanName, RootBeanDefinition mbd, @Nullable Object[] args) {
                if (mbd.isSingleton()) {
                    Object sharedInstance = getSingleton(beanName);
                    if (sharedInstance!= null && args == null) {
                        return sharedInstance;
                    }
                }
                // Instantiate bean instance.
                BeanWrapper instanceWrapper = null;
                if (mbd.isPrototype()) {
                    instanceWrapper = new PrototypeBeanWrapper();
                } else {
                    instanceWrapper = new BeanWrapperImpl();
                }
                instanceWrapper.setWrappedInstance(doCreateBean(beanName, mbd, args));
                synchronized (mbd.singletonLock) {
                    if (mbd.isSingleton()) {
                        addSingleton(beanName, instanceWrapper.getWrappedInstance(), mbd);
                    }
                }
                return instanceWrapper.getWrappedInstance();
            }
           ```
           　　对于每一个 Bean Name，Spring 将检查它的 Bean 定义是否为单例模式，如果是单例模式，那么将尝试获取之前已经创建好的共享实例。如果存在已创建的共享实例，并且当前请求不带参数，则直接返回共享实例；否则，继续执行后续的实例化逻辑。

           　　6. 使用 Autowire 注入 Bean 属性值
           ```java
            /**
             * Populate the property values of the given bean instance using autowiring by type or name.
             */
            @SuppressWarnings("deprecation")
            private void populateBean(String beanName, Object bean, @Nullable RootBeanDefinition mbd) {
                if (mbd == null ||!mbd.hasPropertyValues()) {
                    return;
                }

                BeanWrapper bw = new BeanWrapperImpl(bean);
                MutablePropertyValues pvs = mbd.getPropertyValues();
                for (PropertyValue pv : pvs.getPropertyValueList()) {
                    PropertyDescriptor pd = bw.getPropertyDescriptor(pv.getName());
                    if (pd == null) {
                        if (logger.isDebugEnabled()) {
                            logger.debug("Ignored illegal property '" + pv.getName() + "' on bean class [" +
                                    bean.getClass().getSimpleName() + "]");
                        }
                        continue;
                    }
                    
                    Method writeMethod = pd.getWriteMethod();
                    if (writeMethod == null) {
                        if (logger.isDebugEnabled()) {
                            logger.debug("No setter defined for property '" + pv.getName() + "' on bean class [" +
                                    bean.getClass().getSimpleName() + "]");
                        }
                        continue;
                    }
                    
                    TypeConverter converter = getTypeConverter();
                    Object value = pv.getValue();
                    if (value instanceof TypedStringValue) {
                        value = ((TypedStringValue) value).getValue();
                    }
                    else if (value instanceof RuntimeBeanReference) {
                        String refName = ((RuntimeBeanReference) value).getBeanName();
                        value = getBean(refName);
                    }
                    else if (value instanceof RuntimeBeanNameReference) {
                        String refName = ((RuntimeBeanNameReference) value).getBeanName();
                        value = getBean(refName);
                    }
                    else if (value instanceof ResourceArrayValue) {
                        Resource[] resources = ((ResourceArrayValue) value).getResources();
                        value = Stream.of(resources).map(r -> {
                            InputStream is = r.getInputStream();
                            BufferedReader br = new BufferedReader(new InputStreamReader(is));
                            StringBuilder sb = new StringBuilder();
                            String line;
                            while ((line = br.readLine())!= null) {
                                sb.append(line).append("
");
                            }
                            br.close();
                            is.close();
                            return sb.toString();
                        }).collect(Collectors.toList());
                    }
                    else if (value instanceof TypedObjectValue) {
                        TypedObject typedObject = ((TypedObjectValue) value).getValue();
                        value = doConvertFromTypedObject(typedObject, converter);
                    }
                    
                    try {
                        ReflectionUtils.invokeMethod(writeMethod, bean, value);
                    }
                    catch (Throwable ex) {
                        throw new IllegalArgumentException("Could not set property '" + pv.getName() + "' of bean '" +
                                beanName + "'", ex);
                    }
                }
            }
           ```
           　　Autowire 功能允许 Spring 根据类型或名称注入 Bean 属性值。使用 Autowire 的 Bean 在初始化完成之后，才会真正完成属性值的注入工作。
          
         　　至此，Spring Boot 的整个启动流程就结束了。这里只是抛砖引玉，实际上还有很多细节需要去探究。下面，我将结合实际案例，给大家展示 Spring Boot 的启动过程更全面的认识。

         # 5.实战案例分析
         　　为了让读者更直观地了解 Spring Boot 的启动流程，我们先用一个实际案例来模拟 Spring Boot 的启动过程。这个案例就是 Spring Boot Starter Web，它提供了构建 RESTful API 或 WEB 应用所需的所有依赖。我们将学习一下 Spring Boot Starter Web 的启动过程，并对比 Spring Boot 的启动流程，看看两者有什么区别。
         　　首先，我们需要在 Maven 项目的 pom.xml 文件中添加 Spring Boot Starter Web 的依赖：
         
         ```xml
         <dependency>
             <groupId>org.springframework.boot</groupId>
             <artifactId>spring-boot-starter-web</artifactId>
         </dependency>
         ```
         
         然后，在 Application.java 文件中编写启动函数：
         
         ```java
         package com.example.demo;
         
         import org.springframework.boot.SpringApplication;
         import org.springframework.boot.autoconfigure.SpringBootApplication;
         
         @SpringBootApplication
         public class DemoApplication {
             public static void main(String[] args) {
                 SpringApplication.run(DemoApplication.class, args);
             }
         }
         ```
         
         当然，为了方便起见，我们也可以直接运行 main 函数，而不是通过 SpringApplication.run()。启动成功后，会打印如下信息：
         
         ```
         Starting application on localhost with PID 18120 (/Users/admin/Documents/workspace/demo/target/classes started by admin in /Users/admin/Documents/workspace/demo)
         No active profile set, falling back to default profiles: default
         Started DemoApplication in 0.83 seconds (JVM running for 1.367)
         ```
         
         此时，我们的 Spring Boot 服务已经启动起来了。接着，我们来分析一下 Spring Boot Starter Web 的启动流程。下面，我们逐步来看 Spring Boot Starter Web 的启动流程。
         
         ## 一、@SpringBootApplication 注解解析
         ```java
         @Target(ElementType.TYPE)
         @Retention(RetentionPolicy.RUNTIME)
         @Documented
         @Inherited
         @SpringBootConfiguration
         @EnableAutoConfiguration
         @ComponentScan(excludeFilters = {
                 @Filter(type = FilterType.CUSTOM, classes = TypeExcludeFilter.class),
                 @Filter(type = FilterType.CUSTOM, classes = AutoConfigurationExcludeFilter.class) })
         public @interface SpringBootApplication {
             
             /**
              * Exclude specific auto-configuration classes such that they will never be applied.
              * @return the classes to exclude
              */
             Class<?>[] exclude() default {};
             
             /**
              * Exclude specific auto-configurations defined via starters. For example, if you want to use the
              * HibernateJpaAutoConfiguration and have excluded the DataSourceAutoConfiguration, setting this property
              * to "org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration" would exclude it from
              * consideration when scanning for auto-configuration candidates.
              * @return the conditions used to exclude some auto-configurations via starters
              */
             Class<? extends Condition>[] excludeConfigurations() default {};
         }
         ```
         从上面的代码可以看到，@SpringBootApplication 注解是 Spring Boot 提供的一个注解。它主要做了以下几件事情：
         
         1. 指定了 @SpringBootConfiguration 注解作为元注解。
         2. 设置了 componentScans，指定自动发现组件所在的包路径。
         3. 设置了 @EnableAutoConfiguration 注解，启用 Spring Boot 的自动配置功能。
         
         因此，相当于我们在 Spring Boot 启动的时候，可以使用 @SpringBootApplication 注解代替配置文件。因此，可以忽略掉上面两个注解。下面，我们再看看 EnableAutoConfiguration 注解的源码。
         
         ## 二、EnableAutoConfiguration 注解解析
         ```java
         @Target({ElementType.TYPE})
         @Retention(RetentionPolicy.RUNTIME)
         @Documented
         @Inherited
         @AutoConfigurationPackage
         @Import({AutoConfigurationImportSelector.class})
         public @interface EnableAutoConfiguration {
             
             /**
              * Explicitly enable certain auto-configurations. If left empty, all available auto-configurations are considered.
              * @return the list of enabled auto-configurations
              */
             Class<?>[] value() default {};
             
             /**
              * Control whether auto-configurations should be activated or not. Can contain condition expressions. By default,
              * auto-configurations are activated based on their specified conditions or a combination thereof. To deactivate them,
              * specify them explicitly within this property or provide a custom {@link EnableAutoConfiguration#exclude()} method.
              * @return the conditions to activate the auto-configurations
              */
             String[] exclude() default {};
             
             /**
              * Control which auto-configurations are disabled. Cannot be combined with {@link EnableAutoConfiguration#exclude()}.
              * Disabled configurations can be re-enabled by removing their corresponding entries in this array or adding an appropriate
              * item to the {@code spring.autoconfigure.exclude} property.
              * @return the list of disabled auto-configurations
              */
             Class<?>[] excludeName() default {};
         }
         ```
         从上面的代码可以看到，EnableAutoConfiguration 注解也是 Spring Boot 提供的一个注解。它主要做了以下几件事情：
         
         1. 设置了 @AutoConfigurationPackage 注解，将主配置类所在的包设置为自动配置类搜索的根目录。
         2. 设置了 @Import(AutoConfigurationImportSelector.class) 注解，加载了 AutoConfigurationImportSelector。该类的作用是在启动过程中查找候选配置类，并按顺序导入这些配置类。
         
         AutoConfigurationImportSelector 的源码如下所示：
         
         ## 三、AutoConfigurationImportSelector 注解解析
         ```java
         public final class AutoConfigurationImportSelector implements DeferredImportSelector, BeanClassLoaderAware, ResourceLoaderAware, BeanFactoryAware {
             
             private ClassLoader beanClassLoader;
             
             private ResourcePatternResolver resourcePatternResolver;
             
             private BeanFactory beanFactory;
             
            ...
             
             @Override
             public void selectImports(DeferredImportSelectorHandler handler) {
                 AutoConfigurationMetadata metadata = AutoConfigurationMetadataLoader.loadMetadata(this.beanClassLoader);
                 
                 AnnotationAttributes attributes = attributesFor(getSpringFactoriesLoaderFactoryClass());
                 List<String> factories = attributes.getStringArrayList("value");
                 if (logger.isDebugEnabled()) {
                     logger.debug("Locating auto configuration files for spring.factories loader: " + factories);
                 }
                 loadFactoryNames(factories, handler);
                 
                 applyExcludes(metadata, attributes, handler);
                 
                 configureFactories(attributes, handler);
             }
             
             private void loadFactoryNames(List<String> factoryNames, DeferredImportSelectorHandler handler) {
                 for (String factoryName : factoryNames) {
                     try {
                         Class<?> factory = Class.forName(factoryName, false, this.beanClassLoader);
                         if (!DeferredImportSelector.class.isAssignableFrom(factory)) {
                             handleError("Auto-configuration factory class [" + factory.getName() +
                                     "] is not a subclass of " + DeferredImportSelector.class.getName());
                             continue;
                         }
                         
                         DeferredImportSelector deferredImportSelector =
                                 asUtility(factory.getDeclaredConstructor().newInstance());
                         deferredImportSelector.selectImports(handler);
                     }
                     catch (Throwable ex) {
                         handleError("Unable to instantiate auto-configuration factory class [" + factoryName + "]", ex);
                     }
                 }
             }
             
             private void applyExcludes(AutoConfigurationMetadata metadata,
                                       AnnotationAttributes attributes, DeferredImportSelectorHandler handler) {
                 List<String[]> excludes = attributes.getStringArrayList("exclude");
                 if (!CollectionUtils.isEmpty(excludes)) {
                     filter(metadata, excludes, handler);
                 }
             }
             
             private void filter(AutoConfigurationMetadata metadata, List<String[]> excludes, DeferredImportSelectorHandler handler) {
                 for (String[] exclude : excludes) {
                     String className = exclude[0];
                     boolean allNested = Boolean.valueOf(exclude[1]);
                     
                     BeanDefinitionBuilder builder = BeanDefinitionBuilder.rootBeanDefinition(FilteredAutoConfiguration.class);
                     builder.addPropertyValue("filterClassName", className);
                     builder.addPropertyValue("includeAllNestedConditions", allNested);
                     
                     String beanName = GENERATED_AUTOCONFIGURATIONS_BASE_PACKAGE + "." + GENERATED_AUTOCONFIGURATIONS_SUFFIX + "_exclude_" + className.replace(".", "_");
                     handler.process(new BeanDefiningAnnotationBeanDefinition(builder.getBeanDefinition(), null, beanName));
                 }
             }
             
             private void configureFactories(AnnotationAttributes attributes, DeferredImportSelectorHandler handler) {
                 Map<String, Object> conditionMap = new HashMap<>();
                 for (String key : attributes.getAnnotation(ConditionalOnClass.class::getName).stringArray("name")) {
                     conditionMap.put(key, Boolean.TRUE);
                 }
                 for (String key : attributes.getAnnotation(ConditionalOnMissingClass.class::getName).stringArray("name")) {
                     conditionMap.put(key, Boolean.FALSE);
                 }
                 
                 List<ConditionEvaluationReport> evaluationReports =
                         ConditionEvaluationReport.get(this.beanFactory).getConditionAndReports(conditionMap);
                 
                 if (!evaluationReports.isEmpty()) {
                     for (ConditionEvaluationReport report : evaluationReports) {
                         if (!report.getOutcomes().isEmpty()) {
                             ConditionEvaluationResult result = report.getOutcomes().iterator().next();
                             String[] classNames = StringUtils.commaDelimitedListToStringArray(result.getComponent());
                             
                             for (int i = 0; i < classNames.length; i++) {
                                 String className = classNames[i].trim();
                                 
                                 BeanDefinitionBuilder builder = BeanDefinitionBuilder.rootBeanDefinition(ConditionalOnClassCondition.class);
                                 builder.addPropertyValue("className", className);
                                 
                                 String beanName = GENERATED_CONDITIONAL_BEANS_BASE_PACKAGE + "." + GENERATED_CONDITIONAL_BEANS_SUFFIX
                                         + "_" + className.replaceAll("\\.", "_").toLowerCase();
                                 handler.process(new BeanDefiningAnnotationBeanDefinition(builder.getBeanDefinition(), null, beanName));
                             }
                         }
                     }
                 }
             }
         }
         ```
         从上面的代码可以看到，AutoConfigurationImportSelector 类是 Spring Boot 的核心类之一，它实现了 DeferredImportSelector 接口，因此它可以控制 Spring 如何导入候选配置类。下面，我们再看看 @ConditionalOnClass 注解的源码。
         
         ## 四、@ConditionalOnClass 注解解析
         ```java
         @Target({ElementType.TYPE, ElementType.METHOD})
         @Retention(RetentionPolicy.RUNTIME)
         @Documented
         @Conditional(OnClassCondition.class)
         public @interface ConditionalOnClass {
             
             /**
              * Specify the fully qualified names of the required classes. At least one of these classes must be present on the classpath.
              * @return the fully qualified names of the required classes
              */
             String[] name();
             
             /**
              * When true, no beans of the underlying technology are registered even if the specified classes are detected. Defaults to
              * {@code false}, meaning any detected classes will trigger auto-configuration. Use this flag for non-standard
              * technologies where being present simply implies usage rather than presence of a concrete implementation.
              * @return whether to skip registration of any detected classes
              */
             boolean assignable() default false;
         }
         ```
         从上面的代码可以看到，@ConditionalOnClass 注解是 Spring 提供的另一个注解。它可以控制 Spring 是否启用某个自动配置类。下面，我们再看看 OnClassCondition 注解的源码。
         
         ## 五、OnClassCondition 注解解析
         ```java
         public class OnClassCondition implements Condition {
             
             private static final Log LOGGER = LogFactory.getLog(OnClassCondition.class);
             
             private final Set<String> classNames;
             
             private final boolean assignable;
             
             public OnClassCondition(Set<String> classNames, boolean assignable) {
                 super();
                 this.classNames = classNames;
                 this.assignable = assignable;
             }
             
             @Override
             public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
                 ConditionMessage.Builder message = ConditionMessage.forCondition(OnClassCondition.class);
                 Environment environment = context.getEnvironment();
                 
                 for (String className : classNames) {
                     Class<?> clazz = getClass(environment, className);
                     
                     if (clazz!= null) {
                         if (this.assignable &&!context.getBeanFactory().getType(className).isAssignableFrom(clazz)) {
                             if (LOGGER.isDebugEnabled()) {
                                 LOGGER.debug(message.found("an incompatible class").items(clazz).describe());
                             }
                             return false;
                         }
                         if (LOGGER.isDebugEnabled()) {
                             LOGGER.debug(message.found("required class").items(clazz).describe());
                         }
                         return true;
                     }
                     
                     if (LOGGER.isInfoEnabled()) {
                         LOGGER.info(message.notFound("class").item(className).describe());
                     }
                 }
                 
                 if (LOGGER.isDebugEnabled()) {
                     LOGGER.debug(message.didNotFind("any of").items(classNames).describe());
                 }
                 return false;
             }
             
             private Class<?> getClass(Environment environment, String className) {
                 try {
                     return ClassUtils.forName(className, environment.getSystemClassLoader());
                 }
                 catch (Exception ex) {
                     return null;
                 }
             }
         }
         ```
         从上面的代码可以看到，OnClassCondition 类是 Spring 提供的 Condition 接口的一个实现类。它主要通过读取 Spring 配置环境和类路径，判断指定的类是否存在，如果存在则返回 true。

