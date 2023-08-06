
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　BeanFactory是一个接口，它定义了getBean()方法用于从BeanFactory中获取bean对象。ApplicationContext继承BeanFactory，并对其功能进行了扩展，使之支持更多的应用场景，比如说消息资源处理、事件传播等。因此BeanFactory和ApplicationContext在Spring框架中的作用并不相同。BeanFactory是Spring框架的基础设施，用来管理应用中的各个对象。它主要负责实例化、配置和定位 bean 对象。ApplicationContext是在BeanFactory的基础上构建的，除了BeanFactory提供的基本功能外，ApplicationContext还增加了以下几方面的功能：
         　　1）面向切面的编程（AOP）
         　　2）应用事件（Application Event）
         　　3）资源访问（Resource Accessing）
         　　4）国际化（i18n）
         　　5）载入多个配置文件（Multiple Configuration Files Loading）
         　　6）Web环境下多个ServletContext共享同一个BeanFactory（Shared Scope of the BeanFactory in a Web Environment）
         　　7）脚本语言（Scripting Language Support）
         　　ApplicationContext并不是BeanFactory的唯一实现类，除了BeanFactory外，还有FileSystemXmlApplicationContext、ClassPathXmlApplicationContext等。BeanFactory和ApplicationContext之间的差异，主要体现在以下几个方面：
           （1）生命周期管理：BeanFactory创建bean的实例后并不会管理它的生命周期，而ApplicationContext会管理bean的整个生命周期。也就是说，BeanFactory只是负责创建bean实例，ApplicationContext通过BeanFactory创建出来的bean才是具有完整生命周期的对象。BeanFactory可以通过配置文件或编码的方式告诉Spring容器如何去创建对象，但是ApplicationContext可以动态地更新配置，所以ApplicationContext更适合于实际项目开发。
           （2）IoC依赖注入：BeanFactory只关注Bean的实例化和配置，但是ApplicationContext除了可以管理Bean的实例化和配置外，还可以实现IoC依赖注入，即将一个对象中的属性值注入另一个对象中。BeanFactory只能通过反射调用无参数构造函数创建Bean对象，而ApplicationContext可以通过很多方式来注入依赖，比如BeanFactoryPostProcessor、BeanPostProcessor、BeanFactoryAware、ApplicationContextAware等。
           （3）上下文信息：ApplicationContext除了包括BeanFactory的所有功能外，还可以提供更多的应用上下文信息，比如容器中bean对象的名字、类型等，ApplicationContext中的getBean()方法就可以根据名称或者类型获取到指定的bean对象。BeanFactory没有这些上下文信息，所以无法根据bean的名字查找相应的对象。
         # 2.基本概念术语说明
         　　BeanFactory和ApplicationContext都属于Spring Framework的核心接口之一，它们共同的父类是BeanFactory。但两者又有本质上的不同。BeanFactory主要用于管理bean对象，ApplicationContext则在BeanFactory的基础上添加了更多特性。BeanFactory除了管理bean对象之外，还负责bean对象的生命周期的管理；ApplicationContext除了BeanFactory提供的基本功能外，还提供了许多额外的功能，比如面向切面的编程（AOP）、资源访问（Resource Accessing）等。为了方便理解，下面先对BeanFactory和ApplicationContext进行简要的介绍。
         　　BeanFactory：BeanFactory是Spring中最基本的IOC容器，它只有两个接口：BeanFactory接口和HierarchicalBeanFactory接口。BeanFactory接口是BeanFactory的主要接口，提供了获取bean的方法，如getBean(name)等；HierarchicalBeanFactory接口继承BeanFactory接口，提供了层次性的BeanFactory，允许BeanFactory成为另一个BeanFactory的委托。BeanFactory用于管理bean的实例化、配置及装配，其生命周期由Spring负责管理。
         　　ApplicationContext：ApplicationContext继承BeanFactory接口，提供了其他更多的功能，比如面向切面的编程、资源访问、事件传播、国际化等。ApplicationContext除了提供BeanFactory的基本功能外，还提供其他一些重要功能，比如支持多种格式的配置文件加载、Web环境下的共享BeanFactory、事件监听机制等。ApplicationContext用于在Spring IOC容器中存储bean，并对bean进行管理，其生命周期也由Spring管理。
         # 3.BeanFactory概述
         　　BeanFactory是Spring框架的核心接口之一，它是一个工厂模式的结构，可以生成各种类型的bean。BeanFactory的主要用途就是管理bean的实例化、配置及装配，其生命周期由Spring框架负责管理。BeanFactory有两种基本角色：BeanFactoryPostProcessor和BeanFactory。BeanFactoryPostProcessor是在BeanFactory初始化之后立刻被执行的特殊BeanFactory，它可以对bean进行加工，比如设置BeanFactory中的数据源等。BeanFactory接口提供了获取bean的方法，如getBean(name)。BeanFactory提供了三种类型的bean定义：普通bean、单例bean、集合bean。BeanFactory提供了控制bean生命周期的方法，如getBean()方法返回之前先触发其生命周期回调方法。BeanFactory的默认行为是单例模式，即一个类的bean只会被实例化一次。如果需要多例模式的话，可以在bean的定义中配置scope属性为“prototype”，这样每次getBean的时候都会创建一个新的bean。
         # 4.ApplicationContext概述
         　　ApplicationContext也是Spring框架的核心接口之一，它继承BeanFactory接口并添加了许多特性，比如资源访问、面向切面的编程、事件传播等。ApplicationContext主要用于替换BeanFactory作为Spring IoC容器的中心组件，ApplicationContext比BeanFactory更复杂，因为ApplicationContext除了管理bean外，还支持其他的特性。ApplicationContext支持多种格式的配置文件加载，包括XML、Properties文件、YAML配置文件等。ApplicationContext提供以下特性：
           （1）getBean(name): 根据bean的名称获取bean对象。
           （2）registerShutdownHook(): 在JVM关闭时自动释放ApplicationContext。
           （3）资源访问: 可以从ApplicationContext中获得各种资源，比如网络连接、数据库链接、邮件服务、文件系统资源等。
           （4）国际化(i18n): 支持不同区域的资源访问。
           （5）基于注解的依赖注入: 通过注释来完成依赖关系的注入。
           （6）事件通知机制: 消息、事件、错误、异常等通知可以进行监听和响应。
           （7）Web应用: 提供Web应用程序上下文，包括ServletContext、HttpServletRequest、HttpServletResponse等。
           （8）多文件配置: 支持加载多个配置文件，可重用相同配置。
         # 5.ApplicationContext和BeanFactory的区别
         　　BeanFactory和ApplicationContext之间存在很大的区别。BeanFactory是一个轻量级的IOC容器，它仅仅用来管理bean对象，而且所有的bean都是单例模式。BeanFactory并不会对bean的生命周期进行管理，它只提供简单轻量级的API。ApplicationContext相对于BeanFactory来说，功能更强大，ApplicationContext还提供支持面向切面的编程（AOP）、资源访问（Resource Accessing）、事件传播等功能。一般情况下，建议优先使用ApplicationContext，因为ApplicationContext提供的功能更加强大，在实际项目开发中有很好的实践意义。
         　　总结来说，BeanFactory是Spring框架中较为简单的IOC容器，其设计目的是用于管理bean的实例化、配置及装配，但BeanFactory只提供最基本的bean管理功能。ApplicationContext是BeanFactory的高级版本，在BeanFactory的基础上添加了许多特性，比如支持多种格式的配置文件加载、事件监听机制、资源访问、国际化等，而且ApplicationContext更适合于实际项目开发。