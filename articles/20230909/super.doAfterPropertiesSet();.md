
作者：禅与计算机程序设计艺术                    

# 1.简介
  

super.doAfterPropertiesSet()方法是在Spring框架中非常重要的一个方法。它的作用就是在Bean实例化之后做一些必要的初始化工作。
在Spring源码中，可以找到这个方法的实现类为AbstractAutowireCapableBeanFactory，以下是该类的注释说明:
```java
/**
 * Simple lifecycle container for a BeanFactory. Doesn't do anything
 * specific other than implementing the Lifecycle interface and calling the
 * lifecycle methods on each bean. As such, it can be used as a standalone
 * lifecycle manager (e.g. for testing purposes).
 * <p>Note that it doesn't call any of the callbacks defined by the
 * ApplicationContext interface (such as ApplicationListener or EnvironmentAware),
 * which is the purpose of Spring's {@link org.springframework.context.support.AbstractApplicationContext} superclass.
 * However, it is possible to register custom implementations of those callback
 * interfaces with this implementation. This allows for interception of
 * application context events before they reach listeners registered through
 * the ApplicationContextFacade interface.
 * @author <NAME>
 * @since 07.12.2003
 * @see #setBeanNameGenerator(BeanNameGenerator)
 * @see #getBeanPostProcessor()
 */
public abstract class AbstractAutowireCapableBeanFactory extends DefaultSingletonBeanRegistry
		implements ConfigurableBeanFactory, SingletonBeanRegistry {
	//...
	/**
	 * Calls the {@code afterPropertiesSet()} method on all singleton beans
	 * in this factory. Invoked at the end of the singleton pre-instantiation
	 * stage, afterPropertyPopulation() being called and before initBeans().
	 * @throws BeansException if a BeanFactoryPostProcessor fails
	 */
	protected void invokeInitMethods(String[] beanNames, String logPrefix)
			throws BeansException {

		for (String beanName : beanNames) {
			Object bean = getSingleton(beanName);

			if (!(bean instanceof InitializingBean)) {
				continue;
			}

			try {
				((InitializingBean) bean).afterPropertiesSet();
			}
			catch (Throwable ex) {
				throw new BeanCreationException(
						beanName + " failed to initialize", ex);
			}

			if (logger.isTraceEnabled()) {
				logger.trace(logPrefix + "Finished initializing of bean '" + beanName + "'");
			}
		}
	}

	/**
	 * Apply the given BeanPostProcessors to the given existing bean instance,
	 * invoking their respective post-processing methods. The returned bean instance may be a wrapped one,
	 * depending on whether there were any BeanPostProcessors applied to it.
	 * <p>The default implementation applies standard BeanPostProcessors. If you want to override this,
	 * make sure to also call the superclass's postProcessBeforeInitialization method as well.
	 * @param existingBean the existing bean instance
	 * @param name the name of the bean
	 * @return the original bean instance, or a wrapped one if there were BeanPostProcessors applied
	 * @throws BeansException in case of errors
	 * @see #postProcessBeforeInitialization(Object, String)
	 * @see javax.annotation.Priority
	 */
	@SuppressWarnings("deprecation")
	protected Object applyBeanPostProcessorsBeforeInitialization(Object existingBean, String name) throws BeansException {
		Object result = existingBean;
		for (BeanPostProcessor processor : getBeanPostProcessors()) {
			result = processor.postProcessBeforeInitialization(result, name);
			if (result == null) {
				break;
			}
		}
		return result;
	}

	/**
	 * Return the list of BeanPostProcessors that will get applied to beans created
	 * by this factory. Can be overridden in subclasses to return custom BeanPostProcessors.
	 * Will include both BeanPostProcessors specified as beans in the BeanFactory plus ones added
	 * dynamically using addBeanPostProcessor.
	 * <p>Note that this method should not instantiate or cache BeanPostProcessors, but rather provide
	 * them lazily based on the needs of individual bean instances. Instantiating processors ahead of time
	 * would mean that they need to be aware of already instantiated singletons etc, just to create some
	 * unnecessary overhead. Caching processors too early would not respect the Spring container's shutdown
	 * semantic properly: A cached processor might still be active during shutdown even though its owning
	 * BeanFactory has been destroyed already. By observing the initialization state of the BeanFactory
	 * itself we can ensure that only fully initialized processors are cached. Note that BeanPostProcessor
	 * objects returned from this method must not depend on the concrete bean type, to avoid eagerly applying
	 * processors to beans that shouldn't qualify.
	 * @return the List of BeanPostProcessors (in decreasing order of precedence)
	 * @see #addBeanPostProcessor
	 * @see #getBeanPostProcessor
	 */
	protected final List<BeanPostProcessor> getBeanPostProcessors() {
		List<BeanPostProcessor> bpps = new ArrayList<>(this.beanPostProcessors);
		bpps.sort(AnnotationAwareOrderComparator.INSTANCE);
		return Collections.unmodifiableList(bpps);
	}
	
	//...
}
```
从上面的注释可以看出，AbstractAutowireCapableBeanFactory是一个抽象类，它提供了初始化容器并管理单例Bean的功能。在这个类中，有一个invokeInitMethods方法用来执行所有单例Bean的afterPropertiesSet方法，这也是调用此方法的入口点。接下来我们会详细讨论此方法如何执行后置属性设置，以及Spring BeanFactory中BeanPostProcessor接口及其角色的关系。
# 2.核心概念
## 2.1 BeanFactory、ApplicationContext
BeanFactory 是 Spring 框架中的顶层设计模式之一，定义了Spring应用的基本功能集。BeanFactory负责实例化、定位、配置应用程序的组件及对象。应用程序一般通过BeanFactory获取所需的服务，BeanFactory的子接口ApplicationContext更进一步扩展了BeanFactory的功能，在BeanFactory基础上增加了面向现代化应用功能（如事件发布、资源访问、国际化等）的支持。

BeanFactory包含两个主要的成员变量：BeanFactory getParentBeanFactory() 返回BeanFactory的父容器，用于继承上下文；HierarchicalBeanFactory HierarchicalBeanFactory getParentBeanFactory() 返回BeanFactory的父容器，用于继承上下文。通常情况下，BeanFactory的父容器可以是另一个ConfigurableBeanFactory类型的BeanFactory。ApplicationContext 是BeanFactory的子接口，除了BeanFactory提供的所有功能外，ApplicationContext还提供以下额外的功能：

1. 事件发布 - ApplicationEventPublisher publishesApplicationEvents to inform interested parties about framework-level events。ApplicationContext允许向感兴趣的其他组件发布广播消息，例如ApplicationEvent。

2. 资源访问 - ResourcePatternResolver provides convenient access to resources in a resource location ，可以使用特定的模式查找多个资源文件。

3. 国际化 - MessageSource enables easy access to localized messages without needing to know the locale of the user。ApplicationContext自动根据用户的语言环境加载相应的MessageResourceBundle。

4. 测试支持 - LoadTimeWeaver supports integration testing with tools like JUnit。

ApplicationContext比BeanFactory多了对各种新特性的支持，比如事件发布、资源访问、国际化、测试支持等。ApplicationContext接口的实现类包括XmlWebApplicationContext、AnnotationConfigApplicationContext等，ApplicationContext的默认实现类为ClassPathXmlApplicationContext，通过读取配置文件实例化Spring容器，当配置文件中包含IoC依赖时，容器会实例化相应的Bean。

## 2.2 BeanDefinition
BeanDefinition 是Spring框架的核心类之一，它存储了关于Bean的配置信息。它由BeanFactory或ApplicationContext创建，并最终被注册到Spring容器中。其代表了一个原始的Bean实例，而Bean是指由BeanFactory或ApplicationContext创建出来的实例。每个Bean都有一个相应的BeanDefinition。

BeanDefinition 可以分为三类：

1. RootBeanDefinition：存储了最原始的Bean配置信息。

2. ChildBeanDefinition：子类，存储的是通过继承ParentBeanDefinition得到的Bean配置信息。

3. AnnotationBeanDefinition：存储了用注解形式定义的Bean配置信息。

RootBeanDefinition 和ChildBeanDefinition 通过 XML 文件进行配置或者通过AnnotationConfigApplicationContext 进行注解配置的方式创建。但是，对于没有使用XML方式的配置，Spring也提供了编程式的BeanDefinitionBuilder来创建BeanDefinition。

## 2.3 ApplicationContextAware
ApplicationContextAware 是一个回调接口，它在Spring容器实例化一个Bean时，会回调该接口的方法。ApplicationContextAware 有且只能有一个实现类，即BeanFactoryAware，ApplicationContextAware 的作用主要是将Spring容器内的数据注入到某些Bean中。