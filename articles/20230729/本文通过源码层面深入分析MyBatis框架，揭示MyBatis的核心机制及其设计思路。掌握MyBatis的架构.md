
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 MyBatis 是apache的一个开源项目，2006年就已经进入了它的官方维基，截止2020年9月1日，它已经成为最流行的ORM框架之一，是一个优秀的持久化框架。相比Hibernate、TopLink、MyBatis等框架，MyBatis更加简单易用，学习成本低，操作数据库也很方便。 MyBatis可以实现定制化SQL、存储过程以及高级映射，灵活方便。它支持多种关系数据库、对各种数据库操作提供统一的接口。所以说 MyBatis 是一个十分好的 ORM 框架。
          在本文中，我们将会从源码角度详细地剖析 MyBatis 的底层实现机制，并尝试给出 MyBatis 中最关键的设计思想。
          # 2.基本概念术语说明
          1. MyBatis 中重要的四个对象:SqlSessionFactoryBuilder、SqlSessionFactory、MapperRegistry、MappedStatement
          2. SqlSessionFactoryBuilder(简称 SFB)：用来创建SqlSessionFactory对象的，可以加载mybatis的配置文件（即sqlMapConfig.xml）并解析生成SqlSessionFacotry
          3. SqlSessionFactory：SqlSession的工厂类，用来产生SqlSession对象。一个应用对应一个SqlSessionFactory，线程不安全，所以需要外部同步访问或请求作用域内的线程来获得SqlSession对象。
          4. MapperRegistry：mapper注册器，用来保存mapper接口信息。每个mapper文件对应一个MapperRegistry，该类负责管理注册的 mapper。
          5. MappedStatement：SQL语句映射类，用来保存一条具体执行的SQL语句相关的信息，如SQL语句类型（SELECT/INSERT/UPDATE/DELETE等），SQL语句，参数映射，结果映射，缓存配置等。 
          # 3. 核心算法原理与操作步骤
          ## （一）SqlSessionFactoryBuilder 的构造函数
          首先，来看一下 SqlSessionFactoryBuilder 类的构造函数：

          ```java
          public SqlSessionFactoryBuilder() {
              // 配置XML文件加载器
              this(new XMLConfigBuilder());
          }
  
          private SqlSessionFactoryBuilder(XMLConfigBuilder xmlConfigBuilder) {
              // 初始化配置信息读取器
              config = new Configuration();
              xmlConfigBuilderDelegate = new XMLConfigBuilderDelegate(config, xmlConfigBuilder);
          }
          ```
          可以看到，SqlSessionFactoryBuilder 通过构造方法注入了一个 XMLConfigBuilder 对象，然后初始化自身的 Configuration 对象和 XMLConfigBuilderDelegate 对象。Configuration 对象是 MyBatis 中用来保存 MyBatis 配置信息的对象。
          XMLConfigBuilder 是 MyBatis 中的一个配置文件解析类，负责解析 MyBatis 配置文件的逻辑。通过构造方法注入了一个 org.w3c.dom.Document 对象，从而完成对mybatis的配置文件解析工作。
          从构造方法我们可以看出来，SqlSessionFactoryBuilder 使用 XMLConfigBuilder 来进行 MyBatis 配置文件的解析工作。
          ## （二）XMLConfigBuilder 和 XMLConfigBuilderDelegate 的构造函数
          下面再来看一下 XMLConfigBuilder 和 XMLConfigBuilderDelegate 的构造函数：
          
          ```java
          public XMLConfigBuilder() {}
  
          /**
           * Constructor that accepts an optional {@link Document} to parse instead of a file name or stream.
           */
          public XMLConfigBuilder(Document doc) {
              this.document = doc;
          }
  
          public XMLConfigBuilderDelegate(Configuration configuration, XMLConfigBuilder xmlConfigBuilder) {
              super(configuration, xmlConfigBuilder);
              if (this.document == null && configuration.getVariables() == null) {
                  throw new BuilderException("No document specified");
              }
          }
          ```
          可以看到，XMLConfigBuilder 和 XMLConfigBuilderDelegate 的构造函数都没有什么特别的地方，都是空参构造方法，仅仅对一些变量进行赋值。其中 XMLConfigBuilder 通过构造方法传入了一个 Document 对象，表示一个 DOM 树结构；XMLConfigBuilderDelegate 会先调用父类的构造方法，然后检查是否存在全局变量，如果不存在则抛出异常。
          ## （三）XMLConfigBuilder 解析配置文件的方法
          接下来我们来看一下 XMLConfigBuilder 的 parse 方法：
          
          ```java
          public void parse() throws Exception {
              if (!configuration.isLoaded()) {
                  loadConfiguration();
              } else {
                  log.debug("The configuration was already loaded.");
              }
          }
  
          private void loadConfiguration() throws Exception {
              if (document == null) {
                  inputSource = Resources.getResourceAsInputSource(configuration.getConfigFile());
              } else {
                  inputSource = new DOMInputSource(document);
              }
  
              XMLMapperEntityResolver entityResolver = new XMLMapperEntityResolver();
              entityResolver.setConfiguration(configuration);
              entityResolver.setEntityResolver(new EntityResolver() {
                  @Override
                  public InputSource resolveEntity(String publicId, String systemId) throws SAXException, IOException {
                      return null;
                  }
              });
              builder = DocumentBuilderFactory.newInstance().newDocumentBuilder();
              builder.setEntityResolver(entityResolver);
              document = builder.parse(inputSource);
  
              configuration.addLoadedFile(inputSource.getSystemId());
              bindNamespaces(document.getDocumentElement());
              parseConfiguration(document.getDocumentElement());
              buildTypeAliases();
              buildPlugins();
              buildObjectWrapperFactory();
              checkDaoDefinitions();
              applyDefaultNamespace(document.getDocumentElement());
          }
          ```
          这个方法主要完成了以下几个步骤：
          1. 判断是否已经加载过 MyBatis 配置文件，如果已经加载过则直接跳到第 7 步，否则进入第 2 步。
          2. 设置 XML 文件输入源。如果 XMLConfigBuilder 构造方法中没有传入 Document 对象，那么就会通过 Resources.getResourceAsInputSource 方法设置输入源为 MyBatis 配置文件，否则输入源为 Document 对象。
          3. 创建 DocumentBuilder 对象，设置 EntityResolver 对象来处理实体引用。
          4. 调用 DocumentBuilder 的 parse 方法解析配置文件，并把解析出的 Document 对象保存到成员变量中。
          5. 如果配置文件中定义了命名空间，则绑定命名空间。
          6. 解析配置文件，包括处理 properties 标签和 settings 标签，解析 mapper 标签，构建 MappedStatement 对象。
          7. 根据全局配置构建 TypeAlias 对象。
          8. 检查 DAO 对象定义。
          ## （四）SqlSessionFactoryBuilder 的 build 方法
          现在我们知道了 SqlSessionFactoryBuilder 的构造函数如何解析 MyBatis 配置文件，接下来我们看一下 SqlSessionFactoryBuilder 的 build 方法：
          
          ```java
          public SqlSessionFactory build(InputStream inputStream) {
              try {
                  XMLConfigBuilder parser = new XMLConfigBuilder(inputStream);
                  return build(parser.parse());
              } catch (Exception e) {
                  throw new IllegalArgumentException("Error building SqlSession.", e);
              } finally {
                  IOUtils.closeQuietly(inputStream);
              }
          }
  
          public SqlSessionFactory build(Reader reader) {
              try {
                  XMLConfigBuilder parser = new XMLConfigBuilder(reader);
                  return build(parser.parse());
              } catch (Exception e) {
                  throw new IllegalArgumentException("Error building SqlSession.", e);
              } finally {
                  IOUtils.closeQuietly(reader);
              }
          }
  
  
          public SqlSessionFactory build(URL url) {
              try {
                  InputStream inputStream = Resources.getUrlAsStream(url);
                  return build(inputStream);
              } catch (IOException e) {
                  throw new IllegalArgumentException("Error building SqlSession from " + url, e);
              }
          }
  
          public SqlSessionFactory build(String resource) {
              URL url = Resources.getResourceURL(resource);
              return build(url);
          }
  
          protected SqlSessionFactory build(Configuration configuration) {
              ErrorContext.instance().resource(configuration.getResource()).activity("building session factory").object(getClass());
              final SqlSessionFactory sqlSessionFactory = new DefaultSqlSessionFactory(configuration);
              for (DeferredLoadEventListener listener : getDeferredLoadEventListeners()) {
                  sqlSessionFactory.getConfiguration().addEventListener(listener);
              }
              return sqlSessionFactory;
          }
          ```
          可以看到，SqlSessionFactoryBuilder 的 build 方法是用来根据不同的输入源（InputStream、Reader、URL、String）构建 SqlSessionFactory 对象，最终都会调用到 build 方法，只是最后返回的对象不同罢了。
          此外，还有一些重载方法也在这里。
          ## （五）SqlSessionFactory 的构造函数和 createSqlSession 方法
          当 SqlSessionFactory 被构建后，我们可以通过 createSqlSession 方法来获取 SqlSession 对象，因为 SqlSession 对象是线程不安全的，所以每次在一个线程里获取 SqlSession 时都需要通过 SqlSessionFactory 来获取。
          前面的部分已经看到 SqlSessionFactoryBuilder 的 build 方法，里面使用了 DefaultSqlSessionFactory 来构建 SqlSessionFactory 对象，那我们来看一下 DefaultSqlSessionFactory 的构造函数：
          
          ```java
          public DefaultSqlSessionFactory(Configuration config) {
              super(config);
              this.executor = new SimpleExecutor(config, this);
              this.typeHandlerRegistry = new TypeHandlerRegistry(config);
              this.scriptingLanguageRegistry = new ScriptingLanguageRegistry(config);
              initPlugins(config);
              initTransactionIsolationLevel();
              initDatabaseId();
              initSettings();
              afterPropertiesSet();
          }
          ```
          可以看到，DefaultSqlSessionFactory 只是简单地调用父类的构造函数，然后保存了一些数据，其中包括 executor、typeHandlerRegistry、scriptingLanguageRegistry、settings、plugins、databaseId。
          然后，还有一个 afterPropertiesSet 方法用来完成一些初始化工作。
          那至于 createSqlSession 方法做了什么？还是从 executor 中获取 SqlSession 对象吧。
          ## （六）SimpleExecutor 的构造函数和 openSession 方法
          我们之前分析过 executor 对象时，是一个 SimpleExecutor 对象。下面我们再来看一下 SimpleExecutor 的构造函数和 openSession 方法。
          ### 1. SimpleExecutor 的构造函数
          ```java
          public SimpleExecutor(Configuration configuration,
                              SqlSessionFactory sqlSessionFactory) {
              super(sqlSessionFactory);
              this.configuration = configuration;
              this.mappedStatements = new HashMap<String, MappedStatement>();
              this.cacheEnabled = configuration.isCacheEnabled();
              cache = new PerpetualCache(configuration);
              flushCacheRequired = true;
              lazyLoadingEnabled = configuration.isLazyLoadingEnabled();
              aggressiveLazyLoading = configuration.isAggressiveLazyLoading();
              multipleResultSetsEnabled = configuration.isMultipleResultSetsEnabled();
              useColumnLabel = configuration.isUseColumnLabel();
          }
          ```
          这个构造函数比较简单，就是保存了一些必要的参数。
          ### 2. SimpleExecutor 的 openSession 方法
          ```java
          public SqlSession openSession() {
              Transaction tx = new ManagedTransaction(localDataSource, localTransactionManager, autoCommit);
              return new DefaultSqlSession(configuration, executor, autoCommit, dirty, nested,
                                      closeConnection, transaction, connectionHolder, nestedOptions);
          }
          ```
          这个方法也是比较简单的，调用了 DefaultSqlSession 的构造函数，并把必要的参数保存到局部变量里。
          ## （七）DefaultSqlSession 的构造函数和 initialize 方法
          ```java
          public DefaultSqlSession(Configuration configuration, Executor executor, boolean autoCommit,
                                   boolean dirty, boolean nested, boolean closeConnection, Transaction transaction, ConnectionHolder holder,
                                   Map<String, Object> nestedOptions) {
              this.configuration = configuration;
              this.executor = executor;
              this.autoCommit = autoCommit;
              this.dirty = dirty;
              this.nested = nested;
              this.closeConnection = closeConnection;
              this.transaction = transaction;
              this.connectionHolder = holder;
              this.mappedStatements = new ConcurrentHashMap<>();
              this.closed = false;
              this.statementHandlers = new ArrayList<>();
              this.parameterHandler = new DefaultParameterHandler();
  
              buildStatementHandlers(configuration);
              initializeDefaults(configuration);
          }
          ```
          这个构造函数比前面很多构造函数要复杂些，它主要用来保存一些参数，然后根据 Configuration 对象来构建 statementHandlers。
          StatementHandler 实际上就是 MyBatis 执行 SQL 的核心类， MyBatis 对各种 SQL 都提供了对应的 StatementHandler 子类，比如 SelectStatementHandler 用于执行 SELECT 语句，InsertStatementHandler 用于执行 INSERT 语句等。
          parameterHandler 对象是一个 ParameterHandler 接口的实现类， MyBatis 将输入参数转换成 PreparedStatement 参数。
          ```java
          private void buildStatementHandlers(Configuration cfg) {
              Set<String> mappers = new HashSet<>();
              mappers.addAll(cfg.getMappers());
              ResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
              for (String mapperLocation : mappers) {
                  String pattern = mapperLocation + "/*" + FileUtil.findSQLStatementPattern();
                  try {
                      Resource[] resources = resolver.getResources(pattern);
                      for (Resource resource : resources) {
                          if (resource.isFile()) {
                              parseStatementNode(cfg, resource.getInputStream());
                          }
                      }
                  } catch (IOException e) {
                      throw new BuilderException("Failed to read resource with pattern '" + pattern + "'", e);
                  }
              }
          }
          
          private void parseStatementNode(Configuration cfg, InputStream inputStream) {
              XPathParser parser = new XPathParser(inputStream, cfg.getVariables());
              List<XNode> list = parser.evalNodes("/mapper/select|/mapper/insert|/mapper/update|/mapper/delete");
              for (XNode node : list) {
                  MappedStatement ms = buildStatementFromNode(cfg, node);
                  if (ms!= null) {
                      mappedStatements.put(ms.getId(), ms);
                  }
              }
          }
          
          private MappedStatement buildStatementFromNode(Configuration cfg, XNode context) {
              String id = context.getStringAttribute("id");
              if (id == null || id.isEmpty()) {
                  throw new BuilderException("statement requires an id attribute");
              }
              
              Class<?> parameterClass = resolveClass(context.getStringAttribute("parameterType"));
              String resultType = context.getStringAttribute("resultType");
              Class<?> resultClass = resolveClass(resultType);
              String statementType = context.getName();
              
              Executor executor = cfg.getDefaultExecutorType() == ExecutorType.BATCH? batchExecutor : simpleExecutor;
              StatementType stmtType = StatementType.valueOf(statementType.toUpperCase(Locale.ENGLISH));
              KeyGenerator keygen = cfg.getKeyGenerator(id);
              Integer fetchSize = context.getIntAttribute("fetchSize");
              Integer timeout = context.getIntAttribute("timeout");
              ParameterMap parameterMap = resolveParameterMap(cfg, context.getStringAttribute("parameterMap"), parameterClass);
              ResultMap resultMap = resolveResultMap(cfg, context.getStringAttribute("resultMap"), resultClass);
              CacheKey cacheKey = constructCacheKey(context, parameterMap, resultMap, fetchSize, timeout, boundSql);
              
              MappedStatement.Builder builder = new MappedStatement.Builder(cfg, id, parameterClass, resultClass, statementType);
              builder.keyGenerator(keygen).fetchSize(fetchSize).timeout(timeout).parameterMap(parameterMap).resultMap(resultMap)
                    .cache(resolveCache(cfg, context)).resultOrdered(context.getBooleanAttribute("resultOrdered", false))
                    .resultSetType(resolveResultSetType(context.getStringAttribute("resultSetType")))
                    .statementId(buildId(id, parameterClass, resultClass, cacheKey)).keyColumns(buildKeyColumns(cfg, id))
                    .executor(executor).language(resolveLanguageDriver(cfg, context));
              MappedStatement statement = builder.build();
              return statement;
          }
          ```
          上面的代码主要用来解析配置文件中的 statement 节点，并构建 MappedStatement 对象。mappedStatements 保存的是所有的 MappedStatement 对象。
          ### 1. initialize 方法
          ```java
          private void initializeDefaults(Configuration cfg) {
              defaultExecutorType = cfg.getDefaultExecutorType();
              defaultStatementTimeout = cfg.getDefaultStatementTimeout();
              defaultFetchSize = cfg.getDefaultFetchSize();
              defaultResultSetType = cfg.getDefaultResultSetType();
              typeAliasRegistry = cfg.getTypeAliasRegistry();
              objectFactory = cfg.getObjectFactory();
              plugins = cfg.getPlugins();
              reflectorFactory = cfg.getReflectorFactory();
              resultSets = new ArrayList<>();
          }
          ```
          这个方法是初始化默认值用的，主要包括设置默认执行器、默认超时时间、默认查询数量、默认结果集类型、TypeAlias 注册器、ObjectFactory 对象、插件列表、反射工厂和结果集合。
          ## （八）总结
          在这一节，我们分析了 MyBatis 的配置流程，包括配置文件的加载、解析和初始化过程，并构造了一个 SqlSessionFactory 对象。
          下一节，我们将会分析 MyBatis 的执行流程，具体是由 Executor 来处理和执行 SQL 语句，以及采用什么策略来缓存结果集。

