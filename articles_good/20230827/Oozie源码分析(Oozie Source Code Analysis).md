
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 Oozie是什么？
Apache Oozie是一个基于Hadoop的一个工作流调度系统，能够将MapReduce、Pig、Hive等不同形式的任务流程化。它具有以下功能：
- 支持复杂的工作流调度
- 可以自动化工作流执行
- 提供了监控和管理界面
- 使用简单，易于上手
其功能最主要的特点之一就是支持流程化的任务。所谓流程化，就是将一个长期的业务过程，分成若干个离散的任务节点组成，每个任务节点可以执行单个或多个MapReduce作业，串联起来就形成了一个完整的业务流程。这样做不仅可以实现业务流程的可视化和可靠性的保证，而且还可以提升整个业务的协同程度、优化资源利用率和节省运行成本。
## 1.2 为什么要分析Oozie源码？
其实在之前的一段时间里，我一直对Oozie这个项目很感兴趣。因为它的功能特性很强，且使用简单，非常符合我对Hadoop生态圈的理解。另外，随着Hadoop的版本迭代，它的源代码也逐渐完善，开发者会不断向该项目提交PR来完善它。因此，当时为了研究学习它的源代码，觉得无从下手，想着只能去阅读开源项目的代码。
不过后来，我的认识慢慢发生变化，越来越多的人开始关注Apache Hadoop生态圈，包括Hadoop项目本身，一些周边产品，如Hive、Spark等等。相比于Hadoop项目，Oozie更像是一个开源项目，它的源代码不是很多人能够看懂的，而是需要通过反编译工具才能看懂。于是乎，作为一个Hadoop爱好者，我开始着迷于Oozie源代码的魅力，并且开始寻找机会分析它的源代码。
## 2.核心组件及其功能解析
Oozie共由四大模块构成：
### (1) Coordinator服务
Coordinator是用来管理工作流实例的，它负责接收客户端提交的工作流请求，并根据工作流定义创建相应的工作流实例。它负责启动workflow action，监控action执行状态，并记录历史信息。Coordinator服务的主要功能如下：
- 接收工作流实例请求
- 执行工作流实例的初始化（设置参数）
- 检查工作流实例是否有环路依赖关系，防止死循环
- 根据工作流实例的依赖关系构建任务图（DAG图），并生成待执行action队列
- 执行action（即job任务）
- 将action执行结果上传至HDFS或者数据库中，并进行持久化存储
- 判断工作流实例是否成功结束
- 将工作流实例的状态信息发送给客户端
Coordinator服务除了负责管理工作流实例外，还可以提供HTTP接口用于外部系统调用。例如，可以通过调用Oozie REST API向Oozie服务器提交工作流实例请求，也可以通过监控页面获取工作流实例的执行进度和详细日志。
### (2) Workflow服务
Workflow是Oozie的核心模块，它是负责定义和执行工作流的。它处理XML配置文件中的工作流定义，并根据配置生成对应的workflow图，生成待执行action队列，同时也负责执行action。Workflow服务的主要功能如下：
- 从HDFS读取workflow定义配置文件
- 生成workflow图
- 根据workflow定义配置生成待执行action队列
- 执行action（即job任务）
- 将action执行结果上传至HDFS或者数据库中，并进行持久化存储
- 获取所有action的执行结果并整合成最终的工作流实例结果
- 通过不同的Action类型，Workflow服务可以调用不同类型的作业执行器，例如，mapreduce、pig等。
- 在Workflow服务内部，存在两个进程池，分别用于执行MapReduce和Pig任务。
### (3) JobType服务
JobType是Workflow服务的子模块，它负责管理action的执行器，如mapreduce、hive等。JobType服务主要功能如下：
- 从xml配置文件读取作业类型属性，例如，作业类型名称、输入目录、输出目录、作业主类名、其他配置项等
- 根据作业类型配置生成相应的作业执行器，如mapreduce、hive等
- 执行作业类型指定的作业（job任务）
- 返回作业的执行状态和结果
### (4) DB存储模块
DB存储模块是Oozie的另一个重要子模块，它负责将工作流实例信息、作业执行日志、作业类型配置、工作流定义配置等保存到数据库中。它提供了统一的接口，使得Oozie的各个组件都可以使用统一的存储方式。
Oozie服务的组件架构如上图所示。
## 3.算法与数据结构
Oozie源代码采用Java语言编写，基本结构比较简单，主要涉及如下几个方面：
- CoordinatorServer类：启动Coordinator服务；
- CoordinatorEngine类：封装了Coordinator服务的各个功能；
- WorkflowAppService类：启动Workflow服务；
- WorkflowServices类：封装了Workflow服务的各个功能；
- JobSubmission类的子类：负责提交不同类型的作业；
- ActionExecutor类：执行action（即job任务）。
其中，CoordinatorEngine和WorkflowServices类中都有非常丰富的数据结构和算法。接下来，我们将逐一解析它们。
### 3.1 CoordinatorEngine类
CoordinatorEngine类是Coordinator服务的核心类，它处理所有的工作流请求。它的主要功能函数如下：
#### a) submitRequest()
```java
public String submitRequest(@Context UriInfo uriInfo,
        @HeaderParam("oozie.remote.user.name") String userName, CoordinatorJobBean coordJob) throws JMSException,
            OozieException {
   ... //校验输入参数

    try {
        addLog(coordJob.getJobId(), "Coord job submission");

        boolean isStart = coordJob.getStatus() == CoordinatorJob.Status.PREMATERMINATED;
        if (!isStart &&!isConcurrencyAllowed(userName)) {
            throw new OozieException("User [" + userName
                    + "] is not authorized to start another coordinator run concurrently for workflow: ["
                    + coordJob.getAppName() + "/" + coordJob.getAppPath() + "]");
        }

        coordJob.setCreatedTime(new Date());
        coordJob.setUser(userName);
        setCwd(coordJob.getConf().getVar(WorkflowAppService.COORD_WORKFLOW_DIR));
        validateCoordExternalId(coordJob);
        if (coordJob.getEndTime()!= null
                && System.currentTimeMillis() > coordJob.getEndTime().getTime()) {
            coordJob.setStatus(CoordinatorJob.Status.KILLED);
            return coordJob.getJobId();
        }

        Path appPath = new Path(coordJob.getAppPath());
        FSUtils.mkdirs(appPath, getAppFileSystem(), true);

        WorkflowJobBean wfJob = loadWFConfig(coordJob.getUser(), coordJob.getConf(), coordJob.getBundleId());

        register(coordJob.getJobId(), coordJob);

        // Create and enqueue all the actions for this coordinator instance based on dependencies
        List<CoordinatorAction> pendingActionsList = createCoordJobs(wfJob, coordJob.getUserName());

        // Kick off pending actions by submitting them to the JobTracker or by executing them directly
        queuePending(pendingActionsList, userName, coordJob.getConsoleUrl(), coordJob.isForced());

        addLog(coordJob.getJobId(), "Coord job created with " + coordJob.getRemainingactions() + " remaining actions.");

        // Inform XLog service of the creation of this Coordinator
        NotificationService.getInstance().generateEvent(XLogService.EVENT_TYPE.COORD_JOB_CREATED,
            coordJob.getJobId(), coordJob.toString(), getUserAndGroup(coordJob), "INFO",
            XLogMessage.LIFECYCLE.END, "");

        coordJob.setStatus(CoordinatorJob.Status.RUNNING);
        update(coordJob.getId(), coordJob);

        addLog(coordJob.getJobId(), "Coord job submitted successfully.");
        return coordJob.getJobId();
    } catch (IOException e) {
        LOG.error("Error creating coordinator job.", e);
        return "";
    } finally {
        IOUtils.closeQuietly(confFile);
    }
}
```
submitRequest()函数处理工作流实例的提交请求，首先校验输入参数的合法性，然后根据权限控制判断用户是否有权启动新的工作流实例，如果有环路依赖关系，则抛出异常。接着构造工作流实例对象，调用createCoordJobs()函数生成所有的action并加入待执行队列，然后启动这些action。最后更新工作流实例的状态，返回工作流实例ID。
#### b) createCoordJobs()
```java
private List<CoordinatorAction> createCoordJobs(WorkflowJobBean wfJob, String user) throws IOException {
    DAGClient<String, Integer> dagClient = initializeDag(wfJob.getDagName(), wfJob.getCredentials());
    Set<Integer> readyTasks = dagClient.getReadyNodes();
    Map<String, Set<String>> sharedDatasets = generateSharedDatasets(wfJob);

    List<CoordinatorAction> coordActions = new ArrayList<>();
    while (!readyTasks.isEmpty()) {
        int taskId = selectTaskToSubmit(dagClient, readyTasks);
        readyTasks.remove(taskId);
        String nodeName = dagClient.getNodeName(taskId);
        NodeDescriptor nodeDesc = dagClient.getNode(nodeName).getValue();
        AbstractMainTransition transition = TransitionFactory.getTransition(nodeDesc, conf, envVars, sharedDatasets);
        String jobXml = XmlUtils.prettyPrint(transition.toJobXml(user)).trim();

        CoordinatorAction coordAction = new CoordinatorAction();
        coordAction.setConf(conf);
        coordAction.setEnvVariables(envVars);
        coordAction.setJobId(idGenerator.generateID(IDGenerator.COORDINATOR_ACTION_ID_PREFIX));
        coordAction.setName(nodeName);
        coordAction.setNominalTime(new Date());
        coordAction.setActionNumber(coordActions.size()+1);
        coordAction.setType(ActionType.COORD_STD_MAPREDUCE);
        coordAction.setConf("<coordinator-app name=\"\" xmlns=\"uri:oozie:coordinator:0.4\">"
                        + "<action>" + jobXml + "</action></coordinator-app>");

        coordActions.add(coordAction);
    }
    return coordActions;
}
```
createCoordJobs()函数通过调用initializeDag()函数初始化DAG图，获取所有ready状态的task，然后选择一个task，调用selectTaskToSubmit()函数，生成一个action并添加到列表中，直到所有task都被处理。这里的selectTaskToSubmit()函数实际上通过策略模式调用不同的选取策略（如FIFO、FairShare、Earliest Deadline First等）来确定下一个待执行的task。
#### c) initializeDag()
```java
private DAGClient<String, Integer> initializeDag(String dagName, Credentials credentials) throws IOException {
    DAG<String, Integer> dag = DagUtils.parseGraph(workflowXml);
    Configuration conf = getBaseConfWithJars();
    conf.set(OozieClient.COORDINATOR_APP_NAME, appName);
    return new DAGClientImpl<>(dag, conf, credentials);
}
```
initializeDag()函数解析DAG图的XML文件，生成DAG图对象，并调用Configuration对象生成基于基础配置的子配置，然后生成DAGClient对象，返回。
#### d) selectTaskToSubmit()
```java
int selectTaskToSubmit(DAGClient<String, Integer> dagClient, Set<Integer> readyTasks) {
    int nextTaskId = Collections.min(readyTasks);
    String taskName = dagClient.getNodeName(nextTaskId);
    TaskSelector selector = SelectorFactory.getSelector(selectorPolicy);
    List<Set<Integer>> preferredSets = SelectorUtils.getPreferedNodeSets(selector, taskName, dagClient);
    preferredSets = SelectorUtils.normalizePreferredLists(preferredSets, dagClient);
    return selector.getNextTask(taskName, preferredSets, dagClient, readyTasks);
}
```
selectTaskToSubmit()函数通过SelectorFactory.getSelector(selectorPolicy)获得选取策略对象，然后调用selector对象的getNextTask()函数选取下一个待执行的task，这里的selectorPolicy可能的值包括FIFO、FairShare、Earliest Deadline First等。
### 3.2 WorkflowServices类
WorkflowServices类是Workflow服务的核心类，它处理所有的工作流请求。它的主要功能函数如下：
#### a) submitCoordinator()
```java
public synchronized void submitCoordinator(final Context context, final CoordinatorJob coordJob,
                                           final URIBuilder uriBuilder, final String logToken)
                                            throws Exception {

    final String jobId = coordJob.getId();

    final App app = new App();
    app.setName(jobId);
    app.setStartTime(DateUtils.formatDateUTC(System.currentTimeMillis()));
    if (coordJob.getGroup()!= null) {
        app.setGroup(coordJob.getGroup());
    } else {
        app.setGroup("default");
    }
    app.setStatus(App.Status.PREP);
    UserGroupInformation ugi = getOozieUser();
    auditLog(context.getRequestURI(), coordJob.getUser(), ugi,
            AuditLog.AuditLogAction.COORD_NEW, jobId, "NA", "Creating new coordinator application id=" + jobId);

    FileSystem fs = getAppFileSystem();
    Path root = new Path(getOozieHome()) ;// this will be replaced with actual coord job dir once it's determined
    Path coordJobDir = new Path(root, jobId);
    FileSystem remoteFs = null;

    try {
        remoteFs = remoteFsGetFilesystem(uriBuilder.getHost());
    } catch (IOException ioe) {
        throw createException(ioe.getMessage(), ioe);
    }

    FSUtils.mkdirs(coordJobDir, fs, DEFAULT_PERM);

    Path coordExternalIdPath = new Path(coordJobDir, COORD_EXTERNAL_ID_FILE);
    if (fs.exists(coordExternalIdPath)) {
        // check if external ID matches requestor UGI
        StringBuilder sb = new StringBuilder();
        InputStream is = null;
        BufferedReader reader = null;
        try {
            is = fs.open(coordExternalIdPath);
            reader = new BufferedReader(new InputStreamReader(is));

            String line;
            while ((line = reader.readLine())!= null) {
                sb.append(line);
            }
        } catch (IOException ioe) {
            throw createException(ErrorCode.E0723, ioe, coordExternalIdPath);
        } finally {
            closeStreams(reader, is);
        }

        String oldExternalId = sb.toString();
        if (!oldExternalId.equals(ugi.getShortUserName())) {
            throw new XServletException(ErrorCode.E0724, oldExternalId, ugi.getShortUserName());
        }

        FSUtils.delete(fs, coordExternalIdPath, false);
    }

    OutputStream os = null;
    try {
        os = fs.create(coordExternalIdPath, true);
        os.write(ugi.getShortUserName().getBytes(UTF_8));
    } catch (IOException ioe) {
        throw createException(ErrorCode.E0725, ioe, coordExternalIdPath);
    } finally {
        closeStreams(os);
    }

    final Path inputPath = new Path(coordJob.getInpath());
    final URI uri = createCoordUri(uriBuilder, coordJob.getId());
    copyLocalFileToHdfs(inputPath, uri, coordJob.getNamespace());

    Properties confProps = coordJob.getConfProperties();
    if (confProps!= null) {
        Path confFilePath = new Path(coordJobDir, CONF_FILE);
        writeConfToFile(confProps, confFilePath, fs, UTF_8);
    }

    // Adding custom headers from HTTP Request Headers
    Properties headers = context.getHeaders();
    for (Object headerKey : headers.keySet()) {
        Object headerValue = headers.get(headerKey);
        if ("content-type".equalsIgnoreCase((String) headerKey) || "accept".equalsIgnoreCase((String) headerKey)) {
            String valueStr = (String) headerValue;
            app.setHeader(valueStr.substring(0, 1).toUpperCase() + valueStr.substring(1), valueStr);
        }
    }

    Path actionDataDir = new Path(coordJobDir, ACTION_DATA);
    final String consoleUrl = createConsoleUrl(uriBuilder, jobId);
    confProps.setProperty(OOZIE_URL, consoleUrl);

    byte[] bytes = XmlUtils.toJsonByteArray(confProps);
    confProps.clear();

    final AtomicBoolean failed = new AtomicBoolean(false);
    try {
        HadoopAccessorService has = Services.get().get(HadoopAccessorService.class);
        has.createFile(uri, fs, coordJobDir, APP_PATH_PREFIX + "/workflow.xml", bytes, null, null);
    } catch (Throwable t) {
        failed.set(true);
        throw createException(t, ErrorCode.E0726, uri.toString());
    }

    // Create the coordinator job directory and files before calling start() so that any subsequent error in starting coord jobs
    // can still delete their working directories as they are marked as FAILED. Otherwise, we may end up deleting an active coord job.
    Path doneDirPath = new Path(coordJobDir, DONE_DIR);
    Path failDirPath = new Path(coordJobDir, FAIL_DIR);
    FSUtils.mkdirs(doneDirPath, fs, DEFAULT_PERM);
    FSUtils.mkdirs(failDirPath, fs, DEFAULT_PERM);

    oozieServer.getJobExecutor().start(new Callable<Void>() {
        public Void call() throws Exception {
            try {
                // If we fail after this point but don't raise an exception here, then CoordJobCallable won't mark us as successful
                QueueingService ss = Services.get().get(QueueingService.class);

                startCoordJob(coordJob, context, uriBuilder, coordJobDir, inputPath,
                             coordJob.getTrackerUri(), coordJob.getMatThrottling(),
                             coordJob.getConcurrency(), coordJob.getOrder(), consoleUrl,
                             coordJob.getTimeout(), coordJob.getMaxretries(), logToken,
                             coordJob.getAcl());
                app.setStatus(App.Status.SUCCEEDED);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                throw createException(ErrorCode.E0727, jobId, ie.getMessage());
            } catch (OozieException oe) {
                handleCoordKill(coordJob, uriBuilder, coordJobDir, fs, actionDataDir,
                                coordJob.getLogRetrievalTries());
                throw oe;
            } catch (Exception e) {
                LOG.warn("Error running coordinator for " + jobId + ": ", e);
                failed.set(true);
                handleCoordKill(coordJob, uriBuilder, coordJobDir, fs, actionDataDir,
                                coordJob.getLogRetrievalTries());
                throw e;
            } finally {
                cleanupCoordDirs(coordJobDir, doneDirPath, failDirPath, fs);
            }
            return null;
        }
    });

    if (failed.get()) {
        throw createException(ErrorCode.E0728, jobId);
    }

    auditLog(context.getRequestURI(), coordJob.getUser(), ugi,
            AuditLog.AuditLogAction.COORD_START, jobId, "NA", "Starting coordinator application id=" + jobId);
}
```
submitCoordinator()函数处理工作流实例的提交请求，首先构造一个新的App对象，然后调用copyLocalFileToHdfs()函数复制本地输入文件到HDFS上的指定路径。接着写入配置文件，并调用HadoopAccessorService.createFile()函数写入workflow.xml到HDFS上。最后调用JobExecutor.start()函数启动coordinator job。
#### b) startCoordJob()
```java
protected void startCoordJob(final CoordinatorJob coordJob, final Context context, final URIBuilder uriBuilder,
                              final Path coordJobDir, final Path inputPath, final String trackerUri,
                              final Long matThrottling, final Integer concurrency, final CoordinatorJob.Order order,
                              final String consoleUrl, final Long timeout, final Integer maxRetries,
                              final String logToken, final CoordinatorAuthorizationToken coordAuthToken)
                               throws Exception {
    Thread runnerThread = new Thread() {
        public void run() {
            final boolean result = CoordCommandLauncher.processCoordCommand(uriBuilder, coordJob, coordJobDir,
                                                                             inputPath, trackerUri, matThrottling,
                                                                             concurrency, order, consoleUrl, timeout,
                                                                             maxRetries, logToken, coordAuthToken);
            notifyParent(result);
        }
    };
    runnerThread.start();
}
```
startCoordJob()函数启动一个线程，调用CoordCommandLauncher.processCoordCommand()函数执行coordinator命令。
#### c) CoordCommandLauncher类
```java
static boolean processCoordCommand(final URIBuilder uriBuilder, final CoordinatorJob coordJob,
                                   final Path coordJobDir, final Path inputPath, final String trackerUri,
                                   final Long matThrottling, final Integer concurrency, final CoordinatorJob.Order order,
                                   final String consoleUrl, final Long timeout, final Integer maxRetries,
                                   final String logToken, final CoordinatorAuthorizationToken coordAuthToken)
                                    throws InterruptedException, ExecutionException,TimeoutException{
    ZKUtil zk = ZKUtil.getInstance();
    CoordJobCallable callable = new CoordJobCallable(uriBuilder, coordJob, coordJobDir, inputPath,
                                                    trackerUri, matThrottling, concurrency, order,
                                                    consoleUrl, timeout, maxRetries, logToken, coordAuthToken);
    Future<Boolean> future = Executors.newSingleThreadExecutor().submit(callable);
    boolean success = future.get(COMMAND_TIMEOUT, TimeUnit.SECONDS);
    return success;
}
```
CoordCommandLauncher.processCoordCommand()函数创建一个CoordJobCallable对象，并用Executors.newSingleThreadExecutor()创建一个单线程的线程池，调用线程池的submit()函数提交任务，并等待任务完成。
#### d) CoordJobCallable类
```java
public class CoordJobCallable implements Callable<Boolean>, Closeable {
    private static final Log LOG = LogFactory.getLog(CoordJobCallable.class);
    private static final int MAX_RETRIES = 3;
    
    private final Context context;
    private final CoordinatorJob coordJob;
    private final Path coordJobDir;
    private final Path inputPath;
    private final String trackerUri;
    private final long matThrottling;
    private final int concurrency;
    private final Order order;
    private final String consoleUrl;
    private final long timeout;
    private final int maxRetries;
    private final String logToken;
    private final CoordinatorAuthorizationToken coordAuthToken;
    
    public CoordJobCallable(Context context, CoordinatorJob coordJob, Path coordJobDir, Path inputPath,
                            String trackerUri, Long matThrottling, Integer concurrency, Order order,
                            String consoleUrl, Long timeout, Integer maxRetries, String logToken,
                            CoordinatorAuthorizationToken coordAuthToken) {
        this.context = context;
        this.coordJob = coordJob;
        this.coordJobDir = coordJobDir;
        this.inputPath = inputPath;
        this.trackerUri = trackerUri;
        this.matThrottling = matThrottling == null? -1L : matThrottling;
        this.concurrency = concurrency == null? 1 : concurrency;
        this.order = order;
        this.consoleUrl = consoleUrl;
        this.timeout = timeout == null? -1L : timeout * 1000; // convert seconds to millis
        this.maxRetries = maxRetries == null? MAX_RETRIES : maxRetries;
        this.logToken = logToken;
        this.coordAuthToken = coordAuthToken;
    }

    protected CoordinatorDriver prepareDriver() throws Exception {
        final YarnScheduler scheduler = new MR2YarnScheduler();
        Configuration yarnConf = YarnConfiguration.loadConfiguration();
        yarnConf.setClass(MRJobConfig.MR_ACLS_CLASS, JobACLsManager.class, JobACLsManager.class);
        rmProxy = new RMProxy(yarnConf, services.get());
        
        LauncherSecurityManager securityManager = new LauncherSecurityManager();
        URL[] urls = new URL[0];
        ClassLoader loader = Thread.currentThread().getContextClassLoader();
        if (loader!= null) {
            urls = launcherContainerClassLoaderFinder(urls, loader);
        }
        securityManager.setupSecurityManager(urls);
        Thread.currentThread().setContextClassLoader(securityManager.getClassLoader());

        final EventHandler eventHandler = getEventHandler(scheduler);
        final RunningJobCache runningJobCache = RunningJobCache.getInstance();
        runningJobCache.init(services.get());
        final ACLStore aclStore = DefaultACLStore.getInstance();
        Runnable runnable = () -> {
        };
        final AclsManager aclsManager = new AclsManagerImpl(aclStore, runnable);

        URI fsDefaultName = new URI(coordJob.getOozieUrl());
        final MRFrameworkClock clock = new MRFrameworkClock();
        final ApplicationMaster appMaster = new CoordinatorAM(clock, coordJob.getId(),
                coordJob.getGroup(), coordJob.getAppName(), coordJob.getAppPath(),
                fsDefaultName, coordJob.getUser(), config, context, services.get(), aclStore,
                conf, eventHandler, listener, info, tokens, timelineEntityGroupId, cluster,
                aclsManager, rmProxy, systemClock, userGroupInformation);

        ContainerLaunchParameters clp = new ContainerLaunchParameters();
        clp.setCommands(Arrays.asList(CoordinatorCommandMapping.buildTokens(coordAuthToken)));
        clp.setEnvironment(Collections.<String, String> emptyMap());
        clp.setApplicationResourceUsageReport(null);
        clp.setNmPrivatePaths(null);
        clp.setNmAddr(null);
        clp.setAuxiliaryServices(Collections.<String, Service>emptyMap());
        List<String> commands = Arrays.asList(
                CoordinatorCommandMapping.toString(clp).split("\\n"));
        LOG.info("Generated command-list:" + commands);
        ApplicationConstants.LOG_DIRS = logs;
        appMaster.initAndRun(commands);

        final CoordinatorDriver driver = new CoordinatorDriver(config, context, services.get(), aclStore,
                conf, clock, appMaster, eventHandler, listener, info, tokens,
                timelineEntityGroupId, cluster, runningJobCache, aclsManager, rmProxy,
                userGroupInformation, securityManager, metricsRegistry);
        driver.start();
        return driver;
    }

    private EventHandler getEventHandler(YarnScheduler scheduler) {
        return new CoordinatorEventHandler(this, scheduler, coordJob.getUser(), coordJob.getGroup(),
                                          coordJob.getAppName(), context, coordJob.getRetryMax(),
                                          coordJob.getExpires(), coordJob.getActionNumber(), outputEvents);
    }
    
    private void waitUntilDone() throws Exception {
        while (driver.getAppState()!= YarnApplicationState.FINISHED
                && driver.getAppState()!= YarnApplicationState.FAILED
                && driver.getAppState()!= YarnApplicationState.KILLED) {
            LOG.info("Waiting until coordination process finishes..");
            Thread.sleep(STATUS_CHECK_INTERVAL);
        }
    }
    
    private synchronized Boolean call() throws Exception {
        try {
            driver = prepareDriver();
            Status status = driver.getCoordJob().getStatus();
            
            if (status == Status.SUCCEEDED) {
                LOG.info("Successfully executed coordinator action!");
                coordStats.incrSucceeded(coordJob.getFrequency());
            } else if (status == Status.FAILED) {
                LOG.info("Coordinator action execution failed!! Check diagnostics...");
                for (WorkflowAction wfa : driver.getCompletedActions()) {
                    if (wfa.getStatus() == WorkflowAction.Status.ERROR) {
                        diag.append("\n*** action " + wfa.getId() + ":\n" + wfa.getErrorMessage());
                    }
                }
                
                if (++numRetries < maxRetries) {
                    diag.append("\nWill retry after waiting for " + retryIntervalMillis + " ms.");
                    
                    Thread.sleep(retryIntervalMillis);

                    notifyParent(Boolean.FALSE);
                    // Notify parent process about retries by returning FALSE instead of TRUE
                    
                    coordStats.incrRetried(coordJob.getFrequency());
                    numRetries++;
                    retryIntervalMillis *= 2; // Increase retry interval exponentially
                    
                    return call();
                    
                } else {
                    LOG.error("Maximum number of retries reached! Will fail coordinator action.");
                    coordStats.incrFailed(coordJob.getFrequency());
                    return Boolean.TRUE;
                }
                
            } else if (status == Status.DONEWITHERROR) {
                LOG.error("Coordinator action completed with errors!! Check diagnostics...");
                for (WorkflowAction wfa : driver.getCompletedActions()) {
                    if (wfa.getStatus() == WorkflowAction.Status.ERROR) {
                        diag.append("\n*** action " + wfa.getId() + ":\n" + wfa.getErrorMessage());
                    }
                }
                
                coordStats.incrFailed(coordJob.getFrequency());
                return Boolean.TRUE;
                
            } else {
                LOG.error("Unknown status received from coordinator engine: " + status);
                coordStats.incrFailed(coordJob.getFrequency());
                return Boolean.TRUE;
            }
            
            return Boolean.TRUE;
            
        } catch (Exception e) {
            diag.append("\n\n" + ExceptionUtils.getStackTrace(e));
            LOG.error("Error processing coordinator action:", e);
            return Boolean.TRUE;
        } finally {
            if (diag.length() > 0) {
                auditLog(context.getRequestURI(), coordJob.getUser(), userGroupInformation,
                         AuditLog.AuditLogAction.COORD_FAILED,
                         coordJob.getId(), "", "Coordinator action execution failed. Diagnostics:\n"
                          + diag.toString());
            }
            closeStreams();
        }
    }
}
```
CoordJobCallable.call()函数处理coordinator命令。首先调用prepareDriver()函数准备执行coordinator job的相关参数，然后循环检查coordinator job的状态，并处理失败的情况。如果coordinator job成功完成，那么更新counter，否则判断是否达到最大重试次数，如果达到了，那么更新counter并结束流程；如果没有达到最大重试次数，那么等待一定时间再次重试。