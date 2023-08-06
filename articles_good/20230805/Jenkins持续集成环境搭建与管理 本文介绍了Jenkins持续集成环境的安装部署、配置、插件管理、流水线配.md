
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 Jenkins是一种基于Java开发的开源CI/CD工具，可以自动化地编译、测试并打包应用，还可以进行自动化的部署、发布工作。本文将会从零开始详细介绍Jenkins的安装部署、配置、插件管理、流水线配置、构建触发器、构建结果查看、构建日志审计、定时任务等方面知识。

         相比于其他的CI/CD工具，如Travis CI、Codeship、CircleCI等，Jenkins更加自由灵活，功能更加强大。它可以实现较复杂的工作流，支持各种类型的SCM（源代码管理系统）、构建节点（执行构建命令的服务器或机器）、插件扩展。另外，Jenkins拥有强大的REST API接口，允许第三方系统调用其各项功能，例如监控、报警、调度等。因此，Jenkins已经成为云计算领域中最流行的CI/CD工具。

         在本文中，你将会了解到以下内容：

         1) Jenkins的安装部署
         2) Jenkins的主要配置文件及它们的作用
         3) Jenkins的插件管理
         4) Jenkins的流水线配置
         5) Jenkins的构建触发器
         6) Jenkins的构建结果查看
         7) Jenkins的构建日志审计
         8) Jenkins的定时任务
         通过上述内容，你可以掌握Jenkins的相关知识技能，利用这些知识解决日常工作中的实际问题，提升工作效率。

        # 2.安装部署
        ## 2.1.下载安装
        由于Jenkins是一个开源软件，所以您可以在官网https://jenkins.io/download/ 上找到最新的版本，下载后直接解压即可启动服务。

        ## 2.2.配置
        安装完成后，需要对一些配置项进行调整。进入jenkins所在目录下的config.xml文件，可以看到很多配置选项，这里仅涉及最常用的配置，其他可选配置请参考官方文档。
        
        ### 2.2.1 设置系统编码
        如果您的系统使用的不是UTF-8编码，则需要修改此配置项，否则可能导致中文乱码或者其他错误。

        ```
        <hudsonHome>PATH_TO_YOUR_JENKINS_HOME</henkinsHome>
        <!-- 修改此处的路径为Jenkins的根目录 -->
        
        <systemMessage encoding="UTF-8">
          <![CDATA[Your customized system message]]>
        </systemMessage>
        <!-- 修改系统消息的编码为UTF-8 -->
        ```

        ### 2.2.2 设置HTTP端口号
        默认情况下，Jenkins的HTTP端口号为8080，如果需要修改，可以在全局配置页面设置。


        ### 2.2.3 创建管理员账户
        使用默认账号登录Jenkins时会提示输入用户名和密码，这是为了保证安全性的考虑。建议创建一个管理员账户，该账户具有所有权限，然后删除默认的admin账户。点击左侧导航栏的“Manage Jenkins”，在页面下半部分找到“Configure Global Security”部分，点击“Create Admin User”按钮创建管理员账户。


        ### 2.2.4 配置SSH连接
        SSH是远程访问Jenkins服务器的标准方法，而通过SSH可以执行远程命令，管理Jenkins服务器，这也是Jenkins推荐的方法。在全局配置页面找到“Configure System”部分，选择“SSH Server”。


        配置好之后，可以通过SSH方式访问Jenkins服务器。

        ## 2.3.启动Jenkins
        启动服务后，打开浏览器输入http://localhost:8080就可以进入Jenkins的主界面了，默认的登录用户名密码都是admin/admin。
        

    # 3.主要配置文件说明
    Jenkisn提供了两个配置文件，一个是全局配置文件jenkins.yaml，另一个是用户配置文件user.properties。

    ## 3.1.jenkins.yaml
    jenkins.yaml是全局配置文件，放在Jenkins的根目录下，默认不做任何修改，只需要知道它的位置即可。
    
    ```
    jenkins:
      mode: NORMAL
      version: 2.222.1
      numExecutors: 2
      labels: docker
      usageMode: EXCLUSIVE
      scmCheckoutRetryCount: 10
    toolLocationConfiguration:
      git: /usr/bin/git
    nodeProperties: []
    ```
    
    - mode：Jenkins运行模式，NORMAL表示正常模式，EXCLUSIVE表示独占模式。在独占模式下，Jenkins只能被单个用户登陆，不能同时处理多个构建任务。如果有多个用户要同时执行构建任务，就需要切换到共享模式。
    - version：当前Jenkins版本号。
    - numExecutors：Jenkins最大的并发执行线程数。
    - labels：标签，用于标记节点。
    - usageMode：节点的使用模式，NORMAL表示共享，EXCLUSIVE表示独占。
    - scmCheckoutRetryCount：SCM检出失败重试次数。
    - toolLocationConfiguration：软件安装路径。

    ## 3.2.users配置文件user.properties
    user.properties文件存储的是Jenkins用户信息。其中包括每个用户的用户名、密码、昵称、E-mail地址、API Token、Token描述等属性。

    ```
    admin=xxxxxxxxxxxxxxxxxxxxx
    test1=$2a$10$YjCxylJXyXxxk7LiSHjgtuERt5wOp2NtdNXXvxmfONULdNIqlCEGW
    Anonymous=anonymous
    xiaoming=xxxxxxxxxxxxxxxxxxx
    ```

    可以用文本编辑器打开该文件，新增用户信息。
    
    # 4.插件管理
    Jenkins的插件是个非常重要的模块，因为它提供了很多实用的功能。Jenkins所有的插件都在管理界面的插件库中进行安装和更新，插件的安装及卸载也均通过这个界面进行。

    ## 4.1.安装插件
    要安装某个插件，首先需要确保您的Jenkins处于运行状态，然后点击左侧导航栏的“Manage Jenkins”，在页面下半部分找到“Manage Plugins”部分，选择“Available”标签页。


    搜索需要安装的插件，找到插件后点击“Install without restart”按钮进行安装。等待插件安装完成即可。
    
    ## 4.2.插件配置
    有些插件需要在全局配置文件jenkins.yaml中进行配置，有些插件还需要在项目配置文件pom.xml或settings.xml中进行配置。
    
    ### 4.2.1.maven插件的配置
    Maven插件用于构建Java项目。在全局配置文件中添加maven安装路径，并在系统设置中启用Maven支持。
    
    ```
    mavenInstaller:
      installers:
        - maven:
            id: '3.6.3'
            url: https://repo1.maven.org/maven2/org/apache/maven/apache-maven/3.6.3/apache-maven-3.6.3-bin.zip
    ```

    ```
    <buildWrappers>
        <wrapper class="hudson.plugins.gradle.GradleBuildWrapper"/>
        <wrapper class="hudson.plugins.maveninfo.MavenInfoBuildWrapper"/>
        <wrapper class="hudson.tasks.Maven$MavenInstallation">
          <name>Maven 3.6.3</name>
          <home>/var/lib/jenkins/tools/hudson.tasks.Maven_MavenInstallation/Maven 3.6.3/</home>
        </wrapper>
      </buildWrappers>
      
      <publishers>
        <hudson.tasks.MavenDeployer plugin="maven-deploy-plugin@2.8.2">
           ...
        </hudson.tasks.MavenDeployer>
      </publishers>
    </project>
    ```

    ### 4.2.2.GitLab插件的配置
    GitLab插件用来与GitLab集成，能够从GitLab获取代码，并且能够在Jenkins中执行构建任务。
    
    ```
    gitlabPluginConfig: 
      apiUri: "https://gitlab.example.com/" 
    ```
    
    ```
    GitLab Plugin Configuration: 
        API URI: https://gitlab.example.com/ 
      
    Pipeline Job > Definition > Build Steps > Add build step > GitLab > Repository URL and credentials: 
      https://gitlab.example.com/mygroup/myproject.git (Repository URL) 
      credentials of mygitlabuser or use Git configuration from project    
```

# 5.流水线配置

## 5.1.介绍

流水线(Pipeline)是Jenkins提供的一种可视化的方式定义构建任务，它使得构建过程更加直观，并且提供了强大的跨平台和跨技术栈的能力。

流水线的定义由多个阶段组成，每个阶段代表一个动作，比如检查代码、编译代码、单元测试、生成artifact、部署到某台服务器等。流水线的每个阶段可以执行不同的脚本，这样可以方便地实现自动化。

## 5.2.创建流水线

### 5.2.1.新建流水线

登录Jenkins首页后，点击左侧导航栏的“New Item”按钮，然后选择“Pipeline”类型创建流水线。


输入流水线名称，点击“OK”按钮，进入流水线的配置页面。

### 5.2.2.配置流水线

在“Pipeline”页面，可以看到如下几个配置区域：

1. General：一般设置，包括项目名称、描述等；
2. SCM：源码管理，选择相应的代码仓库；
3. Build Triggers：构建触发器，可以设置定时触发，也可以手动触发；
4. Pipeline：流水线配置，即流水线阶段；
5. Post-build Actions：构建后操作，用于指定流水线结束后执行的操作，比如发送通知邮件等。

#### 5.2.2.1.General

在General区域，填写项目名称、描述等基本信息。


#### 5.2.2.2.SCM

在SCM区域，根据实际需求选择代码仓库的类型，比如Git、SVN等。选择代码仓库的URL地址和凭据。


#### 5.2.2.3.Build triggers

在Build triggers区域，设置构建触发器，包括定时触发、轮询策略、Webhook触发等。

- 定时触发：可以选择每天、每周或每月定时执行构建任务。
- 轮询策略：选择轮询间隔时间，当代码发生变化时，立刻执行构建任务。
- Webhook触发：通过调用外部系统的API，向Jenkins发送消息，触发构建任务。


#### 5.2.2.4.Pipeline

在Pipeline区域，可以定义流水线阶段。按需配置不同任务，比如执行shell脚本、构建Docker镜像、发布Docker镜像等。


#### 5.2.2.5.Post-build Actions

在Post-build Actions区域，可以设置构建后操作，包括发送邮件通知、制品包归档等。


### 5.2.3.保存流水线

保存流水线的配置，并应用。点击页面右上角的“Save”按钮保存配置，应用。


### 5.2.4.启动流水线

保存配置后，即可启动流水线，点击页面左上角的“Build Now”按钮。


## 5.3.流水线变量

流水线变量是流水线的一个特殊功能，它可以让多个阶段共享同样的数据。一般来说，流水线变量可以用于在多个阶段传递数据，比如构建号、代码版本号等。

### 5.3.1.添加变量

点击流水线配置页面的“Variables”选项卡，可以看到如下页面：


点击“Add Variable”按钮添加变量，如下图所示：


填写变量名称、默认值等信息，点击“OK”按钮保存。

### 5.3.2.引用变量

流水线阶段可以使用变量，变量名采用${ }形式引用，如${BUILD_NUMBER}。


变量也可以定义在多个阶段共用。

# 6.构建触发器

构建触发器主要用于控制流水线的运行时间，包括定时触发器、触发外部事件触发器等。

## 6.1.定时触发器

在流水线配置页面的“Build Triggers”区域，可以设置定时触发器。


通过定时触发器，可以满足一些高级的自动化场景，比如定期清理过期的数据、进行数据库备份等。

## 6.2.触发外部事件触发器

触发外部事件触发器主要用于在外部系统中触发Jenkins的构建任务，比如通过代码仓库中触发构建，通过其他CI/CD工具的webhook触发构建等。

在流水线配置页面的“Build Triggers”区域，点击“Trigger builds remotely”链接，进入“Trigger builds remotely”页面。


在“Build when a change is pushed to GitLab”区域，勾选“Push events”。设置完成后，点击页面底部的“Save”按钮保存配置，返回流水线配置页面。

# 7.构建结果查看

构建结果查看主要用于实时查看构建进度、结果等。

## 7.1.实时日志输出

流水线执行过程中，在“Console Output”区域可以实时输出日志，并且可以过滤日志级别。


## 7.2.构建摘要

在流水线执行结束后，会显示构建的结果和耗时等信息。点击“Build Summary”链接，可以查看详细的信息。


## 7.3.构建详情页

构建结果页面展示了整个流水线执行过程的细节。


# 8.构建日志审计

构建日志审计用于记录流水线的执行情况，如各阶段花费的时间，各阶段是否成功，流水线是否有报错等。

点击流水线配置页面的“AuditTrail”选项卡，可以查看构建日志。


# 9.定时任务

定时任务是在特定时间触发流水线执行。在Jenkins管理界面，点击左侧导航栏的“Manage Jenkins”，在页面下半部分找到“Configure Tasks”部分，点击“Cron Tab”按钮，进入“Cron Tab”页面。


在“Cron Tab”页面，可以添加定时任务。
