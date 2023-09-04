
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## Jenkins 是什么?
Jenkins 是一款开源CI&CD服务器，它提供自动化构建、部署、测试、监控等功能。它的优点是免费并且支持多种平台和语言。Jenkins目前拥有超过7万个插件和近2万个用户。

Jenkins的主要功能包括：
- 持续集成（Continuous Integration，简称CI）: 将所有开发者代码合并到主干后自动编译、测试，确保软件质量。
- 持续交付（Continuous Delivery/Deployment，简称CD）: 通过自动部署的方式频繁更新产品或服务，以实现业务需求不断的快速迭代。
- 持续测试（Continuous Testing）: 在整个开发生命周期中自动执行测试脚本，发现及时修复软件中的bug。
- 持续分析（Continuous Analysis）: 对项目的代码质量和风格进行分析，持续改进开发流程和提升代码质量。

## Jenkins 安装配置
### 安装Jenkins
在安装Jenkins之前，请确认你的操作系统是否满足要求，要求至少要有Java运行环境。比如CentOS Linux7以上版本或者Ubuntu Server14.04 LTS以上版本。如果你的操作系统不满足要求，你可以选择虚拟机或者云服务器等方式获取一个Linux环境。

然后下载Jenkins安装包并解压到你希望安装的位置，比如/opt/jenkins目录下。
```
$ wget http://mirrors.tuna.tsinghua.edu.cn/jenkins/redhat-stable/latest/jenkins-2.190.1.war
$ sudo mkdir /var/lib/jenkins
$ sudo chown jenkins:jenkins /var/lib/jenkins
$ sudo su - jenkins
$ cd /opt/jenkins
$ java -jar jenkins-2.190.1.war
```
等待Jenkins启动成功后，打开浏览器访问http://localhost:8080，你会看到Jenkins的欢迎页面，然后根据页面提示设置初始管理员账户密码。


接着我们可以创建一个新用户，将其设置为管理员。点击“Manage Jenkins” -> “Configure Global Security” -> “Add User”，输入用户名和密码，并勾选“Make this user an admin”。


最后点击“Save and Apply”保存设置。

### 配置插件
Jenkins的很多特性都需要相应的插件才能工作，而我们第一步也是最重要的一步就是安装这些插件。进入“Manage Jenkins” -> “Manage Plugins”管理插件界面。


Jenkins提供了丰富的插件生态，其中很多插件都是免费的。搜索并安装以下插件：
- Git Plugin
- GitHub plugin (optional)
- Sonarqube Scanner for Jenkins

Git Plugin用于从远程仓库拉取代码，GitHub plugin用于连接GitHub，Sonarqube Scanner用于代码质量检查。除此之外还有很多其他插件可供选择，但由于篇幅原因，这里就不一一列举了。

### 配置Jenkins
#### 创建项目
点击“New Item”创建新的项目，填写项目名称和描述。选择“Freestyle Project”作为类型，点击“OK”创建项目。


#### 设置源码库
点击“General” -> “Source Code Management” -> “Git”，输入git地址并保存。


#### 添加构建触发器
点击“Build Triggers” -> “Poll SCM”，配置定时任务，每隔一段时间检测代码更新并触发构建。


#### 指定构建环境
点击“Build Environment”->“Provide Node & Label During Build”，指定Jenkins节点和标签。如果要发布到Docker Hub，也需要指定Docker镜像标签。


#### 添加构建步骤
点击“Build”->“Add build step” -> “Execute shell”，添加构建命令，比如python setup.py install。


#### 添加凭证
点击“Credentials”->“Jenkins”->“Global credentials(unrestricted)”->“Add Credentials”，添加Jenkins凭证，如GitHub私钥等。


#### 添加构建后操作
点击“Post-build Actions”->“Publish JUnit test result report”，设置junit测试报告路径。如果要发布Docker镜像，还可以添加“Deploy Docker Image”构建后操作。


#### 测试项目
点击“立即构建”按钮手动触发一次构建，观察构建日志和构建结果。


### 部署Flask项目到生产环境
当我们的项目测试通过后，就可以把代码部署到生产环境了。首先，我们需要创建一台新服务器，然后安装Jenkins，并使用Git clone或者SVN checkout代码。然后，根据上面的配置方法设置好Jenkins环境，构建部署脚本，添加环境变量，配置定时任务等。

配置完毕之后，只要定时任务开启，每次代码更新都会自动触发一次构建，测试完成后就会自动部署到生产环境。