
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Jenkins是一个开源的基于Java开发的自动化服务器，主要用于支持敏捷开发流程，让软件的构建、测试和发布过程变得透明可重复。Jenkins拥有强大的插件生态系统，支持众多语言的编译、打包、测试等工作流，能够轻松实现WebHooks、Github、GitLab、Bitbucket等主流SCM工具的集成，并提供丰富的接口供外部服务调用。
# 2.基本概念术语
## 2.1 Jenkins的功能特点
- 代码质量管理（Code Quality Management）：通过插件如Sonarqube、FindBugs等进行代码质量分析。
- 智能构建触发器（Smart Build Triggers）：可以根据指定条件（比如，每次代码push后自动构建）自动触发构建流程。
- 插件集成（Plugin Integration）：Jenkins社区有很多好用的插件可以帮助提升工作效率。
- 版本控制集成（Version Control Integration）：支持包括SVN、Git在内的众多SCM工具的集成。
- 丰富的扩展接口（Extensive API）：允许用户编写脚本或Groovy程序集成到Jenkins中。

## 2.2 Jenkins的术语
### 2.2.1 Master节点
Master节点是一个运行着Jenkins master程序的机器，它负责调度执行任务并分配任务给各个节点，同时接受各个节点发送来的命令。Master节点会在后台启动一个WebServer服务，可以通过浏览器访问其Web界面，同时支持通过API方式进行远程控制。

### 2.2.2 Slave节点
Slave节点是运行着Jenkins agent程序的机器，它监听Master节点的请求并执行相关任务，例如编译、测试、打包等。

### 2.2.3 项目配置(Project)
项目配置即Jenkins用来记录项目配置信息的文件，其包含了该项目所需的一切信息，如源码仓库地址、编译环境配置、构建触发规则、执行步骤、定时执行设置等。Jenkins可以同时管理多个项目配置。

### 2.2.4 视图(View)
视图用来整合多个项目，对外展示给其他用户查看，只需要关注视图中展示的项目即可，不需要知道每个项目的细节信息。

### 2.2.5 插件(Plugin)
插件是Jenkins提供的额外能力模块，它可以在不修改Jenkins源代码的情况下增添新的功能特性。有些插件可能是商业插件或者收费插件，这取决于商业许可协议。

### 2.2.6 配置文件(Config file)
配置文件指的是Jenkins的各种设置，包括Master节点配置、Slave节点配置、插件配置、用户权限配置、邮件通知配置等。配置文件保存在/var/lib/jenkins目录下。

## 2.3 Jenkins的组成结构

## 2.4 Jenkins的安装部署
这里以CentOS7为例，介绍一下Jenkins的安装部署过程。

1. 安装Jenkins

```bash
sudo yum install java -y 
wget -O /etc/yum.repos.d/jenkins.repo https://pkg.jenkins.io/redhat-stable/jenkins.repo 
rpm --import https://pkg.jenkins.io/redhat-stable/jenkins.io.key 
sudo yum update -y && sudo yum install jenkins -y 
```

2. 启动Jenkins

```bash
systemctl start jenkins 
systemctl enable jenkins 
```

3. 配置防火墙

为了让外部主机访问Jenkins，还要配置防火墙放行Jenkins端口。
```bash
firewall-cmd --zone=public --add-port=8080/tcp --permanent 
firewall-cmd --reload 
```
4. 在浏览器中打开http://IP:8080，输入默认密码admin，进入登录页面。

5. 创建第一个管理员账户

创建一个管理员账户，方便管理Jenkins。点击“Manage Jenkins”下的“Manage Users”，然后点击“添加新用户”。输入相关信息之后保存。此时，创建的这个管理员账户就可以用浏览器登录Jenkins并进行管理。