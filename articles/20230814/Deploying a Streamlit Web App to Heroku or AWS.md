
作者：禅与计算机程序设计艺术                    

# 1.简介
  

随着Web应用的流行以及云计算的普及，越来越多的人开始在线上部署自己的Web应用。对于新手来说，如何快速地将一个Python/R等脚本语言编写的Web应用部署到云服务器上，或者以服务的方式提供给其他开发者使用是一个十分重要的问题。本文将介绍如何将Streamlit应用部署到Heroku和Amazon Web Services（AWS）平台上的方法。 

为了让读者更直观地理解什么是Heroku和AWS，以及两者之间的区别，我们先简要介绍一下这两个云服务器平台。 

## Heroku
Heroku是一个面向开发人员的PaaS (Platform-as-a-Service)产品，它提供了一个免费的公共服务器环境，可以托管各种类型的Web应用程序。用户只需注册账号、上传代码、设置域名并启动应用程序即可，完全不需要自己管理服务器硬件。Heroku支持多种编程语言、框架和数据库等。

Heroku官方网站：https://www.heroku.com/

## Amazon Web Services（AWS）
AWS是一个云服务提供商，提供基于公有云的基础设施，包括计算资源、存储空间、网络连接、数据库服务和分析服务等。AWS提供了多种不同的计算服务，如EC2（Elastic Compute Cloud）、ECS（Elastic Container Service）等，可以满足不同类型的应用需求。AWS还提供了很多的存储、网络和数据库服务，包括S3（Simple Storage Service）、EBS（Elastic Block Store）、DynamoDB等，可以帮助开发者快速部署自己的Web应用。

AWS官方网站：https://aws.amazon.com/cn/

# 2.前期准备工作
## 2.1 安装Heroku CLI
如果您没有安装过Heroku CLI，请按照下面的链接安装：

https://devcenter.heroku.com/articles/getting-started-with-python#set-up

## 2.2 注册Heroku账号
首先，需要注册Heroku账号。访问Heroku官方网站 https://signup.heroku.com/login 点击“Sign up”按钮注册账号，填写相关信息即可。

## 2.3 安装Git客户端
需要安装Git客户端。如果你已经安装了Git，可以跳过这一步。

Windows下载地址：https://gitforwindows.org/

Mac或Linux系统直接通过命令行输入`sudo apt install git`或`sudo yum install git`进行安装即可。

## 2.4 配置Git账户信息
配置您的Git账户名称和邮箱，确保提交记录中显示正确的用户名和邮箱。

```bash
$ git config --global user.name "your name"
$ git config --global user.email "your email"
```

## 2.5 安装Streamlit
安装Streamlit，可以使用pip安装：

```bash
$ pip install streamlit
```

# 3.将Streamlit应用部署到Heroku上
Heroku是最流行的PaaS云服务器平台之一，本文将演示如何将Streamlit应用部署到Heroku平台上。部署过程包含四个步骤：

1. 初始化Heroku仓库
2. 添加文件至Git仓库
3. 创建Heroku应用
4. 推送代码至Heroku应用

## 3.1 初始化Heroku仓库
使用Heroku之前，需要创建一个新的Git仓库。然后，使用如下命令初始化：

```bash
$ mkdir myapp # 创建目录
$ cd myapp    # 进入目录
$ heroku create myapp  # 在Heroku上创建应用，名为myapp
```

这个命令会在Heroku上创建一个名为myapp的新应用。此时，本地文件夹已经关联远程Git仓库，任何修改都可以同步到Heroku上。

## 3.2 添加文件至Git仓库
如果要部署的是现有的Streamlit应用，则需要把整个应用的文件添加至Git仓库。否则，可以创建一个示例应用并把它添加至Git仓库。

把所有文件添加至Git仓库：

```bash
$ git init           # 初始化Git仓库
$ git add.          # 将文件添加至暂存区
$ git commit -m "init"   # 提交文件至Git仓库
```

把特定文件添加至Git仓库：

```bash
$ git add filename.py     # 将单个文件添加至暂存区
```

## 3.3 创建Heroku应用
创建完Heroku仓库后，可以使用Heroku命令行工具创建应用。登录Heroku控制台，选择“New”->“Create new app”，输入应用名和所属账户（如果已有的话）。按提示完成创建过程。

创建完应用后，会得到一个URL，可以通过该URL访问该应用。

## 3.4 推送代码至Heroku应用
将代码推送到Heroku应用上：

```bash
$ git push heroku master        # 推送master分支的代码到Heroku上
```

这样，应用就部署成功了！你可以通过指定端口号访问应用，也可以用域名访问。

# 4.将Streamlit应用部署到AWS上
除了部署到Heroku平台上，另一种流行的云服务器平台是Amazon Web Services（AWS），本文将介绍如何将Streamlit应用部署到AWS上。部署过程也包含四个步骤：

1. 创建EC2实例
2. 设置安全组规则
3. 配置SSH密钥
4. 将代码推送至EC2实例

## 4.1 创建EC2实例
登录AWS控制台，选择“服务”->“计算”->“Amazon EC2”，进入EC2主页。点击“启动实例”。

选择“Amazon Linux AMI”，点击“下一步: 配置实例详细信息”继续。

在“选择可用区域”中选择实例所在的区域，点击“下一步: 增加存储”。

根据需要调整存储大小，建议至少为10GB。点击“下一步: 标签”。

可以在“标签”页面设置一些标签，这些标签可以用来标记该实例，方便对其进行查找。点击“下一步: 选择运行方式”。

根据应用的运行要求，选择实例的类型和数量。通常情况下，推荐选择t2.micro等较小型实例。点击“下一步: 配置安全组”。

默认情况下，实例具有开放的入站和出站网络访问权限，如果需要限制访问，可以添加相应的安全组规则。点击“下一步: 键对”。

这里可以选择是否要创建新的密钥对，也可以选择已有的密钥对。点击“下一步: 审核和启动”。

最后，点击“启动实例”按钮启动实例。

## 4.2 设置安全组规则
登录AWS控制台，选择“服务”->“VPC”->“安全组”，进入安全组页面。选择实例所在的安全组，点击“编辑”按钮。

如果实例仅作为Web服务器使用，则可以允许HTTP和HTTPS访问；如果还要作为数据库服务器使用，还要允许MySQL或PostgreSQL访问等。

保存安全组设置。

## 4.3 配置SSH密钥
为了使得本地机器和EC2实例之间可以通信，需要配置SSH密钥。登录AWS控制台，选择“服务”->“计算”->“Amazon EC2”，进入EC2主页。选择实例，点击“描述”按钮查看主机名和IP地址。

在本地电脑上打开命令行窗口，输入以下命令，生成密钥对：

```bash
$ ssh-keygen -b 2048 -f ~/.ssh/ec2_rsa_keypair
```

这个命令会生成一个RSA密钥对，私钥保存在当前用户目录下的`.ssh/ec2_rsa_keypair`，公钥保存在同级目录下的`ec2_rsa_keypair.pub`。

将公钥内容复制到EC2实例的SSH授权密钥列表中：

```bash
$ cat ~/.ssh/ec2_rsa_keypair.pub | pbcopy   # 把公钥内容复制到粘贴板
```

切换到EC2控制台，选择“实例状态”选项卡，点击对应实例ID。选择“操作”->“启动实例”，启动该实例。等待实例启动完成后，点击左侧菜单栏中的“网络和安全”。

选择“查看安全信息”，查看实例的IPv4公网IP地址。记住该地址，之后会用到。

## 4.4 将代码推送至EC2实例
登录本地计算机，执行如下命令：

```bash
$ ssh -i ~/.ssh/ec2_rsa_keypair ec2-user@<ip address>   # 使用SSH连接实例
```

其中，`<ip address>`是刚才记录的公网IP地址。第一次连接会提示是否信任主机，输入yes即可。

使用如下命令安装最新版本的Node.js：

```bash
$ sudo yum update -y      # 更新软件包
$ curl -sL https://rpm.nodesource.com/setup_16.x | sudo bash -      # 安装Node.js
$ sudo yum install nodejs -y   # 安装Node.js
```

确认Node.js安装成功：

```bash
$ node -v         # 查看Node.js版本
```

安装npm：

```bash
$ npm install -g npm       # 安装npm
```

使用npm安装streamlit：

```bash
$ npm install -g streamlit    # 安装streamlit
```

确认streamlit安装成功：

```bash
$ streamlit hello            # 测试streamlit
```

将应用代码推送到EC2实例：

```bash
$ scp -r ~/myapp/* ec2-user@<ip address>:~/myapp/  # 将代码推送到实例
```

这里，`~/myapp/`是应用的根目录，请替换成实际的应用路径。

切换到EC2实例，启动Streamlit服务：

```bash
$ streamlit run myapp/myscript.py   # 启动Streamlit服务
```

测试应用，浏览器打开<http://localhost:8501>或其它端口（取决于部署时指定的端口号）。

部署成功！

# 5.总结
本文主要介绍了如何将Streamlit应用部署到Heroku和AWS平台上，以及相应的部署流程和注意事项。Heroku和AWS都是流行的云服务器平台，两者各有特色，读者可以自行选择适合自己的部署方案。希望本文对大家有所帮助，祝大家新年快乐！