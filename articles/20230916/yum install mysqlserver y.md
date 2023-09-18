
作者：禅与计算机程序设计艺术                    

# 1.简介
  

许多Linux服务器运行Mysql数据库时，都需要配置环境变量PATH。如果不设置好环境变量，很可能导致各种各样的问题，比如找不到mysql命令、无法启动服务等。所以，正确配置环境变量可以有效地避免这些问题。

本文将带领读者如何正确配置环境变量PATH，并成功安装Mysql数据库服务。

# 2.安装mysql
在正式配置环境变量前，需要先安装Mysql数据库服务。

## 2.1 CentOS安装MySQL
centos默认已经自带了mysql服务，所以只需要用以下命令即可完成安装。

```bash
sudo yum install mysql-server -y
```


下载对应版本安装包后，执行以下命令进行安装。（安装过程中会提示输入root密码）

```bash
sudo rpm -ivh mysql-community-release-el7-5.noarch.rpm
sudo yum update
sudo yum install mysql-server -y
```

查看已安装的mysql服务：

```bash
sudo netstat -lntp | grep mysqld # 查看mysql服务是否开启
```

## 2.2 Ubuntu安装MySQL

Ubuntu系统默认没有安装mysql，需要手动安装。

首先，需要添加mysql的官方源，然后更新源列表并安装mysql：

```bash
sudo apt-get update
sudo apt-get install mysql-server -y
```

查看已安装的mysql服务：

```bash
sudo netstat -ltnp | grep mysqld # 查看mysql服务是否开启
```

# 3.配置环境变量PATH
在使用mysql之前，必须配置好环境变量PATH。

## 3.1 配置环境变量
编辑配置文件~/.bashrc或~/.bash_profile文件，添加如下两行代码：

```bash
export PATH=/usr/local/mysql/bin:$PATH 
source ~/.bashrc
```

以上两行代码的作用分别是：

1. 添加mysql的可执行文件路径到PATH环境变量中；
2. 执行更新后的配置文件，使得修改立即生效。

## 3.2 验证环境变量
执行下面的命令测试是否成功配置环境变量：

```bash
which mysql
```

如果输出/usr/local/mysql/bin/mysql，则表示环境变量配置成功。

# 4.启动mysql服务
启动mysql服务之前，必须保证系统中已经安装好mysql。

## 4.1 centos启动mysql服务
centos启动mysql服务命令如下：

```bash
sudo systemctl start mysqld.service # 启动mysql服务
sudo systemctl enable mysqld.service # 设置mysql服务开机启动
```

## 4.2 ubuntu启动mysql服务
ubuntu启动mysql服务命令如下：

```bash
sudo service mysql restart # 重启mysql服务
sudo /etc/init.d/mysql stop|start|restart|status # 启动、停止、重启或查看mysql服务状态
```

# 5.创建数据库和用户
登录mysql数据库后，必须创建一个数据库并授予该数据库访问权限给一个用户。

```sql
create database test; # 创建名为test的数据库
grant all privileges on test.* to 'username'@'%' identified by 'password'; # 授权username对test数据库拥有所有权限，允许任意IP访问
flush privileges; # 更新权限表，使修改立即生效
```

这里，我们使用最简单的授权方式，即允许任意IP访问。实际应用中，还需考虑安全性，比如使用加密传输密码等措施。