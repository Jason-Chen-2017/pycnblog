
作者：禅与计算机程序设计艺术                    

# 1.简介
  

什么是Ansible？
Ansible是一个开源的配置管理工具，其目的是通过Playbook（剧本）文件，自动化地对远程主机进行配置管理、部署软件、应用服务等。它采用模块化编程方式，并提供强大的功能特性用于简化复杂的部署任务，可以实现自动化配置管理，使得系统管理员不必手动编写脚本或命令即可完成工作。由于其高度模块化设计、可扩展性及任务自动化能力，广泛被用作基础设施自动化、自动化运维、DevOps自动化等领域。

2.安装
在Ubuntu上安装Ansible
首先需要安装pip包管理器，因为后续的安装需要用到该包管理工具。

sudo apt-get update && sudo apt-get upgrade -y 

sudo apt-get install python-setuptools -y 

sudo easy_install pip 

1.安装ansible角色管理工具ansible-galaxy：

sudo pip install ansible-galaxy 

安装ansible包：

sudo apt-get install software-properties-common -y 

sudo apt-get install python-software-properties -y 

sudo add-apt-repository ppa:rquillo/ansible -y 

sudo apt-get update -y 

sudo apt-get install ansible -y 

2.创建远程主机的SSH免密码登陆
需要保证所有远程主机都支持SSH无密码登陆。如果需要对远程主机进行管理，则需要做如下操作：

创建免密钥登陆的用户

将当前用户添加到远程主机的授权列表中，以便可以免密钥登录

在本地主机上生成密钥对，并将公钥拷贝到远程主机的授权文件中。如：

ssh-keygen -t rsa # 生成密钥对

cat ~/.ssh/id_rsa.pub | ssh user@remotehost "mkdir -p.ssh && chmod 700.ssh && touch.ssh/authorized_keys && chmod 600.ssh/authorized_keys && cat >>.ssh/authorized_keys"

其中user代表远程主机的用户名，remotehost代表远程主机的IP地址或域名。

也可以使用root用户在远程主机上执行以上操作：

su root

ssh-keygen -t rsa

exit

cat /root/.ssh/id_rsa.pub | ssh root@remotehost'mkdir -p.ssh;chmod 700.ssh;touch.ssh/authorized_keys;chmod 600.ssh/authorized_keys;cat >>.ssh/authorized_keys'

其中root@remotehost表示远程主机的ip或者域名，此处使用的时root账户登陆并生成免密钥。

3.测试是否成功设置免密钥登陆
可以使用以下命令验证：

ssh user@remotehost # 连接远程主机

输入yes后回车，尝试免密钥登陆。

4.测试Ansible版本

sudo ansible --version # 查看Ansible版本

如果出现版本号信息，则表示安装成功。


5.克隆ansible官方的playbook仓库

git clone https://github.com/ansible/ansible-examples.git 

cd ansible-examples/playbooks 

6.修改hosts文件

打开inventory目录下的hosts文件：

vi inventory/hosts 

将所有远程主机加入到该文件中，包括本机（localhost），格式示例：

[webservers]

localhost

www.example.com

[dbservers]

mysqlserver.example.com

7.运行playbook

在ansible-examples/playbooks目录下，运行以下命令：

ansible all -m ping # 测试远程主机连通性

ansible webservers -a "/bin/echo 'Hello World'" # 执行远程主机的命令

ansible dbservers -i inventory/prod -m yum -a "name=httpd state=present" # 安装httpd服务