
作者：禅与计算机程序设计艺术                    

# 1.简介
  

华为云虚拟化管理平台（HuaWei Cloud Management Platform，简称HCM）是一个集成、统一、高效的虚拟化管理解决方案。在过去的几年里，华为云的虚拟化管理已经不断发展壮大。通过HCM，可以实现资源池、主机池、云服务器池等多个资源池的管控；可以通过统一的界面管理众多租户、项目、VPC网络、镜像、密钥等资源，并提供监控告警、报表统计等功能；支持VMware、KVM、Docker容器等主流虚拟化技术，满足用户的各种需求。
本文将介绍华为云第一款基于OpenStack设计的虚拟化管理产品——华为云虚拟机管理器MCM（Cloud Virtual Machine Manager）。MCM是在华为云上运行着的一个OpenStack服务，主要用于对虚拟机进行创建、删除、变更、监控等生命周期管理操作。因此，MCM可广泛应用于企业IT系统中的虚拟化管理场景中。
# 2.主要特点
MCM具有以下主要特点：

1. 可视化界面：提供了简单易用的Web界面，使得管理员可以方便地管理虚拟机，实现虚拟化管理的自动化和降低管理成本。

2. 安全性：采用了Kerberos认证机制，保障数据安全、访问权限控制。同时，还提供了数据加密存储、审计日志记录等安全措施。

3. 操作便利性：整体页面清晰、结构合理，操作流程简洁明了。

4. 监控告警：提供了完善的监控指标和告警功能，对虚拟机状态实时掌握，及时发现异常情况。

5. 可扩展性：具备良好的可扩展性，能够快速响应业务发展需求。

6. 支持多种虚拟化技术：目前MCM支持VMware、KVM、Docker等主流虚拟化技术。

7. 兼容OpenStack接口规范：与OpenStack社区保持高度一致性，提供完整的API，覆盖OpenStack生态圈。

8. 拥有强大的开发能力：采用Python语言编写，拥有优秀的工程实践和架构理念。
# 3.MCM架构图
为了更好理解MCM的架构，下图是MCM的架构图：

MCM由两大模块构成：前端和后端。前端负责提供用户操作界面，包括仪表盘、资源管理、虚拟机管理、监控告警等；后端负责提供API接口、数据中心的资源池管理、安全管理、虚拟机生命周期管理、告警信息查询等功能。

MCM后端架构分为服务层、存储层和消息层。服务层负责处理请求，向存储层获取资源信息，并向消息层发送通知或指令。存储层则是数据库层，存储所有的数据，如VM、镜像、密钥、用户、用户组等，是MCM最重要的模块之一。消息层则用于通信，向VM发送生命周期变更通知，或者向管理员发送警告信息。
# 4.安装部署
## 4.1 前提条件
* 安装了Ubuntu操作系统
* 有root权限
* OpenStack已成功安装，可以直接使用Keystone登录（Keystone是OpenStack身份验证、授权、注册等的基础设施）
## 4.2 安装步骤
1. 更新apt源：
  ```bash
  sudo apt update -y
  ```
2. 安装MySQL：
  ```bash
  sudo apt install mysql-server python-mysqldb libapache2-mod-wsgi -y
  ```
  
3. 配置MySQL：
  ```bash
  sudo sed -i "s/#bind-address*/bind-address = 0.0.0.0/" /etc/mysql/mysql.conf.d/mysqld.cnf
  sudo systemctl restart mysql.service
  ```

4. 创建数据库：
  ```bash
  mysql -u root -p
  CREATE DATABASE hwcvmmanager DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;
  GRANT ALL PRIVILEGES ON *.* TO 'hwcvmmanager'@'%' IDENTIFIED BY 'hwcvmmgrpwd';
  FLUSH PRIVILEGES;
  quit;
  ```
  注：“hwcvmmanager” 为用户名，“hwcvmmgrpwd” 为密码

5. 下载安装包：
  ```bash
  wget https://repo.huaweicloud.com/mcm/huaweicloud-mcm-v1.1.0.tar.gz
  tar zxvf huaweicloud-mcm-v1.1.0.tar.gz
  cd huaweicloud-mcm-v1.1.0
  ```

6. 配置数据库连接信息：
  ```bash
  cp mcm/settings_local.py.sample mcm/settings_local.py
  vim mcm/settings_local.py
  # 修改如下信息
  DATABASES = {
    'default': {
      'ENGINE': 'django.db.backends.mysql',
      'NAME': 'hwcvmmanager',
      'USER': 'hwcvmmanager',
      'PASSWORD': '<PASSWORD>',
      'HOST': 'localhost',
      'PORT': '3306',
      }
  }
  ```

7. 初始化数据库：
  ```bash
 ./manage.py syncdb --noinput
 ./manage.py migrate
  ```

8. 添加服务到systemd：
  ```bash
  cp systemd/hws-cloudkitty-api.service ~/.config/systemd/user/
  cp systemd/hws-cloudkitty-processor.service ~/.config/systemd/user/
  cp systemd/hws-mcm-agent.service ~/.config/systemd/user/

  chmod +x ~/./config/systemd/user/*.service
  
  systemctl daemon-reload
  systemctl enable --now hws-cloudkitty-api.service
  systemctl enable --now hws-cloudkitty-processor.service
  systemctl enable --now hws-mcm-agent.service
  ```

9. 配置NTP服务：
  ```bash
  sudo timedatectl set-ntp true
  ```

10. 配置环境变量：
  ```bash
  echo "export PATH=/usr/bin:/usr/sbin:/bin:/sbin" >> ~/.bashrc
  source ~/.bashrc
  mkdir ~/.venv && virtualenv ~/.venv/mcm
 . ~/.venv/mcm/bin/activate
  pip install -r requirements.txt
  ```

11. 配置web server：
  ```bash
  sudo a2enmod wsgi
  sudo cp apache/mcm.conf /etc/apache2/sites-available/
  sudo a2ensite mcm
  sudo systemctl reload apache2.service
  ```

12. 启动web server：
  ```bash
 ./start_all.sh
  ```

13. 浏览器打开 http://your-ip-or-domain ，登录MCM，默认账号密码：<PASSWORD>