
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Oracle GoldenGate是Oracle公司推出的高性能、可靠的数据库实时数据传输工具，它可以帮助企业实现跨越现代化的数据仓库和数据集市，在不同的分布式环境之间同步数据。其由两个部分组成，分别为分布式接收器（GoldenGate Replicat）和分布式发送器（GoldenGate Streamer）。

本文将从分布式接收器、分布式发送器及它们的交互流程角度出发，详细介绍Oracle GoldenGate的高可用架构与配置，并对比其它主流的数据库实时数据传输工具的架构及配置方法。

## 1. GoldenGate 的组成部分

1. 事务日志记录（Transaction Logging）模块

	事务日志记录模块负责将变更数据的元信息（例如，更新类型、主键值等），通过二进制日志传送到目标端数据库，并存储到归档库中。

2. 主备模式切换（Master-Slave Switching）模块

	该模块用于检测和自动处理对任何一个节点的主备角色的切换。当主节点出现故障时，从节点将自动成为新的主节点，以保证高可用性。

3. 消息传送（Messaging）模块

	该模块负责网络通信功能，包括远程过程调用（RPC）、网络消息传递（NMP）以及数据库文件传输协议（DBTFTP）等。它在传输层之上提供业务逻辑协调能力。

4. 一致性管理（Consistency Management）模块

	一致性管理模块用于检测、解决或缓解差异数据。主要用于多主数据库间的数据一致性问题。

5. 数据变化捕获（Data Change Capture）模块

	数据变化捕获模块用于捕获源数据库中的数据变更。该模块能够识别出指定表中的插入、删除或修改的数据记录。

6. 元数据记录（Metadata Recording）模块

	元数据记录模块能够保存所有的DDL、DML语句以及系统视图的信息。该模块能够用于审计、数据迁移、数据同步以及其他用途。

7. 命令解析器（Command Parser）模块

	命令解析器模块能够将源数据库中的SQL命令解析成标准的PL/SQL语言，并根据需要执行对应的操作。

## 2. GoldenGate 接收器的部署架构

在接收器安装之前，需确认以下几点：

1. Oracle Database 的版本兼容

	要使用Oracle GoldenGate，必须要使用符合Oracle Database产品线版本的软件。此外，如果启用了CDC（Change Data Capture）功能，则接收器所在主机的Oracle Database版本还需要能够支持CDC功能。

2. 操作系统的平台兼容

	Oracle GoldenGate 运行于x86架构的Linux和Windows Server操作系统上。其最低要求是64位的操作系统。

3. 安装目录的权限

	Oracle GoldenGate 需要安装到本地磁盘上，并且必须具有相应的读写权限。建议使用root用户进行安装。

4. 配置文件

	在部署Oracle GoldenGate接收器之前，需准备好配置文件。一般来说，配置文件由三部分构成：
	1) 监听端口（Listener Port）：指定用于接收数据传输的端口号。
	2) 控制文件（Control File）：指定用于管理接收器配置和状态的元数据。
	3) 资源定义文件（Resource Definition File）：定义GoldenGate组件和物理资源之间的映射关系。

### 2.1 GoldenGate 接收器的物理机安装

假设有两台物理机serverA 和 serverB，且都已经成功安装Oracle database，接下来，可以按照以下步骤进行安装：

1. 创建目录和符号链接

	在serverA和serverB上分别创建目录：
	```
	mkdir -p /u01/gg_rec
	mkdir -p /u01/gg_home
	mkdir -p /u01/gg_txn
	```
	
	然后，创建符号链接：
	```
	ln -s /u01/gg_home /goldengate
	```
	注意：这里需要修改符号链接的路径以适配当前环境。

2. 设置环境变量

	编辑/etc/profile文件，添加以下两行：
	```
	export ORACLE_HOME=/usr/local/oracledb
	export PATH=$PATH:$ORACLE_HOME/bin:/u01/gg_home/bin
	```

3. 安装Oracle GoldenGate 接收器

	将Oracle GoldenGate接收器安装包拷贝至 serverA和serverB任意一台机器。

	进入接收器安装包所在目录，执行如下命令：
	```
	./runInstaller
	```
	然后按提示选择“INSTALL”选项，等待安装完成即可。

	安装成功后，会在/u01/gg_home目录下生成以下几个重要的文件：
	```
	/u01/gg_home/bin/ggsci (GGSCI command interpreter)
	/u01/gg_home/dirprm/ggos.params (GGOS parameters file)
	/u01/gg_home/dirprm/mgr.ctl (manager control file)
	/u01/gg_home/dirprm/registry.loc (registry location file)
	/u01/gg_home/dirprm/sqlnet.ora (database network configuration)
	/u01/gg_home/dirprm/tnsnames.ora (database listener address mapping)
	```

4. 配置并启动GGSCI

	登录到serverA上，编辑/u01/gg_home/dirprm/ggos.params 文件：
	```
	edit param
	GG_INSTALLDIR=<接收器安装目录>
	GG_GDOCTORCONF=DEFAULT:<接收器安装目录>/ggsdoctortask.conf
	SQLHOST=<listener地址>
	SQLPORT=<listener端口>
	SQLSERVICENAME=<服务名>
	SQUELCHMODE=false
	GGFTRACE=true
	TRCLEVEL=3
	TRCSTATISTICS=none
	TRCCATEGORIES=all
	TRCGROUPS=none
	```
	其中：<listener地址> 是本机或者是VIP地址；<listener端口> 是本机Oracle listener的端口号；<服务名> 是可以唯一标识你的接收器的名称。
	示例如下：
	```
	edit param
	GG_INSTALLDIR=/u01/gg_home
	GG_GDOCTORCONF=DEFAULT:/u01/gg_home/ggsdoctortask.conf
	SQLHOST=localhost
	SQLPORT=1521
	SQLSERVICENAME=orcl
	SQUELCHMODE=false
	GGFTRACE=true
	TRCLEVEL=3
	TRCSTATISTICS=none
	TRCCATEGORIES=all
	TRCGROUPS=none
	```

	配置完成后，启动GGSCI：
	```
	cd $ORACLE_HOME/bin
	./ggsci
	```
	输入 start mgr 来启动manager进程。

5. 配置GG_ROOT下的环境变量

	进入$ORACLE_HOME/network/admin目录，执行以下命令创建listener：
	```
	lsnrctl add <<listener>> <<protocol>> <<host>> <port> <SID>
	```
	示例：
	```
	lsnrctl add orcl tnsdb unix localhost:1521
	```

6. 在另一台物理机上安装 GOLDENGATE 发送器

	将Oracle GoldenGate 发送器安装包拷贝至另一台物理机serverB。同样地，进入发送器安装包所在目录，执行如下命令：
	```
	./runInstaller
	```
	然后按提示选择“INSTALL”选项，等待安装完成即可。

7. 配置 GGSCI 并启动 manager

	登录到serverB上，编辑/u01/gg_home/dirprm/ggos.params 文件：
	```
	edit param
	GG_INSTALLDIR=<发送器安装目录>
	GG_GDOCTORCONF=DEFAULT:<发送器安装目录>/ggsdoctortask.conf
	SQLHOST=<listener地址>
	SQLPORT=<listener端口>
	SQLSERVICENAME=<服务名>
	SQUELCHMODE=false
	GGFTRACE=true
	TRCLEVEL=3
	TRCSTATISTICS=none
	TRCCATEGORIES=all
	TRCGROUPS=none
	```
	其中：<listener地址> 是本机或者是VIP地址；<listener端口> 是本机Oracle listener的端口号；<服务名> 是可以唯一标识你的接收器的名称。

	配置完成后，启动 GGSCI：
	```
	cd $ORACLE_HOME/bin
	./ggsci
	```
	输入 start mgr 来启动 manager 进程。

8. 配置接收器端的 tnsnames.ora 文件

	接收器需要访问发送器数据库，因此需要修改接收器端的 tnsnames.ora 文件。编辑 $ORACLE_HOME/network/admin/tnsnames.ora 文件，加入如下内容：
	```
	orclrecv =
	  (DESCRIPTION=
	    (ADDRESS=(PROTOCOL=tcp)(HOST=localhost)(PORT=1522))
	    (CONNECT_DATA=(SERVICE_NAME=orclsend))
	  )
	```
	其中：(HOST=localhost) 需要根据实际情况修改；(PORT=1522) 是接收器上的 listener 端口号。

至此，接收器的安装配置已完成。

### 2.2 GoldenGate 接收器的虚拟机安装

由于虚拟机无法做到完美契合物理机的性能，因此不推荐这种部署方式。但可以使用VMWare或VirtualBox等虚拟化软件创建一个含有两块硬盘的虚拟机，然后在上面安装接收器。

首先，在宿主机上创建一个目录：
```
mkdir ~/gg_rec
```
然后，将Oracle GoldenGate的安装包拷贝到宿主机的这个目录。进入目录，解压安装包，执行如下命令：
```
./runInstaller -d.
```
完成安装后，查看安装后的安装目录：
```
find. -name ggsci | xargs ls -alh
```
可以看到：
```
total 88K
drwxr-xr-x 2 root root  4.0K Sep 19 16:55./
drwxr-xr-x 6 root root  4.0K Aug 19 10:23../
-rwxr-xr-x 1 root root    17 Feb  5  2019 README*
-rwxr-xr-x 1 root root   77K Sep 19 16:55 ggsci*
drwxr-xr-x 2 root root  4.0K Oct  3 13:03 lib/
drwxr-xr-x 2 root root  4.0K Mar  4 10:33 opt/
drwxr-xr-x 5 root root  4.0K Sep 19 16:55 product/
drwxr-xr-x 2 root root  4.0K Nov 20 17:58 sqllib/
```
安装成功后，可以在宿主机的宿主机所在目录下找到以下几个重要文件：
```
./product/12.2.0/oggcore_12.2.0.0.0_linux64.rpm
./product/12.2.0/pgtarinstaller_12.2.0.0.0_linuxamd64.bin
./product/12.2.0/pscripthandler_12.2.0.0.0_linux64.rpm
```
把这些文件放入到虚拟机的共享目录（mount point）中。比如，在 VMware 中配置 NFS 分布式共享就可以这样设置：
```
sudo yum install nfs-utils # 先安装nfs相关的软件包
sudo mkdir /mnt/gg_share # 创建共享目录
sudo chmod 777 /mnt/gg_share # 修改权限
echo "/mnt/gg_share *(rw,sync,no_subtree_check)" >> /etc/exports # 添加共享目录到 exports 文件
sudo exportfs -a # 重新加载共享目录
showmount -e 192.168.0.101 # 查看共享目录是否正确设置
```
创建完毕后，就需要配置虚拟机了。

首先，需要设置一张虚拟机网络适配器，设置它的 IP 地址，并允许其上网。

然后，安装所需的依赖包，例如，如果要使用 MySQL，需要安装 mysql-connector-java 。

最后，配置 GGSCI ，编辑 /u01/gg_home/dirprm/ggos.params 文件：
```
edit param
GG_INSTALLDIR=<共享目录>/gg_home
GG_GDOCTORCONF=DEFAULT:<共享目录>/gg_home/ggsdoctortask.conf
SQLHOST=<虚拟机的IP地址>
SQLPORT=<MySQL listener 端口>
SQLSERVICENAME=<服务名>
SQUELCHMODE=false
GGFTRACE=true
TRCLEVEL=3
TRCSTATISTICS=none
TRCCATEGORIES=all
TRCGROUPS=none
```
然后，启动 GGSCI ，输入 start mgr ，登录到物理机上执行以下步骤：

首先，启动 MySQL 服务：
```
sudo systemctl start mysqld.service
```

然后，启动 listener 服务：
```
sudo lsnrctl start
```

创建数据库：
```
CREATE DATABASE test;
```

启动 manager 服务：
```
cd $ORACLE_HOME/bin
./ggsci
start mgr
```