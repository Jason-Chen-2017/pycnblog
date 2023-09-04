
作者：禅与计算机程序设计艺术                    

# 1.简介
  

MongoDB是基于分布式文件存储的数据库。作为NoSQL数据库之一，它支持的数据结构非常灵活，是当前NoSQL数据库中功能最丰富、最强大的一个。由于其高性能、高可用性、易扩展等特点，使得它在web应用、移动开发、物联网(IoT)、地理信息系统(GIS)等领域得到广泛应用。
本篇博文主要阐述了MongoDB的基础知识、应用场景、安装配置及使用方法，以及运维管理相关的技能要求。希望通过阅读本篇博文可以对读者有所帮助。


# 2.基本概念和术语
## 2.1 MongoDB基本概念
MongoDB是一个文档型数据库，由C++编写而成，旨在为Web应用提供可扩展的高性能服务器端数据库解决方案。该数据库将文档存储为BSON（二进制JSON）格式，并支持动态查询语言（如SQL），适用于分布式和分片集群部署。文档数据结构简化了关系数据库表格的设计，允许嵌套文档、数组、及多种数据类型。

MongoDB中的集合（Collection）类似于关系数据库中的表（Table）。集合中存储着文档（Document）数据，文档包含字段（Field）和值（Value）。每个文档之间可以存在很多关系。集合存在于数据库中命名空间（Namespace）中。MongoDB使用_id字段作为默认主键。

## 2.2 MongoDB术语表
|  词汇   |                             解释                             |
|:-------:|:------------------------------------------------------------:|
| Database|                    数据库，同MySQL的database                     |
| Collection |                集合，类似于MySQL的table或SQL Server的view                 |
| Document |               一条记录，类似于MySQL的row或SQL Server的record                |
| Field |           字段，类似于MySQL的column或SQL Server的field            |
| Value |              值，存储在字段中的数据，比如字符串、整数、浮点数等               |
| Query Language | 查询语言，类似于SQL，用于从数据库中查询数据 |
| Index |       索引，加速查询的一种机制，类似于SQL的索引或Nosql的secondary index       |


## 2.3 MongoDB集群架构
MongoDB集群是一个主/副本集结构，所有写入请求都要经过主节点处理，然后同步到副本集其他成员上。任何时候，集群中只会有一个主节点负责处理写操作，其他成员则扮演备份角色，当主节点失效时，另一个节点自动接管继续提供服务。副本集通常由若干个节点组成，主节点和副本集成员通信主要靠心跳检测和副本集选举。

下图展示了一个典型的MongoDB集群架构：




每个MongoDB节点都会运行三个进程：mongod（数据库进程）、mongos（查询路由器）、mongodump/mongorestore（数据迁移工具）。mongod用于存储数据，mongos用于查询路由，mongodump/mongorestore用于数据备份。另外还包括两个辅助进程，第一个是后台任务进程（即oplog tailer），第二个是统计信息收集进程（即dbstats）。

## 2.4 Mongodb安装与配置
### 2.4.1 安装前准备
#### 2.4.1.1 操作系统要求
由于目前市面上的Linux发行版本大多数都是基于RedHat或CentOS的，因此，本文以CentOS 7为例进行安装和配置说明。其它Linux发行版本的安装配置请参考相关文档。

#### 2.4.1.2 系统环境变量设置
MongoDB的包存放在/opt目录下，为了方便管理，需要添加/opt目录到系统环境变量PATH里。执行以下命令修改环境变量：

```shell
sudo vi /etc/profile
```

在最后一行新增如下两行：

```
export PATH=/opt/mongodb/bin:$PATH
export LD_LIBRARY_PATH=/opt/mongodb/lib/:$LD_LIBRARY_PATH
```

刷新配置文件：

```shell
source /etc/profile
```

#### 2.4.1.3 创建数据目录
MongoDB的数据目录一般位于/data/db，需提前创建。执行以下命令创建目录：

```shell
sudo mkdir -p /data/db
```

### 2.4.2 使用yum源安装MongoDB
MongoDB提供了各种平台的预编译安装包，可以使用yum源直接安装。首先启用MongoDB的软件仓库：

```shell
sudo yum install https://repo.mongodb.org/yum/redhat/$releasever/mongodb-org/4.4/x86_64/RPMS/mongodb-org-4.4.repo
```

然后安装MongoDB：

```shell
sudo yum install mongodb-org
```

启动MongoDB服务：

```shell
sudo systemctl start mongod
```

### 2.4.3 配置MongoDB
MongoDB的配置存储在mongod.conf文件中。默认情况下，此文件位于/etc/mongod.conf。打开配置文件：

```shell
sudo vi /etc/mongod.conf
```

修改以下参数：

- bindIp：绑定IP地址，默认为localhost，这里修改为0.0.0.0，允许外部访问。
- port：端口号，默认为27017，如果修改，需要同时修改MongoDB客户端连接串的端口号。
- dbpath：数据文件路径，默认值为/var/lib/mongo。
- logpath：日志文件路径，默认值为/var/log/mongodb/mongod.log。
- journal：开启journal来实现持久化，默认不开启。

保存退出后重启MongoDB服务：

```shell
sudo systemctl restart mongod
```

验证MongoDB是否正常运行：

```shell
mongo --eval "db.runCommand({ connectionStatus: 1 })"
```

输出结果中应包含serverInfo、ok两个字段，表示连接成功。

### 2.4.4 MongoDB客户端连接串
客户端可以通过连接串来访问MongoDB数据库。连接串中包括了主机名、端口号、数据库名称。连接串的语法为：

```
mongodb://[username:password@]host1[:port1][,...hostN[:portN]][/[defaultauthdb]?options]
```

其中，`username`，`password`，`defaultauthdb`，`options`为可选项。

示例：

```
mongodb://myDBReader:<EMAIL>:27017,otherHost.net:27017/?replicaSet=test&readPreference=primaryPreferred
```

说明：

- `myDBReader`为用户名；
- `myDBPassword`为密码；
- `myReplicaSet`为副本集名称；
- `myapp.example.com`和`otherHost.net`为主机域名或者IP；
- `27017`为端口号；
- `/defaultauthdb?options`为默认权限数据库及连接选项。

更多关于连接串的详细信息，请参考官方文档：https://docs.mongodb.com/manual/reference/connection-string/