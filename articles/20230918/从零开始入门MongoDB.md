
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 一句话概括MongoDB
MongoDB是一个基于分布式文件存储的数据库系统。它支持丰富的数据类型、查询及索引管理能力，并提供高效的复制及自动故障转移功能。
## MongoDB优点
- 支持复杂的查询功能：MongoDB支持丰富的查询条件和运算符，使得数据访问变得十分灵活。
- 数据聚合和分析：使用map-reduce操作可以对数据进行聚合和分析，同时还支持分片集群，可以有效处理海量数据。
- 可扩展性：支持横向扩展，利用多台服务器组成集群，可轻松应对大数据量的需求。
- 安全性：通过角色权限控制，保证数据安全，防止恶意攻击或数据泄露。
## MongoDB缺点
- 不擅长处理关系型数据：MongoDB不擅长处理关系型数据，因此在关系型数据库领域有些工作任务会显得十分繁琐。
- 没有事务支持：由于MongoDB是非关系型数据库，没有提供事务支持，如果需要实现业务上的ACID特性，就需要自己实现机制。
- 查询语言有限：MongoDB查询语言的语法较为简单，但也有诸多限制，不能够实现复杂的查询功能。
# 2.安装部署
## 安装过程
### 下载MongoDB
- 将下载好的压缩包解压到指定目录，如`C:\Program Files\MongoDB`。
### 配置环境变量
打开环境变量编辑器，找到系统变量Path对应的项，双击编辑，添加以下内容:
```
C:\Program Files\MongoDB\Server\3.4\bin
```
这样配置好环境变量后，命令行里输入 `mongo`，就可以进入MongoDB交互模式了。
### 初始化数据库
启动MongoDB后，首先要初始化一个数据库。假设我们的数据库名称为testdb，运行以下命令：

```
use testdb; // 切换至testdb数据库
db.createUser({user:"admin",pwd:"admin",roles:[{role:"root",db:"admin"}]}) // 创建管理员用户
```

执行上述命令后，系统会提示输入密码两次，密码需符合要求。创建成功后，可以用如下命令登录数据库：

```
mongo -u admin -p admin --authenticationDatabase "admin"
```

出现下列提示时，表明连接成功：

```
>Successfully authenticated as principal admin on admin database
>mongos>
```

也可以直接在命令行中输入密码完成认证：

```
>db.auth("admin","admin")
{
	"ok": 1,
	"authenticated": 1
}
```

至此，数据库准备完毕。
# 3.基本概念术语说明
## 文档(Document)
在MongoDB中，一个JSON对象即是一个文档。文档类似于关系型数据库中的行记录，字段类似于关系型数据库中的列。一条文档可以有多个键值对，每个键对应的值都可以是不同类型的。
## 集合(Collection)
文档组成的集合即为集合。集合是文档的容器，类似于关系型数据库中的表格。集合中的所有文档结构相同，可以根据特定条件查找、修改或者删除文档。
## 数据库(Database)
数据库是集合的容器。数据库内可以存放多个集合，每个集合可以视作一个小的关系型数据库。
## 服务器(Server)
服务器是MongoDB的运行环境。可以将其看做一个独立的服务程序，负责维护整个数据库系统。每个服务器可以承载多个数据库，每一个数据库在不同的服务器上存储。
## 客户端工具
MongoDB提供了许多客户端工具用于操作数据库，包括：
- Mongo Shell：是MongoDB自带的交互式Javascript Shell，可以用来对数据库进行各种操作，是学习MongoDB的最佳工具。
- Compass：是MongoDB官方推出的MongoDB GUI工具，用于简化对MongoDB的操作。
- Robomongo：是基于Mongo Shell的另一种GUI工具，功能更加强大。
- Robo 3T：是由MongoDB基金会推出的一款基于图形界面的MongoDB客户端工具。