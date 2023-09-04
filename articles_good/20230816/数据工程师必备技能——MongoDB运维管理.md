
作者：禅与计算机程序设计艺术                    

# 1.简介
  

MongoDB 是一种开源分布式 NoSQL 数据库。它的特点是在易用性、高性能及自动故障转移的前提下提供了快速、灵活的数据处理能力。数据仓库、日志分析系统、电商网站、社交网络、消息推送服务等都依赖 MongoDB 来存储和处理海量的数据。

为了能够有效地运维 MongoDB ，需要掌握一些重要的技术工具及方法。掌握 MongoDB 的运维管理知识，能够帮助你更好地管理 MongoDB 集群，降低运维成本，保障生产环境的数据安全可靠。以下将介绍 MongoDB 运维管理的知识体系。

# 2.基本概念术语说明
## 2.1 NoSQL（非关系型数据库）
NoSQL 指的是 Not only SQL，意即不仅仅是 SQL 语言，还包括键-值对、文档、图形、列族等各种非关系型数据库技术。

在实际应用中，MongoDB 有着极快的读写速度，也支持水平扩展，因此在很多场景中被广泛采用。

除了 MongoDB 以外，目前业界还有 Cassandra、HBase 和 Redis 等流行的 NoSQL 技术。

## 2.2 MongoDB 运维管理概览

MongoDB 运维管理一般分为四个阶段：

- 配置管理：包括服务器硬件配置管理、部署安装配置管理、日志管理和监控管理。

- 权限管理：包括用户管理、角色管理、资源管理。

- 备份恢复：包括备份策略设计、备份方案选择、增量备份和复制集机制。

- 慢查询分析：包括慢查询日志采集、慢查询日志解析和优化建议。

## 2.3 MongoDB 运维管理内容体系

本次分享的内容，将从以下方面全面介绍 MongoDB 运维管理知识：

1. MongoDB 简介
2. 基本概念术语说明
3. MongoDB 操作与常用命令介绍
4. 配置管理
5. 权限管理
6. 备份恢复
7. 慢查询分析

# 3. MongoDB 简介
## 3.1 MongoDB 简介

MongoDB 是一种开源分布式 NoSQL 数据库，由 C++ 语言编写而成。它最初开发于 2007 年，最初用于解决大规模 Web 应用数据存储的需求。

由于 MongoDB 使用 BSON 数据格式，因此可以很容易地在客户端和服务器之间进行数据交换。同时，MongoDB 提供了丰富的数据类型，支持动态 schema，使得其适合用于企业级数据仓库。

## 3.2 MongoDB 特性

- MongoDB 支持分布式文件存储：可以将数据文件存放在不同的物理机器上，通过增加更多的机器来实现横向扩展。

- MongoDB 支持自动分片：当数据量过大时，Mongo 将数据自动拆分成多个片段（chunk），并在后台自动建立索引。这样，在进行分组、排序、查找时，无需考虑数据的全局性，只需考虑某个 shard 上的数据即可。

- MongoDB 支持自动故障切换：当某台机器出现故障时，另一个机器可以立即接管其上的数据库，保证数据库可用性。

- MongoDB 支持查询调优：可以通过调整查询方式，如使用索引或扫描不同的数据模式来优化查询效率。

# 4. 基本概念术语说明

## 4.1 MongoDB 中的 Collection 和 Document

### 4.1.1 Collection

Collection 是 MongoDB 中储存数据的逻辑结构，类似于关系型数据库中的表格。每个 Collection 都会有一个唯一的名字，可以由若干条 Documents 构成。

### 4.1.2 Document

Document 是 MongoDB 中存储数据的最小单位。一条 Document 可以是一条记录，也可以是一个嵌套的对象。字段 (Field) 和值 (Value) 的对表示了一个键值对，也就是说，每个 Document 可以包含若干个键值对。

Document 在创建、更新、删除时，都不会影响其他 Document。但是，如果文档修改频繁，可能会导致性能问题。因此，需要定期对 Collection 进行维护，删除陈旧的 Document，让文档数量保持在一个合理的范围内。

## 4.2 MongoDB 中的 Index

Index 是 MongoDB 中用来快速查询、排序数据的一种数据结构。在集合里创建一个索引后，就可以通过这个索引来快速检索数据。

一个 Collection 可以有多个 Index，每个 Index 会加快相应查询的速度。但是，索引也会占用磁盘空间，所以在创建索引之前，应当仔细评估 Collection 的大小、字段分布、索引需要覆盖的查询情况等因素。

## 4.3 MongoDB 中的 Replica Set

Replica Set 是 MongoDB 实现高可用性的方式之一。它由多个 mongod 进程组成，且每一个进程负责处理一部分数据。

一个 Replica Set 中的所有节点具有相同的数据集合。当其中任何一个节点发生故障时，另一个节点会接管其工作。Replica Set 提供了冗余和数据持久化能力，可以抵御部分节点损坏或网络分区带来的影响。

# 5. MongoDB 操作与常用命令介绍

## 5.1 创建数据库

```bash
use db_name   # 创建数据库db_name
```

## 5.2 删除数据库

```bash
db.dropDatabase()    # 删除当前数据库
use new_db          # 连接新数据库new_db
db.dropDatabase()    # 删除新数据库new_db
```

## 5.3 创建集合

```bash
db.createCollection("collection_name")     # 创建集合collection_name
```

## 5.4 删除集合

```bash
db.collection_name.drop()      # 删除集合collection_name
```

## 5.5 插入数据

```bash
db.collection_name.insertOne({"key": "value"})       # 插入一条数据
db.collection_name.insertMany([{"key": "value"},...])  # 插入多条数据
```

## 5.6 查询数据

```bash
db.collection_name.find()                                # 查询所有数据
db.collection_name.findOne({"key": "value"})              # 根据条件查询第一条数据
db.collection_name.find({"key": {$gt: value}})             # 查询指定字段大于指定值的记录
db.collection_name.find().sort({"key": -1})                # 对结果按指定字段排序
```

## 5.7 更新数据

```bash
db.collection_name.updateOne({"key": "value"},{"$set": {"key": "new_value"}})    # 更新第一条匹配的数据
db.collection_name.updateMany({"key": "value"},{"$set": {"key": "new_value"}})   # 更新所有匹配的数据
```

## 5.8 删除数据

```bash
db.collection_name.deleteOne({"key": "value"})    # 删除第一条匹配的数据
db.collection_name.deleteMany({"key": "value"})   # 删除所有匹配的数据
```

# 6. 配置管理

## 6.1 操作系统配置

- 设置系统时区：可以使用 date 命令查看系统时区，使用 timedatectl 命令设置系统时区。

  ```bash
  sudo timedatectl set-timezone Asia/Shanghai
  ```

- 为 MongoDB 指定最大打开文件句柄数限制：

  ```bash
  ulimit -n 64000
  ```

  ​	注：64000 是系统默认值，根据实际需要设置合理的值。

- 为 MongoDB 分配内存：

  ```bash
  vi /etc/security/limits.conf
  mongodb soft nofile 64000
  mongodb hard nofile 64000
  ```

  ​	注：mongodb 改为实际使用的用户名，nofile 要大于等于打开的文件数量。

## 6.2 文件系统配置

- 设置 MongoDB 数据目录：

  ```bash
  mkdir -p /data/mongodb/{data,log}
  chown -R mongodb.mongodb /data/mongodb
  chmod -R 770 /data/mongodb
  ```

  ​	注：chown 修改文件的属主，chmod 修改文件的权限。

- 配置 journaling：

  MongoDB 默认关闭 journaling，因为开启 journaling 会影响写操作性能，并且数据不完整可能造成数据恢复困难。在生产环境中建议开启 journaling。

  ```yaml
  systemLog:
    destination: file
    path: "/var/log/mongodb/mongod.log"
    logAppend: true
  storage:
    engine: wiredTiger
    journal:
      enabled: true           # 开启 journaling
  ```

  ​	注：此处配置为 mongod 服务端的配置，也需要将该配置文件放在 /etc/mongod.conf 或 /etc/mongos.conf 。

## 6.3 日志管理

MongoDB 自带的日志功能比较简单，主要提供两个级别的日志信息：info 和 warn。在默认情况下，MongoDB 只打印 info 级别的信息。

### 6.3.1 查看日志

- 方法一：登录到 MongoDB 客户端，输入如下命令查看日志：

  ```bash
  show logs
  ```

  ​	注：执行完命令之后，会显示当前所在的 log 文件名称和最新日志记录的序号。使用 `tail` 命令，可以实时追踪最新日志的变化：

  ```bash
  tail --follow /var/log/mongodb/mongod.log
  ```

- 方法二：进入 MongoDB 的数据目录，查看日志文件。

  ```bash
  cd /data/mongodb/log/
  ls -lrt mongod.log*
  tail -f *         # 使用 tail 命令实时追踪最新日志的变化
  ```

### 6.3.2 日志级别

- debug：调试日志，输出非常详细的调试信息。

- info：一般信息，输出程序运行过程中的正常状态信息。

- warning：警告信息，输出潜在的问题提示。

- severe：严重错误信息，输出不可恢复的错误信息。

- fatal：致命错误信息，输出无法继续运行的错误信息。

可以通过启动参数 `--verbose` 或 `--quiet` 来控制日志级别。

```bash
--quiet      # 只打印 error 及以上级别的日志
--verbose    # 打印所有级别的日志
```

另外，还可以通过配置文件修改日志级别：

```yaml
systemLog:
  destination: file
  path: "/var/log/mongodb/mongod.log"
  logAppend: true
  logLevel: severe
```

## 6.4 监控管理

监控 MongoDB 时，首先要关注 CPU、内存、网络 I/O、硬盘利用率等各项指标，尤其是它们随时间变化趋势。

MongoDB 提供了几个标准的监控接口：

1. MongoDB shell：mongo 命令的输出，包括各种性能指标和系统状态信息。

2. MongoDB 日志：主要包括 opcounters 指标，统计数据库的读、写、查询次数。

3. Database Monitoring Tools：第三方监控工具，比如 Prometheus、Zabbix 等。

# 7. 权限管理

权限管理是保障数据安全、控制数据访问的重要环节。

## 7.1 用户管理

MongoDB 中，用户管理包括两部分：

1. 添加用户：`db.createUser()`

   ```bash
   use admin          # 切换到 admin 数据库
   db.createUser({
           user: "username",
           pwd: "password",
           roles: [ { role: "readWriteAnyDatabase", db: "admin" } ]
       })
   ```

2. 删除用户：`db.dropUser()`

   ```bash
   db.dropUser("username")
   ```

## 7.2 角色管理

角色管理是基于用户定义的一系列权限规则，通过给用户赋予角色，可以控制用户对特定资源的访问权限。

1. 添加角色：`db.createRole()`

   ```bash
   use admin          # 切换到 admin 数据库
   db.createRole({
         role: "roleName",
         privileges: [{ resource: { anyResource: true }, actions: ["anyAction"] }],
         roles: []
     })
   ```

2. 更新角色：`db.updateRole()`

   ```bash
   db.updateRole(
          "roleName",
          {
              privileges: [
                  { resource: { anyResource: false }, actions: ["anyAction"] },
                  { resource: { collection: "anyCollection" }, actions: ["anyAction"] }
              ],
              roles: ["roleNameA", "roleNameB"],
              changeStream: null
          }
      )
   ```

3. 删除角色：`db.dropRole()`

   ```bash
   db.dropRole("roleName")
   ```

## 7.3 资源管理

资源管理是 MongoDB 中一种抽象概念，指的是数据库、集合或是任意资源。不同的角色可以对不同资源赋予不同的权限，以达到细粒度的访问控制目的。

# 8. 备份恢复

## 8.1 备份原则

- 尽可能避免在业务高峰期做备份，减少对业务的影响。
- 定期全量备份，根据需要选择增量备份。
- 备份文件尽可能小，并考虑压缩。
- 备份应该在线进行，防止数据库卡死或故障导致备份失败。

## 8.2 备份方案

- 手动全量备份：这种备份方式需要人工介入，在业务低谷期执行。

  ```bash
  mongodump --host <host> --port <port> --out /backup/<database>/<collection>/
  ```

- 定时全量备份：这种备份方式可以设置计划任务，定期执行备份。

- 自动增量备份：这种备份方式不需要人工介入，后台自动监测数据变动，生成增量备份。

  通过 oplog 实现，oplog 记录对数据库的所有写入操作，包括增删改，然后读取 oplog 生成增量备份。

## 8.3 复制集

复制集是一个集群，用来实现数据备份和容灾。复制集由一个 Primary（主要节点）和一个或多个 Secondary（从节点）组成。Primary 会把数据同步给 Secondary。如果 Primary 宕机，则自动选举出新的 Primary；Secondary 如果长时间没有收到心跳包，则会认为其失效，重新选举新的 Primary。

- 手动启动复制集：

  ```bash
  mongod --replSet myReplSetName --port 27017
  ```

  启动第一个成员，添加选项 `--replSet`。

- 添加成员：

  ```bash
  mongo --port 27017
  rs.add("localhost:<port>")   # 添加成员
  ```

- 删除成员：

  ```bash
  mongo --port 27017
  rs.remove("localhost:<port>")   # 删除成员
  ```

- 故障转移：

  ```bash
  mongo --port 27017
  rs.freeze("<numhours>")   # 停止服务，等待数据同步完成
  rs.stepDown()               # 下线当前 PRIMARY，等待选举出新的 PRIMARY
  ```

## 8.4 备份恢复演练

### 8.4.1 安装 MongoDB

- 安装软件包：

  ```bash
  wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2004-4.4.4.tgz
  tar xzf mongodb-linux-x86_64-ubuntu2004-4.4.4.tgz
  sudo cp bin/* /usr/local/bin/
  ```

- 配置 MongoDB：

  ```bash
  sudo vi /etc/mongod.conf
  bindIp: 0.0.0.0        # 允许外部连接
  port: 27017            # 设置端口号
  replication:
    replSetName: myReplSetName   # 设置复制集名
  ```

  启动 MongoDB：

  ```bash
  sudo systemctl start mongod
  ```

### 8.4.2 测试 MongoDB

- 登录 MongoDB：

  ```bash
  mongo --host localhost --port 27017
  ```

- 测试数据导入：

  ```bash
  use testdb
  db.testcol.insert({'name': 'John', 'age': 30});
  exit
  ```

### 8.4.3 手工全量备份

- 登录 MongoDB：

  ```bash
  mongo --host localhost --port 27017
  ```

- 执行全量备份：

  ```bash
  use testdb
  var backupDir = '/tmp/' + Math.random();
  var dumpCmd ='mongodump --host'+ 'localhost' +'--port'+ '27017' +'--db'+ 'testdb' +'--out'+ backupDir;
  db.runCommand({custom:{command:'echo \"' + Date() + '\" >'+ backupDir + '.timestamp'}});
  db.runCommand({custom:{command:dumpCmd}});
  exit
  ```

  此脚本将在 `/tmp/` 目录下随机产生一个目录，保存当前日期，并执行 mongodump 命令备份数据。

- 检查备份结果：

  ```bash
  tree /tmp/9a5d9d8d-c6aa-4b5b-ab1c-dc6cc3f2bfbb
  ├── backup_metadata.json
  └── testdb
      └── testcol
          ├── data
          │   └── testcol.ns
          ├── metadata.json
          └──noindex.ns
  ```

### 8.4.4 手工增量备份

- 配置复制集：

  ```bash
  sudo vim /etc/mongod.conf
  replication:
    replSetName: myReplSetName
    oplogSizeMB: 1024   # 设置 oplog 大小
  ```

  重启 MongoDB：

  ```bash
  sudo systemctl restart mongod
  ```

- 测试数据写入：

  ```bash
  mongo --host localhost --port 27017
  use testdb
  db.testcol.insert({'name': 'Mike', 'age': 35});
  exit
  ```

- 执行增量备份：

  ```bash
  var backupDir = '/tmp/' + Math.random();
  var dumpOplogCmd ='mongodump --host'+ 'localhost' +'--port'+ '27017' +'--db'+ 'local' +'--collection'+ 'oplog.rs' +'--query \'{ts:{$gt:Timestamp(0)}}\' --out'+ backupDir + '/' + 'oplog';
  db.runCommand({custom:{command:'mkdir -p'+ backupDir}});
  db.runCommand({custom:{command:dumpOplogCmd}});
  ```

  此脚本将 oplog 的 ts 大于 0 的记录导出到指定的目录，作为增量备份。

- 检查备份结果：

  ```bash
  tree /tmp/f8f2d0d9-4cc7-47d7-befb-1c3b7c76f603
  └── oplog
      ├── local.ns
      └── oplog.bson
  ```