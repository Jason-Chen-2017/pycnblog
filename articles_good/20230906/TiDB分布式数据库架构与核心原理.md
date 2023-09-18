
作者：禅与计算机程序设计艺术                    

# 1.简介
  

TiDB 是 PingCAP 公司 2017 年开源的分布式 HTAP（Hybrid Transactional/Analytical Processing）数据库产品，其目标是在强一致性、高性能和易用性之间找到平衡点。TiDB 的特点是融合了传统的 RDBMS 和 NoSQL 的最佳特性，具备水平扩展能力、高可用特性、强一致性和实时 HTAP 查询功能等优秀特性。本文从整体架构、集群设计、核心组件、事务模型、存储机制等方面对 TiDB 分布式数据库系统的架构进行全面剖析。

# 2.概览

## 2.1 整体架构


上图是 TiDB 的整体架构。

整体架构分成三个主要模块：Server 层、Client 层和存储层。其中 Server 层和 Client 层分别对应 PD （Placement Driver）和 TiDB 两个组件，存储层则是承载所有数据及相关元信息的 MySQL 兼容存储引擎。PD 通过选举的方式在多个节点间分配数据位置，并且通过 etcd 或其他服务发现机制自动感知各个节点故障并做出调度决策；TiDB 通过 SQL Parser 解析客户端提交的 SQL 请求，将其转化为键值对形式的命令请求，并通过查询优化器生成相应的执行计划；然后将请求发送给指定的 TiKV 节点处理，并最终返回结果给客户端。

## 2.2 集群设计

TiDB 支持部署多种规模的集群，包括单机版和集群版。单机版适用于测试和开发环境，集群版适用于生产环境。

### 2.2.1 硬件配置建议

| 配置项 | 推荐配置      |
| ------ | ------------- |
| CPU    | 8核+          |
| 内存   | 32G+          |
| 磁盘   | SSD 至少 100G |

以上为推荐配置。可以根据实际需求调整配置。比如，对于读密集型业务，可以使用较低的配置如 4C+16G+SSD 运行 TiDB 集群。

### 2.2.2 集群拓扑建议

TiDB 集群通常由 PD、TiDB、TiKV 组成，这些组件均可横向扩展，并使用资源隔离和副本机制保证高可用。


如上图所示，一个典型的 TiDB 集群由三台机器组成：一主一从的 PD 服务器，负责元数据的管理和调度；一台 TiDB 服务器负责响应 SQL 请求；多台 TiKV 服务器负责存储数据和计算。

由于 PD 会承担集群调度的职责，因此 PD 的数量一般需要配置为奇数，建议配置为 3 个或者 5 个。TiDB 可以部署多个实例，但为了避免单点故障，建议每个集群只部署一个 TiDB 实例。

### 2.2.3 部署建议

不同版本的 TiDB 集群部署方式不同，推荐按照以下步骤进行部署：

1. 安装 Docker


2. 拉取镜像文件

```bash
docker pull pingcap/tidb:latest
```

3. 创建目录映射

```bash
mkdir -p /data/pd0,/data/pd1,/data/pd2
mkdir -p /data/tikv0,/data/tikv1,/data/tikv2
```

4. 启动 PD

```bash
docker run --name pd0 \
  -d \
  --restart=always \
  -p 2379:2379 \
  -p 2380:2380 \
  -v /etc/localtime:/etc/localtime:ro \
  -v $PWD/conf/pd.toml:/etc/pd/pd.toml \
  -v /data/pd0:/var/lib/pd \
  pingcap/pd:latest \
  --name="pd0" \
  --data-dir="/var/lib/pd" \
  --client-urls="http://0.0.0.0:2379" \
  --peer-urls="http://0.0.0.0:2380"
```

5. 修改 PD 参数

```bash
sed -i's/^enable-auto-compaction = true$/enable-auto-compaction = false/' conf/pd.toml
sed -i's/^replicate-mode ="async"/replicate-mode = "semi-sync"/g' conf/pd.toml # 可选
sed -i's/# disable-txn-size-limit = true/disable-txn-size-limit = false/g' conf/pd.toml # 可选
```

6. 启动 TiKV

```bash
docker run --name tikv0 \
  -d \
  --restart=always \
  -p 20160:20160 \
  -p 20180:20180 \
  -v /etc/localtime:/etc/localtime:ro \
  -v $PWD/conf/tikv.toml:/etc/tikv/tikv.toml \
  -v /data/tikv0:/var/lib/tikv \
  pingcap/tikv:latest \
  --addr="0.0.0.0" \
  --advertise-addr="${POD_IP}:20160" \
  --store="/var/lib/tikv" \
  --path="/var/lib/tikv/kv" \
  --run-ddl=true \
  --config=/etc/tikv/tikv.toml
```

7. 修改 TiKV 参数

```bash
sed -i's/^defaultcf\.block-cache-size = "\<size\>"$/defaultcf.block-cache-size = "8GB"/g' conf/tikv.toml
sed -i's/^writecf\.block-cache-size = "\<size\>"$/writecf.block-cache-size = "4GB"/g' conf/tikv.toml
sed -i's/^lockcf\.block-cache-size = "\<size\>"$/lockcf.block-cache-size = "1GB"/g' conf/tikv.toml
sed -i's/^titan\/enabled = false$/titan\/enabled = true/g' conf/tikv.toml
sed -i's/^raftdb\/block-cache-size = ""$/raftdb\/block-cache-size = "256MB"/g' conf/tikv.toml
```

8. 启动 TiDB

```bash
docker run --name tidb \
  -d \
  --restart=always \
  -p 4000:4000 \
  -p 10080:10080 \
  -v /etc/localtime:/etc/localtime:ro \
  -v $PWD/conf/tidb.toml:/etc/tidb/tidb.toml \
  pingcap/tidb:latest \
  --store="tikv" \
  --path="127.0.0.1:2379" \
  --config=/etc/tidb/tidb.toml
```

9. 验证集群正常运行

登录 TiDB 控制台 http://$HOST_IP:10080 ，查看集群状态，如果看到 TiDB、PD、TiKV 服务状态都处于健康状态，证明集群已经正常运行。

```sql
tidb> show status;
```

## 2.3 核心组件

### 2.3.1 PD

PD 是 TiDB 分布式数据库系统的关键组件之一。PD 以集群的形式对整个 TiDB 集群中的信息进行统一管理和协调。PD 由多个节点组成，可以通过集群中的任意一个节点访问，提供服务发现和监控告警功能。每个 PD 节点会对外提供如下服务：

- **集群成员管理**：PD 维护集群中各个节点的信息，包括 IP、端口号、角色等。PD 在内部采用 Raft 协议进行数据复制和日志同步，使得集群数据具有高度的容错能力。PD 提供 RESTful API 对外提供服务管理接口。

- **分区调度策略**：PD 根据集群中各个 Region 的大小、数据访问频率等因素，将 Region 划分为不同的物理空间，从而实现不同数据类型的快速检索。PD 使用 Raft Group 和 Bucket 来组织 Region，每一个 PD 节点都会负责其中一部分 Bucket。

- **副本调度策略**：当某个节点出现故障或下线时，PD 将自动检测到异常并将其上的 Region 下线，同时选出新的副本存放位置。

- **元数据存储**：PD 中存储着集群中所有的元信息，包括集群的基本配置信息、Region 信息、副本信息、调度信息等。PD 还提供 gRPC 和 HTTP 接口，用户可以通过这些接口查询集群信息，并实时接收集群变更通知。

### 2.3.2 TiDB

TiDB 是开源分布式 HTAP (Hybrid Transactional/Analytical Processing) 数据库，支持在线事务处理（OLTP）和分析处理 (HTAP)。TiDB 的目的是通过提供一个高度兼容 MySQL 的接口，让用户无缝切换现有的应用，从而达到降低迁移成本和零改造成本的目的。目前 TiDB 已经是 CNCF (Cloud Native Computing Foundation) 孵化项目。

TiDB 使用 Go 语言编写，主要组件如下：

#### Query Engine

Query Engine 是 TiDB 中的计算和查询模块，它主要负责 SQL 的解析、预处理、优化、生成执行计划、统计信息收集等工作。

#### Storage

Storage 是数据存储模块，主要负责数据的写入和读取，以及对数据的维护和安全保护。它包含多种存储引擎，如 TiFlash、RocksDB、TiKV、HBase 等，可以根据需要灵活选择。

#### Distributed Transactions

Distributed Transactions 是分布式事务组件，用于支持跨越多个 Region 的事务。它支持 ACID 属性，包括原子性（Atomicity）、一致性（Consistency）、隔离性（Isolation），持久性（Durability）。

#### TIKV

TiKV 是分布式 KV 存储引擎，由 Rust 语言编写，主要用来存储真正的数据。它提供了丰富的功能，包括持久化、安全、调度等。

#### PD/TiDB 通信协议

PD/TiDB 通信协议是 TiDB 的内部通信协议，用于集群内部的各种组件之间的信息交换和协作。它遵循 HTTP+JSON 或 gRPC 协议，支持 RPC 调用、流量调度等功能。

### 2.3.3 TiKV

TiKV 是 TiDB 分布式数据库系统的关键组件之一。TiKV 以存储为核心，通过 Raft 协议实现数据分布式存储和数据副本的一致性，通过 Coprocessor 模块支持在线计算和聚合运算。它的主要功能如下：

- 数据存储：TiKV 使用底层的 Rust 实现了一个高吞吐量、低延迟的 KV 存储引擎。它的性能超过了目前市面上同类产品，尤其是在云平台环境下的部署。

- 数据分片和副本：TiKV 将数据的存储按照 Region 切分成多个范围，每个 Range 都保存了相同索引的 Key-Value 数据。数据在多个节点上副本，形成多副本冗余的数据存储，能够有效的防止数据丢失。

- 消息订阅：TiKV 支持发布/订阅模式，允许应用程序订阅指定的数据变化，从而实现缓存刷新、消息推送等功能。

- 事务：TiKV 具备完整的 ACID 事务特性，包括事务的提交、回滚、隔离级别等。

- Coprocessor 模块：TiKV 提供 Coprocessor 模块，支持在线计算和聚合运算，极大的提升了查询效率。Coprocessor 可以减少与 TiKV 的网络通信次数，减轻 TiKV 的压力。

## 2.4 存储机制

### 2.4.1 MVCC

MVCC（Multiversion Concurrency Control）即多版本并发控制，是一种并发控制方法，能够确保并发访问存储在数据库中的数据时的正确性，支持高效的读写操作。基于此，TiDB 提供了一套完整的 ACID 事务功能，包括事务的开启、结束、提交、回滚、并发控制等。

TiDB 使用两阶段提交（Two-Phase Commit，2PC）算法来保证事务的一致性。相比于单纯的提交阶段，两阶段提交引入 Prepared 阶段，在这个阶段，Coordinator 准备好事务的执行计划，包括分配事务标识符等，并将该事务记录在各个节点上的 Undo Log 上，等待所有参与者完成协商，最后提交事务。若任何参与者因为某些原因失败，则 Coordinator 首先会取消之前的事务，恢复数据的原始状态。

通过 MVCC 技术，TiDB 可以提供高效且一致的事务并发控制能力，并且能够保证在任意时间点，同一条记录的快照数据只能有一个线程对其进行读写。另外，TiDB 利用 Region 的切分机制，将数据分布到不同的节点，进一步提高了数据容灾能力。

### 2.4.2 行列混合存储

行列混合存储即将数据按照行和列的形式存储在一起，根据访问的热点区域将数据分布到不同的地方。行列混合存储的优点是可以有效的压缩数据，节约存储空间，并支持高效的访问。

TiDB 使用 LSM Tree（Log Structured Merge Tree，日志结构合并树）作为其存储引擎。LSM Tree 是一款类似 BTree 的数据结构，但是它的结构是通过顺序写日志而不是在内存构造的。TiDB 在数据插入时，先将其写入 WAL 文件，之后再异步地刷入 SST 文件中。这么做的目的是为了保证数据在崩溃后仍然可以回滚，保证数据的完整性。SST 文件中保存着排序好的 Key-Value 数据，通过 LSM Tree 的合并算法，能够快速定位、查询 Key-Value 数据。

除了支持 LSM Tree 以外，TiDB 还支持按照列族存储和范围查询。按照列族存储的意思是将不同类型的数据按照不同的列族存储，例如将主键、索引列、数据列按不同的列族存储，这样可以有效的提高查询效率，并且能够降低热点数据的影响。范围查询则可以减少不必要的扫描过程，加快查询速度。

### 2.4.3 Region 切分机制

Region 是 TiDB 内在的概念，也是数据分布和数据隔离的单位。Region 是指一个数据集合，是一个逻辑隔离的存储单元，每个 Region 都会分配一段连续的 KeyRange，并且不同 Region 中的 KeyRange 彼此不会重叠。Region 分裂和副本的切换都是通过调度器进行自动化的。当某个节点的负载过高时，会触发副本的动态迁移，或者主动发起 split 操作，将 Region 分裂成两个 Region，进而增加节点的负载。

TiDB 使用 Pre-Split 机制，在创建表的时候就将 Region 切割开。Pre-Split 可以有效的解决空 keyrange 的问题，减少了内存的占用，提高了读写效率。

### 2.4.4 Raft Group 和 Bucket

Raft Group 和 Bucket 是 TiDB 中用来管理 Region 的概念。Raft Group 是一个独立的系统，由一组编号连续的节点组成，通过 Raft 共识协议实现数据的一致性。每个 Raft Group 有自己对应的编号，默认情况下，每个 Raft Group 的 ID 为 0，并通过 PD 的 Configuration Change 命令进行修改。Bucket 是一个逻辑概念，用于封装 Region，在物理上被切分成多个范围，并对 Range 的读写操作进行归约。

## 2.5 事务模型

TiDB 支持两种事务模型：

- 乐观锁事务：以一定概率检测冲突并重试，直到成功为止。这种事务模型以较低的代价实现并发控制，适用于绝大多数 OLTP 场景。

- 悲观锁事务：完全基于索引加锁，直到成功为止。这种事务模型可能会导致死锁，适用于一些核心交易场景。

TiDB 的悲观锁事务模型，采用的是 SELECT FOR UPDATE 语句。对于普通的 SELECT 语句，TiDB 会对索引项加共享锁，以确保数据最新。而对于 SELECT FOR UPDATE 语句，TiDB 会对索引项加排他锁，阻止其他事务对这些行的更新和删除，直到当前事务提交或者回滚。

## 2.6 SQL 性能优化

TiDB 从 v2.1 版本开始，引入了优化器来加速 SQL 执行，优化器会根据统计信息、执行计划、硬件资源情况等综合因素，生成最优的执行计划。

优化器的优化目标是消除不必要的计算和 IO，提升 SQL 执行的性能。优化器基于成本模型，估计每条 SQL 语句的执行成本，并据此选择执行计划。

### 2.6.1 SQL 调优工具

TiDB 提供了 SQL 调优工具，能够分析 SQL 执行计划，识别和诊断慢速 SQL，并推荐优化方案。

- explain：explain 命令可以打印 SQL 执行计划，帮助用户理解查询的执行路径和消耗的时间。
- slow query：slow query 日志可以记录执行时间超过阈值的 SQL，帮助用户定位慢速 SQL。
- stmt-summary：stmt-summary 日志可以记录执行过程中最耗时的 SQL，包括总的执行时间、CPU 时间、等待时间和 Buffer 命中次数等。

### 2.6.2 SQL 优化技巧

- 使用索引覆盖：尽可能使用索引覆盖查询条件，避免回表查询，提升查询性能。

- 小表驱动大表：SQL 中的 from 子句中小的表应该放在前面，大的表放在后面。

- 读随机写，写随机读：对于写比较多的场景，可以考虑将其设置为随机读写，降低写放大带来的性能损失。

- 避免大范围扫描：如果查询涉及范围大的数据，则应尽量缩小范围，避免产生过多数据，减少网络传输和计算量，提升查询性能。

- 使用 limit offset 代替 count(*)：使用 limit 时，可以提前终止查询，减少遍历次数，提升查询性能。

- 优化 subquery：对于嵌套的 subquery，可以在外层设置过滤条件，将不需要的结果过滤掉，减少计算量和 IO，提升查询性能。

- 批量插入：对于批量插入的数据，应按照批次大小和数据大小进行分组，进行批量插入，减少网络传输和存储压力。

## 2.7 数据导入

TiDB 支持多种数据导入的方式，包括 csv 文件导入、Dumpling 工具导入、lightning 工具导入、TiDB Lightning（实验特性）导入。

### 2.7.1 CSV 文件导入

TiDB 支持直接导入 CSV 文件。CSV 文件的第一行是表头，第二行开始是数据。TiDB 会解析表头，并根据表头的内容，将 CSV 文件导入到对应的表中。

```
mysql> CREATE TABLE mytable (id INT NOT NULL PRIMARY KEY AUTO_INCREMENT, name VARCHAR(255), age INT);
Query OK, 0 rows affected (0.11 sec)

mysql> LOAD DATA INFILE '/path/to/csvfile.csv' INTO TABLE mytable;
Query OK, 3 rows affected, 1 warnings (0.05 sec)
Records: 3  Duplicates: 0  Warnings: 1
```

LOAD DATA 命令支持自定义数据类型，并且会自动对字段进行类型转换。除此之外，还可以通过 SET 语句设置其它导入参数，例如忽略错误行。

```
SET @@session.tidb_skip_utf8mb4=TRUE; // 设置忽略UTF-8字符集编码的警告
LOAD DATA INFILE '/path/to/csvfile.csv' INTO TABLE mytable (id, name, age INT UNSIGNED);
```

### 2.7.2 Dumpling 工具导入

TiDB 同时也提供 Dumpling 工具，用于导出数据，并且支持导入到 TiDB。Dumpling 不仅可以导出的速度快，而且导出的结果与 TiDB 本身的存储格式兼容，可以直接导入到 TiDB。

Dumpling 的使用方法如下：

1. 创建一个用于导出数据的连接：

```
./dumpling -u root -h 127.0.0.1 -P 4000 -t 16 -f "/tmp/test.sql"
```

参数说明：

- u 表示用户名，root 用户名固定为 root。
- h 表示主机地址，默认为本地。
- P 表示端口，默认为 4000。
- t 表示并发数，默认为 4。
- f 表示导出文件的路径，默认为 stdout。

2. 执行导出命令：

```
./dumpling -u root -h 127.0.0.1 -P 4000 -t 16 -f "/tmp/test.sql" > test.log 2>&1 &
```

### 2.7.3 lightning 工具导入

Lightning 是 TiDB 官方提供的一个实验性的工具，用于导入外部数据到 TiDB。Lightning 只支持导入 CSV 文件，并且要求 CSV 文件必须符合如下格式：

- 每行为一个数据，数据之间以制表符 `\t` 分隔。
- 第一行是表头，第二行开始是数据。

```
id       int     not null auto_increment primary key,
name     varchar(255),
age      int
```


Lightning 的使用方法如下：

1. 下载 lightning 工具

```
wget https://download.pingcap.org/tidb-toolkit-{version}-linux-amd64.tar.gz && tar xvf tidb-toolkit-{version}-linux-amd64.tar.gz && cd tidb-toolkit-{version}/bin
```

2. 创建配置文件 config.yaml

```
cat >> config.yaml << EOF
# log level: info, debug, warn, error
log-level: "info"

# the server address of target database which has a cluster type is TiDB or TiFlash
tidb-server-address: 127.0.0.1:4000

# the path of csv file to import
file:./employees.csv

# table name in the target database
database: "mydb"
table: "employees"

# separator used for columns, '\t' by default if omitted
separator: ","

# whether skip header line when parsing input file, `false` by default if omitted
header: true

# charset of csv file, only support utf8 and gbk currently, `utf8` by default if omitted
charset: "utf8"

# specify load data mode, "replace", "insert ignore", "append", each row by default if omitted
mode: "replace"

# specify number of concurrent worker, 1 by default if omitted
threads: 1

# specify session variables passed to TiDB
vars:
  tidb_distsql_scan_concurrency: 10
EOF
```

更多详细信息，请参考官方文档：[Import Data into TiDB Using Lightning](/reference/tools/tidb-lightning/import-example.md)。

3. 执行导入命令

```
./tidb-lightning -config./config.yaml
```

## 2.8 数据安全性

TiDB 使用权限验证和加密传输数据来保证数据安全。

### 2.8.1 权限验证

TiDB 提供了权限验证系统，对 SQL 请求进行身份验证，只有经过认证的用户才可以执行 SQL。

TiDB 使用账户和密码的方式进行认证，并支持基于角色的访问控制（Role-based Access Control，RBAC），以满足复杂业务权限需求。

### 2.8.2 数据加密

TiDB 使用 TLS 加密传输所有数据。数据传输过程中的所有数据都加密传输，不管是 SQL 还是数据，都具有强烈的安全属性。

TiDB 支持以下几种加密方式：

- TLS：使用 TLS 协议对数据进行加密传输，使用加密算法如 AES、RSA 等保护数据安全。
- SSL：与 TLS 类似，但是更加稳定，同时兼容老版本 MySQL。
- CA：颁发数字证书，客户端可以校验证书的有效性，确保数据安全。

## 2.9 结论

TiDB 是一款开源分布式 HTAP（Hybrid Transactional/Analytical Processing）数据库，基于 TiKV 存储引擎构建，兼顾了传统数据库和 NoSQL 数据库的优势。TiDB 提供完善的生态，包括周边生态，例如 TiSpark、TiDB Operator、BR、Backup&Restore 等，能够满足用户的各种场景和需求。