
作者：禅与计算机程序设计艺术                    

# 1.简介
  


Redis是一个开源的高性能key-value数据库，它提供了多种数据结构实现，包括字符串、哈希表、列表、集合、有序集合等，可以将数据存储在内存中，具有优秀的读写性能。其内置了一些基本命令，通过这些命令可以完成对数据的CRUD（Create/Read/Update/Delete）操作，通过不同类型的键值数据库之间进行相互转换，方便开发人员快速实现项目功能。

Redis支持主从复制，可以实现读写分离，为集群提供容错性；Redis支持发布订阅模式，可以实现消息的发布和订阅功能；Redis支持事务机制，支持跨多个key之间的操作；Redis支持Lua脚本语言，可以实现复杂的业务逻辑；Redis支持持久化功能，可以将数据存入磁盘，重启之后不会丢失数据；Redis支持缓存淘汰策略，当达到限制的时候，可以自动删除相应的缓存数据。

同时，Redis还支持许多模块扩展，如RedisBloomFilter、RedisGraph、RediSearch等，可进一步提升功能和性能。除此之外，Redis还有很多内部指令，可以通过执行内部命令，实现特定功能。因此，掌握Redis权限配置技巧，能够帮助我们更加灵活地控制Redis服务器的访问权限，提升系统的安全性。

本文主要介绍如何配置Redis的权限管理。首先，介绍一下Redis中角色概念及其权限设置方式。然后，介绍Redis授权模型及授权命令。最后，给出几个实际案例，用以阐述Redis权限配置技巧。

# 2. Redis角色及权限

## 2.1 概念

Redis有以下四个角色：

- 主节点(Master Node)：主节点用来接收其他节点的连接请求，并处理其他节点发送来的命令请求。

- 副本节点(Slave Node)：副本节点接受主节点的连接请求，将主节点的数据集拷贝一份到自己的数据库中，作为自己的数据备份。当主节点发生故障时，副本节点可以顶替他成为新的主节点继续服务。

- 哨兵(Sentinel)：哨兵是Redis官方推荐的一个解决主节点选举和故障转移的方案。哨兵通常部署在Redis集群的外部，监控各个主节点是否正常工作，根据投票数或者时间，选举出一个新的主节点。如果某个主节点无法正常工作，则由哨兵触发故障转移，让另一个副本节点提升为新的主节点，之前的主节点成为新的副本节点。

- 客户端(Client)：客户端是一个运行在任意机器上的Redis应用，可以向Redis服务器发送命令请求，获取响应结果。

每个角色都拥有不同的权限设置。下面介绍各个角色的权限设置及其区别。

## 2.2 主节点权限

主节点拥有所有Redis命令的权限。其权限可以通过配置文件修改或通过授权命令设置。配置文件方式是在redis.conf文件中进行如下配置：

```
bind 127.0.0.1 # 默认绑定localhost，仅限于本地连接
protected-mode no # 是否启用保护模式，默认关闭，设置为yes即启用保护模式，只允许特定IP访问
port 6379 # redis端口号
tcp-backlog 511 # TCP连接队列长度
timeout 0 # 命令超时时间，单位毫秒
tcp-keepalive 300 # 空闲连接保持时间，单位秒
daemonize yes # 是否以守护进程的方式启动
supervised systemd # 以系统托管的方式启动，systemd是Linux系统下的一种服务管理器，这里我们使用systemd托管redis服务
pidfile /var/run/redis_6379.pid # 指定pid文件路径
logfile "" # 指定日志文件路径，为空表示不输出日志文件
always-show-logo yes # 在启动信息里显示Logo
save 900 1 # 每隔多少秒执行一次bgsave命令，负责保存快照
save 300 10 # 如果300秒内有至少10次修改操作，则执行一次bgsave命令
save 60 10000 # 如果60秒内有至少10000次修改操作，则执行一次bgsave命令
stop-writes-on-bgsave-error yes # 是否在出现错误时停止写入操作
rdbcompression yes # 是否压缩RDB文件
dbfilename dump.rdb # RDB文件的名称
dir./ # RDB文件保存目录
replica-serve-stale-data yes # 当追赶数据的时候是否提供旧数据
replica-read-only yes # 是否禁止客户端修改数据
repl-diskless-sync no # 是否使用无盘同步
repl-diskless-sync-delay 5 # 无盘同步延迟，单位秒
repl-ping-slave-period 10 # 与副本节点的ping周期，单位秒
repl-timeout 60 # slave端命令等待超时时间，单位秒
slave-priority 100 # slave优先级，数字越小优先级越高
min-slaves-to-write 0 # 执行写操作时最少需要多少个副本节点确认
min-slaves-max-lag 10 # 执行写操作时，最多不能超过最大延迟时间
lazyfree-lazy-eviction no # 是否开启延迟释放内存功能
lazyfree-lazy-expire no # 是否开启延迟释放过期数据功能
lazyfree-lazy-server-del no # 是否开启延迟释放删除功能
replica-lazy-flush no # 是否延迟复制数据到副本节点
appendonly no # 是否开启AOF持久化功能
appendfilename "appendonly.aof" # AOF文件名称
appendfsync everysec # AOF刷盘策略，有三个选项everysec/always/no
no-appendfsync-on-rewrite no # 是否在AOF重写时阻塞主线程
auto-aof-rewrite-percentage 100 # AOF重写条件，超过改百分比时触发重写
auto-aof-rewrite-min-size 64mb # AOF重写最小文件大小，单位字节
aof-load-truncated yes # 是否载入截断的AOF文件
lua-time-limit 5000 # Lua脚本运行时间上限，单位微秒
slowlog-log-slower-than 10000 # 执行慢查询时记录的时长，单位微秒
slowlog-max-len 128 # 慢查询日志的最大长度
latency-monitor-threshold 0 # 慢查询监控阈值，单位微秒
notify-keyspace-events "" # key空间事件通知，包括hset、del等
hash-max-ziplist-entries 512 # hash类型使用的最大索引节点个数
hash-max-ziplist-value 64 # hash类型使用的索引值最大字节数
list-max-ziplist-entries 512 # list类型使用的最大索引节点个数
list-max-ziplist-value 64 # list类型使用的索引值最大字节数
set-max-intset-entries 512 # set类型使用的最大元素个数
zset-max-ziplist-entries 128 # zset类型使用的最大索引节点个数
zset-max-ziplist-value 64 # zset类型使用的索引值最大字节数
activerehashing yes # 是否激活Rehash功能，打开的话会占用更多资源，降低效率
client-output-buffer-limit normal 0 0 0 # 客户端输出缓冲区限制，包括normal/slave/pubsub三种模式
hz 10 # 时钟频率，用于内部计时任务
dynamic-hz yes # 是否动态调整时钟频率，默认为yes
aof-rewrite-incremental-fsync yes # 是否逐步重写AOF文件
```

配置文件中的参数非常多，这里只介绍上面涉及到的几个参数。bind参数指定Redis监听的IP地址，默认值为127.0.0.1，一般不建议开放远程连接。protected-mode参数设定是否启用保护模式，默认为否。port参数指定Redis的监听端口，默认值为6379。其他参数一般默认即可。

要设置Redis权限，可以使用命令`CONFIG SET`进行配置，该命令可以在运行过程中动态修改Redis的参数。例如：

```
127.0.0.1:6379> CONFIG GET protected-mode // 查看当前配置
1) "protected-mode"
2) "no"

127.0.0.1:6379> CONFIG SET protected-mode yes // 设置保护模式为yes
1) "OK"
```

设置完后，只有白名单的IP才能访问Redis。

## 2.3 副本节点权限

副本节点权限受主节点权限的限制，只能执行部分Redis命令，如AUTH、PING、INFO等。由于副本节点直接拷贝主节点的数据，因此不可修改数据。主节点故障时，需要手动切换到另一个副本节点。

副本节点的配置文件示例如下：

```
slaveof <masterip> <masterport> // 指定主节点IP和端口
protected-mode no
port 6379
tcp-backlog 511
timeout 0
tcp-keepalive 300
daemonize yes
pidfile /var/run/redis_6379.pid
logfile ""
always-show-logo yes
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
dbfilename dump.rdb
dir./
replica-serve-stale-data yes
replica-read-only yes
repl-diskless-sync no
repl-diskless-sync-delay 5
repl-ping-slave-period 10
repl-timeout 60
slave-priority 100
min-slaves-to-write 0
min-slaves-max-lag 10
lazyfree-lazy-eviction no
lazyfree-lazy-expire no
lazyfree-lazy-server-del no
replica-lazy-flush no
appendonly no
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
lua-time-limit 5000
slowlog-log-slower-than 10000
slowlog-max-len 128
latency-monitor-threshold 0
notify-keyspace-events ""
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-entries 512
list-max-ziplist-value 64
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
activerehashing yes
client-output-buffer-limit normal 0 0 0
hz 10
dynamic-hz yes
aof-rewrite-incremental-fsync yes
```

slaveof参数指定了该副本节点的主节点地址。

要设置Redis权限，可以使用命令`SLAVEOF NO ONE`，将当前副本节点停止复制，并且取消之前配置的所有主节点。此时只能通过从库进行连接，执行写操作。

## 2.4 哨兵权限

哨兵权限也是受主节点权限限制。哨兵作为Redis官方推荐的选举和故障转移方案，拥有监控主节点状态、选举新主节点等权限。但是，哨兵不会参与任何业务数据，仅作为维护Redis集群的辅助工具。

哨兵的配置文件示例如下：

```
sentinel monitor mymaster <masterip> <masterport> <quorum> // 配置监控的主节点
sentinel auth-pass mymaster password // 配置监控的密码
sentinel down-after-milliseconds mymaster <milliseconds> // 配置监控超时时间
sentinel failover-timeout mymaster <milliseconds> // 配置故障转移超时时间
sentinel parallel-syncs mymaster <num> // 配置并行同步数量
sentinel notification-script mymaster <scriptname> // 配置通知脚本名称
sentinel client-reconfig-script mymaster <scriptname> // 配置重新配置脚本名称
sentinel config-epoch mymaster <epoch> // 当前配置纪元
sentinel current-epoch 0 // 当前纪元
sentinel known-slave mymaster <hostname> <ip> <port> // 已知副本节点
sentinel known-replica mymaster <hostname> <ip> <port> // 已知从库节点
```

monitor参数用于配置监控的主节点，auth-pass参数用于配置监控的密码，down-after-milliseconds参数用于配置监控超时时间，failover-timeout参数用于配置故障转移超时时间，parallel-syncs参数用于配置并行同步数量，notification-script参数用于配置通知脚本名称，client-reconfig-script参数用于配置重新配置脚本名称。known-slave参数用于配置已知副本节点，known-replica参数用于配置已知从库节点。

要设置Redis权限，可以使用命令`SENTINEL SET`设置哨兵的各种参数，或者使用ACL授权，将哨兵的权限分配给用户。

## 2.5 客户端权限

客户端权限受限于普通用户权限。客户端连接Redis时，不需要用户名和密码。若开启保护模式，需要提供密码才可连接。

# 3. Redis授权模型

Redis支持基于角色的授权模型，可以细粒度控制每个命令的权限，以及限制某些用户组只能执行特定的命令。基于角色的授权模型共分为两类：

- 用户级别：基于角色的授权最基础的单位，一个用户对应一个角色。用户级别的授权在整个Redis集群范围内有效。

- 命令级别：基于角色的授权除了可以针对单个用户授权，还可以针对具体命令授权。命令级别的授权只在特定数据库中有效，并且只能授予超级管理员权限的用户执行。

下面介绍两种授权模式。

## 3.1 用户级别授权

Redis提供了ACL授权系统，允许管理员创建用户和赋予用户权限。其语法如下：

```
useradd username password // 创建用户
userdel username // 删除用户
acl setusername username // 修改用户名
acl getuser username // 获取用户信息
acl whoami // 获取当前用户信息
acl gentoken username // 生成访问令牌
acl token username secret // 使用令牌登录
acl log target [count|reset] // 查看访问日志
aclforget username // 清空用户权限
acl save // 保存ACL配置
acl load // 从持久化文件加载ACL配置
```

其中，useradd命令用于创建用户，userdel命令用于删除用户，acl setusername命令用于修改用户名，acl getuser命令用于获取用户信息，acl whoami命令用于获取当前用户信息，acl gentoken命令用于生成访问令牌，acl token命令用于使用令牌登录，acl log命令用于查看访问日志，aclforget命令用于清空用户权限，acl save命令用于保存ACL配置，acl load命令用于从持久化文件加载ACL配置。

要设置用户级别的Redis权限，可以使用命令`ACL SETUSER`进行设置，该命令带有用户相关的参数。例如：

```
127.0.0.1:6379> ACL SETUSER user on +@all -@admin allkeys -@hello worldkeys +ping +info ~x* del get
OK
```

第一个参数为用户名，第二个参数为密码，第三个参数为权益，第四个参数为命令限制。+表示添加权限，-表示删除权限，@表示所属角色，allkeys表示所有key，worldkeys表示所有key前缀，ping表示执行PING命令，info表示执行INFO命令，~x*表示匹配以x开头的命令，del表示执行DEL命令，get表示执行GET命令。

## 3.2 命令级别授权

Redis提供了COMMAND命令，用于查看当前服务器支持的命令列表。其语法如下：

```
COMMAND INFO command [command...] // 获取命令信息
COMMAND GETKEYS pattern... // 获取命令key
COMMAND COUNT // 获取命令数量
COMMAND SEARCH keyword... // 搜索命令
```

其中，COMMAND INFO命令用于获取命令信息，command参数为命令名称，COMMAND GETKEYS命令用于获取命令key，pattern参数为通配符，COMMAND COUNT命令用于获取命令数量，COMMAND SEARCH命令用于搜索命令。

Redis服务器支持命令级别的授权，但默认没有启用。要启用命令级别的授权，需要配置redis.conf文件，找到`command-permission-block-warning`参数，将该参数的值设为NO。另外，也可通过ACL SETUSER命令给超级管理员分配全部权限，然后再使用ACL DELUSER命令删除所有普通用户。

配置完成后，通过AUTH命令或acl gentoken命令登录Redis，输入正确的密码或访问令牌，即可执行超级管理员权限下的所有命令。

对于普通用户，可以使用命令AUTH mypassword或acl gentoken username mypassword生成访问令牌，再输入这个访问令牌登录Redis。这样就可以执行被授予权限的命令。

# 4. 实际案例

下面给出几个实际案例，展示Redis权限配置技巧。

## 4.1 普通用户执行写操作

假设有一个普通用户alice，希望使用Redis写数据，但是又不想给alice完全的命令权限。为了防止alice滥用权限，可以给alice授权写入数据的命令：

```
127.0.0.1:6379> AUTH alice
127.0.0.1:6379[1]> SET mykey hello
OK
```

如预期，alice成功写入了一个key为mykey的字符串。但是，alice还是可以执行其他的命令，比如PING、INFO等。

为了限制alice只能执行指定的命令，可以使用ACL SETUSER命令给alice分配权限，命令如下：

```
127.0.0.1:6379> AUTH alice
127.0.0.1:6379[1]> USERSET alice
1) (integer) 1
2) (nil)
```

这条命令给alice分配了默认用户级别的权限。alice只能执行指定的命令，比如SET和AUTH命令。其他的命令无法执行。

为了进一步限制alice只能执行指定数据库的命令，也可以使用ACL SETUSER命令给alice分配权限：

```
127.0.0.1:6379> ACL SETUSER alice on >password@0 +SET @mydb +del @mydb
OK
```

这条命令给alice分配了数据库0和mydb两个数据库的权限。在mydb数据库中，alice只能执行SET和DEL命令。其他的命令无法执行。

为了限制alice只能执行命令get和del，可以使用命令MODE：

```
127.0.0.1:6379> MODE alice GET,DEL
1 OK
127.0.0.1:6379> get foo
(nil)
127.0.0.1:6379> del bar
1
```

如预期，alice只能执行GET和DEL命令。其他命令无法执行。

## 4.2 管理员执行超级管理员命令

假设有两个用户，一个叫做bob，另一个叫做tom。bob是Redis普通用户，tom是Redis超级管理员。为了防止bob滥用权限，bob应该只能执行指定的命令。而tom却可以执行所有命令。bob可以使用命令ACL GENTOKEN或AUTH生成访问令牌，然后再输入这个令牌登录Redis。如此，bob就只能执行被授予权限的命令，而tom可以执行所有命令。

为了限制tom只能执行指定的命令，可以使用ACL SETUSER命令给tom分配权限：

```
127.0.0.1:6379> AUTH tom
127.0.0.1:6379[1]> USERSET tom
1) (integer) 1
2) (nil)
```

这条命令给tom分配了默认用户级别的权限。tom只能执行指定的命令，比如USERSET命令。其他的命令无法执行。

为了进一步限制tom只能执行指定数据库的命令，也可以使用ACL SETUSER命令给tom分配权限：

```
127.0.0.1:6379> ACL SETUSER tom on >password@0 +SET @mydb +del @mydb
OK
```

这条命令给tom分配了数据库0和mydb两个数据库的权限。在mydb数据库中，tom只能执行SET和DEL命令。其他的命令无法执行。

为了限制tom只能执行命令get和del，可以使用命令MODE：

```
127.0.0.1:6379> MODE tom GET,DEL
1 OK
127.0.0.1:6379> get foo
(nil)
127.0.0.1:6379> del bar
1
```

如预期，tom只能执行GET和DEL命令。其他命令无法执行。