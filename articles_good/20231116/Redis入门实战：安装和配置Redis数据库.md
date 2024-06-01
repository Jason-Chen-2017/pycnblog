                 

# 1.背景介绍


## 什么是Redis？
Redis是一个开源的高性能键值对（key-value）存储数据库。它支持多种数据结构，包括字符串、散列、列表、集合、有序集合等。它的主从复制、发布/订阅、事务处理、LUA脚本、LRU缓存、自动淘汰策略等功能都使得它在很多应用场景中都有着突出的优点。许多公司如微博、微信、淘宝等都在使用Redis作为其缓存服务或消息队列服务。Redis目前已经由Redis Labs的Dave Morin负责维护并进行持续开发。
## 为什么要用Redis？
Redis的主要优势之一就是速度快。由于数据存在内存中，相比于磁盘数据库随机读取的效率要高很多，因此Redis能实现非常高的数据读写速度。另外，Redis支持数据的备份，即master-slave模式的数据分布，也能够用于集群架构下的缓存共享。除此之外，Redis还有几个额外的优点：
* 支持多种数据结构：Redis提供丰富的多种数据结构，包括字符串、散列、列表、集合、有序集合等。
* 数据过期功能：对于不经常访问的数据，可以设置相应的过期时间，让Redis自动删除。
* 事务性操作：Redis支持批量执行命令，事务 guarantees the atomicity and consistency of multiple operations within a single step，这使得Redis操作具有原子性，一致性，隔离性和持久性等特性。
* Lua脚本：Lua脚本是一种基于脚本语言的服务器端编程技术。Redis的所有操作都是原子性的，这使得使用脚本可以轻松实现复杂的功能。
* LRU缓存淘汰策略：Redis支持LRU（Least Recently Used，最近最少使用）缓存淘汰策略，该策略将使得Redis缓存中的对象逐渐失去生命力，直到达到最大容量限制。
* 命令监控：Redis提供了命令监控功能，能够记录所有客户端发送的命令，包括执行时长、次数、命令类型、命令参数等。
以上这些优点，使得Redis成为很多大型网站和应用使用的缓存方案或消息队列方案。所以，学习Redis，就等于学习使用缓存、消息队列，并且了解其背后的原理及特点。
## 安装Redis
### Windows环境下安装Redis

1. 打开Windows资源管理器，找到Redis-x64-xxx.zip压缩包。右击选择“属性”，勾选“解压档案到当前目录”复选框，然后单击确定。

2. 在命令提示符窗口输入“cd redis-x64-xxx\”，切换至解压后的文件夹。

3. 执行“redis-server.exe”命令启动Redis服务器。

4. 执行“redis-cli.exe”命令启动Redis命令行界面，输入命令“set foo bar”设置一个键值对，“get foo”获取刚才设置的值。如果一切正常，输出应该为“bar”。

### Linux环境下安装Redis
Redis在Linux环境下的安装，主要依赖编译环境。如果你没有编译环境，那么只能通过源码安装。

1. 从Redis的官方网站上下载最新版本的源码包。

2. 使用以下命令进行编译安装：

```
$ wget http://download.redis.io/releases/redis-4.0.9.tar.gz
$ tar xzf redis-4.0.9.tar.gz
$ cd redis-4.0.9
$ make
$ sudo make install
```

3. 配置Redis：

编辑`/etc/redis/redis.conf`配置文件，修改数据库存放路径、绑定IP地址等信息。例如：

```
bind 127.0.0.1 # bind to local host only
port 6379 # default port is 6379
dir /var/lib/redis # set database directory
logfile "/var/log/redis/redis-server.log" # log file path
dbfilename "dump.rdb" # dump file name for persistence
```

4. 启动Redis服务器：

```
$ sudo service redis start
```

5. 测试Redis是否正常运行：

连接Redis服务器，查看状态信息：

```
$ redis-cli
127.0.0.1:6379> info
```

关闭Redis服务器：

```
$ sudo service redis stop
```

这样Redis就安装完成了。

## 配置Redis
Redis的配置包括两个文件：redis.conf和redis.sentinel.conf。前者是常规配置，后者是哨兵模式的配置。本文只讨论常规模式的配置，哨兵模式的配置另起一节介绍。

一般来说，配置信息都保存在redis.conf文件里，默认情况下，这个文件的位置如下所示：

```
/usr/local/etc/redis.conf (Mac OS X) or /etc/redis/redis.conf (linux)
```


### 基本配置选项

#### daemonize no|yes

指定Redis是否以守护进程的方式运行。如果设置为no，则Redis不会在后台运行，如果设置为yes，则Redis以守护进程方式运行。一般情况下，推荐设置为yes。

#### dir./path/to/data/directory

指定Redis数据文件所在的目录。如果不设置，Redis会把数据文件放在启动Redis服务器的目录下。

#### dbfilename filename

指定Redis数据文件名。默认的文件名为dump.rdb。

#### loglevel debug|verbose|notice|warning

指定日志级别。debug表示打印出非常详细的信息，包括性能消耗，网络通信等；verbose表示打印出比较详细的信息，但可能含有一些私密信息；notice表示只打印重要的警告或者错误信息；warning表示只打印警告信息。默认的日志级别为notice。

#### logfile stdout|filename

指定日志文件路径。如果设置为stdout，则表示输出日志到标准输出；否则，表示输出日志到指定的文件。如果不设置，则不会创建日志文件。

#### pidfile filepath

指定pid文件路径。

#### include config_file(s)

包含其他配置文件。可以在同一配置文件中包含多个include选项，也可以使用多个配置文件。Redis会按顺序处理配置文件，因此如果出现重复的选项，后面的配置文件会覆盖前面的配置文件中的相同选项。

#### maxmemory bytes|megabytes|gigabytes

设置Redis的最大可用内存。当内存超过限定值时，Redis会先尝试清除旧的内容，再添加新内容，以便保持在最大限度内利用内存。Redis会优先回收空闲内存而不是立即释放内存。

#### maxmemory-policy volatile-lru|allkeys-lru|volatile-random|allkeys-random|volatile-ttl|noeviction

设置当内存超出限制时，Redis的内存淘汰策略。volatile-lru表示只淘汰最近最少使用（LRU）的过期键值对；allkeys-lru表示只淘汰最近最少使用（LRU）的键值对；volatile-random表示随机淘汰过期键值对；allkeys-random表示随机淘汰键值对；volatile-ttl表示优先淘汰即将过期的键值对；noeviction表示禁止淘汰任何键值对。

#### notify-keyspace-events KEA|Ex|Pn|Pw|Kg|Sr|To|Ll

设置键空间通知的事件类型。如果某些事件发生时，需要通知Redis，可以使用notify-keyspace-events选项。KEA指Key event: del,expire,del,hset,hdel,rpush,lpush,sadd,zadd等；Ex表示expired事件；Pn表示pattern事件，比如给某个键匹配的通配符发生变化等；Pw表示psubscribe事件；Kg表示del，flushdb，flushall，rename等ActionEvent，可以从数据库中获取相关数据；Sr表示syncronization事件，比如Sentinel节点重新连接等；To表示touch事件；Ll表示logfile rotation事件。

#### client-output-buffer-limit normal|replica  soft limit  hard limit  overflow action

设置Redis客户端输出缓冲区限制。对于写入操作频繁的客户端（比如在写日志），Redis可能会积累太多的输出缓冲区，导致客户端无法及时消费数据。为了防止这种情况，Redis可以设置输出缓冲区的限制。通常情况下，Redis的客户端输出缓冲区默认为16MB，如果数据量过大，则Redis会清空输出缓冲区，让客户端消费掉一些数据。但是，如果客户端的消费能力不能跟上Redis生产数据的速度，就会造成输出缓冲区越来越满，从而引发客户端阻塞甚至超时。

normal表示对于普通的客户端，输出缓冲区大小是16MB；replica表示对于副本客户端，输出缓冲区大小是64MB。soft limit表示软限制，hard limit表示硬限制，overflow action表示当超过硬限制时采取的措施。当硬限制被触发后，如果采用了block选项，则Redis会阻塞发送客户端请求，直到系统缓冲区有足够的空闲空间；如果采用了truncated选项，则Redis会删除旧的内容，只保留新的内容；如果采用了noeviction选项，则Redis直接返回错误信息，并拒绝执行命令。

#### slowlog-log-slower-than milliseconds

指定慢查询日志的时间阈值。如果某一条命令花费的时间超过指定的微秒数，则会被记录到慢查询日志里面。慢查询日志可以通过slowlog get N命令来获取最近N条慢查询日志。

#### slowlog-max-len number

指定慢查询日志的最大长度。当慢查询日志的长度超过指定的数量时，最早的日志会被自动删除。

#### latency-monitor-threshold milliseconds

指定监控延迟命令的执行时间阈值。如果一条命令执行的时间超过指定阈值，则Redis会记录这条命令的执行时间，并通过INFO commandstats命令统计延迟命令的执行时间。

#### hash-max-zipmap-entries entries|poweroftwo

设置最大的哈希表大小。Redis在哈希表中用压缩编码来节约内存，当哈希表的元素个数超过一定数量时，Redis会自动启用哈希表压缩编码。但是，由于压缩编码的开销，压缩后的哈希表大小也受到影响。

#### hash-max-zipmap-value valuebytes

设置哈希表压缩编码的最大阈值。只有字符串的长度大于或等于该值时，才会被哈希表压缩。

#### list-max-ziplist-size size

设置最大的列表大小。Redis使用压缩列表来保存列表数据，当列表的元素个数小于列表平均大小的一半且列表数据总长度小于64字节时，Redis会使用压缩列表。压缩列表的优势是节省内存空间。

#### list-compress-depth depth

设置压缩列表的压缩深度。压缩深度决定了压缩列表压缩的程度，如果元素个数较少，可以设置为1；如果元素个数较多，可以适当增加压缩深度。

#### set-max-intset-entries entries

设置整数集合的最大元素个数。Redis可以将整数用集合的形式保存，对于较大的整数集合，用整数集合会节省内存空间。

#### zset-max-ziplist-entries entries

设置有序集合的最大元素个数。Redis使用压缩列表来保存有序集合数据，当有序集合的元素个数小于有序集合平均大小的一半且有序集合数据总长度小于64字节时，Redis会使用压缩列表。压缩列表的优势是节省内存空间。

#### zset-max-ziplist-value valuebytes

设置有序集合压缩编码的最大阈值。只有元素的成员长度大于或等于该值时，才会被压缩。

#### activerehashing yes|no

激活重置哈希槽的功能。当设置为yes时，如果某个key的Slot发生了变化，Redis会自动地将key移动到新的Slot。

#### lazyfree-lazy-eviction yes|no

指定是否启用延迟释放内存功能。当设置为yes时，如果有过期的Key，则不会立即释放内存，而是将其标记为待释放。待释放的Key在过期时间到了之后才会被真正释放。

#### slave-priority n

指定副本的优先级。Redis Sentinel可以根据优先级来选择同步或从库。优先级越低的副本，同步所需时间越长。

#### min-slaves-to-write num|min

指定至少有多少个副本处于可写状态才能接受写命令。min表示至少有一个副本处于可写状态。

#### min-slaves-max-lag milliseconds

指定最小的从库延迟时间。Redis Sentinel可以根据延迟时间来判定主从关系是否正常。

#### requirepass password

指定访问密码。如果配置了访问密码，客户端必须通过AUTH命令提供密码才可以执行命令。

#### rename-command oldname newname

重命名Redis的内部命令。通过该选项，可以自定义Redis的内部命令的名称。

### 深入理解Redis配置

#### AOF持久化

AOF持久化全称Asynchronous Append Only File，是Redis的持久化机制之一。Redis的AOF持久化机制就是将Redis的命令追加到文件末尾，每条命令都以Redis协议的格式保存。Redis重启的时候，会通过恢复文件中的命令来重新构建数据结构。

默认情况下，Redis只开启AOF持久化功能，没有RDB持久化功能。如果想同时开启AOF和RDB两种持久化机制，可以配置appendonly yes和save ""。如果想要只开启AOF持久化功能，可以配置appendonly yes。如果只是想临时关闭AOF持久化功能，可以配置appendonly no。

AOF持久化主要解决的问题是数据完整性问题。因为Redis是采用了Write Once Read Many的模式，数据只允许写入一次，但却可以被任意多个客户端读取。AOF持久化可以记录每次对数据的修改，包括增删改操作。只要还原出之前的状态，就可以回复到之前的状态，确保数据完整性。

#### RDB持久化

RDB持久化全称Redis DataBase，是Redis的持久化机制之一。RDB持久化机制就是定期将Redis的内存数据以快照的方式写入磁盘，以实现灾难恢复的目的。

Redis默认不会开启RDB持久化功能。如果需要使用RDB持久化功能，可以在redis.conf文件里通过配置save或者bgsave指令来设置RDB的持久化策略。save指令用于配置固定时间间隔进行RDB持久化操作，bgsave指令用于在后台异步地执行RDB持久化操作。

RDB持久化的原理很简单，就是周期性地将Redis的所有数据以二进制文件的形式保存到磁盘。也就是说，每次RDB持久化操作都会生成一个新的快照文件，不同的快照之间只有命令集不同，具体的差异可以通过Diff工具进行分析。由于RDB持久化机制的原因，Redis在故障时，可以将损坏的数据集恢复到最新状态，大大减少了数据恢复的时间。

#### 主从复制

Redis的主从复制机制是一种数据冗余的手段。当一个Redis服务器以主服务器的身份运行时，它会把自己的数据全量同步给从服务器。从服务器接收到数据后，可以对外提供读请求或者写请求。主从复制可以实现读写分离，既可以提高读性能，又可以避免单点故障。

当一个主服务器宕机时，可以快速将其中一个从服务器升级为新的主服务器，继续提供服务。还可以设置多个从服务器以提高读性能。除了传统的主从复制外，Redis还提供了Sentinel模式，Sentinel模式可以更好地管理Redis的主从复制。Sentinel模式可以实现主从复制的自动ailover，也就是说，当主服务器宕机时，Sentinel会自动选择一个从服务器，将其升级为新的主服务器，保证服务的连续性。

#### 哨兵模式

Sentinel模式是Redis高可用模式之一，可以实现Redis服务器的自动故障转移。Sentinel是基于Redis集群来实现的。Redis集群是由多个主从节点组成的一个分布式集群，而Sentinel则是一个独立的进程，它独立地运行，发现并监控Redis主从节点的状态。

Sentinel模式中，Redis集群的每个主节点都配置了Sentinel。当一个主节点出现问题时，Sentinel可以检测出来，并通过API向应用方返回错误信息，通知应用程序切换到另一个主节点。通过Sentinel模式实现的主从自动切换，可以避免单点故障带来的风险。

#### 内存管理机制

Redis在内存管理方面做了大量工作。首先，它使用了内存池技术，将分配的内存划分成特定大小的块，然后管理这些块。这样可以减少内存碎片的产生，提升内存利用率。其次，它采用了惰性删除和定期删除两种删除策略。惰性删除不会立即将已删除的键值对从内存中删除，而是使用标记和删除的方案，等到内存不足时再一次性删除。定期删除则是每隔一段时间（默认10分钟）执行一次内存检查，扫描那些过期的键值对并删除。

最后，Redis采用了对象缓存技术，在内存中缓存对象的副本。这样可以避免频繁地从磁盘加载对象，降低了访问对象的延迟。

#### 客户端链接

Redis支持长连接，通过长连接可以有效地节省客户端和Redis之间的TCP连接建立和断开的时间。Redis还支持客户端的输出缓冲区，可以对输出流量进行限制。

Redis对客户端连接进行了限制，通过timeout配置项可以设置客户端闲置时间，超过这个时间还没收到命令，则断开连接。Redis对客户端的输出流量进行限制，通过client-output-buffer-limit配置项可以控制客户端输出缓冲区的大小和行为。

#### 数据结构

Redis支持丰富的数据结构，包括字符串、散列、列表、集合、有序集合等。对于字符串数据类型，它支持动态扩容，扩容时只申请新的内存，不拷贝老的数据。对于哈希数据类型，它采用的是压缩链表法来优化存储效率。列表和集合分别采用的是双向链表和哈希表的方式，提供了高效的插入和删除操作。有序集合则采用的是跳跃表（Skip List）数据结构，它以链表的形式组织元素，并且元素按照大小排列。

Redis的哈希数据类型可以使用ziplist、hashtable和dict三种编码方式，其中ziplist是压缩列表，在元素个数小于512个时采用，可以节约内存空间；hashtable是哈希表，在数据比较均匀时采用，查找速度更快；dict是字典，是在hashtable基础上的进一步优化。

对于Redis的链表结构来说，Redis在插入和删除数据时，都会重新构造整个链表，这样对于性能会有一定的影响。所以，在Redis中，建议尽量避免在头部或者尾部频繁插入和删除数据，而是选择合适的插入位置。