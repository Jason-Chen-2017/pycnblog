
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Redis 是当下最火热的开源 NoSQL 数据库之一，它提供了强大的键值对存储能力，支持多种数据结构类型（字符串、哈希表、列表、集合、有序集合），并通过内部功能实现了强大的持久化、事务处理等功能。

作为一个高性能的内存数据库，Redis 在大规模集群环境中表现突出，被广泛应用于缓存、消息队列和搜索引擎等领域。

在很多场景下，使用 Redis 的命令行客户端可以代替 Web 界面进行管理，提升效率。比如，定时任务调度可以使用 redis-cli 来设置，而对于复杂查询或数据统计分析则需要直接操作 Redis 数据。

本文主要介绍 Redis 的命令行接口，如何安装及启动，以及提供的不同命令及其用法。希望能够帮助读者更好的了解和使用 Redis 。

# 2.安装与启动
## 安装
Redis 可以通过源代码编译安装，也可以下载预编译好的二进制版本安装，或者通过 Docker 或其他云服务快速部署。

为了方便演示，我们将安装最新版本的 Redis ，执行以下命令即可安装最新版 Redis 到 Ubuntu 操作系统上：

```bash
sudo apt update && sudo apt install -y build-essential tcl
wget http://download.redis.io/releases/redis-6.0.9.tar.gz
tar xzf redis-6.0.9.tar.gz
cd redis-6.0.9
make
```

以上命令会下载最新版源码包，解压后编译出可执行文件 redis-server 和 redis-cli ，存放在 src 目录下。

## 启动
默认情况下，Redis 只监听本地主机的 6379 端口，可以通过指定配置文件启动：

```bash
./redis-server /path/to/redis.conf
```

其中 `/path/to/redis.conf` 是 Redis 配置文件路径，一般默认为 `redis.conf`。

也可以直接启动 Redis 服务：

```bash
./redis-server --daemonize yes
```

执行完该命令后，Redis 服务就会以后台模式运行，并监听本地的 6379 端口。

此外，还可以在运行时修改配置参数：

```bash
./redis-server --port 6380 --bind 192.168.0.100 --maxmemory 1gb
```

其中 `--port` 指定 Redis 监听的端口号，默认为 6379；`--bind` 指定 Redis 绑定哪个 IP 地址，默认为 127.0.0.1；`--maxmemory` 设置最大可用内存，默认为 0 表示无限制。

## 测试
启动完成后，可以测试 Redis 是否正常工作：

```bash
./redis-cli ping
PONG
```

如果看到 PONG ，说明 Redis 已正常运行。

# 3.命令介绍

Redis 支持丰富的命令，包括基础的 KEY-VALUE 操作指令、LIST 指令、HASH 指令、SET 指令、ZSET 指令、发布订阅指令、脚本指令、服务器指令等。

本文主要介绍 Redis 的命令行接口所提供的一些常用的命令。

## SET、GET

SET 命令用于添加、修改字符串类型的 KEY-VALUE 对，语法如下：

```bash
SET key value [EX seconds] [PX milliseconds] [NX|XX]
```

* `key` : 要设置的 KEY 。
* `value` : 要设置的值。
* `[EX seconds]` : 为过期时间，单位为秒。
* `[PX milliseconds]` : 为过期时间，单位为毫秒。
* `[NX|XX]` : 如果设置了 NX 参数，只有 name 不存在的时候才进行设置操作；如果设置了 XX 参数，只有 name 存在的时候才进行设置操作。

示例：

```bash
redis> SET mykey "hello world"
OK
redis> GET mykey
"hello world"
redis> SET mykey "goodbye" EX 5 # 设置有效期为 5 秒
OK
redis> GET mykey
"goodbye"
redis> TTL mykey
4
redis> SET myotherkey "foo bar" NX # 如果 myotherkey 不存在，才进行设置操作
OK
redis> SET myotherkey "bar baz" XX # 如果 myotherkey 存在，才进行设置操作
(error) ERR no such key
```

GET 命令用于获取指定 KEY 的值，语法如下：

```bash
GET key
```

示例：

```bash
redis> GET foo
(nil)
redis> SET foo "Hello World!"
OK
redis> GET foo
"Hello World!"
redis> DEL foo
(integer) 1
redis> GET foo
(nil)
```

DEL 命令用于删除指定的 KEY，语法如下：

```bash
DEL key1 [key2] [...]
```

示例：

```bash
redis> SET foo "hello"
OK
redis> SET bar "world"
OK
redis> MSET foo hello bar world
OK
redis> KEYS *
[b'foo', b'bar']
redis> DEL foo bar
(integer) 2
redis> KEYS *
[]
```

MSET 命令用于一次设置多个 KEY-VALUE 对，语法如下：

```bash
MSET key value [key value...]
```

示例：

```bash
redis> MSET foo hello bar world
OK
redis> MGET foo bar nonexist
['hello', 'world', None]
redis> KEYS *
[b'foo', b'bar']
redis> DEL foo bar
(integer) 2
redis> KEYS *
[]
```

## INCR、DECR

INCR 命令用于对指定 KEY 中的整数值做加 1 操作，语法如下：

```bash
INCR key
```

DECR 命令用于对指定 KEY 中的整数值做减 1 操作，语法如下：

```bash
DECR key
```

示例：

```bash
redis> SET counter 10
OK
redis> INCR counter
(integer) 11
redis> DECR counter
(integer) 10
```

注意：INCR 和 DECR 操作仅适用于整型数字。如果给定 KEY 中存储的值不是整数，则返回错误信息。

## APPEND、SUBSTR

APPEND 命令用于追加内容到指定的字符串类型的 KEY 上，语法如下：

```bash
APPEND key value
```

SUBSTR 命令用于从指定的字符串类型 KEY 中获取子串，语法如下：

```bash
SUBSTR key start end
```

* `start` : 子串起始位置，第一个字符索引值为 0 。
* `end` : 子串结束位置。

示例：

```bash
redis> SET mykey "Hello World"
OK
redis> APPEND mykey "!"
(integer) 13
redis> GET mykey
"Hello World!"
redis> SUBSTR mykey 6 11
"World"
```

## SADD、SPOP、SISMEMBER

SADD 命令用于向指定的集合添加元素，语法如下：

```bash
SADD key member1 [member2]...
```

* `key` : 集合名称。
* `memberN` : 要加入集合的成员。

SPOP 命令用于随机弹出集合中的一个元素，语法如下：

```bash
SPOP key
```

* `key` : 集合名称。

SISMEMBER 命令用于判断某个值是否是一个集合的成员，语法如下：

```bash
SISMEMBER key member
```

示例：

```bash
redis> SADD myset "apple" "banana" "cherry"
(integer) 3
redis> SISMEMBER myset "banana"
(integer) 1
redis> SPOP myset
"apple"
redis> SPOP myset
"cherry"
redis> SPOP myset
(nil)
redis> SISMEMBER myset "pear"
(integer) 0
```

## ZADD、ZREM、ZRANGE

ZADD 命令用于添加元素到有序集，语法如下：

```bash
ZADD key score1 member1 [score2 member2]...
```

* `key` : 有序集名称。
* `score1` : 元素分数。
* `member1` : 元素名称。

ZREM 命令用于移除有序集中的元素，语法如下：

```bash
ZREM key member1 [member2]...
```

* `key` : 有序集名称。
* `memberN` : 要移除的元素名称。

ZRANGE 命令用于获取有序集中指定区间内的元素，语法如下：

```bash
ZRANGE key start stop [WITHSCORES]
```

* `key` : 有序集名称。
* `start` : 起始索引值，范围在 0 到 -1 之间。
* `stop` : 结束索引值，范围在 0 到 -1 之间。
* `[WITHSCORES]` : 可选参数，以元组形式返回分数。

示例：

```bash
redis> ZADD myzset 1 "apple"
(integer) 1
redis> ZADD myzset 2 "banana"
(integer) 1
redis> ZADD myzset 3 "cherry"
(integer) 1
redis> ZRANGE myzset 0 -1 WITHSCORES
[b'apple', b'1', b'banana', b'2', b'cherry', b'3']
redis> ZREM myzset "banana"
(integer) 1
redis> ZRANGE myzset 0 -1 WITHSCORES
[b'apple', b'1', b'cherry', b'3']
```