
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## Memcached 和 Redis 的区别？
Memcached 是一款高性能的内存缓存系统，它支持多种数据结构，包括字符串、哈希表、列表、集合等，能够提供快速的访问。它的主要功能是在内存中存储键值对（key-value）数据，通过简单的 API 可以快速地存取这些数据。相比于其他类型的缓存系统，Memcached 更擅长处理简单的数据类型，如文本、图片、视频等，而且不需要设置过期时间，可以设置较大的内存容量。Redis 以其更为复杂的功能而闻名，支持丰富的数据类型，包括字符串、哈希表、列表、集合、有序集合等，能够有效地解决业务中的各种数据问题。

## 为什么要用 Memcached 或 Redis ？
由于快速的访问速度和可靠性要求，对于一些敏感的热点数据，或者需要经常读取的数据，可以使用 Memcached 或 Redis 来缓存。在大型网站或应用服务器上，将热点数据缓存在内存中可以明显提升访问效率。如果访问请求频繁到数据库，那么将热点数据缓存到 Redis 中可以减少数据库的负载，从而提升网站的整体响应速度。除此之外，还可以通过队列技术实现实时通知，比如用户评论、购物车更新等。

## Memcached 和 Redis 在哪些领域的应用比较广泛？
Memcached 和 Redis 都可以用于缓存数据，但它们最主要的区别在于应用范围。一般情况下，Memcached 只用于做临时缓存，例如 web 页面的静态资源文件、短信验证码等。Redis 则常用来作为持久化的缓存，如数据查询结果缓存、会话缓存、商品推荐缓存等。另外，Memcached 支持分布式部署，而 Redis 不支持。因此，当单机内存无法支撑缓存需求时，可考虑使用 Redis 作为集群模式进行部署。

# 2.Memcached 使用场景及原理
## 1)、Memcached的主要作用
Memcached 提供了一个基于内存的高速缓存服务。其主要作用是用来存储小块数据（如图片、视频、页面片段等），以提高网站的访问速度。这种技术的优点是快速读取、写入、删除，缺点是不能存储海量数据。所以，Memcached 只适合那些缓存相对固定的、不经常变动的数据，否则效率可能会降低。例如，Memcached 可以缓存最近登录的用户信息、热门电影的前五名榜单等。

## 2)、Memcached工作原理
Memcached 将数据保存在内存中，利用LRU（least recently used）算法对内存中的缓存进行回收。这里的缓存指的是内存中的存储空间，而不是磁盘上的硬盘缓存。如果需要获取某个数据，Memcached 会先检查这个数据是否已经在缓存中。如果没有的话，Memcached 会从内存中加载这个数据并返回给客户端。如果缓存中没有，Memcached 会联系相应的服务器来获取这个数据。

## 3)、Memcached 安装配置
Memcached 有两个版本，分别是 memcached 1.x 和 memcached 2.x 。本教程使用的 memcached 版本是 1.4.13 ，安装完成后，就可以使用它了。

1、下载 Memcached 

到 http://memcached.org/downloads 上下载最新版本的 Memcached ，目前最新版是 1.4.13 。

2、安装 Memcached

将下载好的压缩包上传至服务器，解压后进入目录执行：

```
./configure --prefix=/usr/local/memcached # 指定安装路径
make && make install
```

3、启动 Memcached 服务

```
/usr/local/memcached/bin/memcached -d # 参数 -d 表示后台运行
```

这样就启动了 Memcached 服务，默认监听 11211 端口，可以通过 telnet 测试一下：

```
telnet localhost 11211
```

输入 quit 退出 telnet 连接。

## 4)、Memcached 操作命令
Memcached 提供了一些操作命令，可以对缓存数据进行增删改查。

### 设置（set）数据
设置缓存数据，语法如下：

```
set key value exptime
```

- key: 数据的唯一标识符
- value: 数据的值
- exptime (optional): 缓存数据的过期时间（单位：秒）。如果不指定，默认为0，表示永不过期。

示例：

```
set username admin 1800   //设置用户名admin的缓存数据，缓存时间为30分钟
set message Hello World    //设置消息“Hello World”的缓存数据
```

### 获取（get）数据
获取缓存数据，语法如下：

```
get key
```

- key: 数据的唯一标识符

示例：

```
get username      //获取用户名admin的缓存数据
get message       //获取消息“Hello World”的缓存数据
```

### 删除（delete）数据
删除缓存数据，语法如下：

```
delete key
```

- key: 数据的唯一标识符

示例：

```
delete username     //删除用户名admin的缓存数据
delete message      //删除消息“Hello World”的缓存数据
```

### 清空（flush_all）缓存
清空整个 Memcached 的缓存，语法如下：

```
flush_all
```

示例：

```
flush_all          //清空整个 Memcached 的缓存
```

# 3.Redis 基本概念及特点
## 1)、Redis概述
Redis 是完全开源免费的，遵守BSD协议，是一个高性能的key-value数据库。它支持数据的持久化，可选的Master-slave复制模型，发布订阅模式，LUA脚本，流水线，事务等。Redis内置了丰富的数据结构，支持多种数据类型的读写，并提供对Redis所有操作的原子性支持。除此之外，Redis还有连接池管理器，客户端分区等功能。Redis是当前 NoSQL（Not Only SQL，非关系数据库）方向最热门的排行第一的方案。

## 2)、Redis能干什么
Redis 可以做很多事情，具体来说：

1. 数据库：Redis 是一种高级键值对数据库，支持字符串，哈希，列表，集合，有序集合，位图，HyperLogLog 等数据类型。Redis 支持多种类型数据之间的交互，并支持同时多个客户端访问同一个数据。

2. 缓存：Redis 提供了一个功能强大的缓存系统。你可以把数据库的热点数据缓存到内存中，再由内存提供速度快的访问。Redis 的持久化功能可以让你在服务器故障之后仍然可以保留缓存数据，并且可以用复制机制实现缓存的共享。

3. 消息队列：Redis 的 List 数据类型提供了消息队列的功能。你可以把消息加入到 List 中，然后再从另一个进程中按顺序消费这些消息。

4. 分布式锁：Redis 提供了分布式锁功能。你可以在不同的 Redis 节点之间建立一系列的锁，只允许在一个节点上进行操作，从而避免多线程竞争导致的逻辑错误。

5. 计数器：Redis 提供计数器功能。你可以在内存中维护计数器，然后把它们发布到 Redis 中。其他客户端可以订阅这些计数器，获得实时的更新。

6. 排序：Redis 通过 SortedSet 数据类型支持排序。你可以把元素及其分数加入到 SortedSet 中，通过 scores 排序来获取按照分数排序后的元素。

除此之外，Redis 支持 Lua 脚本语言，可以让你运行自定义的命令。

## 3)、Redis和Memcached的区别
Redis 和 Memcached 都是高性能的 key-value 存储，但是它们有以下几个不同点：

1. 数据类型：Redis 支持丰富的数据类型，包括字符串，散列，列表，集合，有序集合和 HyperLogLog 等。Memcached 只支持简单的字符串键值。

2. 数据备份：Redis 支持主从同步，可以进行备份，意味着你可以拥有多个 Redis 服务器，每个服务器存储一部分数据。如果你的数据发生变化，只需要让 Redis 执行主服务器同步，其他的服务器则可以获取实时数据。Memcached 不支持备份。

3. 分配策略：Redis 可以根据配置项自动分配内存，虽然性能上优于 Memcached，但却不够灵活。Memcached 也可以通过调整预留内存的方式来限制内存的使用。

4. 多样的客户端：除了常用的 Python，Java，C# 等语言，Redis 支持众多编程语言，如 Ruby，JavaScript，PHP，Perl，Tcl 等。Memcached 只有官方的客户端支持。

综合来说，选择 Redis 的原因可能是因为它支持更多的数据类型，支持数据的持久化，支持主从同步，有丰富的客户端支持，并且它的持久化机制使得它很适合用来做缓存和消息队列。