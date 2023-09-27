
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1什么是Redis？
Redis (Remote Dictionary Server) 是一种开源（BSD许可）的高性能键值对存储数据库。它可以用作数据库、缓存和消息中间件。Redis支持数据的持久化，可以将内存中的数据保存在磁盘上，并可通过 Redis 指令调出，还可以灵活地设置备份策略，实现灾难恢复和高可用性。除此之外，Redis 提供了许多特性，如发布/订阅、事务、过期等。Redis 的开发语言是 C 语言，但其 API 与 Python 和 Ruby 一样直观易懂。由于 Redis 支持主从同步机制，因此它可用于构建可伸缩的分布式缓存服务。

## 1.2为什么要使用Redis作为缓存？
缓存作为提升网站响应速度和降低后端负载的重要手段之一，也是现代 web 应用程序中最常用的技术。一般情况下，数据是从后端数据库查询并呈现给用户。当多个请求同时访问相同的数据时，可能造成性能瓶颈。这时候，就需要引入缓存来解决这一问题。缓存就是在内存中存储部分数据副本，以方便下次访问。缓存的好处主要有以下几点：

1. 减少数据库查询次数，提升响应速度；
2. 将热门数据放在内存中，加快访问速度；
3. 分担服务器压力，避免单点故障带来的影响；
4. 缓存共享，降低数据源的负载；
5. 数据一致性，防止脏数据导致系统崩溃或错误；

## 1.3Redis可以用来做什么？
Redis 可以用来做很多方面的事情，包括：

1. 缓存 - 用作高速数据交换的存储器，用于临时的数据库查询和输出。比如网站首页上的新闻、产品列表、用户信息等；
2. 消息队列 - 用于实现异步任务的队列。例如，排队系统可以基于 Redis 来处理延迟的任务执行，提高系统整体处理能力；
3. 实时计数 - 以固定间隔更新计数器。例如，应用程序可以使用 Redis 来统计网站访问量，并在一定时间内展示给用户；
4. 分布式锁 - 通过基于 Redis 的分布式锁实现分布式资源互斥访问。例如，允许多台机器同时编辑同一个文档；
5. 群聊聊天室 - Redis 集群可以提供高并发的聊天应用。每条消息都可以缓存在 Redis 中，并利用集群功能进行分片；
6. 会话缓存 - 使用 Redis 可以有效地存储和检索用户 session 数据。因为 session 数据一般都是临时性的，且不应该保存在数据库中。Redis 提供了更好的性能和可靠性。
7. 购物车 - 可以用 Redis 来存储用户的购物车数据，在页面显示时直接从 Redis 获取数据快速显示。如果发生支付失败的情况，也只需简单修改 Redis 中的数据即可。
8. 其它缓存业务 - Redis 还有很多其它有用的功能，例如搜索结果缓存、反垃圾邮件、点击率统计、商品推荐等。这些功能都可以在不同的场景下发挥作用。

总而言之，Redis 可以用来做很多方面的事情，在不同的场景下发挥它的优势。因此，学习如何使用 Redis 作为缓存是一个很有价值的技能。

# 2.基本概念术语说明
## 2.1Redis数据结构
Redis 有五种数据类型：string(字符串类型)，hash(哈希类型)，list(列表类型)，set(集合类型)，zset(sorted set: 有序集合)。其中，String 类型是简单的 key-value 对，其他四个则是复杂的数据结构。

### 2.1.1 String 类型
String 是 Redis 最基础的数据类型，一个 String 类型的 value 最大不能超过 512M。String 可以是数字或者字符串，可以调用 Redis 自带的一些命令对 String 操作。String 类型的命令包括 SET 和 GET。如下面示例代码所示：

```python
redis> SET mykey "Hello"
OK # 设置成功
redis> GET mykey
"Hello" # 返回 Hello
```

### 2.1.2 Hash 类型
Hash 类型是一个 string 类型的 field 和 value 的映射表，所有的 field 都是唯一的，对字段和值分别做映射。Hash 可以用来存储对象，比如用户个人信息。Hash 的命令包括 HSET、HGET、HMGET、HDEL。如下面示例代码所示：

```python
redis> HSET user:1 username John Doe
(integer) 1 # 设置成功，返回被设置为 1 的 field 个数
redis> HSET user:1 age 29
(integer) 1 # 设置成功，返回被设置为 1 的 field 个数
redis> HGET user:1 username
"John Doe" # 返回指定 field 的 value
redis> HMGET user:1 age gender
1) "29" # 返回多个 field 的 value
2) "(nil)" # 如果没有找到对应的值，返回 nil
redis> HDEL user:1 age
(integer) 1 # 删除指定的 field 和 value，返回被删除的个数
```

### 2.1.3 List 类型
List 类型是一个有序的项列表，列表项可以重复。List 可以实现一个队列，非常适合用于队列的场景，可以获取或者删除指定范围的元素。List 的命令包括 LPUSH、RPUSH、LPOP、RPOP、LINDEX、LLEN、LTRIM。如下面示例代码所示：

```python
redis> RPUSH myqueue "item1"
(integer) 1 # 添加到右侧，返回列表长度
redis> RPUSH myqueue "item2"
(integer) 2 # 添加到右侧，返回列表长度
redis> LRANGE myqueue 0 -1
1) "item1" # 从左至右取出列表所有项
2) "item2"
redis> LPOP myqueue
"item1" # 删除并返回左侧第一项
redis> RPOP myqueue
"item2" # 删除并返回右侧第一项
redis> LINDEX myqueue 1
"item2" # 返回指定位置的项
redis> LLEN myqueue
(integer) 1 # 返回列表长度
redis> LTRIM myqueue 0 0
OK # 只保留第一个项
redis> LTRIM myqueue -1 -1
OK # 只保留最后一个项
```

### 2.1.4 Set 类型
Set 类型是一个无序集合，集合内不包含重复的元素，而且元素只能是 string 类型。Set 可以实现标签云这种功能，可以快速查找某个元素是否存在于集合内。Set 的命令包括 SADD、SREM、SCARD、SISMEMBER、SINTER、SUNION、SDIFF、SSCAN。如下面示例代码所示：

```python
redis> SADD myset "item1" "item2" "item3"
(integer) 3 # 添加三项到集合，返回添加的个数
redis> SREM myset "item2" "item4"
(integer) 1 # 从集合中移除两项，返回移除的个数
redis> SCARD myset
(integer) 2 # 返回集合长度
redis> SISMEMBER myset "item1"
(integer) 1 # 判断 item1 是否在集合中，存在返回 1，不存在返回 0
redis> SINTER myset otherset
... # 返回两个集合的交集
redis> SUNION myset otherset
... # 返回两个集合的并集
redis> SDIFF myset otherset
... # 返回第一个集合与第二个集合的差集
redis> SSCAN myset 0 MATCH item* COUNT 1000
... # 根据匹配模式搜索集合内的项，每次最多返回 1000 项
```

### 2.1.5 Zset 类型
Zset 类型是 String 类型和 Double 类型的集合，每个元素都会关联一个 double 类型的分值。Zset 可以根据分值对元素进行排序。Zset 的命令包括 ZADD、ZSCORE、ZRANK、ZREVRANK、ZINCRBY、ZCARD、ZCOUNT、ZRANGE、ZREM、ZLEXCOUNT、ZRANGEBYLEX、ZREMRANGEBYLEX。如下面示例代码所示：

```python
redis> ZADD myzset 1 "item1" 2 "item2" 3 "item3"
(integer) 3 # 添加三项到 zset，并给每个项赋值分值，返回添加的个数
redis> ZSCORE myzset "item2"
"2" # 返回指定项的分值
redis> ZRANK myzset "item2"
(integer) 1 # 返回指定项的索引，以 0 为起始索引
redis> ZREVRANK myzset "item2"
(integer) 1 # 返回指定项的倒序索引，以 0 为起始索引
redis> ZINCRBY myzset 1 "item2"
"3" # 修改指定项的分值，并返回新的分值
redis> ZCARD myzset
(integer) 3 # 返回集合内元素数量
redis> ZCOUNT myzset 1 2
(integer) 2 # 返回指定分值区间内元素数量
redis> ZRANGE myzset 0 -1 WITHSCORES
1) "item1"
2) "1"
3) "item2"
4) "3"
5) "item3"
6) "2"
redis> ZREM myzset "item2"
(integer) 1 # 删除指定项，返回被删除的个数
redis> ZLEXCOUNT myzset [aaa (zzz]
(integer) 2 # 返回指定区间内元素数量
redis> ZRANGEBYLEX myzset [a [c
1) "item1"
2) "item2"
redis> ZREMRANGEBYLEX myzset [b [c
(integer) 1 # 删除指定区间内元素，返回被删除的个数
```

## 2.2Redis网络模型
Redis 是一个典型的 TCP Socket 服务。客户端与 Redis 的通信使用 RESP (REdis Serialization Protocol) 协议，RESP 协议是 Redis 自己定义的一种数据序列化格式。Redis 网络模型由以下几个部分组成：

### 2.2.1 命令请求
客户端向 Redis 发出命令请求，通常是以纯文本形式发送。命令请求一般由四部分组成：

1. 请求类型（如：SET、GET、HSET、LPUSH）。
2. 待操作的 Key 或多个 Keys 。
3. 待操作的参数。
4. 命令选项（如：EX=60表示设置缓存过期时间为60秒）。

### 2.2.2 命令回复
Redis 对命令请求进行解析，得到待操作的 Key 或多个 Keys ，然后在后台执行相关操作，并将执行结果以二进制形式编码，并通过 TCP Socket 发送给客户端。编码后的执行结果采用 RESP 协议，即 Redis 自己的自定义协议。

### 2.2.3 连接建立
Redis 服务端接收到命令请求后，创建相应连接，并与客户端完成 TCP Socket 通讯。

### 2.2.4 授权验证
Redis 提供了安全认证功能，要求客户端必须通过 AUTH 指令提供密码才能发出命令请求。

## 2.3Redis应用场景

### 2.3.1 缓存
缓存可以提升网站访问速度，让用户获得更流畅的体验。缓存的应用场景非常广泛。例如，对于经常访问的数据（如：商品详情页），可以先把数据缓存到 Redis 中，再直接返回给用户，避免了频繁访问数据库。对于静态资源（如：图片、CSS、JS 文件），也可以通过缓存的方式加快用户访问速度。

### 2.3.2 计数器
Redis 可以用作计数器，例如，网站每日新增用户数、网站活跃用户数、评论数、推送通知数等。可以通过 INCR 和 DECR 命令对计数器进行增减操作，并设定过期时间，使计数器自动清零。

### 2.3.3 消息队列
Redis 也可以用作消息队列。比如，可以使用 Redis 的 LIST 数据结构实现消息队列。生产者进程可以把消息放入 Redis 的 LIST 中，消费者进程则可以从 LIST 中获取消息进行处理。LIST 是双向链表，最新消息插入的右侧，最老消息插入的左侧。

### 2.3.4 分布式锁
为了保证不同服务之间的资源访问的正确性，需要实现分布式锁功能。Redis 在多个节点之间提供了分布式锁功能，通过获取锁和释放锁的操作实现资源的独占访问。

### 2.3.5 限流
Redis 也可以用来限制用户的请求频率，限制访问接口的请求频率。例如，在短时间内对同一个用户发出大量请求是比较危险的，可能会造成服务器压力过大甚至导致宕机，因此可以通过 Redis 的计数器和窗口机制实现用户请求的限流。

# 3.核心算法原理和具体操作步骤以及数学公式讲解

## 3.1原子性和事务
Redis 本身支持事务，事务可以一次执行多个命令，从而确保数据一致性。Redis 的事务具有原子性，事务中的所有命令都会被串行执行。如果事务中的某一条命令执行失败，那么整个事务都不会被执行，也就是说，事务是一个不可分割的操作序列。

事务的开始标记是一个 MULTI 关键字，事务的结束标记是一个 EXEC 关键字。MULTI 执行之后进入事务模式，此时 Redis 不直接执行事务中请求的命令，而是将它们保存到一个事务队列里。若在事务队列里的所有命令都被执行，则执行 EXEC 命令，执行事务。否则，只对已经处于事务模式下的客户端进行回滚操作。

举例如下：

```python
redis> MULTI # 开启事务
OK
redis> SET foo bar # 队列命令1
QUEUED
redis> DEL foobar # 队列命令2
QUEUED
redis> EXEC # 执行事务，成功
(error) NO such key 'foobar'
redis> get foo # 查看 foo 的值
"bar"
```

## 3.2过期失效键自动删除策略
Redis 提供两种过期策略：惰性删除和定期删除。

### 3.2.1惰性删除策略
Redis 默认是使用惰性删除策略。当执行 EXPIRE 命令设置过期时间时，只是设置了一个过期事件，并不会立刻对该键进行过期处理，而是在下一次访问该键时才触发过期。这么做的目的是提高性能，不会每一次访问都耗费 CPU 去检查该键是否过期。

### 3.2.2定期删除策略
定期删除策略每隔一段时间（默认 10 分钟），Redis 会随机抽取一些设置了过期时间的键，检查其是否过期，如果过期的话，就自动删除。这个过程称为 RDB（Redis DataBase） saving。RDB 是一个 point-in-time 快照，保存了 Redis 当前时刻的数据状态。

RDB 每次启动时，会优先加载 RDB 文件，也就是最新生成的 RDB 文件。RDB 的缺点是丢失数据，意味着可能丢失一小部分数据。

定期删除策略能够保证数据的最终一致性，因为它会在某些极端情况下丢失数据。但是定期删除策略的执行频率比惰性删除策略的频率要低得多。所以 Redis 同时使用了两者，通过合理配置来选择。

## 3.3持久化
Redis 支持两种持久化方式：RDB 和 AOF。

### 3.3.1RDB持久化
RDB 是 Redis 数据的持久化方案。它是定时持久化操作，定期保存 Redis 的数据到磁盘。RDB 文件可以配置，保存的数据包括快照和指令，快照即当前时刻的 Redis 数据，指令记录 Redis 读写的命令。

RDB 加载时，Redis 会先读取保存的快照文件，然后再重演保存的指令来恢复数据，恢复之前的工作进度。这样可以保证数据完整性。

RDB 的优点是可以进行灾难恢复，只需最近的一个 RDB 文件就可以完全还原 Redis，而不需要依赖于磁盘上的数据。但是 RDB 缺点是需要停机，一旦停止服务，Redis 就无法再接受命令。

### 3.3.2AOF持久化
AOF（append only file）是 Redis 的另一种持久化方案。相对于 RDB，AOF 使用追加日志的方式记录所有对数据库的写入操作，然后同步到硬盘。AOF 可以配置，保存的数据包括指令，跟 RDB 类似。

AOF 的优点是可以在保证数据完整性的前提下，增加持久化层，即使服务器宕机，之前的写操作也不会丢失。AOF 启动时，优先加载 AOF 文件，从而可以较好的实现高可用。

AOF 的缺点是文件太大，长时间运行时，会占用更多的磁盘空间。另外，在保存大量数据时，AOF 的速度可能会慢于 RDB。

## 3.4主从复制
Redis 提供了主从复制功能，可以实现多个 Redis 实例的数据共享。

多个 Redis 实例之间采用主从复制，可以实现读写分离，一个主 Redis 负责写数据，多个从 Redis 负责读数据。

主从复制可以有三个阶段：

1. 复制偏移量：slave 连接 master 时，master 报告自己的复制偏移量。
2. 数据同步：master 持续收到 slave 的写入命令，并将其复制到其他 slave 上。
3. 后台重建 replication backlog：当 slave 数据丢失时，需要重新建立连接，主动与其他 slave 断开连接，通过 PSYNC 命令同步数据。

主从复制可以用于实现读写分离，在主库进行写操作时，其他从库可以帮助分担读操作的负载。

## 3.5哨兵机制
Redis Sentinel 可以实现 Redis 集群的监控，故障转移和通知。

Sentinel 的工作模式如下：

1. Sentinel 进程运行，监听集群里的 master 和 slave 。
2. 当 master node 发生故障切换时，sentinel 通知其他 sentinel ，其他 sentinel 根据配置决定投票给哪个 slave node 成为新的 master 。
3. 当 client 连接 master 时，通过 sentinel ，client 首先连接任意一个 sentinel ，向 sentinel 获取当前 master 地址，并向 master 进行命令请求。
4. 如果 master node 下线，sentinel 会选举出新的 master 。

## 3.6集群分布式技术
Redis Cluster 实现了 Redis 分布式集群。

Redis Cluster 分布式集群提供了一下几个功能：

1. 分区和复制：所有数据按照 slot （分片）的方式存储在多个节点上，通过主从结构实现数据复制。
2. 失效转移：当某个节点出现故障时，将它上的 slot 移动到其他节点，保持 Redis Cluster 的高可用性。
3. 管理工具：Redis Cluster 提供了 redis-cli 工具和 RedisInsight 管理界面，方便管理员查看集群信息和调试问题。

# 4.具体代码实例和解释说明
这里以 Python 客户端为例，编写代码实例，详细介绍 Redis 的各种数据类型及操作。

## 4.1String类型操作
### 4.1.1设置和获取
String 类型是 Redis 的最基本的数据类型，用 SET 和 GET 命令可以设置和获取值。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.set('name', 'Alice') # 设置 key 为 name 值为 Alice
print r.get('name').decode() # 获取 key 为 name 的值，并打印
```

输出：

```
Alice
```

### 4.1.2批量设置和获取
批量设置和获取可以使用 mset 和 mget 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

data = {'k1': 'v1', 'k2': 'v2', 'k3': 'v3'}
r.mset(data) # 批量设置 k-v
print r.mget(['k1', 'k2']).decode() # 批量获取 k1,k2 的值
```

输出：

```
['v1', 'v2']
```

### 4.1.3删除
删除操作可以使用 del 命令，删除指定 key。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.delete('name') # 删除名为 name 的 key
```

### 4.1.4过期时间
设置过期时间可以使用 expire 方法。

```python
import time
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.setex('temp_key', 10, 'temporary_value') # 设置 temp_key 的值并设置过期时间为 10s
print r.exists('temp_key') # 查询 key 是否存在，返回 True
time.sleep(10) # 等待过期
print r.exists('temp_key') # 查询 key 是否存在，返回 False
```

输出：

```
True
False
```

### 4.1.5查找和替换
查找和替换可以使用 replace 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.set('old_key', 'old_value') # 设置 old_key 的值为 old_value
print r.get('old_key').decode() # 获取 old_key 的值，返回 old_value
new_value = r.replace('old_key', 'new_value') # 替换 old_key 的值为 new_value，返回旧值 old_value
print new_value.decode() # 获取替换后的新值，返回 new_value
```

输出：

```
old_value
new_value
```

## 4.2Hash类型操作
Hash 类型是一个 string 类型的 field 和 value 的映射表，所有的 field 都是唯一的。

### 4.2.1设置和获取
设置和获取 hash 类型数据可以使用 hset 和 hget 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.hset('user', 'name', 'Alice') # 设置 user 的 name 属性值为 Alice
print r.hget('user', 'name').decode() # 获取 user 的 name 属性值，返回 Alice
```

输出：

```
Alice
```

### 4.2.2批量设置和获取
批量设置和获取可以使用 hmset 和 hmget 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

data = {'age': 29, 'gender': 'female'}
r.hmset('user', data) # 批量设置 user 的属性
print r.hmget('user', ['age', 'gender']) # 批量获取 user 的属性值，返回元组 ('29', 'female')
```

输出：

```
[u'29', u'female']
```

### 4.2.3删除
删除属性可以使用 hdel 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.hset('user', 'name', 'Bob') # 设置 user 的 name 属性值为 Bob
r.hdel('user', 'name') # 删除 user 的 name 属性
```

### 4.2.4获取属性列表
获取属性列表可以使用 hkeys 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

data = {'name': 'Carol', 'age': 30}
r.hmset('user', data) # 批量设置 user 的属性
print r.hkeys('user').decode() # 获取 user 属性列表，返回 ['name', 'age']
```

输出：

```
['name', 'age']
```

### 4.2.5获取属性值列表
获取属性值列表可以使用 hvals 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

data = {'name': 'David', 'age': 31, 'gender':'male'}
r.hmset('user', data) # 批量设置 user 的属性
print r.hvals('user').decode() # 获取 user 属性值列表，返回 ['David', '31','male']
```

输出：

```
['David', '31','male']
```

## 4.3List类型操作
List 类型是一个有序的项列表，列表项可以重复。

### 4.3.1设置和获取
设置和获取 list 类型数据可以使用 lpush 和 rpop 方法。

lpush 插入左边，rpop 弹出右边。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

for i in range(3):
    r.lpush('mylist', str(i)) # 插入到左侧
print r.lrange('mylist', 0, -1).decode() # 获取所有项，返回 ['2', '1', '0']
print r.rpop('mylist').decode() # 弹出右侧，返回 '0'
```

输出：

```
['2', '1', '0']
'0'
```

### 4.3.2删除
删除项可以使用 lrem 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

for i in range(3):
    r.lpush('mylist', str(i)) # 插入到左侧
r.lrem('mylist', 0, '1') # 删除值为 1 的所有项
print r.lrange('mylist', 0, -1).decode() # 获取所有项，返回 ['0', '2']
```

输出：

```
['0', '2']
```

### 4.3.3获取长度
获取长度可以使用llen方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

for i in range(3):
    r.lpush('mylist', str(i)) # 插入到左侧
print r.llen('mylist') # 获取长度，返回 3
```

输出：

```
3
```

### 4.3.4获取指定区间项
获取指定区间项可以使用 lrange 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

for i in range(5):
    r.lpush('mylist', str(i)) # 插入到左侧
print r.lrange('mylist', 0, 1).decode() # 获取第0到第1项，返回 ['4', '3']
print r.lrange('mylist', -2, -1).decode() # 获取倒数第2到倒数第1项，返回 ['1', '0']
```

输出：

```
['4', '3']
['1', '0']
```

### 4.3.5截断列表
截断列表可以使用ltrim方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

for i in range(5):
    r.lpush('mylist', str(i)) # 插入到左侧
r.ltrim('mylist', 0, 2) # 截断列表，只保留第0到第2项
print r.lrange('mylist', 0, -1).decode() # 获取所有项，返回 ['4', '3', '2']
```

输出：

```
['4', '3', '2']
```

### 4.3.6队列操作
队列操作可以模拟栈和队列。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.lpush('stack', 'apple') # 插入 apple 到栈顶
r.lpush('queue', 'banana') # 插入 banana 到队列头部
print r.rpop('stack').decode() # 弹出栈顶，返回 apple
print r.lpop('queue').decode() # 弹出队列头部，返回 banana
```

输出：

```
apple
banana
```

## 4.4Set类型操作
Set 类型是一个无序集合，集合内不包含重复的元素。

### 4.4.1添加成员
添加成员可以使用 sadd 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.sadd('myset', 'a', 'b', 'c') # 添加 a b c 到集合 myset
```

### 4.4.2删除成员
删除成员可以使用 srem 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.sadd('myset', 'a', 'b', 'c') # 添加 a b c 到集合 myset
r.srem('myset', 'b', 'c') # 删除 b c 项
```

### 4.4.3获取长度
获取长度可以使用 scard 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.sadd('myset', 'a', 'b', 'c') # 添加 a b c 到集合 myset
print r.scard('myset') # 获取集合长度，返回 3
```

输出：

```
3
```

### 4.4.4判断成员是否存在
判断成员是否存在可以使用 sismember 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.sadd('myset', 'a', 'b', 'c') # 添加 a b c 到集合 myset
print r.sismember('myset', 'a') # 判断 a 是否属于集合 myset，返回 True
print r.sismember('myset', 'd') # 判断 d 是否属于集合 myset，返回 False
```

输出：

```
True
False
```

### 4.4.5求交集和并集
求交集和并集可以使用 sinter 和 sunion 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.sadd('set1', 'a', 'b', 'c') # 创建集合 set1
r.sadd('set2', 'b', 'c', 'd') # 创建集合 set2
print r.sinter('set1','set2').decode() # 求交集，返回 ['b', 'c']
print r.sunion('set1','set2').decode() # 求并集，返回 ['a', 'b', 'c', 'd']
```

输出：

```
['b', 'c']
['a', 'b', 'c', 'd']
```

### 4.4.6求差集
求差集可以使用 sdiff 方法。

```python
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.sadd('set1', 'a', 'b', 'c') # 创建集合 set1
r.sadd('set2', 'b', 'c', 'd') # 创建集合 set2
print r.sdiff('set1','set2').decode() # 求差集，返回 ['a']
```

输出：

```
['a']
```

### 4.4.7随机获取成员
随机获取成员可以使用 spop 方法。

```python
import random
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

r.sadd('myset', 'a', 'b', 'c') # 添加 a b c 到集合 myset
rand_member = r.spop('myset') # 随机获取集合中的一个项
if rand_member is not None:
    print rand_member.decode() # 打印随机项
else:
    print 'Set is empty.' # 如果集合为空，打印提示信息
```

输出：

```
e.g.: 'b'
```