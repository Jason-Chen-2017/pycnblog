
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Redis是一个开源的高性能键值对数据库。它支持多种类型的数据结构，包括字符串、哈希表、列表、集合、有序集合等。Redis提供了许多数据结构的操作命令，用于设置键值对，获取键值，删除键值对，数据排行等。本文主要介绍Redis中的5个基础命令——String（字符串）、Hash（散列）、List（列表）、Set（集合）、Sorted Set（有序集合）。并且结合一些使用场景，详细阐述每个命令的用法和优缺点。

# 2. 背景介绍
Redis是一个开源的高性能键值对数据库，它支持丰富的数据结构，如字符串、哈希表、列表、集合、有序集合。Redis提供了丰富的接口操作命令，可以通过命令实现数据的读写，缓存处理，消息队列，计数器，集群管理，发布/订阅等功能。在实际的应用中，redis可以用来作为缓存，数据库的一种非关系型数据库，通过键值对存储。Redis具有如下优点:

1. 速度快
2. 支持丰富的数据结构
3. 可用于缓存，消息队列，计数器等
4. 易于操作

# 3. 基本概念术语说明

1. 数据类型
Redis支持五种基本的数据类型：STRING（字符串），HASH（散列），LIST（列表），SET（集合），SORTED SET（有序集合）。

2. 连接redis
Redis客户端可以通过不同的方式连接Redis服务器，如客户端连接本地Redis服务端或者远程Redis服务端，通过配置Redis的启动参数也可以指定是否允许远程连接。

3. Key
每一个key都是一个二进制序列，用于唯一标识数据库中的数据对象，键的命名规则遵循一定约定，可以使用ASCII字符集，不能包含空格或特殊字符，最大512字节。
通常情况下，为了保证键值的唯一性，通常在设计时会添加业务前缀或其他信息，以便区分不同类型的键值，降低发生冲突的可能性。

4. Value
与Key相对应，Value也是一个二进制序列，通常表示数据的值。不同于Key，Value可以包含任何东西，比如一个数字，一个字符串，一个复杂的结构体等。但是对于某些特定的数据结构，比如hash和list，Value只能是某种特定格式的。

5. Expire时间
过期时间指的是键值对在Redis中的存活时间，过期后将被自动删除。用户可以调用EXPIRE命令或者配置文件设置超时时间。当Redis中的键值过期之后，如果没有再次被访问，则不会被加载到内存中。

# 4. String（字符串）命令
## 设置String类型
String数据类型是最基本的数据类型，即一个key对应一个value。对String类型的set/get/delete命令进行简单介绍：

```
set key value [EX seconds|PX milliseconds] [NX|XX]
```

- `set` 命令用来设置键值对；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `value` 是需要保存的数据；
- `[EX seconds]` 指定键的过期时间(秒)，过期之后数据就无法获取，但依然占用内存空间；
- `[PX milliseconds]` 指定键的过期时间(毫秒)，过期之后数据就无法获取，但依然占用内存空间；
- `[NX|XX]` NX 表示只有 name 不存在的时候才设置 value， XX 表示只有 name 存在的时候才设置 value，不设置默认为 NX 模式。

```
get key
```

- `get` 命令用来获取键值对；
- `key` 参数是键名，用于获取对应的value值。

```
del key [key...]
```

- `del` 命令用来删除键值对；
- `key` 参数是键名，用于删除指定的键值对。

```
strlen key
```

- `strlen` 命令用来获取string类型值的长度；
- `key` 参数是键名，用于获取对应的value值的长度。

## 获取、设置String类型的数据

```
redis> set name "runoob"
OK
redis> get name
"runoob"
```

# 5. Hash（散列）命令
## 设置、获取、删除Hash类型

```
hset key field value
```

- `hset` 命令用来设置散列数据类型；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `field` 参数是字段名，每个字段都是一个字符串，用来标识唯一的属性名；
- `value` 是需要保存的数据。

```
hmset key field1 value1 field2 value2...
```

- `hmset` 命令是 hmset 命令的一个批量版本；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `field` 参数是字段名，每个字段都是一个字符串，用来标识唯一的属性名；
- `value` 是需要保存的数据。

```
hget key field
```

- `hget` 命令用来获取散列数据类型；
- `key` 参数是键名，用于获取指定的键值对；
- `field` 参数是字段名，用于获取指定属性的值。

```
hkeys key
```

- `hkeys` 命令用来获取所有字段名；
- `key` 参数是键名，用于获取指定对象的所有字段名。

```
hvals key
```

- `hvals` 命令用来获取所有属性值；
- `key` 参数是键名，用于获取指定对象的所有属性值。

```
hlen key
```

- `hlen` 命令用来获取属性数量；
- `key` 参数是键名，用于获取指定对象的属性数量。

```
hexists key field
```

- `hexists` 命令用来判断是否存在某个字段；
- `key` 参数是键名，用于获取指定对象的指定字段是否存在。

```
hdel key field [field...]
```

- `hdel` 命令用来删除一个或多个字段；
- `key` 参数是键名，用于删除指定对象的一个或多个字段。

## 示例

```
redis> hset user_info name "runoob" age "18" gender "male"
(integer) 2
redis> hgetall user_info
 1) "name"
 2) "runoob"
 3) "age"
 4) "18"
 5) "gender"
 6) "male"
```

# 6. List（列表）命令
## 设置、获取、删除List类型

```
lpush key value [value...]
```

- `lpush` 命令用来插入元素到列表的左侧；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `value` 是需要保存的数据。

```
rpush key value [value...]
```

- `rpush` 命令用来插入元素到列表的右侧；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `value` 是需要保存的数据。

```
llen key
```

- `llen` 命令用来获取列表长度；
- `key` 参数是键名，用于获取指定对象的列表长度。

```
lrange key start stop
```

- `lrange` 命令用来获取列表中指定范围内的元素；
- `key` 参数是键名，用于获取指定对象的指定范围内的元素；
- `start` 是起始位置（从0开始计数），`-1` 为列表尾部；
- `stop` 是结束位置（从0开始计数），`-1` 为列表尾部。

```
linsert key before|after pivot value
```

- `linsert` 命令用来在列表指定元素之前或之后插入元素；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `before` 或 `after` 是关键字，表示在哪边插入；
- `pivot` 是指定元素，表示要在这个元素之前或之后插入；
- `value` 是需要插入的数据。

```
ltrim key start stop
```

- `ltrim` 命令用来截取列表指定范围内的元素；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `start` 是起始位置（从0开始计数），`-1` 为列表尾部；
- `stop` 是结束位置（从0开始计数），`-1` 为列表尾部。

```
lrem key count value
```

- `lrem` 命令用来删除列表中等于给定值的元素；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `count` 是删除个数，`-1` 为全部删除；
- `value` 是指定值。

```
lindex key index
```

- `lindex` 命令用来根据索引获取元素；
- `key` 参数是键名，用于获取指定对象的指定索引位置元素；
- `index` 是索引值（从0开始计数）。

```
lset key index value
```

- `lset` 命令用来修改列表中指定位置元素的值；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `index` 是索引值（从0开始计数），`-1` 为列表尾部；
- `value` 是需要保存的数据。

```
lpop key
```

- `lpop` 命令用来弹出列表第一个元素；
- `key` 参数是键名，用于弹出并返回指定对象的第一个元素。

```
rpop key
```

- `rpop` 命令用来弹出列表最后一个元素；
- `key` 参数是键名，用于弹出并返回指定对象的最后一个元素。

## 使用场景

- 消息队列，任务队列，操作日志记录，Leaderboards，微博好友推荐系统。

## 示例

```
redis> lpush mylist "world"
(integer) 1
redis> lpush mylist "hello"
(integer) 2
redis> rpush mylist "foo"
(integer) 3
redis> lrange mylist 0 -1
1) "hello"
2) "world"
3) "foo"
redis> ltrim mylist 1 -1
OK
redis> lrange mylist 0 -1
1) "world"
```

# 7. Set（集合）命令
## 添加元素到Set

```
sadd key member [member...]
```

- `sadd` 命令用来向集合中添加元素；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `member` 是待加入的元素。

## 删除元素从Set

```
srem key member [member...]
```

- `srem` 命令用来删除集合中的元素；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `member` 是待删除的元素。

## 查看集合成员

```
scard key
```

- `scard` 命令用来查看集合中元素的数量；
- `key` 参数是键名，用于获取指定对象的成员数量。

```
smembers key
```

- `smembers` 命令用来查看集合中的所有元素；
- `key` 参数是键名，用于获取指定对象的所有成员。

```
sismember key member
```

- `sismember` 命令用来查看集合中是否存在指定的元素；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `member` 是待查询的元素。

```
spop key
```

- `spop` 命令用来随机弹出集合中的一个元素；
- `key` 参数是键名，用于随机弹出指定对象的一个元素。

## 使用场景

- 共同关注，共同喜好，共同收藏，去重，交集。

## 示例

```
redis> sadd myset "one"
(integer) 1
redis> sadd myset "two"
(integer) 1
redis> sadd myset "three"
(integer) 1
redis> scard myset
(integer) 3
redis> smembers myset
1) "one"
2) "two"
3) "three"
redis> sismember myset "four"
(integer) 0
redis> spop myset
"three"
redis> smembers myset
1) "one"
2) "two"
```

# 8. Sorted Set（有序集合）命令
## 插入元素到Sorted Set

```
zadd key score1 member1 score2 member2...
```

- `zadd` 命令用来向有序集合中插入元素；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `score` 是成员的分值，可以是一个负数，用来排序；
- `member` 是待加入的成员。

## 根据权重移除元素

```
zrem key member [member...]
```

- `zrem` 命令用来从有序集合中移除元素；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `member` 是待移除的成员。

## 根据权重获取元素

```
zrange key start stop [withscores]
```

- `zrange` 命令用来按照分值排序获取有序集合中的元素；
- `key` 参数是键名，用于获取指定对象的元素；
- `start` 和 `stop` 分别表示开始位置和结束位置；
- `withscores` 表示带上分值，默认只显示成员。

```
zrevrange key start stop [withscores]
```

- `zrevrange` 命令用来按照分值倒序获取有序集合中的元素；
- `key` 参数是键名，用于获取指定对象的元素；
- `start` 和 `stop` 分别表示开始位置和结束位置；
- `withscores` 表示带上分值，默认只显示成员。

```
zrank key member
```

- `zrank` 命令用来获取指定成员在有序集合中的排名；
- `key` 参数是键名，用于获取指定对象的成员排名；
- `member` 是待查询的成员。

```
zrevrank key member
```

- `zrevrank` 命令用来获取指定成员在有序集合中的排名（按分值倒序排列）；
- `key` 参数是键名，用于获取指定对象的成员排名；
- `member` 是待查询的成员。

```
zscore key member
```

- `zscore` 命令用来获取指定成员的分值；
- `key` 参数是键名，用于获取指定对象的成员分值；
- `member` 是待查询的成员。

```
zcard key
```

- `zcard` 命令用来获取有序集合中元素的数量；
- `key` 参数是键名，用于获取指定对象的元素数量。

## 修改元素的分值

```
zincrby key increment member
```

- `zincrby` 命令用来增加指定成员的分值；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `increment` 是待增量；
- `member` 是待修改的成员。

```
zremrangebyrank key start stop
```

- `zremrangebyrank` 命令用来按照排名范围移除元素；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `start` 和 `stop` 分别表示开始位置和结束位置。

```
zremrangebyscore key min max
```

- `zremrangebyscore` 命令用来按照分值范围移除元素；
- `key` 参数是键名，用于唯一标识数据库中的数据对象；
- `min` 和 `max` 分别表示最小分值和最大分值。

## 使用场景

- 有序地展示排行榜。

## 示例

```
redis> zadd myzset 1 "one"
(integer) 1
redis> zadd myzset 2 "two"
(integer) 1
redis> zadd myzset 3 "three"
(integer) 1
redis> zrange myzset 0 -1 withscores
1) "one"
2) "1"
3) "two"
4) "2"
5) "three"
6) "3"
redis> zrevrange myzset 0 -1 withscores
1) "three"
2) "3"
3) "two"
4) "2"
5) "one"
6) "1"
redis> zrank myzset "one"
(integer) 0
redis> zrank myzset "two"
(integer) 1
redis> zrank myzset "three"
(integer) 2
redis> zscore myzset "one"
"1"
redis> zscore myzset "two"
"2"
redis> zscore myzset "three"
"3"
redis> zcard myzset
(integer) 3
redis> zincrby myzset 10 "two"
"12"
redis> zrange myzset 0 -1 withscores
1) "one"
2) "1"
3) "two"
4) "12"
5) "three"
6) "3"
redis> zremrangebyrank myzset 0 1
(integer) 2
redis> zrange myzset 0 -1 withscores
1) "two"
2) "12"
3) "three"
4) "3"
redis> zremrangebyscore myzset "-inf" "(12"
(integer) 1
redis> zrange myzset 0 -1 withscores
1) "two"
2) "12"
3) "three"
4) "3"
redis> zremrangebyscore myzset "(12" "+inf"
(integer) 2
redis> zrange myzset 0 -1 withscores
1) "three"
2) "3"
```