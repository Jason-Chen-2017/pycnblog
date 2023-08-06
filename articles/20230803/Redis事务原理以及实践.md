
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1995年，爱立信（Apollo Systems）在研究员梅尔·迪克斯特利（Michael Dickey）的帮助下，提出了事务处理技术概念，并基于此设计了Redis数据库。随后Redis逐渐成为了开源项目，被越来越多的开发者使用。Redis是一个高性能、可扩展的键-值存储数据库。
         
         在实际应用场景中，客户端需要执行多个命令，而这些命令需要满足以下两个条件：1. 操作多个数据；2. 需要保证操作的原子性。要实现这一点，就需要一种机制可以将多条命令打包起来，然后一次性执行，且不会因某一条命令失败而导致整个过程的终止。Redis提供了事务功能来支持这种需求。
         
         事务是由一组命令组成的不可分割的工作单元，一个事务中的所有命令都要么都执行成功，要么都不执行。事务最主要的特性就是原子性。事务是一个串行执行的一系列命令，中间不会有其他客户端的命令请求进来影响当前事务的执行。如果一个事务执行失败或者遇到错误，那么只有它自己会回滚到事务开始前的状态，不会影响到其他事务的执行。Redis事务功能可以用于单个Redis命令或多个Redis命令构成的复杂业务逻辑。
         
         通过本文，希望能让读者对Redis事务有一个清晰、全面的认识，并且能够运用好Redis事务，提升自己的编程水平。本文假设读者已经熟悉Redis相关知识，至少具备使用Redis基本命令的能力。
         
         # 2.基本概念术语说明
         ## 2.1.Redis事务
         Redis事务是指Redis服务器执行一个队列命令集，该命令集要么都执行成功，要么都不执行，这个过程不是原子的。事务从Redis2.0版本引入。
         
         ## 2.2.事务模式
         Redis事务有两种运行模式：
         - 普通模式（默认模式）
         - 阻塞模式
         在普通模式下，客户端发送的所有命令都会排队等候服务器的响应。在事务执行完毕之前，其他客户端不能发送新的命令给服务器。在阻塞模式下，客户端发送的每条命令都会等待服务器响应。当服务器接收并处理完上一条命令之后，才会处理下一条命令。
         
         ## 2.3.Redis命令
         Redis支持的命令很多，例如字符串类型（string）命令、列表类型（list）命令、集合类型（set）命令等等。本文只讨论Redis事务和原子性相关的命令。
         
        ###  2.3.1.Redis String类型命令
         String类型命令包括SETNX、GETSET、MSET、MGET、APPEND、INCR、DECR等命令。关于String类型的命令，本文只讨论需要修改数据的命令，也就是说，需要读取并改变数据的值的命令。因为String类型命令既不需要加锁也没有隔离级别。

         1. SETNX 命令：设置指定键的值，若指定的键不存在，则进行赋值。
         ```redis
         redis> SETNX mykey "hello"
         (integer) 1
         redis> GET mykey
         "hello"
         redis> SETNX mykey "world"
         (integer) 0
         redis> GET mykey
         "hello"
         ```

         设置键mykey的值为hello，成功返回1，已存在时则返回0，且该键仍然保持其原值的hello。

         如果把SETNX命令替换成SET命令，则在已存在时会覆盖原值，而不是更新：

         ```redis
         redis> SET mykey "hello"
         "OK"
         redis> SET mykey "world"
         "OK"
         redis> GET mykey
         "world"
         ```

         使用SETNX命令比使用SET命令更安全，因为SET命令在数据量较大的情况下会存在并发问题。

         2. MSET 命令：批量设置多个键的值。
         ```redis
         redis> MSET k1 "hello" k2 "world"
         "OK"
         redis> GET k1
         "hello"
         redis> GET k2
         "world"
         ```

         将k1和k2设置为hello和world，成功返回OK。

         3. MSETNX 命令：批量设置多个键的值，但只在所有键都不存在时，才成功。
         ```redis
         redis> DEL k1 k2
         (integer) 2
         redis> MSETNX k1 "hello" k2 "world"
         (integer) 1
         redis> MSETNX k3 "foo" k4 "bar"
         (integer) 0
         redis> MGET k1 k2 k3 k4
         (nil)
         (nil)
         "foo"
         "bar"
         ```

         删除k1和k2两个键后，再尝试批量设置k1为hello，k2为world和k3为foo的情况。由于k1和k2已经存在，所以批量设置失败，仅设置k3为foo。成功设置的键值对包括k1=hello，k3=foo。

         4. GETSET 命令：设置指定键的值并返回原来的值。
         ```redis
         redis> SET mykey "hello"
         "OK"
         redis> GETSET mykey "world"
         "hello"
         redis> GET mykey
         "world"
         ```

         将mykey设置为hello，获取其旧值并设置为world。

         5. APPEND 命令：追加值到指定键。
         ```redis
         redis> SET mykey "hello"
         "OK"
         redis> APPEND mykey " world!"
         (integer) 12
         redis> GET mykey
         "hello world!"
         ```

         将mykey的值附加上" world!"，成功返回长度12，新值为"hello world!".

         6. INCR 命令：将指定整数类型的键增加指定的增量值。
         ```redis
         redis> SET mycounter 10
         "OK"
         redis> INCR mycounter
         (integer) 11
         redis> INCRBY mycounter 5
         (integer) 16
         ```

         用一个整数类型（这里假定为计数器）的键mycounter初始化为10，然后递增其值并打印，结果为11，再递增其值增加5，结果为16。

         7. DECR 命令：将指定整数类型的键减去指定的增量值。
         ```redis
         redis> SET mycounter 10
         "OK"
         redis> DECR mycounter
         (integer) 9
         redis> DECRBY mycounter 5
         (integer) 4
         ```

         用一个整数类型（这里假定为计数器）的键mycounter初始化为10，然后递减其值并打印，结果为9，再递减其值减去5，结果为4。

        ### 2.3.2.Redis List类型命令
         List类型命令包括LPUSH、RPUSH、LPOP、RPOP、LTRIM、LLEN、LINDEX、LINSERT、LRANGE、LREM等命令。List类型命令需要对操作的List进行排他性操作，因此需要加锁。
        
         1. LPUSH 命令：添加元素到左侧。
         ```redis
         redis> RPUSH mylist "world"
         (integer) 1
         redis> LLEN mylist
         (integer) 1
         redis> LPUSH mylist "hello"
         (integer) 2
         redis> LLEN mylist
         (integer) 2
         redis> LRANGE mylist 0 -1
         (error) ERR Operation against a key holding the wrong kind of value
         redis> LIST mylist
         ^[[Bhello
         ^[[Cworld
         ```

         从右边推入一个元素“world”，成功返回插入后的List长度为1，再从左边推入“hello”成功返回长度为2。查看该List元素时会报错（因为该命令应该用REDIIS server端的指令才能正常工作）。使用LIST命令查看时，元素会显示为^[[B和^[[C这样的特殊字符，这是因为在交互模式下，Redis向Terminal输出的命令文本和执行后的命令结果可能由于编码不同，导致显示异常。

         2. RPUSH 命令：添加元素到右侧。
         ```redis
         redis> RPUSH mylist "world"
         (integer) 1
         redis> RPUSH mylist "hello"
         (integer) 2
         redis> LRANGE mylist 0 -1
         (error) ERR Operation against a key holding the wrong kind of value
         redis> LIST mylist
         ^[[Bhello
         ^[[Cworld
         ```

         从右边推入另一个元素“world”，成功返回插入后的List长度为1，再从右边推入“hello”成功返回长度为2。查看该List元素时会报错。使用LIST命令查看时，元素会显示为^[[B和^[[C这样的特殊字符。

         3. LPOP 命令：删除并返回List左侧第一个元素。
         ```redis
         redis> RPUSH mylist "world"
         (integer) 1
         redis> RPUSH mylist "hello"
         (integer) 2
         redis> LPOP mylist
         "hello"
         redis> LRANGE mylist 0 -1
         (error) ERR Operation against a key holding the wrong kind of value
         redis> LIST mylist
         ^[[Cworld
         ```

         添加两个元素到mylist中，然后弹出左侧的第一个元素，打印结果，成功弹出“hello”。查看该List元素时会报错。使用LIST命令查看时，元素会显示为^[[C这样的特殊字符。

         4. RPOP 命令：删除并返回List右侧第一个元素。
         ```redis
         redis> RPUSH mylist "world"
         (integer) 1
         redis> RPUSH mylist "hello"
         (integer) 2
         redis> RPOP mylist
         "world"
         redis> LRANGE mylist 0 -1
         (error) ERR Operation against a key holding the wrong kind of value
         redis> LIST mylist
         ^[[Bhello
         ```

         添加两个元素到mylist中，然后弹出右侧的第一个元素，打印结果，成功弹出“world”。查看该List元素时会报错。使用LIST命令查看时，元素会显示为^[[B这样的特殊字符。

         5. LTRIM 命令：截取List元素，删除多余元素。
         ```redis
         redis> RPUSH mylist "a"
         (integer) 1
         redis> RPUSH mylist "b"
         (integer) 2
         redis> RPUSH mylist "c"
         (integer) 3
         redis> RPUSH mylist "d"
         (integer) 4
         redis> LTRIM mylist 0 2
         OK
         redis> LRANGE mylist 0 -1
         (error) ERR Operation against a key holding the wrong kind of value
         redis> LIST mylist
         ^[[Ba
         ^[[Cb
         ^[[Cc
         ```

         添加四个元素到mylist中，然后截取其中三个元素保留第一个，第二个，第三个元素，成功返回OK。查看该List元素时会报错。使用LIST命令查看时，元素会显示为^[[Ba，^[[Cb和^[[Cc这样的特殊字符。

         6. LINDEX 命令：返回指定索引处的元素。
         ```redis
         redis> RPUSH mylist "a"
         (integer) 1
         redis> RPUSH mylist "b"
         (integer) 2
         redis> RPUSH mylist "c"
         (integer) 3
         redis> LINDEX mylist 1
         "b"
         redis> LRANGE mylist 0 -1
         (error) ERR Operation against a key holding the wrong kind of value
         redis> LIST mylist
         ^[[Ba
         ^[[Cb
         ^[[Cc
         ```

         添加三个元素到mylist中，查询索引为1处的元素，结果为“b”，查看该List元素时会报错。使用LIST命令查看时，元素会显示为^[[Ba，^[[Cb和^[[Cc这样的特殊字符。

         7. LINSERT 命令：在List中指定位置插入元素。
         ```redis
         redis> RPUSH mylist "hello"
         (integer) 1
         redis> RPUSH mylist "world"
         (integer) 2
         redis> LINSERT mylist BEFORE "world" "foo"
         (integer) 3
         redis> LRANGE mylist 0 -1
         (error) ERR Operation against a key holding the wrong kind of value
         redis> LIST mylist
         ^[[Bhello
         ^[[Cfoworld
         ```

         添加两个元素到mylist中，然后在“world”前面插入“foo”，成功返回插入后的List长度为3。查看该List元素时会报错。使用LIST命令查看时，元素会显示为^[[Bhello，^[[Cfoworld这样的特殊字符。

         8. LLEN 命令：返回List的长度。
         ```redis
         redis> RPUSH mylist "hello"
         (integer) 1
         redis> RPUSH mylist "world"
         (integer) 2
         redis> LLEN mylist
         (integer) 2
         redis> LRANGE mylist 0 -1
         (error) ERR Operation against a key holding the wrong kind of value
         redis> LIST mylist
         ^[[Bhello
         ^[[Cwolrld
         ```

         添加两个元素到mylist中，然后查询该List的长度，结果为2。查看该List元素时会报错。使用LIST命令查看时，元素会显示为^[[Bhello，^[[Cwolrld这样的特殊字符。

         9. LRANGE 命令：返回List中指定范围内的元素。
         ```redis
         redis> RPUSH mylist "a"
         (integer) 1
         redis> RPUSH mylist "b"
         (integer) 2
         redis> RPUSH mylist "c"
         (integer) 3
         redis> RPUSH mylist "d"
         (integer) 4
         redis> LRANGE mylist 0 2
         (error) ERR Operation against a key holding the wrong kind of value
         redis> LIST mylist
         ^[[Aa
         ^[[Bb
         ^[[Bc
         ```

         添加四个元素到mylist中，查询索引0~2处的元素，结果分别为“a”，“b”和“c”。查看该List元素时会报错。使用LIST命令查看时，元素会显示为^[[Aa，^[[Bb和^[[Bc这样的特殊字符。

         10. LREM 命令：删除List中指定数量和值相同的元素。
         ```redis
         redis> RPUSH mylist "a"
         (integer) 1
         redis> RPUSH mylist "b"
         (integer) 2
         redis> RPUSH mylist "a"
         (integer) 3
         redis> RPUSH mylist "b"
         (integer) 4
         redis> LREM mylist -1 "b"
         (integer) 2
         redis> LRANGE mylist 0 -1
         (error) ERR Operation against a key holding the wrong kind of value
         redis> LIST mylist
         ^[[Aa
         ^[[Cc
         ```

         添加四个元素“a”和“b”，然后重复添加“a”和“b”，查询List中的所有元素，结果为“a”，“a”，“b”，“b”，只剩下两次“a”和“b”。使用LREM命令删除掉所有值为“b”的元素，只剩下“a”和“a”。查看该List元素时会报错。使用LIST命令查看时，元素会显示为^[[Aa，^[[Cc这样的特殊字符。

         上述所有命令除了SETNX、MSETNX和INCR/DECR外，都是不需要加锁的。

        ### 2.3.3.Redis Hash类型命令
         Hash类型命令包括HSET、HGET、HMSET、HMGET、HDEL、HEXISTS、HINCRBY等命令。Hash类型命令需要对操作的Hash进行排他性操作，因此需要加锁。
         
         1. HSET 命令：设置字段的值。
         ```redis
         redis> HSET myhash field1 "value1"
         (integer) 1
         redis> HGETALL myhash
         (error) ERR Operation against a key holding the wrong kind of value
         redis> HGET myhash field1
         "value1"
         ```

         对名为myhash的Hash对象添加一个名为field1的字段，其值为“value1”，成功返回1。查看该Hash的所有字段时会报错。使用HGETALL命令查看时，字段会显示为空。使用HGET命令查找某个字段时，会正确得到其值。

         2. HMSET 命令：批量设置字段的值。
         ```redis
         redis> HMSET myhash field1 "value1" field2 "value2"
         "OK"
         redis> HGETALL myhash
         (error) ERR Operation against a key holding the wrong kind of value
         redis> HGET myhash field1
         "value1"
         redis> HGET myhash field2
         "value2"
         ```

         对名为myhash的Hash对象批量添加字段，成功返回OK。查看该Hash的所有字段时会报错。使用HGETALL命令查看时，字段会显示为空。使用HGET命令查找某个字段时，会正确得到其值。

         3. HGET 命令：获取字段的值。
         ```redis
         redis> HSET myhash field1 "value1"
         (integer) 1
         redis> HGETALL myhash
         (error) ERR Operation against a key holding the wrong kind of value
         redis> HGET myhash field1
         "value1"
         redis> HGET myhash notexist
         (nil)
         ```

         对名为myhash的Hash对象添加一个名为field1的字段，其值为“value1”，成功返回1。查看该Hash的所有字段时会报错。使用HGETALL命令查看时，字段会显示为空。使用HGET命令查找某个字段时，会正确得到其值。对于不存在的字段，会返回(nil)。

         4. HMGET 命令：批量获取字段的值。
         ```redis
         redis> HMSET myhash field1 "value1" field2 "value2"
         "OK"
         redis> HGETALL myhash
         (error) ERR Operation against a key holding the wrong kind of value
         redis> HMGET myhash field1 field2 notexist
         (nil)
         "value1"
         "value2"
         ```

         对名为myhash的Hash对象批量添加字段，成功返回OK。查看该Hash的所有字段时会报错。使用HGETALL命令查看时，字段会显示为空。使用HMGET命令批量查找多个字段时，会正确得到其值。对于不存在的字段，会返回(nil)。

         5. HDEL 命令：删除一个或多个字段。
         ```redis
         redis> HSET myhash field1 "value1" field2 "value2" field3 "value3"
         (integer) 3
         redis> HDEL myhash field1 field2
         (integer) 2
         redis> HGETALL myhash
         (error) ERR Operation against a key holding the wrong kind of value
         redis> HGET myhash field3
         "value3"
         redis> HGET myhash field1
         (nil)
         redis> HGET myhash field2
         (nil)
         ```

         对名为myhash的Hash对象添加三个字段，分别为field1="value1"，field2="value2"和field3="value3"，成功返回三个。然后使用HDEL命令删除field1和field2两个字段，成功返回2。查看该Hash的所有字段时会报错。使用HGET命令查找某个字段时，会正确得到其值。对于被删除的字段，会返回(nil)。

         6. HEXISTS 命令：判断是否存在某个字段。
         ```redis
         redis> HSET myhash field1 "value1" field2 "value2" field3 "value3"
         (integer) 3
         redis> HEXISTS myhash field1
         (integer) 1
         redis> HEXISTS myhash notexist
         (integer) 0
         ```

         对名为myhash的Hash对象添加三个字段，分别为field1="value1"，field2="value2"和field3="value3"，成功返回三个。然后使用HEXISTS命令检查field1字段是否存在，结果为1，表示存在。检查notexist字段是否存在，结果为0，表示不存在。

         7. HINCRBY 命令：对数字型字段的值做增量操作。
         ```redis
         redis> HSET myhash counter 10
         (integer) 1
         redis> HINCRBY myhash counter 5
         (integer) 15
         redis> HGET myhash counter
         "15"
         redis> HINCRBY myhash notexist 5
         (integer) 5
         redis> HGET myhash notexist
         "5"
         ```

         对名为myhash的Hash对象添加一个名为counter的字段，其值为10，成功返回1。使用HINCRBY命令对counter做增量操作，成功返回15。使用HGET命令获取counter字段的值，结果为“15”。再对notexist字段做增量操作，没有这个字段，所以会自动创建一个。获取notexist字段的值，结果为“5”。