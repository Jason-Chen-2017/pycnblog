
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2019年，互联网蓬勃发展，各种互联网平台日渐繁盛，网络服务提供商纷纷推出了基于云计算的服务，比如阿里云、腾讯云、百度云等，而对于数据层面的存储则以分布式数据库 Cassandra、HBase、MongoDB 为代表。为了应对日益增长的访问量和海量的数据，这些分布式数据库都加上了一层高速缓存，来提升系统整体的性能。那么什么是缓存呢？
          
          ## 概念定义
           “缓存”（Cache）指的是临时存放计算机数据的小容量存储器，通常存储在内存中，用于快速访问数据，以改进系统效率。它可以减少处理器时间，提高计算机系统的运行速度。而memcached和redis都是缓存产品，它们的主要区别就是存储位置不同。memcached是一个纯内存的缓存，它的存储空间是相对较小的，不能存放太多数据，因此不能用来作为分布式缓存，它的优点是简单、快速。redis是完全开源免费的缓存产品，支持丰富的数据结构，并且支持主从服务器模式，能够满足分布式缓存需求。
          ## Memcached
          memcached是一个开源的，高性能的分布式内存对象缓存系统，用作先进的动态WEB应用。memcached支持多种数据结构，如字符串、散列、列表、集合和有序集合。它还支持简单的协议，包括ASCII文本协议和二进制协议。memcached最初由Danga Interactive公司开发，后来成立于SourceForge社区。memcached支持全内存模式和分布式模式，其中全内存模式适用于单个服务器，分布式模式适用于多台服务器。
      　　Memcached 使用 SASL (Simple Authentication and Security Layer) 认证模块进行用户验证，默认情况下，没有启用 SASL 。为了安全起见，建议您关闭 Memcached 的匿名登录功能，即在配置文件中将 "-o -m" 参数设置为 "no_auth"。
      
      ### 安装配置
        ```sh
        sudo apt-get update && sudo apt-get install memcached
        
        # 查看memcached是否安装成功
        ps aux | grep'memcached'
        ```
      ### 配置项解析
      **基本配置**：
      - -p 指定端口号，默认端口是11211。
      - -u 用户权限认证，默认为无。如果指定了此参数，需要先启动memcached服务，然后再创建认证文件（下面会提到）。
      - -l 本地IP地址，默认为 INADDR_ANY （0.0.0.0），表示监听所有网络接口。
      - -d 以守护进程方式运行，在后台运行 memcached 服务。
      - -r 设置最大内存使用量，单位MB。默认为64MB，可根据需要设置。
      **内存管理配置**：
      - -m 设置内存大小，单位MB。默认为64MB，但是推荐设置为物理内存的一半或更大。
      - -M 指定内存耗尽策略。memcached 会淘汰一部分数据，保证内存总量不超过设定值。可选值如下：
        - volatile-ttl：当达到内存限制时，将优先清除过期的项；
        - allkeys-lru：当达到内存限制时，删除最近最少使用的项；
        - volatile-random：当达到内存限制时，随机清除过期的项；
        - allkeys-random：当达到内存限制时，随机删除一个项；
        - noreply：默认的回应模式，避免长时间等待。
      **连接池配置**：
      - -c 设置最大连接数，默认为1024。
      - -P 设置每个套接字的缓冲区大小。默认为4K。
      - --disable-watchdog 不启用 watchdog 检查机制。
      - --enable-flush命令通过 TCP 请求实现 flush 命令。
      **日志记录配置**：
      - -t 设置内存分配失败时的错误信息打印频率，单位秒。默认值为60秒。
      - -vv 详细日志模式，打印所有命令执行的信息。
      - -v 一般日志模式，打印常规信息。
      
      **SASL 认证**
      如果需要开启 SASL 认证，可以参考下面的操作步骤：
      1. 创建认证文件 auth.txt ，内容为用户名和密码。
      ```
      username:password
      user1:<PASSWORD>
      user2:pass2
     ...
      ```
      2. 在启动 memcached 时增加 "-u /etc/memcached/auth.txt" 参数。
      ```sh
      sudo memcached -u /etc/memcached/auth.txt
      ```
      
      ### 使用 Memcached
      1. 设置键值对
      ```
      set key value [ex seconds] [px milliseconds] [nx|xx]
      ex seconds : 将键值对的过期时间设置为 seconds 秒。
      px milliseconds : 将键值对的过期时间设置为 milliseconds 毫秒。
      nx : 当且仅当 key 不存在时，才对其设置值。
      xx : 当且仅当 key 存在时，才对其设置值。
      ```
      ```sh
      # 设置键值为value的键值对，过期时间设置为60秒
      set name "redis"

      # 设置键值为value的键值对，过期时间设置为30分钟
      set name "memcache" ex 1800
      ```
      2. 获取键值对的值
      ```
      get key
      ```
      ```sh
      # 获取name对应的键值对的值
      get name
      ```
      3. 删除键值对
      ```
      del key [key...]
      ```
      ```sh
      # 删除name和age对应的键值对
      del name age
      ```
      4. 清空所有缓存
      ```
      flush_all
      ```
      
      ### 数据类型
      Memcached 支持以下几种数据类型：字符串（string），散列（hash），列表（list），集合（set），有序集合（sorted set）。以下是在命令行操作时，常用的几个命令：
      1. 设置字符串
      ```
      set key value [flags] [exptime] [bytes]
      flags : 表示存储的类型，通常用数字表示，如果是数字则不用这个参数，例如 set k v 1 。
      exptime : 表示键值的过期时间，单位秒。
      bytes : 表示数据长度，单位字节。
      ```
      ```sh
      # 设置键值为value的字符串类型键值对
      set strkey "hello world"
      ```
      2. 获取字符串
      ```
      get key
      ```
      ```sh
      # 获取strkey对应的键值对的值
      get strkey
      ```
      3. 设置散列
      ```
      hset key field value
      ```
      ```sh
      # 添加键值对field:value到键值对集key中
      hset myhash field1 "Hello"
      ```
      4. 获取散列
      ```
      hget key field
      ```
      ```sh
      # 获取myhash对应的键值对集的所有字段及值
      hgetall myhash
      ```
      5. 删除散列中的元素
      ```
      hdel key field [field...]
      ```
      ```sh
      # 从键值对集key中删除field字段
      hdel myhash field1
      ```
      6. 设置列表
      ```
      lpush key value
      rpush key value
      lrange key start end
      ltrim key start stop
      lindex key index
      ```
      ```sh
      # 插入元素"world"到列表"mylist"的左侧
      lpush mylist "world"

      # 插入元素"hello"到列表"mylist"的右侧
      rpush mylist "hello"

      # 获取列表"mylist"从索引0开始到索引4结束的元素
      lrange mylist 0 4

      # 获取列表"mylist"的前五个元素
      lrange mylist 0 4

      # 删除列表"mylist"的所有元素
      ltrim mylist 1 0

      # 获取列表"mylist"的第一个元素
      lindex mylist 0
      ```
      7. 删除列表中的元素
      ```
      lrem key count value
      ```
      ```sh
      # 从列表"mylist"中移除所有值为"hello"的元素
      lrem mylist 0 "hello"
      ```
      8. 设置集合
      ```
      sadd key member [member...]
      scard key
      smembers key
      spop key
      sismember key member
      ```
      ```sh
      # 添加元素"hello"到集合"myset"
      sadd myset hello

      # 获取集合"myset"的元素个数
      scard myset

      # 获取集合"myset"的所有成员
      smembers myset

      # 随机获取集合"myset"的一个元素
      spop myset

      # 判断元素"hello"是否属于集合"myset"
      sismember myset hello
      ```
      9. 删除集合中的元素
      ```
      srem key member [member...]
      ```
      ```sh
      # 从集合"myset"中删除元素"hello"
      srem myset hello
      ```
      10. 有序集合
      ```
      zadd key score member [score member...]
      zcard key
      zcount key min max
      zincrby key increment member
      zrange key start end [withscores]
      zrevrange key start end [withscores]
      zrangebylex key min max [limit offset count]
      zrevrangebylex key max min [limit offset count]
      zrank key member
      zrem key member [member...]
      zremrangebylex key min max
      zremrangebyrank key start stop
      zremrangebyscore key min max
      ```
      ```sh
      # 添加元素"member1"到有序集合"myzset"，值为100
      zadd myzset 100 member1

      # 获取有序集合"myzset"的元素个数
      zcard myzset

      # 统计有序集合"myzset"中值为100的元素的个数
      zcount myzset 100 100

      # 对元素"member1"的值加100
      zincrby myzset 100 member1

      # 获取有序集合"myzset"中值为[0, 100)的元素列表
      zrange myzset 0 100 withscores

      # 获取有序集合"myzset"中值为[0, 100)的元素列表(降序)
      zrevrange myzset 0 100 withscores

      # 根据字典序范围查询有序集合"myzset"的元素列表
      zrangebylex myzset "[aaa" "(g" limit 0 100

      # 根据字典序范围查询有序集合"myzset"的元素列表(降序)
      zrevrangebylex myzset "(g" "[aaa" limit 0 100

      # 查询元素"member1"在有序集合"myzset"中的排名
      zrank myzset member1

      # 删除元素"member1"及其对应的值
      zrem myzset member1

      # 删除有序集合"myzset"中值为[100, +inf)的元素
      zremrangebyscore myzset 100 +inf

      # 删除有序集合"myzset"中排名在[start, stop]之间的所有元素
      zremrangebyrank myzset 0 10
      ```
      
      ### 分片集群
      Memcached 提供了分片集群的能力，它可以让多个 Memcached 节点组成一个分布式集群，所有的操作请求均按照 key 路由到各个 Memcached 节点中进行处理，实现数据共享和负载均衡。
      
      1. 配置
      首先，安装好 Memcached 集群环境。在每台服务器上，创建一个 Memcached 服务启动脚本，然后编辑该脚本，添加以下几行：
      
      ```
      #!/bin/bash
      echo $MEMCACHED_PORT is starting...
      exec /usr/bin/memcached -c /etc/memcached/$MEMCACHED_PORT.conf -u /etc/memcached/auth.txt
      ```
      
      每台服务器的 /etc/memcached/ 目录下创建一个子目录，命名规则为 conf_${port} ，${port} 为该 Memcached 节点的端口号，例如 conf_11211。在该目录下创建配置文件 memcached.conf ，内容如下：
      
      ```
      port ${port}           # 设置 Memcached 服务监听的端口号
      bind 127.0.0.1          # 只允许本地连接
      server_verbosity 1     # 设置日志输出级别，1为只显示错误信息
      hash cash              # 设置 Hash 算法，这里设置为一致性 Hash
      num_threads 4          # 设置线程数量，这里设置为 4 个
      ```
      
      最后，将所有节点上的启动脚本放在同一个目录，并将该目录下的 memcached.conf 文件拷贝到所有 Memcached 节点上的相同目录下。
      
      2. 操作
      Memcached 通过 key 路由到各个 Memcached 节点，所有操作请求均由这些节点完成。以下为一些常用操作的例子：
      
      > 添加键值对
      ```
      set key value
      ```
      
      > 获取键值对的值
      ```
      get key
      ```
      
      > 删除键值对
      ```
      delete key
      ```
      
      > 清空所有缓存
      ```
      flush_all
      ```