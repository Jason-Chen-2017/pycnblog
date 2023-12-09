                 

# 1.背景介绍

Redis是一个开源的高性能的key-value数据库，它支持数据的持久化，可基于内存（Redis）或磁盘（Disk）。Redis 提供多种语言的 API。Redis 可以在18s内处理100W个请求，每秒处理100K个设置操作（SET, GET, DELETE），每秒处理110K个字符串操作（GET, SET, DELETE），每秒处理80K个列表操作（LPUSH, RPUSH, LPOP, RPOP, LRANGE），每秒处理110K个有序集合操作（ZADD，ZCARD，ZCOUNT，ZRANGE），每秒处理100K个集合操作（SADD，SCARD，SDIFF，SINTER，SUNION），每秒处理100K个哈希操作（HSET，HGET，HDEL）。Redis 支持通过 Lua 脚本定制命令（redis.call）。
