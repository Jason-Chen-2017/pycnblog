                 

# 1.背景介绍

Redis（Remote Dictionary Server，远程字典服务器）是一个开源的高性能key-value存储系统，由Salvatore Sanfilippo编写，并于2009年发布。Redis支持数据的持久化，可以将内存中的数据保存在磁盘中，重启的时候可以再次加载进行使用。Redis不仅仅支持简单的key-value类型的缓存，同时还提供列表、集合、有序集合及哈希等数据结构的存储。

Redis支持网络操作，可以用于远程通信。它采用ANSI C语言编写，并遵循BSD协议，提供免费的源代码。Redis通过Redis Persistence提供数据持久性功能：可以将内存中的数据保存在磁盘中，当服务重启时加载进行使用；同时通过Replication提供数据备份和读写分离功能：可以将数据复制到多台服务器上，为读操作提供多个服务器。

# 2.核心概念与联系
## 2.1 Redis基本概念
### 2.1.1 Redis基本概念介绍
Redis是一个开源的高性能key-value缓存系统,主要由C语言编写,遵循BSD协议,提供免费源代码,支持网络操作,适用于远程通信。Redis采用ANSI C语言编写,并遵循BSD协议,提供免费的源代码。Redis通过Redis Persistence提供数据持久性功能：可以将内存中的数据保存在磁盘中,当服务重启时加载进行使用；同时通过Replication提供数据备份和读写分离功能：可以将数据复制到多台服务器上,为读操作提供多个服务器。
### 2.1.2 Redis核心组件介绍
#### 2.1.2.1 Redis客户端与服务端关系
客户端与服务端之间是一种主从关系:客户端向redis发送命令请求;redis接收命令后执行相应操作并返回结果给客户端;客户端接收结果并处理或显示给最终用户(如浏览器)。这种模式称为“请求/响应”模式或“客户机/服务器”模式。常见redis客户端有php redis、java Jedis等等;常见redis服务端有rediss、rediss-cluster等等;常见rediss-cluster包含了对rediss集群管理和监控功能;常见php redis包含了对php redis扩展管理和监控功能;常见java Jedis包含了对java Jedis扩展管理和监控功能;常见python redis包含了对python redis扩展管理和监控功能;常见go redis包含了对go redis扩展管理和监控功能;常见nodejs redis包含了对nodejs redis扩展管理和监控功能;常见ruby redis包含了对ruby redis扩展管理和监控功能;常见lua redix包含了对lua redix扩展管理和监控功能;常见perl redix包含了对perl redix扩展管理和监控功能;常见shell redix包含了对shell redix扩展管理和监控功能;常见ios redix包含了对ios redix扩展管理和监控功能;常见android redix包含了对android redix扩展管理和监控功能;常见windows phone redix包含了对windows phone redix扩展管理和监控功능