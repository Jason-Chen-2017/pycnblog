
作者：禅与计算机程序设计艺术                    

# 1.简介
  


MySQL 是一种开源的关系型数据库管理系统（RDBMS），由瑞典iaDB公司开发。MySQL 是目前使用最广泛的数据库之一。很多企业在采用 MySQL 的过程中，都面临着配置文件优化的问题。本文将从配置文件优化的几个方面入手进行介绍，包括文件路径设置、参数设置、日志设置等，并提供一些具体的操作步骤和解决方案。希望能够对读者提高理解力、解决工作中遇到的配置优化问题，进一步提升数据库的性能。
# 2. 文件路径设置；

## 2.1 安装目录选择

由于 MySQL 默认安装目录一般是 /usr/local/mysql 下，为了方便管理及后续维护，建议将 MySQL 安装到一个比较独立的文件夹下，如 /data/mysql 中。
```bash
[root@localhost ~]# cd /data && mkdir mysql && chmod -R 777 mysql # 创建文件夹并授权
[root@localhost ~]# ln -s /data/mysql /usr/local/mysql # 创建软链接
```

## 2.2 配置文件选择

MySQL 配置文件一般放在 /etc/my.cnf 或 /etc/mysql/my.cnf 中。由于 /etc/my.cnf 在 CentOS 7 上已经过时，所以我们推荐使用 /etc/mysql/my.cnf 来作为 MySQL 配置文件。此外，建议根据 MySQL 的版本选择对应的配置文件名，如 my.cnf-5.7，my.cnf-8.0，my.cnf-8.0.19，分别对应 MySQL 5.7，MySQL 8.0，MySQL 8.0.19。
```bash
[root@localhost ~]# ls /etc/ | grep'mysql'
mysql/              mariadb/           mariadb.conf.d/    mariadb-files.list
mariadb-5.5         mariadb-10.3       mysql/             mysql_config.sh
mariadb-10.2        mariadb-10.2.26-el mariadb.conf.backup
```

## 2.3 数据文件选择

数据文件一般存放在数据目录中，默认情况下，MySQL 的数据目录是 /var/lib/mysql。建议将数据文件放置到独立的数据分区上，如 /data/mysql/data。
```bash
[root@localhost ~]# mkdir -p /data/mysql/data && chown -R mysql:mysql /data/mysql/data # 创建文件夹并授权
[root@localhost ~]# vim /etc/my.cnf # 修改配置文件
datadir=/data/mysql/data # 设置数据目录
log-error=/data/mysql/data/error.log # 设置错误日志位置
slow-query-log-file=/data/mysql/data/slow.log # 设置慢查询日志位置
long_query_time=3 # 设置慢查询阈值，单位秒
```

## 2.4 日志设置

日志记录是对 MySQL 运行过程的重要依据。建议开启 MySQL 的 SQL 慢查日志和慢查询日志，便于定位分析慢查询。对于大数据量的应用场景，建议开启 Binlog，即 Binary log。
```bash
[root@localhost ~]# vim /etc/my.cnf # 修改配置文件
log_bin=mysql-bin.log # 启用 binlog
expire_logs_days = 7 # 设置日志保留天数
max_binlog_size = 1G # 设置单个 binlog 文件大小
binlog_format = ROW # 设置 binlog 存储引擎为行级
server_id = 1 # 指定 server id
slow-query-log = on # 开启慢查询日志
slow-query-log-file = /data/mysql/data/slow.log # 设置慢查询日志位置
long_query_time = 3 # 设置慢查询阈值，单位秒
general_log = on # 开启 general log
general_log_file = /data/mysql/data/mysql-general.log # 设置 general log 日志位置
```

# 3. 参数优化；

## 3.1 InnoDB 表引擎参数优化

InnoDB 表引擎是 MySQL 默认的支持事务的表类型，而其参数也需要进行优化。主要的参数如下：

1. innodb_buffer_pool_size：innodb_buffer_pool_size 选项定义了用于缓存数据的内存缓冲池的大小。这个值可以设置为物理内存的 70%~80% 左右，因此可以考虑先用 free 命令查看剩余内存空间。

2. innodb_flush_log_at_trx_commit：该参数决定何时将 InnoDB 日志写入磁盘。默认值为 1，表示每个事务提交时都会写入日志。如果把该参数设置为 0，则表示每秒钟将日志刷新到磁盘一次，可能会影响事务的实时性，但也会减少日志开销，提高性能。

3. innodb_io_capacity：该参数定义了 InnoDB 磁盘子系统的最大 IOPS（Input/Output Operations Per Second）值。它用来控制 InnoDB 访问磁盘的速度，降低 IOPS 可以提高性能。

4. innodb_write_io_threads：该参数定义了用于处理 InnoDB 后台写操作的线程数量。默认值为 4，可适当调整以获取更好的性能。

5. thread_concurrency：该参数定义了服务器同时使用的最大线程数量，可以通过 ulimit -u 查看当前用户限制。如果超过了限制，可以考虑调小该值或增加 root 用户的权限。

6. sort_buffer_size：该参数定义了排序操作所需的内存大小。默认值为 2M，可以适当调整以获得更佳性能。

```bash
[root@localhost ~]# vim /etc/my.cnf # 修改配置文件
[mysqld]
innodb_buffer_pool_size=8G # 设置 innodb 缓冲池大小为 8GB
innodb_flush_log_at_trx_commit=0 # 每秒钟刷新日志，降低日志写入次数
innodb_io_capacity=2000 # 设置 innodb 磁盘 IOPS 为 2000
innodb_write_io_threads=8 # 设置用于处理 InnoDB 后台写操作的线程数量为 8
thread_concurrency=64 # 设置服务器最大线程数为 64
sort_buffer_size=256K # 设置排序操作所需的内存大小为 256KB
```

## 3.2 MyISAM 表引擎参数优化

MyISAM 表引擎不支持事务，其参数优化主要基于文件的大小进行调整。

1. key_buffer_size：key_buffer_size 选项定义了用于缓冲索引的内存大小。默认值为 8MB，建议设置为物理内存的 20%~30%。

2. read_buffer_size：read_buffer_size 选项定义了 MyISAM 扫描索引的缓冲区大小。默认值为 16KB，可以适当调整以获得更佳性能。

3. read_rnd_buffer_size：read_rnd_buffer_size 选项定义了 MyISAM 使用的缓冲区大小。默认值为 256KB，可以适当调整以获得更佳性能。

4. concurrent_insert：concurrent_insert 选项定义了是否允许多个进程同时插入同一个表。默认为 2，表示允许两个进程同时插入同一个表，可以适当调整以获得更佳性能。

```bash
[root@localhost ~]# vim /etc/my.cnf # 修改配置文件
[mysqld]
key_buffer_size=16M # 设置用于缓冲索引的内存大小为 16MB
read_buffer_size=1M # 设置 MyISAM 扫描索引的缓冲区大小为 1MB
read_rnd_buffer_size=1M # 设置 MyISAM 使用的缓冲区大小为 1MB
concurrent_insert=2 # 设置是否允许多个进程同时插入同一个表为 2
```

# 4. 启动优化；

## 4.1 删除不需要的日志文件

删除掉不需要保留的日志文件，比如 slow.log 和 error.log，避免占用过多的磁盘空间。
```bash
[root@localhost ~]# rm /var/log/mysql/slow.log.* /var/log/mysql/error.log.*
```

## 4.2 关闭不需要的服务

关闭不需要的服务，比如 MySQL 服务端和客户端。
```bash
[root@localhost ~]# systemctl stop mysqld # 停止 mysql 服务端
[root@localhost ~]# systemctl disable mysqld # 禁止自动启动 mysql 服务端
[root@localhost ~]# systemctl stop mariadb # 停止 mariadb 服务端
[root@localhost ~]# systemctl disable mariadb # 禁止自动启动 mariadb 服务端
[root@localhost ~]# systemctl stop mysql # 停止 mysql 客户端
[root@localhost ~]# systemctl disable mysql # 禁止自动启动 mysql 客户端
```

## 4.3 切换 mysql 服务端

通过以下命令切换 mysql 服务端：
```bash
[root@localhost ~]# mv /usr/local/mysql /usr/local/mariadb
[root@localhost ~]# ln -s /usr/local/mariadb /usr/local/mysql
```
注意：如果是使用 yum 安装的 mysql，则无需修改软连接，直接使用 yum install 替换即可。