
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　数据工程师作为整个企业的数据支撑和运营的重要角色，掌握其相关技能可以实现对数据的整体把握、处理、存储、安全、监控等全生命周期管理。而与此同时，数据分析师也成为各行各业所不可或缺的重要岗位。数据分析师在处理海量数据时，需要有高效的SQL语言水平，对大数据系统的性能优化能力尤其要求。由于数据量越来越大，数据分析师的SQL理解和使用能力要求也越来越强。数据工程师则是基于数据科学和技术领域的专业知识和工具进行更加细致的工作。相信本课程能够帮助大家快速上手SQL编程，熟悉PostgreSQL数据库的安装配置与优化，进而提升个人职场竞争力。
         # 2.什么是SQL?
          SQL(Structured Query Language)是一种用于管理关系型数据库的语言，用于存取、查询、更新和删除关系数据库中的数据。它具备完整的ACID特性，包括原子性（Atomicity），一致性（Consistency），隔离性（Isolation），持久性（Durability）。它是一个标准化的语言，由国际标准组织ISO（International Organization for Standardization）制定，并得到各个厂商和数据库系统的支持。 SQL语言有很多功能，包括SELECT、UPDATE、INSERT、DELETE、CREATE TABLE、DROP TABLE、ALTER TABLE、JOIN、WHERE、GROUP BY、HAVING、ORDER BY、UNION、MINUS、SUBQUERY、CROSS JOIN等，这些命令可以用来进行数据的增删改查、表的创建、修改、删除等操作。
        # 3.什么是PostgreSQL?
         PostgreSQL是一种开源的关系数据库管理系统，支持SQL语言。它的开发者在过去几十年里一直坚持创新，致力于提供一个高效率、可靠、可扩展的产品。 PostgreSQL能够快速、灵活地处理大量数据，能够处理复杂的查询，并且具有高度安全性和可靠性。它是目前世界上最流行的关系数据库管理系统，被誉为Oracle的克隆品。通过课程的学习，学生将能够编写符合PostgreSQL语法的SQL语句，并且搭建自己的数据库集群，实现真正意义上的“大数据”应用。
        # 4.为什么要学习PostgreSQL？
         在学习SQL之前，首先应该明确自己学习的目的。如果你只是为了熟悉SQL，那么在实际应用中遇到性能瓶颈时，可能就无从下手了。当数据量逐渐变大时，SQL的执行速度也会变慢。同时，如果没有相应的数据库设计，或许你的SQL语句运行得还不错，但最终可能会带来巨大的性能问题。这时候，需要转向更适合大规模数据的PostgreSQL。
        # 5.课程安排及硬件准备
         - 时间：3小时
         - 语言：中文、英文均可
         - 硬件要求：
            操作系统：Linux/Windows Server 2008+，Ubuntu 16+
            CPU：1核以上
            RAM：1GB以上
            硬盘空间：20GB以上
            网络条件：良好的网络连接

         # 6.第一章 入门
          # 6.1 安装与环境配置
           本章节将详细介绍如何安装、配置PostgreSQL服务器，并检查服务是否正常运行。
          ## 6.1.1 安装PostgreSQL服务器
          #### Ubuntu Linux
           使用apt-get安装PostgreSQL服务器：
             sudo apt-get update && sudo apt-get install postgresql
          #### Windows
          ## 6.1.2 配置PostgreSQL服务器参数
           配置文件一般位于/etc/postgresql/文件夹下。如安装包默认安装，配置文件为/etc/postgresql/9.x/main/postgresql.conf。
           打开该文件，根据需求调整以下参数：

           | 参数名称        | 推荐值     | 描述           |
           | :-------------: |:----------:| ----------------|
           | shared_buffers   | 至少256MB | 设置共享内存缓冲区大小，该大小决定了缓存的最大容量；建议设置物理内存的70%-80%左右。|
           | effective_cache_size | 至少512MB | 设置可用的总内存大小，该大小决定了系统能够使用的物理内存上限；建议设置为物理内存的80%-90%左右，否则可能会导致系统swap.|
           | work_mem    | 至少1MB | 设置每个连接的内存缓冲区大小，该大小决定了排序、hash操作的内存开销。建议设置物理内存的25%-30%左右。|
           | maintenance_work_mem    | 至少1MB | 设置维护模式下的内存缓冲区大小，该大小决定了vacuum、analyze操作的内存开销。建议设置物理内存的10%-20%左右。|
           | max_connections  | 600-1000  | 设置最大允许连接数量，建议设置为服务器的实际负载能够承受的最大连接数量。|

          其他参数根据实际情况进行调整，通常不需要调整。
         ### 6.1.3 检查PostgreSQL服务状态
          可以通过以下命令查看PostgreSQL服务状态：
          ```shell
          $ service --status-all | grep postgres
          ```
          如果输出结果中active (running)字样，说明Postgresql服务器正在正常运行。
         ### 6.1.4 创建数据库用户
          默认情况下，PostgreSQL服务启动后，会自动创建一个名为postgres的超级用户。这个用户拥有所有的权限，可以通过这个用户登录数据库并创建其它用户。因此，通常不需要再创建其它用户。
          当然，如果需要的话，也可以通过以下命令创建一个新的普通用户：
          ```sql
          CREATE USER newuser WITH PASSWORD 'password';
          ```
          此处的newuser为新用户的用户名，密码为password。创建成功后，新用户可以使用如下方式登录数据库：
          ```sql
          psql -U newuser -W
          Password for user newuser: password
          ```
          输入正确的密码后，即可登录数据库。

          **注意**：请牢记新用户的用户名和密码，之后可能会用到。


        # 7.第二章 SQL优化与PostgreSQL优化
          本章节主要介绍SQL优化和PostgreSQL数据库优化的方法。
         ## 7.2 SQL优化方法
           SQL优化方法分为静态优化和动态优化。
          - 静态优化指的是在编译期间优化查询计划，例如通过索引优化、查询重写、统计信息收集、查询优化器的选择等。
          - 动态优化指的是在运行期间优化查询计划，例如通过SQL Tuning Utility、pgstattuple等工具监测数据库负载并进行SQL语句的调优。
          下面介绍一些SQL优化方法。
         ### 7.2.1 慎用SELECT *
           SELECT * 会导致查询扫描整个表，可能会导致查询计划出现性能问题。应尽量避免在不清楚需要的字段的情况下使用SELECT * 。
         ### 7.2.2 LIKE前置条件筛选
           LIKE前置条件筛选可以加快LIKE查询的执行速度，先对待匹配的列进行筛选操作，然后再利用LIKE进行匹配。
         ### 7.2.3 分组过滤条件下推
           分组过滤条件下推是指先对数据进行聚集，然后在聚集基础上进行过滤，这样可以减少计算量，缩短查询的时间。
         ### 7.2.4 使用NOT EXISTS代替NOT IN
           NOT EXISTS不会返回重复的行，比NOT IN效率高，应优先考虑NOT EXISTS。
         ### 7.2.5 删除重复索引
           对于联合索引，如果其中某个列存在唯一值，则该列上的索引可以标记为唯一索引，不要对该列建立索引。删除重复索引可以减少索引的大小，提高索引的效率。
         ### 7.2.6 索引分裂
           索引分裂可以使查询处理更有效率。索引分裂策略包括等分和等距离。
         ### 7.2.7 修改操作需要索引
           对某些修改操作，比如UPDATE、DELETE等，修改的列上需要建立索引。
         ### 7.2.8 ORDER BY LIMIT分页
           如果查询涉及排序，而且排序后的记录比较多，则采用ORDER BY LIMIT分页的方式可以提升查询效率。
         ## 7.3 PostgreSQL优化方法
         ### 7.3.1 查询优化器选择
           在PostgreSQL中，可以使用EXPLAIN命令查看查询优化器选择的查询计划，其中包括基准测试、估算算法、索引扫描、索引合并等过程。
           EXPLAIN输出中包括的字段包括：

           - Id：查询中每一个操作的序列号，编号越小，表示操作优先级越高。
           - Selectivity：表示查询条件与索引之间的匹配程度。
           - Filter：表示查询条件。
           - Rows：表示扫描的行数。
           - Width：表示扫描的字节数。
           - Shared Hit Blocks：表示共享块命中次数。
           - Local Hit Blocks：表示本地块命中次数。

           可以结合实际情况，选择合适的查询计划。
         ### 7.3.2 索引结构
           PostgreSQL支持BTree、Hash、Gin、Brin等索引结构。其中Btree是最常用的索引结构，支持范围查找、排序、散列分布等操作。另外，GIN索引可以实现精确匹配、多重匹配、数组元素匹配等操作，有效防止注入攻击。
           由于PostgreSQL支持多种索引结构，因此选择最适合业务场景的索引结构非常重要。
         ### 7.3.3 调整AutoVacuum参数
           AutoVacuum是一个定时任务，在一定时间内扫描死行并释放资源，减轻数据库压力。可以通过以下命令查看当前AutoVacuum参数：
           ```sql
           SHOW autovacuum_enabled;
           SHOW vacuum_cost_delay;
           SHOW vacuum_cost_limit;
           ```
           可以根据实际情况，调整这些参数，以减少AutoVacuum的频率和影响范围。
         ### 7.3.4 锁等待超时时间设置
           Lock wait timeout参数用来设置事务等待锁超时的时间。设置长短依据数据库的负载、事务大小、机器性能等因素。
         ### 7.3.5 临时表和内存表
           根据业务场景，选择适合的临时表和内存表可以降低数据库压力，提升查询效率。
         ### 7.3.6 监控数据库负载
           在生产环境中，需要持续跟踪数据库的负载情况。可以通过监控日志、系统监控、采样统计等方式进行。
         # 8.第三章 PostgreSQL数据库性能优化
          本章节将详细介绍PostgreSQL数据库的性能优化方法。
         ## 8.3 测试硬件配置
          为了做好性能优化，首先需要了解服务器硬件配置。
         ### 8.3.1 CPU性能
          在高性能服务器上部署PostgreSQL数据库，CPU性能是其关键特征。建议使用Intel Xeon或者AMD EPYC服务器，其中Xeon CPU一般具有AVX2指令集，能够获得更好的查询性能。
         ### 8.3.2 内存性能
          内存性能也很重要，一般情况下，内存的大小直接决定着数据库的处理能力。如果内存太小，则会导致缓存不足，甚至可能导致内存溢出错误。如果内存过小，也可能导致系统swap频繁。
         ### 8.3.3 I/O性能
          磁盘I/O也是影响数据库性能的关键因素。建议使用SSD固态硬盘。另外，建议使用专用的RAID卡提高磁盘IOPS。
         ### 8.3.4 网速
          网速也会影响数据库性能。建议选择低延迟的网络链路。
         ## 8.4 服务器性能测试工具
         ### 8.4.1 pg_top
         pg_top是PostgreSQL官方提供的一款性能监视工具，可以监测数据库的性能，包括CPU占用、内存占用、磁盘读写、网卡收发包速率、连接数、进程数、负载均衡等指标。安装命令如下：
         ```shell
         sudo apt-get install pg_top
         ```
         ### 8.4.2 pt-query-digest
         pt-query-digest是另一款开源的MySQL性能分析工具，能够解析慢日志，并按照一定规则生成报告。安装命令如下：
         ```shell
         wget https://www.percona.com/downloads/percona-toolkit/LATEST/binary/tarball/pt-query-digest-latest.tar.gz
         tar xzf pt-query-digest*.tar.gz
         cd percona-toolkit-*
        ./configure
         make
         make install
         ```
         ### 8.4.3 pgbadger
         pgbadger是一款开源的PostgreSQL日志分析工具，可以生成HTML报告，展示慢日志、慢查询、复制延迟等信息。安装命令如下：
         ```shell
         git clone https://github.com/dalibo/pgbadger.git
         cd pgbadger
         sudo make install
         ```
         ### 8.4.4 pgwatch2
         pgwatch2是一款开源的PostgreSQL性能监视工具，可以监测PostgreSQL集群的各种指标，包括CPU占用、内存占用、磁盘IO、网络流量等，并生成图表展示。安装命令如下：
         ```shell
         curl -fsSL https://install.directadmin.com/pgwatch2/install.sh | bash
         ```
         ### 8.4.5 PgBouncer
         PgBouncer是一款开源的PostgreSQL连接池，可以提高数据库连接的复用率。安装命令如下：
         ```shell
         sudo apt-get update && sudo apt-get install pgbouncer
         ```
         ## 8.5 服务器性能测试方法
         ### 8.5.1 测试场景
         测试场景需要反映生产环境中典型负载模式。
         ### 8.5.2 测试方法
         测试方法可以包括压力测试、负载测试、峰值测试等。
         ### 8.5.3 压力测试
         压力测试是对数据库处理请求能力的评估。压力测试时长一般为一天，周期性进行。常用的压力测试工具有Apache Bench和PgBench。Apache Bench是Apache官方提供的一个压力测试工具，安装命令如下：
         ```shell
         sudo apt-get install apache2-utils
         ```
         使用示例如下：
         ```shell
         ab -n 100000 -c 100 http://localhost/
         ```
         表示每秒发送100个请求，共发送10万次请求。
         PgBench是由PostgreSQL提供的一个压力测试工具，安装命令如下：
         ```shell
         sudo apt-get install postgresql-contrib
         ```
         使用示例如下：
         ```shell
         pgbench -c 10 -T 60 testdb
         ```
         表示开启10个连接并保持60s，运行testdb数据库。
         ### 8.5.4 负载测试
         负载测试是数据库在一段时间内接收请求的能力，即响应时间。负载测试可以模拟日常业务高峰期，并对响应时间进行监控。
         ### 8.5.5 峰值测试
         峰值测试是模拟一个突发流量时刻的数据库性能。峰值测试可以发现数据库的各种性能瓶颈，如硬件资源不足、垃圾回收、网络传输等。
         ## 8.6 数据库性能优化
         ### 8.6.1 连接池
         通过使用连接池，可以有效减少数据库连接数，提高数据库吞吐量。PostgreSQL提供了两种连接池实现：PgBouncer和PgPool-II。
         #### PgBouncer
         PgBouncer是PostgreSQL官方发布的连接池组件，支持预连接、负载均衡、连接池管理等功能。
         ##### 安装PgBouncer
         ```shell
         sudo apt-get update && sudo apt-get install pgbouncer
         ```
         ##### 配置PgBouncer
         PgBouncer的配置文件路径为/etc/pgbouncer/pgbouncer.ini，主要配置项包括：
         - max_client_conn：服务器最大允许连接数，建议设置为服务器的实际负载能够承受的最大连接数量。
         - default_pool_size：默认连接池大小，建议设置为实际需要的最小连接数。
         - reserve_pool_size：保留连接池大小，默认为0。当服务器负载增加时，可以将reserve_pool_size设定的值增加，以预留足够的空闲连接供服务器使用。
         - server_round_robin：是否启用轮询机制，默认为true。
         - ignore_startup_parameters：忽略连接池参数，防止连接失败。
         - log_queries：是否记录慢查询。
         - stats_period：统计信息刷新间隔，默认为60。
         - application_name：应用名称，可以在PgBouncer的日志中看到。
         #### PgPool-II
         PgPool-II是基于PgBouncer开发的高可用连接池。
         ##### 安装PgPool-II
         ```shell
         sudo apt-get install pgpool2
         ```
         ##### 配置PgPool-II
         PgPool-II的配置文件路径为/etc/pgpool2/pgpool.conf，主要配置项包括：
         - port：监听端口，默认为9999。
         - listen_address：监听地址，默认为环回地址。
         - backend_hostname：后端服务器的主机名。
         - backend_port：后端服务器的端口。
         - pooling_mode：连接池模式，默认off。
         - pid_file：pid文件位置。
         - logging_module：日志模块，默认为stderr。
         - num_init_children：初始化子进程个数。
         - admin_users：管理员用户名列表，用于远程监控。
         - ignore_startup_parameters：忽略连接池参数，防止连接失败。
         - client_idle_timeout：客户端空闲超时时间，默认0。
         - server_idle_timeout：服务器空闲超时时间，默认60。
         - dns_lookup_time_out：DNS解析超时时间，默认10。
         - server_connect_timeout：服务器连接超时时间，默认15。
         - query_timeout：查询超时时间，默认0。
         ### 8.6.2 查询优化器
         不同版本的PostgreSQL查询优化器的特性和优化方式都不同。最新版的PostgreSQL 13采用基于成本的优化器，能够给出更好的查询计划。
         #### PG_STAT_ALL_TABLES视图
         PG_STAT_ALL_TABLES视图提供了最详细的表级别的性能统计信息。
         ##### 安装PG_STAT_ALL_TABLES扩展
         ```sql
         CREATE EXTENSION pg_stat_all_tables;
         ```
         ##### 查看表级别性能统计信息
         ```sql
         SELECT relname, seq_scan, idx_scan FROM pg_stat_all_tables WHERE schemaname='public' AND n_live_tup > 0;
         ```
         #### 基于成本的优化器
         PostgreSQL 13采用基于成本的优化器，能够生成更加优秀的查询计划。
         ##### 配置成本模型
         成本模型文件路径为/etc/postgresql/13/main/cost_based.yaml，里面包含各种操作的权重、函数调用的次数、IO代价等。
         ```yaml
         query_tree_cache_size: 1000
         statement_mem: 100MB
         from_collapse_limit: 8
         join_collapse_limit: 8
         force_parallel_mode: off
         min_parallel_relation_size: 10KB
         cpu_tuple_cost: 0.01
         io_tuple_cost: 0.02
         network_tuple_cost: 0.01
         memory_tuple_cost: 0.05
         parallel_setup_cost: 1000
         parallel_tuple_cost: 0.1
         parameter_selectivity: 0.3
         plan_cost_type: dynamic
         random_page_cost: 4
         effective_io_concurrency: 2
         operator_precedence_warning: off
         disable_index_joins: on
         lop_multi_range_opt_min_len: 10
         enable_seqscan: true
         enable_mergejoin: true
         enable_nestloop: true
         enable_indexscan: false
         enable_bitmapscan: false
         enable_tidscan: false
         log_duration: on
         debug_print_plan: false
         enable_custom_plans: []
         custom_plans: {}
         buffer_cache_size: 256MB
         shared_buffers: 1GB
         temp_buffers: 8MB
         work_mem: 16MB
         maintenance_work_mem: 64MB
         max_prepared_transactions: 10
         wal_buffers: 16MB
         default_statistics_target: 1000
         random_seed: 1234
         checkpoint_segments: 32
         checkpoint_completion_target: 0.5
         max_wal_senders: 0
         max_replication_slots: 0
         archive_command: ''
         archive_timeout: 0
         track_commit_timestamp: off
         recovery_min_apply_delay: 0
         hot_standby: off
         allow_non_superusers_to_modified_gtid_set: on
         min_recovery_end_pos: 0/0
         apply_batch_size: 100
         max_locks_per_transaction: 64
         fsync: on
         data_checksums: off
         pg_stat_statements.max: 1000
         pg_stat_statements.track: all
         pg_stat_statements.save: on
         pg_stat_statements.stat_statements_max: 5000
         explain_verbose: off
         jsonb_array_element_warning: off
         log_parser_stats: off
         check_function_bodies: on
         standard_conforming_strings: on
         backslash_quote: safe_encoding
         sql_inheritance: on
         planner_cost_constants: array_1=0.01::real,cpu_operator_cost=0.0025::real,cpu_tuple_cost=0.001::real,random_page_cost=4.0::real,memory_maintenance_workers=0::integer,bgwriter_lru_maxpages=100::integer,bgwriter_lru_multiplier=2.0::real,geqo_effort=5.0::real,default_statistics_target=-1.0::integer,effective_cache_size=5GB::bigint
         index_lookup_info: off
         use_secondary_engine: on
         default_with_oids: off
         enable_partitionwise_aggregate: on
         enable_partitionwise_join: on
         allow_system_table_mods: superuser
         pg_hint_plan.enable_hint: on
         pg_hint_plan.enable_hint_log: on
         timescaledb.enable_collective_insert: off
         ```
         ##### 查询优化器提示
         可以使用EXPLAIN ANALYZE命令获取PostgreSQL的查询优化器的提示，即启发式的查询计划。
         ```sql
         EXPLAIN ANALYZE SELECT * FROM t1 JOIN t2 ON t1.id = t2.id;
         ```
         ### 8.6.3 服务器性能调整
         #### TCP参数优化
         TCP参数优化可以提升服务器的网络性能。
         ##### sysctl设置
         ```shell
         echo "net.core.rmem_max=16777216" >> /etc/sysctl.conf
         echo "net.core.wmem_max=16777216" >> /etc/sysctl.conf
         echo "net.ipv4.tcp_rmem='4096 87380 16777216'" >> /etc/sysctl.conf
         echo "net.ipv4.tcp_wmem='4096 65536 16777216'" >> /etc/sysctl.conf
         echo "net.ipv4.tcp_tw_reuse=1" >> /etc/sysctl.conf
         echo "net.ipv4.ip_local_port_range='10000 65535'" >> /etc/sysctl.conf
         sysctl -p
         ```
         ##### TCP_DEFER_ACCEPT
         ```shell
         echo "net.ipv4.tcp_defer_accept=0" >> /etc/sysctl.conf
         sysctl -p
         ```
         #### 磁盘IO优化
             优化磁盘IO可以提升数据库的磁盘访问速度。
         ##### ioengine设置
         ```shell
         echo "scsi_mod.use_blk_mq=1" >> /etc/multipath.conf
         multipath -Fvg
         blkdiscard -f /dev/sda*
         dd if=/dev/zero of=/mnt/largefile bs=1M count=1k oflag=direct
         blockdev --setra 64k /dev/sda1
         blockdev --setfra 64k /dev/sda1
         sed -i "/^Defaults.*requiretty$/ s/^/#/" /etc/sudoers
         sed -i "/^sed -i$/ d" /var/spool/cron/root
         sed -i "/^rm -rf \/tmp$/ d" /var/spool/cron/crontabs/root
         sync
         swapon -a
         ```
         ##### 磁盘阵列优化
         RAID 0 或 RAID 1 阵列可以提升磁盘IOPS，适用于多数场景。
         ##### 文件系统优化
         ext4 格式的文件系统能够支持大文件，建议使用。
         ```shell
         mkfs.ext4 /dev/sdb
         mount /dev/sdb /data
         echo "/dev/sdb /data ext4 defaults 0 2" >> /etc/fstab
         ```
         btrfs 文件系统适用于内存密集型场景。
         ```shell
         mkfs.btrfs /dev/sdb1
         mount /dev/sdb1 /data
         mkdir -p /data/pgsql
         chown postgres:postgres /data/pgsql
         echo "/dev/sdb1 /data btrfs defaults 0 2" >> /etc/fstab
         ```
         XFS 文件系统能够提供更快的性能，适用于高吞吐量场景。
         ```shell
         mkfs.xfs /dev/sdb1
         mount /dev/sdb1 /data
         mkdir -p /data/pgsql
         chown postgres:postgres /data/pgsql
         echo "/dev/sdb1 /data xfs defaults 0 2" >> /etc/fstab
         ```
         ZFS 文件系统能够提供高可用性和可伸缩性，适用于大数据场景。
         ```shell
         zpool create pool raidz /dev/sdb1
         zfs set atime=off pool/pgdata
         mkdir -p /data/pgsql
         chmod 777 /data/pgsql
         echo "/dev/mapper/pool-pgdata /data/pgsql xfs noatime 0 0" >> /etc/fstab
         echo "zfs set atime=off pool/pgdata" >> /etc/rc.local
         ```