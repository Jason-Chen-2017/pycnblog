
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         MySQL 是最流行的关系型数据库之一，拥有广泛的应用市场，是一个成熟、稳定的产品。但是，随着业务的发展，高并发访问场景下，数据库的性能可能会出现各种瓶颈。其中，最严重的一种情况就是 MySQL 的死锁问题。

         　　什么是死锁？

         在多线程环境中，如果两个或多个线程相互持有对方需要的资源而不释放，就会导致死锁现象的发生。在一个事务中，若某一资源被占用，则其他事务不能继续完成，就会造成死锁，各个事务都无法正常结束，称之为死锁。

　　为什么会产生死锁呢？

　　死锁的产生是由于并发进程中的锁分配不当或者死锁检测与恢复策略不当所致，它是一种非常隐蔽且复杂的问题。

　　产生死锁的四个必要条件：

　　1）互斥条件：指进程独占资源，即同时申请临界资源；

　　　　　　　　　 
　　2）请求和保持条件：指进程已经提出了一个资源请求，但资源可能一直处于保持状态，不能被强占；

　　　　　　　　　 
　　3）不剥夺条件：指进程获得的资源在未使用完毕之前不能强行剥夺，只能按照其占用的顺序释放；

　　　　　　　　　 
　　4）环路等待条件：指在发生死锁时，必然存在一条链路上的进程——资源的循环等待，如 A 等待 B，B 等待 C，C 等待 A，这样形成了一个环路结构。

　　MySQL 死锁产生的原因

　　在 MySQL 中，不同类型的死锁都会引起不同的报错信息，包括：

　　- “Deadlock found when trying to get lock; try restarting transaction”：这种错误一般发生在 AUTO_INCERMENT 锁竞争或 insert/update 语句产生表锁的情况下，在这种情况下，一般都是由于自增主键回滚，导致其它线程未能及时释放锁。

　　- “Lock wait timeout exceeded; try restarting transaction”：这种错误一般发生在多个客户端试图获取同一张表的相同记录锁，但由于锁的兼容性导致无法获取锁，导致客户端被阻塞在请求上，永远不会得到锁，造成死锁。

　　- “Too many requests waiting for this table”：此类错误一般发生在删除表时的 DROP TABLE 上，由于该表仍被某个客户端使用，导致无法删除成功，一直处于等待锁状态。

 

# 2.MySQL 死锁分析与处理

  ## 2.1 概述

  ###  2.1.1 死锁概念

  当多个事务并发执行过程中，如果涉及的资源类别相同，每个事务在获取资源之前，先向系统申请进入此资源的排他锁或共享锁（也叫读锁或写锁）。申请过程如下：

  1) 每个事务首先申请自己需要的资源的排他锁（写锁），当申请到资源后便持有该资源，直至事务结束，释放资源。
  
  2) 如果其他事务的资源申请失败，则申请者则阻塞在原地等待，直至其他所有资源申请者释放自己的资源。
  
  如果多个事务在无限期等待下，仍不能获得所需资源，就产生了死锁。
  
 ![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpdmVuX3BuZw?x-oss-process=image/format,png)

  有两种解决死锁的方法：

  1）抢占资源法：系统在给某些进程分配资源的时候，先确定哪些进程是死锁的祸首，然后直接将这些死锁进程给予杀死命令，从而实现进程间的正常调度。这种方法比较简单，但是容易造成系统负载过高。

  2）超时回退法：系统设置一段时间，超过这个时间限制，系统才会自动进行死锁回滚。这种方法能够有效防止死锁发生，但是也会影响效率。

  ### 2.1.2 死锁日志

  当 MySQL 服务运行时，如果发生死锁，会自动生成死锁日志文件。死锁日志默认存放在 MySQL 数据目录下的 error.log 文件中，可以通过设置“show variables like '%log%'”来查看日志相关配置项。当日志级别设置为“info”，并且打开了引擎盖锁，系统会记录所有发生的死锁。

  ```
  [Note] InnoDB: Highest supported file format is Barracuda.
  [Warning] InnoDB: Starting a new session, Transaction model changed from "null" to "pessimistic".
  INFO 2016-05-17T11:50:11.622753Z mysqld got signal 11 ;
This could be due to timeout (handled by core) or memory exhaustion (not handled by core). Consider increasing thread stack size or increasing max_connections.

InnoDB: Operating system error number 11 in a file operation.
InnoDB: The error means mysqld received a signal requesting an I/O operation and was unable to complete it within the specified time period. It may indicate heavy disk activity or other filesystem problems. Check the operating system error log for additional information.
  ```

  ```
  ERROR 1213 (40001): Deadlock found when trying to get lock; try restarting transaction
  ```

  ```
  ERROR 1205 (HY000): Lock wait timeout exceeded; try restarting transaction
  ```

  ```
  ERROR 1062 (23000): Duplicate entry 'xxxx' for key 'PRIMARY'
  ```

  ### 2.1.3 死锁检测与恢复

  检测死锁的条件：

  1) 设置全局参数innodb_deadlock_detect：开启死锁监测功能。
  
  2) 执行set innodb_print_all_deadlocks=on，将所有死锁的信息打印到日志中。
  
  3) 使用show engine innodb status命令查看死锁信息。
  
  如果发现死锁，则通过以下方式进行恢复：
  
  1) 使用rollback命令回滚当前事务。
  
  2) 将启动参数innodb_kill_blocking_transaction设置为ON，表示系统应当终止那些妨碍死锁的事务。
  
     注：innodb_kill_blocking_transaction参数会降低mysql服务器的并发度，因为它将导致更多的锁冲突。
  
   3) 通过show processlist命令查看当前活动的事务列表。
  
  4) 根据死锁日志文件或show engine innodb status命令获取死锁信息。
  
  5) 根据死锁信息，分析出死锁的周期性及产生的原因，并制定相应的恢复办法。

  ## 2.2 死锁原理分析

  ### 2.2.1 概述

  理解死锁的关键是理解每个事务在申请资源时可能遇到的两种情况。第一个情况是申请资源失败，也就是阻塞；第二种情况是申请不到资源，无法继续执行，那么如何解决死锁呢？

  ### 2.2.2 死锁的三种情况

  #### 1、互斥条件

  对某资源进行S锁，则不能再对该资源进行任何其他类型的锁请求，直到该事务释放S锁。例如：事务1已对A资源做S锁，事务2申请对A资源做X锁，就会发生互斥条件的死锁。

  #### 2、请求和保持条件

  事务1已对资源A做X锁，事务2也想申请对资源A做X锁，但是事务2必须等到事务1释放A的X锁才能继续，事务2又申请对资源B做X锁，事务1也想申请对资源B做XLOCK，但是事务1必须等到事务2释放资源B的锁才能继续。也就是说，在任意时刻，只能有一个事务持有对资源的锁，另一个事务必须等待该事务释放锁之后才能继续。

  #### 3、非抢占条件

  对于已经获得锁的事务来说，新来的事务必须按照申请的顺序获得锁，不能插队。例如：事务1已对资源A做S锁，事务2申请对资源A做S锁，则会发生非抢占条件的死锁。

  ### 2.2.3 死锁的预防

  死锁预防包括三个方面：破坏互斥条件、缩小请求范围、排序请求。

  - **破坏互斥条件**：数据库设计时要尽量保证不会发生互斥条件的死锁，比如设置索引，使数据插入和更新变得更加安全。
  - **缩小请求范围**：在发生死锁时，系统会试图找到一种规避死锁的方法。首先它会尝试强制转换一些锁，比如升级IX锁为IS锁。其次，它还可以采用一些手段减少并发度，如分摊开销，按阶段分配锁，保证公平性。
  - **排序请求**：根据锁的轻重缓急，把对相同资源的请求划分优先级，从而减少死锁的发生。

  ### 2.2.4 死锁的解除

  死锁的解除是通过判断是否存在死锁，然后让进程释放锁，让进程重新按照申请的顺序获得锁。如果所有的锁都被占用，而且已经有多个事务在等待资源，那么这时只能等待直到有其他进程释放锁，系统才能继续运行。

  ### 2.2.5 死锁的检测与恢复

  检测死锁有两种方式：

  1. 设置参数innodb_deadlock_detect。
  2. set innodb_print_all_deadlocks=on，将所有死锁的信息打印到日志中。

  检测死锁后，通过show engine innodb status命令查看死锁信息。

  如果发现死锁，则通过以下方式进行恢复：

  1. 使用rollback命令回滚当前事务。
  2. 将启动参数innodb_kill_blocking_transaction设置为ON，表示系统应当终止那些妨碍死锁的事务。
  3. 通过show processlist命令查看当前活动的事务列表。
  4. 根据死锁日志文件或show engine innodb status命令获取死锁信息。
  5. 根据死锁信息，分析出死锁的周期性及产生的原因，并制定相应的恢复办法。

