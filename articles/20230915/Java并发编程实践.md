
作者：禅与计算机程序设计艺术                    

# 1.简介
  


在多线程领域，要实现高效的并发编程需要对相关的概念、术语以及核心算法有比较深入的理解。本书将通过讲解一些精髓的并发编程知识，帮助读者进一步提升对并发编程的理解，并掌握有效的并发解决方案。本书的主要内容包括并发基础知识、线程池、锁、并发容器、同步工具类等，希望能够帮助读者开发出更加健壮、可靠以及高性能的并发应用。


作者简介：唐炳权，4年工作经验，计算机科学与技术专业，曾任职于微软亚洲研究院，现任职于中国移动通信集团信息化部。本书的目的是为了帮助读者提升对并发编程的理解，以及构建自己独特的并发模式，其中包含了“共享内存模型、线程安全性、可见性、原子性、指令重排序、happens-before原则、线程状态、阻塞队列、生产消费模式、栅栏模式、分离器模式、闭包锁、信号量”等多个方面的知识。并且提供相应的代码实例及实战分析，能助力读者加深对并发编程的理解。
# 2.准备
阅读本书前，建议您具备如下的基础知识：

1. Java语言基础知识：了解面向对象、封装、继承、多态等概念，掌握泛型、注解、反射、异常处理、流处理等语法特性；

2. 操作系统相关知识：掌握操作系统的进程/线程、调度、内存管理、同步互斥机制、文件I/O等基础知识；

3. 数据结构相关知识：了解链表、哈希表、堆栈、队列、树形数据结构、图形数据结构等基本的数据结构；

4. 设计模式相关知识：了解常用设计模式如单例模式、工厂模式、策略模式、适配器模式、观察者模式、模板方法模式等，掌握这些模式的设计思想；

5. Linux相关知识：了解Linux内核中的进程、线程、协程等概念，以及各种锁、同步原语的实现方式。

# 3.目录
1. 并发编程概论
    1. 什么是并发？为什么要并发？
        1.1. 并发是什么？ 
        1.2. 为什么要并发？ 
    1. 并发的优缺点
        1.3. 并发的优点
        1.4. 并发的缺点
        1.5. 并发存在的问题
    1. Java 5 中引入的并发特性
        1.6. Executor Framework
        1.7. Fork/Join 池
        1.8. Java Memory Model（JSR-133）
        1.9. CyclicBarrier 和 Phaser
    1. 其他重要概念
        1.10. 竞争条件（Race Condition）
        1.11. 临界区（Critical Section）
        1.12. 可见性（Visibility）
        1.13. 有序性（Ordering）
        1.14. 原子性（Atomicity）
        1.15. 指令重排（Instruction Reordering）
        1.16. happens-before 关系
        1.17. 同步类型
            1.17.1. 互斥同步 
            1.17.2. 条件变量同步 
            1.17.3. 屏障同步 
        1.18. 线程状态
        1.19. 阻塞队列
        1.20. 生产消费模式
        1.21. 分离器模式
        1.22. 闭包锁
        1.23. 信号量 
        1.24. 执行框架 
    1. 测试与调优
        1.25. 测试并发程序 
        1.26. 避免死锁 
        1.27. 监控并发程序 
    1. 小结 
2. 共享内存模型
    2.1. 什么是共享内存模型？ 
    2.2. 内存间共享问题
    2.3. volatile关键字
        2.3.1. volatile 的作用
        2.3.2. volatile 不能保证原子性 
        2.3.3. volatile 的内存语义 
        2.3.4. 使用场景 
    2.4. synchronized关键字
        2.4.1. synchronized 锁的升级过程 
        2.4.2. synchronized 锁的优化 
    2.5. 锁消除与锁粗化
        2.5.1. 锁消除 
        2.5.2. 锁粗化 
    2.6. 适应性锁 VS 非适应性锁
        2.6.1. 适应性锁 
        2.6.2. 非适应性锁 
    2.7. 小结 
3. 线程安全性 
    3.1. 何为线程安全？ 
    3.2. 对象创建时不必要的同步
    3.3. 不正确地发布对象
    3.4. 可变对象的并发访问 
    3.5. 线程本地存储 
        3.5.1. 什么是线程本地存储？ 
        3.5.2. ThreadLocal 类 
        3.5.3. 线程局部变量的内存泄露问题 
    3.6. Copy-on-Write 机制
        3.6.1. Copy-on-Write 是什么？ 
        3.6.2. CopyOnWriteArrayList 类 
        3.6.3. CopyOnWriteArraySet 类 
    3.7. 小结 
4. 可见性与同步
    4.1. 可见性问题 
        4.1.1. 可见性问题是什么？ 
        4.1.2. synchronized 关键字如何保证可见性？ 
        4.1.3. volatile 关键字的可见性问题 
    4.2. 双端同步（Baririer Synchronization）
        4.2.1. Baririer Synchronization 是什么？ 
        4.2.2. CyclicBarrier 类 
        4.2.3. Phaser 类 
    4.3. 延迟初始化（Lazy Initialization）
        4.3.1. 概念 
        4.3.2. DCL(Double Check Locking)模式 
    4.4. 顺序一致性内存模型
        4.4.1. JSR-133 内存模型 
        4.4.2. Volatile 的禁忌之处 
    4.5. 小结 
5. 线程池 
    5.1. 概念 
        5.1.1. 线程池的概念 
        5.1.2. Executor Framework 的目的 
        5.1.3. Executors 中的各个方法 
    5.2. ThreadPoolExecutor 类 
        5.2.1. 创建线程池的方法 
        5.2.2. 线程池的生命周期 
        5.2.3. 执行任务 
        5.2.4. 设置线程池的参数 
        5.2.5. 提交拒绝策略 
    5.3. ScheduledThreadPoolExecutor 类 
    5.4. 线程池的关闭 
    5.5. 扩展线程池 
        5.5.1. 自定义线程工厂 
        5.5.2. 定制线程名称 
    5.6. 小结 
6. 同步工具类
    6.1. CountDownLatch 类 
    6.2. CyclicBarrier 类 
    6.3. FutureTask 类 
    6.4. Semaphore 类 
    6.5. Exchanger 类 
    6.6. LockSupport 类 
    6.7. Condition 接口 
    6.8. 小结 
7. 阻塞队列 
    7.1. 概念 
        7.1.1. 阻塞队列的概念 
        7.1.2. 为什么要使用阻塞队列？ 
    7.2. BlockingQueue 接口 
        7.2.1. put() 方法 
        7.2.2. take() 方法 
        7.2.3. offer() 方法 
        7.2.4. poll() 方法 
        7.2.5. add() 方法 
        7.2.6. element() 方法 
        7.2.7. remove() 方法 
        7.2.8. contains() 方法 
        7.2.9. clear() 方法 
        7.2.10. size() 方法 
        7.2.11. toArray() 方法 
        7.2.12. remainingCapacity() 方法 
        7.2.13. drainTo() 方法 
        7.2.14. iterator() 方法 
        7.2.15. spliterator() 方法 
        7.2.16. isParallel() 方法 
        7.2.17. getQueue() 方法 
    7.3. ArrayBlockingQueue 类 
    7.4. LinkedBlockingQueue 类 
    7.5. PriorityBlockingQueue 类 
    7.6. DelayQueue 类 
    7.7. SynchronousQueue 类 
    7.8. 小结 
8. 生产者消费模式
    8.1. 概念 
        8.1.1. 生产者消费模式的概念 
        8.1.2. 模式的特点与应用场景 
    8.2. 共享资源的访问控制
        8.2.1. 采用 synchronized 或 ReentrantLock 进行资源访问控制 
        8.2.2. 多线程模式下的线程封闭 
        8.2.3. volatile 变量与内存可见性 
        8.2.4. CountDownLatch 类 
    8.3. BlockingQueue 作为生产者与消费者之间的缓冲区 
        8.3.1. 在 BlockingQueue 上使用循环等待 
        8.3.2. 使用超时等待 
        8.3.3. 记录任务完成情况 
        8.3.4. 消费者的失败恢复 
        8.3.5. 生产者与消费者的异常处理 
    8.4. 小结 
9. 栅栏模式
    9.1. 概念 
        9.1.1. 栅栏模式的概念 
        9.1.2. 为什么要使用栅栏模式？ 
    9.2. Barrier 类 
    9.3. RunnableFuture 接口 
    9.4. 小结 
10. 分离器模式
    10.1. 概念 
        10.1.1. 分离器模式的概念 
        10.1.2. 为什么要使用分离器模式？ 
    10.2. AcquireShared 接口 
    10.3. ReleaseShared 接口 
    10.4. SmallTasks 类 
    10.5. LargeTask 类 
    10.6. Computation 类 
    10.7. CustomFuture 类 
    10.8. 小结 
11. 闭包锁（Closure Locks） 
    11.1. 概念 
        11.1.1. 闭包锁的概念 
        11.1.2. 为什么要使用闭包锁？ 
    11.2. ReadWriteLock 接口 
    11.3. ReentrantReadWriteLock 类 
    11.4. 小结 
12. 信号量（Semaphore） 
    12.1. 概念 
        12.1.1. 信号量的概念 
        12.1.2. 为什么要使用信号量？ 
    12.2. acquire() 方法 
    12.3. release() 方法 
    12.4. tryAcquire() 方法 
    12.5. tryRelease() 方法 
    12.6. availablePermits() 方法 
    12.7. permits() 方法 
    12.8. 小结