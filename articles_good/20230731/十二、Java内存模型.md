
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1995 年的一份文档中首次明确地定义了 Java 内存模型（JMM），它把 Java 虚拟机中的多线程之间的共享变量存取过程抽象成为主存、本地内存和每个线程的工作内存三个部分，并且说明如何在这些内存之间有效地同步数据。

         2003 年，HotSpot JVM 实现了 JMM，并通过内存屏障等技术来强制实施其内存模型。在 HotSpot VM 中，Java 对象是存在于堆内存中的，而栈则存放在线程私有的虚拟内存中——但其实栈上的局部变量也是存放在局部变量表（Local Variable Table，即“局部变量数组”）中的。

         2011 年，为了提升 JVM 的性能，Sun Microsystems 从 JDK 1.5 开始就引入了分代收集算法（Generational Collection），将堆划分成多个独立的空间，每一个空间用作新生代（Young Generation）或老年代（Old Generation）。显然，这种划分让对象的生命周期被细化到了两个相对孤立的时间段——初始分配（Allocation）期和复制（Copying/Evacuation）期。

         在默认情况下，JVM 将堆内存分为新生代和老年代，新生代主要用于对象较少的生命周期短的对象，老年代一般用于存放长期存活的对象。对于堆上对象的布局，JVM 给出了一套规则：

           1. 每个对象都有一个 12B 的头信息；
           2. 对齐指针：如果对象的大小不是 8B 的整数倍，那么它的大小会向上增加到下一个 8B 的整数倍；
           3. 对象头中的 Mark Word 用于记录对象是否已经被垃圾回收器扫描过，以及对象的哈希码值；
           4. 如果对象是一个数组，那么数组长度会被保存在头信息的第一个字段中；
           5. 如果对象是一个字符串或者包装类，那么字符数组地址会被保存在头信息的第二个字段中；
           6. 对象中的每个字段都由一个 Field Descriptors 描述；

         通过这套规则，JVM 可以轻松地计算指针偏移量，进而读写对象中的字段。由于头信息和对齐指针的存在，JVM 就可以保证每个对象的开头都是一个有效的地址，从而可以实现对齐访问。另外，JVM 使用线程私有的工作内存来保存当前线程正在运行的方法相关的数据，在方法执行时，编译器可以自由选择哪些数据需要存放在寄存器、栈上还是工作内存中。

         通过 JMM，JVM 能够以一种一致且廉价的方式管理内存，这是它成为高度优化的 Java 平台的一个重要原因。

         # 2.基本概念术语说明
         ## 2.1.主存和本地内存
         JMM 把主存和本地内存划分为三种类型：
           1. 主存（Main Memory）：也称为物理内存，是直接存取硬件的内存，所有处理器可读取的存储空间。在实际计算机系统中，主存通常被系统连续分配，虚拟内存只是对主存的抽象。
           2. 本地内存（Local Memory）：也称为私有内存，是处理器自己的内存，相对于主存来说，其速度要快得多。本文所讨论的内存区域就是指的本地内存。
           3. 方法区（Method Area）：属于堆内存的一部分，主要存储已被加载的类的元数据、静态变量、常量池和一些框架代码。

         本地内存又分为线程本地内存（Thread-local memory，TLM）和非线程本地内存（Non-thread-local memory，NTLM）。本地内存只为当前线程服务，不同线程之间不能共享。因此，每个线程只能拥有自己独立的 TLM 和 NTLM，其他线程则无法直接访问它们。

         下图展示了 JVM 内存区域示意图：
        ![JVM内存区域示意图](https://i.imgur.com/v7tT7uF.png)
         上图中的堆内存包括年轻代、老年代和永久代，这几块区域都是在同一块物理内存上进行分配的。而方法区和线程私有内存（TLM）是通过虚拟内存完成的，当他们需要扩容时，操作系统会先通过 swap 分配内存，再把不使用的内存归还给操作系统。

         根据 HotSpot VM 的实现，其中的永久代（Permanent Generation）已经成为历史遗留产物，这一部分是由 JVM 自动划分的，并不会出现在真实的物理内存中。JDK 9 将永久代替换成元空间（Metaspace），但是仍然保留了永久代的命名。

        ## 2.2.工作内存和主存
        JMM 规定了每个线程只能拥有自己的本地内存，并且规定线程对变量的所有操作都必须在自己的本地内存中进行，不能直接访问别的线程的本地内存。

        当线程运行一个方法的时候，这个方法的代码和数据会被加载到线程的工作内存（working memory）中。线程的工作内存存储着方法调用的参数、局部变量、临时变量等。

        当一条线程正在执行某个方法时，另一个线程可能正在修改该方法的局部变量的值。为了避免竞争，JMM 提供了 volatile 和 synchronized 关键字，来禁止编译器和处理器缓存，使得各线程都能看到内存中最新的值。

        当线程退出或者执行某个特定的方法时，工作内存就会释放掉，所有的更改都会失效。线程间相互隔离，互不干扰。

        ## 2.3.内存间交互协议
        为了实现线程之间数据的共享和传递，JMM 通过提供内存间交互协议（Memory Interactions Protocols）来约束内存的访问方式。

        JMM 的内存间交互协议包括以下两种：
           1. Happens-Before Ordering 法则：如果 A happens-before B，那么 A 对共享变量的写入操作一定发生在 B 对其的读操作之前。这是 JMM 内存模型的关键，因为它提供了happens-before的定义，可以通过它来判定一个操作在另一个操作之前执行，从而确定各个线程的操作顺序。

           2. Load-Store Barriers 机制：通过插入 Load-Store Barrier 指令，可以告诉处理器或编译器，在特定位置上对变量的读/写操作是原子的，从而可以避免或最小化数据同步的问题。例如，volatile 关键字的写操作后面隐含着一个 Store-Store Barrier 指令，用来确保前面的写入操作对其他处理器的可见性。

        ## 2.4.并发编程模型
        根据 JMM 的设计原则，JVM 支持两种并发编程模型：

         - 1. 共享内存模型（Shared Memory Model）：所有线程共同访问同一个内存，所有变量都通过主内存来通信，线程之间通过协作完成任务。
         - 2. 消息传递模型（Message Passing Model）：线程间没有共享内存，所有变量都在线程之间传递消息。由于线程之间通信需要花费时间，因此效率比共享内存模型低很多。

        # 3.Java 内存模型的关键特征及其实现原理
        JMM 主要关注以下几个方面：

         - 原子性（Atomicity）：对变量的读/写操作是原子的，要么全部成功，要么全部失败，不会出现读到半点不完整数据的情况。
         - 可见性（Visibility）：当一个线程修改了一个共享变量的值之后，新值对于其他线程来说是可以立即看得到的。换句话说，就是对于其他线程来说，该变量是更新后的最新值。
         - 有序性（Ordering）：程序执行的顺序按照程序代码的先后顺序执行。

        ## 3.1.原子性
        JMM 指定所有的变量的读/写操作都是原子操作，不可分割。因此，状态改变具有原子性，这点是通过 volatile 和 synchronize 来做到的。

        ### 3.1.1.volatile 关键字
        Volatile 是 Java 语言提供的一种有序可靠的同步机制，可以保证多线程访问volatile变量时的可见性和原子性。其作用是：

         - 保证可见性：当一个线程修改了 volatile 变量的值，新值对于其他线程是立即可见的。
         - 保证原子性：volatile 变量的读/写操作是原子的，因此能保证线程安全。

        正因为 volatile 变量保证了原子性，才允许多个线程同时读写一个变量，而不需要加锁。volatile 变量读写过程如下：

         - 线程 1：首先读取 volatile 变量，发现此时值为 0；
         - 线程 2：然后线程 2 也读取 volatile 变量，发现此时值为 0；
         - 线程 1：接着线程 1 修改 volatile 变量的值，改为 1；
         - 线程 2：最后线程 2 修改 volatile 变量的值，改为 1。

        在以上过程，volatile 变量保证了修改的可见性，但不是原子操作，线程 1 修改 volatile 变量的值后，线程 2 会立刻读取修改后的 volatile 变量，结果可能是 0 或 1。为了保证原子性，线程 1 需要在修改 volatile 变量后，线程 2 需要等待线程 1 操作完成后才能读取修改后的 volatile 变量。

        ### 3.1.2.synchronized 关键字
        Synchronized 是 Java 语言提供的另一种同步机制，主要用来保护临界资源的独占访问。其作用是确保一次只有一个线程可以访问临界资源，从而防止多个线程同时访问共享变量带来的线程安全问题。

        Synchronized 同步锁是可重入的，也就是同一线程在持有锁的情况下，再次申请该锁是可以成功的。同一个线程在每次获取锁时都应尽力不要将时间消耗在获取锁和释放锁上，以免影响其他线程的运行。而且Synchronized 是悲观锁，即认为任何时候都可能产生线程安全问题，因此其获得锁和释放锁时都需要进行必要的判断和处理。

        Synchonized 同步块的结构如下：

        ```java
        synchronized(this){ // 可选的对象参数
            // 临界区代码
        }
        ```

        Synchronized 关键字加锁后，线程将一直持有锁直到执行完临界区代码块，释放锁。但是，如果临界区代码块中的线程访问的是同一个对象，这可能会造成死锁。因此，建议在同步块的起始处仅声明所需的对象参数。

        ### 3.1.3.synchronized 实现原理
        Synchronized 关键字是通过操作系统的互斥量（Mutex）实现的，这是一个基于原子操作的同步工具。当多个线程访问同一个对象时，只有一个线程能成功地获取锁，后进入临界区执行代码，其他线程则要阻塞等待，直至获取锁的线程释放锁为止。

        Mutex 是一种特殊的进程同步信号，它是一个抽象概念，无法直接被应用程序控制。在 Mutex 的内部，存在着一个等待队列，被暂停的进程将在队列中等待。当可用资源变得可用时，Mutex 将唤醒一个等待进程，并将它转移到就绪态。

        如图所示，当多个线程试图获取相同的锁时，只有一个线程能成功地获取到锁，其他的线程将被阻塞，进入等待队列。等到互斥量被释放时，等待队列中的一个进程被唤醒，并被允许获取锁，进入临界区执行代码。

       ![](https://i.imgur.com/tKZlZFM.png)
        
        Synchronized 是通过操作系统互斥锁来实现的。线程在申请锁时，操作系统会检测是否有其他线程已经获取了该锁，如果已经有线程获得了锁，那申请锁的线程就进入等待状态，直到锁被释放后，该申请线程才能获得锁，继续执行。

        操作系统互斥锁的两种主要实现方式：

        1. 基于原语的互斥锁：互斥锁是由操作系统提供的原语函数。例如 Windows 中的互斥体对象 Mutex，它是一个内核对象，在用户态与内核态都能访问。用户态的线程对 Mutex 对象加锁后，便获得了对该对象的独占访问权限。
        2. 用户级锁（User Level Locks）：这种锁是由用户态的线程库提供的 API 函数。例如 pthread_mutex_lock() 函数，它通过系统调用（sycall）向内核发送请求，获得互斥锁。当锁被另一个线程请求时，pthread_mutex_lock() 函数会阻塞直到锁被释放。

        此外，为了保证线程安全，Synchonized 要求：

        1. 不允许随意降低同步对象的可见性；
        2. 不允许无限扩展同步对象，限制最大数量；
        3. 不允许嵌套同步；
        4. 对同一个对象不允许使用多个同步块；
        5. 同步块的结束必须要有匹配的 release 操作；
        6. 每个对象都应该有且只有一个对应的 monitor。

    ## 3.2.可见性
    可见性是指当一个线程修改了一个共享变量的值之后，新值对于其他线程来说是可以立即看得到的。换句话说，就是对于其他线程来说，该变量是更新后的最新值。

    JMM 通过提供 volatile 和 synchronized 来保证原子性和可见性。其中，volatile 是通过不断刷新主存中的值来达到可见性的效果。而 synchronized 是通过操作系统的互斥锁来实现的。

    可见性是通过 volatile 关键字和 synchronized 关键字来实现的。

    ### 3.2.1.volatile 可见性
    Volatile 变量的读/写操作是原子的，但不能完全保证可见性。因为其它线程的工作内存中可能没有重新从主存中刷新过的最新的值。

    为了解决此问题，Java 语言规范专门定义了一个Volatile 访问器（Volatile Accessor），当一个Volatile变量被写了新值后，它会通知其他线程中，从而强制使线程都从主存中重新读取该变量。这样，其他线程就能正确读取到最新的值。

    ### 3.2.2.synchronized 可见性
    当多个线程同时访问一个对象中的某个变量时，对变量的写操作要么全部执行，要么全部不执行。但是，当多个线程访问同一个变量时，却可能出现不可见现象。原因是 JVM 和处理器可能会对线程的调度进行乱序执行，导致各线程看到的共享变量状态不同步。

    为了解决这个问题，JMM 提供了 volatile 和 synchronized 关键字。volatile 可以保证变量的可见性，即一个线程修改变量的值，其他线程能立即感知到变化。而 synchronized 可以保证原子性，即一个线程对变量的写操作是串行的，彼此互斥；多个线程同时访问时，也能防止因竞争而导致的数据不一致。

    ### 3.2.3.内存屏障（Memory Barrier）
    为了实现可见性，编译器和处理器在生成代码时，会在指令序列中加入内存屏障（Memory Barrier）。内存屏障的作用是在单线程环境中，会实现对 volatile 变量的内存可见性，但在多线程环境中，却不能保证可见性。

    Java 内存模型中定义了 8 个 volatile 变量访问规则。

    1. rule1: a volatile writehappens before every volatile read in the same thread;
    2. rule2: a volatile writehappens before a volatile write in the same thread;
    3. rule3: a volatile writehappens before any subsequent non-volatile read or write in the same thread;
    4. rule4: a volatile writein one thread is immediately visible to all threads in the same cache line (at some undefined time after the write completes); and
    5. rules5 and 6 apply when multiple processors are present; and
    6. rule7: an unlock on a monitor synchronizes with the final store of a volatile variable by that thread; and
    7. rule8: an unlock on a monitor synchronizes with the final store of any previously acquired volatile variables by that thread, regardless of order.

    如图所示，规则 1 表示 volatile 变量写操作与后续 volatile 变量读操作总是伴随着一起出现，即一个 volatile 写操作先于 volatile 读操作。规则 2 表示一个 volatile 写操作先于另一个 volatile 写操作，但两者必须属于同一线程。规则 3 表示 volatile 写操作先于任意后续非 volatile 读或写操作，但必须属于同一线程。规则 4 表示一个线程对 volatile 变量写操作后，该变量的新值立即对所有线程可见（即时处于下一个处理器缓存行的位置）。规则 5 和规则 6 表示在多处理器环境中，volatile 写操作需要在一个处理器缓存行中发布（将新值刷入主存），保证对其他处理器缓存行的线程可见性。规则 7 表示 unlock 操作同步到 volatile 变量的最终 store 操作，即一个线程释放锁，必须在后面对该变量进行 store 操作。规则 8 表示 unlock 操作同步到之前已经 acquire 过的 volatile 变量的 store 操作。

   ![](https://i.imgur.com/kvU8UWG.png)

    通过内存屏障，JMM 为多线程环境下的 volatile 可见性提供保障。
    
    ### 3.2.4.双重检查锁定（Double Checked Locking）
    Double Checked Locking 是一种优化同步策略。它指的是先检查锁有没有被其他线程抢占过，如果没有抢占，才真正创建对象，否则就直接返回旧对象。

    但是，DCL 只适用于单例模式，在多线程环境下并不适用，所以 DCL 仅作为参考，并不是 Java 内存模型中的一条规则。

    ## 3.3.有序性
    有序性是指程序执行的顺序按照程序代码的先后顺序执行。

    JMM 通过内存屏障来禁止指令重排序。

    ### 3.3.1.指令重排序
    为了提高性能，处理器和编译器在生成代码时，可能会对指令进行重排序。但是，为了正确地执行有序性，编译器和处理器需要注意以下几点：

    1. 数据依赖关系（Data Dependency）：如果一个变量的后续操作需要用到这个变量的最新值，则称之为数据依赖。数据依赖关系包括两种情况：

       - 1. 写后读（Write After Read，RAW）：一个线程写了某个变量之后，之后的操作读了这个变量，则这个变量为 RAW 数据依赖。
       - 2. 读后写（Read After Write，WAR）：一个线程读了某个变量之后，之后的操作写了这个变量，则这个变量为 WAR 数据依赖。

      JMM 以数据依赖关系为边界，将指令按数据依赖关系分类：
      - 1. 指令之间不存在数据依赖关系，可乱序执行；
      - 2. 当前线程内，volatile 变量的写操作先于后续读操作，在当前线程内可见，故无需关心；
      - 3. 某个线程对某个 volatile 变量的写操作先于其他线程对同一 volatile 变量的读操作，这种情况下，为了正确执行有序性，读操作也需要串行化；
      - 4. 不同线程之间，如果存在数据依赖，需要根据具体的依赖关系来确定指令执行顺序。

    基于以上分类，JMM 提供了 4 种内存屏障指令来禁止指令重排序：

    1. LoadLoad 屏障：用于处理 LOAD – LOAD 依赖性，在当前线程内，前序volatile变量的load操作先于后序volatile变量的load操作，在当前线程内可见，故无需关心。

    2. StoreStore 屏障：用于处理 STORE – STORE 依赖性，在当前线程内，前序volatile变量的store操作先于后序volatile变量的store操作，故无需关心。

    3. LoadStore 屏障：用于处理 LOAD – STORE 依赖性，在当前线程内，前序volatile变量的load操作先于后序volatile变量的store操作，故需要串行化。

    4. StoreLoad 屏障：用于处理 STORE – LOAD 依赖性，在当前线程内，前序volatile变量的store操作先于后序volatile变量的load操作，故需要串行化。

    ### 3.3.2.volatile 的有序性
    指令重排序和 volatile 变量的可见性有助于确保 volatile 变量的有序性。

    volatile 变量的赋值操作不必视为内存屏障，但 volatile 变量的读操作，必须要插入内存屏障。

    对 volatile 变量的写操作后，volatile 变量自身的值会在其他线程中立即可见，但 volatile 变量之前的内存操作可能会被重排序。

    如果写操作和后续读操作（或写操作）之间不存在数据依赖关系，则读操作不会受到影响。如果写操作和后续读操作（或写操作）之间存在数据依赖关系，读操作可能会延迟到写操作之后，具体取决于编译器和处理器的实现。

    ### 3.3.3.synchronized 的有序性
    synchronized 同步块的内存语义是将其前后的指令都串行化，从而使得整个同步块内的代码按先后顺序执行。如果 synchronized 语句块中既有 volatile 变量的读写操作，也有普通变量的读写操作，则需要根据具体的读写操作来确定指令的执行顺序。

    # 4.Java 内存模型详解
    4.1.概述
    在了解了 Java 内存模型的基本概念之后，下面详细地介绍一下 Java 内存模型。

    4.2.重排序
    指令重排序是指编译器和处理器为了提高性能，将指令按照它们在机器代码中的顺序执行的能力。指令重排序可以分为以下几类：

    1. 编译器优化导致的重排序：编译器在生成代码时，根据某些条件，可能会对指令进行重排序。
    2. 指令并行导致的重排序：处理器在运行时，可能会对指令进行并行执行。
    3. 内存系统导致的重排序：由于指令的执行顺序受到内存系统的影响，处理器可能会对指令进行重排序。

    在编译器优化过程中，指令重排序也会造成不准确性，因为优化后的指令与原指令的相对顺序不一定正确。

    4.3.Happens-Before 规则
    Happens-Before 规则是 JMM 的基本原则，它指定了两个操作之间的顺序。如果一个操作的结果要对另一个操作可见，那么这两个操作之间必须满足 Happens-Before 规则。比如，在单个线程中，程序顺序执行的结果必须要对后续操作可见，则程序的执行顺序满足 Happens-Before 规则。

    Happens-Before 规则定义了如下四种情况：

    1. 程序顺序规则：一个线程中的每个操作，happens-before 于该线程中的任意后续操作。程序顺序规则表示单线程环境下的顺序性。
    2. 监视器锁规则：对一个锁的解锁，happens-before于随后对这个锁的加锁。
    3. volatile变量规则：对一个volatile变量的写操作，happens-before于任意后续对这个变量的读操作。
    4. 传递性规则：如果操作A happens-before操作B，且操作B happens-before操作C，那么操作A happens-before操作C。

    上述规则可以用图表示如下：

    ```
    Program Order Rule     : <==> {a} 
    Monitor Lock Rule      : <==|{}|{a}>    
    Volatile Variable Rule : <===|{}|={a}>  
    Transitive Rule        : <==|<||>|{}|>   
    ```

    程序顺序规则：<==> 表示两个操作可以在任意顺序执行。
    
    监视器锁规则：一个解锁操作happens-before于随后对这个锁的加锁。这里的<==|{}|{a}>是指对一个锁的解锁操作happens-before于任意后续对这个锁的加锁。{}代表操作不能并发执行。
    
    Volatile变量规则：对一个volatile变量的写操作happens-before于任意后续对这个变量的读操作。<===|{}|={a}>描述了volatile变量写操作happens-before于任意后续对这个变量的读操作。
    
    传递性规则：如果操作A happens-before操作B，且操作B happens-before操作C，那么操作A happens-before操作C。

    ```
    A happens-before C   // 操作A happens-before操作C。
    ------------------|    传递性规则。
    |-----------------|    操作B happens-before操作C。
                      \// 操作A happens-before操作B。
    ```

    在单个线程中，程序顺序规则是 JMM 的最强规则。其他三个规则都是根据程序顺序来推导出来的。


    4.4.原子性与内存屏障
    原子性是 JMM 的一个很重要的特性。原子性保证了多线程环境下共享变量的一致性，从而提供了一种方式来防止指令重排序，保证共享变量的可见性。

    原子操作是指一个操作或者多个操作要么全部执行并且执行结果与不执行都一样，要么全部不执行。在 CPU 执行指令时，对数据的读/写操作就是原子操作。

    JMM 内存模型通过在指令序列中插入内存屏障（Memory Barrier）来禁止指令重排序，从而保证原子性。

    JMM 提供了 8 个原子性内存操作来实现可见性与有序性：

    1. lock 指令：对主内存的变量进行写操作之前，需要加锁，这个动作是一个原子性操作。
    
    2. unlock 指令：对主内存的变量进行写操作之后，释放锁，这个动作是一个原子性操作。
    
    3. read 指令：从主内存中读取变量的值，这个动作是个 volatile 变量的读操作，不会出现数据不一致问题。
    
    4. load 指令：从局部变量区（工作内存）中读取变量的值，这个动作是个 volatile 变量的读操作，不会出现数据不一致问题。
    
    5. use 指令：对 volatile 变量的读操作。
    
    6. assign 指令：给工作内存中的变量赋值，这个动作是一个 volatile 变量的写操作，不会出现数据不一致问题。
    
    7. store 指令：将工作内存中的变量值刷新到主内存中，这个动作是个 volatile 变量的写操作，会将更新过的值发布到其他线程，从而确保可见性。
    
    8. write 指令：将局部变量的值刷新到主内存中，这个动作是个普通变量的写操作，不会出现数据不一致问题。

    内存屏障是一组处理器指令，用来禁止特定类型的处理器重排序。在 JMM 中，有三种类型的内存屏障：

    1. LoadLoad 屏障：它确保 loads 先于 loads 前的 volatile 读操作，即前序volatile变量的load操作先于后序volatile变量的load操作，在当前线程内可见，故无需关心。
    
    2. StoreStore 屏障：它确保 stores 先于 stores 后的 volatile 写操作，即前序volatile变量的store操作先于后序volatile变量的store操作，故无需关心。
    
    3. LoadStore 屏障：它确保 loads 先于 stores 后的 volatile 读写操作，即前序volatile变量的load操作先于后序volatile变量的store操作，故需要串行化。
    
    4. StoreLoad 屏障：它确保 stores 先于 loads 后的 volatile 读写操作，即前序volatile变量的store操作先于后序volatile变量的load操作，故需要串行化。

    这些内存屏障的具体作用是通过插入相应的指令，使得处理器不会对内存中同一变量的读/写操作进行重排序，保证了原子性与有序性。

    4.5.volatile 的内存语义
    本节介绍 volatile 关键字在内存模型中的语义。

    1. volatile 变量的写-读内存语义

    ```java
    volatile int a = 0;
    public void writer(){
        a = 1;
    }
    public void reader(){
        if(a == 1){}
    }
    ```

    假设一个线程调用writer()方法，这时会触发volatile变量的写操作，这时这个写操作之前的读操作以及之后的写操作都不能插入内存屏障，保证了可见性。

    假设另一个线程调用reader()方法，这时会触发volatile变量的读操作，会读取到volatile变量的最新值，之前的读操作以及之后的写操作都不能插入内存屏ancellationToken，保证了一致性。

    2. volatile 变量的写-写内存语义

    ```java
    volatile int b = 0;
    public void writer(){
        b = 1;
    }
    public void otherWriter(){
        b = 2;
    }
    ```

    假设一个线程调用writer()方法，这时会触发volatile变量的写操作，这时这个写操作之前的读/写操作以及之后的读/写操作都不能插入内存屏障，保证了可见性。

    假设另一个线程调用otherWriter()方法，这时会触发volatile变量的写操作，这时这个写操作之前的读/写操作以及之后的读/写操作都不能插入内存屏障，保证了可见性。

    但是需要注意的是，volatile 变量的写-写操作还是不能保证原子性的，因为可能存在数据竞争。

    在多线程环境下，volatile 变量的读写操作应该遵循 happens-before 规则，所以在 Java SE 5.0 版本中增加了 volatile 的内存语义。

