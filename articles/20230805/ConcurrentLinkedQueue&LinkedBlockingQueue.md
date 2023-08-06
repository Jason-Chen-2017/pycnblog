
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1996年Java 5引入了并发包java.util.concurrent。随后在Java 6中又加入了两个新的队列类——ConcurrentLinkedQueue和LinkedBlockingQueue。它们都实现了阻塞队列（BlockingQueue），即它规定生产者线程和消费者线程必须按照先入先出的顺序访问队列中的元素。这里将分别介绍ConcurrentLinkedQueue和LinkedBlockingQueue的工作原理、基本概念以及区别。本文只讨论ConcurrentLinkedQueue，因为LinkedBlockingQueue的工作原理类似于ConcurrentLinkedQueue。
         
        # 2.ConcurrentLinkedQueue
        ## 2.1 背景介绍
         Java集合框架中提供了一些线程安全的队列，如BlockingQueue、ArrayBlockingQueue等。但这些队列往往需要开发人员手动去控制同步机制，而且在高并发情况下容易出现数据不一致的问题。因此，Java 5.0之后，Java提供了两个新的线程安全队列——ConcurrentLinkedQueue和LinkedBlockingQueue。下面从工作原理和数据结构两方面进行分析。
        ### 2.1.1 如何保证线程安全？
        ConcurrentLinkedQueue通过使用CAS(Compare And Swap)操作保证对内部节点的原子性修改。其内部使用一个单向链表实现。首先将新节点添加到链表尾部，然后通过CAS操作将新节点指向头结点，这样就完成了节点插入。同样，当需要移除队首节点时，也是通过CAS操作将头结点指向头结点的下一个节点，完成节点的删除。
        在多线程环境下，ConcurrentLinkedQueue通过锁机制进行同步。队列在声明和初始化时会自动获取一个锁，所有对队列的访问均需要获取锁后才能进行，保证了队列的线程安全。下面是一个ConcurrentLinkedQueue的简单示意图：
        
        
        通过上面的图示可以看出，ConcurrentLinkedQueue基于链表的数据结构，每个节点都存储着数据及指向下一个节点的引用，通过CAS操作，实现对节点的原子操作，进而保证了线程安全。
        ### 2.1.2 数据结构
         ConcurrentLinkedQueue基于单向链表实现，其中每个节点包括data域和next域。其中data域存放数据值，next域指向下一个节点的引用。具体的数据结构如下图所示：
         
        
         每个节点可以看作是一个结构体，包含两个成员变量：data保存实际的值，next指针指向下一个节点；还有一个静态内部类Node，用于保存数据的节点。由于采用的是单向链表，所以每个节点只能指向它的前驱节点。因此，tail指向最后一个元素的前驱，head指向第一个元素。
         
         当添加元素的时候，在尾部创建一个新的节点，然后设置它的next域指向head（使得新节点成为head的前驱），然后尝试更新head，成功的话就返回true，否则继续竞争失败直到成功为止。但是如果在多线程的场景下，可能多个线程竞争一个位置，导致多个线程把相同的节点同时加到链表尾部，造成链表存在环形结构，导致逻辑上的错误。所以ConcurrentLinkedQueue在实际应用中要配合其他的同步措施比如ReadWriteLock等一起使用才比较安全。
 
        # 3.LinkedBlockingQueue
        ## 3.1 背景介绍
         LinkedBlockingQueue是另一种线程安全且功能丰富的BlockingQueue。它继承于AbstractQueue，实现了BlockingQueue接口。与ConcurrentLinkedQueue不同，LinkedBlockingQueue允许创建大小受限制的队列，支持可选的容量(capacity)参数，即队列大小不能超过指定的大小，超出的元素将被拒绝添加。另外，LinkedBlockingQueue提供了一个带有超时时间的offer方法，能够指定等待时间，防止调用线程一直阻塞。另外，LinkedBlockingQueue提供了一个put方法，可以无限期地阻塞调用线程。
         LinkedBlockingQueue底层由链接表（LinkedList）实现，链表中元素都是按先进先出的顺序排序，LinkedBlockingQueue既是BlockingQueue，又是Deque。具体来说，LinkedBlockingQueue允许从头部或尾部进出元素，这意味着可以在队列两端进行扩展。
        ### 3.1.1 如何保证线程安全？
         LinkedBlockingQueue通过锁机制进行同步。队列在声明和初始化时会自动获取一个锁，所有对队列的访问均需要获取锁后才能进行，保证了队列的线程安全。对于节点的插入和删除，LinkedBlockingQueue也使用了原子性的CAS操作。具体来说，插入操作使用putLock锁进行排他锁，插入元素时获取锁，然后进行元素的添加，在将当前节点的next指针设置为之前节点的next指针后，尝试更新head或tail指针，若成功则返回，否则重试。删除操作使用takeLock锁进行排他锁，删除元素时获取锁，然后遍历链表查找元素，找到匹配的元素后，将当前节点的next指针设置为之前节点的next指针后，尝试更新head指针或tail指针，若成功则返回，否则重试。
        ### 3.1.2 数据结构
         LinkedBlockingQueue底层的数据结构就是由链表构成的。每个节点包括数据域data和指向下一个节点的引用next。

         LinkedBlockingQueue主要包含四个重要的属性：head（队首）、tail（队尾）、count（元素数量）、items（实际数组）。

         - head：队列头指针，指向队首元素
         - tail：队列尾指针，指向队尾元素的下一个空位
         - count：当前队列中的元素数量
         - items：存储数据的数组

         下面给出LinkedBlockingQueue的完整数据结构。

                                  head
                                      |  
                                      V
                                     +---+-----+
                                 --->|item1|<----
                                   tail     next
                                    /|\           \
                                null |            \
                                      +-------------+    

        如上图所示，LinkedBlockingQueue的结构分为两部分：head到tail之间的一段是存储数据元素的数组，也就是items。每当插入一个元素或者删除一个元素时，都会修改head和tail指针，确保队列始终处于一个有效范围之内。count记录了实际存储的数据的个数。
        ### 3.1.3 put() 和 take() 方法
         LinkedBlockingQueue 提供了一个put()方法用来往队列中存入元素，默认情况下该方法不会阻塞，如果队列已满，则会抛出异常。同样，还有对应的 take()方法用来从队列中取出元素，此方法同样不会阻塞，如果队列为空，则会抛出异常。但是 LinkedBlockingQueue 提供了一些变体方法，比如：
         ```
             public void put(E e); // 抛出InterruptedException，put操作超时时不会抛出TimeoutException
             public boolean offer(E e, long timeout, TimeUnit unit) throws InterruptedException; // 指定超时时间，超时则返回false
             public E take(); // 抛出InterruptedException，获取元素超时时不会抛出TimeoutException
             public E poll(long timeout, TimeUnit unit) throws InterruptedException; // 获取元素超时时间，超时则返回null
         ``` 
         可以看到，除了直接抛出InterruptedException外，put和poll两种方法提供超时参数，超时时间以TimeUnit类型表示，超时则返回对应类型的默认值，而不是抛出InterruptedException。offer方法相比于put方法更强大，当指定超时时间timeout时，如果队列已满则立即返回false，避免阻塞线程。
         上述方法提供了非阻塞的方法，当资源可用时立即返回，否则在超时时间内阻塞等待。为了获取阻塞方法，比如put()、take()，可以使用以下方式：
         ```
             public synchronized void put(E e) {
                 while (count == items.length) {
                     try {
                         wait();
                     } catch (InterruptedException ex) {}
                 }
                 if (tail == items.length) {
                     tail = 0;
                 }
                 items[tail] = e;
                 ++tail;
                 ++count;
                 notifyAll();
             }
 
             public synchronized E take() {
                 while (count == 0) {
                     try {
                         wait();
                     } catch (InterruptedException ex) {}
                 }
                 Object x = items[head];
                 if (++head == items.length) {
                     head = 0;
                 }
                 --count;
                 notifyAll();
                 return (E)x;
             }
         ```   
         将方法定义为同步，使用wait()和notifyAll()方法实现线程间通信。当队列资源不可用时，阻塞在这个方法上。
         put()方法首先检测当前队列是否已经满，如果是则进入循环，直到队列可用。然后插入新元素到队列末尾，通知所有阻塞的线程。与take()方法一样，head和tail指针通过判断数组边界来调整。
         take()方法类似，但是从队首取得元素，并且清除旧元素。
         ## 4.总结
         本文从工作原理和数据结构两个角度介绍了ConcurrentLinkedQueue和LinkedBlockingQueue，ConcurrentLinkedQueue基于CAS操作实现了线程安全的队列，通过锁机制保证线程安全；LinkedBlockingQueue是在ConcurrentLinkedQueue基础上的改良版本，具有额外的空间限制能力，提供了可选容量参数，并且提供put()和take()方法提供阻塞功能。
         从源码的角度分析了ConcurrentLinkedQueue和LinkedBlockingQueue的基本原理，以及各自的数据结构。希望通过这两篇文章，能让读者对ConcurrentLinkedQueue和LinkedBlockingQueue有一个更深入的理解。