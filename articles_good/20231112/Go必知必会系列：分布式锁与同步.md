                 

# 1.背景介绍


## 概念
对于并发编程来说，锁是非常重要的机制之一。在某些场景下，多个进程或线程需要访问共享资源时，需要对共享资源进行加锁（Lock），以保证数据的完整性、正确性和一致性。由于锁会阻塞线程或进程的执行，因此如果没有特别注意并发安全的问题，就会造成死锁、资源浪费等问题。

在分布式环境中，锁也不例外。在这种情况下，为了确保数据在不同节点上的正确访问和更新，就需要使用分布式锁（Distributed Lock）。分布式锁是一种基于主从复制模式的分布式协调服务实现的可重入锁，它能确保在不同的节点上只允许一个客户端持有锁，并且在客户端宕机或者调用 unlock 操作后释放锁。

当然还有一些其他类型的锁，如互斥锁（Mutex）、读写锁（ReadWriteLock）、条件变量（Condition Variable）等，但本文主要讨论分布式锁。

## 为什么要用分布式锁
为了提升分布式环境下的可用性和性能，很多公司都开始逐渐将应用部署到多台机器上。而随之而来的一个重要的挑战就是如何保证这些应用之间的数据一致性。这其中最简单的方法就是使用分布式锁。

假设我们有两个服务 A 和 B，它们都需要操作同一个数据库表。那么如果服务 A 在对该表进行操作过程中，服务 B 此时想对这个表进行操作，就会出现数据不一致的问题。因为两个服务分别运行在不同的服务器上，所以他们访问的是不同的内存地址，对于同一个数据来说，只能有一个服务在修改它。

为了解决这个问题，可以给数据库增加分布式锁，使得只有一个服务可以获取到锁，然后再操作数据库。另外，还可以通过一些手段比如缓存同步来避免数据库之间的同步等待。但是引入分布式锁还带来了一系列复杂的问题，比如锁冲突时的处理，请求超时后的处理，锁的失效时间的设置等。

## 什么是可重入锁？
所谓可重入锁（Reentrant Lock）就是在获得锁的情况下，能够再次获得锁，也就是说当前线程已经持有了某个锁，当再次申请的时候仍然可以使用相同的锁。这就意味着无论这个线程有多少次进入临界区，都不需要额外花费时间去阻塞线程，避免了死锁。

## 什么是偏向锁？
所谓偏向锁（Biased Lock）就是在没有竞争发生的前提下，减少锁对象的切换次数。它的基本思路是在每次进入临界区的时候，持有锁的线程都会认为自己是第一个进入的，进而避免发生锁冲突。这样的话，无需进行全锁，只需先尝试获取偏向锁，获取失败再尝试完全自旋锁。而在抢占式低延迟环境中，这是一种高效的锁策略。

# 2.核心概念与联系
## 分布式锁
分布式锁是一个用于在分布式环境下控制对共享资源的访问的工具。它通过保持独占的方式来防止多个进程或线程同时访问共享资源，从而达到不同进程或线程之间对共享资源的并发访问的统一管理。分布式锁包括两类，一是基于网络的分布式锁；二是基于文件的分布式锁。

基于网络的分布式锁主要依赖于Zookeeper或Etcd等分布式协调服务实现的基于主从架构。其流程如下图所示：

1. 客户端请求获取锁
2. 服务端验证客户端是否有效，如果是有效的则返回true，否则返回false
3. 如果返回值为true，则进入等待状态，直到被唤醒或者超时
4. 服务端释放锁

基于文件的分布式锁则主要通过文件锁来实现。其流程如下图所示：

1. 客户端打开文件
2. 检查是否存在锁标识
3. 如果不存在，则创建锁标识并写入到文件中
4. 如果存在，则读取锁标识中的PID值，判断是否为当前进程ID，若是则删除锁标识并返回成功，否则判断是否超过超时时间，若超过则删除锁标识并返回失败，否则进入等待状态

## 可重入锁
可重入锁又称递归锁或嵌套锁，指的是在一个线程在持有锁的期间，可以在同一个线程中再次申请此锁。也就是说，如果线程获得了某个锁，那么再次申请时可以直接忽略。这样就保证了同一线程可以在不同的方法或函数内部调用该锁，而不会导致死锁或其它问题。

## 偏向锁
偏向锁是JDK 1.6之后才引入的一个锁优化方式。当一个线程在锁的竞争中，处于一直竞争状态，因而一直无法获取到锁时，这个线程会把锁的偏向设置为线程id，以便于之后该线程再次请求锁时，可以提前获得锁，减少上下文切换的消耗。同时，当线程不再持有锁时，释放锁的操作还是正常的，只是不再需要做偏向相关的操作。

## Java中的锁
在Java语言中，提供了四种锁类型：

1. Synchronized关键字
2. ReentrantLock类
3. ReadWriteLock接口
4. StampedLock接口

### Synchronized关键字
Synchronized关键字是用于方法、代码块或构造函数的同步的一种锁。它能够提供线程间的互斥访问，即一次只有一个线程可以持有某个对象锁。使用 synchronized 关键字声明的代码块可以隐式地按照顺序进行同步。

```java
public class SynchronizedDemo {

    private static int count = 0;
    
    public void increase() {
        for (int i = 0; i < 1000000; i++) {
            count++;
        }
    }
    
    public static void main(String[] args) throws InterruptedException {
        final SynchronizedDemo demo = new SynchronizedDemo();
        
        Thread t1 = new Thread(() -> {
            try {
                demo.increase();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
        
        Thread t2 = new Thread(() -> {
            try {
                demo.increase();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
        
        t1.start();
        t2.start();
        
        t1.join();
        t2.join();
        
        System.out.println("Count: " + count); // Count: 2000000
    }
}
```

从代码中可以看到，这是一个简单的计数器类的示例，我们用两种线程同时调用 increase 方法，实际上是想要统计一百万次的累加结果。由于 synchronized 是一种排他锁，所以在这个例子里，如果两个线程同时执行 increase 方法，则它们是串行化执行的，最终的结果是一样的。换句话说，synchronized 只能保证一个线程一个时间地进入临界区，不能保证多个线程同时进入临界区。

### ReentrantLock类
ReentrantLock类是较新的同步类，它能够提供更灵活的锁定和同步控制功能。它可以指定等待时间、公平锁等参数。除此之外，它还支持各种锁策略，如公平锁、读写锁、可重入锁等。

ReentrantLock比synchronized具有更多的特性，如可重入性、锁降级、锁升级、定时锁等。其中，可重入性是指一个线程在持有锁的情况下，可以再次获得该锁，而不需要重新申请锁。锁降级是指从一个重入锁降级为非重入锁。锁升级是指从一个非重入锁升级为重入锁。定时锁是指在一定的时间内自动释放锁，如果业务逻辑需要持续执行，则不会因为锁一直被占用的情况导致死锁。

下面看一个简单的示例：

```java
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class ReentrantLockDemo {

    private static int count = 0;
    
    private Lock lock = new ReentrantLock();
    
    public void increase() {
        lock.lock();
        try {
            for (int i = 0; i < 1000000; i++) {
                count++;
            }
        } finally {
            lock.unlock();
        }
    }
    
    public static void main(String[] args) throws InterruptedException {
        final ReentrantLockDemo demo = new ReentrantLockDemo();
        
        Thread t1 = new Thread(() -> {
            try {
                demo.increase();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        
        Thread t2 = new Thread(() -> {
            try {
                demo.increase();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        
        t1.start();
        t2.start();
        
        t1.join();
        t2.join();
        
        System.out.println("Count: " + count); // Count: 2000000
    }
}
```

这是一个简单的计数器类的示例，用两个线程同时调用 increase 方法。但是不同的是，这里采用了 ReentrantLock 对象作为同步锁，它提供了更多的锁控制功能。我们创建一个 ReentrantLock 对象，并在 try-finally 语句块中释放锁。这样，即使在异常的情况下也能保证锁一定被释放。

### ReadWriteLock接口
ReadWriteLock接口是Java 5引入的新接口，它用来管理对共享资源的读和写访问。它包括一个分离的读锁和一个独占的写锁，允许多个线程同时读取资源，但只允许单个线程进行写入。ReadWriteLock接口由两个接口组成：ReadLock和WriteLock。ReadLock用于读取共享资源，而WriteLock用于写入共享资源。

ReadWriteLock比Synchronized提供了更细粒度的同步控制，能够最大程度地提升系统的吞吐量。

```java
import java.util.Random;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class RWLockDemo {

    private String name;
    private int age;
    
    private ReadWriteLock rwLock = new ReentrantReadWriteLock();
    private Lock readLock = rwLock.readLock();
    private Lock writeLock = rwLock.writeLock();
    
    public RWLockDemo(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    public void setName(String name) {
        writeLock.lock();
        try {
            this.name = name;
        } finally {
            writeLock.unlock();
        }
    }
    
    public String getName() {
        readLock.lock();
        try {
            return name;
        } finally {
            readLock.unlock();
        }
    }
    
    public void setAge(int age) {
        writeLock.lock();
        try {
            this.age = age;
        } finally {
            writeLock.unlock();
        }
    }
    
    public int getAge() {
        readLock.lock();
        try {
            return age;
        } finally {
            readLock.unlock();
        }
    }
    
    public static void main(String[] args) {
        Random random = new Random();
        
        RWLockDemo demo = new RWLockDemo("", 0);
        
        Runnable writerThread = () -> {
            while (true) {
                try {
                    Thread.sleep(random.nextInt(100));
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                
                if (demo.getName().isEmpty()) {
                    demo.setAge(random.nextInt());
                    demo.setName(Thread.currentThread().getName());
                    System.out.printf("%s wrote %d and changed to %s.\n",
                            Thread.currentThread().getName(), demo.getAge(), demo.getName());
                } else {
                    System.out.printf("%s is writing or reading...\n", Thread.currentThread().getName());
                }
            }
        };
        
        for (int i = 0; i < 10; i++) {
            Thread thread = new Thread(writerThread);
            thread.start();
        }
    }
}
```

上面是一个简单的读写锁示例，其中包括两个线程，一个读线程和一个写线程。读线程和写线程都要获取读写锁才能访问共享资源，但读锁和写锁之间是互斥的，即一次只能有一个线程持有写锁，而多个线程持有读锁。同时，每个线程只能持有自己的锁，而不能共享锁。

### StampedLock接口
StampedLock是一个接口，它提供了一种乐观读写锁，它支持批量地从结构化存储读取数据。使用StampedLock，就可以在线程间共享资源时，实现一种高效且低延迟的乐观并发控制。

StampedLock接口中包含两个方法：tryOptimisticRead()和validate()。

tryOptimisticRead()方法试图获取乐观读锁，而不真正锁定任何资源。如果真的需要获取资源，则再调用 validate()方法来确认乐观读锁是否有效。

validate()方法验证获取到的乐观读锁是否有效。如果获取到的锁是一个过期锁，则会重试，直到获取到有效的锁。如果获取到的锁是一个有效的锁，则会释放所有之前获得的锁，以确保资源不会被其他线程打断。

```java
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.StampedLock;

public class StampedLockDemo {

    private long value;
    
    private StampedLock stampedLock = new StampedLock();
    
    public long getValue() {
        long stamp = stampedLock.tryOptimisticRead();
        long v = value;
        if (!stampedLock.validate(stamp)) {
            Lock wLock = stampedLock.writeLock();
            long ws = wLock.tryLock();
            try {
                v = value;
            } finally {
                wLock.unlock();
            }
        }
        return v;
    }
    
    public boolean updateValue(long delta) {
        long stamp = stampedLock.writeLock();
        try {
            value += delta;
            return true;
        } finally {
            stampedLock.unlock(stamp);
        }
    }
    
    public static void main(String[] args) {
        StampedLockDemo demo = new StampedLockDemo();
        
        Runnable readerThread = () -> {
            while (true) {
                System.out.println(Thread.currentThread().getName() + ": " + demo.getValue());
                try {
                    TimeUnit.SECONDS.sleep(1);
                } catch (InterruptedException e) {
                    break;
                }
            }
        };
        
        Runnable updaterThread = () -> {
            while (true) {
                demo.updateValue((long) Math.random() * 10);
                System.out.println(Thread.currentThread().getName() + " updated.");
                try {
                    TimeUnit.SECONDS.sleep(1);
                } catch (InterruptedException e) {
                    break;
                }
            }
        };
        
        new Thread(readerThread).start();
        new Thread(updaterThread).start();
    }
}
```

上面是一个简单的乐观读写锁示例，其中包括一个读线程和一个写线程。读线程会获取乐观读锁，然后尝试读取数据，如果数据被修改，则会升级为悲观读锁，读取最新的数据。写线程会获取写锁，然后尝试更新数据。

# 3.核心算法原理及操作步骤
## Zookeeper分布式锁原理
Zookeeper分布式锁原理很简单，可以参考Zookeeper官网[http://zookeeper.apache.org/]关于分布式锁的相关文档。

## 可重入锁原理
可重入锁又称递归锁或嵌套锁，指的是在一个线程在持有锁的期间，可以在同一个线程中再次申请此锁。也就是说，如果线程获得了某个锁，那么再次申请时可以直接忽略。这样就保证了同一线程可以在不同的方法或函数内部调用该锁，而不会导致死锁或其它问题。

具体的算法原理如下：

1. 判断当前线程是否已经持有锁
2. 如果已经持有锁，则计数器加1
3. 如果未持有锁，则尝试获取锁
4. 如果获取到锁，则计数器置1
5. 获取锁成功，返回true
6. 如果获取锁失败，则释放之前已经申请的锁，重新开始获取锁
7. 当计数器大于1，则认为已经获取到了锁，返回true
8. 当计数器等于1，则证明该线程已经退出了递归，可以释放锁，返回false

## 偏向锁原理
偏向锁是JDK 1.6之后才引入的一个锁优化方式。当一个线程在锁的竞争中，处于一直竞争状态，因而一直无法获取到锁时，这个线程会把锁的偏向设置为线程id，以便于之后该线程再次请求锁时，可以提前获得锁，减少上下文切换的消耗。同时，当线程不再持有锁时，释放锁的操作还是正常的，只是不再需要做偏向相关的操作。

具体的算法原理如下：

1. 当前线程获取锁
2. 如果锁不可偏向线程，则将线程id记录为锁的偏向线程id
3. 如果锁可偏向线程，则判断当前线程是否是偏向线程
4. 如果当前线程是偏向线程，则进入代码块，将计数器置为1
5. 如果当前线程不是偏向线程，则在循环中获取锁
6. 每次获取锁时，将锁的偏向线程id置空，从而偏向锁失效
7. 将锁设置为非偏向状态
8. 执行代码块
9. 释放锁

## Java中的锁实现原理

### synchronized关键字的原理
Synchronized关键字是用于方法、代码块或构造函数的同步的一种锁。它能够提供线程间的互斥访问，即一次只有一个线程可以持有某个对象锁。使用 synchronized 关键字声明的代码块可以隐式地按照顺序进行同步。

具体的算法原理如下：

1. 创建一个锁对象
2. 获得锁对象，进行同步，释放锁对象

### ReentrantLock类
ReentrantLock类是较新的同步类，它能够提供更灵活的锁定和同步控制功能。它可以指定等待时间、公平锁等参数。除此之外，它还支持各种锁策略，如公平锁、读写锁、可重入锁等。

具体的算法原理如下：

1. 创建一个ReentrantLock对象
2. 请求获取锁
3. 如果当前线程已经拥有该锁，则加锁计数器自增1
4. 如果当前线程没有持有该锁，则首先尝试获取该锁
5. 如果获取到锁，则锁的持有者为当前线程，加锁计数器为1
6. 如果获取锁失败，则重复第2步，直到获取到锁或者超时
7. 请求释放锁
8. 如果锁的持有者为当前线程，则检查锁计数器，如果计数器为1，则释放锁，否则将锁的持有者置为空，减小锁计数器，如果计数器等于1，则释放锁

### ReadWriteLock接口
ReadWriteLock接口是Java 5引入的新接口，它用来管理对共享资源的读和写访问。它包括一个分离的读锁和一个独占的写锁，允许多个线程同时读取资源，但只允许单个线程进行写入。ReadWriteLock接口由两个接口组成：ReadLock和WriteLock。ReadLock用于读取共享资源，而WriteLock用于写入共享资源。

具体的算法原理如下：

1. 创建一个ReadWriteLock对象
2. 获取写锁
3. 获取读锁
4. 释放读锁
5. 释放写锁

### StampedLock接口
StampedLock接口是Java 8引入的新接口，它提供了一种乐观读写锁，它支持批量地从结构化存储读取数据。使用StampedLock，就可以在线程间共享资源时，实现一种高效且低延迟的乐观并发控制。

具体的算法原理如下：

1. 创建一个StampedLock对象
2. 获取一个写锁或读锁
3. 使用乐观锁更新变量的值
4. 提交事务，释放写锁或读锁

# 4.具体代码实例和详细解释说明
## Java版本
笔者写文章时，使用的Java版本是OpenJDK 11.0.11+1。

## Zookeeper分布式锁实践
这里以Apache Curator Framework库的InterProcessSemaphoreV2作为示例来展示如何使用Zookeeper来实现分布式锁。

### 安装Zookeeper服务
Zookeeper服务安装比较简单，可以直接下载安装包进行安装即可。

### Maven工程构建配置
```xml
<dependency>
  <groupId>org.apache.curator</groupId>
  <artifactId>curator-recipes</artifactId>
  <version>2.13.0</version>
</dependency>
```

### 编写代码
```java
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.framework.recipes.locks.InterProcessSemaphoreV2;

public class DistributedLockExample {

    private static final String CONNECTION_STRING = "localhost:2181";
    private static final String LOCK_PATH = "/my-locks";
    private static final String CLIENT_NAME = "client-1";

    public static void main(String[] args) throws Exception {

        CuratorFramework client = CuratorFrameworkFactory.newClient(CONNECTION_STRING, CLIENT_NAME);
        client.start();

        InterProcessSemaphoreV2 semaphore = new InterProcessSemaphoreV2(client, LOCK_PATH, 1);

        // acquire the lock
        semaphore.acquire();

        try {

            // critical section of your code that requires locking here...

            System.out.println("Doing something...");

        } finally {

            // release the lock
            semaphore.release();
        }

        client.close();
    }
}
```

### 测试效果
启动Zookeeper服务，并启动工程，观察控制台输出日志信息。正常情况下，当有多个客户端同时运行时，只有一个客户端能成功获取到锁，其它客户端只能阻塞等待，以实现锁的互斥访问。

## 可重入锁实践
### 源码实现
```java
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class MyLock {

    private Map<Thread, AtomicInteger> locks = new HashMap<>();

    public void lock() {
        Thread currentThread = Thread.currentThread();
        AtomicInteger atomicInteger = locks.computeIfAbsent(currentThread, k -> new AtomicInteger(1));
        while (atomicInteger.get()!= 0) {
            // spin wait on the Atomic integer until it reaches zero
        }
    }

    public void unlock() {
        Thread currentThread = Thread.currentThread();
        AtomicInteger atomicInteger = locks.get(currentThread);
        if (atomicInteger == null || atomicInteger.decrementAndGet() <= 0) {
            locks.remove(currentThread);
        }
    }
}
```

### 测试效果
```java
public class Main {

    public static void main(String[] args) {
        MyLock myLock = new MyLock();

        Thread thread1 = new Thread(() -> {
            myLock.lock();
            System.out.println("Thread-1 acquired the lock");
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                myLock.unlock();
                System.out.println("Thread-1 released the lock");
            }
        }, "thread1");

        Thread thread2 = new Thread(() -> {
            myLock.lock();
            System.out.println("Thread-2 acquired the lock");
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                myLock.unlock();
                System.out.println("Thread-2 released the lock");
            }
        }, "thread2");

        thread1.start();
        thread2.start();
    }
}
```

### 输出结果
```
Thread-1 acquired the lock
Thread-2 acquired the lock
Thread-1 released the lock
Thread-2 released the lock
```

## 偏向锁实践
### 源码实现
```java
import sun.misc.Unsafe;

public class MyLock {

    private volatile int state = 0;           // 锁定状态
    private volatile int ownerId = -1;       // 锁持有线程ID

    Unsafe unsafe;                            // Unsafe类
    int addressOffset;                        // 成员变量address的偏移量

    public MyLock() {
        try {
            unsafe = Unsafe.getUnsafe();        // 获取Unsafe实例
            Class<?> clazz = getClass();
            addressOffset = unsafe.arrayBaseOffset(clazz.getDeclaredField("state"));   // 获取成员变量address的偏移量
        } catch (NoSuchFieldException e) {
            throw new Error(e);
        }
    }

    public void lock() {
        int currentOwnerId = getCurrentOwnerId();    // 获取当前持有线程ID
        int expectedState = getState();              // 获取当前锁定状态

        if (currentOwnerId == getThreadId()) {      // 如果当前线程已持有锁，则加锁计数器自增1
            setState(expectedState + 1);             // 更新锁定状态
        } else {                                      // 如果当前线程没有持有锁，则获取锁
            if (unsafe.compareAndSwapInt(this, addressOffset, expectedState, expectedState | LOCKED_BIT)) {
                ownerId = getThreadId();               // 设置线程ID
                setState(expectedState + 1);         // 更新锁定状态
            } else {                                  // 如果获取锁失败，则自旋等待
                doSpinWait();                         // 通过doSpinWait方法自旋等待
            }
        }
    }

    /**
     * 通过doSpinWait方法实现自旋等待
     */
    private void doSpinWait() {
        int counter = SPIN_COUNT;                   // 初始化自旋次数
        while ((getState() & LOCKED_BIT)!= 0 && counter-- > 0) {     // 如果锁定状态位为1，则进行自旋等待，直到获取到锁
            Thread.yield();                           // yield方法导致当前线程暂停
        }
    }

    public void unlock() {
        int currentOwnerId = getCurrentOwnerId();            // 获取当前持有线程ID
        int expectedState = getState();                      // 获取当前锁定状态

        if (currentOwnerId == getThreadId() && expectedState >= 1) {   // 如果当前线程已持有锁，且锁定状态不为0，则释放锁
            if (unsafe.compareAndSwapInt(this, addressOffset, expectedState, expectedState - 1)) {
                if (expectedState == 1) {                     // 如果锁定状态为1，则表示之前的线程也处于自旋状态，需要唤醒它
                    unparkCurrentOwner();                    // 唤醒线程
                }
            }
        }
    }

    /**
     * 获取当前持有线程ID
     */
    private int getCurrentOwnerId() {
        return unsafe.getIntVolatile(this, addressOffset);     // 获取当前持有线程ID
    }

    /**
     * 获取当前锁定状态
     */
    private int getState() {
        return unsafe.getIntVolatile(this, addressOffset);     // 获取当前锁定状态
    }

    /**
     * 设置当前锁定状态
     */
    private void setState(int newState) {
        unsafe.putOrderedInt(this, addressOffset, newState);   // 设置当前锁定状态
    }

    /**
     * 获取当前线程ID
     */
    private int getThreadId() {
        return Thread.currentThread().getId();                  // 获取当前线程ID
    }

    /**
     * 唤醒当前线程持有的锁
     */
    private void unparkCurrentOwner() {
        Thread currentOwner = Thread.of(ownerId);                 // 根据锁持有线程ID获取线程对象
        if (currentOwner!= null) {                               // 如果线程对象不为空，则唤醒它
            currentOwner.unpark();                                // unpark方法使线程从等待中恢复，变成可执行状态
        }
    }


    /*
     * 状态位定义
     */
    private static final int LOCKED_BIT = 1 << 31;          // 锁定状态位
    private static final int SPIN_COUNT = 1000;             // 自旋次数
}
```

### 测试效果
```java
public class Main {

    public static void main(String[] args) {
        MyLock myLock = new MyLock();

        Thread thread1 = new Thread(() -> {
            myLock.lock();
            System.out.println("Thread-1 acquired the lock");
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                myLock.unlock();
                System.out.println("Thread-1 released the lock");
            }
        }, "thread1");

        Thread thread2 = new Thread(() -> {
            myLock.lock();
            System.out.println("Thread-2 acquired the lock");
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                myLock.unlock();
                System.out.println("Thread-2 released the lock");
            }
        }, "thread2");

        thread1.start();
        thread2.start();
    }
}
```

### 输出结果
```
Thread-1 acquired the lock
Thread-2 acquired the lock
Thread-1 released the lock
Thread-2 released the lock
```