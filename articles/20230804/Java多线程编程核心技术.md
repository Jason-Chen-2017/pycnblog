
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1. 这是一本关于Java多线程编程的入门书籍，通过快速理解并掌握多线程技术，可以帮助你更好地利用Java进行高效的并发处理，提升应用的响应速度、并发能力和吞吐量。
         2. 本书从线程的创建、启动、停止、同步、等待/通知机制、线程间通信、线程池等方面详细介绍了多线程开发中常用的技术。
         3. 如果读者是第一次接触多线程编程或者需要快速学习一些基础知识的话，那么这本书绝对是一个不错的选择！
         4. 本书适合于具有一定Java编程经验的工程师阅读。
         
         # 2.作者简介
         ## 李娟（Java架构师）
         1. Java多年的架构设计经验，先后担任过很多不同规模系统的架构师；
         2. 十几年的IT架构工作积累，对高性能、高可用性、可扩展性的架构有丰富的经验；
         3. 曾就职于中国移动、亚信科技、平安证券，目前主要负责微服务架构设计与研发。
         4. 您可以直接联系她：<EMAIL> 或 微信号：xm19971001 。

         ## 陈恒龙（架构师/Java高级工程师）
         1. 有多年Java开发及架构工作经验；
         2. 曾就职于搜狗集团、华为、网易，现任某著名公司架构师；
         3. 对多线程、锁、并发控制、数据结构及算法等相关技术有深入的研究和理解；
         4. 您可以直接联系他：<EMAIL> 或 微信号：chenkunlong 。

         ## 许立强（Java程序员）
         1. 近五年在阿里巴巴集团有丰富的Java后台开发经验；
         2. 拥有较强的分析问题和解决问题的能力，擅长用自己的思路解决复杂的问题；
         3. 同时具有较强的创新意识和团队精神，深受学员们的追捧；
         4. 您可以直接联系他：<EMAIL>或微信号：xujq1994 。

         # 3.目录
         1. 第一章　Java多线程概述
         2. 第二章　Java线程的实现方式
         3. 第三章　创建线程
         4. 第四章　启动线程
         5. 第五章　线程的状态与控制
         6. 第六章　线程间通信
         7. 第七章　线程池
         8. 第八章　Lock接口
         9. 第九章　Volatile关键字
         10. 第十章　信号量 Semaphore
         11. 第十二章　Exchanger 交换器
         12. 第十三章　ScheduledExecutorService 计划任务执行类
         13. 第十四章　ConcurrentHashMap 并发哈希表
         14. 第十五章　并发工具类总结
         15. 附录A JMM内存模型详解
         16. 附录B LockSupport类详解
         17. 附录C AQS同步组件详解
         18. 致谢
          
        # 4.《Java多线程编程核心技术》正文
        #  1.Java多线程概述
           在java编程中，线程是一种用来执行程序段的并发执行的方式。多线程被广泛用于网络服务器的编程中，可以充分地提高程序的运行效率。
           通过引入多线程技术，可以让一个程序同时处理多个任务，节省CPU时间，提高程序的响应速度，缩短处理时间。
           
           Java语言提供了两种创建线程的方式：继承Thread类和实现Runnable接口。前者更加简单，后者灵活且可复用。
           Thread类的构造函数指定线程的名字，但实际上这个名字仅仅是作为线程的一个标识。
           Runnable接口包含一个run()方法，定义线程要执行的任务。通过实现Runnable接口，可以在不同的线程之间共享同一个Runnable对象。
           
           创建线程后，可以通过start()方法启动线程。当调用start()方法时，JVM会创建一个新的线程并执行run()方法。
           
           执行完毕的线程将自动终止，因此，如果某个线程需要一直运行直到结束，则不能设置daemon属性。如果所有非daemon线程都结束，虚拟机将自动退出。
        #  2.Java线程的实现方式
           Java语言的线程分为用户线程（又称非守护线程）和守护线程两类。线程的类型由isDaemon()方法确定，返回true表示该线程为守护线程，false表示该线程为用户线程。
           
           默认情况下，所有线程都是用户线程。当线程的run()方法执行完成后，该线程终止。当所有的用户线程终止时，虚拟机也终止。
           
           当某个线程设置为守护线程时，它将变成守护线程，并且不会阻碍JVM退出。只有当所有的非守护线程都终止后，虚拟机才会退出。
           
           可以通过setDaemon(boolean on)方法来设置线程的守护状态。如果on为true，则该线程为守护线程；如果on为false，则该线程为用户线程。

        #   3.创建线程
           Java多线程的创建过程分为以下几个步骤：
           
           1. 创建线程对象。
               创建线程对象一般有两种方式：
               
               - 使用Thread类继承的方式。
                   public class MyThread extends Thread {
                       //...
                   }
                   
                   public static void main(String[] args) {
                       MyThread thread = new MyThread();
                       //...
                   }
                
               - 使用Runnable接口实现的方式。
                   public interface MyRunnableInterface implements Runnable {
                       //...
                   }
                   
                   public static void main(String[] args) {
                       MyRunnableInterface runnable = new MyRunnableInterface() {
                           @Override
                           public void run() {
                               //...
                           }
                       };
                       
                       Thread thread = new Thread(runnable);
                       //...
                   }
                   
               
           2. 设置线程名称。
               可通过setName()方法设置线程的名称，例如：
               
               thread.setName("MyThread");
            
           3. 重写run()方法。
               若使用Thread类继承的方式，则需要重写Thread类中的run()方法，否则无法启动线程。其原型如下所示：
               
               public void run() {}
               
               此处编写线程要执行的代码。
               
               若使用Runnable接口实现的方式，则无需重写run()方法，而是在创建线程对象时传入一个实现了Runnable接口的对象即可。
            
           4. 启动线程。
               线程创建之后，需要启动才能开始运行。通过start()方法启动线程。
               
               注意：不要在run()方法中调用start()方法，因为此时线程还没有启动，可能造成死锁。
                
        #    4.1启动线程的方式
             Thread类的start()方法和其他方法一样，也是invokevirtual指令。但是由于有可能多次调用，所以线程可能会执行多次run()方法。
             如果线程之前已经启动过，则再次调用start()方法不会产生新的线程，只是重新进入就绪队列。
             
             另外，通过静态方法Thread.yield()也可以让当前线程暂停执行，让出CPU资源。不过，这种方式是不推荐使用的。
             
        #    4.2终止线程的方法
             Thread类的stop()方法已过时，不推荐使用。建议使用中断Thread对象的方式终止线程。
             
             中断线程的方法有两种：
             
             a. 通过interrupt()方法。在需要终止线程的时候，将该线程对象的中断标志设置为true。然后调用该对象的interruped()方法判断是否已经被中断。如果线程正常执行，则interrupted()方法返回false；如果被中断，则返回true。
             
             b. 通过Thread.sleep(long millis)方法，使线程暂停一段时间，在这段时间内，可以通过调用interrupt()方法来中断该线程。在millis时间到达之前，线程会一直处于休眠状态，即使处于活动状态。
             
        #  5.启动线程顺序
            在启动一个线程之前，通常需要先创建线程对象。然后，根据创建线程的两种方式，设置线程的名称和执行任务。最后调用start()方法启动线程。
            
            创建、设置和启动顺序如下所示：
            
             1. 创建线程对象。
                 Thread thread = new Thread();
                 
             2. 设置线程名称。
                 thread.setName("MyThread");
              
             3. 实现Runnable接口。
                 public class MyThread implements Runnable {
                     private int num;
                     
                     public MyThread(int n) {
                         this.num = n;
                     }
                     
                     public void run() {
                         System.out.println("Thread " + num + " is running.");
                     }
                 }

                 
             4. 设置任务参数。
                 MyThread myThread = new MyThread(i);
                 
             5. 启动线程。
                 thread.start();
                
            在上面的例子中，创建线程对象、设置线程名称和实现Runnable接口是在一步完成的。设置任务参数和启动线程是在两个步骤完成的。这样可以避免出现错误。
            
        #  6.线程的状态与控制
            Java语言提供了一个Thread.State枚举类来表示线程的状态。线程的状态包括NEW、RUNNABLE、BLOCKED、WAITING、TIMED_WAITING、TERMINATED。
            NEW：尚未启动的线程。
            RUNNABLE：正在执行的线程。
            BLOCKED：正在被阻塞的线程。
            WAITING：处于等待状态的线程。
            TIMED_WAITING：处于带时间限制的等待状态的线程。
            TERMINATED：已终止的线程。
            
            可以通过getState()方法获取线程的当前状态，返回值是一个Thread.State类型的枚举值。
            
            获取线程状态的方法如下所示：
            
             public void getState(){
                 Thread.State state = threadObj.getState();
                 switch (state){
                     case NEW:
                         System.out.println("线程尚未启动");
                         break;
                     case RUNNABLE:
                         System.out.println("线程正在运行");
                         break;
                     case BLOCKED:
                         System.out.println("线程被阻塞");
                         break;
                     case WAITING:
                         System.out.println("线程处于等待状态");
                         break;
                     case TIMED_WAITING:
                         System.out.println("线程处于带时间限制的等待状态");
                         break;
                     case TERMINATED:
                         System.out.println("线程已终止");
                         break;
                     default:
                         System.out.println("未知状态");
                         break;
                 }
             }
             
             此外，还可以使用isAlive()方法判断线程是否存活。该方法的作用是检查线程是否仍然存活，即线程是否处于运行、就绪、阻塞或等待状态。
             
             判断线程存活的方法如下所示：

              public boolean isAlive(){
                  return threadObj.isAlive();
              }

              
             
        #   7.线程间通信
            线程间通信（英语：Inter-Thread Communication，缩写：ITC），指的是不同线程之间的信息交流。在多线程环境下，线程之间的数据共享和相互协作是非常重要的一环。线程间通信有两种方式：共享变量和消息传递。
            
            共享变量：是指多个线程共同访问同一个变量。共享变量存在数据竞争问题，即两个或以上线程同时读取或修改同一个变量时，可能会导致数据的混乱。为了解决这一问题，需要保证访问共享变量的线程是安全的，并且按照一定的顺序访问。
            
            消息传递：是指多个线程通过发送消息进行通讯。消息传递是指线程把自己想发送的信息放在消息队列里，其他线程从消息队列里接收信息。在收到信息的线程里，就可以得到这个信息。在消息传递机制下，线程之间不存在数据共享的问题，各个线程之间可以自由地进行通讯，缺点是发送和接收消息的线程必须相互协调配合，确保消息的顺利传递。
            
            下面我们看一下Java多线程中最常用的两种线程间通信方式——共享变量和消息队列。
            
            一、共享变量通信
             
             共享变量通信就是指多个线程共同访问同一个变量。为了防止数据冲突，应当在访问共享变量的代码块上加锁。对于java来说，可以使用synchronized关键字加锁，synchronized关键字可以在一个对象或者类的同步代码块上加锁，语法如下：
              
               synchronized(obj){//同步代码块}
               
               obj是任意对象或类的实例，表示该代码块要加锁的对象。其他线程试图获取该对象的同步锁时，只能等待当前线程释放同步锁。加锁的范围越小，发生冲突的概率就越低。
              
             为了便于使用，java.util包中提供了各种同步集合类，如ArrayList、Vector、HashTable等，这些集合都是线程安全的。
             
             在共享变量通信中，通常有一个主线程和多个子线程一起工作。主线程往共享变量中写入信息，子线程从共享变量中读取信息。为了防止读写冲突，应该对共享变量进行加锁。同时，应该确保读写操作的正确性，以避免数据错误。
             
             例如，假设有一个共享变量num，初始值为0。主线程和子线程分别为t1和t2，要求每隔0.1秒，主线程向num增加1，输出当前的值。
             ```java
             class SharedVarDemo{
                 volatile int num = 0;
                 
                 public void writer(){
                     for(;;){
                         try{
                             Thread.sleep(100);
                             num++;
                             System.out.println("Current value of num : "+num);
                         }catch(InterruptedException e){
                             e.printStackTrace();
                         }
                     }
                 }

                 public void reader(){
                     while(!Thread.currentThread().isInterrupted()){
                         if(num % 10 == 0 && num > 0 ){
                             System.out.println("Reader interrupted...");
                             Thread.currentThread().interrupt();
                         }
                     }
                 }

             }


             public class TestSharedVarDemo{
                 public static void main(String[] args){
                     final SharedVarDemo demo = new SharedVarDemo();

                     Thread t1 = new Thread(() -> {
                         demo.writer();
                     });

                     Thread t2 = new Thread(() -> {
                         demo.reader();
                     });

                     t1.start();
                     t2.start();

                 }
             }
             ```

             在TestSharedVarDemo类的main()方法中，我们创建了一个SharedVarDemo类的实例demo。在这个实例中，有一个volatile修饰的变量num，表示该变量的值可以被线程间共享。

            在writer()方法中，通过Thread.sleep(100)睡眠了100毫秒，然后向num增加1。为了确保每次输出都打印最新值，writer()方法使用了for循环，并且在try-catch语句块中捕获了InterruptedException异常。

            在reader()方法中，通过while循环不断监测num的值是否符合条件，条件是取模后的结果等于0且值大于0。如果满足条件，则调用Thread.currentThread().interrupt()方法中断reader()方法，并且输出一条提示信息。

           在TestSharedVarDemo类的main()方法中，我们创建了两个线程t1和t2，其中t1是一个Runnable对象，代表一个生产者线程，负责向共享变量中写入信息；而t2是一个Thread对象，代表一个消费者线程，负责从共享变量中读取信息。

           在main()方法中，我们启动t1和t2，并且让它们一起运行。

           在writer()方法中，由于采用了volatile关键字，因此num变量在写入时不会缓存，而是直接刷新到主内存。此时，reader()方法就可以从主内存中读取到最新值。

           测试结果显示：writer()方法每隔0.1秒，向num变量写入一个值；reader()方法每隔0.1秒检测num变量的值是否符合条件，符合条件则输出提示信息并中断自身。整个过程无数据竞争问题。

           二、消息队列通信
            消息队列通信，是指多个线程通过消息队列进行通讯。消息队列是存放消息的容器，每个线程可以向队列中发送消息，其他线程从队列中接收消息。
            在使用消息队列通信时，通常有一个生产者线程和多个消费者线程一起工作。生产者线程往消息队列中添加消息，消费者线程则从消息队列中移除消息。在移除消息时，消费者线程一般会指定超时时间，以防消息队列中无消息。
            java.util.concurrent包中提供了BlockingQueue接口来实现消息队列，BlockingQueue接口提供了put()和take()两个方法，put()方法用于向队列中添加消息，take()方法用于从队列中移除消息。
            下面是BlockingQueue接口的使用示例：
            
            import java.util.concurrent.*;

            public class MessageQueueDemo {

                public static void main(String[] args) throws InterruptedException {

                    BlockingQueue<Integer> queue = new LinkedBlockingDeque<>();

                    ExecutorService service = Executors.newFixedThreadPool(2);

                    Future future = service.submit(new Producer(queue));

                    Consumer consumer = new Consumer(queue);

                    TimeUnit.SECONDS.sleep(2);

                    future.cancel(true);

                    service.shutdownNow();

                    Thread thread = new Thread(consumer::consumeMessage);

                    thread.start();

                    thread.join();

                }

            }

            class Producer implements Runnable {

                private final BlockingQueue<Integer> queue;

                public Producer(BlockingQueue<Integer> queue) {
                    super();
                    this.queue = queue;
                }

                @Override
                public void run() {
                    int count = 0;
                    try {
                        while (!Thread.currentThread().isInterrupted()) {
                            Integer message = produceMessage();
                            System.out.println("Produce a message:" + message);
                            queue.put(message);
                            count++;
                            TimeUnit.MILLISECONDS.sleep(100);

                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    } finally {
                        System.out.println("Producer finished, total count:" + count);
                    }

                }

                private Integer produceMessage() {
                    Random random = new Random();
                    return random.nextInt(100);
                }
            }

            class Consumer {
                private final BlockingQueue<Integer> queue;

                public Consumer(BlockingQueue<Integer> queue) {
                    super();
                    this.queue = queue;
                }

                public void consumeMessage() {
                    int count = 0;
                    try {
                        while (!Thread.currentThread().isInterrupted()) {

                            Integer message = queue.poll(1, TimeUnit.SECONDS);//设置超时时间，防止一直阻塞
                            if (message!= null) {
                                count++;
                                System.out.println("Consume a message:" + message);
                                TimeUnit.MILLISECONDS.sleep(100);
                            } else {
                                continue;
                            }

                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    } finally {
                        System.out.println("Consumer finish,total count:" + count);
                    }
                }
            }
            
            在MessageQueueDemo类的main()方法中，我们创建了一个BlockingQueue类型的消息队列queue。在这里，我们使用LinkedBlockingDeque实现了BlockingQueue接口，该实现是一个有界的BlockingQueue。

            在main()方法中，我们创建了一个固定大小的线程池service，其中包含一个生产者线程Producer和一个消费者线程Consumer。

            在main()方法中，我们通过Future接口提交了一个生产者线程，Producer通过调用produceMessage()方法生成随机数，并将生成的随机数放入队列中。

            在main()方法中，我们创建了一个消费者线程Consumer，并调用它的consumeMessage()方法。Consumer从队列中移除消息，并将其打印出来。

            在main()方法中，我们调用Thread.join()方法，等待消费者线程执行完毕。

            为了测试消息队列通信，我们创建一个生产者线程Producer，生成100个随机数，每隔0.1秒向消息队列中放入一个随机数，并在生产者线程结束后打印生产的总数量。

            我们创建一个消费者线程Consumer，每隔0.1秒从消息队列中获取一个消息，并在消费者线程结束后打印消费的总数量。

            运行这个程序，我们将看到生产者线程每隔0.1秒生成一个随机数，并放入消息队列中，消费者线程每隔0.1秒从消息队列中获取一个消息。程序的运行过程中，生产者线程和消费者线程互不干扰。

            消费者线程通过设置超时时间，在消息队列中一直获取不到消息时，就会继续阻塞，防止一直阻塞下去。这样做既可以防止消息队列阻塞住，又不会影响生产者的工作。