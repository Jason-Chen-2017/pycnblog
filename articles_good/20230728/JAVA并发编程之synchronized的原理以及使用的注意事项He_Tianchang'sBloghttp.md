
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　“同步”这个词在多线程编程中是一个重要的概念，它定义了多个线程执行同一个任务时对共享资源的访问控制。Java提供了synchronized关键字来支持对共享资源的同步访问。本文将从基本概念、原理、使用方法、优缺点、适用场景、锁优化等方面对synchronized进行详细介绍。

         # 2.基本概念
         　　“同步”（Synchronization）是指计算机按照设计者意图来运行程序的一段时间段，并且不被其他进程打断，因此同步可以实现数据完整性的保证和程序的正确性。由于资源共享带来的复杂性和独占锁的性能问题，使得多线程编程十分困难。而在java中，提供了一种称为同步的机制，通过该机制能够实现多线程之间的通信和协调。所谓同步，就是当多个线程同时访问某个资源时，保证只能有一个线程访问该资源，其他线程则要等待。

         　　为了更好地理解synchronized的概念，先了解一下“互斥量”（Mutex）的概念。互斥量又称信号量，是在进程间通信中的一种工具。信号量用来控制某一资源可供多少进程进行访问。互斥量实际上是利用操作系统提供的信号量机制实现的，其工作原理如下：创建一个初始值为1的信号量，每当某个进程试图获取互斥量时，如果此时信号量的值大于0，那么就把信号量减1，并进入临界区；如果信号量的值等于0，那么就阻塞当前进程，直到其他进程释放了互斥量，并将信号量加1后唤醒该进程。释放互斥量时，需要将信号量加1。

         # 3. synchronized 的原理及使用方法
         　　在Java中，每个对象都对应一个monitor锁，当不同的线程尝试对相同对象的同步块进行加锁时，只有一个线程能成功地获取锁并继续执行，其他线程必须等待前一个线程释放锁之后才能抢夺。加锁过程相当于尝试获取互斥量，锁的获取过程其实是利用CAS指令实现的，每次更新锁的计数器，当计数器不等于0时，说明锁已经被其他线程获取过，当前线程需等待；当计数器为0时，说明此刻没有其他线程获得锁，当前线程就可以获取锁。释放锁的过程则是将计数器设置为0即可。

         　　1.同步方法
           当一个类的方法上有synchronized关键字时，它作用于整个方法，所以当两个线程同时调用该方法时，只有一个线程能进入该方法并执行，另一个线程必须等待第一个线程结束后才能再次调用该方法。

           ```
           public class SynchronizedDemo {
               
               private int count = 0;
                
               public synchronized void increase() throws InterruptedException{
                   for(int i=0;i<1000000;++i){
                       ++count;
                   }
               }
               
               public static void main(String[] args) throws InterruptedException {
                   
                   final SynchronizedDemo demo = new SynchronizedDemo();
                   
                   Thread t1 = new Thread(){
                       
                       @Override
                       public void run() {
                           try {
                               demo.increase();
                           } catch (InterruptedException e) {
                               e.printStackTrace();
                           }
                           
                       };
                       
                   };
                   
                   Thread t2 = new Thread(){
                       
                       @Override
                       public void run() {
                           try {
                               demo.increase();
                           } catch (InterruptedException e) {
                               e.printStackTrace();
                           }
                           
                       };
                       
                   };
                   
                   long start = System.currentTimeMillis();
                   t1.start();
                   t2.start();
                   t1.join();
                   t2.join();
                   long end = System.currentTimeMillis();
                   System.out.println("count=" + demo.count);
                   System.out.println("cost time:"+(end-start)+"ms");   //输出结果：count=2000000 cost time:75ms
               
           }
           
           ```

           
           在SynchronizedDemo类的increase()方法上添加synchronized关键字，这表示所有对该方法的调用都必须由同一个线程执行。即使是不同线程调用该方法也会串行执行。


         　　2.同步代码块

         　　可以使用同步代码块来实现同步。语法形式如下：

         　　```
         　　public void method(){
         　　synchronized(this){ //同步代码块的入口
         　　　　　　 //需要同步的代码
         　　}
         　　}
         　　```
          
         　　synchronized的作用范围是整个代码块，即括号里的内容。当两个线程同时执行method()方法时，只有一个线程能成功地获取锁并进入同步代码块内执行，其他线程必须等待前一个线程执行完毕后才能继续执行。

         　　在同步代码块中，不能出现以下三种情况：

           - 对变量赋值语句
           - 对象实例化语句
           - 方法调用语句



         　　但是可以通过以下方式绕开上述限制：

           - 将变量声明为volatile类型，在方法外面声明
           - 通过局部变量缓存中间结果的方式避免对共享变量的修改

        
           ```
           public class SynchronizedBlockDemo {
           
               volatile boolean flag = true;
               
               public void printA() {
                   while(!flag) {}
                   System.out.print("a");
                   flag = false;
               }
               
               public void printB() {
                   while(flag) {}
                   System.out.print("b");
                   flag = true;
               }
               
               public static void main(String[] args) throws InterruptedException {
                   
                   final SynchronizedBlockDemo block = new SynchronizedBlockDemo();
                   
                   Thread t1 = new Thread(){
                       
                       @Override
                       public void run() {
                           for(int i=0;i<10;++i){
                               block.printA();
                           }
                       };
                       
                   };
                   
                   Thread t2 = new Thread(){
                       
                       @Override
                       public void run() {
                           for(int i=0;i<10;++i){
                               block.printB();
                           }
                       };
                       
                   };
                   
                   t1.start();
                   t2.start();
                   
                   t1.join();
                   t2.join();
                   
               }
               
           }
           
           ```

           
           为了实现“打印AB”的效果，main()方法里面创建了一个SynchronizeBlockDemo类型的对象block，并创建两个线程t1和t2。t1和t2分别负责打印“A”和“B”，并且通过volatile的flag属性进行通信。两个线程通过while循环一直等待对方的打印动作完成后，再通知对方自己已经准备完毕。这样实现的目的是使得t1和t2交替打印，而不是同时打印。

       　　3. 线程间的通信

           可以通过wait()和notify()/notifyAll()方法来实现线程间的通信。wait()方法让线程处于等待状态，直到其他线程调用了notify()/notifyAll()方法通知才恢复执行，否则一直保持等待状态。比如：

           ```
           public class CommunicationDemo {
               
               Object lock = new Object();
               
               public void sendMsg(String msg) {
                   try {
                       synchronized(lock) {
                           if(!"over".equals(msg)) {
                               wait(); //等待接收消息
                           }
                           System.out.println("send msg:" + msg);
                       }
                   } catch (InterruptedException e) {
                       e.printStackTrace();
                   }
               }
               
               public void receiveMsg(final String msg) {
                   new Thread(){
                       
                       @Override
                       public void run() {
                           try {
                               sleep((long)(Math.random()*10));//随机休眠
                               synchronized(lock) {
                                   notify();
                                   System.out.println("receive msg:" + msg);
                                   lock.notifyAll();//通知所有的线程
                                   lock.notify();//通知优先级最高的一个线程
                                   sleep((long)(Math.random()*10));//随机休眠
                                   lock.wait();
                               }
                           } catch (Exception e) {
                               e.printStackTrace();
                           }
                       };
                       
                   }.start();
               }
               
               public static void main(String[] args) {
                   
                   final CommunicationDemo comm = new CommunicationDemo();
                   
                   comm.receiveMsg("hello");
                   comm.receiveMsg("world");
                   comm.receiveMsg("over");
                   
                   new Thread(){
                       
                       @Override
                       public void run() {
                           comm.sendMsg("hi!");
                       };
                       
                   }.start();
               
           }
       
           ```

           
           此例中，有三个线程，分别是comm.sendMsg(), comm.receiveMsg("hello"), comm.receiveMsg("world")。其中comm.sendMsg()是发送消息的线程，其余两线程都是收取消息的线程。sendMsg()线程首先判断是否接收完毕，如果没有接收完毕，就一直wait()，直到接收完毕才再次发送消息。其他线程也是一样，接收完毕后就notify()主线程，然后一起向主线程发送消息。这里的通信机制就是基于对象的锁机制实现的。
        
       　　4. synchronized 和 Lock

           Java的1.5版本引入了新的并发包java.util.concurrent，其中包含了两个重要的类：ReentrantLock和ReadWriteLock。这两个类都提供了比synchronized更灵活、功能更强大的锁机制。

           ReentrantLock类相较于synchronized来说更具扩展性和灵活性，并且具有响应中断的能力，并且提供了一些高级功能，如尝试非阻塞地获取锁，超时设置等。而且，它还可以绑定多个条件，方便线程间的同步。

           ReadWriteLock接口提供了读锁和写锁，允许多个线程同时读同一个资源，而只允许一个线程对某个资源进行写入。这种方式允许多线程共同访问一个共享资源，但又防止写操作冲突，提高了程序的并发性能。

           从效率的角度来看，Lock接口比synchronized更快，因为它底层采用的是基于monitor锁的方案，而monitor锁的效率要比wait/notify机制高很多。但是，Lock接口也比synchronized更复杂，使用起来也更加复杂。

           在性能要求比较苛刻的情况下，可以考虑使用Lock接口来代替synchronized关键字。

       　　5. 使用建议

           synchronized关键字应该只用于同步线程间的数据访问，不应该用于同步控制流，如if、for、while等。如果要控制流程，推荐使用Lock或Condition接口来实现。

           如果某个代码块或方法不是性能关键点，则尽可能使用更简单的synchronized代码块或方法，避免发生额外的同步开销。如果存在多线程竞争的情景，则推荐使用Lock和线程池来提升并发处理能力。

       　　6. synchronized和ReentrantLock的区别

           |              | Synchronized           | ReentrantLock                        |
           | ------------ | ---------------------- | ------------------------------------ |
           | 可重入性      | 不可重入                | 可重入                              |
           | 线程等待策略 | 公平锁                  | 非公平锁                            |
           | 锁的机制      | 对象监视器              | CAS+内存屏障                         |
           | 锁语义        | 无                     | 可中断、可延迟                      |
           | 锁升级        | 否                     | 是                                  |
           | 性能开销       | 小                     | 大                                  |
           | 公平锁        | 没有                   | 有                                  |
           | 条件          | 没有                   | 有                                  |
           | 中断响应      | 支持                   | 支持                                |
           | 锁池          | 没有                   | 无限数量                             |

