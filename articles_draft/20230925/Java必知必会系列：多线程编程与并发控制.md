
作者：禅与计算机程序设计艺术                    

# 1.简介
  

多线程（Thread）是现代操作系统提供的一种并发执行的方式，它允许多个任务同时执行，从而提高了程序的运行效率。Java提供了多线程开发模型，通过JUC包（java.util.concurrent）提供的工具类和框架可以实现多线程编程。在本教程中，我们将详细介绍Java多线程编程的相关知识，包括以下几个方面：

1. Java Thread类
2. 同步机制——synchronized关键字、volatile关键字
3. 线程间通信——wait()方法、notify()方法、notifyAll()方法
4. 线程池及其参数设置
5. Executor框架和ExecutorService接口
6. JUC中的Atomic原子类
7. JUC中的锁以及同步器接口
8. FutureTask类及其使用方式
9. CountDownLatch类及其使用方式
10. CyclicBarrier类及其使用方式
11. Semaphore类及其使用方式

# 2.Java Thread类
## 2.1 创建线程
创建线程需要先创建一个Thread类的子类，并重写run()方法。可以通过两种方式创建线程：
- 通过继承Thread类创建线程：创建一个新的Thread类的子类，并重写run()方法，然后调用start()方法启动线程。
```java
public class MyThread extends Thread {
    @Override
    public void run() {
        System.out.println("Hello World");
    }

    public static void main(String[] args) throws InterruptedException {
        MyThread myThread = new MyThread(); // 第1步：创建线程对象
        myThread.start();                   // 第2步：启动线程
    }
}
```
- 通过实现Runnable接口创建线程：创建一个实现Runnable接口的类，并重写run()方法，然后把该类的实例传递给Thread构造器，最后调用start()方法启动线程。
```java
class MyRunnable implements Runnable{
    @Override
    public void run() {
        try {
            for (int i = 0; i < 5; i++) {
                TimeUnit.SECONDS.sleep(1);
                System.out.println(i + " Hello");
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

public class Main {
    public static void main(String[] args) {
        MyRunnable runnable = new MyRunnable();

        Thread thread = new Thread(runnable);   // 第1步：创建线程对象
        thread.start();                          // 第2步：启动线程
    }
}
```
## 2.2 终止线程
可以通过调用interrupt()方法终止一个正在运行的线程，该方法会抛出InterruptedException异常，并且会清除线程的中断状态。如果要让线程正常终止，应该在退出循环或等待的地方捕获InterruptedException异常并处理掉。如果在线程执行过程中抛出InterruptedException异常，则该异常会终止该线程的执行。如下所示：
```java
class MyRunnable implements Runnable{
    private boolean isRunning = true;

    @Override
    public void run() {
        while (isRunning) {
            System.out.println("Hello!");
            try {
                TimeUnit.MILLISECONDS.sleep(100);
            } catch (InterruptedException e) {
                System.err.println("Interrupted...");
                isRunning = false;
                break; // 中断线程
            }
        }
        System.out.println("Exiting the thread.");
    }

    public void stop() {
        isRunning = false;
    }
}

public class Main {
    public static void main(String[] args) throws InterruptedException {
        MyRunnable runnable = new MyRunnable();

        Thread thread = new Thread(runnable);   // 第1步：创建线程对象
        thread.start();                         // 第2步：启动线程

        TimeUnit.SECONDS.sleep(3);              // 暂停3秒，便于观察输出结果
        runnable.stop();                        // 停止线程
        thread.join();                           // 等待线程结束

        System.out.println("Done!");
    }
}
```
此时线程会被正常终止。打印出的日志：
```
Hello!
Hello!
Hello!
Interrupted...
Exiting the thread.
Done!
```
## 2.3 获取线程信息
获取线程的名字、优先级、ID等信息可以使用Thread类的getters方法。
```java
MyRunnable runnable = new MyRunnable();
Thread thread = new Thread(runnable);   // 第1步：创建线程对象
thread.setName("MyThread");             // 设置线程名
System.out.println("Name: " + thread.getName());    // 第3步：获取线程名
System.out.println("Priority: " + thread.getPriority());     // 第4步：获取线程优先级
System.out.println("ID: " + thread.getId());        // 第5步：获取线程ID
```
此外，还可以调用Thread类的静态方法currentThread()方法来获取当前线程。
```java
MyRunnable runnable = new MyRunnable();
Thread thread = new Thread(runnable);   // 第1步：创建线程对象

Thread currentThread = Thread.currentThread();      // 获取当前线程
System.out.println("Current thread name: " + currentThread.getName());    // 第2步：获取当前线程名
System.out.println("Current thread priority: " + currentThread.getPriority());     // 第3步：获取当前线程优先级
System.out.println("Current thread ID: " + currentThread.getId());        // 第4步：获取当前线程ID
```