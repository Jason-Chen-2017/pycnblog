
作者：禅与计算机程序设计艺术                    
                
                
## C++11之后出现了多线程、异步编程等新特性，如何在这些特性的基础上实现更高效的并行程序成为一个重要问题。本文从C++11标准引入的线程库开始，逐步介绍C++语言中线程及并发编程相关的内容，并且介绍《C++ Primer Plus》第十三版对线程和并发编程方面的部分重点讲解。
# 2.基本概念术语说明
## C++11版本引入了线程(thread)和同步(synchronization)机制，下面对这些术语进行简单阐述。
### 线程（Thread）
线程是进程的一个执行单元。一个进程可以由多个线程组成，各个线程间共享内存空间和文件描述符资源，能够有效提升并发运行效率。创建线程的函数是pthread_create()，它接受一个线程参数结构体，包括线程的启动函数、函数的参数等信息。在Linux/Unix系统下，pthread_create()会创建一个新的线程并返回其标识符，这个标识符是一个整型变量，可用作后续操作的句柄。

```c++
#include <pthread.h>

void* threadFunc(void*) {
    // do some work here...
    return nullptr;
}

int main() {
    pthread_t tid;   // thread id
    int rc = pthread_create(&tid, nullptr, threadFunc, nullptr);

    if (rc == 0) {
        printf("thread created successfully.
");
        // more code to join or detach the thread...
    } else {
        perror("pthread_create failed!");
        exit(-1);
    }

    return 0;
}
```

### 协程（Coroutine）
协程是一种比线程更加轻量级的实现线程的方案。它可以在一个线程里暂停并切换到其他地方继续执行，通过栈的方式来保存当前状态，不需要像线程那样需要分配独立的内存空间。C++20提供了对协程的支持，但目前还是实验性质。

```c++
struct coroutine {
    bool operator()(std::stop_token st) noexcept {
        for (;;) {
            co_await wait_for_io();
            process_io();

            while (!work_.empty()) {
                auto task = std::move(work_.front());
                work_.pop();

                co_await resume_with([&]() -> coro<void> {
                    task();
                    yield;
                });
            }
        }
    }

   private:
    queue<function<void()>> work_;
};

auto start_coroutine() {
    coroutine cobj{};
    return cobj({});
}
```

### 锁（Lock）
锁是保证线程安全访问共享资源时所用的工具。每个线程都持有一个锁，当其他线程试图获得这个锁的时候，就需要等待。在同一时间只允许一个线程持有锁，其他线程必须等当前线程释放锁才能获得。在C++11版本中，为了保证性能，通常使用互斥锁或读写锁。

```c++
class MyClass {
   public:
    void lock() { mutex_.lock(); }

    void unlock() { mutex_.unlock(); }

   private:
    mutex mutex_;
};
```

### 条件变量（Condition Variable）
条件变量是用于阻塞某个线程，直到某些特定条件满足之后才唤醒的工具。调用wait()方法使线程进入阻塞状态，调用notify()/notifyAll()方法通知阻塞状态的线程。在C++11版本中，条件变量是由condition_variable类模拟出来的。

```c++
class MyClass {
   public:
    void signal() { cv_.notify_one(); }

    void broadcast() { cv_.notify_all(); }

   private:
    condition_variable cv_;
};
```

### Future（未完待续）

