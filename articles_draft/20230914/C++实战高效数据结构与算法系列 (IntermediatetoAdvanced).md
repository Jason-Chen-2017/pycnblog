
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1. 数据结构
数据结构（Data Structure）是计算机存储、组织数据的方式，它是计算机科学的一门基础学科，是构建算法和程序的重要基石之一。数据结构最主要的功能就是用来管理数据的存取、修改、检索等操作。数据结构的分类可以分成逻辑结构和物理结构两大类。逻辑结构是指按照数据之间的关系将数据组织在一起的方法，包括数组、链表、队列、栈、树、图等；而物理结构则是指数据如何在计算机内存储并实现访问，比如顺序表、散列表、二叉搜索树、平衡二叉树、堆、B+树等。

## 2. 算法
算法（Algorithm）是用于解决特定问题或计算任务的一组规则、指令或者脚本，它是用来描述怎样从输入中得到输出，算法代表着计算模型的设计方案，指导计算机执行相关指令或运算过程。目前应用最广泛的算法有排序算法、搜索算法、查找算法、贪婪算法、动态规划算法、字符串匹配算法、哈希算法、回溯算法、贪心算法、朴素贝叶斯算法、K最近邻算法、线性规划算法等。

## 3. C++语言
C++（C plus plus，中文为“加拿大海滩”）是一种支持多种编程范式的高级程序设计语言，其提供了诸如面向对象的、命令式、函数式编程等多种特性，可广泛应用于系统开发、网络服务、图像处理、嵌入式系统、数据库系统、图形学、科学计算等领域。

## 4. 本系列文章
本系列文章分为上下册共计五卷，共同构成了一套完整的数据结构与算法学习教程，适合具有扎实的基础知识、掌握数据结构的基础者。每一卷的文章内容相对独立，相互之间没有交集，阅读本系列文章将会更容易理解、记忆、扩展、运用到实际工作当中。

## 5. 作者简介
赵旭阳，北京邮电大学电子工程及信息科学系博士研究生，现任CTO，负责公司核心业务系统的研发，精通数据结构和算法，曾就职于阿里巴巴、腾讯、百度等大型互联网公司，获得过全国各个高校的计算机竞赛一等奖、二等奖和三等奖，并期待用自己的努力帮助更多的人摆脱信息茧房。微信号：zhaoaoxiang1917。

# 2. C++实战数据结构与算法 2——队列
## 1. 概述
队列（queue）是有序集合，它只允许在尾部添加元素，并从头部删除元素。队列通常被用作先进先出（FIFO:First In First Out）的数据结构。如下图所示：


从上图可以看出，队列里有三个元素，分别为A、B、C。我们可以使用两个栈来实现队列，一个栈用来保存入队元素，另一个栈用来保存出队元素。入队时，我们将元素压入第一个栈，出队时，我们再弹出第二个栈。

## 2. 定义
### （1）简单队列
#### 2.1 节点结构
```c++
template <typename T>
struct Node {
    T data;
    Node<T>* next;

    Node(const T& value):data(value),next(nullptr){};
};
```

#### 2.2 队列结构
```c++
template <typename T>
class Queue {
  private:
    Node<T>* head = nullptr; // 队首指针
    Node<T>* tail = nullptr; // 队尾指针

  public:
    void push(const T& item);    // 入队
    bool pop();                  // 出队
    const T& front() const;      // 获取队首元素
    int size() const;            // 返回队列大小
};
```

### （2）循环队列
#### 2.1 节点结构
```c++
template <typename T>
struct Node {
    T data;
    Node<T>* next;

    Node(const T& value):data(value),next(nullptr){};
};
```

#### 2.2 队列结构
```c++
template <typename T>
class CircularQueue {
  private:
    static const int MAXSIZE = 100;   // 最大容量
    int rear = -1;                    // 队尾指针
    int front = 0;                    // 队首指针
    int size_ = 0;                    
    std::unique_ptr<Node<T>> arr[MAXSIZE];

  public:
    CircularQueue():rear(-1),front(0),size_(0){};
    ~CircularQueue(){
        while(!empty()){
            pop();
        }
    };
    
    // 判断队列是否为空
    bool empty() const{return size_ == 0;};
    
    // 入队操作
    bool enqueue(const T& item){
        if(isFull()) return false;
        
        auto p = std::make_unique<Node<T>>(item);
        ++rear;
        arr[rear] = std::move(p);
        ++size_;
        return true;
    };
    
    // 出队操作
    bool dequeue(){
        if(isEmpty()) return false;

        --size_;
        ++front;
        if(front > rear){
            front -= MAXSIZE + 1;
            rear -= MAXSIZE + 1;
        }
        return true;
    };
    
    // 获取队首元素
    const T& getFront() const{
        assert(!isEmpty());
        return arr[(front + ((rear - front) % MAXSIZE)) % MAXSIZE]->data;
    };
    
    // 获取队列大小
    int size() const{return size_;};
    
    // 判断队列是否已满
    bool isFull() const{return size_ >= MAXSIZE;};
};
```