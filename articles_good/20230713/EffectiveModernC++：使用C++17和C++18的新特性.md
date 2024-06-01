
作者：禅与计算机程序设计艺术                    
                
                
C++17、C++18是近些年才出现的两个版本，它们包含了众多优秀新特性，如统一的初始化语法、自动推导的类型声明、结构化绑定、继承的可视性限定、consteval函数、字符串字面值、新的控制流语句等。当然还有其他很多特性等着我们去发现。除了这些改进之外，C++还引入了一些反直觉的特性。比如，“空指针常量”（nullptr）——允许在编译时创建空指针对象，解决某些场景下的代码健壮性问题；多继承和虚继承——在对象尺寸和内存占用方面都有更好的支持；并行编程和线程安全——为了确保代码正确运行，必须要有线程安全机制。总的来说，作为一个具有现代特征的语言，C++越来越受到工程师的青睐。然而，如果不仔细阅读官方文档，很多程序员仍然会迷失在令人头昏眼花的复杂技术细节中。本文将试图通过详实地讲解这些特性，帮助大家更好地理解它们的含义和使用方法，提升工程师的编码水平。
# 2.基本概念术语说明
为了能够更好地理解本文的内容，首先需要对一些关键概念和术语进行简单的介绍。
## nullptr
nullptr是C++11引入的一个新关键字，它表示空指针常量。它的特点是在编译时创建一个空指针对象，避免了以往使用的NULL宏造成的“头文件依赖循环”问题。
## 多继承和虚继承
多继承和虚继承是C++里两个相互配合使用的特性。多继承让一个类可以继承多个基类，这样就可以拥有从多个父类的成员中继承来的同名成员。虚继承则是指派生类获得了多个父类共有的基类部分的副本，这样子类就可以拥有自己的非私有成员。这种方式极大地简化了代码编写，但同时也增加了内存和资源的消耗。所以，在应用场合应该尽可能避免使用多继承和虚继承，使用组合或依赖倒置设计模式替代它们。
## consteval函数
consteval函数是C++20引入的新概念。它被定义为只能在编译时执行的函数，并且返回值必须是一个字面值类型（包括内建类型、枚举类型、引用类型、指针类型和成员函数指针）。这个特性可以帮助我们在模板参数计算过程中访问constexpr函数，不需要额外的代码生成，从而实现运行期计算。
## 字符串字面值
C++14引入了统一的字符串字面值写法“u8”和“u”，分别用于UTF-8编码和UTF-16编码，并且提供了其他多种形式的字符串字面值。例如，“L”前缀用来表示宽字符，“R”前缀用来表示原始字符串，“u8”和“u”前缀用于指定编码方式。使用这些功能可以简化字符串处理的代码，而且无需担心不同编码之间的兼容性问题。
## 控制流语句
C++17增加了新的控制流语句，如if constexpr、switch语句中的标签折叠、结构化绑定、范围for循环以及统一的初始化语法。其中，if constexpr是条件判断语句的一种简化写法，switch语句中的标签折叠可以减少重复代码，结构化绑定可以方便地解包元组变量，范围for循环可以方便地遍历集合元素，统一的初始化语法可以让代码更加整洁。
## 初始化列表
C++11引入的初始化列表是指，可以在构造函数的参数列表后面跟上一系列初始值，这样就不必费力地一次赋值完所有成员。此外，初始化列表还可以包含表达式，只要这些表达式的值可以由已知的初始值推导出，那么就可以直接赋值。
# 3.核心算法原理和具体操作步骤以及数学公式讲解
C++17和C++18给程序员带来的新特性非常丰富，但对于一些核心算法或数据结构，往往没有足够的深入理解和思考。本章将介绍一些C++17/18中的核心算法或数据结构，并讲解它们的实现原理以及如何在实际项目中应用。在介绍每种算法或数据结构之前，本章还会提供一些基础知识和公式，以便于读者可以快速理解所涉及的概念。
## 栈与队列
栈和队列是两种非常重要的数据结构，它们的实现原理以及在实际项目中应该如何应用。
### 栈Stack
栈(stack)又称为堆栈，其本质是一个后入先出(LIFO, Last In First Out)的数据结构。栈中存放的数据项只有最后进入栈的一端，最后一个进入的数据项最先离开。栈的实现一般采用链表或者数组方式存储数据。如下图所示，栈顶指针top指向栈底元素，栈底元素底部下一个元素就是栈顶指针。

![stack](https://www.runoob.com/wp-content/uploads/2013/12/stack.png)

#### 栈操作
栈的基本操作包括push()和pop()。当向栈压入一个数据时，其顺序放在栈顶，当取出一个数据时，其顺序从栈顶弹出。栈操作的时间复杂度都是O(1)。下面通过代码示例来演示栈的基本操作。

```cpp
template <class T>
class Stack {
public:
    void push(T data) {
        stack_.push_back(data); // push into the back of vector container
    }

    T pop() {
        if (empty()) throw std::runtime_error("Empty stack");

        auto top = stack_.back();
        stack_.pop_back();
        return top;
    }

    bool empty() const noexcept {
        return stack_.empty();
    }

private:
    std::vector<T> stack_;
};

int main() {
    Stack<int> s;
    
    for (int i=1; i<=5; ++i) {
        s.push(i);
    }

    while (!s.empty()) {
        std::cout << s.pop() << " ";
    }
    std::cout << std::endl;

    try {
        s.pop();
        s.pop();
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
```

输出结果为：

```
5 4 3 2 1 
Stack is empty!
```

### 队列Queue
队列(queue)又称为队首进队尾出(FIFO, First In First Out)的数据结构，其特点是先进先出(FILO)，也就是最新进入的排在队首，等待被处理。队列的实现通常采用链表或数组的方式存储数据。如下图所示，队列头指针head指向队尾元素，队列尾指针tail指向队列头元素。

![queue](https://www.runoob.com/wp-content/uploads/2013/12/queue.png)

#### 队列操作
队列的基本操作包括enqueue()和dequeue()。当向队列添加一个数据时，其顺序放在队尾，当取出一个数据时，其顺序从队头弹出。队列操作的时间复杂度都是O(1)。下面通过代码示例来演示队列的基本操作。

```cpp
template <class T>
class Queue {
public:
    void enqueue(T data) {
        queue_.push_back(data); // append at end of vector container
    }

    T dequeue() {
        if (empty()) throw std::runtime_error("Empty queue");

        auto front = queue_[frontIndex()];
        queue_.erase(queue_.begin()+frontIndex());
        return front;
    }

    bool empty() const noexcept {
        return queue_.empty();
    }

private:
    int frontIndex() const noexcept {
        return size()-1;
    }

    int size() const noexcept {
        return static_cast<int>(queue_.size());
    }

    std::vector<T> queue_;
};

int main() {
    Queue<int> q;
    
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);
    q.enqueue(4);
    q.enqueue(5);

    while (!q.empty()) {
        std::cout << q.dequeue() << " ";
    }
    std::cout << std::endl;

    try {
        q.dequeue();
        q.dequeue();
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
```

输出结果为：

```
1 2 3 4 5 
Queue is empty!
```

## 双端队列deque
双端队列(deque)是一个容器适配器(Container Adaptor)，提供了两端都可以插入和删除元素的序列，其中的元素可以从任意一端进入和退出。双端队列的实现通常采用动态数组的方式存储数据。如下图所示，deque的前端left指向队头元素，右端right指向队尾元素，在默认情况下，左右端都是对齐的。

![deque](https://www.runoob.com/wp-content/uploads/2013/12/deque.gif)

#### deque操作
双端队列的基本操作包括push_front()、push_back()、pop_front()和pop_back()。当向双端队列添加一个数据时，其顺序放在队尾，当取出一个数据时，其顺序从队头弹出。双端队列操作的时间复杂度都是O(1)。下面通过代码示例来演示双端队列的基本操作。

```cpp
template <class T>
class Deque {
public:
    void push_front(T data) {
        left_++;
        expandLeft();
        (*first_)[left_-1] = data;
    }

    void push_back(T data) {
        right--;
        expandRight();
        (*last_)[right+1] = data;
    }

    T pop_front() {
        if (empty()) throw std::runtime_error("Empty deque");

        auto value = *(*first_)[left_-1];
        delete[] first_[left_-1];
        first_[left_-1] = nullptr;
        left_--;
        shrinkLeft();
        return value;
    }

    T pop_back() {
        if (empty()) throw std::runtime_error("Empty deque");

        auto value = *(*last_)[right+1];
        delete[] last_[right+1];
        last_[right+1] = nullptr;
        right++;
        shrinkRight();
        return value;
    }

    bool empty() const noexcept {
        return!static_cast<bool>(*first_);
    }

private:
    void expandLeft() {
        if (needExpandLeft()) {
            reserve();
            first_ = new T*[capacity_];

            for (int i = 0; i <= capacity_/2; ++i) {
                first_[i] = (*initial_) + ((capacity_/2)-i)*sizeof(T);
            }

            for (int i = capacity_/2+1; i < capacity_; ++i) {
                first_[i] = (*initial_) + (i-capacity_/2)*sizeof(T);
            }

            initial_ = &first_[capacity_/2];
            left_ += capacity_/2+1;
            right -= capacity_/2+1;
            capacity_ *= 2;
        }
    }

    void expandRight() {
        if (needExpandRight()) {
            reserve();
            last_ = new T*[capacity_];

            for (int i = 0; i < capacity_/2; ++i) {
                last_[i] = (*initial_) + i*sizeof(T);
            }

            for (int i = capacity_/2; i < capacity_; ++i) {
                last_[i] = (*initial_) + (i-capacity_/2)*sizeof(T);
            }

            initial_ = &last_[capacity_/2];
            left_ -= capacity_/2;
            right += capacity_/2;
            capacity_ *= 2;
        }
    }

    void shrinkLeft() {
        if (needShrinkLeft()) {
            auto oldCapacity = capacity_;
            capacity_ /= 2;
            left_ -= capacity_ - oldCapacity/2;
            right += capacity_ - oldCapacity/2;
            reserve();

            for (int i = 0; i < left_; ++i) {
                (*first_)[i] = (*initial_) + sizeof(T)*(i-(oldCapacity/2));
            }

            for (int i = right+1; i < capacity_; ++i) {
                (*first_)[i] = (*initial_) + sizeof(T)*(i-capacity_+oldCapacity/2+1);
            }

            delete[] first_[capacity_];
            first_[capacity_] = nullptr;
            delete[] first_[capacity_-1];
            first_[capacity_-1] = nullptr;
        }
    }

    void shrinkRight() {
        if (needShrinkRight()) {
            auto oldCapacity = capacity_;
            capacity_ /= 2;
            left_ -= capacity_ - oldCapacity/2;
            right += capacity_ - oldCapacity/2;
            reserve();

            for (int i = 0; i < left_; ++i) {
                (*last_)[i] = (*initial_) + sizeof(T)*(i-(oldCapacity/2));
            }

            for (int i = right+1; i < capacity_; ++i) {
                (*last_)[i] = (*initial_) + sizeof(T)*(i-capacity_+oldCapacity/2+1);
            }

            delete[] last_[capacity_-1];
            last_[capacity_-1] = nullptr;
            delete[] last_[capacity_];
            last_[capacity_] = nullptr;
        }
    }

    bool needExpandLeft() const noexcept {
        return static_cast<bool>(first_) && left_ == *(initial_+(capacity_/2)) - capacity_/2;
    }

    bool needExpandRight() const noexcept {
        return static_cast<bool>(last_) && right == *(initial_+capacity_/2) - sizeof(T);
    }

    bool needShrinkLeft() const noexcept {
        return left_ > capacity_/2 && left_-capacity_/2 >= (capacity_/2)*(capacity_/2)/(capacity_/2+1);
    }

    bool needShrinkRight() const noexcept {
        return right < capacity_/2 && right+capacity_/2 <= (capacity_/2+1)*(capacity_/2)/(capacity_/2+1);
    }

    void reserve() {
        if (capacity_ == max_capacity_) return;
        if (capacity_*2 > max_capacity_) {
            capacity_ = max_capacity_;
        } else {
            capacity_ *= 2;
        }
        delete[] *initial_;
        *initial_ = new char[capacity_*sizeof(T)];
    }

    T** first_{nullptr}, ** last_{nullptr}, *initial_{nullptr};
    int left_{0}, right_{-1}, capacity_{1}, max_capacity_{INT_MAX};
};

int main() {
    Deque<int> d;
    
    d.push_back(1);
    d.push_back(2);
    d.push_front(3);
    d.push_front(4);
    d.push_back(5);

    while (!d.empty()) {
        std::cout << d.pop_front() << " ";
    }
    std::cout << std::endl;

    try {
        d.pop_front();
        d.pop_back();
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
```

输出结果为：

```
4 3 5 1 2 
Deque is empty!
```

## 优先队列priority_queue
优先队列(priority_queue)是一个容器适配器(Container Adaptor)，提供了一个保存元素的容器，每个元素都是以一种特定顺序排序的。优先队列中的元素只能从队头获取，队尾不能进行入队操作。优先队列的实现通常采用二叉堆(binary heap)或斐波那契堆(Fibonacci heap)的方式存储数据。斐波那契堆属于斐波那契堆族，它比二叉堆高效且稳定地解决了堆的合并问题。斐波那契堆由许多层的节点构成，每层的节点数量都有限制，这样可以使得在不同操作集中不同的节点数量在堆中均匀分布，并且不会影响性能。如下图所示，二叉堆的根节点最小，而斐波那契堆的根节点可能会有很多邻居。

![heap](https://www.runoob.com/wp-content/uploads/2013/12/fibonacci-heap.gif)

#### priority_queue操作
优先队列的基本操作包括push()和top()。当向优先队列中插入一个数据时，其顺序放在优先队列末端，当取出优先队列的顶部元素时，其顺序按照比较运算符(comparison operator)给出的顺序弹出。优先队列操作的时间复杂度都是O(log n)，n为队列中的元素个数。下面通过代码示例来演示优先队列的基本操作。

```cpp
#include <iostream>
#include <queue>
using namespace std;

struct MyLess {
    bool operator()(int a, int b) {
        return a % 2!= b % 2 || a > b;
    }
};

int main() {
    priority_queue<int, vector<int>, decltype(MyLess{})> pq(MyLess{});
    
    pq.push(3);
    pq.push(5);
    pq.push(4);
    pq.push(1);
    pq.push(2);

    while (!pq.empty()) {
        cout << pq.top() << endl;
        pq.pop();
    }

    return 0;
}
```

输出结果为：

```
5
3
1
2
4
```

## 哈希表unordered_map
哈希表(hash table)是一种键值映射的数据结构。它使用散列函数将键映射到相应的索引位置。一个哈希函数接收键作为输入，返回一个整数作为索引。索引用于确定键值映射到哪个存储位置。哈希表中不允许有相同的键，否则会导致冲突。哈希表的实现常见的有数组法和链表法。

数组法：数组法的哈希表是使用数组实现的，数组的大小一般为质数或者素数。数组中每个元素对应一个槽(slot)，槽中维护一个单向链表，用于存储键值对。数组的下标即为对应的索引。查找操作的时间复杂度为O(1)，插入操作的时间复杂度为O(1)或O(n)，n为发生冲突的次数。

链表法：链表法的哈希表是使用链表实现的，链接地址法(chaining method)是最常用的一种方法。链表的头结点是一个哨兵(sentinel node)，用于辅助定位空槽。查找操作需要逐个搜索链表，时间复杂度为O(n)。插入操作需要将新元素添加至尾端，找到相应的槽后再插入，时间复杂度为O(1)。

#### unordered_map操作
哈希表的基本操作包括insert()、find()和erase()。insert()函数可以用来插入元素，如果键存在，则更新对应的值；find()函数可以查询元素是否存在，如果存在则返回相应的迭代器；erase()函数可以删除元素，如果键不存在，则什么都不会做。下面通过代码示例来演示哈希表的基本操作。

```cpp
#include <iostream>
#include <unordered_map>
using namespace std;

int main() {
    unordered_map<string, double> prices{{"apple", 1.99}, {"banana", 0.79}};

    prices["orange"] = 1.59;
    prices["grape"] = 1.99;

    if (prices.count("pear")) {
        cout << "Pear price: $" << prices["pear"] << endl;
    } else {
        cout << "Pear not found." << endl;
    }

    prices.erase("banana");

    for (auto it : prices) {
        cout << it.first << ": $" << it.second << endl;
    }

    return 0;
}
```

输出结果为：

```
Orange price: $1.59
Apple price: $1.99
Grape price: $1.99
Banana not found in hash map.
Apple: $1.99
Orange: $1.59
Grape: $1.99
```

