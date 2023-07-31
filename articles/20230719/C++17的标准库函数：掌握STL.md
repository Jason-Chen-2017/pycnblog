
作者：禅与计算机程序设计艺术                    
                
                
C++的标准库提供了很多常用的数据结构和算法，如数组、链表、队列、栈等，以及多种函数，能够帮助程序更加高效地运行。其中，容器类模板（Container class templates）和算法类模板（Algorithmic class templates）是最常用的两个类模板，所以必备的知识点之一就是理解它们的机制及其优劣势，如何选择合适的类型。STL也是C++中最具影响力的编程技术之一，在C++11之后，才逐渐形成完整体系，成为各大开发人员必备的工具箱。本文从如下几个方面全面阐述了C++17中的新特性——即C++标准库的更新内容，并通过对一些重要的容器类模板和算法类模板进行解析，帮助读者了解这些容器类模板和算法类模板的机制和操作方法。
# 2.基本概念术语说明
## 2.1 STL(Standard Template Library)简介
STL是指Standard Template Library的缩写，是C++的一组通用数据结构和算法模板。它包括六个主要组件：容器类模板、迭代器类模板、算法函数模板、表达式模板、内存管理函数模板和异常处理类模板。STL的目的是为了统一数据结构和算法的接口，实现可移植性和可重用性，并增加语言的功能性和灵活性。通过继承和组合这些模板，可以快速创建符合需求的数据结构和算法。STL支持高效率、线程安全和大量实用的算法，且易于学习和使用。
## 2.2 数据结构与迭代器
### 2.2.1 序列容器
C++中有以下五种序列容器：

1. vector: 支持随机访问的动态数组容器，可以在任意位置插入或删除元素。vector<int> vec;表示创建一个整数型的空向量。vec.push_back(n)向尾部添加一个元素n；vec[i]返回第i个元素，vec.size()返回向量长度，vec.empty()判断是否为空，vec.erase(it)删除元素，vec.insert(pos, val)在指定位置插入值val。

2. list: 双向链表容器，支持迭代器遍历和插入删除操作。list<int> lst;表示创建一个整数型的空列表。lst.push_front(n)在头部插入一个元素n，lst.pop_front()删除头部元素，lst.push_back(n)在尾部插入一个元素n，lst.pop_back()删除尾部元素。其他操作类似vector。

3. deque: 可以看作是动态数组容器的扩展版本，具有双端队列的性质。deque<int> dq;表示创建一个整数型的空双端队列。dq.push_back(n)和dq.pop_back()分别在队尾和队首添加或删除元素；dq.push_front(n)和dq.pop_front()分别在队首和队尾添加或删除元素；dq[i]返回第i个元素，dq.at(i)返回第i个元素，dq.front()返回队首元素，dq.back()返回队尾元素，dq.size()返回队列长度，dq.empty()判断是否为空。

4. forward_list: 是单向链表容器，只能向前遍历。forward_list<int> flst;表示创建一个整数型的空单向列表。flst.push_front(n)在队首插入一个元素n，flst.pop_front()删除队首元素。其他操作类似list。

5. array: 固定大小的数组容器。array<int, 10> arr{1, 2, 3};表示创建一个整数型的10元素数组arr，并初始化元素值为1，2，3。arr.fill(0)将数组所有元素的值设置为0；arr[i]和arr.at(i)都可以获取第i个元素的值，但不能修改元素值。

### 2.2.2 关联容器
C++中还有三种关联容器：

1. set：集合，存储唯一的键值对，根据键自动排序，键不能重复。set<string> s; s.insert("hello"); // 插入字符串hello到集合s中。s.count("world") 返回字符串"world"在集合s中出现的次数，而不存在则返回0。

2. map：映射，存储键值对，根据键自动排序。map<string, int> mp; mp["apple"] = 100; // 插入键值对{"apple":100}到映射mp中。mp.find("banana") 返回一个指向键为"banana"的元素的指针或引用，如果查找失败则返回一个空指针或引用。

3. unordered_map：哈希映射，存储键值对，以哈希表方式存储。unordered_map<string, int> umap; umap["apple"] = 100; // 插入键值对{"apple":100}到哈希映射umap中。

### 2.2.3 堆
C++中有一个堆（heap）数据结构，可以用来实现优先级队列。

1. priority_queue：是一个容器适配器，它提供对顶元素访问和弹出操作的高效率。priority_queue<int> pq; 表示创建一个整数型的空优先级队列。pq.emplace(n) 插入元素n到优先级队列中。pq.top() 获取优先级队列中的顶元素，而不弹出该元素。pq.pop() 删除优先级队列中的顶元素。

2. make_heap / push_heap / pop_heap / sort_heap：用于构造堆，向堆中插入元素，从堆中取出最大元素，建立最大堆，并将堆化数组（vector/deque）排序。

### 2.2.4 迭代器
C++中的迭代器是一种类模板，是一种特殊类型的对象，允许不同类的容器共享同一组元素，并且可以在不同的时间点访问相同元素。C++的迭代器支持向前移动、后退移动、随机访问，以及比较、减法和增法操作。

C++迭代器有两种类型：

1. 输入迭代器（Input Iterator）：只能读取元素，不能写入元素，只支持++、--、==、!=、*、->。例如，istream_iterator<int> isit; 表示一个输入迭代器，可以用来读取整数流。isit++; ++isit; *isit; isit->；

2. 输出迭代器（Output Iterator）：只能写入元素，不支持++、--、==、!=、*、->，但是可以赋值。例如，ostream_iterator<int> osit(cout, " "); 表示一个输出迭代器，可以用来向cout输出多个整数，间隔一个空格。*osit = 3; ++osit; *osit = 5; 。

