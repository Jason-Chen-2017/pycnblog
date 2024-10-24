
作者：禅与计算机程序设计艺术                    

# 1.简介
  


数据结构和算法是计算机科学的基础课。但是，在实际应用中，学习者往往缺乏对数据结构和算法知识的理解、掌握、运用能力。因此，本书以实践为主线，全面、系统地介绍数据结构和算法的理论知识和应用技巧。

本书将围绕两个主题展开：

1. 高级编程技术：面向对象设计模式、模板元编程、泛型编程、并行计算等；

2. 数据结构和算法分析：基于时间复杂度、空间复杂度、asymptotic notation的分析方法及其应用、快速排序、二叉树、哈希表、堆栈、队列、链表等。

通过实践教学的方式，本书能够帮助读者加强对数据结构和算法的理解、掌握、运用，并提升个人的竞争力和技术水平。

本书适合对数据结构和算法感兴趣、有一定经验的读者阅读。但不限于此，读者在阅读过程中也可获得大量有益的信息。本书可作为数据结构和算法方面的参考书籍，也可以作为自我学习的数据结构和算法入门教材。

# 2.作者简介

郭嘉鸿，男，博士，现任华为软件研究院首席研究员、开源软件基金会创始理事。博士毕业于清华大学，获物理学博士学位，主要研究方向是分布式存储、云计算和区块链网络，并主持了多个重点项目。长期从事软件开发工作，参与过大大小小的商业和开源项目的研发，对技术的热情和追求永远存在。出版有专著《Java多线程权威指南》、《深入理解Java虚拟机（第二版）》、《企业架构模式》等。同时，他也是ACM/IEEE、USENIX等国际会议的主要演讲嘉宾。曾受邀担任ACM和IEEE计算机 Society会议联合主席，参加国内外高水平会议，撰写教育、培训、咨询等内容，曾主编英文版《数据结构与算法分析》。

# 3.购买链接

本书是《数据结构与算法分析》系列的第八本书。购买链接如下：

京东购买链接：https://item.jd.com/12749762.html?cu=true&utm_source=tongyongbao&utm_medium=tuiguang&utm_campaign=tongyongbao&utm_term=edm_jd

当当购买链接：http://product.dangdang.com/29006011.html

天猫购买链接：https://detail.tmall.com/item.htm?spm=a230r.1.14.12.6b565e9bsnyXqJ&id=584471492911&cm_id=140105335569ed55e27b&abbucket=10

亚马逊购买链接：https://www.amazon.cn/%E9%AB%98%E7%BA%A7%E7%BC%96%E7%A8%8B%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84-%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%AD%A6%E4%B9%A0-Guan-Jiaxin-ebook/dp/B07R6KRQFR/?_encoding=UTF8&pd_rd_w=nXJMh&pf_rd_p=c1704bf4-ccce-4f81-afbe-8fe77f4c5c40&pf_rd_r=9GWFZV2PZDVBKGKCSWC0&refRID=9GWFZV2PZDVBKGKCSWC0&th=1

# 4.目录
## 第一部分 导论
### 1.1 本书概要
### 1.2 数据结构和算法概览
### 1.3 相关背景介绍
### 1.4 内容安排
## 第二部分 C++基础
### 2.1 C++简介
### 2.2 语言基础
### 2.3 对象、类与面向对象
### 2.4 STL(标准模板库)
#### 2.4.1 容器概述
#### 2.4.2 序列式容器
#### 2.4.3 关联式容器
#### 2.4.4 STL算法概述
### 2.5 函数式编程
#### 2.5.1 函数概述
#### 2.5.2 Lambda表达式
#### 2.5.3 std::function
### 2.6 模板元编程
#### 2.6.1 模板概述
#### 2.6.2 模板类型参数
#### 2.6.3 偏特化与重载
### 2.7 内存管理
#### 2.7.1 new/delete运算符
#### 2.7.2 malloc()/free()函数
#### 2.7.3 new[]/delete[]运算符
#### 2.7.4 std::shared_ptr/std::unique_ptr
#### 2.7.5 RAII资源获取即初始化
#### 2.7.6 auto_ptr失效性
#### 2.7.7 代价低廉的动态内存分配器
### 2.8 异常处理
#### 2.8.1 异常处理概述
#### 2.8.2 try/catch语句
#### 2.8.3 throw关键字
#### 2.8.4 noexcept关键字
### 2.9 语法总结
### 2.10 扩展阅读
## 第三部分 高级编程技术
### 3.1 面向对象设计模式
#### 3.1.1 概念
#### 3.1.2 创建型设计模式
##### 3.1.2.1 Abstract Factory模式
##### 3.1.2.2 Builder模式
##### 3.1.2.3 Prototype模式
##### 3.1.2.4 Singleton模式
#### 3.1.3 结构型设计模式
##### 3.1.3.1 Adapter模式
##### 3.1.3.2 Bridge模式
##### 3.1.3.3 Composite模式
##### 3.1.3.4 Decorator模式
##### 3.1.3.5 Facade模式
##### 3.1.3.6 Flyweight模式
#### 3.1.4 行为型设计模式
##### 3.1.4.1 Chain of Responsibility模式
##### 3.1.4.2 Command模式
##### 3.1.4.3 Interpreter模式
##### 3.1.4.4 Iterator模式
##### 3.1.4.5 Mediator模式
##### 3.1.4.6 Memento模式
##### 3.1.4.7 Observer模式
##### 3.1.4.8 State模式
##### 3.1.4.9 Strategy模式
### 3.2 模板元编程
#### 3.2.1 定义
#### 3.2.2 基本原则
#### 3.2.3 历史及当前
#### 3.2.4 类型实例
#### 3.2.5 用例实例
#### 3.2.6 扩展阅读
### 3.3 泛型编程
#### 3.3.1 概念
#### 3.3.2 抽象数据的定义
#### 3.3.3 约束条件
#### 3.3.4 基于容器的泛型编程
##### 3.3.4.1 泛型向量
##### 3.3.4.2 泛型队列
##### 3.3.4.3 泛型优先队列
##### 3.3.4.4 泛型映射
##### 3.3.4.5 泛型集
#### 3.3.5 基于函数对象的泛型编程
##### 3.3.5.1 lambda表达式
##### 3.3.5.2 std::function
#### 3.3.6 扩展阅读
### 3.4 并行计算
#### 3.4.1 OpenMP
##### 3.4.1.1 指令集
##### 3.4.1.2 工作sharing方式
##### 3.4.1.3 parallel for loop
##### 3.4.1.4 数据共享方式
##### 3.4.1.5 数据依赖关系
##### 3.4.1.6 任务间依赖关系
#### 3.4.2 CUDA编程模型
##### 3.4.2.1 GPU编程模型
##### 3.4.2.2 编程例子
#### 3.4.3 MPI(Message Passing Interface)
##### 3.4.3.1 概念
##### 3.4.3.2 使用场景
##### 3.4.3.3 通信模型
##### 3.4.3.4 MPI程序示例