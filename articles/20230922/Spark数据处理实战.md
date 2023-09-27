
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Spark是一个快速、通用、基于内存的集群计算系统。它可以用于机器学习、流处理、网页搜索、图形分析等领域。本书旨在帮助读者了解Spark框架的基础知识，包括RDD（弹性分布式数据集）、DataFrames（数据帧）、SQL（结构化查询语言）、MLlib（机器学习库）、GraphX（图形处理）、Streaming（流处理）。同时也将介绍一些在实际工作中遇到的最常见的问题以及解决方案。最后还会提供一些拓展阅读建议，帮助读者进一步学习Spark。作者：陈浩博，中科院信息所研究员，Apache Spark PMC成员，曾任职于微软亚洲研究院、新浪微博、知乎等互联网公司。
# 2.版本要求
本书的正文结构、插图排版都已经达到了较高的标准，力求易懂易读。而为了保持最新和正确的知识，我推荐读者使用最新版本的Spark(2.4.0)及其相关工具进行实践。如果读者的开发环境没有配置好的话，可以使用云服务或本地虚拟机的方式尝试。我在撰写本书时使用的Spark版本为2.4.0。
# 3.读者对象
本书面向具有一定编程基础和Spark使用经验的工程师、架构师、数据科学家等。对于一些经常接触但可能不熟悉Spark的读者，本书也可以作为入门参考。
# 4.前言
《Spark数据处理实战》是一本开源图书，基于Apache Spark 2.4.0版本编写。本书的主要目的就是通过案例驱动的方式，全面介绍Spark的各种功能模块和使用技巧。为了更好地传播Spark技术，作者采取了以下措施：
- 将Spark源码贴近原理，尽量详细阐述每个组件的设计思路和实现原理；
- 在每个章节的结尾给出练习题，读者可以自行完成相关功能的代码实现；
- 每个章节的后面都附有参考文献，方便读者了解Spark的发展历史、优秀案例，提升对Spark技术的理解；
- 本书的每一节都包含上演操作Spark的示例代码，便于读者快速上手体验Spark并理解原理。
# 5.如何购买
本书已经开源在GitHub上，欢迎广大读者提Issue或者PR参与到本书的翻译或编写过程中。如果你希望获得定制的付费版权，或者需要关注本书的更新动态，欢迎联系作者微信号hongbinchao，备注“Spark实战”。

# 目录
* [第1章 Apache Spark概述](chapter1/README.md)
  * [1.1 Apache Spark简介](chapter1/section1.1.md)
  * [1.2 核心概念](chapter1/section1.2.md)
  * [1.3 发展历史与特点](chapter1/section1.3.md)
  * [1.4 安装部署与配置](chapter1/section1.4.md)
  * [1.5 集群管理器配置](chapter1/section1.5.md)
* [第2章 数据处理的核心机制——RDD](chapter2/README.md)
  * [2.1 RDD简介](chapter2/section2.1.md)
  * [2.2 RDD操作详解](chapter2/section2.2.md)
  * [2.3 RDD持久化与缓存](chapter2/section2.3.md)
  * [2.4 RDD依赖关系与宽窄依赖](chapter2/section2.4.md)
  * [2.5 文件存储与分区](chapter2/section2.5.md)
  * [2.6 RDD转DataFrame与DataFrame操作](chapter2/section2.6.md)
  * [2.7 Datasets简介](chapter2/section2.7.md)
* [第3章 DataFrame API与SQL](chapter3/README.md)
  * [3.1 DataFrame概述](chapter3/section3.1.md)
  * [3.2 SQL基础语法](chapter3/section3.2.md)
  * [3.3 DataFrame操作详解](chapter3/section3.3.md)
  * [3.4 外部数据源与表管理](chapter3/section3.4.md)
  * [3.5 性能调优](chapter3/section3.5.md)
  * [3.6 使用场景](chapter3/section3.6.md)
* [第4章 机器学习库MLlib](chapter4/README.md)
  * [4.1 MLlib概述](chapter4/section4.1.md)
  * [4.2 模型训练与预测](chapter4/section4.2.md)
  * [4.3 特征抽取与转换](chapter4/section4.3.md)
  * [4.4 文本处理与分类](chapter4/section4.4.md)
  * [4.5 模型评估与调优](chapter4/section4.5.md)
  * [4.6 使用场景](chapter4/section4.6.md)
* [第5章 GraphX API与图论](chapter5/README.md)
  * [5.1 GraphX简介](chapter5/section5.1.md)
  * [5.2 GraphX操作详解](chapter5/section5.2.md)
  * [5.3 GraphX性能优化](chapter5/section5.3.md)
  * [5.4 使用场景](chapter5/section5.4.md)
* [第6章 流处理API——Structured Streaming](chapter6/README.md)
  * [6.1 Structured Streaming简介](chapter6/section6.1.md)
  * [6.2 Structured Streaming原理与特性](chapter6/section6.2.md)
  * [6.3 基于Schema的数据类型化](chapter6/section6.3.md)
  * [6.4 Structured Streaming操作详解](chapter6/section6.4.md)
  * [6.5 时间窗口与水位线](chapter6/section6.5.md)
  * [6.6 输出模式](chapter6/section6.6.md)
  * [6.7 状态管理与容错](chapter6/section6.7.md)
  * [6.8 使用场景](chapter6/section6.8.md)
* [第7章 分布式计算](chapter7/README.md)
  * [7.1 分布式架构概览](chapter7/section7.1.md)
  * [7.2 MapReduce编程模型](chapter7/section7.2.md)
  * [7.3 Spark编程模型](chapter7/section7.3.md)
  * [7.4 Spark应用的部署方式](chapter7/section7.4.md)
  * [7.5 Spark的资源管理](chapter7/section7.5.md)
  * [7.6 调试、监控与调优](chapter7/section7.6.md)
* [第8章 扩展阅读](chapter8/README.md)