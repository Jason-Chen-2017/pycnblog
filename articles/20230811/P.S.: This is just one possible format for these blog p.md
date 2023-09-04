
作者：禅与计算机程序设计艺术                    

# 1.简介
         

## （1）算法领域
概括来说，算法领域就是研究如何设计有效、高效的算法，从而解决计算机相关问题，提升计算机系统性能的领域。

算法领域研究的是对问题求解的方法，目的是为了找出一个最优解或近似解，这个方法可以采用不同的计算模型、搜索技术或优化技术来实现。

算法由五个方面组成：

1. 数据结构
2. 算法分析
3. 运用计算机模型和技术
4. 应用
5. 工程

## （2）算法分类
算法按照输入数据的规模及特征可以分为三种类型：

1. 确定性算法（Deterministic algorithm）——每个输入只对应唯一输出
2. 非确定性算法（Nondeterministic algorithm）——每一步的输出都可能不同
3. 随机算法（Randomized algorithm）——输出的分布并不确定，只有当输入相同时才会产生相同的输出

根据运算复杂度，算法又可以分为四类：

1. 简单算法（Simple algorithm）——时间复杂度在多项式时间内
2. 平摊分析算法（Amortized analysis algorithm）——先用较低的时间复杂度猜测后面的时间复杂度，再将猜测的结果和实际的运行时间比较
3. 线性时间算法（Linear time algorithm）——时间复杂度是n的多项式级别
4. 超多项式时间算法（Polynomial-time algorithm）——时间复杂度是n的指数级别

根据算法所需存储空间，算法又可以分为两类：

1. 单位存储空间算法（Space-efficient algorithm with unit storage space）——需要的存储空间随数据大小呈线性增长
2. 辅助存储空间算法（Space-efficient algorithm with auxiliary storage space）——算法需要额外的辅助存储空间来存放中间变量，但不能超过总体数据量的某个比例

根据算法是否能通过抽象化处理某些特定问题，算法又可以分为两类：

1. 非抽象算法（Concrete algorithm）——对于某些具体的问题，算法可以直接用具体的数据结构和算法实现
2. 抽象算法（Abstract algorithm）——算法本身并不是针对具体问题，而是借助某些形式语言来描述一系列操作，然后再通过具体的数据结构和算法进行求解

## （3）目标函数
目标函数定义了算法在满足约束条件下求解的问题的准则，目标函数常用于衡量算法的优劣。

目标函数主要包括以下几种：

1. 期望期望的目标函数（Expected expected objective function）——即总体期望值和期望值的乘积
2. 期望最大化目标函数（Expected maximization objective function）——即求得最大的期望值
3. 最小费用最大化目标函数（Minimizing the total cost of solving each subproblem to maximize overall profit）——即最小化子问题的整体代价，最大化整体利润