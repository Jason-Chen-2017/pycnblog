
作者：禅与计算机程序设计艺术                    

# 1.简介
  

随着城市居民生活水平的提高，人们越来越注重节约能源、保护环境以及从事更环保的活动。在这个过程中，解决住宅、商贸中心等个别住户负荷过大的问题就成为很重要的课题之一。近年来，随着电力系统的不断升级和改造，与其说是利用交流电进行有效率的分布式发电，不如采用具有灵活性的微电网(MicroGrid)的方式实现这一目标。微电网的特点就是能够根据需要实时调度电力供应，使得每个用户都可以获得足够的可用电量。然而，微电网也带来了新的烦恼——如何选择合适的负荷抑制策略？一个好的负荷抑制策略能够减少热电偶然产生的负载波动，确保系统平稳运行，并且保持整体效率。本文将讨论负荷抑制策略对微电网负荷收敛速度的影响，并试图找出一种或几种能够有效提升微电网负荷收敛速度的方法。
# 2.概念定义及术语说明
## 2.1 什么是负荷？
所谓负荷(Load)，是指一定时间内的电能消耗。负荷又分为电力负荷(Power load)、流量负荷(Flow rate load)和功率因素负荷(Power factor load)。一般来说，正式定义为“是在一定时间、一定空间范围内，通过一个交流设备或电路传输的能量”，用符号$L=\frac{P}{t}$表示，其中$P$为功率，$t$为时间。

在当代电力系统中，电力是主要的生产力形式。由交流电系统或直流电系统生成的电能可用于热交换器等电能转换设备，进而通过电压差转变成其他形式的能量。例如，一台发电机的输入电压决定了其能耗，而输出电压则代表了单位时间内的实际能量输出能力。因此，电力负荷直接决定了电厂的输出功率，也是衡量电厂性能的主要指标之一。

## 2.2 什么是负荷抑制？
负荷抑制(Load shedding)是指在系统运行过程中，降低一定的负荷，以保证系统的持续稳定运行。负荷抑制有多种方式，包括物理上关闭某些电线，利用交流电控制系统上限电压降低负荷，或者采用其他方法排除过载现象。负荷抑制有助于减少不必要的能量浪费，同时还能避免过热现象、温差导致的频繁故障。常见的负荷抑制方法包括过载保护、功率分配和价格调整三类。

## 2.3 什么是微电网？
微电网(MicroGrid)是一种能够根据需要实时调度电力供应的电力网络，能够实现极大的经济规模化、节省运营成本。它将许多小型家庭用电设施连接在一起组成一个整体。微电网的应用范围不仅局限于电力领域，还涉及到其他产业领域，如空气净化、环保、农业、生态、金融、通信、医疗等。

## 2.4 什么是负荷收敛？
负荷收敛(Convergence)是指系统的实际负荷趋向于忽略或抑制对另一个负荷的响应，即系统中的其他负荷都处于饱和状态。负荷收敛意味着电能系统运行平稳，且不存在风险隐患。负荷收敛对电力系统的运行至关重要，因为存在的风险隐患都会影响到系统的运行。

负荷收敛可以简单理解为两个电路之间总负荷的比例，如果某一负荷远远大于另一负荷，则另一负荷会长期处于饱和状态，系统也无法正常工作。但是，由于存在各种各样的原因导致负荷收敛的现象，必须找到一种方法让负荷达到均衡状态，以维持系统的运行。

# 3.算法原理及具体操作步骤
## 3.1 模糊模型
### 3.1.1 描述
负荷抑制策略的研究可以划分为两步，第一步是建模，第二步是求解。建模阶段需根据实测数据生成模拟电网模型，该模型要能够反映不同负荷抑制策略对负荷收敛的影响，才能得出结论。

建模时，根据已有的数据，建立一个虚拟电力系统，该电力系统可能包含若干集散负荷点、电力枢纽、线路以及传动设备。每一组数据要对应于一个节点，节点之间的线路和设备反映了交流负荷形成的过程。

### 3.1.2 模型假设
#### 3.1.2.1 集散负荷项模型假设
在节点间的线路反映的是绝对值而不是相对值，所以该模型将所有节点视为绝对值。此外，假设只有一个集散负荷项，且集散负荷项出现在任意两个节点的某一条线上。因此，模型可写为：
$$p_i=p_{j}+\delta p \forall i, j$$

#### 3.1.2.2 次集聚负荷项模型假设
假设某个节点出现多条线路上的次集聚负荷项，且每个次集聚负荷项的损失率相同。例如，有一个节点有四条线路，分别连到了三个节点，假设线路a上的次集聚负荷项对整个节点的损失率为$\gamma_a$，线路b上的次集聚负荷项对整个节点的损失率为$\gamma_b$，线路c上的次集聚负荷项对整个节点的损失率为$\gamma_c$。那么该节点的次集聚负荷模型为：
$$p_i=p_{a}+\gamma_ap_{b}+\gamma_cp_{c}$$

#### 3.1.2.3 负荷抑制模型假设
在节点间可存在两种类型的负荷项：一是从属负荷项，二是自身负荷项。两种负荷项可以共存，但不能共享容量。所以，在满足节点间的线路容量约束条件下，如下两式成立：
$$p_i+\delta p \leq P_e,\quad\forall i$$
$$-\delta p\leq p_i\leq P_s,\quad\forall i$$

其中$P_e$是节点电压上限，$P_s$是节点电压下限。

#### 3.1.2.4 网络边界条件模型假设
假设模型外的线路或外部条件不发生变化，也就是说，没有随时间改变的负荷项。这类边界条件也被称为固定负荷项。

因此，模型可写为：
$$p_i=p_{j}+Q_it_i,\forall (i,j)\in E;\quad p_i=p_{j}+Q_et_i,\forall i\not\in V$$

其中，$E$是网络的边集合，$V$是网络的顶点集合，$Q_e$是各条边的容量，$t_e$是各条边的时长。

#### 3.1.2.5 参数估计模型假设
还需要考虑的是，线路和节点参数的估计误差。线路参数可以由两种方式估计：一种是直接用已知参数计算得到，另一种是估计参数在给定的容量范围内的最佳取值，如压力、电阻、电感等。节点参数的估计可以通过热力学方程或其它微观方程得到。

#### 3.1.2.6 参数估计模型假设
最后，还可以假设一些常用的特殊情况。例如，网络的分布式特性，如节点之间的距离、导线损耗、电压漂移等。

### 3.1.3 模型目标
#### 3.1.3.1 节点电压偏差目标函数
对于任意节点$i$, 有：
$$\Delta v_i^n=\frac{\sum_{\forall e}(Q_et_i)^2}{\sum_{\forall e}Q_et_i}-1$$

式中，$v_i^n$是节点$i$的仿真电压，$\Delta v_i^n$是节点$i$的仿真电压偏差。这里的目标函数是为了保证节点仿真电压的绝对误差，减小节点的电压异常行为。

#### 3.1.3.2 系统发电量目标函数
在微电网中，系统发电量往往是优化目标。因此，需要建立一个系统发电量的目标函数。这个目标函数的构造需要结合微电网本身的特点和负荷数据的测量值。

系统发电量目标函数可以写成如下形式：
$$f=\int_{0}^{T}\rho t_i^2\left(\dot{m}_i-\dot{m}_{ref}\right)^2dt_i+\alpha\cdot\left[\sum_{i=1}^Np_i^2-\mu^2\cdot N_p^2\right]$$

式中，$N_p$是发电站数量，$\rho$是电荷密度，$\mu$是电阻率，$\dot{m}_i$是节点$i$的真实平均流量。目标函数主要考虑以下方面：
* 最大化系统发电量，即保证节点的真实流量接近真实平均流量，以增加系统整体发电量；
* 抑制集散负荷，防止单节点爆炸效应；
* 限制发电站数量，避免功率集中爆炸。

#### 3.1.3.3 时域平衡目标函数
为了避免时域不平衡现象，需要加入时域平衡目标函数。该目标函数的构造和节点电压偏差目标函数类似，目的是减少相邻节点电压差异，消除交流潮流。

## 3.2 提案方案
### 3.2.1 使用随机优化算法
现实中，负荷抑制策略往往依赖于复杂的电力系统模型、复杂的优化算法和理论知识。基于此，我们建议采用随机优化算法作为负荷抑制策略的核心算法。

随机优化算法是一种基于概率论的优化算法。其主要思想是随机地初始化变量，然后迭代更新变量的值，以寻找全局最优解。

具体操作步骤如下：

1. 初始化参数：先确定每条线路的容量和时长，确定起始节点、终止节点及各节点的容量上下限，确定发电站数，确定网络的电压范围。
2. 生成初始解：随机生成一系列初始解，将这些初始解存储起来。
3. 对初始解评价：计算每一个初始解对应的系统发电量。
4. 更新目标函数：根据上一步的结果，设置一个目标函数，此时的目标函数可以是系统发电量的优化目标，也可以是其他指标的优化目标。
5. 执行迭代：重复执行以下操作：
    a. 从初始解集合中随机选取一个解作为当前解。
    b. 在当前解的基础上，依照一定概率调整某条线路的容量或时长。
    c. 根据新线路容量/时长调整线路参数，重新计算线路矩阵。
    d. 判断新线路容量/时长是否有明显好坏，如果有，记录下此次迭代的优良值；如果没有，判定此次迭代无变化。
    e. 如果有优良值，结束迭代，否则返回步骤b继续执行。
6. 返回最优解。
7. 综合以上，随机优化算法能够自动找到一种较优的负荷抑制策略，能够更加准确地预测负荷收敛速度。