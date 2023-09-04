
作者：禅与计算机程序设计艺术                    

# 1.简介
  

numpy（Numerical Python library）是一个用于科学计算和数据分析的Python库，它提供了矩阵运算、统计函数、随机生成等功能。该库提供了一个与Python标准库中的random模块相似的接口，即`np.random`。

此外，numpy还提供了一些数据集（数据结构），包括线性代数数据类型ndarray（N-dimensional array）、索引数组Index、日期时间数据类型Datetime64、结构化数组DataFramr和标签数组LabelBinarizer，以及很多其他有用的工具。本文主要讨论其中的随机数生成器`np.random.RandomState`，这是一种具有确定性的伪随机数生成器，可以生成伪随机序列。在生成随机数之前，通过设置种子值，可以使得相同的算法在每次运行时都产生相同的随机数序列。通过指定不同的种子值，可以得到不同的随机数序列。

np.random.seed()的用法如下：

    `np.random.seed(seed=None)`
    
其作用是设置一个种子值，所有由np.random生成的随机数都将根据这个种子值进行初始化。如果不给定参数，则默认使用当前时间作为种子值。同一个种子值会生成完全相同的随机数序列。因此，当需要重复生成相同的随机数序列时，可以通过设置相同的种子值来实现。

由于设置种子值的目的，对于相同的代码段多次运行可能得到不同的结果。但为了确保可重复性，应当固定好使用的种子值，或者通过某些方法保证每次运行产生不同的种子值。通常情况下，直接调用一次`np.random.seed()`就可以了。例如，以下程序会生成10个不同的随机数序列：

    ```
    import numpy as np
    
    for i in range(10):
        np.random.seed(i+1) # 设置不同种子值
        print(np.random.rand(5))
    ``` 

# 2.基本概念和术语
## 2.1 概率分布
随机变量X的概率分布，定义了X取某个值或某范围内的值的可能性。假设X是一个离散型随机变量，那么X的概率分布就是它各个取值出现的频率。假设X是一个连续型随机变量，那么X的概率密度函数（probability density function, PDF）描述了它取任意一个值时的概率。概率分布图示如下所示:

常见的概率分布有正态分布（normal distribution）、泊松分布（poisson distribution）、指数分布（exponential distribution）、均匀分布（uniform distribution）、卡方分布（chi-squared distribution）、二项分布（binomial distribution）、超几何分布（hypergeometric distribution）等。

## 2.2 概率质量函数
概率质量函数（PMF）或者叫分布函数（distribution function），描述的是离散型随机变量X的概率质量，即给定X的取值为x，则P(X = x)。如果是连续型随机变量，则概率密度函数（PDF）。

## 2.3 分布（Distribution）、样本（Sample）、抽样（Sampling）
分布（Distribution）：从总体中抽取样本，并赋予其真实含义的过程，通常是依据某种概率模型建立的。分布可以类比为面试题的答案，例如常见的正态分布，就是用来描述人的身高分布。

样本（Sample）：从分布中抽取的一组数据点，这些数据点按照一定顺序排列，组成了一组样本。例如，在天气预报领域，很多地区的天气状况都可以用一组样本来描述。

抽样（Sampling）：从总体中按一定规律选取样本。例如，调查了20个北京市的居民的年龄，就涉及到对20个人的年龄进行抽样。

## 2.4 参数（Parameter）、估计（Estimate）、样本（Sample）、推断（Inference）
参数（Parameter）：从概率分布中衍生出的变量，描述了分布的全部特征。例如，正态分布的参数有均值（mean）和标准差（standard deviation），均值为整个样本的中心位置，而标准差代表了分布的宽度。

估计（Estimate）：利用已知的数据来推断未知数据的过程，也就是给出一组数据，估计一个未知的或隐藏的参数。例如，我们知道每个人的身高和性别，可以用这两者来估计相应的人群的平均身高和性别。

样本（Sample）：在估计过程中，所用到的一组数据。例如，要估计一个人的身高，我们可能会用他的一组身高作为样本。

推断（Inference）：从样本数据中推导出关于总体的各种信息。例如，我们可能会从采集到的样本数据中发现身高偏低的人群，并得出结论说这个群体比其他群体更容易患肺癌。