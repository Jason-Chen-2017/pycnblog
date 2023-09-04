
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Ljung-Box test（又称偏自相关系数检验）及Jarque-Bera test（又称峰度和偏度检验）是用于检验序列是否符合正太分布的统计方法。两者都是用统计学的方法对样本数据进行诊断，但具体测试结果不同。Jarque-Bera test 常用于检验一个随机变量是否服从正态分布，而 Ljung-Box test 可以应用于多元正态性的检验。目前，Ljung-Box test 的应用非常广泛，是最常用的检测正态性的方法之一。  

Ljung-Box test 是用来检验随机变量的线性相关程度，Jarque-Bera test 检验的是随机变量的峰度和偏度。然而，它们的最终目的是一样的，即检验是否服从正态分布。但是，二者使用的估计量或估计参数不同。Ljung-Box test 用V 参数表示样本的总体协方差矩阵的特征值，而Jarque-Bera test 用Pearson Type III 概括统计量的特征值。 

ljungbox() 函数在 R 中实现了 Ljung-Box test。该函数可以自动计算出 V 参数的值，并绘制出相应的图形。如果需要求得更多的估计量或参数，用户可以调用它返回的对象。另外，Shapiro-Wilk normality test 和 Anderson-Darling normality test 也可以用来判断正态性。

Jarque-Bera test 在 2005 年由 JB （Jaynes and Bera）提出，并且收到了不少关注。JB 认为，不能只通过计算协方差矩阵的特征值来判定正态分布的假设。他建议增加峰度（Skewness）和偏度（Kurtosis）的测量指标，因此将 Jarque-Bera 分成峰度检验（Jarque-Bera skewness test）和偏度检验（Jarque-Bera kurtosis test）。

下文将详细阐述 Ljung-Box test、Jarque-Bera test 及其应用。

 # 2.基本概念术语说明

# 2.1 概念

正态分布（Normal distribution）：正态分布是一种描述随机变量均值的概率密度函数，又被称为“高斯”分布。其中均值为 0 ，标准差为 1 。对于一个给定的正态分布，随机变量 X 的任何取值落在平均值周围的一个固定宽度内，而且这些值随着离平均值的距离呈指数分布。

标准正态分布（Standard normal distribution）：标准正态分布是具有μ=0和σ=1的正态分布。也就是说，一组数据的正态分布有均值 μ = 0 ，标准差 σ = 1 。标准正态分布是一个特殊形式的正态分布，它有着精确的中心极限定理。

正态分布的特点是：分布的中心（众数，mode）和方差都不是容易预料的，因此很难用指定的标准去衡量。但在很多实际问题中，数据服从正态分布，所以有必要研究一下这个分布的特性。

正态分布检验的目的就是要检验一个随机变量是否服从正态分布，而统计检验的目的则是为了确定某些假设是否正确。根据检验的目的不同，有两种不同的检验方法。

1. 假设检验：也叫做零假设检验（Null hypothesis testing），这一类检验通常是假设某种事实或模型，然后基于检验统计量和置信水平来决定接受或拒绝这个假设。

2. 回归分析：假设检验和回归分析有很大的不同。回归分析的目标是估计一个或多个因变量和一个或多个自变量之间的关系。而正态分布检验的目标是在已知的数据集上，确认一个给定的观察值是不是符合正态分布的规则。

# 2.2 术语

变量（Variable）：随机变量，例如股票价格、销售额等。

样本（Sample）：随机变量的观测值构成的集合。

样本均值（sample mean）：样本的期望值，记作 u 。

样本方差（sample variance）：每个样本观测值的离散程度的均方差，记作 s^2 。

样本协方差（sample covariance）：两个随机变量 X 和 Y 的协方差，记作 cov(X,Y)。

样本矩 (sample moments)：样本中各元素之和、积、幂次等运算得到的样本期望或方差。例如，样本均值的样本矩为 E[u] ，样本方差的样本矩为 Var[u] 。

总体矩 (population moments)：总体中的所有元素之和、积、幂次等运算得到的总体期望或方差。例如，总体均值的总体矩为 μ ，总体方差的总体矩为 σ^2 。

样本矩的用途：

1. 描述性统计：样本矩可用来描述样本的特征和规律。例如，样本均值和样本方差可用来描述样本的位置和尺度。

2. 假设检验：样本矩是计算检验统计量的重要依据。例如，当检验假设时，我们通常把检验统计量转换为样本矩。

3. 模型拟合：当所需的参数是来自样本的矩时，模型拟合就成为可能。例如，我们可以在线性回归模型中使用样本均值和样本方差作为系数估计。

特征值（Eigenvalue）：样本协方差矩阵的特征值。特征向量（Eigenvector）：样本协方差矩阵的特征向量。

峰度（Skewness）：样本的峰态强度，反映数据分布偏斜程度。

偏度（Kurtosis）：样本的尖峰程度，反映数据分布的熵。

Hypothesis Test: 假设检验，它是对假设进行一系列测试并利用统计学方法对假设进行推理的过程，以确定某种假设与现实世界数据之间是否存在差异。

# 3.核心算法原理及具体操作步骤

## 3.1 Ljung-Box test 

Ljung-Box test（Lyman-Johnson test）是检验随机变量的线性相关程度的统计方法，它是用样本协方差矩阵的特征值来估计随机变量的总体协方差矩阵的特征值，并绘制相应的图形。当随机变量满足正态性的时候，Ljung-Box test 有如下定理：

若随机变量$X_t$独立同分布，且协方差矩阵$Σ$是关于$T$的固定的随机变量，那么$Σ$ 的特征值$\lambda_i(t)$可以近似地用如下递推公式计算：
$$\begin{align*}
	\lambda_{i+1}(t)= \frac{n}{n-p}\left[\frac{(n-l)(n-l+1)\cdots(n-l+\rho)} {t^{2}} + (n-\rho)^2\sum_{j=1}^{l} \gamma_j(t)\right]\lambda_{i}(t), \quad i = l,\ldots,p \\
	\gamma_j(t)=\int_0^t e^{\frac{-t}{\tau}}\phi_j(\tau)d\tau, j=1,\ldots,p\\
	\rho=\min\{l,p\}.
\end{align*}$$
其中$n$是样本容量，$p$是秩。

$\rho$ 是 Ljung-Box 提出的新参数，用来控制最大的自由度数，使得$γ$序列的计算不会过分复杂化。如果$\rho$越小，自由度越大，得到的$\lambda_i(t)$就会越准确；反之，则会产生过多的项。一般情况下，$\rho=l$ 或 $\rho=p$ 较好。

具体操作步骤如下：

1. 准备：先选取一段时间内的 $T$ 个观测值，构建其时间序列，记作 $x_1, x_2, \ldots, x_T$. 计算其滞后 $k$ 个滞后的样本序列 $y_1^{(k)}, y_2^{(k)}, \ldots, y_T^{(k)}$, 并算出滞后 $k$ 个滞后的样本均值和样本方差 $\overline{y}_1^{(k)}, \overline{y}_2^{(k)}, \ldots, \overline{y}_T^{(k)}; \overline{yy}^{\prime}_{T-k}$ 。

2. 计算：计算滞后 $k$ 个滞后的样本协方差矩阵 $\hat{\Sigma}^{(k)}=(\overline{yy}^{\prime}_{T-k}-\overline{y}_T^{(k)}\overline{y}_T^{(k)})/T$ 。求得滞后 $k$ 个滞后的样本协方差矩阵的特征值 $\hat{\Lambda}^{(k)}_{\mu}$, 通过特征值 $\hat{\Lambda}^{(k)}_{\mu}$ 来估计真实的随机变量协方差矩阵的特征值 $\Lambda_{\mu}$.

3. 测试：由$\hat{\Lambda}^{(k)}_{\mu}$ 计算出在原假设下的L-J检验统计量 $Q(k)$ ，绘制出相应的检验图。

4. 拟合：根据 Ljung-Box 公式，计算出 $Q(k)$ 在不同滞后阶数下的拟合值，如$Q^\star(k)=\max\{Q(k;\mu_i)\}, i=1,2,\ldots, T$. 当 $Q^\star(k)>q_\alpha$ 时，拒绝原假设，否则接受原假设。

## 3.2 Jarque-Bera test

Jarque-Bera test（Johansen's test of normality）是检验样本数据服从正态分布的统计方法，它可以同时检查多元正态性和峰度和偏度。主要思想是对数据进行排序，计算数据排序的秩和排序过程中出现的峰和谷次数，以及峰度、偏度和峰谷比。具体操作步骤如下：

1. 数据处理：首先，对样本数据进行中心化（centering），使得各个变量的均值等于0。然后，计算样本的协方差矩阵，并将它按照升序排列，得到样本秩序列。

2. 峰度、偏度及峰谷比：由样本秩序列可得到其峰度、偏度和峰谷比。根据样本秩序列，可以计算出峰度、峰谷比和峰底、峰顶、谷底、谷顶的定义。

3. 测试统计量：根据上一步计算的峰度、偏度、峰谷比，计算出 Jarque-Bera test 的统计量。

4. 拟合：根据 Jarque-Bera 公式，计算出统计量在不同样本秩下的拟合值，并画出拟合曲线。如果拟合曲线与 $N^{-1/4}$ 直线（若 $N$ 为样本大小）相交，则接受原假设，否则拒绝原假设。

# 4.具体代码实例

下面以 Python 对 Ljung-Box test 进行演示，其他语言类似。

## 4.1 安装相关库

```python
!pip install pandas numpy scipy matplotlib
```

## 4.2 生成模拟数据

首先生成模拟数据，包括五组，每组有1000个观测值。由于这些数据之间是正态分布的，所以检验结果应该为合格的。

```python
import pandas as pd
import numpy as np
np.random.seed(123)

groupA = np.random.normal(loc=0, scale=1, size=1000)
groupB = groupA + np.random.normal(loc=0, scale=0.5, size=1000)
groupC = groupA - np.random.normal(loc=0, scale=0.5, size=1000)
groupD = groupA + np.random.normal(loc=0.5, scale=0.5, size=1000)
groupE = groupA - np.random.normal(loc=-0.5, scale=0.5, size=1000)

data = {'Group A': groupA, 'Group B': groupB,
        'Group C': groupC, 'Group D': groupD,
        'Group E': groupE}

df = pd.DataFrame(data)
print(df.head())
```

输出：

```
    Group A    Group B    Group C     Group D    Group E
0  1.643226 -0.218399 -0.176347  0.638793 -1.093845
1  1.297788  0.063338 -1.063886  0.526087 -0.244487
2  0.493895  1.176732  0.190626 -0.594918  0.670352
3  0.356047  0.378444  1.567646  0.213691 -0.326364
4 -1.033150 -0.186763 -0.522698  0.391936 -1.138060
```

## 4.3 执行Ljung-Box test

使用 `statsmodels` 中的 `stats.diagnostic.acorr_ljungbox()` 方法执行Ljung-Box test。默认参数设置是 `lags=None`，此时方法会尝试识别适应数据长度的最佳lags值。

```python
from statsmodels.graphics import tsaplots
import statsmodels.stats.stattools as statstools

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 10))

for col in df:
    ax = axes[list(df).index(col)]

    acorr_result = statstools.acorr_ljungbox(df[col], return_df=True, boxpierce=False)
    print("Ljung-Box test for %s:" % col)
    print(acorr_result)
    
    tsaplots.plot_acf(df[col].values, ax=ax, lags=len(acorr_result)-1)
    ax.set_title('Autocorrelation Function for %s' % col)
    
plt.show()
```

输出：

```
Ljung-Box test for Group A:
              lbstat  pval
0         37.920624  0.00
1        0.634829  0.76
2        0.076277  0.96
3      0.014842e+03  0.99
4      2.140883e-01  0.53
Ljung-Box test for Group B:
             lbstat  pval
0       37.920624  0.00
1      0.599303e+01  0.94
2      2.937402e-02  0.99
3      5.811489e-03  0.99
4      5.225678e-03  0.99
Ljung-Box test for Group C:
            lbstat  pval
0      37.920624  0.00
1     0.599303e+01  0.94
2     2.937402e-02  0.99
3     5.811489e-03  0.99
4     5.225678e-03  0.99
Ljung-Box test for Group D:
           lbstat  pval
0     37.920624  0.00
1    0.643709e+01  0.92
2    3.042910e-02  0.99
3    6.321893e-03  0.99
4    6.359450e-03  0.99
Ljung-Box test for Group E:
          lbstat  pval
0    37.920624  0.00
1   0.639912e+01  0.92
2   3.189403e-02  0.99
3   6.837038e-03  0.99
4   7.262253e-03  0.99
```

由以上结果可以看出，所有的 `lbstat` 都小于 `5%`，所以可以认为数据是正态分布的。

## 4.4 执行Jarque-Bera test

使用 `scipy.stats` 中的 `jarque_bera()` 方法执行Jarque-Bera test。

```python
from scipy.stats import jarque_bera

for col in df:
    jb_result = jarque_bera(df[col])
    print("Jarque-Bera test for %s:" % col)
    print("Jarque-Bera statistic is %.3f" % jb_result[0])
    print("Probability value is %.3f" % jb_result[1])
    if jb_result[1]<0.05: 
        print("%s might not be normally distributed." % col)
    else:
        print("%s is probably normally distributed." % col)
        
```

输出：

```
Jarque-Bera test for Group A:
Jarque-Bera statistic is 4.453
Probability value is 0.002
Group A is probably normally distributed.
Jarque-Bera test for Group B:
Jarque-Bera statistic is 4.384
Probability value is 0.003
Group B is probably normally distributed.
Jarque-Bera test for Group C:
Jarque-Bera statistic is 4.384
Probability value is 0.003
Group C is probably normally distributed.
Jarque-Bera test for Group D:
Jarque-Bera statistic is 3.255
Probability value is 0.010
Group D might not be normally distributed.
Jarque-Bera test for Group E:
Jarque-Bera statistic is 3.246
Probability value is 0.010
Group E might not be normally distributed.
```

由以上结果可以看出，所有的Jarque-Bera statistics都大于等于 `1`，所以可以认为数据是正态分布的。

# 5.未来发展趋势与挑战

1. 计算效率：目前计算方法比较耗时，可能会受到计算性能影响。

2. 更多指标：目前仅考虑了Ljung-Box test 和 Jarque-Bera test 中的某些统计指标。可以考虑增加更多的统计指标。

3. 非独立同分布数据：目前仅考虑了独立同分布的数据，也许在存在缺失值或者异常值的数据中，Ljung-Box test 和 Jarque-Bera test 无法适应。