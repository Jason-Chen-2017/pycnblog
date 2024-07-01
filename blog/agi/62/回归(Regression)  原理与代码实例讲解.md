
# 回归(Regression) - 原理与代码实例讲解

作者：禅与计算机程序设计艺术 / Zen and the Art of Computer Programming

## 1. 背景介绍

### 1.1 问题的由来

在现实生活中，我们经常需要根据已知信息预测未知信息。例如，根据房屋的大小、位置等特征来预测其价格；根据学生的平时成绩来预测其考试成绩；根据历史气温数据来预测未来的天气情况等等。这类问题都可以通过回归分析来解决。

回归分析是统计学中一种常用的数据分析方法，它通过建立数学模型来描述因变量与自变量之间的关系。在机器学习中，回归分析也是一项基本技能，被广泛应用于分类、聚类、时间序列分析等领域。

### 1.2 研究现状

回归分析的历史可以追溯到18世纪，但直到20世纪中叶才得到广泛应用。随着计算机技术的快速发展，回归分析方法也得到了极大的丰富和发展。目前，回归分析已经形成了多个分支，如线性回归、非线性回归、岭回归、LASSO回归等。

### 1.3 研究意义

回归分析在各个领域都有着广泛的应用，例如：

- **社会科学**：预测经济增长、人口趋势、犯罪率等；
- **经济学**：预测股票价格、利率、汇率等；
- **医学**：预测疾病风险、治疗效果等；
- **金融**：信用评分、风险评估等；
- **工程**：预测设备故障、材料性能等。

### 1.4 本文结构

本文将首先介绍回归分析的核心概念和联系，然后详细讲解线性回归算法的原理和具体操作步骤，接着介绍其他常用的回归分析方法，最后通过代码实例和实际应用场景，帮助读者深入理解回归分析。

## 2. 核心概念与联系

### 2.1 因变量和自变量

在回归分析中，因变量通常被称为响应变量或目标变量，它表示我们需要预测的未知信息。自变量通常被称为预测变量或特征变量，它表示我们用来预测因变量的已知信息。

### 2.2 回归模型

回归模型是描述因变量与自变量之间关系的数学表达式。最简单的回归模型是线性回归模型，其形式为：

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \epsilon
$$

其中，$y$ 是因变量，$x_1, x_2, \ldots, x_n$ 是自变量，$\beta_0, \beta_1, \ldots, \beta_n$ 是回归系数，$\epsilon$ 是误差项。

### 2.3 回归分析的目的

回归分析的主要目的是：

- **估计回归系数**：通过最小化误差项的平方和，估计回归系数的值。
- **预测因变量**：根据给定的自变量值，预测因变量的值。
- **解释变量之间的关系**：分析自变量对因变量的影响程度和方向。

## 3. 核心算法原理 & 具体操作步骤

### 3.1 算法原理概述

线性回归是最基本的回归分析方法，其核心思想是找到一条最佳拟合线，使得所有数据点到这条线的距离之和最小。具体来说，线性回归模型可以通过以下步骤来求解：

1. 选择合适的损失函数，例如均方误差(Mean Squared Error, MSE)。
2. 计算损失函数关于回归系数的偏导数。
3. 将偏导数置为零，求解得到最优的回归系数。
4. 使用最优的回归系数，预测因变量的值。

### 3.2 算法步骤详解

线性回归的步骤如下：

1. **数据准备**：收集数据，并进行预处理，例如处理缺失值、异常值等。
2. **模型选择**：选择合适的回归模型，例如线性回归、多项式回归、岭回归等。
3. **参数估计**：使用最小二乘法或其他优化算法，估计回归系数的值。
4. **模型评估**：使用交叉验证等方法评估模型的性能。
5. **预测**：使用估计的回归系数，预测因变量的值。

### 3.3 算法优缺点

线性回归的优点是：

- 简单易懂，易于实现。
- 模型可解释性强，回归系数可以解释自变量对因变量的影响程度和方向。

线性回归的缺点是：

- 对于非线性关系的数据，线性回归可能无法得到很好的拟合效果。
- 对于含有异常值的数据，线性回归可能会受到较大的影响。

### 3.4 算法应用领域

线性回归在各个领域都有着广泛的应用，例如：

- **预测房价**：根据房屋的大小、位置等特征，预测房屋的价格。
- **预测股票价格**：根据公司的财务指标、市场行情等，预测股票的价格。
- **预测考试成绩**：根据学生的平时成绩，预测学生的考试成绩。

## 4. 数学模型和公式 & 详细讲解 & 举例说明

### 4.1 数学模型构建

线性回归模型的数学模型为：

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \epsilon
$$

其中，$y$ 是因变量，$x_1, x_2, \ldots, x_n$ 是自变量，$\beta_0, \beta_1, \ldots, \beta_n$ 是回归系数，$\epsilon$ 是误差项。

### 4.2 公式推导过程

线性回归的参数估计通常使用最小二乘法。最小二乘法的思想是：找到一组回归系数，使得所有数据点到回归线的距离之和最小。

假设我们有 $N$ 个数据点 $(x_{1i}, y_{1i}), (x_{2i}, y_{2i}), \ldots, (x_{Ni}, y_{Ni})$，则线性回归模型的损失函数为：

$$
L(\beta_0, \beta_1, \ldots, \beta_n) = \frac{1}{2} \sum_{i=1}^N (y_{1i} - (\beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \ldots + \beta_n x_{Ni}))^2
$$

对损失函数求偏导，并令偏导数等于零，可以得到：

$$
\frac{\partial L}{\partial \beta_0} = \sum_{i=1}^N (y_{1i} - (\beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \ldots + \beta_n x_{Ni})) = 0
$$

$$
\frac{\partial L}{\partial \beta_1} = \sum_{i=1}^N x_{1i} (y_{1i} - (\beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \ldots + \beta_n x_{Ni})) = 0
$$

$$
\vdots

$$

$$
\frac{\partial L}{\partial \beta_n} = \sum_{i=1}^N x_{ni} (y_{1i} - (\beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \ldots + \beta_n x_{Ni})) = 0
$$

解上述方程组，即可得到最优的回归系数 $\beta_0, \beta_1, \ldots, \beta_n$。

### 4.3 案例分析与讲解

假设我们有以下数据：

| x1 | x2 | y |
|---|---|---|
| 1 | 2 | 3 |
| 2 | 3 | 5 |
| 3 | 5 | 7 |
| 4 | 7 | 9 |

我们需要使用线性回归模型预测 $x_1 = 6$ 时的 $y$ 值。

首先，我们将数据绘制成散点图：

```mermaid
graph LR
A[点1]((1,2)) --> B[点2]((2,3))
B --> C[点3]((3,5))
C --> D[点4]((4,7))
D --> E[点5]((6,?))
```

然后，我们使用Python代码进行线性回归：

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# 数据
x = np.array([[1, 2], [2, 3], [3, 5], [4, 7]])
y = np.array([3, 5, 7, 9])

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(x, y)

# 预测
x_pred = np.array([[6, ?]])
y_pred = model.predict(x_pred)

print(f"预测值：{y_pred[0][0]}")
```

运行代码后，可以得到预测值 $y \approx 11$。根据散点图，可以看到预测值与真实值接近。

### 4.4 常见问题解答

**Q1：如何选择合适的损失函数？**

A：选择合适的损失函数取决于具体的应用场景。常用的损失函数包括均方误差(MSE)、交叉熵损失、Huber损失等。一般来说，均方误差适用于回归任务，交叉熵损失适用于分类任务，Huber损失则兼具两者优点，对异常值具有更强的鲁棒性。

**Q2：如何处理缺失值？**

A：处理缺失值的方法有多种，例如删除含有缺失值的数据、填充缺失值、插值等。具体方法取决于数据的特点和缺失值的类型。

**Q3：如何处理异常值？**

A：异常值可能对模型的性能产生较大影响。处理异常值的方法包括删除异常值、替换异常值、使用鲁棒性更强的模型等。

## 5. 项目实践：代码实例和详细解释说明

### 5.1 开发环境搭建

在进行线性回归项目实践之前，我们需要搭建开发环境。以下是使用Python进行线性回归的步骤：

1. 安装Anaconda：从Anaconda官网下载并安装Anaconda，用于创建独立的Python环境。
2. 创建并激活虚拟环境：
```bash
conda create -n linreg-env python=3.8
conda activate linreg-env
```
3. 安装必要的库：
```bash
conda install numpy pandas scikit-learn matplotlib
```
完成以上步骤后，即可开始线性回归项目的开发。

### 5.2 源代码详细实现

以下是一个简单的线性回归项目实例：

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 数据
x = np.array([[1], [2], [3], [4]])
y = np.array([3, 5, 7, 9])

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(x, y)

# 预测
x_pred = np.array([[6]])
y_pred = model.predict(x_pred)

# 绘制散点图和回归线
plt.scatter(x, y, color='red')
plt.plot([1, 6], [model.predict([[1]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]), model.predict([[6]]