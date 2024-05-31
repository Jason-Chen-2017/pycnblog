# Metropolis-Hastings算法原理与代码实战案例讲解

## 1.背景介绍

### 1.1 概率模型与采样的重要性

在许多领域中,如统计学、机器学习、计算物理学等,我们经常需要从复杂的概率分布中进行采样。然而,这些概率分布通常是高维且复杂的,使得直接从中采样变得非常困难。这就需要一种高效的采样方法,能够从任意给定的概率分布中生成样本,这对于模拟、估计和推断都是非常重要的。

### 1.2 常见采样方法及其局限性

常见的采样方法包括拒绝采样(Rejection Sampling)、重要性采样(Importance Sampling)等。然而,这些方法在处理高维和复杂分布时往往效率低下或者失效。例如,拒绝采样需要一个包围目标分布的简单分布,而在高维情况下,找到这样的包围分布是非常困难的。

### 1.3 马尔可夫链蒙特卡罗方法(MCMC)

为了解决上述问题,马尔可夫链蒙特卡罗(Markov Chain Monte Carlo, MCMC)方法应运而生。MCMC方法通过构造一个马尔可夫链,使其稳态分布正是我们所需要的目标分布,从而可以有效地从目标分布中采样。Metropolis-Hastings算法就是MCMC方法中最著名和最通用的一种算法。

## 2.核心概念与联系

### 2.1 马尔可夫链(Markov Chain)

马尔可夫链是一个随机过程,它的下一个状态只依赖于当前状态,而与过去的状态无关。形式上,对于一个状态空间$\mathcal{S}$,如果一个随机过程$\{X_t\}$满足:

$$P(X_{t+1}=x|X_t=x_t,X_{t-1}=x_{t-1},\ldots,X_0=x_0)=P(X_{t+1}=x|X_t=x_t)$$

则称$\{X_t\}$是一个马尔可夫链。

### 2.2 稳态分布(Stationary Distribution)

对于一个马尔可夫链,如果存在一个分布$\pi$,使得无论初始状态如何,经过足够长的时间后,马尔可夫链的分布会收敛到$\pi$,则称$\pi$是该马尔可夫链的稳态分布。形式上:

$$\lim_{t\rightarrow\infty}P(X_t=x|X_0=x_0)=\pi(x),\quad\forall x_0\in\mathcal{S}$$

### 2.3 细致平稳条件(Detailed Balance Condition)

细致平稳条件是马尔可夫链达到稳态分布的一个充分条件。对于一个转移核$P(x\rightarrow y)$,如果对任意的$x,y\in\mathcal{S}$,都有:

$$\pi(x)P(x\rightarrow y)=\pi(y)P(y\rightarrow x)$$

则$\pi$就是该马尔可夫链的稳态分布。

### 2.4 Metropolis-Hastings算法

Metropolis-Hastings算法就是通过构造一个满足细致平稳条件的马尔可夫链,使其稳态分布正是我们所需要的目标分布$\pi(x)$,从而可以从$\pi(x)$中有效地采样。

该算法的核心思想是:从当前状态$x_t$出发,首先提出一个新的候选状态$y$,然后根据一定的接受率$\alpha(x_t,y)$决定是否接受这个候选状态。如果接受,则令$x_{t+1}=y$;否则,保持原状态,即$x_{t+1}=x_t$。通过这种方式,最终马尔可夫链的稳态分布就会收敛到目标分布$\pi(x)$。

## 3.核心算法原理具体操作步骤

Metropolis-Hastings算法的具体步骤如下:

1. 初始化:选择一个初始状态$x_0$,并给定目标分布$\pi(x)$和提议分布$q(x\rightarrow y)$。

2. 对于第$t$次迭代:
   
   a. 从提议分布$q(x_t\rightarrow\cdot)$中采样一个候选状态$y$。
   
   b. 计算接受率:
      $$\alpha(x_t,y)=\min\left\{1,\frac{\pi(y)q(y\rightarrow x_t)}{\pi(x_t)q(x_t\rightarrow y)}\right\}$$
   
   c. 从$U(0,1)$中采样一个随机数$u$。
   
   d. 如果$u\leq\alpha(x_t,y)$,则接受候选状态,令$x_{t+1}=y$;否则,保持原状态,令$x_{t+1}=x_t$。

3. 重复步骤2,直到马尔可夫链收敛。

4. 在收敛后,$x_t$的分布就近似于目标分布$\pi(x)$,从而可以从$\pi(x)$中采样。

需要注意的是,提议分布$q(x\rightarrow y)$的选择对算法的效率有很大影响。一个好的提议分布应该尽可能接近目标分布,这样可以提高接受率,从而加快收敛速度。

## 4.数学模型和公式详细讲解举例说明

### 4.1 细致平稳条件的证明

我们来证明,如果一个马尔可夫链的转移核$P(x\rightarrow y)$满足细致平稳条件,那么$\pi(x)$就是它的稳态分布。

证明:对于任意的$x,y\in\mathcal{S}$,我们有:

$$\begin{aligned}
\pi(y)&=\sum_{x'\in\mathcal{S}}\pi(x')P(x'\rightarrow y)\\
      &=\sum_{x'\in\mathcal{S}}\pi(y)P(y\rightarrow x')\quad\text{(由细致平稳条件)}\\
      &=\pi(y)\sum_{x'\in\mathcal{S}}P(y\rightarrow x')\\
      &=\pi(y)
\end{aligned}$$

因此,$\pi(y)$是马尔可夫链的稳态分布。

### 4.2 Metropolis-Hastings算法的细致平稳性

我们来证明,Metropolis-Hastings算法所构造的马尔可夫链满足细致平稳条件,从而其稳态分布就是目标分布$\pi(x)$。

对于算法中的转移核$P(x\rightarrow y)$,我们有:

$$\begin{aligned}
\pi(x)P(x\rightarrow y)&=\pi(x)\left[q(x\rightarrow y)\alpha(x,y)+r(x)\mathbb{I}(x=y)\right]\\
                      &=\pi(x)q(x\rightarrow y)\min\left\{1,\frac{\pi(y)q(y\rightarrow x)}{\pi(x)q(x\rightarrow y)}\right\}+\pi(x)r(x)\mathbb{I}(x=y)\\
                      &=\begin{cases}
                        \pi(y)q(y\rightarrow x), & \text{if }\pi(y)q(y\rightarrow x)\leq\pi(x)q(x\rightarrow y)\\
                        \pi(x)q(x\rightarrow y), & \text{if }\pi(y)q(y\rightarrow x)>\pi(x)q(x\rightarrow y)
                      \end{cases}\\
                      &=\pi(y)P(y\rightarrow x)
\end{aligned}$$

其中,$r(x)=1-\sum_{y\neq x}q(x\rightarrow y)$是保持原状态的概率,而$\mathbb{I}(\cdot)$是示性函数。

由此可见,Metropolis-Hastings算法所构造的马尔可夫链满足细致平稳条件,因此其稳态分布就是目标分布$\pi(x)$。

### 4.3 一维正态分布采样的例子

假设我们想从一个一维正态分布$\mathcal{N}(\mu,\sigma^2)$中采样,其概率密度函数为:

$$\pi(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

我们可以使用Metropolis-Hastings算法,选择提议分布为另一个正态分布$\mathcal{N}(x,\tau^2)$,其中$\tau$是一个超参数,控制了提议分布的方差。

在这种情况下,接受率$\alpha(x,y)$可以简化为:

$$\alpha(x,y)=\min\left\{1,\exp\left(-\frac{(y-\mu)^2-(x-\mu)^2}{2\sigma^2}\right)\right\}$$

通过不断地从提议分布中采样候选状态,并根据上述接受率决定是否接受,最终我们就可以从目标正态分布$\mathcal{N}(\mu,\sigma^2)$中获得样本。

## 5.项目实践:代码实例和详细解释说明

以下是一个使用Python实现Metropolis-Hastings算法从一维正态分布中采样的代码示例:

```python
import numpy as np
import matplotlib.pyplot as plt

# 目标分布的参数
mu = 0
sigma = 1

# 提议分布的参数
tau = 0.5

# 初始状态
x_init = 0

# 迭代次数
num_iter = 10000

# 初始化
x = x_init
samples = [x]

# Metropolis-Hastings算法
for i in range(num_iter):
    # 从提议分布中采样候选状态
    y = np.random.normal(x, tau)
    
    # 计算接受率
    alpha = min(1, np.exp(-(y - mu)**2 / (2 * sigma**2)) / np.exp(-(x - mu)**2 / (2 * sigma**2)))
    
    # 决定是否接受候选状态
    u = np.random.uniform()
    if u < alpha:
        x = y
    
    # 记录样本
    samples.append(x)

# 绘制样本分布
plt.figure(figsize=(8, 6))
plt.hist(samples, bins=50, density=True)
plt.plot(np.linspace(-5, 5, 100), 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(np.linspace(-5, 5, 100) - mu)**2 / (2*sigma**2)), 'r-', lw=2)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Samples from Normal Distribution')
plt.show()
```

代码解释:

1. 首先,我们设置目标正态分布$\mathcal{N}(0,1)$的参数$\mu=0,\sigma=1$,以及提议分布$\mathcal{N}(x,\tau^2)$的参数$\tau=0.5$。

2. 初始化初始状态$x_\text{init}=0$,迭代次数`num_iter=10000`,并创建一个空列表`samples`来存储采样的结果。

3. 进入Metropolis-Hastings算法的主循环:
   
   a. 从提议分布$\mathcal{N}(x,\tau^2)$中采样一个候选状态$y$。
   
   b. 计算接受率$\alpha(x,y)$,在本例中,由于目标分布和提议分布都是正态分布,接受率可以简化为:
      $$\alpha(x,y)=\min\left\{1,\exp\left(-\frac{(y-\mu)^2-(x-\mu)^2}{2\sigma^2}\right)\right\}$$
   
   c. 从$U(0,1)$中采样一个随机数$u$。
   
   d. 如果$u<\alpha(x,y)$,则接受候选状态$y$,令$x=y$;否则,保持原状态。
   
   e. 将当前状态$x$添加到`samples`列表中。

4. 循环结束后,`samples`列表中就存储了从目