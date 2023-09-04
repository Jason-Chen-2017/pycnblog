
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Probabilistic PCA (PPCA) 是一种降维技术，它可以将高维数据转换为低维数据，同时保持尽可能多的数据信息。相对于传统的主成分分析 (PCA)，PPCA 在降维的同时还可以保留原始数据的部分方差，并且在降维后保留原始数据的最大子空间。基于贝叶斯统计的 PPCA 技术有效地解决了 PCA 的两个主要缺点：即数据量不足时可能会出现“旁路”效应；另一方面，传统的 PCA 会丢失原始数据的一些方差信息，进而造成信息损失。PPCA 通过引入噪声信念，强化了原先已经存在的数据信息，从而更好地保留了原始数据信息。

传统的 PPCA 可以通过极大似然估计 (MLE) 方法进行求解，但由于计算复杂度较高，因此一般只适用于小型数据集。近年来，随着概率图模型 (PGM) 和变分推断方法的发展，基于 PGM 的 PPCA 有了广泛的应用。本文将详细介绍 PPCA 的原理、核心概念以及相关数学知识，并给出具体的代码实现。最后，我们还会给出 PPCA 的未来研究方向以及挑战。

# 2. 基本概念
## 2.1 二次判别分析
正如大多数机器学习领域一样，PPCA 使用二次判别分析 (QDA) 来对训练数据进行分类。假设有 $N$ 个样本点 $(x_1, y_1), \cdots,(x_N,y_N)$ ，其中每个样本点 $x_i$ 都属于标签类别 $y_i = c_j(j=1,\cdots,K)$ 。令 $\pi_k$ 为第 $k$ 个类的先验概率，则 QDA 模型的似然函数为：

$$
L(\theta, \phi; X, Y) = \prod_{i=1}^N p(y_i|x_i;\theta,\phi) 
\propto \prod_{k=1}^K \pi_k^{n_k} \prod_{i:y_i=k}\left[\frac{1}{\sqrt{(2\pi)^d |\Sigma_k|}}\exp\left(-\frac{1}{2}(x_i-\mu_k)^T\Sigma^{-1}_k(x_i-\mu_k)\right)\right] \\
\text{where } n_k=\sum_{i:y_i=k} 1,\quad d=D, \quad \Sigma_k=\frac{1}{n_k-1}(X_k-m_k)(X_k-m_k)^T
$$

其中，$\theta=[\pi_1,\cdots,\pi_K]$ 为参数向量，$D$ 为样本的维度。$\phi=[A_1,\cdots,A_K]$ 是类内协方差矩阵，$X=(x_1,\cdots,x_N)^T$, $Y=(y_1,\cdots,y_N)^T$ 为特征向量和标记向量，$X_k = \{ x_i : y_i=k \}$ 表示第 $k$ 个类的数据集，$m_k$ 为第 $k$ 个类的均值向量， $|\Sigma_k|$ 为第 $k$ 个类的总体协方差矩阵的逆。

为了求得最优解，通常采用交叉熵作为损失函数，其表达式为：

$$
\mathcal{C}(\theta, \phi)=-\log L(\theta, \phi; X, Y) = -\sum_{i=1}^N \log p(y_i|x_i;\theta,\phi)+const.
$$

这里的 const 为常数项，无需关注。

## 2.2 概率图模型
概率图模型（Probabilistic Graphical Model, PGM）是对已知联合分布 $p(x,y)$ （或称为马尔科夫网络）建模的理论基础。用图结构表示联合分布：节点 $X_i$ 表示随机变量 $x_i$，$Y_j$ 表示随机变量 $y_j$，存在隐含边 $(X_i,Y_j)$ 表示它们之间的依赖关系。根据链规则，联合分布可以表示为条件独立分布之积。例如，对于联合分布 $p(x,y,z)$，其可表述为：

$$
p(x,y,z)=p(x,z)p(y|x).
$$

隐含变量的个数越多，图结构就越复杂，且难以直接观测到。然而，考虑到很多情况下变量之间存在依赖关系，通过观测得到的变量可能无法完全决定结果，所以需要依靠变量间的依赖关系。

基于 PGM 的 PPCA 可以借助概率图模型中的生成模型和判别模型来刻画数据生成过程和结果之间的依赖关系。具体来说，生成模型刻画了隐变量与观测变量之间的联系，即希望隐变量的值能够控制观测变量的值。判别模型刻画隐变量和结果的关系，即希望利用观测到的变量信息来预测结果。因此，PPCA 通过构建一个因果图模型 (causal graph model) 来对模型进行建模。如下图所示：


上图展示了一个典型的因果图模型。图中的箭头表示隐变量 $Z$ 与观测变量 $X$ 或 $Y$ 的因果关系。灰色框中的箭头表示变量间的非因果关系。黑色圆圈代表观测变量，蓝色圈代表隐变量，白色矩形代表模型输出。生成过程由左边的方框表示，右边的箭头表示父子节点的对应关系，边上的权重表示边缘概率。判别过程由左下角的白色矩形表示，该矩形代表结果的概率分布。

## 2.3 变分推断
变分推断是从概率图模型中采样的一种策略。变分推断可以获得关于模型参数 $\theta$ 和 $\phi$ 的近似后验分布，而无需事先知道精确的条件分布。变分推断的目标是寻找一种方法，使得相对于真实模型 $q_\phi(Z\mid X,Y)$ ，拟合的模型 $q_{\tilde{\phi}}(Z\mid X,Y)$ 具有更好的近似期望。变分推断的一个具体实现是变分EM算法。

变分EM算法包括两步：第一步为E步，求解在当前参数下各个隐变量的期望，即求得后验分布 $p_{\theta}(Z\mid X,Y)$ ;第二步为M步，优化拟合的参数 $\theta$ 和 $\phi$ 以最小化交叉熵损失函数，即找到使得似然函数最大化的模型参数。

变分EM算法的基本想法是首先对模型进行初始化，然后重复执行以下两个步骤，直至收敛：

1. E步：固定模型参数，使用后验分布 $p_{\theta}(Z\mid X,Y)$ 对各个隐变量进行迭代，即更新每个隐变量的期望 $\bar{q}_\phi(Z_i\mid X,Y)$
2. M步：固定所有隐变量的期望，最大化似然函数 $L(\theta, \phi; X, Y)$ 。

## 2.4 信息论
信息论是数理统计学的一门基础学科。信息论研究的是信息的度量、编码与传输。在 PPCA 中，我们可以使用信息论中的理论来衡量高维数据在低维空间下的“有效性”。

信息的度量指标有熵和互信息。熵表示系统混乱程度的度量。给定数据集合 $S$ ，其熵定义为：

$$
H(S)=-\sum_{i=1}^N p(x_i) \log_2 p(x_i) 
$$

这里 $p(x_i)$ 表示第 $i$ 个样本点属于 $S$ 的概率。互信息表示 $X$ 和 $Y$ 之间的相关性。若 $X$ 和 $Y$ 独立，则其互信息为：

$$
I(X;Y)=0
$$

若 $X$ 和 $Y$ 相关，则其互信息为：

$$
I(X;Y)=\sum_{x\in\chi}p(x) \sum_{y\in\chi'}p(y) \log \frac{p(xy)}{p(x)p(y)}
$$

这里 $\chi$ 和 $\chi'$ 分别表示 $X$ 和 $Y$ 的取值集合。若 $X$ 和 $Y$ 的独立性很弱，则互信息 $I(X;Y)>0$ 。

PPCA 的目的是使得降维后的数据可以达到最大的有效信息提升。因此，我们需要衡量降维前后数据之间的互信息变化。更具体地说，PPCA 需要使得在给定数据集 $S$ 下，高维数据点 $x_i$ 和低维数据点 $\tilde{x}_i$ 的互信息尽可能大。具体的公式形式如下：

$$
\begin{align*}
I(X;Y)&=\sum_{x\in\chi}p(x) \sum_{y\in\chi'}p(y) \log \frac{p(xy)}{p(x)p(y)}\\&=\sum_{x\in\chi}p(x) \sum_{y\in\chi'}\tilde{p}_{z|yz}(y) \log \frac{\tilde{p}_{z|yz}(y)}{\tilde{p}_{z|x}(x)}\log \frac{p(xy)}{\tilde{p}_{z|x}(x)}\log \frac{\tilde{p}_{z|yz}(y)}{\tilde{p}_{z|x}(x)}\\
&\geq \sum_{x\in\chi}p(x) \sum_{y\in\chi'}\tilde{p}_{z|yz}(y) I(Z;Y) + H(Z)\\
\end{align*}
$$

这里 $\tilde{p}_{z|yz}(y)$ 是隐变量 $Z$ 在条件 $Y=y$ 时的值，$H(Z)$ 表示 $Z$ 的熵。等号右侧第一项表示不增加 $Z$ 信息的情况下，$X$ 和 $Y$ 之间的互信息增益；第二项表示增加了 $Z$ 的信息的情况下，$X$ 和 $Y$ 之间的互信息损失。当 $\tilde{p}_{z|yz}(y)<\tilde{p}_{z|x}(x)$ 时，等号右侧第一项大于等于等号右侧第二项。

# 3. PPCA 的核心原理
## 3.1 潜在变量表示
潜在变量（latent variable）是概率模型的重要组成部分。在 PPCA 中，我们认为隐变量 $Z$ 是维数比 $X$ 更少的变量，表示低维空间中的数据点。相应的，我们可以在 $Z$ 上施加约束条件，得到一个新的空间。这种约束条件就是潜在变量表示 (latent representation)。

设 $Z$ 的分布为 $q_\phi(Z\mid X)$ ，我们希望 $Z$ 和 $X$ 之间的映射是一个非线性的变换。这样的映射可以让潜在变量的表示符合实际需求，比如希望潜在变量中包含更多的信息、希望潜在变量具有某种自然的分离性等。具体地，潜在变量的表示可以由如下公式定义：

$$
Z=\phi(X;\theta) + \epsilon
$$

这里 $\phi(X;\theta)$ 表示隐变量 $Z$ 的表示，$\epsilon$ 表示噪声项，$\theta$ 表示参数向量。参数向量 $\theta$ 的大小决定了潜在变量的表示的复杂度。

为了找到一个合适的 $\phi$ 函数，我们可以通过最大化数据的对数似然函数（对数似然函数表示模型在给定数据集上计算的对数概率密度函数）来训练参数 $\theta$ 。训练过程中，我们将使用训练数据集 $X$ 来最大化似然函数，而优化过程则使用验证数据集 $V$ 。在优化过程中，我们希望使得参数 $\theta$ 和隐变量 $Z$ 的期望之间的距离尽可能小。也就是说，希望 $\mathbb{E}[q_{\phi}(Z\mid X)]$ 和 $q_{\phi}(Z\mid V)$ 尽可能相似。

## 3.2 混合高斯分布
PPCA 使用一个 $K$ 维的高斯分布 $p(Z;m_k, S_k)$ 来表示隐变量 $Z$ 的分布。$m_k$ 为第 $k$ 个类中心，$S_k$ 为协方差矩阵。如果数据集 $X$ 服从多个高斯分布，那么隐变量的分布也将服从这些分布的混合分布。假设数据集 $X$ 被分割为 $K$ 个类别，每个类别的中心向量为 $m_k$ ，协方差矩阵为 $S_k$ 。那么：

$$
Z\sim \sum_{k=1}^Kp(Z|X=x_i;\theta_k)=\sum_{k=1}^K \alpha_kp(Z|X=x_i;\theta_k)
$$

这里 $\alpha_k$ 为第 $k$ 个类的占比。根据潜在变量表示公式，可以得到：

$$
\begin{aligned} Z &= \phi(X;\theta) + \epsilon \\ &= \sum_{k=1}^K \alpha_k \phi(x_i;\theta_k) + \epsilon \\ & \sim \sum_{k=1}^K \alpha_k N(Z|m_k,\sigma_k^2I+\tau_kS_k) + \epsilon \end{aligned}
$$

其中，$\sigma_k$ 为方差超参数，$\tau_k$ 为平滑项。$\sigma_k$ 和 $\tau_k$ 的选择对最终的结果有着至关重要的作用。

## 3.3 激活函数
激活函数（activation function）用于将潜在变量的表示压缩到某一指定的范围，或者将其限制在某个范围内。PPCA 中的激活函数可以分为线性激活函数和非线性激活函数。线性激活函数可以认为是恒等映射，即 $g(z)=z$；而非线性激活函数往往可以把潜在变量的表示空间进行压缩，使其更易于处理。

例如，Sigmoid 函数是一种常用的非线性激活函数。其表达式为：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

激活函数也可以加入到模型的搭建中。具体来说，给定隐变量 $Z$ 的值，我们可以定义对应的潜在变量的分布 $q_\phi(Z\mid X)$ 。然后，再利用激活函数 $g$ 将 $Z$ 的分布限制在指定范围内：

$$
q_\phi(Z\mid X)=\int g(z) q_\phi(Z=z\mid X) dz
$$

这个技巧对 PPCA 的结果影响很大。例如，使用 Sigmoid 函数可以防止过大的 $Z$ 导致过拟合。此外，使用不同的激活函数还可以提高鲁棒性，使得 PPCA 可以适应不同的数据集。

## 3.4 协方差约束
协方差约束（covariance constraint）是 PPCA 的关键参数。它用来控制潜在变量的方差，并保证潜在变量之间的相关性。协方差矩阵越大，则潜在变量之间的相关性越强，反之亦然。

通常，协方差矩阵是通过方差贡献最大化得到的。在 PPCA 中，我们通过将协方差矩阵的分块结构化为 $K$ 个独立的协方差矩阵，来进行分块处理。具体来说，协方差矩阵可以分解为：

$$
S=\sum_{k=1}^K \tau_k (\sigma_k I + T_k S_k)
$$

其中，$\sigma_k$ 和 $\tau_k$ 为方差和平滑项，$T_k$ 是锤子矩阵。$T_k$ 的元素为：

$$
T_k=\frac{1}{\sqrt{d}}[e^{\frac{-\frac{1}{d}\left(\vec{x}-\vec{m}_k\right)^T \Sigma^{-1}_k\left(\vec{x}-\vec{m}_k\right)}{2}}]^{T},\forall k=1,\cdots,K, \forall x\in\chi
$$

这里 $\vec{x}$, $\vec{m}_k$ 为数据点 $x$ 和类别 $k$ 的中心向量，$\Sigma^{-1}_k$ 为第 $k$ 个类的逆协方差矩阵。此时，$S$ 的分块形式为：

$$
S=\sum_{k=1}^K \tau_k (\sigma_k I + \frac{1}{\sqrt{d}}T_k S_k)
$$

协方差矩阵的分块结构对结果的影响非常大。例如，增加 $\tau_k$ 可以减少潜在变量之间信息的共享，从而使得潜在变量之间更加独立；而减小 $\sigma_k$ 可以让潜在变量更加分散，进一步促进不同类的区分度。

## 3.5 噪声项
噪声项（noise term）是 PPCA 的另外一个重要参数。它可以起到一定程度上的平滑作用，并避免模型陷入局部最优解。

噪声项的引入可以通过为隐变量添加噪声来完成。噪声项可以表示为：

$$
\epsilon\sim N(0,\beta^{-1}I),\quad \beta>0
$$

通过设置较大的 $\beta$ 值，可以鼓励模型不要过于相信噪声项，进而得到稳定的结果。但是，过大的噪声会导致模型的不稳定性，容易欠拟合。

# 4. 具体算法
## 4.1 估计方差和偏移
在进行 PPCA 时，首先需要确定参数 $\theta$ 。一般来说，可以通过极大似然估计的方法估计参数。

具体地，令：

$$
\begin{equation*}
\begin{split}
    &q(Z|X,Y,\theta)=\int N(z|m(x),s(x))q(Z|X,Y,\theta)dz\\
    &=\int N(z|m(x),s(x))p_\theta(Z,Y=y|X=x)dz
\end{split}\\
\end{equation*}
$$

其中，$q(Z|X,Y,\theta)$ 表示隐变量 $Z$ 在给定数据点 $(X,Y,\theta)$ 的后验分布；$m(x)$ 为隐变量的均值函数，$s(x)$ 为协方差函数；$p_\theta(Z,Y=y|X=x)$ 为在给定输入 $X=x$ 的条件下，隐变量 $Z$ 和标记变量 $Y=y$ 的联合概率分布。

为了估计模型参数 $\theta$ ，我们可以使用 EM 算法或变分 EM 算法来最大化似然函数。具体地，对于每个样本点 $x_i$ ，可以得到关于参数 $\theta$ 的似然函数：

$$
\begin{equation*}
\begin{split}
    &\ln p_\theta(Z,Y=y_i|X=x_i)=\ln N(z_i|m(x_i),s(x_i))+\\
    &\ln \gamma(y_i|\alpha_{y_i}) + \ln p_\theta(Z,Y=y_i|X=x_i)-\\
    &\ln q_\phi(Z|X=x_i)-\ln \gamma(y_i|\alpha_{y_i}) \\ 
    &=\frac{1}{2}\left\{z_i^T(s(x_i)^{-1}+\beta^{-1}I)z_i+m(x_i)^T s(x_i)^{-1} z_i+\\
    &y_iz_i-(m(x_i)^Ts(x_i)^{-1} m(x_i)+\alpha_{y_i})\right\}+\beta^{-1}\ln p(x_i)
\end{split}\\
\end{equation*}
$$

这里 $\beta$ 为噪声项，$\alpha_k$ 为第 $k$ 个类的先验概率。$z_i$ 表示 $x_i$ 的潜在变量表示。通过最大化对数似然函数，可以得到参数 $\theta$ 的估计值。

然后，为了获得隐变量的估计值，可以使用变分推断的方法。首先，求得隐变量 $Z$ 的后验分布：

$$
q(Z|X,Y,\theta)=\int N(z|m(x),s(x))q(Z|X,Y,\theta)dz
$$

然后，对后验分布进行变分，得到新的分布 $q_{\tilde{\phi}}(Z\mid X,Y)$ 。

## 4.2 降维
PPCA 的降维操作比较简单。基本思路是：

1. 计算潜在变量的均值向量 $m_k$ 和协方差矩阵 $S_k$ 
2. 根据隐变量的后验分布 $q_{\tilde{\phi}}(Z\mid X,Y)$ 计算降维后的低维数据点 $\tilde{x}_i$ 。

具体地，对每一个类别 $k$ ，我们可以计算：

$$
m_k=\frac{1}{n_k}\sum_{i:y_i=k}z_i, \quad S_k=\frac{1}{n_k-1}\sum_{i:y_i=k}(z_i-m_k)(z_i-m_k)^T
$$

然后，根据后验分布，计算降维后的低维数据点 $\tilde{x}_i$ 。

## 4.3 可视化
在降维之后，我们可以将数据可视化，以便观察数据是否呈现出良好的特征结构。常用的可视化方式有核密度估计 (kernel density estimation, KDE) 和轮廓系数 (scree plot) 。

核密度估计表示数据的分布曲线。假设 $Z$ 服从高斯分布，我们可以计算每一个数据点 $x_i$ 的密度值 $p(Z\leq z_i)$ ，并绘制一条曲线来反映这些密度值的分布。

轮廓系数表示特征数量的度量。假设 $Z$ 的降维后的数据点构成矩阵 $\tilde{X}$ ，那么：

$$
R^2_k=\frac{1}{n-k}\sum_{i=k+1}^n \tilde{x}_i^T\tilde{x}_i
$$

上式表示第 $k$ 个类别的特征方差。如果所有的特征方差构成的曲线呈现出明显的聚集区域，则说明特征数量较少。否则，说明特征数量较多，需要进一步降维。

# 5. 代码实现
下面的例子展示了如何使用 PPCA 来降维和可视化二维数据集。

```python
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def log_likelihood(data, centers, covs):
    """
    Compute the log likelihood of each point in data under a mixture of normal distributions
    
    Args:
        data: [num_samples, num_features] array representing input points
        centers: list of mean vectors for each component
        covs: list of covariance matrices for each component
        
    Returns:
        log likelihood values for all samples under each component
    """

    def compute_log_likelihood(x, mu, sigma):
        """
        Compute the log likelihood of a single sample under a normal distribution
        
        Args:
            x: a single input point 
            mu: mean vector of the normal distribution
            sigma: covariance matrix of the normal distribution
            
        Returns:
            log likelihood value for the given input and parameters
        """
        inv_cov = np.linalg.inv(sigma)
        diff = x - mu
        return (-np.dot(diff, np.dot(inv_cov, diff))/2
                - np.log(np.linalg.det(sigma)/2/np.pi)**0.5)

    num_samples, _ = data.shape
    log_probs = []
    for center, cov in zip(centers, covs):
        prob = np.zeros([num_samples])
        norm_dist = multivariate_normal(mean=center, cov=cov)
        for i in range(num_samples):
            prob[i] = norm_dist.pdf(data[i])
        # add a small constant to avoid numerical instability when computing logarithms
        max_prob = prob.max()
        prob += np.spacing(1)*max_prob
        log_prob = np.log(prob)
        log_probs.append(log_prob)
    return log_probs


if __name__ == '__main__':
    # generate random dataset with two clusters
    data, labels = datasets.make_blobs(n_samples=1000, n_features=2, cluster_std=1., shuffle=True, random_state=None)

    # estimate parameters using maximum likelihood
    init_centers = [[data[labels==i].mean(axis=0)] for i in set(labels)]
    init_covs = [(data[labels==i]-init_centers[i][0]).T @ (data[labels==i]-init_centers[i][0])/len(data[labels==i])
                 for i in set(labels)]
    alpha = [float(len(data[labels==i]))/(len(data)) for i in set(labels)]
    params = {'alpha': alpha, 'centers': init_centers, 'covs': init_covs}

    # compute log likelihood for current initialization
    ll = sum([params['alpha'][i]*log_likelihood(data, params['centers'], params['covs'])[i].sum()\
              for i in range(len(set(labels)))])

    # perform alternating minimization for parameter optimization
    tol = 1e-3   # tolerance for convergence criterion
    max_iter = 1000  # maximum number of iterations
    em_iter = 10  # maximum number of iterations for one epoch of EM algorithm
    beta = 1e-5  # noise variance hyperparameter
    tau = 1     # smoothing hyperparameter
    converged = False
    iteration = 0
    while not converged and iteration < max_iter:
        prev_ll = ll

        # E step: update expectation of latent variables by approximating posterior
        exp_qz = {}
        exp_pz = {}
        for i in range(em_iter):
            old_exp_qz = exp_qz if len(exp_qz)!=0 else None

            # compute expected joint probability
            pz = np.array([[multivariate_normal.pdf(data[j], mean=params['centers'][l], cov=params['covs'][l]+beta*np.eye(2))]
                           for j in range(len(data)) for l in range(len(set(labels)))])
            qz = pz * ((1./beta)+(np.ones((len(data), len(set(labels))))/tau)*(1./len(set(labels))))
            qz /= qz.sum(axis=1)[:, np.newaxis]

            # approximate expected joint probability via softmax approximation
            temp = qz.copy()
            for l in range(len(set(labels))):
                temp[:, l] -= np.max(temp[:, l])
            exp_qz[i] = np.exp(temp)/(np.exp(temp).sum())
            
            # update expected prior probability based on new expected joint probability
            if old_exp_qz is None or np.linalg.norm(old_exp_qz[i]-exp_qz[i]) > tol**2:
                r_kl = np.concatenate(([0.], np.cumsum(np.array([params['alpha'][l] for l in range(len(set(labels)))]))[:-1]), axis=0)[labels]
                exp_pz[i] = exp_qz[i]/r_kl[:, np.newaxis]
            else:
                break
        
        # M step: optimize model parameters based on updated expectations of latent variables
        for k in range(len(set(labels))):
            weighted_data = data[labels==k]
            w_weights = exp_pz[list(exp_qz.keys())[-1]][:, k]*len(weighted_data)
            weights = w_weights/w_weights.sum()
            new_centers = weighted_data.T @ weights
            new_cov = np.cov(weighted_data.T, aweights=weights.reshape((-1,)), bias=False)
            params['centers'][k] = new_centers.ravel().tolist()
            params['covs'][k] = new_cov + beta*np.eye(len(new_cov))
        
        # calculate new log likelihood
        curr_ll = sum([params['alpha'][i]*log_likelihood(data, params['centers'], params['covs'])[i].sum()\
                       for i in range(len(set(labels)))])

        # check for convergence
        if abs(curr_ll-prev_ll) < tol:
            print('Convergence achieved after %d iterations!'%iteration)
            converged = True
        elif iteration >= max_iter-1:
            print('Maximum number of iterations reached!')
        else:
            print('Iteration: %d, Log Likelihood: %.3f'%(iteration, curr_ll))

        iteration += em_iter

    # visualize results
    _, ax = plt.subplots(figsize=(12, 8))
    colors = ['red', 'blue']
    for i in range(len(set(labels))):
        idx = np.where(labels == i)[0]
        ax.scatter(data[idx, 0], data[idx, 1], color=colors[i])

    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16)
    plt.show()

    # visualize low dimensional embedding
    transformed = []
    for i in range(len(set(labels))):
        means = params['centers'][i]
        covariances = params['covs'][i]
        dist = np.random.multivariate_normal(means, covariances, size=1000)
        transformed.extend(dist)
    reduced_dim_data = np.vstack(transformed[:len(data)])

    _, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(reduced_dim_data[:, 0], reduced_dim_data[:, 1], marker='.')

    ax.set_xlabel('Component 1', fontsize=16)
    ax.set_ylabel('Component 2', fontsize=16)
    plt.show()
```