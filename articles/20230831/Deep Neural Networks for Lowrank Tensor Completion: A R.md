
作者：禅与计算机程序设计艺术                    

# 1.简介
  

低秩张量Completion（LTC）是一种机器学习任务，其中，目标是利用少量训练数据去预测或填充大量缺失值的数据。当前，LTC研究主要集中在监督学习、无监督学习、半监督学习等几类方法上。本文根据相关工作的成果综述，提出了一种用于低秩张量Completion的神经网络模型——DEEP NEURAL NETWORKS FOR LOW-RANK TENSOR COMPLETION (DNTC)。本文首先介绍了低秩张量Completion的基本概念、术语，并对比了不同于传统张量Completion的特征，如不同于矩阵Completion的零元素以及自动特征选择。然后，提出了DNTC的结构设计和原理，重点分析了不同的正则化技术及其对DNTC的影响。之后，从优化算法角度出发，对DNTC进行了优化，并分析了不同优化算法的优缺点。最后，通过实验结果展示了DNTC的有效性和优势。

DNTC能够对缺失值进行建模，同时还可以利用大量的标签数据进行训练，因此可应用于实际场景。作者认为，DNTC有望成为解决现实世界中大规模低秩张量Completion的重要工具。此外，作者希望本文的研究能够为对低秩张量Completion领域更深入、全面的理解打下基础。
# 2.基本概念、术语说明
## 2.1 低秩张量Completion
低秩张量Completion（Low Rank Matrix Completion or LMC）是指利用少量数据，依据这些数据的统计特性，预测或填充大量缺失值的过程。它的特点是假设张量的每一个分量都由一组互相独立的基向量表示，即张量的秩较小。因而，它属于非负矩阵理论中的范畴，属于张量分析和统计学习的重要方向。近年来，低秩张量Completion已经得到越来越多的关注，其原因之一是其在工业领域的广泛应用。例如，手表上的个性化时钟显示，地震监测、医疗保健、财务风险评估、信息安全检测等都需要完成大量缺失值的预测任务。

对于给定的低秩张量X，假设其共分裂成两个子张量A和B，记作$X = AB$，则称B为低秩部分(Low rank part)，A为高秩部分(Higher rank part)。由于B中每个元素都由一组基向量表示，所以B又称为低秩分量(Low rank component)。对于任意一组向量$\boldsymbol{v}$，如果存在某个特征向量列组$\boldsymbol{W}=\{\boldsymbol{w}_1,\cdots,\boldsymbol{w}_r\}$使得$\|\boldsymbol{v}-AB\|_2^2=min_{\boldsymbol{w}\in \mathbb{R}^p}|(\boldsymbol{v}-AB)\cdot\boldsymbol{w}|, p=\dim(\mathrm{span}(\boldsymbol{W}))$,那么就可以说$\boldsymbol{v}$是由$\boldsymbol{W}$生成的。当$\|\boldsymbol{v}-AB\|=0$时，说明$\boldsymbol{v}$可以由张量A和B的元素完全恢复，并且该恢复过程可以在线性时间内完成。因此，低秩张量Completion就是寻找合适的基向量集，以便求解$\arg\min_{AB}\|\cdot\|_F^2$最小化问题。

为了保证高准确率，LMC往往采用大数据量、复杂数据模式的手段，例如高维信号处理、时序序列分析等。此外，LMC往往面临的是多种异常情况，如含有噪声、异质数据、缺失值、稀疏性等。因此，LMC也被称为病态张量Completion（ill-posed tensor completion），它是一个具有挑战性的问题。

低秩张量Completion既包括矩阵Completion，又包括张量Completion。对于矩阵Completion来说，每一个元素都是由一组互相独立的基向量表示，因此要求出的基向量的个数就等于元素的个数；而对于张量Completion来说，每一个分量都由一组基向量表示，因此要求出的基向量的个数就等于秩的大小。因此，两者之间存在着区别。

## 2.2 低秩张量
低秩张量（Low Rank Tensor）是具有着很好的低秩特性的张量。其满足三阶同构关系：即任取三个矩阵$(\mathcal{A},\mathcal{B},\mathcal{C})\in \mathbb{R}^{n\times d}\times \mathbb{R}^{d\times m}\times \mathbb{R}^{m\times r}$,其中n,m,d,r分别代表行、列、低秩维度、剩余维度。张量X具有低秩特性，当且仅当存在这样的矩阵$(\mathcal{A},\mathcal{B},\mathcal{C})$满足下列条件：

$$
\begin{align*}
&\text{For any }x\in X:\quad x=\mathcal{A}^\top\mathcal{B}^\top\mathcal{C}y\\
&\text{where }\mathcal{A}^\top\mathcal{B}^\top\mathcal{C}=(\underset{\mathcal{M}\in \mathbb{R}^{m\times n}}{\operatorname{vec}}\mathcal{M}\otimes (\underset{\mathcal{N}\in \mathbb{R}^{d\times m}}{\operatorname{vec}}\mathcal{N}\otimes \underset{\mathcal{P}\in \mathbb{R}^{r\times d}}{\operatorname{vec}}\mathcal{P}),\quad y=(\underset{\boldsymbol{a}\in \mathbb{R}^m}{\operatorname{col}}\boldsymbol{a}\otimes (\underset{\boldsymbol{b}\in \mathbb{R}^d}{\operatorname{col}}\boldsymbol{b}\otimes \underset{\boldsymbol{c}\in \mathbb{R}^r}{\operatorname{col}}\boldsymbol{c}))\quad\forall \mathcal{A},\mathcal{B},\mathcal{C}
\end{align*}
$$ 

这里，$vec(\mathcal{M})=(\mathcal{M}_{ij})$表示将矩阵$\mathcal{M}$转换为一维向量，$\otimes$表示张量积。对于$\mathcal{A}$、$\mathcal{B}$、$\mathcal{C}$，各自有m、d、r个元素，分别对应着第i个矩阵的列、第j个矩阵的行、第k个矩阵的元素。$vec$运算符把矩阵映射到列向量，再通过张量积实现不同矩阵之间的连接。因而，如果X具有低秩特性，那么X的每个元素都可以通过三元组$(\mathcal{A},\mathcal{B},\mathcal{C})$恢复。

低秩张量Completion试图寻找到一个张量$(\mathcal{A},\mathcal{B},\mathcal{C})$，使得任一张量X都满足条件：$X=\mathcal{A}^\top\mathcal{B}^\top\mathcal{C}y$.相应地，缺失值处的元素由基向量的贡献来决定。因此，对于给定的张量X，低秩张量Completion的目标是求解一组基向量$\{\mathcal{A},\mathcal{B},\mathcal{C}\}$，它们能够使得以下误差最小：

$$
\|\mathcal{A}^\top Y-\mathcal{B}^\top Z+\mathcal{C}^\top W\|_{F}^2+f_X\|Y-Z-WZ\|_{F}^2+\frac{\lambda}{2}\left\|\sum_{i=1}^d\left(U_i^{\top} U_i-I_m\right)\right\|_{F}^2+\frac{\mu}{2}\left\|\sum_{j=1}^d\left(V_j^{\top} V_j-I_n\right)\right\|_{F}^2+\frac{\kappa}{2}\left\|\sum_{k=1}^r\left(W_k^{\top} W_k-I_o\right)\right\|_{F}^2+\text{other terms}
$$

其中，$Y$和$Z$是观测到的矩阵，$W$是已知矩阵或已知的值。$\{U_i,V_j,W_k\}$是分解后的基向量集合。$\|\cdot \|_F^2$表示Frobenius norm，$I_m$、$I_n$和$I_o$分别表示单位矩阵。$f_X$是惩罚项，通常为均方误差，即$f_X=\|\cdot\|_F^2/\mid\mathcal{A}\mid^2$。其他的惩罚项通常用于控制模型复杂度，如Laplacian正则化、Hebbian自适应学习率、Dropout、共轭梯度法等。

## 2.3 深度神经网络（DNNs）
深度神经网络（Deep Neural Network, DNNs）是一种基于生物神经网络机理构建的机器学习模型，由多个简单层组成，并能够模拟大脑神经系统的多层感知过程。深度神经网络由输入层、隐藏层和输出层组成，其中隐藏层由多个神经元组成，接收输入、计算输出并传递信号到下一层。在本文所述的DNTC模型中，输入层接受原始数据，然后进入隐藏层，最后输出预测结果。DNTC模型使用多层感知器（MLPs）作为基本的结构单元。

## 2.4 正则化技术
正则化（Regularization）是一种用于防止过拟合的方法。正则化技术的目的是使模型在训练过程中不受过大的影响，从而使模型对测试数据集的效果好一些。正则化可以分为先天正则化和后天正则化。

先天正则化是指模型结构本身就带有一些限制条件，比如限制参数数量，或者限制参数的范围。比如对于线性回归模型来说，参数通常是一个系数矩阵，这种限制使得模型容易过拟合。这类正则化往往可以直接施加到损失函数上，如加入L2正则化项，使得系数的二范数（即Euclidean Norm）小于某个阈值。

后天正则化则是在训练过程中通过对代价函数增加惩罚项来实现的。由于神经网络通常较复杂，难以直接得到全局最优解，后天正则化往往是采用启发式的方式来对代价函数添加额外的惩罚项，以此来使得模型不至于过拟合。比如dropout、Lasso、Ridge、Elastic Net等正则化方式都属于后天正则化。

# 3. DEEP NEURAL NETWORKS FOR LOW-RANK TENSOR COMPLETION （DNTC)
## 3.1 模型结构设计
DNTC的模型结构可以分为编码器模块、中间层模块和解码器模块。

### 3.1.1 编码器模块
编码器模块主要由两部分组成，第一部分是一个多层感知器（MLP），用来编码低秩张量中的部分数据，第二部分是一个残差块，用来在特征层上进行特征融合。编码器的输出形式为$\hat{Y}=MLP(Y)$，其中Y是未缺失的数据。

### 3.1.2 中间层模块
中间层模块由多个ResNetBlock构成，每一个ResNetBlock由两个卷积层、BN层和ReLU激活层组成，第二个卷积层对原始输入Y做卷积。

$$
Y\to ResNetBlock(Y)+Y
$$ 

ResNetBlock是对残差网络中的基本模块的改进。由第一个卷积层、BN层和ReLU激活层组成，第二个卷积层对原始输入Y做卷积，然后与第一个卷积层的输出做元素相加。这样做的目的是保持原始输入的图像特性。

### 3.1.3 解码器模块
解码器模块由一个多层感知器（MLP）组成，它接受编码器模块的输出、低秩张量的秩r、剩余维度d、观测数据y和置信度矩阵$\Gamma$作为输入，输出预测张量。预测张量$\hat{X}$的计算方法如下：

$$
\hat{X}=softmax(\Gamma^\top MLP([Y,r,d]+y))
$$

其中$\Gamma$是一个置信度矩阵，它记录着每一个观测数据的置信度。$[Y,r,d]$是编码器模块的输出，$y$是观测数据。$softmax(\cdot)$是对置信度进行归一化，使得每一个元素的范围在$[0,1]$之间。

## 3.2 正则化设计
在模型训练过程中，需要加入正则化项以避免模型过拟合。本文设计了四种正则化方法，它们是L1正则化、L2正则化、范数约束、共轭梯度法。

### 3.2.1 L1正则化
L1正则化是对模型的权重施加直接的惩罚项，来减轻过拟合现象。

$$
loss+L_1*\Omega(W)=\frac{1}{N}\sum_{i=1}^N\{f_X(W\odot X_i+b)-y_i\}^2+\lambda * \sum_{l=1}^L \|W_l\|_1
$$ 

其中，$L_1$是超参数，用来调整惩罚项的强度，$\Omega(W)$表示W的L1范数。

### 3.2.2 L2正则化
L2正则化也是对模型的权重施加直接的惩罚项，但与L1正则化不同，L2正则化直接平方了权重，因此会产生更大的惩罚力度。

$$
loss+L_2*\Omega(W)=\frac{1}{N}\sum_{i=1}^N\{f_X(W\odot X_i+b)-y_i\}^2+\lambda * \sum_{l=1}^L \|W_l\|_2^2
$$ 

### 3.2.3 范数约束
范数约束的思想是让权重满足某一特定范数的限制，而不是让权重太大或太小。因此，通过限制模型的权重分布达到一定范围，可以防止过拟合。

### 3.2.4 共轭梯度法
共轭梯度法是一种基于拉格朗日乘子的迭代方法，用于解决凸优化问题。它通过引入拉格朗日乘子，来降低模型的复杂度。DNTC的模型是一个凸优化问题，因此可以用共轭梯度法来进行优化。

## 3.3 优化算法
优化算法是指模型参数更新的具体方案，它是模型训练过程的关键一步。本文对两种优化算法进行了比较，包括Adam和SGD。

### 3.3.1 Adam
Adam是最近提出的一种基于梯度下降的优化算法，它结合了动量法、RMSProp和Adagrad三种策略。Adam的特点是自适应调整学习速率，使得模型在训练初期快速接近最优解，并在后续迭代中逐步收敛。

$$
\begin{align*}
v_{t}&=\beta_1 v_{t-1}+(1-\beta_1)g_t \\
m_{t}&=\beta_2 m_{t-1}+(1-\beta_2)g_t^2 \\
\hat{v}_t&=\frac{v_{t}}{(1-\beta_1^t)} \\
\hat{m}_t&=\frac{m_{t}}{(1-\beta_2^t)}\quad\beta_1\neq 0,\beta_2\neq 0 \\
W' &= W - \alpha\frac{\hat{v}_t}{\sqrt{\hat{m}_t}+\epsilon}\\
\end{align*} 
$$

其中，$W'$表示模型的权重；$t$表示迭代次数；$v_t$、$m_t$、$\beta_1$、$\beta_2$是ADAM算法的参数；$g_t$是模型在当前迭代获得的梯度；$\alpha$是学习率；$\epsilon$是为了防止分母出现零的极小值。

### 3.3.2 SGD
随机梯度下降（Stochastic Gradient Descent，SGD）是一种典型的优化算法，它每次只考虑一个训练样本，并基于梯度下降更新参数。然而，SGD存在缺陷，因为它只能针对整个训练集进行一次迭代，导致模型无法学到局部最优解。

## 3.4 实验验证
本文使用MNIST手写数字数据集进行实验验证。实验设置是训练一个DNTC模型，对每一个训练样本，赋予不同的置信度，来模拟不同的异常情况，包括随机缺失、随机错误、稀疏缺失和重复缺失。为了评估模型的性能，本文计算了模型在测试集上的精度、召回率、F1分数以及运行时间。实验结果表明，DNTC模型在准确率、召回率和运行时间上均超过了最先进的低秩张量Completion方法。