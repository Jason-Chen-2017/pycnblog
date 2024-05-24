
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


支持向量机（Support Vector Machine，SVM）是一种二类分类模型，属于监督学习方法，主要用于分类和回归问题。它利用训练数据构建一个超平面或超曲面将数据分割开来，使得不同类别的数据间距离最大化，间隔最大化。支持向量机可以有效解决样本不均衡的问题，且在高维空间中仍然有效。

本系列教程介绍了支持向量机中的最优化算法——坐标下降法、KKT条件以及拉格朗日对偶性。先从简单的线性可分支持向量机开始，逐步推广到更一般的非线性支持向量机，最后讨论了其拓展。
# 2.核心概念与联系
## 2.1 线性可分支持向量机
线性可分支持向量机（Linear Separable Support Vector Machine, LSSVM）是一种二类分类模型，也是本系列教程的起点。LSSVM 可以通过求解以下优化问题得到分界超平面：
$$
\begin{align*}
&\underset{\boldsymbol w}{\text{max}}&\quad \sum_{i=1}^m \alpha_i-\frac{1}{2}\left(\boldsymbol w^T \boldsymbol w+\rho\right)\\
&s.t.&\quad y_i(x_i^T \boldsymbol w+\theta)=1,\forall i\\
&\quad\quad&\quad\quad 0\leq\alpha_i\leq C,\forall i
\end{align*}
$$
其中$\boldsymbol x_i$表示输入向量，$y_i$表示样本类别，$C>0$是软间隔正则化参数。$\alpha_i$是拉格朗日乘子，$\rho$是松弛变量。

对于线性可分问题来说，上述约束条件是等号约束条件，因此可以使用KKT条件进行求解：
$$
\begin{aligned}
\alpha_i &=-\dfrac{\delta_{ik}(w^Tx_k+b)}{\delta_{ki}x_k}\\[1ex]
w &=\sum_{i=1}^{n}\alpha_iy_ix_i \\[1ex]
b &= y_j-w^Ty_j
\end{aligned}
$$
其中$n$为样本数量，$x_k$表示第$k$个输入向量，$y_k$表示第$k$个样本类别，$\delta_{ik}$表示第$i$个样本对第$k$个约束的偏导，$w$和$b$分别是模型的参数。

对于非线性可分问题，可以使用核函数的方法进行转换，引入核函数作为转换的中间层，并利用核函数作为特征空间的特征来提升泛化能力。首先定义核函数：
$$
K(x_i,x_j)=\exp(-\gamma||x_i-x_j||^2)
$$
其中$\gamma$是一个控制权值变换的超参。假设输入空间$\mathcal{X}$和输出空间$\mathcal{Y}$都是欧式空间，那么$\gamma$越小意味着数据的重合度越高；而$\gamma$越大意味着数据的非线性影响越小。当$\gamma=0$时，就是采用线性核函数。引入核函数后，约束条件变成：
$$
\begin{align*}
&\underset{\boldsymbol w}{\text{max}}&\quad \sum_{i=1}^m \alpha_i-\frac{1}{2}\left(\boldsymbol w^T \boldsymbol K \boldsymbol w+\rho\right)\\
&s.t.&\quad y_i(x_i^T \boldsymbol w+\theta)=1,\forall i\\
&\quad\quad&\quad\quad 0\leq\alpha_i\leq C,\forall i
\end{align*}
$$
## 2.2 非线性支持向量机
### 2.2.1 核技巧
核函数能够有效地处理非线性问题。当原始空间$\mathcal{X}$不是欧氏空间时，可以通过将输入数据映射到高维特征空间$\mathcal{H}$中，然后采用核函数的方式来处理非线性问题。核函数将原始数据通过非线性变换映射到特征空间$\mathcal{H}$中，使得机器学习算法在特征空间中进行分类和回归。常用的核函数有多项式核、高斯核等。

假设输入空间$\mathcal{X}$和输出空间$\mathcal{Y}$都是$\mathbb{R}^d$的子集，存在一个映射函数$K:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}$。则可以根据核函数的定义写出判别式：
$$
f(x)=\mathrm{sign}(w^Tx+b)\quad=\mathrm{sign}(\sum_{i=1}^nw_iy_ik(x_i,x)+b),
$$
其中$k(x_i,x_j)$表示$x_i$和$x_j$的核函数值。特别地，如果$K(x_i,x_j)=\sigma(x_i^Tx_j)$，则称$K$为径向基核函数（radial basis function）。径向基核函数经常被用于SVM中的核技巧。

核技巧的基本思想是，通过一个非线性变换将输入空间投影到一个更容易处理的高维空间，然后在高维空间中采用核函数来分类和回归。核技巧的主要优点是它可以在保持非线性拟合能力的同时大幅减少计算复杂度。SVM的实现通常依赖于核函数及其导数。

### 2.2.2 模型选择
#### 2.2.2.1 软间隔与硬间隔
线性可分支持向量机只能处理线性分界，所以不能很好地处理非线性数据。为了克服这个缺陷，我们需要考虑非线性支持向量机。

支持向量机具有软间隔特性，即允许某些误分类样本。而硬间隔支持向量机（hard margin support vector machine, HSVM）要求所有样本都必须正确分类。

软间隔和硬间隔之间的关键区别在于如何处理支持向量。软间隔通过增加松弛变量来控制间隔，允许一些样本被错误分类。相反，硬间隔支持向量机通过惩罚所有支持向量，使得它们彼此间隔足够大。为了防止过拟合，HSVM通常会设置一个限定超参$C$，使得支持向量的个数至多为$C$。

#### 2.2.2.2 支持向量的选择
支持向量机模型的一个重要的特性是它只关心支持向量，因为其他样本对决策没有贡献。但实际上，所有样本都可能成为支持向量。如何确定哪些样本是支持向量呢？一种办法是预先设置一个阈值$\epsilon$，只有那些违背约束条件的样本才被视作支持向量。另一种办法是用结构风险最小化的准则来选择支持向量，即只选取罕见的边缘样本。但是这种方式比较困难，通常采用启发式的方法，如启发式搜索法（hill climbing algorithm）。

#### 2.2.2.3 不同的目标函数
除了软间隔的线性可分支持向量机外，还有其他类型的支持向量机，如无核的非线性支持向量机、带核的非线性支持向量机和半正定核的支持向量机。这些类型的支持向量机都有着不同的目标函数，而且选择核函数的作用也不同。

无核的非线性支持向量机（Nonlinear SVM without Kernel）直接在输入空间$X$上进行训练，即直接在输入向量上进行分类。通过在输入空间中构造合适的超曲面，来避开训练样本中存在的噪声点。它的损失函数如下：
$$
\begin{align*}
&\underset{\boldsymbol w}{\min}&\quad \frac{1}{2}\|w\|^2+\lambda\|W\|_F\\
&\text{s.t.}&\quad y_i(w^Tx_i+\theta)-1\geqslant 0,\forall i
\end{align*}
$$
其中$\lambda>0$控制正则化强度，$W=[w_1;w_2]$表示所有样本的权重。它利用拉格朗日乘子的对偶形式进行优化，给出解：
$$
\hat{w}=(X^TX+I_N\lambda W^TW)^{-1}X^TY+X^TW.
$$
其中$I_N$是一个$N\times N$的单位矩阵。

带核的非线性支持向量机（Nonlinear SVM with Kernel）是无核支持向量机的扩展。它通过隐式地将原始输入数据映射到高维特征空间，来获得非线性拟合能力。核函数可以看做是在低维特征空间中对输入数据的内积。带核的非线性支持向量机的目标函数如下：
$$
\begin{align*}
&\underset{\boldsymbol w}{\min}&\quad \frac{1}{2}\|w\|^2+\lambda\|W\|_F\\
&\text{s.t.}&\quad y_i(w^T\phi(x_i)+b)-1\geqslant 0,\forall i\\
&s.t.&\quad k(x_i,x_j)\geqslant 1,\forall i,j
\end{align*}
$$
其中$\lambda>0$控制正则化强度，$\phi$是特征映射函数，$k(x_i,x_j)$表示$x_i$和$x_j$的核函数值。它的损失函数如下：
$$
\begin{align*}
&\underset{\boldsymbol w,b,\lambda}{\min}&\quad \frac{1}{2}\|w\|^2+\lambda\|W\|_F\\
&\text{s.t.}&\quad y_i(w^T\phi(x_i)+b)-1\geqslant 0,\forall i\\
&s.t.&\quad \mathbf{q}_i\cdot\mathbf{q}_j\leqslant M\Delta_{ij},\forall i,j
\end{align*}
$$
其中$\Delta_{ij}=y_i\neq y_j$, $M$是容忍度参数。这个目标函数进一步限制了两个相同类的样本必须在特征空间中的距离至少为$M$，这样可以避免因类内噪声导致的不可靠分类。

半正定核的支持向量机（Semi-Positive Definite SVM）是带核的支持向量机的特殊情况。它的目标函数如下：
$$
\begin{align*}
&\underset{\boldsymbol w,b,\lambda}{\min}&\quad \frac{1}{2}\|w\|^2+\lambda\|\Sigma\|_2\\
&\text{s.t.}&\quad y_i(w^T\phi(x_i)+b)-1\geqslant 0,\forall i\\
&s.t.&\quad \mathbf{q}_i\cdot\mathbf{q}_j\leqslant M\Delta_{ij},\forall i,j
\end{align*}
$$
其中$\Sigma$是一个半正定的对称矩阵，限制了两个不同的类必须在特征空间中的距离至少为$M$。