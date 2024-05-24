
作者：禅与计算机程序设计艺术                    

# 1.简介
         
19世纪末，当时还叫数学家莱昂哈德·马尔可夫斯基（Leonhard Euler）在欧洲地区发现了积分函数和微分方程，从此引申出了一整套符号语言，以及解决分析问题的方法论，例如微积分、函数论、力学、工程等等。这些方法论后来成为著名的通用科学的理论基础。
         
         而近些年，随着计算机的发明、机器学习的进步，自动化推理系统的普及以及海量数据的涌现，神经网络模型的训练技艺也越来越高级，越来越能够处理复杂的非线性关系。基于上述前沿的技术革命，关于函数的导数的研究热潮似乎正在复苏。

         2017年，阿兰·图灵奖得主马文·弗里德曼（<NAME>）等人发表了一篇题目为“Can Neural Networks Reveal the Equations of Motion?”的文章，通过对一些微观物体的运动进行模拟实验，以及由动力学相互作用所生成的牛顿三角法模型，证明了神经网络模型能够学习到物质运动的 governing equations 。

         2018年，中科院自动化所长江学者邹仁升（Lance Zhang）团队通过实验发现神经网络可以学习到一些运动规律，从而对自然界的一系列运动机理进行建模。该论文被国际顶尖期刊IEEE Transactions on Neural Networks and Learning Systems接受。

         随着这一研究领域的成熟，我们可以看到函数的导函数（differential equation）在神经网络学习中的应用。这个问题的关键就是如何将神经网络中的非线性激活函数（activation function）表示为导函数形式。从数学上来说，通过求导操作就能得到导函数表达式。而这一思路，正是本文要探讨的内容。

         本文的主要贡献有两个方面。第一，我们将提出两种学习导函数的方式——前馈神经网络（feedforward neural network）与反向传播（backpropagation）。对于导函数学习，前馈神经网路与反向传播这两种方法是最常用的两种方法。第二，我们通过实验验证了这种导函数学习方法在学习各种函数的导函数方面的有效性。 

         为什么要学习导函数？因为导函数描述了函数变化率或斜率的行为，它是许多物理定律、社会现象和数学模型的基础。如电场力、温度场变化、流体流动方程式、热传导方程式、物理波动方程式、动力学方程等。利用导函数学习，可以对复杂的非线性关系进行抽象，并对数据进行解释，从而提高模型的预测精度。

         
         # 2.基本概念术语说明
         ## 概念与定义
         *导函数* $\mathrm{d}y/\mathrm{d}x=\frac{\partial y}{\partial x}$ 是函数 $y=f(x)$ 在点 $x$ 的切线。换句话说，它是函数 $y$ 在 $x$ 方向的斜率。导函数的取值一般满足不连续条件，即导函数为 0 和无限大的地方，$dy/dx$ 表示函数在 $x$ 轴方向上的一个单位长度，$\frac{dy}{dx}$ 表示函数在 $x$ 轴方向上的一个单位梯度。
         **导数** （derivative）是指导函数斜率的线性近似，因此导数也可以理解为函数的微分。我们把$u=du/dx$,记作$\frac{du}{dx} = \frac{\partial u}{\partial x}$,则有：
        $$
            \frac{\partial u}{\partial x} = \lim_{h\rightarrow 0}\frac{u(x+h)-u(x)}{h}, h>0
        $$ 
        **偏导数** (partial derivative) 是指导函数斜率在任一维度上的分量。若偏导数存在，则称导数是偏导数的导数。
         ## 符号说明
         - $y$: 函数 $y=f(x)$ 的值。
         - $x$: 变量 $x$ 的取值。
         - $z$: 变量 $z$ 的取值。
         - $f$: 一元函数 $f(x)$ 。
         - $g$: 一元函数 $g(z)$ 。
         - $u$: 变量 $u$ 的取值。
         - $v$: 变量 $v$ 的取值。
         - $\frac{\partial f}{\partial x}$: 函数 $f(x)$ 对 $x$ 的偏导数。
         - $\frac{\partial^n f}{\partial x^n}$: 函数 $f(x)$ 对 $x$ 的$n$次偏导数。
         - $\mathbf{W}$ : 权重矩阵。
         - $B$: 激活函数的输入加权和。
         - $\alpha$: L2 正则项系数。
         - $\lambda$: L2 正则项的权重衰减参数。
         - $\epsilon$: 学习速率。
         - ${\bf z}^i$: 第 $i$ 个隐藏层单元的输入。
         - ${\bf a}^j$: 第 $j$ 层网络输出。
         - $K$: 输出层的单元数。
         - $p_k$: 类别 $k$ 的概率。
         - $w_k$: 类别 $k$ 的权重。
         - $    ilde{w}_k$: 类别 $k$ 的一阶导数。
         - $\bar{b}_k$: 类别 $k$ 的二阶导数。
         - $
abla_{    heta^{(l)}}$: 参数 $    heta^{(\ell)}$ 关于损失函数 $\mathcal{L}$ 的梯度。
         - $
abla_{\boldsymbol{    heta}}$：损失函数关于参数集合 $\boldsymbol{    heta}$ 的梯度。
         - $\delta_j^{[l]}$: 误差项。
         - $a_i^{(l-1)}$: 第 $i$ 个隐藏层单元的输出。
         - $J(    heta)$: 模型的损失函数。
         - $
abla_    heta J(    heta)$: 模型的参数集合 $    heta$ 关于损失函数 $J(    heta)$ 的梯度。
         - $\gamma$: 向前传播的损失函数。
         - $\hat{y}^{(i)}$: 预测值。
         - $\hat{y}$: 最终预测值。
         - $D_{    ext{train}}$: 训练集的大小。
         - $D_{    ext{test}}$: 测试集的大小。
         - $\mathcal{X}$: 输入空间。
         - $\mathcal{Y}$: 输出空间。
         - $\phi$: 映射 $\mathcal{X}$ 到 $\mathcal{Y}$ 的非线性变换。
         ## 定义与公式
         ### 不定积分
         设 $f$ 是关于变量 $x$ 的单调递增函数，$I$ 是开区间 $[a, b]$ ，$f(x), g(x)$ 在 $[a, b]$ 上处处可导。那么，$fg|_I$ 可以看作在 $I$ 上关于 $x$ 的不定积分，记做 $fg(x)$ 。且满足如下形式：
         $$\begin{equation*}
             fg(x) = \int_a^b\left[\int_{f(a)}^{f(x)}\frac{dg'}{dx}(s)\right]ds + \int_a^x\left[\int_{g(a)}^{g(x)}\frac{df'}{dx}(s)\right]ds 
         \end{equation*}$$
         其中，$f'(x)>0,\quad g'(x)>0$,且 $f(a), f(b), g(a), g(b)>0$. 如果 $f(x)\leq g(x)$ ，则 $f'$ 可微分，可写为：
         $$\begin{equation*}
             f'(x) = \lim_{h\rightarrow 0}\frac{f(x+h)-f(x)}{h}.
         \end{equation*}$$
         当 $f(x)$ 在 $x=a$ 时趋于 0 时，$g(x)$ 可导，$g'(x)$ 在 $x=a$ 时趋于正无穷，故右边第二项等于 $-\infty$；当 $f(x)$ 在 $x=b$ 时趋于 $+\infty$ 时，$g(x)$ 可导，$g'(x)$ 在 $x=b$ 时趋于负无穷，故右边第二项等于 $+\infty$ 。所以，$fg|_I$ 属于某个特定形式，且只有当 $f(x)>g(x)$ 时，才会给出有意义的结果。
         
         ### 折线积分
         折线积分是指定义在闭区间 $[a, b]$ 或开区间 $(a, b)$ 上的关于曲线 $y=f(x)$ 的路径的形状积分，即：
         $$\begin{equation*}
              \int_{a}^{b}y\,dx = F(b) - F(a).
         \end{equation*}$$
         这里，$F(x)$ 是 $y$ 在 $x$ 处的曲线的曲率半径。
         
         当 $f(x)$ 连续，曲线的坐标方法有：
         $$
             f(x) = kx + m, \quad dx = \frac{1}{h}\sqrt{(f(x+h)-f(x))^2+(f(x)+f(x-h)-2f(x))^2},\quad\forall h
eq 0.
         $$
         令 $r = \frac{dx}{h}$，则：
         $$\begin{equation*}
              r = \frac{\Delta y}{\Delta x} = \frac{kf(x+h)+m-kfx-m}{h}\approx \frac{f''(x)}{2}\cdot h.
         \end{equation*}$$
         根据 Lagrange 插值法，取 $    au = (x+h)/2$，有：
         $$\begin{align*}
             &F(    au) = F((x+h)/2)\\
             &= [F(x) + (x+h-x)\frac{dF(x)}{dx}] + [(x+h)/2 - (x+h/2)]\frac{dF((x+h)/2)}{dx}\\
             &= F(x) + (x+h-x)r + (    au-x)(\frac{dF(x)}{dx} + \frac{dF(x+h)}{dx})\cdot h\\
             &= F(x) + (x+h-x)r + \frac{h}{2}(\frac{dF(x)}{dx} + \frac{dF(x+h)}{dx}).
         \end{align*}$$
         折线积分的计算公式：
         $$\begin{equation*}
             \int_{a}^{b}f(x)dx = F(b) - F(a) + C
         \end{equation*}$$
         其中，$C$ 是关于 $f(x)$ 的某些初值要求，比如 $\int_{a}^{a+h}f(x)dx = F(a+h) - F(a)$ 。
         
     