
作者：禅与计算机程序设计艺术                    

# 1.简介
  


在统计学、经济学、金融领域，由于复杂性、非线性、不确定性等原因，如何求解优化问题是至关重要的问题。本文将介绍一些经典的最优化算法，包括梯度下降法、牛顿法、拟牛顿法、BFGS方法、L-BFGS方法、Powell方法、Conjugate Gradient方法、Limited Memory Broyden-Fletcher-Goldfarb-Shanno (L-MGBFS)方法、Sequential Least Squares Programming (SLSQP)方法、Simulated Annealing方法、Stochastic Gradient Descent with Momentum (SGD-M)方法、Adagrad方法、Adadelta方法、RMSprop方法、Adam方法。并会结合实际应用案例介绍它们的特点和适用场景。

# 2. 算法背景介绍

## 梯度下降法(Gradient Descent) 

梯度下降法（Gradient descent）是一种基于导数的迭代优化算法。其基本思想是在函数的山峰上寻找下坡路线，使得函数值降低，即下降最快。通过计算目标函数的梯度（斜率），可以得到该方向上的下降速度。

具体而言，给定初始点$x_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. $k=0$
2. $x_{k+1} = x_k - \eta\nabla f(\mathbf{x}_k)$
3. while满足停止条件，执行以下操作
   * if $|f(\mathbf{x}_{k+1}) - f(\mathbf{x}_k)| < \epsilon$,则停止迭代，得到局部最小值
   * else 设$t_k=\frac{|f(\mathbf{x}_{k+1}) - f(\mathbf{x}_k)|}{|\nabla f(\mathbf{x}_k)|}$，则$\eta = \frac{\eta}{\sqrt{t_k}}$，并令$k=k+1$

其中，$\nabla f(\mathbf{x})$表示$f$在$\mathbf{x}$处的一阶偏导数，$\epsilon$是一个精度参数。

## 牛顿法(Newton Method)

牛顿法（Newton method）是一种基于海瑟矩阵的迭代优化算法，主要用于解决非线性方程组及其他包含高维变量的最优化问题。其基本思想是对函数进行二阶泰勒展开，得到二阶导数的泰勒公式，然后根据泰勒公式求出泰勒展开式的根，从而得到接近极小值点。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. $\mathbf{p}_k=-\nabla f(\mathbf{x}_k)$
2. 如果$\nabla^2 f(\mathbf{x}_k)\neq\mathbf{0}$,那么：
   1. 利用海瑟矩阵公式计算海瑟矩阵$\mathbf{H}_k=\nabla^2 f(\mathbf{x}_k)$
   2. 利用迭代公式计算更新向量$\mathbf{p}_{k+1}=(-\nabla^2 f(\mathbf{x}_k))^{-1}\nabla f(\mathbf{x}_k)$
3. 如果$\nabla^2 f(\mathbf{x}_k)=\mathbf{0}$,那么:
   1. 使用单位阵代替海瑟矩阵$\mathbf{H}_k$
   2. 使用普通梯度下降方法计算$\mathbf{p}_{k+1}=-\nabla f(\mathbf{x}_k)$
4. $x_{k+1} = x_k + \eta\mathbf{p}_{k+1}$
5. while满足停止条件，执行以下操作
   * 如果$f(x_{k+1})<f(x_k)$,则停止迭代，得到局部最小值；否则继续迭代。

## 拟牛顿法(Quasi Newton Methods)

拟牛顿法（Quasi-Newton method）是一种基于拟合二阶模型的迭代优化算法。其基本思想是利用损失函数的二阶导数矩阵来近似海瑟矩阵。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 如果损失函数是二次函数或者更简单的函数形式，则使用BFGS算法；如果损失函数是海森矩阵函数或自然梯度下降收敛较慢的情况，则使用DFP算法。
2. 对于BFGS算法，需要计算海森矩阵。具体做法是计算当前梯度$\nabla f(\mathbf{x}_k)$和上一次梯度$\nabla f(\mathbf{x}_{k-1})$的共轭梯度矩阵，并用它们来计算海森矩阵。海森矩阵可用于近似目标函数的一阶和二阶导数，并且对拟牛顿法的收敛性具有较好的保证。
3. DFP算法也称为拟牛顿法（拟牛顿法最初起源于牛顿法中的拟牛顿方法）。其基本思想是采用差分拟合的思想。损失函数是由权重乘子矩阵$\Theta$和二次型矩阵$Q$决定的。在每一步迭代中，需要计算矩阵$H$的逆，并用于拟合矩阵$Q$。但是计算逆矩阵$H^{-1}$代价很高，所以DFP算法通常会使用牛顿法来逼近逆矩阵。
4. 拟牛顿法还有许多变种算法，例如L-BFGS算法和DFP算法，有限元法，缩减法等。这些算法都有着各自的优缺点，但在一般情况下，拟牛顿法都比普通牛顿法收敛速度要快，而且精度也相对较高。

## BFGS方法(Broyden–Fletcher–Goldfarb–Shanno)

BFGS算法（Broyden–Fletcher–Goldfarb–Shanno）是最流行的拟牛顿法之一，也是一种基于海森矩阵的迭代优化算法。其基本思想是计算损失函数的一阶和二阶导数，然后将它们合并成海森矩阵，最后采用标准线性代数的方法求解海森矩阵的逆矩阵。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 初始化搜索方向$\mathbf{d}_0=-\nabla f(\mathbf{x}_0)$,迭代次数$i=0$
2. 循环直到满足迭代终止条件
   * 计算搜索方向$\mathbf{s}_i=-\nabla f(\mathbf{x}_{i-1}+\alpha_i\mathbf{d}_{i-1})$，其中$\alpha_i$是下降因子。
   * 更新海森矩阵$B_{ij}=\gamma_is_is^\top+\delta_{i-1}y_iy_i^\top$，其中$\delta_{i-1}$表示$B$矩阵是否已经收敛。
   * 更新搜索方向$\mathbf{d}_i=\frac{1}{\beta_i}(\gamma_is_{i-1}-y_i)$。其中$\gamma_i$和$\beta_i$是固定的下降因子。
   * 当$B$矩阵已经收敛时，退出循环；否则继续循环。
3. 计算海森矩阵的逆矩阵$\mathbf{H}^{-1}_k=\frac{1}{B^{-\top}B^{-1}}\left[B^{-1}B^{-\top}\right]^{-1}B^{-1}$，并使用迭代公式获得更新向量$\mathbf{p}_{k+1}=\mathbf{H}^{-1}_kd_k$。
4. $x_{k+1} = x_k + \eta\mathbf{p}_{k+1}$
5. while满足停止条件，执行以下操作
   * 如果$f(x_{k+1})<f(x_k)$,则停止迭代，得到局部最小值；否则继续迭代。

## L-BFGS方法(Limited Memory BFGS)

L-BFGS方法（Limited memory BFGS）是一种改进版的拟牛顿法，它对拟牛顿法的内存消耗进行了限制，提升算法性能。其基本思想是使用最近$m$次搜索方向的信息，而不是仅仅使用当前搜索方向的信息。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 初始化搜索方向$\mathbf{d}_0=-\nabla f(\mathbf{x}_0)$,迭代次数$i=0$
2. 初始化历史矩阵$H_0$为零矩阵。
3. 循环直到满足迭代终止条件
   * 计算搜索方向$\mathbf{s}_i=-\nabla f(\mathbf{x}_{i-1}+\alpha_i\mathbf{d}_{i-1})$，其中$\alpha_i$是下降因子。
   * 更新历史矩阵$H_{i,:}$和$H_{:,i}$
   * 根据历史矩阵$H_{i-m\leq j< i}$计算迭代矩阵$W_{i-m+1:i,\hat{s}}=(H^{-1}_{i-m+1:i})_{ij}(H^{-T}_{i-m+1:i})\hat{s}$
   * 计算迭代矩阵$Y_{i-m+1:i}=\left[(I-WH_{i-m+1:i})H^{-1}_{i-m+1:i}Y_{i-m+1:i-1}\right]_{n\times n}$。其中$n$表示维度。
   * 计算$A_i$和$c_i$，使得搜索方向$\mathbf{s}_i$被近似成两个向量：$\tilde{\mathbf{s}}_i=\mu_i\mathbf{z}_i+[\lambda_i]\nabla f(\mathbf{x}_{i-1}+\alpha_i\mathbf{d}_{i-1})$，其中$\lambda_i$和$\mu_i$是固定的下降因子，$\mathbf{z}_i$是迭代矩阵$Z_i$的第$i$列，并满足约束条件$||\mathbf{z}_i||_2=1$，其中$Z_i=H^{-1}_{i-m+1:i}X_{i-m+1:i}$。
   * 更新搜索方向$\mathbf{d}_i=W_{i-m+1:i}\left[A_iB_{i-m+1:i}^-1C_i+[\lambda_i]\nabla f(\mathbf{x}_{i-1}+\alpha_i\mathbf{d}_{i-1})\right]$
   * $X_{i}=\left[I-WB_{i-m+1:i}A_i\right]X_{i-1}+Y_{i-m+1:i}\mathbf{z}_i^\top$
   * $Z_i=H^{-1}_{i-m+1:i}X_{i-m+1:i}$
   * 当迭代次数达到最大次数$N$时，退出循环；否则继续循环。
4. 计算海森矩阵的逆矩阵$\mathbf{H}^{-1}_k=\frac{1}{B^{-\top}B^{-1}}\left[B^{-1}B^{-\top}\right]^{-1}B^{-1}$，并使用迭代公式获得更新向量$\mathbf{p}_{k+1}=\mathbf{H}^{-1}_kd_k$。
5. $x_{k+1} = x_k + \eta\mathbf{p}_{k+1}$
6. while满足停止条件，执行以下操作
   * 如果$f(x_{k+1})<f(x_k)$,则停止迭代，得到局部最小值；否则继续迭代。

## Powell方法(Powell's method)

Powell方法（Powell’s method）是一种迭代优化算法。其基本思想是沿着一个单纯形路径（simplex path）进行迭代，使得函数值下降最快。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 初始化搜索方向$\mathbf{p}_0=(-\nabla f(\mathbf{x}_0),...,-\nabla f(\mathbf{x}_0)),...\,(+\nabla f(\mathbf{x}_0),...,\nabla f(\mathbf{x}_0))$.
2. 初始化搜索半径$\rho$
3. while满足迭代终止条件
   * 对每个坐标轴，沿着单纯形路径进行迭代，使得目标函数下降最快。具体地，在第$i$个坐标轴上，求解$\Delta_if(\mathbf{x}_{k-1}+\Delta p_{\rm min}+\Delta q_{\rm max})$与$\Delta_jf(\mathbf{x}_{k-1}+\Delta p_{\rm min}+\Delta q_{\rm max})$之间的最小值，并令$\Delta_ix_i=x_{k-1}[i]+\Delta p_{\rm min}[i]$和$\Delta_jx_j=x_{k-1}[j]+\Delta q_{\rm max}[j]$,其中$0\leq\Delta p_{\rm min}[i],\Delta q_{\rm max}[j]<\rho$。
   * 更新搜索半径$\rho$。
   * 当搜索半径达到精度要求时，退出循环。
4. 当所有坐标轴都已经完成了完整的单纯形路径迭代后，计算全局最小值。

## Conjugate Gradient方法(Conjugate Gradient Method)

Conjugate Gradient方法（Conjugate gradient method）是一种基于伪逆矩阵的迭代优化算法。其基本思想是利用矩阵分解来求解目标函数的梯度和海瑟矩阵，从而减少计算量和存储空间。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 初始化搜索方向$\mathbf{p}_0=-\nabla f(\mathbf{x}_0)$,梯度为零向量，迭代次数$i=0$
2. 循环直到满足迭代终止条件
   * 计算梯度$-\nabla f(\mathbf{x}_{k-1}+\alpha_i\mathbf{p}_{k-1})$，并计算预conditioner矩阵$B_i$。
   * 计算搜索方向$-\Delta\mathbf{p}_i=B_i^{-1}\nabla f(\mathbf{x}_{k-1}+\alpha_i\mathbf{p}_{k-1})+\beta_iB_i^{-1}B_i\Delta\mathbf{p}_{i-1}$。其中$\beta_i$是固定的下降因子。
   * 计算超松弛因子$\mu_i=\frac{\langle\nabla f(\mathbf{x}_{k-1}),\Delta\mathbf{p}_i\rangle}{\langle\Delta\mathbf{p}_{i-1},\Delta\mathbf{p}_{i-1}\rangle}$.
   * 更新搜索方向$\mathbf{p}_{i+1}=\mathbf{p}_{i}+\mu_i\Delta\mathbf{p}_i$。
   * 当超松弛因子等于零时，退出循环。
   * 计算更新步长$\alpha_{i+1}=(\alpha_{i-1}-\frac{\langle\nabla f(\mathbf{x}_{k-1}),\Delta\mathbf{p}_i\rangle}{\langle\Delta\mathbf{p}_{i-1},\Delta\mathbf{p}_{i-1}\rangle})/\mu_i$。
   * 在相应坐标轴上更新步长为$\alpha_i$，并更新对应位置。
   * 计算下一次迭代时的坐标$x_{k+1}=x_{k-1}+\alpha_{i+1}\Delta\mathbf{p}_{i+1}$。
   * 当达到最大迭代次数时，退出循环。
   * 更新梯度$\mathbf{g}_i=\nabla f(\mathbf{x}_{k-1}+\alpha_{i+1}\mathbf{p}_{i+1})$。
3. $x_{k+1} = x_{k}+\sum_{i=0}^i\alpha_{i}\mathbf{p}_{i}$

## Limited Memory Conjugate Gradient Method

L-MGCG方法（Limited memory conjugate gradient method）是一种改进版的CG方法。其基本思想是对残差向量的更新施加限制，从而提升收敛速度和精度。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 初始化搜索方向$\mathbf{p}_0=-\nabla f(\mathbf{x}_0)$,梯度为零向量，迭代次数$i=0$
2. 初始化缓存矩阵$S_0=I_{n\times n}$，残差向量$\mathbf{r}_0=-\nabla f(\mathbf{x}_0)$
3. 循环直到满足迭代终止条件
   * 计算残差向量$B_{i-1}=\left(\begin{array}{cc}I&-S_{i-1}\\S_{i-1}^{-1}&I\end{array}\right)^{-1}\left(\begin{array}{c}\nabla f(\mathbf{x}_{i-1})\\0\end{array}\right)$。
   * 更新缓存矩阵$S_{i}=\left(\begin{array}{cc}S_{i-1}&-B_{i-1}^{T}\\B_{i-1}&\Lambda\end{array}\right)$。其中$\Lambda$是一个对角矩阵。
   * 更新搜索方向$-\Delta\mathbf{p}_i=B_{i-1}^{-1}\nabla f(\mathbf{x}_{i-1})+\beta_iB_{i-1}^{-1}B_{i-1}\Delta\mathbf{p}_{i-1}$。其中$\beta_i$是固定的下降因子。
   * 计算超松弛因子$\mu_i=\frac{\langle\nabla f(\mathbf{x}_{i-1}),\Delta\mathbf{p}_i\rangle}{\langle\Delta\mathbf{p}_{i-1},\Delta\mathbf{p}_{i-1}\rangle}$.
   * 更新搜索方向$\mathbf{p}_{i+1}=\mathbf{p}_{i}+\mu_i\Delta\mathbf{p}_i$。
   * 当超松弛因子等于零时，退出循环。
   * 计算更新步长$\alpha_{i+1}=(\alpha_{i-1}-\frac{\langle\nabla f(\mathbf{x}_{i-1}),\Delta\mathbf{p}_i\rangle}{\langle\Delta\mathbf{p}_{i-1},\Delta\mathbf{p}_{i-1}\rangle})/\mu_i$。
   * 在相应坐标轴上更新步长为$\alpha_i$，并更新对应位置。
   * 更新残差向量$\mathbf{r}_{i+1}-=B_{i-1}^{-1}\left(\nabla f(\mathbf{x}_{i-1})+\beta_iB_{i-1}^{-1}B_{i-1}\Delta\mathbf{p}_{i-1}\right)+B_{i}^{-1}B_{i-1}\Delta\mathbf{p}_i$。
   * 当$\Vert\mathbf{r}_{i+1}\Vert_2^2\le\epsilon_2^2$或$\Vert\Delta\mathbf{p}_{i+1}\Vert_2\le\epsilon_1$时，退出循环。
   * 计算下一次迭代时的坐标$x_{i+1}=x_{i-1}+\alpha_{i+1}\Delta\mathbf{p}_{i+1}$。
   * 更新梯度$\mathbf{g}_i=\nabla f(\mathbf{x}_{i-1}+\alpha_{i+1}\mathbf{p}_{i+1})$。
   * $i=i+1$
4. $x_{k+1} = x_{k}+\sum_{i=0}^i\alpha_{i}\mathbf{p}_{i}$

## Sequential Least Square Programming

SLSQP方法（Sequential least squares programming）是一种迭代优化算法。其基本思想是拟合二次型函数。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 设置$N$为约束条件个数，设置$\epsilon$为迭代终止阈值。
2. 初始化$k=0$，设置$\phi_0=f(\mathbf{x}_0)$，设置$x_0=\mathbf{x}_0$。
3. while满足迭代终止条件
   * 计算梯度$\nabla f(\mathbf{x}_{k})$。
   * 计算海瑟矩阵$H_{k}=J(\mathbf{x}_{k})^TH(\mathbf{x}_{k})$。
   * 计算线性规划器$A_{k}=-H^{-1}J(\mathbf{x}_{k})^T$。
   * 计算线性规划器的置信度范围$t_k=\mathrm{max}\{\beta t_{k-1},\theta\}$，其中$\beta$为一个参数，$\theta$为一个很小的正数。
   * 计算线性规划器的绝对误差范围$a_k=\mathrm{max}\{\alpha a_{k-1},\xi\}$，其中$\alpha$为一个参数，$\xi$为一个很小的正数。
   * 通过线性规划器最小化$f(\mathbf{x}_{k+1})$，以得到新的变量取值。具体方法是，求解线性规划问题：
      $$
          \min_{\Delta x_{k+1}} \quad ||J(\mathbf{x}_{k})(\Delta x_{k+1})|| \\ 
          s.t.\quad (\Delta x_{k+1},...,a_k)\in [l,u],\{J(\mathbf{x}_{k}),h(\mathbf{x}_{k+1})\}_{k=0}^{k+1},t_{k}
   $$
   * 判断迭代是否成功。具体判断标准是，线性规划器的绝对误差是否在$a_k$范围内。
   * 计算误差$e_k=f(\mathbf{x}_{k+1})-f(\mathbf{x}_k)-\nabla f(\mathbf{x}_{k})\cdot(\mathbf{x}_{k+1}-\mathbf{x}_{k})$。
   * 判断迭代是否满足精度要求。具体判断标准是，绝对误差是否在$a_k$范围内，残差是否小于阈值。
   * 执行完毕，$k=k+1$。
4. return $\mathbf{x}_{k+1}$。

## Simulated Annealing

模拟退火（simulated annealing）是一种基于温度变化的迭代优化算法。其基本思想是随机游走，随着温度的升高，接受新解的概率越来越小，使得陷入局部极值时能够跳出。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 设置初始温度$\tau_0$和降温速率$\gamma$。
2. 初始化$\mathbf{x}_k=\mathbf{x}_0$。
3. 循环直到满足迭代终止条件
   * 生成一个新解$\mathbf{x}_{k+1}$。
   * 如果$f(\mathbf{x}_{k+1})<f(\mathbf{x}_k)$，则接受该解，否则接受一定概率的新解。具体计算方法是，计算接受概率$p=\exp(-(f(\mathbf{x}_{k+1})-f(\mathbf{x}_k))/(k\gamma))$。如果$p>r$，则接受新解；否则接受一定概率的旧解。
   * 计算新的温度$\tau_k'=cooling\_rate(\tau_k)$。
   * 如果$\tau_{k'}<\epsilon$，则退出循环；否则继续循环。
4. return $\mathbf{x}_{k+1}$。

## Stochastic Gradient Descent with Momentum (SGD-M)

带动量的随机梯度下降（Stochastic Gradient Descent with Momentum）是一种基于一阶矩估计的迭代优化算法。其基本思想是利用动量法，在梯度下降的基础上引入一阶矩估计，增强梯度的估计精度。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 设置动量参数$\beta$。
2. 初始化参数$\mathbf{v}_0=\beta\nabla f(\mathbf{x}_0)$。
3. while满足迭代终止条件
   * 对于每一个样本$(\mathbf{x}_{i},y_i)$,计算梯度$g_i=\nabla f(\mathbf{w}_i+\mathbf{v}_k)$。其中$\mathbf{w}_i$是模型的参数，$\mathbf{v}_k$是速度参数。
   * 利用梯度下降更新速度参数$\mathbf{v}_{k+1}=\beta\mathbf{v}_k+\eta g_i$。
   * for i=1 to num_samples
       * 对于第$i$个样本$(\mathbf{x}_{i},y_i)$,更新参数$\mathbf{w}_i\leftarrow\mathbf{w}_i-\eta g_i$。
   * return $\mathbf{x}_{k+1}$。

## Adagrad方法(Adaptive Gradient Method)

Adagrad方法（Adaptive gradient method）是一种基于参数分裂的迭代优化算法。其基本思想是对每个参数对应的梯度平方进行累加，然后根据这个统计信息调整步长大小。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 设置分裂因子$R_t=\frac{\eta}{1+R_{t-1}^Tx}$。
2. 初始化参数$R_0=\mathbf{0}$。
3. while满足迭代终止条件
   * 计算梯度$g_k=\nabla f(\mathbf{x}_k)$。
   * 更新参数$R_{k+1}=R_kt^Tg_k$。
   * 更新参数$x_{k+1}=x_k-\frac{\eta}{\sqrt{R_{k+1}}}g_k$。
   * 返回$\mathbf{x}_{k+1}$。

## Adadelta方法(Adapative Delta learning rate method)

Adadelta方法（Adapative delta learning rate method）是一种基于自适应学习率的迭代优化算法。其基本思想是对每个参数对应的梯度平方和参数的平均平方误差进行累加，然后根据统计信息调整步长大小。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 设置分裂因子$R_t=\frac{\eta}{1+R_{t-1}^Tx}$。
2. 初始化参数$R_0=\mathbf{0}$，$E[g^2]_0=\mathbf{0}$，$E[\Delta x^2]_0=\mathbf{0}$。
3. while满足迭代终止条件
   * 计算梯度$g_k=\nabla f(\mathbf{x}_k)$。
   * 更新参数$E[g^2]_{k+1}=0.9\cdot E[g^2]_{k}+0.1\cdot g_kg_k$。
   * 更新参数$R_{k+1}=R_kt^Te[g^2]_{k+1}/(\sqrt{E[g^2]_{k+1}}+\epsilon)$。
   * 更新参数$E[\Delta x^2]_{k+1}=0.9\cdot E[\Delta x^2]_{k}+0.1\cdot (\Delta x_k)^2$。
   * 更新参数$x_{k+1}=x_k-\frac{\eta}{\sqrt{R_{k+1}+\epsilon}}\Delta x_k$。
   * 返回$\mathbf{x}_{k+1}$。

## RMSprop方法(Root Mean Squared Propagation)

RMSprop方法（Root mean squared propagation）是一种基于自适应学习率的迭代优化算法。其基本思想是对每个参数对应的梯度平方和平方均方根进行累加，然后根据统计信息调整步长大小。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 设置分裂因子$R_t=\frac{\eta}{\sqrt{1-(R_{t-1}^Tw)})}$。
2. 初始化参数$R_0=\mathbf{0}$，$E[g^2]_0=\mathbf{0}$。
3. while满足迭代终止条件
   * 计算梯度$g_k=\nabla f(\mathbf{x}_k)$。
   * 更新参数$E[g^2]_{k+1}=0.9\cdot E[g^2]_{k}+0.1\cdot g_kg_k$。
   * 更新参数$R_{k+1}=R_t/(\sqrt{E[g^2]_{k+1}}+\epsilon)$。
   * 更新参数$x_{k+1}=x_k-\frac{\eta}{R_{k+1}}g_k$。
   * 返回$\mathbf{x}_{k+1}$。

## Adam方法(Adam optimization algorithm)

Adam方法（Adaptive moment estimation）是一种基于自适应学习率的迭代优化算法。其基本思想是对梯度以及各项指标的偏差平方和各项指标的偏差和估计进行累加，然后根据统计信息调整步长大小。

具体而言，给定初始点$\mathbf{x}_0$,初始步长$\eta$,目标函数$f(\mathbf{x})$,求解其局部最小值的算法如下:

1. 设置分裂因子$R_t=\frac{\eta}{1-\beta^t}\sqrt{1-\beta^t}$。
2. 初始化参数$m_0=\mathbf{0}$，$v_0=\mathbf{0}$，$\beta_1=0.9$,$\beta_2=0.999$,$\epsilon=10^{-8}$。
3. while满足迭代终止条件
   * 计算梯度$g_k=\nabla f(\mathbf{x}_k)$。
   * 更新参数$m_k=0.9\cdot m_{k-1}+0.1\cdot g_k$。
   * 更新参数$v_k=0.999\cdot v_{k-1}+0.001\cdot g_kg_k$。
   * 更新参数$\hat{m}_k=\frac{m_k}{1-\beta_1^k}$。
   * 更新参数$\hat{v}_k=\frac{v_k}{1-\beta_2^k}$。
   * 更新参数$x_{k+1}=x_k-\frac{R_t}{\sqrt{\hat{v}_k+\epsilon}}\hat{m}_k$。
   * 返回$\mathbf{x}_{k+1}$。