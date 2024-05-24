
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 什么是动量法？
动量法（Momentum optimizer）是一种优化算法，其思想是利用“物体运动中的惯性”来加速梯度下降过程。在机器学习领域中，动量法通常用于处理凸函数，即目标函数是连续可微的。由于深度神经网络（DNNs）具有高度非线性特性，因此在训练过程中存在很多局部最小值或鞍点，导致SGD算法在更新参数时容易陷入困境。动量法通过模拟真实世界的物体运动，来控制参数的更新方向，从而减少随机梯度下降（SGD）中的震荡并快速到达全局最优解。
## 1.2 为何需要使用动量法？
在深度学习中，优化器（Optimizer）是一个重要的组件，用于控制模型的参数更新。使用合适的优化器可以提高模型的训练速度、提升模型的泛化性能、防止过拟合等。但是，SGD算法有一个明显的缺陷——容易陷入局部极小值或鞍点。为了解决这个问题，人们开发了一些改进版的优化算法，如动量法、Adam优化器、Adagrad优化器等。下面我们就来了解一下动量法。
# 2.基本概念术语说明
## 2.1 动量
动量（momentum）是物体运动中的一种现象，它表示物体受力后保留的一定量的速度，就是物体运动中惯性的一个方面。动量法试图用这种方式来建立起一定的惯性，使得参数朝着能够有效降低代价的方向进行更新。它主要依赖于历史上的速度信息，即前面的迭代步已经完成的梯度。因此，动量法认为前面几次迭代步已经朝着优化方向移动了一定的距离，这一路上的每一步都应该由此引导。动量法使用一个超参数 $\beta$ 来控制历史速度对当前速度的影响程度。当 $\beta=0$ 时，动量法退化成SGD；当$\beta$较大时，会产生更大的动量，使得参数更新受惯性的影响变强；当$\beta$接近于无穷大时，参数更新将收敛到局部最优值。
## 2.2 共轭梯度法CG
共轭梯度法（Conjugate Gradient method，CG）是一种迭代优化算法，被广泛应用于求解非线性方程组。它采用矩阵分解的方法，把目标函数矩阵形式化，用正定矩阵 $A$ 和负定矩阵 $B$ 分解成两个正交矩阵，便于计算。在迭代过程中，算法首先计算预conditioned残差 $r_k = b - A x_k$ ，然后解线性方程组 $\hat{p}_k = (B^T r_{k-1})^{-1} B^T r_k$ （B^T是B的转置）。这样就可以得到搜索方向 $p_k = \hat{p}_{k-1} + cg_k$ 。CG的优势在于它的收敛速度很快，而且在计算上也比较简单。
## 2.3 二阶方法BFGS
BFGS（Broyden–Fletcher–Goldfarb–Shanno，BFGS）是一种线性精确型方法，适用于求解线性最小二乘问题。该算法利用海瑟矩阵（Hessian matrix）作为拟牛顿法的近似，海瑟矩阵是函数在一维搜索方向上的二阶导数的矩阵，可以方便地计算海瑟矩阵的逆矩阵。当目标函数为二次型时，可以使用BFGS算法，取得非常好的效果。
# 3.核心算法原理及具体操作步骤
## 3.1 梯度下降法
首先，初始化参数，令 $v_0=0$，然后执行以下循环：
$$x^{t+1}=x^t-\alpha\nabla f(x^t)$$
其中，$t$ 表示第 $t$ 次迭代，$\alpha$ 是步长。根据动量法的思想，引入历史速度 $v_t$ 的概念，将迭代步替换为：
$$v^{t+1}= \beta v^t+(1-\beta)\nabla f(x^t)\\
x^{t+1}=x^t-v^{t+1}$$
其中，$\beta$ 是动量因子，取值范围为 [0,1]，通常取 0.9。这个公式对应到代码实现就是：
```python
for i in range(iterations):
    if t == 0:
        # 初始化参数，令 v_0=0
        v = np.zeros(w.shape[0])

    grad = compute_gradient(X, y, w)  # 计算梯度
    v = beta * v + (1 - beta) * grad  # 更新历史速度
    w -= learning_rate * v   # 更新参数
```
## 3.2 共轭梯度法
共轭梯度法可以看作是在梯度下降法的基础上加入了预conditioning的步骤。共轭梯度法首先计算预conditioned残差 $r_k = b - Ax_k$ ，然后解线性方程组 $q_k = -(A^* q_{k-1})_{k-1} + r_k$ （A^*是A的伪逆矩阵），得到 $p_k = (B^T r_{k-1})^{-1} B^T r_k$ ，然后更新参数：
$$x^{t+1} = x^t + \alpha p_k$$
其中，$\alpha$ 是步长，是固定的。这个公式对应到代码实现就是：
```python
def conjgrad(X, y, w):
    def fisher_matrix_vector(p):
        """计算 Fisher 矩阵"""
        h = compute_hvp(p)
        return h @ p

    def compute_hvp(p):
        """计算 Hessian-vector product"""
        grad = compute_gradient(X, y, w)
        jacobian = compute_jacobian(X, y, w)

        v = torch.Tensor([grad]).reshape(-1) + alpha * p

        for _ in range(cg_iters):
            z = projector(v - lr * jacobian @ projection(v))
            y = momentum(y, z)
            v = proj_y + z

        return (proj_y - y).reshape((-1,)) / alpha

    def projector(z):
        norm = torch.norm(z)
        if norm < epsilon:
            return z
        else:
            return epsilon * z / norm

    def projection(v):
        return v - (v.T @ Q_k @ v) / (Q_k.T @ Q_k) * Q_k.T

    Q_k = None
    Q_k_1 = None
    iteration = 0
    prev_eigval = float('inf')
    maxiter = 100
    epsilon = 1e-5

    for it in range(maxiter):
        iteration += 1
        residual = b - X @ w
        pk = conjgrad_step(residual)
        prev_w = w
        prev_y = pk
        delta = 0

        while True:
            Ap = fisher_matrix_vector(prev_y)

            alpha = -torch.sum((prev_y * Q_k_1).T @ pk) / ((prev_y.T @ Q_k_1) ** 2)

            next_w = prev_w + alpha * prev_y
            diff = next_w - prev_w
            next_y = pk + diff

            eigval = (next_y.T @ Q_k @ next_y)[0][0].item()

            if eigval > prev_eigval or math.isclose(eigval, prev_eigval):
                break

            delta *= 2
            prev_eigval = eigval

        w = next_w
        prev_eigval = eigval

        try:
            Q_k = fisher_matrix(X, y, w)
            Q_k_1 = Q_k
            print("Iteration {}: eigval={:.4f}".format(it, eigval))
        except RuntimeError:
            pass


def conjgrad_step(residual):
    global Q_k_1, Q_k
    y = Q_k_1 @ residual
    a = y.T @ y

    if a <= epsilon:
        return y

    b = y.T @ res
    d = c = 0

    for _ in range(cg_iters):
        Ap = compute_hessian_vector(b, c, d, y)
        alpha = (a - b * c) / (c * Ap + epsilon)

        d = alpha * b + d
        c = sqrt(d**2 + epsilon)
        b = sqrt(b**2 + epsilon)

        if abs(Ap) <= epsilon:
            break

    d /= c
    alpha /= c

    pk = -Q_k_1.T @ residual + d * y

    return pk
```
这里的 `fisher_matrix` 函数用来计算 Fisher 矩阵，`compute_hessian_vector` 函数用来计算 Hessian-vector product，`projector`，`projection` 函数用来对搜索方向进行约束。
## 3.3 BFGS方法
BFGS方法利用海瑟矩阵（Hessian matrix）作为拟牛顿法的近似。在每一次迭代中，先计算梯度 $g_k=\nabla f(x^k)$ ，然后利用海瑟矩阵 $H_{k-1}$ 求出候选更新方向 $d_k=-H_{k-1}^{-1}g_k$ ，再用梯度下降的方式更新参数，直到收敛。公式如下：
$$x^{t+1}=x^t+\gamma_k d_k$$
其中，$\gamma_k$ 是给定的反向传播率。这个公式对应到代码实现就是：
```python
for i in range(iterations):
    grad = compute_gradient(X, y, w)  # 计算梯度
    s = search_direction(X, y, w, grad)  # 搜索方向

    # 没有完全求解的情况
    gamma = linesearch(X, y, w, grad, s)

    # 更新参数
    w += gamma * s
```
其中，`linesearch` 函数用来计算反向传播率 $\gamma_k$。