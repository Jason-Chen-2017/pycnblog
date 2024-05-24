
作者：禅与计算机程序设计艺术                    

# 1.简介
  

支持向量机（Support Vector Machine，SVM）是一种用于分类和回归的机器学习模型，其特点在于通过构造超平面将数据划分到多个空间中，使得每一个数据点都在超平面的正负方向上。因此SVM可广泛应用于模式识别、图像处理、生物信息、文本分类、网络安全等领域。

# 2.Basic Concepts and Terminology
## 2.1 Supervised Learning Problem Formulation
SVM作为监督式学习方法，所要解决的问题可以描述为：给定一个训练样本集合$D=\{(x_i,y_i)\}_{i=1}^{n}$，其中$x_i\in R^p$表示输入变量，$y_i \in {-1,+1}$表示输出变量(类别标签)，试求解能够最大化训练集目标函数$F(w,b)$，即:
$$
\max_{w,b}\sum_{i=1}^ny_i(wx_i+b)-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_jx_i^Tx_j
$$
其中$\alpha_i>0$, $\forall i$. $F(w,b)$为经验风险(empirical risk)，由损失函数构成。

## 2.2 Optimization Objective Function
为了找到全局最优解，可以采用结构风险最小化原则(structural risk minimization principle)。结构风险最小化原则认为，对优化问题而言，最小化经验风险同时满足约束条件是最优的，即：
$$
R_{\text{exp}}(\hat{f})+\lambda J(\theta)=\sum_{i=1}^nL(y_i,\hat{f}(x_i))+\lambda R(\theta),\\
\text{where }\hat{f}=\sum_{i=1}^m\alpha_i y_i K(x_i,x'),\text{ where }K(x_i,x')= \phi(x_i)^T\phi(x')
$$
结构风险最小化原则可以通过正则化的方式进行规范。

通过拉格朗日乘子法，可以证明SVM的目标函数的最优化等价于下列KKT条件：
$$
\begin{cases}
    \nabla L(w,b)+\alpha_i y_ix_i^T=0&\forall i,\\
    y_i(wx_i+b)\geq 1-\xi_i&\forall i,\alpha_i >0,\xi_i=0,\forall i.\\
    y_i(wx_i+b)< 1-\xi_i&\forall i,\alpha_i =0,\xi_i<\zeta_i,&\forall i.\xi_i <\frac{C}{\nu},\zeta_i=0,\forall i.\\
    0<\alpha_i&\forall i,1\leqslant i\leqslant n.\\
    \alpha_i+\zeta_i\leq C,\forall i.&\zeta_i=\xi_i-1/\nu,0\leqslant i\leqslant n.
\end{cases}
$$
其中$\psi(u)=log(1+e^{u})$是一个双曲正切函数。KKT条件保证了优化目标的充分必要条件，即:

1. 原点在约束边界上；
2. 对偶问题的约束被滞后；
3. 满足充分弱覆盖。

## 2.3 Hinge Loss
SVM中的损失函数定义为软间隔分类器下的凸二阶范数，称为“0-1损失”或“Hinge Loss”。其定义如下：
$$
L(y_i,(wx_i+b))=\max(0,1-(wy_i+(b)))= \max(0,1-y_iw^tx_i-b).
$$
当$y_iw^tx_i+b\geq 1$时，$L(y_i,(wx_i+b))=0$；否则，$L(y_i,(wx_i+b))$不等於$0$。

# 3. Core Algorithm and Theory
## 3.1 Primal Optimization
首先，考虑线性不可分情况下的拉格朗日函数，其对应的原始问题为：
$$
\min_{\alpha}\quad\frac{1}{2}\|\bar{\alpha}\|^2 + \sum_{i=1}^n\alpha_i-\sum_{i=1}^n\alpha_iy_i\left<\bar{\alpha}_i, x_i\right>,\\
\text{s.t.}~\alpha_i\geq 0, \forall i.
$$
利用KKT条件，可以得到最优化问题的解为：
$$
\max_{\alpha} \quad\sum_{i=1}^n\alpha_i-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_jx_i^Tx_j,~s.t.~\sum_{i=1}^n\alpha_iy_i=0,\alpha_i\geq 0, \forall i.
$$

根据拉格朗日乘子法的解析表达式，可以得到最优解：
$$
\hat{\alpha}=({\bf X}^Ty+\lambda I)^{-1}(\bf y),
$$
其中，${\bf X}$为$x_i$组成的矩阵，$\lambda$为松弛变量。

当训练样本满足$y_i(wx_i+b)>1-\xi_i$的情况时，$\xi_i$取值为0。当训练样本满足$y_i(wx_i+b)< 1-\xi_i$的情况时，$\xi_i$大于零，且$\frac{1}{\nu}\leqslant \xi_i \leqslant \frac{C}{\nu}$, 称为违背边界条件的例子。

假设存在违背边界的实例点，也就是$\frac{1}{\nu}>\xi_i>\frac{C}{\nu}$, 此时需要引入松弛变量$\epsilon_i$，再重新计算拉格朗日函数：
$$
\min_{\alpha}\quad\frac{1}{2}\|\bar{\alpha}\|^2 + \sum_{i=1}^n\alpha_i-\sum_{i=1}^n\alpha_iy_i\left<\bar{\alpha}_i, x_i\right>-\sum_{i=1}^n\epsilon_i,\text{ s.t. }\alpha_i\geq 0, \forall i; \\
\xi_i\geq \frac{C}{\nu}-\epsilon_i, ~ \forall i;\epsilon_i\geq 0,\forall i.
$$
得到最优化问题：
$$
\max_{\alpha} \quad \sum_{i=1}^n\alpha_i-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_jx_i^Tx_j + \sum_{i=1}^n\epsilon_i + \frac{C}{\nu}\sum_{i=1}^n\xi_i\\
\text{s.t. }\sum_{i=1}^n\alpha_iy_i=0, \alpha_i\geq 0, \forall i; \epsilon_i\geq 0,\forall i;\xi_i\geq 0,\forall i.
$$
然后，引入松弛变量$\epsilon_i$ 和 ${\bf z}$向量，令：
$$
g_i=-\left[\begin{array}{c}\psi(\xi_i)\\ y_i\psi(-\xi_i) \end{array}\right],~~ h_i=\left[y_i\psi(-\frac{C}{\nu}), \frac{C}{\nu}\psi(-\frac{C}{\nu})\right].
$$
那么KKT条件可以写为：
$$
\begin{cases}
    g_i^T\bar{\alpha}_i+\frac{1}{2}\bar{\alpha}_ig_i^T\bar{\alpha}_i=h_i^T\bar{\alpha}_i &\forall i,\\
    \bar{\alpha}_i\geq 0&\forall i,\\
    \left<g_i,z\right>_+=\bar{\alpha}_i& \forall i, j=1,2;\\
    \epsilon_i+1/C\xi_i\geq 0&\forall i, \\
    \bar{\beta}_i=1/C&\forall i=1,..., m.
\end{cases}
$$

## 3.2 Dual Optimization
在此处我们重点研究对偶问题。首先，由最优化问题可以导出拉格朗日函数：
$$
L(\bar{\alpha}, \bar{\beta})=\sum_{i=1}^n\alpha_i-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_jx_i^Tx_j+\sum_{i=1}^m\beta_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\beta_i\beta_jy_iy_jx_i^Tx_j,
$$
其中，$\alpha=(\alpha_1,...,\alpha_n)^T$和$\beta=(\beta_1,...,\beta_m)^T$是拉格朗日乘子。所以：
$$
\nabla_\beta L(\bar{\alpha}, \bar{\beta})=\sum_{i=1}^n\beta_iy_jx_i^Tx_j- \sum_{i=1}^m\beta_iy_jx_i^Tx_j =0,~~~\beta_i=0\quad \forall i\neq k,~~k=1,..,m
$$
该方程说明对偶问题保证的是系数$\beta_k$的优化不受其他任何参数的影响。注意到$\sum_{i=1}^n\alpha_iy_i=0$，因此：
$$
\begin{align*}
&\nabla_\alpha L(\bar{\alpha}, \bar{\beta})=\sum_{i=1}^n\alpha_iy_jx_i^Tx_j+\sum_{i=1}^n\beta_iy_jx_i^Tx_j=0,\\\
&\alpha_i\geq 0,\forall i.
\end{align*}
$$
再用拉格朗日乘子法导出对偶问题的解：
$$
\bar{\alpha}=({\bf X}^Ty+\lambda\bar{\beta})^{-1}(\bf e_n),~~ \bar{\beta}=({\bf Y}^TX+\lambda\bar{\alpha})^{-1}(\bf e_m),~~~\bar{\alpha}=(\alpha_1,\alpha_2,...\alpha_n)^T,~\bar{\beta}=(\beta_1,\beta_2,...,\beta_m)^T.
$$
其中，$\bf e_n=(1,1,...,1)^T$,$\bf e_m=(1,1,...,1)^T$为单位向量。这样就得到了SVM对偶问题的解。

最后，讨论核技巧。核技巧通过非线性变换把输入空间映射到高维特征空间，从而避免使用原空间内的核函数。如果输入空间是高维欧式空间，那么对应的核函数就是径向基函数。具体过程如下：

1. 定义核函数$K(x,z):R^d\times R^d\rightarrow R$。例如，$K(x,z)=\exp(-\gamma ||x-z||^2)$。这里，$d$是输入空间的维度，$\gamma>0$是核函数的超参数。

2. 在对偶问题中，先将$X$映射到高维空间$\mathcal{H}$：

   $$
   \mathcal{H}:=\left\{(\phi(x_1),...,\phi(x_n)),x\in R^d\right\}.
   $$
   
   其中，$\phi:\mathcal{X}\rightarrow\mathcal{H}$是映射函数。

3. 再定义核矩阵$K_{\mathcal{H}}\in R^{\vert\mathcal{H}\vert\times \vert\mathcal{H}\vert}$：

   $$
   K_{\mathcal{H}}:= \left[K(x_i,x_j)\right]_{ij}=[K(\phi(x_i),\phi(x_j))]_{ij}
   $$
   
   其中，$x_i$对应于$\phi(x_i)$。这样就可以利用核技巧来提升SVM的性能。

4. 当输入空间$\mathcal{X}$很高维时，核矩阵$K_{\mathcal{H}}$占用空间过大。因此，可以在核矩阵的构造过程中采用一些技巧来减小它所占用的空间。

   a. 对称半正定矩阵。如果$K_{\mathcal{H}}$的某些对角元素$K_{\mathcal{H}}_{ii}$存在，使得$K_{\mathcal{H}}_{jj}>0$，则称$K_{\mathcal{H}}$是对称半正定矩阵。这种核矩阵可以使用更有效的方法来构造。例如，可以利用Tikhonov正则化来构造：

      $$
      K_{\mathcal{H}}=\text{diag}(k_1,k_2,...,k_n)(\Phi\Phi^T+\sigma I)^{-1},~~~k_i\in R,~~~\sigma>0.
      $$
      
      其中，$\Phi$是输入数据的矩阵，其每行对应于输入空间中的一个向量。$\sigma$是正则化参数。

      b. 可以采用随机梯度下降来构造核矩阵。这种方法不需要直接计算出完整的核矩阵。首先，构造核函数对应的拉普拉斯算子：

      $$
      \mathcal{L}_K=\left(\begin{matrix}
          (-\frac{1}{2}\Gamma\Gamma^T+\sigma I)&I\\
          I&(-\frac{1}{2}\Lambda\Lambda^T+\sigma I)
        \end{matrix}\right),~~~\sigma>0.
      $$
      
      其中，$\Gamma=\left(\begin{matrix}K(x_1,x_1)\\K(x_2,x_2),\cdots,K(x_n,x_n)\end{matrix}\right)$，$\Lambda=\left(\begin{matrix}K(y_1,y_1)\\K(y_2,y_2),\cdots,K(y_m,y_m)\end{matrix}\right)$。

      然后，随机初始化核矩阵$K_{\mathcal{H}}$，并通过随机梯度下降更新它：

      $$
      K_{\mathcal{H}}^{(t+1)}=(\Sigma_1+\Lambda_1\Sigma_2)^{-1}\Omega_{\Delta t}\\
      \Sigma_1^{-1}\Sigma_2^{-1}\mathcal{L}_K\left(K_{\mathcal{H}}^{(t)}\right)K_{\mathcal{H}}^{(t)}+\sigma I\left(K_{\mathcal{H}}^{(t)}\right)K_{\mathcal{H}}^{(t)}
      $$
      
      其中，$\Sigma_1$和$\Sigma_2$分别是训练数据$X$的协方差矩阵和验证数据$Y$的协方差矩阵；$\Lambda_1$和$\Lambda_2$分别是训练数据$X$的特征值和验证数据$Y$的特征值；$\Omega_{\Delta t}$是关于$\Delta t$的随机波动项；$\mathcal{L}_K$是拉普拉斯算子；$K_{\mathcal{H}}^{(t)}\in R^{\vert\mathcal{H}\vert\times \vert\mathcal{H}\vert}$是第$t$次迭代时的核矩阵。

# 4. Example Code
## SVM for Binary Classification in Python
```python
import numpy as np

class SVM:
    def __init__(self, kernel="linear", C=1.0, epsilon=0.1, max_iter=100):
        self.kernel = kernel # choice of kernel function ("linear" or "rbf")
        self.C = C           # soft margin parameter
        self.epsilon = epsilon   # tolerance value to check convergence
        self.max_iter = max_iter
    
    def fit(self, X, y):
        n, p = X.shape
        
        if isinstance(y, list):
            y = np.array([int(label) for label in y])

        self._gram_matrix(X)
        self._solve_dual()
        
        iters = 0
        old_objective = float("inf")
        new_objective = self._compute_objective()

        while abs(old_objective - new_objective) > self.epsilon and iters < self.max_iter:
            alpha_changed = 0

            # Passes over all instances in dataset
            for i in range(n):
                E_i = self._E(i)[0]

                # Computes step size alpha_i using line search method
                alpha_i = self.alpha[i]

                # Makes sure that no step is taken outside the limits
                alpha_low = max(0, alpha_i - self.C)
                alpha_high = min(self.C, alpha_i + self.C)
                
                if alpha_i == alpha_low:
                    l, h = alpha_low, alpha_high
                elif alpha_i == alpha_high:
                    l, h = alpha_high, alpha_low
                else:
                    l, h = alpha_i - self.C, alpha_i + self.C

                c1 = E_i - y[i]*self.bias - 1
                c2 = E_i - y[i]*self.bias + 1
                
                if alpha_i == l:
                    dalpha_i = min(self.C, c1)/self.t[i]
                elif alpha_i == h:
                    dalpha_i = max(-self.C, c2)/self.t[i]
                else:
                    dalpha_i = c2*y[i]/self.t[i]
                    
                # Updates alpha_i with direction and step size dalpha_i
                alpha_new_i = alpha_i - dalpha_i

                if alpha_new_i <= alpha_low:
                    alpha_new_i = alpha_low
                elif alpha_new_i >= alpha_high:
                    alpha_new_i = alpha_high

                if alpha_new_i!= alpha_i:
                    alpha_changed += 1
                    alpha_i = alpha_new_i

                    # Recomputes Gram matrix and bias term after update
                    self._update_Gram_and_bias(X[i], y[i], alpha_i)
            
            # Update parameters beta based on updated alphas
            self._update_params()
            
            # Check objective value change to determine stopping condition
            old_objective = new_objective
            new_objective = self._compute_objective()
            
            print("Iteration:",iters,"Objective Value:",new_objective)
            iters += 1
            

    def _gram_matrix(self, X):
        n = len(X)
        gamma = 1 / X.var() ** 2

        if self.kernel == 'rbf':
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    diff = X[i,:] - X[j,:]
                    K[i,j] = np.exp(-gamma * np.dot(diff, diff))
        else:
            K = np.dot(X, X.T)

        self.K = K


    def _compute_objective(self):
        obj = 0.5 * np.dot(self.alpha, np.dot(self.K, self.alpha)) - self.bias
        return obj
    
        
    def _update_Gram_and_bias(self, xi, yi, alpha_i):
        beta_old = self.bias
        
        if yi*(np.dot(self.K[:,:], self.alpha) + beta_old)*yi > 1:
            self.bias -= ((self.K[self.sv_index,:][:,self.sv_index]).sum()*self.alpha[self.sv_index]).sum()\
                        + yi*alpha_i + beta_old - yi*((self.K[self.sv_index,:][:,self.sv_index]*self.alpha[self.sv_index]).sum())
                        
        sv_index = [idx for idx,value in enumerate(self.alpha) if value > 0]
        
        G = self.G[sv_index,:]
        a = self.alpha[sv_index]
        A = self.A[sv_index,:]
        L = self.L[sv_index,:]
        
        Gamma = np.vstack((-a**2*G.T + a*A.T + a*L.T, -(a*G.T + a*A.T + a*L.T))).T
        Lambda = np.vstack((a*G.T + a*A.T + a*L.T, -(a**2*G.T + a*A.T + a*L.T))).T
        
        self.K[:,:] = (np.linalg.inv(self.sigma*np.eye(len(self.sv_index))+Gamma)).dot(Lambda)
        
        
    def _solve_dual(self):
        n, m = self.K.shape
        
        # Find support vectors and their corresponding indices
        sv = []
        sv_index = []
        alpha = []
        G = []
        A = []
        L = []
        
        for i in range(n):
            if self.y[i] == 1:
                alpha_i = self._svm_qp(i)
                
                if alpha_i > 1e-10:
                    sv.append(i)
                    sv_index.append(i)
                    alpha.append(alpha_i)
                    G.append(self.K[i,:])
                    A.append(self.y[i]*self.K[:,i])
                    L.append(np.dot(self.K[:,i],self.alpha))
        
        self.sv = sv
        self.sv_index = sv_index
        self.alpha = np.array(alpha)
        self.G = np.vstack(G)
        self.A = np.vstack(A)
        self.L = np.vstack(L)
        
        # Initialize parameters beta and bias terms
        self._update_params()

    
    def _svm_qp(self, i):
        Q = np.outer(self.y[i]*self.K[:,i],self.y[i]*self.K[:,i])
        q = -np.ones(Q.shape[0])*self.y[i]*self.K[i,i]
        G = np.vstack((-np.eye(len(self.alpha)), np.eye(len(self.alpha))))
        h = np.hstack((np.zeros(len(self.alpha)), self.C*np.ones(len(self.alpha))))
        A = np.vstack((np.ones(1), -np.ones(1)))
        b = np.array([[0],[1]])
        solvers.options['show_progress'] = False
        res = linprog(q, A_ub=G, b_ub=h, A_eq=A, b_eq=b)
        
        return res.x[0]

        
    def predict(self, X):
        pred_labels = []

        for i in range(len(X)):
            fx = self._decision_function(X[i])
            pred_labels.append(fx)

        return pred_labels


    def _decision_function(self, x):
        return sum([(self.alpha[i] * self.y[i] * self.kernel(self.K[i, :], x)) for i in self.sv_index]) + self.bias



def rbf_kernel(x, z):
    gamma = 1 / x.var() ** 2
    return np.exp(-gamma * np.dot(x - z, x - z))


if __name__=="__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data[:100,:]
    y = iris.target[:100]
    
    clf = SVM(kernel='rbf', C=1.0, epsilon=1e-4, max_iter=1000)
    clf.fit(X, y)
    preds = clf.predict(X)
    accuracy = np.mean(preds==y)
    print("Accuracy:",accuracy)
```