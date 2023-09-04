
作者：禅与计算机程序设计艺术                    

# 1.简介
  

支持向量机（Support vector machines, SVMs）是一种监督学习分类方法，其目的是找到一个分离超平面将数据划分为不同的类别。而支持向量回归(Support vector regression, SVR)则是在该基础上对输出变量进行预测。本文通过一个案例系统地介绍了SVM、SVR算法的具体概念，并用Python实现了一个案例，展示了SVM、SVR算法的具体应用场景及优点。
# 2.支持向量机 (Support Vector Machine, SVM) 
## 2.1 概念
支持向量机（Support vector machine, SVM）是一种二类分类模型，它通过优化最大间隔边界，来学习输入空间中的样本。最大间隔边界使得分类边界尽可能远离噪声点，并且在保证所有数据的正确分类的同时，最小化支持向量到边界之间的距离。

支持向量机由两部分组成：
- 线性支持向量机(Linear support vector machine, SVM with linear kernel): 支持向量机的一种常用形式，直接寻找最优超平面。它的学习任务就是寻找一个定义域为特征空间的超平面，通过几何间隔最大化或者几何间隔和数据距离之和最小化的方式获得这个超平面。这样的超平面将输入空间分成两部分，一部分对应于负例，另一部分对应于正例。
- 非线性支持向量机(Nonlinear support vector machine, SVM with non-linear kernel): 在输入空间中存在着复杂的非线性关系时，可以使用非线性核函数来构造SVM模型。常用的非线性核函数有径向基函数(Radial basis function)， sigmoid 函数，高斯函数等。采用非线性核函数可以获得比线性核更强大的分类能力。

## 2.2 线性SVM
### 2.2.1 原始问题描述
给定输入空间X∈Rn和输出空间Y={-1,+1}，我们的目标是学习一个映射φ: X→Y，其中φ(x)≥0当且仅当y=1，即判断实例x属于正类；否则，φ(x)<0，即判断实例x属于负类。

假设训练集T={(x1,y1),(x2,y2),...,(xn,yn)},其中xi∈Rn是实例的特征向量，yi∈{-1,+1}是实例的标签，i=1,2,...,n表示实例个数。

原始问题需要确定一个超平面φ,通过计算得到使得Φ(xi)=−1或者1的概率最大化的目标函数，即：


其中，θ=(w,b)是超平面的参数向量，wi∈R是超平面的法向量，b∈R是超平面的截距项。εi(1>=i>=n)称为松弛变量或间隔，表示训练样本x[i]到超平面的距离。Φ(xi)是超平面上的预测值，如果σ(wi*xi+bi)>0，则预测为1，否则预测为-1。

### 2.2.2 拉格朗日函数
为了求解原始问题，我们可以使用拉格朗日函数。首先我们把原始问题写成如下标准型：


其中，λi(1>=i>=n)称为拉格朗日乘子，μi(1>=i>=n)称为松弛变量。λi是拉格朗日乘子，可以使对偶问题(dual problem)求解变得容易。

### 2.2.3 对偶问题
原始问题是一个求解凸二次规划问题，但是为了便于分析、处理和解码，通常会选择约束最严格的合页损失函数作为目标函数。因此，我们可以构造等价的对偶问题。首先，我们考虑目标函数关于θ的导数为零的问题：


然后我们考虑约束条件，也就是拉格朗日乘子是什么的问题：


以上两个约束条件联立起来得到对偶问题：

&\quad \quad %5Chboxmin_{b,\boldsymbol{\epsilon}}\quad&\qquad&\qquad&\qquad&%5CEpsilon\equiv%20%5CHbar%20\Omega%20%5CBigg%20\Lambda%5E%7B-1%7D%5B%5Chat%7By%7D%28%5Cmathbf{x}%29%5Cgeqslant0%5D%20%5E%7Bm-1%7D%20\sum_{j=1}^m\sigma(\kappa_jy_j%20%5CBigg%20\mu_jy_j)+\frac{1}{2}\sum_{i=1}^{n}(\boldsymbol{\epsilon}_{i%2Bj}-b)\sigma(\mu_iy_i%20%5CBigg%20\sigma_i%28y_i%20\boldsymbol{x}_i%5ET%5E%7BH%7DR%5E%7Bk-1%7Dx_i%5E%7BT%7DH%7Dy_i)-\frac{1}{2}\sum_{i=1}^{n}\boldsymbol{\epsilon}_{ii}%5E%7B2%7D+\frac{1}{2}\rho%5ET%5E%7BH%7DR%5E%7Bk-1%7D%5CEpsilon-c&plus;\gamma%5EB%5Cleft(\sum_{i=1}^{n}\boldsymbol{\epsilon}_{ii}\right),&%20%5Cforall%20i%3D1%2C2%2C%20%5Cldots%2Cn,%20%5Cforall%20j%3D1%2C2%2C%20%5Cldots%2Cm.&\quad\text{(Primal problem)} \\
&\quad\qquad&\qquad&\qquad&\qquad&\qquad&\qquad&\qquad&\quad\text{(Dual problem)}