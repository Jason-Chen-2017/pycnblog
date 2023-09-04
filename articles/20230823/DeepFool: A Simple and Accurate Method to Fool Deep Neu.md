
作者：禅与计算机程序设计艺术                    

# 1.简介
  

DeepFool 是一种基于梯度的方法，可以用于对抗深度神经网络。它的名字来源于"fooling"这个动词的古意，即欺骗，让模型误认为目标类别为所选输入图像的标签。深度神经网络中的权重参数使得它很容易受到扰动的影响，而通过分析扰动来欺骗神经网络对于目标图像分类的结果是一个非常有效的手段。

DeepFool 算法最早提出是在 2015 年的 CVPR 会议上。当时，它已经被多个研究人员使用了起来，并且在许多任务中都取得了不错的效果。比如，它在 ImageNet 大赛中以第一名的成绩夺冠，在 AwA 数据集上的分类精度也高于其他同类方法。

今天，DeepFool 算法已经成为很多领域的标准技巧之一，如图像恶意检测、生成攻击样本、对抗训练等领域。虽然它仍然是一个新颖的算法，但它具有很强的实用性，能够快速准确地生成对抗样本。

# 2.相关工作
DeepFool 的主要优点就是简单而有效，它不需要知道目标模型的任何信息或参数，直接使用输入图像作为攻击对象，并迭代优化输入图像来逐渐逼近原始图像。

相关工作有三种类型：基于梯度的攻击；基于扰动的攻击；基于随机梯度的攻击。

## （1）基于梯度的攻击

基于梯度的攻击算法利用损失函数（loss function）的一阶导数或者二阶导数来衡量输入图像对目标类别的影响。然后选择沿着该方向最大化损失函数值，使得输出分类发生变化，这种攻击方式需要选择对抗样本和原始样本之间的差距，并且需要计算整个图像的所有像素点的梯度值，其时间复杂度较高。

## （2）基于扰动的攻击

基于扰动的攻击算法通过增加输入图像中的小扰动来改变模型的预测结果，例如，给图像添加少量随机噪声、旋转图像、剪切图像，以此来达到对抗目的。这种攻击方式简单、快速且隐蔽，但是由于没有考虑到图像内部的重要区域，因此往往效果不理想。

## （3）基于随机梯度的攻击

基于随机梯度的攻击算法通过从输入图像中抽取随机的扰动（perturbation），然后迭代更新图像，直至满足指定的扰动大小和扰动范围。这种攻击方式在一定程度上克服了基于扰动的攻击的缺陷，它可以在一定时间内完成对抗样本的构建，但由于其随机性可能会导致不同输入得到相同的对抗样本，而且迭代过程可能需要更长的时间。

# 3.DeepFool 算法
## （1）原理
DeepFool 方法的原理是，对于给定的目标类别 t ，定义一个罚项 $\mathcal{R}(x)$ 来描述对于当前输入图像 x 和分类器 f(x) 预测的每类的响应，其中 $f$ 表示一个深度卷积神经网络 (CNN)。$\mathcal{R}$ 函数定义如下：

$$\mathcal{R}(x)=\max_{\theta \in \Theta}I_{t}(\theta)(-1/\|\nabla_{\theta}\mathcal{L}_{\theta}(x,\hat{y})\|^2)$$

其中，$\theta$ 是网络的权重参数集合，$\Theta$ 表示所有可能的权重参数集合，$\mathcal{L}_{\theta}(x,\hat{y})$ 表示给定输入 $x$ 和真实标签 $\hat{y}$ 的损失函数。$(-1/\|\nabla_{\theta}\mathcal{L}_{\theta}(x,\hat{y})\|^2)$ 分子负号表示优化方向，分母 $(-1/\|\nabla_{\theta}\mathcal{L}_{\theta}(x,\hat{y})\|^2)$ 为距离因子，用来缩短对抗样本的搜索范围。

为了生成对抗样本，DeepFool 使用迭代优化的方式，一步步将原始图像 $x$ 逼近至其预测概率最大的那个类别。具体来说，设 $\bar{x}_{i+1}=x_i-\alpha_ix_i+\sum_{j=1}^m a_jx_j$，其中 $\alpha_i>0$ 为学习速率，$a_j\geqslant 0$ 为扰动参数，$j=1,\cdots,m$ 表示扰动个数。第 i 次迭代的目标是求 $\arg\max_{x'}I(\hat{y},h(x'))=-\frac{1}{\|\nabla_{x'}\mathcal{L}(x',\hat{y})\|}$。这里，$h$ 是输入图像 $x'$ 的分类结果，$\mathcal{L}$ 表示分类损失函数。最终 $m$ 个扰动参数 $\{a_j\}_{j=1}^m$ 将会决定着最终对抗样本的形状。

迭代过程中，当目标分类发生变化时，停止迭代。如果目标类别一直没有变化，则继续迭代，直至达到迭代次数或指定的搜索范围。

## （2）迭代优化
DeepFool 在迭代过程中，计算梯度 $\nabla_{\theta}\mathcal{L}_{\theta}(x,\hat{y})$ 需要完整计算代价函数 $\mathcal{L}_{\theta}(x,\hat{y})$，这一步耗时很长，并且占用内存资源过多。所以作者对代价函数进行估计，采用局部近似方法，只需要计算代价函数的一阶导数。首先计算损失函数的一阶导数：

$$\delta_i=\frac{1}{2}\left[\frac{\partial\mathcal{L}_{\theta}(x_i+\epsilon_iy_i,-)\partial y_i}{\partial x_i}-\frac{\partial\mathcal{L}_{\theta}(x_i,-)\partial y_i}{\partial x_i}\right]$$

其中，$\epsilon_i$ 为扰动因子，$y_i$ 为一组单位向量，代表改变量对目标函数的影响。

利用上面得到的一阶导数，DeepFool 可以计算 $\nabla_{\theta}\mathcal{L}_{\theta}(x,\hat{y})$ 。具体方法为，令 $\phi(\alpha)=\mathcal{L}_{\theta}(x_i+\alpha y_i,\hat{y})$，其中 $\alpha=\|\nabla_{\theta}\mathcal{L}_{\theta}(x,\hat{y})\|/n$，$n$ 为步长大小。那么：

$$\frac{d\phi(\alpha)}{d\alpha}=\frac{1}{n}[\mathcal{L}_{\theta}(x_i+(\alpha+\epsilon_i)y_i,\hat{y})-\mathcal{L}_{\theta}(x_i+(1-\epsilon_i)y_i,\hat{y})]-\frac{\epsilon_i-1+\epsilon_i}{\epsilon_i-\delta_i+1}$$

由此可得：

$$\frac{d\phi(\alpha)}{d\alpha}=[\mathcal{L}_{\theta}(x_i+(\alpha+\epsilon_i)y_i,\hat{y})-\mathcal{L}_{\theta}(x_i-(1-\epsilon_i)y_i,\hat{y})]-\frac{\epsilon_i-1+\epsilon_i}{\epsilon_i-\delta_i+1}$$

求得：

$$\alpha=-\frac{1}{\delta_i}\ln(\frac{\mathcal{L}_{\theta}(x_i+\epsilon_iy_i,\hat{y})-\mathcal{L}_{\theta}(x_i,(1-\epsilon_i)y_i,\hat{y})}{\mathcal{L}_{\theta}(x_i,-)+\epsilon_iy_i-\delta_iy_i})$$

也就是说，DeepFool 每次迭代的目标是寻找使得梯度 $\nabla_{\theta}\mathcal{L}_{\theta}(x,\hat{y})$ 中对应输入 $x_i$ 的系数尽可能小的解，即 $[-1/\|\nabla_{\theta}\mathcal{L}_{\theta}(x,\hat{y})\|,1]$ 之间的值。在求得相应的 $\alpha$ 时，注意到 $\delta_i=\frac{1}{2}\left[\frac{\partial\mathcal{L}_{\theta}(x_i+\epsilon_iy_i,-)\partial y_i}{\partial x_i}-\frac{\partial\mathcal{L}_{\theta}(x_i,-)\partial y_i}{\partial x_i}\right]$, $\epsilon_i$ 取值范围在 [0,0.5] 之间，所以 $\alpha$ 的计算不会超过限制。

## （3）扰动参数的选择
扰动参数 $a_j$ 的选择对 DeepFool 的性能有比较大的影响。文献中对 $a_j$ 的选择方法有两种，分别是：

1. Fixed step size：固定步长法。对每个扰动参数 $j$，确定步长 $\alpha_j$ ，使得对抗扰动 $\tilde{x}^{adv}_{ij}$ 最小化以下约束：

   $$\min_{\tilde{x}^{adv}_{ij}}\frac{1}{m}\sum_{l=1}^m I(\hat{y},h(\tilde{x}^{adv}_{ij}(l)))+\lambda J_{\ell}(a^{adv}_{ij}(l))$$

   这里，$J_\ell(x)$ 表示 Lp 范数，$\lambda$ 是超参数，$m$ 为扰动参数 $\{\alpha_j\}$ 中的元素数量。

2. Adaptive step size：自适应步长法。对于每一个扰动参数 $\ell$ ，先确定步长 $0<\alpha_j=\sigma^2_j\cdot r_j^\ell<r_j^\ell$ ，再计算对抗扰动 $\tilde{x}^{adv}_{ij}$ 并计算对应代价函数：

   $$J^{\ell}(a^{\ell})=\frac{1}{m}\sum_{k=1}^m\frac{1}{n}\sum_{l=1}^na^{adv}_{jk}(l)^P\mathcal{L}_{\theta}(\tilde{x}_{kl},\hat{y})+\lambda R_{\ell}^T\eta_{\ell}$$

   这里，$\sigma_j$ 为梯度下降步长的初始值，$\eta_j$ 为梯度下降步长的学习速率，$P$ 表示梯度下降步长的形状。

基于上面两种方法，文献还试验了多种参数选择策略，包括：

1. Linear increase：线性增加。即 $\{\alpha_j\}=(0,s,...,2s,...,2s-1,0)$.
2. Geometric increase：几何级数增加。即 $\{\alpha_j\}=(c_j,c_j^2,...,\lfloor\frac{(j+1)}{2}\rfloor c_j^2+1)$。
3. Reciprocal geometric increase：倒置几何级数增加。即 $\{\alpha_j\}=(c_j^(-1),c_j^(-2),...,\lfloor\frac{(j+1)}{2}\rfloor c_j^(-2)+1)$。
4. Logarithmic increase：对数级数增加。即 $\{\alpha_j\}=(e^{-j}, e^{-j+1},..., 1)$。
5. Exponential decrease：指数减少。即 $\{\alpha_j\}=(\exp(-1),..., \exp(-N))$。

这些策略均基于“小尺度”的参数选择。另外，为了防止出现拉伸现象，还可以加入随机噪声来控制扰动参数的范围，如加上固定的幅度和方向的扰动。

## （4）数学证明
### （4.1）数学期望
假设输入图像 $x_i$ 对目标类别的分类误差为 $I(\hat{y},h(x_i))$，而扰动参数 $\ell$ 的误差 $\epsilon_i\Delta_i$ 。对输入图像 $x_i$ 的第 $j$ 个元素 $x_i[j]$ 求导，有：

$$\frac{\partial I(\hat{y},h(x))}{\partial x_i[j]}=\frac{\partial h(x)-1}{h(x)}\frac{\partial\mathcal{L}_{\theta}(x,\hat{y})}{\partial x_i[j]}+\mathcal{H}_{\theta}(x)[j]\frac{\partial\mathcal{L}_{\theta}(x,\hat{y})}{\partial x_i[j]}$$

其中，$\mathcal{H}_{\theta}(x)$ 表示输入 $x$ 的 Hessian 矩阵，是一个 $w\times w$ 矩阵，对角线元素为 $0$。假设沿着梯度方向，$z_i=-\frac{\partial I(\hat{y},h(x))}{\partial x_i[j]}$。那么，对于扰动参数 $\ell$ ，有：

$$I(\hat{y},h(x_i-\epsilon_i\Delta_i))[j]=I(\hat{y},h(x_i))-\epsilon_i z_i[j]$$

设扰动后图像 $x_i-\epsilon_i\Delta_i$ 的损失函数为 $J_{\ell}(a_{ij}(l))$，则：

$$J_{\ell}(a_{ij}(l))=\frac{1}{n}\sum_{k=1}^n\frac{1}{n}\sum_{l=1}^na^{adv}_{ik}(l)^P\mathcal{L}_{\theta}(\tilde{x}_{lk},\hat{y})+\lambda R_{\ell}^T\eta_{\ell}$$

求导：

$$\frac{\partial J_{\ell}}{\partial a_{ij}(l)}=\frac{1}{n}\sum_{k=1}^n\frac{1}{n}\sum_{l=1}^na^{adv}_{ik}(l)^Pa_{ij}(l)\mathcal{L}_{\theta}(\tilde{x}_{kl},\hat{y})+\lambda R_{\ell}\eta_{\ell}$$

设 $B_{\ell}$ 表示误差函数 $I(\hat{y},h(x_i-\epsilon_i\Delta_i))[j]$ 的一阶导数，$\mathcal{G}_{\ell}^{adv}$ 表示扰动后图像 $x_i-\epsilon_i\Delta_i$ 的损失函数的一阶导数。那么，有：

$$\frac{\partial J_{\ell}}{\partial a_{ij}(l)}=\frac{1}{n}\sum_{k=1}^n\frac{1}{n}\sum_{l=1}^na^{adv}_{ik}(l)^PB_{\ell}\mathcal{G}_{\ell}^{adv}\eta_{\ell}$$

这里，$\mathcal{G}_{\ell}^{adv}$ 有：

$$\frac{\partial\mathcal{L}_{\theta}(\tilde{x}_{kl},\hat{y})}{\partial\tilde{x}_{kl}[j]}=-\frac{1}{2}\left[x_i[j]+\epsilon_i\Delta_i[j]-\tilde{x}_{kl}[j]\right]$$

令 $\omega_{\ell}=\mathcal{G}_{\ell}^{adv}/B_{\ell}$，即 $\omega_{\ell}$ 是扰动前后两张图像的梯度之比。那么，上面的表达式可以写作：

$$\frac{\partial J_{\ell}}{\partial a_{ij}(l)}=\frac{1}{n}\sum_{k=1}^n\frac{1}{n}\sum_{l=1}^na^{adv}_{ik}(l)^Pb_{ij}(l)\omega_{\ell}\mathcal{G}_{\ell}^{adv}\eta_{\ell}$$

当 $\eta_{\ell}>0$ 时，$\mathcal{G}_{\ell}^{adv}$ 正比于 $\eta_{\ell}$ ，因此 $\omega_{\ell}$ 趋向于正值；当 $\eta_{\ell}$ 不变时，$\mathcal{G}_{\ell}^{adv}$ 趋向于零，因此 $\omega_{\ell}$ 趋向于无穷大。因此，当 $\eta_{\ell}$ 不变时，$a^{adv}_{ij}(l)>0$，对于任意 $\ell$ ，根据最优性条件，$a^{*}_{ij}$ 是使得 $J_{*}(a_{ij}(l))$ 最小的解，即 $J_*$ 实际上是扰动后的损失函数。

### （4.2）精心设计的正则化项
为了避免过拟合，作者在损失函数中引入了正则化项：

$$J_{final}(\hat{y},\pi(x;a_{ij}(l)),a_{ij}(l))=-\frac{1}{n}\sum_{k=1}^n\sum_{l=1}^na^{adv}_{ik}(l)^P\mathcal{L}_{\theta}(\tilde{x}_{kl},\hat{y})+\lambda\Omega_{\ell}(a_{ij}(l))$$

这里，$\pi(x;a_{ij}(l))$ 表示对抗扰动 $\tilde{x}^{adv}_{ij}(l)$ 。$\Omega_{\ell}(a_{ij}(l))$ 为正则化项。有三种正则化项选择策略：

1. $\Omega_{\ell}(a_{ij}(l))=a_{ij}(l)^2$
2. $\Omega_{\ell}(a_{ij}(l))=(1-a_{ij}(l))^2$
3. $\Omega_{\ell}(a_{ij}(l))=\beta\|a_{ij}(l)\|^q$

其中，$\beta$ 为超参数，$q$ 为 Lq 范数。三个策略都能控制扰动的大小和范围。

## （5）实施细节
实现细节：

1. 测试图片和 $\hat{y}$ 通过命令行参数传入。
2. 参数 $\{a_{ij}(l)\}_{ij=1}^n\subseteq[0,1]^m$ 初始化为 0 或随机。
3. 更新规则：

   $$a^{adv}_{ij}(l+1)={\rm arg\,min}_{\eta}J_{\ell}(a_{ij}(l)+\eta)$$

   其中，$\eta$ 是步长。

4. 直到收敛或者迭代次数超过限制。