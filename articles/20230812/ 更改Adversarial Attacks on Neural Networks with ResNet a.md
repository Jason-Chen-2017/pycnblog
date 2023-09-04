
作者：禅与计算机程序设计艺术                    

# 1.简介
  

近年来，人们发现神经网络（NN）很容易受到对抗性攻击，对抗性攻击是指通过对NN的输入、输出或权重进行恶意修改来干扰模型的预测结果，导致错误分类或预测错误。目前，针对NN的对抗性攻击方法分成两大类：白盒攻击与黑盒攻击。白盒攻击利用对网络结构、参数、训练数据等的分析信息进行攻击，黑盒攻击则通过对攻击者不知道的信息进行攻击。NN的防御性攻击研究也主要集中在白盒攻击上，而黑盒攻击相关的研究也越来越少。
最近，作者团队借鉴了FGSM、PGD、DeepFool等多种白盒攻击方法的思路，提出了一个新颖的ResNet的防御性攻击模型——RSD-PGD，其相对于传统的ResNet防御方法有较大的优势。本文的目标是系统阐述RSD-PGD攻击方法的原理及其安全性，并试图总结出这种攻击方法在ResNet上的优势。
# 2.基本概念术语说明
对抗性攻击：通过对神经网络的输入、输出或权重进行恶意修改来干扰模型的预测结果，导致错误分类或预测错误。
白盒攻击与黑盒攻击：白盒攻击利用对网络结构、参数、训练数据等的分析信息进行攻击，而黑盒攻击则通过对攻击者不知道的信息进行攻击。
目标函数：定义攻击过程中希望达到的目的，例如使得预测错误，或者最大化预测概率等。
Adversarial Example：对抗样本就是一张图片，对抗攻击的目的就是把正常的图片变成一个对抗样本。当给定一个正常样本X，可以用梯度下降法或其他优化算法来最小化目标函数Loss(X) ，从而生成一个对抗样本X'，使得Loss'(X')尽可能地接近于Target Loss，即X'更加适合作为对抗样本对模型进行攻击。
梯度反转：在对抗性攻击的过程中，每一步优化都需要同时考虑正确标签和错误标签。这个过程称作梯度反转，可以让优化器不断调整输入向量，直至目标函数改变方向。因此，梯度反转对生成的对抗样本有重要影响，它决定了攻击成功与否。
批量大小：一般情况下，批量大小对应于一次迭代中处理的数据个数。一次迭代需要计算梯度值，如果批量太小，则每个批次的梯度值计算代价高昂；如果批量太大，则效率低下，无法充分利用GPU资源。
损失函数：指模型训练过程中使用的用于衡量预测结果误差的函数。比如，交叉熵损失函数通常会被用来训练分类模型。在对抗性攻击中，还需要设计一种新的损失函数，来尽量减轻模型对抗攻击的效果。
输入范畴：输入范畴是指攻击者能够控制的输入值范围。如图像、文本等不同类型数据的输入，都属于不同的范畴。
超参数：是指模型训练过程中出现的参数，如学习率、优化算法、权重衰减率等。
正则化项：是指用于限制模型复杂度的手段。如L2正则化用于防止过拟合，L1正则化用于避免稀疏参数。
基于扰动的对抗攻击：采用随机扰动的方式对输入向量进行扰动，来产生对抗样本。
微步扰动：微步扰动是指攻击者设置很小的步长，每一次的更新幅度就比较小，这样既能够增加攻击的难度，又可以快速得到有效的结果。
# 3.核心算法原理和具体操作步骤以及数学公式讲解
## (1). RSD-PGD 算法流程图

图中的Input为原始输入x，Target为模型应该预测的类别，$\epsilon$ 为攻击步长，$K$ 表示重复攻击次数。$w_i$ 是待学习的参数，在这里假设为ResNet的weights。FGSM是一种常用的梯度上升方法，用来求取当前点到目标点的梯度方向，然后沿着该方向进行一定步长的更新，来产生对抗样本。

## (2). RSD-PGD 算法步骤
### 求解梯度方向 $\theta = \nabla L(\hat{y}, y)$
计算梯度: $g_{adv} = \nabla_\theta E_{\eta}(w_i+\eta\cdot g_{loss})$
其中$E_{\eta}$表示随机扰动后的参数估计。$L(\hat{y}, y)$表示损失函数，由模型输出$\hat{y}$ 和真实类别 $y$ 决定。在本文中，$L$ 函数是交叉熵函数，$g_{loss}$ 表示的是模型输出向量 $\hat{y}$ 的梯度，可以通过梯度反转方法求得。
### FGSM算法求解步长 $\eta$
计算步长: $\eta=\frac{\epsilon}{||g||}$
其中$g$ 表示梯度方向，$\epsilon$ 为攻击步长。
### 对抗扰动 $\Delta x = \eta\cdot g_{adv}$
将步长与梯度方向乘积作为攻击扰动，得到对抗扰动。
### 更新参数 $w_i+\Delta x$
将对抗扰动与原始参数值加和作为下一次攻击的初始参数。

### 重复 K 次攻击
最后，K次对抗扰动后，返回最优对抗样本。

## (3). RSD-PGD 算法数学公式
$w^\prime_i=argmin_w L(f(w), y)+\lambda RSD(\theta,\rho_k,\alpha_k)(w)$

其中$L$表示损失函数，$f(w)$表示模型，$y$表示标签，$\lambda$为惩罚系数，$\rho_k$和$\alpha_k$分别为参数$(w^{\rm k})$的标准差和均值的模。

RSD($\theta,\rho_k,\alpha_k$)表示软化正则化项，有以下几种选择：

1.$RSD(Lipshitz,$ $\alpha): RSD(L, \alpha)=\frac{\alpha}{2}\sum_{l=1}^{L} ||w_l||^2_2+(\frac{1-\alpha}{\alpha})\left\|w^{\top}\right\|^{2}_2$

   $RSD(L, \alpha): RSD(L, \alpha)=\frac{\alpha}{2}\sum_{l=1}^Lw_l^T w_l + (\frac{1-\alpha}{\alpha})\left\|w^{\top}\right\|_F^2$, Frobenius norm.
   
2.$RSD(\beta_k,\gamma_k):\quad RSD(\beta_k,\gamma_k)=\frac{\gamma_k}{2}\sum_{m=1}^{M}\left|\frac{\partial f^{(m)}(w^{\rm k})}{\partial w_i}\right|+\frac{(1-\gamma_k)\beta_k}{2}||w^{\rm k}||^2_2$

   $\quad$其中$f^{(m)}(w^{\rm k})$表示第$m$层的激活函数的导数。

3.$RSD(w_i, \lambda_k):\quad RSD(w_i, \lambda_k)=\lambda_kw_i+\frac{1}{2}\lambda_k I_d$

   $I_d$表示单位矩阵，$d$表示模型的维度。

# 4.具体代码实例和解释说明
## (1). Pytorch实现
```python
import torch
from torchvision import models
import copy
import numpy as np

class PGDAttack:
  def __init__(self, model, epsilon=0.3, num_steps=100, step_size=0.01, 
               random_start=True, loss_func='CrossEntropy'):
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.random_start = random_start
    if loss_func == 'CrossEntropy':
      self.loss_func = nn.CrossEntropyLoss()

  def attack(self, inputs, targets):
    # Predict the label of original input before attack
    pre_label = self.predict(inputs)

    # Initialize adversarial example to be same shape as original image
    adv_inputs = inputs.clone().detach().requires_grad_(True)
    
    if self.random_start:
      # Randomly initialize starting point in epsilon ball around original input
      noise = np.random.uniform(-self.epsilon, self.epsilon, size=inputs.shape).astype(np.float32)
      adv_inputs += torch.tensor(noise).to(inputs.device)
      
    for _ in range(self.num_steps):
      
      outputs = self.model(adv_inputs)
      _, preds = torch.max(outputs, dim=1)
      loss = self.loss_func(outputs, targets)

      grads = torch.autograd.grad(loss, [adv_inputs])[0]

      adv_inputs.data.add_(self.step_size * torch.sign(grads.data))
      delta = adv_inputs - inputs

      # Projection onto epsilon ball
      perturbation = torch.clamp(delta, min=-self.epsilon, max=self.epsilon)
      adv_inputs = inputs + perturbation
      
            
    adv_images = torch.clamp(adv_inputs, min=0., max=1.)
    return adv_images
  
  def predict(self, images):
    predictions = self.model(images)
    predicted_labels = torch.argmax(predictions, axis=1)
    return predicted_labels

if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = models.resnet18(pretrained=True).to(device)
  model.eval()

  test_image = transforms.ToTensor()(test_image).unsqueeze(0).to(device)

  target_label = 100
  attack = PGDAttack(model, epsilon=0.3, num_steps=100, 
                     step_size=0.01, random_start=True)

  adv_image = attack.attack(test_image, torch.LongTensor([target_label]).to(device))
  utils.save_image(adv_image[0], save_path)
```