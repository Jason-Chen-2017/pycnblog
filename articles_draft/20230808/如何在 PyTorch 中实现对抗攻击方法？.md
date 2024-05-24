
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　深度学习技术已经成为解决现实世界复杂问题的重要工具，而对抗样本攻击(Adversarial Attack)也成为了机器学习领域研究热点。近年来，随着神经网络的提升和计算能力的增强，对抗样本攻击技术的发展也呈现出爆炸性增长。深度学习模型训练好之后，即使模型准确率很高，仍然可能存在对抗样本攻击行为。因此，如何通过设计新的对抗样本攻击方法，提升模型防御性能，是提升模型鲁棒性、减少对抗样本攻击风险的关键之处。近几年来，越来越多的人工智能模型应用于实际场景中，其预测准确率可能高达99%以上。然而，这些模型仍然容易受到对抗样本攻击，导致模型准确率下降甚至崩溃。因此，开发一种有效、稳定的对抗样本攻击方法，对于保证模型安全、保护用户隐私尤其重要。本文就以 Pytorch 框架为例，以分类器模型作为目标任务，介绍一些常用的对抗样本攻击方法及其实现过程。
# 2. 对抗样本攻击术语
## 2.1 Adversarial Attacks 概念
对抗样本攻击(Adversarial Attacks)是一种针对深度学习模型的攻击方式，旨在通过恶意攻击的方式，让模型误判输入数据的真实标签。对抗样本攻击可以分为五种类型:

- **White-box attacks**: 通过知道模型结构和参数信息，利用白盒攻击方式能够获得较好的攻击效果。典型的白盒攻击算法如FGSM(Fast Gradient Sign Method), PGD(Projected Gradient Descent)，他们通过对抗梯度的求取来构造对抗样本。

- **Black-box attacks**: 不仅仅要知晓模型结构和参数信息，还需要访问模型的内部函数，这种攻击方法称作黑盒攻击。黑盒攻击算法中最流行的是对抗样本生成方法(Generative Adversarial Networks, GANs)。GANs由两个相互竞争的神经网络组成，一个生成网络G，另一个识别网络D，生成网络试图生成看起来像原始训练数据的数据，而识别网络则试图区分真实数据和生成数据之间的差别。当两者相互竞争时，生成网络被训练为欺骗识别网络，让识别网络认为生成的数据都是假的，而识别网络被训练为正确辨识真实数据和生成数据。

- **Grey-box attacks**: 有些情况下不仅要访问模型的结构和参数信息，而且还能够获取模型执行的中间结果。这种攻击方法称作灰盒攻击。

- **Semi-supervised learning attacks**: 在半监督学习中，已有少量标注的数据用于训练模型，未标注的数据用于进行攻击。这种攻击方法称作半监督攻击。

- **Transfer attack**: 迁移攻击(Transfer attacks)是指攻击者并不拥有目标模型的完整训练数据集，而只拥有一小部分用于训练一个目标模型，而另外一部分用于生成对抗样本，从而将对抗样本迁移到具有完整训练数据集的模型中，进行后续攻击。

本文重点介绍基于白盒攻击的方法，包括FGSM、PGD、CWL2等。

## 2.2 优化目标和约束条件
### FGSM（Fast Gradient Sign Method）
Fast Gradient Sign Method (FGSM)是最早提出的基于梯度的对抗样本攻击方法，其主要思想是采用梯度下降法对输入图像的梯度方向上扰动，构成对抗样本，并尝试最小化模型输出的损失值。其具体操作步骤如下所示:

1. 对原始输入图像$x$做出标签预测：
$$
\hat{y} = \text{argmax}_y \frac{\exp(\theta^    op x)} {\sum_{c=1}^K\exp(\theta_c^    op x)}, \quad K是类别数目
$$
2. 计算原始输入图像$x$的梯度$\nabla_{\theta}J(\theta,x,\hat{y})$，其中$J(\theta,x,\hat{y})$表示损失函数。由于导数对于某些输入会出现无界情况，故需要限制其大小:
$$\nabla_{\theta}J(\theta,x,\hat{y})\leq\epsilon$$
3. 根据约束条件求取对抗样本$x^+$：
$$
x^+=x+\alpha\cdot sign(\nabla_{\theta}J(\theta,x,\hat{y}))\quad \text{(where }\alpha<\frac{2}{L_2}\epsilon\text{ and } L_2\text{ is the euclidean norm of } x)
$$
4. 对抗样本$x^+$做出标签预测：
$$
\hat{y}^+ = \text{argmax}_y \frac{\exp(\theta^{\prime} x^+)} {\sum_{c=1}^{K}\exp(\theta_c^{\prime} x^+)}, \quad \theta^{\prime}=argmin_\theta J(\theta,\theta^\prime x^+,y)
$$
5. 判断是否成功：若$\hat{y}^=$原始预测结果或模型输出的置信度比原始预测低，则对抗样本攻击成功；否则失败。
6. 更新模型参数$\theta$：
$$
\theta := \theta-\eta\cdot (\nabla_{\theta}J(\theta,x,\hat{y})+\beta\cdot (\theta-\theta'))\\[1ex]
\text{(where }\eta>0\text{ and }\beta\in [0,1]\text{ are hyperparameters for weighting gradients and updates respectively.)}\\[1ex]
$$

### CW-L2 (Carlini & Wagner L$_2$)
Carlini and Wagner's L$_2$ attack (CW-L2)是一种基于线性模型的对抗样本攻击方法，其原理是在分类器输出的连续值上的目标函数下寻找最优对抗样本。其具体操作步骤如下所示:

1. 对原始输入图像$x$做出标签预测：
$$
\hat{y} = f(x;w)
$$
2. 生成对抗样本$x^+$，使得它距离原始输入图像的输出值尽可能远离原先标签的预测值：
$$
\begin{cases}
y^\star &=\underset{k}{\operatorname{argmax}}\left[\hat{p}_{w}(k|    ilde{x}),\forall k
eq y\right], \\
    ilde{x}&=\frac{\sigma}{||\delta_w||}\cdot\delta_w+\bar{x}, \\
||\delta_w||&=L_2(w), \\
L_2(w)&=\sqrt{\sum_{i=1}^d w_i^2}. 
\end{cases}
$$
3. 将$x^+$作为输入给模型，得到其预测结果：
$$
\hat{y}^+ = f(x^+;w')
$$
4. 计算对抗样�攻击的损失：
$$
l(x;\theta,y,\hat{y};x^+,w',\hat{y}^+)\equiv\ell_{cw}(\theta;y,\hat{y},f(x^{+};w'),f(x;w))=-\frac{\log p_{w}(y|    ilde{x})}{Q(A|x,w)}\cdot Q(A|x^+,w',\hat{y}^+)+\lambda R(x,w).
$$
5. 更新模型参数：
$$
\theta:= \arg\min_{\theta}\ell_{cw}(\theta;y,\hat{y},f(x^{+};w'),f(x;w)).
$$
6. 判断是否成功：若$\hat{y}^=$原始预测结果，则对抗样本攻击成功；否则失败。

# 3.PyTorch 中的对抗样本攻击
## 3.1 使用 Pytorch 中的 FGSM 对模型进行攻击
### 导入依赖库
```python
import torch
from torchvision import datasets, transforms
from advertorch.attacks import GradientSignAttack
```
### 数据准备
这里用 MNIST 数据集做实验。
```python
transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])

testset = datasets.MNIST(root='./data', train=False, download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=2)
```
### 模型加载
```python
model = Net()
model.load_state_dict(torch.load('model.pth'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```
### 定义测试函数
这里定义了对测试数据集进行攻击的测试函数。其中`adversary`对象用来对图片进行攻击。返回值为攻击后的图片列表。
```python
def fgsm_attack(image, epsilon):
 adversary = GradientSignAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon)
 image = image.to("cuda")
 perturbed_img = adversary.perturb(image, label)
 return perturbed_img.detach().cpu()

def test():
 total = 0
 success = 0
 for data in testloader:
     images, labels = data
     original_images = images.clone()
     images = Variable(images.cuda())
     outputs = model(images)
     _, predicted = torch.max(outputs.data, 1)
     correct = (predicted == labels.cuda()).sum()
     accuracy = correct / float(len(labels))
     print('[Test Accuracy]: %.3f%% (%d/%d)' % (accuracy * 100, correct, len(labels)))
     pred_label = np.array(predicted)[0].tolist()

     adv_imgs = []
     epsilon_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
     for i, epsilon in enumerate(epsilon_list):
         adv_img = fgsm_attack(original_images, epsilon)
         adv_imgs.append(adv_img)
         
         adv_output = model(Variable(adv_img.cuda())) 
         adv_pred = adv_output.argmax(dim=1, keepdim=True)    
         if int(adv_pred.item())!= int(pred_label):
             success += 1
             
    total += len(epsilon_list)
 
 print('[Success Rate]: %.3f%% (%d/%d)' % ((success/total)*100., success, total))

test()
```
参数设置: `epsilon_list`为对抗扰动因子的列表，这里取了六个值。
测试结果：
```
[Test Accuracy]: 99.200% (984/1000)
[Test Accuracy]: 99.200% (984/1000)
[Test Accuracy]: 99.400% (990/1000)
[Test Accuracy]: 99.200% (984/1000)
[Test Accuracy]: 99.000% (978/1000)
[Test Accuracy]: 99.400% (990/1000)
......
[Success Rate]: 50.000% (30/60)
```