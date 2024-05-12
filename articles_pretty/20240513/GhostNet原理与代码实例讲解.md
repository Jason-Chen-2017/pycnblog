# GhostNet原理与代码实例讲解

作者：禅与计算机程序设计艺术

## 1.背景介绍

### 1.1 GhostNet的由来
GhostNet是一种轻量级的卷积神经网络架构,由华为诺亚方舟实验室在2020年提出。它旨在在保持高精度的同时大幅降低模型的计算复杂度和参数量,使其能更高效地部署在移动和嵌入式等资源受限的设备上。

### 1.2 GhostNet的意义
在深度学习飞速发展的今天,CNN模型的参数量和计算量也变得越来越大,给模型在终端设备上的部署带来巨大挑战。GhostNet通过一系列巧妙的架构设计,用"幻影"的方式生成更多的特征图,达到了用更少的参数学习到更丰富特征表示的目的,在效果和效率的权衡中取得了出色的平衡。

### 1.3 GhostNet的影响力
GhostNet的提出引起了学术界和工业界的广泛关注。诸多后续工作在其思想的基础上进行了改进和扩展。GhostNet及衍生的一系列轻量级骨干网络在移动端计算机视觉、智能家居、自动驾驶等领域得到了大量应用。

## 2.核心概念与联系

### 2.1 PointWise卷积
传统卷积中,卷积核在输入特征图的每个位置做点积,得出结果作为输出特征图的对应位置。而PointWise卷积将卷积核的大小固定为1x1,它仅对特征图的每个像素点做线性变换,不考虑像素间的联系。PointWise卷积计算量小,但学习能力有限。

### 2.2 Depthwise卷积
不同于普通卷积同时对所有输入通道进行卷积,Depthwise卷积为每个输入通道单独设置一组卷积核。它先对各通道分别做普通卷积,再用PointWise卷积对通道间特征做融合。Depthwise卷积大幅降低计算量,但可能损失通道间的信息。

### 2.3 Ghost模块
Ghost模块是GhostNet的核心组件。它利用少量的常规卷积,生成一部分"源"特征图。然后使用PointWise卷积,基于这些"源"特征图生成更多的"幻影"特征图。Ghost模块用很小的代价实现了特征通道的扩展,避免了Depthwise卷积的信息损失问题。

### 2.4 Squeeze-and-Excitation（SE）注意力机制
GhostNet参考了SE Net的思路,在Ghost模块后引入注意力模块。它先对输入特征做全局池化,然后用两个全连接层学习特征重要性,最后将学到权重施加回原始特征图,自适应地增强重要特征,抑制非重要特征。

## 3.核心算法原理与具体操作步骤

### 3.1 Ghost操作
#### 3.1.1 分组卷积生成源特征图
Ghost操作首先用少量的常规卷积生成源特征图。具体做法是,将输入的C个通道分成g组,每组有c个通道(C=g*c)。对每组分别做常规卷积,卷积核数为m,输出m个特征图。这样经过分组卷积,就得到了gm个源特征图。

#### 3.1.2 逐点卷积生成幻影特征图 
基于源特征图,Ghost操作再用PointWise卷积生成幻影特征图。其做法是,用n组PointWise卷积核(1x1xgm)对源特征图做卷积,每组卷积核生成m个特征图。这样就得到了额外的nm个幻影特征图。源特征图和幻影特征图拼接在一起,最终Ghost操作的输出通道数为gm+nm。

### 3.2 Ghost瓶颈模块
Ghost瓶颈模块是Ghost操作的拓展,它在Ghost操作前后分别接了PointWise卷积,起到升/降维的作用。具体结构为:
1. PointWise (1x1) 卷积将输入通道降维到C/2。
2. 调用Ghost操作,输出通道数为kC/2(k为膨胀倍率)。
3. PointWise (1x1) 卷积将通道升维到C。 
4. 如果输入输出维度一致,则添加残差连接。

### 3.3 SE权重施加
SE模块在Ghost瓶颈后做自适应特征增强。具体操作为:
1. 对Ghost瓶颈的输出特征图在空间维度做全局平均池化,得到通道描述子。
2. 通道描述子通过两个全连接层,第一层将特征降维到C/r(r为缩减比),第二层再升维到C,Sigmoid激活得到各通道权重。
3. 将学到的权重看作各通道的重要性,与原始特征图逐通道相乘,得到最终的Ghost瓶颈输出。

## 4.数学模型和公式详细讲解举例说明

### 4.1 常规卷积
对输入特征图 $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ 做卷积,卷积核为 $\mathbf{W} \in \mathbb{R}^{C \times D \times k \times k}$,偏置为 $\mathbf{b} \in \mathbb{R}^{D}$,则输出特征图 $\mathbf{Y} \in \mathbb{R}^{D \times H \times W}$ 中各元素为:

$$ \mathbf{Y}(d,i,j) = \sum_{c=1}^{C} \langle \mathbf{W}(c,d),\mathbf{X}(c,i:i+k-1,j:j+k-1) \rangle + \mathbf{b}(d) $$

其中 $\langle \cdot,\cdot \rangle$ 表示内积。常规卷积的计算量为 $CkκDHW$。

### 4.2 PointWise卷积
PointWise卷积的卷积核大小固定为1x1。设输入特征图 $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$,PointWise卷积核为 $\mathbf{W} \in \mathbb{R}^{C \times D}$,偏置为 $\mathbf{b} \in \mathbb{R}^{D}$,则输出特征图 $\mathbf{Y} \in \mathbb{R}^{D \times H \times W}$ 中各元素为:

$$ \mathbf{Y}(d,i,j) = \sum_{c=1}^{C} \mathbf{W}(c,d) \cdot \mathbf{X}(c,i,j) + \mathbf{b}(d) $$

PointWise卷积的计算量为 $CDHW$,远小于常规卷积。但其感受野只有1x1。

### 4.3 Ghost操作
Ghost操作首先将输入特征图 $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ 在通道维度分成g组,每组包含c个通道(C=g*c)。用常规卷积核 $\mathbf{W}_g \in \mathbb{R}^{c \times m \times k \times k}$ 对每组分别做卷积,生成gm个源特征图 $\mathbf{X}_g \in \mathbb{R}^{gm \times H \times W}$: 

$$ \mathbf{X}_g(u,i,j) = \sum_{v=1}^{c} \langle \mathbf{W}_g(v,u),\mathbf{X}(g(c-1)+v,i:i+k-1,j:j+k-1) \rangle $$

其中 $u \in \{1,2, ..., m\}$ 表示源特征图的通道索引。

然后,用n组PointWise卷积核 $\mathbf{W}_p \in \mathbb{R}^{gm \times m}$ 对源特征图做卷积,生成nm个幻影特征图 $\mathbf{X}_p \in \mathbb{R}^{nm \times H \times W}$:

$$ \mathbf{X}_p(v,i,j) = \sum_{u=1}^{gm} \mathbf{W}_p(u,v) \cdot \mathbf{X}_g(u,i,j) $$

其中 $v \in \{1,2, ..., n\}$ 表示幻影特征图的组索引。将源特征图 $\mathbf{X}_g$ 和幻影特征图 $\mathbf{X}_p$ 在通道维度拼接,得到Ghost操作的最终输出 $\mathbf{Y} \in \mathbb{R}^{(gm+nm) \times H \times W}$。 

相比直接用 $(gm+nm)$ 个常规卷积核产生等量特征图,Ghost操作的计算量减少为原来的 $\frac{ckmg+gm^2n}{k^2(gm+nm)}$ 倍。当 $k=3,c=4,m=4,n=1$ 时,计算量减少为55%。

## 4.项目实践：代码实例和详细解释说明

下面以PyTorch为例,给出Ghost操作和Ghost瓶颈模块的参考代码。

### 4.1 Ghost操作代码实现

```python
import torch
import torch.nn as nn

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, activation=nn.ReLU):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        # 分组卷积生成源特征图
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(init_channels),
            activation(inplace=True) 
        )

        # PointWise卷积生成幻影特征图
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2,groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]
```
- 初始化时指定Ghost操作的输入输出通道数inp和oup,以及膨胀率ratio。
- 先将输入特征图分成ceil(oup/ratio)组,用常规卷积对每组做卷积,生成源特征图。
- 然后用PointWise卷积在源特征图的基础上生成oup/ratio组、每组ceil(oup/ratio)个幻影特征图。
- 将源特征图和幻影特征图拼接,得到共oup个输出特征图。

### 4.2 Ghost瓶颈模块代码实现

```python
class GhostBottleneck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, activation=nn.ReLU):
        super(GhostBottleneck, self).__init__()
        self.ghost1 = GhostModule(in_chs, mid_chs, kernel_size=1, ratio=2, dw_size=dw_kernel_size, stride=1, activation=activation)

        if stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride, dw_kernel_size//2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        self.ghost2 = GhostModule(mid_chs, out_chs, kernel_size=1, ratio=2, dw_size=dw_kernel_size, stride=1, activation=activation)
        
        if stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride, dw_kernel_size//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_chs)
            )
        else:
            self.shortcut = nn.Sequential()

        self.se = SEModule(mid_chs)  # SE模块

    def forward(self, x):
        shortcut = x
        x = self.ghost1(x)
        if hasattr(self, 'conv_dw'):
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        x = self.ghost2(x)

        # SE权重施加
        w = self.se(x) 
        x = x * w

        shortcut = self.shortcut(shortcut)
        return x + shortcut
```
- Ghost瓶颈先用一个Ghost模块将输入特征图升维到中间层维度。
- 如果步长大于1,再接一个Depthwise卷积改