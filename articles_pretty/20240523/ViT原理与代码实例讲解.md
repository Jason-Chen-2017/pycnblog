# ViT原理与代码实例讲解

作者：禅与计算机程序设计艺术

## 1.背景介绍
### 1.1 计算机视觉中的挑战
### 1.2 卷积神经网络(CNN)的局限性  
### 1.3 Transformer在NLP领域的成功
### 1.4 将Transformer应用到计算机视觉的尝试

## 2.核心概念与联系
### 2.1 Vision Transformer(ViT)模型概述
#### 2.1.1 将图像分割为patches
#### 2.1.2 将patches映射为embedding
#### 2.1.3 添加position embedding
#### 2.1.4 使用Transformer编码器提取全局特征
#### 2.1.5 分类头对特征进行分类
### 2.2 ViT与CNN的对比
#### 2.2.1 局部特征 vs. 全局特征
#### 2.2.2 平移不变性 vs. 位置敏感性
#### 2.2.3 感受野 vs. 全局感受野
### 2.3 ViT的优势与局限
#### 2.3.1 对长距离依赖建模能力强
#### 2.3.2 泛化能力更好
#### 2.3.3 需要更大的数据集和计算开销

## 3.核心算法原理具体操作步骤
### 3.1 图像分割为patches
### 3.2 线性投影生成embedding
### 3.3 加入位置编码(position embedding)
### 3.4 Transformer编码器结构详解  
#### 3.4.1 Multi-Head Attention
#### 3.4.2 MLP Block
#### 3.4.3 Layer Normalization
#### 3.4.4 Residual Connections
### 3.5 分类头(Classification Head)结构

## 4.数学模型和公式详细讲解举例说明
### 4.1 图像分割为patches
$$
\mathbf{x}_p = \mathbf{x}_{cls} + \sum_{i=1}^N \mathbf{x}_i * \mathbf{E}
$$
其中 $\mathbf{x}_p$ 为patch tokens，$\mathbf{x}_{cls}$ 为分类token，$\mathbf{x}_i$ 为第i个patch，$\mathbf{E}$ 为线性投影矩阵。

### 4.2 Transformer编码器计算过程
$$
\begin{aligned}
\mathbf{z}_0 &= [\mathbf{x}_{cls}; \mathbf{x}_p^1\mathbf{E}; \cdots; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{pos} \\
\mathbf{z}^\prime_l &= \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}\\
\mathbf{z}_l &= \text{MLP}(\text{LN}(\mathbf{z}^\prime_l)) + \mathbf{z}^\prime_l \\
\mathbf{y} &= \text{LN}(\mathbf{z}^0_L)
\end{aligned}
$$

其中$\mathbf{E}_{pos}$是可学习的位置编码，$\text{MSA}$是Multi-Head Self Attention，$\text{MLP}$是多层感知机模块，$\text{LN}$为Layer Normalization，$L$为Transformer的层数。

### 4.3 Multi-Head Attention计算过程

$$
\begin{aligned}
\text{MSA}(\mathbf{X}) &= [\mathbf{Z}_1; \cdots \mathbf{Z}_h]\mathbf{W}^O \\
\text{其中} \quad \mathbf{Z}_i &= \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i) \\
\mathbf{Q}_i &= \mathbf{X} \mathbf{W}_i^Q,\quad
\mathbf{K}_i = \mathbf{X} \mathbf{W}_i^K,\quad
\mathbf{V}_i = \mathbf{X} \mathbf{W}_i^V
\end{aligned}
$$

$h$ 是attention头数，$\mathbf{W}_*$ 是可学习的参数矩阵。

## 5.项目实践：代码实例和详细解释说明
### 5.1 导入需要的库

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
```

### 5.2 ViT模型定义

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_ratio,
                  attn_drop=attn_drop_ratio, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
```

- 定义ViT模型类，初始化各个组件，包括patch embedding、position embedding、Transformer编码器、分类头等。  
- 使用`nn.Parameter`将class token和position embedding定义为可学习参数。
- Transformer编码器由多个`Block`组成，每个Block包含Multi-Head Attention和MLP。
- 最后对class token进行Layer Norm，然后通过分类头进行分类。

### 5.3 PatchEmbed定义

```python
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) 
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()  
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
```

- `PatchEmbed`模块用于将图像分割为patches并映射为embedding。
- 使用`nn.Conv2d`将patches映射到指定维度的embedding。
- 对得到的embedding进行flatten和transpose操作，最终输出形状为`[B, num_patches, embed_dim]`。

### 5.4 Transformer编码器Block定义

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

- Block是Transformer编码器中的一层，由Multi-Head Attention和MLP组成。
- 使用Layer Normalization和Residual Connection。
- 可以根据需要添加Dropout和DropPath用于正则化。

### 5.5 Attention模块定义

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

- Attention模块实现了Multi-Head Self Attention。
- 用一个Linear层生成query、key、value。
- 将query与key做点积得到attention map，然后计算softmax得到attention权重。
- 将attention权重与value相乘，然后reshape和再次进行投影。

### 5.6 MLP模块定义

```python
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

- MLP模块由两个Linear层组成。
- 使用GELU激活函数和Dropout。

### 5.7 模型训练与测试代码

```python
def train(net, trainloader, criterion, optimizer, epochs=10, device='cuda'):
    net.train()
    for epoch in range(epochs):
        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        print(f'Epoch: {epoch+1}, Loss: {train_loss/(batch_idx+1)}, Acc: {correct/total}')

def test(net, testloader, criterion, device='cuda'):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)