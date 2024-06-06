# MAE原理与代码实例讲解

## 1. 背景介绍

### 1.1 自监督学习的兴起

近年来,自监督学习(Self-supervised Learning)在计算机视觉、自然语言处理等领域取得了令人瞩目的成就。自监督学习旨在从大规模无标注数据中学习通用的数据表征,从而减少对人工标注数据的依赖。其核心思想是利用数据本身的内在结构和规律,构建一个预测任务,通过训练模型完成该任务,使其学会数据的有效表征。

### 1.2 MAE的提出

在计算机视觉领域,Kaiming He等人于2021年提出了一种名为MAE(Masked Autoencoders)的自监督学习框架。MAE借鉴了自然语言处理中的掩码语言模型(如BERT),将掩码思想引入视觉领域。通过随机掩盖图像块并训练模型重建原始图像,MAE能够学习到图像的高层语义特征。MAE一经提出就在学术界引起了广泛关注,并迅速成为视觉自监督学习的新范式。

## 2. 核心概念与联系

### 2.1 Autoencoder

Autoencoder(自编码器)是一种经典的无监督学习模型。它由编码器(Encoder)和解码器(Decoder)两部分组成。编码器将输入数据映射到低维的隐空间,解码器则从隐空间重建原始输入。通过最小化重建误差,Autoencoder能够学习到数据的压缩表征。

### 2.2 Masked Language Model

Masked Language Model(MLM)是一种自监督学习范式,最初应用于自然语言处理领域。其代表性工作是BERT(Bidirectional Encoder Representations from Transformers)。MLM随机掩盖输入序列中的部分token,并训练模型预测被掩盖的token。通过这种方式,模型能够学习到上下文信息和词语之间的关系。

### 2.3 MAE与Autoencoder和MLM的关系

MAE将Autoencoder和MLM两种思想巧妙地结合在一起。与经典的Autoencoder不同,MAE在输入图像上随机掩盖一部分图像块,仅对可见图像块进行编码。解码器则需要根据可见图像块的表征,重建完整的原始图像。这一过程与MLM非常类似,只不过MAE是在图像块级别进行掩码和预测。通过这种掩码自编码机制,MAE能够学习到图像的全局结构和高层语义信息。

## 3. 核心算法原理与具体操作步骤

### 3.1 图像分块与掩码

MAE首先将输入图像划分为若干个大小相等的图像块(patch)。然后,从这些图像块中随机选择一部分进行掩码,即将其像素值置为0。掩码比例通常设置为75%左右。图像分块可以使用滑动窗口或非重叠分块的方式进行。

### 3.2 编码器

编码器只对未被掩码的图像块进行编码。首先,将这些可见图像块展平并线性映射到隐空间。然后,加入可学习的位置嵌入(position embedding),以引入图像块的位置信息。接着,将嵌入向量输入Transformer编码器,通过自注意力机制建模图像块之间的全局依赖关系。编码器的输出是可见图像块的隐向量表征。

### 3.3 解码器

解码器需要根据可见图像块的表征,重建完整的原始图像。首先,将可见图像块的隐向量表征和被掩码图像块的可学习掩码token拼接在一起,形成完整的隐向量序列。然后,将该序列输入Transformer解码器,通过自注意力机制和跨注意力机制建模图像块之间的依赖关系。解码器的输出是所有图像块的像素值重建。

### 3.4 重建损失与训练

MAE的训练目标是最小化重建损失,即原始图像与重建图像之间的均方误差(MSE)。由于编码器只能访问可见图像块的信息,因此解码器必须学会利用全局上下文信息来推断被掩码图像块的内容。通过这种方式,MAE能够学习到图像的高层语义特征。模型训练通常采用Adam优化器和余弦学习率调度策略。

## 4. 数学模型与公式详细讲解

### 4.1 图像分块

给定一张大小为$H \times W$的图像$\mathbf{x}$,将其划分为$N$个大小为$P \times P$的图像块$\mathbf{x}_i, i=1,2,\dots,N$。图像块的总数$N$计算公式为:

$$N = \lfloor \frac{H}{P} \rfloor \times \lfloor \frac{W}{P} \rfloor$$

其中,$\lfloor \cdot \rfloor$表示向下取整操作。

### 4.2 掩码操作

从$N$个图像块中随机选择$M$个进行掩码,掩码比例为$\rho$。被掩码的图像块集合为$\mathcal{M}$,未被掩码的图像块集合为$\mathcal{U}$。掩码操作可以表示为:

$$
\mathbf{x}_i^{\prime} = 
\begin{cases}
\mathbf{0}, & i \in \mathcal{M} \\
\mathbf{x}_i, & i \in \mathcal{U}
\end{cases}
$$

其中,$\mathbf{x}_i^{\prime}$表示掩码后的图像块,$\mathbf{0}$表示全零向量。

### 4.3 编码器

将未被掩码的图像块$\mathbf{x}_i, i \in \mathcal{U}$展平并线性映射到$D$维隐空间,得到嵌入向量$\mathbf{e}_i$:

$$\mathbf{e}_i = \mathbf{W}_e \cdot \text{flatten}(\mathbf{x}_i) + \mathbf{b}_e, \quad i \in \mathcal{U}$$

其中,$\mathbf{W}_e \in \mathbb{R}^{D \times (P^2 \cdot C)}$和$\mathbf{b}_e \in \mathbb{R}^D$分别为可学习的权重矩阵和偏置向量,$C$为图像的通道数。

然后,加入可学习的位置嵌入$\mathbf{p}_i \in \mathbb{R}^D$,得到最终的输入嵌入向量$\mathbf{z}_i$:

$$\mathbf{z}_i = \mathbf{e}_i + \mathbf{p}_i, \quad i \in \mathcal{U}$$

接着,将$\mathbf{z}_i$输入Transformer编码器,通过自注意力机制建模可见图像块之间的全局依赖关系:

$$\mathbf{h}_i = \text{Transformer-Encoder}(\mathbf{z}_i), \quad i \in \mathcal{U}$$

其中,$\mathbf{h}_i \in \mathbb{R}^D$为第$i$个可见图像块的隐向量表征。

### 4.4 解码器

将可见图像块的隐向量表征$\mathbf{h}_i$和被掩码图像块的可学习掩码token $\mathbf{m} \in \mathbb{R}^D$拼接在一起,形成完整的隐向量序列$\mathbf{h}^{\prime} \in \mathbb{R}^{N \times D}$:

$$
\mathbf{h}_i^{\prime} = 
\begin{cases}
\mathbf{m}, & i \in \mathcal{M} \\
\mathbf{h}_i, & i \in \mathcal{U}
\end{cases}
$$

然后,将$\mathbf{h}^{\prime}$输入Transformer解码器,通过自注意力机制和跨注意力机制建模图像块之间的依赖关系:

$$\mathbf{y}_i = \text{Transformer-Decoder}(\mathbf{h}_i^{\prime}), \quad i=1,2,\dots,N$$

其中,$\mathbf{y}_i \in \mathbb{R}^{P^2 \cdot C}$为第$i$个图像块的像素值重建。

### 4.5 重建损失

MAE的训练目标是最小化重建损失,即原始图像块$\mathbf{x}_i$与重建图像块$\mathbf{y}_i$之间的均方误差(MSE):

$$\mathcal{L}_{\text{recon}} = \frac{1}{M} \sum_{i \in \mathcal{M}} \| \mathbf{x}_i - \mathbf{y}_i \|_2^2$$

其中,$\| \cdot \|_2$表示$L_2$范数。需要注意的是,重建损失只计算被掩码图像块的重建误差。

## 5. 项目实践:代码实例与详细解释

下面以PyTorch为例,给出MAE的核心代码实现。

### 5.1 图像分块与掩码

```python
def patchify(imgs, patch_size):
    """
    将图像划分为大小为patch_size的图像块
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def random_masking(x, mask_ratio):
    """
    对图像块进行随机掩码
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore
```

### 5.2 编码器

```python
class PatchEmbed(nn.Module):
    """
    将图像块嵌入到隐空间
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)
        return x

class MAEEncoder(nn.Module):
    """
    MAE编码器
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        x = self.patch_embed(x)
        x += self.pos_embed[:, :x.shape[1], :]
        x = x * mask.unsqueeze(-1)  # apply mask
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x
```

### 5.3 解码器

```python
class MAEDecoder(nn.Module):
    """
    MAE解码器
    """
    def __init__(self, patch_size=16, embed_dim=768, depth=8, num_heads=16, out_chans=3):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (224 // patch_size) ** 2 + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn