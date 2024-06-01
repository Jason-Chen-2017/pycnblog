
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在近几年来，语言模型预训练技术取得了非常大的进步。语言模型的本质就是用自回归序列到序列(Seq2Seq)的方式预训练一个神经网络模型。传统的 Seq2Seq 模型的训练通常是通过最大似然损失函数，通过反向传播更新参数，这种方法虽然简单易懂，但是速度慢而且收敛困难，甚至出现梯度消失或爆炸等现象。为了解决这一问题，研究者们提出了基于注意力机制(Attention Mechanisms)的 Seq2Seq 模型，通过注意力模块强化解码器对输入序列的注意力学习，从而使得生成结果更加准确。受益于这个模型，很多自然语言处理任务都得到了改善。比如语言翻译、文本摘要、文本补全、机器问答、文本分类等。
近些年，越来越多的人关注了无监督预训练技术，因为它可以提高模型的泛化能力，减少样本需求，且效果比监督预训练好。最近，一种新的 Seq2Seq 模型——Autoregressive Transformer (AT)被提出来，其特点是采用自回归性质的网络结构来实现解码，因此在推断时只需要一次完整的循环即可生成结果，并不需要像传统的 Seq2Seq 模型一样使用带噪声的输入。另外，AT 使用全连接层代替 RNN 来完成 Self-Attention 操作，这大幅降低了计算量，也减少了参数数量。同时，AT 使用线性 attention 进行运算，这使得其训练速度更快，并节省了显存空间。但是，由于 AT 的自回归性质，它只能进行左到右或者右到左的单向编码，不能够像传统 Seq2Seq 模型那样进行双向编码。
那么，如何将这两种模型结合起来，既能够获得传统 Seq2Seq 模型的好处（即速度快，并且可用于双向编码），又能够具有独有的自回归特性？这就是今天文章要讨论的问题。
# 2.相关工作
## 2.1 Seq2Seq 模型
Seq2Seq 模型由两个子模块组成：编码器(Encoder)和解码器(Decoder)。编码器接受源序列作为输入，并通过 RNN 或者 CNN 对输入进行编码，输出的是一个固定长度的上下文向量。然后将上下文向量送入解码器中，解码器依次生成每个时间步的目标词，直到生成结束。Seq2Seq 模型的主要优点包括：
* 速度快：Seq2Seq 模型的编码和解码过程都是一次循环，所以速度比较快。
* 可扩展性：Seq2Seq 模型可以使用任意类型的 RNN 或 CNN，可以在不同领域之间迁移学习。
* 标签信息丢弃：Seq2Seq 模型生成的目标词不依赖于原始标签，因此可以去除标签信息。
## 2.2 AutoRegressive Transformer (AT)
AutoRegressive Transformer 是一种 Seq2Seq 模型，它的特点是在解码过程中只需一次完整的循环，从而不需要像传统 Seq2Seq 模型那样通过输入噪声来模拟生成过程。AT 使用全连接层代替 RNN 来实现 Self-Attention 操作，这样可以大大降低计算复杂度，并降低参数数量。其具体构造如下图所示：
其中 $x_t$ 为当前时刻的输入，$\overline{x}_{<t}$ 表示输入序列的前 $t-1$ 个元素。$H$ 和 $W$ 分别表示隐藏层大小和词嵌入大小。
### 2.2.1 Encoder
编码器接收源序列作为输入，将其映射为一个固定长度的上下文向量 $\mathbf{c}=\left[\mathbf{h}_{\mathrm{enc}, 1}, \ldots, \mathbf{h}_{\mathrm{enc}, T}\right]$ ，其中 $T$ 为序列的长度。$\mathbf{h}_{\mathrm{enc}}$ 可以视作是编码器在时刻 $t$ 时隐含状态，其维度等于隐藏单元个数。其具体计算方法如下：
$$\begin{aligned}
\mathbf{h}_{\mathrm{enc}} &=\operatorname{Encoder}(\mathbf{x}) \\
&=\operatorname{LSTM}(E(\mathbf{x})) \\
&\text { where } E:\mathbb{R}^{d_{\text {input }}}\rightarrow\mathbb{R}^{d_{\text {hidden}}} \text { is the embedding function}\\
\end{aligned}$$
这里，$\operatorname{Encoder}$ 是一个 RNN 单元，如 LSTM。
### 2.2.2 Decoder
解码器根据 $\mathbf{c}$ 初始化自身的隐含状态 $\mathbf{h}_{\mathrm{dec}, t=0}$ 。在第 $t$ 次循环迭代时，利用上一步的隐含状态 $\mathbf{h}_{t-1}$ 和 $\mathbf{y}_{t-1}$ 来生成第 $t$ 个目标词 $\mathbf{y}_{t}$。目标词的计算方式如下：
$$\mathbf{y}_{t}=\operatorname{MLP}(E(\mathbf{y}_{t-1})\odot\alpha_{t-1}^{\top})\quad\text{(Eq.1)}$$
这里，$\alpha_{t-1}$ 是 decoder 在时刻 $t-1$ 时生成的概率分布，$E(\cdot)$ 是词嵌入矩阵，$\odot$ 是 hadamard 乘积符号，$MLP$ 是两层全连接网络。
### 2.2.3 Cross Entropy Loss
为了训练 Seq2Seq 模型，通常使用交叉熵损失函数。对于每个时间步 $t$，交叉熵损失可以计算如下：
$$L_{t}=-\log p_{\theta}(\mathbf{y}_{t}|X^{\leq t}),\quad \forall t=1,\ldots,T,$$
其中，$p_{\theta}(\cdot)$ 是模型的输出分布，$\theta$ 表示模型的参数。AT 使用上述损失函数进行参数优化，其更新规则如下：
$$\begin{aligned}
\nabla_{\theta}\mathcal{L}&=-\frac{1}{T} \sum_{t=1}^T \log p_{\theta}(\mathbf{y}_{t}|X^{\leq t}) \\
&=-\frac{1}{T} \sum_{t=1}^T \log \left(\sigma\left(\tilde{\mathbf{w}}\left(\mathbf{h}_{t-1}, \overline{h}_{t}, \mathbf{y}_{t-1}\right)\right)\right),\quad \forall t=1,\ldots,T\\
\end{aligned}$$
其中，$\tilde{\mathbf{w}}$ 是 attention weights，其计算方式如下：
$$\tilde{\mathbf{w}}=\operatorname{softmax}\left(\operatorname{Linear}\left(\operatorname{ReLU}\left(\operatorname{Linear}\left(\mathbf{h}_{t-1}, H\right)+\operatorname{Linear}\left(\overline{h}_{t}, W\right)+\operatorname{Linear}\left(\mathbf{y}_{t-1}, U\right)\right), V\right)\right). $$
这里，$V$ 表示 attention 权重大小。$\overline{h}_{t}$ 是 decoder 在时刻 $t-1$ 时隐含状态与 context vector 组合后的结果。$\sigma$ 函数是 sigmoid 函数，用于将 MLP 的输出转换为概率值。
## 2.3 Combine Models into a Single Model
为了结合这两种模型，作者设计了一个新模型——"Fast Autoregressive Transformer (FAT)"。 FAT 的主要特点是：
* 只需要一次完整的循环，因此速度相较于其他 Seq2Seq 模型更快。
* 不仅可以使用左到右或者右到左的单向编码，还可以使用双向编码。
* 可以一次性计算整个序列的损失，而不是像其他 Seq2Seq 模型那样逐个计算。
具体地，作者把左边 encoder 中的 LSTM 替换为 "Slow Multihead Attention"，把右边 decoder 中 Self-Attention 替换为 "Fast Fully Connected Softmax"，然后用 "Slow Multihead Attention" 生成 attention 矩阵，用 "Fast Fully Connected Softmax" 生成 decoder 输出。这两个模块的计算量都是 $O(n^2)$，因此整体速度还是很慢的。为了加速这一部分的计算，作者设计了一个混合方法——"hybrid forward pass"。具体地，当某个位置没有出现在源序列时，作者直接把词嵌入置零，然后把所有 attention weights 都置零，就不会再进行计算。
# 3. 具体原理及操作步骤
## 3.1 Slow MultiHead Attention （慢速多头注意力）
Slow MultiHead Attention 是一种特殊的注意力机制，它的特点是：
* 接受左边的输入序列 $\mathbf{x}$ 和左边的隐藏状态 $\mathbf{h}_{l-1}$，输出 $\alpha_{l}^{\top}$。
* 通过一个由多个头部组成的多个注意力子模块，每一个子模块都可以看做是一个带输入门的门控线性单元 (Gated Linear Units)，然后用软性切分将 $\alpha_{l}^{\top}$ 划分为 $K$ 个子集。
具体计算方法如下：
$$\begin{aligned}
Q_{\ell}[i]=\operatorname{Linear}\left(\mathbf{x}, Q_{\ell, i}\right)\\
K_{\ell}[j]=\operatorname{Linear}\left(\mathbf{h}_{l-1}, K_{\ell, j}\right)\\
V_{\ell}[k]=\operatorname{Linear}\left(\mathbf{h}_{l-1}, V_{\ell, k}\right)\\
e_{\ell}[i, j] =\left\|\operatorname{KeyRelu}\left(\operatorname{QueryMap}(Q_{\ell}[i]) + KeyMap(K_{\ell}[j])\right)\right\|_{2} / \sqrt{d_{\text {key }}} \\
\alpha_{\ell}^{\top}=Softmax\left(e_{\ell}\right) \\
\beta_{\ell}^{\top}=\operatorname{Dropout}\left(Softmax\left(e_{\ell}\right)\right) \\
r_{\ell}=\operatorname{MultiHeadMatMul}\left(Q_{\ell}, K_{\ell}, V_{\ell}, \alpha_{\ell}^{\top}\right) \\
\end{aligned}$$
其中，$i$, $j$, $k$ 分别对应于三个输入嵌入向量的下标；$\ell$ 表示第几个注意力子模块；$d_{\text {key }}$ 是键值的维度。这里，$KeyMap(K_{\ell}[j])$ 是通过一个非线性变换 $\operatorname{KeyRelu}$ 将 $K_{\ell}[j]$ 变换到同一空间内；$QueryMap(Q_{\ell}[i])$ 和 $KeyMap(K_{\ell}[j])$ 分别是两个不同的线性变换，分别作用在查询和键上，产生两个不同的嵌入向量。接着，$\operatorname{MultiHeadMatMul}$ 是将三个输入向量的注意力权重相乘，然后用它们按一定方式拼接，得到最终的输出。最后，$\beta_{\ell}^{\top}$ 是对 $\alpha_{\ell}^{\top}$ 按照比例进行 dropout。
## 3.2 Hybrid Forward Pass（混合正向传递）
为了加速混合模型的推断阶段，作者设计了一套混合方法。具体地，当某个位置没有出现在源序列时，混合模型会直接跳过该位置的解码，不进行任何计算。这一部分的计算量很小，因此可以忽略不计。但是，如果某一位置出现了噪声或者需要被翻译的词语可能比较少，则可以通过设置阈值来决定是否跳过。作者选择了词频阈值，并基于不同语言的实际情况调整了阈值。在实际运行的时候，作者设定了一个全局阈值，这个阈值与词典大小成正比。
## 3.3 Fast Fully Connected Softmax （快速完全连接SoftMax）
Fast Fully Connected Softmax 是一种新的注意力机制，它的特点是：
* 接受上一步的隐含状态 $\mathbf{h}_{t-1}$ 和源序列 $\mathbf{x}$，输出注意力权重 $\alpha_{t}^{\top}$。
* 用一个注意力权重矩阵 $\Omega_{\ell}$ 投影 $\mathbf{h}_{t-1}$ 和 $\mathbf{x}$，并利用一个点积运算符 $\odot$ 来计算注意力权重。
具体计算方法如下：
$$\begin{aligned}
\alpha_{t}^{\top}=&\operatorname{FCSoftmax}\left(\Omega_{\ell} \circ (\mathbf{h}_{t-1}, \mathbf{x})\right) \\
\hat{\alpha}_{t}^{\top}=&\frac{1}{\sum_{j=1}^{m_{\text {tokens }}} e_{t[j]}} \cdot \operatorname{tanh}\left(\Omega_{\ell} \circ (\mathbf{h}_{t-1}, \mathbf{x})\right) \\
\left(\Omega_{\ell} \circ (\mathbf{h}_{t-1}, \mathbf{x})\right)_{\mu m, n}=&\operatorname{QueryMap}\left(\mathbf{h}_{t-1}\right)_{\mu m} \cdot \operatorname{KeyMap}\left(\mathbf{x}\right)_{\nu n}+U_{\mu n}+\text { bias }\\
\end{aligned}$$
其中，$m$ 表示输入序列的长度，$n$ 表示词表大小；$\mu$, $\nu$ 分别表示目标序列和源序列中的下标；$U_{\mu n}$ 和 $\text {bias}$ 是可学习的参数。这里，$\operatorname{QueryMap}$, $\operatorname{KeyMap}$ 是分别将输入向量映射到查询空间和键空间的矩阵。$\operatorname{FCSoftmax}$ 函数是在 MLP 中引入 softmax 函数的变种，用于完成注意力权重的归一化，而后面给出的公式即为 $\operatorname{FCSoftmax}$ 的计算公式。
## 3.4 Overall Architecture （总体架构）
下图展示了 FAT 模型的总体架构：

* **Encoder** : 左侧。通过左边的 encoder 生成固定长度的上下文向量 $\mathbf{c}$.
* **Decoder** : 右侧。先初始化自身的隐含状态 $\mathbf{h}_{\mathrm{dec}, t=0}$, 然后每一步循环迭代都接受上一步的隐含状态 $\mathbf{h}_{t-1}$ 和 $\mathbf{y}_{t-1}$ 来生成第 $t$ 个目标词 $\mathbf{y}_{t}$。
* **Cross entropy loss** : 每一步循环都计算损失值 $L_t$.
* **Attention** : 根据当前的上下文向量 $\mathbf{c}$, 使用 **Slow Multihead Attention** 来生成注意力权重 $\alpha_{l}^{\top}$ 。
* **Gating mechanism**: 使用 **Fast Fully Connected Softmax** 来生成注意力权重 $\alpha_{t}^{\top}$.
## 3.5 Training and inference （训练和推断）
训练时，各个子模块联合优化。损失值 $L_t$ 与 $p_{\theta}(\cdot)$ 参数之间的关系遵循交叉熵损失函数。
测试时，作者先用右侧的 decoder 初始化 $\mathbf{h}_{\mathrm{dec}, t=0}$ 。然后，在 $t$ 时刻，如果当前时刻输入的 token 已经出现在源序列中，则用第 $t$ 个词来推断下一个词；否则的话，则直接跳过当前时刻的计算，并将对应的输入令牌替换为零。由于作者只在需要生成的位置才计算注意力权重，因此可以通过这种方式大大提升效率。
# 4. 代码实现及示例
## 4.1 数据准备
## 4.2 环境配置
本文采用 python 3.6+ 和 PyTorch >= 1.3+ 版本。
```bash
!pip install transformers datasets torchaudio jieba
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import cross_entropy
from transformers import BertTokenizer, BertModel, AdamW
from datasets import load_dataset
import torchaudio
import jieba
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased').to(device)
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, trg_vocab, max_len=512):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.data = data
        self.max_len = max_len
        
    def tokenize(self, sentence):
        tokens = ['[CLS]'] + tokenizer.tokenize(sentence[:self.max_len - 2]) + ['[SEP]']
        return tokenizer.convert_tokens_to_ids(tokens)
    
    def encode_source(self, source):
        encoded = self.tokenize(' '.join([char for char in jieba.cut(source)])).unsqueeze_(0).to(device)
        mask = ((encoded!= model.config.pad_token_id)*1).bool().to(device)
        padding_length = model.config.max_position_embeddings - len(encoded)
        padded = torch.cat((encoded, torch.zeros(padding_length, dtype=torch.long)), dim=0)
        
        return {'input_ids': padded.to(device), 'attention_mask': mask.to(device)}

    def decode_target(self, target):
        ids = self.tokenize('[CLS]'+' '.join(['[MASK]'] * len(target))+'[SEP]')
        labels = self.tokenize(target)[1:-1].to(device)
        input_mask = ((ids!=model.config.pad_token_id)*1).bool().to(device)
        
        return {'input_ids': ids.unsqueeze_(0).to(device), 
                'attention_mask': input_mask.unsqueeze_(0).to(device)}, {'labels': labels.to(device)}
    
    def __getitem__(self, index):
        source, target = self.data['en'][index], self.data['de'][index]
        src_dict = self.encode_source(source)
        trg_inputs, trg_outputs = self.decode_target(target)
        return src_dict, trg_inputs, trg_outputs
    
    def __len__(self):
        return len(self.data['en'])
    
train_data = TranslationDataset({'en': list(load_dataset('wmt14')['train']['translation']['en']),
                                 'de': list(load_dataset('wmt14')['train']['translation']['de'])}, 
                                tokenizer.get_vocab(), tokenizer.get_vocab())
val_data = TranslationDataset({'en': list(load_dataset('wmt14')['validation']['translation']['en']),
                               'de': list(load_dataset('wmt14')['validation']['translation']['de'])},
                              tokenizer.get_vocab(), tokenizer.get_vocab())
test_data = TranslationDataset({'en': list(load_dataset('wmt14')['test']['translation']['en']),
                                'de': list(load_dataset('wmt14')['test']['translation']['de'])},
                               tokenizer.get_vocab(), tokenizer.get_vocab())
def compute_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                         shift_labels.view(-1))
                     
optimizer = AdamW(params=model.parameters(), lr=5e-5)  
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)   
for epoch in range(10):
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True) 
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)  
    scheduler.step()
    total_loss = []   
    num_batches = len(train_loader)
    print(f'Epoch {epoch}:')
    for idx, (src_dict, trg_inputs, trg_outputs) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(**src_dict, **trg_inputs)['last_hidden_state']  
        loss = compute_loss(outputs, trg_outputs['labels'].squeeze_(0))
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        if (idx+1)%20==0 or (idx+1)==num_batches:
            print('\tBatch:', f'{idx+1}/{num_batches}', '\tLoss:', round(loss.item(), 4))
            
    avg_loss = sum(total_loss)/len(total_loss)
    print(f'\tTraining average loss:{round(avg_loss, 4)}')
    
    # Validation
    valid_loss = []    
    num_batches = len(val_loader)
    with torch.no_grad():
        for idx, (src_dict, trg_inputs, trg_outputs) in enumerate(val_loader):  
            outputs = model(**src_dict, **trg_inputs)['last_hidden_state']  
            loss = compute_loss(outputs, trg_outputs['labels'].squeeze_(0))
            valid_loss.append(loss.item())
            if (idx+1)%20==0 or (idx+1)==num_batches:
                print('\tValidation Batch:', f'{idx+1}/{num_batches}', '\tLoss:', round(loss.item(), 4))
                
    avg_valid_loss = sum(valid_loss)/len(valid_loss)
    print(f'\tValidation average loss:{round(avg_valid_loss, 4)}')
```
## 4.3 模型架构
Bert-based 的 Encoder-Decoder 模型。
# 5. 分析
## 5.1 速度上的差异
本文所提出的模型 Speed Transformer 比 BERT 提供的显著性能优势。但是，Speed Transformer 需要使用额外的组件来模仿 RNN 模型的自回归属性，因此训练速度会慢一些。但是，它的速度快到足以支持真实生产的任务，比如用于 NLP 的机器翻译，例如 Google Translate。
## 5.2 模型的参数数量差异
Speed Transformer 的模型参数数量比 BERT 小，这是因为 Speed Transformer 只保留必要的子模块，例如 Slow Multihead Attention 和 Fast Fully Connected Softmax。这些子模块的计算量都很小，因此可以在实际场景中训练和推断。相比之下，BERT 的参数数量通常达到数百万。
## 5.3 是否需要编码器和解码器
本文提出的模型没有要求使用固定的编解码器架构。因此，可以自由选择想要使用的子模块，而不受限于 RNN 或卷积的限制。此外，对哪些模块施加注意力，以及使用多少注意力头部，都可以自行调节。
## 5.4 其他缺陷
* 仅采用全连接层的 Attention 并不足以捕获全局特征。目前，仅靠注意力机制来学习全局信息的模型仍然存在一些问题，比如错误解读和歧义。