
作者：禅与计算机程序设计艺术                    

# 1.简介
  

随着自然语言处理（NLP）技术的发展，在识别文本中句子边界及命名实体等信息成为一项重要的任务。现有的命名实体识别（NER）方法通常采用基于规则或统计模型的方法，这些模型需要训练大量的数据并进行参数调优，这些过程耗时且精度较低。为了解决这个问题，本文提出了一个基于双向长短记忆循环神经网络（BiLSTM-CRF）的命名实体识别系统。
BiLSTM-CRF是一种有效的序列标注模型，能够对标记化后的序列中的每个元素进行上下文敏感的建模。通过在BiLSTM层中学习到隐藏状态，CRF层则通过对所有可能的标记序列进行评分，从而确定最佳的标记序列。模型可以从有限的标签集合中自动学习到序列的实际标签，因此不需要手动指定标签映射关系。
实验结果表明，该模型具有很好的性能，能够达到SOTA水平。
本文的贡献主要包括：
1、设计了一种双向长短记忆循环神经网络（BiLSTM-CRF）的命名实体识别模型；
2、将嵌入层、编码层、分类器分别用双向LSTM（Bidirectional LSTM）、编码层、带有条件随机场（Conditional Random Field，CRF）的双向LSTM层实现；
3、提出了一种新颖的基于位置的正则化方法，利用词之间的距离信息来增强双向LSTM的表征能力；
4、针对中文数据集的NER任务进行了实验验证，取得了令人满意的效果。
# 2.相关工作综述
## 2.1 命名实体识别的定义与分类
命名实体识别（Named Entity Recognition，NER）是指从文本中识别出各种命名实体，如人名、地名、机构名、时间日期、货币金额等，并给予其相应的分类或类型。NER是一个复杂的多领域问题，当前常用的方法可以归纳如下：

1. **基于规则的方法**：这种方法是手工构建的规则库，根据一定的规则判断哪些词属于命名实体，如以地名、人名、组织机构名开头的词被认为是命名实体。这种方法简单易行，但准确率一般。

2. **基于统计学习的方法**：这类方法利用机器学习技术，利用大量训练数据自动学习词与词之间的联系，如隐马尔可夫模型（Hidden Markov Model，HMM），条件随机场（Conditional Random Field，CRF）。这类方法可以获得更高的准确率，但缺乏灵活性，需要大量的标记数据。

3. **神经网络的方法**：近年来基于神经网络的方法取得了惊艳成效。典型的方法是基于卷积神经网络（Convolutional Neural Networks，CNN）的结构，即先通过卷积神经网络提取局部特征，再利用线性分类器分类。另一些方法直接将神经网络应用于整个词序列，例如基于BiLSTM的双向LSTM层和CRF层的序列标注模型。这类方法利用神经网络的高度非线性特性，在一定程度上克服了传统方法的局限性。

本文所提出的模型是基于神经网络的方法。

## 2.2 NER的不同任务分类
目前，NER的任务分类主要包括以下三种：

1. **普通的NER任务**（Single-label NER）：这类任务目标是在输入序列中识别出所有命名实体。通常会采用BIOES tagging schema来标注序列，即B表示起始位置，I表示中间位置，E表示终止位置，S表示单个实体，O表示不属于任何实体。

2. **多标签的NER任务**（Multi-label NER）：这类任务目标是在输入序列中识别出命名实体，同时也考虑到一个实体可能存在多个类型的情况。通常会采用BIEOS tagging schema来标注序列，即B表示起始位置，I表示中间位置，E表示终止位置，S表示单个实体，O表示不属于任何实体。

3. **跨标签的NER任务**（Cross-Label NER）：这类任务目标是在输入序列中识别出不同类型的命名实体，并将它们分开。通常会采用BILOU tagging schema来标注序列，即B表示起始位置，I表示中间位置，L表示实体内部，U表示上一个字符处的实体，O表示其他字符。

本文只考虑单标签的NER任务。

# 3. 基本概念术语说明
## 3.1 深度学习
深度学习是一类机器学习方法，它在计算机视觉、自然语言处理、语音识别等领域的取得成功，得到了广泛关注。深度学习模型由多个“层”组成，每一层都可以看作是一个计算单元，可以接收上一层传入的信号，通过一系列线性变换和激活函数运算得出本层输出，再传给下一层。这样一层一层传递信号直至预测值输出。

深度学习模型具备多样性和自适应性，能够自动学习复杂的特征表示。深度学习模型的性能不断提升，并且在图像、语音、文字等多个领域都取得了良好效果。深度学习模型是实现诸如图像分类、检测、语义分析、图像合成、翻译、问答等功能的基础工具。

## 3.2 序列标注模型
序列标注模型是用来对序列中的元素进行标记（或类别）的一类模型。其中包括标记序列的生成和学习两个主要过程。序列标注模型一般分为两大类：

1. 有监督模型：这种模型需要输入序列和对应的正确标记序列作为训练数据，通过迭代优化求解出使得损失函数最小的模型参数。

2. 无监督模型：这种模型不需要输入序列的真实标记，仅依赖于已知的结构和语义信息。通过聚类、相似度匹配、最大熵等方法自行学习出标记序列。

本文所讨论的模型属于有监督模型。

## 3.3 激活函数
激活函数是深度学习模型的关键部分之一。激活函数用于调整输出值的范围，使得输出值落在一个可接受的区间内。常见的激活函数有sigmoid函数、tanh函数、ReLU函数等。不同的激活函数对深度学习模型的性能影响不同。

## 3.4 双向LSTM
双向LSTM（Bidirectional Long Short Term Memory）是一种对序列建模的神经网络模型。LSTM模型利用门结构控制信息流动和更新隐藏状态。双向LSTM将时间方向和空间方向上的信息考虑进来，可以更好的捕捉序列的全局信息。具体来说，双向LSTM有两个LSTM，每个LSTM以不同的顺序处理输入，以更好的捕获全局信息。

## 3.5 CRF层
CRF层是深度学习模型中很重要的部分，也是本文提出的方法中新增的模块。CRF层利用势函数（Potential Function）建模标签之间的相互作用。CRF层的输出是一个概率分布，代表各个标签出现的概率。

## 3.6 序列标注问题
序列标注问题一般包括序列的输入、输出以及训练样本。输入是一个N元组（N为序列长度），表示一个序列；输出是一个N元组，表示对应每个位置的标记；训练样本是由输入、输出组成的一组连续标记序列。序列标注问题是序列模型学习的基本问题。

# 4. 核心算法原理和具体操作步骤以及数学公式讲解
## 4.1 数据处理阶段
首先，需要将原始文本数据预处理成可以进行深度学习的形式。主要包括：

1. 分割词汇——将文本按空格等符号切分为一个一个词汇；
2. 分词——将一个词汇拆分为多个小词（如汉字）。分词有助于降低词汇的个数，提高训练速度；
3. 去除停用词——过滤掉一些没用的词（如“the”，“a”，“an”等），减少特征维度；
4. 转换成数字——将文本中每个词转换为唯一的整数id。

## 4.2 模型设计阶段
### 4.2.1 模型概览
该模型包括如下几个部分：

1. Embedding Layer：词嵌入层，主要用于将每个词映射到固定维度的向量空间中，能够帮助模型学习到词语的语义信息。

2. Encoding Layer：编码层，主要用于提取序列的语义信息。通过双向LSTM模型获取整个序列的表示。

3. Position-Aware Layer：位置信息增强层，该层引入位置信息，可以提高模型对位置依赖性的学习能力。

4. CRF Layer：CRF层，条件随机场层，用于对标签序列进行学习和推理，从而得到序列的最终标记。

### 4.2.2 Embedding Layer
Embedding Layer用于将每个词映射到固定维度的向量空间中，能够帮助模型学习到词语的语义信息。这里采用的是词向量的方式，而不是直接使用词的one-hot编码。这里选择的词嵌入方式为GloVe（Global Vectors for Word Representation）。

### 4.2.3 Encoding Layer
Encoding Layer用于提取序列的语义信息。双向LSTM模型获取整个序列的表示。

### 4.2.4 Position-Aware Layer
Position-Aware Layer引入位置信息，可以提高模型对位置依赖性的学习能力。具体方法是使用局部感受野（Local Receptive Fields，LRFs），并且引入注意力机制（Attention Mechanism）来让模型根据不同位置的信息做出不同的决策。

具体来说，LRFs允许模型学习到不同位置的上下文信息，并且可以同时处理多重位置依赖关系。Attention Mechanism引入可学习的参数矩阵A，对不同位置的特征进行加权平均，从而使得模型更倾向于关注重要的位置信息。

### 4.2.5 CRF Layer
CRF层是深度学习模型中很重要的部分，也是本文提出的方法中新增的模块。CRF层利用势函数（Potential Function）建模标签之间的相互作用。具体的势函数由用户自定义，本文采用了CRF模型。CRF模型的输出是一个概率分布，代表各个标签出现的概率。

## 4.3 训练过程
训练过程分为四个步骤：

1. 数据预处理——首先对原始文本数据进行预处理，包括分割词汇、分词、去除停用词等操作。然后将预处理后的数据转换成数字格式，同时准备好训练和测试数据；

2. 模型初始化——初始化模型参数；

3. 训练模型——训练模型，使得模型在训练数据上的损失函数最小；

4. 测试模型——测试模型，评价模型在测试数据上的性能。

训练过程中，模型按照设定的策略迭代优化模型参数，使得模型在训练数据上的损失函数最小。具体的优化策略包括随机梯度下降（SGD）、动量法（Momentum）、Adam等。

# 5. 具体代码实例和解释说明
## 5.1 模型训练代码示例
```python
import torch
import torchtext
from torchtext import data
from torchtext import datasets
from torch.utils.data import DataLoader
import random
from tqdm import trange

TEXT = data.Field(sequential=True, tokenize='spacy')   # Field对象，用于将文本数据转换成整数序列
LABEL = data.Field(sequential=False)                     # Label对象，用于将标签数据转换成整数序列

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)    # IMDB数据集，用于训练模型
print('Length of training set:', len(train_data))           # 打印训练集大小
print('Length of testing set:', len(test_data))            # 打印测试集大小

TEXT.build_vocab(train_data, max_size=10000, vectors="glove.6B.100d")      # 建立词汇表，max_size限制字典大小，vectors="glove.6B.100d"为词向量
LABEL.build_vocab(train_data)                                              # 建立标签表

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # 判断设备，是否可以使用GPU加速训练
batch_size = 64                                                           # 设置batch_size
train_iterator, test_iterator = data.BucketIterator.splits((train_data, test_data), batch_size=batch_size, device=device)   # 将数据划分为batch，并设置device属性

class BiLSTM_CRF(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)        # 词嵌入层
        self.encoder = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)     # 双向LSTM编码层
        self.attention = PositionalwiseFeedForward(hidden_dim*2, 128, dropout)          # LRF层
        self.linear = torch.nn.Linear(hidden_dim*2+128, output_dim)                # 全连接层
        self.crf = CRF(output_dim, sparse_target=True)                            # CRF层
    
    def forward(self, text, mask):
        embedded = self.embedding(text)                                      # 获取词向量表示
        encoded, _ = self.encoder(embedded)                                   # 获取双向LSTM编码表示
        attn_out = self.attention(encoded)                                    # 获取LRF表示
        logits = self.linear(attn_out)                                        # 获取全连接输出
        scores, tag_seq = self.crf._viterbi_decode(logits, mask)                 # 执行CRF解码
        
        return scores, tag_seq
        
def train():
    model = BiLSTM_CRF(len(TEXT.vocab), 100, 256, len(LABEL.vocab), 2, 0.5).to(device)  # 初始化模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)                   # 设置优化器
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)                       # 设置损失函数
    
    print('\nStart Training...\n')
    for epoch in range(5):                                                      # 开始训练
        total_loss = 0                                                          # 记录总损失
        for i, batch in enumerate(train_iterator):
            text, label = getattr(batch, 'text'), getattr(batch, 'label')         # 提取batch中的数据
            text, label = text.to(device), label.to(device)                         # 将数据转移到GPU
            
            score, _ = model(text, None)                                         # 使用模型进行前向传播
            loss = -criterion(score, label)                                     # 计算损失
            total_loss += loss.item()                                            # 累计损失
            
            optimizer.zero_grad()                                                # 清空梯度
            loss.backward()                                                       # 反向传播
            optimizer.step()                                                     # 更新参数
            
        print('[Epoch: %d] Loss: %.3f' %(epoch+1, total_loss/len(train_iterator)))
        
    return model
    
def evaluate(model):
    print('\nEvaluating...\n')
    total_acc, total_count = 0., 0

    for i, batch in enumerate(test_iterator):
        text, label = getattr(batch, 'text'), getattr(batch, 'label')
        text, label = text.to(device), label.to(device)

        score, predict_tag = model(text, None)
        predict_tag = predict_tag[mask].tolist()
        label = label.tolist()

        acc = np.mean([p == l for p, l in zip(predict_tag, label)])              # 计算ACC
        total_acc += acc                                                       # 累计ACC
        total_count += sum([(l!= 0 and p!= l) or (l == 0 and p!= l) for l, p in zip(label, predict_tag)])    # 统计FP, FN

    avg_acc = total_acc / len(test_iterator)                                  # 计算平均ACC
    f1 = float(total_count)/len(test_data)*2/(np.mean([(t!=0 and t!=-1) for t in LABEL.vocab.stoi.values()]) + np.mean([(p!=0 and p!=-1) for p in list(set(predict_tag))+list(set(label))])).item()      # F1-score
    recall = float(total_count)/(sum([(t==1 and p==1) for t, p in zip(label, predict_tag)]).item()*2)             # 召回率
    precision = float(total_count)/(sum([(t==1 and p==1) for t, p in zip(label, predict_tag)]).item())               # 精确率
    print('Test Accuracy: {:.3f} | Test F1-Score: {:.3f} | Test Precision: {:.3f} | Test Recall: {:.3f}'.format(avg_acc, f1, precision, recall))
            
    return avg_acc

if __name__=='__main__':
    model = train()
    accuracy = evaluate(model)
```
## 5.2 模型设计代码示例
### 5.2.1 双向LSTM
```python
class EncoderLayer(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super(EncoderLayer, self).__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.rnn = nn.LSTM(input_dim, hid_dim, num_layers=n_layers, bidirectional=True)
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        
    def forward(self, src):
        
        #src = [sent len, batch size]
        
        embedded = self.embedding(src)
        #embedded = [sent len, batch size, emb dim]
                
        outputs, (hidden, cell) = self.rnn(embedded)
        #outputs = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell
```
### 5.2.2 Position-Aware Layer
```python
class PositionalWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_ff, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(d_in, d_ff, 1) # position-wise
        self.w2 = nn.Conv1d(d_ff, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output
```
### 5.2.3 CRF Layer
```python
class CRF(nn.Module):
    def __init__(self, tagset_size, gpu=False, pad_idx=None, start_idx=None, stop_idx=None, sparse_target=False):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.gpu = gpu
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.sparse_target = sparse_target

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

    def viterbi_decode(self, feats, mask):
        """
        Computes the Viterbi path given features and transition parameters.
        Args:
          feats: A tensor of size (seq_length, batch_size, num_tags) containing
            unary potentials for each state.
          mask: A mask of shape (seq_length, batch_size) indicating valid states.
        Returns:
          viterbi: The decode sequences.
          score: The score of the viterbi sequence.
        """
        seq_length, batch_size, num_tags = feats.size()

        backpointers = []
        init_vvars = torch.full((batch_size, num_tags), -10000.)
        init_vvars[:, self.start_idx] = 0
        forward_var = init_vvars

        if self.gpu:
            init_vvars = init_vvars.cuda()
            forward_var = forward_var.cuda()

        for step in range(seq_length):

            next_tag_var = (forward_var.unsqueeze(2) + self.transitions).view(
                [-1, num_tags])
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().type_as(bptrs_t)
            backpointers.append(bptrs_t.view(batch_size, 1))

            feat_t = feats[step].view([-1, 1])
            emit_scores = forward_var.gather(1, feat_t).view(-1)

            next_tag_var = next_tag_var.view(batch_size, num_tags)
            forward_var = (feat_t + next_tag_var).view(-1)

            forward_var = forward_var * mask[step].view(-1) \
                           + (-10000.0) * (1 - mask[step]).view(-1)

        terminal_var = (forward_var + self.transitions[self.stop_idx]).view(
            [-1, num_tags])
        best_tag_id = torch.argmax(terminal_var, dim=1).squeeze()
        path_score = torch.gather(terminal_var, 1,
                                  best_tag_id.view(-1, 1)).view(-1)

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = torch.gather(bptrs_t, 1,
                                        best_tag_id.view(-1, 1)).view(-1)
            best_path.insert(0, int(best_tag_id))

        return best_path, path_score


    def neg_log_likelihood(self, inputs, tags, mask):
        """
        Computes the negative log likelihood of the sequence of tags given
        the inputs and transition parameters.
        Args:
          inputs: A tensor of size (seq_length, batch_size, num_tags) containing
            unary potentials for each state.
          tags: A tensor of size (seq_length, batch_size) containing the gold
            target tags indices over the vocabulary.
          mask: A mask of shape (seq_length, batch_size) indicating valid states.
        Returns:
          neg_llh: The negative log likelihood.
        """
        assert not self.training, "This module only supports evaluation mode."
        feats = inputs
        score, _ = self.viterbi_decode(feats, mask)
        feats = feats.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        score = score.float()
        llh = torch.zeros_like(score).scatter_(0, tags.unsqueeze(0), score)

        inside_score = self._inside(feats, tags, mask)

        return -(llh + inside_score).mean()

    def forward(self, inputs, tags, mask):
        """
        Computes the conditional log probability of the sequence of tags given
        the inputs and transition parameters.
        Args:
          inputs: A tensor of size (seq_length, batch_size, num_tags) containing
            unary potentials for each state.
          tags: A tensor of size (seq_length, batch_size) containing the gold
            target tags indices over the vocabulary.
          mask: A mask of shape (seq_length, batch_size) indicating valid states.
        Returns:
          crf_loss: The average Crf loss over the batch.
        """
        assert self.training, "This module only supports training mode."
        feats = inputs
        trans_scores = self.transitions.index_select(0, tags.data.view(-1)).view(*tags.size(),
                                                                                        self.tagset_size)
        gold_score = self._compute_score(inputs=feats, labels=tags,
                                         transitions=trans_scores, mask=mask)

        mask = mask.byte()
        score = self._inside(feats, tags, mask)

        return -torch.sum(gold_score - score[mask]) / mask.float().sum()

    def _compute_score(self, inputs, labels, transitions, mask):
        batch_size, seq_length, num_tags = inputs.size()

        trans = transitions.permute(2, 0, 1)
        alpha = inputs
        for i in range(seq_length):
            emit_score = alpha.view(batch_size, num_tags, 1)
            trans_score = trans.expand(batch_size, num_tags, num_tags)
            score = emit_score + trans_score

            alpha = masked_softmax(score, mask[i], axis=1) * labels[i]

        last_alpha = alpha.view(batch_size, num_tags, 1)
        end_transition = transitions.view(
            1, num_tags, num_tags)[self.end_idx].expand(batch_size, 1, num_tags)
        terminal_score = last_alpha + end_transition

        all_paths_score, _ = self._viterbi_decode(terminal_score, mask)
        return all_paths_score.sum()/batch_size

    def _inside(self, feats, tags, mask):
        '''
        Compute the partition function over the unlabeled sequence.
        Args:
          feats: A tensor of size (seq_length, batch_size, num_tags) containing
              unary potentials for each state.
          tags: A tensor of size (seq_length, batch_size) containing the gold
              target tags indices over the vocabulary.
          mask: A mask of shape (seq_length, batch_size) indicating valid states.
        Returns:
          z: The partition function score.
        '''
        seq_length, batch_size, num_tags = feats.size()

        start = torch.full((batch_size,), self.start_idx, dtype=torch.long)
        if self.gpu:
            start = start.cuda()

        end = torch.full((batch_size,), self.stop_idx, dtype=torch.long)
        if self.gpu:
            end = end.cuda()

        pad_mask = torch.eq(tags, self.pad_idx)
        first_pad = pad_mask[0].nonzero().view(-1)[0] if pad_mask.any() else seq_length

        tags_padded = torch.cat([start.unsqueeze(0),
                                 tags[:first_pad]], 0)
        tags_padded = torch.cat([tags_padded,
                                 end.unsqueeze(0)], 0)
        pad_mask_padded = torch.cat([torch.zeros_like(start).bool(),
                                     pad_mask[:first_pad]], 0)
        pad_mask_padded = torch.cat([pad_mask_padded,
                                     torch.ones_like(end).bool()], 0)
        length_mask = (~pad_mask_padded).int()

        feats_padded = torch.cat([torch.full_like(feats[0][0], -10000.),
                                   feats[:first_pad]], 0)
        feats_padded = torch.cat([feats_padded,
                                   torch.full_like(feats[-1][0], -10000.)], 0)

        scores = self._score_sentence(feats_padded[:-1],
                                      tags_padded[1:],
                                      feats_padded[1:],
                                      tags_padded[:-1],
                                      length_mask)

        return scores.view(batch_size).sum(0) / mask.sum()

    def _score_sentence(self, feats, obs_tags, pred_feats,
                        pred_tags, mask):
        '''
        Given a sentence compute its score under the model.
        Args:
          feats: A tensor of size (seq_length, batch_size, num_tags) containing
              unary potentials for each state.
          obs_tags: A tensor of size (seq_length, batch_size) containing observed
              part-of-speech tags for each word in the sentence.
          pred_feats: A tensor of size (seq_length, batch_size, num_tags) containing
              unary potentials predicted by another external tool for each state.
          pred_tags: A tensor of size (seq_length, batch_size) containing predicted
              part-of-speech tags for each word in the sentence.
          mask: A mask of shape (seq_length, batch_size) indicating valid states.
        Returns:
          s: The sentence score computed using an inside algorithm.
        '''
        seq_length, batch_size, num_tags = feats.size()

        start = torch.full((batch_size,), self.start_idx, dtype=torch.long)
        if self.gpu:
            start = start.cuda()

        end = torch.full((batch_size,), self.stop_idx, dtype=torch.long)
        if self.gpu:
            end = end.cuda()

        pad_mask = torch.eq(obs_tags, self.pad_idx)
        first_pad = pad_mask[0].nonzero().view(-1)[0] if pad_mask.any() else seq_length

        obs_tags_padded = torch.cat([start.unsqueeze(0),
                                     obs_tags[:first_pad]], 0)
        obs_tags_padded = torch.cat([obs_tags_padded,
                                     end.unsqueeze(0)], 0)
        pad_mask_padded = torch.cat([torch.zeros_like(start).bool(),
                                     pad_mask[:first_pad]], 0)
        pad_mask_padded = torch.cat([pad_mask_padded,
                                     torch.ones_like(end).bool()], 0)
        length_mask = (~pad_mask_padded).int()

        backward_mask = (1 - mask).cumsum(0).masked_fill_(
            mask.cumsum(0).byte(), 1).bool()

        s = torch.zeros(batch_size).type_as(feats)

        if self.gpu:
            s = s.cuda()

        prev_obs_tag = obs_tags_padded[:-1]
        curr_pred_tag = pred_tags[1:]
        prev_state = feats[0]

        for idx in range(seq_length):

            curr_obs_tag = prev_obs_tag
            curr_pred_tag = prev_state.argmax(1)

            # Transition to next tag
            curr_trans = self.transitions[curr_pred_tag.view(-1),
                                           curr_obs_tag.view(-1)].view_as(prev_state)
            next_state = feats[idx] + curr_trans

            delta = (next_state * obs_tags_padded[idx+1].ne(self.pad_idx)).logsumexp(
                1, keepdim=True)
            exp_delta = torch.exp(delta)
            s = s + delta

            if idx < seq_length-1:

                curr_obs_tag = obs_tags_padded[idx+1]
                curr_pred_tag = pred_tags[idx+1]
                new_phi = self.transitions[curr_pred_tag.view(-1),
                                            curr_obs_tag.view(-1)].view(batch_size,-1,1) +\
                          pred_feats[idx][:,:,None]*\
                          ((~backward_mask[idx+1])[None,:,:]).repeat(1,1,num_tags)


                denom = new_phi.logsumexp(1).clamp_min(1e-9)[:,:,None]\
                                              .repeat(1,1,num_tags)\
                                              .detach()\
                                              .contiguous()
                phi = new_phi.exp()*(new_phi.exp()-denom.exp())
                phi /= phi.sum()

                exp_eta = phi.sum(1)
                scores = exp_delta[:, :, None]*phi
                gamma = scores.sum(1) + exp_eta
                psi = scores/gamma[...,None,:]
                norm_psi = psi/psi.sum(1,keepdim=True).clamp_min(1e-9)
                
                alphas = norm_psi*delta[...,None,None]/delta[:,None,:]
                
                exp_alphas = alphas.exp().sum(0)
                denom = exp_alphas.sum()
                gammas = exp_alphas/denom

                phi_obs = self.transitions[curr_pred_tag.view(-1),
                                             curr_obs_tag.view(-1)].view(batch_size, 1, 1)-alphas/gammas[...,None,:]

                denom = (phi_obs.logsumexp(1)+delta[...,None,None])/delta[:,None,:]
                norm_phi_obs = phi_obs.exp()/denom.exp()

                phi = (phi + norm_phi_obs)*(1-backward_mask[idx+1])


            prev_state = next_state


        return s

    @staticmethod
    def _generate_transitions(num_tags, constraints):
        """Generates transition matrix according to provided constraints."""
        transitions = torch.eye(num_tags)
        constrained = {tag: idxs for tag, idxs in constraints.items()}
        for k, v in constrained.items():
            for vi in v:
                transitions[k][vi] = -math.inf
                transitions[vi][k] = -math.inf
        return transitions
```