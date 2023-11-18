                 

# 1.背景介绍


GPT (Generative Pre-trained Transformer) 大模型是一种基于transformer机器翻译模型的自然语言生成模型，其中包含了多达十亿参数的神经网络模型，能够根据输入序列生成连续文本输出。
随着NLP和智能助手的迅速普及，越来越多的人开始将GPT大模型应用到日常工作中，用于业务流程自动化任务，例如，在电商平台进行商品销售订单自动生成、在HR系统进行员工培训考核结果生成、在客服系统进行咨询问题自动回复等。
GPT大模型可以用于实现自动化业务流程自动化的关键技术之一，但是，如何使用它来完成各类业务流程自动化任务，并对其发展产生重大影响？本文将全面剖析GPT大模型的技术原理与应用，并通过实践案例介绍如何用GPT大模型构建企业级的自动化业务流程自动化应用。
# 2.核心概念与联系
## GPT模型简介
GPT（Generative Pre-trained Transformer）模型是一个深度学习模型，它使用预训练好的transformer网络结构，并对语料库进行微调，进而生成连续文本，并且被证明比LSTM、GRU等标准RNN结构生成器更有效。
该模型由两部分组成：编码器（Encoder）和解码器（Decoder）。
### Encoder：
编码器主要负责把输入序列转换成上下文向量，也就是把输入序列中的每个单词或者符号映射成为一个固定维度的向量表示。
### Decoder：
解码器负责基于上下文向量生成目标序列。解码器接收上一步的输出作为当前输入，然后生成下一步的输出。
### Masked Language Model（MLM）：
Masked Language Model（MLM）用于解决文本生成任务中，输入数据中存在缺失的部分或不需要关注的内容，即所谓的遮蔽语言模型。MLM使用随机遮盖语言模型（RandLM），这个模型把输入序列中的任意位置替换成[MASK]标签，模型就会预测这个标签对应的单词或符号。当标签被预测正确时，模型就知道输入数据中有遮蔽的部分，否则就会认为生成的数据没有遮蔽的部分。因此，MLM可以帮助模型预测出遮蔽的部分并填补完整。
### Next Sentence Prediction（NSP）：
Next Sentence Prediction（NSP）用于解决语言模型的上下文理解问题，即判断两个句子之间是否有逻辑关系。NSP模型在做语言建模的时候，会同时给两个句子加上不同的标签，比如is_next和not_next。当模型看到两个连续的句子，且前者是后者的下一句话时，就会预测为is_next，反之则预测为not_next。这也是通过NSP模型来解决生成文本时的上下文理解的问题。
### 模型架构：
整体的模型架构如下图所示：
## RASA(Reinforcement Learning Assistance Systems) 框架
RASA框架是由Rasa Technologies公司于2018年推出的开源AI领域的机器人助手软件，旨在为企业提供一系列的解决方案，包括实体识别、意图识别、对话管理、自然语言生成、交互式学习、持续集成、监控以及部署支持。
RASA通过在自然语言理解与表达方面建立领先的技术水平，将自然语言处理模块与智能对话模块紧密结合，能够通过将自然语言交流场景映射到自然语言理解和生成的算法，完成复杂的聊天机器人的功能。RASA框架的精妙之处在于，它不仅能够实现自然语言理解与表达模块的功能，还能够通过强大的交互学习机制，有效地提升模型的鲁棒性和易用性。
# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## GPT模型——Transformer模型
本节将从Transformer模型的基础原理和Transformer模型的GPT模型架构角度出发，详细介绍GPT模型的基本原理和GPT模型的架构。
### Transformer模型的基础原理
Transformer模型是2017年由Google Brain团队提出的论文“Attention is All You Need”的变种，它的主要特点是在标准的RNN、CNN等传统模型的基础上，使用注意力（attention）机制解决长序列信息建模的问题。这里，作者们提出了一个全新的自注意力机制，使得模型可以充分捕获到输入序列之间的关联性，而不是依赖于固定的时间步长。为了做到这一点，作者们设计了一个基于注意力机制的新颖结构——Self-Attention层，其目的是要允许模型学习到输入序列内部的联系，并非只是依赖于单个的时间步长。
#### Self-Attention层
首先，输入序列x是由若干个序列元素组成的，例如，对于文本分类任务来说，输入序列就是一串文本序列，其长度为$T$；对于图片分类任务来说，输入序列就是一张张图像序列，其长度为$H \times W \times C$ 。
然后，Self-Attention层首先计算输入序列x的每个元素之间的相似性矩阵Q。假设输入序列x的第i个元素是xi，那么相似性矩阵Qi就为：
$$ Q_{i,:} = v^TQ_iv $$ （1）
其中，$Q_i$是一个长度为$n$的向量，代表输入序列第i个元素的特征向量；$v$也是一个长度为$n$的权重矩阵；$n$通常远小于$T$或$C$。在实际计算过程中，Self-Attention层使用softmax函数将上述相似性矩阵Qi归一化，得到标准化后的权值系数Si：
$$ Si_{ij} = \frac{exp(Q_{i,j})}{\sum_{k=1}^{n}{exp(Q_{i,k})}} $$ （2）
其目的就是让不同输入序列元素之间的关系都能够得到比较准确的表示，而不是只考虑相邻的时间步。
接下来，Self-Attention层还要计算每个输入序列元素与输出序列所有元素的相关性矩阵K和V，并将它们乘以相应的权重矩阵。由于输入序列中的每一个元素都与整个输出序列相关，所以输出序列也必须包含所有可能的输出情况。所以，Self-Attention层还要将输入序列与输出序列的全连接层连接起来。
最后，Self-Attention层需要为模型学习到的关联性赋予权重。这一点通过一个软正交函数完成，即$\text{Softmax}(Wx+b)$，其中W和b是可学习的参数。
综上，Self-Attention层的作用是为每个输入序列元素分配权重，从而实现输入序列之间的关联性建模。
#### Transformer模型的GPT模型架构
GPT模型中，Encoder是基于Self-Attention层的堆叠。GPT模型的Decoder与其他的模型类似，也是由多个Self-Attention层和全连接层组成。区别在于，GPT模型的Decoder不再依赖于固定的时间步长，而是直接通过前一步的输出来预测下一步的输出。
Encoder与Decoder的Stacked Self-Attention块如下图所示：
### GPT模型
#### 生成概率计算
GPT模型的目标是生成连续的文本序列，因此，除了计算输入序列的相关性矩阵外，还需要计算目标序列的相关性矩阵。由于目标序列是由输入序列导出的，因此，目标序列的相关性矩阵只需要从输入序列中抽取几个元素即可。那么，怎样确定哪些元素才是合适的呢？这就是GPT模型中关键的一步——生成概率计算。
##### 采样策略
在生成序列时，我们采用采样策略。具体来说，GPT模型不直接给出完整的序列，而是以一个标记的token作为输入，要求模型生成该token之后的下一个token，并重复此过程直至生成结束或达到指定长度限制。这种采样策略称作贪婪搜索（greedy search）。但这种方法效率低下，且难以控制生成效果。
为了降低贪婪搜索的困难程度，GPT模型采用带噪声的采样策略。具体来说，每一步生成都是从一个概率分布中选择一个token。但这种方法要求模型能够产生连贯且逼真的文本。GPT模型使用另一种策略，即通过熵正则项约束模型生成的文本。具体来说，对于每一个token，模型都必须为其分配一个概率，并且这个概率应该与模型的预测输出有关。具体而言，模型的预测输出会指导模型下一步的行为，并且会影响模型对这个token的选择。为此，GPT模型会使用语言模型（language model）计算当前的输入token的联合概率。语言模型衡量的是当前输入token的历史状态，以及之后的输出序列。因此，语言模型对于序列生成任务具有不可替代的作用。
GPT模型通过以下公式计算token的生成概率：
$$ P(token_i|token_1^{i-1}, past_{i-1}) = \frac{\text{softmax}(LMS_{token_i}|past_{i-1})}{\sum_{t\in V}{\text{softmax}(LMS_{t}|past_{i-1})}} $$ （3）
其中，$V$为输入语料库中的所有可能的token集合；$past_i$表示模型的前$i-1$个输出token，是一个$batch \times seq \times dim$的tensor，其中dim为模型隐层维度。$LMS_{token_i}$表示模型预测当前token为token_i的概率，可以由下面的公式计算：
$$ LMS_{token_i} = g(h_i)^T\log p(token_i|\text{all previous tokens in sequence}; w_i) $$ （4）
其中，$g(h_i)$是模型的输出表示；$p(token_i|\text{all previous tokens in sequence}; w_i)$是语言模型，表示模型的预测输出序列中第i个token的联合概率。
##### Nucleus Sampling（扰动抽样）
Nucleus Sampling（扰动抽样）是一种自适应抽样技术，能够控制模型生成的连贯性。具体来说，Nucleus Sampling会限制模型可能选取的token，以减少模型生成连贯性较差的文本。
Nucleus Sampling的基本思想是，从模型预测出的可能性分布中，按照一定的概率将高置信度的token折叠掉，并只保留一定比例的最低置信度的token。这样做的目的是，尽量避免模型过于依赖于某一类的token，而丢弃其他类的token。
具体来说，Nucleus Sampling将模型预测出的概率分布按概率大小排序，并选取一定比例的最小概率（即$P_\tau(\theta)$，$\theta$是置信度阈值）的token，其余的token则舍弃。其中，$\tau$是超参，控制模型的置信度，通常取值为$0.5$或$0.9$。
##### 回溯策略
在贪心搜索方法中，如果遇到模型不能预测的token，只能跳过该token，从头开始生成新的序列。在实际生产环境中，这种策略存在很大的局限性。如果模型无法继续生成新序列，则只能停止生成。但是，模型对停滞状态的判断并不是完全可靠的。有时候，模型虽然不能生成新序列，但是仍然有可能进入到更糟糕的局面，这就是所谓的回溯问题。为了防止这种情况发生，GPT模型引入了回溯策略。具体来说，当模型不断错乱地生成序列时，模型会尝试回退到之前生成的序列，并尝试修正错误。具体而言，模型会记录之前生成的序列及其对应的概率，并利用这些信息来修正当前序列的生成。
#### 训练优化策略
##### Negative Sampling
Negative Sampling是一种训练策略，能够减少模型的内存消耗。具体来说，模型以负样本的方式损失目标序列，鼓励模型学习到较短的、语法正确的序列。而正样本则对应着完整的、可理解的序列。Negative Sampling将正样本视作真实信号，负样本视作虚假信号，因此，模型能够更多关注正样本。
GPT模型使用Negative Sampling，具体来说，为了训练GPT模型，模型需要输入一个序列和一个标签，其中标签表示输出序列的概率分布。在Negative Sampling中，模型不仅需要正确地预测目标序列的token，而且还需要预测一些错误的token。为此，模型会随机地从输入序列中抽取一定数量的负样本token，并把它们拼接在目标序列的末尾。
例如，给定一个输入序列：“The quick brown fox jumps over the lazy dog”，模型需要生成一个目标序列，如：“the quick brown dog barks”。为了训练模型，模型需要为每个token分配一个概率，而模型也会为一个输入序列可能的所有可能的目标序列分配一个概率。为此，模型可以随机地从输入序列中抽取一个负样本，如“fox jump over eagerly”；然后，模型就可以在正样本和负样本两种情况下，为目标序列分配一个概率，并最大化这个概率。
##### Knowledge Distillation
Knowledge Distillation是一种训练策略，能够增强模型的泛化能力。具体来说，当模型以弱监督方式学习时，模型会获得较差的性能。但如果模型能够以强化学习的方式学习到知识，那么它就能提升它的泛化能力。
知识蒸馏是指利用一个较好（也称为teacher）的预训练模型，去增强（distill）一个较差（也称为student）的模型的性能。知识蒸馏的原理是：Teacher模型学习复杂的任务，并且输出了易于理解的中间表示。而Student模型接收到了这些中间表示，并且试图复现Teacher的表现。知识蒸馏的目标就是让Student模型学会从Teacher那里学习到知识。知识蒸馏一般分为两步：第一步，Teacher模型根据特定任务对模型进行训练；第二步，Student模型接收到Teacher的中间表示，并根据自己的任务进行fine tuning，以增强模型的性能。
GPT模型使用了知识蒸馏的方法，具体来说，它使用一个深度监督网络（DSN）作为Teacher模型，负责对输入的句子进行分类。DSN是一个多层感知机网络，它接受句子的词汇表，并对输入的句子进行分类。给定一个输入句子，DSN可以输出二分类结果，其中0表示不属于某个类别，1表示属于某个类别。学生模型（GPT）则是一个Transformer模型，它接受DSN的中间表示，并尝试复现DSN的表现。GPT模型使用学生模型的中间表示作为输入，并输出了一系列的token。最后，GPT模型会用正负样本的训练方法，进一步增强模型的性能。
# 4.具体代码实例和详细解释说明
## 数据集准备
本次实验使用了一个基于MovieLens的评价数据集。数据集包含了6040个用户对4800部电影的评价，共有10个属性：
- userId：用户ID
- movieId：电影ID
- rating：用户对电影的评分
- timestamp：评价时间戳
- releaseYear：电影发行年份
- genre：电影风格
- title：电影名称
- imdbUrl：IMDb链接
- tmdbUrl：TMDB链接
## 模型搭建
### DSN模型
DSN模型是一个多层感知机网络，它接受句子的词汇表，并对输入的句子进行分类。给定一个输入句子，DSN可以输出二分类结果，其中0表示不属于某个类别，1表示属于某个类别。DSN模型的代码如下：

```python
import torch
from transformers import BertTokenizer, BertModel


class DSNModel(torch.nn.Module):
    def __init__(self, num_labels, hidden_size):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        
        self.classifier = torch.nn.Linear(hidden_size*2, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            
        last_hidden_state = outputs[0]   # 获取BERT模型的最后一层隐藏层输出
        cls_embedding = last_hidden_state[:, 0]    # 获取BERT模型的[CLS] token的隐含表示
        logits = self.classifier(torch.cat([last_hidden_state, cls_embedding], dim=-1))   # 将BERT模型的最后一层隐藏层输出和[CLS] token的隐含表示拼接起来，输入到分类器中，得到分类的结果
        
        return logits
    
```

DSN模型继承于`torch.nn.Module`，初始化的时候，调用`BertTokenizer`和`BertModel`类，获取BERT模型的tokenzier和bert。设置一个分类器`classifier`，用来将BERT模型的最后一层隐藏层输出和[CLS] token的隐含表示拼接起来，输入到分类器中，得到分类的结果。`forward()`函数定义了DSN模型的前向传播过程。
### GPT模型
GPT模型是一个基于Transformer模型的语言模型。GPT模型的Decoder同其他模型一样，也是由多个Self-Attention层和全连接层组成。不同的是，GPT模型的Decoder不再依赖于固定的时间步长，而是直接通过前一步的输出来预测下一步的输出。

```python
import torch
from transformers import GPT2Config, GPT2Model

class GPTModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout_prob):
        super().__init__()

        config = GPT2Config(vocab_size=vocab_size, 
                            n_embd=embedding_size, 
                            n_head=hidden_size//64, 
                            n_layer=num_layers, 
                            resid_pdrop=dropout_prob, 
                            embd_pdrop=dropout_prob,
                            attn_pdrop=dropout_prob)

        self.gpt2 = GPT2Model(config)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        encoder_outputs = self.gpt2.encoder(input_ids=input_ids,
                                            attention_mask=attention_mask)
        
        decoder_outputs = self.gpt2.decoder(inputs_embeds=None,
                                            attention_mask=None,
                                            inputs_embeds=decoder_input_ids)
        return decoder_outputs
    
    def generate(self, input_ids, attention_mask, max_length, temperature, top_k, top_p, device='cpu'):
        generated_tokens = []
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids, attention_mask, input_ids)
                
                next_token_logits = outputs[..., -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)

                if next_token == self.eos_token:
                    break
                
                generated_tokens.append(next_token.tolist())
                    
                input_ids = torch.cat((input_ids, next_token.unsqueeze(-1)), dim=-1)
                
        return generated_tokens[:-1]

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

```

GPT模型继承于`torch.nn.Module`，初始化的时候，创建了一个`GPT2Config`对象，设置了GPT模型的配置。创建了`GPT2Model`对象，用来初始化GPT模型。设置了一个分类器`classifier`。`forward()`函数定义了GPT模型的前向传播过程，`generate()`函数定义了GPT模型的生成过程。生成过程中，每次迭代都会先通过GPT模型生成一个token，然后把它加入到输入序列中，并作为下一次输入。
### 训练过程
#### 数据预处理
加载数据集并进行预处理，构造train DataLoader和test DataLoader。数据集经过预处理后，用BERT Tokenizer进行分词，然后对每个句子进行padding，使得长度为512。
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def load_data():
    data = pd.read_csv('../datasets/movie_reviews.csv')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    sentences = data['review'].values[:2000]
    labels = data['sentiment'].values[:2000]

    encoded_sentences = [tokenizer.encode(sent, add_special_tokens=True, truncation=True, max_length=512) for sent in sentences]
    padded_sentences = pad_sequences(encoded_sentences, maxlen=512, padding="post", dtype="long")

    X_train, X_test, y_train, y_test = train_test_split(padded_sentences, labels, test_size=0.1, random_state=42)

    train_dataset = TensorDataset(torch.LongTensor(X_train),
                                  torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.LongTensor(X_test),
                                 torch.FloatTensor(y_test))

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=8)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=8)

    return train_dataloader, test_dataloader
```

#### 训练DSN模型
创建DSN模型，并进行训练，返回训练好的模型。
```python
import torch

dsn_model = DSNModel(num_labels=2, hidden_size=768)
optimizer = torch.optim.AdamW(params=filter(lambda x: x.requires_grad, dsn_model.parameters()), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    total_loss = 0.0
    correct = 0
    count = 0
    for i, (input_ids, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = dsn_model(input_ids.cuda(), None, None)
        loss = criterion(output, label.cuda().long())
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
        total_loss += loss.item()*label.size(0)
        correct += (predicted.cpu()==label.numpy()).sum()
        count += len(label)
    print("Epoch {}/{} Train Loss:{:.4f} Acc:{:.4f}".format(epoch+1, 10, total_loss/count, correct/count))
    

print("Training Done!")
torch.save(dsn_model.state_dict(), './models/dsn.pth')
```

#### 训练GPT模型
创建GPT模型，并进行训练，返回训练好的模型。
```python
import torch

train_loader, val_loader = load_data()
gpt_model = GPTModel(vocab_size=len(tokenizer), embedding_size=768, hidden_size=768, num_layers=12, dropout_prob=0.1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpt_model.to(device)
optimizer = torch.optim.AdamW(params=filter(lambda x: x.requires_grad, gpt_model.parameters()), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)


def train(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    counter = 0
    for idx, sample in enumerate(loader):
        input_ids = sample[0].to(device)
        target_ids = sample[0].to(device)
        mask = sample[1].to(device)
        target_mask = mask[:, 1:].contiguous()
        preds = model.forward(input_ids=input_ids,
                              attention_mask=mask,
                              decoder_input_ids=target_ids[:, :-1])
        pred_logits = preds.logits[:, :-1, :].contiguous().view(-1, preds.logits.shape[-1])
        real_logits = target_ids[:, 1:].contiguous().view(-1)
        loss = criterion(pred_logits, real_logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        _, preds = torch.max(preds.logits, dim=-1)
        same = torch.eq(preds, target_ids[:, 1:]) * target_mask[:, 1:]
        accu = torch.mean(same.float())

        running_loss += loss.item()
        running_correct += accu.item()
        counter += 1
    avg_loss = running_loss/counter
    accuracy = running_correct/counter
    return {'loss':avg_loss,'acc':accuracy}


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    counter = 0
    for idx, sample in enumerate(loader):
        input_ids = sample[0].to(device)
        target_ids = sample[0].to(device)
        mask = sample[1].to(device)
        target_mask = mask[:, 1:].contiguous()
        preds = model.forward(input_ids=input_ids,
                              attention_mask=mask,
                              decoder_input_ids=target_ids[:, :-1])
        pred_logits = preds.logits[:, :-1, :].contiguous().view(-1, preds.logits.shape[-1])
        real_logits = target_ids[:, 1:].contiguous().view(-1)
        loss = criterion(pred_logits, real_logits)

        _, preds = torch.max(preds.logits, dim=-1)
        same = torch.eq(preds, target_ids[:, 1:]) * target_mask[:, 1:]
        accu = torch.mean(same.float())

        running_loss += loss.item()
        running_correct += accu.item()
        counter += 1
    avg_loss = running_loss/counter
    accuracy = running_correct/counter
    return {'loss':avg_loss,'acc':accuracy}


best_val_acc = float('-inf')
total_steps = len(train_loader)*10
for epoch in range(10):
    train_result = train(gpt_model, train_loader, optimizer, scheduler, criterion, device)
    val_result = evaluate(gpt_model, val_loader, criterion, device)
    print("Epoch:{}/{}, train loss:{:.4f}, train acc:{:.4f}, val loss:{:.4f}, val acc:{:.4f}\n".format(epoch+1, 10, train_result['loss'], train_result['acc'], val_result['loss'], val_result['acc']))
    if best_val_acc<val_result['acc']:
        best_val_acc = val_result['acc']
        torch.save({'epoch':epoch+1,'state_dict':gpt_model.state_dict()}, '../results/checkpoint.tar')
        

print("Training Done!")

checkpoint = torch.load('../results/checkpoint.tar')
gpt_model.load_state_dict(checkpoint['state_dict'])
```