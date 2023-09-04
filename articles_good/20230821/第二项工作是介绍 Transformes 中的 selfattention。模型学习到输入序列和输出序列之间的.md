
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Self-Attention机制（来自论文 Attention Is All You Need）是 Transformers 模型的核心组成部分之一。它的主要思想是允许模型能够同时关注输入序列中的不同位置的信息。这一机制使得模型能够捕捉到不同位置上的相关信息，从而能够处理长输入序列的问题。

传统的RNN、CNN等模型通常是根据输入单词的顺序进行建模。而在Transformers中，每一个token都可以被看做是input sequence的一部分。因此，Transformers模型中的self-attention机制则不再局限于固定单个时间步的状态表示，而是将整个input sequence视作input representation。这样做的一个重要原因是：从本质上来说，语言是一种表征形式，而非像图像或音频一样只是连续的一段数据流。

图1展示了一个简单的示例。假设有一个输入句子"The quick brown fox jumps over the lazy dog"，那么该句子可以被转换为词序列[“the”, “quick”, “brown”, “fox”, “jumps”, “over”, “lazy”, “dog”]。这里的每个词都可以作为一个input token，然后经过编码器编码得到一个向量表示。在传统RNN模型中，每个时间步只能看到当前的时间步及之前的历史信息，不能观察后面的信息。但在Transformers中，每个token都可以作为整体来考虑，也可以考虑其前后的信息。比如说，如果当前token是"fox"，就应该知道这个词后面跟着"quick"、"brown"、"jumps"等关键词；或者如果当前token是"over"，也许要找寻一下与之对应的"lazy"、"dog"两词。如此一来，自注意力机制就可以帮助模型捕捉到不同位置的信息。



图1: 图示了一个简单输入句子的Encoder-Decoder结构，其中包括self-attention机制。Encoder模块主要负责把输入序列表示为一个固定长度的向量表示，Decoder模块则是使用解码器对生成结果进行推理并得到最终的预测序列。

# 2.基本概念术语说明
## 2.1 Self-Attention
Self-Attention就是指模型学习到输入序列和输出序列之间的全局关联性，它通过计算输入序列中每个词与其他所有词之间的注意力权重来实现这一点。举个例子，对于输入序列 "The quick brown fox jumps over the lazy dog"，假设模型需要生成 "A quick brown fox jumped over a sleeping lion." 的输出序列。模型需要同时考虑到源序列和目标序列的信息。可以这样理解：当生成下一个单词时，模型只关心自己现在正在生成的内容以及与此相关的那些已经生成的单词。Self-Attention就能够帮我们解决这个问题。具体做法是，模型首先计算出输入序列中每个词的向量表示。然后，模型在这些向量之间应用注意力机制，得到每个词与其他所有词之间的注意力权重。最后，模型利用这些权重调整各个向量的权重，以便能够生成目标序列。


图2: Self-Attention的计算过程。左侧为Encoder的过程，右侧为Decoder的过程。红色方框为注意力计算区域，绿色框为结果输出区域。

## 2.2 Transformer层级架构
Transformes由多个Transformer层级构成。每个层级都由多头注意力机制和基于位置的前馈网络(position-wise feed-forward networks)两个模块组成。

多头注意力机制即使将输入序列中的所有元素与其他元素建立联系，也是采用了多头注意力机制。事实上，每个层级的多头注意力机制的个数可以是不同的。由于每个头可以有效地关注不同范围的输入序列信息，所以这使得模型学习到全局关联性。对于每个头，模型可以学习到不同位置上的相关信息。

基于位置的前馈网络即是两层全连接网络，它的目的是将每个向量映射到另一个空间维度。在Transformer中，这种网络一般用于处理位置编码信息。

## 2.3 Positional Encoding
Positional encoding是一种用来描述相对位置关系的编码方式。Positional encoding可以帮助模型更好地关注距离当前词或者位置很远的词。Positional encoding的目的就是为了将绝对位置信息转化成相对位置信息，因为相同的词出现在不同的位置会影响上下文特征。

Positional encoding可以由以下公式来表示：PE_{pos, 2i} = sin(\frac{pos}{10000^{\frac{2i}{d_model}}})，PE_{pos, 2i+1} = cos(\frac{pos}{10000^{\frac{2i}{d_model}}})，其中 pos 表示当前词的位置，d_model 是模型的维度。具体来说，给定序列的长度L，位置编码矩阵 P 是一个 L x d_model 的矩阵。

## 2.4 Scaled Dot-Product Attention
Scaled dot-product attention是在标准的dot-product attention之上增加了一个缩放因子来控制softmax的温度。具体来说，Scaled dot-product attention的公式如下：
$$
\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
其中 Q、K 和 V 分别是查询、键和值。softmax函数用来计算注意力权重，其计算公式如下：
$$
\text{softmax}(x_{i j})=\frac{\exp (x_{i j})}{\sum_{j=1}^{n}\exp (x_{i j})}
$$
除此之外，还可以使用dropout来防止过拟合。

## 2.5 Multi-head Attention
Multi-head attention其实就是通过多个头来并行计算注意力。每个头都可以关注到不同的输入范围，因此可以获取到不同的注意力权重。因此，模型可以学习到不同位置上的相关信息。具体来说，每个头的公式如下：
$$
\text{Attention}(Q_{\alpha}, K_{\beta}, V_{\gamma})=\text{Concat}(\text{Head}_1,\dots,\text{Head}_h)\\
\text{Head}_i= \text{Attention}(\text{Q}_iW_q^{\alpha},\text{K}_iW_k^{\beta},\text{V}_iW_v^{\gamma})
$$
其中 α、β 和 γ 分别表示第 i 个头的权重。因此，整个模型的计算可以分解为多个头的并行计算，提升模型的性能。

## 2.6 Feed Forward Networks
Feed forward networks是用于处理输入数据的前馈神经网络。在Transformer模型中，feed forward network是由两层线性变换组成的，它们分别是多层感知机（MLP），激活函数 ReLU，输出层。用公式表示的话：
$$
FFN(x)=ReLU(G(xW_1)+b_1)\circ F(xW_2)+b_2
$$
其中 G 函数表示第一层 MLP，F 函数表示第二层 MLP，⊗ 为 hadamard 乘积符号。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 3.1 Self-Attention计算公式
### （1）计算Query、Key、Value矩阵
Query矩阵的计算公式如下：
$$
Q=WQ\in R^{l_q\times n_q}
$$
其中 WQ 是可训练的参数，Rq是随机初始化的矩阵，Rq是 Wq 的正交矩阵，n_q是 Query 矩阵的列数，l_q是 Query 矩阵的行数。类似的，Key 和 Value 矩阵的计算公式如下：
$$
K=WK\in R^{l_k\times n_k}\\
V=WV\in R^{l_v\times n_v}
$$
其中 l_k、l_v 和 n_k、n_v 分别是 Key 矩阵、Value 矩阵的行数和列数。

### （2）计算注意力权重
注意力权重的计算公式如下：
$$
A=softmax(\frac{QK^T}{\sqrt{d_k}})
$$
其中，QK^T 是矩阵乘积运算。softmax 函数计算注意力权重，其计算公式如下：
$$
\text{softmax}(x_{i j})=\frac{\exp (x_{i j})}{\sum_{j=1}^{n}\exp (x_{i j})}
$$
除此之外，还可以使用dropout来防止过拟合。

### （3）计算输出矩阵
输出矩阵的计算公式如下：
$$
\text{out}=A(VW^\top+\bar{p})\in R^{l_v\times m}
$$
其中 A 为注意力权重矩阵，V 为 Value 矩阵，m 是输出矩阵的列数，p 为位置编码矩阵。

## 3.2 Transformer层级架构


图3: Transformer模型的结构示意图。左侧为Encoder的过程，右侧为Decoder的过程。第一个Encoder是Encoder Layer，它由两个子层：Multi-head Attention 层和 Position-wise Feed-Forward 层组成。第二个Encoder也是同样的结构。最后，Encoder的输出和输入向量长度相同。第三个Decoder是Decoder Layer，它与第一个Encoder完全相同，区别在于 Decoder 增加了第三个子层 Masked Multi-head Attention 层。Masked Multi-head Attention 层用来遮蔽掉目标序列里已经生成的词。最后，Decoder的输出和输入向量长度相同。除此之外，还有 Source Language Embedding 层和 Target Language Embedding 层。Source Language Embedding 层和 Target Language Embedding 层用来将输入句子转换成词向量。

## 3.3 Scaled Dot-Product Attention 和 Multi-head Attention 的组合


图4: Scaled Dot-Product Attention 和 Multi-head Attention 的组合示意图。左侧为Encoder的过程，右侧为Decoder的过程。每个子层都是先完成 Scaled Dot-Product Attention 操作，然后在完成并行运算。然后将结果拼接起来得到整个输出。

## 3.4 Positional Encoding 和 Feed Forward Network 结合


图5: Positional Encoding 和 Feed Forward Network 结合示意图。左侧为Encoder的过程，右侧为Decoder的过程。输出矩阵经过 Linear Transformation 和 DropOut 后，和 Positional Encoding Matrix 相加，形成新的输出矩阵。新的输出矩阵输入 Feed Forward Network 进行进一步的处理。

# 4.具体代码实例和解释说明
## 4.1 数据集：用于文本翻译任务的数据集是 IWSLT'14 English-German 翻译任务数据集。
### （1）下载数据集
```python
import torchtext
from torchtext.datasets import TranslationDataset
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')
train_dataset, valid_dataset, test_dataset = TranslationDataset(
    path='iwslt2016', exts=('.en', '.de'), fields=(
        ('src','src_raw'), ('trg', 'trg_raw')), 
    filter_pred=lambda x: len(vars(x)['src']) > 0 and len(vars(x)['trg']) > 0,
    tokenizer=tokenizer)
```
### （2）加载数据集
```python
BATCH_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    def create_padded_sequence(seqs):
        seqs = [torch.LongTensor(seq).to(device) for seq in seqs]
        lens = torch.LongTensor([len(seq) for seq in seqs]).to(device)
        padded_seqs = nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=PAD_IDX)
        return padded_seqs, lens
    
    def merge_two_dicts(x, y):
        z = x.copy()   # start with x's keys and values
        z.update(y)    # modifies z with y's keys and values & returns None
        return z

    # sort by source sequence length to use packed_padded_sequences later on
    sorted_batch = sorted(batch, key=lambda x: len(x['src']), reverse=True)

    # extract fields from different dictionaries into separate lists
    batches = [{key: [] for key in sorted_batch[0]} for _ in range(len(sorted_batch))]
    for data in sorted_batch:
        for key in data:
            batches[-1][key].append(getattr(data, key))
            
    # convert list of dicts into dictionary of lists
    result = {}
    for key in sorted_batch[0]:
        result[key], lengths = zip(*[(seqs[idx], len(seqs[idx]))
                                      for idx, seqs in enumerate(batches)])
        
    result = {key: torch.stack(values)
              for key, values in result.items()}
    
    # add additional fields required for transformer model
    result["masks"] = ~(result["src"].eq(PAD_IDX)).unsqueeze(-2)
    result["target"] = merge_two_dicts(result, {"src": None,
                                                 "trg": None,
                                                 "masks": None})["trg"]
    result["target_mask"] = (~merge_two_dicts(result, {"src": None,
                                                       "trg": None,
                                                       "masks": None})
                             ["trg"].eq(PAD_IDX)).unsqueeze(-2)
    result["target_shift"] = torch.cat((merge_two_dicts(result, {"src": None,
                                                                "trg": None,
                                                                "masks": None})
                                        ["trg"][...,:1]*0,
                                       merge_two_dicts(result, {"src": None,
                                                                "trg": None,
                                                                "masks": None})
                                        ["trg"][...,:1]), -1)[...,:-1,:]
    
    return result
    
train_iter = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_iter = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_iter = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
```
### （3）定义模型结构
```python
class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        assert hid_dim % n_heads == 0
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, mask):
                
        # layer normalization
        src = self.self_attn_layer_norm(src)
        src = self.ff_layer_norm(src)
                
        # multi-head attention
        _src, _ = self.self_attention(src, src, src, mask)
        
        # dropout
        src = self.dropout(_src)
                    
        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        # dropout
        src = self.dropout(_src)
        
        return src
    

class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()
        
        self.masked_multihead_attn = MultiHeadAttentionLayer(hid_dim,
                                                             n_heads,
                                                             dropout,
                                                             device)
        self.encoder_decoder_attn = MultiHeadAttentionLayer(hid_dim,
                                                              n_heads,
                                                              dropout,
                                                              device)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_dec_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        # masked multi-head attention
        _trg, _ = self.masked_multihead_attn(trg, trg, trg, trg_mask)
        
        # dropout
        trg = self.dropout(_trg)
        
        # residual connection
        trg = self.self_attn_layer_norm(trg + _trg)
        
        # encoder-decoder attention
        _trg, attention = self.encoder_decoder_attn(trg, enc_src, enc_src, src_mask)
        
        # dropout
        trg = self.dropout(_trg)
        
        # residual connection
        trg = self.enc_dec_attn_layer_norm(trg + _trg)
        
        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        # dropout
        trg = self.dropout(_trg)
        
        # residual connection
        trg = self.ff_layer_norm(trg + _trg)
        
        # update masked target sequence
        output_dict = {'output': trg,
                       'attention': attention}
        
        return output_dict
    
    
class TransformerModel(nn.Module):
    def __init__(self,
                 encoder_layers,
                 decoder_layers,
                 src_vocab_size,
                 trg_vocab_size,
                 pad_idx,
                 device):
        super().__init__()
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(HID_DIM, N_HEADS,
                                                         PF_DIM, DROPOUT, device)
                                             for _ in range(encoder_layers)])
        
        self.decoder_layers = nn.ModuleList([DecoderLayer(HID_DIM, N_HEADS,
                                                         PF_DIM, DROPOUT, device)
                                             for _ in range(decoder_layers)])
        
        self.src_embedding = nn.Embedding(src_vocab_size, HID_DIM)
        self.trg_embedding = nn.Embedding(trg_vocab_size, HID_DIM)
        
        self.positional_encoding = positional_encoding(MAX_LEN,
                                                          HID_DIM,
                                                          device)
        
        self.fc_out = nn.Linear(HID_DIM, trg_vocab_size)
        
        self.dropout = nn.Dropout(DROPOUT)
        
        self.scale = torch.sqrt(torch.FloatTensor([HID_DIM])).to(device)
        
        self.pad_idx = pad_idx
        self.device = device


    def make_src_mask(self, src):
        # src shape: (batch size, src sentence length)
        src_mask = (src!= self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # apply same mask to all timesteps per batch dimension
        src_mask = src_mask.expand(src_mask.shape[0], MAX_LEN, src_mask.shape[2])
        
        return src_mask
    

    def make_trg_mask(self, trg):
        # trg shape: (batch size, trg sentence length)
        trg_pad_mask = (trg!= self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        trg_len = trg.shape[1]
            
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len),
                                               dtype=torch.uint8,
                                               device=self.device)).bool()
        
        # unsqueeze and expand to match target dimensions (batch size, trg sent len, max input length)
        trg_pad_mask = trg_pad_mask.expand(trg_pad_mask.shape[0],
                                           trg_pad_mask.shape[1],
                                           src_mask.shape[1])
        
        # boolean combination of sub_mask and pad_mask
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return trg_mask

    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # embedding and adding positional encoding
        src_emb = self.dropout(self.src_embedding(src) * self.scale +
                               self.positional_encoding[:src.shape[1],:])
        trg_emb = self.dropout(self.trg_embedding(trg) * self.scale +
                               self.positional_encoding[:,:trg.shape[1]])
        
        # passing through encoder layers
        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_mask)
            
        
        # decoding
        for layer in self.decoder_layers:
            
            output_dict = layer(trg_emb, src_emb,
                                 trg_mask, src_mask)
                
            attention = output_dict['attention']
            
            # multiply attention weights by destination embeddings to 
            # focus on corresponding parts of the input sequence
            attention = attention @ self.trg_embedding.weight.transpose(0, 1)
            
            context = attention.sum(dim=-1)/attention.shape[-1]
            
            # concatenation of destination embeddings with attention based weights
            trg_emb = torch.cat((context[...,None,:], trg_emb), dim=-1)
            
            
        
        # output fully connected layer
        logits = self.fc_out(trg_emb)
        
        return logits

    
# set hyperparameters
ENC_LAYERS = 3
DEC_LAYERS = 3
HID_DIM = 512
N_HEADS = 8
PF_DIM = 2048
DROPOUT = 0.1

LEARNING_RATE = 0.0005
CLIP = 1
N_EPOCHS = 10

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create model object
model = TransformerModel(ENC_LAYERS, DEC_LAYERS,
                         SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, PAD_IDX, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1) 


# training loop
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = run_epoch('train', model, optimizer, criterion, train_iter)
    valid_loss = run_epoch('validation', model, criterion, valid_iter)
    
    end_time = time.time()
    
    scheduler.step(valid_loss)
    
    print(f"Epoch: {epoch+1}, Train loss: {train_loss:.3f}, Valid loss: {valid_loss:.3f}, Time: {(end_time - start_time)/60:.2f} minutes")
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './transformer_translation.pt')