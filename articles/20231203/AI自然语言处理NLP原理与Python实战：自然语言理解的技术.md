                 

# 1.背景介绍

自然语言处理（Natural Language Processing，NLP）是人工智能（AI）领域的一个重要分支，旨在让计算机理解、生成和处理人类语言。自然语言理解（Natural Language Understanding，NLU）是NLP的一个子领域，旨在让计算机理解人类语言的含义和意图。

在过去的几年里，NLP和NLU技术取得了显著的进展，这主要归功于深度学习和大规模数据的应用。这些技术已经被广泛应用于各种领域，如机器翻译、情感分析、文本摘要、语音识别、对话系统等。

本文将深入探讨NLP和NLU的核心概念、算法原理、具体操作步骤以及数学模型公式。同时，我们将通过具体的Python代码实例来解释这些概念和算法。最后，我们将讨论未来的发展趋势和挑战。

# 2.核心概念与联系

在NLP和NLU领域，有几个核心概念需要理解：

1.自然语言（Natural Language）：人类通常使用的语言，如英语、汉语、西班牙语等。
2.自然语言处理（NLP）：计算机对自然语言进行处理的技术，包括文本分类、情感分析、命名实体识别、语义角色标注等任务。
3.自然语言理解（NLU）：NLP的一个子领域，旨在让计算机理解人类语言的含义和意图，包括语义解析、意图识别、实体识别等任务。

NLP和NLU之间的联系如下：NLP是NLU的基础，NLU是NLP的一个子集。NLP涉及到更广的语言处理任务，而NLU则更关注语言的含义和意图。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解

在NLP和NLU领域，有几种核心算法和技术，包括：

1.词嵌入（Word Embedding）：将单词映射到一个高维的向量空间中，以捕捉词汇之间的语义关系。常用的词嵌入方法有Word2Vec、GloVe等。

2.循环神经网络（Recurrent Neural Network，RNN）：一种递归神经网络，可以处理序列数据，如文本。RNN的主要问题是长距离依赖性问题，即难以捕捉远离当前时间步的信息。

3.长短期记忆（Long Short-Term Memory，LSTM）：一种特殊的RNN，通过引入门机制来解决长距离依赖性问题，能够更好地捕捉远离当前时间步的信息。

4.注意力机制（Attention Mechanism）：一种用于关注输入序列中特定部分的技术，可以帮助模型更好地捕捉关键信息。

5.Transformer：一种基于自注意力机制的模型，能够更好地捕捉长距离依赖性，并在许多NLP任务上取得了State-of-the-art的性能。

6.BERT：一种预训练的Transformer模型，通过Masked Language Model和Next Sentence Prediction两个任务进行预训练，能够在多种NLP任务上取得出色的性能。

具体的算法原理和操作步骤以及数学模型公式详细讲解将在后续的部分中逐一介绍。

# 4.具体代码实例和详细解释说明

在本节中，我们将通过具体的Python代码实例来解释NLP和NLU的核心概念和算法。

## 4.1 词嵌入

### 4.1.1 使用GloVe训练词嵌入

```python
from gensim.models import Word2Vec

# 加载预训练的GloVe模型
model = Word2Vec.load("glove.6B.100d.txt")

# 查看词汇表
print(model.wv.vocab)

# 查看单词的词向量
print(model["king"])
```

### 4.1.2 使用Word2Vec训练词嵌入

```python
from gensim.models import Word2Vec

# 加载文本数据
sentences = [["hello", "world"], ["hello", "how", "are", "you"]]

# 训练Word2Vec模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 查看词汇表
print(model.wv.vocab)

# 查看单词的词向量
print(model["hello"])
```

### 4.1.3 使用FastText训练词嵌入

```python
from gensim.models import FastText

# 加载文本数据
sentences = [["hello", "world"], ["hello", "how", "are", "you"]]

# 训练FastText模型
model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 查看词汇表
print(model.wv.vocab)

# 查看单词的词向量
print(model["hello"])
```

## 4.2 RNN

### 4.2.1 使用PyTorch训练RNN模型

```python
import torch
import torch.nn as nn

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, 1, self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.out(out[:, -1, :])
        return out

# 加载文本数据
sentences = [["hello", "world"], ["hello", "how", "are", "you"]]

# 转换为PyTorch的tensor
x = torch.tensor(sentences)

# 初始化RNN模型
model = RNN(input_size=1, hidden_size=10, output_size=1)

# 训练RNN模型
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(100):
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, torch.tensor([1.0]))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
```

## 4.3 LSTM

### 4.3.1 使用PyTorch训练LSTM模型

```python
import torch
import torch.nn as nn

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, 1, self.hidden_size)
        c0 = torch.zeros(1, 1, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.out(out[:, -1, :])
        return out

# 加载文本数据
sentences = [["hello", "world"], ["hello", "how", "are", "you"]]

# 转换为PyTorch的tensor
x = torch.tensor(sentences)

# 初始化LSTM模型
model = LSTM(input_size=1, hidden_size=10, output_size=1)

# 训练LSTM模型
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(100):
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, torch.tensor([1.0]))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
```

## 4.4 Attention Mechanism

### 4.4.1 使用PyTorch实现Attention Mechanism

```python
import torch
import torch.nn as nn

# 定义Attention模型
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden):
        attn_weights = torch.softmax(self.linear(hidden), dim=1)
        return torch.bmm(attn_weights.unsqueeze(2), hidden.unsqueeze(1))

# 加载文本数据
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(text)
        tokens = [tokens[i] for i in range(len(tokens)) if len(tokens[i]) <= self.max_len]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<unk>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<pad>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "<s>"]
        tokens = [tokens[i] for i in range(len(tokens)) if tokens[i] != "</s>"]
        tokens = [tokens[i] for i in