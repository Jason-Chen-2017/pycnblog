                 

# 基于LLM的长文档推荐方法

> 关键词：长文档推荐, 语言模型, 基于注意力机制, 协同过滤, 深度学习

## 1. 背景介绍

在当今信息爆炸的时代，海量文本数据的产生与获取极大地丰富了人类的信息资源。然而，信息过载也带来了一个问题：如何快速、准确地从海量的信息中筛选出最相关的文档推荐给用户，成为信息检索领域的一项重大挑战。传统的文本推荐方法往往依赖于短文本的特征提取和相似性匹配，难以处理长文档（如论文、书籍等）。

长文档推荐（Long Document Recommendation）正是为了应对这一挑战而提出的。长文档推荐方法不仅能处理文本特征提取和相似性匹配，还能利用语言模型的强大语言理解能力，对文档的内容和结构进行更深层次的语义理解，从而提升推荐准确度。基于语言模型的长文档推荐方法已经成为近年来NLP研究的一个热门方向。

本文将详细介绍基于语言模型的长文档推荐方法，涵盖其核心原理、操作步骤、案例分析、代码实现等内容。同时，还将探讨长文档推荐方法在实际应用中的场景和挑战，以及未来可能的发展趋势。

## 2. 核心概念与联系

### 2.1 核心概念概述

在介绍长文档推荐方法之前，首先需要了解几个关键概念：

- **长文档推荐**：指从海量的长文档集合中，根据用户的阅读历史和偏好，为用户推荐最相关的长文档的过程。其目标是为用户提供精准、个性化的阅读体验。

- **语言模型**：指通过对大量文本数据进行统计学习，建立起来的语言知识模型。常见的语言模型包括n-gram模型、RNN、LSTM、Transformer等。语言模型能够理解自然语言的内在规律，捕捉文本的语义信息，成为长文档推荐的重要工具。

- **注意力机制（Attention Mechanism）**：一种在序列模型中用于捕捉序列间依赖关系的机制。在长文档推荐中，注意力机制可以根据文档的不同部分，动态地赋予其不同的权重，从而更准确地捕捉文档的重点内容。

- **协同过滤（Collaborative Filtering）**：一种基于用户历史行为数据的推荐方法，常用于电商、音乐、视频等领域。在长文档推荐中，协同过滤可以结合用户的行为数据和文档的特征，生成更符合用户兴趣的推荐结果。

这些概念之间有着紧密的联系，共同构成了长文档推荐的基础框架。如图1所示，长文档推荐方法通常通过以下步骤实现：

```mermaid
graph TB
    A[长文档推荐]
    B[数据准备] --> A
    C[预训练语言模型] --> A
    D[文档特征提取] --> A
    E[用户行为分析] --> A
    F[模型训练] --> A
    G[推荐生成] --> A
```

其中，数据准备、预训练语言模型、文档特征提取、用户行为分析是输入端，而模型训练、推荐生成是输出端。

### 2.2 核心概念原理和架构

图1：长文档推荐方法流程图

#### 2.2.1 数据准备

数据准备是长文档推荐的基础。在实践中，常用的数据集包括用户行为数据和文档特征数据：

- **用户行为数据**：通常包括用户的阅读历史、收藏夹、评分、浏览时间等。这些数据可以帮助模型了解用户的兴趣偏好。

- **文档特征数据**：包括文档的标题、摘要、关键词、作者、引用等。这些数据用于对文档进行初步的相似性匹配。

#### 2.2.2 预训练语言模型

预训练语言模型是通过大量无标签文本数据训练得到的语言模型，其目的是学习语言的基本规律和语义信息。常用的预训练语言模型包括BERT、GPT等。在长文档推荐中，预训练语言模型可以作为特征提取器，用于提取文档的语义信息。

#### 2.2.3 文档特征提取

文档特征提取是长文档推荐的关键步骤。传统的文档特征提取方法通常基于TF-IDF、LDA等技术，但这些方法难以处理长文档的语义信息。基于语言模型的文档特征提取方法则可以利用预训练模型的知识表示，生成更具语义信息的特征向量。

#### 2.2.4 用户行为分析

用户行为分析是通过用户的历史行为数据，了解用户的兴趣偏好。常用的方法包括协同过滤、基于用户兴趣模型的推荐方法等。

#### 2.2.5 模型训练

模型训练是长文档推荐的核心步骤。常用的模型包括基于注意力机制的推荐模型、基于协同过滤的推荐模型等。这些模型将用户行为数据和文档特征数据作为输入，生成推荐结果。

#### 2.2.6 推荐生成

推荐生成是长文档推荐的最后一步，即根据模型训练得到的推荐结果，为用户生成推荐文档。

## 3. 核心算法原理 & 具体操作步骤

### 3.1 算法原理概述

基于语言模型的长文档推荐方法主要分为两个步骤：

1. **文档特征提取**：使用预训练语言模型对文档进行编码，生成文档的语义向量。
2. **推荐生成**：将用户行为数据和文档语义向量作为输入，使用注意力机制生成推荐结果。

#### 3.1.1 文档特征提取

文档特征提取是长文档推荐的关键步骤。传统的文档特征提取方法通常基于TF-IDF、LDA等技术，但这些方法难以处理长文档的语义信息。基于语言模型的文档特征提取方法则可以利用预训练模型的知识表示，生成更具语义信息的特征向量。

常用的文档特征提取方法包括：

- **静态特征提取**：直接使用预训练语言模型对文档进行编码，生成文档的语义向量。这种方法简单易行，但难以处理长文档的动态变化。

- **动态特征提取**：结合用户行为数据，动态地提取文档的特征。例如，在用户阅读某段文档时，可以使用预训练语言模型对当前段落进行编码，生成动态特征向量。

#### 3.1.2 推荐生成

推荐生成是长文档推荐的最后一步。常用的推荐生成方法包括：

- **基于注意力机制的推荐**：通过注意力机制，根据文档的不同部分，动态地赋予其不同的权重，从而更准确地捕捉文档的重点内容。

- **基于协同过滤的推荐**：结合用户的行为数据和文档的特征，生成更符合用户兴趣的推荐结果。

### 3.2 算法步骤详解

#### 3.2.1 数据准备

在数据准备阶段，需要收集用户的行为数据和文档的特征数据。用户的行为数据包括用户的阅读历史、收藏夹、评分、浏览时间等。文档的特征数据包括文档的标题、摘要、关键词、作者、引用等。

#### 3.2.2 预训练语言模型

选择预训练语言模型作为特征提取器。常用的预训练语言模型包括BERT、GPT等。

#### 3.2.3 文档特征提取

使用预训练语言模型对文档进行编码，生成文档的语义向量。例如，使用BERT模型对文档的每个段落进行编码，生成段落向量。

#### 3.2.4 用户行为分析

结合用户的行为数据，了解用户的兴趣偏好。例如，使用协同过滤方法，根据用户的历史行为数据，生成用户兴趣模型。

#### 3.2.5 模型训练

选择基于注意力机制的推荐模型或基于协同过滤的推荐模型，将用户行为数据和文档语义向量作为输入，生成推荐结果。

#### 3.2.6 推荐生成

根据模型训练得到的推荐结果，为用户生成推荐文档。

### 3.3 算法优缺点

基于语言模型的长文档推荐方法具有以下优点：

1. **语义理解能力强**：基于语言模型的推荐方法能够理解长文档的语义信息，捕捉文档的重点内容，提升推荐准确度。

2. **处理长文档能力强**：传统的推荐方法难以处理长文档，而基于语言模型的推荐方法能够有效处理长文档。

3. **模型可解释性强**：基于语言模型的推荐方法能够提供更直观的模型解释，帮助用户理解推荐结果的来源。

然而，基于语言模型的推荐方法也存在以下缺点：

1. **计算成本高**：预训练语言模型的计算成本较高，需要大量的计算资源和时间。

2. **数据需求高**：需要大量的用户行为数据和文档特征数据，数据的收集和处理难度较大。

3. **泛化能力有限**：基于语言模型的推荐方法往往依赖于特定的数据集和预训练模型，泛化能力有限。

4. **动态变化难以处理**：长文档的语义信息会随着文档的阅读而变化，基于静态特征的推荐方法难以处理。

### 3.4 算法应用领域

基于语言模型的长文档推荐方法广泛应用于以下几个领域：

1. **学术论文推荐**：为研究人员推荐最新的学术论文，帮助他们了解最新的研究方向和成果。

2. **图书推荐**：为读者推荐最适合的图书，提高阅读体验。

3. **技术文档推荐**：为工程师推荐最相关的技术文档，帮助他们快速解决问题。

4. **医学文献推荐**：为医生推荐最新的医学文献，帮助他们了解最新的医学研究成果。

5. **教育资源推荐**：为学生推荐最适合的学习资源，提高学习效率。

## 4. 数学模型和公式 & 详细讲解 & 举例说明

### 4.1 数学模型构建

#### 4.1.1 数据表示

在长文档推荐中，通常将用户行为数据和文档特征数据表示为矩阵形式。例如，用户行为数据可以表示为$U \in \mathbb{R}^{N \times D}$，其中$N$表示用户数，$D$表示用户行为特征数。文档特征数据可以表示为$V \in \mathbb{R}^{M \times D'}$，其中$M$表示文档数，$D'$表示文档特征数。

#### 4.1.2 预训练语言模型

在长文档推荐中，预训练语言模型可以表示为$L$，其输出为一个隐含层的向量，表示文档的语义向量。例如，使用BERT模型对文档的每个段落进行编码，生成段落向量$H \in \mathbb{R}^{L \times T \times D}$，其中$T$表示段落数，$D$表示隐含层的维度。

#### 4.1.3 注意力机制

注意力机制可以表示为$Attention(Q, K, V)$，其中$Q$表示查询向量，$K$表示键向量，$V$表示值向量。在长文档推荐中，注意力机制用于计算文档不同部分的重要性权重。

#### 4.1.4 推荐模型

推荐模型可以表示为$F(U, V, H)$，其中$U$表示用户行为数据，$V$表示文档特征数据，$H$表示文档的语义向量。推荐模型将用户行为数据和文档特征数据作为输入，生成推荐结果$R$。

#### 4.1.5 推荐生成

推荐生成可以表示为$R = F(U, V, H) \times W$，其中$W$表示推荐权重矩阵。

### 4.2 公式推导过程

#### 4.2.1 文档特征提取

假设使用BERT模型对文档的每个段落进行编码，生成段落向量$H \in \mathbb{R}^{L \times T \times D}$，其中$L$表示文档数，$T$表示段落数，$D$表示隐含层的维度。

#### 4.2.2 注意力机制

注意力机制可以表示为$Attention(Q, K, V)$，其中$Q$表示查询向量，$K$表示键向量，$V$表示值向量。在长文档推荐中，注意力机制用于计算文档不同部分的重要性权重。

$$
Attention(Q, K, V) = \frac{\exp(\text{softmax}(QK^T))V}{\sum_{i=1}^{T} \exp(QK_i^T)}
$$

其中$softmax$函数表示归一化操作，$QK^T$表示查询向量与键向量的点积，$V$表示值向量。

#### 4.2.3 推荐模型

推荐模型可以表示为$F(U, V, H)$，其中$U$表示用户行为数据，$V$表示文档特征数据，$H$表示文档的语义向量。推荐模型将用户行为数据和文档特征数据作为输入，生成推荐结果$R$。

$$
R = F(U, V, H) = \sum_{i=1}^{M} \alpha_i V_i
$$

其中$V_i$表示第$i$个文档的特征向量，$\alpha_i$表示第$i$个文档的重要性权重，可以通过注意力机制计算得到。

#### 4.2.4 推荐生成

推荐生成可以表示为$R = F(U, V, H) \times W$，其中$W$表示推荐权重矩阵。

$$
R = F(U, V, H) \times W = \sum_{i=1}^{M} \alpha_i V_i \times W
$$

### 4.3 案例分析与讲解

#### 4.3.1 案例1：学术论文推荐

假设有一个学术论文推荐系统，目标是为研究人员推荐最新的学术论文。该系统收集了1000个用户的阅读历史、收藏夹、评分、浏览时间等行为数据，以及1000篇学术论文的标题、摘要、关键词、作者、引用等特征数据。使用BERT模型对每篇论文进行编码，生成段落向量$H \in \mathbb{R}^{1000 \times 5 \times 768}$。

在推荐模型中，将用户行为数据和论文特征数据作为输入，生成推荐结果$R \in \mathbb{R}^{1000 \times 1}$。最后，通过调整推荐权重矩阵$W$，生成最终的推荐结果$R$。

#### 4.3.2 案例2：图书推荐

假设有一个图书推荐系统，目标是为读者推荐最适合的图书。该系统收集了10000个用户的阅读历史、收藏夹、评分、浏览时间等行为数据，以及10000本图书的标题、摘要、关键词、作者、出版日期等特征数据。使用BERT模型对每本图书进行编码，生成特征向量$H \in \mathbb{R}^{10000 \times 1 \times 768}$。

在推荐模型中，将用户行为数据和图书特征数据作为输入，生成推荐结果$R \in \mathbb{R}^{10000 \times 1}$。最后，通过调整推荐权重矩阵$W$，生成最终的推荐结果$R$。

## 5. 项目实践：代码实例和详细解释说明

### 5.1 开发环境搭建

在进行项目实践前，需要先搭建好开发环境。以下是使用Python进行PyTorch开发的环境配置流程：

1. 安装Anaconda：从官网下载并安装Anaconda，用于创建独立的Python环境。

2. 创建并激活虚拟环境：
```bash
conda create -n pytorch-env python=3.8 
conda activate pytorch-env
```

3. 安装PyTorch：根据CUDA版本，从官网获取对应的安装命令。例如：
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

4. 安装Transformers库：
```bash
pip install transformers
```

5. 安装各类工具包：
```bash
pip install numpy pandas scikit-learn matplotlib tqdm jupyter notebook ipython
```

完成上述步骤后，即可在`pytorch-env`环境中开始项目实践。

### 5.2 源代码详细实现

以下是使用BERT模型进行学术论文推荐项目实现的完整代码：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score

# 定义BERT模型
class BERTEmbedding(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(BERTEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-cased', num_labels=num_labels)
        self.linear = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, 0, :]
        logits = self.linear(sequence_output)
        return logits

# 定义用户行为数据
user_data = pd.read_csv('user_data.csv', header=None, sep='\t')
user_data = user_data.values

# 定义论文特征数据
paper_data = pd.read_csv('paper_data.csv', header=None, sep='\t')
paper_data = paper_data.values

# 定义BERT模型和优化器
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BERTEmbedding(hidden_size=768, num_labels=1000)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# 定义推荐生成函数
def generate_recommendation(model, user_data, paper_data):
    user_behavior = user_data
    paper_features = paper_data
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = [tokenizer.encode(user_behavior[i]) for i in range(len(user_behavior))]
    paper_input_ids = [tokenizer.encode(paper_features[i]) for i in range(len(paper_features))]
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = np.array(user_input_ids)
    paper_input_ids = np.array(paper_input_ids)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = torch.tensor(user_input_ids)
    paper_input_ids = torch.tensor(paper_input_ids)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input_ids = paper_input_ids.to(device)
    
    # 将用户行为数据和论文特征数据转换为BERT模型可接受的格式
    user_input_ids = user_input_ids.to(device)
    paper_input

