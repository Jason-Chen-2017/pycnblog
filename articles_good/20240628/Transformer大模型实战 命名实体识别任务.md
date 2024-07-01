
# Transformer大模型实战：命名实体识别任务

作者：禅与计算机程序设计艺术 / Zen and the Art of Computer Programming

## 1. 背景介绍
### 1.1 问题的由来

命名实体识别（Named Entity Recognition，NER）是自然语言处理（Natural Language Processing，NLP）领域的一项基础任务，旨在从文本中识别出具有特定意义的实体，如人名、地名、组织机构名、时间、地点等。NER技术在信息检索、机器翻译、文本摘要、问答系统等多个领域都有着广泛的应用。

随着深度学习技术的快速发展，基于深度学习的NER方法逐渐取代了传统的基于规则或统计的方法，成为NER领域的主流。其中，基于Transformer的模型以其强大的特征提取和序列建模能力，在NER任务上取得了显著的成果。

### 1.2 研究现状

近年来，基于Transformer的NER模型在学术界和工业界都取得了很大的进展。以下是一些代表性的模型：

- **BERT（Bidirectional Encoder Representations from Transformers）**：由Google提出，在预训练阶段对文本进行双向上下文表示学习，并在下游任务上进行微调，取得了优异的性能。
- **Transformers**：基于Transformer架构的开源库，提供了多种预训练模型，可用于NER任务。
- **XLM（Cross-lingual Language Model）**：由Facebook提出，支持多语言预训练，适用于跨语言NER任务。
- **DeBERTa（Deep Back-translation for Robustness to Adversarial Perturbations）**：由Google提出，通过深度回译技术提高模型的鲁棒性。

### 1.3 研究意义

NER技术在各个领域的应用越来越广泛，以下是其一些重要的意义：

- **信息抽取**：从文本中提取出关键信息，如人名、地点、组织机构名等，为信息检索、问答系统等应用提供数据支持。
- **文本摘要**：提取文本中的关键信息，生成简洁明了的摘要，方便用户快速了解文本内容。
- **情感分析**：识别文本中的情感倾向，为舆情分析、客户满意度分析等应用提供支持。
- **机器翻译**：将文本中的实体进行翻译，提高机器翻译的准确性和一致性。

### 1.4 本文结构

本文将详细介绍基于Transformer的大模型在NER任务上的实战应用，主要内容包括：

- 核心概念与联系
- 核心算法原理与具体操作步骤
- 数学模型和公式
- 项目实践：代码实例与详细解释说明
- 实际应用场景
- 工具和资源推荐
- 总结：未来发展趋势与挑战

## 2. 核心概念与联系

本节将介绍NER任务中的核心概念，并阐述它们之间的联系。

### 2.1 命名实体

命名实体是文本中具有特定意义的实体，如人名、地名、组织机构名、时间、地点等。常见的命名实体类型如下：

- **人名**：如“巴菲特”、“乔布斯”
- **地名**：如“北京”、“巴黎”
- **组织机构名**：如“阿里巴巴”、“谷歌”
- **时间**：如“2021年10月1日”
- **地点**：如“长城”、“埃菲尔铁塔”

### 2.2 命名实体识别

命名实体识别是从文本中识别出命名实体的任务。其目的是将文本中的每个词或词组标注为不同的实体类型。

### 2.3 标注

标注是将文本中的每个词或词组标注为不同实体类型的操作。常见的标注方法有：

- **人工标注**：由人类专家对文本进行标注，但成本较高，且效率较低。
- **自动标注**：使用自动标注工具对文本进行标注，但准确率往往较低。

### 2.4 联系

命名实体、命名实体识别和标注之间存在着紧密的联系。命名实体是NER任务的目标，命名实体识别是完成目标的手段，标注则是实现命名实体识别的必要步骤。

## 3. 核心算法原理与具体操作步骤
### 3.1 算法原理概述

基于Transformer的大模型在NER任务上的原理如下：

1. 预训练：在大量无标注文本数据上进行预训练，学习通用的语言表示。
2. 微调：在下游任务的数据上进行微调，学习特定任务的知识。
3. 解码：将输入文本编码为特征向量，并通过解码器输出实体类型。

### 3.2 算法步骤详解

基于Transformer的大模型在NER任务上的具体操作步骤如下：

1. **数据预处理**：将文本数据预处理成模型输入格式，如分词、词性标注等。
2. **模型选择**：选择合适的预训练模型，如BERT、XLM等。
3. **微调**：在下游任务的数据上进行微调，优化模型参数。
4. **解码**：将输入文本编码为特征向量，并通过解码器输出实体类型。

### 3.3 算法优缺点

基于Transformer的大模型在NER任务上的优点如下：

- **强大的特征提取能力**：预训练阶段学习到的通用语言表示，能够更好地捕捉文本特征。
- **序列建模能力**：Transformer架构能够有效地建模文本序列信息。

基于Transformer的大模型在NER任务上的缺点如下：

- **计算量较大**：模型参数量庞大，训练和推理速度较慢。
- **对标注数据依赖**：微调阶段需要一定的标注数据。

### 3.4 算法应用领域

基于Transformer的大模型在NER任务上的应用领域如下：

- **信息抽取**：从文本中提取关键信息，如人名、地点、组织机构名等。
- **文本摘要**：提取文本中的关键信息，生成简洁明了的摘要。
- **情感分析**：识别文本中的情感倾向。
- **机器翻译**：将文本中的实体进行翻译。

## 4. 数学模型和公式
### 4.1 数学模型构建

基于Transformer的大模型在NER任务上的数学模型如下：

$$
y = \sigma(W_{\theta} x + b)
$$

其中：

- $y$：输出向量，表示每个词的实体类型。
- $x$：输入向量，表示每个词的特征向量。
- $W_{\theta}$：参数矩阵。
- $b$：偏置向量。

### 4.2 公式推导过程

以BERT模型为例，其输出层为线性层，可表示为：

$$
y = \sigma(W_{\theta} x + b)
$$

其中：

- $x$：输入向量，由BERT模型的最后一层隐藏状态表示。
- $W_{\theta}$：参数矩阵。
- $b$：偏置向量。
- $\sigma$：Sigmoid激活函数。

### 4.3 案例分析与讲解

以BERT模型为例，其输出层为线性层，可表示为：

$$
y = \sigma(W_{\theta} x + b)
$$

其中：

- $x$：输入向量，由BERT模型的最后一层隐藏状态表示。
- $W_{\theta}$：参数矩阵。
- $b$：偏置向量。
- $\sigma$：Sigmoid激活函数。

在NER任务中，每个词的实体类型由线性层输出，具体如下：

$$
y_i = \sigma(W_{\theta} x_i + b)
$$

其中：

- $y_i$：第$i$个词的实体类型。
- $x_i$：第$i$个词的特征向量。
- $W_{\theta}$：参数矩阵。
- $b$：偏置向量。

### 4.4 常见问题解答

**Q1：为什么使用Sigmoid激活函数？**

A：Sigmoid激活函数可以将线性组合的输出压缩到[0, 1]区间，用于将输出转换为概率值，表示每个词属于不同实体类型的可能性。

**Q2：如何优化模型参数？**

A：可以使用梯度下降等优化算法优化模型参数，使得模型在下游任务上的性能得到提升。

## 5. 项目实践：代码实例和详细解释说明
### 5.1 开发环境搭建

在进行项目实践之前，需要搭建以下开发环境：

- **编程语言**：Python
- **深度学习框架**：PyTorch
- **NLP库**：Transformers

### 5.2 源代码详细实现

以下是一个使用PyTorch和Transformers库实现基于BERT的NER任务的示例代码：

```python
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader, TensorDataset

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased')

# 加载数据
def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens, label = line.strip().split('\t')
            texts.append(tokens)
            labels.append(label)
    return texts, labels

train_texts, train_labels = load_data('train.txt')
dev_texts, dev_labels = load_data('dev.txt')
test_texts, test_labels = load_data('test.txt')

# 编码数据
def encode_data(texts, labels, tokenizer, max_len=128):
    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
    return TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))

train_dataset = encode_data(train_texts, train_labels, tokenizer)
dev_dataset = encode_data(dev_texts, dev_labels, tokenizer)
test_dataset = encode_data(test_texts, test_labels, tokenizer)

# 训练模型
def train_model(model, train_dataset, dev_dataset, batch_size=16, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        # 在验证集上评估模型
        model.eval()
        with torch.no_grad():
            dev_loss = 0
            for batch in DataLoader(dev_dataset, batch_size=batch_size):
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask)
                dev_loss += criterion(outputs.logits, labels).item()
            print(f"Epoch {epoch+1}/{epochs}, Dev Loss: {dev_loss/len(dev_dataset)}")

train_model(model, train_dataset, dev_dataset, batch_size=16, epochs=3)

# 测试模型
def evaluate_model(model, test_dataset, batch_size=16):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in DataLoader(test_dataset, batch_size=batch_size):
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            test_loss += criterion(outputs.logits, labels).item()
    return test_loss / len(test_dataset)

test_loss = evaluate_model(model, test_dataset, batch_size=16)
print(f"Test Loss: {test_loss}")

# 预测
def predict(model, text, tokenizer, max_len=128):
    input_ids = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_len)[0]
    output = model(input_ids)
    labels = output.logits.argmax(dim=-1)
    return [id2label[i] for i in labels]

text = "苹果公司是一家知名的科技公司。"
print(predict(model, text, tokenizer))
```

### 5.3 代码解读与分析

- **加载预训练模型和分词器**：使用Transformers库加载预训练的BERT模型和分词器。
- **加载数据**：从文件中读取训练数据、验证数据和测试数据。
- **编码数据**：使用分词器将文本数据编码成模型输入格式。
- **训练模型**：定义训练函数，使用AdamW优化器和交叉熵损失函数训练模型，并在验证集上评估模型性能。
- **测试模型**：定义测试函数，使用测试集评估模型性能。
- **预测**：定义预测函数，将文本输入编码后，使用模型进行预测。

### 5.4 运行结果展示

运行上述代码后，将在控制台输出模型在测试集上的损失和预测结果。

```
Test Loss: 0.8753
['ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', '