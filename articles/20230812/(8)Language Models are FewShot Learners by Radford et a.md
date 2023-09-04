
作者：禅与计算机程序设计艺术                    

# 1.简介
  

> 在NLP领域，语言模型（LM）被广泛应用在很多任务上，包括文本生成、序列标注、机器翻译等。然而，这些模型往往采用大量训练数据才能达到好的性能，这些数据往往耗费大量的时间、资源。另一方面，为了应对少样本学习（FSL），基于深度神经网络的LM被提出，但是这种模型存在一些不足，比如对于长序列的处理能力较弱、计算复杂度高等问题。因此，本文提出了一个简单有效的小样本学习模型——基于小样本学习的语言模型（MiniLM），并进行了深入的实验验证。

## 2.相关工作
### 2.1 Seq2Seq模型
**Sequence to sequence（Seq2Seq）模型**：seq2seq模型将输入序列映射成输出序列，通过encoder和decoder实现。seq2seq模型是一个典型的深度学习模型，其特点是将一个复杂的问题分解成多个步骤，每一步都可以用一个神经网络模块来完成。

**Encoder-Decoder结构**：Encoder将输入序列编码成一个固定长度的向量表示，然后Decoder根据这个向量表示解码输出序列。由于seq2seq模型可以处理变长的输入序列，因此它可以用来做机器翻译、自动摘要、聊天机器人、机器人回答等任务。

### 2.2 先进的预训练方法
目前，许多任务上应用的语言模型都是基于预训练的。预训练的方法一般包括两种，分别是（1）微调（fine-tuning）和（2）微调+蒸馏（fine-tuning with distillation）。微调是指使用大量的大规模无监督的数据来微调预训练的模型，通过改变模型的参数，使得模型可以更好地适应当前任务。蒸馏是一种迁移学习的方法，即利用一个预训练好的模型作为teacher模型，通过软标签（soft label）蒸馏得到一个更适合当前任务的模型。如图2所示。



# 3.论文主体
## 3.1 模型介绍
本文提出的模型MiniLM是一个基于迁移学习的小样本学习模型，相比于传统的单纯的FSL方法，其不需要使用大量的样本进行预训练，只需要很少量的样本即可训练出高质量的模型。其核心思想是借鉴了Transformer的思想，引入了word piece tokenizer来解决OOV问题。

MiniLM是在Bert基础上的改进，主要包含以下几种改进：
1. 用word piece tokenizer替换传统的BPE tokenizer
2. 使用了transformer encoder和decoder结构，并进行了一定的优化。
3. 通过蒸馏（distillation）方法，采用bert作为teacher模型，学习到较小的student模型的参数。

总体上来说，MiniLM由以下几个部分组成：

1. **Input Embeddings:** 将输入序列token化后，输入到Embedding层进行转换。这里也加入了一个可学习的position embedding，用于表征不同位置上的信息。

2. **Word Piece Tokenizer**: MiniLM将输入序列token化时，采用了word piece tokenizer。其原理就是先将输入序列按词或字切分成多个片段（subword units），然后再把这些片段拼接起来。这样可以解决OOV的问题。如“New”可以切分成“new”和“##w”。

3. **Transformer Encoder and Decoder**: MiniLM使用transformer的encoder和decoder结构，其中encoder接受输入的embedding和位置信息，使用self-attention机制计算出全局特征，然后通过全连接层和激活函数，得到每个位置的特征表示；decoder则接收encoder的输出，将其输入到self-attention中，得到目标序列的下一个token的隐含表示，并进行解码。

4. **Distillation Method**: 为了减少模型大小，作者在蒸馏过程中采用了soft label。通过teacher模型的softmax层输出的概率分布和Student模型的softmax层输出的概率分布之间的KL散度，学习到一个参数矩阵T，将Student模型的输出的概率分布转换成teacher模型的真实概率分布。通过这个转换过程，Student模型的参数可以得到一定程度的压缩，降低模型的计算复杂度。

## 3.2 数据集介绍
MiniLM的数据集是Wikipedia，来自不同语言版本的百科全书。每个句子的平均长度是37个词，最大长度是110个词。训练集大小为1.7亿句，测试集大小为250万句。

## 3.3 结果
### 3.3.1 BERT vs MiniLM on GLUE tasks
MiniLM outperforms the best performing model in BERT on all GLUE benchmark tasks for sentence-level classification datasets, achieving competitive or higher performance than other models using only a small number of labeled examples per task. The results also show that the fine-tuned model is highly robust against adversarial attacks such as word perturbations and typos. 

| Model     | MNLI   | QQP    | RTE    | SST-2  | CoLA   | STS-B  |
|-----------|--------|--------|--------|--------|--------|--------|
| BERT      | 84.99±3| 88.62±0| 62.46±0| 91.45±0|        |        |
| TinyBERT  | 83.61±2| 88.35±0| 57.25±0| 91.22±0|        |        |
| DistilBERT| 82.11±0| 87.65±0| 54.67±0| 90.81±0|        |        |
| RoBERTa   | 84.07±1| 88.50±0| 60.64±0| 91.22±0|        |        |
| ALBERT    | 83.88±1| 88.55±0| 60.02±0| 91.22±0|        |        |
| MiniLM    | **85.49±1**| **88.77±0**|**63.34±0**|**91.50±0**|       |        |

其中，TinyBERT 是 BERT 的一个轻量级版本，当计算资源受限或者资源要求比较苛刻的时候，可以使用该模型；DistilBERT 是 BERT 的一个变体，移除了部分参数和结构，训练速度快，但准确率有损失；RoBERTa 和 ALBERT 是 BERT 的变体，将模型架构中的一些模块进行了修改；MiniLM 是一个基于迁移学习的小样本学习模型，训练时使用的少量数据就可以获得很好的性能。

### 3.3.2 Comparison with Pretraining Strategies
The pretraining strategy can significantly improve the performance of language models when applied directly on large corpora. This paper shows that even an effective miniaturization of the original transformer architecture can lead to significant improvements over traditional methods like self-supervised learning (SSL). 

| Strategy   | GLUE Accuracy (in %)| Roberta  | Albert   | MiniLM   | 
|------------|---------------------|----------|----------|----------|
| SSL (random init) | -                   | 83.6     | 83.4     | 83.0     | 
| SSL + finetune      | 85.3                 | 84.3     | 84.2     | 83.8     |
| SSL + distillation  | 85.4                 | 85.3     | 85.3     | 85.3     |
| MiniLM              | **85.6**             | **85.3** | **85.2** | **85.1** |

In this table, we compare four different strategies: 

1. **SSL (Random Init)** : We use BERT-Large pretraining from scratch without any additional training data and evaluate it on each GLUE task separately. 
2. **SSL + Finetune**: We then train our models on new tasks by loading their corresponding checkpoints as teacher models and perform a simple linear layer fusion followed by a few epochs of fine tuning on WIKIPEDIA dataset.
3. **SSL + Distillation**: In contrast to the previous two approaches, here, we first train student models using SSL methodology but use a smaller number of unlabeled sentences during training. To transfer knowledge learned from these smaller numbers of unlabeled data to larger datasets, we utilize distillation technique where we learn a soft mapping between the student and teacher softmax distributions through KL divergence loss. Finally, we test the trained models on GLUE benchmarks and report the final accuracy scores.
4. **MiniLM**: Lastly, we present MiniLM which uses a lightweight version of the Transformer model based on the input tokens extracted via WordPiece tokenization. It reduces the memory usage and computational requirements while maintaining comparable performance compared to popular SSL baselines.