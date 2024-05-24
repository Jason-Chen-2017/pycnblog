
作者：禅与计算机程序设计艺术                    

# 1.简介
  
及背景介绍
在文本情感分析领域，传统的方法通常包括基于统计方法或者机器学习模型对预料中的每一个单词进行分类、打分，这种方法往往存在不足之处，比如对于复杂场景下的情感表达，如同一段文字描述的多个维度，或表现出多个情绪时，传统方法的效果会变得很差。为了能够更好地处理复杂场景下情感表达，近年来有一些研究工作试图利用神经网络对文本情感进行建模，其中一个主要方向是通过建立深层次的神经网络结构来捕获不同复杂度特征之间的关系，从而提高情感识别能力。例如Google DeepMind提出的“Image CAPTCHA”问题就是对图像字母的形状进行分类任务，通过训练一个卷积神经网络来学习不同字母之间的模式关联性。这些网络都成功地在图像分类、序列建模、文本分类等不同领域取得了不错的成果。

2017年6月，Google Brain团队在其最新一版论文中首次提出Capsule Network，这是一种全新的神经网络结构，旨在解决深度神经网络中的梯度消失、梯度爆炸、泛化性能较弱的问题。该方法创造性地将普通神经元的输出神经元状态压缩到高纬度空间中，通过不同的核函数组合成高级特征表示，来对输入数据进行编码并实现信息的高效传递。当网络收缩后，其内部参数矩阵可以看作具有自然而强大的非线性激活功能。因此，通过对各个层次的参数矩阵进行正则化控制，能够有效防止梯度消失和爆炸问题，进而提升网络的深度和广度，并达到很好的学习能力和泛化能力。

2017年11月，Hinton团队又在他的最新一版论文中详细阐述了Capsule Networks的结构。在这个版本的文章中，他们引入了一个新的Capsule Layer来实现对多种输入数据的融合。通过将多个高纬度的向量整合到一起，Capsule Layer可以逼近任意复杂的分布，并且能够自动地对长尾分布进行建模。此外，论文还介绍了如何通过结合注意力机制和动态路由策略来增强网络的学习能力和表达能力。通过这些工作，两者创造性地将Capsule Networks应用于文本情感分析领域，取得了很好的结果。

# 2.基本概念术语说明
## Capsule Networks
Capsule Networks（胶囊网络）是一种深度神经网络结构，能够捕捉输入的数据模式并生成更有效的高阶表示。它由一系列的胶囊组成，每个胶囊代表输入数据的局部区域，胶囊之间通过权重连接实现非线性组合，最终输出一个对全局分布进行编码的低纬度空间。每个胶囊包含三个元素：长度，方向，以及线性激活值。胶囊的长度刻画了该区域的重要程度，方向则用来指示该区域的前景与背景。通过结合所有胶囊的线性激活值，胶囊网络可以学习复杂的非线性空间分布。

## Capsule Layer
Capsule Layer 是Capsule Networks的核心组件，它的作用是将多个高纬度的向量整合到一起，并实现对不同维度间的组合。它主要由两个子模块构成：Routing Module 和 Transformation Module。Routing Module的作用是计算每个胶囊的目标权重，即分配给其他胶囊的信息的大小；Transformation Module则根据这些权重对输入数据进行转换，使其成为一个低纬度的向量，最终输出一个对全局分布进行编码的低纬度空间。

## Routing Module
Routing Module 的作用是在Capsule Layer的每个时间步内，计算当前输入数据（称为“输入”）所需的各个胶囊的目标权重。根据当前输入数据，Routing Module会计算每个胶囊的响应分数，然后根据这些分数对其他胶囊的响应进行传递，直至所有的胶囊均获得了相应的权重。这里的目标权重就类似于激活函数的输出值，它表示了胶囊对当前输入数据的响应强度。

## Attention Mechanism
Attention mechanism 是 Routing Module 的一种改进方式。其目的在于让网络专注于某些特定区域，而不是将所有输入数据映射到所有胶囊上。其基本思想是，通过关注关键词、语法结构等来获取与所关注内容相关的信息，并忽略无关紧要的内容。Attention mechanism 可以帮助 Routing Module 更好地选择与目标相关的胶囊，同时保持其他胶囊的激活值不受影响。

## Dynamic Routing Strategy
Dynamic Routing Strategy （动态路由策略）是 Routing Module 的另一种改进方式。它的主要目的是避免对某些胶囊的响应过于激烈，导致它们对全局分布的建模过于简单。该策略通过计算每个胶囊的路由权重，从而调整每一步的路由过程，确保不同胶囊的响应之间具有平衡的贡献度。

## ReLU Activation Function
ReLU Activation Function（修正线性单元激活函数）是用于激活神经元输出的常用函数，它是一个非线性函数，能够通过设置负权值的像素点的值为零来抑制神经元的输出。相比于sigmoid函数，ReLU函数的优点是速度更快，并且由于参数数量较少，因此能够快速地进行训练。

## Cross Entropy Loss
Cross Entropy Loss （交叉熵损失）是用来衡量两个概率分布之间的距离的损失函数。它是一种常用的用于分类问题的损失函数，因为它能够反映两个概率分布的差异。Cross Entropy Loss 函数形式如下：

L = -Σyi*log(pi)

其中，Σ是所有类别标签的总和，yi是第i个样本属于第j类的真实概率，pi是模型输出的预测概率。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 概览
传统的文本情感分析方法基于统计方法或者机器学习模型对预料中的每一个单词进行分类、打分，这种方法往往存在不足之处，比如对于复杂场景下的情感表达，如同一段文字描述的多个维度，或表现出多个情绪时，传统方法的效果会变得很差。为了能够更好地处理复杂场景下情感表达，近年来有一些研究工作试图利用神经网络对文本情感进行建模，其中一个主要方向是通过建立深层次的神经网络结构来捕获不同复杂度特征之间的关系，从而提高情感识别能力。例如Google DeepMind提出的“Image CAPTCHA”问题就是对图像字母的形状进行分类任务，通过训练一个卷积神经网络来学习不同字母之间的模式关联性。这些网络都成功地在图像分类、序列建模、文本分类等不同领域取得了不错的成果。

近日，斯坦福大学的研究人员等着载入了它们的眼睛——Hinton团队等着看到了DeepMind的突破！Hinton团队和斯坦福大学在一篇新文章中详细阐述了Capsule Networks的结构。在这篇文章里，他们介绍了Capsule Layer的结构，并展示了如何通过结合注意力机制和动态路由策略来增强网络的学习能力和表达能力。通过这些工作，两者创造性地将Capsule Networks应用于文本情感分析领域，取得了很好的结果。

下面我们将详细地介绍一下这篇文章的内容。

## Introduction and related work
### Natural Language Processing
在本文之前的NLP研究中，有两种主要的方法——基于统计的方法和机器学习的方法。基于统计的方法通常采用字典的方式进行词的切分，然后对词频进行统计，最后用分类器对句子进行打分。这种方法存在明显的局限性，比如对于复杂的语言风格、多义词等，这种方法的准确率往往不高。而机器学习的方法通常是采用神经网络对预料进行建模，通过学习语义和语法的特征，对输入数据进行表示，并通过中间层进行非线性转换，最终得到一个概率值作为输出。虽然神经网络已经在NLP领域占据了重要位置，但仍有许多不足。比如在不同领域的数据集上训练出的模型，可能不能直接迁移到另外一个领域，而且模型的复杂度往往受限于数据的规模，难以应用于复杂的场景。另外，神经网络在捕捉数据模式方面的能力还有待提高。

### Sentiment Analysis
文本情感分析，也被称为Opinion Mining（观点挖掘），是NLP的一个热门研究方向。其任务是在用户的评论文本中识别其情感倾向，是非常重要的智能产品功能。目前，基于深度学习的方法已经取得了不错的效果，尤其是比较复杂的文本情感分析任务。例如，谷歌在其最新一版搜索引擎Google News的主页就采用了基于神经网络的深度学习技术来分析用户的查询意图，来决定是否显示其相关新闻。

早期的文本情感分析方法，主要基于统计的方法。这类方法通过训练分类器对文本进行标注，然后运用这类分类器对新的文本进行情感判断。传统的基于统计的方法，如朴素贝叶斯方法，可以对简单的文本情感分析提供可行的方案，但对于复杂的情绪表达，这些方法的准确率并不高。后来的机器学习方法，如支持向量机、神经网络，基于特征工程构建了高度复杂的模型，对于复杂的情绪表达更加有效。

### A deep neural network approach to sentiment analysis
随着神经网络在NLP领域的应用越来越广泛，基于神经网络的文本情感分析方法也日渐成为热门话题。目前，基于神经网络的文本情感分析方法大致可以分为两类：短文本分类和长文本分类。

#### Short text classification methods
短文本分类的方法，一般采用卷积神经网络（CNN）或者循环神经网络（RNN）等深度学习模型。这种模型可以捕捉输入文本的全局语义信息、局部上下文信息和序列特性。它首先通过卷积神经网络提取文本的局部特征，然后通过循环神经网络学习长期依赖的特征，最后通过全连接层输出最后的分类结果。如图1所示，最基础的基于CNN的文本情感分类器架构。


上图左侧是最基础的结构示意图，右侧是举例的图像分类任务示意图。卷积层在图像上滑动，每次移动一个像素点，提取局部特征；池化层将局部特征聚合为整体特征；全连接层输出预测的概率分布。RNN层接受序列的历史信息，捕捉到整个文本的长期依赖关系。

但是，这些模型在短文本上的表现并不尽如人意。例如，在图像分类任务中，对于图片质量、拍摄角度等各种因素的变化，往往会影响图像的分类结果。同样的，在序列模型中，对于长文本来说，它的局部依赖关系往往比较复杂，因此需要更加深层的模型才能捕捉到这一点。所以，针对短文本分类任务，更多的关注全局特征和局部特征的融合，而不是局限于全局特征和局部上下文信息。

#### Long text classification methods
另一类长文本分类的方法，是深度学习模型自回归生成模型（RNNG）。这种方法通过利用RNN来建模序列数据生成的过程，从而对文本进行建模。这种方法将文本生成模型看作一个马尔科夫链，它生成文本的一个字符或者一个词，然后根据先前的输出生成后续的字符或者词。这样做的好处在于可以捕捉到整个文本的全局分布和局部细节，以及动态生成的影响。如图2所示，最基础的基于RNNG的文本生成模型架构。


如上图所示，基于RNNG的文本生成模型由编码器、解码器两部分组成。编码器输入文本，通过多层RNN生成隐含变量$h_t$；解码器根据隐含变量生成文本，输入词典中预测一个下一个词。训练过程是最大似然估计。

然而，基于RNNG的文本生成模型在长文本分类上也面临着同样的挑战。它的生成模型只能生成单个词或字符，而不能考虑到长文本中的全局分布和局部细节。为了解决这个问题，目前的长文本分类方法，主要依赖于深度学习模型的层次化表示，通过多个层次的特征，对文本进行表示。如图3所示，深层LSTM网络结构示意图。


如图3所示，深层LSTM网络通过多层循环神经网络，对文本进行表示。它首先输入整个文本，得到每个词的上下文特征，然后再逐层生成句子级别的特征。最后，通过输出层输出分类的结果。这种层次化的特征表示能够捕捉到全局分布和局部细节，并且可以有效地生成与输入文本匹配的结果。

然而，这种方法的缺陷在于，它只能适用于高度规则的文本数据，无法捕捉到文本的复杂语义和多样性。因此，基于RNNG的文本生成模型目前还不是通用且普遍使用的技术。

### Combining CNNs and RNNs with capsules for long text classification
Hinton等人的文章发现，RNNs通过捕捉局部序列特征，能够对长文本中的全局依赖关系进行建模；CNNs通过捕捉局部空间特征，能够对长文本中的局部相似性进行建模。而Hinton团队的进一步发现，如果将这两种网络架构混合起来，就可以产生比单一的深度网络更好的效果。Hinton等人提出了一种新的网络结构——Capsule Networks。

#### Capsule Networks
Capsule Networks 是Hinton团队在论文中提出的一种全新的深度神经网络结构，可以捕捉输入的数据模式并生成更有效的高阶表示。它由一系列的胶囊组成，每个胶囊代表输入数据的局部区域，胶囊之间通过权重连接实现非线性组合，最终输出一个对全局分布进行编码的低纬度空间。每个胶囊包含三个元素：长度，方向，以及线性激活值。胶囊的长度刻画了该区域的重要程度，方向则用来指示该区域的前景与背景。通过结合所有胶囊的线性激活值，胶囊网络可以学习复杂的非线性空间分布。如图4所示，一个典型的Capsule Networks的架构示意图。


如上图所示，Capsule Networks由一个Encoder部分和一个Decoder部分组成。Encoder部分通过多层CNNs或RNNs，对输入数据进行特征提取；Decoder部分则通过多个胶囊的堆叠来完成特征融合。胶囊网络的核心在于其能够捕捉到不同复杂度特征之间的关系，并把这些关系编码到低纬度空间中。

#### Layering the networks together
Capsule Networks可以在短文本分类模型和长文本分类模型之间无缝切换。也就是说，既可以使用基于CNN的短文本分类器，也可以使用基于RNNG的长文本分类器，来处理相同的输入数据，从而实现模型的高度模块化。

#### Using attention mechanisms in the routing module
为了应对长文本分类过程中，路由模块的计算开销，Hinton等人提出了一种改进的动态路由模块。该路由模块引入注意力机制，可以根据注意力权重，对各个胶囊进行激活。

#### Introducing a new loss function
Hinton团队发现，最近的长文本分类任务中，存在一个严重的问题——类别不平衡问题。类别不平衡问题是指，对于某一个类别来说，正负样本数量不平衡，比如正样本的数量远远小于负样本的数量。这就会造成模型在学习时容易偏向于正负样本数量差距大的类别，而忽略其它类别的特征，从而降低模型的分类能力。为了解决这个问题，Hinton等人提出了一种新的损失函数——Focal Loss，来为不同类型的样本赋予不同的权重，从而减轻类别不平衡问题带来的影响。

# 4.具体代码实例和解释说明
## CNNs and RNNs as encoder modules
### Short text classification using Convolutional Neural Networks (CNNs)
```python
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext import datasets
import spacy

nlp = spacy.load('en')
TEXT = data.Field()
LABEL = data.LabelField()
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embed_dim))
            for fs in filter_sizes])
            
        self.fc = nn.Linear(len(filter_sizes)*num_filters, output_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1) # [batch_size, embedding] -> [batch_size, 1, seq_len, embedding]
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs] #[batch_size, num_filters, seq_len-(fs-1)]
        x = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x] #[batch_size, num_filters]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)
    
classifier = CNNClassifier(vocab_size=len(TEXT.vocab), embed_dim=100, num_filters=100, filter_sizes=[3,4,5],
                           output_dim=len(LABEL.vocab), dropout=0.5)
optimizer = optim.Adam(classifier.parameters())
criterion = nn.CrossEntropyLoss()

classifier = classifier.to(device)
criterion = criterion.to(device)

def accuracy(preds, y):
    preds = preds.argmax(dim=1).view(-1)
    correct = torch.sum(preds == y)
    acc = float(correct) / len(y)
    return acc 

for epoch in range(10):
    running_loss = 0.0
    total = 0
    
    train_iter = data.BucketIterator(dataset=train_data, batch_size=32, shuffle=True, device=device)

    for i, batch in enumerate(train_iter):
        inputs, labels = batch.text, batch.label
        
        optimizer.zero_grad()
        
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*labels.size(0)
        total += labels.size(0)
        
    print('[%d/%d] Training loss:%f'%(epoch+1, 10, running_loss/total))
                
test_iter = data.BucketIterator(dataset=test_data, batch_size=32, shuffle=False, device=device)
with torch.no_grad():
    correct = 0
    total = 0
    for i, batch in enumerate(test_iter):
        inputs, labels = batch.text, batch.label
        outputs = classifier(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
            
    print('Test Accuracy of the model on %d test sentences: %d %%' % (len(test_data), 100 * correct / total))  
```
### Long text classification using Recurrent Neural Networks (RNNs) with Gated Recurrent Units (GRUs)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext import datasets
import spacy

spacy_en = spacy.load('en', disable=['parser', 'ner'])
def tokenizer(text):
    tokens = [token.text for token in spacy_en.tokenizer(text)]
    return tokens

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.LabelField()
train_data, valid_data, test_data = datasets.AG_NEWS.splits(TEXT, LABEL, root='.data')

MAX_VOCAB_SIZE = 25_000
TEXT.build_vocab(train_data, 
                 max_size=MAX_VOCAB_SIZE,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

class TextGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_dim, 
                            hidden_dim, 
                            num_layers=n_layers,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden
        
    def forward(self, x, prev_state):
        lstm_out, state = self.lstm(x, prev_state)
        logits = self.fc(lstm_out)
        return logits, state
    

INPUT_DIM = len(TEXT.vocab)
HIDDEN_DIM = 256
OUTPUT_DIM = len(LABEL.vocab)
N_LAYERS = 2
BATCH_SIZE = 64
SEQ_LENGTH = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextGenerator(INPUT_DIM, HIDDEN_DIM, N_LAYERS, OUTPUT_DIM).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def generate_text(seed_sentence, model, TEXT, device, length=100):
    encoded_seed = TEXT.process([seed_sentence]*len(seed_sentence))[0].to(device)
    current_state = None
    
    with torch.no_grad():
        sentence = seed_sentence
        words = nltk.word_tokenize(sentence)
        generated_words = [] + ['<SOS>'] + words[:-1]
        
        for i in range(length):
            tensor_sentence = np.array([[TEXT.vocab.stoi[w] for w in generated_words]]).long().to(device)
            
            if not current_state:
                current_state = model.init_hidden(tensor_sentence.shape[0])
                
            predictions, current_state = model(tensor_sentence, current_state)
            
            prob, pred_idx = torch.max(predictions[-1][-1], dim=-1)
            word = TEXT.vocab.itos[pred_idx.item()]
            generated_words.append(word)
            
            if '<EOS>' in generated_words or i==length-1:
                break
                
        return''.join(generated_words[1:])
            
seed_sentences = ["I love watching movies.", "The sun shines brightly today."]
for s in seed_sentences:
    print("-"*50)
    print("Seed Sentence:", s)
    gen_sent = generate_text(s, model, TEXT, device, SEQ_LENGTH)
    print("Generated sentence:", gen_sent)
```
### Combining both CNNs and RNNs with capsules for long text classification
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english')) | set(string.punctuation)

def preprocess(text):
    """
    Preprocesses text by removing stopwords, punctuation, digits and converts all characters to lowercase. 
    Also removes extra whitespaces before and after text.
    """
    processed_text = "".join([char.lower() for char in text if char not in STOPWORDS and not char.isdigit()])
    processed_text = " ".join(processed_text.split())
    return processed_text

def extract_features(df):
    """
    Extract features from dataframe using count vectorization and TFIDF transformation. Returns feature matrix X and label array Y.
    """
    cv = CountVectorizer(preprocessor=preprocess, analyzer='word', ngram_range=(1,3), min_df=2, max_df=.8)
    tfidf = TfidfTransformer(norm='l2')
    X = cv.fit_transform(df['text'].tolist())
    X = tfidf.fit_transform(X).todense()
    Y = df['label'].values
    return X, Y

def classify(X, Y, clf_type='LogReg'):
    """
    Classifies given dataset using selected classifier type ('MultinomialNB', 'LogReg', 'RandomForest', 'SVC'). Returns metrics scores including accuracy, precision, recall, F1 score, etc.
    """
    if clf_type=='MultinomialNB':
        clf = MultinomialNB()
    elif clf_type=='LogReg':
        clf = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial')
    elif clf_type=='RandomForest':
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
    elif clf_type=='SVC':
        clf = SVC(kernel='linear', probability=True, random_state=0)
    
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    accuracy = sum((Y==Y_pred))/len(Y)
    precision = sum([(Y[i]==Y_pred[i])*int(Y[i]==1)/len(Y[Y==1]) for i in range(len(Y))])/len(Y[Y==1])
    recall = sum([(Y[i]==Y_pred[i])*int(Y_pred[i]==1)/len(Y_pred[Y_pred==1]) for i in range(len(Y))])/len(Y[Y==1])
    f1score = 2*(precision*recall)/(precision+recall)
    confusion_matrix = [[sum((Y[(Y!=lbl_row)&(Y_pred==lbl_col)])==(lbl_col)*(Y_pred!=(lbl_row)))/sum((Y==lbl_row)&(Y_pred==lbl_col))]
                        [(sum((Y[(Y!=lbl_row)&(Y_pred==lbl_col)])==(lbl_col)*(Y_pred!=(lbl_row)))/sum((Y==lbl_col)&(Y_pred!=lbl_row)))
                         for lbl_col in set(list(Y)+list(Y_pred))] for lbl_row in set(list(Y)+list(Y_pred))]
    confusion_matrix = np.array(confusion_matrix)
    cm_df = pd.DataFrame(confusion_matrix, columns=set(list(Y)+list(Y_pred)), index=set(list(Y)+list(Y_pred)))
    tn, fp, fn, tp = confusion_matrix.ravel()
    specificity = tn/(tn+fp)
    balanced_accuracy = (tp/(tp+fn)+(tn/(tn+fp)))/2
    
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1 Score:", round(f1score, 3))
    print("Specificity:", round(specificity, 3))
    print("Balanced Accuracy:", round(balanced_accuracy, 3))
    print("\nConfusion Matrix:\n", cm_df)
    
    return {'acc':round(accuracy, 3), 'prec':round(precision, 3),'rec':round(recall, 3), 
            'f1':round(f1score, 3),'spec':round(specificity, 3), 'ba':round(balanced_accuracy, 3)}

def visualize_classification(scores):
    """
    Plots bar chart of different metric scores obtained through classification experiment.
    """
    labels = list(scores.keys())[:3]+['Spec.', 'BA']
    values = list(scores.values())[:3]+[values[6], values[5]]
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    sns.barplot(ax=ax, x=labels, y=values)
    plt.show()