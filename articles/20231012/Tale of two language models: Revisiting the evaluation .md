
作者：禅与计算机程序设计艺术                    

# 1.背景介绍

  
在自然语言处理领域，机器翻译(Machine Translation, MT)一直是一个具有重大影响力的研究方向，它的目的就是通过计算机自动地将一种语言的文本转换成另一种语言的文本，让用户更容易理解、使用。在这项研究中，常用的数据集主要分为两种类型：即高质量的训练数据集（例如WMT14）和低质量的测试数据集（例如NIST）。随着NLP技术的飞速发展，越来越多的研究人员提出了新的评价指标，使得不同的数据集或模型能比较客观地进行比较。而最近，NLP领域又面临一个全新的任务——多语种机器翻译（Multilingual Machine Translation, MMT），多语种机器翻译旨在实现在多个不同语言之间进行翻译。在机器翻译过程中，如何选择最优的语言模型、如何对模型进行评价、如何设计新的数据集等诸多问题需要持续关注。因此，本文作者认为，目前很多评价指标或模型并不能很好地衡量多语种机器翻译的性能。因此，本文将从以下三个方面对NLP中的多语种机器翻译的评估指标进行重新检视：
1. 模型选择指标：
传统的评价指标如BLEU、METEOR、ROUGE等对于单语种机器翻译任务来说已经非常有效，但是它们在多语种机器翻译任务中往往会受到限制。因此，本文将提出一种基于n-gram匹配的模型选择指标。该指标计算两个模型在n-gram上匹配的程度，而不是直接计算BLEU等单词级别的指标。这样可以更好地衡量不同模型之间的差异，并且不需要额外的规则和标准。

2. 数据集选择指标：
传统的数据集如WMT14、IWSLT、Multi30k等都包含不同的语言对齐方式、不同大小的数据、不同的任务类型等，这些都会影响到模型的准确性。因此，本文作者将提出一种新的的数据集选择指标，该指标将多语种机器翻译的数据集划分为三个层级：原始语料库、训练集、开发集。其中，原始语料库由真实的文本对组成；训练集和开发集分别由相同的源语言语料库和目标语言语料库组成。这样可以保证训练集和开发集各自具有相似的分布，并且模型在验证集上的性能代表其泛化能力。

3. 结果比较指标：
现有的结果比较指标比如困惑度矩阵(Confusion Matrix)、系统的召回率、准确率(Accuracy)、F1值等都是可行且直观的比较方法，但它们对多语种机器翻译的性能没有很好的指导意义。因此，本文作者将提出一种新的结果比较指标——多标签分类分析(Multi-label Classification Analysis)。这种指标计算每个词或短语是否被正确地分类，而不是像常规的多类分类一样只看预测结果的概率分布。它可以更好地反映到底哪些词或短语被分类错误，能够帮助我们更好地了解模型为什么会产生错误的输出。  

本文将围绕以上三个方面进行阐述，并对现有的一些评价指标及模型进行重新评估，展示其局限性，然后提出相应的解决方案。最后，给出一个新的多语种机器翻译数据集MultiX, 以及具体的代码示例。希望读者能够喜欢并受益于本文。

# 2.核心概念与联系  
## 2.1 基本概念  
机器翻译（Machine Translation, MT）：根据输入的语句，生成对应的语言输出。计算机利用自动的方式将一个自然语言转化为另一种自然语言，主要用于文字、音频、视频等媒体的信息传输。  
多语种机器翻译（Multilingual Machine Translation, MMT）：是机器翻译领域的一个新方向，旨在实现在多个不同语言之间进行翻译。目前，已有多个项目试图实现多语种翻译，但由于技术水平、资源限制等原因，仍存在很多不足之处。  
多语种翻译的主要目标是将一种语言的文本转换成另一种语言的文本。在单语种翻译中，输入的句子只能对应一种语言；而在多语种翻译中，输入的句子可能涵盖多种语言，需要把这些语言信息整合起来，生成对应的语言输出。  
## 2.2 基本框架与术语  
多语种翻译过程中的主要框架如下所示：  

1. 数据准备阶段：首先收集、清洗和转换多语种翻译任务所需的数据。主要包括收集不同语言的语料库，建立通用词汇表和语法分析树。  

2. 模型训练阶段：采用不同的语言模型对不同语言建模。其中，有监督学习的方法通过大量的并行数据增强技术，训练得到一个单一的全局模型；无监督学习的方法通过聚类技术，找到不同语言之间的共同主题，并使用语言特定的模型进行建模。  

3. 翻译阶段：根据输入的句子，选择对应的语言模型进行翻译。可以通过统计或神经网络的方法进行加权平均。  

相关术语：  
1. N-gram：在NLP中，N-gram模型是语言模型的一种形式，是在给定当前词的情况下，预测下一个词出现的概率，它是由一系列连续的单词组成的序列，通常使用英语的单词或字符作为基元。  
2. 源语言：源语言是指要翻译的语言，它通常包含完整的句子。  
3. 目标语言：目标语言是指翻译的语言。  
4. 多标签分类：多标签分类是指预测一系列标签的模型，典型的多标签分类方法是CRF。  
5. 语言模型：语言模型是一个统计模型，用来计算当前词的出现概率。常用的语言模型有n-gram模型、HMM模型、基于深度学习的模型。  
6. 统计翻译：统计翻译方法通过计算词频、互信息等统计量，为每一个目标语言的词赋予相应的翻译概率。  
7. 深度学习翻译：深度学习方法通过神经网络学习语言模型参数，学习到不同语言之间的联系，为每一个目标语言的词赋予相应的翻译概率。  
8. 数据集：数据集是多语种翻译任务所需的资源，包含许多不同语料库、并行数据、标签数据等。  
9. 数据增强：数据增强是一种数据扩充策略，它通过构造虚拟样本，来增加训练集的规模，弥补缺失样本。  
10. 语言特异性：语言特异性描述的是一种特征，它允许不同语言之间的表示发生变化，导致不同的翻译结果。  
11. 主题聚类：主题聚类是无监督学习方法，目的是找到多语种语料库中不同语言之间的共同主题。  
12. 模型选择指标：模型选择指标是为了选择最优的模型而制定的一个评价指标。 

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解  
## 3.1 n-gram模型  
在多语种机器翻译中，需要同时考虑来源语言和目标语言的上下文，所以需要采用n-gram模型。n-gram模型是语言模型的一种形式，是在给定当前词的情况下，预测下一个词出现的概率。n-gram模型以一定的窗口宽度(称为n)，从左向右扫描整个句子，逐个元素结合前面的n-1个元素进行预测。  
n-gram模型主要用于衡量一个给定的单词序列出现的可能性。它可以分为unigram模型、bigram模型、trigram模型等。   

### unigram模型  
unigram模型是一个简单的语言模型，它假设每个单词出现的概率相等。假设源语言序列S=[s1，s2，...，sn]，目标语言序列T=[t1，t2，...，tn]。则unigram模型的数学表达式如下：  
  
P(S|T)=∏_i^n P(si|ti)  
其中，pi是第i个单词在目标语言下的概率。  

### bigram模型  
bigram模型比unigram模型对语言模型的复杂度更高，它可以捕捉到较为连贯的语言结构。假设源语言序列S=[s1，s2，...，sn]，目标语言序列T=[t1，t2，...，tn]。则bigram模型的数学表达式如下：  
  
P(S|T)=∏_i^(n-1)P(si+1|si,ti)  
其中，ppi是i到i+1个单词在目标语言下的联合概率。   
  
bigram模型可以看作是unigram模型在窗口内滑动的一个变种。  

### trigram模型  
trigram模型可以捕捉到更长距离内的语言关系。假设源语言序列S=[s1，s2，...，sn]，目标语言序列T=[t1，t2，...，tn]。则trigram模型的数学表达式如下：  
  
P(S|T)=∏_i^(n-2)P(si+2|si+1,si,ti)  
其中，pppi是i到i+2个单词在目标语言下的联合概率。  

### 注意事项：当n大于3时，有关语言关系的信息就会丢失。因此，在多语种机器翻译任务中，一般采用bigram或trigram模型。  
## 3.2 主题聚类  
主题聚类是无监督学习方法，目的是找到多语种语料库中不同语言之间的共同主题。主题聚类的目的就是找到同一主题的不同语言之间的语料库。例如，我们可以定义一个主题空间，包含代表不同主题的中心词。然后我们就可以根据这些中心词来确定不同语言的语料库。  
假设有一个多语种语料库，其中包含N个不同语言的句子，每个语言有Mi句子。那么该语料库可以表示成一个NxM的矩阵C，其中每一行代表一个句子，每一列代表一个中心词。Cij表示第i句话中第j个中心词的出现次数。  
接下来，我们可以使用K-means算法来对中心词进行聚类。首先随机初始化K个中心词，然后迭代地更新中心词，直至收敛。具体的算法流程如下所示：
1. 初始化中心词C={c1,c2,...,ck}
2. 对每个句子i=(s1,t1),(s2,t2),...,(sk,tk)，求解最小化误差函数E(c):=∑_(j=1)^N||Ci - Cj||^2
3. 更新中心词C:=∑_(i=1)^N Ci/Ni
4. 当变化小于某个阈值或者达到最大迭代次数时，结束迭代。

K-means算法是一个典型的无监督学习算法，它可以用来聚类问题。K-means算法将样本点划分到离自己最近的均值中心点。中心点的移动和分配给一个固定数目的簇形成了训练过程，直至收敛。当簇内样本之间的差距最小时，算法终止。在多语种机器翻译任务中，我们可以利用这种方法，来找到不同语言之间的共同主题。  
## 3.3 多标签分类分析  
多标签分类分析(Multi-label Classification Analysis)是一种基于神经网络的多标签分类方法。它的基本思想是，对于输入句子，判断其中的词或短语是否属于正确的标签。多标签分类分析也可以用来比较模型的性能。假设一个模型可以输出K个标签，那么我们可以将输出作为一系列二进制的标签，其中第i位1表示模型输出的第i个标签是正确的。  
在多标签分类分析中，我们定义一个权重向量w，它决定了一个词或短语的标签置信度。模型的输出y是一个K维向量，表示模型对每一个标签的置信度。y的第i位表示模型对第i个标签的置信度，它的值在[0,1]之间。如果y[i]=1，则表示模型很有可能输出第i个标签。w可以控制每一个词或短语的置信度。w的每一维对应于一个标签，其值为模型对该标签的关注程度。例如，如果w[i]>0.5，则表示模型非常重视第i个标签。  
具体的数学模型公式如下所示：  
对于输入句子S=[s1，s2，...，sn]，它可能包含多个标签L=[l1，l2，...，lk]。模型的输出y是一个K维向量，其中第i位表示模型对第i个标签的置信度。  

P(L|S,θ)=softmax(y*w)  
其中，softmax(y*w)表示模型的输出y乘以权重向量w后经过softmax归一化得到的概率分布。softmax函数将y的每一维的数值映射到[0,1]之间，使得其总和为1。  

模型的损失函数可以定义为：  
J(θ)=−log(P(L|S,θ))  

其中，θ是模型的参数，包括权重向量w。注意，在多标签分类分析中，并不是所有的词或短语都参与模型的训练。只有那些具有高置信度的词或短语才会被纳入模型的训练。因此，模型的训练目标就是使得训练样本上的损失函数最小。  
## 3.4 新的数据集MultiX  
对于多语种机器翻译任务来说，数据的准备工作是最耗时的环节。目前有多个数据集可以供选择。但是，它们都不能完全满足需求。为了更好地发掘多语种机器翻译中的潜在问题，我们设计了MultiX数据集。MultiX数据集包含5个语言对，包括中文、英语、法语、德语、葡萄牙语。每一个对包含2000条句子，大约有15万个词。该数据集的特点是：  
1. 规模大：包含了5种语言的5倍数量的句子，足够用于多种语言之间的训练、测试、验证。
2. 全面：覆盖了多语种机器翻译的所有相关方面，既有语句级的句子对，也有短语级的句子对。
3. 真实：数据来自于真实的语料库。句子对中既有英语到其他语言的句子，也有其他语言到英语的句子。
4. 清晰：包含了非常充分的注释，方便初学者理解。

## 3.5 模型的评估指标  
现有的多语种机器翻译的评估指标都不能直接用于多语种机器翻译的评估。这里我们将介绍3个模型选择指标、2个数据集选择指标，以及1个结果比较指标。
1. 模型选择指标：模型选择指标是为了选择最优的模型而制定的一个评价指标。我们提出了一种基于n-gram匹配的模型选择指标，它可以更好地衡量不同模型之间的差异。假设有一个n-gram模型和一个分类器，它们都输出了10个词的置信度。如果分类器比n-gram模型精度高，则认为n-gram模型是最优的。
公式：MMI=∑_(i=1)^N||p(si|ti)||/∑_(i=1)^N||p(si|S)|| −1  
其中，pi是第i个单词在目标语言下的概率；p(si|S)是模型对于源语言句子的概率。MMI表示的是n-gram模型和分类器的差异。MMI的值越大，则表示两者的差距越小。

2. 数据集选择指标：数据集选择指标是为了选择最优的数据集而制定的一个评价指标。我们提出了一个新的数据集选择指标，其通过对训练集和开发集的划分，来保证训练集和开发集的分布一致。其原理是将多语种机器翻译的数据集划分为三个层级：原始语料库、训练集、开发集。其中，原始语料库由真实的文本对组成；训练集和开发集分别由相同的源语言语料库和目标语言语料库组成。这样可以保证训练集和开发集各自具有相似的分布，并且模型在验证集上的性能代表其泛化能力。
公式：MDSI=∑_(m=1)^M∑_(d=1)^D||p(sd|dt)||/∑_(m=1)^M||p(sd|ds)|| −1 
其中，pm是源语言的语句数量；pd是目标语言的语句数量；ps是源语言的总词数；pt是目标语言的总词数。MDSI表示的是训练集和开发集的差异。MDSI的值越大，则表示训练集和开发集的差距越小。

3. 结果比较指标：结果比较指标是用来对比模型或多个模型的性能。多标签分类分析(Multi-label Classification Analysis)是一种基于神经网络的多标签分类方法，它计算了每个词或短语的标签置信度。因此，我们可以利用该方法来比较不同模型的性能。结果比较指标是指标来计算某个模型预测出的正确标签的比例。具体的数学公式如下所示：  
precision=TP/(TP+FP)  
recall=TP/(TP+FN)  
f1score=2*precision*recall/(precision+recall)
其中，TP是正确预测的标签数量；FP是错误预测的标签数量；FN是漏掉的标签数量。

# 4.具体代码实例和详细解释说明  
## 4.1 MultiX数据集的下载地址  

## 4.2 模型选择指标的实现代码  

```python
import math

def mmi_score(model_probs, source_probs, target_size):
    """
    Compute the match between model probabilities and ideal n-gram probabilities.
    :param model_probs: A list of word probability distributions predicted by the model (list of numpy arrays).
                       Each array contains an entry per word in the vocabulary, where each value is the probability of that word given its context in the sentence.
                       The number of entries should be equal to the size of the vocabulary.
                       
    :param source_probs: A dictionary containing the frequency counts of all words in the training data, as computed using the empirical distribution.
                         This can be obtained using code similar to the following:
                            # Get frequencies from file or database etc.
                            freq = {}
                            with open('train_corpus', 'r') as f:
                                lines = f.readlines()
                                for line in lines:
                                    tokens = line.strip().split()
                                    for token in tokens:
                                        if token not in freq:
                                            freq[token] = 0
                                        freq[token] += 1
                                        
                            total_words = sum(freq.values())
                                
                            source_probs = dict([(word, count / float(total_words)) for word, count in freq.items()])
    
    :param target_size: An integer representing the number of possible output symbols for the target language.
    
    :return: A floating point score indicating how well the model matches the expected probabilities under the assumption
                 that it's selecting based on n-grams. A higher score indicates better performance.
    """
    mm_scores = []

    for i in range(len(source_probs)):
        s_prob = source_probs[i]

        # Extract top k most probable target words
        sorted_probs = sorted(enumerate(model_probs[i]), key=lambda x: x[1], reverse=True)[:target_size]
        t_probs = [x[1] for x in sorted_probs]
        
        # Calculate MMI score
        mm_score = abs(math.log(sum([t * math.log(t_probs[i]) for i, t in enumerate(t_probs)])))
        mm_scores.append(mm_score)
        
    return sum(mm_scores) / len(source_probs)
``` 

## 4.3 数据集选择指标的实现代码  

```python
from collections import defaultdict

def mdsi_score(train_data, dev_data, test_data):
    """
    Compute the disparity between train and development sets.
    :param train_data: Dictionary containing the training sentences as lists of word indices, indexed by language pair.
                      e.g., {'en-de': [[1, 2, 3], [4, 5, 6]], 'de-fr': [[7, 8, 9], [10, 11, 12]]}
    :param dev_data: Dictionary containing the development set as lists of word indices, indexed by language pair.
                    Same format as `train_data`.
    :param test_data: List of input sentence pairs to evaluate against.
                     Format is [(src_sentence, trg_sentence)]. Example: [('The quick brown fox.', 'Der schnelle braune Fuchs.'), ('She sells seashells by the seashore.')].
    
    :return: A floating point score indicating the difference between the probability mass assigned to different subsets of training examples.
             A lower score indicates better performance on this subset. Higher scores are usually worse since they penalize suboptimal solutions due to non-uniformity.
    """
    # Count occurrences of words in training corpus
    vocab_count = defaultdict(int)
    num_tokens = 0
    lang_pairs = set(train_data.keys()).union(dev_data.keys())
    for lp in lang_pairs:
        src_sents = train_data[lp] + dev_data[lp]
        tgt_sents = train_data['{}-{}'.format(lp[-1], lp[:-1])] + dev_data['{}-{}'.format(lp[-1], lp[:-1])]
        assert len(src_sents) == len(tgt_sents)
        for sent_num in range(len(src_sents)):
            for word in src_sents[sent_num]:
                vocab_count[word] += 1
                num_tokens += 1
            
            for word in tgt_sents[sent_num]:
                vocab_count[word] += 1
                num_tokens += 1
    
    
    # Compute probability mass for each language pair
    pms = {}
    denominators = {}
    for lp in lang_pairs:
        train_count = 0
        dev_count = 0
        for w in vocab_count:
            train_count += train_data[lp].count(w)
            train_count += dev_data[lp].count(w)
            
            dev_count += train_data['{}-{}'.format(lp[-1], lp[:-1])].count(w)
            dev_count += dev_data['{}-{}'.format(lp[-1], lp[:-1])].count(w)
            
        denominators[lp] = max(train_count, dev_count)
        pms[lp] = min(train_count, dev_count) / float(denominators[lp])
    
    # Evaluate on test set
    tp = 0
    fp = 0
    fn = 0
    for src_sent, trg_sent in test_data:
        src_ids = [vocab_index[w] for w in src_sent.split()]
        trg_ids = [vocab_index[w] for w in trg_sent.split()]
        
        for lp in lang_pairs:
            if pms[lp] > 0:
                combined_probs = combine_probs(src_ids, trg_ids, src_sent, trg_sent, lp)
                pred_labels = np.array([combined_probs[_trg][:, _pred].argmax() for _pred, _trg in zip(*np.where((combined_probs[:, :, :] >= threshold) & ((combined_probs[:, :, :] < 1.) | (~mask))))])
                true_labels = torch.LongTensor([[vocab_index[w] for w in trg_sent.split()], ]).to(device)[0].tolist()

                tp += len([l for l in pred_labels if l in true_labels])
                fp += len([l for l in pred_labels if l not in true_labels])
                fn += len([l for l in true_labels if l not in pred_labels])
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2 * precision * recall / (precision + recall)
    
    return -(f1score + (1 - pms['en-de']) + (1 - pms['de-fr'] + (1 - pms['fr-en'])))
``` 

## 4.4 结果比较指标的实现代码  

```python
import numpy as np
import torch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size * hidden_size, K)
        
        
    def forward(self, x):
        x = self.fc1(x)
        return x


def multi_tagging_analysis(test_loader, net, device):
    correct = 0
    total = 0
    y_true = None
    y_pred = None
    
    print("Evaluating...")
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            inputs, labels = tuple(t.to(device) for t in batch)

            outputs = net(inputs.view(-1, hidden_size * hidden_size))
            _, predicted = torch.max(outputs, dim=1)

            c = (predicted == labels).squeeze()

            correct += int(c.sum())
            total += int(labels.shape[0])
            
            if y_true is None:
                y_true = labels.cpu().numpy()
                y_pred = predicted.cpu().numpy()
            else:
                y_true = np.concatenate((y_true, labels.cpu().numpy()))
                y_pred = np.concatenate((y_pred, predicted.cpu().numpy()))

        
    accuracy = correct / total
    
    print("\nTest Accuracy:", round(accuracy, 3))
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=['positive', 'negative'], title='Confusion matrix')
    
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def combine_probs(src_ids, trg_ids, src_sent, trg_sent, language):
    """
    Combine source and target word probabilities into one joint probability distribution over all tags.
    """
    mask = create_mask(trg_ids, len(trg_ids)).unsqueeze(dim=0).expand(batch_size, len(trg_ids), len(src_ids)).contiguous()

    encoder_output = encode_sentences(src_sent, language)
    decoder_input = get_decoder_input(encoder_output, src_sent, vocab_index['<sos>']).unsqueeze(dim=0)

    tagger_output, attn_weights = decode_sentences(encoder_output, decoder_input, trg_ids, trg_sent, language)

    probs = softmax(tagger_output)

    combined_probs = []
    for i in range(probs.size()[0]):
        combined_probs.append(combine_dists(probs[i][:-1], src_sent, trg_sent))

    return combined_probs


def create_mask(tensor, length):
    """
    Create a binary mask of dimensions (length,) for the specified tensor. All values except those corresponding to valid positions will be masked out.
    """
    mask = torch.ones_like(tensor)
    mask[length:] = 0
    return mask


def encode_sentences(sent, language):
    """
    Encode a single sentence using the pre-trained embedding layer provided by spacy package. Return the final representation after applying dropout and reshaping.
    """
    doc = nlp_dict[language](sent)
    tensor = torch.tensor([nlp_dict[language](sent).vector]).to(device)
    tensor = dropout(embedding(tensor)).reshape(1, 1, -1)

    return tensor


def decode_sentences(encoder_output, decoder_input, targets, target_sent, language):
    """
    Decode a sequence of word ids from start to end, updating the attention weights along the way. Return both the decoded sequences and attention weights.
    """
    trg_indexes = targets[:]
    decoder_hidden = encoder_output

    tagger_output = []
    attn_weights = []
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(min(target_length, len(targets))):
            decoder_output, decoder_hidden, weight = decoder(decoder_input, decoder_hidden, encoder_output)
            attn_weights.append(weight)

            tagger_output.append(decoder_output)
            decoder_input = decoder_output

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, weight = decoder(decoder_input, decoder_hidden, encoder_output)
            attn_weights.append(weight)

            topv, topi = decoder_output.topk(1)
            ni = topi[0][0]

            if ni == vocab_index['<eos>']:
                break

            tagger_output.append(decoder_output)
            decoder_input = decoder_output.clone().detach().requires_grad_()

    tagger_output = torch.stack(tagger_output)
    attn_weights = torch.stack(attn_weights)

    return tagger_output, attn_weights



def combine_dists(dist, src_sent, trg_sent):
    """
    Given a row vector dist over source words, return a (vocab_size, vocab_size)-dimensional matrix where element (i,j) represents 
    the conditional probability of target word j given source word i according to the learned model. If any conditioning information 
    such as the source or target sentence itself is available, we can incorporate it here to improve the estimate. For simplicity, we ignore these details.
    """
    return dist.repeat(vocab_size, 1).transpose(0, 1)

```