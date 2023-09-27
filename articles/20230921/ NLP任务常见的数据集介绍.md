
作者：禅与计算机程序设计艺术                    

# 1.简介
  

近年来，自然语言处理（Natural Language Processing， NLP）技术逐渐成为深度学习领域中的热门方向。而大量的应用场景都需要基于海量文本数据进行处理，因此收集、整理、标注相关的数据集成了构建模型所需的重要基础。但不同的数据集又往往存在较大的差异性，因此在选择合适的数据集时还需要综合考虑各方面因素。本文将从以下几个方面详细介绍NLP任务常见的数据集：
- 数据大小及规模
- 数据分布
- 数据类别
- 数据格式
- 处理流程和工具
2.数据集一览
## 数据集介绍
### CoLA(Corpus of Linguistic Acceptability)
CoLA数据集是一个加英语维基百科训练数据集，由William Bowman创建，目的是检查文本是否满足语法和语法上的逻辑约束。该数据集共包括106,570个句子，其中18,928个句子正确，剩余的97,642个句子属于错误。句子错误主要有两种类型：词法错误和语义错误。词法错误包括错别字、拼写错误、语法结构不符合等；语义错误包括指代消失、主谓倒置、依赖不明确等。如果模型判断出句子不符合语法要求，则很可能出现语义错误，反之，则属于词法或句法上的错误。
下载地址:https://github.com/google-research/bert/blob/master/glue_data/CoLA.zip  
原始数据形式如下：
```text
- [CLS] The dog is on the table. [SEP] <acceptable>
- [CLS] A man, a plan, a canal : Panama! [SEP] <unacceptable>
- [CLS] He drank too much. [SEP] <unacceptable>
```
### SST-2(Stanford Sentiment Treebank)
SST-2数据集也是一个加英语维基百科训练数据集，由Jay Alammar、Yiqing Liu和Richard McCaw担任研究员，发布于2012年。它是一个情感分析数据集，共包括两个标签：<positive>表示积极的情感，<negative>表示消极的情感。每一条评论是由一个句子和对应的标签组成。评论作者给出的评价一般带有褒贬意识，因此不一定准确。例如，评论“这个餐厅非常好吃！”有褒义，但是仍然具有一定的情感倾向，所以其标签可能是<positive>。
下载地址:http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip  
原始数据形式如下：
```text
( ( ( sentiment ) ) sst_label ) # <tree>
```
### MRPC(Microsoft Research Paraphrase Corpus)
MRPC数据集也是一个加英语维基百科训练数据集，由David Potter、Waleed AbuEl-Haij和Gang Wei担任研究员，发布于2013年。它是一个语义相似度数据集，主要是对句子进行重复并生成新句子，要求模型能够判断出两个句子之间的语义相似度。
下载地址:http://www.aclweb.org/anthology/D17-1052.pdf  
原始数据形式如下：
```text
Quality: Positive	Sentence 1: I like this car quite a bit.	Sentence 2: This car is very nice for me to own.	Label: 1|0
```
### RTE(Recognizing Textual Entailment)
RTE数据集也是一个加英语维基百科训练数据集，由Jinho Lee、Zhili Shen、Lily Deng、Xiaoming Chen和Amy Yu担任研究员，发布于2014年。它是一个文本蕴含关系数据集，要求模型判断一个假设句与一个证据句之间的蕴含关系。
下载地址:https://cogcomp.seas.upenn.edu/Data/RTE/  
原始数据形式如下：
```text
News Article 1: Google and Apple are starting a partnership.
News Article 2: Apple and Google have announced they will be forming a new partnership.
Label: Yes/No entails.
```
### QQP(Quora Question Pairs)
QQP数据集也是一个加英语维基百科训练数据集，由Zhihua Zhang、Le Tao Tian、Xiangting Wang、Yu Rui Gao和Ke Sida担任研究员，发布于2016年。它是一个问答匹配数据集，要求模型判断两条对话问题是否相关。
下载地址:https://github.com/facebookresearch/ParlAI/blob/master/downloads.md  
原始数据形式如下：
```text
Question 1: What kind of movies did Tim Urban ever play in?
Question 2: How do you make toast with garlic bread?

Answer: Tim Urban played some hit movies, such as Pulp Fiction, Mission Impossible, Finding Dory, etc., but he never directed any films himself. 
To make toast with garlic bread, follow these steps:

1. Slice the garlic into small pieces
2. Add salt and pepper to the garlic
3. Toast the slices together in butter until golden brown
4. Stir the garlic mixture with your hands while cutting into a disc shape
5. Drop the garlic disc onto hot flatbread or buttered toast