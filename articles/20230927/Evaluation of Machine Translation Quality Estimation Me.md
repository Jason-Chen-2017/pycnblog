
作者：禅与计算机程序设计艺术                    

# 1.简介
  

机器翻译领域的一个重要任务就是质量评估(Quality Estimation)，即确定一个翻译候选是否达到预期质量水平。传统的机器翻译质量评估方法基于统计学模型或规则方法，往往难以对不同数据集和翻译模型进行客观公正的比较。最近几年，越来越多的研究人员提出了一些新颖的方法来提高机器翻译质量评估的准确性、效率和效果。本文将以WMT数据集和评测指标为例，总结现有的机器翻译质量评估方法，并给出其具体应用。
# 2.Background and Terminology
机器翻译质量评估旨在衡量一个翻译候选文本的质量，由两个目标组成：(i)对单个翻译候选文本进行打分，并且可区分高质量、低质量等；(ii)将这些文本分组，建立一个对整个翻译系统质量的评价。两种目标可以看作是相关的，但又不完全相同。例如，当面对同一个句子，许多机器翻译系统会提供多个候选翻译，每种翻译都有不同的得分。但这种情况下，如果选择其中某一个或几个好的翻译作为最终的输出，则需要考虑到系统的平均质量水平。因此，一般而言，对单个翻译候选文本的评分要优于对整个系统的评价，而且两者之间存在着密切联系。

基于统计模型或规则方法的质量评估有很多种方式，包括语言模型、统计机器翻译、信用奖励模型、单词级别注意力（word-level attention）、序列建模方法等。近年来，神经网络和非监督学习的方法正在迅速发展。然而，在实践中，目前尚无统一的机器翻译质量评估方法，这是一个值得探索的问题。

# 3.Major Approaches for Evaluating Machine Translations
机器翻译质量评估方法主要可以分为以下三类：

1. Rule-based methods：按照固定策略和标准，对翻译结果进行判断。如符号替换法（literal translation），规则替换法（rule-based machine translation）等。
2. Statistical models：基于统计模型的机器翻译质量评估方法。如n-gram语言模型（n-gram language model）、词级别注意力（word-level attention）等。
3. Neural networks：利用神经网络模型自动学习翻译质量评价函数。如条件随机场（Conditional Random Field, CRF）、注意力机制（Attention Mechanism）、双向LSTM（Bidirectional LSTM）等。

除此之外，还有一些方法试图通过分析或改进训练数据的质量来评估机器翻译系统。如，计算翻译错误（error analysis）、机器翻译改进（machine translation improvement）、特征工程（feature engineering）等。

# 4.Approach Used in This Paper: Word-Level Attention with an n-gram Language Model 
## Introduction
Word-level attention mechanism has been used to improve the quality of neural machine translation (NMT) systems by modeling the dependencies between words in a sentence. The basic idea is that each word should be translated based on its contextual information from other related words, which helps them express more coherent thoughts or ideas. However, there are many ways to calculate such contextual information, including using self-attention mechanisms that rely on biases derived from previous decoder hidden states and alignment scores computed during training. In this paper, we focus on another approach called word-level attention combined with an n-gram language model as the basis for evaluating NMT outputs. 

An n-gram language model captures the probability distribution of next tokens given a sequence of tokens, where n refers to the number of consecutive words considered at once. We can use an n-gram language model trained on a large corpus of parallel sentences to evaluate the quality of individual translations produced by an NMT system. Specifically, if the output of the NMT system matches a high probability assigned by the n-gram language model, then it could indicate that the translation is good; otherwise, it may not be trustworthy enough. Moreover, since the language model provides a global view of the language structure, it also enables us to compare different NMT systems directly against one another without any need for retraining them or relying on fixed evaluation metrics like BLEU score.


## Methodology
We use two approaches for evaluating the quality of an NMT system's outputs: 

1. Predicting probabilities assigned by an n-gram language model to unigrams/bigrams constructed from the original source text and the corresponding target translation. 
2. Calculating the normalized mutual information (NMI) between the predicted target text and the reference translation provided by human annotators. 


### Predicting Probabilities Using an n-gram Language Model
To predict probabilities assigned by an n-gram language model to unigrams/bigrams in the input sentence, we follow these steps:

1. Split the input sentence into tokens, i.e., words.
2. Construct bigrams/unigrams from these tokens, respectively, up to order k. These bigrams will form our hypothesis space, and the likelihoods associated with them will determine how likely a particular translation hypothesis is. For example, if k=3 and we have the phrase "I love watching movies", then the set of hypotheses would include the following three-grams: {("I", "love"), ("love", "watching"), ("watching", "movies")}. Each bigram consists of a tuple of adjacent words, whereas an unigram is just a single word. Note that some words may appear multiple times within a sentence, so bigrams with repetition count only once while unigrams count all occurrences separately.
3. Use the n-gram language model to compute the probability of each bigram/unigram under the assumption that they occur independently. The probability estimate tells us what the probability of seeing each possible translation hypothesis is given the observed input sentence. To do this, we first tokenize both the input sentence and the reference translation into lists of subwords (or optionally characters). Then, we feed these tokenized sequences through the language model and obtain log probabilities for each hypothesis (log P(hypothesis)). Finally, we take the sum over all hypotheses and exponentiate to get the total log probability of the translation candidate. The resulting value corresponds to the probability assigned by the language model to the translation candidate, represented as a float in the range [0,1]. 

The n-gram language model works well because it assumes that translations are constructed sequentially according to patterns discovered in the training data. If we consider separate inputs and outputs in isolation, the language model will assign low probabilities to out-of-context pairs and give noisy estimates overall. On the other hand, when coupled with other features, such as word alignments or contextual knowledge extracted from external resources, the n-gram language model can provide valuable insights into the translation quality beyond traditional metrics like BLEU.  

### Comparing Different Systems Directly with Human Annotations
In addition to comparing the predictions made by the NMT system itself, we also want to see whether using human annotations instead improves the results. Specifically, we assume that human translators have annotated a subset of the test sets with their own judgments about the quality of translations obtained using various MT systems. We train a supervised classifier (e.g., logistic regression) on these annotations along with the NMT system's prediction, and report performance metrics such as accuracy, precision, recall, and F1 score. By doing this, we hope to identify cases where human expertise can lead to better translation outcomes than NMT alone. 

Note that performing direct comparisons requires access to annotation labels, but we note that the availability of such labels typically depends on the specific task being performed and on the amount of effort put forth by the annotators. It is worth keeping in mind that even though humans perform accurate annotation tasks, they may still make mistakes or misunderstandings. Therefore, the performance reported here should always be taken with caution and must be interpreted in terms of its benefits relative to conventional automatic evaluation metrics.