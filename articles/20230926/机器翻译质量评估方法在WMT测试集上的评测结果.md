
作者：禅与计算机程序设计艺术                    

# 1.简介
  

&emsp;&emsp;近年来，机器翻译系统的性能逐渐提高，而在性能的提升过程中，对于翻译质量的评估也越来越重要。从传统的基于统计模型的词级别的评价方法到现代的基于神经网络的单词或句子级别的评价方法，都给了不同程度的改善。然而，如何有效地衡量一个系统的翻译质量，仍然是一个值得探索的问题。本文就试图回顾一下自然语言处理（NLP）领域最流行的机器翻译质量评估方法——BLEU、TER、METEOR以及SMOOTH，并且分析其在WMT测试集上得到的各项结果。

在开始之前，让我们先对评价方法进行一些基本介绍。

# 2.评价方法
## BLEU
&emsp;&emsp;Bilingual Evaluation Understudy (BLEU) 是由NIST于2002年提出的一种自动化的评估方法。它的目标是在机器翻译中准确评估生成的句子的多样性和一致性。它通过计算参考语句和生成语句之间的n-gram重合率(即将生成句子中的n个词元与参考句子中的相应n个词元匹配所获得的分值)，然后取这些分值的加权平均来衡量生成句子的多样性。

## TER
&emsp;&emsp;Translation Error Rate (TER) 是另一种机器翻译质量评估指标。该方法主要用于统计机器翻译系统产生错误的词汇占比。它与BLEU类似，也是通过计算参考语句和生成语句之间的n-gram相似性来衡量生成句子的多样性。但不同之处在于，TER不仅考虑语法上的差异，还包括拼写上的差异，而BLEU只考虑语法上的差异。因此，TER能够更全面地反映系统的翻译质量，适用于需要更全面的评估的场景。

## METEOR
&emsp;&emsp;Metric for Evaluation of Translation Output (METEOR) 是一个基于短语级别的机器翻译质量评估方法。它对两个句子都进行分词并计算n-gram重合率，再根据得分情况赋予权重，最后取这些分值的加权平均作为整个句子的质量评价。

## SMOOTH
&emsp;&emsp;SMOOTHe Automatic MT Evaluation and Orchestration System (SMOOTH) 是由微软亚洲研究院提出的自动化的机器翻译评估方法。它的目标是建立起自动化的评估体系，自动收集、分析和报告所有翻译系统的质量数据。它利用了其他四种方法的优点，对机器翻译质量进行多维度的评估。

# 3.WMT 2014评测任务
&emsp;&emsp;今年的评测任务是“WMT14 News Translation Task”，主要关注新闻英文到中文的翻译质量。该任务共计177,946对英文和中文新闻文章，其中中文翻译对齐标签完全正确的数据有177,688对。剩余的3588对则分别属于以下五类：

1. Segmentation errors: 段落划分错误
2. Alignment errors: 对齐错误
3. Tokenization errors: 分词错误
4. Post-editing errors: 后期修改错误
5. Syntactic structure errors: 句法结构错误

在此基础上，三种较新的机器翻译质量评估方法，即BLEU、TER和METEOR，都是采用词级的计算方式。本文拟选取WMT14的news test数据集作为分析对象，依据已有的相关文献进行实验验证，并找出两种方法之间的差距。