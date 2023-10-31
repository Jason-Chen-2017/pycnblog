
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


机器翻译（MT）是自然语言处理中的一个子领域，主要目的是实现将一种语言的语句或文本转换成另一种语言的形式。根据不同的任务需求、所需翻译对象、资源、质量、目标语言等不同条件，机器翻译可以分为四种类型：文本翻译、语音翻译、手写翻译和图像翻译。文本翻译又可细分为自动文本翻译和人工文本翻译两种。本文主要讨论自动文本翻译技术。
自动文本翻译是指通过计算机将一种语言的数据文本转换为另一种语言数据文本的过程。自动文本翻译包括句法分析、语义理解、翻译生成、并行化处理、计算资源优化等多个环节。机器翻译领域对于自动文本翻译技术的应用非常广泛，其在许多应用场景下都有重要作用。如，搜索引擎、聊天机器人、智能助手、视频监控、视频编辑、网页翻译、报刊杂志电子化等。其中，最常见的应用场景就是网页翻译。例如，当用户访问百度翻译时，若遇到需要翻译的中文页面，则调用百度翻译API进行翻译。下面我们先了解一下机器翻译的相关术语。
# 2.核心概念与联系
## 1.词汇表
### 1.1 词汇表
| 英文 | 中文 | 描述 |
| ---- | ---- | ---- |
| Machine Translation | 机器翻译 | 从一种语言翻译成另外一种语言的过程 |
| Natural Language Processing (NLP) | 自然语言处理（Natural Language Understanding，Natural Language Generation，NLU，NLG） | 技术研究和开发人员用来处理及运用自然语言的系统、方法和技术 |
| Word-based model | 基于词的模型 | 根据词的相似性或上下文关系进行翻译的统计模型 |
| Neural machine translation | 神经机器翻译（Neural MT） | 通过神经网络来实现机器翻译的技术 |
| Sequence to sequence learning | 序列到序列学习 | 使用LSTM（长短期记忆网络）或者RNN（循环神经网络）来实现神经机器翻译 |
| Encoder-decoder framework | 编码器—解码器框架 | 将输入序列映射到输出序列的序列到序列学习方法 |
| Syntactic analysis | 句法分析 | 分割句子成词、标注句法结构和句法树等过程 |
| Parsing tree | 语法分析树 | 表示句子中各个词语之间的逻辑关系的树形结构 |
| Semantic understanding | 意义理解 | 对文本内容含义的理解 |
| Dependency parsing | 依存分析 | 根据句法分析树确定每个词语与其他词语之间的关联 |
| Stroke-level modeling | 笔画级别建模 | 以笔画为单位对汉字进行表示，是传统机器翻译的基础 |
| Character-level models | 字符级别模型 | 以字符为单位对汉字进行表示，是一些比较新颖的机器翻译方法 |
| Recurrent neural network (RNN) | 循环神经网络 | 模型中的基本运算单元，用于处理序列数据 |
| Long short term memory networks (LSTM) | 长短期记忆网络 | RNN的变体，能够捕获时间上的顺序关系 |
| Backpropagation through time (BPTT) | 反向传播时间 | 用梯度下降法训练RNN参数的迭代方式 |
| Greedy decoding | 贪婪解码 | 在生成翻译结果时采用最大概率策略 |
| Beam search decoding | 束搜索解码 | 在生成翻译结果时采用多重采样的方式 |
| Penalty functions | 惩罚函数 | 用于控制生成翻译结果的多样性 |
| Preprocessing | 数据预处理 | 删除停用词、过滤无意义词等处理 |
| Postprocessing | 后处理 | 调整结果的风格、句法等 |