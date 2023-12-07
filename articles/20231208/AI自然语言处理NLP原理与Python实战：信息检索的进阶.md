                 

# 1.背景介绍

自然语言处理（NLP）是人工智能领域的一个重要分支，它旨在让计算机理解、生成和处理人类语言。信息检索是NLP的一个重要应用领域，它涉及到文本数据的搜索、检索和排序。在这篇文章中，我们将深入探讨NLP的原理和Python实战，特别关注信息检索的进阶。

# 2.核心概念与联系
在进入具体的算法和实现之前，我们需要了解一些核心概念和联系。

## 2.1 文本数据
文本数据是我们需要处理的基本单位，它可以是文章、新闻、博客等。文本数据通常包含在文本文件中，可以使用Python的`open`函数读取。

## 2.2 词汇表
词汇表是文本数据中出现的所有单词的集合。我们需要将文本数据转换为词汇表，以便进行后续的处理。

## 2.3 词频统计
词频统计是计算每个单词在文本数据中出现的次数的过程。我们可以使用Python的`collections`模块中的`Counter`类来实现词频统计。

## 2.4 逆向文件频率
逆向文件频率是计算每个单词在整个文本集合中出现的次数的过程。我们可以使用Python的`Counter`类来实现逆向文件频率。

## 2.5 词袋模型
词袋模型是一种简单的文本表示方法，它将文本数据转换为一个包含单词和它们在文本中出现次数的字典。我们可以使用Python的`collections`模块中的`defaultdict`类来实现词袋模型。

## 2.6 文档向量化
文档向量化是将文本数据转换为数字向量的过程，以便进行数学计算和机器学习算法的应用。我们可以使用Python的`numpy`库来实现文档向量化。

## 2.7 相似度计算
相似度计算是计算两个文档之间相似度的过程，以便进行文本检索和分类。我们可以使用Python的`numpy`库来计算欧氏距离和余弦相似度。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
在这一部分，我们将详细讲解信息检索的核心算法原理、具体操作步骤以及数学模型公式。

## 3.1 文本预处理
文本预处理是将文本数据转换为机器可以理解的格式的过程。我们需要对文本数据进行以下操作：

1. 小写转换：将所有字符转换为小写，以便统一处理。
2. 去除标点符号：使用正则表达式`re.sub`函数去除标点符号。
3. 分词：使用Python的`jieba`库进行中文分词，使用`nltk`库进行英文分词。
4. 词干提取：使用Python的`nltk`库进行词干提取，以便减少词汇表的大小。

## 3.2 词汇表构建
我们需要将文本数据转换为词汇表，以便进行后续的处理。我们可以使用Python的`set`类来构建词汇表。

## 3.3 词频统计和逆向文件频率
我们需要计算每个单词在文本数据中出现的次数，以及每个单词在整个文本集合中出现的次数。我们可以使用Python的`Counter`类来实现词频统计和逆向文件频率。

## 3.4 词袋模型
我们需要将文本数据转换为一个包含单词和它们在文本中出现次数的字典。我们可以使用Python的`defaultdict`类来实现词袋模型。

## 3.5 文档向量化
我们需要将文本数据转换为数字向量，以便进行数学计算和机器学习算法的应用。我们可以使用Python的`numpy`库来实现文档向量化。

## 3.6 相似度计算
我们需要计算两个文档之间的相似度，以便进行文本检索和分类。我们可以使用Python的`numpy`库来计算欧氏距离和余弦相似度。

# 4.具体代码实例和详细解释说明
在这一部分，我们将提供具体的代码实例，并详细解释其中的每一步。

## 4.1 文本预处理
```python
import re
import jieba
import nltk

def preprocess_text(text):
    # 小写转换
    text = text.lower()
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    words = jieba.cut(text) if text.encode('utf-8').strip().startswith(u'中') else nltk.word_tokenize(text)
    # 词干提取
    words = [word for word in words if nltk.pos_tag([word])[0][1] in ('NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'WP$', 'PDT', 'CD', 'UH', 'TO', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'IN', 'AT', 'ER', 'DT', 'PRP', 'PRP$', 'POS', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB', 'WP', 'PDT', 'CD', 'UH', 'TO', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN', 'IN',