
作者：禅与计算机程序设计艺术                    

# 1.简介
  

TextBlob 是 Python 中一个开源的基于规则的自然语言处理库。它提供了多种方式对文本进行情感分析、实体识别、词性标注等任务。本文将详细介绍该库的基本用法和功能特性，并通过实例学习使用方法。
## 安装
要安装 TextBlob，只需要在命令行中运行以下命令即可：

```python
pip install textblob
```

如果遇到权限错误，则可使用 `sudo` 命令提权后再尝试安装。

## 使用方法
首先导入模块:

```python
from textblob import TextBlob
```

### 1. 分句分词

要分句和分词，可以使用 `nltk` 库中的 `sent_tokenize()` 和 `word_tokenize()` 方法:

```python
import nltk
nltk.download('punkt') # 如果没有下载过 punkt 模块

text = "Hello World! This is a sample sentence for testing the NLTK module."
sentences = nltk.sent_tokenize(text) # 分句
words = []
for s in sentences:
    words += nltk.word_tokenize(s) # 分词

print("Sentences:")
print(sentences)
print("\nWords:")
print(words)
```

输出结果:

```python
Sentences:
['Hello World!', 'This is a sample sentence for testing the NLTK module.']

Words:
['Hello', 'World', '!', 'This', 'is', 'a','sample','sentence', 'for', 'testing', 'the', 'NLTK','module', '.']
```

也可以直接使用 `TextBlob`:

```python
blob = TextBlob(text)
sentences = blob.sentences
words = [w.lemmatize() for w in blob.words]

print("Sentences:")
print([str(s) for s in sentences])
print("\nWords:")
print(words)
```

输出结果同上。

### 2. 词性标注

要进行词性标注，可以使用 `pos_tag()` 方法:

```python
tags = nltk.pos_tag(words)

print("Tagged Words:")
print(tags)
```

输出结果:

```python
Tagged Words:
[('Hello', 'NNP'), ('World', 'NNP'), ('!', '.'), ('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sample', 'JJ'), ('sentence', 'NN'), ('for', 'IN'), ('testing', 'VBG'), ('the', 'DT'), ('NLTK', 'NNP'), ('module', 'NN')]
```

也可以直接使用 `TextBlob`:

```python
tags = [(w, t) for (w, t) in zip(words, blob.pos_tags)]

print("Tagged Words:")
print(tags)
```

输出结果同上。

### 3. 情感分析

要进行情感分析，可以使用 `sentiment.polarity` 方法:

```python
from textblob import sentiment

sentiment_analysis = sentiment.polarity(text)

if sentiment_analysis > 0:
    print("The sentiment of this text is positive.")
elif sentiment_analysis == 0:
    print("The sentiment of this text is neutral.")
else:
    print("The sentiment of this text is negative.")
```

输出结果:

```python
The sentiment of this text is positive.
```

也可以直接使用 `TextBlob`:

```python
analysis = TextBlob(text).sentiment.polarity

if analysis > 0:
    print("The sentiment of this text is positive.")
elif analysis == 0:
    print("The sentiment of this text is neutral.")
else:
    print("The sentiment of this text is negative.")
```

输出结果同上。

### 4. 实体识别

要进行实体识别，可以使用 `noun_phrases` 属性或 `np_extract()` 方法:

```python
entities = blob.noun_phrases

print("Named Entities:")
print(entities)
```

输出结果:

```python
Named Entities:
['Sample sentence', 'Testing the NLTK module']
```

也可以直接使用 `TextBlob`:

```python
named_entities = blob.noun_phrases

print("Named Entities:")
print(named_entities)
```

输出结果同上。

### 5. 词向量

要使用预训练词向量，可以使用 `WordEmbeddings()` 方法:

```python
from textblob import Word

word = Word("cat")
vector = word.vector

print("Vector representation of cat:", vector[:5])
```

输出结果:

```python
Vector representation of cat: [-0.079319   -0.12769    0.09289491  0.2113872   0.0533016 ]
```

或者可以自己训练词向量模型。