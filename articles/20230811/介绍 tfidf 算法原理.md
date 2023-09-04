
作者：禅与计算机程序设计艺术                    

# 1.简介
         

TF-IDF（Term Frequency - Inverse Document Frequency）是一种统计方法，用来评估一份文档中的词语对于一个检索词的重要程度。TF-IDF 是一组用于信息检索与文本挖掘的算法。它主要是通过反映一篇文档中某个词语的频率和其在整个文档集中的位置距离来评估该词语的价值。TF-IDF 的权重可以衡量一个单词对文档整体的重要性，而非某个特定的文档。
# 2.基本概念
## （1）TF(term frequency)
在给定一个文档 D 中，词项 t 在文档 D 中的出现次数，称作词项 t 的词频 (term frequency)，记做 TF(t,D)。这里的词项 t 可以是一个完整的词或是一个短语。
例如，文档 "the quick brown fox jumps over the lazy dog" 中，词项 "quick" 和词项 "jumps" 的词频分别为 1 次和 2 次，而词项 "fox", "over", "lazy", "dog" 的词频都为 0 次。
## （2）IDF(inverse document frequency)
如果某个词项 t 至少出现在 N 个文档中，则称该词项具有很高的信息量 (information content)，或者具有独特性 (uniqueness)。那么，IDF(t) 就是衡量一个词项 t 对文档集 D 的文档重要性的方法。显然，当某个词项只出现在其中一小部分文档中时，它的 IDF 就较低；但如果某个词项经常出现在文档集中，则它的 IDF 就可能很高。IDF 是根据文档集 D 中的文档数量 N 来计算的。

IDF(t) = log(N / |{d ∈ D: t ∈ d}|)

其中，D 为文档集，|⋅| 表示集合 A 的大小。上式表示词项 t 的逆向文档频率，它表示某词项 t 在 D 中出现的概率。

## （3）TF-IDF
为了将 TF 和 IDF 结合起来进行评估，可以定义 TF-IDF 权重。TF-IDF 权重等于 TF 乘以 IDF，即：

Tfidf(t,D) = TF(t,D) * IDF(t)

tfidf 代表 Term Frequency-Inverse Document Frequency，即“词频-逆向文档频率”。这个名称的由来是为了区别于其他词项提升的方法。

# 3.算法原理及具体操作步骤
TF-IDF 算法工作流程如下图所示：


## （1）数据准备
首先需要准备文本文档，并用自然语言处理工具清洗、标注、分词等预处理。然后利用倒排索引构建文档集合 D，其中每条文档记录了每个文档中的所有词项及其词频。倒排索引指的是索引形式为“词→文档”的索引表格，记录了每一个词出现过的文档列表。这里假设 D 共有 n 个文档，每个文档有一个唯一标识符 docId，并且已按顺序排序。对于每个文档 i，在倒排索引中，记录了词项及其对应的文档列表。比如，倒排索引中若存在词项 w1 在文档 i 中出现，则在相应行的文档列表中添加 i。具体地，D[i]=[w1,w2,...,wn]，其中 di=(wi,fi)，wi 表示第 j 个词项，fi 表示第 j 个词项在第 i 个文档中出现的次数。另外，对于每一个词项 wi，通过计算公式 |{d ∈ D: wi ∈ d}| ，计算出它的逆文档频率 idf(wi)。

## （2）计算 TF
接下来，算法计算文档集 D 中各个词项的 TF 值，即词项出现次数除以文档总词数。具体地，第 i 个文档的 TF 值为：

tf(wi,docId)=log10(1+fi)/∑log10(1+fj), where fi=frequency of word wi in document i and ∑fj is total number of words in all documents that contain word wi.

公式中 log10(x) 表示以 10 为底的 x 的对数。

## （3）计算 IDF
算法计算每个词项的 IDF 值，即反映词项在文档集中出现的次数的负对数值。具体地，对每一个文档 i，计算出包含该文档的文档个数 N：

N = number of documents containing word wi

最后，计算每个词项的 IDF 值为：

idf(wi)=log10(N/|{d ∈ D: wi ∈ d}| )

## （4）计算 TF-IDF
算法将 TF 和 IDF 结合得到 TF-IDF 值，即：

Tfidf(wi,docId) = tf(wi,docId) * idf(wi)

# 4.代码实现
```python
import math

def get_word_freq(docs):
"""
获取词频字典

:param docs: list, 每个元素为文档的词列表 [[word11, word12...],[word21, word22...]....]
:return: dict, 词频字典 {"word": freq}
"""
word_dict = {}
for doc in docs:
# 遍历每篇文档的所有词
for word in set(doc):
if word not in word_dict:
word_dict[word] = 0
# 词频统计
word_dict[word] += 1
return word_dict


def get_total_words(word_dict):
"""
获取所有文档的总词数

:param word_dict: dict, 词频字典 {"word": freq}
:return: int, 所有文档的总词数
"""
sum_num = 0
for value in word_dict.values():
sum_num += value
return sum_num


def get_doc_count(word, word_dict, docs):
"""
获取词在文档集中出现的文档个数

:param word: str, 要查询的词
:param word_dict: dict, 词频字典 {"word": freq}
:param docs: list, 每个元素为文档的词列表 [[word11, word12...],[word21, word22...]....]
:return: int, 词在文档集中出现的文档个数
"""
count = 0
for i in range(len(docs)):
if word in docs[i]:
count += 1
return count


def compute_tf(word, word_dict, docs, index):
"""
计算词项的 TF 值

:param word: str, 要查询的词
:param word_dict: dict, 词频字典 {"word": freq}
:param docs: list, 每个元素为文档的词列表 [[word11, word12...],[word21, word22...]....]
:param index: dict, 倒排索引 {"word":[docid]}
:return: float, 词项的 TF 值
"""
# 获取词频
fi = word_dict[word]
# 获取文档的总词数
total_words = get_total_words(word_dict)
# 获取词在文档集中出现的文档个数
N = len([True for _ in range(len(index)) if word in index[_]])
# 计算 TF 值
tf = math.log((fi + 1) / (total_words + N))
return tf


def compute_idf(word, word_dict, docs, index):
"""
计算词项的 IDF 值

:param word: str, 要查询的词
:param word_dict: dict, 词频字典 {"word": freq}
:param docs: list, 每个元素为文档的词列表 [[word11, word12...],[word21, word22...]....]
:param index: dict, 倒排索引 {"word":[docid]}
:return: float, 词项的 IDF 值
"""
# 获取词在文档集中出现的文档个数
N = get_doc_count(word, word_dict, docs)
# 计算 IDF 值
idf = math.log(float(len(docs)) / max(1, N))
return idf


def compute_tfidf(word, word_dict, docs, index):
"""
计算词项的 TF-IDF 值

:param word: str, 要查询的词
:param word_dict: dict, 词频字典 {"word": freq}
:param docs: list, 每个元素为文档的词列表 [[word11, word12...],[word21, word22...]....]
:param index: dict, 倒排索引 {"word":[docid]}
:return: float, 词项的 TF-IDF 值
"""
# 获取词项的 TF 值
tf = compute_tf(word, word_dict, docs, index)
# 获取词项的 IDF 值
idf = compute_idf(word, word_dict, docs, index)
# 计算 TF-IDF 值
tfidf = tf * idf
return tfidf


def build_inverted_index(docs):
"""
构建倒排索引

:param docs: list, 每个元素为文档的词列表 [[word11, word12...],[word21, word22...]....]
:return: dict, 倒排索引 {"word":[docid]}
"""
inverted_index = {}
for i in range(len(docs)):
# 遍历文档的所有词
for word in set(docs[i]):
# 如果该词还没有加入倒排索引，则初始化一个列表
if word not in inverted_index:
inverted_index[word] = []
# 将当前文档的编号加入到列表
inverted_index[word].append(i)
return inverted_index


if __name__ == '__main__':
# 假设有四篇文档，第一篇文档 "the quick brown fox jumps over the lazy dog"，第二篇文档 "apple pie is delicious"，第三篇文档 "the cat chased the mouse and caught it"，第四篇文档 "dog park restaurants are popular nowaday"
# 分割文档为词列表
docs = [["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"], ["apple", "pie", "is", "delicious"],
["the", "cat", "chased", "the", "mouse", "and", "caught", "it"], ["dog", "park", "restaurants", "are", "popular", "nowaday"]]

# 获取词频字典
word_dict = get_word_freq(docs)
print("词频字典:\n", word_dict)

# 获取所有文档的总词数
total_words = get_total_words(word_dict)
print("所有文档的总词数:", total_words)

# 获取词在文档集中出现的文档个数
count = get_doc_count("cat", word_dict, docs)
print("\"cat\" 在文档集中出现的文档个数:", count)

# 构建倒排索引
inverted_index = build_inverted_index(docs)
print("倒排索引:\n", inverted_index)

# 计算词项的 TF-IDF 值
tfidf = compute_tfidf("cat", word_dict, docs, inverted_index)
print("\"cat\" 的 TF-IDF 值:", tfidf)
```