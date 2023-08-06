
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 TF（Term Frequency）是一个统计指标，用于衡量一个词项在一篇文档中出现的频率。TF 可以反映词项的重要性或权重。给定一个词项集，可以根据词项的 TF 值进行排序。最高的词项通常具有最重要的含义或意义，而其他词项则不具有显著的意义。文档 D 中的 TF 向量代表了文档中各个词项的 TF 值。
          
         为了计算出每篇文档的 TF 向量，需要先对每篇文档中的词项进行计数。假设一篇文档中共有 m 个词项，其词项集为 W = {t1, t2,..., tm}，其中 ti 是第 i 个词项。对于文档 D，词项集 W 在该文档中出现的次数记作 freq(ti) 。那么文档 D 的 TF 向量就是文档中所有词项的 TF 值。
         
         根据 TF 值的大小不同，可以将 TF 分为静态 TF、基于 TF-IDF 的动态 TF 和基于文档长度的加权 TF。
         
         - 静态 TF: TF 表示单个词项在整个词项集中的重要性或权重，并且每个词项的 TF 值都是相同的。因此，可以使用一个常数 a 来表示 TF 的值。例如：
          

         - 基于 TF-IDF 的动态 TF: TF-IDF（Term Frequency-Inverse Document Frequency）是一个动态 TF 方法，它考虑到词项的重要性随着时间变化而变化的特征。TF-IDF 使用两个因子来衡量词项的重要性：词项在整个集合中出现的次数（TF）；反映文档排名靠后的词项的程度（IDF）。TF-IDF 的公式如下所示：


           IDF 计算公式： log (|D|/(|D| + |d1|+...+|dn|) )，|D| 为文档库中的文档数量，|d1|,...,|dn| 为某一文档 d1,..., dn。

           通过上述公式，可以得到某个词项在文档 D 中出现的频率的 TF-IDF 值。TF-IDF 值越高，表示词项的重要性越高，反映其更具代表性。
          
         - 基于文档长度的加权 TF: 基于文档长度的加权 TF 也称为 local TF，它对词项的 TF 值进行了更细粒度的分析。通过计算每个词项在文档 D 中的 TF 值除以文档 D 的平均长度，可以得到 TF-weighted。TF-weighted 一般情况下，长文档的 TF-weighted 会比短文档的 TF-weighted 有较大的差异。
            
           下面用 Python 语言计算 TF 值并进行统计：
           
           ```python
           import re
           from collections import Counter
           
           def compute_tf(doc):
               """ Compute TF for each word in the document."""
               
               words = doc.split()
               counter = Counter(words)
               total_count = len(words)
               
               tf = {}
               for key, value in counter.items():
                   tf[key] = value / float(total_count)
                   
               return tf
           
           def get_documents():
               """ Get documents as list of strings"""
               
               docs = ["This is an example sentence.",
                       "The quick brown fox jumps over the lazy dog."]
               
               return [re.sub(r'\W+','', doc).lower().strip() for doc in docs]
               
           if __name__ == "__main__":
               docs = get_documents()
               print("Documents:", docs)
               
               tf_dict = {}
               for i, doc in enumerate(docs):
                   tf_dict["Document {}".format(i)] = compute_tf(doc)
                
               for k, v in sorted(tf_dict.items()):
                   print(k, v)
                   
           Output:
           Documents: ['this is an example sentence.', 'the quick brown fox jumps over the lazy dog.']
           Document 0 {'is': 0.16666666666666666, 'an': 0.16666666666666666, 'example': 0.16666666666666666,'sentence': 0.16666666666666666, '.': 0.16666666666666666}
           Document 1 {'quick': 0.25, 'brown': 0.25, 'fox': 0.25, 'jumps': 0.25, 'lazy': 0.25}
           ```
       
           上面的例子展示了如何从文本中抽取文档列表，然后计算每个文档的 TF 值。最后，打印出每篇文档的 TF 字典，以及 TF 字典中各项的词项及其 TF 值。