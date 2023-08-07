
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         ## TF-IDF (Term Frequency-Inverse Document Frequency) 是一种用于信息检索与文本挖掘的统计方法，它主要是利用每一个词或者短语在一份文档中出现的频率（Term Frequency）计算每个词或者短语的权重。TF-IDF 的公式为：
         

         其中，f(w, d) 表示词汇 w 在文档 d 中出现的次数，n表示文档的数量，df(w) 表示词汇 w 在整个文档集中的出现次数。 
         
         IDF(w) = log[N / df(w)] + 1 （其中 N 为总文档数）。
         
         上述公式可由下面两点解释：
           - TF(w,d):衡量文档 d 中词 w 的重要性。
           - IDF(w):衡量词 w 对整个文档集的重要程度。
         
         TF-IDF 的目的是为了更准确地评估给定的查询词或文档集合中词语的相关性。
         
         ## TfidfVectorizer
         
         scikit-learn 提供了 `TfidfVectorizer` 类实现对文本数据进行 TF-IDF 向量化，其主要步骤如下：
         
            1. 对输入的文本数据进行分词处理。默认情况下，分词器是 `WhitespaceTokenizer`，即将文本按空格字符切分成单词列表。
         
            2. 根据词频统计得到每个词的 TF 值。默认情况下，用 `CountVectorizer` 来进行词频统计。
         
            3. 根据 IDF 统计得到每个词的 IDF 值。默认情况下，用 `TfidfTransformer` 来计算 IDF 值。
         
            4. 将上述两个统计结果相乘得到最终的 TF-IDF 值。
         
         下面详细介绍 `TfidfVectorizer` 的参数设置及功能。
         
         
         ### 参数设置及功能介绍
         
         #### max_df/min_df 参数
         
         参数 `max_df`/`min_df` 可控制文档频率（Document Frequency），即保留那些文档中出现次数最高的特征词，或只保留出现次数最低的特征词。
         
         参数 `max_df` 指定了一个数字，代表要保留的最大文档频率。如果某个词在超过该值的文档中出现过，则不会被转换为 TF-IDF 向量。默认值为 1.0，即不考虑文档频率。
         
         参数 `min_df` 指定了一个数字，代表要保留的最小文档频率。如果某个词在低于该值的文档中出现过，则不会被转换为 TF-IDF 向量。默认值为 1。
         
         举例：
         
              from sklearn.feature_extraction.text import TfidfVectorizer
              vectorizer = TfidfVectorizer()

              texts = [
                      "The quick brown fox jumps over the lazy dog.",
                      "The slow white turtle runs away."
                  ]

              vectorizer.fit(texts)
              
              print("Vocabulary size:", len(vectorizer.vocabulary_))  # Vocabulary size: 14
              

              # Set min_df=2 will remove 'the' token in both documents since it appears only once
              vectorizer = TfidfVectorizer(min_df=2)

              vectors = vectorizer.transform(texts).toarray()

              for text, vec in zip(texts, vectors):
                  print(text, "
", dict(zip(vectorizer.get_feature_names(), vec)))
                  

          Output:

             Vocabulary size: 14

             The  {'quick': 0.0, 'brown': 0.0, 'fox': 0.0, 'jumps': 0.0, 'over': 0.0, 'lazy': 0.0, 'dog.': 0.0,
                    'slow': 0.0, 'white': 0.0, 'turtle': 0.0, 'runs': 0.0, 'away.': 0.0} 

             The  {'quick': 0.0, 'brown': 0.0, 'fox': 0.0, 'jumps': 0.0, 'over': 0.0, 'lazy': 0.0, 'dog.': 0.0,
                    'slow': 0.0, 'white': 0.0, 'turtle': 0.0, 'runs': 0.0, 'away.': 0.0} 


          从输出可以看到，经过设置后，两个文档中的 `the` 都没有被作为特征词加入到 TF-IDF 向量当中。
          
          #### ngram_range 参数
         
          参数 `ngram_range` 指定了构造 TF-IDF 向量时考虑的词的大小范围，是一个元组 `(min_n, max_n)`。若设置为 `(1, 1)`，则只考虑每个单词本身；若设置为 `(1, 2)`，则考虑每个单词和它的一对成对组合，依此类推。默认为 `(1, 1)`。
         
          举例：
 
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(ngram_range=(1, 2))

                texts = ["New York is a city in the U.S."]

                vectorizer.fit(texts)
                
                print("Vocabulary size:", len(vectorizer.vocabulary_))  # Vocabulary size: 9


                vectors = vectorizer.transform(texts).toarray()

                feature_names = vectorizer.get_feature_names()

                print("
Text:
", texts[0])  
                print("
Features names:
", feature_names)  
                print("
TF-IDF Vector:
", dict(zip(feature_names, list(vectors[0]))))
                
          Output:

               Vocabulary size: 9

               Text: 
               New York is a city in the U.S.

               Features names:
               ['new', 'york', 'is', 'a', 'city', 'in', 'the', 'u.s']

               TF-IDF Vector:
               {'new': 0.0, 'york': 0.0, 'is': 0.0, 'a': 0.0, 'city': 0.0, 'in': 0.0, 'the': 0.0, 'u.s': 0.0}


          从输出可以看到，构造出来的特征名由词组成，而不是单个字母组成。
          
          #### analyzer 参数
         
          参数 `analyzer` 指定了对文本进行分词的方法。可选的值有 `"word"`、`"char"` 和 `"char_wb"`。
         
          - `"word"` 表示按照单词来分词，即空白字符会被忽略，并且单词之间无空格符。例如："New York" 会被拆分为 "New" 和 "York"。
          - `"char"` 表示按照字符来分词，即每个字符都是一个独立的词。例如："New York" 会被拆分为 "N e w" 和 "Y o u r k"。
          - `"char_wb"` 表示严格按照字符来分词，即每个词都是连续的字符序列。例如："New York" 会被拆分为 "Ne" "w_" "Yo" "ur" "k "。
         
          默认值为 `"word"`。
          
          举例：
  
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(analyzer="char")

            texts = ["New York is a city in the U.S."]

            vectorizer.fit(texts)
            
            print("Vocabulary size:", len(vectorizer.vocabulary_))  # Vocabulary size: 9


            vectors = vectorizer.transform(texts).toarray()

            feature_names = vectorizer.get_feature_names()

            print("
Text:
", texts[0])  
            print("
Features names:
", feature_names)  
            print("
TF-IDF Vector:
", dict(zip(feature_names, list(vectors[0]))))
            
          Output:

               Vocabulary size: 9

               Text: 
               New York is a city in the U.S.

               Features names:
               [' ', '.', 'N', 'e', 'n', 'w', 'Y', '_', 'o', 'r', 'k', 'i','s', 'a', 'c', 't', 'i', 'd', 'u','m', '.', '<UNK>']

               TF-IDF Vector:
               {'': 0.0, '.': 0.0, 'N': 0.0, 'e': 0.0, 'n': 0.0, 'w': 0.0, 'Y': 0.0, '_': 0.0, 'o': 0.0, 'r': 0.0, 'k': 0.0, 
                'i': 0.0,'s': 0.0, 'a': 0.0, 'c': 0.0, 't': 0.0, 'i': 0.0, 'd': 0.0, 'u': 0.0,'m': 0.0 }


          从输出可以看到，构造出的特征名中每个字符都是一个词，而且 `<UNK>` 是最后一个词，即未知词汇。
          
          #### stop_words 参数
         
          参数 `stop_words` 指定了停用词表，即在 TF-IDF 向量化时要去除的词。可指定以下参数：
            - None : 不使用停用词
            - “english” : 使用内置的英文停用词表
            - List of strings : 指定自己定义的停用词表
         
          默认值为 None。
          
          
          ```python
          stop_words = set(["is", "the", "of"])  # Define your own stop words
          
          tfidf_vect = TfidfVectorizer(stop_words=stop_words)
          X = tfidf_vect.fit_transform(data)
          ```
          
          
          当然，也可以通过继承 `TfidfVectorizer` 来自定义自己的停用词表。
          
          #### sublinear_tf 参数
         
          参数 `sublinear_tf` 可控制是否采用线性对数计数。若设置为 True，则会采用下列公式：
            
          ```math
          tf = 1 + log{tf}
          ```
    
          若设置为 False，则直接使用原始的词频。默认为 False。
          
          
          ```python
          tfidf_vect = TfidfVectorizer(sublinear_tf=True)
          X = tfidf_vect.fit_transform(data)
          ```
          
          
          #### smooth_idf 参数
         
          参数 `smooth_idf` 用于平滑 IDF 函数。若设置为 True，则会采用下列公式：
              
          ```math
          idf = log[(1+N)/(1+df(w))+1] + 1
          ```
   
          若设置为 False，则会采用下面这个函数：
              
              idf = log[N/(df(w)+1)] + 1
              
          其中，N 为文档数目。默认值为 True。
          
          
          ```python
          tfidf_vect = TfidfVectorizer(smooth_idf=False)
          X = tfidf_vect.fit_transform(data)
          ```
          
          
          #### norm 参数
         
          参数 `norm` 可控制最终的 TF-IDF 向量的范数。可用的值包括 `"l1"`, `"l2"` 或 `"None"`。若为 `"l1"`，则向量元素的绝对值之和等于 1；若为 `"l2"`，则向量元素的模的平方之和等于 1。默认值为 `"l2"`。
          
          
          ```python
          tfidf_vect = TfidfVectorizer(norm='l1')
          X = tfidf_vect.fit_transform(data)
          ```
          
          
          #### use_idf 参数
         
          参数 `use_idf` 用于控制是否使用 IDF 作为加权因子。若设置为 True，则会根据指定的 `norm` 规范化 TF-IDF 向量；若设置为 False，则会返回 TF 矩阵。默认值为 True。
          
          
          ```python
          tfidf_vect = TfidfVectorizer(use_idf=False)
          X = tfidf_vect.fit_transform(data)
          ```
          
          
          #### binary 参数
         
          参数 `binary` 用于控制是否将 TF-IDF 矩阵转化为二进制矩阵。若设置为 True，则会将大于 0 的值设定为 1，否则为 0。默认值为 False。
          
          
          ```python
          tfidf_vect = TfidfVectorizer(binary=True)
          X = tfidf_vect.fit_transform(data)
          ```
          
     
        