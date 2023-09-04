
作者：禅与计算机程序设计艺术                    

# 1.简介
         

        中文分词（Chinese Word Segmentation，CWS）是中文信息处理的一个基础子任务，它用于将连续的文字分割成独立的词或短语。在自然语言理解、信息检索、文本挖掘、机器翻译等诸多领域都扮演着重要角色。本篇文章通过对主流中文分词工具的介绍，梳理其基本原理、使用方式和特点，力求为读者提供一个较为系统的学习参考。
        # 2.基本概念术语
        - 词：中文中具有完整意义的最小单位。如“中国”、“美国”等。
        - 字：汉字、英文字母等组成的最小单位。如“中”、“国”、“大”等。
        - 句子：由若干个词组成的完整陈述句。如“我爱中国，中国是一个美丽的国家！”。
        - 单词：由一个或多个字组成的一个词。如“中”，“国”，“一”等。
        - 文档：由若干个句子组成的一个完整的文本。如新闻报道、电影脚本、学术论文等。
        - 分词器：一种基于规则的程序或模型，用来把一个未分词的文档自动分割成句子、词、字等短小的元素。
        
        # 3.常用中文分词工具概览
        ## 3.1 Jieba分词器（Python版）
        
        Jieba分词器是Python语言编写的开源项目，遵循MIT协议，能够有效地实现中文分词。其主要功能包括：
        1. 搜索引擎模式：适合搜索引擎构建倒排索引时使用的分词方法，可以准确识别出文本中的关键词。
        2. 全模式：试图将句子中所有的可能成词组合进行切分，试图从中提取出最有价值的片段。它的缺点是速度慢，但是返回结果精确度高。
        3. 精确模式：会将中文分词给做到很精确，但是速度慢，不一定适合所有场合。
        4. HMM标注：使用HMM模型对词语后面的词符号进行标注，从而保证词语之间的边界。
        5. 字典及自定义词典：支持用户自己扩展字典，使得分词效果更加精确。
        6. 用户词典优先级高于默认词典，用户可以将一些生僻词汇加入自己的字典中，防止被切分。
        7. 支持繁体分词。
        
        安装：
        
           pip install jieba
        
        使用：
        
            >>> import jieba
            >>> sentence = "这是一些中文语句"
            >>> words = jieba.lcut(sentence)
            >>> print("/".join(words))
            
            是/此/些/中/文/句/子
        
        示例：
        ```python
        import jieba

        sentences = [
            '李小福是创新办主任也是云计算方面专家；何凯娟则是大家喜欢的企业法规 architect',
            '因特尔前副总裁史蒂夫·乔布斯因健康原因辞去CEO职位，接替他的是格林尼治天文台科学家吉姆·克拉克。',
            '如何快速准确地拍摄一张照片？这里有一个简单的教程',
            '简单说一下微信的构架和工作原理',
            '研究生命起源，得出结论 evolution'
        ]

        for sent in sentences:
            seg_list = jieba.cut(sent, cut_all=False)  # 默认是精确模式
            output = '/'.join(seg_list)
            print(' '.join(output).encode('utf-8'))
        ```
        
        ## 3.2 pkuseg分词器（C++版）
        
        pkuseg分词器是清华大学开发的基于DAG结构的中文分词工具，支持两种模式，可以同时进行精确分词和模式分词。pkuseg支持繁体中文分词。
        
        安装：
        
           pip install pkuseg
        
        使用：
        
            >>> import pkuseg
            >>> seg = pkuseg.pkuseg() # 以默认配置初始化实例
            >>> text = "世界上只有一种真正的英雄主义,那就是认真正直，不计得失。"
            >>> res = seg.cut(text)
            >>> print('/'.join(res))

            世界/上/有/只/一/种/真正/的/英雄主义,/，/那/里/就/唯/一/的/真正/的/英雄主义,/那/才/是/真正/的/英雄主义,/，/不/计/得失/。/。
        
        示例：
        ```python
        from pkuseg import pkuseg

        segmentor = pkuseg()   # 初始化实例

        texts = ['这是一个例子',
                '今天天气不错，应该开心']

        for line in texts:
          seg_list = segmentor.cut(line)
          result = "/ ".join(seg_list)
          print("{}\t{}".format(result.strip(), len(result)))
        ```
        
        ## 3.3 THULAC分词器（C++版）
        
        THULAC分词器是一个开源的中文分词工具，使用HMM和词库实现，速度快准确率高。THULAC能识别并正确切分以下类别的中文字符：

        1. 中文字符；
        2. 阿拉伯数字；
        3. 空格、制表符、换行符；
        4. 标点符号；
        5. HTML标记；
        6. 英文字符、英文词汇、非中文词汇；
        7. 未登录词。
        
        安装：
        
           brew install thulac
         
         或下载源码编译安装：
           
           git clone https://github.com/thunlp/THULAC.git
           cd THULAC && make && sudo make install
         
        使用：
        
            >>> import thulac
            >>> lac = thulac.thulac(seg_only=True) # 只进行分词，不进行词性标注
            >>> text = "俄罗斯联邦储蓄银行决定于1月1日起暂停支付欧元区和瑞士法郎的存款利息。"
            >>> res = lac.cut(text)
            >>> print('/'.join(res))

            俄罗斯联邦储蓄银行/决定/于/1月1日/起/暂停支付欧元区和瑞士法郎的存款利息/。/。
        
        示例：
        ```python
        from thulac import thulac

        tagger = thulac(seg_only=True)    # 只进行分词，不进行词性标注

        texts = ['这是一个例子',
                 '今天天气不错，应该开心']

        for line in texts:
          seg_list = tagger.cut(line)
          result = "/ ".join(seg_list)
          print("{}\t{}".format(result.strip(), len(result)))
        ```
        
        ## 3.4 Stanford分词器（Java版）
        
        Stanford分词器是一个java语言编写的中文分词工具，在Corenlp、CRF4j、Maxent等工具的基础上开发，能够实现高准确率的分词，且提供了API接口。stanford分词器还附带了词性标注模块。
        
        安装：
        
           brew install java
           cd /usr/local/opt/apache-maven@3.6/libexec
          ./bin/mvn dependency:get -Dartifact=edu.stanford.nlp:stanford-segmenter:3.9.1
           cp ~/.m2/repository/edu/stanford/nlp/stanford-segmenter/3.9.1/stanford-segmenter-3.9.1.jar ~/stanford-segmenter.jar
        
        使用：
        
            >>> from nltk.tokenize import StanfordTokenizer
            >>> st = StanfordTokenizer('../stanford-segmenter.jar') # 设置分词器路径
            >>> text = "俄罗斯联邦储蓄银行决定于1月1日起暂停支付欧元区和瑞士法郎的存款利息。"
            >>> res = st.tokenize(text)
            >>> print(' '.join(res))

            俄罗斯 联邦 储蓄 银行 决定 于 一月 一日 起 暂停 支付 欧元区 和 瑞士 法郎 的 存款 利息 。
        
        示例：
        ```python
        from nltk.tokenize import StanfordTokenizer

        tokenizer = StanfordTokenizer('/Users/didi/Documents/stanford-segmenter.jar')      # 设置分词器路径

        texts = ['这是一个例子',
                 '今天天气不错，应该开心']

        for line in texts:
          tokens = tokenizer.tokenize(line)
          result = " ".join(tokens)
          print("{}\t{}".format(result.strip(), len(result)))
        ```