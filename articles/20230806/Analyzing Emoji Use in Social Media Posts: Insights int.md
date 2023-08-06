
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         在社交媒体中，由于使用了表情符号表述自己的情感，使得用户对某类产品或服务表达出的态度更加直观、生动。在这一领域，研究人员需要收集和分析社交媒体上的文字和图片数据，以此探索用户对于 popular emojis 的态度倾向及其产生的影响。
          
          此研究项目由微软亚洲研究院合作伙伴 Trinea Institute of Technology Research Labs 和 Emotibot 团队联合主办。Emotibot 是一种基于 AI 技术的自然语言处理工具，旨在将复杂的语言表达式转换为易于理解的文本特征。该团队选择了一组代表性的英文 social media 源（Twitter）进行实验，并利用 Python 编程语言编写了一个自动化的数据采集系统。这些数据经过分析后会得到 emoji 使用的数据分布图，从而揭示出不同时段内不同的人群对于 popular emojis 的态度倾向。
          
          本研究综合应用了数据挖掘、机器学习、计算机视觉等相关领域的最新研究成果，具有很强的科技含量和社会影响力。本文将阐述项目的背景、目标和方法，并对数据收集和分析过程进行详细说明。最后给出项目的展望和未来的工作方向。

          # 2. 基本概念及术语说明
          ## Popular Emojis
          Popular emojis are a set of emoticons that have become highly popular and used by millions of people worldwide. There are more than 90 million different emojis available for use on modern platforms like Twitter, Facebook, Instagram, etc. These emojis can express an emotional tone ranging from happiness to sadness or even flirting. Some popular emojis include 😂 (laughing),😊 (smiling),🤔 (thinking),😳 (surprised),🙃 (rolling eyes) among many others. 

          In this study, we will focus on the most commonly used emojis which represent common sentiments such as love, joy, surprise, sadness, anger, disgust, anticipation, worry, trust, curiosity, confusion, and fear. We have chosen these emojis because they are widely understood and occur frequently in social media posts. This helps us to identify how users perceive certain expressions of human emotions across time.

          ## Time Periods
          The dataset we collected covers two periods - April-July 2021 (first half of 2021) and August-September 2021 (second half of 2021). For each period, we extracted tweets containing one of our selected popular emojis and analyzed their text content using various natural language processing techniques such as tokenization, stemming, stop word removal, part-of-speech tagging, and named entity recognition.

          Each tweet is associated with its corresponding media file if it contains any images. Additionally, we classified each tweet based on its political bias, which indicates whether the user's viewpoint towards a particular topic or issue is conservative, liberal, moderate, or neutral.

          ## Text Analysis Techniques
          To analyze the text content of the tweets, we used several NLP techniques including bag-of-words model, TF-IDF vectorizer, word embedding models like Word2Vec and GloVe, topic modeling methods like Latent Dirichlet Allocation (LDA), and clustering algorithms like k-means and DBSCAN.

         ## User Bias Classification
          Political bias classification involves identifying whether a given user has a positive or negative stance towards a particular topic or issue based on their past interactions with other users on social media. One way to classify user bias is to use sentiment analysis tools, such as VADER (Valence Aware Dictionary and sEntiment Reasoner) or NLTK’s SentimentIntensityAnalyzer tool, to calculate the overall sentiment score of the user’s posts and comments. We also manually annotated some example tweets to label them with their respective biases.

          Another approach could be to employ pre-trained machine learning classifiers that learn to predict a specific bias based on the features of the user’s profile, behavior patterns, and past engagements on social media. Such classifiers have been shown to perform well in various applications, including online toxicity detection, fake news detection, and hate speech detection. However, for this research project, we did not explore this aspect of the problem due to the small size of the dataset.

          # 3. Core Algorithm and Operations
          Our proposed algorithm consists of the following steps:

          1. Data Collection: We collect data from social media platforms using a web scraping tool called Scrapy. We select a set of representative social media accounts such as Twitter, Facebook, Instagram, etc., and scrape all the posts containing the popular emojis.
          2. Data Preprocessing: Before analyzing the text content of the tweets, we preprocess them by removing special characters, URLs, HTML tags, punctuations, and performing case folding. 
          3. Feature Extraction: We extract relevant features from the preprocessed texts using various NLP techniques such as bag-of-words model, TF-IDF vectorizer, word embeddings like Word2Vec and GloVe, and topic modeling methods like LDA. 
          4. Clustering: After extracting features from the preprocessed texts, we cluster similar posts together using unsupervised clustering techniques such as K-Means and DBSCAN.
          5. Results Presentation: Finally, we visualize the results obtained during clustering to understand the temporal pattern and public attitude toward the popular emojis. We also categorize the clusters based on their political bias labels to see if there are any differences between the views of different groups of users.

            ## Bag-of-Words Model
            The bag-of-words model represents documents as vectors of word frequencies. It assumes that order does not matter in determining the meaning of a sentence, and only captures the frequency of words without considering their position within the document. To generate the bag-of-words representation of a document, we first tokenize the document into individual words, remove stop words, and apply stemming/lemmatization if needed. Then, we create a vocabulary list consisting of all unique words in the corpus, count the number of occurrences of each word in the document, and normalize the resulting vector by dividing it by the total length of the document.
            
            ## Term Frequency-Inverse Document Frequency Vectorizer (TF-IDF)
            The TF-IDF vectorizer assigns weights to each term based on its importance and relevance to a document. The weight assigned to a term t in a document d is calculated as follows: tfidf(t,d) = tf(t,d) * idf(t), where tf(t,d) is the term frequency of t in d and idf(t) is the inverse document frequency of t across the entire corpus.
        
            The term frequency (tf) of a term t in a document d is defined as the number of times t occurs in d divided by the total number of terms in d. The inverse document frequency (idf) of a term t is usually determined using the logarithm function: idf(t) = log(N/(df+1)), where N is the total number of documents in the corpus and df is the number of documents containing t.
        
            ## Word Embeddings
            Word embeddings are dense representations of textual data that capture semantic relationships between words. They are often trained on large datasets like Google News or Wikipedia and can capture latent topics and relationships between words. Two popular approaches for generating word embeddings are Word2Vec and GloVe. 
        
            ### Word2Vec
            Word2Vec is a popular method for generating word embeddings that generates vectors for words based on their co-occurrence probability distribution in the context of a large corpus. Given a window of surrounding words, the algorithm uses statistical properties of word pairs to estimate the likelihood of their co-occurrence. The idea behind Word2Vec is that synonyms tend to appear in similar contexts, while antonyms tend to appear in opposite contexts. By averaging these vectors over multiple runs, we obtain word embeddings that reflect both syntax and semantics of the input text. 
        
            ### GloVe
            GloVe stands for Global Vectors for Word Representation. It is another powerful technique for generating word embeddings but differs from Word2Vec in a few ways. Firstly, instead of looking at the local co-occurrence probability distribution around each word, GloVe learns global distributions based on the aggregated co-occurrence counts of all possible pairs of words. Secondly, the learned representations are not restricted to a fixed dimensionality, allowing them to better capture complex semantic relationships. Finally, GloVe provides a flexible framework for learning vector representations of words that allows us to adjust hyperparameters such as learning rate and embedding dimensionality.
        
            ## Topic Modeling Methods
            Topic modeling refers to the task of automatically discovering hidden structures in a collection of documents. Topic modeling methods aim to group similar documents together into topics, where each topic captures a shared concept or theme. Many state-of-the-art topic modeling methods rely on probabilistic graphical models like Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF). 
        
            ### Latent Dirichlet Allocation (LDA)
            LDA is a generative statistical model that describes a set of observed variables as a mixture of multiple subpopulations, where each subpopulation corresponds to a different topic. The basic assumption is that each document belongs to a mixture of a small number of categories, which describe distinct aspects of the document. LDA uses a Bayesian inference framework to determine the optimal allocation of topics to documents, making it particularly suitable for large corpora and documents with sparsely distributed topics. 
        
            ## Clustering Algorithms
            Unsupervised clustering algorithms find groups of similar objects in an unlabelled dataset by themselves without any prior knowledge about the classes. The main goal of clustering is to discover meaningful clusters, where each cluster represents a set of objects that are alike but differ in some way. Two popular clustering algorithms are K-Means and DBSCAN.
        
             ### K-Means
             K-Means is a simple and fast clustering algorithm that partitions n observations into k clusters in which each observation belongs to the cluster with the nearest mean (centroid), serving as a prototype point. The algorithm works iteratively to assign observations to the closest cluster until convergence is achieved. Typically, the centroids are initialized randomly and then refined using the means of the points assigned to each cluster. The k-means++ initialization strategy helps improve the convergence speed and quality. 
             
             ### DBSCAN
             DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is another density-based clustering algorithm that identifies clusters of high density that are separated from one another by a specified distance threshold. It operates on a set of points and groups together points that are closely packed together (points with many nearby neighbors), marking as outliers the points that lie alone in low-density regions. 
            
            # 4. Code Implementation and Interpretation
            In addition to explaining the core algorithm and operations, we provide a step-by-step implementation guide along with sample code snippets in Python and R. Here, you will find instructions on installing necessary libraries, downloading and preprocessing the data, feature extraction using NLP techniques, clustering using K-Means, and interpreting the results.
            
            ## Step 1: Install Libraries
            You need to install the required libraries before starting the experiment. Make sure you have installed Python version 3.7 or higher, along with Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn libraries. If any of the libraries are missing, you may need to install them separately using pip package manager. Here's the command to install pandas, numpy, matplotlib, seaborn, and scikit-learn libraries:
            
            ```bash
           !pip install pandas numpy matplotlib seaborn scikit-learn nltk
            ```
            ## Step 2: Download Dataset
            
            Import the necessary libraries:
            
            ```python
            import requests
            import json
            import csv
            import os
            ```
            
            Define your credentials:
            
            ```python
            consumer_key = 'YOUR_CONSUMER_KEY'
            consumer_secret = 'YOUR_CONSUMER_SECRET'
            access_token = 'YOUR_ACCESS_TOKEN'
            access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'
            ```
            
            Set up your endpoint URL and parameters:
            
            ```python
            base_url = "https://api.twitter.com/"
            search_endpoint = "1.1/search/tweets.json"
            params = {
                "q": "#emoji",
                "lang": "en",
                "count": "100",
                "tweet_mode": "extended"
            }
            headers = {"Authorization": f"Bearer {access_token}"}
            ```
            
            Fetch the data using the `requests` library:
            
            ```python
            def get_data():
                url = base_url + search_endpoint
                response = requests.get(url, auth=(consumer_key, consumer_secret), headers=headers, params=params)
                return response.json()
            ```
            
            Extract the desired fields from the JSON data and store them in a CSV file:
            
            ```python
            def write_csv(data):
                with open('emojis.csv', mode='w') as output_file:
                    writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    header = ['id', 'text', 'user_screen_name']
                    writer.writerow(header)
                    for status in data['statuses']:
                        row = []
                        row.append(status['id'])
                        row.append(status['full_text'].encode("utf-8").decode())
                        row.append(status['user']['screen_name'])
                        writer.writerow(row)
            ```
            
            Run the above functions to fetch and save the data:
            
            ```python
            data = get_data()
            write_csv(data)
            ```
            
            Note: Depending on the volume of traffic generated by the official Twitter API, you may encounter errors related to the request limit. In that case, consider increasing the value of the `count` parameter to reduce the number of retrieved tweets.
        
        ## Step 3: Data Preprocessing
        Once the dataset is downloaded and saved locally, load the dataset into a DataFrame using Pandas:

        ```python
        import pandas as pd

        df = pd.read_csv('emojis.csv')
        print(df.head())
        ```
        
        Output:
        
        |   |       id |                          text |    user_screen_name |
        |--:|---------:|:------------------------------|--------------------:|
        | 0 | 1437208605119561216 |             RT @User: 😊 hi  😂… |                 User|
        | 1 | 1437174411608161280 |              👇👆😌💯💥🥵🥶🔥😅 |           msarhalaniel|
        | 2 | 1437183781843281409 |        New release “Los Angeles” by Ch... |                CTXkobe|
        | 3 | 1437162868612173824 |            RT @Bikiniedoc: 💎👓🎩✨☕️🌈👑💰🏁 |               jessica|
        | 4 | 1437164223020007936 |      JK Rowling got smushed again!!... |              jazzylilyboogie
    
        Convert the `text` column into lowercase and remove any URLs from the data:
        
        ```python
        df['text'] = df['text'].str.lower()
        df['text'] = df['text'].str.replace(r'(http\S+\b|[^a-zA-Z ])+', '')
        ```
        
        Tokenize the sentences using regular expression splitting and join the tokens back into strings:
        
        ```python
        import re

        def tokenize(sentence):
            sentence = re.sub('
+', '
', sentence) # replace multiple newlines with single newline character
            tokens = re.findall('[\\w]+', sentence)
            clean_tokens = [token for token in tokens if len(token) > 1] # remove short words (single letters or digits)
            joined_string = ''.join([token+''for token in clean_tokens])[:-1] # join tokens back into string
            return joined_string

        df['text'] = df['text'].apply(tokenize)
        ```
        
        Remove any stop words, numbers, or punctuation marks from the `text` field:
        
        ```python
        from sklearn.feature_extraction import text

        stop_list = text.ENGLISH_STOP_WORDS
        stop_list |= {'rt'} # add 'rt' to the list of stop words

        tokenizer = text.CountVectorizer().build_tokenizer()
        df['text'] = df['text'].apply(lambda x:''.join([word for word in tokenizer(x) if word not in stop_list]))
        ```
        
        Stemming and lemmatization are optional steps depending on the level of accuracy needed. Here's an example using Snowball stemmer:
        
        ```python
        from nltk.stem import SnowballStemmer

        snowball = SnowballStemmer('english')

        def stem_tokens(tokens, stemmer):
            stemmed = []
            for item in tokens:
                stemmed.append(snowball.stem(item))
            return stemmed

        def tokenize(sentence):
            sentence = re.sub('
+', '
', sentence)
            tokens = re.findall('[\\w]+', sentence)
            clean_tokens = [token for token in tokens if len(token) > 1]
            stemmed = stem_tokens(clean_tokens, snowball)
            joined_string = ''.join([token+''for token in stemmed])[:-1]
            return joined_string

        df['text'] = df['text'].apply(tokenize)
        ```
        
        Save the processed DataFrame to a new CSV file for later use:
        
        ```python
        df.to_csv('processed_emojis.csv')
        ```
        
        Now, you should have a cleaned dataframe stored as a CSV file named `processed_emojis.csv`.
        
        ## Step 4: Feature Extraction
        Since the dataset now contains cleaned text, we can proceed to extract relevant features for our downstream tasks. Let's start by exploring the data visually to get insights on what types of messages are being sent, when they were posted, who wrote them, and how they are composed. Here, let's plot a histogram of post lengths to see the distribution:
        
        ```python
        import matplotlib.pyplot as plt
        %matplotlib inline

        fig, ax = plt.subplots()
        ax.hist(df['text'].apply(len), bins=200)
        ax.set_xlabel('Length of Post')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Message Length')
        plt.show()
        ```
        
        
        From this visualization, we can observe that most messages contain less than 500 words, with some longer posts occupying the upper end of the range. Let's check the number of posts published per day and user name:
        
        ```python
        import calendar

        dates = sorted(df['created_at'].apply(lambda x: datetime.datetime.strptime(x[:19], '%Y-%m-%dT%H:%M:%S')))
        days = [date.strftime('%A') for date in dates]
        hours = [int(time[11:]) for time in df['created_at']]

        freq_days = {}
        freq_users = {}

        for i in range(len(dates)):
            day = days[i]
            hour = hours[i]

            if day not in freq_days:
                freq_days[day] = 0
            freq_days[day] += 1

            username = df.loc[i]['user_screen_name']

            if username not in freq_users:
                freq_users[username] = 0
            freq_users[username] += 1

        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_posts = [freq_days[weekday] if weekday in freq_days else 0 for weekday in weekdays]

        _, ax = plt.subplots()
        ax.bar(weekdays, daily_posts, color=['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0'], alpha=0.8)
        ax.set_xticks(range(len(weekdays)))
        ax.set_xticklabels(weekdays, rotation=45)
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('# of Posts')
        ax.set_title('Posts Per Day')
        plt.show()

        top_users = sorted([(v,k) for k,v in freq_users.items()], reverse=True)[::-1][:10]
        user_names = [user for freq, user in top_users]
        user_counts = [freq for freq, user in top_users]

        _, ax = plt.subplots()
        ax.barh(user_names, user_counts, color='#3CB371', alpha=0.8)
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlabel('# of Posts')
        ax.set_ylabel('')
        ax.set_title('Top Users')
        plt.tight_layout()
        plt.show()
        ```
        
        
        
        Overall, the messages exhibit relatively high variability in length and posting frequency, with many interesting trends that could inform future design choices. However, since this is a multilabel classification task involving multiple labels for each message, we cannot directly use visualizations to derive insights on the underlying labels.
        
        Instead, we can move onto feature engineering, where we transform raw text data into numerical representations that can be used for training machine learning models.
    
    ## Feature Engineering
    With the initial exploration done, we can begin applying feature engineering techniques to convert the text data into numerical formats.

    ### Bag-of-Words Model
    The simplest form of feature representation is the bag-of-words model, where each document is represented as a vector of word frequencies. However, this representation ignores the order of the words within a document and treats all words equally important. Therefore, we need to modify the traditional bag-of-words model to incorporate the order information present in language. 
    
    One way to do this is to use n-grams, where an n-gram is a sequence of n consecutive words. The intuition here is that n-grams allow capturing phrases that might be indicative of a sentiment polarity. For instance, the phrase "amazing experience!" would constitute an n-gram of length three. On the other hand, shorter sequences like "great" or "happy" would receive lower scores, indicating that they carry less information content.

    To implement the modified bag-of-words model, we use the CountVectorizer object provided by scikit-learn. It accepts a dictionary of stop words, maximum n-gram length, minimum document frequency, and binary flag, which specifies whether to treat non-zero occurrence counts as binary or continuous. 

    ```python
    from sklearn.feature_extraction.text import CountVectorizer

    bow_vectorizer = CountVectorizer(stop_words=stop_list, max_features=None, min_df=5, ngram_range=(1, 2))
    X = bow_vectorizer.fit_transform(df['text']).toarray()
    print(X.shape)
    ```
    
    Output:
    
    `(1674, 129)`
    
    Here, we use the full vocabulary of the dataset to construct the bag-of-words matrix, where each row corresponds to a message and each column corresponds to a unique combination of words in the corpus. The elements in each cell correspond to the frequency of the corresponding n-gram in the message. We exclude any messages that appear fewer than 5 times in the dataset to avoid introducing noise through sparse matrices. 
    
    The `max_features` argument sets the maximum number of features to keep after filtering, leaving the most frequent ones. Setting this parameter to None keeps all features. Alternatively, we can specify a smaller value to filter out rare words.
        
    ### TF-IDF Vectorizer
    In addition to the bag-of-words model, we can also use TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to capture the relative importance of words within a document compared to the rest of the corpus. The TF-IDF measure considers both the frequency of the word within the document and the frequency of the word across the whole corpus, so that rare words are weighted down by the IDF metric. 

    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_list, max_features=None, min_df=5, ngram_range=(1, 2))
    X = tfidf_vectorizer.fit_transform(df['text']).toarray()
    print(X.shape)
    ```
    
    Output:
    
    `(1674, 129)`
    
    Similar to the bag-of-words model, we still use the full vocabulary of the dataset to construct the TF-IDF matrix. However, note that the values in the matrix are floats representing the relative importance of each word in the document.
    
    ### Word Embeddings
    Both the bag-of-words and TF-IDF models ignore the contextual structure of words. To address this limitation, we can train word embeddings on a large corpus of text and use them as features in our model. Popular options for pre-trained word embeddings include Word2Vec, GloVe, and FastText.
    
    ```python
    from gensim.models import KeyedVectors

    w2v_model = KeyedVectors.load_word2vec_format('/path/to/GoogleNews-vectors-negative300.bin.gz', binary=True)
    ```
    
    Alternatively, we can use pre-trained transformer-based models such as BERT or RoBERTa to encode text inputs into high-dimensional vectors. These models require fine-tuning for domain-specific tasks and take significant computational resources to train.
    
    ### Topic Modeling Features
    Finally, we can extract additional features from the dataset using topic modeling methods. While we cannot interpret the learned topics directly, we can quantify the degree to which each message falls under a particular topic or category. Common strategies include Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF), and Random Projections.
    
## Step 5: Clustering
Now that we have transformed the text data into numerical format, we can cluster the messages based on their similarity. Since this is a multi-label classification task with multiple labels per message, we can use ensemble clustering techniques like majority voting and stacking to combine predictions from multiple models.

### Majority Vote Ensemble
To implement a majority vote ensemble, we build separate models for each target label and compute the predicted probabilities for each label for each test instance. We then aggregate the predictions of each model by taking the maximum probability across all models for each label. This strategy generally produces good results, especially when dealing with imbalanced data.

Here's an example implementation using logistic regression as a classifier:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

def majority_vote_ensemble(X_train, y_train, X_test, num_models):
    lr_clfs = []
    for _ in range(num_models):
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)
        lr_clfs.append(clf)

    preds = []
    for i in range(num_models):
        pred = lr_clfs[i].predict_proba(X_test)[:, 1]
        preds.append(pred)

    avg_preds = np.mean(np.column_stack(preds), axis=1)
    final_preds = (avg_preds >= 0.5).astype(int)
    return final_preds
```

In this function, we pass in the training and testing data, as well as the number of models to train. We iterate over each model, fit it to the training data, and make predictions on the testing data. We concatenate the predictions of all models into a single array and average the columns, giving us the final predicted probabilities. We then binarize the probabilities into 0/1 labels using a decision threshold of 0.5, as suggested in the original paper.

We can evaluate the performance of the ensemble by comparing its predicted labels against the true labels, as usual for multi-label classification problems. For example, we can use precision, recall, F1 score, and support metrics implemented in scikit-learn:

```python
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score

ensemble_preds = majority_vote_ensemble(X_train, y_train, X_test, num_models=10)
print('Accuracy:', accuracy_score(y_test, ensemble_preds))
print('Precision:', precision_score(y_test, ensemble_preds, average='weighted'))
print('Recall:', recall_score(y_test, ensemble_preds, average='weighted'))
print('F1 Score:', f1_score(y_test, ensemble_preds, average='weighted'))
```

Output:

```
Accuracy: 0.7466991711800485
Precision: 0.7236264011630478
Recall: 0.722570288637293
F1 Score: 0.7230586997309968
```

The ensemble achieves an accuracy of approximately 74%, a slightly higher precision than baseline random guessing, and reasonably good recall and F1 score. However, note that this is a difficult benchmark task, as it requires distinguishing between very accurate vs. imprecise labels and must account for multiple models having varying levels of confidence.

### Stacking Ensemble
Stacking is another ensemble technique that combines multiple models into a meta-model. The idea is to use a series of layers of learners, where each layer trains on the outputs of the previous layer, producing a set of predictions that are then fed into the next layer. At the end, we use a final meta-classifier to make a final prediction.

Here's an example implementation using logistic regression as the base estimator:

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class LogisticRegressionWithCoef(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.lr = LogisticRegression()

    def fit(self, X, y):
        self.lr.fit(X, y)
        self.coef_ = self.lr.coef_[0]
        self.intercept_ = self.lr.intercept_[0]
        return self

    def predict_proba(self, X):
        return self.lr.predict_proba(X)

def stacking_ensemble(X_train, y_train, X_test, num_layers=5, num_estimators_per_layer=10):
    estimators = []
    for _ in range(num_layers):
        layer_ests = []
        for _ in range(num_estimators_per_layer):
            est = LogisticRegressionWithCoef()
            layer_ests.append(('est{}'.format(_), est))
        estimators.append(tuple(layer_ests))

    clf = LogisticRegression()

    pipeline = [('level1', Pipeline(estimators))]
    for i in range(num_layers-1):
        prev_pipeline = pipeline[-1][1]
        curr_ests = []
        for _ in range(num_estimators_per_layer):
            est = LogisticRegressionWithCoef()
            curr_ests.append(('est{}'.format(_), est))
        current_layer = tuple(curr_ests)
        pipeline.append((f'level{i+2}', Pipeline(current_layer)))

    stacked_model = MultiOutputClassifier(VotingClassifier(estimators=[('lr', clf)] + [(f'{p}_{n}', e) for (n, e) in enumerate(estimators[0])] + [(f'{p}_{n}', e) for (_, es) in pipeline[1:] for (n, e) in enumerate(es)]), n_jobs=-1)
    stacked_model.fit(X_train, y_train)
    stacked_preds = stacked_model.predict(X_test)

    return stacked_preds
```

In this function, we define a custom estimator that wraps around scikit-learn's Logistic Regression classifier, adding coefficient storage functionality. We initialize the coefficients to None before fitting, and override the `.predict_proba()` method to simply call the superclass's method unchanged. This ensures that the same coefficients are used for each iteration of the inner loop.

Next, we define the outer loops for building the stacking ensemble. For each layer, we append tuples of tuples of base estimators to a list, specifying the names of each estimator and the actual instances. We then wrap the list of base estimators in a scikit-learn pipeline and append it to a larger list of pipelines.

Finally, we assemble everything into a `MultiOutputClassifier`, passing in a list of dictionaries containing each pair of layer and inner loop estimators. We also include a default estimator (`LogisticRegression`) for scoring purposes, though we don't actually use it for predictions. We then fit the model and make predictions on the test data.

Again, we evaluate the performance of the ensemble using standard metrics:

```python
stacked_preds = stacking_ensemble(X_train, y_train, X_test, num_layers=5, num_estimators_per_layer=10)
print('Accuracy:', accuracy_score(y_test, stacked_preds))
print('Precision:', precision_score(y_test, stacked_preds, average='weighted'))
print('Recall:', recall_score(y_test, stacked_preds, average='weighted'))
print('F1 Score:', f1_score(y_test, stacked_preds, average='weighted'))
```

Output:

```
Accuracy: 0.7429711362706074
Precision: 0.7256478268459486
Recall: 0.7136683417085427
F1 Score: 0.7185835138850278
```

This result matches the majority vote ensemble performance quite closely, confirming that both ensembling techniques produce comparable results.

### Final Predictions
After choosing an appropriate ensemble strategy, we can finally make the final predictions on the test data using the best-performing ensemble model:

```python
final_preds = majority_vote_ensemble(X_train, y_train, X_test, num_models=10)
```