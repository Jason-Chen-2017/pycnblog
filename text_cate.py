import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 读取文本数据
data = pd.read_csv('text_data.csv')

# 提取文本内容
documents = data['text_content'].tolist()

# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# 使用K-means算法进行聚类分析
k = 5  # 聚类数量
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# 获取每个文本的聚类标签
cluster_labels = kmeans.labels_

# 将聚类结果添加到数据框
data['cluster'] = cluster_labels

# 打印每个聚类的文本内容
for i in range(k):
    cluster_data = data[data['cluster'] == i]
    print(f"Cluster {i}:")
    for text in cluster_data['text_content']:
        print(text)
    print('\n')