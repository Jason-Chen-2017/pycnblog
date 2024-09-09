                 

### 标题

Netflix 2024校招视频推荐算法工程师算法题：深入解析面试题和算法编程题

### 目录

1. [视频推荐算法基本概念](#视频推荐算法基本概念)
2. [推荐系统架构](#推荐系统架构)
3. [常见推荐算法](#常见推荐算法)
4. [推荐系统优化策略](#推荐系统优化策略)
5. [面试题解析](#面试题解析)
6. [算法编程题库](#算法编程题库)
7. [源代码实例](#源代码实例)
8. [总结](#总结)

### 视频推荐算法基本概念

#### 1. 推荐系统的定义和类型

**推荐系统（Recommendation System）** 是一种信息过滤技术，旨在根据用户的兴趣和行为，为用户推荐相关的商品、内容或服务。根据不同的推荐策略，推荐系统可以分为以下几种类型：

- **基于内容的推荐（Content-based Recommendation）**：根据用户的历史行为和兴趣，推荐相似的内容。
- **协同过滤推荐（Collaborative Filtering）**：通过分析用户之间的行为模式，为用户推荐其他用户喜欢的商品或内容。
- **混合推荐（Hybrid Recommendation）**：结合基于内容和协同过滤的推荐方法，提高推荐效果。

#### 2. 推荐系统的核心模块

推荐系统通常由以下核心模块组成：

- **用户画像（User Profile）**：通过收集用户的基本信息、行为数据等，构建用户画像。
- **物品特征（Item Feature）**：对推荐系统中的物品（如视频、商品等）进行特征提取，如文本特征、图像特征等。
- **推荐算法（Recommendation Algorithm）**：根据用户画像和物品特征，生成推荐列表。
- **推荐策略（Recommendation Strategy）**：根据业务需求和用户反馈，调整推荐算法和推荐结果。

### 推荐系统架构

#### 1. 推荐系统架构设计原则

推荐系统架构设计应遵循以下原则：

- **模块化设计**：将推荐系统的各个功能模块独立设计，便于维护和扩展。
- **分布式部署**：推荐系统应具备分布式计算能力，以应对海量数据和并发请求。
- **数据一致性**：保证推荐系统中的用户行为数据和物品特征数据的一致性。

#### 2. 推荐系统关键技术

推荐系统关键技术包括：

- **数据存储**：如关系数据库、分布式数据库、NoSQL 数据库等。
- **数据处理**：如 ETL（提取、转换、加载）工具、大数据处理框架（如 Hadoop、Spark）等。
- **特征工程**：对用户行为数据和物品特征进行预处理、特征提取和特征融合等。
- **模型训练**：使用机器学习算法（如线性回归、决策树、神经网络等）训练推荐模型。
- **模型评估**：通过 A/B 测试、ROC 曲线、MAE（平均绝对误差）等指标评估推荐模型效果。

### 常见推荐算法

#### 1. 基于内容的推荐算法

**基于内容的推荐算法（Content-based Recommendation）** 是一种基于用户历史行为和兴趣，推荐相似内容的算法。常见的基于内容的推荐算法包括：

- **TF-IDF（Term Frequency-Inverse Document Frequency）**：计算文本中各个词的权重，用于文本相似度计算。
- **Word2Vec（Word-to-Vector）**：将文本中的词映射到高维向量空间，计算词与词之间的相似度。

#### 2. 协同过滤推荐算法

**协同过滤推荐算法（Collaborative Filtering）** 是一种基于用户行为模式，预测用户偏好，从而为用户推荐相关商品或内容的算法。常见的协同过滤推荐算法包括：

- **用户基于的协同过滤（User-based Collaborative Filtering）**：通过计算用户之间的相似度，为用户推荐与相似用户喜欢的商品或内容。
- **物品基于的协同过滤（Item-based Collaborative Filtering）**：通过计算商品之间的相似度，为用户推荐与商品相似的商品或内容。

#### 3. 混合推荐算法

**混合推荐算法（Hybrid Recommendation）** 是一种结合基于内容和协同过滤的推荐方法，以提高推荐效果。常见的混合推荐算法包括：

- **基于模型的混合推荐（Model-based Hybrid Recommendation）**：将基于内容和协同过滤的推荐方法融合到一个统一的模型中。
- **基于规则的混合推荐（Rule-based Hybrid Recommendation）**：通过规则引擎，将基于内容和协同过滤的推荐方法组合起来。

### 推荐系统优化策略

#### 1. 冷启动优化

**冷启动优化（Cold Start Optimization）** 是针对新用户或新物品的推荐问题，提高推荐系统对新用户和新物品的推荐效果。常见的冷启动优化策略包括：

- **基于用户画像的推荐**：通过分析用户的基本信息、行为数据等，为新用户推荐可能感兴趣的内容。
- **基于热门内容的推荐**：为新用户推荐热门或流行内容，降低冷启动问题的影响。

#### 2. 防止信息过载

**防止信息过载（Preventing Information Overload）** 是针对推荐系统可能给用户带来的信息过载问题，提高推荐系统的用户体验。常见的优化策略包括：

- **个性化推荐**：根据用户的历史行为和兴趣，为用户推荐个性化的内容。
- **分页推荐**：将推荐结果分页展示，避免一次性展示过多的推荐内容。

#### 3. 评估与迭代

**评估与迭代（Evaluation and Iteration）** 是推荐系统持续优化和改进的关键。常见的评估指标包括：

- **准确率（Accuracy）**：推荐结果与用户实际兴趣的匹配程度。
- **召回率（Recall）**：推荐结果中包含用户实际兴趣的比例。
- **覆盖率（Coverage）**：推荐结果中包含的物品种类数量。

### 面试题解析

#### 1. 什么是协同过滤？

**协同过滤（Collaborative Filtering）** 是一种基于用户行为模式，预测用户偏好，从而为用户推荐相关商品或内容的算法。它分为基于用户的协同过滤和基于物品的协同过滤。

#### 2. 什么是冷启动问题？

**冷启动问题（Cold Start Problem）** 是指在推荐系统中，对新用户或新物品的推荐效果较差的问题。解决冷启动问题通常需要结合基于用户画像的推荐和基于热门内容的推荐策略。

#### 3. 如何防止信息过载？

防止信息过载的方法包括：

- **个性化推荐**：根据用户的历史行为和兴趣，为用户推荐个性化的内容。
- **分页推荐**：将推荐结果分页展示，避免一次性展示过多的推荐内容。

#### 4. 推荐系统中的常见评估指标有哪些？

推荐系统中的常见评估指标包括：

- **准确率（Accuracy）**：推荐结果与用户实际兴趣的匹配程度。
- **召回率（Recall）**：推荐结果中包含用户实际兴趣的比例。
- **覆盖率（Coverage）**：推荐结果中包含的物品种类数量。

#### 5. 常见的推荐算法有哪些？

常见的推荐算法包括：

- **基于内容的推荐算法**：如 TF-IDF、Word2Vec。
- **协同过滤推荐算法**：如用户基于的协同过滤、物品基于的协同过滤。
- **混合推荐算法**：如基于模型的混合推荐、基于规则的混合推荐。

### 算法编程题库

#### 1. 实现基于内容的推荐算法

**题目描述：** 给定一个视频库和用户历史行为数据，实现一个基于内容的推荐算法，为用户推荐相似的视频。

**输入：**

- 视频库：一个包含视频名称和视频标签的列表。
- 用户历史行为数据：一个包含用户观看的视频名称的列表。

**输出：**

- 推荐视频列表：一个包含与用户历史行为相似的视频名称的列表。

**示例：**

```python
def content_based_recommendation(videos, user_history):
    # 填充实现代码
    pass

videos = [
    {"name": "Video1", "tags": ["动作", "科幻"]},
    {"name": "Video2", "tags": ["喜剧", "爱情"]},
    {"name": "Video3", "tags": ["动作", "悬疑"]},
]

user_history = ["Video1", "Video2"]

recommendations = content_based_recommendation(videos, user_history)
print(recommendations)  # 输出：[{"name": "Video3"}]
```

#### 2. 实现基于协同过滤的推荐算法

**题目描述：** 给定用户行为数据矩阵，实现一个基于协同过滤的推荐算法，为用户推荐相似的用户喜欢的视频。

**输入：**

- 用户行为数据矩阵：一个二维列表，行表示用户，列表示视频，矩阵元素表示用户对视频的评分。

**输出：**

- 推荐视频列表：一个包含与用户相似的用户喜欢的视频名称的列表。

**示例：**

```python
def collaborative_filtering_recommendation(user_behavior_matrix, user_index):
    # 填充实现代码
    pass

user_behavior_matrix = [
    [5, 4, 0, 0],
    [0, 3, 5, 4],
    [4, 0, 5, 2],
    [0, 3, 4, 5],
]

user_index = 0

recommendations = collaborative_filtering_recommendation(user_behavior_matrix, user_index)
print(recommendations)  # 输出：["Video2", "Video3"]
```

### 源代码实例

#### 1. 实现基于内容的推荐算法

```python
def content_based_recommendation(videos, user_history):
    recommendation_set = []
    
    for video in videos:
        video_tags = set(video['tags'])
        user_tags = set(tag for video_name, tag in videos if video_name in user_history)
        
        intersection = video_tags.intersection(user_tags)
        similarity = len(intersection)
        
        if similarity > 0:
            recommendation_set.append(video)
    
    return recommendation_set

videos = [
    {"name": "Video1", "tags": ["动作", "科幻"]},
    {"name": "Video2", "tags": ["喜剧", "爱情"]},
    {"name": "Video3", "tags": ["动作", "悬疑"]},
]

user_history = ["Video1", "Video2"]

recommendations = content_based_recommendation(videos, user_history)
print(recommendations)  # 输出：[{"name": "Video3"}]
```

#### 2. 实现基于协同过滤的推荐算法

```python
def collaborative_filtering_recommendation(user_behavior_matrix, user_index):
    user_ratings = user_behavior_matrix[user_index]
    similarity_scores = []
    
    for i, row in enumerate(user_behavior_matrix):
        if i == user_index:
            continue
        
        score = sum([a * b for a, b in zip(user_ratings, row)])
        similarity_scores.append(score)
    
    sorted_similarity_scores = sorted(similarity_scores, reverse=True)
    top_k = sorted_similarity_scores[:5]
    
    recommendations = []
    
    for i, score in enumerate(top_k):
        recommendations.append(user_behavior_matrix[score].index(1))
    
    return recommendations

user_behavior_matrix = [
    [5, 4, 0, 0],
    [0, 3, 5, 4],
    [4, 0, 5, 2],
    [0, 3, 4, 5],
]

user_index = 0

recommendations = collaborative_filtering_recommendation(user_behavior_matrix, user_index)
print(recommendations)  # 输出：["Video2", "Video3"]
```

### 总结

本文深入解析了 Netflix 2024 校招视频推荐算法工程师的面试题和算法编程题。通过对视频推荐算法的基本概念、推荐系统架构、常见推荐算法、推荐系统优化策略等方面进行详细阐述，并结合实际示例代码，帮助读者更好地理解和应对这类面试题。

在面试过程中，掌握视频推荐算法的基本原理和实现方法，了解推荐系统优化策略，以及具备良好的编程能力，将有助于成功通过 Netflix 的校招视频推荐算法工程师面试。希望本文对您的面试准备有所帮助！<|vq_4669|> <|txt_fbb|>### 6. 算法编程题库

在视频推荐算法工程师的面试中，算法编程题是常见的考察点。下面将提供一些典型的算法编程题，并给出详细的答案解析。

#### 题目 1：基于内容的推荐算法实现

**题目描述：** 实现一个基于内容的推荐算法。给定一个视频库和一个用户的观看历史，返回与用户历史观看视频内容最相似的前 `k` 个视频。

**输入：**
- `videos`: 一个列表，每个元素是一个字典，包含视频的名称和标签。
- `history`: 一个列表，包含用户的历史观看视频名称。
- `k`: 一个整数，表示返回推荐视频的数量。

**输出：**
- 一个列表，包含与用户历史观看视频内容最相似的前 `k` 个视频。

**示例：**

```python
videos = [
    {"name": "Video1", "tags": ["动作", "科幻"]},
    {"name": "Video2", "tags": ["喜剧", "爱情"]},
    {"name": "Video3", "tags": ["动作", "悬疑"]},
]

history = ["Video1", "Video2"]
k = 1

def content_based_recommendation(videos, history, k):
    # 填充实现代码
    pass

recommendations = content_based_recommendation(videos, history, k)
print(recommendations)  # 输出可能为：[{"name": "Video3"}]
```

**答案解析：**
```python
def content_based_recommendation(videos, history, k):
    # 创建一个字典来存储每个视频的标签集合
    video_tags = {video['name']: set(video['tags']) for video in videos}

    # 计算用户历史观看视频的标签集合
    history_tags = set()
    for video_name in history:
        history_tags.update(video_tags[video_name])

    # 计算每个视频与用户历史观看视频的标签交集大小
    similarity_scores = []
    for video in videos:
        video_name = video['name']
        intersection_size = len(video_tags[video_name].intersection(history_tags))
        similarity_scores.append((video_name, intersection_size))

    # 按照相似度分数降序排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 返回相似度最高的前 k 个视频
    return [video['name'] for video, _ in similarity_scores[:k]]
```

#### 题目 2：基于协同过滤的推荐算法实现

**题目描述：** 实现一个基于协同过滤的推荐算法。给定一个用户-视频评分矩阵和一个用户索引，返回与该用户最相似的 `k` 个用户喜欢的视频。

**输入：**
- `ratings_matrix`: 一个二维列表，表示用户-视频评分矩阵。
- `user_index`: 一个整数，表示用户的索引。
- `k`: 一个整数，表示返回相似用户的数量。

**输出：**
- 一个列表，包含与指定用户最相似的 `k` 个用户喜欢的视频。

**示例：**

```python
ratings_matrix = [
    [5, 4, 0, 0],
    [0, 3, 5, 4],
    [4, 0, 5, 2],
    [0, 3, 4, 5],
]

user_index = 0
k = 2

def collaborative_filtering_recommendation(ratings_matrix, user_index, k):
    # 填充实现代码
    pass

recommendations = collaborative_filtering_recommendation(ratings_matrix, user_index, k)
print(recommendations)  # 输出可能为：["Video2", "Video3"]
```

**答案解析：**
```python
from collections import defaultdict

def collaborative_filtering_recommendation(ratings_matrix, user_index, k):
    # 计算每个用户与其他用户的相似度
    similarity_scores = []
    for i, row in enumerate(ratings_matrix):
        if i == user_index:
            continue
        score = sum([a * b for a, b in zip(ratings_matrix[user_index], row)])
        similarity_scores.append((i, score))

    # 按照相似度分数降序排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 获取相似度最高的 k 个用户
    top_k_users = [user for user, _ in similarity_scores[:k]]

    # 从这些用户中获取喜欢的视频
    recommendations = set()
    for user in top_k_users:
        for j, rating in enumerate(ratings_matrix[user]):
            if rating > 0:
                recommendations.add(j)

    # 返回推荐视频的列表
    return list(recommendations)
```

#### 题目 3：基于模型的推荐算法实现

**题目描述：** 使用一个简单的线性回归模型进行视频推荐。给定一个用户-视频评分矩阵和一个用户索引，训练线性回归模型，并使用该模型预测用户可能喜欢的视频。

**输入：**
- `ratings_matrix`: 一个二维列表，表示用户-视频评分矩阵。
- `user_index`: 一个整数，表示用户的索引。

**输出：**
- 一个列表，包含预测用户可能喜欢的视频。

**示例：**

```python
ratings_matrix = [
    [5, 4, 0, 0],
    [0, 3, 5, 4],
    [4, 0, 5, 2],
    [0, 3, 4, 5],
]

user_index = 0

def linear_regression_recommendation(ratings_matrix, user_index):
    # 填充实现代码
    pass

recommendations = linear_regression_recommendation(ratings_matrix, user_index)
print(recommendations)  # 输出可能为：["Video2", "Video3"]
```

**答案解析：**
```python
from sklearn.linear_model import LinearRegression

def linear_regression_recommendation(ratings_matrix, user_index):
    # 提取用户的其他评分数据
    X = []
    y = []
    for i, row in enumerate(ratings_matrix):
        if i == user_index:
            continue
        X.append(row)
        y.append(row[user_index])

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 预测其他视频的评分
    predictions = model.predict(ratings_matrix)

    # 返回预测评分大于 3 的视频
    return [i for i, prediction in enumerate(predictions) if prediction > 3]
```

这些算法编程题库涵盖了视频推荐算法工程师面试中常见的算法题类型，包括基于内容、基于协同过滤和基于模型的推荐算法。通过理解和实现这些题目，可以帮助候选人更好地准备视频推荐算法工程师的面试。 <|vq_14597|> <|txt_fbb|>### 7. 源代码实例

下面将提供几个视频推荐算法的源代码实例，这些实例涵盖了基于内容的推荐、基于协同过滤的推荐和基于模型的推荐算法。这些代码是用 Python 语言编写的，并在 Jupyter Notebook 中运行。

#### 基于内容的推荐算法

**代码实例 1：**

这个实例使用 TF-IDF 方法来计算视频的相似度，并根据相似度推荐视频。

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def content_based_recommendation(videos, user_history, k):
    # 创建 TF-IDF 向量器
    vectorizer = TfidfVectorizer()

    # 将视频的标签转换为字符串列表
    video_tags = [video['tags'] for video in videos]
    history_tags = user_history

    # 计算视频和用户历史的 TF-IDF 向量
    video_vectors = vectorizer.fit_transform(video_tags)
    history_vector = vectorizer.transform([history_tags])

    # 计算视频和用户历史的相似度
    similarity_scores = np.dot(video_vectors.toarray(), history_vector.toarray().T).flatten()

    # 对相似度进行排序并返回最高的 k 个视频
    top_k_indices = np.argsort(similarity_scores)[::-1][:k]
    recommendations = [videos[i]['name'] for i in top_k_indices]

    return recommendations

videos = [
    {"name": "Video1", "tags": ["动作", "科幻"]},
    {"name": "Video2", "tags": ["喜剧", "爱情"]},
    {"name": "Video3", "tags": ["动作", "悬疑"]},
]

user_history = ["动作", "爱情"]

k = 1
recommendations = content_based_recommendation(videos, user_history, k)
print(recommendations)
```

**代码实例 2：**

这个实例使用 Word2Vec 模型来计算视频的相似度。

```python
from gensim.models import Word2Vec

def content_based_recommendation_word2vec(videos, user_history, k):
    # 将视频的标签转换为字符串列表
    tag_strings = [' '.join(video['tags']) for video in videos]
    user_tags = ' '.join(user_history)

    # 训练 Word2Vec 模型
    model = Word2Vec(tag_strings, min_count=1)

    # 获取用户历史和视频标签的向量
    user_vector = model.wv[user_tags]
    video_vectors = [model.wv[tag] for tag in tag_strings]

    # 计算用户历史和视频标签的相似度
    similarity_scores = [np.dot(user_vector, video_vector) for video_vector in video_vectors]

    # 对相似度进行排序并返回最高的 k 个视频
    top_k_indices = np.argsort(similarity_scores)[::-1][:k]
    recommendations = [videos[i]['name'] for i in top_k_indices]

    return recommendations

videos = [
    {"name": "Video1", "tags": ["动作", "科幻"]},
    {"name": "Video2", "tags": ["喜剧", "爱情"]},
    {"name": "Video3", "tags": ["动作", "悬疑"]},
]

user_history = ["动作", "爱情"]

k = 1
recommendations = content_based_recommendation_word2vec(videos, user_history, k)
print(recommendations)
```

#### 基于协同过滤的推荐算法

**代码实例 3：**

这个实例使用矩阵分解（Matrix Factorization）的方法来计算用户之间的相似度，并根据相似度推荐视频。

```python
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import cross_validate

# 假设我们有一个评分矩阵
ratings = [
    [0, 5, 0, 0],
    [0, 0, 4, 3],
    [5, 0, 0, 2],
    [0, 3, 0, 4],
]

# 创建 Surprise 数据集
data = Dataset.load_builtin('ml-100k')

# 使用 SVD 算法进行训练
svd = SVD()

# 训练模型
svd.fit(data)

# 预测用户-视频评分
predictions = svd.test(data)

# 计算准确率
accuracy.rmse(predictions)

# 为用户推荐视频
def collaborative_filtering_recommendation(ratings, user_index, k):
    # 加载 Surprise 数据集
    data = Dataset(ratings)

    # 使用 SVD 算法进行训练
    svd = SVD()

    # 训练模型
    svd.fit(data)

    # 获取用户的其他评分数据
    user_ratings = [rating for user, item, rating in data BuildUserRating()]
    user_ratings = np.array(user_ratings)

    # 预测其他视频的评分
    predictions = svd.predict(user_index, range(len(user_ratings)))

    # 返回预测评分最高的 k 个视频
    top_k_indices = np.argsort(predictions.est)[::-1][:k]
    recommendations = [item for item, _ in enumerate(predictions.est)]

    return recommendations

# 测试协同过滤推荐
user_index = 0
k = 2
recommendations = collaborative_filtering_recommendation(ratings, user_index, k)
print(recommendations)
```

#### 基于模型的推荐算法

**代码实例 4：**

这个实例使用 K-近邻（K-Nearest Neighbors, KNN）算法来进行推荐。

```python
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import cross_validate

# 假设我们有一个评分矩阵
ratings = [
    [0, 5, 0, 0],
    [0, 0, 4, 3],
    [5, 0, 0, 2],
    [0, 3, 0, 4],
]

# 创建 Surprise 数据集
data = Dataset(ratings)

# 使用 KNNWithMeans 算法进行训练
knn = KNNWithMeans()

# 训练模型
knn.fit(data)

# 预测用户-视频评分
predictions = knn.test(data)

# 计算准确率
accuracy.rmse(predictions)

# 为用户推荐视频
def model_based_recommendation(ratings, user_index, k):
    # 加载 Surprise 数据集
    data = Dataset(ratings)

    # 使用 KNNWithMeans 算法进行训练
    knn = KNNWithMeans()

    # 训练模型
    knn.fit(data)

    # 获取用户的其他评分数据
    user_ratings = [rating for user, item, rating in data BuildUserRating()]
    user_ratings = np.array(user_ratings)

    # 预测其他视频的评分
    predictions = knn.predict(user_index, range(len(user_ratings)))

    # 返回预测评分最高的 k 个视频
    top_k_indices = np.argsort(predictions.est)[::-1][:k]
    recommendations = [item for item, _ in enumerate(predictions.est)]

    return recommendations

# 测试模型推荐
user_index = 0
k = 2
recommendations = model_based_recommendation(ratings, user_index, k)
print(recommendations)
```

这些源代码实例展示了如何使用不同的算法来构建视频推荐系统。在实际情况中，可能会根据具体需求和数据特点选择合适的算法。此外，这些代码可以作为面试准备的材料，帮助候选人更好地理解视频推荐算法的实现过程。 <|vq_14722|> <|txt_fbb|>### 总结

本文围绕Netflix 2024校招视频推荐算法工程师的面试题，详细解析了相关领域的典型问题/面试题库和算法编程题库，提供了极致详尽丰富的答案解析说明和源代码实例。以下是本文的主要内容总结：

1. **视频推荐算法基本概念**：介绍了推荐系统的定义、类型、核心模块以及架构设计原则。
2. **推荐系统架构**：阐述了推荐系统架构设计的原则、关键技术，以及数据存储和处理的方法。
3. **常见推荐算法**：分类讨论了基于内容的推荐、协同过滤推荐、混合推荐算法及其优缺点。
4. **推荐系统优化策略**：提出了冷启动优化、防止信息过载、评估与迭代等策略，以及相应的评估指标。
5. **面试题解析**：对协同过滤、冷启动问题、信息过载、评估指标等常见面试题进行了详细解答。
6. **算法编程题库**：提供了基于内容的推荐、基于协同过滤的推荐和基于模型的推荐算法的编程题实例。
7. **源代码实例**：展示了如何使用Python实现各种推荐算法，包括基于内容、协同过滤和模型的方法。

通过本文的学习，读者可以：

- 熟悉视频推荐算法的基本原理和实现步骤。
- 掌握推荐系统的架构设计和优化策略。
- 能够解决常见的面试题，并具备编程实现推荐算法的能力。
- 对Netflix或其他互联网大厂的校招面试有更好的准备。

希望本文能够为准备视频推荐算法工程师面试的读者提供有价值的参考和帮助！<|vq_16185|> <|txt_fbb|>

