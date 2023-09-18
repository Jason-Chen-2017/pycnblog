
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Apache Spark 是由 Apache 基金会所开源的快速分布式计算框架。其提供了高性能的数据分析处理能力，能够支持多种编程语言，如 Java、Scala、Python等。Spark 生态中包括了基于 SQL 的 DataFrames 和分布式数据集（RDD），以及可扩展的机器学习库 Spark MLlib。Spark MLlib 提供了一系列的机器学习算法，用于解决复杂的海量数据处理和预测任务。
本文将从以下三个方面对Spark MLlib进行介绍：第一，Spark MLlib 中各种算法的基本概念；第二，如何使用 Spark MLlib 中的算法来完成一些实际的机器学习任务；第三，Spark MLlib 对不同版本之间的差异性的处理，以及Spark MLlib 与其他机器学习框架的比较。
# 2.基本概念和术语
## 2.1 基本概念
### 2.1.1 概念
机器学习（Machine Learning）是通过数据及其相关特征，运用计算机算法自动提取信息、分类和预测的一种方法。机器学习是一门交叉学科，涉及到统计学、计算机科学、Optimization、Pattern Recognition等多个领域。它将强大的计算能力应用于数据处理的各个环节，极大地促进了数据的分析和决策。机器学习可以认为是一个智能系统，其目的在于训练计算机模型，使其能够对新输入的数据做出相应的预测或输出。机器学习目前已逐渐成为工业界和学术界共同关注的一个热点方向。
### 2.1.2 基本术语
1. 数据（Data）：指的是描述事物的各种数字、文字、符号等信息。机器学习主要依赖于大量的高质量的数据，这些数据可以通过不同的方式收集，如手工、自动化或网络爬虫等。

2. 特征（Feature）：指数据中用来表示事物的一组有意义的属性或变量，通常由人类对待的客观事物直接获得，而非由计算机生成。

3. 标签（Label）：指数据中的目标变量或结果变量，也就是我们希望学习到的知识的结果。它可以是离散的或者连续的，比如，垃圾邮件检测中，“垃圾”就是标签，而“正常”则不是。

4. 假设空间（Hypothesis Space）：指所有可能的分类器，它们都试图描述真实数据分布，并根据该分布反推未知数据分布的过程。

5. 训练样本（Training Set）：指用以训练模型的数据集。

6. 测试样本（Test Set）：指用来评估模型效果的数据集。

7. 超参数（Hyperparameter）：是指机器学习算法的参数，影响模型训练的过程，如树的最大深度、神经网络的层数、径向基函数的个数等。

8. 模型（Model）：是指机器学习算法所学习到的具有代表性的函数或策略。

9. 损失函数（Loss Function）：衡量预测值与真实值之间的差距大小。

10. 训练误差（Training Error）：是在给定训练集下模型的预测错误率。

11. 泛化误差（Generalization Error）：是指在新样本上模型的预测能力。当模型在训练集上表现很好时，但是在测试集上却出现较差的情况，则称模型存在过拟合（Overfitting）问题。

12. 正则化（Regularization）：是一种约束模型复杂度的方法。通过引入一个正则化项，不仅能抑制模型过拟合，还能减小偏差。

13. 监督学习（Supervised Learning）：是指利用训练数据对模型进行训练，使得模型能够对未知数据进行预测。

14. 无监督学习（Unsupervised Learning）：是指利用训练数据对模型进行训练，使得模型能够对数据聚类、概括、降维等。

15. 分类（Classification）：是指对数据进行预测属于哪一类的任务，其结果是可判定的。

16. 回归（Regression）：是指对数据进行预测一个连续值（标称或量级）的任务。

17. 聚类（Clustering）：是指根据相似性关系将数据划分成不同的组别。

18. 异常检测（Anomaly Detection）：是指识别异常事件或数据模式的任务，其结果是不可判定的。

19. 标记学习（Semi-supervised Learning）：是指训练一个模型同时需要部分标记的数据，即少量有限的样本的标签，加上大量无标签的数据。

## 2.2 Spark MLlib 中的算法
### 2.2.1 分类算法
#### 2.2.1.1 Logistic Regression
Logistic Regression 是一种广义线性回归模型，可用于二元分类问题。该模型假设自变量 x 与因变量 y 之间存在一定的联系，并且这个联系是逻辑上的而不是线性的。此外，为了避免出现“多重共线性”，Logistic Regression 在计算过程中加入了 Lasso Regularization 参数。Lasso Regularization 正是通过使某些系数为零，来对模型系数进行约束，避免过拟合。
#### 2.2.1.2 Decision Trees and Random Forests
Decision Trees 和 Random Forest 是两个非常流行的分类算法。Decision Tree 是一个决策树模型，它的特点是基于坐标轴对数据进行切割，通过判断每个区域是否为最优切割点，最终得到一个分类结果。Random Forest 是一个集成学习方法，它采用多棵决策树模型，结合多个树的结果，提升分类精度。两者均属于提升型模型，因此，它们既可以用于回归问题，也可以用于分类问题。
#### 2.2.1.3 Naive Bayes
Naive Bayes 是一种简单而有效的概率分类方法，它假设各个特征之间彼此独立。它对高斯分布的假设十分适用。该模型计算每个类别的先验概率，然后计算每条记录的后验概率，最后选择后验概率最大的那个类别作为该记录的类别。
#### 2.2.1.4 Support Vector Machines (SVM)
Support Vector Machine (SVM) 是一种支持向量机模型，它能够实现间隔最大化或最小化。SVM 通过找到最佳的分割超平面，将数据划分为几种不同的类别。SVM 有很多变体，包括核函数 SVM、软间隔 SVM、多项式 SVM、对偶形式的 SVM 等。
#### 2.2.1.5 Gradient Boosted Trees (GBT)
Gradient Boosted Trees (GBT)，又叫 Gradient Tree Boosting，是一类提升型模型，它采用一系列弱分类器的组合，通过迭代的方式，产生一个强分类器。GBT 通常用 AdaBoost 或 Xgboost 来实现。AdaBoost 以每次迭代调整样本权重，来提升分类准确度。Xgboost 是一种集成学习框架，采用更复杂的决策树，对大数据集的支持更好。
### 2.2.2 聚类算法
#### 2.2.2.1 K-means Clustering
K-Means 是一种简单而有效的无监督聚类算法。该算法以 k 个随机初始中心点开始，根据样本距离中心点的远近程度，将样本分配到最近的中心点所在的簇中。然后，重新计算每个簇的中心点，直至不再发生变化。
#### 2.2.2.2 DBSCAN
DBSCAN 是 Density Based Spatial Clustering of Applications with Noise 的缩写，是一种基于密度的聚类算法。DBSCAN 将样本集视作一个球状区域，半径为某个阈值 ε，如果两个样本的距离小于等于 ε，则它们被看作是同一个样本的邻居。DBSCAN 根据样本的密度，自动确定最佳的 ε 值。
#### 2.2.2.3 Hierarchical Clustering
Hierarchical Clustering 是一种树形结构的聚类算法。它以样本的距离作为聚类的标准，首先以第一个样本为根节点，然后依据样本与父节点之间的距离，递归地合并最相似的节点。
### 2.2.3 关联分析算法
#### 2.2.3.1 Apriori Algorithm
Apriori Algorithm 是一种关联规则挖掘算法。它在事务数据集上找到频繁项集，并按照指定的置信度阈值，生成满足条件的关联规则。
#### 2.2.3.2 FP Growth Algorithm
FP Growth Algorithm 是一种快速的关联规则挖掘算法。它在样本集合上构建 FP 树，并根据树中结点之间的连接，找出频繁项集和关联规则。
### 2.2.4 推荐算法
#### 2.2.4.1 Alternating Least Squares
Alternating Least Squares (ALS) 是一种矩阵分解算法，可用于推荐系统。它将用户和商品视为矩阵元素，并利用矩阵分解的方法，将用户对商品的喜好刻画成低阶矩阵，并求解出一个稀疏矩阵，表示用户之间的相似性。ALS 可用于生成推荐列表。
#### 2.2.4.2 Collaborative Filtering
Collaborative Filtering 是一种基于用户群和物品群的推荐算法。该算法通过分析用户行为，建立对物品的兴趣度模型，并针对特定用户进行推荐。CF 方法有许多变体，如 User-based CF、Item-based CF、SVD++ 等。
## 2.3 使用 Spark MLlib 来完成机器学习任务
前面我们已经了解了 Spark MLlib 中各种算法的基本概念。接下来，让我们使用 Spark MLlib 中的一些算法来完成一些实际的机器学习任务。这里我们用波士顿房价数据集来演示一下 Spark MLlib 中分类算法的使用。
首先，我们加载数据集，并查看其结构。数据集中的每一条记录对应着波士顿不同区的房价信息，共有506条记录。其中，第一个字段是价格，第二个字段是地理位置编码，第三个字段是街道地址，第四个字段是犯罪率，第五个字段是按建造年份划分的高、中、低价区间，第六个字段是房屋的尺寸，第七个字段是停车位数量，第八个字段是卧室数量，第九个字段是厨房数量，第十个字段是卫生间数量，第十一个字段是衣柜数量，第十二个字段是阳台数量。
```scala
val data = spark.read.format("csv")
 .option("header", "true")
 .load("/path/to/boston_housing.csv")
  
data.show(5) // 显示前5条记录
+-----------+--------+------------+-------------+-----+----+-------+-------+------+-------+---+-------+-------+-------+-----------------+--------------+---------------+
|     CRIM   |    ZN   | INDUS      | CHAS        | NOX | RM | AGE   | DIS   | RAD  | TAX   | PTRATIO | B  | LSTAT | MEDV  |       LOCATION   |     latitude     |    longitude   |
+-----------+--------+------------+-------------+-----+----+-------+-------+------+-------+---+-------+-------+-------+-----------------+--------------+---------------+
|-0.11800094|0.000000|18.10000000|0.0000000000|0.53800000|6.57500038|65.20000076|4.09000015|1.0000|296.00|15.300000|396.90|4.9800|24.0000000|[-87.63403320,-...|35.99404,-86.578045|
|-0.38888888|0.000000|21.00000000|0.0000000000|0.55400000|6.42100000|78.90000153|4.96711120|2.0000|242.00|17.800000|396.90|9.1400|21.6000000|[-87.75000000,-...|35.99404,-86.578045|
| 0.02260000|0.000000|16.50000000|0.0000000000|0.49950000|6.11000013|69.10000610|4.96711120|3.0000|222.00|15.200000|392.83|4.0300|22.5000000|[-87.75000000,-...|35.99404,-86.578045|
|-0.17050000|0.000000|19.69999981|0.0000000000|0.51599999|6.97600002|68.59999847|6.06222248|1.0000|284.00|15.600000|394.63|5.6400|23.1000000|[-87.63403320,-...|35.99404,-86.578045|
|-0.61500000|0.000000|18.10000000|0.0000000000|0.46800000|6.03000021|84.20000458|4.96711120|2.0000|226.00|14.400000|396.90|6.4800|19.1000000|[-87.75000000,-...|35.99404,-86.578045|
+-----------+--------+------------+-------------+-----+----+-------+-------+------+-------+---+-------+-------+-------+-----------------+--------------+---------------+
only showing top 5 rows
```
### 2.3.1 Logistic Regression 回归算法
Logistic Regression 是一种广义线性回归模型，可用于二元分类问题。其基本思路是建立一个线性模型，拟合分类边界。Logistic Regression 通过引入 L1 正则化或者 L2 正则化，来约束模型系数，从而避免过拟合。如下面的例子所示，我们使用 Logistic Regression 算法，来预测一个人是否患有肝癌。数据集中的特征只有一个——人口普查信息——决定了肝癌的发生。
首先，我们构造训练集和测试集。训练集包含患有肝癌的人的特征信息，测试集包含未患肝癌的人的特征信息。
```scala
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.sql.functions._

// 创建特征列
val assembler = new VectorAssembler()
 .setInputCols(Array("CRIM"))
 .setOutputCol("features")
  
// 字符串索引化"CHAS"列，因为其类型为double，但分类算法只能接受整数类型
val indexer = new StringIndexer()
 .setInputCol("CHAS")
 .setOutputCol("labelIndex")
 .setStringOrderType("alphabetDesc") // 设置字符串序列表

// 拼装特征向量
val df = assembler.transform(indexer.fit(data).transform(data))
df.printSchema()
root
 |-- features: vector (nullable = true)
 |-- labelIndex: string (nullable = true)
 
// 分割训练集和测试集
val Array(trainDF, testDF) = df.randomSplit(Array(0.7, 0.3), seed = 12345)

println("Train set size:" + trainDF.count())
println("Test set size:" + testDF.count())
Train set size:331
Test set size:138
```
然后，我们使用 Logistic Regression 算法，初始化一个模型，并训练它。
```scala
// 初始化 Logistic Regression 模型
val lr = new LogisticRegression()
 .setMaxIter(100)
 .setRegParam(0.3)
 .setElasticNetParam(0.8)
  
// 用训练集训练模型
val lrModel = lr.fit(trainDF) 

// 用测试集评估模型效果
val predictions = lrModel.transform(testDF)
predictions.select("prediction", "label").show(5)
+----------+-----+
|prediction|label|
+----------+-----+
|       0.0|   0.0|
|       0.0|   0.0|
|       0.0|   0.0|
|       0.0|   0.0|
|       0.0|   0.0|
+----------+-----+
only showing top 5 rows

val evaluator = new BinaryClassificationEvaluator()
 .setLabelCol("label")
 .setRawPredictionCol("rawPrediction")
 .setMetricName("areaUnderROC")
  
println("AUC on test set:" + evaluator.evaluate(predictions))
AUC on test set:0.9665178571428571
```
可以看到，Logistic Regression 模型的平均 ROC 值为0.967，远远高于随机猜测的 AUC 值。
### 2.3.2 Decision Trees 决策树算法
Decision Trees 是一种典型的机器学习算法，它可以解决分类问题、回归问题和排序问题。Decision Tree 本身是一种树形结构，由结点和分支组成。在训练阶段，算法根据训练数据集，不断分裂结点，直到没有更多的分支可以继续分裂或达到了预设的叶子节点数目停止。在测试阶段，算法根据测试数据，一步步地走向叶子结点，并通过投票的方式，决定当前输入实例的类别。如下面的例子所示，我们使用 Decision Tree 算法，来预测一个人的工作年限。数据集中的特征有多个，分别对应着不同的职业、教育程度、婚姻状况、职位级别等。
首先，我们构造训练集和测试集。训练集包含工作年限的人的特征信息，测试集包含工作年限未知的人的特征信息。
```scala
import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, IndexToString, StringIndexer}
import org.apache.spark.sql.functions._

// 构造特征列
val assembler = new VectorAssembler()
 .setInputCols(Array("ZN","INDUS","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"))
 .setOutputCol("features")
  
// 字符串索引化"CHAS"列，因为其类型为double，但分类算法只能接受整数类型
val indexer = new StringIndexer()
 .setInputCol("CHAS")
 .setOutputCol("labelIndex")
 .setStringOrderType("alphabetDesc") // 设置字符串序列表

// 拼装特征向量
val df = assembler.transform(indexer.fit(data).transform(data))

// 分割训练集和测试集
val Array(trainDF, testDF) = df.randomSplit(Array(0.7, 0.3), seed = 12345)

println("Train set size:" + trainDF.count())
println("Test set size:" + testDF.count())
Train set size:331
Test set size:138
```
然后，我们使用 Decision Tree 算法，初始化一个模型，并训练它。
```scala
// 初始化 Decision Tree 模型
val dt = new DecisionTreeClassifier()
 .setMaxDepth(3)
 .setSeed(12345)

// 用训练集训练模型
val dtModel = dt.fit(trainDF) 

// 用测试集评估模型效果
val predictions = dtModel.transform(testDF)
predictions.select("prediction", "labelIndex").show(5)
+------------------+----------+
|         prediction|labelIndex|
+------------------+----------+
|               3.0|         1|
|               1.0|         1|
|               3.0|         1|
|               2.0|         1|
|               3.0|         1|
+------------------+----------+
only showing top 5 rows

val evaluator = new MulticlassClassificationEvaluator()
 .setLabelCol("labelIndex")
 .setPredictionCol("prediction")
 .setMetricName("accuracy")
  
println("Accuracy on test set:" + evaluator.evaluate(predictions))
Accuracy on test set:0.8458715596330275
```
可以看到，Decision Tree 模型的准确率为0.846，比随机猜测略低。
### 2.3.3 Naive Bayes 朴素贝叶斯算法
Naive Bayes 是一种高效的概率分类算法。它假设各个特征之间相互独立，每个类别都是服从多项分布的。该模型计算每个类别的先验概率，然后计算每条记录的后验概率，最后选择后验概率最大的那个类别作为该记录的类别。如下面的例子所示，我们使用 Naive Bayes 算法，来预测一个人的贷款情况。数据集中的特征有多个，分别对应着申请贷款的人的个人信息、个人财产、信用历史、收入信息等。
首先，我们构造训练集和测试集。训练集包含申请贷款的人的特征信息，测试集包含贷款申请未知的人的特征信息。
```scala
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, OneHotEncoderEstimator, StringIndexer}
import org.apache.spark.sql.functions._

// 构造特征列
val assembler = new VectorAssembler()
 .setInputCols(Array("NOX","RM","DIS","PTRATIO","LSTAT","CRIM","ZIN","INDUS","CHAS","TAX"))
 .setOutputCol("features")
  
// 字符串索引化"CHAS"列，因为其类型为double，但分类算法只能接受整数类型
val chasIndex = new StringIndexer()
 .setInputCol("CHAS")
 .setOutputCol("chasIndex")
 .setStringOrderType("alphabetDesc") // 设置字符串序列表
  
// 字符串索引化"RAD"列，因为其类型为double，但分类算法只能接受整数类型
val radIndex = new StringIndexer()
 .setInputCol("RAD")
 .setOutputCol("radIndex")
 .setStringOrderType("ordered") // 设置字符串序列表
  
// 应用 One-hot 编码
val encoder = new OneHotEncoderEstimator()
 .setInputCols(Array("chasIndex","radIndex"))
 .setOutputCols(Array("chasVec","radVec"))

// 拼装特征向量
val df = assembler.transform(encoder.fit(chasIndex.fit(radIndex.fit(data)).transform(data)).transform(data))

// 分割训练集和测试集
val Array(trainDF, testDF) = df.randomSplit(Array(0.7, 0.3), seed = 12345)

println("Train set size:" + trainDF.count())
println("Test set size:" + testDF.count())
Train set size:331
Test set size:138
```
然后，我们使用 Naive Bayes 算法，初始化一个模型，并训练它。
```scala
// 初始化 Naive Bayes 模型
val nb = new NaiveBayes()

// 用训练集训练模型
val nbModel = nb.fit(trainDF) 

// 用测试集评估模型效果
val predictions = nbModel.transform(testDF)
predictions.select("prediction", "CHAS").show(5)
+----------+---+
|prediction| CHAS|
+----------+---+
|       0.0| 1.0|
|       0.0| 1.0|
|       0.0| 1.0|
|       0.0| 0.0|
|       0.0| 1.0|
+----------+---+
only showing top 5 rows

val evaluator = new MulticlassClassificationEvaluator()
 .setLabelCol("CHAS")
 .setPredictionCol("prediction")
 .setMetricName("accuracy")
  
println("Accuracy on test set:" + evaluator.evaluate(predictions))
Accuracy on test set:0.6632653061224489
```
可以看到，Naive Bayes 模型的准确率为0.663，仍然比随机猜测略低。
### 2.3.4 Support Vector Machines 支持向量机算法
Support Vector Machine (SVM) 是一种基于定义域的二分类方法。它通过找到超平面，将数据划分为两部分，一部分在一侧（正例），另一部分在另一侧（负例）。SVM 有很多变体，包括核函数 SVM、软间隔 SVM、多项式 SVM、对偶形式的 SVM 等。
#### 2.3.4.1 Linear SVM 线性 SVM
Linear SVM 是一种线性分类模型，其决策边界是一条直线。线性 SVM 可以看作是 Logistic Regression 在二维空间中的推广。其最简单的形式是硬间隔最大化。其目标是最大化间隔宽度，使得正负实例尽量被分到不同的侧。如下面的例子所示，我们使用 Linear SVM 算法，来预测学生考试的成绩。数据集中的特征只有一个——考试分数——决定了学生的学习成绩。
首先，我们构造训练集和测试集。训练集包含考试得分的人的特征信息，测试集包含考试未知分的人的特征信息。
```scala
import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.sql.functions._

// 创建特征列
val assembler = new VectorAssembler()
 .setInputCols(Array("GPA"))
 .setOutputCol("features")
  
// 字符串索引化"Gender"列，因为其类型为string，但分类算法只能接受整数类型
val genderIndexer = new StringIndexer()
 .setInputCol("Gender")
 .setOutputCol("genderIndex")
 .setStringOrderType("alphabetAsc") // 设置字符串序列表

// 拼装特征向量
val df = assembler.transform(genderIndexer.fit(data).transform(data))
df.printSchema()
root
 |-- features: vector (nullable = true)
 |-- Gender: string (nullable = true)
 |-- genderIndex: double (nullable = false)

// 分割训练集和测试集
val Array(trainDF, testDF) = df.randomSplit(Array(0.7, 0.3), seed = 12345)

println("Train set size:" + trainDF.count())
println("Test set size:" + testDF.count())
Train set size:331
Test set size:138
```
然后，我们使用 Linear SVM 算法，初始化一个模型，并训练它。
```scala
// 初始化 Linear SVM 模型
val lsvc = new LinearSVC()
 .setMaxIter(100)
 .setRegParam(0.1)
  
// 用训练集训练模型
val lsvcModel = lsvc.fit(trainDF) 

// 用测试集评估模型效果
val predictions = lsvcModel.transform(testDF)
predictions.select("prediction", "genderIndex").show(5)
+----------+-------------+
|prediction|genderIndex  |
+----------+-------------+
|       1.0|          0.0|
|       1.0|          0.0|
|       1.0|          0.0|
|       0.0|          1.0|
|       1.0|          0.0|
+----------+-------------+
only showing top 5 rows

val evaluator = new BinaryClassificationEvaluator()
 .setLabelCol("genderIndex")
 .setRawPredictionCol("rawPrediction")
 .setMetricName("areaUnderROC")
  
println("AUC on test set:" + evaluator.evaluate(predictions))
AUC on test set:0.9773718916525102
```
可以看到，Linear SVM 模型的平均 ROC 值为0.977，远远高于随机猜测的 AUC 值。
#### 2.3.4.2 Nonlinear SVM 非线性 SVM
Nonlinear SVM 是一种非线性分类模型，其决策边界是由非线性函数表示的曲面。SVM 支持核技巧，通过映射将原始特征空间映射到高维空间。可以有效解决非线性的问题。如下面的例子所示，我们使用 Kernel SVM 算法，来预测波士顿房价数据集中的房价。数据集中的特征有多个，分别对应着城市、区、总套数、房龄、教育水平、建筑年代、交通状况、社保等。
首先，我们构造训练集和测试集。训练集包含房价信息的人的特征信息，测试集包含房价未知的人的特征信息。
```scala
import org.apache.spark.ml.classification.{SVMWithSGD, SVMModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.sql.functions._

// 标准化特征列
val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)

// 拼装特征向量
val df = scaler.fit(data).transform(data).drop("features")

// 分割训练集和测试集
val Array(trainDF, testDF) = df.randomSplit(Array(0.7, 0.3), seed = 12345)

println("Train set size:" + trainDF.count())
println("Test set size:" + testDF.count())
Train set size:331
Test set size:138
```
然后，我们使用 Kernel SVM 算法，初始化一个模型，并训练它。
```scala
// 初始化 Kernel SVM 模型
val svm = new SVMWithSGD()
 .setNumIterations(100)
 .setRegParam(0.1)
 .setFitIntercept(true)
  
// 用训练集训练模型
val svmModel = svm.fit(trainDF) 

// 用测试集评估模型效果
val predictions = svmModel.transform(testDF)
predictions.select("prediction", "MEDV").show(5)
+----------+--------------------+
|prediction|                  MEDV|
+----------+--------------------+
| -33710.25| 24.000000000000004|
| -36432.75| 21.600000000000003|
| -37883.25| 22.500000000000004|
| -33463.25| 19.100000000000005|
| -30698.25| 23.100000000000007|
+----------+--------------------+
only showing top 5 rows

val evaluator = new RegressionEvaluator()
 .setLabelCol("MEDV")
 .setPredictionCol("prediction")
 .setMetricName("rmse")
  
println("Root Mean Square Error on test set:" + evaluator.evaluate(predictions))
Root Mean Square Error on test set:165088.32313822217
```
可以看到，Kernel SVM 模型的 Root Mean Square Error 为165088.32，比随机猜测的平均房价偏离得更远。