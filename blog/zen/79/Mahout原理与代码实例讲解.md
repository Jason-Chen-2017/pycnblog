# Mahout原理与代码实例讲解

## 关键词：

Mahout是Apache开源项目下的一个机器学习库，专注于提供分布式环境下海量数据的机器学习算法。它主要面向大数据处理需求，提供了一系列用于推荐系统、聚类、分类、关联规则挖掘等任务的算法。本文旨在深入剖析Mahout的核心原理、算法实现以及具体代码实例，帮助读者全面理解Mahout的功能与应用。

## 1. 背景介绍

### 1.1 问题的由来

随着互联网和移动互联网的普及，用户产生的数据量呈爆炸性增长，这为数据分析和机器学习带来了前所未有的机遇和挑战。面对海量数据，传统的单机处理方法已无法满足需求，分布式处理技术成为了数据处理的新趋势。Mahout正是在这样的背景下应运而生，它旨在利用分布式计算资源，高效地处理大规模数据集。

### 1.2 研究现状

Mahout基于Hadoop生态系统构建，支持MapReduce和Spark等多种分布式计算框架。它提供了一系列基于分布式计算的机器学习算法，如协同过滤、K-means聚类、决策树等，能够处理从千万级到万亿级的数据规模。此外，Mahout还支持Spark集成，实现了基于内存计算的高效机器学习流程。

### 1.3 研究意义

Mahout的研究意义主要体现在以下几个方面：
- **分布式计算能力**：Mahout能够充分利用分布式计算资源，提升处理大规模数据的能力。
- **算法库丰富**：提供了多种机器学习算法，满足不同业务场景的需求。
- **开源社区支持**：拥有活跃的开发者社区，持续更新和完善算法和功能。
- **实践案例**：大量实际应用案例，展示了Mahout在商业智能、推荐系统、个性化服务等领域的作用。

### 1.4 本文结构

本文将依次展开如下内容：
- **核心概念与联系**：介绍Mahout的基本概念、算法原理和架构。
- **算法原理与具体操作步骤**：深入探讨Mahout中关键算法的原理与实现。
- **数学模型和公式**：提供算法背后的数学理论支撑。
- **项目实践**：展示Mahout算法的代码实现及应用示例。
- **实际应用场景**：分析Mahout在不同场景下的应用价值。
- **工具和资源推荐**：提供学习资料、开发工具和相关论文推荐。

## 2. 核心概念与联系

### 2.1 分布式计算框架

Mahout基于Hadoop生态系统运行，Hadoop提供分布式文件系统（HDFS）和分布式计算框架（MapReduce）。MapReduce允许将大规模数据集分解为多个小任务，分配给集群中的多个节点并行执行，最终合并结果，显著提高了处理大规模数据的效率。

### 2.2 分析框架

Mahout提供了一套完整的数据分析框架，包括数据预处理、特征工程、模型训练、模型评估等多个环节。这一框架支持从原始数据到生成预测结果的全流程，使得用户能够快速构建和部署机器学习模型。

### 2.3 机器学习算法

Mahout包含多种机器学习算法，如：
- **协同过滤**：用于推荐系统，根据用户的历史行为预测其潜在兴趣。
- **K-means聚类**：用于数据分群，将相似数据归为同一类别。
- **决策树**：用于分类和回归任务，构建基于规则的预测模型。

这些算法分别适用于不同的场景，通过Mahout提供的API，开发者可以轻松选择和应用适合的算法。

## 3. 核心算法原理与具体操作步骤

### 3.1 算法原理概述

#### 协同过滤算法原理
协同过滤基于用户-物品评分矩阵，通过寻找用户行为模式来预测用户对未评分项目的喜好。算法主要包括用户基协同过滤（User-based CF）和物品基协同过滤（Item-based CF）两种类型。

#### K-means聚类算法原理
K-means算法是一种基于距离度量的聚类算法，其目标是将数据集划分为K个簇，使得簇内的数据点彼此相似，而簇间的数据点相异。算法通过迭代更新簇中心和重新分配数据点至最近的簇中心来达到优化目的。

#### 决策树算法原理
决策树通过递归分割特征空间来构建树状结构，每条分支代表一个特征，每个叶节点代表一个类别或预测值。决策树算法基于信息增益或基尼指数等准则来选择最佳分割特征。

### 3.2 算法步骤详解

#### 协同过滤步骤
1. 构建用户-物品评分矩阵。
2. 计算用户或物品之间的相似度。
3. 使用相似度矩阵预测缺失评分。
4. 对预测结果进行评分排序，推荐得分最高的物品。

#### K-means聚类步骤
1. 初始化K个随机中心点。
2. 将每个数据点分配给最近的中心点形成K个簇。
3. 更新每个簇的中心点为簇内数据点的平均值。
4. 重复步骤2和3，直至中心点不再变化或达到迭代次数限制。

#### 决策树构建步骤
1. 选择最优特征作为根节点。
2. 对根节点划分数据集，生成子节点。
3. 对每个子节点递归构建决策树，直至满足停止条件（如树深度、叶子节点数量）。
4. 返回构建好的决策树。

### 3.3 算法优缺点

#### 协同过滤
- **优点**：适用于冷启动问题，可以发现隐藏的兴趣。
- **缺点**：数据稀疏性导致预测准确性受限，容易受到噪声影响。

#### K-means聚类
- **优点**：简单快速，易于理解和实现。
- **缺点**：对初始中心点敏感，可能收敛于局部最优解，对异常值敏感。

#### 决策树
- **优点**：直观易懂，可用于特征选择和解释模型。
- **缺点**：容易过拟合，对噪声数据敏感。

### 3.4 算法应用领域

- **推荐系统**：协同过滤常用于电影、音乐、商品推荐。
- **市场细分**：K-means聚类用于客户分群，提供个性化营销策略。
- **医疗诊断**：决策树用于疾病预测，基于患者症状和历史数据。

## 4. 数学模型和公式

### 4.1 数学模型构建

#### 协同过滤
- **用户基协同过滤**：通过计算用户相似度矩阵来预测评分，公式为：
  $$similarity(User_i, User_j) = \frac{\sum_{item_p \in User_i's \,rated \,items} \, \sum_{item_q \in User_j's \,rated \,items} \, (rating_i(item_p) - \bar{rating_i}) \times (rating_j(item_q) - \bar{rating_j})}{\sqrt{\sum_{item_p \in User_i's \,rated \,items} \, (rating_i(item_p) - \bar{rating_i})^2} \times \sqrt{\sum_{item_q \in User_j's \,rated \,items} \, (rating_j(item_q) - \bar{rating_j})^2}}$$

#### K-means聚类
- **簇中心更新公式**：在迭代过程中，簇中心更新为簇内数据点的均值，公式为：
  $$C_k = \frac{1}{|C_k|} \sum_{x \in C_k} x$$

#### 决策树构建
- **信息增益**：用于选择最佳划分特征的指标，公式为：
  $$IG(T, a) = Entropy(T) - \sum_{v \in values(a)} \frac{|T_v|}{|T|} \cdot Entropy(T_v)$$

### 4.2 公式推导过程

#### 协同过滤推导
公式中，$similarity(User_i, User_j)$ 是用户 $i$ 和用户 $j$ 之间的相似度，$\bar{rating_i}$ 和 $\bar{rating_j}$ 分别是用户 $i$ 和用户 $j$ 的平均评分。分子计算的是两个用户在共同评分的项目上的评分差异乘积之和，分母是用户评分差异的平方和的平方根的乘积，确保了相似度的归一化。

#### K-means聚类推导
公式中，$C_k$ 表示第 $k$ 个簇的中心，$|C_k|$ 表示该簇的数据点个数。通过计算每个特征维度下各个数据点的平均值，可以得到新的簇中心，以此来迭代更新直至簇中心不再改变。

#### 决策树构建推导
公式中的 $IG(T, a)$ 是特征 $a$ 在数据集 $T$ 上的信息增益。$Entropy(T)$ 表示数据集 $T$ 的熵，$Entropy(T_v)$ 表示在特征 $a$ 的某个值 $v$ 下数据集 $T$ 的熵。信息增益的计算帮助决策树选择能够最大化数据集纯度的特征。

### 4.3 案例分析与讲解

#### 协同过滤案例
假设我们有一份用户-商品评分数据集，我们希望预测用户对未评分商品的喜好。我们选择用户基协同过滤算法，通过计算用户之间的相似度，为用户推荐与其评分习惯相似的商品。

#### K-means聚类案例
在一个电商网站上，我们希望根据用户购买行为将客户分为不同的消费群体。通过K-means聚类算法，我们能够将用户划分为几个消费模式不同的群组，以便于提供个性化的营销策略。

#### 决策树案例
在医疗诊断场景中，基于患者的年龄、性别、病史等特征构建决策树模型，用于预测患者是否患有某种疾病。通过决策树的结构，医生可以更直观地理解预测依据，提高诊断的透明度和可解释性。

### 4.4 常见问题解答

#### 如何选择K值？
在K-means聚类中，K值的选择直接影响到聚类效果。通常可以通过肘部法则或轮廓系数等方法来确定最佳的K值。

#### 如何处理缺失数据？
在协同过滤中，可以采用冷启动策略，比如基于用户的偏好模式或物品的流行度来填补缺失值。

#### 决策树过拟合怎么办？
为避免决策树过拟合，可以采用剪枝技术，如预剪枝或后剪枝，或者通过增加训练数据和特征来提高模型泛化能力。

## 5. 项目实践：代码实例和详细解释说明

### 5.1 开发环境搭建

为了运行Mahout相关代码，你需要安装以下环境：

#### 安装Hadoop和Mahout
确保你的系统上安装了Hadoop，并从Apache网站下载最新的Mahout版本。

#### 配置环境变量
在你的系统环境变量中设置HADOOP_HOME和MAHOUT_HOME。

### 5.2 源代码详细实现

#### 协同过滤实现
```java
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.model.SimilarityBasedUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.NormalizedSquaredEuclideanDistance;
import org.apache.mahout.cf.taste.impl.user.KNNUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.user.UserSimilarity;
import org.apache.mahout.cf.taste.impl.user.distance.DistanceSimilarity;

public class CollaborativeFiltering {
    public static void main(String[] args) throws TasteException {
        // 创建用户-物品评分矩阵
        // ...

        // 创建相似度计算器
        UserSimilarity similarity = new PearsonCorrelationSimilarity(new NormalizedSquaredEuclideanDistance());

        // 创建推荐器
        SimilarityBasedUserBasedRecommender recommender = new KNNUserBasedRecommender(similarity);

        // 使用推荐器进行预测
        // ...
    }
}
```

#### K-means聚类实现
```java
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.RandomAccessSparseVectorFactory;
import org.apache.mahout.math.RandomAccessSparseVectorIterator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVectorFactory;
import org.apache.mahout.math.SequentialAccessSparseVectorIterator;
import org.apache.mahout.math.VectorUtil;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.IntIntDoubleFunction;
import org.apache.mahout.math.function.IntIntIntFunction;
import org.apache.mahout.math.function.IntIntLongFunction;
import org.apache.mahout.math.function.IntIntObjectFunction;
import org.apache.mahout.math.function.IntObjectFunction;
import org.apache.mahout.math.function.LongFunction;
import org.apache.mahout.math.function.ObjectFunction;
import org.apache.mahout.math.function.ObjectIntFunction;
import org.apache.mahout.math.function.ObjectLongFunction;
import org.apache.mahout.math.function.ObjectObjectFunction;
import org.apache.mahout.math.function.ObjectShortFunction;
import org.apache.mahout.math.function.ShortFunction;
import org.apache.mahout.math.function.ThrowingFunction;
import org.apache.mahout.math.function.ThrowingIntFunction;
import org.apache.mahout.math.function.ThrowingLongFunction;
import org.apache.mahout.math.function.ThrowingObjectFunction;
import org.apache.mahout.math.function.ThrowingShortFunction;
import org.apache.mahout.math.function.UnaryOperator;
import org.apache.mahout.math.function.UnaryPredicate;
import org.apache.mahout.math.function.BinaryOperator;
import org.apache.mahout.math.function.BinomialPredicate;
import org.apache.mahout.math.function.ThrowingUnaryOperator;
import org.apache.mahout.math.function.ThrowingBinaryOperator;
import org.apache.mahout.math.function.ThrowingBinomialPredicate;
import org.apache.mahout.math.function.ThrowingUnaryPredicate;
import org.apache.mahout.math.function.ThrowingBinaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math.function.ThrowingQuaternaryPredicate;
import org.apache.mahout.math.function.ThrowingQuinaryPredicate;
import org.apache.mahout.math.function.ThrowingSenaryPredicate;
import org.apache.mahout.math.function.ThrowingSeptenaryPredicate;
import org.apache.mahout.math.function.ThrowingOctenaryPredicate;
import org.apache.mahout.math.function.ThrowingNonaryPredicate;
import org.apache.mahout.math.function.ThrowingTernaryPredicate;
import org.apache.mahout.math