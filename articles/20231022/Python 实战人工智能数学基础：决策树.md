
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


决策树（decision tree）是一种用于分类、回归分析的机器学习方法。决策树由一个根节点、分支节点和终端节点组成，根节点表示最初进行判断的数据集的样本，分支节点表示对样本进行划分的属性或特征，终端节点表示叶子结点上的结果，例如，目标变量离散化后取值分布最大的类别。其优点是简单直观，易于理解和实现，并可以处理多维数据。但是，它也存在一些缺陷，如可能出现过拟合现象、不稳定性等。
# 2.核心概念与联系
## 2.1 基本术语
- 根节点：决策树中的第一个结点，从这里开始分叉，形成不同子树。
- 分支节点：分叉过程产生的新结点。
- 终端节点：叶子结点，表示结果。
- 父亲节点：指向该结点的上一级结点。
- 孩子节点：指当前结点分支的两个子结点。
- 属性：决策树构建的依据，在决策树算法中通常是指某个变量或变量组合，用来描述数据。
- 结点的度（Degree）：该结点子树中结点的个数。
- 深度（Depth）：树的层数。
- 高度（Height）：树中所有分支的长度。
- 内部节点：除根节点和叶子节点外的其它所有结点。
- 外部节点：根节点和叶子节点之外的所有结点。
- 概率（Probability）：事件发生的概率。
- 信息熵（Entropy）：表示随机变量的无序程度。
- 信息增益（Information Gain）：表示得知特征X的信息而使得类Y的信息的不确定性减少的程度。
- 信息增益比（Gain Ratio）：表示在同等条件下选择特征X的信息所获得的期望信息增益与随机选择该特征的信息增益的比值。
- 基尼系数（Gini Impurity）：衡量二元分类问题的不纯度。
## 2.2 ID3算法
ID3算法（Iterative Dichotomiser 3，缩写为ID3）是一个基于信息增益准则的决策树生成算法。该算法的主要特点是独裁和迭代，即每次只处理数据的一部分，通过一步步增加分裂的特征范围，逐渐生长出完整的决策树。
### 2.2.1 原理
1. 对数据集D进行切分：
    - 将D划分为K个互斥的子集，每个子集包含相同的标签；
    - 如果所有元素属于同一类Ck，则停止切分；否则继续。
    
2. 在切分的每一种方式下，计算信息增益：
    - 以第k个特征A为基准将D分割成若干子集Di，计算Di中属于各类的样本数目Dik和样本总数Dk，以及不属于Ak类的样本数目Dki和样本总数Dk；
    - 定义信息增益为信息增益=Wk-Uk*log2(Uk/(Duk+epsilon))，其中Wk为经验熵，Uk为属性Ak的信息熵，epsilon>0可防止除零错误；
    
3. 返回具有最大信息增益的切分特征：
    - 返回具有最大信息增益的特征作为最佳切分特征；
    - 如果此时还有特征可以切分，则返回第2步重新进行计算，否则停止计算。

### 2.2.2 举例
假设我们要用ID3算法构造如下的决策树：
首先选取根结点，然后计算剩余特征的各项信息增益。由于根结点只有一类，所以信息增益为0，因此停止切分。
然后再选取最好的两个特征，分别为Outlook和Temperature。
- Outlook:由于总体的占比很小，所以无法做出决定性的划分。但根据Outlook和Temperature之间的相关性，可以将低热，高热数据划入左分支，高冷，低冷数据划入右分支。可以得到如下的决策树：
- Temperature:在Outlook和Temperature两者之间，可计算出如下信息增益：
Outlook：好->差，坏->好，平->差，有下列公式计算：
$$
Info(Outlook)=\sum_{i=1}^{N}(P_i-P'_i)log_2(P_i/P'_i)\\
P'_i=\frac{N_i}{N}\\
P_i=\frac{\sum_{j}^NP_jx_ij}{\sum_{j}x_ij}\\
N_i=|\{j|x_{ij}\}=k\\
x_ij=\{0,1\},\forall i,\forall j
$$
Temperature：好->差，坏->好，平->差，有下列公式计算：
$$
Info(Temperature)=\sum_{i=1}^{N}(P_i-P'_i)log_2(P_i/P'_i)\\
P'_i=\frac{N_i}{N}\\
P_i=\frac{\sum_{j}^NP_jx_ij}{\sum_{j}x_ij}\\
N_i=|\{j|x_{ij}\}=k\\
x_ij=\{(-inf,-1],(-1,1),(1,inf)\},\forall i,\forall j
$$
综合Outlook的和Temperature的增益，可以知道Outlook的信息增益为0.69，Temperature的信息增益为0.76。
那么选取Outlook的增益更大，因此Outlook为基准切分。根据Outlook的值，把数据分成两个部分：
- 好->差：数据越少，越容易被识别为好，所以选择这一类数据的样本较少。
- 不属于好->差的其他情况。
按照类似的方式，求Temperature的信息增益和选择切分属性，得到如下的决策树：
最后，回溯到根结点，完成决策树的构造。
## 2.3 C4.5算法
C4.5算法（Critical 4.5，缩写为C4.5）是在ID3算法的基础上进行了改进，是对ID3的改进版本。它的基本思想是，在对数据进行切分的时候，优先考虑能够降低基尼指数的属性，而不是仅仅考虑信息增益大的属性。这样做的原因是，当样本集中含有不同规模的类时，信息增益偏向于具有较多样本数目的那些类，这就可能会导致过拟合。C4.5算法采用启发式方法，对连续值型属性使用二分法对其进行切分，对离散值型属性采用多数表决的方法。
### 2.3.1 原理
1. 初始化：
    - 设置根结点；
    - 对于每个实例：
        - 根据实例的特征值来选择加入到路径上的最近的父结点。

2. 对每个非叶子结点，计算其分裂后的基尼指数：
    - 从所有的实例中，根据当前结点的特征分成两部分：
        - 使用当前结点的特征值为真的实例子集。
        - 使用当前结点的特征值为假的实例子集。
    
    - 计算出：
        - 左子结点的基尼指数Gi(D|A) = Σ[(k / |D|) * (p(k) * log2(p(k)))] + [(l / |D'|) * ((1 - p(k')) * log2((1 - p(k'))))]
        - 右子结点的基尼指数Gi(D|!A) = Σ[(k / |D|) * ((1 - p(k)) * log2((1 - p(k))))] + [(l / |D'|) * (p(k') * log2(p(k')))]
        
    - 基尼指数较大的是最佳分裂特征。

3. 生成决策树：
    - 当结点的子结点数目达到某个阈值（比如5）或者没有更多的特征可以分裂的时候，就停止分裂。
    - 否则，按照基尼指数进行特征的分裂。
    - 每次生成一个新的子结点的时候，为它分配一个唯一标识符。
    - 把训练实例送到相应的子结点，同时更新结点的统计信息。
    - 返回根结点。
    
### 2.3.2 举例
假设我们要用C4.5算法构造如下的决策树：
初始化阶段，每个实例对应到根结点上，所以先找出距离最近的父结点。
对于非叶结点Outlook，计算一下基尼指数：
- 左子结点的基尼指数为：
$$
Gi(D|Outlook=Sunny)=\frac{2/8}{8} \cdot 1 \cdot log_2(\frac{2/8}{8})+\frac{6/8}{8} \cdot (1-\frac{1}{3}) \cdot log_2((1-\frac{1}{3}))=\frac{2}{8}\\
Gi(D|Outlook=Overcast)=\frac{4/8}{8} \cdot 1 \cdot log_2(\frac{4/8}{8})+\frac{4/8}{8} \cdot (1-\frac{1}{3}) \cdot log_2((1-\frac{1}{3}))=\frac{2}{8}
$$
- 右子结点的基尼指数为：
$$
Gi(D|Outlook=Rain)=\frac{2/8}{8} \cdot (1-\frac{2}{3}) \cdot log_2((1-\frac{2}{3}))+\frac{6/8}{8} \cdot (\frac{2}{3}) \cdot log_2(\frac{2/3}{8})=\frac{2}{8}\\
Gi(D|Outlook=Foggy)=\frac{2/8}{8} \cdot (1-\frac{1}{3}) \cdot log_2((1-\frac{1}{3}))+\frac{6/8}{8} \cdot (1-\frac{2}{3}) \cdot log_2((1-\frac{2}{3}))=\frac{2}{8}
$$
选取基尼指数较大的作为分裂特征，在Outlook为Sunny的情况下分裂出两个结点。
在Outlook为Sunny的结点里，选择Temperature的值作为切分特征，进行分裂：
- 左子结点的基尼指数为：
$$
Gi(D|Outlook=Sunny, Temperature<-10)=\frac{2/6}{6} \cdot 1 \cdot log_2(\frac{2/6}{6})+\frac{4/6}{6} \cdot (1-\frac{1}{3}) \cdot log_2((1-\frac{1}{3}))=\frac{2}{6}\\
Gi(D|Outlook=Sunny, Temperature=-5<=T<=-2)=\frac{2/6}{6} \cdot (1-\frac{1}{3}) \cdot log_2((1-\frac{1}{3}))+\frac{4/6}{6} \cdot (1-\frac{2}{3}) \cdot log_2((1-\frac{2}{3}))=\frac{2}{6}\\
Gi(D|Outlook=Sunny, Temperature=-2<=T<=-1)=\frac{2/6}{6} \cdot (1-\frac{2}{3}) \cdot log_2((1-\frac{2}{3}))+\frac{4/6}{6} \cdot (\frac{1}{3}) \cdot log_2(\frac{1/3}{6})=\frac{2}{6}\\
Gi(D|Outlook=Sunny, Temperature=-1<=T<=0)=\frac{2/6}{6} \cdot (\frac{1}{3}) \cdot log_2(\frac{1/3}{6})+\frac{4/6}{6} \cdot (\frac{1}{3}) \cdot log_2(\frac{1/3}{6})=\frac{1}{3}
$$
- 右子结点的基尼指数为：
$$
Gi(D|Outlook=Sunny, Temperature>=10)=\frac{4/6}{6} \cdot (\frac{1}{3}) \cdot log_2(\frac{1/3}{6})+\frac{2/6}{6} \cdot (\frac{2}{3}) \cdot log_2(\frac{2/3}{6})=\frac{1}{3}\\
Gi(D|Outlook=Sunny, Temperature=1<=T<=2)=\frac{4/6}{6} \cdot (1-\frac{2}{3}) \cdot log_2((1-\frac{2}{3}))+\frac{2/6}{6} \cdot (\frac{1}{3}) \cdot log_2(\frac{1/3}{6})=\frac{1}{3}\\
Gi(D|Outlook=Sunny, Temperature=2<=T<=5)=\frac{4/6}{6} \cdot (1-\frac{1}{3}) \cdot log_2((1-\frac{1}/3)))+\frac{2/6}{6} \cdot (1-\frac{2}/3) \cdot log_2((1-\frac{2}/3))=\frac{1}{3}\\
Gi(D|Outlook=Sunny, Temperature>=5)=\frac{4/6}{6} \cdot (1-\frac{1}/3) \cdot log_2((1-\frac{1}/3))+\frac{2/6}{6} \cdot (1-\frac{1}/3) \cdot log_2((1-\frac{1}/3))=\frac{1}{3}
$$
选取基尼指数较大的作为分裂特征，在Outlook为Sunny且Temperature>=10的情况下分裂出两个结点。
在Outlook为Sunny的结点里，选择Humidity的值作为切分特征，进行分裂：
- 左子结点的基尼指数为：
$$
Gi(D|Outlook=Sunny, Temperature>=10, Humidity<0.6)=\frac{2/2}{2} \cdot (1-\frac{1}{3}) \cdot log_2((1-\frac{1}/3))+\frac{0/2}{2} \cdot (1-\frac{2}/3) \cdot log_2((1-\frac{2}/3))=\frac{0}{2}\\
Gi(D|Outlook=Sunny, Temperature>=10, Humidity>=0.6)=\frac{0/2}{2} \cdot (1-\frac{1}/3) \cdot log_2((1-\frac{1}/3))+\frac{2/2}{2} \cdot (1-\frac{1}/3) \cdot log_2((1-\frac{1}/3))=\frac{0}{2}
$$
- 右子结点的基尼指数为：
$$
Gi(D|Outlook=Sunny, Temperature>=10, Humidity<0.6)+Gi(D|Outlook=Sunny, Temperature>=10, Humidity>=0.6)=\frac{2/2}{2} \cdot (1-\frac{1}/3) \cdot log_2((1-\frac{1}/3))+ \frac{0/2}{2} \cdot (1-\frac{2}/3) \cdot log_2((1-\frac{2}/3))+\frac{0/2}{2} \cdot (1-\frac{1}/3) \cdot log_2((1-\frac{1}/3))+\frac{2/2}{2} \cdot (1-\frac{1}/3) \cdot log_2((1-\frac{1}/3))=\frac{1}{2}
$$
选取最佳切分特征，按照Humidity<0.6和Humidity>=0.6的两种情况对数据进行分割，得到如下的决策树：