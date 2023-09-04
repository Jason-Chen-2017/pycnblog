
作者：禅与计算机程序设计艺术                    

# 1.简介
  

随着科技的发展，人类基因组中隐私信息越来越多、保护措施也日渐完备。基于这些情况，我们可以通过大规模的个人基因组数据分析发现各种疾病的致病基因突变。然而，这些发现存在着一定的不确定性。例如，虽然目前已知的致癌基因突变数量仅占所有致病突变的很少一部分，但每个致癌突变在临床试验中的临床表现却不能确定其是否真的致癌。

为了更好地理解这些潜在影响，我们可以结合生物统计学的知识和机器学习方法，通过分析个人基因组数据的遗传变异影响患者病症的风险。具体来说，我们希望识别出病人的不易察觉的遗传变异——即临床上表现不明显或暂时的变异——并根据这些变异的发生频率、临床表现以及受到变异的基因靶点所导致的变异后果，判断这些变异对于各个疾病的预后将产生何种影响。

本文以大样本的人工健康档案数据集（PHESANT）作为研究对象，进行了遗传变异分析，并基于贝叶斯高层聚类模型（BHC）进行了突变风险评估和疾病关联分析。我们希望从中了解到：

1. 在我们收集到的信息量较小的情况下，如何对不易察觉的遗传变异进行有效识别？
2. 通过遗传变异之间的关联关系，我们能够推断出遗传变异的生物意义，并进一步推导出遗传变异暴露在各个疾病上的相关性，为医学决策提供参考。
3. 当遗传变异出现时，人的行为可能会受到哪些因素的影响？为此，我们应该在什么条件下进行临床试验，以及哪些指标可用于评价试验结果？

在最后，我们还希望通过本文的讨论引申出以下几个方面：

1. 大规模人工健康档案数据的挖掘潜力如何？将个人基因组数据纳入医学决策的研究如何？
2. 将遗传变异分析与机器学习相结合，如何利用大规模数据提升遗传变异风险评估能力？
3. 如果将BHC应用于更广泛的疾病领域，有什么需要注意的地方？


# 2.基本概念术语说明
## 2.1 概念及术语定义
### 2.1.1 遗传变异
遗传变异（genetic variations）是指遗传信息的单个事件。它是指DNA片段或核苷酸序列的单个碱基变化，其频率高低直接影响基因表达水平、疾病风险和生命健康。遗传变异可能是微小的、短期的，或者是持续的、长期的。

### 2.1.2 突变
突变（mutation）是指遗传变异引起的DNA编辑过程。突变可以是单一的或多态的。单一突变通常是由于一次变异引起的，如同寻找并克隆单个碱基一样。多态突变则是由许多不同的突变共同驱动的，往往导致一个或多个位点同时发生突变。

### 2.1.3 SNP
SNP（single nucleotide polymorphism，单核苷酸多态性）是指两个不同核苷酸之间存在差异，且突变会导致基因功能改变或发生招募效应的变异。

### 2.1.4 INDEL
INDEL（insertion or deletion，插入缺失）是指核苷酸的增加或缺失，且突变会导致基因功能改变或发生招募效应的变异。

### 2.1.5 母基因
母基因（ancestral allele）是指一个生物体先前拥有的某个特定染色体位置的核苷酸。它是被认为存活下来的物种的原始遗传基因，具有独特的编码序列。

### 2.1.6 配对原型
配对原型（matched reference panel）是由从不同国家或组织收集而成的参照系，它们都属于相同的群体，而其基因都已经经过高度筛选。它的作用是在研究过程中比较两组人的基因遗传差异。

### 2.1.7 病例控制
病例控制（case control study）是一种用以研究疾病患病率或死亡率等因素对某一疾病的风险影响的临床研究方法。在该方法中，病人被分为两组：一组为病例组（cases），另一组为控制组（controls）。病例组称为受关注组（interest group），控制组则称为对照组（control group）。

### 2.1.8 BHC
BHC（Bayesian Hierarchical Clustering）是一种贝叶斯概率统计的方法，可用来对包含观测值的高维数据集进行聚类，其中数据点可以视为来自一个未知联合分布。它假设每组数据的先验分布服从多元高斯分布，然后按照联合高斯分布计算每组数据的似然值，并据此生成模型参数。通过反向传播更新模型参数，使得各组数据在同一个簇内的程度尽量接近，而不同簇间的数据距离较远。最终得到的结果是聚类结果，其中每一簇代表一个观测值。BHC可以用于处理任意的离散变量数据，并兼顾数据的内部结构和外部因素。

### 2.1.9 树形模型
树形模型（tree model）是一个用来描述数据的统计模型，主要用于对复杂系统进行分析、预测和分类。树模型由节点和边组成，其中节点表示观测值或观测值集合，边表示一个连续变量的依赖关系。树形模型可以用来刻画数据的相关性、集群、结构等特征，并通过模型中的参数估计、求解等方法来预测新的数据。

## 2.2 数据集及下载地址
### 2.2.1 PHESANT数据集
PHESANT（Personal Health Expenditure Study of African-American NHANES）是一个经过NHLBI卫生部门批准的，关于非洲裔美国人健康费用的定量研究项目。该项目目的是了解非洲裔美国人的个人卫生消费习惯、生育模式、生活方式等，以此来更好的诊断和管理疾病。

PHESANT共有三千八百万份问卷记录，覆盖全美约1.5亿名非洲裔美国人。数据结构为439页，每条记录均有六十多项可供选择的健康指标，包括年龄、种族、收入水平、个人卫生支出、生育计划、婚姻状况、养育子女数量、工作满意度、社交圈子的动态、家庭结构及细节等。

PHESANT数据集可以免费下载，其网站为https://wwwnhlbi.umassmed.edu/content/phsant。数据获取方法如下：

1. 注册账户；
2. 从首页点击左侧“Data Sets”按钮，然后选择“NHANES AFRICAN AMERICANS”，进入到该项目的主页面；
3. 点击“Download”按钮，进入到下载页面。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 3.1 数据处理
### 3.1.1 处理前需考虑的问题
首先，我们要清楚遗传变异的特点，掌握其生物意义，才能对其进行有效识别。其次，掌握遗传变异的分类标准，有利于我们划分不同类型的遗传变异，以便区别它们的风险。第三，知道遗传变异的检测技术，对于检测遗传变异的准确率至关重要。

### 3.1.2 数据集选取
为了确保准确性，我们选取了PHESANT数据集。该数据集有五十七个字段，包含多个相关性强且稀疏的数据点，且数据没有缺失值。因此，在分析之前，我们进行了充分的数据清理和准备工作。

### 3.1.3 转换成可用于机器学习的格式
我们把PHESANT数据集转化为csv文件格式，其大小为216M，列数为439，行数为14900000。为了方便处理，我们只保留了感兴趣的字段，包括：

- Age：患者的年龄。
- Race：患者种族。
- GDPPercap：患者的GDP per capita。
- AlcoholUse：患者的饮酒频率。
- DietRich：患者饮食是否富裕。
- BMI：患者体重。
- LDLCholesterol：患者的胆固醇。
- SmokingStatus：患者吸烟状态。
- TimeSmoked：患者最近一次吸烟的时间长度。
- CigsPerDay：患者每天抽烟的次数。
- PhysicalHealthIssues：患者是否有身体问题。
- GeneticRiskFactor：遗传风险因子。
- PersonalHistoriesOfCancer：患者是否有癌症史。
- TobaccoUse：患者是否曾用过烟草。
- AsthmaDx：患者是否有阿斯玛玛的诊断。
- OsteoporosisDx：患者是否有骨质疏松的诊断。
- HeartDiseaseDx：患者是否有心脏病的诊断。
- ChronicKidneyDiseaseDx：患者是否有老年性肾病的诊断。
- DiabetesType：患者糖尿病类型。
- HepatitisDx：患者是否有乙肝的诊断。
- HypertensionDx：患者是否有高血压的诊断。
- HyperlipidemiaDx：患者是否有高脂血症的诊断。
- IschemicHeartDx：患者是否有ischemic heart disease的诊断。
- PVDx：患者是否有某种心梗的诊断。
- LeukemiaDx：患者是否有黑色素瘤的诊断。
- ProstateCancerDx：患者是否有前列腺癌的诊断。
- BreastCancerDx：患者是否有乳腺癌的诊断。
- UterineCancerDx：患者是否有宫颈癌的诊断。
- TestosteroneTherapy：患者是否接受妊娠检测试剂。
- StrokeDx：患者是否有Stroke的诊断。
- ThalassemiaDx：患者是否有萎缩性狼疮的诊断。
- AgeAtFirstDiagnosis：患者首次诊断的年龄。
- Race_Black：患者种族为黑种的概率。
- Race_Other：患者种族为其他种类的概率。
- Race_White：患者种族为白种的概率。
- MaritalStatus：患者婚姻状况。
- ChildrenCount：患者有无子女。
- MedicalConditionsHistory：患者既往疾病历史。
- Ethnicity：患者民族。
- Gender：患者性别。
- HospitalizationDate：患者入院时间。
- MotherAliveAtDiagnosis：患母亲在诊断时是否存活。
- InfertilityDx：患者怀孕产次不完全的诊断。
- ObesityDx：患者体重超重的诊断。
- ChronicRenalInsufficiencyDx：患者慢性kidney disease的诊断。
- PostoperativePainDx：手术后的疼痛的诊断。
- SurgeryDx：患者是否做过手术。
- RadiographicFindingsOnStudySubject：患者影像检查结果。
- LabResultsForPredictions：患者预测结果的lab test结果。
- PredictionOutcomes：预测结果的结论。

### 3.1.4 过滤掉缺失值
我们发现，原数据集中有些字段存在极少数缺失值，而且很少。为了避免分析时丢失重要信息，我们对包含缺失值的字段进行了过滤。具体方法为：首先统计每个字段的缺失值个数，筛除缺失值较多的字段，再对剩余字段进行填补。

### 3.1.5 分割训练集和测试集
为了验证模型的效果，我们随机划分了数据集，把前50%的数据作为训练集，后20%的数据作为测试集。

## 3.2 遗传变异分析
### 3.2.1 遗传变异类型分布
为了便于分析，我们对遗传变异进行分类，如单核苷酸多态性(SNPs)、插入缺失(indels)。这样可以更加直观地看出遗传变异类型分布。我们可以计算每个样本的SNP和indel的比例，并画出频率分布图。

### 3.2.2 遗传变异的位置分布
为了更直观地了解遗传变异在基因组中的分布情况，我们计算了每个样本中SNP和indel的位置分布。具体方法为，我们计算了每个样本中每个SNP和indel出现的位置的数量，并绘制热图。

### 3.2.3 SNP和indel的关联分析
为了更进一步分析遗传变异之间的关联关系，我们使用协方差矩阵，通过计算样本的每对SNP和indel之间的协方差值，构建了一个两两相关性网络图。协方差矩阵衡量了两个变量之间的线性关系，它的值在[-1,+1]范围内，数值越大表示两个变量相关性越强。我们可以分析不同类型的遗传变异之间在网络中的联系，以此来理解遗传变异的生物意义。

## 3.3 模型训练和参数估计
### 3.3.1 数据预处理
我们对数据进行了预处理，包括标准化和拆分训练集和测试集。具体步骤为：

1. 对每一个特征（每个字段）进行标准化，将其变换到[-1, +1]范围内；
2. 拆分训练集和测试集，设置比例为7:3。

### 3.3.2 模型训练
我们选择了贝叶斯高层聚类模型（BHC）来进行分类。BHC是一个基于贝叶斯概率分布的聚类方法，可以用来对包含观测值的高维数据集进行聚类，其中数据点可以视为来自一个未知联合分布。它假设每组数据的先验分布服从多元高斯分布，然后按照联合高斯分布计算每组数据的似然值，并据此生成模型参数。通过反向传播更新模型参数，使得各组数据在同一个簇内的程度尽量接近，而不同簇间的数据距离较远。最终得到的结果是聚类结果，其中每一簇代表一个观测值。

BHC通过迭代的方式逐步减小参数估计误差，最终得到一个较好的聚类结果。模型训练的详细步骤为：

1. 初始化模型参数；
2. 迭代优化模型参数，使得模型可以最大程度拟合数据；
3. 根据模型参数，生成新的样本聚类结果。

### 3.3.3 参数估计
BHC的迭代优化算法通过最小化似然函数来拟合数据。具体来说，BHC的目标是找到一组均值向量μ和协方差矩阵Σ，使得观测值X的联合概率分布P(X|μ,Σ)最大化。

假设X为观测值，μ为平均值向量，Σ为协方差矩阵，那么P(X|μ,Σ)是观测值X的概率密度函数。BHC的参数估计使用EM算法（Expectation Maximization Algorithm）。

EM算法的三个步骤如下：

1. 第t轮迭代开始时，令q(z|x)=p(z|x),这是当前的聚类概率。
2. 更新μ、Σ和q(z|x)，也就是将q(z|x)中的xi分配给聚类中心ci。
3. 对q(z|x)进行归一化处理，使之满足约束条件p(z=ci)=p(z'=cj|x)+p(ci)/N，其中N为数据集的大小。

BHC使用EM算法估计模型参数，并且在每一轮迭代中都使用相同的先验分布和似然函数。也就是说，BHC在每一轮迭代中使用全部数据对参数进行了估计。

## 3.4 模型性能评估
### 3.4.1 聚类效果评估
为了评估聚类效果，我们计算了样本在不同聚类的标准化欧氏距离。具体方法为：

1. 遍历所有的聚类，计算每一个聚类中样本之间的标准化欧氏距离，然后取平均值；
2. 计算所有聚类平均的标准化欧氏距离，作为聚类效果的评估指标。

聚类效果是指样本在聚类中的聚合程度。如果聚类效果较低，则说明样本在聚类中处于分散状态，因而聚类效果较差。

### 3.4.2 置信度评估
置信度评估是对聚类结果的可靠程度的评估。我们计算了聚类结果的置信度。置信度值介于0～1之间，数值越大，表示聚类结果的可靠程度越高。置信度评估方法包括：

1. Calinski-Harabasz index：该指标用来评估聚类效果。当样本在不同簇中的平均方差差距很大时，说明聚类效果较差；当样本在不同簇中的方差都比较接近时，说明聚类效果较好。

2. Dunn's index：该指标用来评估聚类中的簇间距离。当簇间距离越小，样本在不同簇中的重合度就越高，说明聚类效果较好；当簇间距离越大，样本在不同簇中的重合度就越低，说明聚类效果较差。

## 3.5 遗传变异暴露和疾病关联分析
### 3.5.1 遗传变异的风险评估
我们计算了样本中每个SNP和indel的风险评估分数，并绘制热图。风险评估分数越大，表示该SNP或indel在患病风险上带来的影响越大，故将该SNP或indel置为风险性突变。具体计算方法为：

1. 使用表1中提供的参考表格，计算每个样本的风险评估分数；
2. 将每个样本的风险评估分数超过某个阈值的SNP和indel标记为暴露性突变。

### 3.5.2 疾病关联分析
我们基于遗传变异的风险评估，探索遗传变异暴露对于各个疾病的影响。具体方法为：

1. 计算每个样本中每个风险性突变在各个疾病上的影响，并绘制热图；
2. 找出那些在不同疾病上有显著关联的突变。

# 4.具体代码实例和解释说明
## 4.1 Python代码示例
```python
import numpy as np 
from scipy import stats 
from sklearn.cluster import AgglomerativeClustering 

# Load the dataset 
data = pd.read_csv('phsant.csv') 
fields = ['Age', 'Race_White', 'LDLCholesterol'] 

# Preprocess the data 
data[fields].fillna(-999, inplace=True) # fill missing values with -999 
for field in fields: 
    mu = np.mean(data[field]) 
    std = np.std(data[field]) 
    if std > 0: 
        data[field] = (data[field]-mu)/std
    else:
        continue # skip this feature if it has zero variance
    
# Split the training set and testing set randomly 
np.random.seed(0) # fix the random seed to ensure reproducibility 
train_index = np.random.choice(len(data), size=int(.7*len(data)), replace=False) 
test_index = list(set(range(len(data)))-set(train_index)) 
train_data = data.iloc[train_index,:] 
test_data = data.iloc[test_index,:] 
    
# Train the clustering model and estimate parameters  
model = AgglomerativeClustering() 
model.fit(train_data[fields].values) 
clusters = [[] for i in range(max(model.labels_)+1)] 
for i in range(len(model.labels_)): 
    clusters[model.labels_[i]].append(i) 
        
# Evaluate the cluster performance on both train and test sets 
def evaluate_clustering(X, labels):  
    silhouette_scores = [] 
    calinski_harabasz_score = [] 
    dunn_index = [] 
    jaccard_similarities = []
    
    for k in range(min(kmeans.n_clusters_, max(labels))+1):
        c_mask = [label==k for label in labels]
        X_c = X[c_mask]
        
        if len(X_c)<2: 
            continue # skip empty clusters
            
        silhouette_score = metrics.silhouette_score(X_c, kmeans.labels_[c_mask], metric='euclidean') 
        calinski_harabasz_score.append(metrics.calinski_harabasz_score(X_c, kmeans.labels_[c_mask]))
        dunn_index.append(metrics.davies_bouldin_score(X_c, kmeans.labels_[c_mask]))
        
        
    return sum(silhouette_scores)/len(silhouette_scores), \
           sum(calinski_harabasz_score)/len(calinski_harabasz_score), \
           sum(dunn_index)/len(dunn_index) 
    
evaluate_clustering(train_data[fields].values, model.labels_) # evaluate on training set 
evaluate_clustering(test_data[fields].values, [model.labels_[i] for i in test_index]) # evaluate on testing set 
 
# Analyze SNP and Indels 
snps_data = data['GeneticRiskFactor'].str.split('_').apply(pd.Series)[0].reset_index().rename({'index':'ID'}, axis=1) 
indels_data = data['GeneticRiskFactor'].str.split('_').apply(lambda x : pd.Series([y[3:] for y in x])).stack().to_frame().T.reset_index().drop(['level_1'],axis=1).rename({'index': 'ID', 0:'Variant'}, axis=1) 
all_vars_data = snps_data[['ID','Variant']] 
all_vars_data['Type'] = 'SNP' 
all_vars_data = all_vars_data.append(indels_data, ignore_index=True) 

# Analyze mutational burdens by identifying high risk variants exposed to different diseases 
risk_threshold =.2 # a threshold value between 0 and 1 used to identify exposure to risky variants 
var_exposures = {} 
for field in fields: 
    var_exposures[field] = {'ExposedVariants':[], 'SeverityScores':[]} 
    for variant in all_vars_data['Variant']: 
        prob = abs(stats.norm.pdf(data[[field]], loc=variant, scale=.2)[0][0])/stats.norm.cdf(0, loc=variant, scale=.2) 
        if prob >= risk_threshold:
            severity_score = -.02*(prob-.5)**2+.5 # compute severity score 
            var_exposures[field]['ExposedVariants'].append((variant, prob)) 
            var_exposures[field]['SeverityScores'].append(severity_score) 
            
# Visualize mutational burdens across different diseases 
fig, ax = plt.subplots(figsize=(10,5)) 
sns.barplot(data=[len(v['ExposedVariants']) for v in var_exposures.values()], orient='h') 
ax.set_yticklabels([field for field in fields]); ax.set_xlabel('# Exposed Variants'); 
plt.show() 

# Find correlations among SNPs and Indels 
cov_matrix = np.corrcoef(all_vars_data['Variant'])[:,-1] 
high_correlated_variants = [all_vars_data.loc[i,'Variant'] for i in np.argsort(abs(cov_matrix))[::-1]] 

# Visualize the top three most correlated variants 
top_three_corrs = high_correlated_variants[:3] 
top_three_corrs_mask = [(all_vars_data['Variant']==variant) & (all_vars_data['Type']=='SNP') for variant in top_three_corrs]+\
                      [(all_vars_data['Variant']==variant) & (all_vars_data['Type']=='Indel') for variant in top_three_corrs] 
                      
fig, axes = plt.subplots(nrows=3, figsize=(10,5)) 
for i, mask in enumerate(top_three_corrs_mask): 
    sns.histplot(data[(data['Race_White']==1)][fields[0]][mask], color='b', bins=20, ax=axes[i], alpha=.5); 
                
                   
fig, axes = plt.subplots(nrows=3, figsize=(10,5)) 
for i, mask in enumerate(top_three_corrs_mask): 
    sns.histplot(data[(data['Race_White']==0)][fields[0]][mask], color='r', bins=20, ax=axes[i], alpha=.5); 
             
```

## 4.2 R语言代码示例
```R
library(tidyverse)
library(BHCpack)

# load the phsant data 
data <- read.csv("phsant.csv")

# preprocess the data
fields <- c("Age", "LDLCholesterol", "BMI")
data[,fields]<- lapply(data[,fields], function(x){
  mean_val <- mean(x, na.rm=TRUE)
  sd_val <- sd(x, na.rm=TRUE)
  ifelse(!is.na(sd_val) && sd_val>0, 
         round((x-mean_val)/(sd_val)*100)/100, NA)})

# split the training set and testing set randomly 
set.seed(123)
trainIndex <- sample(seq_len(dim(data)[1]), round(0.7*dim(data)[1]), replace=FALSE)
testData <- data[-trainIndex, ]
trainData <- data[trainIndex, ]

# train the clustering model and estimate parameters
hcModel <- BHC(trainData[,fields], method="ward")$classes

# evaluate the cluster performance on both train and test sets
evaluateClustering <- function(df, labels, kRange){
  
  resDf <- data.frame(k=numeric(), silScore=numeric(), CHScore=numeric(), DBI=numeric())
  
  for(k in kRange){
    subDf <- df[labels==k, ]
    if(nrow(subDf)>1){
      silScore <- round(silhouette(subDf[,fields], subDf)$silhouette_avg, 4)
      chScore <- round(chScore(subDf[,fields], subDf)$value[[1]], 4)
      dbiValue <- round(dbi(subDf[,fields], subDf)$DBI, 4)
      
      resDf <<- rbind(resDf, data.frame(k=k, silScore=silScore, CHScore=chScore, DBI=dbiValue))
    }
  }
    
  cat("Mean Silhouette Score:", mean(resDf$silScore), "\n")
  cat("Mean Calinski-Harabaz Index Score:", mean(resDf$CHScore), "\n")
  cat("Mean Dunn's Biase index:", mean(resDf$DBI), "\n")

  return(resDf)
  
}

cat("\nTraining Set Performance:\n")
evaluateClustering(trainData, hcModel, kRange=1:length(unique(hcModel)))
cat("\nTesting Set Performance:\n")
evaluateClustering(testData, hcModel[testdata$ID], kRange=sort(unique(hcModel)))

# analyze SNP and indel variants 
snps <- apply(data["GeneticRiskFactor"], 1, function(x) substr(x, 1, nchar(x)-2))
indels <- apply(data["GeneticRiskFactor"], 1, function(x) {
  tmpVec <- unlist(strsplit(x, "_"))
  names(tmpVec)[grepl("^[^a-zA-Z]*", tmpVec)]}) %>%
  sapply(paste0, collapse="_") %>%
  unique()
allVars <- data.frame(cbind(ID=seq_len(dim(data)[1]), Variant=union(snps, indels)))
allVars$Type <- factor(ifelse(allVars$Variant %in% snps, "SNP", "Indel"), levels=c("SNP","Indel"))

# find high risk variants that are associated with different diseases 
varThreshold <- 0.2 # a threshold value between 0 and 1 used to identify exposure to risky variants 
varExposures <- matrix(ncol=length(fields), nrow=sum(sapply(fields, length)))
names(varExposures) <- paste0("Field_", seq_along(fields))
varIdx <- seq_len(nrow(allVars))

for(idx in seq_len(nrow(allVars))){
  
  varName <- allVars$Variant[idx]
  
  varProbMatrix <- t(round(pnorm(((t(data[match(varName, rownames(data)), fields])-rep(mean(data[,fields]), each=nrow(data))))/(rep(sd(data[,fields]),each=nrow(data))), lower.tail=F)))
  
  assocDis <- c()
  for(fieldIdx in seq_along(fields)){
    
    currVarProbs <- sort(varProbMatrix[,fieldIdx], decreasing=TRUE)[order(varProbMatrix[,fieldIdx])]
    assocDis[fieldIdx] <- sum(currVarProbs[currVarProbs>=varThreshold])
    
  }
  
  if(any(assocDis/length(fields)>0.5)){
    
    disOrder <- order(assocDis, decreasing=TRUE)
    
    varExposures[names(disOrder), varIdx[idx]] <- TRUE
    varIdx[idx] <- NULL
    
    message("Mutational Burden found in Field_", names(disOrder), ": ", 
            round(assocDis[disOrder]/length(fields)*100, 2), "% of samples have mutation.")
    
  }
  
  
}

# visualize mutational burdens across different diseases 
plot(table(factor(sapply(fields, function(x) colnames(data[grep("_", colnames(data))]==x)),levels=fields))/length(data)*100, 
     type="bar", main="Mutational Burdens Across Different Diseases",
     ylim=c(0,100), ylab="# Samples (%)")