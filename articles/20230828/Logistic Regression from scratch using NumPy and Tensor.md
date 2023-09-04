
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Logistic Regression是一个非常常用的分类模型。它的基本逻辑是：假设存在某个变量X与一个因变量Y之间的关系，通过学习这个关系，能够预测出一个映射函数f(x)将输入数据x转化为输出结果y。对于二分类问题来说，它的输入数据x可以是特征向量，输出结果y只有两种可能值，即正例或者反例。一般情况下，Logistic Regression通过求得的映射函数计算得到概率P(Y=1|X)，即给定输入X，预测其所属的类别。Logistic Regression的学习算法就是训练模型参数θ使得预测结果P(Y=1|X)尽可能接近真实标签y。

2.Logistic Regression与Linear Regression
在讨论Logistic Regression之前，先简单回顾一下线性回归中的相关知识。线性回归的假设是输入变量X与一个因变量Y之间存在线性关系，即Y=β0+β1*X+ε，其中β0和β1是回归系数，ε是一个误差项。当目标变量Y是连续型变量时，线性回归就称为回归问题；而当目标变量Y是二值型变量（即0或1）时，线性回归又被称为逻辑回归（Logistic Regression）。

线性回归可以解决很多问题，比如房价预测、销售额预测、销售量预测等。但是它有一个致命缺陷——只能处理实数类型的数据。例如，你不能用它来判断是否买房子还是租房子；也不能用来估计股票价格涨跌。因此，线性回归常常被用来作为基础模型来构建更复杂的机器学习模型。

Logistic Regression是一种特殊的线性回归模型，可以用来处理二值型变量的问题。它与线性回归最大的不同之处在于：它输出的是一个在0到1之间的概率值，而不是像线性回归那样的连续值。换句话说，Logistic Regression的输出是一个“概率”而不是“确定性的值”。这意味着，Logistic Regression可以用来进行二元分类任务，比如二进制分类、多元分类等。在实际应用中，常用的是分类准确率。

3.Logistic Regression中的基本概念及术语
Logistic Regression最重要的三个要素：sigmoid 函数、代价函数、训练过程。首先，我们来看一下Sigmoid函数。

4.Sigmoid函数
sigmoid函数的定义域是R，值域为[0,1]。它是S形曲线函数，即在坐标轴上任取一点(x, y)，都有：

f(x)=1/(1+exp(-y))=(e^-y)/(1+e^-y), 当y是负无穷大时，等于1/2; 当y是正无穷大时，等于1; 其他情况在区间[0,1]内均匀分布。

sigmoid函数的图像如下图所示：


其导函数为：

f'(x)=-f(x)(1-f(x)), f(x)>0, -1<f(x)<1; f'(x)=f(x)-f(x)^2, f(x)=0; f'(x)=-f(x)+f(x)^2, f(x)=1

由此可见，sigmoid函数在取值为(0,1)时的斜率为0，值域为(0,1)。

接下来，我们再来看一下代价函数。

5.代价函数
代价函数是指模型预测的准确性。采用损失函数的方法衡量预测准确性的方法是不对的，因为这种方法无法提供模型内部决策的信息。所以，模型的性能评价往往需要结合多方面信息才能得到。Logistic Regression通常会选择交叉熵作为代价函数，即：

J(θ)=−logP(Y|X;θ), Y=1, P(Y=1|X;θ)={1 \over 1+exp(-\theta^{T}X)}=sigmoid(\theta^{T}X)

cross entropy: J(θ)=−yln(p)+(1-y)ln(1-p)

where y is the true label of X with value either 0 or 1, p is the predicted probability of Y being equal to 1 given X and θ is the model parameter vector. The cross entropy function measures the average number of bits needed to identify the correct class label y for a given instance x. Cross entropy can be used as an objective function in binary classification problems.