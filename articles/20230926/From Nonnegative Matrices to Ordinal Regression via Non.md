
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Non-negative matrix factorization (NMF) has been one of the most popular topic modeling algorithms for decomposing non-negative data matrices into low rank factors. While NMF is capable of generating meaningful topics from the sparse input data and finding interpretable relationships among them, its applications in ordinal regression problems are limited due to some key issues that we need to address before using it for such purpose. In this paper, we present an algorithm called NTF (Non-negative tensor factorization), which extends previous work on Non-negative Matrix Factorization (NMF) by incorporating additional constraints related to ordinal variables. We also propose a new cost function for ordinal regression based on the Wasserstein distance between the predicted scores and true scores along with their respective cardinalities, which captures the notion of goodness-of-fit while accounting for ordinal variable information. We demonstrate our approach through extensive experiments on multiple datasets including rating prediction, movie recommendation, and customer feedback analysis. The proposed method significantly outperforms other state-of-the-art methods on these tasks and provides insights into how the latent factors relate to each other and underlying relationships among different features. 

# 2.相关术语
## 2.1 Non-negative Matrices
A matrix $X$ is said to be positive if all elements of the matrix satisfy the condition $\forall_{i,j}(x_{ij} \geq 0)$, where $x_{ij}$ represents the element at row i and column j of X. A negative value denotes a missing or unknown value in the matrix. Similarly, a matrix is said to be negative if all elements satisfy the condition $\forall_{i,j}(x_{ij} \leq 0)$. Otherwise, it is called zero-sum. An example of a non-negative matrix would be ratings of movies given by users, where every user gives only a single rating for a particular movie within a range [1,5]. 

In machine learning and pattern recognition, non-negative matrix factorization (NMF) refers to a type of decomposition technique used to extract patterns from a set of observations and organize them into a smaller set of uncorrelated components. It is widely used in applications like image compression, text clustering, and recommender systems. For instance, when analyzing user preferences for products, the user ratings can be represented as a non-negative matrix. By performing dimensionality reduction, NMF tries to find small subspaces that best explain the variation in the user ratings. These subspaces contain valuable information about the user's interests, behaviors, and preferences.

## 2.2 Positive Definite Matrices
Positive definite matrices are matrices whose transpose dot product matrix is always greater than zero. They have positive eigenvalues. Examples include covariance matrices of Gaussian processes or dissimilarity matrices in clustering.

## 2.3 Cardinality
The cardinality of a binary random variable indicates the number of possible outcomes for the variable. An ordinal variable is defined over a finite domain consisting of ordered pairs $(\Omega,\leq)$, where $\Omega=\{(\omega_1,\dots,\omega_n)\}, \omega_1<\cdots<\omega_n$, and $\leq$ is a partial order on the tuples. Let us consider an ordinal variable $Y = (\omega_1,..., \omega_k)$, where $y_{\pi(1)},... y_{\pi(m)}$ represent the mth smallest item after sorting $\Omega$. Then, the ordinal variable Y has k levels or labels, corresponding to each possible ordering of the items. The maximum level count corresponds to a completely ordered pairing, which means there exists no lower or higher ranked item. Therefore, the lowest possible level of the ordinal variable is considered to be level 1. If there are ties in any position, then all tied positions are assigned equal values. Mathematically, let $\tau$ be the set of all possible complete orderings of the items. For a specific tuple $(y_1,\ldots, y_k)$, the index $\pi(i)$ indicates the ranking of the $i$-th item. Therefore, we can write:
$$\pi(i) = |\tau\{j:y_j=i\}|$$
where $|\tau\{j:y_j=i\}|$ represents the total number of times $i$ appears as the $j$-th smallest item in any valid ordering $\tau$. Consequently, the level count of an ordinal variable Y is simply the sum of all $\pi(i)$ for all distinct values $i$ in $\Omega$. 

For example, suppose the set of items is {apple, orange, banana}. We want to define an ordinal variable over these items that assigns a numerical score to each item based on its popularity. Suppose the scores are: apple -> 3, orange -> 2, banana -> 1. This defines an ordinal variable Y with three levels. Now let's assume two more items are added to the list - carrot and kiwi, both of which receive a score of 5. The updated ordinal variable definition becomes:

item | score
---- | -----
apple    | 3  
orange   | 2  
banana   | 1  
carrot   | 5  
kiwi     | 5  

Now, the ordinal variable Y now has five levels, corresponding to each possible combination of the items' scores. There exist many possible orders of the items but two interesting ones are: 

1. First come, first served (FCFS): (apple, orange, banana, carrot, kiwi).

2. Last come, first served (LCFS): (carrot, kiwi, apple, orange, banana).

Each of these combinations are equally likely and imply different interpretations of the situation. The choice of ordering scheme will impact downstream analysis of the data.

## 2.4 Multi-way Splitting
Multi-way splitting involves partitioning a multi-way dataset into groups of objects based on various criteria. One common use case is grouping customers based on their purchasing behavior, where the criteria may involve age group, gender, location, and purchase history. Another example could be grouping computer programs by programming language, operating system, and application category. Here, the goal is to reduce redundancy in the data and identify groups that share similar characteristics. 

## 2.5 Multiple Kernel Learning
Multiple kernel learning (MKL) is a general framework that allows combining several kernels to capture complex interactions between features across different dimensions. MKL achieves this by training separate models for each kernel and integrating them together using meta-parameters learned during training. Various optimization techniques are available to learn the parameters of individual kernels and optimize their interaction. Two well known types of kernels that can be combined using MKL are linear and radial basis functions (RBF). 

# 3.NTF Methodology
## 3.1 Problem Formulation
Ordinal regression is a statistical task that involves predicting a target variable with respect to a discrete set of categories. However, ordinality typically requires defining a metric for measuring distances between successive points on the scale of the ordinal variable. Standard metrics like Euclidean distance or Manhattan distance cannot handle the fact that successive points do not necessarily have a unique ordering. Additionally, taking absolute differences between successive points may lead to inconsistent predictions since larger magnitude differences might occur because of more extreme situations. To solve this problem, we formulate ordinal regression as a constraint satisfaction problem (CSP) where we aim to minimize a loss function subject to certain constraints. Specifically, we assume that there exists a non-negative matrix $W$ of size d x n, where $d$ is the number of latent factors and $n$ is the number of samples, and a non-negative vector b of length n, representing the observed ordinal scores. Given the above assumptions, we seek to learn a model that maps the original feature vectors $X$ onto the latent space $W^\intercal X + b$ while minimizing the mean squared error between the predicted and actual ordinal scores. Moreover, the latent factors should reflect the strength of correlation among the ordinal scores while ensuring that they remain non-negative. Finally, we enforce the continuity property of the ordinal sequence by requiring that adjacent ranks differ by exactly one unit.

To perform this task effectively, we extend traditional NMF methods by introducing additional constraints related to ordinal variables. Specifically, instead of restricting the weights $w_k$ to be non-negative, we introduce two sets of coefficients $a_l,b_l$ and weights $c_l$ for each level l to model the probability distribution over the possible ranks that a sample can obtain at level l. Specifically, we assume that the rank probabilities at level l are given by a logistic sigmoid function, $$\sigma(z)=\frac{1}{1+e^{-z}}, z=a_l^T x+\beta_l.$$ The parameters $\beta_l$ are scalar biases applied to the logit transformation, and $\beta_l$ controls the tradeoff between the informativeness and smoothness of the model at each level.

Additionally, we modify the regularization term of traditional NMF by adding terms that penalize deviation from uniform distributions over the possible ranks. Specifically, we add a penalty term $\lambda_l \|w_l\|_1 + \rho_l \|v_l\|_1$ to the objective function, where $\lambda_l$ and $\rho_l$ control the relative importance of the L1 and L2 norm penalties respectively. Furthermore, we fix the bias parameter $\alpha$ to zero, making sure that the weights and biases remain non-negative throughout the entire process.