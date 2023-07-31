
作者：禅与计算机程序设计艺术                    
                
                
Local Linear Embedding (LLE) is a well-known manifold learning method that has been widely used in various fields such as image processing and bioinformatics to reduce the dimensionality of high dimensional data sets. It works by embedding the high-dimensional input points onto a low-dimensional manifold where each point's neighbors are closely packed together. LLE achieves this goal using a set of linear transformations whose outputs follow Gaussian distribution. In contrast to other manifold learning methods like Isomap or t-SNE which rely on pairwise distances between points, LLE models the local relationships among all the points based on their geometric structure and distance distributions. 

Recently, deep neural networks have shown impressive performance in solving complex tasks like object recognition, speech recognition, and natural language understanding. However, they may struggle with large and high-dimensional inputs due to memory limitations and slow convergence time. To address these issues, researchers proposed techniques like batch normalization, dropout, and stochastic depth to improve generalization ability of DNNs. But none of these techniques consider the underlying geometry of the input space, let alone its nonlinear dependencies.

In this article, we will explore the inner workings of LLE from an algorithmic perspective and discuss some insights obtained through experiments on popular datasets like MNIST and CIFAR10 to identify potential applications in modern machine learning systems. We hope that our analysis can shed some light on the current state-of-the-art approaches and guide future research directions towards more robust deep neural network training and inference processes. 

# 2.基本概念术语说明
## Manifold Learning
Manifold learning is a class of unsupervised machine learning algorithms that aim to find a low-dimensional representation of data while preserving the topology or geometry of the original space. It involves finding a mapping function $\phi$ that maps the input variables $X \in R^m$ to a lower-dimensional space $Y \in R^n$, such that the points that are close to each other in the original space are mapped closer to each other in the new space. The most commonly used approach is Locally Linear Embedding (LLE), which uses a small number of basis vectors to represent a high-dimensional space. 

## Linear Transformation
A linear transformation is any transformation that preserves lines and angles within Euclidean space. Given a matrix A ∈ R^{m×n}, a vector x ∈ R^n, and another vector y = Ax, then y is called the transformed vector. We denote the set of linear transformations acting on a given vector space X as T(X). For example, if X is a two-dimensional Euclidean space R^2, then T(R^2) refers to the set of matrices A that transform column vectors into row vectors. If A transforms vectors v into w, then the composition of A and B gives us A′B, which transforms v into w under the condition that both A and B preserve lines and angles. 

### Orthogonal Matrix
An orthogonal matrix is a square matrix with unit determinant and zero trace. Any n × n orthonormal matrix is also considered to be an orthogonal matrix. Specifically, it means that if U is an orthonormal matrix, then U−1U^TU = I. 

### Span
The span of a set of vectors in a vector space is the set of all possible linear combinations of those vectors. Specifically, the span of {v₁,…,vn} is the subspace of R^n consisting of all linear combinations of vi with coefficients cij = 0, 1,..., n-1. Thus, any vector of R^n lies in the span of at least one subset of its components, and we call the smallest such subset the base of the span. The union of several bases forms a higher dimensional space than any single basis, but each individual basis itself only spans a smaller dimensional subspace. Mathematically, the span of a finite collection of vectors is the intersection of their bases, so there may be infinitely many bases in total. By considering the base vectors that are not spanned by others, we obtain a basis for the complementary subspace that is spanned by the rest of the vectors. 

## Distance Measurements
We use different types of distance measures to define the similarity between pairs of points in high-dimensional space: 

1. Manhattan Distance: This measure calculates the sum of absolute differences along each coordinate axis.

2. Euclidean Distance: This measure calculates the square root of the squared difference between corresponding coordinates of two points.

3. Cosine Similarity: This measure calculates the cosine of the angle formed between two vectors projected into a common subspace.

## Neighborhood Graph 
The neighborhood graph of a dataset consists of vertices representing the data points, and edges connecting adjacent data points within a specified radius. The edge weights indicate the similarity of the two connected data points based on the chosen distance metric. Commonly used neighborhood graphs include k-nearest neighbor (KNN) graphs, Radius Neighbors (RadiusNN) graphs, and Fixed Radius NN (FRNN) graphs.

## Global Linear Model
Suppose we have a dataset D = {(xᵢ,yᵢ)}_{i=1}^N, where xi ∈ R^m represents the features of the i-th observation and yi ∈ R represents the target variable. The global linear model attempts to fit a linear regression model to predict the target value ŷi as a linear combination of the features xᵢ, i.e., ŷi = βˆT xᵢ, where βˆ is the estimated parameter vector. The global linear model assumes that the relationship between the feature values and the target variable follows a linear relationship across all observations. 

## Subspace Projection
Given a matrix A ∈ R^{d×n} and a vector b ∈ R^d, the projection of b onto the range of A is the solution to the equation AA'b = b. Denoting the projection operator P_A(·), we write bP_A(·) := P_A(b) and say that b projects onto the range of A. Similarly, given a matrix B ∈ R^{k×d} and a vector c ∈ R^k, the projection of c onto the kernel of B is the solution to the equation BBc = c. Denoting the projection operator P_B(·), we write cP_B(·) := P_B(c) and say that c projects onto the kernel of B.

## Principal Component Analysis (PCA)
PCA is a technique used to decompose a multivariate dataset into a set of uncorrelated variables that explain a maximum amount of variance in the data. It works by projecting the data onto a hyperplane that maximizes the variance along each direction. PCA finds the eigenvectors of the covariance matrix of the data and chooses the ones with largest eigenvalues as the principal components. Then, it scales the dimensions of the data accordingly and reconstructs them back into the original space. Here, we choose to take the first m principal components as the final representations, where m << d. Intuitively, we want to capture the most important features of the data without losing too much information in terms of noise or outliers.

