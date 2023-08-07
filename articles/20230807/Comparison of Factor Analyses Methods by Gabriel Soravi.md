
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1.1 专题介绍
         
         "Comparison of factor analyses methods" is a topic that attracts many researchers and engineers in various fields due to its wide application scope, high impact on the field, and diverse development directions. Here we are going to compare several factor analysis techniques based on their advantages and disadvantages. We will introduce common concepts used in these techniques such as data matrix, rotation of factors, latent variables, canonical variates, common patterns, and interpretation of factors. 
         1.2 Objective
         The objective of this blog post is to provide an overview of common concepts related to factor analysis including but not limited to data matrix, rotation of factors, latent variables, canonical variates, common patterns, and interpretation of factors. Furthermore, it should help readers understand the pros and cons of different factor analysis techniques, so they can select one technique that best fits their needs. In addition, through detailed explanations and illustrative examples, this article aims to educate readers about factor analysis. Moreover, the authors also aim to promote healthy discussions between practitioners, technologists, and researchers in order to improve our understanding of this fascinating topic. Finally, we hope that this article will lead to fruitful collaborations and further developments in the field of factor analysis.
         1.3 Methodology

         This paper presents five main contributions: i) Reviewing some important terminologies used in factor analysis, ii) Exploring the concept of data matrix and presenting a brief explanation of each term involved with it; iii) Describing different types of rotations possible for factor analysis, iv) Introducing basic concepts of canonical variates and giving an example; v) Providing insights into what makes up a typical factor analysis model and how to interpret results obtained using PCA or FA techniques. 

         2 Conceptualization
         In factor analysis (FA), we have two entities – observations/samples and variables/factors. Each variable or factor can be thought of as contributing to a set of unobserved variables called “latent variables” or “factor scores”. Latent variables represent unknown underlying structures in the observed dataset which contribute towards explaining the variance of the original variables. Using these latent variables, we can then use them to predict future values of the original variables. The process of obtaining meaningful latent variables from observed datasets involves four major steps: 

1. Data Preparation: It involves cleaning, transforming, and preprocessing the raw data before performing any analysis.

2. Data Transformation: After preparing the data, the next step is to normalize or standardize it. Normalization refers to scaling all variables to lie between zero and one while Standardization refers to scaling the variables to have mean zero and unit variance.

3. Determining the Number of Factors: Based on the nature of the data, we determine the number of factors needed for modeling the relationships among the variables. Higher-order factors can often capture more complex dependencies than low-order ones.

4. Rotating Factors: Once the factors are determined, we need to rotate them such that they have maximum correlation with each other. The rotated factors become the basis vectors against which we project the original variables onto. There are multiple ways to perform factor analysis, ranging from simple principal component analysis (PCA) to advanced multivariate analysis techniques like partial least squares regression (PLS).

         3. Basic Terminology
         Before proceeding with detailed descriptions, let’s first discuss some fundamental terms associated with factor analysis. These include:

         **Data Matrix**: A data matrix is simply a matrix containing the observed variables along with their corresponding measurements or responses. For example, if there are N samples and P variables, then the data matrix D would consist of N rows and P columns.

         **Factor Score Vector**: A factor score vector represents the projection of the observation onto the selected direction or factor. It contains only numeric values and has no interpretable meaning unless it is combined with additional information.

         **Latent Variable**: A latent variable is an attribute that contributes towards explaining the variance of the original variables. It does not directly influence the measured value of the variable. Instead, it influences the behavior of the system by causing changes in the output.

         **Rotation of Factors**: Rotation refers to the process where new base vectors are chosen such that they maximize the correlation between the existing bases and the remaining variables after accounting for the effects of removed dimensions. Two popular approaches are SVD and QR decomposition.

         **Canonical Variates**: Canonical variates are variables whose variances are equal across all components of the resulting factor analysis model. They are typically defined as linear combinations of the original variables. 

         4 Data Matrix
         4.1 Introduction
         
         The first step in factor analysis is to prepare the data matrix X. The data matrix D consists of N rows and P columns, where N represents the number of samples and P represents the number of variables. 

         If there are missing values or outliers in the data, they need to be imputed, otherwise the algorithm may not converge properly. Additionally, we might want to check whether the variables are normally distributed or follow some specific distribution. The Box-Cox transformation, logarithmic transformation, or range normalization can be applied to achieve normality.

         4.2 Columns Correlation Structure Analysis
         
         To analyze the correlations structure of the data matrix X, we compute the pairwise correlation coefficients between every pair of variables. By plotting the correlation coefficient matrix, we can detect clusters of highly correlated variables and decide which groups of variables are redundant or informative. Redundant variables carry little information and could potentially be removed during feature selection. Informative variables contribute significantly to the explained variance of the data.

         4.3 Rows Covariance Structure Analysis
         
         Next, we examine the covariance structure of the data matrix X by computing the sample covariance matrices of individual samples or cases. In particular, we look for large positive semidefinite submatrices within each sample covariance matrix. Large positive semidefinite matrices indicate strong multicollinearity among the variables in that case, suggesting the need for variable reduction.

         5 Rotational Mechanisms
         
         Once we have prepared the data matrix, we need to determine the optimal number of factors k to explain the largest amount of variance in the data. Typically, we start with a small number of factors k = 1, 2, or 3, and increase k until we obtain satisfactory results. 

         Since the dimensionality of the data increases with k, we cannot visualize all k factors simultaneously. Therefore, we need to choose a subset of factors to display at once. One way to do this is to consider the loadings or weights of each factor, which measure the importance of each latent variable relative to the others. The magnitude of the loading indicates the degree of contribution of the corresponding variable to each factor. When we plot the weight vectors, we observe a tradeoff between the proportion of total variance captured by each factor and its intrinsic dimensionality. As a result, we need to balance the strengths of both properties when selecting the number of factors.

         6 Types of Rotations
         
         Four types of rotations are commonly used in factor analysis: canonical rotations, quasi-orthogonal rotations, oblique rotations, and modified oblique rotations. In the following sections, we'll explore these methods in detail.

         7 Canonical Rotations
         
         The most straightforward approach to rotation is canonical rotation. It means constructing orthonormal basis vectors consisting of the first k eigenvectors of the sample covariance matrix of X. Each factor score vector zi is then constructed by multiplying xi by the corresponding column vector vi.

         8 Quasi-Orthogonal Rotations
         
         Another useful method of rotation is the quasi-orthogonal rotation. In this case, we assume that the factors form a valid orthogonal set, and apply Gram-Schmidt orthogonalization to construct them. We then calculate the adjusted coordinates of each observation based on the factored bases, allowing us to reconstruct the original variables without losing any information.

         9 Oblique Rotations
         
         An alternative approach is to use oblique rotations. In this case, we find an orthogonal matrix T such that T*T' = I, where I denotes the identity matrix. Then, we apply T to the data matrix X and extract the same set of factors as with canonical rotations.

         10 Modified Oblique Rotations
         
         Our final rotation mechanism is the modified oblique rotation. In this case, we again define an orthogonal matrix T and apply it to the data matrix X, producing new factor score vectors zi'. However, instead of applying the same T to all factors, we adjust the rotation matrix T_j for each factor j separately according to certain criteria. 

         The simplest criterion is the square root of eigenvalues of the cross-covariance matrix Cj = E[x'z']*inv(E[zz']) - I. This criterion ensures that each factor approximately forms a plane perpendicular to the other factors and captures nearly half of the total variation. The adjustment matrix T_j is computed as T^(-1)*(cj'*sqrt(cj)), where * denotes the Hadamard product.