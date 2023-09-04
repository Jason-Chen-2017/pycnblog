
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Support vector machines (SVMs) are one of the most popular classification algorithms used today for both supervised and unsupervised learning tasks. In this article, we will discuss how support vectors work and their importance in SVM models. 

# 2.什么是支持向量机？
Support vector machine is a type of binary classifier that uses a hyperplane to split data into two classes. The hyperplane is chosen such that it maximizes the margin between the two classes while minimizing the errors within each class. SVM is widely used for image processing, text analysis, and pattern recognition applications. Here's an illustration:


In this example, there are two classes represented by circles and squares respectively, where dots represent training examples. A line can be drawn through these points to separate them into two regions with equal distances from the separating hyperplane. The area inside the dotted lines is called the "margin". As shown in the picture above, large margins give more freedom to the decision boundary and allow the model to classify new examples accurately. But too small or no margin leads to overfitting and poor generalization performance on new data. SVM solves this problem by finding the optimal hyperplane based on a trade-off between the margin size and misclassification errors.

Let's break down some key concepts related to SVMs:

1. Hyperplanes: A hyperplane is a straight line that splits space into two parts. We use hyperplanes as decision boundaries in SVM models to make predictions about new data.
2. Margin: The distance between the nearest points to the hyperplane defines the margin. It controls the width of the decision boundary created by the hyperplane. 
3. Support vectors: The training instances closest to the hyperplane are known as support vectors. They are responsible for fitting the hyperplane and contribute significantly towards building an accurate model. If any of these instances become outliers, the model may perform poorly.  
4. Soft margins: When using SVMs, the margin between the classes does not have to be strictly defined. Soft margins relax this constraint and allow the hyperplane to capture only the support vectors effectively disregarding any other irrelevant features. This makes soft margins useful when dealing with complex datasets that contain noise.  

Now let’s understand how support vectors help improve SVM models. 

# 3.什么是支持向量？
In traditional linear models like logistic regression and linear discriminant analysis (LDA), all the variables except for the ones corresponding to the target variable are considered independent. However, in real world scenarios, many features could potentially affect the output. Therefore, instead of ignoring such features, they can also be taken into account during modeling. Support vector machines do exactly that - they select just those features which are contributing most toward making correct predictions. These features are called support vectors. Let’s see why supporting these important features has significant effect on SVM models. 

Suppose we have a dataset consisting of three features, X1, X2, and X3. For simplicity, assume that the target variable y belongs to two categories, say 'Class-1' and 'Class-2'. Our goal is to train a SVM model so that it can correctly predict whether a new instance belonging to Class-1 or Class-2 based on its values of X1, X2, and X3. To achieve this, we can define our hyperplane as follows:

hyperplane = W1 * X1 + W2 * X2 + W3 * X3 = 0
  
where,  
W1, W2, and W3 are the weights assigned to each feature along the hyperplane. 
We want to maximize the margin between the hyperplane and the closest point to the plane i.e., the maximum value of Σ(wx - y)^2, where x is a new instance to be classified and wx is the dot product of w and x. One way to optimize this function is to take derivative with respect to W1, W2, and W3. This gives us the direction in which we need to move our hyperplane to increase the margin. Once we find the minimum gradient descent step for our objective function, we get:

δ = [w1_new - w1; w2_new - w2; w3_new - w3] / ||[w1_new - w1; w2_new - w2; w3_new - w3]||
 
This formula says that if we want to decrease the margin by changing the coefficients (W1, W2, and W3), then we should change them in the direction of δ. Hence, moving in this direction would increase the margin between the hyperplane and the closest point to the plane. However, since we don't want to cross the hyperplane, we want to minimize ||δ|| or Σδ^2. Mathematically, this can be expressed as:

min { Σy*(Wx) } subject to Σy*(Wx)*x ≥ 1 - γ min{Σ(Wx)^2}, for γ > 0

where,

y ∈ {-1,+1}, |y| = 1    (Binary labels).

Now, consider a new instance, x', whose label is unknown. We want to determine its membership to either Class-1 or Class-2 based on the given training set. Since we know that the hyperplane separates the two classes and the closest point to the hyperplane can only lie outside the margin, we cannot simply compute the dot product of the weights and the new instance x'. Instead, we introduce slack variables s1 and s2 associated with each sample, which allow the hyperplane to slide outside the margin. Mathematically, the new prediction rule becomes:


decision = sign([−(W1 * X1') + s1 ; −(W2 * X2') + s2 ; −(W3 * X3') + s3]) 
= sign(-((W1 * X1') - s1))
 


If y' = +1, s1 >= 0 and s2 <= 0. Similarly, if y' = -1, s1 <= 0 and s2 >= 0. In other words, the samples located at the extremes of the margin are given higher priority than the samples located near the center. Thus, once we identify the support vectors using the kernel trick, we can discard the non-support vectors easily without affecting the accuracy of our model.