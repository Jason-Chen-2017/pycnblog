
作者：禅与计算机程序设计艺术                    

# 1.简介
  

TOPSIS（Technique for Order Preference by Similarity to Ideal Solution）是一个衡量指标选择方法，由苏哈托·多明戈斯（Shantanu Dameji）提出。该方法最早应用于市场营销领域，其优点是能够根据各个目标的权重，对决策者的建议进行排序。

TOPSIS是一个模糊综合评价方法，它适用于目标及其相互矛盾的问题。通过将所有目标按与最优解的距离进行比较，可以判断哪些目标更重要。距离越小，就意味着相应的目标越重要。TOPSIS可用于处理多目标优化问题，如产品配置、项目调度等。

在本文中，我会为读者提供一个基于Python语言的实现TOPSIS的方法。希望对大家有所帮助！

# 2. Concepts and terminologies introduction
## 2.1 Problem statement definition
Let’s consider a multi-criteria decision making problem where there are multiple alternatives with different criteria to be evaluated. We need to select the alternative that has highest satisfaction to all criteria at the same time while ensuring that no one criterion is completely ignored or reduced due to any factor other than the selected alternative. In other words, we need to maximize the Pareto frontier, which means an area within a set of points where no point lies above another point on the line connecting them. Mathematically, it can be written as follows:

min (max(z_i - xi + u_i,0) / w_i)^+, i = 1 to n 

where z_i denotes the weighted sum of each criterion for alternative i, xi denotes the preferred weight assigned to each criterion, u_i represents the unfairness or penalty value associated with alternative i, and w_i represents the importance given to each criterion respectively. The + sign after the maximum function ensures that if all criterias have negative impact then the denominator will become zero and therefore, the score will not change and remains constant equal to infinity. This method assigns higher preference values to solutions that satisfy most criteria but do not satisfy some. Therefore, it considers trade-offs between different criteria rather than just choosing the best solution based on a single criterion alone.
## 2.2 Criteria and weights assignment
In this section, we will define the specific criteria to be used and their corresponding weights according to our preferences. Let's assume we want to evaluate three criteria A, B, and C. For each criterion, we have to assign a positive integer value representing its importance. Suppose, A has a higher weighting factor compared to B and C while B and C both have similar weighting factors. Then, the corresponding weights for these criteria would be:

Weight for A = 3

Weight for B = 2

Weight for C = 1

The larger the absolute value of a weight, the greater the influence it has on determining the ranking order among alternatives. If two or more criteria have the same weight, they may perform equally important roles during evaluation.
## 2.3 Unfairness penalties
If there exists any alternative whose satisfaction levels are significantly lower than those of other alternatives, it becomes difficult to compare such alternatives accurately using only objective criteria. To handle this issue, we use the concept of unfairness or penalty. It specifies the degree of discrimination against certain groups of criteria. More specifically, the lower the unfairness penalty, the greater the impact it has on evaluating the overall quality of each alternative. An unfairness penalty value typically ranges from 0 to ∞. Let's assume that we don't want to allow any particular group of people to receive a high unfairness penalty value, so let's give them a very low penalty value (e.g., 0.01). However, we also want to make sure that no one gets an extreme unfairness penalty because of overly strict standards. Therefore, we might choose to adjust the penalties accordingly depending on the context and goals of the analysis.