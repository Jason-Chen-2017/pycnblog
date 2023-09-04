
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 What is convexity?
Convexity in optimization refers to the fact that a function or curve can be made into a single line segment between any two points without going below it on either side, even if there are infinitely many other points along its way. It means that all local minima are also global minima, and they form a convex set which surrounds the minimum value with zero curvature at every point within. 

The word "convex" has several meanings:

1. In geometry, convex shapes have no holes or intersections except those between their edges; this property makes them ideal for physical applications such as welding and printing machines. 

2. In economics, convex functions provide a simple, mathematically tractable model for representing supply and demand curves. By studying these functions, analysts can determine when markets will be efficient, how much profits could be obtained, and what the equilibrium price should be. 

3. In operations research and computer science, convex optimization problems arise from various fields such as machine learning, signal processing, computer vision, and finance. These problems involve solving for an optimal trade-off among different objectives or constraints based on some measure of risk.

## 1.2 Why do we care about convexity?
We care about convexity because convex functions possess certain desirable properties that make them useful in a wide range of optimization contexts. Some of these include:

1. Concave functions (also called concave downwards) cannot reach their lowest point, so they may not have a unique global minimum. However, convex functions always have one and only one global minimum, making them very important in many practical situations where you need to find the best solution.

2. The shape of a convex function's graph is typically flat at its optima, whereas in general convex functions can have kinks or bumps in their graphs. This allows us to identify and exploit certain features of our problem domain that contribute most to the objective function's value. For example, if we're looking for a maximal margin classifier in support vector machines (SVM), we might want to focus on regions where positive and negative examples are relatively far apart.

3. If we know that the solutions to a given optimization problem are contained within a certain region defined by a convex function, then we can use more advanced techniques like gradient descent algorithms to search for better solutions. Gradient descent works well when the objective function is both continuous and differentiable, but often times it becomes difficult to apply it directly to non-convex optimization problems due to their high-dimensional nature. Instead, we can use methods like Lagrangian multipliers or barrier functions to convert the non-convex problem into a sequence of convex subproblems whose solutions we can combine using the standard approach of gradient descent.

4. Many optimization problems can be converted into convex ones by introducing auxiliary variables. For instance, suppose we want to minimize the sum of squares $f(x)=\sum_i x^2$. One possible way to rewrite the problem as a quadratic programming problem is to introduce an auxilary variable $\xi=\sqrt{\sum_i x^2}$ and minimize $g(\xi)=\xi$, subject to the constraint that $g'(x)\geq 0$ everywhere, i.e., $(x-\bar{x})^T Q^{-1} (\frac{x-\bar{x}}{\norm{x-\bar{x}}} - \frac{\bar{x}-x}{\norm{\bar{x}-x}}) = \left<Q^{-1}\nabla f(\bar{x}), \frac{x-\bar{x}}{\norm{x-\bar{x}}} \right>_{\infty}=0$. Then the original problem can be equivalently written as minimizing $\xi$ while ensuring that each component of $x$ satisfies $g'(x)\leq 0$, giving rise to a new convex optimization problem with the same optimal solution. This technique applies to a variety of problems in statistics, econometrics, and operation research, including maximum likelihood estimation and portfolio optimization.