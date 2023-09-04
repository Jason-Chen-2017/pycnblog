
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是动态规划？
>Dynamic Programming, also known as DP for short, is a technique used to solve problems by breaking them down into smaller sub-problems and storing the results of those sub-problems so that they can be reused later on in the solution process. Dynamic programming is mainly used when there are overlapping sub-problems in solving an optimization problem or if optimal solutions to each sub-problem has already been computed. The goal of dynamic programming is to find out the optimum solution for a given problem. It uses memoization method which stores the intermediate results of function calls and avoids redundant computations.

动态规划（英文缩写 DP）是一个在计算机科学中应用较为普遍的求解问题的方法，它主要利用子问题的递推关系来解决一个大型问题。动态规划往往会用到备忘录方法来存储中间结果，避免重复计算，从而达到优化的问题求解过程中的最优解。

## 一维动态规划和二维动态规划之间的区别是什么？
“一维”和“二维”动态规划有什么不同呢？为什么要使用二维动态规划而不用一维动态规划呢？下面我们就来看一下。

### 一维动态规划
>In simple terms, one dimensional dynamic programming refers to a type of dynamic programming where only one variable is involved i.e., it deals with decision making problems where we have to choose between different options based upon some criteria. 

简单来说，一维动态规划就是指一种单变量动态规划，它用于处理与某个变量相关的决策问题。例如，在买卖股票、石子游戏等场景中，我们都可以把它作为一维动态规划来做。

### 二维动态规划
>Two dimensional dynamic programming, on the other hand, involves using two variables – either variables that represent state information or decisions taken at different stages of the game - to formulate a decision problem. In this approach, the choices made along one axis determine the outcome of the subsequent stage, hence the name "two dimensional". 

另一方面，二维动态规划则是利用两个变量来表示状态信息或不同的阶段所作出的决策，这种方法主要用于形成决策问题。在这种方法下，沿着某一轴的选择将决定之后的一阶段，因此它的名称是“二维”。

### 为何需要二维动态规划？
两者之间存在着本质的差异。由于“一维”动态规划只能解决一种场景，所以它很难处理那些需要考虑多种情况的复杂问题。例如，在背包问题中，一维动态规划并不能直接用来解决，因为我们不知道最终选定哪几个物品才能获得最大价值。但是，如果我们把这个问题转换成二维的形式，就可以通过考虑每个物品选择与否以及选择时所处的状态，来找到一种全局最优的解决方案。这样的话，“二维”动态规划就能够解决这一类问题。

### 什么时候应该使用一维动态规划，什么时候应该使用二维动态规划呢？
在实际应用中，无论是在一个还是多个变量的条件下，都会存在很多场景需要使用动态规划，但是具体采用哪一种方式还需要依据具体的需求和限制。下面，我会结合一些实际例子来阐述二维动态规划的作用及其适应场景。

## 二维动态规划的几种常见场景
以下是二维动态规划在实际场景中的应用：

1. 股票交易问题
股票交易问题是一个典型的二维动态规划问题。给定一个长度为n的数组prices，其中第i个元素代表第i天的股票价格，再给定一个整数k，代表交易次数。假设第i天交易的价格为p_i，那么我们希望在第j次交易结束后手上持有股票的价格为max(p[j]+p[i])，也就是说，我们希望在这次交易结束后买入的股票价格高于之前任何一次交易的价格。如何在给定的交易次数内完成最大收益呢？这是一道具有挑战性的组合优化问题。

2. 编辑距离
编辑距离，又称Levenshtein距离，是指两个字符串之间由一个转化成另一个所需的最少操作次数。动态规划有广泛的应用，包括字符串匹配、序列对齐、文本编辑、词袋模型等领域。在文本编辑问题中，我们希望在两个字符串之间找到一条经过尽可能少次操作（插入、删除、替换）的路径，使得这两个字符串相等。同样地，如何找出一条经过最小数量的操作使得两个字符串相等也是一项具有挑战性的组合优化问题。

3. 矩阵链乘法
矩阵链乘法是另一种典型的二维动态规划问题。给定一个由n个矩阵组成的序列，其中第i个矩阵Ai*Bj，其中Ai*Bij是第i个矩阵与所有其他矩阵相乘的结果，以及一个系数序列p=[p1,p2,...,pn]，其中pi代表第i个矩阵的秩。我们的目标是找到一个最小代价的顺序，用这n个矩阵相乘，即C=A1*B1+...+An*Bn。如何找到这样一个最小代价的顺序，也是一道具有挑战性的组合优化问题。

4. 旅行售货机问题
旅行售货机问题是指有一个售货船和N个商品，每个商品有自己的重量和价值。我们希望在不超过背包容量M的情况下，选择一些商品，将他们放入背包，且总重量不超过背包的总重量。同时，所选择的商品不能超过M件，并且希望尽可能高效地完成售卖。这是一个具有挑战性的多项式时间多重集合优化问题。

## 参考资料
- https://en.wikipedia.org/wiki/Dynamic_programming
- Leetcode