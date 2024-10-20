
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 动态规划（Dynamic Programming）是一种高效且灵活的算法解决方案，用于最优化计算、组合优化或最优化控制问题。它旨在通过对复杂问题的子问题进行积累，一步步地推导出一个带有最优值或者最优策略的解。动态规划被广泛应用于电路设计、资源分配、图形处理等领域。例如，线性规划、分组规划、任务调度、生产计划、最短路径等都可以使用动态规划方法求解。

           在生物信息学中，动态规划也经常作为求解问题的工具，比如序列比对、基因识别、结构预测、蛋白质结构预测等。特别是在序列比对领域，DP算法可以有效地找到一条“最优”的序列比对路径，而不需要费力求取所有可能的比对方案。

           本文将着重阐述DP在生物信息学中的一些应用及其背后的基本概念。

         # 2.基本概念术语说明
         ## （1） 无后效性
         在动态规划中，即使对于同样的输入，一个解法的计算可能会花费多次，但它的结果总是唯一的。换句话说，就是只要给定了某些输入，那么就一定会有一个唯一的输出。这叫做动态规划的无后效性。

         ## （2） 滚动数组/表格
         动态规划算法通常使用滚动数组/表格的方式存储中间结果，避免重复计算，从而提高算法性能。简单来说，就是用两个变量分别指向当前位置和之前的位置的数组，然后按顺序填充表格中的元素。
         
            int f[MAXN]; //表示存放最大值

            memset(f, -INF, sizeof(int)*MAXN); //初始化为-INF

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    // 更新f[j]的值...
                }
                swap(f, g); //更新数组指针
            }

          上面的例子展示了一个典型的滚动数组的形式，其中f[j]表示当前位置前j个字符组成的字符串的最长回文串长度。当i增大时，f指针也随之增加，保证始终保留最新计算出的元素。这一策略能够避免由于填满整个数组而导致空间不足的问题。

       ## （3） 状态转移方程
       动态规划的核心思想是利用子问题的最优解构造出全局的最优解，这种情况下状态转移方程一般遵循以下规则：

        dp[i][j] = max(dp[i-1][j], dp[i-1][j+1]) + cost(i,j) //cost函数表示当前位置到最远距离的代价

      上式表示根据当前位置（i,j）的上下左右四个位置的不同情况选择一个值，所选值的意义是判断以(i,j)为中心的最长回文字串的长度。cost函数用来计算代价，也可以理解为转移时的额外开销。
     
       可以看出，状态转移方程是一个二维数组，为了方便计算，需要满足三个条件：
        * 第一行和第一列表示的是初始状态；
        * 每一个元素只能由上面的两种状态转移得到；
        * 不能用自身或之前的状态计算，只允许依赖先前已经计算好的状态。
       
       当然，实际上这种限制并不是严格的，因为往往可以通过一些技巧消除掉冗余计算。


     # 3.核心算法原理和具体操作步骤以及数学公式讲解
     ## （1） 最长公共子序列（LCS）
      LCS问题可以描述为寻找一个字符串S1和另一个字符串S2之间的最长公共子序列。这是个具有代表性的动态规划问题，它经常出现在生物信息学领域，如序列比对、基因识别等。
      
      ### 动态规划的求解过程
      首先，设dp[i][j]为S1的第i个字符和S2的第j个字符匹配的最长子序列的长度。

      如果S1的第i个字符等于S2的第j个字符，那么我们只需考虑前两者之间匹配的部分，即`dp[i-1][j-1]`；

      如果它们不同，我们就只能从前者选或者从后者选，选哪个呢？如果从前者选的话，则需要考虑与前者相同的字符，那么还剩下n-1个字符，因此应该是`dp[i-1][j]`，但前者是包括第i个字符的，而后者是不包括第i个字符的，所以还需要加上1；如果从后者选，则类似的道理。

      综合上面两点，得：

      `dp[i][j] = max(dp[i-1][j], dp[i-1][j-1])+1 if S1[i]==S2[j] else max(dp[i-1][j], dp[i][j-1])`

      最后，为了找出最长公共子序列，我们需要从倒数第二行最后一个元素开始向上回溯，每次记录一个匹配字符，最终得到的序列就是这个最长公共子序列。

      ### 例子

      比如，我们要找出字符串"ABCDGH"和字符串"AEDFHR"之间的最长公共子序列，对应的LCS矩阵如下：

      |   A  |  B   |  C   | D    | G    | H    |
      |------|------|------|------|------|------|
      |      | 0    | 0    | 0    | 0    | 0    |
      |  A   | 1    | 0    | 0    | 0    | 0    |
      | E D F| 0    | 0    | 0    | 0    | 0    |
      | H R  | 0    | 0    | 0    | 0    | 0    |
      |     R| 0    | 0    | 0    | 0    | 0    |

      从上面的矩阵可以看到，只有矩阵中斜线右侧的格子有值，其他都是0。所以，最长公共子序列为："ADH"，对应于矩阵中的位置是"E D H"。

     ## （2） 最短编辑路径（SED）
     SED问题可以描述为确定一个字符串S1和另一个字符串S2之间的最小编辑路径长度。

      ### 动态规划的求解过程
      与LCS类似，假设dp[i][j]为S1的第i个字符和S2的第j个字符之间的最小编辑路径长度，这个长度可以表示成下面的三种方式之一：

      （1）把S1[i]替换为S2[j]的最小编辑路径长度，为`dp[i-1][j-1]+1`。
      （2）把S1[i]删除的最小编辑路径长度，为`dp[i-1][j]+1`。
      （3）把S2[j]插入的最小编辑路径长度，为`dp[i][j-1]+1`。

      因此，

      `dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])+1`

      也就是说，每一步都有三种选择，取决于当前位置是否需要替换、删除、还是插入。注意这里的加号需要变成负号才能表示消耗的代价。

      最后，为了找出最短编辑路径，我们可以沿着矩阵的左上角方向，逐步地回溯。

      ### 例子

      比如，我们要找出字符串"ABCDEFG"和字符串"ACDEBGF"之间的最短编辑路径，对应的SED矩阵如下：

      |   A  |  C   |  D   | E    | B    | G    |
      |------|------|------|------|------|------|
      |      | 7    | 6    | 5    | 4    | 3    |
      |  A   | 6    | 5    | 4    | 3    | 2    |
      | C DE BF| 5    | 4    | 3    | 2    | 1    |
      |   G F | 4    | 3    | 2    | 1    | 0    |
      |       | 3    | 2    | 1    | 0    | INF  |

      从矩阵中可以看到，编辑路径的最小代价值为1，对应于矩阵中的位置是'C D E B F G',反映了这样一个操作序列：

      （1）把'D'替换为'B';
      （2）把'E'删除;
      （3）把'B'插入;
      （4）把'C'插入;
      （5）把'D'插入。

      因此，最短编辑路径为1，但这个序列也是最短的。

      # 4.具体代码实例和解释说明
      接下来，我将给出两个关于序列比对的实际例子，并与大家一起探讨相应的DP算法。

      ## 例1——序列比对
      ### 问题描述
      已知两个序列$s_1=(s_{11},s_{12},\cdots, s_{1m})^T$和$s_2=(s_{21},s_{22},\cdots, s_{2n})^T$, 求它们之间的最长公共子序列$lcs(s_1,s_2)$。
      
      $s_1=(s_{11},s_{12},\cdots, s_{1m})^T=\{a_1, a_2,\cdots, a_m\}$ 和 $s_2=(s_{21},s_{22},\cdots, s_{2n})^T=\{b_1, b_2,\cdots, b_n\}$, 求$\max \limits_{\substack{1\leqslant i\leqslant m \\ 1\leqslant j\leqslant n}} \{d_{ij}\mid d_{ij}=s_{1i}-s_{2i}     ext{ or } (a_i=b_j)\}$。

      意义：判断两个字符串序列是否相似，计算相似度。相似度计算方法为取两个字符串序列任意两个字符差值的绝对值，若差值为0则对应位置字符相同，此时取这两个位置字符的差值的最大值。

      举例：

      比较"AGGTAB"和"GXTXAYB", 他们的相似度为: $\left\{|\operatorname*{abs}(9-7)|+\operatorname*{abs}(4-2)|+\operatorname*{abs}(13-11)|+\operatorname*{abs}(7-9)|+\operatorname*{abs}(8-7)|+\operatorname*{abs}(8-8)|+\operatorname*{abs}(7-7)|+\operatorname*{abs}(8-8)|+\operatorname*{abs}(7-7)|+\operatorname*{abs}(1-2)|+\operatorname*{abs}(6-5)|+\operatorname*{abs}(5-4)|\right\}$=$6$.


      定义$f[i][j]$表示两个序列的前$i$个字符组成的子序列和$s_2$的前$j$个字符组成的子序列的最长公共子序列的长度。

      $$
      f[i][j]=\begin{cases}
      0,&i=0\\
      0,&j=0\\
      f[i-1][j-1]+1 &\quad     ext{(if }s_1[i]=s_2[j]    ext{)}\\
      \max \Bigg(\{f[k][j]\mid k<i\wedge s_1[k]=s_2[j]}\Bigg)+1&\quad     ext{(otherwise)}\end{cases}\\
      k=0,1,\cdots,i-1
      $$

      ### 分析
      1. 输入：两个序列$s_1,s_2$，长度分别为$m,n$。
      2. 计算量级：$O(mn)$。
      3. 需要返回$f[m][n]$。
      4. 根据提示，我们构建动态规划的状态转移方程如下：

      ```python
      f[i][j]=max(f[i-1][j],f[i][j-1]) if s1[i]!=s2[j] else f[i-1][j-1]+1
      return f[m][n]
      ```
      此处使用了滚动数组的方法，使得空间复杂度降低至$O(min(m,n))$。

      ## 例2——序列比对优化
      ### 问题描述
      已知两个序列$s_1=(s_{11},s_{12},\cdots, s_{1m})^T$和$s_2=(s_{21},s_{22},\cdots, s_{2n})^T$, 求它们之间的最长公共子序列$lcs(s_1,s_2)$。
      
      $s_1=(s_{11},s_{12},\cdots, s_{1m})^T=\{a_1, a_2,\cdots, a_m\}$ 和 $s_2=(s_{21},s_{22},\cdots, s_{2n})^T=\{b_1, b_2,\cdots, b_n\}$, 求$\max \limits_{\substack{1\leqslant i\leqslant m \\ 1\leqslant j\leqslant n}} \{d_{ij}\mid d_{ij}=s_{1i}-s_{2i}     ext{ or } (a_i=b_j)\}$。

      次数优化：采用两层循环，计算每个字母出现次数，减少计算量。

      举例：

      比较"AGGTAB"和"GXTXAYB", 使用次数优化后的相似度为: $\left\{|\operatorname*{abs}(9-7)|+\operatorname*{abs}(4-2)|+\operatorname*{abs}(13-11)|+\operatorname*{abs}(7-9)|+\operatorname*{abs}(8-7)|+\operatorname*{abs}(8-8)|+\operatorname*{abs}(7-7)|+\operatorname*{abs}(8-8)|+\operatorname*{abs}(7-7)|+\operatorname*{abs}(1-2)|+\operatorname*{abs}(6-5)|+\operatorname*{abs}(5-4)|\right\}$=$6$.

      定义$c[k]$表示字符$k$在$s_1$中的出现次数，定义$d[k]$表示字符$k$在$s_2$中的出现次数。

      $$
      c[k]=\{i\in [1,m]\mid s_1[i]=k\}
      $$
      $$
      d[k]=\{i\in [1,n]\mid s_2[i]=k\}
      $$

      定义$f[i][j]$表示两个序列的前$i$个字符组成的子序列和$s_2$的前$j$个字符组成的子序列的最长公共子序列的长度。

      $$
      f[i][j]=\begin{cases}
      0,&i=0\\
      0,&j=0\\
      f[i-1][j-1]+1 &\quad     ext{(if }s_1[i]=s_2[j]    ext{)}\\
      \max \Bigg(\{f[k][j]\mid k<i\wedge s_1[k]=s_2[j]}\Bigg)+1&\quad     ext{(otherwise)}\end{cases}\\
      k=0,1,\cdots,i-1
      $$

      ### 分析
      1. 输入：两个序列$s_1,s_2$，长度分别为$m,n$。
      2. 计算量级：$O(m+n)$。
      3. 需要返回$f[m][n]$。
      4. 根据提示，我们构建动态规划的状态转移方程如下：

      ```python
      count=[[0]*26 for _ in range(2)] #[count of each char at position i and j]
      ans=0 
      for i in range(len(s1)):
          count[(i+1)%2][ord(s1[i])-ord('a')]+=1
          ans+=len(set(range(26)).intersection([ord(s)-ord('a') for s in set(s1[:i+1])]))#times when two strings differ at pos i+1 
          print(ans)
          for j in range(len(s2)):
              count[(i+1)%2][ord(s2[j])-ord('a')]+=1
              ans-=len(set(range(26)).intersection([ord(t)-ord('a') for t in set(s2[:j+1])])) 
              print(ans)
      ```

      此处的计算量主要为计数，时间复杂度为$O(mn)$，改进后的版本节省了一部分计算量。