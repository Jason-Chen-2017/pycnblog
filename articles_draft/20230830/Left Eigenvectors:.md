
作者：禅与计算机程序设计艺术                    

# 1.简介
  
：
左特征向量(left eigenvector)是矩阵的一种重要性质，它可以用来表示矩阵的一些特殊结构和特性。本文将介绍一下左特征向量的概念、如何计算得到左特征向量、以及其应用。

# 2.定义和基本概念：
左特征向量（left eigenvector）: 如果一个对称矩阵$A \in \mathbb{R}^{n\times n}$存在非零向量$v_1$，使得对于任意的非零向量$x=(x_1,\cdots, x_n)$，有：$$Av_1=λv_1,\forall v_1 $$则称$v_1$为矩阵$A$的左特征向量，$\lambda$为对应的特征值。左特征向量又称特征向量，与特征值相对应。

一般来说，对于一个实对称矩阵$A$，其特征向量$v$和特征值$\lambda$之间有如下关系：$$v^TAv=\lambda^2,$$其中，$v^Tv>0$.因此，如果$\lambda>0$,则$v$是$A$的特征向量；否则，$v$不是$A$的特征向量。

在实际中，对称矩阵往往都是由实数构成的，但由于数值计算的限制，我们经常遇到对称矩阵是近似或稀疏矩阵。因此，为了保证算法的准确性和稳定性，通常会对矩阵进行预处理，如转置或初等变换等，从而使得矩阵满足一定条件，例如严格对角阵、非负实数的主子空间等。

# 3.基本算法：
计算矩阵$A$的左特征向量，可以分为以下几个步骤：

1. 对矩阵$A$进行初等变换，直到满足某些条件。这些条件包括：矩阵$A$是一个实数矩阵，当且仅当它的对角线元素均为正时才为严格对角阵。
2. 判断是否存在严格的特征值。若没有，则不存在左特征向量。
3. 如果存在严格的特征值，则确定相应的特征向量。

具体操作步骤如下：
1. 若矩阵$A$的对角线元素都为正或者为负，则直接取出对应位置的特征向量即可。即：
    - 当$A$是严格对角阵，则：$$v_i=[\delta_{ii},-(\delta_{ij}\delta_{jk})^{-1}A_{kj}]$$
    - 当$A$不是严格对角阵，但是其对应的特征值有重根，即存在多个对应特征向量时，选取一个即可。
    - 当$A$不是严格对角阵并且没有重根，则不存在左特征向量。
   （注：这里的$[\delta_{ii},-(\delta_{ij}\delta_{jk})^{-1}A_{kj}]$就是矩阵$[I-\frac{\sigma_i}{\sigma_j}(e_j\otimes e_k)]A$的特征向量。）
   
2. 通过迭代的方法求解特征值与特征向量。首先，初始化特征值、特征向量和方向向量。然后，对于每个向量$v_1$，进行下述迭代：
    a. 求解方程$Av_1 = λv_1$中的$λ$，记为$λ_1$。
    b. 根据方程$Av_1 = λv_1$得到新的特征向量$v_2$，记为：$$v_2=[v_1,-\frac{v_1^TA_iv_1}{v_1^Tv_1}v_1]$$
    c. 比较$v_1$和$v_2$的模长，若相差很小则跳出循环，此时得到的$λ_1$即为特征值。

举个例子，考虑矩阵：$$A=\begin{bmatrix}-2&-1\\-1&2\end{bmatrix}$$，用初等变换可知$A$是实数矩阵，且严格对角阵。设$v_1=[\delta_{11}=2,\delta_{12}=-1]^T$为对应的特征向量。则，根据方程$Av_1 = λv_1$，有：$$[-2(2)-(-1)(-1)+(2)(-1)]=(-2+1+1)=0$$，即$λ_1=-1$。再假设$v_2=[v_1,-\frac{v_1^TA_iv_1}{v_1^Tv_1}v_1]$为对应的特征向量。则：$$A[v_2-v_1][v_2-v_1]^T=-2(2)^2+(1)(-1)=-4.$$又：$$[-2(-1)-(-1)(2)+(2)(-1)]=(-2-1+2)=0$$，也即$λ_2=-1$。由此可得$v_1$与$v_2$是矩阵$A$的特征向量，对应的特征值为$-1$和$-1$。

# 4.代码实现：
python代码实现：

```python
import numpy as np
from scipy import linalg 

def left_eigvecs(a): 
    # 初始化参数
    m, n = a.shape
    
    # 进行初等变换
    if is_strict_diag(a):
        vecs = np.eye(m)*np.sign(a).diagonal() 
        vals = a.diagonal().copy()  
    elif has_repeated_root(a):
        idx = repeated_root_idx(a)  
        val, vec = linalg.eig(a[:,idx])
        idx = np.argmax(abs(val))
        vecs = [vec[:,idx]]   
        vals = [-val[idx], val[idx]]
    else:
        return None
        
    while True:
        new_vals = []
        new_vecs = []
        
        for i in range(len(vecs)):
            tmp = solve(a, vecs[i]) 
            norm = abs(tmp/linalg.norm(tmp)).reshape((-1,))
            # print("norm:", norm)
            j = norm.argmax()  
            val = max(norm)/min(norm)     
            if (new_vals and abs(val)<abs(new_vals[-1])) or not new_vecs:
                new_vals.append(val*vals[i//2])   
                new_vecs.append((vecs[i]+tmp/norm[j]*vecs[i].dot(tmp))*val**0.5)  
            else: 
                break  
                
        if len(new_vecs)==len(vecs)+1:   
            vecs += new_vecs[-1:]   
            vals += list(new_vals[-1:])
        else:
            vecs[:] = new_vecs  
            vals[:]=new_vals
            break
            
    return vecs, vals  

def solve(mat, vec):    
    """求解Ax=b，返回Ax"""
    res = mat@vec   
    return res
 
def is_strict_diag(a):
    """判断是否严格对角阵"""
    eps = np.finfo(float).eps
    diag = a.diagonal()
    return all(abs(d)>eps for d in diag) 
    
def has_repeated_root(a):
    """判断是否有重复根"""
    eps = np.finfo(float).eps
    _, eigval, _ = linalg.svd(a)
    return any(abs(eigval[i]-eigval[i-1])<eps for i in range(1,len(eigval)))
    
def repeated_root_idx(a):
    """找出重复根的列索引"""
    eps = np.finfo(float).eps
    u, s, vt = linalg.svd(a)
    rank = sum(s>eps)
    idx = np.argsort(u[:,rank:], axis=0)[::-1][:,:rank]
    return idx[0,:]    
```