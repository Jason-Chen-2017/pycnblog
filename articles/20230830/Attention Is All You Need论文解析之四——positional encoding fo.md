
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Transformer模型是当前最热门的NLP任务的基础模型之一。Transformer的关键点在于引入了“self-attention”机制，即每一个位置对其他所有位置都可以attend。自然语言处理的输入序列一般来说是长的，而且每个位置的特征往往很重要，因此加入这种“全局注意力”的机制可以提高模型的性能。而在传统RNN模型中，由于时间上的依赖关系，只能从当前时刻到历史序列的某一个时刻进行建模，因此难以捕获全局的信息。
为了解决这个问题，transformer通过学习得到一个函数f(x)将输入序列转换成另外一种形式，使得每个位置的表示都能够充分考虑上下文信息。其中，f(x)是一个多层非线性变换，通过学习，可以把输入序列转换成新的形式。在多头注意力机制中，可以让不同的子空间共享权重。也就是说，每一次查询都会关注不同的子空间，这样既能够捕获全局信息又不会产生冗余信息。最后，通过使用残差连接和LayerNorm等正则化技术，使得模型的训练更加稳定，并保证泛化能力。图1展示了 transformer 的整体结构。
但是，对于非序列数据（图像、视频）或序列数据的变种（如对话系统），单纯的 Transformer 模型就不能直接用了，需要改进。这就是positional encoding的作用。

positional encoding 是一种对非序列数据的嵌入方式，它会添加一些额外的信息来帮助模型编码数据中的相对位置关系。该方法主要用于构建表征时序信息的循环神经网络（RNN）模型。其主要思想是在每个位置向量上增加一定的信息，这些信息来源于嵌入矩阵和位置索引，比如位置编码（Positional Encoding）。


# 2.基本概念术语说明
## 2.1 Positional encoding 位置编码
Positional encoding 是一种对非序列数据的嵌入方式，它会添加一些额外的信息来帮助模型编码数据中的相对位置关系。其基本思想是：给输入序列添加一串额外的位置编码，而不是简单地将每个位置视作一个独立的输入。因为相邻位置的距离通常比较小，所以编码后的位置坐标对模型的预测行为应该更有帮助。位置编码的目的是给模型提供关于各个位置之间的相对关系的更多信息，这样模型就可以学会如何利用这种信息来增强特征表示。

### 为什么要加上位置编码呢？
那么，为什么要给输入序列添加位置编码呢？如果不加的话，模型可能只是记住输入数据本身，而缺乏全局信息。假设输入序列是由$x_i$组成的，那么无论我们怎样处理$x_i$，模型都无法识别出这个序列的全局特征。举个例子，如果我们想学习到输入序列中的语法关系（前后关联），那么模型只能看到$x_{i-1}$和$x_i$，而无法利用整个序列的信息。在另一个例子，如果我们想学习到输入序列中潜在的模式（如序列出现模式，如循环序列等），那么模型也只看到每个元素的一部分，而没有整体的信息。

因此，当输入数据具有全局特性时，我们可以通过对原始输入数据加上位置编码的方式来增强模型的表达能力。

### 位置编码存在的意义
除了上面介绍的两个原因之外，还有第三个原因，即位置编码可以帮助模型学习到长期依赖关系。这是由于位置编码的引入，我们可以学到不同位置之间的相互影响。例如，在图像分类任务中，不同的位置对应着不同的特征区域，但是这些区域却是按照顺序排列的，这就使得模型可以获得局部信息，但对于整个图像的全局信息却无能为力。引入位置编码之后，我们便可以为模型学习到各个位置间的长期依赖关系，这样模型才能对图像的全局特征做出更准确的预测。

## 2.2 Self-Attention 自注意力机制
Self-Attention 机制是 transformer 模型的核心。它的基本思路是在计算时将一系列的输入向量（Query、Key 和 Value 三者组成的元组）在不同的子空间之间进行Attention，从而实现序列数据的全局特性的编码。

### Query、Key 和 Value 的含义
在 self attention 中，输入序列 x 可以被看做是一个向量集合 {x^1,...,xn} ，其中 $x^1,...,xn$ 是序列的 n 个位置向量。每一个位置向量的维度一般都是 d，即 $x\in R^{n \times d}$ 。

Query、Key 和 Value 分别是 self attention 中的三个矩阵。Query 和 Key 矩阵的维度分别为 $m\times d$ 和 $n\times d$ ，而 Value 矩阵的维度为 $n\times m$ 。具体来说，

* Query 矩阵：由前面的输入序列 x 求取的权重矩阵，矩阵中每个元素表示每个位置对该位置之前的元素的影响程度。
* Key 矩阵：由输入序列 x 求取的权重矩阵，矩阵中每个元素表示每个位置对其之前的所有元素的影响程度。
* Value 矩阵：由输入序列 x 汁取的值矩阵，矩阵中每个元素表示每个位置的特征值。

### Attention Score 函数
Attention score 函数 f(\cdot,\cdot) 是 self attention 运算的核心函数，它接受两个输入向量作为输入，输出一个标量，用来衡量两个向量之间的相关程度。不同的 Attention score 函数可以得到不同的结果，例如 dot-product attention 或 additive attention 等。

### Scaled Dot-Product Attention (SDPA)
Scaled Dot-Product Attention 是最简单的 Attention score 函数。其基本思路是：对于 Query i 对应的 Key j ，计算 q_ik 和 k_jk 的点积，除以根号(维度)大小，得到 Attention Score aij 。然后，通过一个 softmax 函数得到 Attention Weights ai 后，再通过 Value 矩阵得到输出向量 vij 。最终，Attention 向量 oi = sum{aij * vij} 对所有的位置求和得到输出序列。

$$
a_{ij}=q_iq_k+bias\\
\text{softmax}(a)=\frac{\exp(a_i)}{\sum_{j=1}^{n}\exp(a_j)}\\
o=\text{tanh}(\text{W}[a]\text{V}^T[x])
$$ 

## 2.3 Multi-Head Attention （MHA）
Multi-Head Attention 即多头注意力机制，是基于 self attention 技术的改进版本。其基本思路是：将多个 self attention 操作组合起来，每个操作在不同子空间下探索不同特征，从而提升模型的表达能力。具体来说，首先将 Q、K、V 通过线性变换 Wq、Wk、Wv 分别映射到 q、k、v 上，然后计算 attentionscores 函数得到 aij 。接着，再次线性变换 Wl 将 attention scores 和 values 组合成 oij 。最后，通过一个 softmax 函数得到 attention weights ai 后，通过加权求和得到输出向量 oi = sum{aij * oij} 。

$$
\begin{aligned}
&\text{(MultiHead)Attention}\\
&Q=[Q_{\beta_1},...Q_{\beta_h}]\\
&K=[K_{\beta_1},...K_{\beta_h}]\\
&V=[V_{\beta_1},...V_{\beta_h}]\\
&\text{where } \beta_i\in\{1,...,h\}\\
&\text{for each head } h\text{:}\\
&\quad \alpha_{ij}^{(h)}=\frac{\exp(Q_{\beta_h}q_iq_k+\epsilon)}{\sum_{j'=1}^{n'}exp(Q_{\beta_h}q_i q_{j'})}\\
&\quad o^{(h)}=\text{softmax}(\text{concat}(\alpha_{ij}^{(h)})V_{\beta_h})\\
&\quad O=[O_{\beta_1},...O_{\beta_h}]=\text{concat}(o^{(1)},...o^{(h)})\\
&\quad O\in \mathbb{R}^{n\times d} \\
&\text{where }\text{concat}(.,.)=\left[\begin{array}{ccccc}
                        o_{\beta_1}^{1}&... & o_{\beta_1}^{n}\\
                       .&.&.\\
                        o_{\beta_h}^{1}&... & o_{\beta_h}^{n}\\
                    \end{array}\right]\\
&\text{and }\epsilon=\text{small constant}\\
\end{aligned}
$$

## 2.4 Relative Positional Encoding (RPE)
Transformer 在设计之初曾经采用了基于相对位置编码的方案，这种方案旨在编码绝对位置信息。然而，相对位置信息也很重要，特别是在图像或视频输入情况下。所以，研究人员提出了基于相对位置编码的方案（Relative positional encodings），使用基于 Query 与 Key 之间的差值来编码位置信息。这种方式下，每个位置的编码并不仅仅由它自身的信息决定，还由其他位置的编码决定。

$$
PE({\bf l})=\sin(\frac{\pi l}{L_{max}}),\cos(\frac{\pi l}{L_{max}})
$$

上述函数 PE 可以将绝对位置编码转换为相对位置编码。其中 L_max 表示最大长度。

$$
RE({\bf l}, {\bf r})=PE({\bf l}-{\bf r}),\forall l\neq r
$$

其中 ${\bf l}$ 和 ${\bf r}$ 分别表示位置坐标，且 ${\bf l}-{\bf r}$ 表示两个位置的差值。如此一来，位置信息就可以通过位置差值隐式地传递出来。