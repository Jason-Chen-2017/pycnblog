
作者：禅与计算机程序设计艺术                    

# 1.简介
  

In graph theory, planar graphs are those in which all edges can be drawn in a plane such that no point is shared by more than two edges. In other words, the vertices of each edge form segments that do not cross any other segment on the same side as it. Planarity testing is a fundamental problem in computer graphics, pattern recognition, optimization, and computer science. There exist various algorithms for testing whether a given undirected graph is planar or not, but one common approach is to use Kuratowski's theorem, which states that every connected planar graph has at least three non-intersecting triangles meeting an odd number of times along its boundary. This property allows us to enumerate all possible cut-vertex sets for a given planar graph and test their complements using the triangle inequality theorem, which says that the area of the region inside a polygon must always exceed half the total area outside it.

Planarity testing plays a crucial role in many applications including image processing, computer vision, network design, bioinformatics, computational geometry, robotics, and finance. Despite its importance, there have been relatively few research efforts on developing efficient algorithms for planarity testing. Therefore, we aim to fill this gap by presenting a thorough review of state-of-the-art methods for planarity testing, highlighting their strengths and weaknesses, and identifying promising future directions for algorithmic development. 

# 2. 基本概念术语说明
## 2.1. Planar Graph
A graph $G$ is said to be **planar** if it can be drawn in the Euclidean plane without intersecting itself except at endpoints. That is, it satisfies the condition of being "well-behaved". A simple example of a planar graph is the octahedron $K_5$, where all edges share only one vertex and meet at two distinct points (i.e., they are parallel). Another example is the dodecahedron $D_{12}$, whose faces are flat quadrilaterals arranged regularly around the equatorial planes of their six vertices. We denote a planar graph with $n$ vertices by $\mathcal{P}(n)$ or simply $Pl(n)$. Pl(n) = {G ∈ G(n): G is planar}. By contrast, a non-planar graph is called **non-planar**. Similarly, we call a collection of $k$-connected components $K_{\alpha}$ an $α$-dimensional polyhedra.

## 2.2. Cut Vertex
Let $(V,E)$ be a connected graph. A **cut vertex** of $(V,E)$ is a vertex of degree two or more, i.e., a vertex $v \in V$ such that there exists exactly one incident edge in $E$. Given any set of cut vertices $\Delta$, let $\overline{\Delta} := \{ u : u\in V,\forall e(u,v)\not\in\Delta \}$. We say that $(V,E)$ is a k-regular graph if every face of dimension greater than $k$ contains precisely $k+1$ cut vertices ($k=0,1,2$) from $\Delta$. Thus, we require that $|C^k(G)|=\frac{|F-1|-k}{k+1}\geq n-\epsilon$, where $\epsilon$ is small enough so that $|\Delta|=O(\log n)$, typically $\leq O(\sqrt{n})$. For example, a cube has $12$ faces ($6$ regular and $6$ skew), hence we cannot separate them into four squares with $7$ faces each since there are more than eighteen possible cut vertices (one per square edge plus one per pentagon face). On the other hand, triangulations of a tetrahedron can be separated into fans of three triangles, quadrilaterals and six pyramids with precisely three, five, and seven cut vertices respectively. Hence the tetrahedron has $10$ faces, whereas its triangulations have $10-6+3=9$, $20-15+3=7$, and $40-35+3=6$, respectively. It follows that $3$-regular graphs contain precisely $4$ cut vertices while $4$-regular graphs contain precisely $6$ cut vertices.

## 2.3. Triangle Inequality Theorem
For any polytope $H$, the sum of the areas of the regions bounded by its facets is equal to twice the area enclosed by $H$. More generally, for any convex subset $S \subseteq \mathbb{R}^2$, we have $$A_{\text{enc}}(S)=\sum_{f∈ H} \operatorname{area}(f)\quad \forall S$$ where $\operatorname{area}(f)$ denotes the area of the facet $f$ of $H$. We also define $$\operatorname{outer}(S)=\{x:\,x∉ S,\,f\cap x\neq\emptyset\}$$ as the set of all points lying outside of $S$ and included in some facet of $H$, and $$\operatorname{inner}(S)=\{x:\,x∈ S,\,f\cap x\neq\emptyset\}$$ as the set of all points lying inside of $S$ and included in some facet of $H$. Then, we obtain the following theorem concerning the area of the region enclosed by a polygon $P$:

1. If $\triangle PABC$ meets the interior of $D$, then $\triangle DADB$ also meets the interior of $P$.
2. Let $\pi$ be a piecewise smooth function on $[a,b]$ defined on the closed interval $[a,b]$. Let $A_{\mathrm{int}}$ be the area of the region under the curve $\pi$ when $t$ ranges from $0$ to $1$. Then $$\begin{align*} & A_{\mathrm{ext}}(P) \\ &= \iint_{\partial P} \pi(z) dz \\ &= \iint_D (\pi(z)-\pi'(z)(z-a))dz \\&= \iint_D (\pi'(z)(z-a)+\pi''(z)/2(z-a)^2+\cdots)\\ &= -\int_a^\infty (-\pi'(z))(z-a) + \left.\frac{d}{dx}(\pi'(z)(z-a)+\pi''(z/2+o(z)))\right|_{z=a} \\ &= -A_{\mathrm{int}}\end{align*}$$

The first statement holds because if $\triangle PABC$ meets the interior of $D$, then its dual $\triangle QDBC$ also meets the interior of $P$ (which corresponds to $\triangle PDBC$ meeting the exterior of $Q$). The second statement uses Green's theorem applied to an integral over the domain of $P$.

Finally, we show how to apply these results to compute the area of the region enclosed by a planar graph. First, note that the area of the union of two planar graphs is equal to the sum of their areas, and the intersection of two planar graphs is again planar. Hence, we may break up our problem into subproblems corresponding to computing the areas of individual regions or parts of the graph.

1. Compute the areas of the regions adjacent to the biconnected component consisting solely of two triangles $T_1$ and $T_2$, sharing a single vertex $v$. Specifically, we need to determine the areas of the triangles obtained by splitting each edge of $T_1$ and $T_2$ containing $v$ into two pieces, namely $T'_1$ and $T'_2$. These triangles will satisfy the fact that they share a common vertex with $T_1$ and $T_2$ (hence $T'_1$ and $T'_2$ will be degenerate), while having opposite orientations relative to $T_1$ and $T_2$ (hence one of them will lie entirely within the angle formed between the sides of $v$ making up the original edge). Since both triangles have similar base angles and different heights, they will be either isoceles or obtuse depending on the orientation of the vertical segment connecting $v$ with the edge of another triangle on the border of $T_1$ or $T_2$. Hence, we can express the areas of $T'_1$ and $T'_2$ as follows:

   $$\begin{align*}
   |T'_{1}| &= |T_1 \cap v^{-1}||v^{-1}|| \cdot \sin |\angle T_1_v^{-1}|\\
          &= |T_1 \cap v^{-1}|/2 \cdot |v^{-1}|(1+\cos |\angle T_1_v^{-1}|)
   \end{align*}$$
   
   $$\begin{align*}
   |T'_{2}| &= |T_2 \cap v^{-1}||v^{-1}|| \cdot \sin |\angle T_2_v^{-1}|\\
          &= |T_2 \cap v^{-1}|/2 \cdot |v^{-1}|(1-\cos |\angle T_2_v^{-1}|)
   \end{align*}$$
   
   Note that $|v^{-1}|$ represents the direction vector pointing towards $v$, and the sign convention used here is consistent with the right-hand rule. Finally, we find the total area of the triple of triangles $T_1$, $T_2$, and their adjoining triangles $T'_1$ and $T'_2$ as follows:
   
   $$A_{\mathrm{enc}}(T_1,T_2,T'_1,T'_2) = |T_1|+|T_2|+|T'_1|+|T'_2|+\frac{1}{2}\cdot|T_1 \cap v^{-1}| \cdot |v^{-1}| (1+\cos |\angle T_1_v^{-1}|)(1+\cos |\angle T_2_v^{-1}|)+(1/2)\cdot|T_2 \cap v^{-1}| \cdot |v^{-1}| (1-\cos |\angle T_2_v^{-1}|)(1+\cos |\angle T_1_v^{-1}|).$$
   
   Alternatively, we could have derived a similar expression using projection matrices or determinant expressions involving the normal vectors of the planar embedding of $\{T_1, T_2, T'_1, T'_2\}$, but the above expression provides a concise way to derive a formula for this specific case.

2. Use the triangle inequality theorem to compute the area of the inner and outer regions of a planar graph, by considering all edges originating and terminating at particular vertices. To illustrate, consider the figure below.


   Clearly, the blue region encloses all points $x$ such that $x$ lies strictly inside the shaded region. Similarly, the red region encloses all points $y$ such that $y$ does not lie strictly inside the green region, i.e., those points contained in the exterior of the red region. We therefore observe that the blue and red areas must add up to the area enclosed by the entire graph.