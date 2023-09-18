
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Shadow mapping 是一种渲染技术，它利用阴影映射技术模拟在光源或其他投射物面的投射阴影效果。在光照计算时，从渲染摄像机生成一个阴影贴图（shadow map），然后通过查找深度纹理的方式，在每个片段位置计算相应的阴影坐标，并基于该坐标采样相应的阴影贴图，实现模拟投射阴影的效果。阴影映射技术能够精确地反映物体表面上的投射阴影效果，而无需进行复杂的光照模型计算和物理模拟，因此可以提高呈现图像的真实感。
Shadow mapping 有多种优化方式，如软阴影（soft shadow）、动态阴影（dynamic shadow）等。由于其利用了深度纹理的逐像素查询功能，对性能的消耗较大，所以一般都配合其它渲染效果一起使用，比如法线贴图、高光贴图、环境光遮蔽（ambient occlusion）。另外，相比传统的阴影映射方法，高级版本的阴影映射还可以采用动态阴影和PCF（Percentage Closer Filtering）等技术，有效降低阴影带来的锯齿状 artifacts。

2.相关术语
1) Depth texture: 一种特殊的Texture类型，主要用于存储场景中物体的距离信息，可以用它来做shadow map。Depth Texture有两种类型：float 和 unsigned int，分别对应深度范围为[0,1]和[0,2^n-1]的范围，其中n代表数字精度。

2) Projection matrix: 投影矩阵，用于将三维空间中的点转换到屏幕空间，得到二维的屏幕坐标。用于将物体投影到视线方向上。

3) View Matrix: 观察矩阵，用于确定光源到视线的转换关系。

4) Light position and direction vectors: 光源位置向量和光源指向向量，分别表示光源位置和光源指向的方向。

5) Shadow buffer: 暗部位缓存，用于存储光源阴影的深度信息。

6) Shadow bias: 阴影偏移值，是在shadow comparison的基础上引入一个微小偏差，防止邻近物体的阴影被绘制在距离阴影比较位置过近的地方。

7) Near plane and far plane: 近平面和远平面，用来设置阴影的显示范围。

# 2.Shadow Mapping Basic Concepts
## 2.1 Shadow Map Introduction
Shadow mapping 的核心思想就是利用深度纹理来计算投射阴影，即根据场景中物体的位置和形状来生成一个投射阴rome贴图，这个贴图可以用来表示场景中的所有对象的阴影，而无需对每个对象进行渲染和计算。由于投射阴影有一定几何形状上的限制，比如立方体，因此可以实现逼真的阴影效果。
## 2.2 Shadow Mapping Algorithm
### 2.2.1 Projecting the Scene onto a Shadow Map
首先需要创建一个投影矩阵ProjectionMatrix，将物体的变换关系投影到视线方向上。然后根据光源位置和方向，计算出投影矩阵中的lightSpaceMVP矩阵，用于计算摄像机到光源的投影映射。最后计算摄像机空间下的深度（z）值，并通过投影矩阵将深度转换到光源空间下的深度坐标。
### 2.2.2 Calculating the Shadow Map Coordinates from the Scene Position
利用深度纹理，就可以计算出每个顶点在投影矩阵下的深度值，根据z值的大小决定相应的纹理坐标。对于每个阴影贴图上的像素，都可以通过读取深度纹理，来确定其对应的场景位置，再利用这个位置和光源方向，计算出阴影坐标，进行阴影贴图采样。
### 2.2.3 Shadow Bias
由于深度值是由相机到物体的距离，因此不可能完美的反映实际的阴影。为了解决这一问题，可以引入一个偏差项(bias)，使得像素的阴影分布更均匀，进一步减少 artifacts。

### 2.2.4 Percentage Closer Filtering (PCF)
除了静态阴影外，Shadow Mapping 还有一种动态阴影的方法——Percentage Closer Filtering，它利用周围的多层深度纹理来更加准确地估计深度值。通过一定的权重函数来过滤掉距离物体很远的像素。这样可以有效地减少 artifacts。

## 2.3 Render Passes with Shadow Maps
### 2.3.1 Rendering Objects without Shadow Maps
首先渲染场景中的不透明物体。
### 2.3.2 Rendering Objects with Shadow Maps
若要渲染包含投射阴影的物体，则需要渲染两个Pass：

第一步：渲染阴影贴图，也就是本文所说的投射阴影贴图，也就是说，假设我们有了一个cube，在相机位置的八个角落有阴影，那么我们的shadowmap贴图中的四个区域就会存有阴影的深度信息。这个过程需要的算法就是之前提到的第一部分——投影矩阵、摄像机和光源的计算以及深度纹理的计算等。
第二步：渲染对象，先对阴影贴图进行采样，计算阴影衰减系数，然后将阴影映射过后的颜色叠加到物体的颜色上，并进行透明混合。

当然，这个时候的物体可能还有高光贴图、法线贴图等等效果也要考虑进去。

第三步：渲染透明物体，同前面一样，但是不渲染阴影贴图，只是直接渲染透明物体。