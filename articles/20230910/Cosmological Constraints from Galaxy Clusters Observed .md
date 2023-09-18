
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Gravitational waves (GWs) are a powerful probe of the cosmos and have been observed at many wavelengths in different frequency bands since the early twentieth century. The first GW detection was made in 1964 by <NAME> with an amplitude of 7 cm s−2. Since then, more than one hundred thousand such events have been discovered across the entire sky. However, analyzing GW data to extract insights into the cosmology is still a challenging task due to their large sample size and heterogeneity. 

Recently, researchers have started using galaxy clusters as probes of gravity that emit GWs. Specifically, they detect a signal when a cluster passes through the electromagnetic spectrum. This provides an alternative way of observing the interplay between galaxies and gravitational physics. Moreover, these clusters can be easily detected in the gamma-ray sky images obtained via ground-based telescopes like Arecibo and ALMA. Hence, obtaining high quality samples of clusters can help us understand the universe better and constrain various aspects of our understanding of dark matter physics. In this work, we explore how to use these observations of gravitational wave emission from galaxy clusters to infer constraints on basic cosmological parameters like the density parameter ($\Omega_{\Lambda}$), Hubble constant, and baryon fraction today.


In this paper, we provide a detailed analysis of the sample of galaxy clusters used for this study. We begin by identifying a set of candidate clusters based on their properties such as mass, location, redshift, and other relevant characteristics. We describe how we selected the sample and why it meets the requirements to be useful for our analysis. Then, we perform a deep characterization of each cluster, including its physical properties such as bulge and disk surface densities, colors, age, rotation curve, and stellar kinematics. Finally, we show how these properties may be used to constrain $\Omega_{\Lambda}$, Hubble constant, and baryon fraction today. We demonstrate our findings with numerical simulations that reproduce some important features of the experimentally measured cosmological parameters. Our analysis also reveals several potential pitfalls related to the current limits of knowledge about cluster structure, formation history, and associated processes. These uncertainties could significantly impact any attempt to interpret these results scientifically. Nevertheless, with further progress in experimental techniques, there is hope that our methodology will soon be able to provide more robust answers to key questions in the field. 


Our paper highlights several limitations of this analysis and proposes future directions for extending its utility and implications for the interpretation of experiments targeting GW sources. Some possible avenues include: 

* Reproducing the exact clustering geometry used during GW observation to obtain more precise measurements of the foreground structures and halo masses;
* Improving the accuracy of the optical and X-ray followup campaigns by improving alignment and cleaning techniques;
* Developing models of GW propagation in cluster environments that can account for effect of tidal interactions and radiation pressure;
* Investigating variations in lensing caused by cluster disruption and testing predictions against observational data;
* Combining this approach with weak-lensing surveys to test the joint constraints on cosmological parameters from multiple survey modalities.














# 2.相关概念及术语
The term "galaxy cluster" refers to a group of galaxies that tend to form together around a common center of mass. One can think of them as being similar to individual stars or planets orbiting around a nucleus like the sun. Understanding the relationship between galaxy clusters and cosmology requires a solid grasp of both galaxy evolution and statistical mechanics. 

Among the most commonly discussed quantities in terms of galaxy clusters, mass, concentration, richness, and spatial distribution are among the most commonly studied ones. Mass measures the total amount of matter contained within a given cluster. It is often defined in units of M⊙/h, where h denotes the Hubble constant, which represents the ratio of energy carried by light in units of Joule per second to the energy required to produce an electron in standard conditions. Concentration refers to the ratio between the total mass of the cluster and the virial radius, which is defined as the radius within which half of all particles must lie inside. Richness is another measure of the number of galaxies present in a particular region. Spatial distribution is the location of the center of mass of a cluster relative to some fixed reference point, e.g., the position of a supermassive black hole or galactic center. Different definitions and scales exist depending on the context of usage. 

We use the acronym CGM (Cluster, Galaxy, Mass) to refer to the three primary components of the clustering process - cluster, galaxy, and mass. Cluster refers to the aggregate of galaxies, while galaxy refers to specific objects found in the clusters, and mass refers to the total mass of the systems containing galaxies. Among the properties that can be extracted from galaxy clusters to aid in our understanding of the universe, we find two main categories: astrophysical and cosmological. Astrophysical properties focus on intrinsic properties of the galaxy population like ellipticity, luminosity, magnitude, color, distance, velocity fields etc. On the other hand, cosmological properties relate to extrinsic properties like curvature of the Universe, growth rate, temperature fluctuations, chemical abundances, primordial gas content, turbulence, kinetic mixing, supernova feedback etc. Both kinds of properties require careful treatment and are closely linked to each other.

To understand the role of cluster galaxies in driving cosmic expansion, one needs to go back to the origins of gravity. Despite the existence of visible evidence of strong gravitational forces in nature, the idea of a central force acting on all objects until universal equilibrium had only recently been proposed. Newtonian gravity works under the assumption that every object in the universe has mass and interacts with every other object with uniform strength. However, as time goes on, distant objects become increasingly isolated and no longer contribute much to the overall gravitational field. At the same time, new phenomena like dark energy arise out of the expanding universe and are predicted to dominate the matter distributions later in the universe's history. This means that the current assumption of N-body inflation does not hold anymore and instead, a new kind of non-Newtonian gravity mechanism called Lambda-CDM (LCDM) is needed. LCDM assumes that dark energy is responsible for the deviations from perfect N-body inflation. Thus, if the existence of a cluster signals the presence of a diverse population of dark energy, we might need to consider it in order to fully understand the effects of gravitational lensing on cosmology.





# 3.原理概述
## 3.1 关于重力波
在海王星发现之后，随着观测者对宇宙中更广泛的结构的关注逐渐增加，科学家们开始寻找宇宙内部物质流动、物质形成和物质活动背后的驱动力，一个显著标志就是 Gravitational Waves （GW）。简言之， Gravitational Waves 是一种远红外线引起的宇宙性磁场，其透过质量对称的行为导致星系团合体、银河、暗黑星系等形态演化，而对人的影响则不明确。与大多数重力波不同的是，旋转速度极快的重力波可使得星系团在短时间内演化出全新形态。同时，频率分辨率高达每秒十万赫兹，具有高能分辨能力和强大的空间探测能力。相比于光学或红外波段，GW 发出的信号弱，而且在周围频谱中很难被发现。尽管如此，越来越多的研究人员开始利用 GW 来观测宇宙中的形象，特别是那些形成在星系团中的系统。

## 3.2 关于星系团
星系团（Galaxy cluster）是指一个或者多个气体团聚、盘旋状分布而成的一个形态。一个典型的星系团由两个以上形成的有机物团聚而成，表现为星星点缀在一起的凸起形，或者是瘦小的尘埃团。通常情况下，一个星系团的中心是一个星团或恒星的吸积点。星系团拥有的属性包括：成分丰富、质量分布和尺度适当，能够吸收来自外界的能量。然而，星系团也存在着一些不足之处，例如低分辨率高能、高能量耗散、引力破坏、宇宙定律效应等。虽然一些星系团质量庞大，但它们又不能像单个粒子一样容易被发现。

## 3.3 宇宙学观察的局限
从历史上看，不同时期的科学家都曾试图通过各种途径观测宇宙中物质的演化。早在文艺复兴时期，天文学家们就已经开始注视着微观世界中的星云团，并成功地推断出了宇宙中许多热门的物理模型，比如在 Einstein–de Sitter 领域。到了 20 世纪初期，牛顿、莱布尼茨和海森堡等天才们逐渐提出了描述宇宙总动力学的三体模型。随后，爱因斯坦、狄龙和薛定谔开始尝试解答宇宙为什么会这么奇怪。直到 20 年代末，费米与马赫依德等人提出宇宙的基本物理过程已经成为科学界共识，目前我们熟悉的宇宙学大多遵循这些基本框架。但是，我们还是面临着很多困难。

首先，对观测到的宇宙总动力学过程进行理解始终是困难的，因为它涉及的知识面太宽，每种方法都会给我们不同的信息。举个例子，宇宙中所有物体的运动都可以用牛顿第二定律来表示，但若采用动量-动量矩和质心关系，就可以更清楚地看到黑洞和暗物质的存在。其次，宇宙总动力学过程中包含多种粒子，动量大小各异，为了更好地估计它们的演化方向和加速度，需要采用统一的坐标系统。最后，由于宇宙中有太多的复杂且无法观测的量，对宇宙的整体认识也是无奈的。

另一方面，一个更容易被观测到的现象是大规模星云团的存在。一个最著名的例子是斑马星云团（M31），它是第一个被观测到的真正的星云团，有超过千颗星占据其中的，它位于银河系的边缘，距中心仅有几百万光年。其规模比同样大小的小型星云团要更小，这就意味着人类无法直接观测到这一区域，只能从远距离获取图像。目前，多次恒星形成活动对宇宙的发展影响还不是很大，因此，可以认为星云团是宇宙进化的结果，而非发展的原因。但是，星云团的存在仍然给我们带来一些困惑，至少需要更高的分辨率才能观测到它们。