
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 概述
染色质结构（Chromatin）是指细胞内所有蛋白质的位置分布，与其功能密切相关。近年来，随着高通量测序技术的广泛应用，对染色质结构及其功能进行高效、定量测定研究成为了必不可少的一环。然而，对于染色质的科普还停留在表面知识层次，并没有从实际问题出发，讨论起染色质结构到底是什么意义。在本文中，作者将从生物信息学视角出发，讨论染色质结构及其功能的基本原理。作者将从染色质结构组成和功能三个方面详细阐述染色质结构及其功能的基本概念，并将这些原理与现有的各种实验方法相结合。作者还将描述他对染色质功能的理解，并展望下一步的研究方向。
## 染色质结构简介
染色质结构可以用基体、染色质链和核苷酸键之间的三维空间分布来表示。染色质结构由四个主要的结构元素组成：染色质模（Chromosomes）、染色质骨架（Chromatin fibers）、DNA碱基（DNA nucleotides）以及DNA结构轨迹（DNA structures）。
### 染色质模
染色质模由许多小型的DNA分子链组成，形成 chromosomes（又称nuclear bodies），它们有两种类型：叶状核苷酸（histone）和组蛋白质（chromatin protein）蛋白。叶状核苷酸是真核生物体中重要的组成单位之一，其构象与染色质结构有关。叶状核苷酸包含在染色质上，并且与核酸发生互补作用，形成不同大小的囊泡，具有不同的颜色和分子质地。每个染色质模都有一个与之对应的核苷酸，由多个连接着核苷酸的蛋白质的双螺旋结构组成。
>图1：染色质模分子结构示意图

根据不同染色质模的特点和功能，染色质结构会呈现不同形式和结构。如上图所示，染色质模主要包括四种不同结构的蛋白质，分别是组蛋白质蛋白、膜状核苷酸蛋白、沟状核苷酸蛋白和修饰核苷酸蛋白。组蛋白质蛋白主要负责DNA聚集和复制，占主导地位；膜状核苷酸蛋白表现出对外周环境的耦合作用，其特异性使染色质结构呈现出非均匀的形态；沟状核苷酸蛋白则主要参与染色质内DNA修复，形成抗性作用；而修饰核苷酸蛋白则主要参与结构调整。因此，不同类型染色质模之间的功能差异会影响染色质结构的演化。
### 染色质骨架
染色质骨架是一个不断向外延伸、不断衔接直线形的结构，由多个不同染色质模团通过亚稳定的轨道相连而成，组成一个蛋白质团簇。它具备高度的结构复杂度，以致于在各个层级上都可观察到多种类型的结构细节。目前已知染色质骨架存在着多种形式，其中两条最常见的是转座（CTCF）和芯片（ChIP）。CTCF是一种冗长的染色质间纤维化物，通过其连接，染色质经过不同的方式进入细胞核，形成染色质同源区，从而调控细胞的多样性。CTCF的发现引起了人们对染色质结构的兴趣，一些研究人员试图利用 CTCF 的特性研究染色质结构和功能。芯片作为一种核酸磁悬浮物，在细胞核表面形成，能够识别和计数特定蛋白质或基因的表达水平。ChIP 还可以用来检测单个蛋白质的功能，其标记物就是染色质功能区域的序列。例如，TATA 修饰序列可以标记特异性 DNA Binding Domain（DDBDs）上的 DNA 结合位点。
>图2：不同染色质骨架的结构特征图解

### DNA 碱基
DNA分子由四种碱基（Adenine（A），Cytosine（C），Guanine（G）和Thymine（T））组成。DNA分子是染色质结构的基本单位，其编码信息通过基带来传输。每个碱基都由四个原子组成，分别是氧原子（U）、氮原子（N）、离子（O）、铅原子（P），共同作用于 DNA 分子的两个核苷酸之间。染色质表面可以看到氧原子对半导体阳离子的排斥作用，形成双绕结构，能够对 DNA 分子进行电泳分离。在 DNA 链的末端，一个碱基的上三角结构被游离的氮原子包裹住，成为偏向于轻碱基的末端。在其他地方，由于 DNA 分子的形态限制，碱基的最外层残基可以固定住一个氢原子的双侧，实现二倍的自由度。DNA的含糖量较低，一般情况下不溶于水。
### DNA 结构轨迹
DNA结构轨迹是指染色质结构的动态演化过程。它以染色质中的一段DNA分子片段为单位，并记录了该片段在不同时间点上所呈现出的基序，同时也反映了其功能和结构变化情况。结构轨迹一般是借助计算机模拟软件生成的，其产生原理与构筑染色质金属结构的步骤类似，即通过核苷酸的螺旋形运动和碱基键的形成，从而使染色质中的蛋白质结构因子受到适当的控制，最终形成一个独特的染色质结构。
>图3：DNA结构轨迹示意图

## 染色质结构功能
### 细胞器官及器件的功能
细胞器官的功能在各个领域各有千秋，如免疫系统的作用、神经元的工作、肿瘤移植等。虽然整个组织的结构都受制于染色质结构，但在一定程度上能调控细胞器官的活动，保障其正常运行。在健康的细胞中，细胞器官在相互作用中形成一个复杂的功能网络，以完成特定的生物任务。除了遵循细胞内的标准流程之外，器官间也会互相调控和合成，形成临床上被认为的“行事顺利”的局面。染色质结构还能够影响器官的形态和功能，如皮质激素的作用、神经元的不同类型之间的关系以及ADORA-295的作用。在某些情况下，通过改变染色质结构，可以改变组织内部各个器官的功能。
### 细胞功能的调控作用
染色质结构是细胞功能调控的关键一环。与其他蛋白质一样，染色质结构的改变会影响RNA免疫、转录调控、激活、抑制细胞器官的功能。其主要原因在于染色质结构的多样性，从而导致不同类型的基因和蛋白质结合时形成不同的结合模式和功能作用。通过改变染色质结构，可调控细胞中几乎所有的蛋白质和器官的功能。这些功能的调控依赖于多种生物调控机理，包括互相作用、信号传导、分子作用和激活因子的作用。在下列几个方面，我们将介绍染色质结构在不同功能上的作用。
#### RNA免疫
染色质结构与RNA结合后，能够产生特殊的靶向RNA-binding protein，该蛋白质能够降解抑制抗原，促进细胞内RNA的翻译。这一过程类似于蛋白质和RNA相互作用的过程，但是染色质结构会影响RNA结合的模式，这种模式决定了蛋白质的表现。当一种RNA结合到染色质结构较多的区域时，其抗原还原力较强，因此在此区域释放的抗原应当能够抑制其活性，而不是像其与蛋白质相互作用时那样损伤RNA的活性。这种区域叫做RNA binding site（RBS）。通过在不同的RBS上释放抗原，还可以在同一种细胞类型或组织条件下选择性地调控细胞的RNA免疫功能。
>图4：不同RBS上释放的抗原对于RNA免疫功能的影响

#### 转录调控
细胞内有很多RNA分子，它们在细胞核内进行复制、转录、翻译、修饰以及各个基因的调控。在结构上，有些蛋白质和RNA结合形成一个特定的复合物，从而调控了特定的基因或RNA的转录。当一个蛋白质或者RNA与染色质结构间隔很远时，它就会被紧邻的染色质结构截获，使得它的功能受限。所以，染色质结构的位置和形状对于RNA的调控非常重要。因此，通过改变染色质结构的形状，就能够调节RNA的转录水平，使得细胞内特定类型的RNA获得更好的加工效果，从而改善细胞的功能。
>图5：染色质结构影响RNA转录调控

#### 细胞激活
在不同的细胞类型中，染色质结构的形态都不同，因此细胞激活往往也有不同之处。DNA-binding domain（DDBDs）、kinase complexes、chromatin remodeling complexes、and chromatin interactions can all regulate the expression levels of different types of genes or proteins in a cell. DDBDs bind to specific sites on DNA with multiple functional roles such as transcription factor binding, promoter activity, replication initiation, and tissue architecture. Kinetochore structures and loops are important components of chromatin fiber organization that control its function in regulating gene expression patterns across an entire cell. Remodeling chromatin creates new regions of chromatin by cleaving enclosing histones and non-histone chromatin proteins, which is critical for maintaining long-term genome stability and enabling signal transduction between cells. Finally, chromatin interactions occur through the dynamic interactions between chromatin compartments and their associated proteins that mediate downstream processes including transcript synthesis, translation, and protein folding. All these mechanisms contribute to the conformation of chromatin, resulting in the regulation of gene expression. In summary, although there may be many overlapping factors affecting cellular function, changes in chromatin structure play a crucial role in shaping and influencing this process.