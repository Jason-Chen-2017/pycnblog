
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Attribute-based image retrieval (ABIR) is an emerging research area that aims to retrieve relevant images based on the attributes of objects or scenes in a query image. Existing ABIR methods mainly focus on retrieving similar images by measuring the similarity between visual features extracted from the images themselves without considering the contextual information beyond the object itself. To address this issue, we propose a new dataset called FGVCA (Fine-Grained Visual Concepts Attributes), which includes over 70,000 fine-grained visual concepts with their corresponding attributes. We also provide two baselines: DAMSM and XBM, which are state-of-the-art deep models trained using FGVCA. These baselines can be used as starting points for further research on attribute-based image retrieval. Our experiments show that these two baselines outperform strong competitors such as CBIR and CNN-based approaches when evaluated on standard benchmarks like CUHK-SYSU and Market-1501. Moreover, our proposed dataset and baseline demonstrate the feasibility and utility of attribute-based image retrieval in a large scale fine-grained visual concept space. Therefore, they could serve as a foundation for future research in this field.
In this paper, we present the following contributions:

1. A novel dataset called FGVCA containing over 70,000 fine-grained visual concepts with attributes annotated at three levels of granularity (i.e., general/specific level, abstract level, and category level). The dataset is designed to cover various aspects of human visual perception ranging from subtle visual cues to highly concrete and semantic properties. 

2. Two baselines: DAMSM and XBM, both of which are deep neural networks trained using the FGVCA dataset. The former uses a fully connected layer followed by a softmax function while the latter performs convolutional pooling after each fully connected layer. Both baselines achieve state-of-the-art performance in terms of mean average precision (mAP) on standard benchmarks like CUHK-SYSU and Market-1501 compared to existing methods like CBIR and CNN-based approaches.

3. An extensive experimental evaluation demonstrating the superiority of DAMSM and XBM in attribute-based image retrieval compared to several other state-of-the-art methods such as CBIR and CNN-based approaches. Experiments reveal that DAMSM significantly outperforms all competing techniques including recent state-of-the-art triplet loss based methods like GNCM or PCBS. This demonstrates that attributed representations learned from FGVCA provide better discriminative power than visual features alone and can lead to significant improvements in attribute-based image retrieval tasks. Furthermore, the relative contribution of individual attributes towards overall visual concepts is well captured in our representation learning process, which provides insights into the underlying visual semantics of fine-grained visual concepts. 

Overall, our work advances the state-of-the-art in attribute-based image retrieval through the introduction of a high-quality, challenging dataset and two effective baselines. While future work should continue exploring different ways of exploiting fine-grained visual concepts, the introduced dataset and baselines offer a solid basis for further research in this area. Finally, our results suggest that attribute-based image retrieval may have applications in various fields where object recognition is involved, such as medical imaging, industrial inspection, augmented reality, and artistic creation. It could help improve many real world problems related to object recognition and understanding.

# 2.数据集介绍
## 数据集名称及介绍
我们提出的FGVCA数据集（Fine-Grained Visual Concepts Attributes）由以下属性组成：

1. 种类(Category): 提供51个丰富的图像特征类别，包括生活用品、交通工具、建筑材料、植物、动物等；
2. 细粒度(Specific Level): 提供19个粗粒度分类(如色彩、纹理、材质等)，每个细粒度又分为20多个具体的子分类；
3. 抽象级(Abstract Level): 提供36个抽象分类(如颜色、形状、大小、位置等)，每个抽象级别又分为若干具体的子分类。

总共有12835张图片，每张图片标注了13个不同维度的属性。我们从每个细粒度(Specific Level)出发，依次构建更小的抽象级(Abstract Level)。这样做可以保证数据的多样性、充实性以及重要性。

FGVCA数据集为广泛应用于机器视觉领域的计算机视觉任务提供了一个重要的数据集。它提供了丰富而详细的图像特征类别及其对应的属性，可用于各种机器学习、深度学习任务中。

## 数据组织结构
我们将FGVCA数据集按如下方式组织：
