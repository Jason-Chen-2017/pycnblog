
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## GAN概述
Generative Adversarial Networks (GANs) 是近几年提出的一种新型的无监督学习方法，它的主要目的是利用生成模型来对数据进行建模，使得机器能够自己创造新的样本。在2014年，GAN被提出并取得了非常成功的成果，之后也被应用到很多领域。通过GAN可以训练出高质量的图像、音频、文本、视频等多种类型的数据。由于GAN的成功，越来越多的研究人员和开发者开始关注并尝试解决GAN的问题，探索其潜在价值，并开发出更加有效、便捷的应用。因此，有必要深入了解GAN的相关知识，了解它的产生背景、产生原因、它的特性、它的优点、它的局限性，以及它的未来展望和发展方向。

## GAN会议概况
GAN（Generative Adversarial Networks）自2014年提出后受到了广泛的关注和关注，由此产生了一批关于GAN的学术会议，这些会议包括CVPR(Computer Vision and Pattern Recognition),ICML(International Conference on Machine Learning)，IJCAI(International Joint Conference on Artificial Intelligence)。其中，ICML是最重要的GAN会议之一，它是主流的GAN会议。ICML 2019上发布了《Generative Adversarial Nets》一文，文章对GAN的基本原理、主要方法、数学推导、基于真实数据的评估以及未来前景进行了详细阐述。2020年的ICML将继续举办，重点关注GAN的最新进展和前沿工作。
除了ICML之外，其他的GAN会议也在不断地举行。包括NeurIPS、ECCV、IJCNN等。另外，还有一些GAN子社区也会组织一些GAN会议，例如CVPR的GAN委员会、NIPS的GAN协会等。

## GAN会议及特色分享
### CVPR 2019 - Generative Adversarial Nets
ICML 2019 将GAN 作为计算机视觉和模式识别领域的一个热门研究课题，所以CVPR也被称为ICML的一个分会场。作为第一届CVPR，CVPR 2019 在2019年4月8-12日在比利时布鲁塞尔举行。一般来说，每年的CVPR都会举办一次GAN研讨会，论坛形式，邀请GAN方面的大牛演讲、分享他们的最新研究成果、经验教训、以及未来的规划。CVPR会议的主题与GAN密切相关，包括但不限于：生成模型、图像合成、风格迁移、多模态生成、可控性、模糊、标签伪造、数据增强、指标评估、分布偏移、增强学习、条件GAN、GAN的应用等。2019 年 ICML 的Generative Adversarial Nets 学术报告主要聚焦于三个方面：原理、方法、实践。

Generative Adversarial Network（GAN）是一个基于生成模型的无监督学习方法。它由两个相互竞争的网络，一个生成器G，一个鉴别器D，组成。生成器负责产生具有真实感的样本，而鉴别器负责辨别生成样本与真实样本之间的差异，并帮助生成器提升性能。两者相互博弈，生成器通过迭代的方式不断试图欺骗鉴别器，逼迫鉴别器判断生成的样本是“假”还是“真”。直至生成器达到足够好的效果，使得鉴别器无法再分辨出生成样本与真实样本的区别，从而结束游戏。

Generative Adversarial Networks 有如下几个特点：

1. 可学习特征空间
GAN 使用了一个可学习的特征空间，让生成器输出的样本看起来更像是真实样本而不是噪声。

2. 生成分布
GAN 把生成分布表示为一个多维的概率分布，这个分布可以由用户指定，也可以由 GAN 自适应调整。

3. 判别能力
GAN 的判别能力强。它可以在很小的样本集上训练，并且可以在不同的条件下生成高保真度的图像。

4. 高阶生成
GAN 可以通过控制隐变量 z 来实现高阶生成，这对生成高质量的图片有着至关重要的作用。

### IJCAI 2020 - Rethinking BiGAN: A Fresh Look at Unsupervised Representation Learning from Biased Data
IJCAI 是中国计算AI领域的一支重要会议。IJCAI 2020 以“Rethinking BiGAN: A Fresh Look at Unsupervised Representation Learning from Biased Data”开幕，邀请国际顶尖的研究者、数据科学家、工程师和企业家一起探讨如何通过自适应的正则化约束条件缓解 BiGAN 模型的过拟合现象。本期报告主要围绕如下问题展开：

1. 什么是 BiGAN？BiGAN 主要解决的是什么样的任务？

2. 为什么 BiGAN 需要正则化？为什么需要优化器？

3. 如何缓解 BiGAN 过拟合？提出了几个不同正则化手段。

4. 如何评估 BiGAN 表现？验证集上的预测误差如何衡量 BiGAN 的好坏？

5. 最后，分享了一些个人观点和想法，希望听众能对该研究领域有所启发。