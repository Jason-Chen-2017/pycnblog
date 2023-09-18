
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在科技领域,无论是计算力还是存储空间都越来越便宜,我们可以利用这些资源开发出更高效、更精准的模型和服务.但同时,我们也发现了一些创新可能,比如物体检测、图像分割、自然语言处理等.通过有效利用机器学习的方法,计算机视觉领域已经从单纯的图像分类/识别演变成了复杂的任务——物体检测/跟踪,图像分割,基于视觉的智能交互系统,甚至可以进行全身动作识别.

人类面对复杂环境时,显得情绪多样、变化快,而且身体活动范围广泛,动作表情丰富.通过摄像头拍摄的人脸视频数据,传统的基于图片的模型往往无法很好地解决这个问题.近年来,计算机视觉领域的发展取得了重大的进步,基于卷积神经网络(CNN)-based的方法在解决各种视觉任务方面取得了突破性的进步.在本文中,我们将介绍一种新的基于CNN的视频情感识别方法——CNN-based facial expression recognition in videos.本文的作者团队在CVPR2017上发表了一篇新的工作,试图通过训练一个卷积神经网络(CNN)-based模型来解决视频情感识别问题.

# 2. 相关研究
## 2.1 Convolutional Neural Networks for Video Classification
之前的基于CNN的视频分类方法,如Long Short Term Memory(LSTM)-based、Hierarchical LSTM(HLSTM)、Fully Convolutional Network(FCN)-based等都是建立在固定长度帧序列的假设基础上的.由于视觉事件发生的瞬间非常短暂,因此短期内的视觉特征难以捕捉全局信息.而对于长期的行为事件,如短时间内连续的暴风雨、骚乱、枪击事件,这一局部的视觉事件依然能够给出足够的信息.因此,将多个相邻的帧当做一个单独的输入不太合适,需要进行复杂的预处理,如视频序列的抽取和裁剪.

为了处理这些问题,Li等人提出了一个新的深度学习框架——视频卷积神经网络(VidNet).该网络将两个路径连接起来:首先,它利用堆叠的卷积层提取局部特征,从而在较低的空间分辨率下学习到时间相关的信息;其次,它利用双向循环神经网络(Bi-RNN),对全局特征进行编码并输出整个视频的分类结果.因此,VidNet既考虑到了局部信息又保留了全局结构,具有良好的健壮性和鲁棒性.

## 2.2 FER+: A Multi-modal Deep Learning Framework for Facial Expression Recognition from Videos
OpenCV中的haar特征,以及用于实时情感识别的基于HOG特征的算法,提供了一种简单、直观的方法来实现视频情感识别.但它们仍存在很多限制,如特征空间大小和数量的限制、特征表示缺乏全局上下文信息、无法捕获静态图像特质.基于此,Li等人提出了一种新的基于深度学习的视频情感识别方法——FER+.该方法基于三个模块来处理不同的视觉信息:光流特征模块、静态图像特征模块和上下文信息模块.

光流特征模块提取了视频中的运动轨迹,通过利用光流场可以捕捉到物体运动的空间和速度信息.静态图像特征模块通过对视频帧的静态图像进行特征提取,可以捕捉到不同视角下的空间分布信息.上下文信息模块采用注意力机制来融合全局上下文信息,包括视频全局结构和视频中的人物动作信息.通过三个模块的组合,该方法可以提取出丰富的全局信息,对视频中人物的情感表达进行更加准确的识别.

## 2.3 Multimodal sentiment analysis on Twitter using convolutional neural networks with linguistic features and visual data
Twitter是一个活跃的微博平台,每天都产生着海量的动态消息.因此,Twitter情感分析一直是个热门话题.传统的基于文本的情感分析方法主要依赖于词汇级情感的标注,缺乏全局上下文信息.但是在当前的语境下,使用视频数据进行情感分析才真正契合现代社会的需求.

Vishwanathan等人在ICWSM2015上发表了一篇新的研究,探索了如何将视觉数据和语言数据结合起来进行情感分析.他们使用了带有预训练权重的AlexNet作为图像模型,将其最后的输出与基于CNN的语言模型(word embedding+GRU)结合,来学习视频中的情感标签.与之前的情感识别方法不同的是,他们使用带有视觉和语言特征的数据集来训练模型,并且得到了更好的性能.

# 3. CVPR2017: A deep learning framework for video emotion recognition based on spatial and temporal feature fusion
CVPR2017提供了一系列的图像和视频数据集,用于训练和评估新的情感识别模型.在本文中,我们要重新审视一下基于CNN的视频情感识别方法——CNN-based facial expression recognition in videos.

## 3.1 模型概述
CNN-based facial expression recognition in videos可以分成以下几个步骤：

1. 数据集准备: 本文使用Fer2013数据集，共有3589条视频数据，其中有1870条训练数据，719条测试数据；共有7 expressions，分别为angry、disgust、fear、happy、sad、surprise、neutral。其中，“neutral”即没有表情或不确定性的情绪。训练数据中包含两千四百五十二张图像，测试数据中包含七十六张图像。
2. 模型设计: 本文采用的模型为AlexNet，它是一个深度卷积网络，由八个卷积层（5x5，stride=1）、三个全连接层组成。模型的输入大小为$227\times 227\times 3$，有25 million parameters。在AlexNet的基础上，对其进行微调，使用ReLU作为激活函数，不使用max pooling。
3. 模型训练: 在FerPlus数据集上进行训练，使用SGD随机梯度下降法优化器，初始学习率为0.001，使用L2正则化，batch size=64，在每轮迭代后减小学习率到0.0001。训练过程中使用early stopping策略，当验证集的准确率停止增长时，停止训练。
4. 预测结果：使用L2范数作为评价指标，以均值方差作为标准化方式，预测每个视频片段的情绪类别及置信度。对每段视频，选取最佳匹配的分类结果，忽略置信度低于阈值的结果。最终的预测结果通过平均所有视频片段的结果得到。

## 3.2 模型架构
模型架构如下图所示：


该模型包括三层卷积层，每层由两次卷积+BN+ReLU构成，与AlexNet类似，前三层的参数个数比AlexNet少。然后是两个线性层，再接一个softmax层。由于我们只进行情绪识别，所以最后一层仅有一个softmax单元。中间过程的输出尺寸为$256\times 8\times 8$，经过dropout层后减少到$256\times 4\times 4$,之后进入全连接层。模型的损失函数为cross entropy loss，训练时使用dropout防止过拟合。

## 3.3 模型超参数设置
训练模型时，我们使用L2正则化，初始学习率为0.001，batch size为64，epoch次数为100。在每轮迭代后减小学习率到0.0001。模型训练的最后，我们选择验证集上的准确率最高的模型作为最终的预测模型。

## 3.4 模型效果
### 3.4.1 数据集划分
本文使用Fer2013数据集，共有3589条视频数据，其中有1870条训练数据，719条测试数据。在训练数据中，有2千四百五十二张图像；在测试数据中，有七十六张图像。

### 3.4.2 模型效果
#### 3.4.2.1 模型准确率
作者训练了两次模型，每次模型训练时间约3小时，测试集的准确率分别为94%和93.5%.其中，第一次模型的训练参数设置为momentum=0.9，batch size=64，epoch数为100，学习率为0.001；第二次模型的训练参数设置为momentum=0.9，batch size=64，epoch数为50，学习率为0.0001。作者对两种模型的准确率都进行了比较。

作者提到，"In this work we do not use any other hyperparameters such as dropout rate or regularization factor. We keep the momentum constant throughout training."，并说道："We also note that the choice of batch size is quite important to achieve good performance and generalize well to new unseen data. In our experiments, we choose a batch size of 64 which is a typical value used for large-scale image classification tasks."

#### 3.4.2.2 模型的可解释性
作者对每个模型的卷积核进行了分析，分析结果显示出模型在不同层对不同表情类型的响应强度。

作者还分析了模型对于人脸的位置、姿态、光照条件等因素的影响。

# 4. 作者总结
本文介绍了一种新的基于CNN的视频情感识别方法——CNN-based facial expression recognition in videos。

基于卷积神经网络的方法，在解决视觉任务方面取得了突破性的进步。本文提出的模型通过多模态的融合方案，提升了视频情感识别的效果。

本文的作者团队在CVPR2017上发表了一篇新的工作，试图通过训练一个卷积神经网络(CNN)-based模型来解决视频情感识别问题。在本文中，作者详细介绍了模型的设计、训练、预测结果等过程。

最后，作者总结了本文的研究收获和未来的发展方向。