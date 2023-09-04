
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　近年来随着计算机视觉领域的火爆，卷积神经网络(Convolutional Neural Network, CNN)在图像分类、目标检测等领域中越来越受到关注。本文将从人工神经网络（Artificial Neural Networks, ANN）到卷积神经网络（Convolutional Neural Networks, CNN），详细剖析CNN是如何工作的，以及为什么能够取得如此成就。
        　　CNN背后的理论基础仍然是人工神经网络的概念和结构。我们首先回顾一下人工神经网络（ANN）的基本概念和结构。
        # 2.基本概念及术语
        ## 2.1人工神经网络（Artificial Neural Networks，ANNs）
       　　人工神经网络（Artificial Neural Networks，ANNs）是由连接各个处理单元(Processing Unit)的多层网络构成。这些处理单元可以是处理器（Processor）或者微处理器（Microprocessor）。每个处理单元都拥有输入端(Input End)，输出端(Output End)，以及多个权重和偏置参数。然后输入数据经过处理单元之间的连接传递，其结果会反馈给最后的输出层。我们可以使用不同的激活函数（Activation Function）来控制网络的非线性和对称性，比如Sigmoid、Tanh、ReLu等。如下图所示：


        上图中的数字表示的是该层神经元的个数。输入层接收原始图像的数据，中间的隐藏层进行特征提取并转换成适合分类的特征向量，最后输出层使用softmax函数计算出分类概率值。ANN是一个高度非线性、高度对称、高度参数化的模型，能够模拟复杂的非线性关系。
        
       ## 2.2卷积神经网络（Convolutional Neural Networks，CNNs）
       　　卷积神经网络（Convolutional Neural Networks，CNNs）是一种基于图像的神经网络模型。它具有以下几个特点：

           - 模型简单、易于训练；
           - 可以捕捉到图像的全局信息；
           - 不仅能够处理图片类别而且能够识别不同大小的目标；
           - 能够学习到局部特征和全局特征；

       　　CNN最主要的创新之处就是使用了卷积（Convolution）运算。CNN通过滑动窗口的方式，对输入的图像进行特征提取，提取出重要的特征，然后再进行分类。卷积是指对输入数据进行滤波，得到一个新的二维数组作为输出。如下图所示：
