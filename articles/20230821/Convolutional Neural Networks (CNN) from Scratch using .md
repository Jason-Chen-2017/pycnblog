
作者：禅与计算机程序设计艺术                    

# 1.简介
  

卷积神经网络（Convolutional Neural Network, CNN）是近年来一个非常热门的研究方向。它能够自动提取图像中的特征并运用其进行分类、检测、跟踪等任务，被广泛应用于视觉领域，取得了很大的成功。但是学习和实现一个完整的CNN模型仍然是一个比较困难的事情，尤其是在没有相关库或框架帮助下的。在这个系列的教程中，我将从头到尾基于Python/NumPy进行卷积神经网络的实现，不需要借助任何现成的库或框架。尽管这样做会比较枯燥，但你可以在自己的项目中参考一下，也可以作为学习或者临时使用机器学习技巧的入门指南。本文作者不是一位专业的科研工作者，只是个爱好者，所以本文不会涉及太多专业词汇。本文适合具备一定编程基础的人阅读，建议阅读顺序如下：

6. Building Complex CNN Architectures: Part I - 使用残差块和Inception模块来构造更加有效的深度CNN。(coming soon)

**Note:** This article is not just about teaching you how to implement CNNs, but also teaches you the fundamental concepts behind them which will help you understand advanced topics like Residual networks or Inception modules. So make sure that you have read through all previous articles before continuing this one as they build on each other. Also note that some concepts may be more easily understood after reading the previous articles because we are building upon the knowledge gained earlier. 

## What You Will Learn

In this series of tutorials, we will learn to build convolutional neural networks from scratch by implementing it using only NumPy and basic Python libraries such as numpy, matplotlib, etc. We will go over every step involved in creating a simple CNN architecture and advance it into more complex architectures by adding residual connections and inception modules. By the end of this tutorial, you will be able to create your own CNN models and experiment with different techniques and ideas to improve performance. If you want to deep dive into these topics further, please refer to the official resources provided at the end of the article.