
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Welcome to part two of our PyTorch tutorial series!

In this article we will be learning about activation functions and how they are used in neural networks. We'll also look into some popular types of activation functions such as ReLU and Softmax. You should have a good understanding of linear algebra concepts before you continue with this tutorial. If you don't know what is meant by "linear algebra", I suggest watching one or more Linear Algebra tutorials on Youtube first before proceeding further. 

We assume that readers have already gone through the previous part of our tutorial which was about building simple neural networks using PyTorch. So if you haven't read it yet, please go back and do so now.  

# 2.激活函数(Activation Function)
An important concept in machine learning and deep learning is the activation function. It is responsible for transforming input data from one form to another. The most common type of activation function is Rectified Linear Unit (ReLU), which simply sets all negative values in an array to zero and leaves all positive values unchanged. Other commonly used activation functions include Sigmoid, Tanh, and Softmax. Let's start with explaining each of these in detail.<|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>><|im_sep|>