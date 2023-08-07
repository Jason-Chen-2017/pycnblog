
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2月9日，我在知乎发布了一条知乎专栏文章《如何学习深度学习？》，里面的热门话题之一就是“机器学习入门”，其中有一个分支专栏计划叫做《机器学习入门课程》，将从零开始带领大家了解机器学习的核心概念、机器学习的应用场景、机器学习的基本算法、模型评估指标以及常用框架（包括TensorFlow、PyTorch和Keras）等，并通过实战案例让大家真正动手实现机器学习任务。
         
         2021年已经过去两个月了，社区上关于“深度学习”的各种教程、框架、资源层出不穷，但学习这些知识并不是一个易事，因为实际上要学习很多内容，而且由于时间和精力的限制，我们需要快速地把一些基础的内容掌握起来，以应对实际的业务需求。
         
         本文是《机器学习入门课程》系列文章中的第十期课，主要以“图像分类”的项目作为实战案例，使用Python语言的开源深度学习框架FastAI来实现这个项目，其余内容会在之后陆续更新完善。
         # 2.FastAI库简介
         FastAI是一个基于Python的深度学习框架，它提供了专门针对深度学习领域的高级API，可以帮助开发人员轻松地构建、训练并部署深度学习模型，解决实际问题。
         
         ### 2.1 安装
         首先安装最新版本的Anaconda，然后运行命令安装fastai库：

         ```python
         conda install -c fastai -c pytorch -c anaconda jupyter
         pip install fastai==2.5.3
         ```
         
         您可以运行以下命令验证是否安装成功：

         ```python
         python -m fastai.utils.show_install
         ```

         ### 2.2 数据集
         FastAI目前支持的常用数据集包括：MNIST、CIFAR-10、ImageNet等。本文选择用CIFAR-10数据集，这是目前最常用的图像分类数据集之一。您可以按照如下方式下载该数据集：

         ```python
         import fastai.vision.all as vision
         
         path = untar_data(URLs.CIFAR)
         tfms = get_transforms()
         data = vision.ImageDataBunch.from_folder(path, valid='test', ds_tfms=tfms, bs=16).normalize(cifarstats)
         ```

         上述代码会自动下载CIFAR-10数据集并划分为训练集和测试集，并指定一些数据增强的方法和图片标准化方法。
         ### 2.3 模型
         在这里，我们将采用一种经典的卷积神经网络——Resnet来完成这个项目。

         ResNet是由微软研究院团队于2015年提出的，是深度学习领域里用到的一个非常著名的模型。它在当时取得了比其他深度学习模型更好的性能，在ImageNet、Places、CIFAR-10等众多视觉任务中都取得了优秀的成绩。
         
         这里，我们将使用Resnet18来进行图片分类。

          ```python
          from torchvision.models import resnet18
          model = nn.Sequential(*list(resnet18(pretrained=False).children())[:-1],
                                nn.AdaptiveAvgPool2d((1,1)))
                                  
          for param in model.parameters():
              param.requires_grad = False
                      
          nclass = len(data.classes)
          head = nn.Linear(in_features=512, out_features=nclass)
          model = nn.Sequential(*(model,head))
          ```

          上述代码定义了一个全连接层作为分类器，并将Resnet18的最后几层（即全连接层）进行了冻结，防止它们的参数被训练。

          