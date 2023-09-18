
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Python Weekly 是一份由 Python 社区成员每周更新的新闻推送。目前，有很多优秀的 Python 框架、库、工具被开发出来，成为 Python 生态中不可缺少的一部分。文章以月刊的方式发布，每期都包括 Python 的最新热点新闻、库和工具等相关信息。

在过去的几年里，Python Weekly 一直坚持每周一次的形式，但随着社区的扩大，发布频率也在逐渐提升。本期 Python Weekly 将于 7 月份进行，将向大家展示最新的 Python 框架、库和工具。

虽然本期主题不局限于 Python，但是我们仍然从 Python 出发，因为作为最受欢迎的语言之一，它也是许多技术领域的主要编程语言。另外，对于已经熟悉 Python 的读者来说，可以参考到更多关于 Python 新功能的分享，对于刚接触 Python 的同学来说，也可以通过本期的文章快速了解 Python 的世界。

好了，今天就到这里。下面是本期 Python Weekly 的文章内容。
# 2. 最新消息

## 2.1 TensorFlow 2.0 正式版发布

TensorFlow 是一个开源机器学习框架，2.0版本已经正式发布，带来了诸如分布式训练、性能优化和改进的特性。

### 2.1.1 TensorFlow 2.0 新特性
- TensorFlow 2.0 全面支持 Keras 模型API，支持更加灵活的模型构建方式；
- 支持 eager execution 模式，可以像搭积木一样组装模型并直接运行；
- 提供了针对性能优化的 TensorFlow Profiler 和 tf.data API；
- 支持自动混合精度（AMP）和半静态数据类型（STFT）。


## 2.2 PyTorch 1.3 发布

PyTorch 是由 Facebook AI Research 团队在 6 月底开源的深度学习框架。

PyTorch 1.3 版本更新了 API，新增了对动态图(Dynamic Graph)的支持。该版本还新增了 ONNX 模型转换器，使得 PyTorch 模型能够转换为 ONNX 模型。

除此外，PyTorch 1.3 版本还新增了以下模块：
1. 对张量库NumPy的完全兼容性，可以方便地与NumPy交互；
2. 使用JIT(just-in-time)编译器，提高了模型的推理性能；
3. 更丰富的预置神经网络模型；
4. 增加了分布式训练和加载函数；



## 2.3 Python 3.8.0a4 发布

Python 3.8.0a4 是一个早期测试版本，它将于2019年7月9日发布稳定版本。主要更新包括：

1. 新语法特性:
   - 赋值表达式
   - 位置参数
2. 改进的标准库:
   - `random`: 性能提升
   - `hashlib` and `hmac`: 支持 FIPS 140-2 compliant hashing algorithm support
   - `math`: 对浮点数进行更好的处理
   - `typing`: 增强的类型注解
3. 其他改进


## 2.4 IPython 7.8.0 发布

IPython 是一个用于科学计算的 Python shell，其目的是提供一个高效的交互式环境，包括命令提示符，代码编辑器，文件浏览器和基于网页的 notebook 系统。

IPython 7.8 版本主要更新如下：

1. NumPy 1.17 支持

   IPython 可以使用 NumPy 1.17 的最新特性。

2. 添加 PEP 572 支持

   此次更新修复了当 Python 函数中的形参被解析时出现异常的问题。

3. 更新内核机制

   IPython 7.8 版增加了一个新的内核管理机制，允许用户配置多个 IPython 内核，并切换它们之间的工作环境。


## 2.5 VSCode 1.38 发布

VSCode 是一个现代化的跨平台源代码编辑器，具有简洁的界面，可同时编辑不同类型的文件，支持扩展插件。

VSCode 1.38 版本更新内容如下：

1. 拼写检查

   当您编辑时，VSCode 会显示拼写建议。如果单词拼写正确或类似于拼写错误的单词，它会用黄色标注。

2. 设置同步

   当您首次登录 VS Code 时，将根据您的 GitHub 帐户设置同步你的所有设置。

3. 支持更多快捷键组合

   您现在可以自定义各种常用的快捷键组合，例如打开 Explorer 窗口或者搜索文件。

4. Jupyter Notebook 插件

   此次更新还包括对 Jupyter Notebook 的支持，并且提供了对单元格运行结果的渲染和可视化能力。


## 2.6 PyCharm 2019.3 发布

PyCharm 是一个专业级的 Python IDE，提供的代码分析、完成、调试和导航工具。

PyCharm 2019.3 版本更新内容如下：

1. Type annotations

   PyCharm 现在支持类型注释，包括函数返回值类型，变量类型和导入的模块的路径。

2. 新的 Python 远程调试器

   在 PyCharm 中，你可以选择在本地启动远程调试器，或者在远程主机上连接到正在运行的远程调试器。

3. 改进的性能

   PyCharm 的性能得到了改善，特别是在工程目录树中浏览项目时。


## 2.7 TensorFlow Addons 0.9.1 发布

TensorFlow Addons 是用于构建 Tensorflow 2.x 子包的地方。

TensorFlow Addons 0.9.1 版本更新内容如下：

1. Image Segmentation APIs (image, video, text):
   
   - Instance Segmentation: 新的实例分割API。
   - Mask RCNN: 使用Faster R-CNN实现Mask RCNN模型。
   
2. Kinetics 400 数据集: 
   
   - 来自Kinetics 400数据集的视频分类模型。
   
3. Model Garden 上的更多模型示例: 
   
   - 包括 TensorFlow SavedModel 的最小化示例，帮助你理解 SavedModel 文件大小的影响。
   - 包括 PixelCNN++ 和 PixelSNAIL 的变体实现，方便你尝试新的结构。
   - 有关在嵌入式设备上部署模型的示例。
   
4. 其他改进。


## 2.8 DataCamp 升级到 Version Control 1.1.2 

DataCamp 是一个面向数据科学教育的平台，支持数据分析、仿真、建模和协作工作。

Version Control 1.1.2 版本更新内容如下：

1. 更精准的语法检测

   已修复的语法检测对 markdown 文件不起作用。

2. 公开或私密项目切换

   用户现在可以自由切换他们的公开项目，而无需通知 DataCamp 管理员。

3. 存储库克隆

    从云端克隆存储库现在需要 DataCamp 订阅。

4. 支持 JupyterLab 4.0 及以上版本


# 3. 技术前沿

## 3.1 PyTorch Lightning: 一款轻量级且易于使用的PyTorch模块

PyTorch Lightning 是一个简单且高度可扩展的 PyTorch 模板，用于加速研究过程。该模块的目标是提供一个简单，统一的方法来组织您的 ML 管道并达成最佳实践。

Lightning 具有以下优点：

1. 快速启动：Lightning 使用最先进的技术默认设置，为您提供即插即用，方便的接口。
2. 可重复性：Lightning 模块允许您存储配置和代码，以便使用相同的超参数进行重新培训。
3. 可移植性：Lightning 的设计目标是尽可能轻松地在任何硬件上运行。
4. 效率：Lightning 以一种高度模块化和自定义的方式组织您的代码。
5. 可扩展性：您可以使用 Lightning 轻松添加自定义代码来扩展自己的模型。


## 3.2 Pytorch Geometric: 用于图神经网络的PyTorch扩展

PyTorch Geometric 是用于图神经网络的高级框架。它基于 PyTorch 的低级张量库，并提供卷积层、池化层、距离度量、聚合层和其他重要组件，用于定义、训练和评估图神经网络。

PyG 具有以下优点：

1. 简化编码：PyG 为常见任务提供基类，如图分类、节点分类、链接预测和链接预测。
2. 统一的 API：PyG 提供统一的 API 来定义、训练和评估图神经网络。
3. GPU 支持：PyG 利用 GPUs 进行快速运算。
4. 丰富的可用组件：PyG 提供了广泛的组件，如图卷积网络、全连接网络、图注意力机制、图池化层、图距离度量等。
5. 高度模块化：PyG 中的组件均可单独使用。


## 3.3 scikit-learn-mooc: 基于Jupyter Notebook的MOOC课程

Scikit-learn 是一个开源的 Python 机器学习库。

MOOC课程 Scikit-learn-mooc 是一个基于 Jupyter Notebook 的免费 MOOC 课程，适用于希望进一步了解机器学习库的初学者。

课程内容涵盖了 scikit-learn 各个模块的基础知识、数据集和评估指标、线性回归、决策树、聚类、降维等内容。


## 3.4 Google AI Language: 面向多种语言的AI模型

Google AI Language Team 是 Google 开发的一系列文本处理服务，包括自动翻译、自动摘要、问答助手、意图识别和多种语言理解模型。

该团队发布了一系列模型，包括下列模型：

1. Neural Machine Translation: 使用 TensorFlow Lite 在 Android、iOS 上实现的神经机器翻译模型。
2. Document Understanding: 一个基于 TensorFlow 的文档理解模型，可以自动理解文本文档的内容。
3. Conversational Assistance: Google 助手助手可以在应用中提供实时语音响应。
4. Natural Language Understanding: 基于 TensorFlow 的自然语言理解模型，提供意图识别、槽填充和查询理解。


## 3.5 Turi Create: 用于快速和直观地构建机器学习产品的工具

Turi Create 是一个用于快速构建机器学习模型的工具，能够帮助您创建自定义模型，并使用 Apple CoreML 或 Apple Core NLP 进行部署。

Turi Create 提供了以下功能：

1. SFrames: Turi Create 中的 SFrame 是一种稀疏矩阵表示，它用于存储和处理非常大的数据集。
2. 图像分类器: Turi Create 包含多个预训练的图像分类器，可帮助您快速进行实验。
3. 导出为 Core ML: Turi Create 可将自定义机器学习模型导出为 Core ML 格式，用于在 iOS、macOS、tvOS、watchOS 上部署。
4. 创建流水线: 通过命令行工具或 Python API，Turi Create 可帮助您轻松地创建机器学习模型的流水线。


# 4. 竞赛与奖项

## 4.1 Kaggle: 用于机器学习和数据科学竞赛的平台

Kaggle 是一个为机器学习和数据科学社区提供竞赛和奖励的平台。

最新获奖情况如下：

1. 第七届 Data Science Bowl 2019: Kaggle 举办的第一届数据科学冠军赛，为解决复杂且实际的反欺诈问题而聚集了一群数据科学爱好者。奖金总额为 $50,000。
2. 第十四届 Data Hackathon: Kaggle 举办的第九届数据科学人才锦标赛，邀请来自全球的顶尖数据科学家参加，共同解决关于“不平衡数据”和“多模态数据”的难题。奖金总额为 $30,000。
3. 第三届 Datacamp World Champions: Kaggle 举办的第五届数据科学冠军赛，为参加比赛的用户提供在线学习环境，帮助参与者提升数据分析技能。奖金总额为 $10,000。


## 4.2 IBM Watson: 大数据分析平台

IBM Watson 是一个构建大数据分析应用程序的平台。

Watson Studio 旨在简化数据科学工作流程，并提供可扩展的工具来构建、训练和部署数据科学模型。

除了其它优势，Watson 还提供了一套完整的套件，包括下列内容：

1. IBM Cloud Pak for Data: 在 IBM Cloud 上部署数据科学应用。
2. AutoAI: 根据大数据的特征和标签，自动生成机器学习模型。
3. OpenScale: 监控模型质量和运行状况，确保您的服务始终处于最佳状态。
4. Watson Knowledge Catalog: 用于发现、整理和组织企业知识的云服务。


## 4.3 AWS DeepRacer League: 一个开源机器人比赛

AWS DeepRacer League 是一项由来自 AWS 的贡献者举办的开源机器人比赛。

比赛设定很简单，目标就是通过编程控制汽车完成特定任务，并以这个游戏模式来验证自己所编写的程序是否有效。

有兴趣的开发者可以参与其中，并获得 AWS 认证开发者，获得奖励，赢取大量奖金。
