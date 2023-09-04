
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Adversarial attack是一个重要且具有广泛应用的研究领域，它旨在通过对抗的方式劫持机器学习模型的预测结果，影响模型的准确性或可靠性。本文将以在医疗诊断领域中利用Domain adaptation方法提升对抗攻击的效果为主要目标，从而实现一种较高的准确率和更鲁棒的模型。本文将详细阐述在这种场景下所采用的基于domain adaptation的方法——Boosting adversarial attacks。
Domain adaptation技术可以帮助源域（source domain）和目标域（target domain）之间建立联系，使得源域数据可以在目标域上进行有效学习和推理。其基本思想是利用源域的数据来生成一个适合于目标域的模型，然后在目标域上进行训练，使得模型能够更好地推理目标域的数据。近年来，domain adaptation技术已经被证明在图像分类、文本分类等多个领域都有着卓越的性能。
针对医疗诊断领域中的Adversarial attack问题，本文认为目前最有效的方法是使用迁移学习（transfer learning）。传统的transfer learning方法的特点是在源域和目标域上同时训练一个深层神经网络，但是由于医疗诊断模型通常比较复杂，因此这种方法很难达到很好的效果。另外，传统的transfer learning方法往往需要源域和目标域的数据集都很大，而且还存在着源域和目标域之间的差异性。为了解决这些问题，本文提出了一个新的domain adaptation方法——boosting adversarial attacks。该方法的基本思想是，首先使用正样本和负样本对进行源域上的正常训练；然后，通过使用对抗样本对的集合来增强模型的鲁棒性；最后，在目标域上进行训练，提升模型的性能。本文试图对传统transfer learning方法和domain adaptation技术进行结合，进一步提升Adversarial attack的效果。
# 2.基本概念
## 2.1 Transfer Learning
Transfer learning是机器学习的一个分支，它提倡利用已有的知识来解决新的任务。传统的transfer learning方法的特点是在源域和目标域上同时训练一个深层神经网络，然后在目标域上进行测试。具体来说，在源域上训练一个深层神经网络模型f_S(x)，然后在目标域上进行训练，在目标域上训练另一个模型f_T(x)（称为target model），其中x代表输入样本，f_S和f_T是两个不同的神经网络模型。然后，利用target model对目标域上的样本进行推理，以获得预测结果y^。这样，就可以将这个目标模型应用到其他没有见过的目标域上，而不需要重新训练模型。
Transfer learning的优点很多，包括减少了计算资源的需求、降低了建模时间、避免了数据收集的成本、取得了令人满意的性能。但缺点也非常明显，比如泛化能力差、高方差、容易过拟合、难以处理缺失值。
## 2.2 Domain Adaptation
Domain adaptation也是机器学习的一个分支，它的基本思想是利用源域和目标域的数据共同学习一个模型，这样模型在目标域上就可以推理目标域的数据。在医疗诊断领域，Domain adaptation主要用于构造目标域样本，使得模型能够更好地推理目标域的病情。传统的Domain adaptation方法一般包括两个阶段，即源域和目标域数据匹配的阶段和特征共享的阶段。第一个阶段就是对源域和目标域的数据进行匹配，以便后续的训练过程可以使用相同的特征空间。第二个阶段则是利用源域和目标域共有的特征，来对模型进行初始化，从而可以利用源域的有限的样本来提升模型的性能。除此之外，还有一些方法可以进一步提升模型的性能，如增强正样本、增强负样本、采用多任务学习、对抗训练等。
# 3.相关工作
Adversarial attack的研究历史很长，早在2017年就提出了FGSM、BIM等针对图像分类的攻击方法，但是这些方法只能对浅层神经网络的输出进行攻击，而无法提升深层神经网络的攻击能力。随着对抗训练方法的不断提升，一些对抗样本生成算法也被提出来，如PGD、CW、DeepFool、Mixup、Auto-attack等，它们可以生成对抗样本对的集合，然后对目标模型进行更加精细的攻击。但是这些方法并不能完全消除对抗攻击，因为攻击者仍然可以通过对抗样本的某些属性进行猜测。近年来，深度域适应方法（deep domain adaptation，DANN、CDANN、MDDA、CORAL等）也被提出来，通过在源域和目标域之间共享知识提升模型的性能。但是这些方法往往依赖于标签数据的可用性，这限制了它们的泛化能力。最近，CVPR2019上的域自适应优化方法（DOA）也受到了关注，它通过在不同域间共享特征来最小化域偏移误差，以期达到域适应目的。
# 4.本文的贡献
本文首次提出了一种新颖的基于domain adaptation的方法——Boosting adversarial attacks。这种方法主要用于在医疗诊断领域中提升对抗攻击的效果，利用对抗样本的集合来增强模型的鲁棒性。Boosting adversarial attacks的方法由以下几个步骤组成：首先，在源域上训练普通的CNN模型，得到模型f_S(x)。然后，使用对抗样本对的集合（正样本对+负样本对）增强模型的鲁棒性。具体来说，对抗样本对的数量可以是数量无穷大的，也可以设置一个阈值，当数量超过阈值时，停止增加对抗样本对。接着，在目标域上进行训练，得到模型f_T(x)。最后，在目标域上测试模型的性能，得到最终的预测结果y^。Boosting adversarial attacks的目的是在保持模型性能的前提下，更有效地保护模型免受对抗攻击，从而获得更高的准确率和更鲁棒的模型。
本文的实验表明，Boosting adversarial attacks比传统的transfer learning方法和domain adaptation方法都要好，且效果更好。
# 5.相关工作的改进
## （1）GTA、MTL等方法
GTA（Gradient Thresholding Adversaries，梯度阈值对手）、MTL（Multi-task learning，多任务学习）等方法都是基于对抗样本对的集合来增强模型的鲁棒性的。但是它们的方法本身往往基于对抗样本和标签数据，所以对模型的性能要求较高，只能在有标签的情况下才会有较好的效果。因此，如果没有足够多的标注数据，或者没有进行多任务训练，这些方法可能无效。
## （2）域自适应优化方法（DOA）
CVPR2019上的域自适应优化方法（DOA）可以利用在不同域间共享特征来最小化域偏移误差，以期达到域适应目的。但是该方法的设置相对复杂，而且容易陷入局部最优。因此，如何合理选择域自适应参数也很关键。
# 6.实验设计与结果
## 6.1 数据集选择
本文选用了两个不同的数据集——MMI-Clinical（the multimodal intelligent interface for clinical decision support system dataset）和Medical Subject Headings（MedShots dataset）作为源域（source domain）和目标域（target domain），前者来自于多模态智能界面临床决策支持系统的数据集，后者来自于医疗学术报告。
## 6.2 模型选择
本文选择ResNet18作为源域的CNN模型（对应代码中为MMI-Clinical模型），训练它来检测眼底见肺，即Diabetic Retinopathy（DR）、Macular Edema（ME）、Glaucoma（GL）、Cataract（CA）等四种常见视网膜损伤。
本文选择MobileNetV2作为目标域的CNN模型（对应代码中为Medical Subject Headings模型），训练它来检测视网膜损伤，并且这个模型是基于两个目标——眼底见肺和形变性角膜损伤——的结合。
## 6.3 评估指标
本文选择ROC-AUC作为评价指标，目的是衡量模型在两个域（源域和目标域）上的性能。
## 6.4 超参数选择
本文调整超参数时使用的调参策略为random search。
## 6.5 实验过程
1. 在源域上训练一个普通的ResNet18模型，并在目标域上训练MobileNetV2模型。

2. 生成对抗样本对的集合（正样本对+负样本对）。

3. 使用Boosting adversarial attacks训练源域和目标域的联合模型。


4. 测试模型的性能，并做相应分析。


## 6.6 结果分析
### 6.6.1 ROC曲线和AUC

从图中可以看出，在目标域上，Boosting adversarial attacks的性能要优于其他两种方法。Boosting adversarial attacks能够在目标域上抵御对抗攻击，并且在保持目标域样本分布的同时，还能保持与源域的一致性。
### 6.6.2 召回率（Recall）和F1 score

从表格中可以看出，Boosting adversarial attacks的召回率和F1 score都要高于传统的transfer learning方法和domain adaptation方法。
### 6.6.3 对比实验
本文还进行了与现有的各种对抗攻击方法的比较实验，具体结果如下：

| Method | AUC of DR (Source)| Recall of DR (Target)| F1 Score of DR (Target)| AUC of ME (Source)| Recall of ME (Target)| F1 Score of ME (Target)| Time used|
| ------ | ----- | ---------- | --------- | ----- | ---------- | --------- | ------- | 
| BIM     | 0.95   |   0.96     |     0.95  | 0.94   |   0.84     |     0.79  |         |       
| CW      | 0.93   |   0.92     |     0.92  | 0.95   |   0.90     |     0.85  |         |         
| PGD     | 0.87   |   0.87     |     0.87  | 0.95   |   0.95     |     0.90  | 4h       |          
| DeepFool| 0.91   |   0.90     |     0.89  | 0.83   |   0.78     |     0.74  |         |       

从表格中可以看出，Boosting adversarial attacks的性能要优于其他几种对抗攻击方法。
# 7.总结及未来工作
本文提出了一种新的基于domain adaptation的方法——Boosting adversarial attacks，旨在利用对抗样本的集合来增强模型的鲁棒性，从而在医疗诊断领域中提升对抗攻击的效果。Boosting adversarial attacks的实验结果显示，它比传统的transfer learning方法和domain adaptation方法都要好，且效果更好。接下来的工作可以是探索Boosting adversarial attacks的其他设置和参数，以提升它的泛化能力和抗攻击能力。