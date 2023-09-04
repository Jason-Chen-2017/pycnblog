
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
随着人们生活节奏的加快、经济发展水平的提升、信息化程度的不断提高，基于图像数据的行人识别技术已经成为当今人们生活的一项重要需求。在这个过程中，目标检测模型准确预测出行人的真实身份，对行人进行识别、跟踪等应用场景都有很大的帮助。然而，现有的行人识别数据集往往面临以下问题：
* 数据量不足：目前主要的数据集仅包含具有相同属性的行人，缺乏异构的跨时代、跨场景、跨地域的人脸数据。
* 不适合于动态环境：目前数据集中的图像和视频仅提供了静态环境下的行人出现的片段，对于行人动态表演、穿戴装备等情况的识别效果较差。
* 多样性较低：不同年龄、姿态、外观、表情等方面的行人分布存在较大差距，难以训练好的模型能够精准识别每一个类别。
为了解决以上问题，作者团队从自身及其他知名行人识别论文中总结提炼了以下观点，设计并构建了一套增量学习的方法，将多个数据集按时间顺序依次作为模型的先验知识源，共同完成整个行人识别任务的学习过程。实验证明，增量学习方法能够有效地提升行人识别的性能。本文旨在系统阐述这一方法的基本思想、关键实现细节、以及通过新构建的数据集提供的新鲜视角。希望通过这个文章，能够帮助读者了解增量学习方法、理解如何应用到行人识别领域，以及探索未来更进一步的研究方向。
## 作者简介
* 张浩，清华大学计算机系博士生
* 曾获中国科学院计算机所长江学者奖、中国计算机学会优秀教师奖、清华大学计算机系优秀学生奖
* 主攻图像处理、人工智能、机器学习领域
## 参考文献
[1] <NAME>, <NAME> and <NAME>. Deeply Supervised Convolutional Networks for Real-time Person Re-identification[C]. IEEE International Conference on Computer Vision Workshop, Salt Lake City, Utah, USA, October 2016: IEEE Xplore.

[2] <NAME>, <NAME>, <NAME>, et al. Person transfer re-id with temporal consistency optimization[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 97-113.

[3] <NAME>, <NAME>, <NAME>, et al. Boosting person re-identification by exploiting heterogeneous information[J]. IEEE Transactions on Multimedia, 2019, PP(99): 1-1.

[4] <NAME>, <NAME>, <NAME>, et al. Person search in surveillance videos via incremental learning[J]. IEEE transactions on pattern analysis and machine intelligence, 2020, PP(99): 1-1.

[5] Zhang, Baoqian, <NAME>, and <NAME>. "Pedestrian detection and re-identification using an incremental deep model." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 01. 2020. 

[6] Zhou, Jiajie, et al. "Incremental convolutional neural network for large-scale pedestrian re-identification." In CVPR, pp. 9798-9807. IEEE, 2019.