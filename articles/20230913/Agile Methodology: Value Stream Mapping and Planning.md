
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Agile（自适应开发）是一种敏捷的方法论，它强调迭代、快速反馈、协作和客户参与。它把流程、工具和团队结构从日常事务中解放出来，变成了工作更有意义的过程。但它同时也引入了一些新的方法论和工具。其中Value Stream Mapping就是属于这种新方法论的一环。它的主要目的是帮助组织以有效的方式制定产品生命周期及其各个阶段所需的资源和人员分配，实现更加灵活的管理。而另一个方法论则是Planning——它描述如何将需求转化成产品功能，包括规划用户故事并将其映射到产品功能上，包括确定用户场景、用户目标、用户期望，进而形成完整的业务模型。两者结合起来可以提升效率和产品质量。本文介绍了Value Stream Mapping的相关概念和步骤。

# 2.基本概念术语说明
## 2.1 Agile Manifesto
为了保证敏捷开发的持续性，一个重要的原则就是“拥抱变化”，即“适应变化”。我们希望通过不断地改进我们的工作方式，来确保产品的竞争力能够在市场上得以显现。因此，我们相信，只有不断学习、应用新技术和掌握新知识，才能真正实现敏捷开发。

Agile Manifesto是一个宣言，它指出敏捷开发的价值观、原则和方法。它共分为12个部分，分别为：Individuals and Interactions over processes and tools，Working software over comprehensive documentation；Customer collaboration over contract negotiation；Responding to change over following a plan；Collaborative learning over isolation；Sustainable development over profitability；Continuous attention to technical excellence；Simplicity over complexity；Best practices over customizations；Face-to-face communication over electronic communication；Working together over communal ownership。敏捷宣言的这些价值观、原则和方法对敏捷开发的实践起到了很大的作用。

## 2.2 Value Stream Mapping
Value Stream Mapping（VSM）是一种方法，用来获取产品生命周期内每个阶段需要的资源和人员分配。传统的生产流程图或工序表不能直观的反映产品生命周期中的实际情况，VSM可以用可视化的方式让团队成员清楚地看到整个生产流程及其各个环节的活动内容，并了解产品每一阶段的资源、物料、人员等信息。VSM的目的有两个方面。一是让团队理解产品生命周期，便于他们合理安排任务和资源；二是能够帮助公司制定各个阶段的计划、资源的配置、人员分配等策略。

VSM的基本原理是：通过对生产流程的条理性呈现，分析各个环节之间的依赖关系，将生产任务划分成不同的阶段，然后根据每个阶段完成的产品数量和质量指标，计算每个阶段需要的人力、物料、工具等资源数量。VSM基于计算机图形系统绘制的流程图，结合实际情况，提供精细化的资源分配建议。

VSM包含以下几个步骤：
1.识别产品生命周期中的关键节点
2.确定每个节点的输入、输出、加工流程
3.给每个节点赋予估算的时间和费用
4.绘制流程图
5.评估每个阶段的实际耗时和产量
6.完善每个阶段的资源分配建议

## 2.3 角色说明
为了完成VSM的过程，团队成员需要具有如下角色：
1. 项目经理：负责收集需求、制定产品路线图、调整计划。
2. 测试人员：完成不同测试场景的功能测试，并根据产品生命周期评估各个节点的性能和效果。
3. 技术研发人员：参与技术方案设计、编码实现和交付。
4. 普通职员：执行需求分析、编写文档、管理团队，完成VSM相关文档的编写。