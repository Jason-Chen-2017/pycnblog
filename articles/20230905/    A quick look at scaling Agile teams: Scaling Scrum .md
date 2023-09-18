
作者：禅与计算机程序设计艺术                    

# 1.简介
  
           
Agile方法是一种快速迭代开发和交付的方式，它鼓励短小精悍的开发周期，也允许团队在每日日程安排中自主调整工作节奏。如今越来越多的公司采用Agile方法作为其开发模式。而随着业务的发展和项目的不断扩大，Agile方法在管理和协同方面也越来越重要。如何让一个大的Scrum团队能够有效地满足项目的需求是一个非常重要的问题。
近年来，Scrum方法已经被证明了它的适用性、效率和生产力。它帮助团队能够快速响应变化并适应用户的需求，同时还减少了沟通成本、避免冲突、提高工作效率。但由于其自身的局限性，一些规模较大的公司越来越依赖于Scrum团队进行项目管理。当团队成员数量增加到一定程度时，它可能面临很多性能和可靠性上的挑战。为了解决这些挑战，许多公司都开始探索更加灵活、具有弹性的Scrum方法。在这种方法中，角色、制度、流程等元素可以灵活配置，使得Scrum团队的大小、结构以及工作方式都可以根据实际需要和环境改变。在这一篇文章中，我们将讨论两种不同类型的Scrum方法——Scrum-of-scrums（SOSS）和混合型Scrum——及其实现机制。最后，我们会对两种方法给出的建议和改进方向。

# 2.相关概念           
## 2.1.Scrum
Scrum是一个敏捷开发过程，该过程基于事件驱动的迭代和增量模型。Scrum包括两个角色：产品负责人（Product Owner）和Scrum Master。产品负责人负责制定产品方向、优先级和功能需求。Scrum Master负责支持团队的实现并确保Scrum的顺利实施。Scrum通过设定目标、计划和反馈三个制度来推动工作。

Scrum通常分为四个阶段：Sprint Planning、Daily Scrum、Sprint Review、Retrospective。Sprint Planning阶段由产品负责人制定本次迭代的内容和任务。Daily Scrum阶段由每个成员在固定的时间段（一般为15分钟）更新自己当前完成的任务情况。Sprint Review阶段由团队评审上一次迭代的结果并检视后续的改进措施。Retrospective阶段则是整个Scrum过程中用来总结和改善学习到的经验教训。

Scrum的四个阶段构建了相互协作和迭代的工作氛围。在Sprint Planning阶段，产品负责人阐述产品的愿景和目标，定义迭代里要完成哪些事情。每天早上，团队开放展示自己的工作进度。他们使用名词陈述自己的工作，避免直接评价个人工作表现，避免激化矛盾、争执或效率低下的气氛。他们关注工作内容而不是个人表现，避免出现沟通上的阻碍。

在Daily Scrum阶段，Scrum团队把今天做好的工作成果报告给其他成员，然后大家一起总结一下自己的工作，以及遇到的困难和挫折。如果发现问题或者新的需求，他们会站在团队的角度出发，帮助团队解决问题。每次Daily Scrum结束的时候，产品负责人就会收集每个人的工作进度，进行综合评估，将整体进度按优先级排序。

在Sprint Review阶段，Scrum团队检查上一轮迭代的结果，对成功和失败的经验教训进行总结，找出下一轮迭代的关键问题。每轮迭代结束都会有团队成员进行总结，讨论如何改善，分享方法和工具。评审会持续到下一次迭代，再继续进行到Sprint Retrospective阶段。

最后，Retrospective阶段的目的就是总结一轮迭代中的收获和教训，提升团队的能力和改进措施。它会让团队了解自己的优点和缺点，从而形成更好的组织和团队文化。

## 2.2.Scrum of scrums (SOSS)
Scrum of scrums (SOSS)是一种由多个Scrum小组组成的管理模式。SOSS把Scrum的核心理念应用到多个团队之间。每个小组称为一个Scrum team，每个team各自承担一部分产品的开发工作。团队间的协作促进了Scrum的适用性和普遍性。SOSS也可以看作是Scrum团队内部的SOSS。

除了Scrum core concepts之外，SOSS还有以下的特征：

1. Multi-tenancy: SOSS teams can work with external clients and stakeholders.
2. Flexible structure: Different size, complexity and management styles are possible within the same group.
3. Evolutionary development: New ideas and features may arise over time as customers require new functionality or change existing requirements.
4. Cross-functional collaboration: All members have a range of skills, including designers, developers, testers, product managers and business analysts. This helps increase team cohesion and reduces communication overhead.
5. Multidisciplinary ownership: Each member is responsible for several parts of the product. This allows for cross-pollination of expertise between different areas of the organization.
6. Global view: The entire SOSS project has a single, consistent view of progress, goals, risks and dependencies across all groups.

## 2.3.Hybrid Scum
混合型Scrum是一种新型的Scrum方法。混合型Scrum融合了Scrum和SOSS的理念。它适用于跨越功能界限的复杂系统，可以满足客户的需求。混合型Scrum以Scrum为基础，但是不是所有的团队成员都参与所有的Scrum活动。不同的角色依据自己的职责来掌控Scrum。参与Scrum的团队可以把更多的时间投入到非Scrum任务上，因此可以最大限度地利用Scrum所提供的价值。

混合型Scrum的特点如下：

1. Time is more valuable than money: Hybrid Scrum prioritizes time by allowing non-Scrum tasks to take priority over Sprint planning, daily standup and sprint review. This frees up resources for important customer needs while still ensuring agility and quality.
2. Simplified roles and responsibilities: The SCRUM role definitions include flexible team structures, distributed teams and multi-disciplinary ownership models.
3. Reach into other systems: The use of integrated platforms like Jira, Confluence and Jenkins make it easy to share information across multiple sources and teams.
4. Continuous integration and delivery: By using continuous integration tools, software releases can be done frequently and regularly, reducing the risk of errors and delays in production.