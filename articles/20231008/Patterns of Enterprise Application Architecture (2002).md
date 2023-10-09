
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Enterprise Application Architecture(EAA)是由IEEE(Institute of Electrical and Electronics Engineers)于1994年提出的，用于帮助企业解决信息系统的结构化、可维护性、复用性、适应性和可用性等问题的一种方法论。其精髓在于划分清楚应用组件之间的边界、定义组件职责、设计组件接口、使用组件间的松耦合关系。从本质上来说，EAA是一个思想体系而不是具体实践方法，而很多公司在实际应用中却滥用或无视这一方法。如今EAA已经成为主流，软件开发工程师需要具备相应的知识才能在复杂的业务环境下更好地建设软件。所以，了解EAA背后的思想和理念可以帮助开发人员理解面向对象、组件化、服务架构、SOA、事件驱动等理论及其实践方法。
# 2.核心概念与联系
2.1 EAA概述
EAA包含以下七个主要的方面:
- 组织架构模式（Organizational Patterns）:涉及到企业的组织架构及其各个部门职能划分、协调流程、沟通方式、资源共享、权利分配等方面，其重要作用是实现企业的目标、使各个业务部门之间能够互相合作、共同完成工作。
- 软件组件模式（Component Patterns）:描述了如何组织和设计软件系统中的不同元素，包括模块、类、服务、数据以及它们之间的交互。它通过封装、继承、组合和多态等特性，帮助开发者更好地管理应用中的复杂性，提高开发效率并减少错误率。
- 服务架构模式（Service Architectural Patterns）:是一种构建分布式系统的架构风格，用于通过远程过程调用（RPC）和消息传递（messaging）两种形式进行通信。该模式通过将应用程序的功能细节与网络通信分离开来，使得应用可以独立地部署、升级、扩展和监控。
- 数据访问模式（Data Access Patterns）:描述了各种数据库系统中数据查询、更新和持久化的方式。它还探讨了最佳的数据存储方案、索引的创建、优化查询速度的方法和缓冲区管理策略等。
- 消息模式（Message Patterns）:是一种用于解耦分布式系统中不同组件之间通信的有效手段。消息通常采用异步通信方式，因此系统可以根据需要灵活处理消息。消息模式一般包括发布/订阅模式、请求/响应模式、命令模式和推送模式等。
- 管理模式（Management Patterns）:描述了企业级应用运行的完整生命周期中的各种管理活动，如性能管理、安全管理、法规遵从性管理、成本管理、变更管理、配置管理、审计日志记录等。
- 事件驱动模式（Event Driven Patterns）:描述了基于事件的异步和非同步通信机制。它主要用于解耦应用组件的功能，同时提供一种事件驱动的编程模型。
2.2 模式分类
EAA的七种模式按照结构、角色、功能、时机以及对系统架构的影响程度进行分类。
- 架构模式按照结构分类，主要分为3种类型：层次型、流程型、功能型；
- 模式按照角色分类，分为创建型、理解型、执行型、协作型；
- 模式按照功能分类，分为编排型、治理型、转换型、协同型；
- 模式按照时机分类，分为静态型、动态型、响应型、演进型；
- 模式按照对系统架构的影响程度分类，分为目的导向型、实施导向型、框架导向型、验证导向型；
架构模式：层次型模式（Layered pattern）、流程型模式（Process pattern）、功能型模式（Feature pattern）。
模式角色：创建型模式（Creation patterns）、理解型模式（Understanding patterns）、执行型模式（Execution patterns）、协作型模式（Collaboration patterns）。
模式功能：编排型模式（Orchestration patterns）、治理型模式（Governance patterns）、转换型模式（Transformation patterns）、协同型模式（Coordination patterns）。
模式时机：静态型模式（Static patterns）、动态型模式（Dynamic patterns）、响应型模式（Responsive patterns）、演进型模式（Evolutionary patterns）。
模式导向：目的导向型模式（Purpose-driven patterns）、实施导向型模式（Implementation-driven patterns）、框架导向型模式（Framework-driven patterns）、验证导向型模式（Verification-driven patterns）。