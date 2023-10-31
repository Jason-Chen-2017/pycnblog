
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 可维护性(Maintainability)
可维护性是指一个软件或系统从设计、开发到运营维护过程中能够持续提供有效的支持而不被损坏、破坏或降低性能。提升软件的可维护性可以有效地防止出现各种问题，包括后期维护困难、升级困难、扩展困难等等。

提升软件可维护性具有重要意义，因为它直接影响了软件的生命周期，并对企业的未来发展、公司盈利能力、市场占有率都产生重大影响。因此，在制定、实施、完善软件开发过程时，不可忽视提高软件的可维护性的重要性。

为了提高软件的可维护性，需要考虑以下几个方面：

1. 健壮性(Robustness):软件应当具有良好的健壮性，保证其稳定运行，即使遇到外部变化也不会发生崩溃或漏洞，同时对环境和外部输入有充分准备和过滤措施；
2. 可用性(Availability):软件应当具备良好的可用性，保证其能够及时响应用户的请求、承受任意大小的流量负载；
3. 可移植性(Portability):软件应当具有良好的可移植性，便于部署、迁移到新的环境中，使得软件可以在多种软硬件平台上运行，同时避免对硬件依赖造成的限制；
4. 性能(Performance):软件应当具有良好的性能，能够满足日益增长的用户需求和数据处理能力要求；
5. 兼容性(Compatibility):软件应当兼容当前主流的操作系统、应用程序、工具、第三方库等；
6. 测试性(Testability):软件应当具有良好的测试性，包括单元测试、集成测试、端到端测试、性能测试等；
7. 文档化(Documented):软件应当有详尽的文档，使得新人快速了解产品功能和使用方法，对故障排查和版本更新提供了参考；
8. 可理解性(Understandability):软件应当易于理解，包括代码结构清晰、命名符合直觉、模块之间关系合理等。

本文将主要探讨如何在实际开发过程中处理提示词中提到的可维护性相关的问题。
# 2.核心概念与联系
## 可维护性
可维护性（Maintainability）是指一个软件或系统从设计、开发到运营维护过程中能够持续提供有效的支持而不被损坏、破坏或降低性能。提升软件的可维护性可以有效地防止出现各种问题，包括后期维护困难、升级困难、扩展困难等等。

## 模块化（Modularity）
模块化（Modularity）是计算机编程中很重要的一个概念。它允许一个复杂的程序由不同、相互独立的子程序组成，每一个子程序只关注自己的功能，并通过接口来与其他子程序通信。模块化可以提高软件的可读性、易维护性和复用性。

模块化可以分为两大类：物理模块化和逻辑模块化。

物理模块化是指把程序按照模块划分到不同的文件中，比如各个函数放在一个文件里，数据结构放在另一个文件里，这样做可以保持每个文件的逻辑独立，便于维护和更新。

逻辑模块化是指把程序按照功能划分成多个模块，每个模块只做自己应该做的事情，各个模块之间通过接口通信。这种模块化方式能够提高程序的可读性、易理解性、降低耦合性，并且易于实现模块的替换、增减、组合。

## 提示词
提示词是用来描述一段文本或代码功能的术语。一般来说，提示词可以分为促进（Incentivize）、激励（Motivate）、支持（Support）、提示（Inform），或者表达疑问（Ask）。一些常见的提示词如下：

1. 好：修复错误、改进特性、优化性能、减少资源消耗；
2. 不好：难以调试、增加复杂度、引入隐私泄露、引入安全风险；
3. 需要考虑：兼容性、可靠性、数据完整性、代码效率、可扩展性、鲁棒性、可维护性；
4. 不要：风险大、功能缺失、性能下降、安全问题、资源浪费；
5. 有必要：适配变化、新增功能、提高可靠性、提高代码质量、提高代码效率；
6. 是否会：出现问题？是否存在风险？是否会影响性能？是否需要修改？是否需要测试？是否需要文档？

提示词反映出开发者对于特定功能的看法或希望。比如，代码注释中经常会出现“待修改”、“待解决”等类似的提示词。这些提示词主要用于引导开发者注意代码中可能存在的不足之处，并让他们参与到代码的维护工作中。

## 技术债务
技术债务（Technical Debt）是指由于开发人员对某些技术或工程概念理解不够深入、对项目的进展方向没有正确估计、对关键技术决策缺乏经验而导致的延误，往往会导致项目甚至整个开发流程变慢、花费更多时间，甚至导致项目失败。技术债务通常属于商业债务，随着时间的推移，它会逐渐积累起来，最终形成巨大的经济损失。

## 演示文稿模板
演示文稿模板示例如下：

# Propose Engineering Best Practice: How to Deal with Maintainability Problems in Prompt Words?
## Background Introduction
Maintainability is the capability of a software or system to continue functioning correctly and provide support without disruptive events such as crashes, damage or degrading performance over its lifetime. By improving the maintainability of software, we can prevent various problems, including difficulties post-maintenance, upgrade issues, and extension limitations.

Software maintainability has significant importance because it directly affects the life cycle of software, which also have a big impact on company future development, profitability, and market share rate. Therefore, it is essential for developing software processes to consider high-quality maintenance of software from the beginning, during implementation and improvement phases.

In order to improve the maintainability of software, several aspects need to be considered:

1. Robustness: Software should possess good robustness that ensures stability running, even under external changes;
2. Availability: Software must have good availability that responds quickly to user requests and handles arbitrary load sizes;
3. Portability: Software should be highly portable so that it can easily be deployed into new environments, while avoiding hardware dependencies;
4. Performance: Software must have good performance to meet increasing demands for users and data processing capabilities;
5. Compatibility: Software should be compatible with current mainstream operating systems, applications, tools, third-party libraries, etc.;
6. Testability: Software must be well-tested, including unit testing, integration testing, end-to-end testing, performance testing, etc.;
7. Documented: Software should have comprehensive documentation that makes novice team members able to understand product features and usage methods, facilitates troubleshooting and version updates;
8. Understandability: Software should be easy to understand, including clear code structure, intuitive naming conventions, logical relationships among modules.

This article will focus on how prompt words related to maintaining quality can be handled when working on actual projects. 

## Core Concept & Connection
### Modularity
Modularity is an important concept in computer programming. It allows complex programs to be broken down into different subprograms, each of which focuses on its own functionality and communicates through interfaces with other subprograms. Modularity improves readability, maintainability, and reusability of software.

There are two types of modularity: physical and logical modularization. Physical modularity involves dividing a program into different files according to modules, where functions go in one file and data structures go in another file. This approach maintains logical separation between files and allows for easier maintenance and updating.

Logical modularization involves breaking a program down into multiple modules, each doing what it needs to do and communicating via interfaces with others. This type of module architecture provides clarity by allowing readers to follow the logic of individual modules rather than scrolling through the entire codebase. It further enables replacing, adding, and removing modules at will, making it easier to develop scalable and flexible software architectures.

### Maintainability
Maintainability refers to the ability of a software or system to persist through time without interruption due to errors, failures, or slowdowns. Increasing software's maintainability can help prevent many problems ranging from increased complexity, technical debt accumulation, security vulnerabilities, to customer frustration and loss of trust.

To increase software's maintainability, some key areas include:

1. Robustness: The software should have good robustness, meaning that it should remain stable even under changing conditions;
2. Availability: The software should have high availability, providing quick responses to user requests and handling arbitrarily large loads;
3. Portability: The software should be highly portable, enabling it to run across numerous platforms without dependency on specific hardware;
4. Performance: The software should achieve good performance levels, meeting constantly growing needs for users' data processing abilities;
5. Compatibility: The software should work seamlessly with all current mainstream computing technologies, ensuring compatibility with popular operating systems, applications, tools, and libraries;
6. Testability: The software should be tested thoroughly, including unit tests, integration tests, end-to-end tests, and performance tests;
7. Documentation: The software should be documented comprehensively, providing explanations and instructions for using it effectively, easing debugging process and keeping track of updates;
8. Readability: The software should be designed with ease of understanding in mind, promoting logical organization and hierarchy within the codebase.

When working on real-world projects, developers often employ language constructs and best practices to convey their ideas and opinions about certain topics. These phrases typically begin with negative keywords like “not”, “don’t,” “avoid,” or “don’t do.” Often these suggestions reflect personal experience and nuances about the project, but they might also be unfounded opinions stemming from misunderstandings or biases caused by introducing incorrect assumptions early on. As engineers, we may need to carefully evaluate and reevaluate our statements and approaches whenever we encounter them again, thinking critically and questioning whether we agree with the suggestion or not.