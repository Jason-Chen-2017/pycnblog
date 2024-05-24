
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是架构？
“架构”这个词汇源自于古罗马哲学家亚里士多德的名言“ἄρχεσθαι εὐάρεστος”，意思是“准备着进入；面临将来的环境”。在计算机世界中，架构指的是软件系统结构设计的总体策略。架构可以由不同的技术、模式或方法组成，它们共同组成了一种架构风格。

## 为什么要用架构？
为了更好的软件开发过程管理和实施，通过对软件系统进行合理的分层、模块化、组件化和依赖关系管理等方式，能够提高代码的可维护性、复用性和扩展性，并降低软件开发过程中的各种风险。因此，架构风格至关重要，它决定着软件项目的整体质量、效率和生命周期。

架构风格主要包括四种：
- 分层架构（Layered Architecture）
- 框架式架构（Framework-Based Architecture）
- 模块化架构（Modular Architecture）
- 服务架构（Service-Oriented Architecture，SOA）

## 架构风格分类
根据架构风格的不同特点，又分为五类：
- 反映架构模式的类型：
    - Client-Server Architecture
    - Peer-to-Peer Architecture
    - Shared Data Architecture
    - Domain-driven Design（DDD）
    - Event-driven Architecture
    - Microservices Architecture
    - Service-Oriented Architecture（SOA）
- 是否支持微服务架构：
    - Onion Architecture
    - Clean Architecture
    - Hexagonal Architecture
- 功能划分方式：
    - Distributed Systems Architecture
    - API-based Architecture
    - Serverless Architecture
    - Smart Cities Architecture
    - Internet of Things (IoT) Architecture
- 实现方式：
    - Framework-based Architecture
    - Code-as-Configuration Architecture
    - Hybrid Cloud Architecture
    - Multilayered Architecture


本文主要讨论第六个风格——“Hexagonal Architecture”，它最初于2005年由<NAME>引入，其核心理念是：“domain services(领域服务)”应该在应用之外实现，不直接受到应用内部的直接调用。

# 2.基本概念和术语
## 什么是领域服务？
领域服务(Domain Services)，也称为业务逻辑层、应用服务层或者业务层，是应用程序的核心功能。它负责处理应用程序的核心业务需求，包含多个具有相同目的的对象或函数。这些对象或函数要么通过独立接口提供服务，要么通过定义良好的消息协议进行通信。领域服务通常由单独的开发团队进行维护，并在应用程序外部实现，如Java中的DAO或Spring的Service。

## 为什么要使用领域服务？
领域服务的优点很多，例如：
- 提高代码的可维护性和复用性，因为它们被隔离开来，使得不同的应用程序可以共享这些服务。
- 允许开发人员通过精心设计的API来进行分布式系统的集成。
- 降低应用程序之间的耦合度，使得各层之间没有强制性的依赖关系。
- 使软件更容易测试，因为它为每一个业务规则都提供了独立的测试用例。
- 可以为复杂的系统添加新功能，而无需修改现有的代码。

## 为什么要使用Hexagonal Architecture?
Hexagonal Architecture 是一个基于微服务架构的设计模式，它把应用的业务逻辑与它的表现层和持久化层完全解耦。在该架构下，应用的业务逻辑被抽象出来，形成领域服务，领域服务以端口的形式暴露给其他组件。这种设计模式将应用的业务逻辑与数据库和Web框架完全解耦，使得应用可以灵活地选择适合自己的数据存储方式以及Web框架。

Hexagonal Architecture 的特点是：
- 强调业务逻辑和数据访问的隔离。
- 通过独立的领域服务来完成任务，而不是把它们融入应用的业务逻辑中。
- 使用独立的端口驱动业务逻辑。
- 支持不同的部署场景：物理机、虚拟机、云计算平台、本地网络等等。
- 可替换性：可选的组件，比如数据库驱动，可以轻易的切换到另一种实现。
- 可测试性：领域服务层可以单独测试，而不需要应用层的代码。

# 3.Hexagonal Architecture的原则
## SOLID原则的倡导者——Clean Architecture
Hexagonal Architecture 与 Clean Architecture 一样，都是遵循SOLID原则的著名架构设计理念。Clean Architecture 由 Eric Evans 在他的书《Clean Architecture: A Craftsman's Guide to Software Structure and Design》中首次提出，并广泛被使用。

Clean Architecture 除了遵循 SOLID原则外，还有以下原则：
- 单一职责原则（Single Responsibility Principle，SRP）
- 依赖倒置原则（Dependency Inversion Principle，DIP）
- Interface Segregation Principle（ISP）
- Open/Closed Principle（OCP）
- Liskov Substitution Principle（LSP）
- Hollywood Principle（HP）

Hexagonal Architecture 和 Clean Architecture 有一些共同点，如下：
- 均采用依赖倒置原则（DIP），这意味着应用的业务逻辑和支撑其运行的支撑层组件不应该直接依赖于应用程序的核心层组件，而应该依赖于抽象的接口。
- 均通过独立的领域服务来完成任务。
- Hexagonal Architecture 不仅使用接口隔离技术，还使用端口和适配器模式对其进行实现。
- 对数据库和Web框架的选择也有很大的影响。

但 Hexagonal Architecture 与 Clean Architecture 有以下不同点：
- Hexagonal Architecture 更关注于业务逻辑的隔离，而且应用的组件也不是一成不变的，会经历一系列变化，所以它更加灵活。
- 由于业务逻辑和支撑其运行的支撑层组件不再直接依赖于应用程序的核心层组件，所以它更容易对这些组件进行单元测试。
- 如果 Hexagonal Architecture 中出现性能问题，Clean Architecture 会更容易发现。
- Clean Architecture 把持一种编程风格，而 Hexagonal Architecture 是一种架构风格。

# 4.Hexagonal Architecture的实现
## Core层：应用的核心层
Core层一般包括应用的核心业务逻辑。它由应用的关键业务实体以及支撑其业务运行的支撑层组件组成。Core层的作用是实现应用的主要功能，包括用户注册、登录、权限控制、搜索等功能。

## Support层：应用的支撑层
Support层一般包括应用的非核心业务逻辑和支撑层组件。它包含Core层所需的其他组件，包括数据访问层、外部API接口、Web框架等。Support层的作用是辅助Core层完成任务，实现其余功能。

## Ports & Adapters层：Hexagonal Architecture的主要组件
Ports & Adapters层是Hexagonal Architecture的主要组件。它由Port和Adapter构成，其中，每个Port代表了一个应用的支撑层组件，而每个Adapter则用于连接不同的实现。Hexagonal Architecture 中的每个元素都封装进去，因此，一个组件可以被其他组件所共享，只需要知道对应的接口即可。

### Port：端口
端口（Port）用来连接外部的组件和Core层。它从某些角度来说，也可以理解为一种规范或协议，它规定了Core层与外部的各个组件之间应该如何交互。Port一般以接口的形式存在，定义了一组访问某个特定功能的方法。

### Adapter：适配器
适配器（Adapter）用于实现两个组件间的通信。它接收来自Core层的请求，然后转换成对外部组件的实际请求，向外提供服务。比如，对于Web应用来说，可以通过HTTP协议来与外部的API进行通信，这样就实现了应用之间的通信。

Hexagonal Architecture 使用到的主要组件就是Port和Adapter。下面举一个简单的例子来说明这些组件是如何工作的：

假设有一个应用需要获取用户信息，其中有一个模块需要连接到外部的用户信息服务。如果直接在Core层直接实现这一功能，那么Core层就需要直接依赖于外部的用户信息服务，这违背了Hexagonal Architecture 的设计原则，所以我们可以在Support层创建一个新的模块，叫做UserInformationPort，并为它创建相应的接口。Core层可以使用UserInformationPort来获取用户信息。同时，创建一个适配器，该适配器可以接收来自Core层的请求，然后将请求转换成对外部用户信息服务的实际请求。具体实现方式就是从外部服务处获取数据，然后再转换成Core层需要的格式，返回给Core层。

## 流程图
Hexagonal Architecture 的流程图如下：



# 5.Hexagonal Architecture 的优缺点
## 优点
Hexagonal Architecture 的优点很多，下面列举几个比较突出的优点：
- 应用的业务逻辑和其他支撑层组件的分离，可以有效减少它们之间的耦合度。
- 提高应用的可维护性，可以更容易的对应用进行改动，不会影响Core层。
- 为应用的不同功能层提供不同的适配器，可以实现组件的可替换性，达到高度的灵活性。
- 每个组件都可以单独测试，且不需要整体的测试环境。
- 可以轻松地对应用进行部署，只需要部署Core层的组件就可以了，而不需要部署整个应用。

## 缺点
Hexagonal Architecture 也有缺点，但是比起普通的架构风格，缺点相对较小。这里列举几个比较突出的缺点：
- 需要了解Hexagonal Architecture 的实现细节才能正确使用它。
- 实现起来比较复杂，需要花费时间来学习相关知识。
- 不能保证应用的完整性，因为它要求按照设计的接口约束来使用其中的组件。
- 在组织架构上可能难以实施，因为它对技术栈、工具链和工程实践有一定要求。