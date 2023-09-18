
作者：禅与计算机程序设计艺术                    

# 1.简介
  

&nbsp;&nbsp;&nbsp;&nbsp;Clean Architecture是由Uncle Bob领衔出版的一本优秀的软件架构著作。这本书的目标就是给开发者提供一种可以指导他们创建清晰、可维护的代码结构的方法论。通过使用该方法论，开发者能够更好地管理软件开发生命周期中各个层面的复杂性。

&nbsp;&nbsp;&nbsp;&nbsp;作为一个经典的软件架构师，Bob深刻的观察到软件系统的设计方式、需求变动带来的系统架构的重构难题等。于是在他的原创著作中提出了Clean Architecture这个概念。

&nbsp;&nbsp;&nbsp;&nbsp;如果你想学习Clean Architecture相关知识并应用在你的项目中，那么你需要阅读这本经典著作，并且根据自己的实际情况进行实践。

# 2.概念和术语
## 2.1 Clean Code
&nbsp;&nbsp;&nbsp;&nbsp;Clean Code即一份干净、整洁的代码，它应该具备如下的特性：

1. 容易理解
2. 修改时不会引入新的错误
3. 适应长期维护
4. 有好的扩展性

## 2.2 Bounded Contexts
&nbsp;&nbsp;&nbsp;&nbsp;Bounded Context（BC）是Domain Driven Design（DDD）中的一个概念。它用来描述一个域的上下文。在BC中，我们可以找到其业务模型、实体、规则、约束、协议等方面所定义的内容。一个系统可能由多个BC组成，每一个BC都有自己独特的业务逻辑和范围。

&nbsp;&nbsp;&nbsp;&nbsp;BC帮助我们将复杂的系统划分为更小、更明确的子系统，让我们只关注当前子系统的业务逻辑。这样我们就可以专注于当前子系统的实现，而其他子系统则可以独立迭代、部署、交付。

## 2.3 Hexagonal Architecture
&nbsp;&nbsp;&nbsp;&nbsp;Hexagonal Architecture（下略）是一个非常流行的软件架构模式。它的主要目的是将应用程序的核心功能隔离开来，使得它不受外界环境影响，从而实现“无框架”的系统开发。

&nbsp;&nbsp;&nbsp;&nbsp;Hexagonal Architecture采用六边形架构，使得应用程序分为：应用层、领域层、基础设施层、通用工具层、框架层和展示层。其中展示层负责接收用户请求，调用业务逻辑层产生相应结果，将结果显示给用户。展示层依赖于基础设施层和领域层，但不能直接依赖于应用层或其他任何层。

&nbsp;&nbsp;&nbsp;&nbsp;使用Hexagonal Architecture可以有效减少依赖关系，避免过多的耦合，同时还能增加系统内聚力，让系统更易于维护和扩展。

## 2.4 Ports and Adapters
&nbsp;&nbsp;&nbsp;&nbsp;Ports and Adapters（下略）是一个设计模式，旨在将应用程序的业务逻辑与外部服务的通信封装起来，从而实现与各种外部服务的解耦。

&nbsp;&nbsp;&nbsp;&nbsp;Port和Adapter模式由两部分组成：端口（ports）和适配器（adapters）。当某个模块需要与外部服务通信时，可以使用端口提供接口；而将外部服务转化为适配器之后再进行交互。这种设计模式实现了“单一职责”原则，它让不同的模块都专注于自己的职责，而不是被繁复的通信细节困住。

&nbsp;&nbsp;&nbsp;&nbsp;另外，Port和Adapter模式还允许系统在不修改现有代码的情况下替换外部服务。通过引入适配器，我们可以用统一的接口来访问不同类型的外部服务，从而实现模块的可移植性。

## 2.5 Layering
&nbsp;&nbsp;&nbsp;&nbsp;Layering（下略）是一种架构风格，它将系统的职责按照功能划分为不同的层级，并且尽量降低层级之间的通信复杂度。

&nbsp;&nbsp;&nbsp;&nbsp;按功能分层，让系统变得更加模块化，每个层级都只关心自己的功能，使得代码的可读性更高、复用率更高。另外，它还可以帮助我们更好地定位问题、解决问题，并防止出现紧耦合。

## 2.6 Single Responsibility Principle
&nbsp;&nbsp;&nbsp;&nbsp;Single Responsibility Principle（SRP）是SOLID原则中的一个。它的原意是“单一职责”，表述为“A class should have only one reason to change.”。它的含义是说，一个类或者模块只能对某一件事情负责，否则就应该拆分成更细粒度的类或者模块。

&nbsp;&nbsp;&nbsp;&nbsp;单一职责原则强调模块的职责范围要尽可能小，并因此可以更轻松地测试、理解和修改。同时，也能更快地找到模块中的错误，避免因过多职责导致的设计混乱。

## 2.7 Open-Closed Principle
&nbsp;&nbsp;&nbsp;&nbsp;Open-Closed Principle（OCP）也是SOLID原则中的一个。它的原意是“开闭原则”，表述为“software entities should be open for extension but closed for modification。”。它的含义是说，软件实体（如类、模块、函数等）应该对扩展开放，对修改封闭。

&nbsp;&nbsp;&nbsp;&nbsp;OCP表明，当我们需要修改系统行为的时候，我们应该通过扩展代码的方式来实现。而不是通过修改系统源代码的方式。这样，系统的可扩展性才会更好。

## 2.8 Dependency Inversion Principle
&nbsp;&nbsp;&nbsp;&nbsp;Dependency Inversion Principle（DIP）也是SOLID原则中的一个。它的原意是“依赖倒置原则”，表述为“Depend upon abstractions not concretions.”。它的含义是说，依赖关系应该向上传递，不要向实现细节靠拢。

&nbsp;&nbsp;&nbsp;&nbsp;DIP认为，高层模块不应该依赖底层模块，二者都应该依赖抽象。换言之，依赖应该是传递的，而非继承的。这是为了让变化点在所有层级都能被捕获到，实现最大程度的模块可移植性。

## 2.9 UML图例
&nbsp;&nbsp;&nbsp;&nbsp;下面我们看一下《Clean Architecture》中的UML图例。



上图是《Clean Architecture》中的UML图例，它将系统架构分为如下的层次：

1. External Interfaces
2. Core Module: 表示系统的核心组件和业务逻辑。
3. Supporting Modules: 支持Core Module的各种外部服务的适配器和插件。
4. Frameworks and Drivers: 用于支持应用层与基础设施层之间的接口转换。
5. Presentation Layer: 负责处理来自用户的输入，并呈现给用户最终的输出。

在这个架构中，核心模块、支撑模块、外部接口和驱动程序层的耦合较低，它们彼此之间可以独立演进。