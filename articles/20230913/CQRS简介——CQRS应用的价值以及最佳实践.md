
作者：禅与计算机程序设计艺术                    

# 1.简介
  

CQRS（Command Query Responsibility Segregation）即命令查询职责分离，是一个应用程序设计模式，它把一个系统分成两部分：命令处理器和查询处理器。分别负责执行命令更新数据、读取数据的动作。这使得系统更容易扩展和维护。 

传统的面向对象编程技术中，应用通常都是由命令处理器负责处理用户输入的事务性操作，比如保存订单信息、创建账户等；而查询处理器则用来处理只读的数据请求，比如读取账户余额、搜索订单列表等。但是在CQRS架构中，两个处理器角色之间存在明显的区别，他们各自完成不同的功能。

命令处理器专注于执行用户的指令，并且只能做到对当前状态的修改。它们可以实现的操作主要是事务性操作，如增删改操作，但不能处理复杂的业务逻辑或者需要依赖其他服务的数据。此外，命令处理器只能基于整个模型的一套API，无法灵活地适应不同客户群体的需求。

相反，查询处理器在任何时候都可以响应查询请求。它的输入参数可以非常自由，因为它不受限于模型层面的约束，因此可以满足各种类型的查询需求。但是它只能从已知的模型中获取结果，并且只能通过对整个模型的一个快照进行响应。因此，查询处理器很难处理复杂的业务逻辑和需要依赖其他服务的数据。

CQRS架构的意义在于：

 - 提供一种更好的隔离、解耦及扩展能力，因为它将命令处理器和查询处理器分别解耦了。
 - 更好地服务于高并发、海量数据等场景，因为可以通过将数据读取操作和写入操作分开来提升性能。
 - 为单个模型增加更多的灵活性，因为允许每个团队/组织根据自身业务需要选择不同的查询处理器。
 - 提升可测试性，因为命令处理器和查询处理器的测试可以被彻底地分离，并独立于模型的其他部分。

在本文中，我将会分享关于CQRS的一些理论基础知识、技术实现方法、应用案例以及最佳实践建议。希望能够帮助读者理解这个领域的理论和实践方面，对自己后续的工作有所裨益。
# 2.基本概念术语说明
## 2.1.命令查询职责分离(CQRS)
CQRS即命令查询职责分离，是一种应用程序设计模式，它把一个系统分成两部分：命令处理器和查询处理器。命令处理器处理用户的指令，并且只能对当前状态的修改；而查询处理器则处理只读的数据请求，而且可以独立于命令处理器运行。两个处理器角色之间存在清晰的区别。

传统上，应用中只有一条信息流通的路径，即从客户端到服务器端。这条路径上的消息就是命令。服务端接受到命令之后，就会根据命令的类型来执行相应的操作。这些操作可能会导致状态的改变，比如更新数据库中的某个字段的值。查询操作不会修改状态，只是返回某个特定的数据。例如，当客户端需要查看账户余额时，它就发送一个只读的查询请求给服务器，服务器将查询数据库中的某些记录并返回给客户端。查询请求一般不需要等待数据库的回应，因此响应速度很快。

命令查询职责分离是一种架构模式，旨在将命令和查询处理器解耦，提升可扩展性。它把应用分成了两部分，分别作为命令处理器和查询处理器存在。命令处理器接收来自客户端的命令，并依据命令的内容更改模型的状态。但是它只能对当前状态进行修改，不能影响其他模型。查询处理器接收来自客户端的查询请求，并提供模型的最新或查询所需的数据。查询处理器可以独立于命令处理器运行，也不会直接修改模型。

## 2.2.架构模式
CQRS架构模式的设计原则是关注点分离。按照这种设计原则，系统中的不同功能模块被分成不同的上下文（context），每个上下文中的实体（entity）仅负责自己的工作。

系统中的实体按功能分成以下几类：

 - 命令实体（command entity）：用于存储所有发出的命令。
 - 查询实体（query entity）：用于存储所有收到的查询请求。
 - 命令处理器（command processor）：处理来自命令实体的命令，并产生事件。
 - 查询处理器（query processor）：处理来自查询实体的查询请求，并产生响应。
 - 事件源（event source）：产生事件的实体，包括命令实体和查询实体。
 - 消息队列（message queue）：用于事件通知和异步处理。

其中，命令实体和查询实体均为无状态，它们仅提供CRUD操作接口，但不存储数据。消息队列通常为分布式系统中的一种组件。

图7-1展示了一个典型的CQRS架构模型。


图7-1 CQRS架构模型示意图

在该架构中，命令实体处理来自客户端的命令，并向事件源发布“命令已受理”事件。事件源负责将事件通知给其他实体，例如命令处理器。命令处理器获取来自消息队列的命令，并执行相应的操作。当操作执行成功时，命令处理器产生一个“命令已完成”事件，并向消息队列发布该事件。同时，它还可以产生其他类型的事件，如“命令执行失败”或“命令超时”。当事件源收到“命令已完成”事件时，它将其通知给查询处理器。查询处理器可以订阅事件源的事件，并从相关的实体中获取数据。

查询实体用于存储来自客户端的查询请求。当查询处理器接收到查询请求时，它可以检查查询是否合法。如果合法，它将根据查询条件从相关的实体中获取数据，然后返回给客户端。

## 2.3.事件驱动架构(EDA)
事件驱动架构（Event-driven Architecture, EDA）是一个分布式应用架构风格，旨在通过事件传递的方式解决系统间通信的问题。它的核心思想是用轻量级的事件代替传统的请求-响应模式。

EDA主要有两种形式：联合发布-订阅（Fedora）和命令查询职责分离（CQRS）。联合发布-订阅架构采用集中式的事件总线，所有的消息都向总线中广播；CQRS架构则是通过命令查询职责分离将读取操作与写入操作分割开来。

EDA架构的特点包括：

 - 异步通信：EDA采用消息队列来异步处理消息，避免了同步调用造成的延迟，提高了系统的并发性。
 - 降低耦合性：EDA架构中，消息的生产者和消费者之间没有强绑定关系，可以独立扩展。
 - 可恢复性：EDA架构支持消息的持久化，可以方便地重新启动、恢复。
 - 可伸缩性：EDA架构天然具有横向扩展能力，可以在集群中动态添加节点。
 - 最终一致性：EDA架构支持最终一致性，不需要强制指定的时间间隔，保证消息的最终一致性。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 3.1.概念
### 3.1.1.命令查询职责分离（CQRS）
CQRS即命令查询职责分离，是一种应用程序设计模式，它把一个系统分成两部分：命令处理器和查询处理器。命令处理器处理用户的指令，并且只能对当前状态的修改；而查询处理器则处理只读的数据请求，而且可以独立于命令处理器运行。两个处理器角色之间存在清晰的区别。

传统上，应用中只有一条信息流通的路径，即从客户端到服务器端。这条路径上的消息就是命令。服务端接受到命令之后，就会根据命令的类型来执行相应的操作。这些操作可能会导致状态的改变，比如更新数据库中的某个字段的值。查询操作不会修改状态，只是返回某个特定的数据。例如，当客户端需要查看账户余额时，它就发送一个只读的查询请求给服务器，服务器将查询数据库中的某些记录并返回给客户端。查询请求一般不需要等待数据库的回应，因此响应速度很快。

命令查询职责分离是一种架构模式，旨在将命令和查询处理器解耦，提升可扩展性。它把应用分成了两部分，分别作为命令处理器和查询处理器存在。命令处理器接收来自客户端的命令，并依据命令的内容更改模型的状态。但是它只能对当前状态进行修改，不能影响其他模型。查询处理器接收来自客户端的查询请求，并提供模型的最新或查询所需的数据。查询处理器可以独立于命令处理器运行，也不会直接修改模型。

### 3.1.2.事件溯源
事件溯源，也称作事件源头追溯，它通过记录事件序列，揭示出一个数据对象的生命周期。事件源一般分为三类：

 - 指令源（command source）：记录所有指令信息，包括命令数据，时间戳，身份标识符等。
 - 元数据源（metadata source）：记录所有元数据信息，包括领域模型中的实体，属性，关联关系等。
 - 数据源（data source）：记录所有数据的变更信息，包括新增、删除、更新数据项。

事件溯源中的事件可以分为两种类型：

 - 指令事件（command event）：记录指令的元数据，包括发起人的ID，指令类型，目标系统名称，目标资源的URI等。
 - 数据事件（data event）：记录数据变更的信息，包括发生时间，谁更新了数据，更新后的数据。

### 3.1.3.消息队列
消息队列，是一个用于传递和接收消息的异步传输结构。消息队列提供了异步通信机制，允许应用组件之间松耦合，并允许通过重复消费保证消息的完整性。消息队列支持两种主要操作：

 - 发布-订阅（publish-subscribe）：消息发布者向指定的主题发布消息，消息订阅者随时可以订阅该主题。
 - 请求-响应（request-response）：请求者发送请求，消息队列将相应的回复发送给请求者。

消息队列可以保证应用的可靠性和可伸缩性。消息队列可以将事件写入磁盘或内存中，等待消费者的拉取。消费者可能由于各种原因暂停或失败，消息队列将重试多次，直至成功。当消息持久化后，消息队列允许检索历史消息，用于审计和分析。

### 3.1.4.事件总线
事件总线，又称为联合发布-订阅（Fedora）架构。它使用集中式的事件总线，所有的消息都向总线中广播。事件总线可以容纳众多的订阅者，每当发生事件时，它将消息推送到每个订阅者处。订阅者可以选择是否接收消息。

## 3.2.CQRS基本原理
### 3.2.1.命令查询职责分离
CQRS即命令查询职责分离，是一种应用程序设计模式，它把一个系统分成两部分：命令处理器和查询处理器。命令处理器处理用户的指令，并且只能对当前状态的修改；而查询处理器则处理只读的数据请求，而且可以独立于命令处理器运行。两个处理器角色之间存在明显的区别。

传统上，应用中只有一条信息流通的路径，即从客户端到服务器端。这条路径上的消息就是命令。服务端接受到命令之后，就会根据命令的类型来执行相应的操作。这些操作可能会导致状态的改变，比如更新数据库中的某个字段的值。查询操作不会修改状态，只是返回某个特定的数据。例如，当客户端需要查看账户余额时，它就发送一个只读的查询请求给服务器，服务器将查询数据库中的某些记录并返回给客户端。查询请求一般不需要等待数据库的回应，因此响应速度很快。

命令查询职责分离是一种架构模式，旨在将命令和查询处理器解耦，提升可扩展性。它把应用分成了两部分，分别作为命令处理器和查询处理器存在。命令处理器接收来自客户端的命令，并依据命令的内容更改模型的状态。但是它只能对当前状态进行修改，不能影响其他模型。查询处理器接收来自客户端的查询请求，并提供模型的最新或查询所需的数据。查询处理器可以独立于命令处理器运行，也不会直接修改模型。

### 3.2.2.命令查询模型
CQRS架构中，模型分为命令模型和查询模型。命令模型用于处理用户的指令，只能对当前状态进行修改；查询模型用于处理只读的查询请求，可以独立于命令模型运行。它们共同构成了CQRS架构。

命令模型一般包括以下元素：

 - 命令实体：用于存储所有发出的命令。
 - 命令处理器：处理来自命令实体的命令，并产生事件。
 - 事件源：产生事件的实体，包括命令实体。
 - 消息队列：用于事件通知和异步处理。

查询模型一般包括以下元素：

 - 查询实体：用于存储所有收到的查询请求。
 - 查询处理器：处理来自查询实体的查询请求，并产生响应。
 - 事件源：产生响应的实体，包括查询实体。
 - 消息队列：用于事件通知和异步处理。

### 3.2.3.单一数据源
CQRS架构中，往往存在多个数据源，每个数据源对应一个命令模型。当命令模型执行完操作后，产生一个对应的事件，同时触发事件的发布，其他的模型（命令模型除外）可以订阅该事件，并从对应的命令模型中获取更新后的状态。这样的设计既可以减少耦合性，也可以实现分离的查询模型和命令模型，最大程度地提升系统的可维护性。

### 3.2.4.事件驱动模型
CQRS架构中，命令模型和查询模型之间的通讯是通过事件驱动模型实现的。命令模型生产事件并触发事件发布，同时订阅该事件。这样可以确保命令模型的状态一定是最新的。同时，命令模型可以产生许多种类的事件，为了让查询模型能够实时感知到状态的变化，查询模型可以订阅来自多个命令模型的事件。这样，当有状态的变化发生时，查询模型就可以获取最新的信息。

### 3.2.5.幂等性
CQRS架构中的命令模型应当具备幂等性，即相同的命令不应该得到不同的结果。命令模型在处理一个命令时，必须保证产生的事件是相同的，不管它是否已经执行过，这可以保证事件的唯一性。

### 3.2.6.最终一致性
CQRS架构中，命令模型和查询模型都是最终一致性的。也就是说，查询模型获取到的信息，会随着系统的变化而逐渐更新。但最终一致性也带来了一定的复杂性。首先，查询模型需要等待足够长的时间才能获得最新的数据，这取决于读取操作时的网络延迟以及其他因素。其次，由于副作用（side effect）的问题，不同命令之间的依赖关系可能导致数据不一致。最后，最终一致性可能导致不可预测性和性能问题。

## 3.3.CQRS的优势
### 3.3.1.分离关注点
CQRS架构把应用中的命令和查询处理器解耦，提升了应用的可维护性。它提高了应用的健壮性和可扩展性，可以在需要的时候扩展应用的处理能力。

### 3.3.2.可维护性
CQRS架构的命令模型和查询模型可以被不同的开发人员、团队和系统进行维护。这可以提升应用的可用性和可维护性。

### 3.3.3.可复用性
CQRS架构可以被应用到多种不同的场景中。例如，同一个系统可以作为命令模型，另一个系统作为查询模型。另外，查询模型可以重用命令模型生成的事件。这可以极大地提升开发效率和复用性。

### 3.3.4.快速响应
CQRS架构的查询模型可以提供快速响应，而且对于后台任务、报表等长期运行的操作来说，这种响应是必须的。

### 3.3.5.降低成本
CQRS架构有助于降低应用的复杂性。它将系统拆分成命令模型和查询模型，可以将系统中的操作分开，实现降低成本。

### 3.3.6.一致性
CQRS架构保证了应用的一致性，这有利于维护数据正确性和完整性。

## 3.4.CQRS的缺陷
### 3.4.1.性能问题
CQRS架构的性能问题主要是由于两个模型（命令模型和查询模型）之间存在网络延迟的问题。命令模型向事件源发布事件后，可能需要花费较长的时间才会被其他模型获取到。此外，CQRS架构要求频繁的事件交互，可能会引起网络拥塞。

### 3.4.2.集中式架构
CQRS架构采用集中式的事件总线架构。虽然它可以降低网络延迟，但它也是一种集中化的架构。这对某些应用来说并不是最优解。

### 3.4.3.高成本
CQRS架构的实现难度比较高，耗费的人力和财力都比较大。不过，针对企业级应用来说，这是一个值得考虑的事情。

# 4.如何实施CQRS
## 4.1.准备工作
### 4.1.1.业务领域建模
首先，需要建立命令和查询模型对应的领域模型。领域模型描述的是业务领域中的实体、聚合根、值对象以及其他概念。

### 4.1.2.建立数据库表结构
接下来，需要建立数据库表结构。数据库表结构一般如下：

 - Command Table: 用于存放待处理的指令信息。
 - Event Table: 用于存放产生的事件信息。
 - ReadModelTable1: 用于存放查询模型1所需的数据。
 - ReadModelTable2: 用于存放查询模型2所需的数据。

### 4.1.3.确定事件发布策略
决定事件发布策略，主要有两种方式：

 - 根据业务事件，直接发布。
 - 根据聚合根，发布聚合根对应的所有事件。

## 4.2.项目实施过程
### 4.2.1.引入CQRS NuGet包
在现有的项目中引入NuGet包：
```csharp
PM> Install-Package MediatR.Extensions.Microsoft.DependencyInjection
PM> Install-Package FluentValidation
PM> Install-Package Autofac
PM> Install-Package Microsoft.EntityFrameworkCore.Tools
```
FluentValidation 用于验证Command模型，Autofac用于IoC容器的注入，MediatR.Extensions.Microsoft.DependencyInjection用于Command的命令处理器。

### 4.2.2.定义Command模型
定义Command模型，需要继承ICommand接口，ICommand接口需要包含两个属性：AggregateId 和 CommandType。AggregateId 表示Command所在的聚合根Id，CommandType表示Command的类型。
```csharp
public class CreateOrderCommand : ICommand
{
    public Guid AggregateId { get; set; }

    public string OrderNumber { get; set; }
    
    // Other properties...
}
```

### 4.2.3.定义CommandHandler基类
CommandHandler基类用于处理命令，需要继承IRequestHandler<TRequest, TResponse>泛型接口。TRequest表示Command模型，TResponse表示Command执行的结果。
```csharp
public abstract class CommandHandlerBase<TRequest, TResponse> : IRequestHandler<TRequest, TResponse>, IDisposable where TRequest : ICommand
{
    protected readonly IMediator _mediator;

    protected CommandHandlerBase(IMediator mediator)
    {
        this._mediator = mediator;
    }

    public virtual async Task<TResponse> Handle(TRequest request, CancellationToken cancellationToken)
    {
        try
        {
            await ValidateRequestAsync(request);

            var response = await ExecuteCommandAsync(request);

            return response;
        }
        catch (Exception ex)
        {
            throw new Exception("An error occurred during the command execution", ex);
        }
    }

    private async Task ValidateRequestAsync(TRequest request)
    {
        if (!await this._validator.ValidateAndThrowAsync(request))
        {
            throw new ValidationException();
        }
    }

    protected abstract Task<TResponse> ExecuteCommandAsync(TRequest request);

    #region Dispose pattern

    private bool disposedValue;

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                // TODO: dispose managed state (managed objects)
                _mediator?.Dispose();
            }

            // TODO: free unmanaged resources (unmanaged objects) and override finalizer
            // TODO: set large fields to null
            disposedValue = true;
        }
    }

    // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
    // ~CommandHandler()
    // {
    //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
    //     Dispose(false);
    // }

    // This code added to correctly implement the disposable pattern.
    public void Dispose()
    {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    #endregion
}
```

### 4.2.4.定义CommandHandler类
定义CommandHandler类，继承CommandHandlerBase基类，实现ExecuteCommandAsync方法，该方法用于处理命令。
```csharp
public class CreateOrderCommandHandler : CommandHandlerBase<CreateOrderCommand, OrderCreatedEvent>
{
    private readonly IRepository<OrderAggregateRoot> _repository;

    public CreateOrderCommandHandler(IRepository<OrderAggregateRoot> repository, IMediator mediator, IValidator<CreateOrderCommand> validator)
        : base(mediator)
    {
        this._repository = repository?? throw new ArgumentNullException(nameof(repository));
        this._validator = validator?? throw new ArgumentNullException(nameof(validator));
    }

    protected override async Task<OrderCreatedEvent> ExecuteCommandAsync(CreateOrderCommand request)
    {
        var order = new OrderAggregateRoot(request.AggregateId, request.OrderNumber);

        await _repository.AddAsync(order);

        var @event = new OrderCreatedEvent(request.AggregateId, request.OrderNumber);
        
        await PublishDomainEventAsync(@event);

        return @event;
    }
}
```

### 4.2.5.注册CommandHander
注册CommandHander，需要在Startup类中配置Autofac容器：
```csharp
private void ConfigureContainer(IServiceCollection services)
{
    ContainerBuilder builder = new ContainerBuilder();

    // Register types for DI here...

    builder.RegisterType<CreateOrderCommandHandler>()
         .AsImplementedInterfaces()
         .InstancePerDependency();

    // Other registrations...

    builder.Populate(services);

    var container = builder.Build();

    DependencyResolver.SetResolver(new AutofacDependencyResolver(container));
}
```

### 4.2.6.定义Query模型
定义Query模型，需要继承IQuery接口，IQuery接口需要包含AggregateId 属性。AggregateId 表示Query所在的聚合根Id。
```csharp
public class GetOrderByAggregateIdQuery : IQuery
{
    public Guid AggregateId { get; set; }
}
```

### 4.2.7.定义QueryHandler基类
定义QueryHandler基类，需要继承IRequestHandler<TRequest, TResponse>泛型接口。TRequest表示Query模型，TResponse表示Query执行的结果。
```csharp
public abstract class QueryHandlerBase<TRequest, TResponse> : IRequestHandler<TRequest, TResponse>, IDisposable where TRequest : IQuery
{
    protected readonly IMediator _mediator;

    protected QueryHandlerBase(IMediator mediator)
    {
        this._mediator = mediator;
    }

    public virtual async Task<TResponse> Handle(TRequest query, CancellationToken cancellationToken)
    {
        try
        {
            var response = await ExecuteQueryAsync(query);
            
            return response;
        }
        catch (Exception ex)
        {
            throw new Exception("An error occurred while processing your request.", ex);
        }
    }

    protected abstract Task<TResponse> ExecuteQueryAsync(TRequest query);

    #region Dispose pattern

    private bool disposedValue;

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                // TODO: dispose managed state (managed objects)
                _mediator?.Dispose();
            }

            // TODO: free unmanaged resources (unmanaged objects) and override finalizer
            // TODO: set large fields to null
            disposedValue = true;
        }
    }

    // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
    // ~CommandHandler()
    // {
    //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
    //     Dispose(false);
    // }

    // This code added to correctly implement the disposable pattern.
    public void Dispose()
    {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    #endregion
}
```

### 4.2.8.定义QueryHandler类
定义QueryHandler类，继承QueryHandlerBase基类，实现ExecuteQueryAsync方法，该方法用于处理查询。
```csharp
public class GetOrderByAggregateIdQueryHandler : QueryHandlerBase<GetOrderByAggregateIdQuery, OrderDetailsDto>
{
    private readonly IRepository<OrderAggregateRoot> _repository;

    public GetOrderByAggregateIdQueryHandler(IRepository<OrderAggregateRoot> repository, IMediator mediator) 
        : base(mediator)
    {
        this._repository = repository?? throw new ArgumentNullException(nameof(repository));
    }

    protected override async Task<OrderDetailsDto> ExecuteQueryAsync(GetOrderByAggregateIdQuery query)
    {
        var aggregate = await _repository.FindByIdAsync(query.AggregateId);
        
        if (aggregate == null)
        {
            throw new NotFoundException($"Order with id '{query.AggregateId}' was not found.");
        }

        return new OrderDetailsDto
        {
            Id = aggregate.Id,
            OrderNumber = aggregate.OrderNumber,
            CreatedAtUtc = aggregate.CreatedAtUtc,
            Status = aggregate.Status
        };
    }
}
```

### 4.2.9.注册QueryHandler
注册QueryHandler，需要在Startup类中配置Autofac容器：
```csharp
builder.RegisterType<GetOrderByAggregateIdQueryHandler>()
         .AsImplementedInterfaces()
         .InstancePerDependency();
```

### 4.2.10.发布事件
发布事件，需要定义事件模型，继承IDomainEvent接口。IDomainEvent接口需要包含AggregateId 属性。AggregateId 表示事件所在的聚合根Id。
```csharp
public class OrderCreatedEvent : IDomainEvent
{
    public Guid AggregateId { get; }

    public string OrderNumber { get; }

    public DateTime CreatedAtUtc { get; }

    public OrderCreatedEvent(Guid aggregateId, string orderNumber)
    {
        AggregateId = aggregateId;
        OrderNumber = orderNumber;
        CreatedAtUtc = DateTime.UtcNow;
    }
}
```

发布事件的方法需要实现IDomainEventDispatcher接口，该接口需要有一个PublishAsync方法。PublishAsync方法的参数为事件模型的实例。
```csharp
public interface IDomainEventDispatcher
{
    Task PublishAsync<TDomainEvent>(TDomainEvent domainEvent) where TDomainEvent : IDomainEvent;
}
```

在Startup类中，注册IDomainEventDispatcher接口的实现：
```csharp
// Register event dispatcher implementation here...
builder.RegisterType<RabbitMqDomainEventDispatcher>()
         .As<IDomainEventDispatcher>()
         .SingleInstance();
```