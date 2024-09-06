                 

### Agent：数字代理的独立决策和行动能力

#### 1. 什么是代理模式？

**题目：** 请解释什么是代理模式，并给出一个简单的示例。

**答案：** 代理模式是一种设计模式，用于为其他对象提供一种代理以控制对这个对象的访问。代理对象负责管理对真实对象的访问，并提供额外的功能，例如安全检查、日志记录等。

**示例：**

```python
class ImageProxy:
    def __init__(self, real_image):
        self.real_image = real_image

    def display(self):
        self.real_image.display()
        print("Displaying image with proxy.")

class RealImage:
    def display(self):
        print("Displaying real image.")

image = RealImage()
proxy = ImageProxy(image)
proxy.display()
```

**解析：** 在这个示例中，`ImageProxy` 类作为 `RealImage` 类的代理。当调用 `proxy.display()` 时，首先会调用 `RealImage` 的 `display()` 方法，然后代理会额外打印一条信息。

#### 2. 代理模式的用途是什么？

**题目：** 请列举代理模式的主要用途。

**答案：** 代理模式的主要用途包括：

- **控制访问：** 通过代理对象控制对真实对象的访问，例如限制权限或执行安全检查。
- **延迟初始化：** 当真实对象需要较大的资源消耗时，代理可以在需要时才初始化真实对象。
- **日志记录：** 在代理中记录操作日志，方便后续分析。
- **缓存：** 使用代理缓存真实对象的方法结果，提高性能。
- **中介：** 在客户端和真实对象之间添加中介，简化交互。

#### 3. 请解释代理模式中的静态代理和动态代理。

**题目：** 请解释静态代理和动态代理，并给出各自的应用场景。

**答案：** 

* **静态代理：** 静态代理在编译时确定代理对象和真实对象的关系。代理类和真实对象通常在同一个包中，并且代理类通常实现与真实对象相同的接口。

应用场景：适用于功能简单的代理场景，例如日志记录、权限控制等。

* **动态代理：** 动态代理在运行时动态生成代理对象，代理对象和真实对象的关系在运行时确定。动态代理通常使用反射（reflection）机制实现。

应用场景：适用于功能复杂的代理场景，例如缓存、中介等。

#### 4. 什么是中介者模式？

**题目：** 请解释中介者模式，并给出一个简单的示例。

**答案：** 中介者模式是一种行为型设计模式，用于降低多个对象之间的耦合度。中介者对象负责协调和通信，从而减少对象之间的直接依赖。

**示例：**

```python
class Mediator:
    def __init__(self):
        self.components = []

    def register(self, component):
        self.components.append(component)

    def notify(self, sender, event):
        for component in self.components:
            if component != sender:
                component.receive(event)

class Component:
    def __init__(self, mediator):
        self.mediator = mediator

    def send(self, event):
        self.mediator.notify(self, event)

    def receive(self, event):
        print(f"Received event: {event}")

mediator = Mediator()
component1 = Component(mediator)
component2 = Component(mediator)
mediator.register(component1)
mediator.register(component2)
component1.send("Hello")
component2.send("World")
```

**解析：** 在这个示例中，`Mediator` 类负责协调 `Component` 对象之间的通信。当 `component1` 发送 "Hello" 时，`Mediator` 将此事件通知给 `component2`，反之亦然。

#### 5. 中介者模式的主要用途是什么？

**题目：** 请列举中介者模式的主要用途。

**答案：** 中介者模式的主要用途包括：

- **降低耦合度：** 通过中介者对象减少对象之间的直接依赖。
- **简化通信：** 中介者负责协调和通信，简化对象之间的交互。
- **扩展性：** 通过中介者对象可以轻松地添加或修改对象之间的交互逻辑。

#### 6. 什么是责任链模式？

**题目：** 请解释责任链模式，并给出一个简单的示例。

**答案：** 责任链模式是一种行为型设计模式，用于将多个对象连接成一条链，每个对象负责处理一部分请求。如果一个对象不能处理该请求，它会将请求传递给链中的下一个对象。

**示例：**

```python
class Handler:
    def __init__(self, successor=None):
        self._successor = successor

    def handle(self, request):
        if not self._successor:
            raise NotImplementedError("handle must be implemented")
        self._successor.handle(request)

class ConcreteHandler1(Handler):
    def handle(self, request):
        if 0 < request <= 10:
            print(f"ConcreteHandler1 handles request {request}.")
        elif self._successor:
            self._successor.handle(request)

class ConcreteHandler2(Handler):
    def handle(self, request):
        if 10 < request <= 20:
            print(f"ConcreteHandler2 handles request {request}.")
        elif self._successor:
            self._successor.handle(request)

class DefaultHandler(Handler):
    def handle(self, request):
        print(f"DefaultHandler handles request {request}.")

chain = ConcreteHandler1(ConcreteHandler2(DefaultHandler()))
chain.handle(5)
chain.handle(15)
chain.handle(30)
```

**解析：** 在这个示例中，`ConcreteHandler1`、`ConcreteHandler2` 和 `DefaultHandler` 组成了一条责任链。当 `chain.handle(5)` 时，请求会被 `ConcreteHandler1` 处理；当 `chain.handle(15)` 时，请求会被 `ConcreteHandler2` 处理；当 `chain.handle(30)` 时，请求会被 `DefaultHandler` 处理。

#### 7. 责任链模式的主要用途是什么？

**题目：** 请列举责任链模式的主要用途。

**答案：** 责任链模式的主要用途包括：

- **分而治之：** 将复杂请求分散到多个对象中处理，降低单个对象的责任。
- **灵活扩展：** 可以动态地添加或移除处理请求的对象，而不影响其他对象。
- **避免循环调用：** 通过责任链模式，可以避免对象之间的循环调用。

#### 8. 什么是解释器模式？

**题目：** 请解释解释器模式，并给出一个简单的示例。

**答案：** 解释器模式是一种行为型设计模式，用于实现语言解析器。解释器模式将解析过程分解为多个对象，每个对象负责解释一个特定的语言元素。

**示例：**

```python
class Expression:
    def interpret(self, text):
        raise NotImplementedError("interpret must be implemented")

class TerminalExpression(Expression):
    def interpret(self, text):
        if text == "buy gold":
            return "buy gold"
        else:
            return None

class NonTerminalExpression(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def interpret(self, text):
        left_result = self.left.interpret(text)
        if left_result:
            right_result = self.right.interpret(left_result)
            return right_result
        return None

context = "buy gold after 3 days"
parser = NonTerminalExpression(
    TerminalExpression(),
    TerminalExpression(),
)

result = parser.interpret(context)
if result:
    print("The final result is:", result)
else:
    print("The context cannot be interpreted.")
```

**解析：** 在这个示例中，`Expression` 类是抽象解释器，`TerminalExpression` 类是终结符解释器，`NonTerminalExpression` 类是非终结符解释器。解析器通过组合这些解释器来解释复杂的语句。

#### 9. 解释器模式的主要用途是什么？

**题目：** 请列举解释器模式的主要用途。

**答案：** 解释器模式的主要用途包括：

- **实现自定义语言：** 通过创建特定的解释器，可以实现自定义语言的解析。
- **模块化解析：** 将解析过程分解为多个模块，提高代码的可维护性和可扩展性。
- **动态解释：** 允许在运行时动态解释和执行代码。

#### 10. 什么是命令模式？

**题目：** 请解释命令模式，并给出一个简单的示例。

**答案：** 命令模式是一种行为型设计模式，用于将请求封装为一个对象，从而使您能够将请求参数化、排队或记录请求日志，同时支持可撤销的操作。

**示例：**

```python
class Command:
    def execute(self):
        raise NotImplementedError("execute must be implemented")

class LightOnCommand(Command):
    def __init__(self, light):
        self.light = light

    def execute(self):
        self.light.on()

class LightOffCommand(Command):
    def __init__(self, light):
        self.light = light

    def execute(self):
        self.light.off()

class NoOperationCommand(Command):
    def execute(self):
        pass

class RemoteControl:
    def __init__(self):
        self.on_commands = []
        self.off_commands = []

    def store_and_execute(self, command):
        command.execute()

    def on_button_pressed(self, command):
        self.on_commands.append(command)

    def off_button_pressed(self, command):
        self.off_commands.append(command)

light = Light()
remote = RemoteControl()
remote.on_button_pressed(LightOnCommand(light))
remote.off_button_pressed(LightOffCommand(light))
remote.store_and_execute()
```

**解析：** 在这个示例中，`RemoteControl` 类负责存储命令对象并执行它们。当按下按钮时，会相应地将命令添加到 `on_commands` 或 `off_commands` 列表中，然后通过 `store_and_execute()` 方法执行命令。

#### 11. 命令模式的主要用途是什么？

**题目：** 请列举命令模式的主要用途。

**答案：** 命令模式的主要用途包括：

- **参数化请求：** 将请求作为对象传递，使其可以参数化、排队或记录。
- **撤销操作：** 通过实现 `undo()` 方法，可以使命令可撤销。
- **队列处理：** 将请求放入队列中，按顺序执行。
- **日志记录：** 可以记录命令的执行情况，便于后续分析。

#### 12. 什么是迭代器模式？

**题目：** 请解释迭代器模式，并给出一个简单的示例。

**答案：** 迭代器模式是一种行为型设计模式，用于提供一种方法顺序访问一个集合中的各个元素，而无需暴露集合的内部表示。

**示例：**

```python
class Iterator:
    def __init__(self, collection):
        self.collection = collection
        self.index = 0

    def has_next(self):
        return self.index < len(self.collection)

    def next(self):
        if self.has_next():
            item = self.collection[self.index]
            self.index += 1
            return item
        return None

class ListIterator(Iterator):
    def __init__(self, collection):
        super().__init__(collection)

    def has_next(self):
        return self.index < len(self.collection)

    def next(self):
        if self.has_next():
            item = self.collection[self.index]
            self.index += 1
            return item
        return None

class Collection:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    def get_iterable(self):
        return self.items

collection = Collection()
collection.add("apple")
collection.add("banana")
collection.add("cherry")

iterator = ListIterator(collection)
while iterator.has_next():
    print(iterator.next())
```

**解析：** 在这个示例中，`Iterator` 类是抽象迭代器，`ListIterator` 类是具体迭代器，`Collection` 类是集合类。`ListIterator` 类实现了迭代器的核心方法 `next()` 和 `has_next()`，用于遍历集合中的元素。

#### 13. 迭代器模式的主要用途是什么？

**题目：** 请列举迭代器模式的主要用途。

**答案：** 迭代器模式的主要用途包括：

- **封装遍历：** 提供一种统一的方式来遍历不同的数据结构。
- **支持反向遍历：** 通过实现反向迭代器，可以支持反向遍历。
- **支持多重遍历：** 可以同时使用多个迭代器对同一数据结构进行遍历。
- **避免暴露内部表示：** 避免在客户端代码中直接访问集合的内部表示。

#### 14. 什么是中介者模式？

**题目：** 请解释中介者模式，并给出一个简单的示例。

**答案：** 中介者模式是一种行为型设计模式，用于降低多个对象之间的耦合度。中介者对象负责协调和通信，从而减少对象之间的直接依赖。

**示例：**

```python
class Mediator:
    def __init__(self):
        self.components = []

    def register(self, component):
        self.components.append(component)

    def notify(self, sender, event):
        for component in self.components:
            if component != sender:
                component.receive(event)

class Component:
    def __init__(self, mediator):
        self.mediator = mediator

    def send(self, event):
        self.mediator.notify(self, event)

    def receive(self, event):
        print(f"Received event: {event}")

mediator = Mediator()
component1 = Component(mediator)
component2 = Component(mediator)
mediator.register(component1)
mediator.register(component2)
component1.send("Hello")
component2.send("World")
```

**解析：** 在这个示例中，`Mediator` 类负责协调 `Component` 对象之间的通信。当 `component1` 发送 "Hello" 时，`Mediator` 将此事件通知给 `component2`，反之亦然。

#### 15. 中介者模式的主要用途是什么？

**题目：** 请列举中介者模式的主要用途。

**答案：** 中介者模式的主要用途包括：

- **降低耦合度：** 通过中介者对象减少对象之间的直接依赖。
- **简化通信：** 中介者负责协调和通信，简化对象之间的交互。
- **扩展性：** 通过中介者对象可以轻松地添加或修改对象之间的交互逻辑。

#### 16. 什么是观察者模式？

**题目：** 请解释观察者模式，并给出一个简单的示例。

**答案：** 观察者模式是一种行为型设计模式，用于定义对象间的一对多依赖，当一个对象的状态发生变化时，所有依赖于它的对象都会自动收到通知。

**示例：**

```python
class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self, event):
        for observer in self._observers:
            observer.update(event)

class Observer:
    def update(self, event):
        print(f"Observer received event: {event}")

subject = Subject()
observer1 = Observer()
observer2 = Observer()
subject.attach(observer1)
subject.attach(observer2)
subject.notify("Event occurred")
```

**解析：** 在这个示例中，`Subject` 类维护一个观察者列表，当状态发生变化时，会通知所有观察者。`Observer` 类实现了 `update()` 方法，用于处理通知。

#### 17. 观察者模式的主要用途是什么？

**题目：** 请列举观察者模式的主要用途。

**答案：** 观察者模式的主要用途包括：

- **事件驱动：** 当对象状态发生变化时，自动通知所有依赖对象。
- **解耦：** 降低对象之间的耦合度，实现对象间的松耦合。
- **动态扩展：** 可以动态地添加或移除观察者。

#### 18. 什么是策略模式？

**题目：** 请解释策略模式，并给出一个简单的示例。

**答案：** 策略模式是一种行为型设计模式，用于定义一系列算法，将每个算法封装起来，并使它们可以相互替换。策略模式允许使用相同接口实现不同的算法变体，从而使算法的变化不会影响到使用算法的客户类。

**示例：**

```python
class Strategy:
    def execute(self, data):
        raise NotImplementedError("execute must be implemented")

class ConcreteStrategyA(Strategy):
    def execute(self, data):
        print(f"Executing strategy A with data: {data}")

class ConcreteStrategyB(Strategy):
    def execute(self, data):
        print(f"Executing strategy B with data: {data}")

class Context:
    def __init__(self, strategy):
        self._strategy = strategy

    def set_strategy(self, strategy):
        self._strategy = strategy

    def execute_strategy(self, data):
        self._strategy.execute(data)

context = Context(ConcreteStrategyA())
context.execute_strategy("Data for strategy A")

context.set_strategy(ConcreteStrategyB())
context.execute_strategy("Data for strategy B")
```

**解析：** 在这个示例中，`Strategy` 类是策略接口，`ConcreteStrategyA` 和 `ConcreteStrategyB` 类是具体策略实现，`Context` 类是使用策略的上下文类。可以通过设置不同的具体策略来改变 `Context` 的行为。

#### 19. 策略模式的主要用途是什么？

**题目：** 请列举策略模式的主要用途。

**答案：** 策略模式的主要用途包括：

- **封装算法变体：** 封装一系列算法变体，使其可以相互替换。
- **解耦：** 降低策略与上下文之间的耦合度，使算法的变化不会影响到上下文。
- **可扩展性：** 可以动态地添加或修改策略，而不需要修改上下文。

#### 20. 什么是模板方法模式？

**题目：** 请解释模板方法模式，并给出一个简单的示例。

**答案：** 模板方法模式是一种行为型设计模式，用于定义一个操作中的算法骨架，将一些步骤延迟到子类中。模板方法模式使得子类可以覆盖算法中的特定步骤，而不必改变整个算法的结构。

**示例：**

```python
class TemplateMethod:
    def template_method(self):
        self.step1()
        self.step2()
        self.step3()

    def step1(self):
        print("Step 1")

    def step2(self):
        print("Step 2")

    def step3(self):
        print("Step 3")

class ConcreteTemplate(TemplateMethod):
    def step2(self):
        print("Modified Step 2")

template = ConcreteTemplate()
template.template_method()
```

**解析：** 在这个示例中，`TemplateMethod` 类定义了一个模板方法 `template_method()`，包含了三个步骤：`step1()`、`step2()` 和 `step3()`。`ConcreteTemplate` 类继承自 `TemplateMethod`，并修改了 `step2()` 的实现。

#### 21. 模板方法模式的主要用途是什么？

**题目：** 请列举模板方法模式的主要用途。

**答案：** 模板方法模式的主要用途包括：

- **定义算法骨架：** 提供一个算法的骨架，使子类可以覆盖部分步骤。
- **代码复用：** 通过模板方法模式，可以避免在多个类中重复编写相同的算法骨架。
- **灵活扩展：** 允许子类在不改变整体算法结构的情况下，扩展或修改特定步骤。

#### 22. 什么是建造者模式？

**题目：** 请解释建造者模式，并给出一个简单的示例。

**答案：** 建造者模式是一种创建型设计模式，用于将一个复杂对象的构建与其表示分离，使得同样的构建过程可以创建不同的表示。

**示例：**

```python
class Builder:
    def build_part_a(self):
        raise NotImplementedError("build_part_a must be implemented")

    def build_part_b(self):
        raise NotImplementedError("build_part_b must be implemented")

    def build_part_c(self):
        raise NotImplementedError("build_part_c must be implemented")

    def get_product(self):
        raise NotImplementedError("get_product must be implemented")

class ConcreteBuilder(Builder):
    def build_part_a(self):
        print("Building part A")

    def build_part_b(self):
        print("Building part B")

    def build_part_c(self):
        print("Building part C")

    def get_product(self):
        return Product()

class Director:
    def __init__(self, builder):
        self._builder = builder

    def construct_product(self):
        self._builder.build_part_a()
        self._builder.build_part_b()
        self._builder.build_part_c()

director = Director(ConcreteBuilder())
director.construct_product()
```

**解析：** 在这个示例中，`Builder` 类是抽象建造者，`ConcreteBuilder` 类是具体建造者，`Director` 类是导演类。导演类负责调用建造者类的方法，按照特定的顺序构建产品。

#### 23. 建造者模式的主要用途是什么？

**题目：** 请列举建造者模式的主要用途。

**答案：** 建造者模式的主要用途包括：

- **构建复杂对象：** 用于构建具有多个组成部分的复杂对象。
- **分离构建和表示：** 将构建过程与产品表示分离，使得构建过程可以独立于产品表示进行修改。
- **代码复用：** 避免在多个构造函数中重复编写相同的构建逻辑。

#### 24. 什么是工厂方法模式？

**题目：** 请解释工厂方法模式，并给出一个简单的示例。

**答案：** 工厂方法模式是一种创建型设计模式，用于定义一个接口用于创建对象，但让子类决定实例化的类是哪一个。工厂方法使一个类的实例化延迟到其子类。

**示例：**

```python
class Creator:
    def create_product(self):
        raise NotImplementedError("create_product must be implemented")

class ConcreteCreatorA(Creator):
    def create_product(self):
        return ProductA()

class ConcreteCreatorB(Creator):
    def create_product(self):
        return ProductB()

class Product:
    def operation(self):
        raise NotImplementedError("operation must be implemented")

class ProductA(Product):
    def operation(self):
        print("Product A operation")

class ProductB(Product):
    def operation(self):
        print("Product B operation")

creator_a = ConcreteCreatorA()
creator_b = ConcreteCreatorB()

product_a = creator_a.create_product()
product_a.operation()

product_b = creator_b.create_product()
product_b.operation()
```

**解析：** 在这个示例中，`Creator` 类是抽象创建者，`ConcreteCreatorA` 和 `ConcreteCreatorB` 类是具体创建者，`Product` 类是产品类，`ProductA` 和 `ProductB` 类是具体产品类。具体创建者决定实例化的具体产品类。

#### 25. 工厂方法模式的主要用途是什么？

**题目：** 请列举工厂方法模式的主要用途。

**答案：** 工厂方法模式的主要用途包括：

- **创建对象：** 用于创建对象，而不需要知道具体创建的是哪个类的实例。
- **解耦：** 将对象的创建和使用分离，降低系统间的耦合度。
- **扩展性：** 可以通过扩展具体创建者类来扩展系统功能，而无需修改现有代码。

#### 26. 什么是抽象工厂模式？

**题目：** 请解释抽象工厂模式，并给出一个简单的示例。

**答案：** 抽象工厂模式是一种创建型设计模式，用于创建一系列相关或相互依赖对象的接口，而不需要明确指定具体类。它提供一个接口，用于创建一组相关对象的实例，隐藏创建逻辑的具体实现。

**示例：**

```python
class AbstractFactory:
    def create_product_a(self):
        raise NotImplementedError("create_product_a must be implemented")

    def create_product_b(self):
        raise NotImplementedError("create_product_b must be implemented")

class ConcreteFactoryA(AbstractFactory):
    def create_product_a(self):
        return ProductA()

    def create_product_b(self):
        return ProductB()

class ConcreteFactoryB(AbstractFactory):
    def create_product_a(self):
        return ProductA()

    def create_product_b(self):
        return ProductB()

class ProductA:
    def operation(self):
        raise NotImplementedError("operation must be implemented")

class ProductB:
    def operation(self):
        raise NotImplementedError("operation must be implemented")

class ProductAC(ProductA):
    def operation(self):
        print("Product AC operation")

class ProductBC(ProductB):
    def operation(self):
        print("Product BC operation")

factory_a = ConcreteFactoryA()
factory_b = ConcreteFactoryB()

product_a_a = factory_a.create_product_a()
product_a_a.operation()

product_b_b = factory_b.create_product_b()
product_b_b.operation()
```

**解析：** 在这个示例中，`AbstractFactory` 类是抽象工厂，`ConcreteFactoryA` 和 `ConcreteFactoryB` 类是具体工厂，`ProductA` 和 `ProductB` 类是产品类，`ProductAC` 和 `ProductBC` 类是具体产品类。具体工厂决定创建具体产品类的实例。

#### 27. 抽象工厂模式的主要用途是什么？

**题目：** 请列举抽象工厂模式的主要用途。

**答案：** 抽象工厂模式的主要用途包括：

- **创建相关对象的组合：** 用于创建一系列相关或相互依赖对象的组合。
- **解耦：** 降低系统间的耦合度，使得产品类之间的依赖关系通过工厂类来管理。
- **扩展性：** 可以通过扩展具体工厂类来扩展系统功能，而无需修改现有代码。

#### 28. 什么是单例模式？

**题目：** 请解释单例模式，并给出一个简单的示例。

**答案：** 单例模式是一种创建型设计模式，用于确保一个类只有一个实例，并提供一个全局访问点来访问这个实例。

**示例：**

```python
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def some_method(self):
        print("Some method of Singleton.")

singleton1 = Singleton()
singleton2 = Singleton()

print(singleton1 is singleton2)  # 输出 True
```

**解析：** 在这个示例中，`Singleton` 类是一个单例类。通过重写 `__new__` 方法，确保创建实例时只创建一个实例，并返回这个实例。`singleton1` 和 `singleton2` 都是同一个实例，因此 `singleton1 is singleton2` 输出 `True`。

#### 29. 单例模式的主要用途是什么？

**题目：** 请列举单例模式的主要用途。

**答案：** 单例模式的主要用途包括：

- **确保唯一实例：** 确保一个类只有一个实例，防止多次创建。
- **全局访问点：** 提供一个全局访问点，方便其他类访问单例实例。
- **资源管理：** 用于管理共享资源，例如数据库连接、文件系统等。

#### 30. 什么是适配器模式？

**题目：** 请解释适配器模式，并给出一个简单的示例。

**答案：** 适配器模式是一种结构型设计模式，用于将一个类的接口转换成客户期望的另一个接口。适配器让原本接口不兼容的类可以在一起工作。

**示例：**

```python
class Adaptee:
    def specific_method(self):
        print("Specific method.")

class Target:
    def target_method(self, arg):
        print(f"Target method with arg: {arg}")

class Adapter(Adaptee, Target):
    def target_method(self, arg):
        super().specific_method()
        print(f"Adapting to target method with arg: {arg}")

adaptee = Adaptee()
target = Target()
adapter = Adapter()

adaptee.specific_method()
target.target_method("arg")
adapter.target_method("arg")
```

**解析：** 在这个示例中，`Adaptee` 类实现了 `specific_method()` 方法，`Target` 类期望实现 `target_method()` 方法。`Adapter` 类继承自 `Adaptee` 和 `Target`，实现了 `target_method()` 方法，并调用 `specific_method()` 方法，从而实现了适配。

#### 31. 适配器模式的主要用途是什么？

**题目：** 请列举适配器模式的主要用途。

**答案：** 适配器模式的主要用途包括：

- **接口转换：** 将一个类的接口转换成客户期望的另一个接口。
- **兼容性：** 使不兼容的接口能够协同工作。
- **复用：** 通过适配器，可以复用现有的类，而无需修改这些类的接口。

