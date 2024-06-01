
作者：禅与计算机程序设计艺术                    

# 1.简介
  

设计模式（Design pattern）是一套被反复使用的、多种多样的面向对象的方法论，用于指导如何解决常见的问题，提高开发人员的效率、可读性、扩展性和稳定性。创建型设计模式，关注类的创建过程；结构型设计模式，关注类和对象的组合；行为型设计模式，关注对象间通信和协作。在很多面试中，还会问到设计模式应用场景以及要不要加以实践。所以掌握设计模式是面试必备技能。
这篇文章我们将回顾并复习一下，常用的13种设计模式，分别是：

1. Singleton模式
2. Factory模式
3. Observer模式
4. Command模式
5. Adapter模式
6. Template Method模式
7. State模式
8. Strategy模式
9. Visitor模式
10. Mediator模式
11. Chain of Responsibility模式
12. Flyweight模式
13. Interpreter模式

# 2.背景介绍
设计模式，即一套被反复使用的、多种多样的面向对象的方法论，最早出现于Gamma等人的著作“Design Patterns - Elements of Reusable Object-Oriented Software”中。本文根据《Design Patterns - Elements of Reusable Object-Oriented Software》进行学习。

设计模式作为经过大量研究、不断实践总结出的适合面对各种实际问题的设计规范，无疑是非常重要的。而且随着软件需求的不断变化和系统工程越来越复杂，理解设计模式、掌握设计模式的原则、应用设计模式、改进设计模式等才能让软件项目走得更远。因此，希望大家能从中受益。

# 3.基本概念术语说明
## 3.1 模式分类
模式按照其使用目的、结构和角色分成三大类：

1. 创建型模式（Creational patterns）：它们帮助你将实例化的责任分担到单独的类中，以便你可以在运行时动态地改变类的实例。比如，单例模式就是一个典型的创建型模式，它提供了一种在整个程序中都只存在唯一的一个实例的机制。另外，还有工厂方法模式、抽象工厂模式、建造者模式等。

2. 结构型模式（Structural patterns）：这些设计模式描述了一些在软件设计中的关键问题，包括对象组合、类继承和对象的装配，使得你可以创建出具有相似功能的不同类或对象，并且让它们可以相互合作。比如，代理模式、桥接模式、适配器模式、组合模式、享元模式等。

3. 行为型模式（Behavioral patterns）：这些设计模式特别关注对象之间的通信和协作。主要用来实现分布式 systems 和网络通信、对象池管理、职责链、模板方法等。比如，命令模式、迭代子模式、Mediator模式、Observer模式等。

## 3.2 设计模式六大原则
以下是软件设计的六大原则，也是设计模式的基本原则。

1. 开闭原则（Open Close Principle，OCP）：Software entities should be open for extension but closed for modification。是说软件实体应该对于扩展是开放的，但是对于修改是关闭的。换言之，意味着客户应该能够在不用重新编译的情况下增加新的功能，但不能更改现有功能的实现。

2. 依赖倒置原则（Dependency Inversion Principle，DIP）：High-level modules should not depend on low-level modules. Both should depend on abstractions. 抽象不应该依赖于细节，而细节应该依赖于抽象。换句话说，要针对接口编程而不是实现编程。

3. 单一职责原则（Single Responsibility Principle，SRP）：A class or module should have only one reason to change。每个模块或者类都应该有一个单一的功能，不能同时处理多个不同的任务。也就是说，一个类只负责一项职责，不能因为自身职责的增加而导致其他职责的降低。

4. 接口隔离原则（Interface Segregation Principle，ISP）：Make fine grained interfaces that are client specific. Separate major functionalities into multiple small ones and let them communicate with each other through their interfaces. 把客户端所需的接口保持小而精，而通过它们与彼此沟通。

5. 迪米特法则（Law of Demeter，LoD）：Talk only to your immediate friends. Only talk to those who you specifically need to know. 不要跟“陌生人”讲话。只跟同事讨论有关当前事务的问题，不要跟不相关的人讲话。

6. 里氏替换原则（Liskov Substitution Principle，LSP）：Objects in a program should be replaceable with instances of their subtypes without altering the correctness of that program. If S is a subtype of T, then objects of type T may be replaced with objects of type S without breaking any client code. 对父类来说，任何地方都可以代替它的子类，而不会影响程序的正确性。

## 3.3 UML图
UML图（Unified Modeling Language，即通用模型링语言）是用来表达系统蓝图的标准语言。它包括元素、关系、动作等，其中元素表示对象及其属性，关系表示对象间的关联，动作表示系统中的交互行为。

下表列出了常用的UML图：

类型|说明
----|----
类图（Class diagram）|描述系统中的对象及其属性、方法、操作及其联系。
状态机图（State machine diagram）|描述系统中对象状态的切换情况。
部署图（Deployment diagram）|描述系统组件之间的部署关系。
用例图（Use case diagram）|描述系统用户及其使用系统时的场景。
活动图（Activity diagram）|描述系统执行活动流程的顺序。
交互概述图（Interaction overview diagram）|描述系统的功能及其与外部系统的交互方式。
序列图（Sequence diagram）|显示对象之间的交互，描述对象生命周期。

除此之外，还有诸如领域特定建模语言（Domain Specific Languages，DSL），可用于创建特殊用途的图形表示形式。

# 4.核心算法原理和具体操作步骤以及数学公式讲解
## 4.1 Singleton模式
Singleton模式属于创建型模式，当需要控制某个类的实例只有一个的时候，可以使用Singleton模式。比如，控制一个日志文件只能生成一个实例。

Singleton模式的定义如下：
> Ensure a class has only one instance and provide a global point of access to it.

单例模式主要涉及三个角色：

1. Singleton: 单例类，在这里是Logger类。
2. Client: 使用该类的代码，比如main函数。
3. Instance variable: 类的静态变量，保存类的唯一实例。

Singleton模式的代码实现如下：

```python
class Logger(object):
    __instance = None

    @staticmethod 
    def getInstance():
        if Logger.__instance == None:
            Logger()
        return Logger.__instance
    
    def __init__(self):
        if Logger.__instance!= None:
            raise Exception("This class is a singleton!")
        else:
            Logger.__instance = self
            
    # Other methods go here...
    
if __name__ == "__main__":
    logger1 = Logger.getInstance()
    logger2 = Logger.getInstance()
    print id(logger1), " ", id(logger2)
```

首先，定义了一个名为Logger的类，然后声明一个私有的静态变量__instance，并设置默认值为None。然后定义了一个静态方法getInstance(),返回类的唯一实例。如果__instance不存在，就调用构造方法创建一个实例并将其赋值给__instance；否则直接返回__instance。

最后，如果该类的实例已经存在（即__instance不是None），则抛出异常，避免多个实例的产生。

对于Client，只需调用getInstance()方法即可获得类的唯一实例。

## 4.2 Factory模式
Factory模式是由单个类的实例化过程分离出来，允许子类决定实例化哪一个类。举个例子，电脑的制造商可能不同，但最终都生产相同的产品——笔记本。这个时候就可以使用Factory模式。

Factory模式的定义如下：
> Define an interface for creating an object, but let subclasses decide which class to instantiate. Factory Method lets a class defer instantiation it uses to subclasses.

简单来说，Factory模式就是为了把实例化的工作分离出去，由子类决定实例化哪个类。

### 4.2.1 普通工厂模式
普通工厂模式也叫简单工厂模式，是Factory模式的一种特殊形式。这种模式下，工厂类负责实现创建所有实例的逻辑，通常是通过创建实例的方法来完成，这就要求工厂类仅仅负责实例化对应的类，至于实例的创建方式则由子类来指定。

一般来说，普通工厂模式包括3个角色：

1. Product：产品接口，声明产品的共性质。
2. Concrete product：产品具体实现类，实现Product接口。
3. Creator：Creator类，即工厂类，用于创建Concrete product类的实例。

普通工厂模式的代码实现如下：

```python
from abc import ABCMeta, abstractmethod

class IShape(metaclass=ABCMeta):
    """产品接口"""
    @abstractmethod
    def draw(self):
        pass
        
class Rectangle(IShape):
    """Rectangle产品"""
    def draw(self):
        print('Drawing rectangle')
        
class Circle(IShape):
    """Circle产品"""
    def draw(self):
        print('Drawing circle')
        
class ShapeFactory(object):
    """工厂类"""
    @staticmethod
    def create_shape(type):
        """创建对应类型的产品"""
        if type =='rectangle':
            return Rectangle()
        elif type == 'circle':
            return Circle()
        else:
            return None


if __name__ == '__main__':
    shape1 = ShapeFactory.create_shape('rectangle')
    shape2 = ShapeFactory.create_shape('circle')
    shape1.draw()
    shape2.draw()
```

首先，定义了两个接口IShape和IColor，它们是产品接口。然后，定义了Rectangle和Circle两个产品类，它们实现IShape接口。最后，定义了一个工厂类ShapeFactory，它是一个静态类，提供一个静态方法create_shape()用于创建指定的产品。注意，这里的工厂类仍然是Factory模式的角色，只是这里不需要实现产品的创建，而是在创建者的内部完成。

对于客户端，通过调用ShapeFactory的create_shape()方法，传入相应参数，即可获取指定的产品。

### 4.2.2 抽象工厂模式
抽象工厂模式是Factory模式的一种变体，它可以创建一系列相关的产品对象，而无需指定具体类。换句话说，它是Factory模式的另一种扩展。抽象工厂模式下，工厂类负责实例化多个产品族中的某一个产品对象，而不需要知道具体的类。

抽象工厂模式包括四个角色：

1. Abstract factory：抽象工厂接口，声明工厂方法用于创建相关的产品。
2. Concrete factory：具体工厂类，实现Abstract factory接口，用于创建具体的产品。
3. Abstract product：抽象产品接口，声明产品的共性质。
4. Concrete product：产品具体实现类，实现Abstract product接口，定义了创建产品对象的具体过程。

抽象工厂模式的代码实现如下：

```python
from abc import ABCMeta, abstractmethod

class IShapeFactory(metaclass=ABCMeta):
    """抽象工厂接口"""
    @abstractmethod
    def create_color(self):
        pass
        
    @abstractmethod
    def create_shape(self):
        pass
        
class ColorFactory(IShapeFactory):
    """色彩工厂类"""
    def create_color(self):
        return Red()
    
class ShapeFactory(IShapeFactory):
    """形状工厂类"""
    def create_shape(self):
        return Square()
    
    
class IColor(metaclass=ABCMeta):
    """抽象产品接口"""
    @abstractmethod
    def fill(self):
        pass
        
class Red(IColor):
    """红色"""
    def fill(self):
        print('Fill red color')
        
class Green(IColor):
    """绿色"""
    def fill(self):
        print('Fill green color')
        
class Blue(IColor):
    """蓝色"""
    def fill(self):
        print('Fill blue color')
        
class ISquare(metaclass=ABCMeta):
    """抽象产品接口"""
    @abstractmethod
    def paint(self):
        pass
        
class Square(ISquare):
    """正方形"""
    def paint(self):
        print('Paint square')
        
class Circle(ISquare):
    """圆形"""
    def paint(self):
        print('Paint circle')


if __name__ == '__main__':
    color = ColorFactory().create_color()
    color.fill()
    
    shape = ShapeFactory().create_shape()
    shape.paint()
```

首先，定义了两个抽象工厂接口IShapeFactory和IColorFactory，它们是抽象工厂接口。然后，定义了两种具体工厂类ColorFactory和ShapeFactory，它们分别实现IShapeFactory和IColorFactory接口。再定义了两种抽象产品接口IColor和ISquare，它们是色彩和形状的共性质。之后，定义了具体的产品类Red、Green、Blue、Square和Circle。

对于客户端，可以通过工厂类，获取两种产品对象。由于每个工厂类只能创建一种产品，所以客户端不需要知道具体的产品类，只需调用相应的工厂方法即可。