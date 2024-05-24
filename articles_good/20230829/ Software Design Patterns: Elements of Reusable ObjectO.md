
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是设计模式？
在面向对象编程(OOP)中，设计模式（Design pattern）是用于面对各种特定问题的经验总结。它提供了一种方法论、通用模板、可重复使用的设计解决方案。软件开发人员可以使用设计模式来改善、提高代码质量、优化系统结构，以及提升软件的扩展性、可用性、并行性等。

## 为什么要学习设计模式？
随着互联网、移动互联网、物联网、云计算等技术的普及，基于网络的应用越来越多，软件架构也越来越复杂。为了应对这一挑战，设计模式被广泛使用。了解设计模式能够帮助开发者提高代码质量、优化系统结构、提升软件的扩展性、可用性、并行性等方面的能力。

## 设计模式的类型
设计模式可以分为三大类：
* 创建型模式（Creational patterns）：用来描述对象的创建过程，如单例模式、抽象工厂模式、建造者模式、原型模式等；
* 结构型模式（Structural patterns）：用来描述软件系统的组成结构，如适配器模式、桥接模式、组合模式、装饰模式、外观模式、享元模式等；
* 行为型模式（Behavioral patterns）：用来描述对象之间相互作用的方式，如策略模式、模板方法模式、观察者模式、迭代子模式、责任链模式、命令模式、状态模式等。

除了这些模式外，还有一些特殊场景下的模式，例如资源管理、并发性控制等。

# 2.设计模式分类
## 创建型模式
### 单例模式（Singleton Pattern）
所谓单例模式就是一个类只有一个实例，而且自行创建这个实例，其他任何时候都访问这个唯一的实例。单例模式主要解决的问题是在系统中只能存在一个对象时如何提供一个全局的访问点。例如，系统要求只能有一个日志记录器存在，那么就使用单例模式。

### 工厂模式（Factory Pattern）
所谓工厂模式就是定义一个用于创建对象的接口，让子类决定实例化哪一个类。工厂方法使一个类的实例化延迟到其子类。工厂模式主要解决的是当一个类不知道它所需要的对象的类的时候，而由其子类来确定其所需的对象类。工厂模式属于创建型模式，又称为虚拟构造器(Virtual Constructor)。

### 抽象工厂模式（Abstract Factory Pattern）
所谓抽象工厂模式就是提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们具体的类。抽象工厂模式提供了一个创建产品族的工厂接口，能够同时创建多个不同种类的产品。抽象工actory模式属于创建型模式，又称为Kit模式(Kitten Pattern)。

### 建造者模式（Builder Pattern）
所谓建造者模式就是将一个复杂对象的构建与它的表示分离，使得同样的构建过程可以创建不同的 representations 。建造者模式的目的是一步一步构造一个复杂的对象，允许用户只通过指定建造者的指令来创建它，用户不需要知道内部的具体构建细节。建造者模式属于创建型模式，又称为Director模式。

### 原型模式（Prototype Pattern）
所谓原型模式就是用原型实例指定创建对象的种类，并且通过拷贝这些原型创建新的对象。这种方式可以在运行时动态地进行对象创建，即通过复制已有的对象来达到新创建对象的目的。原型模式属于创建型模式。

## 结构型模式
### 适配器模式（Adapter Pattern）
所谓适配器模式就是把一个接口转换成客户希望的另一个接口。适配器模式使得原本由于接口不兼容而不能一起工作的那些类可以一起工作。适配器模式属于结构型模式。

### 桥接模式（Bridge Pattern）
所谓桥接模式就是把事物与其具体实现分开，从而使他们可以各自独立的变化。它是一种 Structural Pattern，主要解决两个稳定性不同但仍需通信的类之间的连接问题。桥接模式属于结构型模式。

### 组合模式（Composite Pattern）
所谓组合模式就是将对象组合成树形结构，使得客户端可以统一的调用组合中的所有对象。组合模式使得客户端可以一致性的处理个别对象以及组合对象，同时还可以对叶子对象和容器对象做统一的处理。组合模式属于结构型模式。

### 装饰模式（Decorator Pattern）
所谓装饰模式就是给一个对象添加一些额外的职责。就增加功能来说，装饰模式比生成子类更灵活。装饰模式属于结构型模式。

### 外观模式（Facade Pattern）
所谓外观模式就是为一个系统创建一个外观角色，这个外观角色负责将系统的多个模块的接口整合起来，这样方便客户端使用。外观模式定义了一个高层接口，这个接口使得这一整套系统更加容易使用。外观模式属于结构型模式。

### 享元模式（Flyweight Pattern）
所谓享元模式就是运用共享技术有效地支持大量细粒度的对象。系统仅保存少量对象，每个对象都包含相同的数据，而对这些数据进行共享。享元模式属于结构型模式。

## 行为型模式
### 模板方法模式（Template Method Pattern）
所谓模板方法模式就是定义一个操作中的算法骨架，而将一些步骤延迟到子类中。模板方法使得子类可以不改变一个算法的结构即可重定义该算法的某些特定步骤。模板方法模式属于行为型模式。

### 策略模式（Strategy Pattern）
所谓策略模式就是定义了算法的族，分别封装起来，让它们之间可以相互替换，此模式让算法的变化，不会影响到使用算法的客户。策略模式属于行为型模式。

### 命令模式（Command Pattern）
所谓命令模式就是将一个请求封装为一个对象，使发出请求的责任和执行请求的责任分割开。命令模式也支持撤销操作。命令模式属于行为型模式。

### 观察者模式（Observer Pattern）
所谓观察者模式就是定义对象间的一对多依赖关系，当一个对象改变状态时，所有依赖于它的对象都会收到通知并自动更新。观察者模式属于行为型模式。

### 迭代子模式（Iterator Pattern）
所谓迭代子模式就是提供一种方法顺序访问一个聚合对象中各个元素，而又无须暴露聚合对象的内部结构。迭代子模式是一种优秀的分离对象之间的 iteration，通过提供一个统一的接口来访问集合元素，而不是暴露底层的实现。迭代子模式属于行为型模式。

### 责任链模式（Chain of Responsibility Pattern）
所谓责任链模式就是使多个对象都有可能接收请求，将这些对象连成一条链，然后沿着这条链传递请求，直到有对象处理它为止。链上的处理者对象必须严格遵守某个特定的接口标准。责任链模式属于行为型模式。

### 状态模式（State Pattern）
所谓状态模式就是允许对象在内部状态发生改变时改变其行为，对象看起来好像修改了其类。状态模式属于行为型模式。

# 3.具体模式介绍
## 单例模式——确保一个类只有一个实例，并提供一个访问它的全局点
单例模式是创建型模式中的一种，其作用是保证一个类仅有一个实例，并提供一个访问它的全局点。单例模式提供了对唯一实例的受控访问。

### 优点
* 对于频繁使用的对象，这是一件非常省事的事情。由于单例对象只需要初始化一次，因此可以降低系统的内存使用。
* 对系统的职责划分有很大的帮助，由于一个类只有一个实例，因而可以避免多重实例化导致的冲突。
* 可以节约系统资源，如果一个对象实例被大量创建，那么就会占用很多的内存，而采用单例模式之后，系统会共享内存，节约资源。

### 缺点
单例模式一般没有接口，扩展比较困难。

### 使用场景
* 有状态的工具类对象，例如日志工具类。
* 需要避免共享的全局变量。
* 有几个子类非常类似，且逻辑复杂，但是它们都要求生成自己的实例时，可以使用单例模式。

### 示例代码
```java
public class Singleton {
    private static final Singleton instance = new Singleton();

    // make constructor private to prevent instantiation from outside
    private Singleton() {}

    public static synchronized Singleton getInstance() {
        return instance;
    }

    // other methods can be defined here...
}
```
以上是一个简单版的单例模式的实现。其实上述的代码已经满足了单例模式的要求。下面是一些典型的使用场景：

#### Lazy Initialization Singleton
该模式懒汉式加载，延迟实例化，提前完成初始化，只有第一次调用getInstance()方法才会真正创建对象。
```java
public class Singleton {
  private volatile static Singleton singletonInstance;
  
  private Singleton (){}

  public static Singleton getSingleton(){
      if (singletonInstance == null){
          synchronized(Singleton.class){
              if (singletonInstance == null){
                  singletonInstance = new Singleton();
              }
          }
      }
      return singletonInstance;
  }
}
```

#### Double Check Locking Singleton
抑制Lazy Initialization Singleton反射攻击，为了防止延迟加载攻击，需要在代码块中加锁，确保每次只有一个线程能成功初始化对象。
```java
public class Singleton{
   private static volatile Singleton instance = null;

   private Singleton(){}

   public static Singleton getInstance(){
       if (instance == null){
           synchronized(Singleton.class){
               if (instance == null){
                   instance = new Singleton();
               }
           }
       }
       return instance;
   }
}
```

#### ThreadSafe Singleton
由于类的构造函数可能会被并发调用，所以需要线程安全的单例模式。单例对象的创建应该是线程安全的，也就是说，同一时间只能有一个线程能实例化单例对象。为了达到这种效果，需要使用volatile关键字声明类实例。
```java
public class Singleton {
    private volatile static Singleton singleton;
    
    private Singleton() {}
 
    public static Singleton getInstance() {
        if (singleton == null) {
            synchronized (Singleton.class) {
                if (singleton == null) {
                    singleton = new Singleton();
                }
            }
        }
        return singleton;
    }
}
```

## 工厂模式——定义一个用于创建对象的接口，让子类决定实例化哪一个类
工厂模式是创建型模式的一种，其角色是定义一个用于创建对象的接口，让子类决定实例化哪一个类。这种类型的设计模式属于创建型模式，它提供了一种创建对象的最佳方式。

### 优点
* 它大大减少了项目工程中的耦合度，当需求变更时，只需要调整派生工厂类就可以了，而无需修改其他部份的代码。
* 当一个产品的生命周期长时，只需要修改工厂类即可，因为产品的派生类都包含了具体产品的创建逻辑。
* 在系统中加入新产品时，只需要添加对应的产品类和派生工厂类，无需做较多的改动。

### 缺点
* 每次增加一个产品时，都需要编写相应的工厂类，费时费力。
* 不容易排错，如果一个产品出现故障，或者产品类结构有变化，则所有的工厂类都需要进行相应的修改。

### 使用场景
* 当一个类不知道它所需的对象的类时。
* 当一个类希望由多个子类中的一个来指定时。
* 当类将创建对象的职责委托给多个帮助类中的某一个时。
* 当对象由第三方提供时。


### 示例代码
```java
// Product Interface
interface Shape {
    void draw();
}

// Concrete Products
class Circle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a circle.");
    }
}

class Rectangle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a rectangle.");
    }
}

// Abstract Factory Interface
abstract class ShapeFactory {
    abstract Shape createShape();
}

// Concrete Factories
class CircleFactory extends ShapeFactory {
    @Override
    Shape createShape() {
        return new Circle();
    }
}

class RectangleFactory extends ShapeFactory {
    @Override
    Shape createShape() {
        return new Rectangle();
    }
}

class Main {
    public static void main(String[] args) {
        // Client code
        ShapeFactory factory = new CircleFactory();
        Shape shape1 = factory.createShape();
        shape1.draw();

        factory = new RectangleFactory();
        Shape shape2 = factory.createShape();
        shape2.draw();
    }
}
```

## 抽象工厂模式——提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们具体的类
抽象工厂模式是创建型模式之一，其主要优点在于隔离了具体类的生成，使得客户端并不知道系统的具体实现，这使得整个系统的结构松耦合。

### 优点
* 提供了一种抽象的工厂方法来负责创建一系列相关的对象。
* 分离了客户端代码和具体的工厂类，使得两者之间的耦合度降低，提高系统的可扩展性。
* 可以更换产品系列中的具体产品。

### 缺点
* 添加新产品时代价很大，每增加一个产品都需要修改抽象工厂和所有具体工厂，增加了系统的复杂度。

### 使用场景
* 一个系统要独立于它的产品的创建、组合和表示时。
* 一个系统想要提供一个产品类的库，而并非必须知道其创建细节时。
* 系统中有多于一个的产品族，而系统只消费其中一族时。
* 系统要求提供一个静态的工厂方法，而不是一个实例化的工厂类时。

### 示例代码
```java
// Component Interfaces
interface Button {
    void paint();
}

interface Checkbox {
    void paint();
}

interface Textbox {
    void paint();
}

// Concrete Components
class WinButton implements Button {
    @Override
    public void paint() {
        System.out.println("Painting a Windows button.");
    }
}

class OSXButton implements Button {
    @Override
    public void paint() {
        System.out.println("Painting an OS X button.");
    }
}

class GnomeCheckbox implements Checkbox {
    @Override
    public void paint() {
        System.out.println("Painting a GNOME checkbox.");
    }
}

class KDECheckbox implements Checkbox {
    @Override
    public void paint() {
        System.out.println("Painting a KDE checkbox.");
    }
}

class WinTextbox implements Textbox {
    @Override
    public void paint() {
        System.out.println("Painting a Windows textbox.");
    }
}

class WebTextbox implements Textbox {
    @Override
    public void paint() {
        System.out.println("Painting a web textbox.");
    }
}

// Abstraction
interface GUIFactory {
    Button createButton();
    Checkbox createCheckbox();
    Textbox createTextbox();
}

// Concrete Implementations
class WindowsGUIFactory implements GUIFactory {
    @Override
    public Button createButton() {
        return new WinButton();
    }

    @Override
    public Checkbox createCheckbox() {
        return new GnomeCheckbox();
    }

    @Override
    public Textbox createTextbox() {
        return new WinTextbox();
    }
}

class MacGUIFactory implements GUIFactory {
    @Override
    public Button createButton() {
        return new OSXButton();
    }

    @Override
    public Checkbox createCheckbox() {
        return new KDECheckbox();
    }

    @Override
    public Textbox createTextbox() {
        return new WebTextbox();
    }
}

// Usage Example
public class Demo {
    public static void main(String[] args) {
        GUIFactory guiFactory = new WindowsGUIFactory();

        Button button = guiFactory.createButton();
        button.paint();

        Checkbox checkbox = guiFactory.createCheckbox();
        checkbox.paint();

        Textbox textbox = guiFactory.createTextbox();
        textbox.paint();
    }
}
```