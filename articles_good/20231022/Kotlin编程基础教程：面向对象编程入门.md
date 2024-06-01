
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 概述
Kotlin是 JetBrains 推出的新一代跨平台语言，主要用于 Android、JVM 和 JavaScript 的开发。它是 Java 在功能性方面的增强版，语法更简洁，并且支持函数式编程。在 2017 年，JetBrains 将 Kotlin 提交给了 Apache Software Foundation（ASF）基金会作为开源项目，并宣布 Kotlin 是 Java 的下一个主要版本。目前，Kotlin 支持几乎所有主流平台，包括 Windows、Linux、macOS、Android、iOS、服务器端等。

Kotlin 是一门静态类型、反射可用的语言，它支持高级特性，如扩展、泛型、协程、委托、DSL （领域特定语言）。由于 Kotlin 可以通过 Java 互操作机制调用 Java 类库，所以 Kotlin 可以很好的与 Java 生态系统进行集成。同时，由于 Kotlin 的语法更接近于 Java，因此一些 Java 程序员也可以很容易地学习 Kotlin ，让程序编写变得更加简单和快速。

Kotlin 拥有自己的编译器，并且可以轻松将其编译为 Java 字节码或 JavaScript 。这种统一的编译器允许 Kotlin 程序员无缝切换到 Java 或 JavaScript 环境。Kotlin 支持多平台编程，因此可以在 JVM 上运行 Kotlin 程序，也可以在 JavaScript 引擎上运行 Kotlin 程序，也可以在 Android 设备上运行 Kotlin 程序。

本系列教程将从以下几个方面介绍 Kotlin 中的面向对象编程知识：

- 基础知识：基本概念、关键字及相关用法。
- 对象：类的定义及创建、继承、接口、内部类。
- 多态：继承、重载、动态类型。
- 抽象：抽象类、接口、注解。
- 枚举：枚举类、对象、序列化。
- 嵌套：成员、局部、内部类。
- 委托：委托模式、代理模式。

## 目标读者
- 对 Kotlin 有初步了解。
- 对计算机编程有基本认识。
- 对面向对象的编程有一定了解。

## 本教程适合人群
- 期望掌握 Kotlin 基础语法。
- 需要了解 Kotlin 的基本面向对象编程知识。
- 想要学习 Kotlin 的面向对象编程知识。

## 作者简介
陈磊，国内首批 Kotlin 产品经理，Android 大牛，推动 Kotlin 成为主流语言。曾任职于阿里巴巴集团。擅长分享技术心得、培训讲师，译制《Kotlin 极简教程》。


# 2.核心概念与联系
## 一、什么是面向对象？
面向对象(Object-Oriented Programming, OOP)是一种抽象思维的编程范式。是一种基于对象的编程方法，具有封装、继承、多态四个特征。其中，封装是指将数据和实现细节隐藏在对象的内部，只暴露必要的接口；继承是指利用已有类的功能，创建新的类；多态是指允许不同类的对象对同一消息作出不同的响应，即对象在不同情形下的表现形式不一样。

在 OOP 中，我们把一个复杂系统分解成若干个相互独立且功能完备的对象。每个对象都是一个小世界，里面封装着自己的状态和行为，而外界访问不到对象的内部信息。通过消息传递和接收，对象间可以互通信息，达到各自功能的复用和扩展。

面向对象编程是一种非常重要的编程范式，因为它提供了一种高层次的抽象，使复杂系统能够被看作由多个对象组成的各自自治的、相互联系的实体。这样，系统中的每一个模块都可以单独构建、测试、调试、修改，从而提升整个系统的质量。

## 二、什么是类？
类是面向对象编程的一个重要概念。在 Java 中，我们用 class 来定义一个类。类中包含了变量和方法，这些变量和方法决定了一个对象的行为。

```java
public class Person {
    private String name; // 姓名
    private int age;     // 年龄

    public void sayHello() {
        System.out.println("Hello!");
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return this.name;
    }
}
```

在这个例子中，Person 是一个类，它包含两个私有变量 `name` 和 `age`，以及两个方法 `sayHello()`、`setName()` 和 `getName()`。其中，`sayHello()` 方法用于打印一条问候语；`setName()` 方法用于设置对象的名字，`getName()` 方法用于获取对象的名字。

类的定义一般包括四个部分：

1. 类的修饰符：用来限定类的访问权限、是否可以被继承等。
2. 类的名称：类名必须遵循驼峰命名法，即第一个单词小写，后续单词首字母大写。
3. 类的类型参数：泛型参数声明，用来指定类的类型。
4. 类的方法：类中定义的各种方法，称为方法。

## 三、什么是对象？
对象是类的实例化结果。当我们创建一个类时，实际上是在告诉编译器创建一个空白的蓝图。只有根据这个蓝图去创建对象，才真正产生内存空间存储数据、行为。

例如：

```java
Person person = new Person();
person.setName("Alice");
System.out.println(person.getName());   // Alice
person.sayHello();                      // Hello!
```

这段代码定义了一个 Person 对象，并调用它的 setName() 方法设置了它的姓名，然后调用 sayHello() 方法打印了一句问候语。

在这里，"new Person()" 创建了一个 Person 对象。`new` 操作符是一个用于创建对象的运算符，它首先找到指定的类，然后根据类中的构造方法对其进行初始化，最后生成一个指向该对象的引用。

对象有三个关键属性：

1. 数据：对象的状态信息，保存在变量中。
2. 方法：对象的行为，保存在方法中。
3. 类型：对象的身份，确定了该对象属于哪个类。

## 四、什么是属性？
属性是指对象的内部状态信息，包括字段（Field）和变量（Variable）。

字段是在类中声明的变量，通常位于类的开头，如 `private int count;`。类中的所有的对象共享相同的字段值。

变量是在类的成员函数中声明的本地变量，只属于当前对象的状态，生命周期只在当前执行路径中有效，离开作用域则自动销毁。如 `int index;`。

## 五、什么是方法？
方法是类的行为。它在类的内部定义，具有特殊的名称、参数列表、返回值、异常处理块等语法特征。

在 Java 中，我们可以使用 `public`, `private`, `protected`, `static`, `final`, `abstract`, `synchronized`, `native`, `transient` 等关键字修饰方法，如：

```java
public static void main(String[] args) {}    // 主函数
public boolean equals(Object obj) {}          // 对象比较
public void test() throws Exception {}        // 测试函数
```

在类中，可以通过 `this` 关键字调用自己内部的属性和方法，如：

```java
public void setAge(int age) {
    if (age < 0 || age > 150) {
        throw new IllegalArgumentException("年龄不能超过150岁！");
    } else {
        this.age = age;
    }
}
```

`setAge()` 方法用来设置对象的年龄，如果年龄小于零或者大于 150，就抛出一个异常。

## 六、什么是构造方法？
构造方法是类的特殊方法，在类被创建时，系统自动调用该方法，用来初始化对象的数据。

Java 中，如果没有显示定义构造方法，系统默认提供一个默认的构造方法，此方法可以不传任何参数，也不需要返回值，但需要确保至少有一个非默认的构造方法，否则编译器无法正确调用该构造方法。

```java
class Car {
    String color;
    String brand;
    
    public Car() {      // 默认构造方法
        color = "red"; 
        brand = "BMW"; 
    }
    
    public Car(String c, String b) {       // 参数构造方法
        color = c; 
        brand = b; 
    }
}
```

Car 类中有两个构造方法，一个是默认构造方法，另一个是带参数的构造方法。

## 七、什么是封装？
封装是指把数据和行为包装在一起，形成一个不可改变的整体，不允许外部直接访问内部的变量和方法，只能通过公共接口访问。通过隐藏数据的访问细节，可以提高代码的安全性和可靠性。

对于 Java 语言来说，访问控制权限有四种级别：

1. default: 包内可见性，即同一个包内可访问，不同包不可访问。
2. protected: 受保护的可见性，即子类可访问，同一包和不同包均可访问。
3. public: 全球可见性，任意位置均可访问。
4. private: 只能本类访问。

## 八、什么是继承？
继承是指从已有的类得到功能上的新增，或者是重新定义已有的类。子类获得父类的所有属性和方法，并且还可以继续添加自己的属性和方法。继承是面向对象编程的重要特点之一。

在 Java 语言中，继承可以使用 `extends` 关键字来实现，如：

```java
class Animal {                    // 父类
    protected String sound;
    
    public void eat() {
        System.out.println("吃东西...");
    }
}

class Dog extends Animal {           // 子类
    public void bark() {
        System.out.println("汪汪叫...");
    }
}

Animal animal = new Dog();              // 实例化子类对象
animal.eat();                            // 调用父类方法
((Dog)animal).bark();                   // 转型调用子类方法
```

在这个例子中，Animal 是一个父类，包含一个 `sound` 属性和一个 `eat()` 方法。Dog 是 Animal 的子类，继承了父类的 `sound` 属性，并添加了自己的 `bark()` 方法。Dog 对象实例化之后，可以通过 `Animal` 引用调用 `eat()` 方法，也可以通过 `(Dog)` 转型操作符转换为 `Dog` 引用调用 `bark()` 方法。

注意：子类对象只能调用父类的方法，不能调用子类独有的属性和方法。

## 九、什么是多态？
多态是指在不同情况下，使用不同的方式来处理对象。在 OOP 思想中，多态意味着程序中的对象具有不同的行为。

在 Java 语言中，多态可以由方法覆盖和方法重载来实现。方法重载就是在一个类中可以有多个同名方法，它们的参数个数或者参数类型不同。方法覆盖就是子类可以重新定义父类中的方法，且不会破坏父类的其他方法。

```java
public class Main {
  public static void main(String[] args) {
      Animal dog = new Dog();
      dog.run();             // 输出“跑……”
      
      Animal cat = new Cat();
      cat.run();             // 输出“走……”
  }
}

class Animal {
  public void run() {
    System.out.println("跑……");
  }
}

class Dog extends Animal {
  @Override
  public void run() {
    System.out.println("跑起来！");
  }
}

class Cat extends Animal {
  @Override
  public void run() {
    System.out.println("走起来！");
  }
}
```

在这个例子中，我们定义了一个 Animal 类和三个子类：Dog、Cat 和 Rabbit。所有动物都具有 `run()` 方法，但是子类 `Dog` 和 `Cat` 的 `run()` 方法有所不同，这就是多态的实现。

## 十、什么是抽象？
抽象是面向对象编程的重要特征之一。抽象表示的是对象的一般性质而不是某些具体的实现细节。抽象可以帮助我们将对象划分成较为简单的、独立的单元，从而更好地理解和设计系统。

抽象可以分为两种类型：

1. 接口抽象：即抽象类和接口。
2. 实现抽象：即抽象类和抽象方法。

接口抽象可以定义一组方法签名，这些方法签名描述了类的外部行为，但不描述其具体实现。抽象类可以有构造方法、抽象方法、普通方法。抽象方法需要被子类实现。

```java
interface Shape {
    double area();
}

abstract class AbstractShape implements Shape {
    abstract double perimeter();

    @Override
    public final double area(){
        return Math.PI * this.perimeter();
    }
}

class Circle extends AbstractShape {
    private double radius;

    public Circle(double r){
        radius = r;
    }

    @Override
    double perimeter(){
        return 2 * Math.PI * radius;
    }
}

Circle circle = new Circle(5);
System.out.println("半径：" + circle.radius);            // 输出：半径：5
System.out.println("周长：" + circle.perimeter());         // 输出：周长：31.41592653589793
System.out.println("面积：" + circle.area());               // 输出：面积：78.53981633974483
```

在这个例子中，我们定义了一个 Shape 接口和一个 AbstractShape 抽象类。Shape 接口定义了一个 `area()` 方法，AbstractShape 类实现了 Shape 接口，并且还有两个抽象方法：`perimeter()` 方法和 `area()` 方法。Circle 类继承 AbstractShape 类并实现 `perimeter()` 方法，并重写了 `area()` 方法，实现了 Shape 接口。

Circle 类有一个构造方法和两个方法，分别是圆的半径和圆周长、面积。我们实例化了一个 Circle 对象，并调用了三个方法，输出了相应的值。

## 十一、什么是注解？
注解是 Java 5.0 引入的新特性，它可以帮助我们在源代码中嵌入元数据。元数据就是数据关于数据的数据，比如一张图片的尺寸、照片拍摄的时间、照片的作者等。元数据可以被程序使用，也可以被编译器或工具使用。

注解可以用于在编译阶段对代码做检查、优化、生成代码、管理依赖关系、甚至可以部署到生产环境。

Java 定义了三种类型的注解：

1. 注解类型注解：即在注解中嵌入注解类型。
2. 元注解：指注解本身也是一个注解，这些注解不是用于代码，而只是用于其他注解。
3. 元素注解：在注解中嵌入数据。

常用的注解类型注解有：@Override、@Deprecated、@SuppressWarnings。

```java
@Override
public void eat(){}

@Deprecated
public void play(){}

@SuppressWarnings("unchecked")
public void test(){
    List list = new ArrayList<>();
    list.add("hello");
    processList(list);
}

void processList(@SuppressWarnings("unused") List<?> l){}
```

在这个例子中，我们定义了三个注解类型注解：@Override、@Deprecated 和 @SuppressWarnings。第一个注解用于修饰方法，第二个注解用于修饰类和方法，第三个注解用于忽略警告信息。

注解类型注解的目的是为了能够在阅读代码的时候对其更加直观易懂，并减少错误的发生。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## 一、继承
### 1.继承概念
继承是面向对象编程中最重要的特征之一。继承是指从已有类得到功能上的新增，或者是重新定义已有的类。子类获得父类的所有属性和方法，并且还可以继续添加自己的属性和方法。继承是 Object 类及其子类的基础。

### 2.继承的优点
- 提高代码的复用率：继承可以实现代码的重复利用。通过继承的方式，可以方便地创建新的类，使得系统架构变得更加灵活、健壮。
- 提高代码的可维护性：继承提供了一种多态的方式，使得代码具有良好的可维护性。子类可以对父类的方法进行重写，实现自己的功能。
- 降低了类的耦合度：继承使得类之间产生了一种关联性，降低了类的耦合度。

### 3.继承的缺点
- 继承过度使用容易导致类结构臃肿难以维护：继承过度使用可能会导致类的结构过于复杂，使得代码难以维护。
- 子类与父类的兼容性较差：继承的特点是有利于代码的复用性和继承，但是过度的继承可能会造成类之间的兼容性问题。如果父类改变了方法的实现，子类的方法调用就会出现问题。
- 多继承：继承只支持单继承，但 Java 支持多继承，而且可以继承多个父类。

### 4.继承的作用
- 子类获取父类的所有属性和方法：子类继承了父类的所有属性和方法，它可以在不需要父类代码的基础上增加新的属性和方法，提高了代码的复用性。
- 子类可以重写父类的方法：子类可以对父类的方法进行重写，以便实现自己的功能。
- 通过组合关系与父类建立联系：通过组合关系可以让子类与父类建立关联关系，扩展父类的功能。

### 5.如何实现继承
使用关键字 `extend` 实现继承，语法如下：

```java
public class ChildClass extends ParentClass{
    // TODO...
}
```

其中，ChildClass 是子类，ParentClass 是父类。

## 二、组合
### 1.组合概念
组合是面向对象编程中另一种重要特征。组合就是使用组合关系来组合对象，也就是说，一个对象可以包含另外一个对象，并且通过这个对象来访问其中的属性和方法。组合可以帮助我们将对象划分成较为复杂的结构，并建立起对象之间的层次关系。

### 2.组合的优点
- 更灵活的结构：通过组合关系，可以构造出比较复杂的对象结构，解决复杂系统的设计问题。
- 更加符合业务逻辑：通过组合关系，可以实现对象的功能。

### 3.组合的缺点
- 不同类的对象耦合度高：组合的特点是建立起关联关系，但是不同类的对象耦合度较高。如果两个类的属性或方法一致，则会造成耦合。

### 4.组合的作用
- 简化对象间的通信：通过组合关系，可以让不同类的对象直接访问彼此的属性和方法，降低了代码的耦合度。
- 允许对象有多个所有者：对象可以通过多个组合关系拥有所有权，实现对象共享，降低了资源的占用。

### 5.如何实现组合
- 使用组合关系建立类之间的关系：创建一个类，其属性或方法都是另一个类的对象。
- 使用成员变量或成员变量数组：将某个类的对象作为属性或成员变量，在成员方法中通过对象的引用访问其属性和方法。

## 三、多态
### 1.多态概念
多态是面向对象编程的重要特性之一，它允许不同的对象对同一消息作出不同的响应。多态的实现方式包括方法重载、方法覆盖、对象替换。

### 2.方法重载
方法重载是指在一个类中可以有多个同名方法，它们的参数个数或者参数类型不同。方法重载的主要原因是为了实现同名方法的多态性。

方法重载的语法如下：

```java
// 方法重载的一般语法
[修饰符] [返回类型] 方法名([参数列表]) {[语句];}

// 方法重载的规则
- 方法名必须相同。
- 参数个数必须不同。
- 参数类型必须不同。

// 实例
public class TestOverload {
   // 方法重载
   public int add(int a, int b) {
      return a+b;
   }

   public float add(float a, float b) {
      return a+b;
   }

   public static void main(String[] args) {
      TestOverload to=new TestOverload();

      // 方法重载调用
      System.out.println(to.add(10,20));   // 输出：30
      System.out.println(to.add(10.5f,20.5f)); // 输出：31.0
   }
}
```

在以上实例中，TestOverload 类提供了两个 add() 方法，它们的签名不同。main() 函数调用这两个 add() 方法，通过方法重载，可以选择对应的方法来执行。

### 3.方法覆盖
方法覆盖（override）是指子类重新定义父类的方法。子类重新定义父类的某个方法，就成为覆盖（override）该方法。方法覆盖的目的是使得子类的方法能够与其父类的方法功能保持一致，但是又能满足子类的特殊需求。

方法覆盖的语法如下：

```java
// 方法覆盖的一般语法
[修饰符] [返回类型] 方法名([参数列表]) {[语句];}

// 方法覆盖的规则
- 方法名、参数列表、返回类型必须相同。
- 子类中不能缩小父类的访问权限。
- 子类不能抛出比父类更多的异常类型。

// 实例
public class SuperClass {
    public int methodA(){
        return 0;
    }
}

public class SubClass extends SuperClass {
    @Override
    public int methodA(){
        return 1;
    }

    public int methodB(){
        return super.methodA()+1;
    }
}

public class Main {
    public static void main(String[] args) {
        SubClass subObj=new SubClass();

        // 方法覆盖调用
        System.out.println(subObj.methodA());    // 输出：1
        System.out.println(subObj.methodB());    // 输出：2
    }
}
```

在以上实例中，SuperClass 类提供了 methodA() 方法，SubClass 类继承 SuperClass 类并重写了 methodA() 方法，同时还提供了 methodB() 方法。Main 类实例化 SubClass 类并调用 methodA() 和 methodB() 方法。

### 4.对象替换
对象替换（substitution）是指用子类对象替换父类对象。当子类对象替代父类对象时，调用子类对象的方法会优先于调用父类对象的方法。对象替换的目的是增强代码的灵活性和可变性。

对象替换的语法如下：

```java
// 对象替换的一般语法
ParentClass parentObj=(ParentClass)childObj;
```

对象替换的优点是灵活性，可以适应变化，例如可以按需替换父类对象，在运行时替换对象。但是，对象替换容易出现问题，如类型转换异常、失去父类的方法实现等。

### 5.总结
多态就是使用父类的对象引用访问子类对象的方法。方法重载、方法覆盖、对象替换都是多态的重要实现方式。

# 4.具体代码实例和详细解释说明
## 一、类和对象
### 1.类定义
#### 语法格式
```java
access_modifier class ClassName{
  // 成员变量
  variable_type variable_name;

  // 构造方法
  public ClassName(){
     // TODO...
  }
  
  // 方法
  access_modifier return_type method_name([parameter_list]){
     // 方法体
  }
}
```

#### 示例
```java
public class Rectangle {
   private double length;
   private double width;
   
   public Rectangle(double len, double wid){
      length = len;
      width = wid;
   }
   
   public double getLength(){
      return length;
   }
   
   public void setLength(double len){
      length = len;
   }
   
   public double getWidth(){
      return width;
   }
   
   public void setWidth(double wid){
      width = wid;
   }
   
   public double area(){
      return length*width;
   }
   
   public double perimeter(){
      return 2*(length+width);
   }
}
```

### 2.对象创建
#### 语法格式
```java
className objectReference = new className(argument_list);
```

#### 示例
```java
Rectangle rect = new Rectangle(10, 20);
```

### 3.成员变量
成员变量（instance variables）是在类中声明的变量，通常位于类的开头，例如：

```java
private int count;
private String name;
```

其中，count 和 name 为成员变量，属于类的状态变量，存储对象的属性。成员变量有四种访问权限：private、default、protected、public。private 代表仅在类内部可见，默认可见性；default 可省略，在同一个包内可见；protected 代表在同一个包内和其子类可见；public 代表全球可见。

### 4.构造方法
构造方法（constructor）是在创建对象时执行的代码块。一个类可以有多个构造方法，但只能有一个无参构造方法。

```java
public class Student {
    private int id;
    private String name;
    private int age;
    
    // 无参构造方法
    public Student() {
        
    }
    
    // 有参构造方法
    public Student(int id, String name, int age) {
        this.id = id;
        this.name = name;
        this.age = age;
    }
}
```

在上例中，Student 类有三个成员变量：id、name 和 age，无参构造方法为空，有参构造方法初始化了三个成员变量。

### 5.方法
方法（method）是在对象中能够完成的操作。方法有三种访问权限：private、default、protected。private 代表仅在类内部可见，默认可见性；default 可省略，在同一个包内可见；protected 代表在同一个包内和其子类可见。

```java
public class Calculator {
    private double result;
    
    // 返回计算结果
    public double calculate(double num1, char operator, double num2) {
        
        switch (operator) {
            case '+':
                result = num1 + num2;
                break;
            case '-':
                result = num1 - num2;
                break;
            case '*':
                result = num1 * num2;
                break;
            case '/':
                result = num1 / num2;
                break;
            default:
                System.out.println("Invalid Operator.");
                break;
        }
        
        return result;
    }
}
```

在上例中，Calculator 类有一个成员变量 result，calculate() 方法接受三个参数：num1、operator、num2。switch 语句根据 operator 执行对应的算术运算并保存结果到 result 变量。

### 6.对象变量
对象变量（object reference variable）是指一个变量，它包含对其他对象的引用。

```java
Employee empRef = null;
empRef = new Employee("John Doe", 25, "Sales");
```

在上例中，empRef 是一个 Employee 对象的引用变量。empRef 指向 Employee 对象，即 empRef 引用了某一个 Employee 对象。

## 二、继承
### 1.定义继承
```java
class superclass{
  // TODO..
}

class subclass extends superclass{
  // TODO...
}
```

### 2.构造方法和方法重写
```java
class Animal{
   public Animal(){
      // TODO...
   }
   
   public void move(){
      // TODO...
   }
}

class Dog extends Animal{
   public Dog(){
      // TODO...
   }
   
   @Override 
   public void move(){
      // 覆写父类的方法，使其适合狗的移动方式
      System.out.println("狗狗正在跑...");
   }
}

public class Main {
   public static void main(String[] args) {
      Dog mydog = new Dog();
      mydog.move(); // 输出：狗狗正在跑...
   }
}
```

在上面示例中，Animal 是父类，Dog 是子类，Dog 继承了父类的 move() 方法，并重写了父类的方法。当创建 Dog 对象并调用 move() 方法时，会输出“狗狗正在跑...”。

### 3.访问控制权限
访问控制权限（Access Control）规定了类、方法、变量、构造方法的可见性。Java 提供了四种访问控制权限：private、default、protected、public。

- private：私有权限，仅在本类中可见。
- default（package-private）：包内可见，默认可见性，也可以省略。
- protected：受保护的可见性，子类可见，同一包内可见。
- public：全球可见，任何地方都可见。

访问控制权限可以使用修饰符来实现：

| Modifier | Access Level                |
| -------- | --------------------------- |
| private  | Only within the class        |
| defalut  | In the package and subclasses |
| protected| In the package and subclasses |
| public   | Everywhere                  |

### 4.多继承
Java 支持多继承，可以在一个类中定义多个父类，例如：

```java
class A{}
class B{}
class C extends A,B{}
```

在上例中，类 C 继承了类 A 和类 B，因此，C 既有类 A 的属性和方法，也有类 B 的属性和方法。

多继承使用逗号 `,` 分隔，子类只能继承一个父类，即只能有一个父类的构造方法。

### 5.避免多继承带来的问题
- 混乱的继承关系：多继承会使类的继承关系混乱，会造成代码的复杂性。
- 深度继承链：如果类的继承层级太深，Java 会报错。

因此，在设计类时，应该尽量保持单一继承关系。

## 三、组合
### 1.定义组合
```java
class employee{
  // TODO...
}

class department{
  // TODO...
  
  employee e; // 创建对象
}
```

### 2.使用组合
```java
department dept = new department();
dept.e = new employee(); // 设置属性
dept.printDetails(); // 调用方法
```

### 3.局限性
- 运行速度慢：每个对象都会分配一个内存空间，组合的方式会使内存使用增加。
- 耦合度高：两个对象之间耦合度高，如果其中一个对象改变，影响范围广。
- 不支持多态：组合无法使用多态。