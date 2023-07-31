
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1994年，<NAME>（图灵奖获得者）发表了著名论文“Design Patterns”的英文版。它将软件开发中经常用到的各种模式和原则总结成一套简单的规则。本系列的书籍围绕着“23种设计模式”出版，是学习面向对象编程、设计模式、并采用设计模式的必读之作。
         本篇文章主要关注23种设计模式中的20种。它们包括创建型模式、结构型模式、行为型模式和J2EE模式等五个方面的一些具体模式。
         
         **目录**

         * 第一章 Introduction
           * 1.1 概述
             * 1.1.1 为什么要学习设计模式？
             * 1.1.2 设计模式的定义及其特点
             * 1.1.3 设计模式分类
             * 1.1.4 为何要在Java中应用设计模式？
             * 1.1.5 抽象工厂模式
             * 1.1.6 单例模式
             * 1.1.7 策略模式
             * 1.1.8 装饰器模式
             * 1.1.9 模板方法模式
             * 1.1.10 命令模式
             * 1.1.11 迭代器模式
             * 1.1.12 观察者模式
             * 1.1.13 中介者模式
             * 1.1.14 状态模式
             * 1.1.15 适配器模式
             * 1.1.16 组合模式
             * 1.1.17 流水线模式
             * 1.1.18 访问者模式
             * 1.1.19 依赖注入模式
             * 1.1.20 最佳实践建议
             * 1.1.21 标准库中的设计模式
           * 1.2 设计模式六大原则
             * 1.2.1 Single Responsibility Principle (SRP)
               * 1.2.1.1 当一个类负责多个职责时，应该拆分出多个类
             * 1.2.2 Open-Closed Principle (OCP)
               * 1.2.2.1 对扩展开放，对修改封闭
             * 1.2.3 Dependency Inversion Principle (DIP)
               * 1.2.3.1 使用接口或抽象类，不要直接依赖于具体实现
             * 1.2.4 Interface Segregation Principle (ISP)
               * 1.2.4.1 不应强制用户去实现不需要的方法
             * 1.2.5 Liskov Substitution Principle (LSP)
               * 1.2.5.1 子类可以替换父类的功能
             * 1.2.6 Law of Demeter (LOD)
               * 1.2.6.1 只与朋友通信
           * 1.3 其他相关概念
             * 1.3.1 模式角色
             * 1.3.2 UML类图
             * 1.3.3 时序图
             * 1.3.4 类关系
             * 1.3.5 依赖关系
             * 1.3.6 方法关系
        * 第二章 创建型模式
          * 2.1 工厂模式
            * 2.1.1 概念
              * 2.1.1.1 简单工厂模式
                * 用一个静态工厂方法来代替new运算符，通过传入的参数，动态地决定创建哪一种产品类的实例；
              * 2.1.1.2 工厂方法模式
                * 在抽象的Creator类中声明了一个工厂方法，用于创建Product的实例；
                * 每当需要创建一个对象的时候，客户端只需要调用这个工厂方法即可；
              * 2.1.1.3 抽象工厂模式
                * 提供了一个创建一系列相关或者相互依赖对象的接口，而无需指定他们具体的类；
                * 通过多重继承或者其他手段，一个工厂可以生产出很多不同风格的产品；
              * 2.1.1.4 迪米特法则
                * 降低类之间的耦合性，建立弱耦合，提高系统的灵活性；
            * 2.1.2 具体案例
              * 简单工厂模式
                * 假设有一个产品类Bird，还有三个子类Duck、Eagle和Woodpecker，它们都需要被创建出来，通过一个静态方法（createBird()）来统一管理各个子类的创建过程。那么如何实现呢？

                ```java
                    public class BirdFactory {
                      public static Bird createBird(String name){
                        if("duck".equals(name)){
                          return new Duck();
                        }else if ("eagle".equals(name)){
                          return new Eagle();
                        } else{
                          return new Woodpecker();
                        }
                      }
                    }
                    
                    // client code
                    Bird duck = BirdFactory.createBird("duck");
                    System.out.println(duck instanceof Duck);   // true
                    Bird eagle = BirdFactory.createBird("eagle");
                    System.out.println(eagle instanceof Eagle); // true
                    Bird woodpecker = BirdFactory.createBird("woodpecker");
                    System.out.println(woodpecker instanceof Woodpecker);//true
                ```

              * 工厂方法模式
                * 以抽象类为基类，派生出几个具体类。然后，在该基类中提供一个工厂方法，每个具体子类返回一个具体的实例。

                ```java
                    //abstract factory pattern
                    
                    abstract class AnimalFactory{
                      public abstract Animal createAnimal();
                    }
                    
                    class DogFactory extends AnimalFactory{
                      @Override
                      public Animal createAnimal(){
                        return new Dog();
                      }
                    }
                    
                    class CatFactory extends AnimalFactory{
                      @Override
                      public Animal createAnimal(){
                        return new Cat();
                      }
                    }
                    
                    interface Animal{}
                    
                  	// concrete classes
                    
                    class Dog implements Animal{
                      public void eat(){
                        System.out.println("Dog is eating.");
                      }
                    }
                    
                    class Cat implements Animal{
                      public void meow(){
                        System.out.println("Cat is meowing.");
                      }
                    }
                    
                    // client code
                    
                    AnimalFactory dogFactory = new DogFactory();
                    Animal dog = dogFactory.createAnimal();
                    dog.eat();   // output: "Dog is eating."
                
                    AnimalFactory catFactory = new CatFactory();
                    Animal cat = catFactory.createAnimal();
                    cat.meow();   // output: "Cat is meowing."
                ```

              * 抽象工厂模式
                * 可以有多个抽象工厂，每个抽象工厂都负责创建一个体系相关的若干产品。

                ```java
                    //abstract factory pattern
                    
                    interface VehicleFactory{
                      Engine createEngine();
                      Transmission createTransmission();
                      Brakes createBrakes();
                    }
                    
                    interface Engine{}
                    
                    interface Transmission{}
                    
                    interface Brakes{}
                    
                    class CarVehicleFactory implements VehicleFactory{
                      @Override
                      public Engine createEngine(){
                        return new CarEngine();
                      }
                
                      @Override
                      public Transmission createTransmission(){
                        return new CarTransmission();
                      }
                
                      @Override
                      public Brakes createBrakes(){
                        return new CarBrakes();
                      }
                    }
                    
                    class PlaneVehicleFactory implements VehicleFactory{
                      @Override
                      public Engine createEngine(){
                        return new PlaneEngine();
                      }
                
                      @Override
                      public Transmission createTransmission(){
                        return new PlaneTransmission();
                      }
                
                      @Override
                      public Brakes createBrakes(){
                        return new PlaneBrakes();
                      }
                    }
                    
                    // concrete classes
                    
                    class CarEngine implements Engine{
                      public String getName(){
                        return "Car engine";
                      }
                    }
                    
                    class CarTransmission implements Transmission{
                      public int getSpeed(){
                        return 60;
                      }
                    }
                    
                    class CarBrakes implements Brakes{
                      public boolean hasABS(){
                        return false;
                      }
                    }
                
                    class PlaneEngine implements Engine{
                      public String getName(){
                        return "Plane engine";
                      }
                    }
                    
                    class PlaneTransmission implements Transmission{
                      public int getSpeed(){
                        return 120;
                      }
                    }
                    
                    class PlaneBrakes implements Brakes{
                      public boolean hasABS(){
                        return true;
                      }
                    }
                    
                    // client code
                
                    VehicleFactory carFactory = new CarVehicleFactory();
                    Engine carEngine = carFactory.createEngine();
                    Transmission carTransmission = carFactory.createTransmission();
                    Brakes carBrakes = carFactory.createBrakes();
                
                    System.out.println("Name of the engine: "+carEngine.getName());
                    System.out.println("Speed of transmission: "+carTransmission.getSpeed());
                    System.out.println("Does it have ABS? "+carBrakes.hasABS());
                
                    VehicleFactory planeFactory = new PlaneVehicleFactory();
                    Engine planeEngine = planeFactory.createEngine();
                    Transmission planeTransmission = planeFactory.createTransmission();
                    Brakes planeBrakes = planeFactory.createBrakes();
                
                    System.out.println("Name of the engine: "+planeEngine.getName());
                    System.out.println("Speed of transmission: "+planeTransmission.getSpeed());
                    System.out.println("Does it have ABS? "+planeBrakes.hasABS());
                ```

              * 建造者模式
                * 根据要求一步步创建一个复杂对象，并允许按步骤逐个构建；

                ```java
                    //builder pattern
                    
                    class PersonBuilder{
                      private final Person person = new Person();
                      
                      public Person buildAge(int age){
                        person.setAge(age);
                        return this.person;
                      }
                      
                      public Person buildGender(String gender){
                        person.setGender(gender);
                        return this.person;
                      }
                      
                      public Person buildHeight(float height){
                        person.setHeight(height);
                        return this.person;
                      }
                      
                      public Person buildWeight(double weight){
                        person.setWeight(weight);
                        return this.person;
                      }
                      
                      public Person buildName(String name){
                        person.setName(name);
                        return this.person;
                      }
                    }
                    
                    class Person{
                      private int age;
                      private float height;
                      private double weight;
                      private String name;
                      private String gender;
                      
                      public int getAge(){return age;}
                      public void setAge(int age){this.age=age;}
                      
                      public float getHeight(){return height;}
                      public void setHeight(float height){this.height=height;}
                      
                      public double getWeight(){return weight;}
                      public void setWeight(double weight){this.weight=weight;}
                      
                      public String getName(){return name;}
                      public void setName(String name){this.name=name;}
                      
                      public String getGender(){return gender;}
                      public void setGender(String gender){this.gender=gender;}
                    }
                    
                    // client code
                
                    PersonBuilder builder = new PersonBuilder();
                    Person john = builder
                             .buildName("John")
                             .buildGender("Male")
                             .buildHeight(1.8f)
                             .buildWeight(80.0)
                             .buildAge(25)
                             .build();
                
                    System.out.println("Name:"+john.getName());
                    System.out.println("Gender:"+john.getGender());
                    System.out.println("Age:"+john.getAge());
                    System.out.println("Height:"+john.getHeight());
                    System.out.println("Weight:"+john.getWeight());
                ```

          * 2.2 抽象工厂模式
            * 构造多个抽象工厂，每一个工厂负责产生独立的对象。
          * 2.3 生成器模式
            * 将一个复杂对象的构建与它的表示分离，使得同样的构建过程可以创建不同的表示；
          * 2.4 原型模式
            * 用原型实例指定创建对象的种类，并且通过拷贝这些原型创建新的对象，不必重新执行创建对象的过程；
          * 2.5 对象池模式
            * 避免频繁创建销毁同类型的对象，可以维护一个可用的对象集合，请求一个对象时，如果集合为空，就新建一个，否则就从集合中取出一个对象使用。
        * 第三章 结构型模式
          * 3.1 适配器模式
            * 将一个类的接口转换成客户希望的另一个接口，使得原本由于接口不兼容而不能一起工作的两个类可以协同工作；
          * 3.2 桥接模式
            * 将抽象部分与它的实现部分分离，使它们都可以独立地变化；
          * 3.3 组合模式
            * 允许将对象组合成树形结构来表现“整体/部分”层次结构；
          * 3.4 装饰器模式
            * 是一种动态地给对象增加额外职责的方式，通过包装一个对象，在运行期间动态地添加一些额外的职责；
          * 3.5 外观模式
            * 是一个用来减少系统的复杂度，降低其耦合度，并更容易使用的设计模式；
          * 3.6 享元模式
            * 在一个应用程序中，相同或相似的对象可以使用相同的资源，这种模式称为对象池；
          * 3.7 代理模式
            * 为其他对象提供一种代理以控制对这个对象的访问；
          * 3.8 单例模式
            * 保证一个类仅有一个实例，并提供一个全局访问点；
        * 第四章 行为型模式
          * 4.1 责任链模式
            * 请求在一条链上传递，直到某一节点处理它为止；
          * 4.2 命令模式
            * 将一个请求封装为一个对象，使发出请求的对象和知道怎么执行一个请求的对象松耦合；
          * 4.3 解释器模式
            * 给定一个语言，定义它的文法表示，并定义一个解释器，这个解释器使用该标识来解释语言中的句子；
          * 4.4 迭代器模式
            * 提供一种方法顺序访问一个聚合对象中的各个元素，而又不暴露其内部的表示；
          * 4.5 中介者模式
            * 用一个中介对象来封装一系列的对象交互，中介者使各对象不需要显式地相互引用，从而使其耦合松散，而且可以独立地改变它们之间的交互；
          * 4.6 备忘录模式
            * 在不破坏封装性的前提下，捕获一个对象的当前状态，并在该对象之外保存这个状态；
          * 4.7 观察者模式
            * 定义对象之间的一对多依赖，当某个对象的状态发生变化时，所有依赖于它的观察者都会得到通知并自动更新自己；
          * 4.8 状态模式
            * 允许对象在其内部状态改变时改变它的行为，对象看起来好像修改了它的类；
          * 4.9 策略模式
            * 分离算法的选择，让算法可以在不影响到客户端的情况下发生变化；
          * 4.10 模板方法模式
            * 将一个方法中的固定流程和变量进行抽象，然后定义一个虚函数来覆盖这些流程，不同的子类可以根据需要重写这些虚函数，从而实现不同的算法；
        * 第五章 J2EE模式
          * 5.1 MVC模式
            * Model-View-Controller模式，把模型（数据），视图（UI），控制器（业务逻辑）分开，可以有效降低耦合度，并增加复用性；
          * 5.2 委托模式
            * 遗留代码使用委托模式传递消息，将客户端与对象之间的交互封装到类中，有利于实现解耦；
          * 5.3 Facade模式
            * 为子系统中的复杂功能提供一个简单的接口，使客户端不必了解子系统的详细信息；
          * 5.4 Observer模式
            * 当一个对象改变状态时，所有依赖于它的观察者都会得到通知并自动更新自己；
          * 5.5 Strategy模式
            * 定义了一系列的算法，并将每个算法封装到具有共同接口的独立类中，使得它们可以相互替换；
          * 5.6 Decorator模式
            * 动态地给一个对象添加一些额外的职责；
          * 5.7 Template Method模式
            * 将一个方法中的固定流程和变量进行抽象，然后定义一个虚函数来覆盖这些流程，不同的子类可以根据需要重写这些虚函数，从而实现不同的算法；
          * 5.8 Mediator模式
            * 用一个中介对象来封装一系列的对象交互，中介者使各对象不需要显式地相互引用，从而使其耦合松散，而且可以独立地改变它们之间的交互；
        * 第六章 小结
          * 设计模式并不是银弹，适用于不同的场景和领域。
          * 有些设计模式虽然简单，但是却非常有效。
          * 需要熟悉各个设计模式背后的概念，理解它们的意义，能够运用它们解决实际的问题。

