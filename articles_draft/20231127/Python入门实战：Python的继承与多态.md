                 

# 1.背景介绍


## 一、继承
- 在面向对象编程中，继承（Inheritance）是指当一个类派生自另一个类时，它可以自动地获得另一个类的所有属性和方法，并可根据需要进行修改或扩展。继承主要用于代码重用，提高代码的可维护性和可读性。通过继承，子类将具有父类的全部功能，同时也拥有自己独特的属性。如此，子类就可以扩展父类的功能或添加新的属性。在继承中，父类称为基类（Base Class），子类称为派生类（Derived Class）。
- 在python中，继承的语法如下:

```python
class ParentClass(object):
    def __init__(self, attr1, attr2):
        self.attr1 = attr1
        self.attr2 = attr2
        
    def method1(self):
        pass
    
class DerivedClass(ParentClass):
    def __init__(self, attr3, *args, **kwargs):
        super().__init__(*args, **kwargs) # call parent constructor with remaining arguments
        self.attr3 = attr3
    
    def method2(self):
        pass
```

- 从上面的代码可以看到，子类`DerivedClass`首先继承了父类`ParentClass`，然后调用了父类的构造函数`__init__`。构造函数的参数还包括自己定义的`attr3`。这样，子类的对象就自动拥有了父类的所有属性和方法。

## 二、多态
- 多态（Polymorphism）是指相同的方法名或函数名可以作用于不同的类型，不同的对象对同一消息会作出不同的响应。在面向对象编程中，多态是指允许不同类型的对象对同一消息作出不同的响应，从而实现“软件工程”中的“灵活性”，让设计者更加专注于业务逻辑的处理。

- 在python中，可以通过多种方式实现多态。以下三个示例分别展示了类的单继承、多继承和多态。

### 1. 单继承
```python
class Shape:
    def draw(self):
        raise NotImplementedError('Subclass must implement abstract method')
        
class Circle(Shape):
    def draw(self):
        print("Drawing a circle")
        
class Rectangle(Shape):
    def draw(self):
        print("Drawing a rectangle")
        
circle = Circle()
rectangle = Rectangle()

shapes = [circle, rectangle]

for shape in shapes:
    shape.draw() # will output "Drawing a circle" and then "Drawing a rectangle" because both classes inherit from the same base class
```

- 在这个例子中，存在两个形状——圆形和矩形。这两个形状都继承自一个抽象类`Shape`，并且有自己的具体实现。这里我们使用了`raise NotImplementedError`方法抛出了一个错误，作为抽象方法，表示这个方法应该由子类来实现。如果没有子类实现该方法，则会产生一个运行时异常。

- `Circle`和`Rectangle`两个类分别实现了自己的`draw()`方法。因此，无论调用哪个对象的`draw()`方法，都会得到不同的输出结果。这是因为多态机制将调用委托给实际对象的实际方法，而非声明在基类中的抽象方法。

### 2. 多继承
```python
class Animal:
    def eat(self):
        print("Animal eating.")
        
class Mammal(Animal):
    def sleep(self):
        print("Mammal sleeping.")
        
class Bird(Animal):
    def fly(self):
        print("Bird flying.")

class Human(Mammal):
    def run(self):
        print("Human running.")
        
class Parrot(Bird):
    def sing(self):
        print("Parrot singing.")

animals = [Human(), Parrot()]

for animal in animals:
    animal.eat()   # Output: Animal eating.
    if isinstance(animal, Mammal):
        animal.sleep() # Output: Mammal sleeping.
    elif isinstance(animal, Bird):
        animal.fly()    # Output: Bird flying.
```

- 在这个例子中，我们创建了四个动物类——哺乳动物、鸟类、人类和鹦鹉。其中，人类和鹦鹉同时也是哺乳动物。由于人类和鹦鹉都是哺乳动物，所以它们也可以调用其父类`Mammal`的`sleep()`方法。与此同时，鹦鹉不是鸟类，所以它无法调用父类`Bird`的`fly()`方法。

- 通过多继承，我们可以组合多个类的功能，即使这些功能之间可能存在冲突。

### 3. 多态性质
- 在多态中，任何类都可以被视作它的父类，或者它的基类。例如，一个子类可以被视为任意其他类型的父类，或者任何其他类型的基类。换句话说，所有的子类都能像它们的父类一样调用方法。这种能力使得对象可以接收、处理和发送数据的方式变得非常灵活。

- 当我们调用一个对象的一个方法时，实际上是在调用对象对应的方法，而不是调用某个特定方法。也就是说，我们并不真正调用方法，而只是告诉编译器我们要调用哪个方法。这样，当我们调用父类的方法时，实际上是在调用子类的同名方法。

- 如果有一个父类带有一个方法，而一些子类都重载了这个方法，那么当我们调用父类的方法时，编译器就会选择哪个子类的方法来调用？

- 在面向对象编程中，只有类才能支持多态性质。只有在运行时才会发生多态性质的影响。也就是说，当我们运行代码的时候，实际上调用的是具体的对象的方法。在编译期间，我们并不能确定到底调用的是哪个子类的哪个方法。