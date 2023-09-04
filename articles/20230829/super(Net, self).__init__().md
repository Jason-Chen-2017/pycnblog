
作者：禅与计算机程序设计艺术                    

# 1.简介
  

python中类的继承有两种方式:第一种是通过调用父类的方法，第二种是直接用父类作为子类的基类。
但直接用父类作为子类的基类存在一个潜在的风险——隐式地调用了父类构造函数，而由于父类构造函数可能需要一些参数，用户并没有真正传入这些参数，这会导致子类对象的初始化失败。

为了解决这个问题，python引入了一个特殊方法__init__(),该方法可以用来控制对象创建时的行为，也就是控制何时、如何初始化对象。在子类中定义自己的__init__方法，可以避免隐式调用父类的构造函数，同时也可以对父类构造函数的参数进行定制化处理。

但是，一般情况下，我们习惯性地将父类的__init__方法作为子类构造器的第一个语句，使得子类构造器无法实现灵活的初始化过程，而且子类构造器逻辑较复杂，阅读起来不方便。因此，为了解决这个问题，python又引入了一个内置函数super()，它可以用来调用父类的方法，即可以将父类的构造函数逻辑转移到子类构造器，并完成相应的初始化工作。

本文介绍一下python中的super()用法，以及super()和__init__()的关系。
# 2.基础概念及术语
## 2.1 继承
继承是面向对象编程的一个重要特征，其基本思想就是子类拥有父类的所有属性和方法（包括私有方法），并且可以扩展自己独有的属性和方法。继承的语法形式如下：

class ChildClass(ParentClass):
    # child class 的属性和方法
## 2.2 方法重写
如果在父类中定义了一个与子类同名的方法，那么子类就可以重写这个方法，重新定义它的功能，这样就拥有了自己特定的功能。比如父类有一个方法叫做display()，子类可以重写这个方法，定义自己的显示效果。

```python
class ParentClass:

    def display(self):
        print("This is the parent class method.")


class ChildClass(ParentClass):
    
    def display(self):
        print("This is the child class method and it will override the parents' method.")
```

当执行ChildClass().display()的时候，打印结果为“This is the child class method and it will override the parents' method.”
## 2.3 super()
Python中有两个内置函数，一个是type()用来获取一个变量的类型，另一个是isinstance()用来判断一个变量是否是某种类型的对象。super()是一个很有用的函数，用于调用父类的属性和方法。super()函数返回的是父类的代理对象，用于调用父类的方法。

```python
class Animal:
    
    def __init__(self, name):
        self.name = name
        
    def eat(self):
        print("%s is eating..." % self.name)
        

class Dog(Animal):
    
    def __init__(self, name):
        super().__init__(name)
        
    def bark(self):
        print("%s is barking." % self.name)
        
    
d = Dog('Buddy')
print(isinstance(d, Animal))    # True
print(issubclass(Dog, Animal))   # True
d.eat()                         # Buddy is eating...
d.bark()                        # Buddy is barking.
```

Dog类继承自Animal类，Dog类重写了父类的eat()方法。调用Dog类的eat()方法的时候，实际上调用的是Dog类重写后的方法，而不是父类的eat()方法。所以，super()函数能够帮助我们调用父类的eat()方法，以此达到继承的目的。

使用super()后，子类无需重复编写父类相同的方法，只需要简单调用super()即可。这对代码的复用、可读性和可维护性都有很大的帮助。

## 2.4 类方法
类方法是一种特殊的实例方法，它允许修改类的状态或行为，不依赖于实例对象。类方法通常用classmethod关键字定义。

举个例子：

```python
class Person:
    
    def __init__(self, name):
        self.name = name
        
    @classmethod
    def create_person(cls, name):
        return cls(name)
        
    def greet(self):
        print("Hello, my name is {}.".format(self.name))
        
        
p = Person.create_person('Alice')
p.greet()     # Hello, my name is Alice.
```

Person类有一个类方法create_person()，它创建一个Person类的实例对象，并返回这个实例对象。调用create_person()方法不需要实例化Person类，而是通过类名称来调用。

注意，@classmethod注解只能修饰一个普通的类方法，不能修饰静态方法staticmethod。因为静态方法是没有实例的，不能访问类的任何属性或方法。

## 2.5 对象
在面向对象编程中，对象指的是具有状态和行为的数据结构。每一个对象都包含数据（状态）和操作（行为）。对象的状态表示对象的当前信息，而对象的行为则表现为其可以接受的消息。

在Python中，每个对象都是实例化自某个特定类的对象。换句话说，如果某个对象属于某个类，那么它也必然是一个实例。