                 

# 1.背景介绍


## 为什么要学习字典？
在数据处理、分析等领域，经常需要处理大量的数据。处理数据的过程中经常会用到字典(Dictionary)或者其他类似容器。Python中有一些模块提供了对字典的支持，比如collections模块中的OrderedDict和defaultdict等。本文将从以下三个方面进行介绍：
- OrderedDict: 有序字典的基本概念、应用场景及Python实现方式；
- defaultdict: 默认字典的基本概念、应用场景及Python实现方式；
- Counter: 计数器类的基本概念、应用场景及Python实现方式；
通过学习这些容器的知识，可以帮助我们更加方便地解决复杂的数据处理问题。
## 为什么要学习类？
在面向对象编程(Object-Oriented Programming, OOP)中，类是一种重要的概念。类是定义对象的蓝图，包括它的属性(attribute)，方法(method)，构造函数(constructor)。Python中有很多高级特性，比如装饰器(decorator), 多态(polymorphism)，抽象类(abstract class)等，都依赖于类的概念。本文将介绍类相关的知识。
## 如何选择合适的内容？
在学习Python中，最重要的一件事情就是掌握其各种模块的使用技巧。由于篇幅限制，本文不会做过多的介绍。建议读者自己去官方文档学习，例如学习Collections模块时，可以先阅读一下collections模块的文档。同时，也可以阅读一些其它优秀的教材，例如“Python数据结构和算法”一书中提到的相关内容。最后，也欢迎大家提交相关的问题或意见。



# 2.核心概念与联系
## 字典（Dictionary）
字典(Dictionary)是一个无序的键值对集合。其中，每个键都是独一无二的，对应一个值。字典的特点是快速访问元素，且允许修改字典的值。Python中的字典可以用{}来表示。
```python
>>> dict = {'name': 'Alice', 'age': 27}
```

### 字典的键（Key）
字典的键(key)必须是不可变类型，一般来说，数字类型、字符串类型、元组类型等都是不可变类型的键，而列表、字典、集合类型则不是。字典的键可以用任意可哈希的对象作为其键，如字符串，数字，元组。

对于重复的键，后面的键值对会覆盖前面的键值对。

示例：
```python
>>> d = {1: 'apple', 2: 'banana'} # key is integer type
>>> print(d[1])   # output: apple
>>> print(d[2])   # output: banana

>>> d = {'apple': 1, 'banana': 2} # key is string type
>>> print(d['apple'])    # output: 1
>>> print(d['banana'])   # output: 2

>>> t = (3, 4) 
>>> d[(t,)] = [5, 6]       # key is tuple with only one element
>>> print(d[(3, 4)])     # output: [5, 6]
```


### 字典的值的类型
字典的值(value)可以是任何类型。在创建字典的时候，如果不指定值，那么所有的键对应的值都默认为None。

示例：
```python
>>> fruit_price = {"apple": 2.5, "orange": 3.0, "pear": 2.75}
>>> print(fruit_price["apple"])      # Output: 2.5
>>> print(type(fruit_price))        # Output: <class 'dict'>
>>> print(fruit_price.keys())       # Output: dict_keys(['apple', 'orange', 'pear'])
>>> print(fruit_price.values())     # Output: dict_values([2.5, 3.0, 2.75])
```

### 字典的方法
字典提供了一些常用的方法，可以用来操作字典。常用的方法如下：

1. `len()`: 返回字典的长度。
2. `clear()`: 清空字典所有项。
3. `get()`: 根据键获取对应的值。
4. `items()`: 获取字典的所有项。
5. `keys()`: 获取字典的所有键。
6. `pop()`: 删除并返回字典指定键的值。
7. `popitem()`: 删除并返回最后一项。
8. `setdefault()`: 设置默认值，如果不存在该键则添加。
9. `update()`: 更新字典。

示例：
```python
>>> fruits = {"apple": 2.5, "orange": 3.0, "pear": 2.75}
>>> len(fruits)          # Output: 3
>>> fruits.clear()
>>> fruits               # Output: {}
>>> fruits.get("banana")  # Output: None
>>> fruits.get("banana", default="Not found.")   # Output: Not found.
>>> fruits.items()       # Output: dict_items([('apple', 2.5), ('orange', 3.0), ('pear', 2.75)])
>>> fruits.keys()        # Output: dict_keys(['apple', 'orange', 'pear'])
>>> fruits.pop("apple")  # Output: 2.5
>>> fruits               # Output: {'orange': 3.0, 'pear': 2.75}
>>> fruits.popitem()     # Output: ('pear', 2.75)
>>> fruits.setdefault("peach", 3.5)           # Output: 3.5
>>> fruits                                   # Output: {'orange': 3.0, 'peach': 3.5}
>>> fruits.update({"grape": 3.2}, cherry=2.5)  # Update multiple items at once
>>> fruits                                   # Output: {'orange': 3.0, 'peach': 3.5, 'grape': 3.2, 'cherry': 2.5}
```

### 字典的应用
字典有着丰富的应用场景，主要有以下几种：

1. 数据映射存储：用于保存和查找数据。字典可以通过键值来获取数据，而且键值不一定非得是整数。
2. 参数传递：通过字典参数传递的方式来向函数传递大量配置信息。
3. 数据聚集：将同一类的数据保存在一起，方便迭代。
4. 对象字典化：将类实例变量和对象属性保存到字典里。
5. 缓存：通过设置不同的字典键来缓存数据，降低计算成本。

## 类（Class）
类(Class) 是一种抽象概念，它描述了一个对象的性质，比如一个矩形、一个圆形等。在计算机程序设计中，类通常被用来创建新的自定义数据类型，以便更好地组织和管理代码。在Python中，类是用`class`关键字来声明的。

### 类的属性
类(Class) 的属性(Attribute) 是类的状态，它指的是类的外部特征，也就是说，类属性通常由类的创建者初始化，然后再保持不变。类的属性可以通过 `self` 来引用。

类属性定义语法：
```python
class Car():
    wheels = 4         # a class attribute

    def __init__(self):
        self.color = "red"  # an instance attribute
    
    def display(self):
        return f"This car has {Car.wheels} wheels and color {self.color}"
```

示例：
```python
car1 = Car()
print(car1.display())    # This car has 4 wheels and color red
print(Car.wheels)        # Accessing the class attribute outside of the class using ClassName.AttributeName
```

### 类的方法
类(Class) 的方法(Method) 是类的功能。它是行为的集合，并且可以包含状态和逻辑。方法可以接受输入参数(Input Parameter)，可以通过 `self` 关键字来引用当前对象。

类方法定义语法：
```python
class MathUtils():
    @staticmethod
    def add(a, b):
        return a + b
        
    @classmethod
    def subtract(cls, a, b):
        return cls(a - b)
    
MathUtils().add(3, 5)   # Using static method without object reference
MathUtils.subtract(3, 5)  # Using class method without creating an object
```

实例方法：
```python
class Animal():
    def eat(self):
        pass
    
class Dog(Animal):
    def bark(self):
        pass
    
dog1 = Dog()
dog1.eat()        # calling parent's method from child class
```

构造方法 `__init__()`：
```python
class Person():
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
person1 = Person("John Doe", 25)
print(person1.name)    # Output: John Doe
print(person1.age)     # Output: 25
```

### 类的继承
类(Class) 的继承(Inheritance) 是面向对象编程的一个重要概念。子类(Subclass) 可以继承父类的属性和方法，也可以重写父类的方法。Python 使用 `:` 来定义一个新类的基类，并在括号内指定多个基类。

子类定义语法：
```python
class ParentClass():
    def __init__(self):
        self.parent_attr = "Parent Attribute"
    
    def parent_method(self):
        return "Calling parent method."
    

class SubClass(ParentClass):
    def __init__(self):
        super().__init__()
        self.sub_attr = "Sub Attribute"
    
    def sub_method(self):
        return "Calling sub method."
    
    def parent_method(self):
        return "Overriding parent method."
```

示例：
```python
pc1 = ParentClass()
sc1 = SubClass()
print(pc1.parent_attr)      # Output: Parent Attribute
print(sc1.parent_attr)      # Output: Parent Attribute
print(sc1.parent_method())  # Output: Overriding parent method.
print(isinstance(sc1, ParentClass))   # True
```

### 抽象类（Abstract Class）
抽象类(Abstract Class) 是只包含抽象方法的类，即不能够创建对象的类。抽象类不能被实例化，只能被继承。抽象类让子类实现父类的抽象方法。抽象方法定义语法：

```python
from abc import ABC, abstractmethod

class AbstractBaseClass(ABC):

    @abstractmethod
    def foo(self, x, y):
        pass

    @abstractmethod
    def bar(self):
        pass
```

## 字典与类之间的关系
### 字典中的值可以是一个类对象
在Python中，字典的键值对可以是任何类型。因此，在字典中可以直接存放类对象。下面的示例中，字典中存放了两个类的对象。

```python
class Employee:
    def __init__(self, id, name, salary):
        self.id = id
        self.name = name
        self.salary = salary

    def __str__(self):
        return f"{self.id}: {self.name} (${self.salary})"


class Company:
    def __init__(self, employees=[]):
        self.employees = employees

    def add_employee(self, employee):
        self.employees.append(employee)

    def get_total_salaries(self):
        total_salaries = 0
        for emp in self.employees:
            total_salaries += emp.salary
        return total_salaries


emp1 = Employee(1, "Alice", 5000)
emp2 = Employee(2, "Bob", 6000)

company = Company()
company.add_employee(emp1)
company.add_employee(emp2)

for e in company.employees:
    print(e)

print(f"Total salaries: ${company.get_total_salaries()}")
```

输出结果：
```
1: Alice ($5000)
2: Bob ($6000)
Total salaries: $11000
```