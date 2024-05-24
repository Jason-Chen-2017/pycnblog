                 

# 1.背景介绍


在Python编程语言中，字典（dictionary）是一种容器类型的数据结构，它存储一个映射关系。比如，键-值对形式的字典可以用来存储用户信息、商品名称及其对应的价格等数据。而类（class）则是面向对象编程中非常重要的组成部分，能够提供更高级的抽象和代码重用机制。在本文中，我们将一起学习Python中的字典和类，探讨它们之间的一些联系和区别。

# 2.核心概念与联系
## 2.1 字典（Dictionary）
字典是另一种容器数据类型，它是一个键-值对集合，其中键必须是不可变对象，比如数字、字符串或者元组；值可以是任何对象。字典的特点是无序且可变的，这意味着字典内元素的顺序可能与插入时的顺序不同。字典可以使用键来访问对应的值，而不必通过索引下标来确定位置。
### 字典的定义语法如下：

```python
my_dict = {key1: value1, key2: value2,..., keyn: valuen}
```

其中，`{ }` 表示一个字典的开始和结束，`,` 分割不同的键值对，`:` 表示键和值的分隔符，`keyi` 和 `valuei` 是字典的一个键值对，可以是任意非空对象。

以下示例创建了一个名为 “employee” 的字典，其中包含了三个键值对：

```python
>>> employee = {'name': 'John', 'age': 27, 'job': 'Software Engineer'}
```

键可以是任何不可变对象，如字符串、数字或元组。要获取字典中的某个值，只需要指定对应的键即可：

```python
>>> print(employee['name'])    # Output: John
>>> print(employee['age'])     # Output: 27
>>> print(employee['job'])     # Output: Software Engineer
```

如果指定的键不存在于字典中，会报错：

```python
>>> print(employee['salary'])   # KeyError:'salary'
```

还可以通过以下方式添加/修改字典中的键值对：

```python
>>> employee['salary'] = 50000
>>> print(employee)           # Output: {'name': 'John', 'age': 27, 'job': 'Software Engineer','salary': 50000}
```

也可以使用更新字典的方式一次性添加多个键值对：

```python
>>> new_data = {'city': 'New York', 'country': 'USA'}
>>> employee.update(new_data)
>>> print(employee)           # Output: {'name': 'John', 'age': 27, 'job': 'Software Engineer','salary': 50000, 'city': 'New York', 'country': 'USA'}
```

如果要删除字典中的某一项，可以使用 `del` 语句：

```python
>>> del employee['salary']
>>> print(employee)           # Output: {'name': 'John', 'age': 27, 'job': 'Software Engineer', 'city': 'New York', 'country': 'USA'}
```

也可以批量删除字典中的某些键值对，但要先将这些键存储在一个列表里：

```python
>>> keys_to_delete = ['name', 'job','salary']
>>> for k in keys_to_delete:
        del employee[k]
        
>>> print(employee)           # Output: {'age': 27, 'city': 'New York', 'country': 'USA'}
```

## 2.2 类（Class）
类（class）是面向对象编程（Object-Oriented Programming，简称 OOP）的基本构建模块。类提供了一种封装、继承和多态的方式，能让我们创建具有相同行为的对象集合。与其他编程语言相比，Python 中的类与其他语言中的类稍有不同，但是仍然是很重要的概念。
### 类的定义语法如下：

```python
class MyClass:
    <statement-1>
   .
   .
   .
    <statement-N>
```

其中 `<statement>` 可以是变量声明、方法定义或其他任何有效的 Python 代码。类定义后，可以使用关键字 `class`，实例化这个类并调用它的属性和方法：

```python
>>> class Employee:
        def __init__(self, name, age):
            self.name = name
            self.age = age
            
        def get_info(self):
            return "Name: {}, Age: {}".format(self.name, self.age)

>>> emp1 = Employee("John", 27)
>>> print(emp1.get_info())      # Output: Name: John, Age: 27
```

在上面的示例中，定义了一个名为 `Employee` 的类，并给出了 `__init__()` 方法，该方法用于初始化对象实例的状态。然后，创建一个新的 `Employee` 对象并调用它的 `get_info()` 方法。另外，类中还可以定义其他的方法，比如计算工资的 `calculate_salary()` 方法，这样就可以将计算工资的逻辑集中到 `Employee` 类中。此外，还可以在 `Employee` 类中定义属性（attribute），比如 `position` 或 `department`。

当创建类时，可以定义构造器（constructor）方法 `__init__()` ，该方法负责对象的初始化工作。类构造器通常接收参数，这些参数用来设置对象的初始状态。构造器在创建对象时自动调用，所以不需要显式地调用构造器。除了构造器之外，还可以定义其他类型的方法。

类之间可以相互继承，继承机制使得子类可以扩展父类的功能。子类可以重新定义父类的属性和方法，同时还可以增加新的属性和方法。继承可以使用关键字 `extends` 来实现：

```python
class Manager(Employee):
    
    def __init__(self, name, age, department):
        super().__init__(name, age)
        self.department = department
        
    def manage(self, employees):
        pass
```

在上述示例中，定义了一个名为 `Manager` 的子类，继承自 `Employee` 类，并新增了 `department` 属性和 `manage()` 方法。构造器 `__init__()` 使用 `super()` 函数调用父类的构造器，并初始化自己的属性 `department`。`manage()` 方法接收一个 `employees` 参数，代表要管理的员工列表，这里没有实际的实现，只是用作示范。

类是一种抽象的概念，因为它只能创建对象，不能直接执行代码。只有创建了对象之后，才可以调用它的属性和方法。因此，在实际开发过程中，我们经常需要结合继承和组合的方式来使用类。