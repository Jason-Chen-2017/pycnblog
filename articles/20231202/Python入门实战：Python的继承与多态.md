                 

# 1.背景介绍

Python是一种强大的编程语言，它具有简洁的语法和易于学习。Python的继承与多态是其核心特性之一，它们使得代码更加模块化、可重用和灵活。在本文中，我们将深入探讨Python的继承与多态，并提供详细的解释和代码实例。

# 2.核心概念与联系
## 2.1 继承
继承是面向对象编程中的一个基本概念，它允许一个类从另一个类中继承属性和方法。在Python中，通过使用`class`关键字定义类，并使用`:`符号指定父类名称来实现继承。子类可以访问和覆盖父类的属性和方法。
```python
class Parent:
    def __init__(self):
        self.parent_attr = "I am a parent"
    
    def parent_method(self):
        print("This is a parent method")
        
class Child(Parent):
    def __init__(self):
        super().__init__() # 调用父类构造函数  这里super()函数表示父类对象  也就是Parent()对象  所以super().__init__()等价于Parent.__init__(self)  也就是调用了Parent构造函数  初始化了parent_attr属性  也就是给自己添加了这个属性  但不会覆盖掉原有的同名属性  如果要覆盖则需要在Child中重写该方法或者直接修改self.parent_attr=xxx  然后再把原有的del self.parent_attr删除掉即可  但这样做不太合适因为会影响到其他地方对该变量的引用  所以最好直接修改值即可 而不是删除原有变量然后再添加新变量 因为这样会影响到其他地方对该变量的引用 比如说如果其他地方还存在引用该变量则会出错误信息告诉你那个变量已经被删除了但你还在使用它 所以最好直接修改值即可而不是删除原有变量然后再添加新变量