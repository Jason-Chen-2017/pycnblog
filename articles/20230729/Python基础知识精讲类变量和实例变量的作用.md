
作者：禅与计算机程序设计艺术                    

# 1.简介
         
19世纪末期，在计算机刚刚开始被用到日常生活中，程序员们发现了一个奇怪的现象：在一个函数内部修改某个变量的值，会影响到该函数外的其他地方。直到十几年后，当我们再次学习编程时，才知道这个现象被称为“变量隐藏”(variable hiding)。
         
         在计算机科学中，变量分为两类：全局变量和局部变量。全局变量存储在内存的不同位置上，不同的程序可以访问同一块内存空间；而局部变量只存在于某些特定范围内，不能被其他程序访问。由于局部变量只存活在其特定的范围内，因此它们可以更好地保护程序数据安全。
         
         在Python中，类似于全局变量的变量叫做“类变量”，它可以被所有实例对象共享。而类似于局部变量的变量叫做“实例变量”，只能被各自对象的实例所拥有。
         
         本文将结合实例来详细讲解什么是类变量、实例变量以及如何使用它们。

       ## 2.基本概念术语说明
       
         ### 2.1 类变量
         “类变量”就是类的成员变量，这些变量属于类的所有实例，所有实例都可以使用该变量。在定义类的时候，通常都会给定初始值，这时候这些变量就成为类的成员变量。也就是说，每一个实例都有自己的一份属于自己的变量副本，这些变量就可以通过类的引用进行访问。
         
         ### 2.2 实例变量
         “实例变量”是指那些在每个实例对象（或者说每个类的实例）自己的数据成员。实例变量在类的声明中以实例变量名定义并初始化。实例变量仅存在于特定的实例对象中，不会与其他实例共享。实例变量可以通过实例对它的引用进行访问。实例变量可以在对象创建的时候指定，也可以在运行时动态设置。
         
         ### 2.3 self参数
         在Python中，self是一个特殊参数，表示的是当前实例的地址。它一般作为第一个参数传入实例方法，代表了当前实例的身份。
         
       ## 3.核心算法原理和具体操作步骤以及数学公式讲解
         
         ### 3.1 类变量的创建
         
             class A:
                 count = 0
                 
            类A的count变量是一个类变量，它的值将被所有实例共享。当创建一个新的实例对象A()时，A().count的初始值为0。然后可以使用以下方式访问和修改count变量：
            
            ```python
            a1=A()
            a2=A()
            
            print(a1.count)   #输出：0
            print(a2.count)   #输出：0
            
            a1.count+=1    #实例a1的count变量增加1
            print(a1.count)   #输出：1
            print(a2.count)   #输出：1
            
            A.count-=1    #类A的count变量减少1
            print(a1.count)   #输出：1
            print(a2.count)   #输出：0
            
            del a1.count     #删除实例a1的count变量
            print(a1.count)   #报错：AttributeError:'A' object has no attribute 'count'
            print(a2.count)   #输出：0
            
            ```
            
            从上面例子可知，实例变量仅存在于各自实例对象中，所以无法通过其他实例直接访问或修改。而类变量则会被所有实例共享，所以可以通过类本身来访问或修改。
            
           ### 3.2 实例变量的创建
           
               class B:
                   def __init__(self):
                       self.value = 0
                       
                b1 = B()
                b2 = B()
                
                print(b1.value)    #输出：0
                print(b2.value)    #输出：0
                
                b1.value += 1      #实例b1的value变量增加1
                print(b1.value)    #输出：1
                print(b2.value)    #输出：0
                
                B.value -= 1       #类B的value变量减少1
                print(b1.value)    #输出：0
                print(b2.value)    #输出：0
                
                del b1.value        #删除实例b1的value变量
                print(b1.value)     #报错：AttributeError:'B' object has no attribute 'value'
                print(b2.value)     #输出：0
               
              从以上例子可知，实例变量是类的属性，可以随着对象的创建而实例化。实例变量只能通过实例来访问和修改。
            
           ### 3.3 self参数
            
            在创建实例对象时，系统自动把对象引用(reference)作为第一个参数传递给类的构造器__init__()。也就是说，实例对象总是从它的构造器调用的方法中接收self参数。
            
           ### 3.4 super函数
            
               class C(object):
                   def __init__(self, x, y):
                       self.x = x
                       self.y = y
                   
               c1 = C(1, 2)
               print(c1.x, c1.y)    #输出：1 2
               
               class D(C):
                   def __init__(self, x, y, z):
                       super().__init__(x, y)    #调用父类的构造器
                       self.z = z
                       
               d1 = D(3, 4, 5)
               print(d1.x, d1.y, d1.z)    #输出：3 4 5
            
               上例中，D继承了C，并且在构造器__init__()中调用了super().__init__(x, y)，这表示调用父类C的构造器__init__()，并将值赋给了实例变量x和y。这样子的话，D的实例就同时具备了C的所有属性。
               
            注意：在定义子类的时候，必须调用父类的构造器__init__()，不然实例化子类时会出现错误。
            
          ### 3.5 属性访问控制
          
            通过以下的代码可以观察到，如果两个类之间存在父子关系，那么子类可以访问父类的属性，而不能修改父类的属性：
            
            ```python
            class E:
                pass
            
            class F(E):
                def __init__(self):
                    self.name = "F"
                    
            f1 = F()
            print(f1.name)   #输出："F"
            
            f1.name = "G"    #报错：AttributeError: can't set attribute
            ```
            
            如果想要让子类也能够修改父类的属性，需要用到一些技巧：
            
            ```python
            class G:
                @property
                def age(self):
                    return self._age
                
                @age.setter
                def age(self, value):
                    if not isinstance(value, int):
                        raise TypeError("age must be an integer")
                    self._age = value
                    
            class H(G):
                def __init__(self):
                    self.name = "H"
                    self._age = 10
            
            h1 = H()
            print(h1.name, h1.age)    #输出："H" 10
            
            h1.age = 20               #修改成功！
            print(h1.age)             #输出："20"
            ```
            
            以上代码中，类G定义了一个属性age，然后将之封装成了一个“具有getter和setter的属性”——age属性由_age实现。这样，子类H就能修改父类的age属性了。

