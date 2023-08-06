
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         Python是一种高级、通用的、解释型、面向对象的脚本语言。它由Guido van Rossum创建于1991年。Python支持多种编程范式，包括面向过程、面向对象、函数式、数据驱动等。Python具有丰富的数据结构、标准库和模块，使其在科学计算、web开发、自动化运维等领域都受到广泛应用。
        
         Python编程语言特性的突出之处在于它的简洁性、易用性和可读性。因此，很多初学者认为学习Python会比其他编程语言更容易入门。同时，越来越多的公司在选用Python进行新项目开发，所以掌握Python编程技能将成为各个公司不可或缺的一项要求。本文是我总结和收集的Python相关面试题，希望能够帮助大家快速了解Python编程语言。
         
         # 2.Python基础
         ## 2.1 标识符(Identifiers)
         ### 2.1.1 命名规则
         - 首字符不能是数字
         - 只能使用字母、下划线(_)或者美元符号($)
         - 不可以使用关键字、保留字和特殊字符
         ```python
         # 合法的标识符
         var = 'Hello World' 
         _var = 'hello world' 
         Var$ = 'foo bar' 
         
         # 非法的标识符
         2var = "Invalid identifier" 
         class = "Not a valid identifier" 
         $var = "Cannot start with special character" 
         import = "Reserved word cannot be used as an identifier" 
         def_func() = "def keyword can't appear in middle of the identifier" 
         ```

         ### 2.1.2 作用域(Scope)
         Python的作用域分为全局作用域和局部作用域两种。

         #### 2.1.2.1 全局作用域（Global Scope）
         全局作用域是最外层的作用域，它定义了所有的全局变量和函数。在函数体内也可以引用全局作用域中的变量。

         #### 2.1.2.2 局部作用域（Local Scope）
         函数内部声明的变量都是局部变量，只对当前函数有效，离开函数后变量也就消失了。但是，如果在函数内部又嵌套了一个函数，那么这个函数的内部声明的变量也是局部变量，但还是属于最内层的局部作用域。

         当一个变量被赋值时，系统首先查找它是否在局部作用域中定义过。如果没有找到，再到上一级作用域中查找，直到全局作用域为止。如果全局作用域中也没有找到，则创建一个新的局部变量。

         这里有一个例子，来展示变量查找顺序：

         ```python
         x = 10          # global scope

         
         def func():
             y = 20      # local scope

             def inner_func():
                 z = 30   # nested function

                 print("x: ", x)    # prints 10
                 print("y: ", y)    # prints 20
                 print("z: ", z)    # prints 30

         
         if __name__ == "__main__":
             func()
             print("global x:", x)     # prints 10
         ```

         在`inner_func()`函数中，我们尝试访问外部函数作用域（即包含`inner_func()`函数的函数作用域，即`func()`函数），但是找不到变量`x`，因为它只能在局部作用域中定义。然后，我们再去查找它的父作用域，即`func()`函数作用域。由于该作用域已经被销毁，我们继续往上查找，最终找到全局作用域中的变量`x`。