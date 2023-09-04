
作者：禅与计算机程序设计艺术                    

# 1.简介
  

函数式编程（Functional Programming）是一种编程范式，它将电脑运算视为数学计算，并且避免使用可变数据（Mutable Data）。函数式编程语言最重要的特征就是只能通过函数来编程，而不能用赋值语句（statement）修改程序状态。因此，函数式编程 languages are all about expressions and functions that return values instead of statements that modify values.

Python拥有许多高阶函数（Higher-order Functions），例如map(), filter()等可以接受函数作为参数的内置函数。这些函数提供了一种简单、便捷的方式来处理集合型数据，并将函数应用到每个元素或选择满足特定条件的元素上。

本文将详细介绍Python中的函数式编程知识。
# 2.基本概念术语说明
## 1. 函数
在计算机科学中，函数是一个独立模块化的代码块，用来实现某个功能。一个函数通常由输入、输出和对一些数据的处理流程组成。在Python中，函数使用def关键字定义，后跟函数名、括号和函数体，如def my_func(param):
    # do something with param
    pass
    
函数具有以下特点:

 - 参数：函数可以有零个或多个参数。
 - 返回值：函数可以通过return语句返回结果。
 - 可调用性：函数是对象，可以像其他任何变量一样被调用。
 - 作用域：函数内部可以使用局部变量，外部无法访问。
 - 默认参数：函数调用时可以设置默认值。
 - 递归函数：函数可以调用自身。
  
## 2. 匿名函数
除了使用def关键字定义普通函数外，Python还允许创建匿名函数（Anonymous Function），即不带函数名的函数。语法如下:

```python
lambda arg1, arg2,... : expression 
```
例如:

```python
sum = lambda x,y:x+y
print sum(2,3)   # output is 5
```
上面例子中的匿名函数只是简单的相加，但是也可以做更复杂的操作。比如:

```python
g = lambda x: x*x + 2*x + 1    # y=x^2+2x+1
h = lambda x: g(x)-9          # z=(x^2+2x+1)-9
z = h(2)                     # output is 4
```
上面例子中的匿�函数f(x)=ax^2+bx+c，其中a,b,c都是待定系数，求方程z=f(x)+9的根。利用两个匿名函数g(x)和h(x)，可以求得方程z=h(x)。

## 3. map()函数
map()函数接收两个参数，第一个参数是一个函数，第二个参数是一个iterable，map()会依次把函数作用于每个元素上，并把结果迭代返回。如果第一个参数是一个列表，则map()也会返回一个新的列表。

举例如下：

```python
numbers = [1, 2, 3, 4, 5]
result = list(map(lambda x: x**2, numbers))
print result      # output is [1, 4, 9, 16, 25]
```
该例中，map()函数把数字列表[1,2,3,4,5]转化成平方后的新列表。

## 4. filter()函数
filter()函数也接收两个参数，第一个参数是一个函数，第二个参数是一个iterable，filter()会依次把函数作用于每个元素上，并只保留结果为True的元素，然后把结果迭代返回。如果第一个参数是一个列表，则filter()也会返回一个新的列表。

举例如下：

```python
numbers = [1, 2, 3, 4, 5]
result = list(filter(lambda x: x%2==0, numbers))
print result     # output is [2, 4]
```
该例中，filter()函数从数字列表[1,2,3,4,5]中选出偶数，并生成新的列表。