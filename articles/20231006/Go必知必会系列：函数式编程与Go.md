
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


函数式编程(Functional Programming)是一种编程范式，它强调计算机程序的运算应该是靠纯函数来完成而不是靠修改可变状态来完成。纯函数指的是输入参数相同则输出结果也相同的函数，没有副作用。也就是说函数要保持独立性、无状态、没有可变状态，因此不能产生 side effect，在同样的输入下总是返回同样的结果。函数式编程就是利用纯函数和高阶函数将计算过程描述为函数式模型，通过组合不同的函数实现各种复杂功能。
Go语言是2009年由Google开发，是一个开源的静态强类型、编译型、并发执行的编程语言。它支持多种编程模式，包括命令式编程、面向对象编程、函数式编程等。作为一门现代化的语言，它的特性和功能极其丰富，已经成为大型公司的必备技能之一。因此，本系列教程将从浅入深地介绍函数式编程，帮助Go开发者快速上手，掌握如何使用Go编写函数式风格的代码。
# 2.核心概念与联系
## 2.1 高阶函数
所谓高阶函数(Higher-order function)，就是能够接受或者返回函数作为参数或者返回值的函数。高阶函数让函数编程变得非常灵活，比如函数可以作为参数传递给其他函数，也可以作为返回值返回给调用者。函数式编程里经常用的高阶函数有map/reduce/filter，用于数据处理；compose/currying用于组合多个函数；fold/accumulate/scan用于遍历集合和树形结构数据;compose这些高阶函数还可以用来构建DSL(domain specific language)。
## 2.2 柯里化 Currying
所谓柯里化Currying，就是将一个多参数函数转换成一系列单参数函数的函数。一般来说，一个函数如果需要接收多个参数，通常将它们组织为一个tuple或列表，然后将其作为参数传给另一个函数。例如：
```
def add(x):
    def inner(y):
        return x + y
    return inner
    
add_one = add(1)
print(add_one(2)) # Output: 3
print(add_one(-3)) # Output: -2
```
但是这种方式太过繁琐，尤其当参数个数很多时，可读性较差。柯里化可以简化这一过程，将原来的函数的第一个参数固定住，之后每次传入第二个参数。这样就可以得到不同参数的结果。例如：
```
from functools import partial
def add(a, b):
    return a+b

add_five = partial(add, 5)
print(add_five(7)) # Output: 12
print(add_five(-2)) # Output: 3
```
## 2.3 闭包 Closure
所谓闭包Closure，就是一个内部函数引用了外部函数变量的环境的函数。它被存储在内存中，并且可以在后续某个时刻访问到这个函数及其相关的参数和局部变量。比如以下代码创建了一个对斜杆的求值函数`evalute`，该函数接收一个字符串表达式作为参数，并返回表达式的结果。这个函数内部又定义了一个匿名函数`evaluate`，这个函数在`evalute`的上下文中执行。
```python
def evaluate():
    def inner(expression):
        result = None
        for token in expression.split(' '):
            if token == '+':
                left, right = stack.pop(), stack.pop()
                result = eval(left) + eval(right)
            elif token == '-':
                left, right = stack.pop(), stack.pop()
                result = eval(left) - eval(right)
            else:
                result = float(token)
            stack.append(result)
        return result
    return inner
```
通过闭包，`inner` 函数可以在 `evalute` 函数的外部调用。如下示例，创建两个求值器，分别在计算`1+2` 和 `3*4`的结果。由于`evaluate` 函数是一个闭包，所以内部函数 `inner` 可以访问到外部函数的变量`stack`。
```python
e1 = evaluate()
e2 = evaluate()
res1 = e1("1 + 2")
res2 = e2("3 * 4")
print(res1)   # output: 3
print(res2)   # output: 12
```