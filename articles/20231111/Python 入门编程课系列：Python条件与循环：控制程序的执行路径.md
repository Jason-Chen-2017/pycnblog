                 

# 1.背景介绍


Python条件语句(Conditional Statement)和循环结构(Looping Structures)，经常被作为学习编程语言的第一步。本系列课程将以Python语言作为教材，介绍如何利用条件语句和循环结构编写程序，帮助读者更加熟悉和灵活地控制程序的执行路径。
条件语句与循环语句都属于程序运行流程中的控制结构，用于对计算机程序进行逻辑判断并根据判断结果选择不同执行路径。因此，掌握条件语句和循环结构对于编写高质量、健壮的代码至关重要。通过本系列课程，读者可以了解Python中条件语句及其语法规则，理解Python中常用的循环语句，包括for语句和while语句等，并能够使用它们实现一些实际应用场景下的功能。
# 2.核心概念与联系
## Python条件语句概览
Python条件语句由if、elif和else三个关键字组成。如下所示：

```python
if condition:
    # code block if condition is true
    
elif other_condition:
    # code block for the first false condition and this one being true
    
else:
    # default code block if all conditions are false
```

条件语句是一种非常常见的控制流程结构，它允许程序基于一系列条件做出不同的决策或动作。在Python语言中，条件语句中使用的条件表达式(condition expression)支持比较运算符(比如==、!=、>、<、>=、<=)，逻辑运算符(比如and、or、not)，布尔类型值True和False。当满足条件表达式时，相应的代码块就会被执行，否则，如果还有其他的条件，则继续检查下一个条件。如果所有条件均不满足，那么程序会执行else子句里面的默认代码块。

## Python循环语句概览
Python语言提供了两种类型的循环结构，即for语句和while语句。

### 1.for语句
for语句用于遍历可迭代对象，每次迭代取出可迭代对象的第一个元素，然后进行处理，直到遍历完成。它的一般形式为：

```python
for var in iterable:
    # loop body executed repeatedly using value of var on each iteration
```

iterable可以是一个序列（list、tuple、set）、字典（dict）或其它可迭代对象。var是一个临时的变量，表示当前正在遍历的元素的值。loop body代表了在每次迭代中要执行的代码。通常情况下，var将从iterable中依次取出元素，并进行处理。

for语句的一个典型用法就是对序列中的每一个元素都执行相同的操作，例如：

```python
numbers = [1, 2, 3, 4]
sum = 0
for num in numbers:
    sum += num
print("The sum is:", sum)   # Output: The sum is: 10
```

上述例子中，for语句用来遍历numbers列表，把每个元素的值添加到sum变量之中。最后打印出sum变量的值。

注意：在for语句中定义的临时变量（如num、sum）只在for语句范围内有效。如果需要在循环外面使用这些变量，需要使用global关键字声明。

### 2.while语句
while语句也称作条件循环语句，它也是一种控制流语句，用于重复执行代码块，直到指定的条件为假。它的一般形式为：

```python
while condition:
    # loop body executed as long as condition evaluates to True
```

condition是一个表达式，只有当该表达式为True时，才会重复执行代码块；当表达式为False时，循环结束。循环体内的代码块通常是一次性的，只能执行一次。while语句的一个典型用法是计算阶乘：

```python
number = int(input("Enter a number: "))    # get user input
factorial = 1     # initialize factorial variable
n = 1             # counter variable starts at 1
while n <= number:
    factorial *= n
    n += 1        # increment counter variable by 1
print("Factorial of", number, "is:", factorial)   # print result
```

上面这个例子中，用户输入一个数字，程序计算它的阶乘并输出结果。由于输入的数据可能不是正整数，因此需要首先把输入数据转换为整数。然后初始化两个变量：factorial为1，用于存储最终的结果；n为1，用于记录当前正在计算的阶乘值。循环条件为n小于等于number，所以程序会一直执行到用户输入的数字阶乘为止。在每次循环迭代中，程序都会更新factorial的值，使得它等于之前的阶乘乘以当前计数器的值。

注意：在while语句中定义的临时变量（如n、factorial）只在while语句范围内有效。如果需要在循环外面使用这些变量，需要使用global关键字声明。

## Python多重条件语句(Ternary Operator)
Python还提供了一个类似C语言的三元运算符(Ternary Operator)。它简化了条件语句的使用方法。

三元运算符由三个运算对象组成：一个布尔表达式、一个真值的表达式和一个假值的表达式。当布尔表达式为True时，运算结果为真值表达式的值；反之，则为假值表达式的值。它的一般形式为：

```python
value = true-expression if boolean-expression else false-expression
```

在上面的表达式中，boolean-expression是一个布尔表达式，true-expression和false-expression都是表达式。如果boolean-expression的值为True，则返回true-expression的值；否则，返回false-expression的值。此处，true-expression和false-expression不能同时为空。

三元运算符常用于代替if...else语句，这样可以减少代码行数，提高代码可读性。举例如下：

```python
x = 'y' if x > 0 else 'n'  
```

上述代码将x的值转化为字符“y”或者“n”，具体选择哪个根据是否大于零决定。