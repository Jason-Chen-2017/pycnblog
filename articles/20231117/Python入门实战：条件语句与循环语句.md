                 

# 1.背景介绍


条件语句（conditional statements）与循环语句（loop statements），是学习编程的必备技能之一。但掌握它们并不是件容易的事情。本文将从基本知识、语法规则、用法介绍等方面对条件语句与循环语句进行讲解。
首先，什么是条件语句与循环语句？
在计算机程序设计中，条件语句用来决定执行代码块的过程是否符合预设条件。而循环语句则用于重复执行特定代码块直到满足某些条件为止。简单来说，条件语句就是根据判断条件不同执行不同的代码分支，循环语句则是多次执行相同的代码段，直到某个条件被满足。
条件语句通常包括if-else、switch-case、逻辑运算符(&&、||、!)等。循环语句通常包括for、while、do-while等。
本文将以简单的示例为主线，逐步深入地讲解条件语句与循环语句的用法及相关概念。
# 2.核心概念与联系
## 2.1条件语句
条件语句的核心概念主要有三个：表达式、条件、语句。其中，表达式指代一个值或变量，条件即是判断条件，决定了执行语句块还是不执行语句块；语句则是要执行的动作。根据表达式的值，条件语句可以分成以下四种类型：
### 2.1.1 if语句
if语句是最基本的条件语句，其一般形式如下:
```python
if 表达式:
    语句
```
若表达式的值为True，则执行语句块中的语句；否则，跳过语句块直接执行后续语句。例如：

```python
num = 9
if num > 5:
    print("num is greater than 5")
print("This statement will always be executed.")
```
输出结果为：
```
num is greater than 5
This statement will always be executed.
```
这里有一个细节需要注意，当if后的条件表达式的结果是False时，程序仍然会继续执行紧跟着的后续语句，而不是忽略后续语句。
### 2.1.2 if...else语句
if...else语句是一个常用的条件语句，其一般形式如下：
```python
if 表达式:
    语句1
else:
    语句2
```
当表达式的值为True时，执行语句1；当表达式的值为False时，执行语句2。例如：

```python
num = 3
if num % 2 == 0:
    result = "Even"
else:
    result = "Odd"
print("The number", num, "is", result)
```
输出结果为：
```
The number 3 is Odd
```
这里还有一个细节需要注意，如果if...else语句没有括号，即if else，那么每个条件都必须独占一行。换句话说，不能将两个条件放在同一行。
### 2.1.3 if...elif...else语句
if...elif...else语句也是一种常用的条件语句，其一般形式如下：
```python
if 表达式1:
    语句1
elif 表达式2:
    语句2
elif 表达式3:
    语句3
....
else:
    默认语句
```
类似于switch-case结构，它提供了多分支条件，只有第一个表达式为True时才执行对应的语句。如果所有的表达式都为False，则执行默认语句。例如：

```python
score = 75
if score >= 90:
    grade = 'A'
elif score >= 80 and score < 90:
    grade = 'B'
elif score >= 70 and score < 80:
    grade = 'C'
elif score >= 60 and score < 70:
    grade = 'D'
else:
    grade = 'F'
print('The student gets', grade)
```
输出结果为：
```
The student gets B
```
### 2.1.4 嵌套if语句
if语句也可以嵌套，即在if或者else语句内部再添加另一个if语句，如此即可实现更复杂的条件判断。例如：

```python
num = -5
if num >= 0:
    if num == 0:
        print("The number is zero")
    elif num > 0:
        print("The number is positive")
    else:
        print("The number is negative")
else:
    print("The number is not valid")
```
输出结果为：
```
The number is negative
```
## 2.2 循环语句
循环语句的作用是重复执行语句块直到满足某些条件为止。循环语句分为两类，分别为迭代语句（iteration statements）和退出语句（exit statements）。其中，迭代语句又分为顺序迭代语句（order iteration statements）和非顺序迭代语句（non-order iteration statements）。
### 2.2.1 for语句
for语句是一种顺序迭代语句，其一般形式如下：
```python
for var in iterable:
    语句
```
var表示一个变量，该变量将在每次循环时依次获得iterable的元素值；iterable表示一个可迭代对象，比如列表、元组、字符串。语句表示要循环执行的代码块。例如：

```python
numbers = [1, 2, 3, 4]
sum = 0
for num in numbers:
    sum += num
print("Sum of the numbers:", sum)
```
输出结果为：
```
Sum of the numbers: 10
```
注意：for语句是一个非常常用的语句，一定要熟练掌握。而且，for语句只能处理可迭代对象，不能处理迭代器对象。因此，推荐尽量使用列表、元组来替代迭代器对象。
### 2.2.2 while语句
while语句是一种非顺序迭代语句，其一般形式如下：
```python
while 表达式:
    语句
```
表达式表示判断条件，当其值为True时，语句块内语句将被执行；否则，跳出循环。例如：

```python
count = 1
total = 0
while count <= 10:
    total += count
    count += 1
print("Total of numbers from 1 to 10 is:", total)
```
输出结果为：
```
Total of numbers from 1 to 10 is: 55
```
### 2.2.3 break语句
break语句是退出循环语句的语句，其一般形式如下：
```python
break
```
break语句会立即退出当前循环，并跳转至下一条语句。例如：

```python
count = 1
total = 0
while True:
    if count > 10:
        break
    total += count
    count += 1
print("Total of numbers from 1 to 10 is:", total)
```
输出结果为：
```
Total of numbers from 1 to 10 is: 55
```
### 2.2.4 continue语句
continue语句也是退出循环语句的语句，但是continue语句不会终止当前循环，而是跳回到循环的开头重新开始新的一轮循环。其一般形式如下：
```python
continue
```
例如：

```python
count = 1
total = 0
while count <= 10:
    if count % 2 == 0:
        count += 1
        continue # skip even number
    total += count
    count += 1
print("Total of odd numbers from 1 to 10 is:", total)
```
输出结果为：
```
Total of odd numbers from 1 to 10 is: 25
```
### 2.2.5 pass语句
pass语句是空语句，是为了保持程序结构完整性而存在的。例如：

```python
while False:
    pass # do nothing here
```
上面代码中的pass语句永远不会被执行，因为它的条件永远为False。实际上，pass可以理解为占位符，在书写代码过程中起到“不做任何事”的效果。
## 2.3 表达式、条件、语句的关系
表达式、条件、语句的关系可以总结为如下几条：
1. 每个语句都由一个表达式组成，但不是所有表达式都表示一个条件。
2. 当执行一个if、while、for、函数调用等语句时，就会产生一个条件。
3. 如果执行的是if-else语句，或者执行到第一次迭代的时候，就产生了条件。
4. 在条件语句和其他语句之间，可能会出现嵌套语句。
5. 有时候多个条件是与的关系，有时候是或的关系。
6. if-elif-else语句的意义在于从多个条件中选择一个满足的条件执行相应的语句块。