                 

# 1.背景介绍


在Python编程语言中，流程控制是指根据条件执行特定代码块的命令序列。包括顺序结构、分支结构（if-else）、循环结构（while-for）。本文将对这些流程控制结构的基本用法及其特点进行讲解，并结合具体的代码实例进一步介绍。

# 2.核心概念与联系
## 顺序结构——从上到下依次执行语句

顺序结构在Python中一般表示为通过缩进的方式来实现。一个Python程序可以由多行语句组成，它们会按顺序被解释器执行。例如：

```python
print("Hello")   # 输出 Hello
x = 1 + 2        # 对 x 赋值
y = x * 3        # 对 y 赋值
z = -7           # 对 z 赋值
print(z)         # 输出 -7
```

## 分支结构——条件判断语句

分支结构主要用于根据条件的真假来决定是否执行某段代码。在Python中，分支结构包括if-else语句、条件表达式等语法元素。如果if语句中的条件判断结果为True，则执行if后的代码块；否则，如果存在对应的else子句，则执行该代码块。例如：

```python
a = int(input("Enter a number: "))   # 用户输入数字
if a % 2 == 0:
    print("Even!")
elif a % 2 == 1:
    print("Odd!")
else:
    print("Invalid input.")
```

条件表达式是一种简化形式的if-else语句。它直接返回True或False，而不是像if-else那样需要判断条件。条件表达式常用在条件判断表达式中，例如：

```python
b = "hello" if len(input()) > 5 else "world"    # 如果用户输入字符长度大于5，则打印 hello，否则打印 world
c = True if (len(s), s)!= (3, 'cat') else False     # 判断两个字符串是否相等
d = None if not result else sum([int(n) for n in str(result)]) / len(str(result))      # 求平均值
```

## 循环结构——重复执行代码块

循环结构主要用于重复执行同一段代码块，直至某个结束条件满足。在Python中，循环结构包括while循环和for循环两种。

### while循环——重复执行代码块，直至满足退出条件

while循环的语法如下：

```python
count = 0       # 初始化变量 count 为 0
while count < 5:
    print(count)
    count += 1
```

在这个例子中，count从0开始自增，当count小于5时，程序会一直运行，每执行一次循环体内的语句就会输出一次计数值，然后再加1。当count达到5时，程序退出while循环。

### for循环——对集合内各项逐一处理

for循环的语法如下：

```python
fruits = ['apple', 'banana', 'orange']          # 创建列表 fruits
for fruit in fruits:                            # 使用 for 循环遍历列表 fruits 的所有元素
    print(fruit)                                # 输出每个元素的值
```

在这个例子中，程序先创建一个列表fruits，里面有三个水果名。然后，程序使用for循环遍历列表fruits的所有元素，每次都取出一个元素，并赋予给变量fruit，然后执行后续代码块。所以，这段代码会把列表中的每个水果名输出一遍。

另外，还可以使用range()函数生成迭代器对象，用作for循环的循环变量。比如：

```python
sum = 0                                         # 初始化变量 sum 为 0
for i in range(10):                             # 生成 0 ~ 9 的数字作为循环变量
    sum += i                                     # 将当前循环变量 i 的值添加到变量 sum 中
print(sum)                                       # 输出最终的求和结果
```

这个例子中，程序使用for循环对0~9范围内的整数进行累加，并最后输出总和。