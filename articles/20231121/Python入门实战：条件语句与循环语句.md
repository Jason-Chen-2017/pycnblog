                 

# 1.背景介绍


条件语句与循环语句是编写计算机程序时经常使用的结构。他们的应用非常广泛，能够极大地提高编程效率和代码可读性。在Python中也提供了相应的语法规则和函数库支持。本文将首先对条件语句和循环语句进行概述，然后逐步深入介绍它们的应用、特点及相关算法。
## 条件语句
条件语句（Conditional Statement）即指根据某种条件判断执行不同的分支代码，根据某个条件表达式的值是否满足特定的条件来执行不同的代码分支。在Python中，条件语句可以包括if、elif、else等关键字，用于处理各种逻辑判断和选择。Python中的条件语句主要包括以下几种类型：

1. if-else语句
   如果条件成立则执行代码块A；否则执行代码块B。

   ```python
   # Example 1: if-else statement
   x = int(input("Enter a number: "))
   
   if x > 0:
       print("Positive")
   else:
       print("Negative or Zero")
   ```
   
2. if-elif-else语句
   有多个条件时可以使用if-elif-else结构。如果第一个条件不满足，则进入第二个条件判断，依次类推。如果所有条件都不满足，则执行最后一个else语句。

   ```python
   # Example 2: if-elif-else statement
   grade = input("Enter your grade (A, B, C, D, F): ")
   
   if grade == "A":
       print("Excellent!")
   elif grade == "B":
       print("Good job!")
   elif grade == "C" or grade == "D":
       print("You need to study more.")
   else:
       print("Better luck next time.")
   ```
   
3. if语句
   在只有一个条件时也可以使用if语句。该语句不需要使用缩进来表示代码块，只需将多行代码放在同一行内即可。

   ```python
   # Example 3: single line if statement
   num = random.randint(-100, 100)
   if num >= 0: print("{} is positive".format(num))
   ```
   
## 循环语句
循环语句（Looping Statement）即指按照顺序重复执行代码块，直到满足特定条件退出循环，或者一直执行下去。在Python中，循环语句主要包括以下几种类型：

1. while循环

   当指定的条件为True时，循环执行指定代码块。一般用于无限循环或循环次数已知。

   ```python
   # Example 4: while loop
   i = 0
   while i < 5:
       print(i)
       i += 1
   ```
   
2. for循环

   对序列元素逐一遍历，一次执行指定代码块。一般用于具有固定数量的元素集合。

   ```python
   # Example 5: for loop
   fruits = ["apple", "banana", "cherry"]
   for fruit in fruits:
       print(fruit)
   ```
   
3. range()函数

   用range()函数来生成一个整数序列，然后用for循环来迭代这个序列。

   ```python
   # Example 6: using range function with for loop
   total_sum = 0
   for i in range(101):   # generates integers from 0 to 99 inclusive
       total_sum += i      # adds each integer to the sum variable
   print("The sum of numbers from 0 to 100 is:", total_sum) 
   ```

以上就是Python中条件语句和循环语句的基本介绍。后续将继续深入探讨其应用、特点及相关算法。期待您的共同建设！😀😁😂🤣😃😄😅😆😉😊😋😎😍😘😗😙😚☺️