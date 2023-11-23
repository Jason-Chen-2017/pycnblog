                 

# 1.背景介绍



流程控制是编程中经常使用的一种基本结构。它可以用来实现数据处理、业务逻辑判断等多种功能。在不同编程语言中，流程控制语法也存在差异，但本文只讨论Python语言中的流程控制语句。

        在Python中，流程控制分成两种类型：顺序执行和选择执行。
- 顺序执行（Sequential Execution）:一般用于对某些操作进行排列组合，按照固定顺序一步步地完成。例如：打印一系列数字，或者读取一个文件的内容并打印出来。Python中使用for循环或者while循环即可实现这种功能。
- 选择执行（Selection Execution）:用于根据条件判断，决定是否执行特定的代码块。例如：if语句、switch语句或者case语句。Python中使用这些语句即可实现流程控制。

本文将通过两个示例——斐波那契数列和猜拳游戏——来讲解Python的流程控制。

    另外，本文还将介绍一些Python中比较重要的流程控制语句以及对应的用法，包括：if语句、for语句、while语句、try...except语句、continue语句、break语句等。希望大家能够从本文中获益。
# 2.核心概念与联系
## 2.1 Python中的流程控制语句概览
Python中的流程控制语句可以分为以下几类：

1. if/else语句：用于根据条件判断是否执行特定代码块；
2. for语句：用于迭代序列或其他可迭代对象（如列表）中的元素；
3. while语句：用于重复执行代码块直到指定的条件满足；
4. try…except语句：用于捕获并处理异常（Exception），防止程序崩溃；
5. continue语句：用于跳过当前循环体的剩余语句，进入下一次循环；
6. break语句：用于退出当前循环体，不再继续执行后续的代码块。

## 2.2 判断语句（if...else...）
判断语句就是让计算机做出选择，根据条件判断结果是否满足某个条件，然后执行相应的代码。Python中的判断语句又分为简单判断语句和复杂判断语句。

### 2.2.1 简单判断语句
- 语法：

   ```
   if expression:
       statement(s)
   else:
       other_statement(s)
   ```
   
- 参数：
   - `expression` :布尔表达式，计算结果为True或False。
   - `statement(s)` :当表达式计算结果为True时执行的代码块，可以是单个语句或者多个语句组成的复合语句。
   - `other_statement(s)` :当表达式计算结果为False时执行的代码块，可以是单个语句或者多个语句组成的复合语句。
- 示例：
  - 求绝对值：

    ```python
    a = -5
    
    # simple absolute value 
    abs_a = a if a >= 0 else (-a) 
    
    print("The absolute value of", a, "is:", abs_a)  
    # Output: The absolute value of -5 is: 5
    ```
  
  - 判断奇偶性：

    ```python
    num = 7
    
    # check even or odd number
    result = 'even' if (num % 2 == 0) else 'odd'
    
    print(num,"is ",result)   
    # Output: 7 is  odd
    ```

### 2.2.2 复杂判断语句
对于复杂的判断条件来说，我们可以使用elif关键字进行判断。

- 语法：

  ```
  if expression1:
      statement1
  elif expression2:
      statement2
  elif expressionN:
      statementN
  else:
      default_statement
  ```

- 参数：
  - `expression` :布尔表达式，计算结果为True或False。
  - `statement` :当表达式计算结果为True时执行的代码块，可以是单个语句或者多个语句组成的复合语句。
  - `default_statement` :当所有前面的表达式都为False时执行的代码块，可以是单个语句或者多个语句组成的复合语句。
- 示例：
  - 检查成绩：

    ```python
    score = 90
    
    if score >= 90:
        grade = 'A'
    elif score >= 80:
        grade = 'B'
    elif score >= 70:
        grade = 'C'
    elif score >= 60:
        grade = 'D'
    else:
        grade = 'F'
        
    print('Your grade is',grade)    
    # Output: Your grade is A
    ```
  
  

## 2.3 循环语句（for...in... and while...）
循环语句就是让计算机按顺序重复执行代码块，直到所有元素都被遍历完。Python中支持的循环语句有for和while两种。

### 2.3.1 for循环
- 语法：

  ```
  for variable in sequence:
      statements
  ```
  
- 参数：
  - `variable` :循环变量，循环过程中每次迭代的值都会被赋值给该变量。
  - `sequence` :可迭代对象，表示需要循环的元素集合。
  - `statements` :循环体，表示需要循环执行的语句。
- 示例：
  - 求和：

    ```python
    sum = 0
    
    for i in range(1, 11):
        sum += i
        
    print('Sum of first 10 numbers:',sum)    
    # Output: Sum of first 10 numbers: 55
    ```
    
### 2.3.2 while循环
- 语法：

  ```
  while condition:
      statements
  ```
  
- 参数：
  - `condition` :布尔表达式，表示循环条件。
  - `statements` :循环体，表示需要循环执行的语句。
- 示例：
  - 斐波那契数列：

    ```python
    n1, n2 = 0, 1
    count = int(input("Enter the number of terms you want to see:"))
    
    if count <= 0:
        print("Please enter a positive integer")
    elif count == 1:
        print("Fibonacci series upto",count,":")
        print(n1)
    else:
        print("Fibonacci series:")
        for i in range(2,count):
            nth = n1 + n2
            print(nth)
            n1 = n2
            n2 = nth
    ```
    
  - 猜拳游戏：

    ```python
    import random
    
    computer_choice = ['rock','paper','scissors']
    user_choice = input("Enter your choice (rock/paper/scissors): ")
    
    while True:
    
        if user_choice not in computer_choice:
            print("Invalid Input! Please choose again.")
            user_choice = input("Enter your choice (rock/paper/scissors): ")
            
        else:
            computer_choice = random.choice(computer_choice)
            
            print("Computer Choice:", computer_choice)
            
            if (user_choice == 'rock') & (computer_choice =='scissors'):
                print("You win!")
                
            elif (user_choice == 'paper') & (computer_choice == 'rock'):
                print("You win!")
                
            elif (user_choice =='scissors') & (computer_choice == 'paper'):
                print("You win!")
                    
            else:
                print("It's a tie!")
                
            play_again = input("\nDo you want to play again? (y/n): ")
            if play_again.lower()!= 'y':
                break
            
    print("\nThank You For Playing!!")
    ```
    
## 2.4 异常处理语句（try...except...）
异常处理语句用来捕获并处理程序运行过程中的错误信息。如果出现错误，则会抛出一个异常，需要根据异常类型进行相应的处理。

### 2.4.1 try...except语句
- 语法：

  ```
  try:
      statements_to_be_executed
  except ExceptionType as e:
      statements_when_exception_occurs
  ```
  
- 参数：
  - `statements_to_be_executed` :异常发生时需执行的代码块，可以是单个语句或者多个语句组成的复合语句。
  - `ExceptionType` :要捕获的异常类型。
  - `e` :代表出现的异常对象。
  - `statements_when_exception_occurs` :当出现异常时需执行的代码块，可以是单个语句或者多个语句组成的复合语句。
- 示例：
  - 除法运算：

    ```python
    try:
        x = int(input("Enter Dividend: "))
        y = int(input("Enter Divisor: "))
        z = x / y
        print("Result:",z)
    except ZeroDivisionError:
        print("Cannot divide by zero!")
    ```
    
    