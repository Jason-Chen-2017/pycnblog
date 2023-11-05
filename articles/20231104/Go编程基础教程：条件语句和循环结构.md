
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Go（又称Golang）是一个开源的静态强类型、编译型、并发型的编程语言，由Google公司推出。Go是为面向计算性能和系统编程设计的静态ally-typed语言，拥有语法简洁和高效的特点。其设计哲学是用简单的方法做最简单的事情，同时也避免不必要的复杂性。
本文将讲解Go语言中经典的条件语句和循环结构，包括if、else、switch、for等语句及其相关语法规则。通过这些结构，你可以快速掌握Go语言的基本功力。学习本文中的知识可以让你更好地理解Go语言，编写出更健壮、可维护的程序。
# 2.核心概念与联系
## 2.1 if else 语句
if else语句用于根据不同的条件执行不同的代码块。它可以对一个或多个布尔表达式进行判断，然后根据表达式的结果执行对应的代码块。
在Go语言中，if else语句是通过关键字"if"和"else"构成的。它的一般形式如下：

```go
if condition1 {
   // code block to be executed if condition1 is true
} else if condition2 {
   // code block to be executed if condition1 is false and condition2 is true 
} else {
   // code block to be executed if both conditions are false
}
```

condition1 和 condition2 是布尔表达式，可以是一个函数调用或者比较运算符的返回值。如果condition1 为true，则会执行第一个代码块；如果condition1 为false，则会检查condition2是否为true，如果condition2也为true，则会执行第二个代码块；否则，会执行第三个代码块。

注意：一般情况下，条件控制语句(if/else)都要加上括号，即使只有一条语句也是如此。例如：

```go
func main() {
    a := 10
    b := 20

    if (a > b){
        fmt.Println("a is greater than b")
    }else{
        fmt.Println("b is greater than or equal to a")
    }
    
    c := "hello world"
    if len(c)<5{
        fmt.Println("length of c less than 5")
    }else if len(c)>5 && len(c)<10{
        fmt.Println("length of c between 5 and 9")
    }else{
        fmt.Println("length of c greater than 9")
    }
    
}
```

输出：

```go
 a is greater than b
 length of c greater than 9
```


## 2.2 switch语句
switch语句类似于C++或Java中的多路分支结构，它提供了一种简单而有效的方式来实现条件跳转。在Go语言中，switch语句是通过关键字"switch"、表达式和花括号{}来构造的，形式如下：

```go
switch expression {
   case value1:
      // code block to be executed if the result of expression equals to value1 
      break 
   case value2:
      // code block to be executed if the result of expression equals to value2 
      break 
  ... 
   default:  
       // code block to be executed if none of above cases is true 
       break
}
```

expression表示的是需要判断的值，value1、value2...是可能出现的值。当switch检测到某个case的值与表达式相等时，就会执行该case对应的代码块。每条case后面都需要有一个break语句，防止代码跳跃到下一条case去执行。default语句是可选的，当switch无法匹配到任何case时，就会执行default对应的代码块。

如下例子：

```go
package main

import "fmt"

func main() {
   var grade string = "B"

   switch grade {
   case "A":
      fmt.Println("Excellent!")
   case "B", "B+":
      fmt.Println("Good job!")
   case "C":
      fmt.Println("You passed.")
   case "D":
      fmt.Println("Better try again.")
   case "F":
      fmt.Println("Sorry, you failed.")
   default:
      fmt.Println("Invalid grade.")
   }
}
```

输出：

```go
Good job!
```

## 2.3 for循环
for循环是Go语言中唯一的循环结构。它的一般形式如下：

```go
for initialization; condition; post {
   // code block to be repeatedly executed until condition becomes false
}
```

其中initialization初始化变量，condition为循环条件，post为每次循环结束后的处理工作。比如，可以在for循环前声明一些局部变量，也可以在循环体内嵌入其他语句。

for循环最常用的地方就是用来遍历数组和切片，示例如下：

```go
package main

import "fmt"

func main() {
   array := [5]int{1, 2, 3, 4, 5}

   for i := 0; i < len(array); i++ {
      fmt.Printf("%d ", array[i])
   }
   fmt.Println()

   slice := []string{"apple", "banana", "orange"}

   for i := 0; i < len(slice); i++ {
      fmt.Printf("%s ", slice[i])
   }
   fmt.Println()
}
```

输出：

```go
1 2 3 4 5 
 apple banana orange 
```

另外，还可以使用range关键字来遍历数组和切片，其形式如下：

```go
for index, element := range array_or_slice {
   //code block to be repeated with every iteration
}
```

index为当前元素的索引，element为当前元素的值。示例如下：

```go
package main

import "fmt"

func main() {
   array := [5]int{1, 2, 3, 4, 5}

   sum := 0

   for _, num := range array {
      sum += num
   }

   average := float64(sum) / float64(len(array))

   fmt.Printf("Sum = %d\nAverage = %.2f\n", sum, average)
}
```

输出：

```go
Sum = 15
Average = 3.00
```