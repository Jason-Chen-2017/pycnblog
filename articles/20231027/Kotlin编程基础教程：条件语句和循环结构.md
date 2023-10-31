
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


条件语句（Statements）和循环结构（Loops）是最基本、最重要的计算机编程语言结构。条件语句和循环结构在实际应用中经常被用到，可以实现对数据的操作控制和重复执行某段代码，并帮助开发者有效地解决程序中的问题。因此，掌握它们非常关键。本文将对Kotlin编程语言提供的条件语句和循环结构进行全面介绍，包括如何使用它们，以及它们是如何工作的。
# 2.核心概念与联系
## 2.1 条件语句
条件语句用于在程序执行过程中根据某个条件判断是否执行特定代码块，从而达到控制流程的目的。Kotlin支持多种条件表达式，包括`if-else`结构、`when`表达式和`switch`结构。
### if-else结构
最简单的条件语句形式就是`if-else`结构。它由一个布尔表达式和两个分支代码块组成。当布尔表达式计算结果为真时，执行第一个分支的代码；否则，执行第二个分支的代码。如下所示：

```kotlin
fun main() {
    var x = 5
    
    // if-else statement example
    if (x > 0) {
        println("Positive number")
    } else {
        println("Negative or zero number")
    }
}
```

输出：

```
Positive number
```

### when表达式
如果需要处理多个分支，可以使用`when`表达式。`when`表达式类似于`switch`结构，但是更加灵活。如下所示：

```kotlin
fun printSign(num: Int) {
    val sign = when (num) {
        in -2..2 -> "zero"
        0 -> "zero"
        num < 0 -> "negative"
        else -> "positive"
    }

    println("$num is $sign.")
}
```

`when`表达式首先比较每个条件，看它的值是否与相应的表达式匹配。如果匹配成功，则执行对应的代码块并退出。如果所有的条件都不匹配，则执行默认的分支。这种方式比使用多个`if-else`语句更简洁，更容易理解。例如，上述代码片段的输出结果为：

```
7 is positive.
-3 is negative.
0 is zero.
2 is positive.
-1 is negative.
```

### switch结构
`switch`结构也是一个选择语句，它的语法和`C/Java`类似。如下示例所示：

```kotlin
fun main() {
    var day = 7

    when (day) {
        1 -> println("Monday")
        2 -> println("Tuesday")
        3 -> println("Wednesday")
        4 -> println("Thursday")
        5 -> println("Friday")
        6 -> println("Saturday")
        7 -> println("Sunday")
        else -> println("Invalid input")
    }

    // same output as above using switch structure
    when (day) {
        1 -> println("Monday")
        2 -> println("Tuesday")
        3 -> println("Wednesday")
        4 -> println("Thursday")
        5 -> println("Friday")
        6 -> println("Saturday")
        7 -> println("Sunday")
        else -> println("Invalid input")
    }
}
```

注意：Kotlin中没有`switch`关键字，所以使用的是`when`。

## 2.2 循环结构
循环结构用于在程序执行过程中重复执行同一段代码或执行满足一定条件的代码块。Kotlin中提供了两种循环结构——`while`循环和`for`循环。
### while循环
`while`循环用于在指定条件下，重复执行代码块。如下示例所示：

```kotlin
fun countDown(n: Int) {
    var i = n
    while (i >= 0) {
        println(i--)
    }
}
```

这个函数接收一个整数参数，表示计数器终止值。然后创建一个`while`循环，其条件是`i>=0`，即只要`i`大于等于0，就继续执行循环体内的代码。循环每次迭代后，打印出`i`的值，并将`i`减1。运行该函数，可以看到输出结果：

```
5
4
3
2
1
```

`while`循环的特点是重复判断条件，直到条件为假才退出循环。因此，使用`while`循环时，务必确保循环条件能够让循环正常结束，否则会导致无限循环。

### for循环
`for`循环是另一种循环结构，它可以用来遍历集合类型的数据，如数组、集合或其他可迭代对象。`for`循环的语法如下：

```kotlin
for (item in collection) {
   // loop body
}
```

其中，`collection`表示需要遍历的数据集合。对于数组或集合等可迭代对象，可以通过`indices`属性或`withIndex()`函数得到索引及元素。此外，还可以通过函数调用形式直接访问元素，如`array[index]`。

如下示例所示，计算数组元素的平方和：

```kotlin
fun sumOfSquares(array: Array<Int>): Int {
    var result = 0
    for (element in array) {
        result += element * element
    }
    return result
}
```

输出结果：

```
9
```