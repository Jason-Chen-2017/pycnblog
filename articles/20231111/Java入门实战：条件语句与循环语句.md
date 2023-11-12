                 

# 1.背景介绍


在计算机编程中，条件语句和循环语句是经常使用的基本结构。对于初级的程序员来说，掌握它们十分重要。本文将从基础知识、应用场景、语法和注意事项等方面进行介绍。
# 2.核心概念与联系
## 2.1 条件语句（Conditional Statement）
条件语句用于基于某种条件执行或不执行某段代码。它可以根据是否满足某个特定的条件，来决定是否执行或跳过某段代码。
### if语句
if语句的基本语法如下：
```java
if(booleanExpression){
   //true执行的代码
}else{
   //false执行的代码
}
```
- booleanExpression 表示判断表达式，是一个布尔表达式，其值可能为true或者false。如果该表达式的值为true则执行`true执行的代码`，否则执行`false执行的代码`。
- `// true执行的代码`和`// false执行的代码`是可选的，可以省略。
- 当多个条件语句需要共用一块代码时，可以使用`if…else if…else`语句：
```java
if (condition1) {
    // condition1 is true, execute this code block
} else if (condition2) {
    // condition2 is true, execute this code block
} else if (condition3) {
    // condition3 is true, execute this code block
} else {
    // none of the conditions are true, execute this code block
}
```
- 在上面的例子中，只有`condition1`为真时才会执行第一组代码；只有`condition1`、`condition2`都为假时才会执行第二组代码；只有`condition1`、`condition2`、`condition3`都为假时才会执行第三组代码；其他情况均不会执行任何代码。
## 2.2 循环语句（Looping Statement）
循环语句用于重复执行某段代码多次。它会根据指定的条件（比如循环次数），让控制流进入到循环体内，然后反复执行该段代码，直至条件达到结束条件为止。
### for循环
for循环的基本语法如下：
```java
for (initialization; condition; increment/decrement) {
    // loop body executed repeatedly while condition is true
}
```
- initialization 是初始化表达式，可以在循环开始前进行一次性赋值。通常会声明一些变量用来记录循环次数。例如：`int i = 0;` 。
- condition 是循环条件，表示循环是否继续。当此表达式的值为false时，循环终止。例如：`i < n` ，表示循环运行n次。
- increment/decrement 是更新表达式，是在每次循环迭代后对计数器变量进行递增或递减的一段代码。如：`i++` 或 `i--` 。
- loop body 是循环体，表示将要重复执行的代码。循环体一般包括一个或多个语句，这些语句将在每一次迭代中执行。
### do-while循环
do-while循环是一种特殊的循环语句，它首先执行一次代码块，然后检查循环条件，如果条件为true，则再次执行代码块，直至条件为false。它的基本语法如下：
```java
do {
    // loop body executed at least once
} while (booleanExpression);
```
- booleanExpression 表示循环条件，其值可能会改变。
- 如果booleanExpression在第一次循环之前就已经为false，那么循环体永远不会被执行。因此，这种循环不能保证至少执行一次。
### while循环
while循环的基本语法如下：
```java
while (booleanExpression) {
    // loop body executed repeatedly until condition becomes false
}
```
- booleanExpression 表示循环条件，其值可能改变。
- 如果booleanExpression在第一次循环之前就已经为false，那么循环体永远不会被执行。因此，这种循环也不能保证至少执行一次。