
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Rust 是一门现代、高效、安全的系统编程语言，拥有较高的运行速度和内存效率。它同时兼具 C 和 Java 的高级特性，并能轻松调用底层系统接口，支持面向对象编程等独特功能。因此，越来越多的企业选择 Rust 来开发基于云服务的大型软件系统。

本教程将通过深入浅出地讲解 Rust 中的条件语句（if-else）和循环结构（for loop 和 while loop），帮助读者熟悉 Rust 编程中的基本语法和逻辑结构。课程假设读者已经掌握了常用编程语言的基本语法、变量、函数等基本概念，并对计算机程序运行原理有一定了解。除此之外，不会涉及太多的编程技巧或工具的使用，希望能够让读者更容易上手 Rust。

# 2.核心概念与联系
## 2.1 条件语句
条件语句是指根据某种条件执行不同的代码分支。在 Rust 中，有三种条件语句：`if-else`，`match`，`while`。下面依次介绍它们的特性。
### 2.1.1 if-else 语句
```rust
if condition {
    // code to be executed when the condition is true
} else if other_condition {
    // alternative code to be executed when the first condition fails
} else {
    // final code block to be executed when all conditions fail
}
```
if-else 语句是最基本的条件语句。当满足某个条件时，就会执行对应的代码块；否则，如果还有其他的条件可以判断，则会进行相应的判断。

### 2.1.2 match 语句
```rust
let x = 5;

match x {
    1 => println!("x is equal to 1"),
    2 | 3 | 4 => println!("x is between 2 and 4 (inclusive)"),
    _ => println!("x is something else")
}
```
match 语句是一个多分枝匹配语句。它的功能类似于 switch case，根据分支条件不同，执行不同的代码块。这里有一个示例程序，根据 `x` 的值，打印不同的消息。

match 语句的一个重要特征是其能够处理所有类型的数据，甚至可以用来处理枚举。例如，你可以定义一个枚举类型，然后用 match 处理该类型的不同值。

```rust
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25
    }
}

println!("{}", value_in_cents(Coin::Nickel)); // Output: 5
```

在这个例子中，我们定义了一个枚举类型 `Coin`，代表硬币种类。然后，我们定义了一个函数 `value_in_cents`，接受一个 `Coin` 参数，返回其面额（以分为单位）。我们用 match 来处理每一种硬币的面额。

### 2.1.3 while 语句
```rust
let mut count = 0;

while count < 10 {
    print!("{} ", count);
    count += 1;
}

println!();
```
while 语句也称为迭代语句。它允许我们重复执行一段代码，直到某个条件变为 false 为止。这里有一个示例程序，利用 while 实现从 0 到 9 的输出。

## 2.2 循环结构
Rust 提供两种循环结构：`for loop` 和 `while loop`。两者都可以用于遍历集合或数组的元素，或者执行指定的次数。

### 2.2.1 for loop
```rust
for i in 1..=10 {
    println!("{}", i);
}
```
for loop 可以用于遍历任何可迭代类型（例如，数组、集合等），也可以用于迭代数字范围。这里有一个示例程序，使用 for loop 从 1 到 10 输出每个数字。

for loop 的另一个常用的功能是逆序遍历一个集合。你可以这样做：
```rust
for &i in &[1, 2, 3].iter().rev() {
    println!("{}", i);
}
// Output: 3 2 1
```

`.iter()` 方法创建一个迭代器，`.rev()` 方法反转其顺序。因此，这里首先遍历 `[1, 2, 3]`，然后再倒着遍历它们。

### 2.2.2 while loop
```rust
let mut num = 0;

while num <= 10 {
    num += 1;
    if num % 2 == 0 {
        continue; // skip even numbers
    }

    println!("{}", num);
}
```
while loop 可用于循环指定次数，也可以用于执行某个条件的循环。这里有一个示例程序，利用 while loop 输出 0 到 10 的偶数。程序首先初始化 `num` 为 0，然后进入一个无限循环。对于每次迭代，程序都会检查 `num` 是否为偶数，如果不是，则会跳过输出这一行。对于偶数，程序直接打印出来。

## 2.3 小结
本章介绍了 Rust 中的条件语句、循环结构以及它们之间的关系。在下一章中，将深入讨论控制流。