
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Rust 是一种现代、快速、安全且可靠的系统编程语言，它的设计目的是为 Web 应用、命令行工具和底层系统开发提供一个良好的语言环境。Rust 的编译器具有惊人的性能，并且支持函数式编程，但由于它在某些方面仍然处于初级阶段，因此仍存在一些学习曲线和陡峭的学习曲线，这让许多新手望而却步。本教程旨在帮助广大 Rust 爱好者快速入门并掌握其基本用法。我将从以下三个方面讲述 Rust 的基础知识：

1. 变量和数据类型
2. 条件语句
3. 循环结构

我会结合具体案例介绍 Rust 语言的基本语法和各种特性。希望可以帮助大家加速了解 Rust 语言，提升编程技能和能力！
# 2.核心概念与联系
## 变量和数据类型
Rust 中的变量可以保存不同类型的信息，Rust 提供了以下几种基本的数据类型：

- 整数类型(i8, i16, i32, i64, u8, u16, u32, u64)
- 浮点型类型(f32, f64)
- 布尔型类型(bool)
- 字符类型(char)
- 元组类型(tuple)
- 数组类型([type; size])
- 切片类型(&[type])
- 指针类型(*const type, *mut type)

每个变量都需要明确指定它的类型，同时还可以对变量进行常量或者不允许修改的限制。
```rust
//声明变量
let x: i32 = 10; //整型变量
let y: char = 'a'; //字符型变量
let z: bool = true; //布尔型变量

//常量
const MAX_SIZE: usize = 10; //无符号整型常量

//不可变性
let mut a: i32 = 10; //mutable 可变性
a = 20; //改变值
```
Rust 的函数参数也可以通过模式匹配的方式赋予默认值，这样就可以方便地调用函数。
```rust
fn print_hello(name: &str, age: Option<u8> = None) {
    println!("Hello {}! Your age is {:?}.", name, age);
}
print_hello("Alice"); // 打印 "Hello Alice! Your age is None."
print_hello("Bob", Some(25)); // 打印 "Hello Bob! Your age is Some(25)."
```
对于集合类型的变量来说，Rust 提供了一系列的操作方法，如集合的创建、遍历、查询等。
```rust
//创建集合
let numbers: Vec<i32> = vec![1, 2, 3]; //向量
let characters: [char; 4] = ['a', 'b', 'c']; //数组

//遍历集合
for number in numbers.iter() {
    println!("{}", number);
}

//查找元素
if let Some(_) = numbers.iter().find(|&x| *x == 2) {
    println!("Found!");
} else {
    println!("Not found.");
}

//映射
numbers.into_iter().map(|x| x + 1).collect::<Vec<_>>();
characters.iter().cloned().collect::<String>();
```
## 条件语句
Rust 中提供了 if 和 match 两种条件判断语句，它们之间的区别如下：

- if 可以用于简单条件判断，当满足条件时执行代码块；
- if...else 可以用于复杂的条件判断，根据条件选择对应的代码块执行；
- match 可以用于多分支条件判断，它比 if...else 更加强大，能够匹配各种不同的情况。
```rust
if condition {
    // code block to be executed when condition is true
}

if condition {
    // code block for if branch
} else {
    // code block for else branch
}

match expression {
    pattern => result, // code block for first matching arm
   ...
    _ => default_result // optional default case (matches any other value)
}
```
match 支持所有的 Rust 数据类型和表达式，包括用户自定义类型。
```rust
struct Color { r: u8, g: u8, b: u8 }
enum Direction { North, East, South, West }

fn main() {
    let color = Color{r: 255, g: 0, b: 0};
    
    match color {
        Color{r, g, b}: if r > 127 && g < 127 && b < 127 {
            println!("This is dark!")
        } else {
            println!("This is light.")
        },
        Direction::North => println!("Go north"),
       .. => println!("All other cases")
    };

    fn add_one(n: Option<&i32>) -> Option<i32> {
        n.map(|&x| x + 1)
    }
    
    assert_eq!(add_one(Some(&3)), Some(4));
    assert_eq!(add_one(None), None);
}
```
## 循环结构
Rust 有三种循环结构，分别是 for 循环、while 循环和 loop 循环。for 循环比较常用，也比较直观易懂。

while 循环可以基于某个表达式的值进行循环，只要这个表达式的值为 true，循环就会一直进行下去。

loop 循环是一个无限循环，它的循环体中必须包含 break 或 continue 来退出循环。
```rust
for num in 0..5 {
   println!("{}", num);
}

let mut count = 0;
while count < 5 {
   println!("{}", count);
   count += 1;
}

let mut counter = 0;
loop {
   println!("{}", counter);
   counter += 1;

   if counter >= 5 {
       break;
   }
}
```