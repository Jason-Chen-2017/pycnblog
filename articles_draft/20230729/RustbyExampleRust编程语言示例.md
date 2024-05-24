
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2010年 9月 15日 ，Mozilla基金会主席兼CEO Ｌ· 安德鲁 斯坦森（Larry 
Stallman）宣布推出Rust编程语言。 它是一种安全、并发、易于学习的新语言，具有独特的静态类型系统和自动内存管理功能，支持FFI（外部函数接口）。该语言被设计为能够胜任系统编程、底层开发、Web开发、分布式计算等领域。它是2015年最受欢迎的开源语言之一。 
         
Rust by Example 是一本基于Rust编程语言的免费电子书。其中的例子都非常简单，而且每个例子都用了实际应用中常用的一些库或者语法，可以帮助读者理解Rust编程语言在实际项目中的运用。而且，作者还提供了详细的代码注释，让读者可以清晰地了解每段代码的作用，加深理解。 

相对于一般的编程语言教材而言，Rust by Example 更注重从零开始学习Rust语言的方式。通过实际的例子练习，可以更好地掌握Rust编程语言的知识和技巧。本书适合初级到中级 Rust 开发人员阅读，并期望通过自己的实践，增强对Rust编程语言的理解和应用能力。 

 # 2.基本概念术语说明
 
 ## Rust 编程语言
 
Rust 是 Mozilla 基金会开发的一门新型编程语言，设计目的是保证系统软件的安全性、并发性和性能。它拥有独特的安全特性，包括高效的内存管理和运行时检查。它的设计目标是在保证不发生数据竞争的同时，达成系统级别的性能。Rust 已经成为开源社区中的热门语言。

 ## Rust 的主要特征
 
- 内存安全：Rust 通过各种方法来保障内存安全，包括所有权系统、借用检查、生命周期注解、模式匹配等等。
- 线程安全：Rust 可以通过消息传递和共享状态并发执行任务，保证线程安全。
- 可靠性保证：Rust 有着严格的错误处理机制，可帮助发现并修复程序中的错误。
- 并发编程模型：Rust 支持通过无共享内存的actor模型进行并发编程。

 ## 标准库
 
 Rust的标准库中包含了一系列丰富的模块，用于实现各种常见功能。常见的功能如文件处理、网络通信、加密、数据库访问、命令行参数解析、单元测试等。
 
  # 3.核心算法原理和具体操作步骤以及数学公式讲解
  
  # 4.具体代码实例和解释说明
  
  
   # 5.未来发展趋势与挑战
   
   # 6.附录常见问题与解答
   
   
   # 7. Rust by Example - Rust 编程语言示例
   
   作者：David Emberger  
   来源：Medium  
   原文链接：https://medium.com/@deemberger/rust-by-example-f6cbef3d2a2e  
   翻译：Mr.Simple   
   发布时间：2018-09-24  
   
   一名软件工程师正在学习 Rust，但他发现 Rust By Example 是一个非常好的资源，可以帮助 him 或她快速上手 Rust。David 在阅读此书时发现了一个事实，那就是“现代”编程语言通常都有很多陷阱和警告，所以即使是初学者也应当小心翼翼。
   
   1. 安装 Rust

首先，需要安装最新版 Rust，可以从官方网站 https://www.rust-lang.org/downloads 中下载对应平台的安装包安装。Rust 依赖于cargo 构建工具，因此也会自动安装 cargo。

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

   2. 打开 Rust by Example

然后，打开 Rust by Example。

```bash
git clone https://github.com/rust-lang-cn/rust-by-example-cn.git
cd rust-by-example-cn
```

你可以看到一个类似的文件目录结构。

   ```
   ├──Cargo.toml
   ├──LICENSE-APACHE
   ├──README.md
   └──src
       ├──bin
       │   └──hello.rs
       ├──collections
       │   ├──hashmap.rs
       │   └──linkedlist.rs
       ├──flow_control
       │   ├──ifelse.rs
       │   ├──looping.rs
       │   └──whilelet.rs
       ├──functions
       │   ├──arguments.rs
       │   ├──closures.rs
       │   ├──default_arguments.rs
       │   ├──fn.rs
       │   ├──generics.rs
       │   ├──input_output.rs
       │   └──methods.rs
       ├──macros
       │   └──custom_derive.rs
       ├──modules
       │   ├──lib.rs
       │   └──multiple_files.rs
       ├──operators
       │   ├──arithmetic.rs
       │   ├──bitwise.rs
       │   ├──comparison.rs
       │   ├──logical.rs
       │   └──type_conversion.rs
       ├──ownership
       │   ├──borrowing.rs
       │   ├──builders.rs
       │   ├──collections.rs
       │   ├──functions_and_methods.rs
       │   ├──pointers.rs
       │   ├──references_and_borrowing.rs
       │   └──slice.rs
       ├──pattern_matching
       │   ├──binding.rs
       │   ├──destructuring.rs
       │   ├──if_let.rs
       │   ├──match_arm.rs
       │   ├──match_guards.rs
       │   └──underscore_pattern.rs
       ├──primitive_types
       │   ├──arrays.rs
       │   ├──tuples.rs
       │   └──variables.rs
       ├──standard_library_types
       │   ├──cell.rs
       │   ├──string.rs
       │   └──vec.rs
       └──syntax
           ├──attribute.rs
           ├──comments.rs
           ├──errors.rs
           ├──format.rs
           ├──lifetime.rs
           ├──macros.rs
           ├──operators.rs
           ├──paths.rs
           ├──ranges.rs
           ├──strings.rs
           └──tests.rs
   ```


   3. 查找适合你的示例

Rust By Example 提供了许多不同主题和难度的示例，但是可能并不是每个人都能找到自己感兴趣的。你可以根据自己的兴趣或需求浏览相关章节，观看视频演示，并试着编写示例代码。例如，David 想要了解列表和哈希表的基本用法。

如果要编写属于自己的示例代码，则需要创建新的 `.rs` 文件。这里有一个模板：

```rust
// src/my_module.rs

pub fn my_function() -> i32 {
    42
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(my_function(), 42);
    }
}
```

David 创建了一个新的 Rust 模块 `my_module`，其中包含了一个叫做 `my_function` 的函数。然后，为了测试这个函数是否正确工作，他添加了一个简单的测试。David 可以通过运行以下命令来测试函数：

```bash
cargo test my_module::tests::it_works
```

他应该会看到类似这样的输出：

```bash
running 1 test
test my_module::tests::it_works... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

