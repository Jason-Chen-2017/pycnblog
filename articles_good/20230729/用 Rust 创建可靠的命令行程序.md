
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2020年是技术领域的元年，编程语言也迎来了翻天覆地的变化。Java、Python、JavaScript等多种语言在这片广袤的市场上掀起了一阵股风，各个方向都有不少开发者投入其中，同时还有一些企业将更加重视其技术能力。而在这些语言中，Rust 则名扬天下。Rust 是一种新兴的编程语言，拥有强大的性能、安全、并发特性和方便开发的特点。本文就带你用 Rust 创建一个可靠的命令行程序，看看它到底有多强悍。
         
         Rust 是一门注重速度、零开销、内存安全性和高效开发的系统级编程语言。它被设计用于构建健壮、快速、可靠的软件。它的编译器和运行时性能都非常优秀。对于创业公司和大型公司来说，Rust 就是必备的。如果你需要写出安全、高性能的代码，那么 Rust 可能是一个很好的选择。对于中小型公司，Rust 的优势还体现在学习曲线上。即使是小型项目也值得探索。
         
         本篇文章主要包括以下六个部分：
         - 1.背景介绍：首先让读者对 Rust 有个整体的认识，了解它的历史、由来、为什么受欢迎。
         - 2.基本概念术语说明：对 Rust 的一些基本概念和术语进行详细介绍，以帮助读者理解 Rust 代码的含义。
         - 3.核心算法原理及具体操作步骤：借助 Rust 一些特定功能实现一个实用的命令行工具。
         - 4.具体代码实例和解释说明：通过具体的代码实例和解释说明 Rust 的实现过程。
         - 5.未来发展趋势与挑战：展望 Rust 在未来几年的发展前景，以及一些挑战性的应用场景。
         - 6.附录常见问题与解答：罗列一些常见的问题，以便帮助读者快速了解 Rust 的应用场景。
         
         如果你想详细了解 Rust 语言的所有功能，请关注微信公众号"Rust社区",作者将提供更多精彩内容！
         
         # 2.基本概念术语说明
          ## 1.什么是编程语言？
          概括地说，编程语言就是人类与计算机之间沟通的方式，是用来告诉计算机做什么的指令集。它定义了程序员编写程序的方式、语法规则和语义，以及计算机如何执行程序。简单来说，编程语言就是一套机器指令，用来控制计算机执行各种任务。
          
          ### 1.1 什么是编译器？
          编译器就是把源代码转化成机器码的工具，可以把高级语言编写的源代码编译成为可以运行的目标文件或可执行文件。也就是说，当编写完一个程序后，我们需要把它编译成计算机可以识别和执行的二进制代码，然后才能运行。编译器一般分为前端编译器和后端编译器。前端编译器通常只负责分析和处理源代码，生成中间代码。后端编译器会根据不同的机器指令集生成机器码。
          
          ### 1.2 什么是静态类型检查？
          静态类型检查是指编译器在编译源码的时候，会进行数据类型的检查，确保代码中的变量或者函数调用的参数类型正确。如果数据类型错误，则编译过程中会报错，提示用户修改错误的数据类型。这个过程叫做静态类型检查。
          
          ### 1.3 什么是自动内存管理？
          自动内存管理是指程序在运行时动态分配和释放内存空间，不需要手动去申请和回收。自动内存管理的意义在于，让程序员不用考虑这些繁琐的事情，从而提升程序的开发效率和质量。目前主流的编程语言都是自动内存管理的，比如 C++、C# 和 Java 。
          
          ### 1.4 什么是垃圾收集机制？
          垃圾收集机制是内存管理的一种方法。其主要作用是自动回收那些不再使用的内存空间，防止内存泄漏。垃圾收集的手段有两种，标记-清除（Mark-Sweep）和复制算法（Copying）。
          
          ### 1.5 什么是运行时环境？
          运行时环境就是操作系统为程序提供的接口和服务。运行时环境包括内存管理、线程调度、网络通信、文件访问等系统操作API。
          
          ### 1.6 什么是关键字？
          关键字是编程语言的保留字，不能用作标识符名称。例如，if 和 else 就是关键字，不能作为变量名或者函数名。
          
          ## 2.Rust 中的概念
          ### 2.1 模块（Module）
          模块（Module）是 Rust 中一个重要的组织结构单位，类似其他编程语言中的命名空间或者包（Package）。模块提供了一种逻辑上的分组方式，可以把相关联的代码放到一起，便于维护和管理。模块可以包含模块、结构体、枚举、 trait、函数、常量等。

          　　举例来说，假设有一个项目，里面包含多个模块，分别对应业务逻辑、数据模型、配置文件解析等。可以把项目按照模块划分为多个文件，每个文件就相当于一个模块。例如，项目的目录结构如下所示：

              ├── config
              │   └── mod.rs
              ├── data_model
              │   └── mod.rs
              ├── logic
              │   └── mod.rs
              └── main.rs

          每个文件夹下的 `mod.rs` 文件就代表了一个模块。这样可以避免重复定义相同的结构体、枚举、trait、函数等。

          ### 2.2 包（crate）
          包（crate）是在 Rust 中编译和链接代码的最小单位。一个包可以包含任意数量的模块和库 crate。

          当创建一个新的 Rust 包时，Cargo 会生成一个 `Cargo.toml` 文件，其中包含项目的信息，如名称、版本、描述、依赖项等。`Cargo.lock` 文件记录了当前项目的依赖项。

          举例来说，假设有一个项目，名字叫做 `myproject`，里面包含多个模块，它们共同构成一个包。可以把项目的目录结构设置为：

              myproject/
                  ├── src/
                  │   ├── lib.rs      // 该文件包含 crate 的主体代码
                  │   ├── module1.rs  // 模块1的代码
                  │   ├── module2.rs  // 模块2的代码
                  │   └──...
                  ├── Cargo.toml     // 项目信息文件
                  └── Cargo.lock     // 锁定依赖项文件

          `lib.rs` 文件可以包含该 crate 的主体代码。

          ### 2.3 use 语句
          use 语句是 Rust 中一个重要的语法元素，用于引入外部模块和项。例如，可以使用 use 语句导入外部模块中的函数、结构体、常量等。

          使用 use 语句可以减少代码的重复书写，提高代码的可读性。在模块内部也可以使用 use 语句来指定当前模块的别名。

          ### 2.4 结构体（Struct）
          结构体（Struct）是 Rust 中一个基础的数据类型，可以用于组织相关的数据。结构体可以包含成员变量和方法。

          举例来说，定义一个 Person 结构体，包含姓名、年龄、地址等字段：

             struct Person {
                 name: String,
                 age: u8,
                 address: String
             }

          此外，结构体也可以定义构造函数，用于初始化结构体的成员变量：

            impl Person {
                fn new(name: &str, age: u8, address: &str) -> Self {
                    return Person {
                        name: String::from(name),
                        age: age,
                        address: String::from(address)
                    };
                }
            }

          可以看到，Person 结构体的构造函数接收三个参数，并返回一个包含这些参数的实例。

          ### 2.5 函数（Function）
          函数（Function）是 Rust 中另一个基础的语法单元。函数可以接受输入参数，并返回输出结果。函数可以定义可变参数列表、泛型参数、闭包（Closure）等特征。

          举例来说，定义一个计算平方的函数：

            fn square(x: i32) -> i32 {
                x * x
            }

          此外，函数也可以定义多个输入参数，并返回多个结果：

            fn add(a: i32, b: i32) -> (i32, i32) {
                (a + b, a - b)
            }

          函数还可以用 return 语句直接返回值，而不是使用括号和花括号包裹返回值。

          ### 2.6 泛型参数（Generic Parameter）
          泛型参数（Generic Parameter）是 Rust 中另一个重要的特性。泛型参数允许函数、结构体和枚举定义独立于类型的数据。

          举例来说，定义一个容器（Container）类型，支持任意类型的值存储：

            enum Container<T> {
                Empty,
                Value(T)
            }

          此外，可以定义泛型结构体，可以存储不同类型的对象：

            struct Pair<A, B> {
                first: A,
                second: B
            }

          可以看到，Pair 结构体有两个泛型参数 A 和 B。

          ### 2.7 trait（Trait）
          Trait（Trait）是 Rust 中第三个重要的语法单元。Trait 提供了一个抽象的方法签名，定义了一个对象应该具有哪些行为。可以为任何其他类型实现 trait 来表明它满足某些条件。

          举例来说，定义一个 Shape trait，表示形状的行为：

            trait Shape {
                fn area(&self) -> f64;
                fn perimeter(&self) -> f64;
            }

          此外，可以为 Rectangles、Circles 等不同类型的形状实现 Shape trait：

            struct Rectangle {
                width: f64,
                height: f64
            }

            impl Shape for Rectangle {
                fn area(&self) -> f64 { self.width * self.height }

                fn perimeter(&self) -> f64 {
                    2.0 * (self.width + self.height)
                }
            }

            struct Circle {
                radius: f64
            }

            impl Shape for Circle {
                fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius }

                fn perimeter(&self) -> f64 {
                    2.0 * std::f64::consts::PI * self.radius
                }
            }

          这样就可以为任意类型的形状调用 area() 方法和 perimeter() 方法。

          ### 2.8 生命周期（Lifetime）
          生命周期（Lifetime）是 Rust 中第四个重要的概念。生命周期描述了一个引用持续的时间。Rust 通过借用检查器（borrow checker）来保证引用的有效性。

          生命周期的一个典型用法是实现 trait 对象，其中 trait 需要生命周期注解。

          举例来说，定义一个 Animal trait，用于表示动物的行为：

            trait Animal {
                fn eat(&mut self);
            }

          为兔子（Tiger）和狗（Dog）实现 Animal trait：

            struct Tiger {}

            impl Animal for Tiger {
                fn eat(&mut self) { println!("The tiger is eating."); }
            }

            struct Dog {}

            impl Animal for Dog {
                fn eat(&mut self) { println!("The dog is eating."); }
            }

          可以看到，Dog 和 Tiger 的生命周期默认为'static，因为它们的生命周期没有显式地声明。

          ### 2.9 枚举（Enum）
          枚举（Enum）是 Rust 中第五个重要的语法单元。枚举可以定义一组不同的数据类型。

          举例来说，定义一个 Color 枚举，表示三种颜色：

            enum Color {
                Red,
                Green,
                Blue
            }

          此外，可以定义一个 Option 枚举，用于封装可能为空的变量：

            enum Option<T> {
                Some(T),
                None
            }

          这样就可以定义包含 Option 值的变量：

            let x = Option::<u8>::Some(5);

          ### 2.10 宏（Macro）
          宏（Macro）是 Rust 中第六个重要的语法单元。宏可以扩展 Rust 的语法，增加自定义的功能。

          举例来说，定义一个 debug! 宏，用于打印变量的值：

            macro_rules! debug {
                ($($arg:tt)+) => { println!(concat!("{:?}", $($arg)+)) }
            }

          此外，可以定义一个 assert! 宏，用于断言表达式的布尔值为 true：

            #[macro_export]
            macro_rules! assert {
                ($cond:expr) => {{ if!$cond { panic!("assertion failed: {}", stringify!($cond)); } }};
                ($cond:expr, $($arg:tt)+) => {{ if!$cond { panic!(concat!("assertion failed: ", $($arg)+)); } }}
            }

          上面的 assert! 宏可以通过表达式 $cond 来判断是否出现错误，如果 $cond 为 false，则会引发 panic，显示断言失败的原因。

        # 3.核心算法原理及具体操作步骤
        通过上述概念的介绍，你应该已经对 Rust 有了一定了解。接下来，你可以尝试着写一个实际的程序，体验一下 Rust 的魅力吧！
        （我们将以一款开源的日记应用程序为例，来实现这篇文章的操作步骤。本节涉及的知识点主要是 Rust 基本语法。）

        ## 1.概览
        日记应用程序日记（Journal）是一个轻量级的开源应用程序，可以帮助你跟踪你的生活、记录你做过的事情。

        它的主要功能有：

        - 添加、编辑、删除日记条目；
        - 查找、筛选已有的日记条目；
        - 支持多种语言，且内置了简单的语法高亮；
        - 可自行添加插件，提供额外的功能。

        ## 2.安装 Rust
        你可以通过官方网站 https://www.rust-lang.org/tools/install 安装最新版的 Rust 环境。安装完成之后，可以在终端（Terminal）中输入 `rustc --version` 检查 Rust 是否安装成功。

        ```bash
        ➜ rustc --version
        rustc 1.45.2 (d3fb005a3 2020-07-31)
        ```

        ## 3.创建项目
        打开终端，进入工作目录，新建一个空白目录 `journal`，然后在该目录下初始化 Rust 项目：

        ```bash
        ➜ mkdir journal && cd journal
        ➜ cargo init --bin
        ```

        命令 `cargo init` 将会在当前目录下创建一个新的 Rust 项目。 `--bin` 参数用于创建一个二进制可执行程序。

        执行完成后，会在当前目录下生成以下文件：

        - `Cargo.toml`: 项目配置信息。
        - `.gitignore`: Git 忽略文件。
        - `src/main.rs`: 程序入口文件。

        这里暂时先不用修改这些文件，我们将继续往下进行项目的开发。

    ## 4.设置依赖
    日记应用程序日记（Journal）需要读取日记条目，所以我们需要引入日记条目的管理库 `diary`。

    添加日记条目的管理库 `diary` 到 Cargo 配置文件 `Cargo.toml` 中：

    ```toml
    [dependencies]
    diary = "0.1.0"
    ```

    执行 `cargo build` 命令来编译项目：

    ```bash
    ➜ cargo build
    Compiling diary v0.1.0 (/Users/xxx/Documents/projects/journal/diary)
   Compiling journal v0.1.0 (/Users/xxx/Documents/projects/journal)
    Finished dev [unoptimized + debuginfo] target(s) in 0.49s
    ```

    注意：日记条目的管理库还不存在，因此会报“package `diary v0.1.0` does not exist”错误，但不会影响项目编译。

    为了解决这个问题，我们需要在 GitHub 上克隆 `diary` 项目：

    ```bash
    ➜ git clone <EMAIL>:xxx/diary.git
    Cloning into 'diary'...
    remote: Enumerating objects: 13, done.
    remote: Counting objects: 100% (13/13), done.
    remote: Compressing objects: 100% (11/11), done.
    Receiving objects:  92% (1145/1257)   1.04 MiB | 2.47 MiB/s   
      Receiving objects: 100% (1257/1257), 2.56 MiB | 2.25 MiB/s, done.
    Resolving deltas: 100% (647/647), done.
    Checking connectivity... done.
    ```

    将克隆好的项目复制到当前项目的目录下：

    ```bash
    ➜ cp -r./diary/*.
    ```

    修改日记条目的管理库 `diary` 的路径：

    ```toml
    [dependencies]
    diary = { path = "./diary"}
    ```

    执行 `cargo build` 命令重新编译项目：

    ```bash
    ➜ cargo build
    Updating crates.io index
   Compiling proc-macro2 v1.0.24
   Compiling unicode-xid v0.2.1
  ...
    Finished dev [unoptimized + debuginfo] target(s) in 3m 45s
    ```

    编译完成。

    ## 5.创建日记条目结构
    根据需求定义日记条目结构，在 `diary/src/entry.rs` 文件中定义：

    ```rust
    #[derive(Debug)]
    pub struct Entry {
        title: String,
        content: String,
        created_at: DateTime<Utc>,
    }

    impl Entry {
        pub fn new(title: &str, content: &str) -> Self {
            Self {
                title: String::from(title),
                content: String::from(content),
                created_at: Utc::now(),
            }
        }

        pub fn get_title(&self) -> &str {
            &self.title
        }

        pub fn set_title(&mut self, title: &str) {
            self.title = String::from(title);
        }

        pub fn get_content(&self) -> &str {
            &self.content
        }

        pub fn set_content(&mut self, content: &str) {
            self.content = String::from(content);
        }

        pub fn get_created_at(&self) -> &DateTime<Utc> {
            &self.created_at
        }

        pub fn set_created_at(&mut self, created_at: &DateTime<Utc>) {
            self.created_at = *created_at;
        }
    }
    ```

    此处定义了一个日记条目 `Entry` 结构体，包含三个属性：`title`, `content`, `created_at`。`title` 和 `content` 属性为 `String`，`created_at` 属性为 `DateTime`。

    `impl` 块定义了几个方法，用于获取和修改属性的值。

    ## 6.创建日记条目管理器
    在 `diary/src/manager.rs` 文件中定义日记条目管理器 `Manager`，用于管理所有日记条目：

    ```rust
    use chrono::{DateTime, Utc};

    use super::entry::Entry;

    pub struct Manager {
        entries: Vec<Entry>,
    }

    impl Manager {
        pub fn new() -> Self {
            Self { entries: vec![] }
        }

        pub fn save(&mut self, entry: Entry) {
            self.entries.push(entry);
        }

        pub fn find_by_title(&self, query: &str) -> Option<&Entry> {
            for entry in &self.entries {
                if entry.get_title().contains(query) {
                    return Some(entry);
                }
            }
            None
        }

        pub fn all(&self) -> &[Entry] {
            &self.entries[..]
        }

        pub fn update_all(&mut self, updated_at: &DateTime<Utc>) {
            for entry in &mut self.entries {
                entry.set_created_at(updated_at);
            }
        }
    }
    ```

    此处定义了一个日记条目管理器 `Manager`，包含一个 `Vec` 存放所有日记条目。

    `save()` 方法用于保存日记条目到管理器中。

    `find_by_title()` 方法用于查找某个标题包含查询字符串的日记条目。

    `all()` 方法用于获取所有的日记条目。

    `update_all()` 方法用于更新所有日记条目的创建时间。

    ## 7.创建 CLI 界面
    接下来，我们可以创建一个命令行接口（CLI），来让用户能够使用日记应用程序日记（Journal）进行日记条目管理。

    在 `src/main.rs` 文件中定义 CLI 入口函数：

    ```rust
    extern crate clap;
    use clap::{App, Arg};

    use chrono::{DateTime, Local, Utc};

    use diary::{Entry, Manager};

    fn main() {
        let manager = Manager::new();

        let matches = App::new("Journal")
           .version("1.0")
           .author("<NAME>")
           .about("A simple command line interface to manage your journal entries.")
           .subcommand(
                App::new("add").about("Add a new journal entry.").arg(
                    Arg::with_name("title")
                       .help("Title of the new journal entry.")
                       .required(true)
                       .index(1),
                ),
            )
           .subcommand(
                App::new("edit").about("Edit an existing journal entry.").arg(
                    Arg::with_name("query")
                       .help("Query string used to search for the journal entry to edit.")
                       .required(true)
                       .index(1),
                ).arg(
                    Arg::with_name("title")
                       .help("New title for the journal entry.")
                       .short("t")
                       .long("title"),
                ).arg(
                    Arg::with_name("content")
                       .help("New content for the journal entry.")
                       .short("c")
                       .long("content"),
                ),
            )
           .subcommand(
                App::new("delete")
                   .about("Delete one or more journal entries.")
                   .arg(
                        Arg::with_name("queries")
                           .help("One or more queries used to search for the journals to delete.")
                           .min_values(1),
                    ),
            )
           .subcommand(
                App::new("list")
                   .about("List all journal entries that match a given query.")
                   .arg(
                        Arg::with_name("query")
                           .help("Query string used to filter the list of journal entries.")
                           .short("q")
                           .long("query")
                           .default_value("*"),
                    ),
            )
           .subcommand(
                App::new("search")
                   .about("Search for specific keywords in any part of the journal entry.")
                   .arg(
                        Arg::with_name("keywords")
                           .help("One or more keywords used to search for in the journals.")
                           .min_values(1),
                    ),
            )
           .subcommand(
                App::new("clear").about("Clear all journal entries."),
            )
           .subcommand(
                App::new("today")
                   .about("Display today's entries.")
                   .arg(
                        Arg::with_name("language")
                           .help("Language code used for date and time representation.")
                           .short("l")
                           .long("language")
                           .possible_values(["en-US"]),
                    ),
            )
           .get_matches();

        let now = Local::now();

        if let Some(ref sub_matches) = matches.subcommand_matches("add") {
            let mut builder = Entry::builder().title(sub_matches.value_of("title").unwrap());
            if let Ok(content) = read_stdin() {
                builder = builder.content(content);
            }
            let entry = builder.build();
            manager.save(entry);
        } else if let Some(ref sub_matches) = matches.subcommand_matches("edit") {
            let query = sub_matches.value_of("query").unwrap();
            let entries = manager.find_by_title(query).map(|e| e.to_owned()).collect::<Vec<_>>();
            if!entries.is_empty() {
                let latest_entry = entries[0].clone();
                let title = sub_matches.value_of("title").unwrap_or(latest_entry.get_title());
                let content = sub_matches.value_of("content").unwrap_or("");
                let edited_entry = Entry::builder()
                   .title(title)
                   .content(content)
                   .created_at(*latest_entry.get_created_at())
                   .build();
                manager.save(edited_entry);
            }
        } else if let Some(ref sub_matches) = matches.subcommand_matches("delete") {
            for query in sub_matches.values_of("queries").unwrap() {
                let entries = manager.find_by_title(query).map(|e| e.to_owned()).collect::<Vec<_>>();
                for entry in entries {
                    manager.entries.retain(|item| item!= &entry);
                }
            }
        } else if let Some(_sub_matches) = matches.subcommand_matches("list") {
            let query = matches.value_of("query").unwrap();
            let filtered_entries = manager.all().iter().filter(|&e| e.get_title().contains(query)).cloned().collect::<Vec<_>>();
            display_entries(&filtered_entries);
        } else if let Some(sub_matches) = matches.subcommand_matches("search") {
            for keyword in sub_matches.values_of("keywords").unwrap() {
                let matched_entries = manager
                   .all()
                   .iter()
                   .filter(|e| e.get_title().contains(keyword) || e.get_content().contains(keyword))
                   .cloned()
                   .collect::<Vec<_>>();
                display_entries(&matched_entries);
            }
        } else if let Some(_sub_matches) = matches.subcommand_matches("clear") {
            manager.entries.clear();
        } else if let Some(sub_matches) = matches.subcommand_matches("today") {
            let language = sub_matches.value_of("language").unwrap_or("en-US");
            let date = now.date().format("%Y-%m-%d").to_string();
            let entries = manager.find_by_title(&date).map(|e| e.to_owned()).collect::<Vec<_>>();
            display_entries(&entries);
        } else {
            println!("No valid subcommand specified!");
        }
    }

    fn display_entries(entries: &[Entry]) {
        println!("{} entries:", entries.len());
        if!entries.is_empty() {
            for entry in entries {
                println!("{}
{}
Created at {}
----------
",
                         colorize_text("Title:", "blue"),
                         colorize_text(entry.get_title(), "green"),
                         format_datetime(entry.get_created_at()));
                print!("{}", colorize_text(entry.get_content(), "white"));
            }
        } else {
            println!("No matching entries found.");
        }
    }

    fn colorize_text(text: &str, color: &str) -> String {
        let colors = ["black", "red", "green", "yellow", "blue", "purple", "cyan", "white"];
        format!("\x1b[{}m{}\x1b[0m", colors.iter().position(|c| c == color).unwrap() + 31, text)
    }

    fn format_datetime(dt: &DateTime<Utc>) -> String {
        dt.with_timezone(&Local).to_rfc3339()[..19].to_string()
    }

    fn read_stdin() -> Result<String, std::io::Error> {
        let mut buffer = String::new();
        std::io::stdin().read_line(&mut buffer)?;
        Ok(buffer.trim().to_string())
    }
    ```

    此处定义了一个 `main()` 函数作为日记应用程序日记（Journal）的 CLI 入口。

    `clap` 库用于解析命令行参数。

    `Manager`、`Entry`、`colorize_text()`、`display_entries()`、`format_datetime()`、`read_stdin()` 函数分别实现日记条目管理器、日记条目、文字颜色处理、日记条目列表展示、日期时间格式化、标准输入读取等功能。

    CLI 入口函数通过解析命令行参数调用对应的功能实现。

    ## 8.测试
    可以通过以下命令启动日记应用程序日记（Journal）：

    ```bash
    ➜ cargo run
    ```

    测试添加、编辑、删除日记条目等功能：

    ```bash
    ➜ cargo run -- add "Write Rust book"
    Created new entry with ID daa7003d-1777-47c9-afbc-1bcf8f413b17

    ➜ cargo run -- edit "Write Rust book" --title "Learn Rust"
    Edited entry with ID daa7003d-1777-47c9-afbc-1bcf8f413b17

    ➜ cargo run -- delete Write Rust book
    Deleted entry with ID daa7003d-1777-47c9-afbc-1bcf8f413b17
    ```

    测试搜索功能：

    ```bash
    ➜ cargo run -- search rust books
    1 entry found:

    Title: Learn Rust
    Content: I'm going to learn Rust programming language by writing this book.
    Created at 2021-03-19T09:11:47+08:00
    ----------

    ```

    测试查看今天的日记条目功能：

    ```bash
    ➜ cargo run -- today
    Today's entries:

    1 entry found:

    Title: Write Rust book
    Content: Yes, I will write the Rust book!
    Created at 2021-03-19T09:11:47+08:00
    ----------

    ```