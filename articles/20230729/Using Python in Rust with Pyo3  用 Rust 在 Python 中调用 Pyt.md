
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 ## 什么是Pyo3?
          Pyo3 是 Rust 和 Python 之间的绑定库,它允许在 Rust 程序中调用 Python 的对象。你可以用 Pyo3 来扩展 Rust 程序或者构建强大的 Python 相关工具。
          ### 为什么要使用Rust来实现Python调用？
          - Python 是一种非常流行的语言，占据着数据分析、机器学习、Web开发等领域的中心地位。
          - Python 的运行效率高，而且支持多种编程范式，包括面向对象的编程、函数式编程和命令式编程。
          - Python 有丰富的第三方库，可以帮助我们解决各种编程问题，提升工作效率。
          - Rust 是 Mozilla Firefox 浏览器、俄罗斯方块等知名应用的基础编程语言。
          - Rust 有活跃的社区，有很多成熟的库可以帮助我们解决日常生活中的复杂问题。
          - Pyo3 可以让我们在 Rust 中方便地调用 Python 中的对象，进而实现一些很酷的功能。
          ### 安装Pyo3
          ```bash
          cargo install pyo3
          ```
          或
          ```bash
          pip install PyO3
          ```
          ### 使用Pyo3
          #### Hello World
          首先，创建一个新项目，Cargo.toml 文件增加以下依赖：
          ```rust
          [dependencies]
          pyo3 = { version = "0.14", features = ["extension-module"] }
          ```
          创建一个叫做 rust_python 的模块文件，里面有一个 hello 函数：
          ```rust
          use pyo3::prelude::*;
          
          fn main() -> Result<(), PyErr> {
              let gil = Python::acquire_gil();
              let py = gil.python();
              
              println!("Hello, Python!");
              
              Ok(())
          }
          ```
          此时，运行这个 crate 会报错：
          ```
          error[E0433]: failed to resolve: could not find `ffi` in the list of imported crates
           --> src/lib.rs:17:9
            |
         17 |     use pyo3::ffi;
            |         ^^^^ could not find `ffi` in the list of imported crates
          ```
          报错原因是因为 pyo3 模块中没有 ffi 模块。需要通过 features 参数开启 extension-module 功能，再次修改 Cargo.toml 文件：
          ```rust
          [dependencies]
          pyo3 = { version = "0.14", features = ["extension-module"] }
          ```
          执行 cargo build 命令后，会看到编译成功消息。此时，可以看到输出内容是“Hello, Python!”。这里只是简单地打印了一条信息，接下来我们尝试调用 Python 对象。

          #### 通过 Pyo3 框架访问 Python 对象
          下面，我们会演示如何在 Rust 中访问 Python 对象，并调用方法和属性。在上面的代码基础上，我们可以加入以下代码来创建 Python 对象：
          ```rust
          #[pyfunction]
          fn create_list(py: Python) -> &PyList {
              let my_list = vec![1, 2, 3];
              let my_obj = py.eval("[1, 2, 3]", None, None)?; // 通过 eval 函数创建列表
              return PyList::new(py, my_list); // 返回 Python List 对象
          }
          ```
          上述代码定义了一个名为 create_list 的 Rust 函数，该函数接收一个 Python 对象作为参数。然后，通过 py.eval 函数执行 Python 语句 "[1, 2, 3]" ，创建了一个列表。最后，通过 PyList::new 方法将 Rust 的 Vec<i32> 对象转换成 Python 的列表对象。注意到我们使用了一个宏 #[pyfunction] ，它表示这是一个 Python 函数，这样 Rust 可以自动将其转换成 Python 可调用对象。
          如果你想要获取列表的长度，可以使用 len 函数：
          ```rust
          #[pyfunction]
          fn get_length(list: &PyList) -> u64 {
              return list.len().unwrap();
          }
          ```
          获取列表的第一个元素也可以使用索引语法：
          ```rust
          #[pyfunction]
          fn get_first_element(list: &PyList) -> i32 {
              return *list.get(0).unwrap().extract::<i32>().unwrap();
          }
          ```
          最后，我们可以通过以下方式调用这些 Rust 函数：
          ```rust
          #[pymodule]
          fn rust_python(_py: Python, m: &PyModule) -> PyResult<()> {
              m.add_wrapped(wrap_pyfunction!(create_list))?;
              m.add_wrapped(wrap_pyfunction!(get_length))?;
              m.add_wrapped(wrap_pyfunction!(get_first_element))?;

              Ok(())
          }
          ```
          在这里，我们定义了一个名为 rust_python 的 Python 模块，并添加了三个函数作为子模块，这些函数分别对应于 Rust 中相应的函数。

          #### 保存并导入模块
          为了能够在其他 Rust 程序中使用刚才编写的模块，我们需要先编译成动态链接库。运行如下命令编译：
          ```bash
          cargo build --release
          ```
          编译完成后，在 target/release 文件夹下找到.so 文件（Windows平台则是 dll 文件）。复制到你想使用的项目目录下即可。另外，还需要把模块注册到 Python 中，如下所示：
          ```python
          import sys
          from pathlib import Path
          so_path = str((Path(__file__).parent / "target" / "release").absolute()) + "/librust_python.so"
          if so_path not in sys.path:
              sys.path.append(str(Path(".").resolve()))
      
          import rust_python as rs
      ```
      上述代码将当前文件夹（也就是 Python 脚本所在文件夹）设置为 Python 环境变量，并将动态链接库路径加到系统 PATH 中。然后，引入之前编写的模块 rs。

      #### 小结
      本文主要介绍了 Pyo3 的使用方法，包括安装、使用示例及使用过程中的注意事项。Pyo3 是 Rust 和 Python 之间最佳搭档，可以为 Rust 带来强大的 Python 生态特性。希望本文对您有所帮助！