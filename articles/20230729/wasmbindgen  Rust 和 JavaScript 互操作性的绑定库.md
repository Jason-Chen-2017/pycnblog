
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         wasm-bindgen 是一款由 Rust 编写的工具链，可以将 Rust 函数导出到 JavaScript 中，让 JS 可以调用 Rust 代码。它使得 Rust 语言可以在浏览器、Nodejs 和其他支持 WebAssembly 的环境中运行，并提供了类型安全和易用性。目前它的最新版本是 0.2.39。本文就wasm-bindgen做一下简单的介绍。
         ## 背景介绍
         Rust 是一个开源系统编程语言，由 Mozilla 基金会开发，2010 年正式推出。Rust 提供了现代化的内存安全保证、线程安全、无数据竞争的多线程模型、以及安全的抽象机制，使得编写健壮、可靠且高效的代码成为可能。同时，Rust 有着丰富的生态系统，其中最突出的就是包管理器 cargo 。
         一旦编写完成后，Rust 可编译成本地可执行文件或生成机器码，但部署到生产环境时，往往需要一个中间层，比如 Apache HTTP Server 或 Nginx ，才能真正运行起来。而在这个过程中，语言之间的交互，尤其是对运行时的动态链接库（DLL）依赖项的处理，就显得尤为重要。wasm-bindgen 就是这样的一款工具链，它可以通过 proc-macro 来实现对 Rust 代码的编译，将 Rust 函数转换为类似于 C API 的绑定代码，然后再通过 emscripten 把这些绑定代码编译成 asm.js 或 WebAssembly 模块，最后把模块导入到浏览器环境中进行动态链接。
         ## 安装wasm-bindgen
        ```sh
        $ cargo install wasm-bindgen-cli
        ``` 
        如果安装成功，应该能够看到如下信息输出：
        ```sh
        info: downloading wasm-bindgen-cli v0.2.39
       ...
        info: installing component 'wasm-bindgen' for 'wasm32-unknown-emscripten'
        ```
        ## 用法
        使用 wasm-bindgen 时，通常需要编写两个 Rust 文件，一个用于定义 Rust 中的函数接口，另一个用于使用 wasm_bindgen::wrap 来标记外部调用的 Rust 函数，并生成 JS 对接文件。首先，Cargo.toml 需要增加以下依赖项：
        ```toml
        [dependencies]
        wasm-bindgen = "0.2"
        
        [lib]
        crate-type = ["cdylib"]
        ```
        在 lib 项下指定 crate-type 为 cdylib，因为要生成一个动态链接库文件，用于在浏览器中加载。注意这里一定要设定正确的 crate-type，否则不能正确构建wasm绑定库。
        ### Rust 代码定义函数接口
        在 src/lib.rs 中，首先定义需要暴露给 JS 的函数。
        ```rust
        #[wasm_bindgen]
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }
        ```
        通过 wasm_bindgen 属性标注该函数，表明它要导出到 JS 环境。add 函数接受两个 i32 参数，返回值为 i32 类型的值。
        ### Rust 代码引用 JS 函数
        当 wasm-bindgen 生成绑定代码时，默认情况下会生成类似于 C API 的接口。如果想访问 JS 对象中的方法，可以使用 js! macro 来调用。
        ```rust
        #[wasm_bindgen]
        pub fn print() {
            let console = web_sys::window().unwrap().console();
            console.log_1(&JsValue::from("Hello from Rust!"));
        }
        ```
        此函数声明了一个名为 window 的变量，并获取其中的 console 对象。然后调用 log 方法打印一段消息到控制台。注意，这种方式只能调用全局作用域下的对象。如果想访问某个元素，例如 document.body 中的属性，则需要先获取该元素的句柄，再进行相关操作。
        ### Rust 代码构建绑定
        在项目根目录下运行如下命令：
        ```sh
        $ wasm-bindgen --out-dir. --target no-modules src/lib.rs
        ```
        上面的命令指定生成绑定文件的目标路径，不生成模块（即把所有导出的 Rust 函数打包到一个文件），还指定了 Rust 源文件位置。默认情况下，生成的文件名为 lib.js ，放在与 Rust 源文件同级的目录下。
        ### JS 端调用 Rust 函数
        在 HTML 文件中，引入刚才生成的 lib.js 文件。
        ```html
        <script type="module">
            import init, * as rust from './lib.js';

            async function run() {
                await init('./lib_bg.wasm');

                const result = rust.add(1, 2);
                console.log(`The sum is ${result}`);
            }
            
            run();
        </script>
        ```
        通过 import 命令导入了 Rust 函数所在的模块，并通过 async 函数初始化 wasm 模块，调用 add 函数求和并打印结果。注意这里的路径参数应当是指向 Rust 生成的 wasm 模块文件路径，而不是 lib.js 。