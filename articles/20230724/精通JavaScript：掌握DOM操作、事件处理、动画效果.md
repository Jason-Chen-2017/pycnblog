
作者：禅与计算机程序设计艺术                    

# 1.简介
         
jQuery是一个很流行的轻量级JavaScript框架，许多网站都在使用它。随着前端技术的发展，越来越多的人开始学习JavaScript开发，但要学好JavaScript还需要一定的时间。因此，很多初学者望而却步，认为JavaScript只是一种编程语言而已，难于进阶。不过，我们也看到了JavaScript的强大功能以及其应用场景的广泛性，越来越多的公司和组织都把JavaScript作为主要的编程语言。所以，掌握JavaScript技巧将成为一个不错的选择。
本书首先介绍了JavaScript的历史、组成及特点；然后介绍了DOM操作、事件处理、动画效果的基础知识；接着深入分析了各种数据结构的实现方法，包括数组、链表、散列表等；最后，通过实际案例带领读者了解和理解这些知识。
# 2.基本概念术语说明
## 2.1 JavaScript概述
JavaScript（简称JS）是一门高级脚本语言，用来给网页添加动态交互功能。由于其轻量级、灵活性、跨平台特性，已经成为网络开发中不可替代的一部分。全世界有上百万个网站使用JavaScript编写客户端程序，如YouTube、Facebook、Google Maps等。同时，JavaScript也是开源项目Node.js的运行环境。所以，掌握JavaScript技术对于开发人员来说尤其重要。
### 2.1.1 相关术语
- ECMAScript: 是JavaScript的规范标准，由国际标准化组织Ecma International制定，并通过ECMA-262这一版正式发布。ECMAScript定义了浏览器脚本语言的基本语法和基本对象。它由一些文档和规格组成，如：
    - ECMA-262: 定义了JavaScript的语法、类型系统和对象模型。
    - ECMA-402: 提供了用于处理数字、日期、及其他区域设置特征的实用工具。
    - ES6/ES2015: 即使版本号不同，ES6和ES2015都是JavaScript的最新规范。它增加了对模板字符串、类、模块、迭代器、生成器、代理、反射和尾调用的支持。
- DOM(Document Object Model): 是W3C组织推荐的处理可视化文档的API。它定义了如何从页面中获取、修改、添加或删除元素，以及如何监听事件。通过DOM，可以更方便地操作HTML和XML文档。
- BOM(Browser Object Model): 是W3C组织针对浏览器窗口对象的模型标准。它提供了访问浏览器窗口属性和行为的方法，例如：open()、setTimeout()等。BOM也提供了操纵cookies、本地存储和全局状态的方法。
- Ajax(Asynchronous JavaScript and XML): 是一种使用 XMLHttpRequest 对象通讯的网络技术。它允许在后台线程中向服务器请求数据，并在接收到响应时更新页面的部分内容，而无需重新加载整个页面。
- JSON(JavaScript Object Notation): 是一种轻量级的数据交换格式。它采用键值对的形式，并具有简单的数据结构。JSON是基于文本的，易于解析，适合于web应用程序间的通信。
- Node.js: 是一个基于Chrome V8引擎的JavaScript运行环境。它使用异步I/O模型，充分利用多核优势，非常适合搭建分布式计算集群。
- npm(Node Package Manager): 是Node.js官方的包管理工具。它可以管理前端组件、后端服务和命令行工具。npm仓库里包含了大量开源库，你可以直接下载使用或者开发自己的插件。
- AMD(Asynchronous Module Definition): 是JavaScript的模块定义规范。它提供声明依赖关系的机制，支持异步加载模块。
- CMD(Common Module Definition): 是Sea.js的模块定义规范。它与AMD类似，但是使用define()函数来定义模块，而不是require()函数。
- gulp: 是一个自动构建工具，能够优化前端资源，比如压缩、合并、编译等。它借助配置文件可以完成诸如LESS转CSS、压缩图片等一系列任务。
- Webpack: 是JavaScript静态模块打包工具。它把各种模块按照依赖关系进行连接，最终生成一个复杂的、可加载的前端资源。Webpack基于AMD模块化标准，使得编写模块化的代码更加简单。
- Babel: 是一款JavaScript转换器，它可以将新一代的JavaScript代码转换为旧浏览器可以识别的、有效的ECMAScript代码。Babel可以让你不用关心浏览器兼容性，直接用现代的JavaScript编码方式去写代码即可。
- jQuery: 是一套轻量级的JavaScript框架。它提供了一系列的函数和方法，简化了前端开发。
- AngularJS: 是Angular Framework的第一个版本。它实现了MVC模式，通过指令扩展了HTML语法，帮助开发者快速创建动态应用。
- React: 是Facebook推出的JavaScript界面库。它采用虚拟DOM，将页面渲染效率提升到了一个新高度。
- Vue.js: 是一款渐进式的JavaScript框架。它采用简洁的语法，可与任何第三方库组合使用。
- TypeScript: 是JavaScript类型的超集。它提供了类型检查、接口、枚举、注解等功能。TypeScript可以在编译期检测出错误，避免运行时异常。
- Closure Compiler: 是JavaScript代码压缩工具。它利用静态分析、类型信息、内联变量、条件语句重写等优化手段减少代码体积。
- ESLint: 是一款JavaScript代码质量检测工具。它可以查找代码中的错误、规约，并提醒开发者改善代码风格和质量。
- Mocha: 是一种JavaScript测试框架，可以执行单元测试、集成测试、浏览器测试等。
- Jest: 是Facebook推出的JavaScript测试框架，可以检测代码的缺陷、性能瓶颈和逻辑错误。
- Karma: 是另一个JavaScript测试框架。它可以集成Mocha，实现端到端的自动化测试。
- Enzyme: 是Airbnb推出的React测试框架。它可以测试组件的渲染结果、事件处理、数据流、样式等。

