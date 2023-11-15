                 

# 1.背景介绍


前言：随着移动互联网、Web前端、后端开发等领域的日渐发展，越来越多的人开始对React和 Redux Thunk 有了兴趣。然而，对于刚接触React的同学来说，掌握这些知识并应用到实际项目中仍存在一些困难，因此，本文将从零开始带领大家学习React的基本用法和异步数据处理过程，帮助大家快速上手。

本文将一步步引导大家学习React，首先从创建React工程开始，然后在组件之间传递数据，在实现一个简单的计数器功能之前，我们需要先了解什么是React、为什么要用React，以及一些React的相关术语。然后我们会创建一个最简单的React计数器应用，然后介绍一下JSX语法，学习如何通过React渲染出DOM元素，同时也会涉及到React的状态和生命周期。最后，我们还会学习如何使用Redux Thunk来处理异步请求，并且利用Redux的状态管理模式来存储数据。最后再总结一下，希望通过本文，大家能够快速入门React和Redux Thunk。欢迎大家分享自己的感受和想法。

## 1.1 为什么要用React?

1. 使用虚拟DOM提升页面渲染效率；

2. 提供更加灵活的方式进行编程；

3. 消除模板语言的痛点，易于维护代码；

4. 支持单向数据流，减少组件之间的耦合性。

5. 拥有庞大的社区支持和生态系统。

## 1.2 React的相关术语
- JSX：一种类似XML的标记语言，可以用JavaScript描述组件的结构和行为。

- Props：组件的属性，是一个对象。父组件可以通过props向子组件传递参数。

- State：组件内部的数据，可以是任意类型。组件可以根据自身的state和外部传入的props计算得到新的state，并触发UI更新。

- Virtual DOM：一种内存中的树状结构，用于存储组件的状态和视图，每当状态或props变化时，重新渲染整个Virtual DOM。

- Component：React中的最小可复用的单元，是一个函数或者类，它负责定义如何显示和处理用户输入。

- Render(): 用来渲染一个React组件，返回一个虚拟节点（Vnode）。

- Parent/Child Components：父级组件可以接受props并渲染子组件，子组件也可以接受props并渲染子组件。

- Controlled Component：表单元素，如input、select等，组件的状态由当前的值决定，不受其他组件影响。

- Uncontrolled Component：非表单元素，如div、span等，组件的状态独立于其他组件。

- Composition：组件组合方式，即组件嵌套组件的方式。

- Higher Order Component：高阶组件，是一个函数，接收一个组件作为参数，返回另一个组件。

- Hooks：一个新特性，使得函数组件可以使用额外的状态和生命周期方法。

## 2. 创建React工程
- 安装Nodejs，具体参考官网。
- 在终端中，切换到所需工作目录，执行以下命令安装create-react-app工具：npm install -g create-react-app 。
- 执行命令npx create-react-app my-app ，其中my-app为工程名称。
- 等待安装完成。
- 切换到工程目录，执行npm start 命令启动服务，浏览器访问http://localhost:3000 查看效果。
```bash
$ npx create-react-app my-app

Creating a new React app in /Users/xxx/my-app.

Installing packages. This might take a couple of minutes.
Installing react, react-dom, and react-scripts with cra-template...

yarn add v1.22.5
info No lockfile found.
[1/4] 🔍  Resolving packages...
warning react-dom@17.0.2 requires a peer of @types/react@>=17.0.0 but none is installed. You must install peer dependencies yourself.
[2/4] 🚚  Fetching packages...
[3/4] 🔗  Linking dependencies...
[4/4] 🔨  Building fresh packages...
success Saved lockfile.
success Saved 3 new dependencies.
info Direct dependencies
└─ react-scripts@4.0.3
info All dependencies
├─ react-dom@17.0.2
├─ react-scripts@4.0.3
└─ react@17.0.2
✨  Done in 9.13s.

Initialized a git repository.

Success! Created my-app at /Users/xxx/my-app
Inside that directory, you can run several commands:

  npm start
    Starts the development server.

  npm run build
    Builds the app for production.

  npm test
    Starts the test runner.

  npm run eject
    Removes this tool and copies build dependencies, configuration files
    and scripts into the app directory. If you do this, you can’t go back!

We suggest that you begin by typing:

  cd my-app
  npm start

Happy hacking!
```