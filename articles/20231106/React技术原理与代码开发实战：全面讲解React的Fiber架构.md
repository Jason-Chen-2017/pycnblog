
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


React 是一个用来构建用户界面的 JavaScript 框架。它提供了 Virtual DOM 的概念，可以有效减少更新渲染所需的DOM操作次数，提升页面性能。它的核心数据结构就是一个树状的数据结构，称为 Fiber（纤维）树。Fiber 树的每一个节点都包含了当前节点需要渲染的所有信息。当 Fiber 树重新渲染时，只会对比和更新其需要变化的部分，而不是整个渲染树。这是 React 在一定程度上解决了视图更新过于频繁的问题。

本文将深入 React 的 Fiber 数据结构，从实现角度全面剖析 React Fiber 的基本原理。让读者能够更加深刻地理解 React Fiber 如何帮助 React 更快、更高效地更新界面。
# 2.核心概念与联系
## 2.1 Fiber 树
React 中最重要的数据结构之一是 Fiber 树。Fiber 树是一个结构化的，递归的数据结构，由一个个小型节点组成，每个节点代表着当前屏幕上可见的元素或组件。React 通过 Fiber 技术重构了调度器，使得它能在更细粒度的层次上进行任务分解，并通过记录工作进度和上下文信息的方式，在工作过程中做到“即时响应”和“中断后恢复”。

Fiber 树中的每个节点都包含以下属性：

1. Type：节点类型，比如 div 或 Text 等
2. Props：节点上的属性对象
3. Children：子节点数组
4. State：节点的状态对象，用于保存组件内的局部状态
5. Update：描述该节点上要执行的更新内容，比如新增，删除或者更改某个节点的类型。
6. Context：表示当前节点的上下文，用来保存一些共享的数据
7. Effects：这个节点上面要执行的副作用列表，比如 componentDidMount 和 componentDidUpdate 这样的生命周期函数
8. Perf：一些性能相关的信息，比如开始时间、结束时间、工作耗时等

Fiber 树的构建过程分为两个阶段，第一个阶段是 JSX 编译阶段，将 JSX 语法转换为 createElement() 方法调用；第二个阶段是 Fiber 对象创建阶段，基于 JSX 产生的调用链，生成 Fiber 对象。Fiber 对象的创建和 Fiber 树的构建过程非常复杂，涉及到了很多底层的实现机制。但是，通过 Fiber 技术，React 可以实现更优秀的更新策略，提升应用的运行效率。

## 2.2 更新策略
React 利用 Fiber 技术构建了一个优先级队列（FiberQueue），里面存放着所有需要更新的 Fiber 对象。当某些事件触发导致视图更新的时候，React 会把这些 Fiber 加入到 FiberQueue 中，然后根据 Fiber 的 priority 属性来排序，从而确定更新顺序。优先级的计算规则如下：

1. Sync 模块更新优先级最高
2. Mount/update 之前的模块更新优先级较高
3. 需要更新样式的模块更新优先级较低
4. 不需要更新的模块更新不予考虑

## 2.3 WorkLoop
React 通过维护一个叫做 WorkLoop 的循环来实现任务调度。WorkLoop 主要职责包括收集优先级最高的 Fiber，按照顺序执行它们的更新。如果遇到优先级较低的 Fiber，就跳过它，并继续处理下一个优先级最高的 Fiber。

首先，WorkLoop 从 FiberQueue 中取出优先级最高的 Fiber，判断它是否存在同步模块。如果存在，就直接进入执行流程，否则，则进入 Module 模块处理流程。

### 2.3.1 Sync 模块更新
Sync 模块更新比较简单，只需要执行一次就可以完成。一般来说，Sync 模块主要包括初始化阶段、渲染阶段和提交阶段。

- 初始化阶段：包括执行 render 函数、创建 FiberRoot 对象、创建第一个 Fiber 对象等。
- 渲染阶段：从根 Fiber 对象开始，依次遍历子节点，创建相应的 Fiber 对象，同时执行 useEffect 和 useState 等钩子函数，收集所有需要渲染的 Fiber。
- 提交阶段：使用 ReactDOM.render 将 root 组件渲染到真实 dom 上。

### 2.3.2 Module 模块更新
Module 模块更新也分为两步，分别是 Before Fiber 和 After Fiber。

#### 2.3.2.1 Before Fiber
Before Fiber 表示的是准备好待更新的 Fiber。Before Fiber 有以下几个特点：

1. 已经遍历到了 Fiber 对象
2. 依赖列表里没有尚未被安装的异步更新
3. 当前的 Fiber 的 childExpirationTime 为 null

#### 2.3.2.2 After Fiber
After Fiber 是更新完毕的 Fiber。After Fiber 有以下几个特点：

1. 已经完成了新的属性和状态的计算
2. 所有的 hook 都是已被调用且执行成功
3. 如果 fiber.effectTag 含有Placement（比如 useState），表示此 fiber 下面还有需要渲染的 fiber，所以该 afterFiber 还不能进入工作循环

### 2.4 小结
Fiber 树是 React 中最重要的数据结构，它既用于描述 UI 组件，又用于执行组件更新时的状态管理。React Fiber 技术通过优先级队列和递归算法，解决了组件更新过于频繁的问题，进一步提升了应用的性能。

本文对 React Fiber 技术的基本原理作了全面剖析。对于 FiberQueue 的分析和 React Fiber 工作循环的分析，给了读者更加深刻的了解。另外，还谈到了 React Fiber 的更新策略，希望通过本文的介绍，能够更好地掌握 React Fiber 相关知识，为日后的学习打下坚实的基础。