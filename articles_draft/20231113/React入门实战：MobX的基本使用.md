                 

# 1.背景介绍


React作为目前最热门的前端框架之一，它的出现使得Web开发者可以快速构建漂亮的用户界面。在过去的十几年里，React已从Facebook开源出来，受到了社区的广泛关注。很多人认为，React是一种JavaScript库，是用于构建用户界面的解决方案。事实上，React是Facebook发布的一套用于构建用户界面的JavaScript工具集，是一个轻量级、可复用的JS库。相对于其它JavaScript框架（如jQuery）而言，React的特点主要有以下两点：

1.声明式编程：用声明的方式描述视图层，而不是直接操作DOM节点。声明式编程让代码更加易读、易理解。这也导致了React推崇数据驱动视图的理念。
2.组件化开发：React将复杂页面划分成多个组件，每个组件只负责一件具体工作。这样做可以提高代码的可维护性、可扩展性及可复用性。

React是通过 JSX 来定义组件结构，这也是一种类似于HTML的标记语言。它允许我们声明式地定义组件的状态及渲染逻辑。与传统的基于类的开发方式不同，React完全由函数式编程支持。因此，React比其他框架更加简单、灵活，适合构建大型项目或是快速迭代新功能。

不过，React也存在一些缺陷，例如性能较低等。为了解决这些问题，Facebook于2017年发布了Flux架构，使用单向数据流思想代替 Redux 管理应用状态。同时还推出了React-Native，能让我们开发原生移动应用。此外，还有一些第三方库比如 Redux 和 MobX 等，提供更加便捷的状态管理。

本文将会对 Redux 和 MobX 进行比较，阐述它们之间的异同，并根据 MobX 的使用教程，带领读者建立自己的 React + MobX 应用，体验 MobX 在实际工程中的应用效果。希望通过本文，读者能够掌握 React + MobX 的开发流程、优势及注意事项，从而帮助自己熟悉和使用这款优秀的状态管理库。

Redux 是 Facebook 提出的一个状态管理库，用于管理JavaScript应用程序中所有的数据。它遵循 Redux 抽象的概念，包括 Store、Action、Reducer、Middleware。在 Redux 中，Store 是一个仓库，保存应用的所有数据；Action 描述动作对象，即要改变 Store 中的哪些数据；Reducer 函数接收 Action 对象并返回新的 Store 状态。为了方便调试和追踪，Redux 提供了 middleware 支持，可帮助我们处理异步请求，日志记录，路由同步等等场景。

与 Redux 不同，MobX 使用观察者模式来监听数据的变化，当被观察的数据发生变化时通知依赖它的组件。MobX 使用的是 getter/setter 方法来获取/修改数据，并且提供了许多 API 来简化编码过程。这使得 MobX 更加轻量级，可以在复杂业务场景下保持高性能。

本文将结合官方文档、示例和实际工程项目，阐述 Redux 和 MobX 的基本概念，并探讨它们之间的异同。最后，我们将着重介绍 MobX 的使用方法，根据实际工程需求来编写一个完整的 MobX 应用。

# 2.核心概念与联系
## 2.1 Redux概念
### 2.1.1 Store
Redux 的核心是 store，它是一个仓库，存储所有的应用状态。我们可以使用 createStore() 方法来创建一个 Redux store，并传入指定 reducer 函数作为参数。

```javascript
import { createStore } from'redux'
const store = createStore(reducer)
```

其中，reducer 函数是一个纯函数，接收旧的 state 和 action，计算出新的 state。action 可以是任何行为，比如设置变量的值、发送 HTTP 请求、显示弹窗等等。每当 store 收到一个 action 时，它会调用 reducer 函数，得到新的 state，然后触发相应的更新。


store 有三种状态：

- 初始状态：初始化完成后的第一个状态。
- 当前状态：指当前应用的状态，由 reducer 根据当前 action 更新而来。
- 恢复状态：当页面刷新后，会恢复到之前保存的状态，即 lastState。

### 2.1.2 Actions
Action 是一个对象，描述发生了什么。它有一个 type 属性，表示 action 的名称。除此之外，action 可以携带其他属性，用于传递数据。actions 通过 store.dispatch() 方法分发给 store。

```javascript
{
  type: "ADD_TODO",
  text: "Learn Redux"
}
```

通常情况下，我们使用 constants 模块来管理 actions 的类型。

```javascript
export const ADD_TODO = 'ADD_TODO';
export const REMOVE_TODO = 'REMOVE_TODO';
//...
```

使用 constants 可以避免拼写错误、忘记修改、造成冲突等问题。

```javascript
dispatch({ type: ADD_TODO, text: 'Learn Redux' });
dispatch({ type: REMOVE_TODO, id: 1 });
```

当需要派发一个 action 时，可以通过 store.dispatch() 方法来实现。

```javascript
store.dispatch({ type: 'ADD_TODO', text: 'Learn Redux' })
```

### 2.1.3 Reducers
Reducers 是 pure function，接收旧的 state 和 action，返回新的 state。reducers 应该是纯净的，这意味着它们不应该修改传入的参数，而且返回的结果一定要是一个新的对象，不能引用旧的对象。

```javascript
function todos(state = [], action) {
  switch (action.type) {
    case ADD_TODO:
      return [...state, {
        id: nextTodoId++,
        text: action.text,
        completed: false
      }]
    // other cases...
    default:
      return state;
  }
}
```

这里，todos reducer 是一个纯函数，用来添加 todo 条目。

### 2.1.4 Middleware
中间件是 Redux 的概念，它是介于 Store 和 Reducer 之间的一层抽象。它是一个函数，接收 Store 的 dispatch() 方法，能够在执行 action 前后进行额外的处理。中间件的目的是为 Redux 增加更多特性，如时间旅行，记录，测试等。

```javascript
const loggerMiddleware = store => next => action => {
  console.log('will dispatch', action);
  let result = next(action);
  console.log('state after dispatch', store.getState());
  return result;
};
```

loggerMiddleware 会在执行 action 前后打印日志信息。

```javascript
const middlewares = [loggerMiddleware];
const store = createStore(rootReducer, applyMiddleware(...middlewares));
```

applyMiddleware() 方法可以把中间件数组作为参数，来创建 store。

### 2.1.5 结论
Redux 的 store 有三个状态：初始状态、当前状态和恢复状态。状态在 store 上被保存在 reducer 函数中，通过 action 进行修改，并通知 store 进行更新。中间件为 Redux 增添了额外的功能。

## 2.2 MobX概念
### 2.2.1 Observables
Observable 是 MobX 的核心概念，它代表了一个可观察的数据结构。任何被 MobX 包装过的对象都是一个可观察的对象，因为它在内部维持了指向它的观察者的链接。当可观察对象被修改时，所有观察者都会收到通知。

```javascript
let numbers = observable([1, 2, 3]);
autorun(() => {
  console.log(`sum is ${numbers.reduce((a, b) => a + b)}`);
});

setTimeout(() => {
  numbers[0] = 2;
}, 1000);
```

这里，numbers 是个被 MobX 包装过的数组。autorun() 函数是一个自动运行的函数，它会在某个数据被修改时自动执行，然后打印出新的总和。setTimeout() 函数会在一秒钟后把数组的第零个元素修改为 2。

### 2.2.2 Computed values and reactions
Computed value 是 MobX 的另一种核心概念，它可以自动计算其他值。当它所依赖的值发生变化时，它也会自动重新计算。

```javascript
@observable count = 0;
@computed get doubledCount() {
  return this.count * 2;
}
console.log(`doubled count is ${this.doubledCount}`);

setTimeout(() => {
  this.count++;
}, 1000);
```

这里，我们定义了两个计数器：count 和 doubledCount。doubledCount 是一个 computed property，它依赖于 count，它返回 count 的双倍值。autorun() 函数会在任意地方用到 computed property，并且在 count 被修改时自动重新计算。

### 2.2.3 Reaction
Reaction 是 MobX 的另一种核心概念，它可以跟踪特定表达式的变动。当表达式的值发生变化时，它会执行指定的回调函数。

```javascript
@observable shoppingList = [];
reaction(() => this.shoppingList.length, () => {
  console.log(`Your shopping list has ${this.shoppingList.length} items.`);
});

setTimeout(() => {
  this.shoppingList.push("milk");
  this.shoppingList.push("bread");
}, 1000);
```

这里，我们定义了一个购物清单列表，并且用 reaction() 函数跟踪其长度。每当 shoppingList 的长度发生变化时，reaction() 的回调函数就会被调用，并打印出新长度。setTimeout() 函数模拟了用户添加商品到购物清单的情况。

### 2.2.4 结论
MobX 有四种核心概念：observables、computed values、reactions、actions。它们都通过依赖收集、发布订阅模式实现自动更新。Observables 将普通 JS 数据结构转变为可观察的，computed values 计算值的变动，reactions 监听特定表达式的变动，actions 发起数据变动。