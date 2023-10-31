
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


React是一个构建用户界面的JavaScript库。它在组件化、声明式编程等方面提供了极大的便利。然而，React中还有一些不太容易理解的概念。本文通过对React中refs与dom的基本概念进行阐述、探讨，并结合实际案例，展示如何在React中使用refs和dom相关的API，帮助你更好的理解和掌握这些概念。
## 什么是refs?
 refs 是一种特殊的 prop，其用于在一个组件上添加一个可访问的 DOM 元素或一个返回对应 DOM 元素的函数。通过 refs 可以获取到真实的 DOM 对象，实现各种交互效果。React 将 ref 存储在组件实例中，当组件的 state 或 props 更新时，ref 也会随之更新。
## 为什么要用refs？
我们都知道在 React 中，状态（state）是驱动 UI 变化的一个重要因素。但是，状态的更新往往是异步的，而组件渲染完成之后才能更新。如果需要在某个时间点对 DOM 做某些操作，则可以通过 refs 来获取这个节点，并调用相应的方法。例如，给一个 input 框设置焦点；触发动画效果；或者对第三方插件的 API 进行封装。
## 有哪些类型的Refs?
React 提供了三种不同类型 Refs:

1. createRef(): 创建一个新的 Ref 对象，其 `.current` 属性初始值为 `null`。可以在 class component 内创建多个 Refs 对象，每创建一个对象，组件都会保存一个引用。

2. useRef(): 返回一个可变的、不可变的 Ref 对象，其 `.current` 属性初始值为传递的第一个参数。可以把此方法看作是 createRef() 的语法糖，可以传递初始值，同时只返回 current 属性。

3. forwardRef(): 高阶函数，接收一个函数作为参数，返回一个组件的 Forwarding Refs。Forwarding Refs 允许父组件控制子组件的行为，一般用在具有复杂内部结构的组件，希望将某些底层逻辑抽象到外面。
## 为什么要区分createRef()与useRef()?
createRef()和useState()类似，都是用来处理组件内部状态变量的问题。但是两者有一个区别就是，useState()只能用于class组件中，不能用于函数组件中，而createRef()可以在函数组件和类组件中使用。所以一般情况下，我们优先使用useState()来管理状态变量，只有在函数组件的局部状态需要被多个组件共享时才使用createRef()。
## useRef的使用场景
useRef主要用途是在函数组件中存放一个可变的可复用的值。比如，我们想在函数组件中保存鼠标悬停位置的数据，就可以利用useRef来保存这个数据，其他组件也可以读取这个数据。还可以用来保存大量不会重新渲染的变量，比如节流函数、计时器、滚动位置等。这样就可以避免额外的性能开销。如下示例代码所示：

```
import { useState, useEffect } from'react';
function Example() {
  const [count, setCount] = useState(0);
  // 使用useRef保存鼠标位置
  const mousePos = useRef({ x: undefined, y: undefined });

  function handleMouseMove(e) {
    mousePos.current = {
      x: e.clientX,
      y: e.clientY
    };
  }

  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <div>
      Mouse position: ({mousePos.current.x}, {mousePos.current.y})<br />
      Count: {count}<br />
      <button onClick={() => setCount(count + 1)}>+</button>
    </div>
  );
}
```

在上面的例子中，我们使用useEffect()监听document上的mousemove事件，并更新鼠标位置信息。同时，我们也使用useRef保存了鼠标位置信息，其他组件可以通过调用ref对象的current属性来访问这个信息。