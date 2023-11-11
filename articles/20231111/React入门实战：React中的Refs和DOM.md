                 

# 1.背景介绍


在React中，有时我们需要获取一些组件渲染后的DOM节点或者某些元素的引用，例如获取输入框的ref、获取滚动条的ref等。这是因为React应用在更新时会重新渲染整个组件树，当我们需要操作某个组件渲染后添加的DOM节点或执行某些特定的操作时，就需要用到refs。但是，要注意的是refs并不是真正意义上的“指针”（pointer），而是对DOM节点的引用。本文将通过三个实例来向您展示React中的refs用法。
## 1.1 为何需要Refs？
一般来说，在React开发中，子组件通过props父组件传递数据，并通过回调函数控制其行为。这意味着，如果父组件需要操作子组件的内部状态或某些DOM元素，只能通过回调函数实现。这种方式对于子组件来说比较直观，但对于父组件来说却不方便。例如，如果希望父组件在用户输入完成之后触发子组件的特定方法，那么父组件就需要监听子组件的数据变化，并且根据数据改变调用相应的方法。然而，这样的设计并不能满足需求，因为子组件的状态可能会随着时间的推移而变得复杂，而且父组件的代码也会变得难以维护。
为了解决这个问题，React提供了refs机制。通过refs，父组件可以访问子组件的DOM元素或某些属性。也就是说，refs机制为父组件提供了一种直接访问子组件内部数据的途径，而不是依赖于子组件的回调函数。更进一步地说，由于refs可以获取到真实的DOM元素，因此父组件可以像操作一般JS对象一样，对子组件进行各种操作。
## 1.2 Refs基本用法
在React中，refs可分为四种类型：
- callback refs: 通过一个函数赋值给ref属性，在组件 componentDidMount 和 componentDidUpdate 时调用。
- createRef() API: 返回一个可用于Refs的对象，可以像其他普通变量一样被赋值。
- string refs: 在之前版本的React中采用字符串形式的refs，虽然仍然可以使用，但不推荐使用。
- legacy context API：旧版的API，不建议使用。
下面我们通过几个实例来阐述React中refs的基本用法。
### 例子1：计数器
```jsx
import React, { useState } from'react';

const Counter = () => {
  const [count, setCount] = useState(0);

  const handleClick = () => {
    console.log('handleClick');
    setCount((prev) => prev + 1);
  };

  return (
    <div>
      <p>{count}</p>
      <button onClick={handleClick}>+</button>
    </div>
  );
};

export default Counter;
```
上面是一个简单的计数器组件，点击按钮可以增加计数值。但是，如果想获取到该组件渲染后生成的p标签和按钮的引用呢？通过refs就可以实现。
```jsx
import React, { useRef, useState } from'react';
import Counter from './Counter'; // 导入计数器组件

const Parent = () => {
  const pRef = useRef();
  const buttonRef = useRef();

  useEffect(() => {
    if (pRef.current && buttonRef.current) {
      console.log(`P标签的文本内容：${pRef.current.textContent}`);
      setTimeout(() => {
        buttonRef.current.click();
      }, 2000);
    }
  }, []);

  return (
    <>
      <h2 ref={pRef}>这是一个计数器</h2>
      <br />
      <Counter />
      <button ref={buttonRef}>点我增加计数器的值</button>
    </>
  );
};

export default Parent;
```
在Parent组件中，我们定义了两个refs：`pRef`和`buttonRef`。然后，在useEffect中，我们判断refs是否存在，如果存在，则可以在DOM渲染结束后使用setTimeout来模拟用户点击事件。我们也可以通过`event.currentTarget`来获取当前元素的引用，然后调用对应的方法。不过，这里为了简单起见，我们直接获取对应元素的引用进行调用。另外，如果需要在多个地方共用同一个refs，则可以把它定义成一个Context，避免重复定义。