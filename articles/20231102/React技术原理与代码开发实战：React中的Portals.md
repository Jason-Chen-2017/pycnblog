
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## Portals 是什么？
Portals 是 React 提供的一个新特性，允许组件“渲染”到特定的 DOM 节点中，而不是整个 body 中。这样可以解决一些弹出框、遮罩层等特殊场景下组件样式相互干扰的问题，提升用户体验。
## 为什么要使用 Portals？
在 React v16.0 中，引入了 Portals 的功能，但是并没有提供官方文档或教程给开发者们讲解如何使用它，所以本文将带领大家一起探讨一下这个重要的功能，并且通过一些实际案例，用最简单易懂的方式讲解它的用法。
# 2.核心概念与联系
## 概念
Portals 是指组件渲染到指定容器的一种机制。可以用于弹出框、模态对话框、自定义滚动条这些场景。其基本结构如下图所示:
上图从下至上依次为 ReactDOM.render() 方法创建的根元素、自定义组件渲染到的目标节点（比如 div）、自定义组件渲染出的内容、根元素内部的内容。

Portals 和 ReactDOM.render() 方法一样需要一个父元素作为根节点，一般情况下，会创建一个div元素作为该父元素，然后将 Portals 渲染到该 div 中。因此，如果要在 ReactDOM.render() 外部渲染 Portal ，则可以在 React 元素外围套一层 div 标签，然后再通过 portals 属性将子元素渲染进去。如：
```javascript
import React from'react';
import ReactDOM from'react-dom';

class App extends React.Component {
  render() {
    return (
      <div>
        <h1>Portal Demo</h1>
        <MyDialog />
      </div>
    )
  }
}

class MyDialog extends React.Component {
  constructor(props) {
    super(props);
    this.el = document.createElement('div');
  }

  componentDidMount() {
    document.body.appendChild(this.el);
  }

  componentWillUnmount() {
    document.body.removeChild(this.el);
  }

  render() {
    const style = {
      position: 'fixed',
      top: 0,
      right: 0,
      bottom: 0,
      left: 0,
      backgroundColor: '#fff'
    };

    return ReactDOM.createPortal(<div style={style}>This is a dialog!</div>, this.el);
  }
}

ReactDOM.render(<App />, document.getElementById('root'));
```
上面例子中，`MyDialog` 是一个自定义组件，实现了 componentDidMount() 和 componentWillUnmount() 方法，用来动态添加或移除 DOM 节点。在 `componentDidMount()` 时，把 DOM 节点插入到了 document.body 中；在 `componentWillUnmount()` 时，把 DOM 节点从 document.body 中移除。然后利用 ReactDOM.createPortal 方法渲染子元素 `<div>` 到自定义创建的 DOM 节点中。这样就实现了将 Dialog 弹出来显示的效果。

除了上述功能之外，Portal 还可以用于解决组件样式相互干扰的问题，例如多个组件都想设置相同的样式，但由于渲染位置不同导致冲突，而 Portals 可以让组件渲染到指定的容器节点内，避免相互影响。

## 联系
React 中的 Portals 主要有以下三个应用场景：

1. 弹出框 Modal：Modal 在 React 中是一种比较常用的模式，可用于展示一些相关信息、操作选项或者表单，是一种复杂的交互形式。因此，当 Modal 需要渲染到某些特定的容器节点时，就可以通过 Portals 来实现弹出效果。另外，还有一种情况就是嵌入其他系统的页面或插件，这种时候也可以通过 Portals 将内容渲染到指定的容器节点中。

2. 自定义滚动条 Scrollbar：很多移动端浏览器或 PC 端浏览器的设计，都会采用自定义滚动条来代替默认的滚动条。此时，就可以通过 Portals 把自定义滚动条渲染到指定的容器节点中，避免其他组件样式影响。

3. 底部工具栏 Footer：许多网页或 APP 的界面都会在底部放置一些常用操作按钮或菜单，此时可以考虑通过 Portals 将组件渲染到某个固定区域，免得影响页面布局。