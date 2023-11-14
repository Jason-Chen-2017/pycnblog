                 

# 1.背景介绍


Error boundaries 是一种用于处理 JavaScript 错误的React组件。它可以捕获其子孙组件树中的 JavaScript 错误，并在组件树中向上传播，而不是导致整个应用崩溃。这是一种处理客户端JavaScript 错误的可靠方式，因为你可以捕获并记录错误，同时仍然显示 UI 的其余部分。

此外，你可以通过渲染一个 fallback UI 来捕获意料之外的错误（比如，网络请求失败或某些其他类型的 JS 错误），从而避免显示空白页或者崩溃整个应用。

本教程将会带领读者完成以下内容：

1.了解什么是 Error boundaries；
2.编写一个简单的 Error boundary component；
3.理解 Error boundaries 对应用的影响；
4.学习如何调试 Error boundaries。
# 2.核心概念与联系
## 2.1 什么是Error boundaries?
首先，我们先看一下官方对Error boundaries的定义：

>Error boundaries are components that catch errors in their child components and display a fallback UI instead of crashing the app. They log errors in development mode only, and don’t affect performance or functionality of the parent component like regular error handling does.

Error boundaries 是一个 React 组件，它可以捕获其子组件中的错误，并展示一个替代 UI 以避免应用崩溃。它仅在开发模式下记录错误，不会影响父级组件的性能和功能，就像一般的错误处理机制一样。

## 2.2 为什么需要Error boundaries？
### 2.2.1 防止意料之外的错误导致应用崩溃

React 的错误处理机制是一个很强大的工具，但是也存在一些缺陷。其中一个主要问题就是，如果某个组件出错了，可能导致其所有后代组件也出错。这会导致整个应用的崩溃。举个例子，如果某个路由组件出错了，那么应用的所有页面都会出现错误提示。

使用 Error boundaries 可以解决这个问题。当某个组件发生错误时，它的子组件及其后代组件都无法正常工作，因此可以展示一个替代的 UI 。这样用户就可以知道，哪里出错了，并且可以重新加载页面、返回首页等。

这种设计方式也是很多库如 Redux、Apollo 或 GraphQL 在处理错误时的方案。它们都是为了保证应用的健壮性，即使由于意料之外的错误导致应用崩溃也可以提供用户友好的体验。

### 2.2.2 提供额外的错误信息给开发者

另一个重要的作用是，提供额外的错误信息给开发者。对于开发者来说，错误日志往往不足以帮助定位错误的根源。但是 Error boundaries 却可以通过 `componentDidCatch` 方法获取到完整的错误信息，包括报错位置、错误类型、错误消息等，从而让开发者更好地分析和修复问题。

## 2.3 Error boundaries组件的属性和用法
### 2.3.1 static getDerivedStateFromError()

要创建一个 Error boundaries 组件，只需继承自 `React.Component`，然后实现静态方法 `static getDerivedStateFromError()`：

```jsx
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    // 更新 state 触发 componentDidCatch()
    return { hasError: true };
  }
  
  componentDidCatch(error, info) {
    console.log('Uncaught error:', error);
    console.log('Info:', info);
  }

  render() {
    if (this.state.hasError) {
      return <h1>Something went wrong.</h1>;
    }

    return this.props.children;
  }
}
```

上面的例子只是简单地渲染了一个固定的文本，表示出错了。实际情况下，你应该渲染一个备用的 UI ，并且在开发环境下输出错误信息。例如，你可以用一个 modal 框来提示错误信息：

```jsx
render() {
  if (this.state.hasError) {
    return (
      <Modal onClose={() => this.setState({ hasError: false })}>
        <h1>Something went wrong.</h1>
        <p>{this.state.errorMessage}</p>
        <pre>{JSON.stringify(this.state.errorInfo)}</pre>
      </Modal>
    );
  }

  return this.props.children;
}
```

### 2.3.2 componentDidCatch()

除了 `getDerivedStateFromError()` 方法之外，Error boundaries 还有一个名为 `componentDidCatch()` 的生命周期方法。这个方法接收两个参数：错误对象（error）和错误信息对象（info）。你可以在这个方法中把错误日志打印出来，或者把错误信息存储起来供后续分析。

```jsx
class ErrorBoundary extends React.Component {
 ...

  componentDidMount() {
    const { children } = this.props;
    try {
      Children.only(children);
    } catch (error) {
      this.setState({
        errorMessage: 'Children must be single element',
        errorInfo: error
      });
    }
  }

  componentDidUpdate(prevProps) {
    const { children } = this.props;
    if (!shallowEqualObjects(children, prevProps.children)) {
      this.setState({ hasError: false });
    }
  }

  render() {
   ...
  }
}
```

上面的示例代码用来检查子元素是否正确地设置，并把错误信息存储在 `state` 中。注意，这里并没有实际渲染任何 UI 。如果你真的想渲染一个 fallback UI ，可以在这个方法中进行。