                 

# 1.背景介绍


React是Facebook在2013年推出的一个用于构建用户界面的JavaScript库。它是一个声明式编程框架（Declarative programming），可以让你轻松地创建可交互的组件。在过去的一年里，React的热度不断上升。目前，React已经成为全球最流行的前端JavaScript框架。相比于其他的前端框架来说，React最大的特点就是它的性能优势。通过虚拟DOM这种“速度与效率”的平衡，React保证了你的应用在运行时的响应速度，使得页面的渲染非常流畅、自然、顺滑。从而吸引更多的Web开发人员加入到React阵营中。本系列教程将带你全面掌握React事件处理机制。首先，我们先回顾一下React的事件机制。
React的事件处理机制主要包括两类：SyntheticEvent和addEventListener方法。
# SyntheticEvent
React在浏览器端实现了一套自己的事件对象，叫做SyntheticEvent。SyntheticEvent继承于浏览器原生事件对象，并添加了额外的属性和方法。SyntheticEvent会自动对一些浏览器行为进行兼容性处理，比如将键盘事件绑定在document而不是body元素上等。
# addEventListener方法
React的事件处理机制由addEventListener方法提供支持。addEventListener方法可以监听HTML元素的各种事件，例如click、mouseover等。addEventListener方法提供了两个参数，第一个参数是要绑定的事件类型，第二个参数是一个函数，该函数会在相应的事件发生时执行。addEventListener方法返回一个removeEventListener方法，调用这个方法可以移除监听器。

# 2.核心概念与联系
## SyntheticEvent
React在浏览器端实现了一套自己的事件对象，叫做SyntheticEvent。SyntheticEvent继承于浏览器原生事件对象，并添加了额外的属性和方法。SyntheticEvent会自动对一些浏览器行为进行兼容性处理，比如将键盘事件绑定在document而不是body元素上等。
## addEventListener方法
React的事件处理机制由addEventListener方法提供支持。addEventListener方法可以监听HTML元素的各种事件，例如click、mouseover等。addEventListener方法提供了两个参数，第一个参数是要绑定的事件类型，第二个参数是一个函数，该函数会在相应的事件发生时执行。addEventListener方法返回一个removeEventListener方法，调用这个方法可以移除监听器。

SyntheticEvent和addEventListener方法之间存在着密切的联系。SyntheticEvent作为React的一个内部模块，用来模拟浏览器原生的事件对象；addEventListener方法作为React暴露给我们的接口，用来注册事件监听器并处理事件。它们之间的关系如下图所示：


## event对象
在addEventListener方法的参数列表中有一个event对象，这个参数代表了触发事件的实际对象。不同类型的事件有不同的event对象。

举例来说，对于点击事件来说，event对象代表的是鼠标点击所在的位置。对于键盘按下事件来说，event对象代表的是被按下的按键。对于拖动事件来说，event对象代表的是鼠标指针的移动轨迹。因此，在addEventListener方法的回调函数中需要获取正确的event对象。

```javascript
function handleClick(event){
  console.log('x:'+ event.clientX); // 获取鼠标点击位置的横坐标值
  console.log('y:'+ event.clientY); // 获取鼠标点击位置的纵坐标值
}

// 使用addEventListener方法注册点击事件监听器
const button = document.getElementById('myButton');
button.addEventListener('click', handleClick);
```