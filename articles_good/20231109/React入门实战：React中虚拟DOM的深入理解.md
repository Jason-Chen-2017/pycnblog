                 

# 1.背景介绍


React(简称`Re`)是一个用于构建用户界面的JavaScript库。Facebook于2013年推出了React，并开源了它的代码。很多公司都在采用它，包括像Netflix、Airbnb、Instagram等。

React本身不是一套完整的解决方案，而是一个UI框架。作为一个UI框架，React只是其中的一层。实际上，React是由一些更底层的模块组成的，比如 ReactDOM 和 React Native。

React主要通过 JSX 的语法创建组件，并且支持 Virtual DOM 模型进行渲染。Virtual DOM 是一种编程概念，是真正的 DOM 的一种抽象表示形式。它是将整个 UI 树转换成一个纯 JavaScript 对象的数据结构，该对象描述 UI 元素及其属性。

因此，如果 UI 不发生变化，则无需重新渲染，只需要比较当前的 Virtual DOM 与上一次渲染时的 Virtual DOM 即可。这样可以提升性能，减少渲染时间，从而达到更好的用户体验。

但是 Virtual DOM 有个重要的问题就是“如何计算 Virtual DOM”？要计算出新的 Virtual DOM，就需要比较两棵树的差异。比如某个节点的文字内容改变了，就需要创建一个新节点，将旧节点替换掉。如果某个节点的属性值改变了，也需要更新这个节点的属性。

本文将重点介绍 Virtual DOM 在 React 中的工作原理。

# 2.核心概念与联系
## 什么是 Virtual DOM？
Virtual DOM（虚拟 DOM）是一个概念，是指用 Javascript 对象来模拟浏览器的 DOM 结构。Virtual DOM 只与界面显示相关，不涉及业务逻辑或数据处理。

简言之，Virtual DOM 就是通过 Javascript 对象而不是实际的 DOM 来描述真实的 DOM，再对比它们的不同，根据不同来决定如何更新真实的 DOM 以达到界面渲染的目的。

## 为什么需要 Virtual DOM？
传统的渲染方式有两种：
- 生成静态HTML页面，然后让浏览器逐步解析，最终生成显示效果；
- 把Javascript代码编译成HTML字符串，通过AJAX请求服务器，返回页面，并动态插入到DOM容器中。这种方式既耗时，又不够灵活。

为了解决上面两种方法的弊端，React 提出了 Virtual DOM 的方案。其基本思想是把整个界面分割成一个个组件，每个组件都有一个 Virtual DOM 树，当状态发生变化时，会重新渲染整个组件的 Virtual DOM 树，最后把这个树中的节点应用到真实的 DOM 上，完成一次 UI 渲染。

## Virtual DOM 与真实 DOM 的区别？
真实的 DOM （Document Object Model，文档对象模型）是一个标准定义，定义了一个表示 XML/HTML 文档的 API。它由各种结点（node）组成，每个结点代表文档中的一小块内容，例如标签、文本或者图像等。


而 Virtual DOM ，是用 Javascript 对象来代替真实的 DOM 实现的一层虚拟化，它和真实的 DOM 之间存在着一一对应的关系。Virtual DOM 通过 JSX 或 createElement 方法产生，然后通过 diff 算法比较两棵 Virtual DOM 树的不同，将变动的部分应用到真实的 DOM 上。


如图所示，Virtual DOM 用数据结构来模拟真实的 DOM，Virtual DOM 可以被认为是一个轻量级的 DOM，因为它只包含数据的信息，不会包含任何的样式或行为。当 Virtual DOM 需要更新的时候，会通过 diff 算法来计算出两个对象的不同，然后只针对不同的部分做局部更新，从而极大的优化渲染效率。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## diff 算法
当 Virtual DOM 树中的某个节点发生变化时，会触发相应的生命周期函数，比如 componentDidMount、componentWillUnmount。React 使用了 diff 算法来比较两棵 Virtual DOM 树的不同。

diff 算法的基本思想是从根节点开始递归地对比每一棵子树，把不同的地方标记出来。算法的关键步骤如下：

1. 对比两棵 Virtual DOM 树的根节点是否相同，如果不同，直接修改真实的 DOM，否则继续下一步；
2. 判断两棵子树的第一个节点是否相同，如果不同，直接修改真实的 DOM，否则进入第三步；
3. 如果第一个节点相同，进入第四步；
4. 比较第一个节点的类型是否相同，如果不同，则删除旧节点并添加新节点；
5. 如果类型相同，比较属性是否相同，如果不同，则更新节点的属性；
6. 递归地对比两个节点的子节点是否相同，如果不同，则更新子节点。

```javascript
function diff(oldTree, newTree){
  // 获取根节点
  let root = document.getElementById("root");

  if (newTree === null || typeof newTree!== "object" || newTree.$$typeof!= Symbol.for("react.element")) {
    // 更新 root 节点的内容
    root.innerHTML = "";

    if (typeof newTree == "string") {
      root.textContent = newTree;
    } else if (typeof newTree == "number") {
      root.textContent = "" + newTree;
    } else if (typeof newTree == "boolean") {
      root.textContent = "" + newTree;
    }
    
    return oldTree;
  }
  
  // 判断类型是否一致
  const tag = newTree.type;
  if (!tag &&!newTree._owner) {
    console.error(`unknow element: ${JSON.stringify(newTree)}`);
    return oldTree;
  }
  
  // 根据 tag 创建 DOM 元素
  let el = document.createElement(tag);
  
  // 设置 key 属性
  if (newTree.key) {
    el.setAttribute("key", newTree.key);
  }

  // 设置 props
  for (let propName in newTree.props) {
    let propValue = newTree.props[propName];
    if (propName === "children") continue;
    setAttribute(el, propName, propValue);
  }

  // 设置 children
  let childNodes = [];
  let prevChild = {};
  let nextChildren = toArray(newTree.props.children).map((child, index) => {
    let key = isString(child)? child : JSON.stringify(child);
    prevChild[key] = oldTree? findOldNode(oldTree.props.children, key) : null;
    return createChild(child, index, prevChild[key]);
  });

  function findOldNode(children, key) {
    if (!children) return null;
    if (isString(children)) return null;
    for (let i = 0; i < children.length; i++) {
      let child = children[i];
      if (isSameElement(child, key)) return child;
      let ret = findOldNode(child.props.children, key);
      if (ret) return ret;
    }
    return null;
  }

  function createChild(vnode, index, prevVdom) {
    if (prevVdom && compareProps(prevVdom.props, vnode.props)) {
      return updateExistingNode(vnode, prevVdom, index);
    } else {
      return createNewNode(vnode, index);
    }
  }

  function updateExistingNode(vnode, prevVdom, index) {
    const sameType = vnode.type === prevVdom.type;
    if (!sameType) {
      removeNode(prevVdom);
      return createNewNode(vnode, index);
    }
    return updateNode(vnode, prevVdom, index);
  }

  function createNewNode(vnode, index) {
    const newEl = document.createElement(vnode.type);
    patchProps(newEl, {}, vnode.props, false);
    insertBefore(newEl, parentEl, index + 1);
    return new VDOM({ type: vnode.type, dom: newEl, props: vnode.props }, [], []);
  }

  function updateNode(vnode, prevVdom, index) {
    const el = prevVdom.dom;
    const prevProps = prevVdom.props;
    const nextProps = vnode.props;
    const newChildren = toArray(nextProps.children);
    patchProps(el, prevProps, nextProps, true);
    const prevChildren = prevVdom.children;
    const commonLength = Math.min(prevChildren.length, newChildren.length);
    let i;
    for (i = 0; i < commonLength; i++) {
      const prevChild = prevChildren[i];
      const nextChild = newChildren[i];
      const key = isString(prevChild)? prevChild : JSON.stringify(prevChild);
      const foundIndex = indexOfKey(nextChildren, key);
      if (foundIndex > -1) {
        const updated = updateExistingNode(nextChild, prevChild, foundIndex);
        newChildren[i] = updated;
        prevChildren[i] = updated;
      } else {
        removeNode(prevChild);
      }
    }
    for (; i < prevChildren.length; i++) {
      removeNode(prevChildren[i]);
    }
    for (; i < newChildren.length; i++) {
      const childVDom = createChild(newChildren[i], i, null);
      appendChild(childVDom, el);
      prevChildren.push(childVDom);
    }
    return prevVdom;
  }

  // 插入位置
  const parentEl = getParent(root);
  let refNode = getRefNode();
  if (refNode) {
    insertAfter(el, refNode);
  } else {
    appendChild(el, parentEl, getNextSibling());
  }

  // 更新前后指针
  movePrevPointers(oldTree, prevChild);
  assignNextPointer(parentEl, el);

  function movePrevPointers(tree, prevChildMap) {
    if (!tree || tree.__vdom) return;
    let key = isString(tree)? tree : JSON.stringify(tree);
    let prevVdom = prevChildMap[key];
    if (prevVdom && prevVdom.dom) {
      prevVdom.next = tree;
    }
    for (let i = 0; i < tree.length; i++) {
      movePrevPointers(tree[i], prevChildMap);
    }
  }

  function assignNextPointer(parentNode, childEl) {
    while ((childEl = childEl.nextElementSibling)) {
      let nodeVdom = getNodeVdom(childEl);
      if (nodeVdom && nodeVdom.__vdom) {
        nodeVdom.next = getNodeVdom(childEl.nextElementSibling).__vdom;
      }
    }
  }

  function toArray(children) {
    return Array.isArray(children)? children : [children];
  }

  function setAttribute(el, name, value) {
    switch (name) {
      case'style':
        setStyle(el, value);
        break;
      default:
        el.setAttribute(name, value);
    }
  }

  function setStyle(el, styleObj) {
    if (isObject(styleObj)) {
      for (const key in styleObj) {
        el.style[key] = String(styleObj[key]);
      }
    } else {
      el.setAttribute('style', styleObj);
    }
  }

  function compareProps(prevProps, nextProps) {
    return shallowEqualObject(prevProps, nextProps);
  }

  function shallowEqualObject(objA, objB) {
    if (objA === objB) {
      return true;
    }
    if (!isObject(objA) ||!isObject(objB)) {
      return false;
    }
    var keysA = Object.keys(objA);
    var keysB = Object.keys(objB);
    if (keysA.length!== keysB.length) {
      return false;
    }
    for (var i = 0; i < keysA.length; i++) {
      var key = keysA[i];
      if (!objB.hasOwnProperty(key) || objA[key]!== objB[key]) {
        return false;
      }
    }
    return true;
  }

  function addEventListener(el, eventName, handler) {
    el.addEventListener(eventName, handler);
  }

  function removeEventListener(el, eventName, handler) {
    el.removeEventListener(eventName, handler);
  }

  function insertBefore(newEl, parentEl, beforeEl) {
    parentEl.insertBefore(newEl, beforeEl);
  }

  function insertAfter(newEl, refEl) {
    if (refEl.nextElementSibling) {
      insertBefore(newEl, refEl.parentNode, refEl.nextElementSibling);
    } else {
      insertBefore(newEl, refEl.parentNode, null);
    }
  }

  function removeNode(vnode) {
    if (!vnode.dom) return;
    unmountComponentAtNode(vnode.dom);
    vnode.dom.remove();
    delete vnode.dom;
  }

  function replaceNode(newVdom, oldVdom) {
    if (!oldVdom.dom) return;
    if (!newVdom) {
      removeNode(oldVdom);
      return;
    }
    if (newVdom.dom) {
      newVdom.dom.parentElement.replaceChild(newVdom.dom, oldVdom.dom);
    } else {
      mountComponentIntoNode(newVdom, oldVdom.dom);
    }
    oldVdom.dom = newVdom.dom;
    oldVdom.children = newVdom.children;
    oldVdom.props = newVdom.props;
  }

  function indexOfKey(arr, key) {
    for (let i = 0; i < arr.length; i++) {
      let item = arr[i];
      let k = isString(item)? item : JSON.stringify(item);
      if (k === key) {
        return i;
      }
    }
    return -1;
  }

  class VDOM{
    constructor(type, dom, props, children, next){
      this.type = type;
      this.dom = dom;
      this.props = props;
      this.children = children;
      this.next = next;
      this.__vdom = true;
    }
  }

  function render(vnode, container) {
    let oldRoot = getNodeVdom(container.firstChild);
    let root = createVdom(vnode);
    patch(container, root, oldRoot);
    if (oldRoot && oldRoot.dom) {
      recollectNodeTree(oldRoot);
    }
  }

  function createVdom(vnode) {
    if (vnode == null || typeof vnode === 'boolean') {
      return null;
    }
    if (typeof vnode ==='string' || typeof vnode === 'number') {
      return new VDOM(null, null, null, vnode);
    }
    let instance;
    if (isFunction(vnode.type)) {
      instance = instantiateComponent(vnode);
      let renderedVdom = render(instance.render(), instance.base);
      instance.renderedVdom = renderedVdom;
      vnode = renderedVdom;
    }
    if (isObject(vnode)) {
      const type = isArray(vnode)? 'div' : vnode.type;
      const props = vnode.props || {};
      let children = flatten(toArray(vnode.props && vnode.props.children)).filter(Boolean);
      children = children.map(createVdom);
      return new VDOM(type, null, props, children);
    }
    throw Error(`Unsupported type: "${vnode}"`);
  }

  function instantiateComponent(element) {
    let instance = new element.type(element.props);
    instance.base = document.createElement(element.type.name);
    return instance;
  }

  function renderComponentToString(instance) {
    return instance.renderedVdom? renderToString(instance.renderedVdom) : '';
  }

  function renderToString(vnode) {
    if (vnode == null || typeof vnode === 'boolean') {
      return '';
    }
    if (typeof vnode ==='string' || typeof vnode === 'number') {
      return '' + vnode;
    }
    let result = '<';
    let attrs = vnode.props;
    let hasAttrs = Boolean(attrs);
    if (hasAttrs) {
      result += vnode.type;
      for (let attr in attrs) {
        if (attr!== 'children') {
          result += ` ${attr}="${escapeHtml(attrs[attr])}"`;
        }
      }
    } else {
      result += vnode.type;
    }
    result += '>';
    if ('children' in attrs) {
      if (isArray(attrs.children)) {
        for (let i = 0; i < attrs.children.length; i++) {
          result += renderToString(createVdom(attrs.children[i]));
        }
      } else {
        result += renderToString(createVdom(attrs.children));
      }
    }
    result += '</' + vnode.type + '>';
    return result;
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function mountComponentIntoNode(vdom, container) {
    let componentInstance = instantiateComponent(vdom);
    let componentVdom = componentInstance.renderedVdom;
    if (!componentVdom) return;
    componentInstance.componentDidMount();
    let firstChild = container.firstChild;
    let afterFirstChild = firstChild? firstChild.nextSibling : null;
    mountIntoContainer(componentVdom, container, afterFirstChild);
  }

  function unmountComponentAtNode(dom) {
    let vdom = getNodeVdom(dom);
    if (vdom.__unmounted) return;
    vdom.__unmounted = true;
    destroyComponentAtNode(vdom);
    clearContainer(vdom.dom);
    cleanupEffects(vdom);
  }

  function mountIntoContainer(vdom, container, after=null) {
    let node = vdom.dom = createDom(vdom);
    injectStyles(vdom);
    insertBefore(node, container, after);
    walk(vdom, performInitialMount);
    walk(vdom, finishClassComponent)
    if (!after) {
      postMountOrder.push(() => {
        triggerRenderTimer();
      });
    }
  }

  function createDom(vdom) {
    let el;
    let tag = vdom.type;
    if (tag === '#text') {
      el = document.createTextNode('');
    } else if (tag === '#comment') {
      el = document.createComment(vdom.props);
    } else {
      el = document.createElement(tag);
      if (vdom.props.onClick) {
        addEventListener(el, 'click', e => {
          executeEventHandlers(vdom.props.onClick, e);
        })
      }
      applyAttributes(el, vdom.props);
    }
    for (let i = 0; i < vdom.children.length; i++) {
      let child = vdom.children[i];
      let childEl = createDom(child);
      appendChild(childEl, el);
    }
    return el;
  }

  function applyAttributes(el, attrs) {
    for (let key in attrs) {
      let val = attrs[key];
      if (key === 'className') {
        el.classList.add(...val.split(/\s+/));
      } else if (key.startsWith('on')) {
        let eventName = key.slice(2).toLowerCase();
        addEventListener(el, eventName, e => {
          executeEventHandlers(val, e);
        });
      } else if (/^xlink:?/.test(key)) {
        el.setAttributeNS(XLINK_NS, key, val);
      } else if (key!== 'children') {
        el.setAttribute(key, val);
      }
    }
  }

  function executeEventHandlers(handlerOrHandlers, event) {
    if (!event) return;
    let handlers = handlerOrHandlers;
    if (!Array.isArray(handlers)) {
      handlers = [handlers];
    }
    for (let fn of handlers) {
      if (fn) {
        callCallback(fn, event);
      }
    }
  }

  function getValueOfEvent(event) {
    return event.target? event.target.value : undefined;
  }

  function getValueForKeyPath(obj, path) {
    try {
      let parts = path.split('.');
      for (let part of parts) {
        obj = obj[part];
      }
      return obj;
    } catch (err) {}
    return undefined;
  }

  function setValueForKeyPath(obj, path, value) {
    try {
      let parts = path.split('.');
      let lastPart = parts.pop();
      let curObj = obj;
      for (let part of parts) {
        curObj[part] = curObj[part] || {};
        curObj = curObj[part];
      }
      curObj[lastPart] = value;
      return true;
    } catch (err) {}
    return false;
  }

  function getProp(vdom, propName, defaultValue) {
    return propName in vdom.props? vdom.props[propName] : defaultValue;
  }

  function getRefValue(refs, refName) {
    return refs[refName].current;
  }

  function mergeRefs(refs,...sources) {
    sources.forEach(source => {
      source && Object.entries(source).forEach(([key, value]) => {
        let setter = value;
        if (setter === null) {
          setter = () => {};
        }
        refs[key] = createRefSetter(setter, refs[key]);
      });
    });
  }

  function createRefSetter(setter, previousRef) {
    return { current: previousRef? previousRef.current : null };
  }

  function setRef(refName, value, callback) {
    let ref = refs[refName];
    if (!ref) return;
    let oldValue = ref.current;
    ref.current = value;
    if (oldValue!== value && callable(callback)) {
      callback(value, oldValue);
    }
  }

  function getInstanceFromVdom(vdom) {
    if (vdom.__instance) return vdom.__instance;
    let parentVdom = vdom.parent;
    while (parentVdom) {
      if (parentVdom.type.prototype instanceof Component) {
        return parentVdom.__instance;
      }
      parentVdom = parentVdom.parent;
    }
    return null;
  }

  function buildEffectArgs(argNames, args) {
    let effectArgs = {};
    argNames.forEach((name, idx) => {
      effectArgs[name] = args[idx];
    });
    return effectArgs;
  }

  function runEffect(effect, dependencies) {
    if (dependenciesChanged(effect, dependencies)) {
      resetComputationDepth();
      invalidate(causeStack);
    } else if (!effect.dirty &&!pendingInvalidations) {
      scheduleUpdate();
    }
    addToPendingEffects(effect);
  }

  function dependenciesChanged(effect, dependencies) {
    return dependencies!== effect.dependencies ||!effect.customCompareFn;
  }

  function invokeEffects() {
    effectsBeingProcessed = pendingEffects.splice(0);
    for (let effect of effectsBeingProcessed) {
      try {
        effect.invoke();
      } catch (err) {
        handleError(err);
      } finally {
        cleanUpEffect(effect);
      }
    }
    effectsBeingProcessed = null;
    pendingEffects = [];
  }

  function addToPendingEffects(effect) {
    pendingEffects.push(effect);
  }

  function scheduleUpdate() {
    scheduledUpdates++;
    scheduleWork(() => {
      flushPassiveEffects();
      if (!scheduledUpdates) return;
      scheduledUpdates--;
      if (shouldScheduleRender()) {
        startBatch(() => {
          stateLock.locked = true;
          try {
            batchUpdatePhase = UpdatePhase.RENDERING;
            renderer.performSyncRefresh();
          } finally {
            stateLock.locked = false;
          }
          batchUpdatePhase = UpdatePhase.NONE;
        });
      }
    });
  }

  function shouldScheduleRender() {
    return Promise.resolve().then(() => {
      return (
        dirtyRoots.size === 1 &&
        (renderer.isPrimaryRenderer ||
        canExitBatchForRenderer(renderer))) &&
        renderer.shouldForceFullRender;
    });
  }

}
```