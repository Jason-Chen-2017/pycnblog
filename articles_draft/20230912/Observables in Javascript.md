
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 什么是Observable?
Observable是一种编程模式，它允许某些变量或表达式依赖于其他变量或表达式的值变化。简单的说，就是当某个变量的值改变时，观察者可以得到通知并自动执行相应的操作。这种行为被称为观察者模式。在Javascript中，Observable主要体现为RxJS。因此，本文重点讨论RxJS。

## 1.2 为什么要用Observable？
1.异步编程：像RxJS这样的ReactiveX库解决了回调地狱的问题。开发者不必嵌套多层回调函数，从而提升代码可读性、可维护性和可测试性。

2.可观测的数据流：在前端应用中，数据是由许多来源产生，需要经过复杂的处理才能呈现给用户。ReactiveX允许将不同数据源的数据转换成统一的观测序列，然后订阅这些序列并做出反应。

3.管理状态：开发者可以更容易地跟踪应用中的状态，比如用户登录、加载状态等。ReactiveX提供了强大的管道机制，让开发者可以轻松创建复杂的状态机。

## 2.Observable基础
### 2.1 创建Observable对象
首先，让我们来学习一下如何创建一个Observable对象。
```javascript
const source = Rx.Observable.of(1, 2, 3); // 使用静态方法of()创建
console.log('source:', source);
```
输出:
```javascript
source: Observable {
  _isScalar: false,
  operator: {
   ...
  },
  sources: []
}
```
其中`_isScalar`属性表示这个Observable是一个集合还是单个值，如果是集合，则值为false；`operator`属性记录了对这个Observable对象的操作，比如filter、map等；`sources`属性保存了这个Observable对象所依赖的所有Observable对象。

我们也可以通过一些方法来创建Observable对象：
```javascript
// 通过创建Observable序列的方式创建
const interval$ = Rx.Observable.interval(1000).take(3); // 每隔一秒发射一个数字，最多发送3次
const range$ = Rx.Observable.range(1, 3); // 发射[1, 2, 3]序列

// 通过subscribe()订阅事件创建
const observable$ = new Rx.Observable((observer) => {
  observer.next(1);
  setTimeout(() => observer.next(2), 1000);
  setTimeout(() => observer.next(3), 2000);
});
```
这里，`interval()`方法会定时发射数字，直到指定的次数为止；`range()`方法直接把范围内的数字发射出来；最后，通过构造器的方式订阅事件来创建Observable对象。

除了使用静态方法创建外，还可以使用RxJS提供的各种Operator来创建Observable对象。

### 2.2 订阅Observable对象
第二步，让我们来学习一下如何订阅一个Observable对象。
```javascript
// 订阅interval()方法创建的Observable对象
const subscription = interval$.subscribe(value => console.log(`received value: ${value}`));
setTimeout(() => subscription.unsubscribe(), 5000); // 停止发射值，等待5秒后取消订阅

// 订阅构造器创建的Observable对象
observable$.subscribe({
  next: (value) => console.log(`received value: ${value}`),
  complete: () => console.log("completed"),
  error: err => console.error("an error occurred:", err)
});
```
这里，`subscribe()`方法可以传入回调函数，每次接收到新值时调用；或者，可以传入对象，分别指定next、complete、error回调函数。此外，可以通过返回的Subscription对象来控制Observable对象生命周期，比如取消订阅。

### 2.3 操作符（Operators）
第三步，让我们来学习一下RxJS提供的各种Operator。

#### map()方法
使用`map()`方法可以把Observable对象发射的每个值映射成另一个值。比如：
```javascript
const numbers$ = Rx.Observable.of(1, 2, 3);
const doubledNumbers$ = numbers$.pipe(
  Rx.operators.map(num => num * 2)
);
doubledNumbers$.subscribe(num => console.log(`doubled number: ${num}`));
```
输出:
```javascript
doubled number: 2
doubled number: 4
doubled number: 6
```
这里，我们使用`pipe()`方法来将多个Operator串联起来，最终生成新的Observable对象。`map()`方法的回调函数接收原始值作为参数，返回新的值。

#### filter()方法
使用`filter()`方法可以过滤掉Observable对象发射的不需要的值。比如：
```javascript
const oddNumbers$ = Rx.Observable.from([1, 2, 3, 4]).pipe(
  Rx.operators.filter(num => num % 2 === 1)
);
oddNumbers$.subscribe(num => console.log(`filtered number: ${num}`));
```
输出:
```javascript
filtered number: 1
filtered number: 3
```
这里，`from()`方法可以把数组转换成Observable序列；`filter()`方法的回调函数接收原始值作为参数，返回true表示保留，返回false表示丢弃。

#### scan()方法
使用`scan()`方法可以计算Observable对象发射值的累计和。比如：
```javascript
const accumulatedSum$ = Rx.Observable.of(1, 2, 3).pipe(
  Rx.operators.scan((acc, cur) => acc + cur)
);
accumulatedSum$.subscribe(sum => console.log(`accumulated sum: ${sum}`));
```
输出:
```javascript
accumulated sum: 1
accumulated sum: 3
accumulated sum: 6
```
`scan()`方法的回调函数接受两个参数：累计和当前值，返回新的累计值。

#### combineLatest()方法
使用`combineLatest()`方法可以合并多个Observable对象发射的最新值。比如：
```javascript
const name$ = Rx.Observable.of("Alice");
const age$ = Rx.Observable.of(25);
const person$ = Rx.Observable.combineLatest(name$, age$, (name, age) => `${name}, ${age}`);
person$.subscribe(p => console.log(`the latest person is: ${p}`));
```
输出:
```javascript
the latest person is: Alice, 25
```
`combineLatest()`方法的参数是一个数组，里面包括所有需要合并的Observable对象；第二个参数是一个回调函数，用来接收合并后的最新值。

#### Subject类
Subject类是一个特殊的Observable对象，可以作为事件发生源或事件观察者。

例如，我们可以用Subject对象实现输入框的实时更新：
```html
<input type="text" #textInput>
<div>{{ text }}</div>
```
```javascript
const subject = new Rx.Subject();
subject.subscribe(event => this.textInput.nativeElement.value = event);
this.textInput.nativeElement.addEventListener("keyup", e => subject.next(e.target.value));
```
我们先创建一个Subject对象，然后订阅它的next()方法，每当文本框失去焦点或文本框内容变化时就会收到事件。接着，我们绑定键盘按键事件，每次按下键盘都会触发事件并更新文本框的内容。

#### BehaviorSubject类
BehaviorSubject继承自Subject类，它会记住最近一次订阅时的消息。我们可以在创建Observable对象时，传入初始值来初始化BehaviorSubject。如下例：
```javascript
const count$ = new Rx.BehaviorSubject(0);
count$.subscribe(x => console.log(x));
count$.next(1); // output: 1
count$.next(2); // output: 2
```
这里，第一次订阅后输出的第一个值就是BehaviorSubject的初始值。之后的输出都是由`next()`方法发出的最新值。