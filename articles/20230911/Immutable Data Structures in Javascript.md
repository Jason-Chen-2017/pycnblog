
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1什么是不可变数据结构？
在计算机编程中，数据结构是指计算机内存储、组织、管理数据的逻辑和方法。不可变数据结构意味着其中的数据元素只能被创建后不能被修改的特性。换句话说，对于这种数据结构来说，所有对数据结构的修改操作都需要创建一个全新的副本，而旧的数据结构则不再使用。为了保证数据不可被修改，通常会给数据结构添加一层保护措施。例如，数组（Array）类型就是一个典型的不可变数据结构。通过这一特性，它可以帮助减少应用中的错误，并提供可预测性和稳定性。
JavaScript 的 Array 是一种最常用的不可变数据结构。在本文中，我将探讨 JavaScript 中最常用的几种不可变数据结构 - Map 和 Set 。但是，这些数据结构并不是唯一的不可变数据结构。还有其他一些数据结构如 Object ，也存在类似的限制。因此，在阅读完本文之后，您应该能够评估不同数据结构之间差异，并根据自己的需求选择合适的数据结构。

## 1.2为什么要使用不可变数据结构？
使用不可变数据结构最大的好处是可以提升性能。原因如下：
- 利用不可变数据结构，就可以避免因数据变化带来的内存泄漏或其他问题。因为数据无法被修改，所以就不存在共享变量的问题了，可以确保数据正确无误地传递到各个组件。
- 使用不可变数据结构可以有效防止 race condition （竞争条件）。由于数据无法被修改，因此可以在多个线程同时访问时保持同步。
- 通过使数据不可变，可使代码更容易理解和测试。相比于可变数据结构，不可变数据结构更易于描述和实现，还可以简化并行开发。

## 1.3本文的目标读者
本文的读者主要包括具有一定编码经验的程序员以及对计算机底层概念有浓厚兴趣的读者。
希望通过本文的学习，读者能够更清楚地了解和理解不可变数据结构，并能够在项目开发中灵活地使用它们。
# 2.Map
## 2.1什么是Map？
Map 是 ECMAScript 中的一个新的数据结构。它类似于哈希表（Hash table），可以存储键值对。其中，每个键对应的值都是独一无二的，也就是说，同样的键不能对应两个不同的值。Map 的主要优点如下：
- Map 是一个抽象数据类型，可以用来表示对象集合；
- Map 提供了一个基于键的集合，可以快速检索某个值的索引位置；
- Map 可以直接映射字符串，数值等任意值，而不需要自定义构造函数。

## 2.2Map的基本语法及操作
### 2.2.1创建Map
```javascript
// 创建空 Map 对象
const myMap = new Map();

// 用数组初始化 Map
const arr = [
  ['name', 'John'],
  ['age', 30],
  ['city', 'New York']
];
const mapFromArr = new Map(arr);

console.log(mapFromArr); // Map { "name" => "John", "age" => 30, "city" => "New York" }
```
通过 `new Map()` 方法可以创建一个 Map 对象。也可以通过传入数组的方式来创建 Map 对象。上面的例子展示了如何通过两种方式创建 Map 对象。
### 2.2.2添加/获取/删除元素
#### 添加元素
可以通过 `set()` 方法向 Map 中添加元素。该方法接受两个参数，第一个参数是键，第二个参数是值。如果键已经存在，则更新对应的值。否则，创建一个新的键值对。
```javascript
myMap.set('key1', 'value1'); // 将键值对添加到 Map 中

myMap.set('key1', 'newValue'); // 更新键 key1 在 Map 中的值

console.log(myMap.get('key1')); // 获取键 key1 在 Map 中的值，输出："newValue"
```
#### 删除元素
可以使用 `delete()` 或 `clear()` 方法从 Map 中删除元素。`delete()` 方法只接受一个参数，即键，如果键存在，则删除对应的键值对；如果键不存在，不会产生任何影响。`clear()` 方法不接受参数，用于清空整个 Map 对象。
```javascript
myMap.delete('key1'); // 从 Map 中删除键 key1

myMap.clear(); // 清空整个 Map 对象
```
### 2.2.3遍历Map
可以通过 `forEach()` 方法对 Map 中的所有元素进行遍历。该方法接受两个参数，第一个参数是一个回调函数，第二个参数是上下文对象 (可选)。回调函数接受三个参数，分别是键、值、Map 本身。可以指定上下文对象，让回调函数内部的 this 指向指定的对象。
```javascript
const items = [];
myMap.forEach((value, key) => {
    console.log(`Key: ${key}, Value: ${value}`);
    items.push([key, value]);
});

console.log(items); // Output: Key: name, Value: John, Key: age, Value: 30...
```
## 2.3Map实现原理及相关工具类库
### 2.3.1Map的底层实现
Map 数据结构是一种散列表 (hash table) 的实现，它的元素是按键值对形式存储的，其中每一个元素由键和值组成。与一般的散列冲突解决方式一样，Map 会采用开放寻址法解决键值对碰撞的问题。对于元素的查找、插入、删除等操作，Map 提供了四个 API 函数：

1. `get(key)`：返回给定键对应的值，如果没有找到该键，则返回 undefined
2. `set(key, value)`：设置键值对，如果键已存在，则更新其值，否则插入新键值对
3. `has(key)`：判断给定的键是否存在于 Map 中
4. `delete(key)`：移除给定键的键值对

Map 对象的原型链上有一个成员叫做 _es6Map，它是一个私有属性，用来存放键值对。那么，具体该怎么实现呢？

#### 属性
首先，Map 对象的属性有：

- `_keys`：保存键的数组
- `_values`：保存值得数组
- `_size`：保存键值对个数的整数
- `_root`：根节点，一个 RBTree 对象或者 null

#### 操作
Map 对象提供了四个 API 方法，分别对应 get、set、has 和 delete 操作，以及一个 forEach 方法用于遍历所有的键值对。这些方法的具体实现如下：

##### `constructor()`
创建一个 Map 对象，构造函数可以接收一个 iterable 对象作为参数，用其中的键值对来填充 Map。

```javascript
let m = new Map([['a', 1], ['b', 2]]);
```

##### `clear()`
清除所有的键值对。

```javascript
m.clear();
```

##### `delete(key)`
删除指定键的键值对，如果键存在，返回 true，否则返回 false。

```javascript
m.delete('a');    // true
m.delete('c');    // false
```

##### `entries()`
返回一个包含键值对的迭代器。

```javascript
for (let [k, v] of m.entries()) {
  console.log(`${k} -> ${v}`);
}
```

##### `forEach(callbackfn [, thisArg])`
调用 callbackFn 依次对于每个键值对执行一次，并传入三个参数：键、值、Map 对象自身。

```javascript
m.forEach(function(value, key, map){
  console.log(key + ':'+ value);
});
```

##### `get(key)`
通过键获取对应的值，返回 undefined 表示不存在该键。

```javascript
m.get('a');      // 1
m.get('c');      // undefined
```

##### `has(key)`
判断键是否存在于 Map 中，返回 true 表示存在，false 表示不存在。

```javascript
m.has('a');       // true
m.has('c');       // false
```

##### `keys()`
返回一个包含所有的键的迭代器。

```javascript
for (let k of m.keys()) {
  console.log(k);
}
```

##### `set(key, value)`
设置键值对，如果键已存在，则更新其值，否则插入新键值对。

```javascript
m.set('d', 3);         // true
m.set('e', 4).        // true
                      // false
                      // false
```

##### `size`
返回键值对数量。

```javascript
m.size;                // 4
```

##### `values()`
返回一个包含所有值得迭代器。

```javascript
for (let v of m.values()) {
  console.log(v);
}
```

#### 哈希函数
Map 内部的哈希函数采用的是 MurmurHash3 算法，该算法在计算效率方面很高，并且可以很好的均匀分布。

#### 红黑树
Map 内部的数据结构是一个 RBTree，它是一种平衡二叉搜索树，每一个结点既可以保存键值对，又可以保存子树。当一个结点多于两颗子树的时候，通过旋转和颜色标记来保持平衡。

# 3.Set
## 3.1什么是Set？
Set 是 ECMAScript 中的另一种新的数据结构，它类似于数组，但只能存储非重复的值。其中，每个值都是独一无二的，不会出现重复的值。Set 的主要优点如下：
- Set 是一个无序且不重复的集合，通过它可以快速地判断某个元素是否属于这个集合。
- 如果想知道 Set 里面有哪些元素，它提供了便利的方法 `values()` 和 `forEach()` 来遍历元素。

## 3.2Set的基本语法及操作
### 3.2.1创建Set
```javascript
// 创建空 Set 对象
const mySet = new Set();

// 用数组初始化 Set
const arr = [1, 2, 3, 3, 2, 1];
const setFromArr = new Set(arr);

console.log(setFromArr); // Set { 1, 2, 3 }
```
Set 可以通过 new 操作符创建，也可以用 Array 或者其它 iterable 对象初始化。上述代码创建了一个空的 Set 对象，然后用数组初始化了 Set。注意，数组中有重复的值，Set 只保留其中一个。
### 3.2.2添加/获取/删除元素
#### 添加元素
可以通过 `add()` 方法向 Set 中添加元素。该方法接受一个参数，即待添加的元素，如果该元素已经存在，则不会发生任何事情。
```javascript
mySet.add(1);     // 将数字 1 添加到 Set 中

mySet.add(1);     // 不发生任何事情

mySet.add('string');   // 将字符串'string' 添加到 Set 中

mySet.add({x: 1});    // 将对象 { x: 1 } 添加到 Set 中
```
#### 删除元素
可以使用 `delete()` 或 `clear()` 方法从 Set 中删除元素。`delete()` 方法只接受一个参数，即元素，如果元素存在，则删除此元素，如果元素不存在，不会发生任何事情。`clear()` 方法不接受参数，用于清空整个 Set 对象。
```javascript
mySet.delete(1);          // 从 Set 中删除数字 1

mySet.delete('string');   // 从 Set 中删除字符串'string'

mySet.clear();             // 清空整个 Set 对象
```
### 3.2.3遍历Set
可以通过 `values()` 方法遍历 Set 对象中的所有元素。该方法返回一个迭代器，可以用 `for...of` 循环或其他迭代器协议方法来遍历。
```javascript
const values = [...mySet.values()];
```
另外，也可以使用 `forEach()` 方法遍历 Set 中的所有元素。该方法接受一个回调函数，每次遇到一个元素都会执行这个回调函数。
```javascript
mySet.forEach(function(value) {
  console.log(value);
});
```