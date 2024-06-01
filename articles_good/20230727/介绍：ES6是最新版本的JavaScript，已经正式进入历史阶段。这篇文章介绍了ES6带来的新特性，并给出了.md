
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2015 年 6 月，ECMAScript 6 (ES6) 诞生，JavaScript 的下一代标准版发布，它带来了很多语法上的改变和新功能，包括：
            - `let` 和 `const` 关键字用于声明变量，在之前只能用 `var` 来声明
            - 模板字符串(template strings)，一种在字符串中嵌入表达式的新方式
            - 箭头函数，可以用来声明函数
            - Classes 类,一个新的语法结构，用于创建对象的蓝图
            - 更易于使用的数据类型，比如 Map、Set、TypedArray
            - 迭代器协议（iterator protocol），让数据结构具有可遍历性
         
         为了更好地理解这些概念，本文会从以下几个方面进行讲解:
         - 对比旧版 JS 有什么不同
         - let 和 const 的使用场景及区别
         - Template literals 的使用方法
         - Arrow functions 的特点和使用方法
         - Class 的定义、继承和实例化
         - Maps/Sets 数据结构和原理
         - Iterators 迭代器协议的作用
         
         本文不会涉及 Node.js 或前端开发相关的内容。
         
         作者：骆昊
         链接：https://juejin.im/post/5b77c9f6e51d450e712ea0fe#heading-4
         来源：掘金
         著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

         # 2.基本概念术语说明
         
         ## 2.1.let 和 const 关键字
          
          `let` 和 `const` 关键字都可以用来声明变量。两者的主要区别如下：
         
           - `let` 声明的变量只在当前块级作用域内有效，而 `const` 声明的变量在整个作用域内有效。
           - 使用 `let` 声明变量后，该变量的值可以重新赋值；使用 `const` 声明变量后，不能再对其重新赋值。
         ```javascript
         // let 可以重新赋值
         let a = 1;
         console.log(a); // output: 1
         a = 2;
         console.log(a); // output: 2
 
         // const 不可以重新赋值
         const b = 1;
         console.log(b); // output: 1
         b = 2; // Uncaught TypeError: Assignment to constant variable.
         console.log(b); // output: 1
         ```
     
         在上面的例子中，`let a` 只在当前 `if` 块作用域内有效，所以第二次输出 `console.log(a)` 时打印的是最新的值。而 `const b` 在整个作用域内有效，所以不能通过修改变量名的方式将 `b` 替换成其他值。
         
         此外，对于同一个变量来说，`let` 和 `const` 的行为也不同：
         
           - 如果这个变量没有被声明，那么 `let` 会被认为是一个全局变量，而 `const` 则报错。
           - 用 `let` 声明的变量可以更改，而 `const` 声明的变量不可以更改。
        ```javascript
         {
             var x = "global";   // 全局作用域下的变量
         }
         if (true){
             let y = 'block';     // if 块作用域下的变量
             
             function f(){
                 for(let i=0;i<5;i++){
                     console.log('inner', i);
                 }
                 console.log("in function", y);
             }
             setTimeout(()=>console.log("setTimeout block scope:",y),1000);
             f();
         }
         console.log("x is global", x);
         console.log("y is undefined outside the block", y);  // 报错，y 没有声明
     
         // 变量替换
         console.log("replace x with other value");
         {
             var x = "otherValue";    // 通过 var 声明，x 是全局变量，因此替换无效
         }
         console.log("new value of x is not replaced because it's a global variable", x);
         ```
        上面的例子中，我们用 `var` 关键字声明了一个全局变量 `x`，然后在一个块作用域里声明了一个变量 `y`。在 `for` 循环中又用 `let` 关键字声明了一个内部变量 `i`，并且在一个定时器回调中访问到了外部变量 `y`。但是由于 `const` 声明的原因，我们不能对变量 `x` 或者 `y` 的值进行修改。
        
        ## 2.2.Template literals
         
         模板字符串是一种在字符串中嵌入表达式的新方式，使用反引号(` `` `)作为界定符，可以在模板字符串中嵌入变量、表达式或其他模板字符串。
         
         下面的示例展示了模板字符串的基本用法：
         
         ```javascript
         // 使用 ${} 插入变量
         let name = "Alice";
         let message = `Hello, ${name}!`;
         console.log(message); // output: Hello, Alice!
         
         // 使用模板字符串插入其他模板字符串
         let header = `<h1>Header</h1>`;
         let article = `${header} <p>${paragraph}</p>`;
         console.log(article); // output: <h1>Header</h1> <p>This is an example paragraph.</p>
         
         // 执行函数并插入到模板字符串中
         function greeting() { return "Howdy"; };
         let greetings = `Hello, ${greeting()}, how are you?`;
         console.log(greetings); // output: Hello, Howdy, how are you?
         ```
         除了上述的插入变量和模板字符串之外，还可以使用各种运算符和条件语句。
         
         模板字符串是一种便利的字符串处理方式，它可以提升代码的可读性，减少出错的可能性，并降低维护成本。
         
         ## 2.3.Arrow Functions
         
         ES6 新增了箭头函数（arrow function）。箭头函数与普通函数相比有以下几个优点：
         
           - 没有自己的 this，它们没有自己的上下文，所以可以很方便地与对象的方法一起使用。
           - 不需要使用 `function` 关键字，直接使用 `=>` 来表示函数体，使得代码更加简洁。
           - 函数体中无法使用 `arguments` 对象，因为它不是函数的一部分。
           - 箭头函数没有自己的 `super` 方法，所以不能用作构造函数。
         
         下面的示例展示了箭头函数的基本用法：
         
         ```javascript
         // 普通函数
         function multiply(x, y) { return x * y; }
         console.log(multiply(2, 3)); // output: 6
         
         // 箭头函数
         let add = (x, y) => x + y;
         console.log(add(2, 3)); // output: 5
         
         // 用数组排序
         let numbers = [3, 1, 4, 1, 5, 9];
         numbers.sort((a, b) => a - b); 
         console.log(numbers); // output: [1, 1, 3, 4, 5, 9]
         
         // 将函数作为参数传递
         let callback = () => console.log("callback called");
         callback(); // output: callback called
         ```
         箭头函数非常适合简短的函数表达式，尤其是在回调函数中。
         
         ## 2.4.Classes
         
         ES6 提供了更加简洁的类的定义方式，称为“类”（class）。类提供了面向对象编程的所有基本元素——属性、方法、构造函数等。
         
         下面的示例展示了类的基本用法：
         
         ```javascript
         class Person {
             constructor(firstName, lastName) {
                 this.firstName = firstName;
                 this.lastName = lastName;
             }
             sayName() {
                 console.log(`${this.firstName} ${this.lastName}`);
             }
         }
         
         let person = new Person("John", "Doe");
         person.sayName(); // output: John Doe
         
         // 使用 super 调用父类构造函数
         class Employee extends Person {
             constructor(firstName, lastName, title) {
                 super(firstName, lastName);
                 this.title = title;
             }
             getFullName() {
                 return `${this.firstName} ${this.lastName}`;
             }
         }
         
         let employee = new Employee("Jane", "Smith", "Manager");
         console.log(employee.getFullName()); // output: Jane Smith
         employee.sayName(); // output: Jane Smith
         ```
         在上面的例子中，我们定义了一个 `Person` 类，它有一个构造函数，接受两个参数 `firstName`、`lastName`，并将它们分别保存到 `this.firstName` 和 `this.lastName` 中。还有个方法 `sayName()`，它打印出完整的名字。接着，我们定义了一个 `Employee` 类，它继承自 `Person` 类，添加了一个构造函数，接受三个参数，其中前两个是父类的参数，第三个是新的参数 `title`。另有两个方法：`getFullName()` 返回完整的名字，和 `sayName()` 一样打印出完整的名字。
         
         注意：如果子类没有定义构造函数，它就会调用父类的构造函数，如果父类没有定义构造函数，就会报错。
         
         类提供了一个面向对象的封装、继承和多态的全新体验。
         
         ## 2.5.Maps/Sets 数据结构
         
         ES6 为集合数据结构引入了两个新的数据结构：`Map` 和 `Set`。
         
         ### Map
         
         `Map` 是一个简单的键值对对象。它的行为类似于 JavaScript 中的对象，但 `Map` 键的范围不限于字符串、数值或其它任何类型，它可以使用任意值。
         
         下面的示例展示了 `Map` 的基本用法：
         
         ```javascript
         // 创建空 Map
         let map = new Map();
         
         // 添加 key-value 对
         map.set('foo', 'bar');
         map.set({}, 'hello object');
         map.set(undefined, null);
         
         // 获取值
         console.log(map.get('foo')); // output: bar
         console.log(map.get({})); // output: hello object
         console.log(map.get(undefined)); // output: null
         
         // 检查是否存在某个键
         console.log(map.has('foo')); // true
         console.log(map.has('baz')); // false
         
         // 删除键值对
         map.delete('foo');
         console.log(map.size); // output: 2
         ```
         在上面的例子中，我们创建了一个空的 `Map` 对象，并添加了一些键值对。我们也可以通过 `get()`、`has()`、`delete()` 等方法来获取、检查和删除键值对。
         
         ### Set
         
         `Set` 是一个简单的集合，它类似于数组，但成员的值都是唯一的。
         
         下面的示例展示了 `Set` 的基本用法：
         
         ```javascript
         // 创建空 Set
         let set = new Set();
         
         // 添加值
         set.add(1);
         set.add('foo');
         set.add({});
         
         // 查看长度
         console.log(set.size); // output: 3
         
         // 判断是否存在某个值
         console.log(set.has('foo')); // true
         console.log(set.has('baz')); // false
         
         // 清空集合
         set.clear();
         console.log(set.size); // output: 0
         ```
         在上面的例子中，我们创建了一个空的 `Set` 对象，并添加了一些值。我们也可以通过 `size` 属性查看集合的大小，通过 `has()` 方法判断某个值是否存在于集合中，通过 `clear()` 方法清空集合。
         
         ## 2.6.Iterators 迭代器协议
         
         Iterator 协议是 ES6 引入的概念，它是一种用来遍历 JavaScript 对象的协议。
         
         当一个对象实现了 Symbol.iterator 属性时，就可以通过 `for...of` 循环来遍历它。
         
         下面的示例展示了如何遍历 `Map` 对象：
         
         ```javascript
         let map = new Map([['one', 1], ['two', 2]]);
         
         for (let entry of map) {
             console.log(entry[0], entry[1]);
         }
         /* output: 
          one 1
          two 2
         */
         ```
         在上面的例子中，我们创建一个 `Map` 对象，并通过 `for...of` 循环来遍历它。`entry` 是迭代器返回的一个结果，它是一个数组 `[key, value]`。
         
         迭代器协议还可以应用于 `Set` 对象，这样就可以直接得到值的集合。

