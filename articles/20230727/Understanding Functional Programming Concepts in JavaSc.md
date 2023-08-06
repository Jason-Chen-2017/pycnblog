
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         ## 一、概述
         函数式编程（functional programming）是一种编程范式，它将计算视为数学函数计算，并避免共享状态和可变数据，所有数据的变化都通过参数传递进行协调。本文作者Noah Gauss，是一名计算机科学家、工程师、创始人、软件工程师，曾任Facebook CEO、Apple Founder，还曾担任微软软件工程师。
         
         函数式编程对于开发人员来说非常重要，因为它赋予了开发人员更多的能力，使他们可以构建更好的软件产品。过去几年里，函数式编程在各个领域都得到越来越多的应用，尤其是在网站和应用程序开发领域，可以帮助解决复杂的问题、提高代码质量、提高效率，并且减少bug等。本文主要介绍JavaScript中的函数式编程，包含两个方面：基础理论和实际运用。
         
         ## 二、背景介绍
         
         19世纪末到20世纪初，纳粹德国实行了一场旷日持久的社会主义革命，给社会带来严重的不平等，其结果是建立起了资本主义制度，使得整个社会陷入动荡不安的状态。直到二十世纪中叶，随着人们对资本主义制度的厌恶，为了摆脱经济危机带来的社会阶级矛盾，出现了左派社会党。左派社会党号称自由民主党，认为应当放弃资本主义的生产方式，摒弃计划经济体系，实现共产主义社会。因此，左派社会党推出了激进的政治路线，即要实现公平正义，保护私有产权，取消垄断资本，打破政府和市场的界限，通过社会主义来实现全面的国家民族的转型。
         
         在20世纪30年代后期，由于苏联的崩溃、东欧剧变、国际金融危机、以及中国的改革开放，发生了十年轰轰烈烈的“反右”运动，社会开始出现左倾错误，社会主义阵营内部也出现了严重的分裂。国内外一些思想解放者认为，阶级斗争已经过时，应该抛弃资本主义道路，转向社会主义道路，提倡“消灭计划经济”，并且废除一切不合宪的政策。于是，上海共青团中央发表了一系列关于社会主义社会的宣言，号召全体公众抛弃资本主义，拥抱社会主义，实践社会主义。1989年1月1日，在中国共产党第九次全国代表大会上，邓小平在一次演讲中指出：“社会主义社会，首先是一个全新的阶段。”社会主义是全新的经济制度，不再以计划经济为主导，而是以市场为基础，通过公平竞争的方式获得财富的分配。
         
         函数式编程是对计算的观念上的一种尝试。它强调计算过程的不可变性，也就是说，数据只应该在函数调用的时候被处理，而不是被修改或者其他操作影响。这种方式往往可以降低代码编写难度，增强代码的可读性和可维护性。函数式编程可以帮助解决一些并非由程序设计者自己能解决的问题，例如排序、过滤、映射、递归等问题。另外，函数式编程还有助于编写正确、易于理解的代码，通过简化代码逻辑，提升编程效率，同时也降低了程序的运行时间和内存占用，从而减轻服务器负载，提高系统的可用性。
         
         从历史上看，函数式编程最早起源于 Haskell 编程语言。Haskell 的函数式编程特性最初是为了支持函数式编程教育，但很快就受到限制，只能用于某些特定领域。直到20世纪70年代末，函数式编程才成为一个真正意义上的编程范式。1987年，美籍华裔计算机科学家张本铭，首次提出了函数式编程的概念。他认为，函数式编程应该成为计算机编程的主流，并推广到整个计算领域。他于1990年在 ACM 上发表的一篇论文中，提出了基于 lambda 演算（lambda calculus）的函数式编程模型。20世纪90年代初，基于 Scheme 语言的 Clojure 和 Common Lisp 开始普及函数式编程。Clojure 是 ClojureScript 的前身，是一个函数式 Lisp 方言，提供了丰富的函数式编程特性。Common Lisp 是以 ANSI Common Lisp 标准为基础的lisp方言，也是目前最流行的函数式编程语言。2007年发布的 Elixir，也是一种基于 Erlang VM 的函数式编程语言。Elixir 提供了轻量且高效的实时编程能力，可以让程序员开发出更加高效和可靠的软件。
         
         当前，函数式编程正在迅速发展，尤其是微服务架构下的异步编程，以及快速发展的前端技术栈 React 和 Angular。函数式编程已经成为开发大规模、分布式、异步、事件驱动的应用程序的必备技能。本文涉及的内容也会随着函数式编程发展的不同阶段而更新。
         
         ## 三、核心概念和术语说明
         
         ### 3.1 函数（Function）
         函数就是一个接受输入参数，返回输出值的表达式或语句。在程序执行过程中，函数可以重复使用，也可以作为模块来引用。函数通常被定义为某个特定任务的实现，可以根据需要参数个数、类型、顺序任意组合调用。
         
         ### 3.2 参数（Parameter）
         参数就是传递给函数的值，是变量的一个别名。当函数调用时，参数值会绑定到相应的参数变量上，被称为形式参数或实际参数。函数的参数一般都具有名称，它可以用来描述该参数所代表的值的含义。
         
         ### 3.3 返回值（Return Value）
         函数的返回值就是函数执行完成后的输出结果。返回值一般都是通过某种形式表示的，通常情况下，返回值的数据类型跟函数声明时的返回类型一致。
         
         ### 3.4 副作用（Side Effect）
         副作用是指一个函数在执行过程中，除了返回结果之外，还会产生其他的影响，比如修改全局变量、改变传入的参数等等。虽然副作用不是好习惯，但是某些场景下确实无法避免，所以也需要关注。
         
         ### 3.5 局部作用域（Local Scope）
         局部作用域指的是变量的生命周期仅限于函数内部。在函数外部访问这个变量时，编译器会报错。函数内部可以通过嵌套函数或闭包的方式实现局部变量之间的交互。
         
         ### 3.6 闭包（Closure）
         闭包就是能够读取其他函数内部变量的函数，创建闭包的函数被称为闭包函数或父函数。可以在闭包函数内部创建另一个函数，并通过内部函数来间接访问外部函数的变量。通过闭包，可以将函数内部的状态保存起来，这样做可以隐藏实现细节，提供更高层次的抽象。
         
         ### 3.7 尾递归优化（Tail Recursion Optimization）
         尾递归优化是一种函数式编程方法，它将递归调用转换成循环，从而减少堆栈空间的消耗。尽管尾递归优化能够有效地减少栈空间消耗，但是也存在一些问题，比如可能会导致性能下降等。尾递归优化在函数调用链中的最后一个函数上才有用，否则无效。
         
         ### 3.8 数组（Array）
         数组是一个线性集合，里面元素按照索引位置排列。JavaScript 中的数组属于一种特殊的对象，它既有属性，又有长度属性。可以利用索引访问数组元素，通过 push() 方法添加元素，利用 pop() 方法删除元素，利用 splice() 方法插入或删除元素，以及 slice() 方法拷贝数组片段。
         
         ### 3.9 对象（Object）
         对象是一个键-值对的集合。对象的每个键都对应一个值，这些值可能是任何类型。JavaScript 中所有的类型都可以视为对象，包括字符串、数字、布尔值等。
         
         ### 3.10 可选参数（Optional Parameter）
         可选参数在函数定义的时候，可以指定其默认值，如果没有传入该参数，则默认使用默认值。可选参数通常用星号标识。
         
         ### 3.11 默认参数（Default Parameter）
         默认参数在函数定义的时候，可以指定默认值，在函数调用的时候，如果没有传入该参数，则使用默认值。默认参数与可选参数类似，都可以有多个，但是在调用时只有一个可以传。
         
         ### 3.12 箭头函数（Arrow Function）
         ES6 提供了箭头函数语法，它允许直接在函数中编写表达式。箭头函数有两种形式，表达式语句（expression statement）和表达式体（expression body）。表达式语句不需要显式的 return 关键字，而表达式体需要用 {} 将函数体括起来。
         
         ```javascript
           let sum = (a, b) => a + b; // expression statement
           let factorial = n => {
             if (n === 0 || n === 1) {
               return 1;
             } else {
               return n * factorial(n - 1);
             }
           }; // expression body
         ```
         
         ### 3.13 偏应用函数（Partial Application Function）
         偏应用函数是指将函数的一些参数固定住，返回一个新的函数，它的参数也已经固定住。这样就可以把一些参数一次性传入，节省重复输入参数的时间。
         ```javascript
           const addFive = add(5);
           
           console.log(addFive(2)); // Output: 7
       
           function add(x) {
             return function(y) {
               return x + y;
             }
           }
       
           Partial application is often used when dealing with functions that take multiple arguments or need to be reused frequently with different values of the same argument types. This approach can significantly reduce code duplication and improve performance. 
         ```
         
         ### 3.14 命令式编程（Imperative Programming）
         命令式编程，也称过程式编程，是一种编程风格，它的特点是以命令的方式进行程序的求解。程序员告诉计算机如何执行程序，而计算机则逐步执行程序指令。命令式编程语言通常使用赋值语句、条件判断语句和循环语句。命令式编程语言往往是一步步执行程序，直到达到预期结果。
         
         ### 3.15 声明式编程（Declarative Programming）
         声明式编程，也叫描述式编程，是一种编程风格，它的特点是采用声明的方式，来描述程序要达到的效果。声明式编程语言通常会自动推导出结果。声明式编程语言往往侧重于计算结果，而非底层的实现机制。
         ```javascript
           const arr = [1, 2, 3];
           const filteredArr = arr.filter(num => num % 2!== 0);
           
           console.log(filteredArr); // Output: [1]
         ```
         
         ### 3.16 元编程（Metaprogramming）
         元编程，也叫编程即编程，是指在编程语言中，写代码来生成代码的过程。元编程在运行时刻可以使用原始数据结构和控制流程，生成适合于当前环境的代码。JavaScript 中的 eval() 和 new Function() 方法都是元编程的例子。
         
         ### 3.17 函数式编程和命令式编程区别
         函数式编程和命令式编程的区别可以总结为以下四点：
         1. 对数据进行操作：函数式编程更倾向于对数据进行操作，也就是说，函数的输入是数据，输出也是数据。命令式编程则更倾向于对数据的表达形式进行操作。
         2. 使用映射函数：函数式编程更倾向于使用映射函数，即输入的数据经过一定的处理后再输出。命令式编程则更倾向于直接对数据进行操作。
         3. 最小化状态：函数式编程不使用可变的状态，也就是说，函数之间互不影响。命令式编程则更倾向于使用可变的状态，为了防止程序运行出错，需要对程序状态进行管理。
         4. 避免共享状态：函数式编程更倾向于使用不可变的状态，这样可以避免多个函数同时修改同一状态，从而避免数据同步问题。命令式编程则更倾向于使用可变的状态，从而引入数据共享问题。
         
         ## 四、核心算法原理和具体操作步骤以及数学公式讲解
         
         ### 4.1 map() 方法
         1. Map() 方法创建一个新数组，其结果是该数组中的每个元素都是调用一个提供的函数后返回的结果。
         2. 创建了一个新数组，然后遍历原始数组，将原始数组的每一项传给回调函数进行处理。将处理之后的结果添加到新数组中，最终得到一个新的数组。
         3. 返回值是一个新的数组，内容是原始数组的每一项经过回调函数处理之后的结果组成的。
         4. 如果原始数组为空，则返回空数组。
         5. 如果回调函数缺失，则返回 undefined。
         
         ```javascript
           const numbers = [1, 2, 3];
           const doubledNumbers = numbers.map(number => number * 2);
           
           console.log(doubledNumbers); // Output: [2, 4, 6]
         ```
         
         ### 4.2 filter() 方法
         1. Filter() 方法创建一个新数组，其中的元素是通过检查数组中符合条件的所有元素。
         2. 创建了一个新数组，然后遍历原始数组，将原始数组的每一项传给回调函数进行处理。如果回调函数返回 true，则该项会被添加到新数组中；如果返回 false，则忽略该项。最终得到一个新的数组。
         3. 返回值是一个新的数组，其中包含满足回调函数条件的元素。
         4. 如果原始数组为空，则返回空数组。
         5. 如果回调函数缺失，则返回 undefined。
         
         ```javascript
           const numbers = [1, 2, 3, 4, 5];
           const evenNumbers = numbers.filter(number => number % 2 === 0);
           
           console.log(evenNumbers); // Output: [2, 4]
         ```
         
         ### 4.3 reduce() 方法
         1. Reduce() 方法对数组中的每个元素依次应用一个函数，将其结果累计为单个值。
         2. 创建了一个新数组，然后遍历原始数组，将原始数组的每一项传给回调函数进行处理。回调函数会接收两个参数：前一个值（accumulator），和当前值（currentValue）。初始值为 accumulator 的初始值，第一个 currentValue 会作为第一个参数传入回调函数。回调函数返回的值会赋值给 accumulator。此时，前一个值的当前值会传入下一次回调函数，继续迭代。最后，accumulator 的值会作为 reduce() 的返回值。
         3. reduce() 方法提供了简洁、优雅的方法来对数组中的值进行操作。
         4. 如果原始数组为空，则返回 undefined。
         5. 如果回调函数缺失，则返回 undefined。
         
         ```javascript
           const numbers = [1, 2, 3, 4, 5];
           const total = numbers.reduce((accumulatedValue, currentValue) => accumulatedValue + currentValue, 0);
           
           console.log(total); // Output: 15
         ```
         
         ### 4.4 forEach() 方法
         1. forEach() 方法为数组中的每个元素都调用一次指定的函数。
         2. 不返回任何值。
         3. 创建了一个新数组，然后遍历原始数组，将原始数组的每一项传给回调函数进行处理。不会返回新数组。
         4. 每个元素都会被遍历到。
         5. 如果原始数组为空，则不会调用回调函数。
         6. 如果回调函数缺失，则不会报错。
         
         ```javascript
           const fruits = ["apple", "banana", "orange"];
           fruits.forEach(fruit => console.log(fruit)); // Output: apple banana orange
         ```
         
         ### 4.5 sort() 方法
         1. Sort() 方法将数组中的元素重新排序。
         2. 排序会直接修改原始数组。
         3. 方法会根据数组元素的 Unicode 表示符号来排序。
         4. 排序是稳定排序，即相等的元素保持原有的相对顺序不变。
         5. 如果原始数组为空，则返回空数组。
         6. 如果回调函数缺失，则按照默认规则排序。
         
         ```javascript
           const unsortedNumbers = [5, 2, 3, 1, 4];
           unsortedNumbers.sort();
           
           console.log(unsortedNumbers); // Output: [1, 2, 3, 4, 5]
         ```
         
         ### 4.6 高阶函数（Higher-Order Functions）
         高阶函数是那些将函数作为参数或者返回值的函数。它们让函数式编程变得更加简单和富有表现力。下面是一个简单的例子：
         
         ```javascript
           function logNumbers(array) {
             array.forEach(console.log);
           }
           const numbers = [1, 2, 3, 4, 5];
           logNumbers(numbers); // Output: 1 2 3 4 5
         ```
         
         此例中，logNumbers() 函数是一个高阶函数，它接受一个数组作为参数，并将数组中的元素打印出来。注意，这里使用的 console.log() 函数不是定义在函数内部的，而是作为全局函数存在的。也就是说，它位于顶层作用域中，因此可以作为回调函数被调用。
         
         与 forEach() 方法类似，map(), filter(), reduce() 等方法也都是高阶函数。它们接受一个回调函数作为参数，并且返回一个新的函数。通过使用高阶函数，我们可以创建功能更强大的函数。
         
         ### 4.7 闭包（Closure）
         闭包是指一个函数和声明该函数的词法环境（即上下文环境）的组合。闭包可以把一些外部变量保存起来，这样使用闭包函数的时候，就不需要每次都传入这些变量。这对异步编程和面向对象编程有非常重要的意义。下面是一个简单的闭包示例：
         
         ```javascript
           function makeAdder(x) {
             return function(y) {
               return x + y;
             }
           }
           
           const add5 = makeAdder(5);
           console.log(add5(2)); // Output: 7
         ```
         
         此例中，makeAdder() 函数接受一个参数 x，并返回一个闭包函数。该闭包函数取代了 add5() 函数，而且其内部有一个匿名函数，在调用 add5() 时，该匿名函数会捕获 x 的值。这样做可以节省空间，并且在调用 add5() 时，可以省去参数传递的麻烦。
         
         ### 4.8 尾调用优化（Tail Call Optimization）
         在函数调用链的最后一个节点处，若该函数是尾调用，则称其为尾调用优化。尾调用优化是函数式编程中的重要优化手段，因为它可以减少栈空间的消耗，从而提高运行速度。
         ```javascript
           function fibonacci(n, first = 0, second = 1) {
             if (n <= 0) {
               return first;
             } else if (n === 1) {
               return second;
             } else {
               return fibonacci(n - 1, second, first + second);
             }
           }
           
           console.log(fibonacci(10)); // Output: 55
         ```
         
         此例中，fibonacci() 函数是一个尾递归函数。尾递归函数的特点是，函数的最后一条语句是返回函数调用语句。其余情况均为正常递归调用。如果一个函数发生尾调用优化，那么它的栈帧就会被弹出，节省栈空间。
         
         ### 4.9 call() 方法
         call() 方法调用一个函数，并以指定的 this 值和若干个参数来替换当前函数的 this 对象和 arguments 对象的值。它返回调用结果。
         
         ```javascript
           const person = {
             name: 'John',
             sayName: function() {
               console.log(`My name is ${this.name}`);
             }
           };
           
           person.sayName(); // Output: My name is John
           person.sayName.call({name: 'Jack'}); // Output: My name is Jack
         ```
         
         此例中，person 是一个对象，包含一个 sayName() 方法。调用 person.sayName() 时，函数内部的 `this` 指向 person 对象。通过 call() 方法可以更改函数执行时 `this` 绑定的对象。这里调用了 sayName() 函数，并将 `this` 指定为 `{name: 'Jack'}` 对象。因此，打印出的结果为 `My name is Jack`。
         
         ### 4.10 apply() 方法
         apply() 方法与 call() 方法相同，但是它接受两个参数，第二个参数是一个数组，该数组将作为 arguments 对象的值传递给函数。
         
         ```javascript
           const multiply = function(x, y) {
             return x * y;
           };
           
           multiply.apply(null, [2, 3]); // Output: 6
         ```
         
         此例中，multiply() 函数是一个普通的函数，它的两个参数分别是 x 和 y。apply() 方法将数组 `[2, 3]` 作为 arguments 对象的值传递给 multiply() 函数。因此，调用 multiply() 函数时，会将 x 设为 2，y 设为 3。因此，打印出的结果为 6。
         
         ### 4.11 bind() 方法
         bind() 方法创建一个新的函数，在 bind() 方法的第一个参数中指定 this 对象的上下文，返回的新函数将忽略原函数的 this 对象，只保留提供的 this 对象。
         ```javascript
           const person = {
             name: 'John',
             greet: function(greeting) {
               console.log(`${greeting}, my name is ${this.name}.`);
             }
           };
           
           const johnGreeting = person.greet.bind(person, 'Hello');
           johnGreeting(); // Output: Hello, my name is John.
         ```
         
         此例中，person 是一个对象，包含一个 greet() 方法。调用 johnGreeting() 函数时，会将函数 johnGreeting() 绑定到 person 对象。johnGreeting() 函数现在拥有 person 对象的上下文，并且会在 greet() 执行时使用 person.greet() 作为内部函数。因此，调用 johnGreeting() 函数时，就会显示 `'Hello, my name is John.'`。
         
         ## 五、具体代码实例和解释说明
         
         ### 5.1 对象字面量的扩展（Object Literal Extension）
         对象字面量的扩展语法是指可以在对象字面量中直接定义 getter 和 setter 函数，即可以在对象字面量中定义 get 属性和 set 属性。getter 函数的行为类似于属性的访问权限，而 setter 函数的行为类似于属性的设置权限。下面是一个简单的示例：
         
         ```javascript
           const user = {
             _age: 0,
             get age() {
               console.log('Getting age...');
               return this._age;
             },
             set age(value) {
               console.log('Setting age...');
               if (typeof value === 'number') {
                 this._age = value;
               } else {
                 throw new Error('Age must be a number.');
               }
             }
           };
           
           console.log(user.age); // Output: Getting age... 0
           
           user.age = 25;
           console.log(user.age); // Output: Setting age... 25
         ```
         
         此例中，user 对象包含了一个私有成员变量 `_age`，以及 getter 和 setter 函数 age()。用户只能通过 getter 函数来读取 age 属性，不能修改 age 属性，只能通过 setter 函数来设置 age 属性，并且 setter 函数会进行数据校验。当用户读取 age 属性时，getter 函数会输出提示信息，而当用户设置 age 属性时，setter 函数会检查是否传入的 value 为数字，然后设置 `_age` 的值。
         
         ### 5.2 函数参数的默认值
         在 ES6 中，函数参数的默认值可以设置为函数表达式，这使得函数参数的默认值可以动态计算。下面是一个简单的示例：
         
         ```javascript
           function calculateArea(width, height = width / 2) {
             return width * height;
           }
           
           console.log(calculateArea(10)); // Output: 50
           console.log(calculateArea(10, 5)); // Output: 25
         ```
         
         此例中，calculateArea() 函数接受两个参数 width 和 height，height 的默认值为 width 的一半。因此，当用户只传入 width 时，默认 height 值为 width / 2，计算得到的面积为 50。当用户同时传入 width 和 height 时，函数会使用 height 来覆盖掉默认值，并计算得到 25 的面积。
         
         ### 5.3 Rest 参数（Rest Parameters）
         所谓 rest 参数，就是指函数定义时，用户传入的参数数量不确定，只知道一定数量的参数，就使用 rest 参数来收集这些参数。rest 参数的语法是三个点（…）加上参数名称。下面是一个简单的示例：
         
         ```javascript
           function sum(...args) {
             return args.reduce((accumulatedValue, currentValue) => accumulatedValue + currentValue);
           }
           
           console.log(sum(1, 2, 3)); // Output: 6
         ```
         
         此例中，sum() 函数接收任意数量的参数，并使用 rest 参数收集这些参数。它将参数数组 args 传递给 Array.prototype.reduce() 方法，并进行求和运算，返回结果。
         
         ### 5.4 Spread Operator（Spread Operator）
         所谓 spread operator（展开运算符），就是指在函数调用中，使用三个点（...）对参数列表进行扩展。下面是一个简单的示例：
         
         ```javascript
           const names = ['Alice', 'Bob'];
           const surnames = ['Smith', 'Taylor'];
           
           const fullNameList = [...names,...surnames].join(', ');
           
           console.log(fullNameList); // Output: Alice Smith, Bob Taylor
         ```
         
         此例中，names 和 surnames 数组分别存有名字和姓氏的列表。通过 spread operator（...）将数组扩展为独立参数，并调用 join() 方法连接两个数组，得到完整的姓名列表。打印出的结果为 `'Alice Smith, Bob Taylor'`。
         
         ### 5.5 Class 的语法糖
         ES6 引入了类（class）的概念，它提供了一种面向对象编程的更高级的写法。类的语法是通过 class 关键字创建的。下面是一个简单的示例：
         
         ```javascript
           class Person {
             constructor(firstName, lastName) {
               this.firstName = firstName;
               this.lastName = lastName;
             }
             
             sayHi() {
               console.log(`Hi, I'm ${this.firstName} ${this.lastName}.`);
             }
           }
           
           const peter = new Person('Peter', 'Parker');
           peter.sayHi(); // Output: Hi, I'm Peter Parker.
         ```
         
         此例中，Person 是一个类，包含构造函数和 sayHi() 方法。通过 new 操作符新建一个 Person 实例，并将 'Peter' 和 'Parker' 作为参数传入。peter 变量将保存 Person 实例，调用 peter.sayHi() 可以看到打印出的消息。
         
         ### 5.6 Promise 对象
         ES6 提供的 Promise 对象是异步编程的基石。Promise 有三种状态：pending（等待中），fulfilled（已成功），rejected（已失败）。下面是一个简单的示例：
         
         ```javascript
           const promise = new Promise((resolve, reject) => {
             setTimeout(() => {
               resolve("Success!");
             }, 2000);
           });
           
           promise.then(result => {
             console.log(result);
           })
           .catch(error => {
             console.error(error);
           });
           // Output after 2 seconds: Success!
         ```
         
         此例中，promise 对象是一个异步操作的承诺，两秒钟后，会将状态置为 resolved，并返回结果为 "Success!"。promise 对象将使用 then() 方法注册一个成功回调函数，用于处理成功的结果。如果 Promise 对象被 rejected（即调用了 reject() 函数），则会进入 catch() 方法注册的异常回调函数。
         
         ### 5.7 Generator 函数
         Generator 函数是ES6提供的一种异步编程方案。它可以让异步操作以同步编程的方式编写。下面是一个简单的示例：
         
         ```javascript
           function* generatorFunc() {
             yield 1;
             yield 2;
             yield 3;
           }
           
           for (let item of generatorFunc()) {
             console.log(item);
           }
           // Output: 1 2 3
         ```
         
         此例中，generatorFunc() 函数是一个 Generator 函数，它使用 yield 关键字返回一个值。调用这个函数并返回的是一个 Generator 对象，而这个 Generator 对象实现了 Iterator 接口。Generator 对象在调用 next() 方法时，遇到 yield 关键字，就会返回 yield 后面紧跟的值。调用完毕后，next() 方法会返回 done: true 的结果。