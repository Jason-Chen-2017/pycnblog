                 

## Node.js 异步编程：事件循环和回调

### 1. Node.js 中的异步编程是什么？

**题目：** 什么是 Node.js 中的异步编程？请简要解释。

**答案：** 异步编程是一种编程范式，允许程序在执行某些任务时，不等待任务完成就继续执行其他任务。在 Node.js 中，异步编程是通过回调函数、Promise 对象和异步/await 语法实现的。

**解析：** Node.js 使用单线程模型，通过事件循环来处理异步操作。在异步编程中，任务（如 I/O 操作）会在完成后通知主线程，主线程可以继续执行其他任务，从而提高程序的性能。

### 2. 事件循环是什么？

**题目：** 事件循环在 Node.js 中有什么作用？请简要解释。

**答案：** 事件循环是 Node.js 中处理异步操作的核心机制。它负责监听异步事件，执行回调函数，并管理定时器、队列等。

**解析：** 事件循环不断地检查任务队列，如果有可执行的回调函数，就将其从队列中取出并执行。事件循环还负责执行定时器回调、处理 I/O 事件等。事件循环是 Node.js 能够实现并发处理的基础。

### 3. 回调函数在 Node.js 中如何使用？

**题目：** 在 Node.js 中，如何使用回调函数处理异步操作？

**答案：** 在 Node.js 中，处理异步操作通常需要传递一个回调函数给异步方法。当异步操作完成时，该方法会调用回调函数，传递结果或错误信息。

**举例：**

```javascript
fs.readFile('example.txt', function (err, data) {
  if (err) {
    console.error('Error:', err);
  } else {
    console.log('Data:', data);
  }
});
```

**解析：** 在这个例子中，`fs.readFile` 方法用于异步读取文件。当文件读取完成后，它将结果传递给回调函数。如果发生错误，回调函数的第一个参数是错误对象；如果操作成功，回调函数的第二个参数是读取到的数据。

### 4. Promise 对象是什么？

**题目：** 什么是 Promise 对象？它在 Node.js 中有什么作用？

**答案：** Promise 对象是一种用于表示异步操作最终完成（成功或失败）及其结果的 JavaScript 对象。在 Node.js 中，Promise 对象用于简化异步操作的编写，提供更明确的错误处理机制。

**举例：**

```javascript
const fs = require('fs').promises;

fs.readFile('example.txt')
  .then(data => console.log('Data:', data))
  .catch(err => console.error('Error:', err));
```

**解析：** 在这个例子中，`fs.readFile` 方法返回一个 Promise 对象。当文件读取成功时，Promise 的 `then` 方法将被调用，并传递读取到的数据；如果发生错误，Promise 的 `catch` 方法将被调用，并传递错误信息。

### 5. 异步/await 语法是什么？

**题目：** 异步/await 语法在 Node.js 中有什么作用？

**答案：** 异步/await 语法是一种用于简化异步代码编写的语法糖，它允许开发者使用同步代码的写法来处理异步操作。

**举例：**

```javascript
async function readData() {
  try {
    const data = await fs.readFile('example.txt');
    console.log('Data:', data);
  } catch (err) {
    console.error('Error:', err);
  }
}
```

**解析：** 在这个例子中，`readData` 函数是一个异步函数，使用 `await` 关键字等待 `fs.readFile` 方法完成。如果方法成功完成，`await` 返回读取到的数据；如果发生错误，`catch` 块会捕获错误并处理。

### 6. Node.js 中如何避免回调地狱？

**题目：** 什么是回调地狱？如何在 Node.js 中避免回调地狱？

**答案：** 回调地狱是一种因回调函数嵌套过多而导致的代码可读性和可维护性降低的问题。在 Node.js 中，可以通过使用 Promise、async/await 或模块化编程等方式来避免回调地狱。

**举例：**

```javascript
// 回调地狱
fs.readFile('example.txt', function (err, data) {
  if (err) {
    console.error('Error:', err);
  } else {
    fs.writeFile('example.txt', data.toString(), function (err) {
      if (err) {
        console.error('Error:', err);
      } else {
        console.log('File written successfully');
      }
    });
  }
});

// 使用 Promise 避免回调地狱
fs.readFile('example.txt')
  .then(data => fs.writeFile('example.txt', data.toString()))
  .catch(err => console.error('Error:', err));
```

**解析：** 在这个例子中，使用 Promise 对象可以避免回调函数的嵌套，使代码更加清晰和易于维护。

### 7. Node.js 中的非阻塞 I/O 是什么？

**题目：** 什么是 Node.js 中的非阻塞 I/O？它有哪些优点？

**答案：** 非阻塞 I/O 是一种编程模型，允许程序在等待 I/O 操作完成时继续执行其他任务。在 Node.js 中，所有 I/O 操作都是非阻塞的，这意味着程序不会在 I/O 操作期间阻塞。

**优点：**

* 提高程序的性能，因为程序可以同时处理多个 I/O 操作。
* 支持并发处理，使程序能够高效地利用系统资源。

### 8. 事件监听器在 Node.js 中有什么作用？

**题目：** 事件监听器在 Node.js 中有什么作用？

**答案：** 事件监听器是 Node.js 中的一个核心概念，用于监听和响应特定的事件。通过事件监听器，Node.js 可以实现模块之间的通信和数据传递。

**举例：**

```javascript
const server = require('http').createServer();
server.on('request', function (req, res) {
  res.end('Hello, World!');
});
server.listen(3000);
```

**解析：** 在这个例子中，`server` 对象监听了 `request` 事件，并在接收到 HTTP 请求时调用回调函数。通过事件监听器，Node.js 可以处理各种事件，如 I/O 操作、定时器等。

### 9. Node.js 中的流是什么？

**题目：** 什么是 Node.js 中的流？它有哪些类型？

**答案：** 流是 Node.js 中用于处理大量数据的抽象概念，允许程序以连续的方式读取或写入数据。

**类型：**

* **可读流（Readable Stream）：** 可以从流中读取数据。
* **可写流（Writable Stream）：** 可以向流中写入数据。
* **双工流（Duplex Stream）：** 同时具有可读和可写功能。
* **变换流（Transform Stream）：** 在数据流中添加或删除数据。

**举例：**

```javascript
const fs = require('fs');
const readable = fs.createReadStream('example.txt');
const writable = fs.createWriteStream('example_copy.txt');

readable.pipe(writable);
```

**解析：** 在这个例子中，`createReadStream` 方法创建了一个可读流，`createWriteStream` 方法创建了一个可写流。使用 `pipe` 方法，可以将可读流的数据传递给可写流，实现数据的读取和写入。

### 10. Node.js 中的 Buffer 对象是什么？

**题目：** 什么是 Node.js 中的 Buffer 对象？它有什么作用？

**答案：** Buffer 对象是 Node.js 中用于处理二进制数据的缓冲区。它允许程序存储、读取和操作二进制数据。

**作用：**

* 用于读取和写入文件。
* 在网络通信中，用于处理二进制数据。
* 在处理 Buffer 对象时，可以执行各种操作，如复制、拼接、切片等。

**举例：**

```javascript
const buffer1 = Buffer.from('Hello, World!');
const buffer2 = Buffer.alloc(10);
buffer1.copy(buffer2, 0, 0, 10);
```

**解析：** 在这个例子中，`Buffer.from` 方法创建了一个包含字符串 'Hello, World!' 的 Buffer 对象。`Buffer.alloc` 方法创建了一个长度为 10 的 Buffer 对象。使用 `copy` 方法，可以将 `buffer1` 的数据复制到 `buffer2` 中。

### 11. Node.js 中的模块是什么？

**题目：** 什么是 Node.js 中的模块？它有哪些特点？

**答案：** 模块是 Node.js 中用于组织和复用代码的基本单元。模块可以将代码分割成独立的、可重用的部分，便于维护和扩展。

**特点：**

* **局部作用域：** 模块内部的变量和函数只在模块内部有效。
* **导出和导入：** 通过 `export` 和 `import` 语句，可以在模块之间共享变量和函数。
* **CommonJS 规范：** Node.js 使用 CommonJS 模块规范，允许模块导出和导入对象、函数、类等。

**举例：**

```javascript
// module.js
exports.add = function (a, b) {
  return a + b;
};

// main.js
const module = require('./module');
console.log(module.add(2, 3)); // 输出 5
```

**解析：** 在这个例子中，`module.js` 模块导出了一个名为 `add` 的函数。`main.js` 模块通过 `require` 方法导入 `module.js` 模块，并调用 `add` 函数。

### 12. Node.js 中的全局对象是什么？

**题目：** 什么是 Node.js 中的全局对象？它有哪些属性和方法？

**答案：** 全局对象（`global`）是 Node.js 中唯一的全局对象，用于存储全局变量、函数和属性。

**属性和方法：**

* **`global`:** 指向全局对象本身。
* **`process`:** 一个包含有关 Node.js 进程信息的对象，例如进程 ID、环境变量等。
* **`Buffer`:** 用于处理二进制数据的 Buffer 对象。
* **`console`:** 用于打印日志和控制台输出的对象。
* **`require`:** 用于导入模块的方法。

**举例：**

```javascript
console.log(process.pid);
console.log(global.Buffer.alloc(10));
```

**解析：** 在这个例子中，`process.pid` 属性用于获取当前进程的 ID，`global.Buffer.alloc(10)` 创建了一个长度为 10 的 Buffer 对象。

### 13. Node.js 中的事件发射器是什么？

**题目：** 什么是 Node.js 中的事件发射器（Emitter）？它有什么作用？

**答案：** 事件发射器（Emitter）是一个用于处理事件和监听器的类，允许程序监听特定事件并响应。

**作用：**

* 在 Node.js 中，许多核心模块都实现了 Emitter 接口，例如 `fs`、`http`、`net` 等。
* 用于实现模块之间的通信和数据传递。

**举例：**

```javascript
const events = require('events');
const emitter = new events.EventEmitter();

emitter.on('message', function (message) {
  console.log('Received message:', message);
});

emitter.emit('message', 'Hello, World!');
```

**解析：** 在这个例子中，`emitter` 对象监听了 `message` 事件，并在接收到事件时调用回调函数。通过 `emit` 方法，可以触发事件并传递数据。

### 14. Node.js 中的非阻塞 I/O 和阻塞 I/O 有什么区别？

**题目：** 什么是 Node.js 中的非阻塞 I/O 和阻塞 I/O？它们之间有什么区别？

**答案：** 非阻塞 I/O 和阻塞 I/O 是两种不同的 I/O 模型，用于处理 I/O 操作。

**非阻塞 I/O：**

* 允许程序在等待 I/O 操作完成时继续执行其他任务。
* I/O 操作不会阻塞程序执行。

**阻塞 I/O：**

* 必须等待 I/O 操作完成才能继续执行后续操作。
* I/O 操作会阻塞程序执行。

**区别：**

* 性能：非阻塞 I/O 更适合高并发场景，能够提高程序性能。
* 代码复杂性：非阻塞 I/O 代码相对复杂，需要处理回调函数。
* 应用场景：非阻塞 I/O 适用于 I/O 密集型任务，如网络通信、文件读写等；阻塞 I/O 适用于计算密集型任务，如数学计算、数据处理等。

**举例：**

```javascript
// 阻塞 I/O
fs.readFile('example.txt', function (err, data) {
  if (err) {
    console.error('Error:', err);
  } else {
    console.log('Data:', data);
  }
});

// 非阻塞 I/O
fs.readFile('example.txt', (err, data) => {
  if (err) {
    console.error('Error:', err);
  } else {
    console.log('Data:', data);
  }
});
```

**解析：** 在这个例子中，`fs.readFile` 方法用于读取文件。第一个例子使用了阻塞 I/O 模型，程序在等待文件读取完成时会被阻塞；第二个例子使用了非阻塞 I/O 模型，程序在等待文件读取完成时可以继续执行其他任务。

### 15. Node.js 中的定时器是什么？

**题目：** 什么是 Node.js 中的定时器？它有哪些方法？

**答案：** 定时器是 Node.js 中用于在指定时间后执行代码的功能。定时器通过 `setTimeout` 和 `setInterval` 方法实现。

**方法：**

* **`setTimeout(callback, delay)`:** 在延迟 `delay` 毫秒后执行 `callback` 函数。
* **`setInterval(callback, interval)`:** 每隔 `interval` 毫秒执行一次 `callback` 函数。

**举例：**

```javascript
// 在 2 秒后执行一次函数
setTimeout(() => {
  console.log('Hello, World!');
}, 2000);

// 每隔 2 秒执行一次函数
setInterval(() => {
  console.log('Hello, World!');
}, 2000);
```

**解析：** 在这个例子中，`setTimeout` 方法将在 2 秒后执行 `console.log` 函数；`setInterval` 方法将每隔 2 秒执行一次 `console.log` 函数。

### 16. Node.js 中的异步/await 语法是什么？

**题目：** 什么是 Node.js 中的异步/await 语法？它如何简化异步代码的编写？

**答案：** 异步/await 语法是 Node.js 中用于简化异步代码编写的语法糖。它允许开发者使用同步代码的写法来处理异步操作。

**如何简化异步代码的编写：**

* **避免回调函数的嵌套：** 通过使用 `await` 关键字，可以将异步操作转换为同步操作，从而避免回调函数的嵌套。
* **增强代码的可读性：** 异步/await 语法使得异步代码更加清晰和易于理解。

**举例：**

```javascript
async function fetchData() {
  try {
    const data = await fs.readFile('example.txt');
    console.log('Data:', data);
  } catch (err) {
    console.error('Error:', err);
  }
}
```

**解析：** 在这个例子中，`fetchData` 函数是一个异步函数，使用 `await` 关键字等待 `fs.readFile` 方法完成。如果操作成功，`await` 返回读取到的数据；如果发生错误，`catch` 块会捕获错误并处理。

### 17. Node.js 中的错误处理机制是什么？

**题目：** Node.js 中有哪些错误处理机制？请简要介绍。

**答案：** Node.js 中有几种常见的错误处理机制：

* **回调函数的错误处理：** 在异步操作中，通过回调函数的第一个参数传递错误对象。
* **Promise 的错误处理：** 使用 Promise 的 `then` 和 `catch` 方法分别处理成功和错误情况。
* **异步/await 的错误处理：** 使用 `try` 和 `catch` 块来捕获和处理异步代码中的错误。

**举例：**

```javascript
// 回调函数的错误处理
fs.readFile('example.txt', function (err, data) {
  if (err) {
    console.error('Error:', err);
  } else {
    console.log('Data:', data);
  }
});

// Promise 的错误处理
fs.readFile('example.txt')
  .then(data => console.log('Data:', data))
  .catch(err => console.error('Error:', err));

// 异步/await 的错误处理
async function fetchData() {
  try {
    const data = await fs.readFile('example.txt');
    console.log('Data:', data);
  } catch (err) {
    console.error('Error:', err);
  }
}
```

**解析：** 在这些例子中，分别展示了回调函数、Promise 和异步/await 语法中的错误处理方法。

### 18. Node.js 中的模块系统是什么？

**题目：** 什么是 Node.js 中的模块系统？它有哪些特点？

**答案：** Node.js 中的模块系统是一种用于组织和复用代码的机制。模块可以将代码分割成独立的、可重用的部分。

**特点：**

* **局部作用域：** 模块内部的变量和函数只在模块内部有效。
* **导出和导入：** 通过 `export` 和 `import` 语句，可以在模块之间共享变量和函数。
* **CommonJS 规范：** Node.js 使用 CommonJS 模块规范，允许模块导出和导入对象、函数、类等。

**举例：**

```javascript
// module.js
exports.add = function (a, b) {
  return a + b;
};

// main.js
const module = require('./module');
console.log(module.add(2, 3)); // 输出 5
```

**解析：** 在这个例子中，`module.js` 模块导出了一个名为 `add` 的函数。`main.js` 模块通过 `require` 方法导入 `module.js` 模块，并调用 `add` 函数。

### 19. Node.js 中的 Buffer 对象是什么？

**题目：** 什么是 Node.js 中的 Buffer 对象？它有什么作用？

**答案：** Buffer 对象是 Node.js 中用于处理二进制数据的缓冲区。它允许程序存储、读取和操作二进制数据。

**作用：**

* 用于读取和写入文件。
* 在网络通信中，用于处理二进制数据。
* 在处理 Buffer 对象时，可以执行各种操作，如复制、拼接、切片等。

**举例：**

```javascript
const buffer1 = Buffer.from('Hello, World!');
const buffer2 = Buffer.alloc(10);
buffer1.copy(buffer2, 0, 0, 10);
```

**解析：** 在这个例子中，`Buffer.from` 方法创建了一个包含字符串 'Hello, World!' 的 Buffer 对象。`Buffer.alloc` 方法创建了一个长度为 10 的 Buffer 对象。使用 `copy` 方法，可以将 `buffer1` 的数据复制到 `buffer2` 中。

### 20. Node.js 中的流是什么？

**题目：** 什么是 Node.js 中的流？它有哪些类型？

**答案：** 流是 Node.js 中用于处理大量数据的抽象概念，允许程序以连续的方式读取或写入数据。

**类型：**

* **可读流（Readable Stream）：** 可以从流中读取数据。
* **可写流（Writable Stream）：** 可以向流中写入数据。
* **双工流（Duplex Stream）：** 同时具有可读和可写功能。
* **变换流（Transform Stream）：** 在数据流中添加或删除数据。

**举例：**

```javascript
const fs = require('fs');
const readable = fs.createReadStream('example.txt');
const writable = fs.createWriteStream('example_copy.txt');

readable.pipe(writable);
```

**解析：** 在这个例子中，`createReadStream` 方法创建了一个可读流，`createWriteStream` 方法创建了一个可写流。使用 `pipe` 方法，可以将可读流的数据传递给可写流，实现数据的读取和写入。

### 21. Node.js 中的事件循环是什么？

**题目：** 什么是 Node.js 中的事件循环？它如何处理异步操作？

**答案：** 事件循环是 Node.js 中用于处理异步操作的核心机制。它负责监听异步事件，执行回调函数，并管理定时器、队列等。

**如何处理异步操作：**

1. 异步操作完成后，将结果放入任务队列。
2. 事件循环不断地检查任务队列，如果有可执行的回调函数，就将其从队列中取出并执行。
3. 事件循环还负责执行定时器回调、处理 I/O 事件等。

**举例：**

```javascript
setImmediate(() => {
  console.log('Immediate');
});

process.nextTick(() => {
  console.log('Next tick');
});

// 输出顺序：Next tick, Immediate
```

**解析：** 在这个例子中，`setImmediate` 和 `process.nextTick` 方法用于设置异步操作。输出顺序为 `Next tick` 和 `Immediate`，因为 `process.nextTick` 的回调函数在事件循环的下一轮执行，而 `setImmediate` 的回调函数在下一轮事件循环的末尾执行。

### 22. Node.js 中的异步编程有哪些优点？

**题目：** Node.js 中的异步编程有哪些优点？

**答案：** Node.js 中的异步编程有以下优点：

1. **非阻塞 I/O：** 异步编程允许程序在等待 I/O 操作完成时继续执行其他任务，提高程序的性能和并发处理能力。
2. **更好的错误处理：** 异步编程提供了更明确的错误处理机制，如 Promise 和异步/await 语法，使得错误处理更加简单和清晰。
3. **代码可读性：** 异步编程使得代码更加简洁和易于理解，减少了回调函数的嵌套和复杂性。

### 23. Node.js 中的 HTTP 服务是什么？

**题目：** 什么是 Node.js 中的 HTTP 服务？它有哪些特点？

**答案：** Node.js 中的 HTTP 服务是一种用于处理 HTTP 请求和响应的机制。它允许 Node.js 创建 Web 服务器，接收和处理客户端的请求。

**特点：**

1. **非阻塞 I/O：** Node.js 使用非阻塞 I/O 模型，允许程序同时处理多个请求，提高并发处理能力。
2. **简单易用：** Node.js 提供了 `http` 模块，使得创建 HTTP 服务非常简单。
3. **扩展性强：** Node.js 支持自定义中间件，可以方便地添加额外的功能。

**举例：**

```javascript
const http = require('http');

const server = http.createServer((req, res) => {
  res.writeHead(200, {'Content-Type': 'text/plain'});
  res.end('Hello, World!');
});

server.listen(3000, () => {
  console.log('Server running at http://localhost:3000/');
});
```

**解析：** 在这个例子中，`http.createServer` 方法创建了一个 HTTP 服务器，并在接收到请求时调用回调函数。`server.listen` 方法用于启动服务器，并指定端口号。

### 24. Node.js 中的进程是什么？

**题目：** 什么是 Node.js 中的进程？它有哪些作用？

**答案：** 进程是 Node.js 中用于管理和运行程序的单元。它包含程序的代码、数据、资源和状态。

**作用：**

1. **并发处理：** 进程可以同时处理多个任务，提高程序的并发处理能力。
2. **资源隔离：** 进程之间相互独立，一个进程的崩溃不会影响其他进程。
3. **负载均衡：** 可以通过创建多个进程来均衡负载，提高程序的性能。

**举例：**

```javascript
const { fork } = require('child_process');

const worker = fork('./worker.js');

worker.on('message', function (msg) {
  console.log('Received message:', msg);
});

worker.send({ data: 'Hello, Worker!' });
```

**解析：** 在这个例子中，`fork` 方法创建了一个新的子进程，并运行 `worker.js` 脚本。子进程通过 `message` 事件接收父进程发送的消息，并通过 `send` 方法发送消息给父进程。

### 25. Node.js 中的全局变量是什么？

**题目：** 什么是 Node.js 中的全局变量？它有哪些？

**答案：** Node.js 中的全局变量是在整个程序中都可以访问的变量。全局变量包括以下内容：

* **`global`:** 指向全局对象。
* **`Buffer`:** 用于处理二进制数据的 Buffer 对象。
* **`process`:** 包含有关 Node.js 进程信息的对象。
* **`console`:** 用于打印日志和控制台输出的对象。
* **`require`:** 用于导入模块的方法。

**举例：**

```javascript
console.log(global === this); // 输出 true
```

**解析：** 在这个例子中，`global` 和 `this` 都指向全局对象，因此 `console.log` 函数可以访问全局变量。

### 26. Node.js 中的事件监听器是什么？

**题目：** 什么是 Node.js 中的事件监听器？它有什么作用？

**答案：** 事件监听器是 Node.js 中用于监听和响应特定事件的机制。事件监听器允许程序在接收到事件时执行相应的回调函数。

**作用：**

1. **实现模块间的通信：** 通过事件监听器，模块可以监听和响应其他模块的事件，实现数据传递和协作。
2. **扩展功能：** 事件监听器使得 Node.js 可以轻松地扩展功能，例如添加自定义事件处理程序。

**举例：**

```javascript
const emitter = require('events').EventEmitter;

const myEmitter = new emitter();

myEmitter.on('message', function (message) {
  console.log('Received message:', message);
});

myEmitter.emit('message', 'Hello, World!');
```

**解析：** 在这个例子中，`myEmitter` 对象监听了 `message` 事件，并在接收到事件时调用回调函数。通过 `emit` 方法，可以触发事件并传递数据。

### 27. Node.js 中的模块是什么？

**题目：** 什么是 Node.js 中的模块？它有哪些作用？

**答案：** Node.js 中的模块是一种用于组织和复用代码的基本单元。模块可以将代码分割成独立的、可重用的部分。

**作用：**

1. **代码复用：** 通过模块，可以将公共代码提取到独立的模块中，便于复用和维护。
2. **隔离作用域：** 模块内部变量和函数只在模块内部有效，避免命名冲突。
3. **模块化编程：** 通过模块，可以实现模块化编程，使代码更加清晰和易于维护。

**举例：**

```javascript
// module.js
exports.add = function (a, b) {
  return a + b;
};

// main.js
const module = require('./module');
console.log(module.add(2, 3)); // 输出 5
```

**解析：** 在这个例子中，`module.js` 模块导出了一个名为 `add` 的函数。`main.js` 模块通过 `require` 方法导入 `module.js` 模块，并调用 `add` 函数。

### 28. Node.js 中的 Buffer 对象是什么？

**题目：** 什么是 Node.js 中的 Buffer 对象？它有什么作用？

**答案：** Buffer 对象是 Node.js 中用于处理二进制数据的缓冲区。它允许程序存储、读取和操作二进制数据。

**作用：**

1. **读取和写入文件：** Buffer 对象可以用于读取和写入文件，处理文件内容。
2. **网络通信：** 在网络通信中，Buffer 对象用于处理二进制数据，如 HTTP 请求和响应。
3. **数据操作：** 在处理 Buffer 对象时，可以执行各种操作，如复制、拼接、切片等。

**举例：**

```javascript
const buffer1 = Buffer.from('Hello, World!');
const buffer2 = Buffer.alloc(10);
buffer1.copy(buffer2, 0, 0, 10);
```

**解析：** 在这个例子中，`Buffer.from` 方法创建了一个包含字符串 'Hello, World!' 的 Buffer 对象。`Buffer.alloc` 方法创建了一个长度为 10 的 Buffer 对象。使用 `copy` 方法，可以将 `buffer1` 的数据复制到 `buffer2` 中。

### 29. Node.js 中的流是什么？

**题目：** 什么是 Node.js 中的流？它有哪些类型？

**答案：** 流是 Node.js 中用于处理大量数据的抽象概念，允许程序以连续的方式读取或写入数据。

**类型：**

1. **可读流（Readable Stream）：** 可以从流中读取数据。
2. **可写流（Writable Stream）：** 可以向流中写入数据。
3. **双工流（Duplex Stream）：** 同时具有可读和可写功能。
4. **变换流（Transform Stream）：** 在数据流中添加或删除数据。

**举例：**

```javascript
const fs = require('fs');
const readable = fs.createReadStream('example.txt');
const writable = fs.createWriteStream('example_copy.txt');

readable.pipe(writable);
```

**解析：** 在这个例子中，`createReadStream` 方法创建了一个可读流，`createWriteStream` 方法创建了一个可写流。使用 `pipe` 方法，可以将可读流的数据传递给可写流，实现数据的读取和写入。

### 30. Node.js 中的全局对象是什么？

**题目：** 什么是 Node.js 中的全局对象？它有哪些属性和方法？

**答案：** 全局对象是 Node.js 中唯一的全局对象，用于存储全局变量、函数和属性。

**属性和方法：**

1. **`global`:** 指向全局对象本身。
2. **`Buffer`:** 用于处理二进制数据的 Buffer 对象。
3. **`process`:** 包含有关 Node.js 进程信息的对象。
4. **`console`:** 用于打印日志和控制台输出的对象。
5. **`require`:** 用于导入模块的方法。

**举例：**

```javascript
console.log(global === this); // 输出 true
```

**解析：** 在这个例子中，`global` 和 `this` 都指向全局对象，因此 `console.log` 函数可以访问全局变量。

