                 

### 文章标题

《Web前端开发：HTML、CSS 和 JavaScript》

### 关键词

- Web前端开发
- HTML
- CSS
- JavaScript
- 响应式设计
- 前端框架
- 前端工程化

### 摘要

本文旨在为初学者和有经验的开发者提供Web前端开发的全面指南。文章从基础概念入手，逐步深入到HTML、CSS和JavaScript的核心知识，并涵盖响应式设计、前端框架以及前端工程化的实战技巧。通过详细的代码示例和项目实战，帮助读者掌握Web前端开发的实用技能，为构建现代网页打下坚实基础。

---

### 《Web前端开发：HTML、CSS 和 JavaScript》目录大纲

#### 第一部分：Web前端开发基础

##### 1.1 Web前端开发概述
###### 1.1.1 Web前端开发的重要性
###### 1.1.2 Web前端开发的技术栈

##### 1.2 HTML基础
###### 1.2.1 HTML文档结构
###### 1.2.2 标签与属性
###### 1.2.3 表单与表单验证

##### 1.3 CSS基础
###### 1.3.1 CSS选择器
###### 1.3.2 布局与定位
###### 1.3.3 伪类与伪元素
###### 1.3.4 动画与过渡效果

##### 1.4 JavaScript基础
###### 1.4.1 基本语法与数据类型
###### 1.4.2 函数与闭包
###### 1.4.3 对象与数组

#### 第二部分：Web前端开发进阶

##### 2.1 响应式设计
###### 2.1.1 媒体查询与断点
###### 2.1.2 常用响应式布局方式
###### 2.1.3 CSS框架简介

##### 2.2 JavaScript高级特性
###### 2.2.1 异步编程
###### 2.2.2 ES6新特性
###### 2.2.3 模块化开发

##### 2.3 前端框架与库
###### 2.3.1 React基础
###### 2.3.2 Vue基础
###### 2.3.3 Angular基础

##### 2.4 前端工程化
###### 2.4.1 Webpack使用
###### 2.4.2 Babel使用
###### 2.4.3 性能优化策略

#### 第三部分：Web前端开发实战

##### 3.1 项目实战一：个人博客网站
###### 3.1.1 项目需求分析
###### 3.1.2 网站设计思路
###### 3.1.3 网站开发步骤
###### 3.1.4 源代码解读

##### 3.2 项目实战二：在线购物平台
###### 3.2.1 项目需求分析
###### 3.2.2 网站设计思路
###### 3.2.3 网站开发步骤
###### 3.2.4 源代码解读

##### 3.3 项目实战三：在线教育平台
###### 3.3.1 项目需求分析
###### 3.3.2 网站设计思路
###### 3.3.3 网站开发步骤
###### 3.3.4 源代码解读

#### 附录

##### 附录A：开发工具与资源
###### 附录A.1 HTML5常用标签与属性
###### 附录A.2 CSS3常用属性与选择器
###### 附录A.3 JavaScript常用方法与函数

##### 附录B：参考文献
###### 附录B.1 相关书籍推荐
###### 附录B.2 在线资源推荐

---

### 第1章：Web前端开发概述

#### 1.1.1 Web前端开发的重要性

Web前端开发是构建网站和网页的核心环节，它决定了用户在使用网站时的体验。前端开发人员需要确保网站在各种设备上都能良好显示，并提供流畅的交互体验。随着互联网的快速发展，Web前端开发的重要性日益凸显，它已成为IT行业中一个热门且具有挑战性的领域。

Web前端开发的主要职责包括：

- **用户界面设计**：创建美观、易用的用户界面。
- **交互逻辑实现**：处理用户的输入和动作，实现动态效果。
- **响应式布局**：确保网站在不同设备和分辨率下都能良好显示。
- **性能优化**：提高网站的加载速度和响应效率。

#### 1.1.2 Web前端开发的技术栈

Web前端开发的技术栈主要包括HTML、CSS和JavaScript三大核心技术。这三者共同构成了Web前端开发的基石。

- **HTML（HyperText Markup Language）**：用于创建网页的结构和内容，它是网页的骨架。
- **CSS（Cascading Style Sheets）**：用于定义网页的样式和布局，它是网页的衣着。
- **JavaScript**：用于添加网页的交互功能，它是网页的“灵魂”。

此外，前端开发中还常用到各种框架、库和工具，例如React、Vue、Angular等框架，以及Webpack、Babel等构建工具。

### 第2章：HTML基础

#### 2.1 HTML文档结构

HTML文档的基本结构如下：

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Document</title>
</head>
<body>
  <!-- 网页内容 -->
</body>
</html>
```

- **DOCTYPE声明**：定义文档类型，确保浏览器以标准模式渲染网页。
- **html**：根元素，包含整个网页的所有内容。
- **head**：头部元素，包含文档的元数据，如字符集、标题等。
- **title**：标题元素，定义网页的标题，显示在浏览器标签上。
- **body**：主体元素，包含网页的实际内容。

#### 2.2 标签与属性

HTML标签用于定义网页中的不同元素，例如标题、段落、图像、表单等。每个标签都有自己的属性，用于描述该元素的特性。

以下是一些常用的HTML标签和属性：

- **h1-h6**：标题标签，`h1`是最高级别的标题，`h6`是最低级别的标题。
- **p**：段落标签，用于定义文本段落。
- **img**：图像标签，用于在网页中显示图像。
- **input**：输入标签，用于创建表单输入框。
- **button**：按钮标签，用于创建按钮。

示例：

```html
<h1>标题</h1>
<p>这是一个段落。</p>
<img src="image.jpg" alt="图像">
<input type="text" placeholder="输入文本">
<button>按钮</button>
```

#### 2.3 表单与表单验证

表单是网页中用于收集用户输入数据的重要组成部分。表单通常包含文本输入框、单选框、复选框、下拉菜单等元素。

```html
<form action="submit.html" method="post">
  <label for="name">姓名：</label>
  <input type="text" id="name" name="name" required>
  
  <label for="email">邮箱：</label>
  <input type="email" id="email" name="email" required>
  
  <input type="submit" value="提交">
</form>
```

表单验证是确保用户输入的数据符合预期格式的重要手段。HTML5提供了一些表单验证属性，如`required`、`type="email"`等，以简化验证过程。

### 第3章：CSS基础

#### 3.1 CSS选择器

CSS选择器用于选择和匹配网页中的元素。CSS选择器包括基本选择器、属性选择器和伪类选择器等。

- **基本选择器**：包括元素选择器、类选择器和ID选择器。
  - 元素选择器：选择所有相同类型的元素。
    ```css
    p {
      color: blue;
    }
    ```
  - 类选择器：选择具有相同类的元素。
    ```css
    .highlight {
      color: red;
    }
    ```
  - ID选择器：选择具有相同ID的元素。
    ```css
    #title {
      font-size: 24px;
    }
    ```

- **属性选择器**：选择具有特定属性的元素。
  ```css
  input[type="text"] {
    background-color: lightgray;
  }
  ```

- **伪类选择器**：选择元素在不同状态下的样式。
  ```css
  a:hover {
    color: red;
  }
  ```

#### 3.2 布局与定位

CSS布局用于定义网页中元素的位置和大小。常用的布局技术包括浮动、定位和Flexbox布局。

- **浮动**：通过设置元素的`float`属性来使元素向左或向右浮动。
  ```css
  .left {
    float: left;
    width: 200px;
    height: 100px;
    background-color: blue;
  }
  ```

- **定位**：通过设置元素的`position`属性来使元素相对于页面或父元素进行定位。
  ```css
  .absolute {
    position: absolute;
    top: 50px;
    left: 100px;
    width: 100px;
    height: 100px;
    background-color: red;
  }
  ```

- **Flexbox布局**：通过使用`display: flex;`属性来创建灵活的布局。
  ```css
  .flex-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  ```

#### 3.3 伪类与伪元素

伪类用于选择元素的不同状态，而伪元素用于选择元素的特殊部分。

- **伪类**：例如`:hover`伪类表示当鼠标悬停在元素上时的状态。
  ```css
  a:hover {
    color: red;
  }
  ```

- **伪元素**：例如`:before`伪元素用于在元素之前插入内容。
  ```css
  p:before {
    content: "Before ";
    color: blue;
  }
  ```

#### 3.4 动画与过渡效果

CSS3引入了动画和过渡效果，使网页元素能够实现更丰富的动态效果。

- **动画**：通过`@keyframes`规则定义动画。
  ```css
  @keyframes rotate {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  .circle {
    width: 100px;
    height: 100px;
    background-color: red;
    animation: rotate 2s linear infinite;
  }
  ```

- **过渡效果**：通过`transition`属性实现元素的平滑过渡效果。
  ```css
  .box {
    width: 100px;
    height: 100px;
    background-color: blue;
    transition: width 2s;
  }

  .box:hover {
    width: 200px;
  }
  ```

### 第4章：JavaScript基础

#### 4.1 基本语法与数据类型

JavaScript是一种解释型、面向对象的脚本语言，它为网页提供了动态交互功能。以下是JavaScript的基本语法和数据类型。

#### 基本语法

```javascript
// 声明变量
var name = "Alice";

// 输出
console.log(name);

// 注释
// 这是一行注释
/* 这是多行注释 */
```

#### 数据类型

JavaScript的数据类型包括：

- **数字（Number）**：表示整数和浮点数。
- **字符串（String）**：表示文本。
- **布尔值（Boolean）**：表示真或假。
- **对象（Object）**：表示各种数据结构。
- **函数（Function）**：表示可执行的代码块。
- **无值（Undefined）**：表示变量未初始化。
- **空值（Null）**：表示一个空对象指针。

#### 4.2 函数与闭包

函数是JavaScript的核心组成部分，用于执行特定任务的代码块。闭包是一种函数对象，它保存了创建该函数时的环境。

```javascript
// 声明函数
function greet(name) {
  return "Hello, " + name;
}

// 调用函数
console.log(greet("Alice"));

// 闭包示例
function makeCounter() {
  let count = 0;
  return function() {
    return count++;
  };
}

const counter = makeCounter();
console.log(counter()); // 1
console.log(counter()); // 2
```

#### 4.3 对象与数组

对象是JavaScript中的核心数据结构，用于存储属性和方法。数组是一种特殊的对象，用于存储一系列元素。

```javascript
// 创建对象
const person = {
  name: "Alice",
  age: 30
};

// 访问对象属性
console.log(person.name); // Alice

// 创建数组
const colors = ["red", "green", "blue"];

// 访问数组元素
console.log(colors[0]); // red

// 数组方法
colors.push("yellow");
console.log(colors); // ["red", "green", "blue", "yellow"]

// 遍历数组
colors.forEach(function(color) {
  console.log(color);
});
```

### 第5章：响应式设计

#### 5.1 媒体查询与断点

响应式设计是Web前端开发中的一项重要技术，它使网页能够适应不同设备和屏幕尺寸。媒体查询（Media Query）是响应式设计的核心。

```css
/* 小于600px的屏幕 */
@media screen and (max-width: 600px) {
  body {
    background-color: lightblue;
  }
}

/* 大于600px的屏幕 */
@media screen and (min-width: 600px) {
  body {
    background-color: lightgreen;
  }
}
```

断点（Breakpoint）是媒体查询中用于定义不同屏幕尺寸的值。常见的断点值包括：

- 小屏幕（Mobile）：小于768px
- 中屏幕（Tablet）：768px到1024px
- 大屏幕（Desktop）：大于1024px

#### 5.2 常用响应式布局方式

常见的响应式布局方式包括：

- **流体布局**：使用百分比宽度，使布局能够自适应屏幕尺寸。
  ```css
  .container {
    width: 80%;
    max-width: 1200px;
    margin: 0 auto;
  }
  ```

- **弹性布局**：使用Flexbox布局，使布局更加灵活。
  ```css
  .container {
    display: flex;
    justify-content: space-between;
  }
  ```

- **栅格系统**：使用预定义的网格布局，快速构建响应式页面。
  ```css
  .row {
    display: flex;
    flex-wrap: wrap;
  }

  .col {
    flex: 1;
    min-width: 300px;
    max-width: 600px;
    padding: 10px;
  }
  ```

#### 5.3 CSS框架简介

CSS框架是一种预定义的CSS样式库，用于简化响应式设计。常见的CSS框架包括Bootstrap、Foundation和Materialize等。

Bootstrap是一个流行的CSS框架，它提供了丰富的组件和预定义样式，使开发响应式网页更加简单。

```html
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
```

```css
.container {
  max-width: 1100px;
  margin: auto;
}
```

### 第6章：JavaScript高级特性

#### 6.1 异步编程

异步编程是一种处理并发和长时间运行任务的编程技术。JavaScript通过回调函数、Promise和异步/await等语法实现了异步编程。

```javascript
// 回调函数
function fetchData(callback) {
  setTimeout(() => {
    callback("Data fetched");
  }, 2000);
}

fetchData(function(data) {
  console.log(data);
});

// Promise
function fetchData() {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      resolve("Data fetched");
    }, 2000);
  });
}

fetchData()
  .then((data) => {
    console.log(data);
  })
  .catch((error) => {
    console.error(error);
  });

// 异步/await
async function fetchData() {
  try {
    const data = await fetchDataAsync();
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}
```

#### 6.2 ES6新特性

ES6（ECMAScript 2015）引入了多项新的语法和功能，使JavaScript编程更加高效和易用。ES6的新特性包括：

- **let和const**：用于声明变量，提供块级作用域。
- **箭头函数**：简化函数声明。
- **模板字符串**：用于构建字符串，支持嵌入变量和表达式。
- **解构赋值**：用于拆分和提取数组或对象中的值。
- **Promise**：用于处理异步操作。
- **类和模块**：提供了面向对象编程的支持。

#### 6.3 模块化开发

模块化开发是一种将代码拆分为可重用模块的编程技术。JavaScript通过模块化可以更好地组织和管理代码。

```javascript
// math.js
export function add(a, b) {
  return a + b;
}

export function subtract(a, b) {
  return a - b;
}

// main.js
import { add, subtract } from "./math.js";

console.log(add(5, 3)); // 8
console.log(subtract(5, 3)); // 2
```

### 第7章：前端框架与库

#### 7.1 React基础

React是一个用于构建用户界面的JavaScript库，它采用声明式编程模型，使开发动态、响应式的网页变得简单。

```jsx
import React from "react";

function Greeting(props) {
  return <h1>Hello, {props.name}</h1>;
}

const element = <Greeting name="Alice" />;
ReactDOM.render(element, document.getElementById("root"));
```

#### 7.2 Vue基础

Vue是一个用于构建用户界面的JavaScript框架，它提供了简洁、灵活的编程模型。

```html
<div id="app">
  <h1>{{ message }}</h1>
</div>
```

```javascript
new Vue({
  el: "#app",
  data: {
    message: "Hello, Vue!"
  }
});
```

#### 7.3 Angular基础

Angular是一个用于构建复杂前端应用程序的JavaScript框架，它提供了完整的功能和丰富的特性。

```html
<!DOCTYPE html>
<html>
  <head>
    <script src="https://unpkg.com/@angular/core@13.0.0-beta.0/fesm5/core.js"></script>
  </head>
  <body>
    <app-my-app></app-my-app>
    <script src="app.js"></script>
  </body>
</html>
```

```javascript
import { platformBrowserDynamic } from "@angular/platform-browser-dynamic";
import { AppModule } from "./app/app.module";

const platform = platformBrowserDynamic();
platform.bootstrapModule(AppModule);
```

### 第8章：前端工程化

#### 8.1 Webpack使用

Webpack是一个现代JavaScript应用的静态模块打包器，它将各种资源模块打包成一个或多个bundle。

```javascript
const path = require("path");

module.exports = {
  entry: "./src/index.js",
  output: {
    filename: "bundle.js",
    path: path.resolve(__dirname, "dist")
  },
  module: {
    rules: [
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"]
      }
    ]
  }
};
```

#### 8.2 Babel使用

Babel是一个用于将ES6+代码转换为向后兼容的JavaScript版本的编译器。它使得开发者可以使用最新的JavaScript特性，而不必担心浏览器的兼容性问题。

```json
{
  "presets": ["@babel/preset-env"],
  "plugins": ["@babel/plugin-proposal-class-properties"]
}
```

```javascript
// index.js
class Greeter {
  constructor(message) {
    this.message = message;
  }
  sayHello() {
    return `Hello, ${this.message}`;
  }
}
```

```javascript
// index.es6
class Greeter {
  constructor(message) {
    this.message = message;
  }
  sayHello() {
    return `Hello, ${this.message}`;
  }
}
```

#### 8.3 性能优化策略

性能优化是Web前端开发中至关重要的环节，它涉及到多个方面，如资源压缩、懒加载、代码分割等。

- **资源压缩**：通过压缩HTML、CSS和JavaScript文件，减少文件的体积。
- **懒加载**：延迟加载非必需的资源，提高页面加载速度。
- **代码分割**：将代码拆分为多个bundle，按需加载，减少初始加载时间。

### 第9章：项目实战一：个人博客网站

#### 9.1 项目需求分析

项目需求如下：

- 创建一个展示个人博客文章的网站。
- 网站应具有简洁明了的布局和良好的用户体验。
- 支持文章的发布、编辑和删除功能。
- 网站应响应式，适配不同设备和屏幕尺寸。

#### 9.2 网站设计思路

设计思路如下：

- 采用React框架，简化开发过程。
- 使用Redux管理应用状态。
- 使用Ant Design组件库，提高开发效率。

#### 9.3 网站开发步骤

1. **环境搭建**：安装Node.js、npm、Create React App等工具。
2. **项目结构**：创建项目文件夹，并初始化项目。
3. **组件开发**：编写React组件，实现页面布局和功能。
4. **状态管理**：使用Redux管理应用状态，实现数据的共享和更新。
5. **路由配置**：使用React Router进行页面路由管理。
6. **部署上线**：将项目打包并上传到服务器，部署到线上环境。

#### 9.4 源代码解读

以下是对项目源代码的详细解读：

```jsx
// App.js
import React from "react";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import Home from "./components/Home";
import About from "./components/About";
import Article from "./components/Article";

function App() {
  return (
    <Router>
      <div className="App">
        <Switch>
          <Route exact path="/" component={Home} />
          <Route path="/about" component={About} />
          <Route path="/article/:id" component={Article} />
        </Switch>
      </div>
    </Router>
  );
}

export default App;
```

以上代码是整个项目的入口文件，它使用React Router进行页面路由管理。通过`<Switch>`和`<Route>`组件，实现了不同路径的页面跳转。

```jsx
// components/Home.js
import React from "react";

function Home() {
  return (
    <div>
      <h1>Home</h1>
      <p>Welcome to my blog!</p>
    </div>
  );
}

export default Home;
```

以上代码是首页组件，它简单地展示了欢迎信息。

```jsx
// components/Article.js
import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";

function Article() {
  const [article, setArticle] = useState(null);
  const { id } = useParams();

  useEffect(() => {
    fetch(`/api/articles/${id}`)
      .then((response) => response.json())
      .then((data) => setArticle(data));
  }, [id]);

  if (!article) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <h1>{article.title}</h1>
      <p>{article.content}</p>
    </div>
  );
}

export default Article;
```

以上代码是文章详情组件，它通过`useParams`钩子获取文章ID，并使用`useEffect`钩子获取文章数据。如果文章数据加载完成，则显示文章内容。

### 第10章：项目实战二：在线购物平台

#### 10.1 项目需求分析

项目需求如下：

- 创建一个在线购物平台，提供商品浏览、添加购物车、下单购买等功能。
- 支持用户注册、登录和权限管理。
- 网站应具有响应式布局，适配不同设备和屏幕尺寸。

#### 10.2 网站设计思路

设计思路如下：

- 采用Vue框架，简化开发过程。
- 使用Vuex管理应用状态。
- 使用Element UI组件库，提高开发效率。

#### 10.3 网站开发步骤

1. **环境搭建**：安装Node.js、npm、Vue CLI等工具。
2. **项目结构**：创建项目文件夹，并初始化项目。
3. **组件开发**：编写Vue组件，实现页面布局和功能。
4. **状态管理**：使用Vuex管理应用状态，实现数据的共享和更新。
5. **路由配置**：使用Vue Router进行页面路由管理。
6. **接口调试**：使用Postman调试API接口。
7. **部署上线**：将项目打包并上传到服务器，部署到线上环境。

#### 10.4 源代码解读

以下是对项目源代码的详细解读：

```vue
<template>
  <div id="app">
    <router-view />
  </div>
</template>
```

以上代码是整个项目的入口文件，它使用了Vue Router进行页面路由管理。

```vue
<template>
  <div>
    <h1>商品列表</h1>
    <div v-for="product in products" :key="product.id">
      <h2>{{ product.name }}</h2>
      <p>{{ product.description }}</p>
      <button @click="addToCart(product)">加入购物车</button>
    </div>
  </div>
</template>
```

以上代码是商品列表组件，它使用了`v-for`指令遍历商品列表，并使用`@click`指令实现加入购物车的功能。

```vue
<template>
  <div>
    <h1>购物车</h1>
    <div v-for="item in cart" :key="item.id">
      <h2>{{ item.name }}</h2>
      <p>{{ item.quantity }} x {{ item.price }}</p>
      <button @click="removeFromCart(item)">移出购物车</button>
    </div>
    <p>总计：{{ total }}</p>
  </div>
</template>
```

以上代码是购物车组件，它使用了`v-for`指令遍历购物车中的商品，并使用`@click`指令实现移出购物车的功能。

### 第11章：项目实战三：在线教育平台

#### 11.1 项目需求分析

项目需求如下：

- 创建一个在线教育平台，提供课程浏览、报名、学习进度跟踪等功能。
- 支持用户注册、登录和权限管理。
- 网站应具有响应式布局，适配不同设备和屏幕尺寸。

#### 11.2 网站设计思路

设计思路如下：

- 采用Angular框架，简化开发过程。
- 使用Angular服务管理应用状态。
- 使用Bootstrap组件库，提高开发效率。

#### 11.3 网站开发步骤

1. **环境搭建**：安装Node.js、npm、Angular CLI等工具。
2. **项目结构**：创建项目文件夹，并初始化项目。
3. **组件开发**：编写Angular组件，实现页面布局和功能。
4. **服务管理**：使用Angular服务管理应用状态，实现数据的共享和更新。
5. **路由配置**：使用Angular Router进行页面路由管理。
6. **接口调试**：使用Postman调试API接口。
7. **部署上线**：将项目打包并上传到服务器，部署到线上环境。

#### 11.4 源代码解读

以下是对项目源代码的详细解读：

```typescript
// app.module.ts
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { CoursesComponent } from './components/courses/courses.component';
import { CourseDetailComponent } from './components/course-detail/course-detail.component';

@NgModule({
  declarations: [
    AppComponent,
    CoursesComponent,
    CourseDetailComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}
```

以上代码是整个项目的入口模块，它导入了项目的所有组件和路由模块。

```typescript
// components/courses/courses.component.ts
import { Component } from '@angular/core';

@Component({
  selector: 'app-courses',
  templateUrl: './courses.component.html',
  styleUrls: ['./courses.component.css']
})
export class CoursesComponent {
  courses = [
    { id: 1, name: 'JavaScript基础', description: '学习JavaScript的基础知识' },
    { id: 2, name: 'React入门', description: '学习React的基础知识' },
    { id: 3, name: 'Vue入门', description: '学习Vue的基础知识' }
  ];
}
```

以上代码是课程列表组件，它定义了一个课程数组，并使用`*ngFor`指令遍历课程列表，并显示每个课程的信息。

```html
<!-- components/course-detail/course-detail.component.html -->
<div *ngIf="course">
  <h1>{{ course.name }}</h1>
  <p>{{ course.description }}</p>
  <button (click)="enrollCourse()">报名</button>
</div>
```

以上代码是课程详情组件，它使用了`*ngIf`指令来条件性地显示课程信息，并使用`(click)`事件绑定来实

