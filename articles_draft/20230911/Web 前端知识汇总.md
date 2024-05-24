
作者：禅与计算机程序设计艺术                    

# 1.简介
  

　　Web 前端，即网页前端开发，是指网站页面的前台设计、美化与功能实现。网站前端开发涉及到的技术有 HTML、CSS、JavaScript、jQuery、Bootstrap、AngularJS、React、Vue.js 等等，掌握这些技术可以让您快速的搭建属于自己的网站，提升职场竞争力。

　　Web 前端开发是一个动态多变的职业领域，因为它要求具备丰富的编程技能，能够解决复杂的问题，同时还要面对诸如兼容性、性能优化、安全性、可维护性、用户体验等方面的挑战。因此，掌握 Web 前端开发的关键在于不断学习和实践。本文通过整理前端开发所需的基础技术，帮助读者加快掌握该领域必备技能。

　　除了掌握前端开发的技术外，文章还会尽量多地探讨一些行业内的应用场景、适用范围，并尝试给出一些实际案例。希望能够帮助读者有效的进行自我成长和职业规划。
# 2.知识图谱
## HTML 篇
HTML(HyperText Markup Language) 是用于创建网页的标记语言，是一种简单的标记语言。包括了各种标签，比如<head>、<title>、<body>等，通过这些标签定义网页的内容结构。HTML 使用UTF-8编码。
### 2.1 HTML基本语法
HTML 代码的基本语法如下：
```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>网页标题</title>
  </head>
  <body>
    <!-- 网页内容 -->
  </body>
</html>
```
其中，`<!DOCTYPE>` 声明文档类型；`<html>` 和 `</html>` 构成了 HTML 文档的根元素；`<head>` 和 `</head>` 包含网页的元数据；`<meta>` 为搜索引擎设置关键字、描述和其他属性；`<title>` 设置网页的标题；`<body>` 和 `</body>` 定义网页的内容。
### 2.2 HTML全局属性
HTML 有很多全局属性，以下是最常用的属性：
- class - 指定一个或多个类名（classname），供 CSS 或 JavaScript 来使用。class 属性允许多个同类元素共存。
- id - 为元素指定唯一的 ID。id 属性可用于在 CSS 或 JavaScript 中对特定的元素进行样式设置。
- style - 为元素指定行内样式。style 属性可用来直接定义元素的样式，而不需借助外部样式表。
- title - 提供额外信息，当鼠标移动到元素上时显示提示。
- data-* - 可用于存储页面私有数据。

## CSS 篇
CSS(Cascading Style Sheets)，层叠样式表，是一种用来表现 HTML 或者 XML 文档样式的计算机语言。CSS 描述了如何展示 HTML 的元素，例如字体、颜色、大小、边框、位置、布局等。CSS 使用UTF-8编码。
### 2.3 CSS选择器
CSS 选择器是 CSS 中非常重要的概念，决定了 CSS 对页面元素的渲染方式。CSS 选择器分为简单选择器、类选择器、ID选择器、群组选择器、子代选择器、后代选择器、并集选择器、伪类选择器、属性选择器等，这些选择器能够精确地选中页面上的元素，并应用对应的样式。
#### 2.3.1 简单选择器
- tag selector - 通过标签名称匹配元素。如 h1 { color: red; } 将使所有 <h1> 元素的文本颜色变为红色。
- class selector - 通过类名称匹配元素。如.red { color: red; } 会把所有 class="red" 的元素的文本颜色变为红色。
- id selector - 通过元素 id 匹配元素。如 #content { font-size: 20px; } 则会将 id="content" 的元素的字体大小设置为20像素。
- attribute selector - 通过元素属性匹配元素。如 a[href] { text-decoration: none; } 可以删除超链接下划线。
#### 2.3.2 复合选择器
- descendant combinator - 通过子代关系匹配元素。如 ul li a { color: blue; } 可以将 <a> 元素直接作为 <ul> 元素的孩子并应用样式。
- child combinator - 通过直接子代关系匹配元素。如 div > p { background-color: yellow; } 可以将直接子代 <p> 元素的背景色设置为黄色。
- adjacent sibling combinator - 通过相邻兄弟关系匹配元素。如 h1 + p { font-weight: bold; } 可以使紧接着 <h1> 元素的第一个 <p> 元素加粗。
- general sibling combinator - 通过一般兄弟关系匹配元素。如 h1 ~ p { border: 1px solid black; } 可以将所有的 <p> 元素与紧随其后的第一个 <h1> 元素之间画一条黑色实线。
- wildcard selector - 可以匹配所有元素，但效率较低，不要滥用。如 * { margin: 0; padding: 0; } 会将所有元素的内边距和外边距清除。
### 2.4 CSS单位及值
CSS 中有多种单位，如 em、px、rem、%、vw、vh、vmin、vmax等，它们之间的区别和联系都需要掌握。CSS 接受的属性值一般为长度、颜色、URL地址、数字、百分比等。
### 2.5 CSS盒模型
CSS 中的盒模型主要由四个部分组成：边框（border）、填充（padding）、边界（margin）、内容（content）。CSS3新增了圆角、盒阴影、透明度、Transforms、Animations等特性。
### 2.6 CSS居中布局
CSS 常用的居中布局有两种，水平居中和垂直居中。
#### 水平居中布局
通过设置左右外边距 auto 来使块级元素水平居中。如下示例代码：
```css
div {
  width: 200px;
  height: 200px;
  background-color: gray;
  position: relative; /* 开启绝对定位 */
  left: 50%; /* 居中 */
  transform: translateX(-50%); /* translate函数替代left负值，浏览器更加支持 */
}
```
#### 垂直居中布局
可以通过设置上下外边距 auto 和上下边框为相同宽度来使块级元素垂直居中。如下示例代码：
```css
div {
  width: 200px;
  height: 200px;
  background-color: gray;
  position: absolute; /* 开启绝对定位 */
  top: 50%; /* 上下外边距50% */
  bottom: 50%;
  margin: 0 auto; /* 自动设置上下边框相等，保证高度一致 */
}
```
## JavaScript篇
JavaScript 是一门基于原型的动态脚本语言，它的作用主要是改变 HTML 和 CSS 的行为，提供用户交互以及动画效果。JavaScript 使用UTF-8编码。
### 2.7 数据类型
JavaScript 有五种基本的数据类型：number、string、boolean、null、undefined，还有数组 Array 和对象 Object。
#### 2.7.1 number 数据类型
number 数据类型用于表示整数和浮点数。由于浮点数计算的精度限制，不推荐使用。
#### 2.7.2 string 数据类型
string 数据类型用于表示字符串，是不可更改的数据类型。
#### 2.7.3 boolean 数据类型
boolean 数据类型只有两个取值：true 和 false，用于表示逻辑值。
#### 2.7.4 null 数据类型
null 数据类型代表空值，也就是不存在的值。JavaScript 中只能使用 null 表示变量的值为空。
#### 2.7.5 undefined 数据类型
undefined 数据类型代表变量没有被赋值，这种类型只有一个值——undefined。
### 2.8 操作符
JavaScript 拥有丰富的操作符，包括算术运算符、关系运算符、逻辑运算符、条件运算符、赋值运算符等。
### 2.9 函数
函数是 JavaScript 中用于执行特定任务的代码块，可以被重复调用。函数可以接受参数和返回值。
### 2.10 对象
对象是 JavaScript 中用于组织数据的一种数据结构，每个对象都有自己的属性和方法。
### 2.11 事件
事件是用户操作网页时的行为，JavaScript 可以监听这些事件，并作出相应的响应。
### 2.12 异步处理
JavaScript 具有异步执行能力，可以使用回调函数或 Promise 处理异步任务。