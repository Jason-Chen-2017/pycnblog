
作者：禅与计算机程序设计艺术                    

# 1.简介
  

CSS (Cascading Style Sheets) 是一种用来表现 HTML 和 XML 文档样式的计算机语言。CSS 提供了许多特性用于控制网页的布局、颜色、字体、边框等外观显示效果。CSS3 则是对 CSS 的更新，提供了更多新的特性，如阴影、转换、动画、多列布局、媒体查询等。本文将对 CSS 进行全面的介绍，并用实例介绍如何通过 CSS 来实现各种各样的设计效果。
本书分为如下几个章节：

1.CSS基础知识：涉及CSS的一些基础知识，如选择器、盒模型、定位、浮动、文本属性、边框、背景、CSS语法。
2.CSS高级技巧：涉及CSS的一些进阶技巧，如布局技巧、混合模式、滤镜、SVG等。
3.实战案例分享：分享作者在实际工作中遇到的一些CSS应用场景，以及相应的解决方案，包括UI组件库、电商网站、新闻站点等。
4.面试题集锦：帮助作者加强面试技巧，提升应聘者的CSS基础素养。
5.扩展阅读：提供CSS参考书籍和学习资源，帮助读者更好地理解CSS的相关知识。
# 2. CSS基础知识
## 1. 什么是CSS？
CSS（层叠样式表）是一个用来描述HTML或XML文档样式的计算机语言。它是一种基于XML的标记语言，因此可以定义可重复使用的样式集合。CSS独立于 HTML 或 XML 文件，文件保存在不同的地方，可通过外部引用或嵌入到HTML中。它可以控制网页的布局、 Typography、 colors、visual styles、交互效果等。
## 2. CSS基本语法
### 1. CSS的规则和声明
在 CSS 中，每一条规则都由两个部分组成，一个是选择器，另一个是声明块。选择器是应用规则的目标元素，声明块是设置这些元素的具体样式。
例如，下面的 CSS 规则：
```css
h1 {
  color: blue;
  font-size: 2em;
}
```
这个规则中，选择器 h1 表示要应用该规则的所有 <h1> 元素；而declarations （即 color:blue 和 font-size:2em）设置了这些元素的颜色和字号。每个声明后面跟着一个分号。
### 2. CSS注释
CSS 中的注释与 JavaScript 类似，可以对 CSS 代码做出解释或者禁用某些样式。
单行注释：
```css
/* This is a single line comment */
```
多行注释：
```css
/* This is
   a multi-line 
   comment */
```
### 3. CSS属性
CSS 有很多属性可以设置元素的样式，下面我们介绍一些常用的属性。
#### 字体系列
* `font-family` - 设置字体系列
* `font-style` - 设置是否斜体
* `font-weight` - 设置粗细
* `font-variant` - 设置是否小型大写字母
* `font-size` - 设置字体大小
* `line-height` - 设置行距
* `text-transform` - 设置文本转换格式
```css
p {
  font-family: Arial, sans-serif; /* 指定多个字体，优先使用 Arial，如果浏览器找不到 Arial，就换其他字体 */
  font-style: italic; /* 斜体字 */
  font-weight: bold; /* 粗体字 */
  font-variant: small-caps; /* 小型大写字母 */
  font-size: 16px; /* 字体大小 */
  line-height: 1.5; /* 行距 */
  text-transform: uppercase; /* 将所有字符转换为大写字母 */
}
```
#### 文本系列
* `color` - 设置字体颜色
* `letter-spacing` - 设置字符间距
* `word-spacing` - 设置单词间距
* `text-align` - 设置水平对齐方式
* `vertical-align` - 设置垂直对齐方式
* `white-space` - 设置空白处理方式
* `text-decoration` - 添加或删除文本装饰
```css
a {
  color: navy; /* 蓝色链接 */
  letter-spacing: 1px; /* 字符间距 */
  word-spacing: 5px; /* 单词间距 */
  text-align: center; /* 水平居中 */
  vertical-align: top; /* 垂直顶部对齐 */
  white-space: nowrap; /* 不折行 */
  text-decoration: underline; /* 下划线 */
}
```
#### 背景系列
* `background-color` - 设置背景颜色
* `background-image` - 设置背景图片
* `background-repeat` - 设置背景图像是否重复
* `background-position` - 设置背景图像位置
* `background-attachment` - 设置背景图像是否固定或随元素滚动
```css
body {
  background-color: #f9f9f9; /* 灰色背景 */
  background-repeat: no-repeat; /* 不重复 */
  background-position: right bottom; /* 在右下角 */
  background-attachment: fixed; /* 固定 */
}
```
#### 渐变系列
* `background-image` - 使用 linear-gradient 函数创建线性渐变背景
* `background-image` - 使用 radial-gradient 函数创建放射性渐变背景
* `background-size` - 设置背景尺寸
* `background-repeat` - 设置背景是否重复
```css
div {
  height: 200px;
  width: 300px;
  border: 1px solid black;
  background-image: linear-gradient(to bottom, red, green); /* 创建从红色到绿色的线性渐变 */
  background-size: cover; /* 填充 */
  background-repeat: repeat-x; /* 横向平铺 */
}
```
#### 边框系列
* `border` - 设置边框宽度、样式和颜色
* `border-top/bottom/left/right` - 分别设置上下左右边框
* `border-radius` - 设置圆角半径
```css
img {
  margin: 10px;
  padding: 5px;
  border: 1px solid gray; /* 边框 */
  border-radius: 5px; /* 圆角 */
}
```
#### 盒模型系列
CSS 为盒模型提供了很多属性，如 `margin`，`padding`，`width`，`height`，`display`，`float`，`clear`。
```css
div {
  margin: 10px; /* 外边距 */
  padding: 5px; /* 内边距 */
  width: 300px; /* 宽度 */
  height: 200px; /* 高度 */
  display: inline-block; /* 显示类型 */
  float: left; /* 浮动方向 */
  clear: both; /* 清除 */
}
```
## 3. CSS选择器
### 1. 类选择器
```css
.box {
  /* 样式 */
}
```
### 2. ID选择器
```css
#myDiv {
  /* 样式 */
}
```
### 3. 标签选择器
```css
div {
  /* 样式 */
}
```
### 4. 属性选择器
```css
[type="checkbox"] {
  /* 样式 */
}
```
### 5. 子选择器
```css
ul > li {
  /* 样式 */
}
```
### 6. 后代选择器
```css
li span {
  /* 样式 */
}
```
### 7. 并集选择器
```css
input[type="submit"], input[type="reset"] {
  /* 样式 */
}
```
### 8. 伪类选择器
```css
a:link {
  /* 样式 */
}
a:visited {
  /* 样式 */
}
a:hover {
  /* 样式 */
}
a:active {
  /* 样式 */
}
```
## 4. 布局技巧
### 1. 盒模型
CSS 的盒模型是指当一个元素被渲染时，它的实际宽度、高度、边距、内边距、外边距、边框和圆角等几何属性都是通过计算获得的。
#### 宽度
当没有给元素设定宽度时，其宽度默认等于父元素的宽度减去左右内边距之和，也就是内容区宽度。如果父元素没有宽度（比如 body），那么宽度会自动根据内容进行调整。可以使用 `width` 属性设置宽度。
#### 高度
当没有给元素设定高度时，其高度默认等于父元素的高度减去上下内边距之和，也就是内容区高度。如果父元素没有高度，那么高度会自动根据内容进行调整。可以使用 `height` 属性设置高度。
#### 内边距
`padding` 属性用于设置元素的内边距。它接受四个值，分别对应上、右、下、左内边距。负值也有效。可以同时设置多个内边距，比如 `padding: 10px 20px;` ，表示上、右、下、左边距均为 10px，左右内边距均为 20px。
#### 外边距
`margin` 属性用于设置元素的外边距。它接受四个值，分别对应上、右、下、左外边距。负值也有效。可以同时设置多个外边距，比如 `margin: 10px 20px;` ，表示上、右、下、左边距均为 10px，左右外边距均为 20px。
#### 边框
`border` 属性用于设置元素的边框。它接受一个或多个值，包括边框宽度、样式和颜色。可以同时设置多个边框，比如 `border: 1px solid black;` ，表示边框宽度为 1px，样式为实线，颜色为黑色。
#### 圆角
`border-radius` 属性用于设置元素的圆角。它接受一个值，指定四个角的半径。可以使用百分比，也可以使用像素值。可以使用逗号分隔不同的值，例如 `border-radius: 5px, 10px, 20px, 10%;` 。第一个值为四个角相同的半径，第二个值为左上角、右上角、右下角、左下角四个角的半径。
```css
div {
  width: 200px;
  height: 200px;
  border: 1px solid black;
  border-radius: 10px; /* 圆角半径 */
}
```
#### 盒模型总结
##### 传统布局模型
传统布局模型主要是基于盒子模型及盒子之间的位置关系进行布局的，主要包括块级元素、行内元素、浮动和绝对定位等。块级元素占据整个容器的宽度，高度默认是自动增加的，且无法设置宽度和高度；行内元素只能在一行内显示，高度和宽度默认均是自动调整的，且不能设置外边距和内边距；浮动元素脱离普通流，并向左或向右偏移，高度和宽度默认是自动调整的，且无法设置宽度和高度；绝对定位元素相对于其最近的已定位（非静态）祖先元素进行定位，并且不会再影响普通流，高度和宽度默认是自动调整的，且无法设置宽度和高度。如下图所示：
##### Flex 布局模型
Flex 布局模型主要用于快速响应式设计。它采用一维化的方式来构建页面布局，使得页面能够自动适配不同尺寸的设备屏幕，通过定义 flex container、flex item、flex properties 可以方便地设置、修改布局。以下为 CSS 对 Flex 布局的支持情况：
- 支持的属性：display、flex-direction、justify-content、align-items、align-content、order、flex-wrap、align-self、flex、shrink、basis
- 不支持的属性：float、clear、vertical-align

如下图所示：