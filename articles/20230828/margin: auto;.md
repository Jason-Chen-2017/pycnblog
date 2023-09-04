
作者：禅与计算机程序设计艺术                    

# 1.简介
  

“margin”是CSS中的一个属性，用于控制盒子元素在父容器中所占据的位置，是相对于父容器的边框线的距离。当设置margin:auto时，元素会自动调整自己的位置使得其居中显示。这个属性非常有用，因为它可以有效地实现不同宽度的父容器之间内容的对齐。
因此，当我们需要将两栏或多栏的内容区分开时（如网页中的左右结构），margin:auto就可以派上用场了。
本文将详细阐述margin:auto属性的工作原理、作用场景、基本语法及相关案例。
# 2.基本概念术语说明
## 2.1 CSS盒模型
CSS盒模型描述的是HTML文档中的所有元素都是一个矩形框。一个矩形框由四个方面组成：边框、内边距、内容、外边距。CSS盒模型允许我们方便地控制这些方面的样式。如下图所示：
- 边框：边框包括元素的边框线、边框角落等，可以通过border属性进行控制。
- 内边距：内边距是指元素内容与边框之间的空白区域，可以通过padding属性进行控制。
- 内容：内容也就是HTML标签之间的文本和图像，也就是元素的主要信息。
- 外边距：外边距是指元素的边框与其他元素的间隙，可以通过margin属性进行控制。

## 2.2 margin:auto的属性值
### 2.2.1 inherit
inherit的值表示该属性继承父元素的margin值。例如，如果父元素没有设置margin值，则子元素也会继承父元素的默认margin值。
```
div {
  margin: 10px; /* 设置父元素的margin值 */
}
p {
  margin: inherit; /* 从父元素继承margin值 */
}
```
### 2.2.2 initial
initial的值表示将margin重置为浏览器的默认值。
```
div {
  margin: initial; /* 将margin重置为默认值 */
}
```
### 2.2.3 unset
unset的值表示还原margin属性到默认值或者继承的初始值。
```
div {
  margin: unset; /* 还原margin属性到默认值或者继承的初始值 */
}
```
## 2.3 margin:auto的作用场景
margin:auto通常应用于块级元素，即display属性值为block或inline-block的元素。一般情况下，块级元素都会独占一行，所以它们可以设置上下margin值，但是不能设置左右margin值。但是，有时候某些时候希望块级元素能够居中显示，此时就需要用到margin:auto。如图所示：

如上图，左右浮动后，每个块级元素将会单独占据一行，导致内容区域宽度不够均匀。此时可以给父容器设置上下margin值，然后将两个子元素设置float:left即可使其居中。
```html
<div class="parent">
  <div class="child1"></div>
  <div class="child2"></div>
</div>

<style>
.parent {
  border: 1px solid black;
  width: 300px;
  height: 200px;
  margin: 0 auto; /* 让父容器水平居中 */
}

.child1,.child2 {
  float: left; /* 将两个子元素向左浮动 */
  width: 100px;
  height: 100px;
  background-color: yellow;
}
</style>
```

如上示例，margin:auto使得父容器内容居中，同时两个子元素左浮动，填满整个父容器。

再如，左右浮动的块级元素之间的间隔太大，想让它们紧贴在一起，此时也可以用到margin:auto，如图所示：

```html
<div class="container">
  <span>This is some text.</span>
  <a href="#">Link</a>
</div>

<style>
.container {
  clear: both; /* 清除浮动 */
  overflow: hidden; /* 防止图片溢出 */
  padding: 10px;
  background-color: #f9f9f9;
}

/* 用margin:auto让左右两边的元素紧贴在一起 */
.container img {
  float: left;
  margin: auto 0;
}

.container span {
  float: left;
  margin: auto 5px;
}

.container a {
  display: block;
  float: right;
  margin: auto;
}
</style>
```

如上示例，首先清除了父元素的浮动，并设置了overflow属性防止图片溢出。然后，分别对两边的元素设置margin:auto，使其紧贴在一起。

总结一下，margin:auto经常用来实现以下几种布局效果：
1. 水平居中：在父容器中加入上下margin值，再将子元素设置为浮动后，就实现了水平居中。
2. 多个块级元素之间间隔固定：在父容器中设置上下margin值，然后在其中某个元素设置clear:both属性，并将其它元素设置为浮动，就实现了多个块级元素之间的间隔固定。