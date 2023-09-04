
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 概述
Margin是一个页面上的边距属性。

它是指元素周围与其他元素之间的距离。

Margin可以用来控制元素的上下、左右间距以及与容器元素之间的距离。

Margin可以设置负值，这样就可以把元素在容器内垂直居中。

CSS中的Margin包括四个方向：top、bottom、left、right。

如下图所示：


## 1.2 为什么要学习Margin？

1. 更好的布局
2. 更多的空间
3. 美化设计
4. 合理的布局

## 1.3 Margin的使用场景

1. 块级元素之间或内联元素之间的距离。
2. 定位子元素与父元素之间的距离。
3. 设置外边距给元素添加空白。
4. 将多个盒子进行对齐。

# 2.基本概念
## 2.1 Margin的含义

Margin是指一个元素与另一个元素或者元素容器边界之间的距离。当两个元素处于同一行时，它们之间的margin就是相邻元素的距离；如果两个元素在两行显示，则它们之间的margin就是两个行之间的距离。margin的值可以通过属性设置，例如: `margin: top right bottom left;`.

Margin是CSS中所有定位属性中最重要的一个属性之一，也是其使用频率最高的属性。

## 2.2 Box Model（盒模型）

HTML文档由多个“框”组成，每一个框都有自己的位置和大小。这些框分为三类：

1. Content Area (内容区域)，包含文本和图像等内容。
2. Padding (内边距)，在内容区域周围创建出一定的空间。
3. Border （边框），用来修饰元素的边缘。


Box Model描述了浏览器如何处理HTML文档中各种元素。

Content Area为HTML文档中实际呈现的内容，Padding是为了使内容区域与边框之间留有一定距离，Border则用于修饰元素的边缘。 

Box Model由margin, border, padding, content四部分构成，其中margin可以分别取左上右下四个方向的偏移值，border则设定边框样式及宽度，padding用于调整内容和边框的间距，content为盒子的内容显示区。

如下图所示：


## 2.3 AutoMargins（自动边距）

当Margin属性设置为auto时，浏览器会根据父元素的宽度和高度计算出适当的边距值。常用属性如：`margin: auto;` `margin-left: auto;` `margin-right: auto;`, `margin: 0 auto;`等。

## 2.4 Collapsing margins（边距折叠）

当两个相邻元素具有相同的Margin时，该边距会自动折叠，即较大的边距值将覆盖较小的边距值。

一般情况下，相邻的block元素默认具有相同的上下margin，但是不同的上下margin不会发生边距折叠。若要实现不同上下margin之间的边距折叠，可以在这些元素之间增加一些其它元素或设置display: inline;样式。

```html
<div style="background-color:#eee;">
  <p>This is the first paragraph.</p>
  <!-- Here we have some extra text to add spacing -->
  <br> 
  <span>This is another small bit of text</span>
</div>
```

```css
/* Without any additional elements or styles */
body {
  font-size: 16px;
}

p {
  background-color: #f00;
  color: white;
  width: 400px;
  height: 100px;
  /* These two properties cause adjacent block elements
     with identical margins to collapse into a single line */
  margin: 20px 0; 
}

/* Adding an element between the paragraphs creates
   some space and causes them not to collapse together */
hr {
  display: none;
}
```