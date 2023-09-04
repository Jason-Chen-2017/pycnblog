
作者：禅与计算机程序设计艺术                    

# 1.简介
  

首先给大家介绍一下我自己吧。我叫陈桢，今年27岁，目前在一家互联网公司担任技术经理。工作中主要负责网站前端开发，同时也参与过后端开发、产品设计和项目管理等工作。
作为一个技术专家来说，对于计算机科学领域来说是必不可少的角色。通过本文的学习，可以加深对CSS（Cascading Style Sheets）的理解和应用，并能快速提升自己的编程能力。
在写作过程中，我会将知识点、示例代码和图片放在一起，有助于读者更全面地理解相关知识。
# 2.前言
CSS一直是Web页面美观、交互性强、可用性高的重要因素之一，也是最具创造力的语言之一。但是CSS却不像其他编程语言一样简单易懂。因此，掌握好CSS的精髓至关重要。
CSS的语法简单，容易上手，但是其灵活的样式设定又需要一定技巧才能正确实现。本文就将带领大家了解CSS的各种属性及其作用，结合实际案例丰富大家的视野。
# 3.基本概念及术语
## CSS语法
CSS (Cascading Style Sheets) 是一种用来表现HTML（超文本标记语言）或XML（可扩展标记语言）文档样式的计算机语言。它是一个声明性的语言，意味着用户通过指定选择器、属性及属性值来添加样式信息。下面是CSS的基本语法规则:

- 每一条CSS规则由选择器，冒号(:)，样式，分号(;)组成；
- 属性采用小驼峰命名法，例如 font-family、margin-top 和 color；
- 颜色值采用十六进制或RGB表示法，例如 #FFFFFF 或 rgb(255,255,255);
- 当多个CSS属性对应同一个元素时，使用空格隔开；
- 使用斜线(/)来注释掉一行或多行代码。

## 块级元素和内联元素
CSS的布局机制决定了块级元素和内联元素的排版方式。

- 段落元素 p 默认为内联元素，它们只能容纳文本、图像或其他内联元素。段落之间默认有一个换行符，也可以使用 margin 把两个相邻的段落进行上下边距调整。
- 大部分常用 HTML 元素都是块级元素，它们可以自动地填充整个宽度并自顶向下摆放。常用的块级元素包括 div、ul、li、h1~h6、p、form、header、section、article、nav、aside、footer、video、audio。

注意：由于不同浏览器的默认样式差异，相同HTML结构在不同浏览器下可能呈现不同的显示效果。为了让HTML文档在所有浏览器下都显示一致的外观，建议通过 CSS 为所有元素设置统一的样式。

## 盒模型
CSS 中定义了两种尺寸计算方式——边框盒模型和内容盒模型。

- 在边框盒模型中，width 和 height 分别指的是 content + padding + border。而 margin 不在其中，因为它只是围绕 content 区域的。
- 在内容盒模型中，width 和 height 分别只包含内容区的内容，padding 不包含在其中。所以 width 和 height 的取值要比实际宽高大一些。

## 层叠样式表
层叠样式表（Cascading StyleSheets，简称 CSS），是一种用来表现 HTML 和 XML 文档样式的计算机语言。它允许web设计人员创建多种类型的网页主题，如文字样式、色彩调配、导航设计、表单布局等，并且可以把这些样式集中应用到整个网站上。层叠样式表的最大优点就是能够轻松实现复杂的网页设计，而且可以根据需要动态更改样式。

当多个层叠样式表应用于同一个HTML文档时，它们所设置的样式将按照以下优先顺序进行应用：

1. 浏览器缺省样式
2. 用户自定义样式表（通常情况下，就是浏览器设置中的“用户样式”选项卡下的样式文件）
3. 作者在样式表中指定的样式
4. 内嵌样式（在 HTML 元素中直接使用 style 属性设置的样式）

通过这种方式，作者可以把特定网页的特殊样式集中存放在样式表中，然后再按需应用到该网页上的各个元素上。这样既可以节省大量时间，提高工作效率，还可以有效地控制网站的外观和感觉。

# 4.CSS属性与作用
## 背景与边框
### background-color

background-color 设置元素的背景颜色。

```css
div {
  background-color: red; /* 设置背景颜色为红色 */
}
```

### background-image

background-image 设置元素的背景图像。

```css
div {
}
```

### background-repeat

background-repeat 设置元素的背景图像是否重复。

```css
div {
  background-repeat: no-repeat; /* 图像不会重复 */
}

/* 如果需要图像在水平或垂直方向上重复 */
div {
  background-repeat: repeat-x; /* 图像在水平方向上重复 */
}

div {
  background-repeat: repeat-y; /* 图像在垂直方向上重复 */
}
```

### background-position

background-position 设置元素的背景图像的起始位置。

```css
div {
  background-repeat: no-repeat; /* 图像不会重复 */
  background-position: center top; /* 将图像置于背景中间，靠上 */
}

/* 可以使用百分比值指定距离 */
div {
  background-repeat: no-repeat; /* 图像不会重复 */
  background-position: left -20px; /* 将图像左侧20px处，靠上 */
}
```

### background-size

background-size 设置元素的背景图像的尺寸。

```css
div {
  background-repeat: no-repeat; /* 图像不会重复 */
  background-position: center bottom; /* 将图像置于背景中心，靠下 */
  background-size: cover; /* 图像完全覆盖背景 */
}

/* 可以指定背景图像的大小 */
div {
  background-repeat: no-repeat; /* 图像不会重复 */
  background-position: center bottom; /* 将图像置于背景中心，靠下 */
  background-size: auto; /* 自动调整背景图像大小 */
}

div {
  background-repeat: no-repeat; /* 图像不会重复 */
  background-position: right bottom; /* 将图像右下角，靠下 */
  background-size: contain; /* 保持背景图像比例缩放 */
}

/* 可以组合使用百分比值和关键字值 */
div {
  background-repeat: no-repeat; /* 图像不会重复 */
  background-position: center bottom; /* 将图像置于背景中心，靠下 */
  background-size: 50% auto; /* 设置背景图像高度占满父容器50%，宽度自适应 */
}
```

### border

border 用于设置元素的边框样式。

```css
div {
  border: solid 1px black; /* 设置边框为实线粗细为1px，颜色为黑色 */
}

/* 可以使用不同样式的边框 */
div {
  border-top: dotted 2px green; /* 上边框设置为虚线粗细为2px，颜色为绿色 */
}

div {
  border-right: dashed 2px blue; /* 右边框设置为点状粗细为2px，颜色为蓝色 */
}

div {
  border-bottom: double 3px yellow; /* 下边框设置为双线粗细为3px，颜色为黄色 */
}

div {
  border-left: solid 1px gray; /* 左边框设置为实线粗细为1px，颜色为灰色 */
}

/* 可以设置四个方向的边框 */
div {
  border-width: 1px 2px 3px 4px; /* 设置四个边框的宽度 */
}

div {
  border-style: solid dashed double dotted; /* 设置四个边框的样式 */
}

div {
  border-color: red green blue orange; /* 设置四个边框的颜色 */
}

/* 可以设置单独的边框 */
div {
  border-top: none; /* 删除上边框 */
}

div {
  border-right: medium none transparent black; /* 修改右边框为中等粗细，颜色透明，底部黑色 */
}

div {
  border-bottom: groove yellow; /* 添加下边框为凹槽边框，颜色为黄色 */
}

div {
  border-left: ridge white; /* 添加左边框为棱锥边框，颜色为白色 */
}
```

### border-radius

border-radius 设置元素的圆角边框。

```css
div {
  border-radius: 10px; /* 设置元素的所有边框圆角半径为10px */
}

div {
  border-top-left-radius: 5px; /* 设置元素的左上角圆角半径为5px */
}

div {
  border-top-right-radius: 1em; /* 设置元素的右上角圆角半径为1em */
}

div {
  border-bottom-right-radius: 3vw; /* 设置元素的右下角圆角半径为3vw */
}

div {
  border-bottom-left-radius: 10%; /* 设置元素的左下角圆角半径为10% */
}
```

## 清除浮动
float 是 CSS 中的一个用来控制元素在页面中的定位方式的属性。如果元素设置了 float ，则该元素会脱离普通文档流，并向左或向右浮动，类似于把该元素的内容贴到了当前行或新行。

```html
<div class="box">
    <p>Some text</p>
</div>
```

```css
.box img {
  float: left; /* 将图片浮动到左边 */
}
.clear {
  clear: both; /* 清除浮动 */
}
```

但是，由于 CSS 是基于盒子模型的，所以虽然图片已经被成功浮动到了左边，但它的周围仍然存在一定的距离。为了消除这种距离，可以添加如下 CSS 代码：

```css
.box::before {
  display: block;
  content: "";
  height: 0;
  visibility: hidden;
}
```

```html
<div class="box"></div>
```

```css
.box {
  overflow: hidden; /* 隐藏溢出的元素 */
}

.box > * {
  float: left; /* 将子元素浮动到左边 */
}

.box::after {
  display: table;
  clear: both; /* 清除浮动 */
}
```

其中 `.box` 是父元素，子元素 `<img>` 和 `<p>` 会随着宽度自动撑满父元素的宽度，而由于 `overflow:hidden;` 对这个父元素无效，所以只能通过清除浮动的方式来解决。

另外，为了避免出现类似于图书封面的环绕边框，可以通过 `:before` 和 `:after` 伪类和 `content` 来规避。