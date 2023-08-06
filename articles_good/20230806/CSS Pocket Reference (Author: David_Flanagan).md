
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 样式表（Style Sheets）语言是用于描述HTML或XML文档呈现方式的一门语言。CSS允许用户通过网页浏览器设置文本的颜色、字体、大小、样式、边框、透明度等。CSS本身独立于HTML和XML之外，因此可以有效地保障网站的可用性及易读性。本文提供一个CSS快速参考，帮助初级CSS开发人员了解其常用属性及语法规则，并可以在实际工作中使用，提高工作效率。

          本文适合以下人员阅读：
           - 对CSS不是很熟悉，但是想学习一下；
           - 有一定前端基础但没有掌握所有的CSS知识；
           - 想提升自己的CSS水平；
           - 需要一份详实的CSS速查手册。

         # 2.CSS基础
         ## 2.1 介绍
         ### 2.1.1 CSS概述
         CSS（Cascading Style Sheets），即层叠样式表（英语：Cascading StyleSheets）。是一种用来表现HTML（超文本标记语言）或者XML文档样式的计算机语言。它允许网页制作者定义诸如字体、尺寸、颜色、间距、对齐方式、图案、渐变、阴影、过渡等风格特征，从而美化网页内容、增强互联网应用效果。

         20世纪90年代末，W3C为了统一各浏览器渲染页面的效果，推出了CSS标准。CSS成为web世界中最重要的技术。CSS1.0时期，仅规定了“盒模型”相关的样式，而CSS2.0引入了更多特性，如定位，文字处理，多列布局，CSS基础选择器，边框模型，轮廓模型，伪类与伪元素，多项视觉效果和动画。

         2011年发布的CSS3，则在CSS2.1的基础上进一步扩充，新增了一些特性，如圆角，倒影，文本阴影，线性渐变，背景图片缩放，转换，动画，字体功能等。

         2017年发布的CSS4，则在CSS3的基础上进一步扩充，新增了一些特性，如自定义变量，环境变量，媒体查询，滤镜，变形，盒子分隔符，多行文本，光标属性，滚动行为，过渡阶段，自定义选择器。

         2021年发布的CSS Level 5，则在CSS4的基础上继续扩充，新增了一些特性，比如宽度高度范围限制，字符集，换行规则，引用文本，强调元素，双向文字方向，对齐内容，选择器组合，插槽，布局模式，高级计数器，分割线，省略号，剪贴纸样式，动画属性，元素内边距，换行容器，文本指示灯，场景切换，用户界面，表单支持，可访问性，连字号修正，竖排字体，正反链条属性，混合动画，标记章节，词干调整，字体渲染控制，面向打印机的样式等。

         ### 2.1.2 CSS书写位置
         在HTML中，样式表通常放在head标签内部，并添加type="text/css"属性值：

         ```html
             <head>
                ...
                 <style type="text/css">
                     /* css code goes here */
                 </style>
             </head>
         ```

         在外部样式表中，也可以使用link标签链接到样式文件：

         ```html
            <!DOCTYPE html>
            <html>
              <head>
                <title>My Page</title>
                <link rel="stylesheet" href="mystyle.css">
              </head>
              <body>
                <!-- content of the page -->
              </body>
            </html>
        ```

        当然，也可以直接在head标签里嵌入CSS代码。当多个样式之间存在冲突时，后面的样式会覆盖前面的样式。

     ## 2.2 CSS语法
     ### 2.2.1 CSS选择器
      CSS样式设置方法，要根据选择器进行相应设置。CSS共有三种选择器：
       - 标签选择器：例如 `h1`、`p`、`a` 等，选取相应标签的内容。
       - class选择器：例如 `.class-name`，选取带有指定类的所有元素。
       - id选择器：例如 `#id-name`，选取具有特定id值的元素。

      另外还有一些其他的选择器：
       - 属性选择器：例如 `[attr=value]` 或 `[attr~=value]`，选取具有指定属性及属性值的元素。
       - 伪类选择器：例如 `:hover` 或 `:active`，选择鼠标悬停或激活状态下的元素。
       - 伪元素选择器：例如 `::before` 和 `::after`，创建一些特殊的几何形状或内容，并将其插入DOM树中的某个位置。
      
      每个选择器都可以有多个属性和样式来进行设置。如：

      ```css
      h1 { font-size: 18px; color: blue;} 
     .intro { margin-bottom: 20px; }
      #myPic { width: 100%; height: auto; }
      a[href^='http'] { text-decoration: none; }
      input[type='submit']:disabled { background-color: gray; }
      ul li:first-child { font-weight: bold; }
      ::selection { background-color: yellow; }
      @media screen and (max-width: 600px){
        body{font-size: 14px;} 
        p { line-height: 1.5em; }
      }
      ```

       - 第一个选择器 `h1` 设置 `font-size` 为 `18px`，`color` 为蓝色。
       - 第二个选择器 `.intro` 设置 `margin-bottom` 为 `20px`。
       - 第三个选择器 `#myPic` 设置 `width` 为 `100%`，`height` 为自动(`auto`)。
       - 第四个选择器 `a[href^='http']` 设置所有以 `http` 开头的 `<a>` 标签的 `text-decoration` 为不显示(`none`)。
       - 第五个选择器 `input[type='submit']:disabled` 设置禁用的 `<input>` 元素类型为提交按钮 (`<input type="submit">`) 的 `background-color` 为灰色。
       - 第六个选择器 `ul li:first-child` 设置 `li` 元素作为子元素的第一个元素 (`ul > li:nth-of-type(n+1)`) 的 `font-weight` 为粗体。
       - 第七个选择器 `::selection` 创建一个伪元素，其样式将被当做用户选择某些内容时的样式。

    ### 2.2.2 CSS属性
     #### 字体属性
      - `font-family`: 设置字体系列，比如 `Arial`, `Helvetica Neue`, `sans-serif`; 默认值为 `Times New Roman`, `Times`, `serif`.
      - `font-size`: 设置字体大小，比如 `16px`, `1.2em`; 默认值为 `16px`.
      - `line-height`: 设置行高，常用单位包括像素(`px`)、分数(`em`)、百分比(`%`)和行距(`line-height`). 默认值为 `normal`(即当前字体大小的倍数).
      - `font-style`: 设置字体风格，可以是 `italic` 或 `oblique`, 默认值为空。
      - `font-weight`: 设置字体粗细，可以是 `bold` 或 `bolder`、 `lighter`, `100` 至 `900` 之间的数字，默认为 `normal`(无倾斜效果).
      - `letter-spacing`: 设置字符间距，默认值为 `normal` (即字间距为0).
      - `word-spacing`: 设置单词间距，默认值为 `normal` (即单词间距为0).
      - `text-align`: 设置文本水平对齐，可以是 `left`, `right`, `center`, `justify`. 默认值为空。
      - `text-transform`: 设置文本转换，可以是 `uppercase`, `lowercase`, `capitalize`, `none`. 默认值为空。
      - `vertical-align`: 设置垂直对齐，可以是 `top`, `middle`, `bottom`, `sub`, `super`, `%`, em`, etc., 默认值为空。

      示例：

      ```css
      /* 设置字体系列 */
      body { font-family: Arial, sans-serif; } 

      /* 设置字体大小 */
      h1 { font-size: 32px; } 

      /* 设置行高 */
      p { line-height: 1.5em; } 

      /* 设置字体风格 */
      span { font-style: italic; } 

      /* 设置字体粗细 */
      strong { font-weight: bolder; } 

      /* 设置字符间距 */
      p { letter-spacing: 2px; } 

      /* 设置单词间距 */
      p { word-spacing: 2px; } 

      /* 设置文本水平对齐 */
      div { text-align: center; } 

      /* 设置文本转换 */
      p { text-transform: uppercase; } 

      /* 设置垂直对齐 */
      td { vertical-align: top; } 
      ```

    #### 颜色属性
     - `color`: 设置文本颜色，值可以是 `red`, `#ff0000`, `rgb()`, `rgba()` 等，默认为黑色。
     - `background-color`: 设置背景颜色，值同 `color`. 默认值为空白色。
     - `opacity`: 设置透明度，值可以是 `0` 至 `1`, 0 表示完全透明，1 表示完全不透明，默认为 `1`.
     
     示例：
     
     ```css
     /* 设置文本颜色 */
     a { color: blue; } 
     nav { color: rgba(255, 0, 0, 0.5); } 
 
     /* 设置背景颜色 */
     body { background-color: white; } 

     /* 设置透明度 */
     img { opacity: 0.5; } 
     ```
     
    #### 边框属性
     - `border`: 设置所有边框，包括边框颜色、样式和宽度。
     - `border-color`: 设置边框颜色。
     - `border-style`: 设置边框样式，值可以是 `solid`, `dashed`, `dotted`, `double`, `groove`, `ridge`, `inset`, `outset`, `none`, `hidden`. 默认值为空。
     - `border-width`: 设置边框宽度，值可以是 `thin`, `medium`, `thick`, 长度值(如 `1px`), 默认值为空。
     - `border-radius`: 设置圆角半径，值可以是 `px` 或 `%` ，可以单独设置 `border-top-left-radius`/`border-top-right-radius`/`border-bottom-right-radius`/`border-bottom-left-radius` 分别改变每个角的半径。
     
     示例：
     
     ```css
     /* 设置所有边框 */
     table, th, td { border: 1px solid black; }
 
     /* 设置边框颜色 */
     button { border-color: red; } 
 
     /* 设置边框样式 */
     hr { border-style: dashed; } 
 
     /* 设置边框宽度 */
     fieldset { border-width: medium; } 
 
     /* 设置圆角半径 */
    .box { border-radius: 10px; } 
     ```
     
    #### 外边距属性
     - `padding`: 设置内边距，可以设置四个方向上的内边距，值可以是 `px` 或 `%`.
     - `margin`: 设置外边距，可以设置四个方向上的外边距，值可以是 `px` 或 `%`.
     
     示例：
     
     ```css
     /* 设置内边距 */
     header { padding: 10px; } 

     /* 设置外边距 */
     footer { margin: 20px; } 
     ```
     
    #### 盒模型属性
     - `display`: 设置元素的显示类型，如 `block`, `inline`, `inline-block`, `none`. 默认值为 `inline`.
     - `position`: 设置元素的定位类型，如 `static`, `relative`, `absolute`, `fixed`, `sticky`. 默认值为 `static`.
     - `float`: 设置元素是否浮动，可以是 `left` 或 `right`. 默认值为空。
     - `clear`: 设置元素的清除方式，可以是 `both`, `left`, `right`. 默认值为 `none`。
     - `overflow`: 设置元素溢出时的处理方式，可以是 `visible`, `hidden`, `scroll`, `auto`, `clip`. 默认值为 `visible`.
     - `overflow-x` / `overflow-y`: 只针对 `overflow` 为 `scroll` / `auto` 时生效，分别设置水平和垂直方向的溢出处理方式，可设置为 `visible`, `hidden`, `scroll`, `auto`. 默认值为 `visible`。
     - `min-width` / `max-width` / `min-height` / `max-height`: 设置元素的最小宽度/最大宽度、最小高度/最大高度，值可以是 `px` 或 `%`.
     - `width` / `height`: 设置元素的宽度/高度，值可以是 `px` 或 `%` 或 `auto` (`auto` 表示由内容决定高度)，默认为 `auto`。
     
     示例：
     
     ```css
     /* 设置元素的显示类型 */
    .container { display: block; } 

     /* 设置元素的定位类型 */
    .box { position: absolute; left: 10px; bottom: 10px; } 

     /* 设置元素是否浮动 */
    .image { float: right; } 

     /* 设置元素的清除方式 */
    .clearfix::after { clear: both; content: ""; display: table; } 

     /* 设置元素溢出时的处理方式 */
     textarea { overflow: hidden; } 

     /* 只针对 overflow 为 scroll/auto 时生效 */
    .scrollbar::-webkit-scrollbar { width: 6px; height: 6px; } 
    .scrollbar::-webkit-scrollbar-track { box-shadow: inset 0 0 6px rgba(0,0,0,0.3); } 
    .scrollbar::-webkit-scrollbar-thumb { background-color: darkgrey; outline: 1px solid slategrey; cursor: pointer; transition: all 0.3s ease-in-out; border-radius: 10px; } 
    .scrollbar::-webkit-scrollbar-thumb:hover { background-color: grey; } 
 
     /* 设置元素的最小/最大宽度/高度 */
    .box { min-width: 200px; max-width: 500px; min-height: 100px; max-height: 500px; } 
 
     /* 设置元素的宽度/高度 */
    .box { width: 100%; height: auto; }     
     ```
     
    #### 其他属性
     - `content`: 定义生成的内容，通常用来创建伪元素。
     - `@import`: 导入外部样式表。
     - `@media`: 根据不同的条件加载不同样式。
     - `@supports`: 判断浏览器是否支持某些特性。
     
     示例：
     
     ```css
     /* 定义生成的内容 */
     ::after { content: "Generated"; } 
 
     /* 导入外部样式表 */
     @import url('https://fonts.googleapis.com/css?family=Roboto');
 
     /* 根据不同的条件加载不同样式 */
     @media only screen and (max-width: 600px) { 
       body { font-size: 14px; } 
     } 
  
     /* 判断浏览器是否支持某些特性 */
     @supports (-moz-appearance: meterbar) { 
       input[type="range"] { appearance: slider; } 
     } 
     ```
    
    # 3.CSS核心技术
    ## 3.1 块级元素和行内元素
    HTML中有两种类型的元素：块级元素和行内元素。下面是它们的区别：
    
    1. 块级元素:

   | Block-level element               | Inline element                |
   | :-------------------------------- |:-----------------------------:|
   | `div`, `h1`-`h6`, `p`, `form`      | `span`, `img`, `input`        |
   
   块级元素占据完整的屏幕宽度，即左右两侧的 padding 和 margin。块级元素通常会另起一行开始排版，并且可以设置宽高和边距属性。
   
   2. 行内元素:
  
   | Inline element                    | Block-level element           |
   | :--------------------------------:|:------------------------------:|
   | `span`, `img`, `input`            | `div`, `h1`-`h6`, `p`, `form`   |
   
   行内元素只占据其父元素所需宽度，不会另起一行开始排版。如果父元素设置了宽高属性，则该元素也会受到影响。

    ## 3.2 盒模型
    盒模型（Box Model）是CSS的一种布局机制，用于生成矩形区域并控制内容的显示。CSS盒模型包括：边框（border）、填充（padding）、内边距（margin）和内容。

    **边框（Border）**
    
    CSS边框的属性如下：
    
    - `border-width`: 设置边框宽度，可以为 `border-top-width`, `border-right-width`, `border-bottom-width`, `border-left-width`, 或 `border-width` 为一个值。
    - `border-style`: 设置边框样式，可以为 `border-top-style`, `border-right-style`, `border-bottom-style`, `border-left-style`, 或 `border-style` 为一个值。
    - `border-color`: 设置边框颜色，可以为 `border-top-color`, `border-right-color`, `border-bottom-color`, `border-left-color`, 或 `border-color` 为一个值。
    
    **填充（Padding）**
    
    CSS填充的属性如下：
    
    - `padding`: 可以为 `padding-top`, `padding-right`, `padding-bottom`, `padding-left`, 或 `padding` 为一个值。
    - `padding-top`, `padding-right`, `padding-bottom`, `padding-left`: 各自设置对应边的填充距离。
    
    **内边距（Margin）**
    
    CSS内边距的属性如下：
    
    - `margin`: 可以为 `margin-top`, `margin-right`, `margin-bottom`, `margin-left`, 或 `margin` 为一个值。
    - `margin-top`, `margin-right`, `margin-bottom`, `margin-left`: 各自设置对应边的外边距距离。
    
    **内容**
    
    CSS内容的属性如下：
    
    - `width`, `height`: 设置内容区域的宽高。
    - `max-width`, `max-height`: 设置内容区域的最大宽高。
    - `min-width`, `min-height`: 设置内容区域的最小宽高。
    - `line-height`: 设置内容行的高度。
    - `white-space`: 设置空白符处理方式。
    - `text-indent`: 设置首行缩进。
    - `text-align`: 设置文本的水平对齐。

    ## 3.3 选择器优先级
    如果两个或两个以上选择器同时满足，则按照它们的顺序依次应用。样式的优先级决定着最终的样式值。CSS有四个级别的优先级：

1. 外部样式表（external style sheet）: 最高优先级，可以控制网页的整体样式。通过 link 标签链接到外部样式表。
2. 内部样式表（internal style sheet）: 中间优先级，可以给指定的 HTML 标签设置特定的样式。通过 style 标签定义。
3. 内联样式（inline styles）: 最低优先级，可以直接在 HTML 标签中设置 style 属性，这种样式只作用于当前元素。
4. 浏览器默认样式: 最后才会应用的样式，一般情况下浏览器都会预先设置好默认样式。

    ## 3.4 居中布局
    CSS中的居中布局主要用到的属性是 `margin`、`padding`、`position`、`display`、`flex`、`grid`等。下面介绍几个常用的方法：

    1. `margin: 0 auto;`：利用 `margin: 0 auto;` 来让块级元素水平居中。

    ```css
   .wrap {
      margin: 0 auto;
      width: 50%;
      height: 100px;
      background-color: green;
    }
    ```

    2. `position: relative; left: 50%; transform: translateX(-50%);`：利用 `position: relative; left: 50%; transform: translateX(-50%);` 来让块级元素水平居中。

    ```css
   .wrap {
      position: relative;
      left: 50%;
      transform: translateX(-50%);
      width: 50%;
      height: 100px;
      background-color: green;
    }
    ```

    3. Flexbox 布局：利用Flexbox布局可以实现各种类型的弹性布局，可以方便实现复杂的垂直、水平居中布局。

    ```css
   .container {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
    }
    ```

    4. Grid 布局：利用Grid布局可以实现网格化的布局，可以方便实现复杂的网格布局。

    ```css
   .container {
      display: grid;
      place-items: center;
      height: 100vh;
    }
    ```

    ## 3.5 清除浮动
    CSS中有一个属性叫做 `clearfix`，它是为了解决掉父级元素中的子元素不能在同一行的问题。

    下面的例子演示了如何使用 `overflow: auto;` 和 `clearfix` 来清除浮动。

    ```css
   .container {
      *zoom: 1; // for IE6 & IE7
      zoom: 1;
    }
   .container:after {
      content: '';
      display: table;
      clear: both;
    }
   .item {
      float: left;
      width: 100px;
      height: 100px;
      background-color: lightblue;
      margin-bottom: 10px;
    }
    ```

    使用 `*zoom: 1;` 兼容IE6/7。

    在IE8下使用 `-ms-clear: both;` 清除浮动。