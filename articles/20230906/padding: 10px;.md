
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 什么是padding？
Padding（空白）是CSS中的属性，用来控制元素内边距（内外边距、内补丁、行间距等）的距离。它可以设置四个方向上的空白距离。
## 1.2 为什么要用padding？
在网页设计中，经常会遇到字体大小不一导致内容排列不一致的情况，比如图标文字、按钮文字、菜单文字等。常规解决方式是增加元素之间的距离，让元素看起来整齐。然而这种做法只能减少空间利用率，不能完全解决字体大小不一的问题。所以需要通过padding控制元素间的距离来解决。除此之外，还可以通过padding改变元素的内容之间的距离。如文本的行距、列表项的垂直对齐等。
## 1.3 padding属性
padding属性接受一个长度或百分比值，或者同时接受四个值分别设置四个方向上的空白距离。语法如下：
```css
/* 设置单侧的内边距 */
padding-top|bottom|left|right: length | % ; 

/* 设置上下左右的内边距 */
padding: top right bottom left; 

/* 设置水平和垂直方向上的内边距 */
padding: horizontal vertical; 

/* 综合设置 */
padding: top right bottom left /* 可以省略前面的某些值 */; 
```
## 1.4 应用场景
一般来说，padding的应用场景主要有以下几种：

1. 增大元素间的距离，提高页面视觉效果；
2. 在元素周围添加空白，增加外观舒适性；
3. 调整行距、列表项的垂直对齐，美化文本；
4. 用作容器包裹其他元素，扩充元素区域，便于布局；
5. 滑动滚动条时显示更多内容，保持良好的用户体验。
# 2.Padding的使用
## 2.1 相同的padding值
给多个元素统一设置相同的padding值很简单，只需要声明一次padding值即可。例如，想要设置上下左右都为20像素的外边距，只需写如下样式代码：
```html
<div class="box">这是第一个div</div>
<div class="box">这是第二个div</div>
<div class="box">这是第三个div</div>

<style type="text/css">
   .box {
        width: 200px;
        height: 200px;
        background-color: pink;
        margin: 20px;
        padding: 20px; /* 设置相同的padding值 */
    }
</style>
```
结果如下：
## 2.2 不一样的padding值
如果想给不同元素设置不同的padding值，比如上边距5像素，下边距10像素，左边距20像素，右边距30像素，可以写多份CSS样式代码，或者用JavaScript动态修改样式。这里举例用多份CSS样式代码实现：
```html
<div class="box box1"></div>
<div class="box box2"></div>
<div class="box box3"></div>

<style type="text/css">
   .box {
        width: 200px;
        height: 200px;
        border: solid 1px #ccc;
        text-align: center;
        line-height: 200px;
        font-size: 30px;
        color: #fff;
        position: relative;
        cursor: pointer;
    }
    
   .box1 {
        padding: 5px; /* 上边距为5像素 */
        background-color: #ff8a8a;
    }
    
   .box2 {
        padding: 5px 10px; /* 上边距为5像素，下边距为10像素，左右边距为默认值 */
        background-color: #6ce26c;
    }
    
   .box3 {
        padding: 5px 20px 10px 30px; /* 上边距为5像素，下边距为10像素，左边距为20像素，右边距为30像素 */
        background-color: #ffa500;
    }
</style>
```
结果如下：