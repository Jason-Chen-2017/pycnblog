
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
在前端开发中，margin:auto可以设置元素居中显示或根据父元素的宽度、高度自动调整间距等功能。但是要注意的是，margin:auto并不是单纯的让某个元素居中，而是能够将多个块级元素设置在一行上并居中显示，并且可以解决一些对称布局的问题。本文将会探讨margin:auto的应用场景及其在实际项目中的应用。

## 基本概念与术语
### 父元素（Parent Element）
即一个块级元素作为另一个块级元素的直接容器时，称之为父元素；例如，div标签作为body标签的直接子元素，则div为body的父元素。

### 兄弟元素（Sibling Elements）
同一个父元素下面的两个相邻的块级元素，叫做兄弟元素；例如，div1和div2都属于父元素div，那么div1和div2都是兄弟元素。

## 用法与适用场景
### margin:auto
如果某个元素设置了margin:auto属性，浏览器会自动调整该元素的上下边距，使得元素居中显示。其具体实现方式如下：

1. 当存在左右外边距(padding)时，不管是否设置width，都会在元素的左右增加相同的内边距(padding)。
2. 元素会水平居中显示，且左右内边距均分。
3. 如果元素的宽度固定或者所属父元素的宽度已知，则margin:auto不会改变元素的宽度，否则将会按比例缩放元素的宽度。
4. 在屏幕尺寸变化的时候，margin:auto会自动更新元素的位置和宽度。

### 使用margin:auto实现水平居中
```html
<div class="parent">
  <div class="child"></div>
</div>

<style type="text/css">
 .parent {
    width: 100%;
    height: 300px;
    background-color: red;
  }
  
 .child {
    width: 200px;
    height: 200px;
    background-color: blue;
    /* 设置margin:auto使元素水平居中 */
    margin: auto;
  }
</style>
```


如上图所示，在设置了margin:auto之后，div元素的左右内边距均分，元素水平居中显示。

### 使用margin:auto实现垂直居中
```html
<div class="parent" style="height: 300px;">
  <div class="child" style="height: 200px;"></div>
  <div class="child" style="height: 200px;"></div>
  <!-- 其他垂直排列的元素 -->
</div>

<style type="text/css">
 .parent {
    display: flex; /* 使用flex布局让div变成一个盒子 */
    align-items: center; /* 使用align-items使得子元素垂直居中 */
    justify-content: space-around; /* 使用justify-content让子元素左右平均分布 */
  }
  
 .child {
    width: 200px;
    height: 200px;
    background-color: blue;
    border: 1px solid black;
  }
</style>
```

如上图所示，设置了display:flex后，设置了align-items:center和justify-content:space-around后，div的子元素会垂直居中，且左右间距均匀分布。

## margin:auto的局限性
1. 不能解决百分比的宽度问题。
2. 当margin:auto的元素被嵌套到其它元素中时，margin:auto不会生效。

## 实践案例
最近我在编写一个电商网站的产品页面的时候，发现当用户刷新页面的时候，图片可能出现错乱的情况，因为图片采用了margin:auto实现居中效果，但由于父元素的宽度并未被计算进去，所以图片可能出现错乱。为了解决这个问题，我通过以下两种方法解决了该问题：

1. 不使用margin:auto进行居中，而是设置元素的left值和right值分别为auto即可。
   ```html
   <div class="parent">
   </div>

   <style type="text/css">
     img {
       max-width: 100%;
       height: auto;
     }

    .parent {
       position: relative;
     }

    .parent::after {
       content:"";
       display: block;
       clear: both;
     }

     img {
       float: left;
       margin: auto;
       padding: 20px;
       min-width: 50%;
     }
   </style>
   ```
   上面这种方法只适用于图片居中的情况，无法解决其他需要居中的场景。

2. 通过使用JavaScript动态获取父元素的宽度，然后将其赋值给img的max-width样式。
   ```javascript
   function setImgWidth() {
     const parent = document.querySelector('.product'); // 获取父元素
     const maxWidth = Math.floor((parent.offsetWidth - (parseInt(getComputedStyle(parent).paddingLeft) + parseInt(getComputedStyle(parent).paddingRight))) / 3); // 根据父元素宽高计算出图片最大宽度
     Array.from(document.querySelectorAll('img')).forEach(function(ele){
       ele.style.maxWidth = `${maxWidth}px`; // 为图片设置最大宽度
     });
   }

   window.onload = function(){
     setImgWidth(); // 初始化调用一次图片宽度设置函数
   };

   window.onresize = function(){
     setImgWidth(); // 当窗口大小发生变化时，重新设置图片宽度
   };
   ```
   此方法利用JavaScript动态获取父元素的宽度，然后将其赋值给img的max-width样式。这里使用了Math.floor方法向下取整，防止图片过小。

总结来说，margin:auto还是很有用的工具，但其局限性也很明显，应当合理地使用才是王道。