
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 一、产品概述
### （1）产品名称：display: block;编辑器插件
### （2）产品定位：解决前端工程师在页面布局中的一些问题，提供便捷的布局功能。
### （3）产品特色：具有简单、快捷、直观的操作方式，适合Web前端工程师上手。
## 二、功能模块及用法
### （1）自动生成CSS代码
#### ①HTML源码转成可视化DOM树结构
#### ②修改节点样式
#### ③保存至本地
### （2）快速响应调整
#### ①双击节点快速添加新元素或显示/隐藏属性
#### ②拖拽节点调整布局
#### ③快捷键操作（暂不支持）
### （3）自定义规则设置
#### ①快速选择不同类型的节点
#### ②拓展更多可用节点类型
#### ③自定义规则覆盖默认规则
### （4）响应式预览效果
#### ①手机端和平板设备上都可以看到预览效果
#### ②将页面放大缩小都可以在预览画布中看到对应效果变化
### （5）其他功能（按需增加）
#### ①复制/粘贴样式块到当前页或别处
#### ②导入/导出页面样式
#### ③浏览器自动刷新更新样式
## 三、基本概念及术语
CSS中的block(块级)元素和inline(内联)元素。
block element 是指那些占据完整宽度的元素，如<div>、<p>等；而inline element 是指那些只占据它需要的宽度的元素，如<span>、<a>等。
CSS中的box model。
Box Model描述了 HTML 文档中各个元素所构成的矩形盒子模型。分为内容区（content area），边框区（border area），内边距区（padding area），外边距区（margin area）。它们的位置关系如下图所示：

### width、height
width、height分别设置了盒子的水平方向宽度和垂直方向高度。如果不设置，则由内部的内容撑开盒子。
示例：
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>width、height</title>
    <style>
     .container {
        border: 1px solid red;
        padding: 10px;
      }
      #box1 {
        background-color: yellow;
        height: 100px; /* 设置盒子高度 */
      }
      #box2 {
        background-color: orange;
        width: 100px; /* 设置盒子宽度 */
      }
      #box3 {
        background-color: greenyellow;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div id="box1"></div>
      <div id="box2"></div>
      <div id="box3"></div>
    </div>
  </body>
</html>
```

### margin、padding、border
margin、padding、border三个属性用于控制盒子的四周空间，margin用于设置四个方向上的距离，padding用于设置内容和边框之间的距离，border用于设置边框的大小、颜色和样式。
示例：
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>margin、padding、border</title>
    <style>
     .container {
        border: 1px solid red;
        margin: 10px; /* 上右下左边距同时设置 */
        padding: 10px; /* 上右下左内间距同时设置 */
      }
      #box1 {
        background-color: yellow;
        height: 100px;
        width: 100px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div id="box1"></div>
    </div>
  </body>
</html>
```

### box-sizing
box-sizing 属性用来设置盒子的边框盒和内边距盒是否计算进内容的宽高里。默认值为 content-box ，表示内容区域的宽高会被加上内边距、边框的宽度，因此盒子实际占据的空间要比设定的大一些。设置为 border-box 时，则不会再加上内边距和边框的宽度，即盒子实际占据的空间等于设定的值。
示例：
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>box-sizing</title>
    <style>
     .container {
        border: 1px solid red;
        padding: 10px;
        box-sizing: border-box;
      }
      #box1 {
        background-color: yellow;
        height: 100px;
        width: 100px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div id="box1"></div>
    </div>
  </body>
</html>
```

### position
position属性用于设置盒子相对于其父容器的定位方式。static表示非定位，relative表示相对定位，absolute表示绝对定位。
示例：
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>position</title>
    <style>
     .container {
        border: 1px solid black;
        position: relative; /* 相对定位 */
        left: 20px; /* 沿着x轴偏移20px */
        top: -10px; /* 沿着y轴向上偏移10px */
      }
      #box1 {
        background-color: yellow;
        height: 100px;
        width: 100px;
      }
      #box2 {
        background-color: blue;
        height: 100px;
        width: 100px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div id="box1"></div>
      <div id="box2"></div>
    </div>
  </body>
</html>
```

### float
float属性用于设置盒子浮动的方式。left表示向左浮动，right表示向右浮动。通过设置clear属性，可以清除浮动。
示例：
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>float</title>
    <style>
     .container {
        border: 1px solid black;
      }
      #box1 {
        background-color: yellow;
        height: 100px;
        width: 100px;
        float: left; /* 浮动到左边 */
        margin-right: 10px; /* 和右边的元素隔离 */
      }
      #box2 {
        background-color: blue;
        height: 100px;
        width: 100px;
        clear: both; /* 清除两侧的浮动 */
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div id="box1"></div>
      <div id="box2"></div>
    </div>
  </body>
</html>
```