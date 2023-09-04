
作者：禅与计算机程序设计艺术                    

# 1.简介
  

布局机制是一个非常重要的组件，在 React Native 中布局主要由 Flexbox 和 Layout Props 两种方式进行实现。Flexbox 是 CSS 中的一种布局系统，可以很好地处理多维空间中的元素排版。Layout Props 是 Facebook 推出的一种新的布局解决方案，它通过 props 来设置布局相关属性，从而实现更灵活的布局。本文将详细介绍这两种布局机制及其应用场景。
# 2.布局系统
## Flexbox
Flexbox 全称 Flexible Box，是 CSS 中的一种布局系统，最早于 1997 年由 W3C 提出，目前已成为主流的布局系统。它的核心思想是，通过定义容器的主轴（main axis）和交叉轴（cross axis），让子元素能够根据不同的 flex-grow、flex-shrink 和 flex-basis 设置确定它们的尺寸和位置。
### 定义
Flexbox 使用 display: flex 或 inline-flex 对父级元素进行定义，定义之后，就可以在该元素内使用 Flexbox 属性对子元素进行布局了。Flexbox 的属性包括：
- display: flex | inline-flex - 将元素设置为 flex 或 inline-flex 模式。
- justify-content - 水平方向上的对齐方式，可选值：flex-start、flex-end、center、space-between、space-around。
- align-items - 垂直方向上的对齐方式，可选值：stretch、flex-start、flex-end、center、baseline。
- align-self - 指定单个子元素的对齐方式，可选值同上。
- flex-direction - 决定主轴的方向，可选值：row、row-reverse、column、column-reverse。
- flex-wrap - 当一条轴线排不下所有的子项时，是否换行，可选值：nowrap、wrap、wrap-reverse。
- flex-flow - flex-direction 和 flex-wrap 的简写形式，属性值为两个单词组成的字符串，中间用空格隔开。
- order - 用来控制布局顺序。默认值为 0 ，值越小，排列顺序越靠前；值越大，排列顺序越靠后。
- flex-grow - 定义子元素的放大比例，值越大，则子元素的尺寸会增大。
- flex-shrink - 定义子元素的缩小比例，值越大，则子元素的尺寸会减少。
- flex-basis - 定义子元素的初始大小。
- align-content - 如果有多根轴线，用这个属性控制这些轴线之间的对齐方式，可选值：flex-start、flex-end、center、space-between、space-around、stretch。
Flexbox 可以很好地处理多维空间中的元素排版，对于简单的页面布局来说，这种布局方法效率很高。然而，Flexbox 有以下限制：
- IE 10 及之前版本不支持 Flexbox。
- 在某些情况下无法实现复杂的布局，比如网格布局等。
## Layout Props
Layout Props 是 Facebook 推出的一种新的布局解决方案，它通过 props 来设置布局相关属性，从而实现更灵活的布局。Layout Props 可以理解为是一套约定好的 API，允许开发者指定视图组件的坐标和尺寸。它的属性包括：
- position - 指定组件的定位类型，可选值：static、relative、absolute、sticky、fixed。
- left/top/right/bottom - 指定相对于父组件的左上角或右上角坐标，或者相对于自身的左上角或右上角坐标。
- width/height - 指定组件的宽度或高度。
- minWidth/minHeight/maxWidth/maxHeight - 指定组件的最小宽度或高度，最大宽度或高度。
- margin/marginLeft/marginRight/marginTop/marginBottom - 指定组件四边的外边距。
- padding/paddingLeft/paddingRight/paddingTop/paddingBottom - 指定组件四边的内边距。
Layout Props 的优点是灵活性高，适用于各种场景。但它也存在一些局限性，比如：
- 不支持传统的浮动和绝对定位，只能通过 left、top、width、height、margin、padding 来控制布局。
- 需要手动计算坐标和尺寸，开发者需要先知道所有子组件的尺寸才能确定自己的尺寸和位置。
总结起来，Flexbox 和 Layout Props 是两种不同的布局系统，各有千秋。如果要做一个简单的页面布局，建议使用 Flexbox，否则可以使用 Layout Props 。