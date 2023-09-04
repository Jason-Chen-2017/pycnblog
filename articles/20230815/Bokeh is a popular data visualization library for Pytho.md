
作者：禅与计算机程序设计艺术                    

# 1.简介
  
及背景介绍
Bokeh 是一款开源数据可视化库，具有简单、高效、交互性强等优点。它最初是为了科研目的设计开发的，随着其功能的不断扩充，已经成为许多领域的必备工具。比如在金融领域，它的图表可以实时反映市场的动向；在科技领域，它可以方便地生成复杂的数据可视化报告；在医疗健康领域，它可以帮助医生快速了解患者的实时状况；在娱乐领域，它可以直观地呈现音乐视频中变化的图像等。由于其强大的交互特性，使得用户可以很轻松地对图形进行调整，制作出令人惊艳的动态图表。虽然很多人认为，“能用就行”，但是Bokeh 的学习曲线还是比较陡峭的。因此，如果你想在实际项目中应用到该库，建议首先去官方文档学习相关知识。以下是一些典型的学习路径供参考：
1）官方文档学习：
http://bokeh.pydata.org/en/latest/docs.html

2）Bokeh 教程：https://www.youtube.com/playlist?list=PLv7mKNHfL8TtSfxJLXR_cCQrMsRNSXTVH

3）中文社区（包括官方QQ群）：
http://www.osgeo.cn/

除了官方文档，我还推荐一些其他的材料：

1）Codecademy：Learn Data Visualization with Python Using Bokeh
https://www.codecademy.com/learn/learn-data-visualization-with-python-using-bokeh
2）Datacamp: Interactive Data Visualizations with Bokeh in Python
https://www.datacamp.com/community/tutorials/interactive-data-visualizations-bokeh
3）LinkedIn Learning: Introduction to Data Visualization with Bokeh
https://www.linkedin.com/learning/introduction-to-data-visualization-with-bokeh
这些资源虽然不是官方文档，但它们能够提供一些额外的学习资料。总之，如果你的目的是只是熟悉 Bokeh 的使用方法，那直接阅读官方文档足够了。如果你想要了解 Bokeh 的原理以及内部实现机制，或者你要在实际项目中应用 Bokeh 来提升产品性能，那么继续阅读本文会对你非常有帮助。
# 2.基本概念术语说明
## 2.1 Bokeh 画布
首先，我们需要知道什么是 Bokeh 画布。在 Bokeh 中，所有图表都绘制在一个画布上，称为 Figure。Figure 可以理解成一个容器，里面有各种各样的组件，包括：
1）Plot：用来绘制图表的区域。
2）Axis：坐标轴。
3）Grid：网格线。
4）Legend：图例。
5）Toolbar：工具栏，用于缩放、移动、下载图片、保存文件等操作。
6）RangeTool：选择范围的工具。
7）HoverTool：悬停提示工具。
8）Title：标题。
9）Label：图表标签。
10）Annotation：图表注释。
通过将不同的组件组合起来，我们可以构造出不同类型的图表，例如散点图、折线图、条形图、箱型图等。

当我们将多个图表放在一起的时候，我们可以把他们放置在同一个画布中。也可以根据业务逻辑拆分画布，比如不同时间段的数据可视化放在一个画布，不同类别的数据放在另一个画chaft。通过这种方式，我们可以同时展示多个图表，并且根据不同的业务需求，自由选择合适的画布布局。

## 2.2 Glyphs 符号
Glyphs 是指构成 Bokeh 图表中的各种元素，如点、线、面等。Bokeh 提供了丰富的符号类型，如矩形、圆形、气泡图、线条等。每种符号都有自己的属性，如颜色、大小、透明度等。可以通过设置符号属性来控制图表的整体风格。

## 2.3 Tools 工具
Tools 是 Bokeh 中的重要组成部分。它提供了丰富的交互功能，如框选、缩放、点击旁边显示信息等。可以通过工具按钮、悬停提示、点击事件、回调函数等来操控图表。

## 2.4 Widgets 小部件
Widgets 是指一些小部件，可以让用户自定义一些参数，如图表的样式、交互模式等。Widgets 可以帮助用户更直观地对图表进行配置，减少繁琐的重复工作。Widgets 大致可以分为两类：
1）基础 Widget：包括文本输入框、下拉菜单、单选按钮、复选框等。
2）组合 Widget：包括滑块、滑竿、进度条等。

## 2.5 数据集
最后，我们需要准备好数据集。我们可以使用 Numpy、Pandas、PySpark、Dask 等库读取本地或远程的数据集，也可以从一些公开的数据源获取数据。一般情况下，我们需要准备好经过处理的数据集。