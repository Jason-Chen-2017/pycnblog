
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Google Data Studio 和Tableau都是免费商用的BI工具，本文将用两个工具制作一个美观的、功能完备的BI仪表盘。具体效果如下图所示：


在这张图中，我们可以看到左侧的“1-Year Sales”表格从头到尾都带有一些设计上的元素：首先，它采用了金色的背景，表格单元格上方也配有描述性的标签；第二，图形标志被放置于表格第一行的位置；第三，所有重要信息都放在最上面，以便用户快速浏览；第四，右边的表格提供了月度销售额的信息，它在视觉效果上突出了数据点，同时还有图例提供直观的数据可读性；最后，页面采用了极具视觉冲击力的字体风格，给人带来一种独特的沉浸式氛围。

这是一个美观且功能完整的BI仪表盘，而且只需要简单的几个步骤就可以完成。那么，如何制作这样一个仪表盘呢？让我们接下来一起了解一下构建这个仪表盘的方法。

# 2.基本概念、术语说明
## 2.1 数据集（Dataset）
数据集通常是指包含多个相关数据源的数据集合。比如，一份销售数据集可能包括了每个月的销售额、每笔订单的金额、顾客的收入等等。而我们要做的就是从这些数据中找出有用的信息并生成易于理解的可视化结果。
## 2.2 维度（Dimension）
维度通常是指对分析进行划分的不同分类标准。比如，一份产品销售数据集可能包括了时间、地点、品牌等作为维度，分别对应着销售的日期、销售的地点、销售的品牌。
## 2.3 度量（Measure）
度量通常是指用来衡量某些变量的定量数据。比如，一份销售数据集可能包括了销售额、订单量、顾客数、利润等作为度量。
## 2.4 属性（Attribute）
属性（Attribute）通常是指描述数据集的一些辅助信息。比如，一份销售数据集可能还会有一些描述性信息，如客户群、货币类型、经营方式等。
## 2.5 聚合（Aggregation）
聚合是指将数据按照不同的维度进行分组或过滤，然后计算出某个度量值。举个例子，我们想知道公司在不同年份的总销售额，就可以按照年份作为维度，计算每个年份的总销售额。这种聚合可以帮助我们更好地了解数据的变化趋势。
## 2.6 主题（Theme）
主题是指用于优化仪表盘外观的一些设计原则。比如，一款红色主题的BI仪表盘可能会充满激情，而一款蓝色主题的仪表盘则会显得生机勃勃。
## 2.7 折线图（Line Charts）
折线图是一种常用的图表类型，它用来呈现随着时间变化的数据。具体来说，它能够显示一条直线或曲线，用于表示变量随时间的变化关系。
## 2.8 柱状图（Bar Charts）
柱状图也是一种常用的图表类型，它用来呈现一段时间内某个维度的变化。具体来说，它将数据通过竖直方向排列，且每个条形或长方形代表一个数据点。
## 2.9 饼图（Pie Charts）
饼图是一种常用的图表类型，它用来呈现某个度量值的占比。具体来说，它将数据以圆弧或扇形的方式分成各个部分，且占比大小由切片颜色、面积大小、位置等决定。
## 2.10 散点图（Scatter Plot）
散点图是一种常用的图表类型，它用来呈现两变量之间的相关性。具体来说，它将数据以一系列的点绘制在平面坐标系上，横轴和纵轴分别表示两个变量。
## 2.11 KPI（Key Performance Indicator）
KPI（关键性能指标）通常是指衡量一件事物质量或完成一项任务的特定指标。比如，一项销售任务的KPI可能是成交率，它用来衡量销售人员能否按时完成销售任务。
## 2.12 排序（Ordering）
排序是指根据指定的规则对数据集中的记录进行重新排列。比如，我们可以根据销售额对销售数据集进行降序排序，将销售最好的项目排在前面。
## 2.13 分组（Grouping）
分组是指对数据集中的记录进行分区或分组。比如，我们可以根据产品类别对销售数据集进行分组，然后再计算出每个类别的平均销售额、最大销售额、最小销售额、总销售额等信息。
## 2.14 可视化（Visualization）
可视化是指将数据转化为易于理解的图形或者图表形式。主要目的在于提高数据的认知效率，帮助人们更好地理解数据。比如，我们可以使用饼图、柱状图、折线图等图表展示公司各年度销售额的变化。
## 2.15 可视化组件（Visualization Components）
可视化组件是指将可视化功能拆分为多个小块，每个小块可以单独实现一个特定的功能。比如，当我们需要对销售数据集进行多维度分析时，我们可以将其拆分为三个可视化组件：总销售额、类别销售额和促销销售额，分别展示每种类型的销售额。
# 3.具体操作步骤
## 3.1 准备数据
首先，我们需要收集一些数据。一般来说，数据需要符合以下要求：
1. 有足够多的维度和度量。
2. 数据需要准确无误。
3. 数据不能太过复杂。
4. 需要添加一些属性来丰富数据。
这里以一个销售数据集为例，假设我们已经收集到了6个维度和1个度量，如下表所示：

| Dimensions | Measure        | Attribute    |
|------------|----------------|--------------|
| Year       | Total Sales    |              |
| Month      | Monthly Sales  |              |
| Country    | Number of Orders| Customer Type|
| City       | Average Order Size|     |
| Product    | Revenue        | Product Name|
| Promotion  | Net Profit     |              |

2.2 将数据导入到BI工具中
现在，我们把这些数据导入到Google Data Studio或Tableau中，具体操作步骤如下：
### Google Data Studio

1. 在Data Studio的顶部菜单栏点击“Connect to data”，选择想要导入的数据源。
2. 查看数据源后，将其拖动至画布区域，然后点击“Add data”。
3. 设置数据集名称并确认数据加载进来。
4. 在字段列表中将要使用的维度和度量拖动至图层面板中。
5. 在“Transform”面板中设置维度和度量转换方式，比如设置各维度相加作为总计销售额、设置平均值作为平均订单大小。

### Tableau

1. 打开Tableau Desktop，进入“Get Data”界面，选择想要导入的数据源。
2. 查看数据源后，将其拖动至画布区域，然后点击“Connect”。
3. 在“Sheet 1”选项卡中将要使用的维度和度量拖动至视觉元素列表中。
4. 在“Calculation”选项卡中设置维度和度量转换方式，比如设置各维度相加作为总计销售额、设置平均值作为平均订单大小。
5. 保存工作表。

## 3.2 创建可视化组件
在制作仪表盘之前，我们需要先创建一些可视化组件。比如，对于销售数据集，我们可以创建一个汇总表格，其中包含6个维度和1个度量。具体操作步骤如下：
### Google Data Studio

1. 点击画布区域的空白处，然后点击“Create visualization”。
2. 选择表格图标，然后设置标题。
3. 从图层面板中将想要使用的维度和度量拖动到“Rows”、“Columns”和“Values”框中。
4. 在“Format”面板中设置单元格大小和字体颜色。
5. 在“Customize”面板中调整其他参数，比如将表格的边框和网格设置为透明。

### Tableau

1. 点击画布区域的空白处，然后点击“Insert > Visualization”。
2. 在弹出的窗口中选择表格图标，然后点击确定。
3. 设置标题。
4. 从视觉元素列表中将想要使用的维度和度量拖动到“Rows”、“Columns”和“Values”框中。
5. 在“Properties”选项卡中设置单元格大小和字体颜色。
6. 在“Options”选项卡中调整其他参数，比如将表格的边框和网格设置为透明。

## 3.3 设计主题和配色方案
在制作仪表盘的时候，我们需要为其配上适合的主题和配色方案。这样才能使我们的仪表盘具有更加吸引人的视觉效果。具体操作步骤如下：
### Google Data Studio

1. 在工具栏中点击“Themes”按钮，然后选择一个喜欢的主题。
2. 如果需要更改主题色调，可以在“Customize theme colors”面板中设置。
3. 关闭该面板，然后点击“Publish”按钮发布仪表盘。

### Tableau

1. 在工具栏中点击“View”按钮，然后点击“Color Palette”按钮。
2. 在弹出的窗口中选择一个喜欢的主题色。
3. 关闭该窗口。

## 3.4 添加注释和说明
在仪表盘中添加一些注释和说明文字，可以增加仪表盘的清晰度。比如，我们可以向大家介绍一下这张表是什么意思，以及数据的来源。具体操作步骤如下：
### Google Data Studio

1. 点击画布区域的空白处，然后点击“Insert”菜单下的“Textbox”图标。
2. 使用键盘输入相关文字，然后点击确定。

### Tableau

1. 点击画布区域的空白处，然后点击“Insert > Text Box”。
2. 使用键盘输入相关文字，然后点击确定。

## 3.5 设置自动刷新频率
由于数据实时性很强，因此我们可以设置自动刷新频率，更新仪表盘上的数据。具体操作步骤如下：
### Google Data Studio

1. 在工具栏中点击“Schedule refresh”按钮。
2. 设置刷新频率，比如每隔五分钟刷新一次。
3. 保存仪表盘。

### Tableau

1. 在工具栏中点击“Schedule”按钮，然后选择“Refresh Frequency”。
2. 设置刷新频率，比如每隔五分钟刷新一次。
3. 关闭该窗口。

## 3.6 设置过滤器
由于不同的业务团队会对数据有不同的分析需求，因此我们可以设置过滤器。这样可以让用户自由地按需筛选出自己感兴趣的数据。具体操作步骤如下：
### Google Data Studio

1. 点击画布区域的空白处，然后点击“Filter”图标。
2. 拖动维度或度量到“Filters”面板中。
3. 设置筛选条件。

### Tableau

1. 点击画布区域的空白处，然后点击“Show Filters”按钮。
2. 拖动维度或度量到“Filter”面板中。
3. 设置筛选条件。

## 3.7 加入交互元素
由于仪表盘的目的是帮助人们更好地理解数据，因此我们应该加入一些交互元素。比如，当用户点击某个具体指标，我们可以显示更详细的图表，方便用户进行分析。具体操作步骤如下：
### Google Data Studio

1. 点击画布区域的空白处，然后点击“Create visualization”。
2. 选择想要使用的可视化元素，比如折线图或柱状图。
3. 拖动维度或度量到“X-axis”、“Y-axis”或“Size”框中。
4. 在“Format”面板中设置颜色、标记符号、图例等。
5. 设置维度或度量的交互行为，比如点击某个具体指标，显示对应的图表。
6. 更新工作表。

### Tableau

1. 点击画布区域的空白处，然后点击“Insert > Visualization”。
2. 选择想要使用的可视化元素，比如折线图或柱状图。
3. 拖动维度或度量到“Fields”面板中。
4. 在“Marks”选项卡中设置颜色、标记符号、图例等。
5. 设置维度或度量的交互行为，比如点击某个具体指标，显示对应的图表。
6. 保存工作表。