
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Tableau Public 是一款开源数据可视化工具，由微软开发并免费提供给个人和团体使用。近年来受到越来越多人的青睐，特别是在商业领域。在这个快速发展的时代，越来越多的人开始接受数据可视化的方式，希望更直观地看到数据背后的故事，从而做出更明智的决策。相信随着时间的推移，Tableau Public 将会成为更多人力资源和管理决策方面的一个不可替代工具。本文将介绍如何使用Tableau Public进行数据分析和可视化。
# 2.基本概念术语说明
## 数据分析
数据分析（Data Analysis）是指从数据中提取有价值的信息，并将这些信息用于报告、决策或其他目的的一系列活动，包括收集数据、分析数据、整理数据、挖掘数据模式、应用模型、总结结果、交流结果及运用知识解决问题等过程。其目的是通过对数据进行处理、提炼、组织和呈现，发现数据中的趋势、关系、规律、模式、核心因素、信息，以及从中找出潜在的问题和机会，以此来帮助公司或机构更好地洞察市场、制定决策、开拓创新。
## 可视化
可视化（Visualization）是利用图表、图像等媒介将数据以图形化的方式呈现出来，以便更好的理解和把握。可视化的作用不仅限于数据的呈现形式，还可以突出数据中最关键的视觉元素，增强用户对数据的感知能力。通过视觉化的数据呈现，能够让数据更容易被人类所理解和理解，从而使得数据的分析、挖掘和决策等工作更加有效。
## Tableau Public简介
Tableau Public是一个基于Web浏览器的交互式数据可视化工具。它提供了丰富的数据源支持，可以直接从各种数据源（如数据库、Excel、CSV文件等）导入数据，并根据数据类型自动生成相应的可视化图表。通过拖放图表、自定义视觉效果、添加文本注释、调整布局，可以完成复杂的可视化作品。
## 使用场景
作为一款开源可视化工具，Tableau Public有很多使用场景。一般来说，当数据量比较小，需要对数据进行快速分析时可以使用。比如，电子商务网站可以用来分析顾客购买行为，市场研究公司可以用来分析销售数据。当数据量比较大，需要进行复杂的分析时，可以使用Tableau Public。比如，保险公司可以通过表格的形式查看不同产品的收益率情况；而房地产公司则可以通过地图的形式直观显示各区房价分布和房价波动变化。由于它使用户界面友好，因此可以在移动设备上访问和使用。另外，还可以将可视化项目分享出去，让其他人也可以访问、评论和使用你的作品。
# 3.Core Algorithm and Operations in Tableau Public
## Introduction to Dashboards and Stories
Tableau Public 中有两种类型的可视化作品：Dashboard 和 Story。其中，Dashboard 可以看成是一个画布，里面包含多个可视化图表，主要用来展示复杂的数据集。Story 可以看成是 Dashboard 的一个精简版，一般只包含一个可视化图表。你可以创建任意数量的 Dashboard，每个 Dashboard 里都可以包含多个 Story。
*Figure 1: Dashboard* 

## Connecting Sources
Tableau Public 支持从许多数据源导入数据，包括数据库、CSV文件、JSON文件等。在连接数据源之前，你需要先安装相应的驱动程序。然后，你可以按照以下步骤连接数据源：

1. 在左侧的导航栏中选择 "Connect"，然后点击 "Connect to New or Existing Data Source" 按钮。

2. 在下拉菜单中选择数据源类型，如数据库、CSV文件、Excel等。

3. 根据数据源类型，填写相应的连接参数，例如主机名、端口号、用户名密码、数据表名等。

4. 验证是否连接成功。如果连接成功，就会出现对应的数据表列表。

*Figure 2: Connecting Sources* 

## Creating Calculations
Tableau Public 支持用户自己编写计算逻辑。你可以在数据表中点击 “New Calculation” 来创建新的计算字段。例如，假设有一个数据表 "Orders"，列 "Price" 和 "Discount" 有相关性，你想创建一个新的计算字段叫 "Discounted Price"，它的计算方式是价格减去折扣。你可以按以下步骤操作：

1. 打开数据表的编辑视图。

2. 右键单击 "Price" 和 "Discount" 两列，选择 "Create Calculation"。

3. 在弹出的窗口中输入 "Discounted Price" 作为计算名称，然后在 "Formula" 框中输入 "=Price-Discount"。注意 "=" 是运算符号，而不是赋值符号。

4. 点击 "OK" 按钮确认新建的计算字段。

*Figure 3: Creating Calculations* 

## Filter and Split By
Tableau Public 提供了丰富的筛选功能，你可以筛选出满足特定条件的数据。例如，假设你有一个数据表 "Customers"，列 "Country" 表示客户所在国家，列 "Sales" 表示该国家的营业额。你想过滤出英国、德国的营业额，可以按以下步骤操作：

1. 打开数据表的编辑视图。

2. 在左侧的 "Fields" 面板中找到 "Country" 和 "Sales" 两个列，然后将它们拖动至 "Rows" 区域。

3. 在 "Marks" 面板中，勾选 "Show All Rows" 以显示所有数据。

4. 点击 "Filter" 标签，进入筛选器页面。

5. 在 "Filters" 区域，双击 "Country" 列，在弹出的下拉菜单中选择 "Equals" 操作符。

6. 在 "Values" 框中输入 "England" 或 "Germany" 之一，然后点击 "Add" 按钮。

7. 再次点击 "Filter" 标签，再次双击 "Country" 列，在弹出的下拉菜单中选择 "Not Equals" 操作符。

8. 在 "Values" 框中输入另一个国家，然后点击 "Add" 按钮。

9. 点击 "Apply" 按钮确认筛选条件。

*Figure 4: Filtering Data* 

除了筛选功能外，Tableau Public 还支持分组功能。你可以按照某个字段的值将数据分成不同的组，方便后续分析。例如，假设有一个数据表 "Products"，列 "Category" 表示产品种类，列 "Price" 表示价格。你想统计不同种类的商品的平均价格，可以按以下步骤操作：

1. 打开数据表的编辑视图。

2. 在左侧的 "Fields" 面板中找到 "Category" 和 "Price" 两个列，然后将它们拖动至 "Columns" 区域。

3. 在 "Marks" 面板中，勾选 "Show Measures" 以隐藏表格内的所有数据。

4. 在 "Calculations" 面板中，点击 "Add Calculation" 按钮。

5. 在弹出的窗口中，输入 "Average Price" 作为计算名称，并输入 "=AVERAGE(Price)"。注意 "=" 不是运算符号，而是函数的语法。

6. 点击 "OK" 按钮确认新建的计算字段。

7. 点击 "Sheet Toolbar" 中的 "Split by" 按钮，选择 "Category" 字段，即可实现分组。

*Figure 5: Splitting Data by Category* 

## Sorting and Ordering
Tableau Public 支持排序功能，你可以按照某一字段的升序或者降序排列数据。例如，假设有一个数据表 "Orders"，列 "Order Date" 表示订单日期，列 "Order Amount" 表示订单金额。你想按照订单日期从早到晚排序，可以按以下步骤操作：

1. 打开数据表的编辑视图。

2. 在左侧的 "Fields" 面板中找到 "Order Date" 和 "Order Amount" 两个列，然后将它们拖动至 "Rows" 区域。

3. 在 "Marks" 面板中，勾选 "Show All Rows" 以显示所有数据。

4. 点击 "Sort" 标签，进入排序页面。

5. 在 "Sort" 区域，双击 "Order Date" 列，在弹出的下拉菜单中选择 "Ascending (A-Z)" 操作符。

6. 点击 "OK" 按钮确认排序条件。

*Figure 6: Sorting Data by Date* 

除了排序功能外，Tableau Public 还支持多维排序功能。你可以按照多个字段同时进行排序，实现复杂的数据分析。例如，假设有一个数据表 "Orders"，列 "Customer Name" 表示客户姓名，列 "Order Date" 表示订单日期，列 "Order Amount" 表示订单金额。你想按照订单日期、订单金额倒序排列，然后再按照客户姓名升序排序，可以按以下步骤操作：

1. 打开数据表的编辑视图。

2. 在左侧的 "Fields" 面板中找到 "Customer Name"、"Order Date" 和 "Order Amount" 三个列，然后将它们拖动至 "Rows" 区域。

3. 在 "Marks" 面板中，勾选 "Show All Rows" 以显示所有数据。

4. 点击 "Sort" 标签，进入排序页面。

5. 在 "Sort" 区域，双击 "Order Date" 列，在弹出的下拉菜单中选择 "Descending (Z-A)" 操作符。

6. 再次点击 "Sort" 标签，双击 "Order Amount" 列，在弹出的下拉菜单中选择 "Descending (Z-A)" 操作符。

7. 最后，点击 "Sort" 标签下的 "Add another sort" 按钮，再次双击 "Customer Name" 列，在弹出的下拉菜单中选择 "Ascending (A-Z)" 操作符。

8. 点击 "OK" 按钮确认排序条件。

*Figure 7: Multidimensional Sorting* 

## Building Custom Visualizations
Tableau Public 提供了丰富的可视化效果，可以满足各种需求。一般情况下，你不需要了解太多计算机图形学的知识，就可以通过拖放图表、自定义视觉效果等方式快速构建出符合要求的可视化作品。例如，假设有一个数据表 "Orders"，列 "Product Type" 表示产品种类，列 "Quantity" 表示产品数量，列 "Revenue" 表示销售额。你想用柱状图表示各产品种类的销售额占比，可以按以下步骤操作：

1. 打开数据表的编辑视图。

2. 在左侧的 "Fields" 面板中找到 "Product Type"、"Quantity" 和 "Revenue" 三个列，然后将它们拖动至 "Columns" 区域。

3. 在 "Marks" 面板中，点击 "Treemap" 图标，选择 "Bar Chart" 作为可视化效果。

4. 点击 "Options" 标签，切换至 "Color Palette" 选项卡。

5. 从颜色调色板中选择喜欢的配色方案，然后关闭弹窗。

6. 点击 "Treemap" 图标，然后点击 "Size" 旁边的圆圈，选择 "Sum of Quantity"。

7. 最后，点击 "Treemap" 图标，然后点击 "Label" 旁边的圆圈，选择 "Product Type"。

*Figure 8: Building Custom Visualizations with Treemap Chart* 

除了基础的可视化效果外，Tableau Public 提供了许多高级的可视化效果，可以帮助你分析出更加丰富的、有意义的结果。例如，假设有一个数据表 "Orders"，列 "Region" 表示区域，列 "Quantity" 表示产品数量，列 "Profit" 表示利润。你想要探究不同区域的产品数量占比以及销售额、利润的影响，可以按以下步骤操作：

1. 打开数据表的编辑视图。

2. 在左侧的 "Fields" 面板中找到 "Region"、"Quantity"、"Revenue" 和 "Profit" 四个列，然后将它们拖动至 "Columns" 区域。

3. 在 "Marks" 面板中，点击 "Sunburst" 图标，选择 "Clustered Bar Chart" 作为可视化效果。

4. 在 "Trellis Selection" 下拉框中，选择 "Region"。

5. 在 "Details" 面板中，点击 "+ Add Detail" 按钮，选择 "Quantity"。

6. 点击 "Customize Color and Size" 按钮，设置自定义颜色和大小。

7. 设置 "Sum of Profit" 为 Y 轴，设置为 "Product Type" 的聚合值，设置为 "Region" 的细分值。

8. 点击 "Update" 按钮，更新图表。

*Figure 9: Exploring Sales Impact Across Regions with Sunburst Chart* 

为了帮助大家熟悉 Tableau Public 的操作技巧，笔者准备了一份 Tableau Public 用户手册，其中详细阐述了 Tableau Public 的基本概念、术语和操作方法。除了本文所涉及的内容外，还有一些高级主题，如数据连接和配置、地理数据可视化等，还待社区的成员们多多贡献！