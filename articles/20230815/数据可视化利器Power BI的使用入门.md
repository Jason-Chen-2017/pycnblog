
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Power BI 是微软推出的一款免费商业智能工具。它是一个基于云的多维分析工具，可以帮助组织快速、直观地处理复杂的数据，并通过直观、易于理解的图表、报告及仪表盘呈现数据。本文将主要围绕 Power BI 的使用入门展开，包括 Power BI 的基础功能、安装配置、使用场景、扩展功能等。

Power BI 作为一个数据可视化工具，其功能远不止于此，涵盖面很广，可以实现各种各样的数据可视化需求。因此，学习如何利用 Power BI 进行数据可视化工作也很重要。

# 2.Power BI 基础功能
Power BI 有以下几个基础功能：

1. 连接到各种数据源，包括关系数据库、Excel 文件、SQL Server、Oracle 和其他多种数据源；
2. 数据模型设计，可以创建多维数据集（Cube）或透视表（Table），然后用预先定义好的维度和指标对其进行分析；
3. 可视化效果设计，可以通过线条、颜色、标记、大小、透明度、图例和标签等方式对数据进行高效的可视化；
4. 报表设计，可以使用 Power BI Desktop 或 Excel 创建丰富的报表；
5. 仪表板设计，可以把不同的数据集组合成单个视图，方便用户查看和管理数据；
6. 发布分享，可以将制作出的 Power BI 内容分享给其他人，或者嵌入到自己提供的 Web 应用中。

以上就是 Power BI 的基本功能，当然还有很多其它的功能，但这些功能基本上覆盖了绝大多数的日常数据可视化需求。

# 3.Power BI 安装配置
Power BI 可以在 Windows 和 Mac 上下载并安装，无需付费。如果需要与服务器集成，则还需要购买许可证。

下面简单介绍下安装过程：

1. 安装 Power BI Desktop：

打开 Microsoft Store，搜索“Power BI Desktop”，找到并安装最新版本的 Power BI Desktop。


2. 配置登录凭据：

首次启动 Power BI Desktop 时，系统会要求输入 Microsoft 账户的用户名和密码。建议创建一个具有足够权限的新 Microsoft 账户来完成登录。


3. 设置数据刷新频率：

在 Power BI Desktop 中，可以在设置页面中设置数据刷新频率，默认为每隔 8 小时刷新一次。一般情况下，越高频的数据刷新速度越快，但同时也可能导致 Power BI 服务出现压力。


4. 配置报表主题：

Power BI Desktop 提供多种预设的报表主题，可以根据自己的喜好来自定义报表外观。点击屏幕右下角齿轮图标 -> “选项” -> “主题” 来设置主题样式。


至此，Power BI Desktop 的安装配置工作已经完成。

# 4.Power BI 使用场景

## 4.1 数据导入

Power BI 可以连接到各种数据源，包括关系数据库、Excel 文件、SQL Server、Oracle 和其他多种数据源。下面介绍如何导入数据：

1. 在 Power BI Desktop 的导航窗格中选择左侧的“获取数据”。


2. 从菜单中选择要导入数据的类型，例如“Excel”，然后选择文件路径。


3. 如果所选文件存在多个工作簿或表格，可选择导入哪些表格或工作簿。


4. 将所需字段拖动到“字段”列表中。


5. 如果需要导入整个文件而不是某个特定表格，请勾选“包括文件中的所有内容”。


6. 当所有设置都完成后，单击“加载”按钮以开始导入。


完成数据导入后，就可以在报表编辑器中开始进行可视化设计。

## 4.2 数据模型设计

在 Power BI Desktop 中，可以创建多维数据集（Cube）或透视表（Table）。数据模型设计可以用预先定义好的维度和指标对数据进行分析。

### 创建 Cube

1. 在报表编辑器的导航窗格中，选择左侧的“数据模型”图标。


2. 选择“新建” -> “多维数据集”或直接按“Alt+Q”键打开快速创造向导。


3. 在向导的第一步“选择数据”，指定要分析的数据源，从而开始数据模型设计。


4. 在向导的第二步“指定关系”，可以手动添加或删除表之间的关联。


5. 在向导的第三步“完成”，可以确认设置是否正确，然后保存数据模型。


完成数据模型设计后，就可以在报表编辑器中开始创建可视化对象。

### 创建 Table

1. 进入数据模型设计画布。


2. 单击右侧的“插入” -> “新表”。


3. 在“查询编辑器”中编写 SQL 查询语句。


4. 执行查询并查看结果。


5. 在“字段”列表中，将查询结果映射到新的表。


6. 根据需要调整列名、格式、排序顺序等属性。


完成 Table 的创建后，就可以在报表编辑器中开始进行可视化设计。

## 4.3 可视化效果设计

在 Power BI Desktop 中，可以通过线条、颜色、标记、大小、透明度、图例和标签等方式对数据进行高效的可视化。

### 可视化类型

Power BI 支持以下几种可视化效果：

1. 折线图：用于显示一段时间内的数据变化趋势；
2. 柱状图：用于显示分类变量随着分类指标变化的程度；
3. 饼图：用于显示分类变量的占比情况；
4. KPI 卡片：用于显示关键业务指标的实时状态；
5. 散点图：用于显示两个变量之间的关系；
6. 直方图：用于显示连续分布数据；
7. 箱型图：用于显示多个分类变量的分布情况。

这里，我们以折线图和柱状图为例，来演示如何进行可视化效果设计。

### 创建折线图

1. 进入可视化画布。


2. 拖动“折线图”控件到报表编辑器的编辑区域。


3. 在“字段”列表中，找到待可视化的字段，然后将它们拖动到“轴”、“值”、“组别”、“标记”、“颜色”、“大小”栏目中。


4. 根据需要调整图表外观。


完成折线图的创建后，即可在报表中查看。

### 创建柱状图

1. 点击右侧的“插入” -> “新可视化效果”。


2. 在“可视化效果”窗格中选择“柱形图”，然后将图表拖动到编辑区域。


3. 在“字段”列表中，找到待可视化的字段，然后将它们拖动到“值”、“轴”、“颜色”、“分组”栏目中。


4. 根据需要调整图表外观。


完成柱状图的创建后，即可在报表中查看。

## 4.4 报表设计

Power BI Desktop 支持丰富的报表设计功能，可以帮助用户创建各种类型的报表，例如日常报告、销售报告、财务报表、策略报表等。下面，我们以创建一个日常报告为例，来演示如何进行报表设计。

1. 在报表编辑器的顶部，单击“新增页签”按钮或按“Ctrl+T”创建新页面。


2. 在新页面上，在“插入”选项卡中选择“文本框”，并将其拉伸到适当位置。


3. 在文本框中输入文字并调整字体、颜色等属性。


4. 在页面的中心位置，拖动“折线图”控件，并调整图表属性。


5. 添加更多图表，重复步骤 4。


6. 调整报表布局，使其更加美观。


完成报表设计后，即可保存并发布报表。

# 5.Power BI 扩展功能

除了 Power BI 本身的基础功能和一些定制化功能，比如定时刷新、邮件订阅等，Power BI 还提供了丰富的扩展插件和 API。

下面介绍几个典型的扩展功能，如通过 REST API 获取数据、创建自定义运算符、开发自定义计算引擎等。

## 通过 REST API 获取数据

Power BI REST API 提供了一种允许外部服务访问 Power BI 数据的方式，方便实现第三方应用对数据进行读取、转换和可视化。

为了使用 REST API，首先需要获得 OAuth 身份验证令牌。你可以在 Power BI Desktop 的设置页面中，找到“开发人员资源”部分下的“获取令牌”按钮。


然后，使用第三方 HTTP 客户端（如 Postman 或 Fiddler）调用 API 并传入相应的参数。


## 创建自定义运算符

Power BI 的运算符库提供了丰富的内置运算符，可以满足绝大多数日常的数据处理需求。但是，仍然有些时候需要一些特殊的运算符支持。

Power BI 提供了两种方式来创建自定义运算符：

1. PQV (Power Query Visualisation) 插件：这是最简单的扩展方式，只需编写 JavaScript 函数代码，即可创建一个新的运算符。

2. M 语言运算符表达式：这种方式相对复杂一些，需要熟悉 M 语言，并且对 DAX（Data Analysis Expressions，Power BI 的数据分析表达式语言）非常了解。

下面，我们以创建自定义 PQV 为例，演示如何扩展运算符库。

### 示例：创建自定义求和运算符

1. 从 GitHub 下载官方示例项目：https://github.com/Microsoft/powerbi-visuals-samples

2. 启动 Visual Studio Code，打开“sum-visual\src\settings.ts”文件。


3. 修改 sum.ts 文件的代码如下：

   ```javascript
   "use strict";
   
   import powerbi from "powerbi-visuals-api";
   import DataView = powerbi.DataView;
   import DataViewCategorical = powerbi.DataViewCategorical;
   import DataViewValueColumn = powerbi.DataViewValueColumn;
   
   export function getSum(data: DataView): number {
       let total: number = 0;
       
       if (!data ||!data.categorical ||!data.categorical.categories ||!data.categorical.values) {
           return null;
       }
       
       const categories: DataViewCategorical[] = data.categorical.categories;
       const values: DataViewValueColumn[] = data.categorical.values;
       
       for (let i = 0, len = Math.max(categories[0].values.length, values[0].values.length); i < len; i++) {
           const categoryIndex: number = categories[0]? categories[0].values[i] : undefined;
           const valueIndex: number = values[0]? values[0].values[i] : undefined;
           
           if (categoryIndex!== undefined && valueIndex!== undefined) {
               // check that both index exist and are not empty strings or null
               if ((typeof categoryIndex ==='string' && categoryIndex.trim()!== '')
                   || typeof categoryIndex === 'number') {
                   
                   total += parseFloat(<any>valueIndex);
               
               } else {
                   console.log('Invalid category');
               }
           } else {
               console.log(`Category at index ${i} does not have a corresponding value`);
           }
       }
       
       return total;
   };
   
   ```

4. 将修改后的文件重新保存，然后回到 Visual Studio Code，在终端窗口中运行“npm install”命令安装依赖项。


5. 构建成功后，将生成的 dist 文件夹中的内容复制到“sum-visual\dist”目录下。


6. 返回 Power BI Desktop，在导航窗格中选择左侧的“扩展”，然后选择“导入本地扩展”按钮。


7. 选择刚才构建好的“sum-visual”文件夹，然后单击“确定”按钮。


8. 导入成功后，在导航窗格中选择左侧的“运算符”，然后选择“导入本地”按钮。


9. 浏览并选择刚才构建好的“sum-visual”文件夹，然后单击“确定”按钮。


10. 导入成功后，你应该可以看到新的“求和”运算符。


11. 你可以继续测试并调试你的运算符，确保它们按照预期工作。