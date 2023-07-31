
作者：禅与计算机程序设计艺术                    
                
                
数据报表制作是数据分析中必不可少的一环。目前市面上有很多商业智能工具可用于报表的设计和制作。其中，Tableau是一个非常优秀的数据报表制作工具，其功能强大、界面简洁、拖拽式操作等特点，已经成为业界主流的数据报表制作工具之一。作为一个用户认可度较高的商业智能工具，它的使用门槛也比较低。但要想充分掌握它并应用到实际工作当中，还是需要有相当丰富的实践经验的。本文将从零开始，带领大家快速入门Tableau。
# 2.基本概念术语说明
## 数据集（Data Set）
在Tableau中，数据集指的是存放数据的实体，通常是一个单独的文件或者数据库中的一组数据。数据集的结构由字段、行以及数据值组成。
![image.png](attachment:image.png)

## 视觉元素（Marks）
视觉元素是指用来呈现数据的图形符号、线条、颜色、大小等方面的元素。每个视觉元素都可以设置相关的参数属性来调整其外观和显示效果。Tableau提供多种类型的视觉元素，如直方图、散点图、折线图、堆积柱状图、地图等。
![image-2.png](attachment:image-2.png)

## 交互（Interaction）
交互是指与用户进行交流的方式，包括鼠标点击、悬停、拖动、缩放、筛选、聚合等。通过交互，用户可以对数据进行筛选、排序、导航、理解等操作。

## 滚动条（ScrollBars）
滚动条是指在视觉元素中出现的一种控制视图范围的组件。通过滚动条，用户可以自由地查看数据分布的全貌。Tableau提供了两种类型的滚动条，水平滚动条和垂直滚动条。

## 计算字段（Calculated Field）
计算字段是指根据其他字段的值进行计算得到新字段的值。Tableau支持基于任何数据的计算字段。

## 样式（Styling）
样式是指影响视觉元素外观和显示方式的属性，如字体、颜色、大小等。Tableau提供了丰富的内置样式库，用户也可以自行创建自定义样式。

## 主题（Theme）
主题是指不同类型的图表、图形、交互、文本等元素的配色方案。Tableau提供多套不同的主题，让用户能够更容易地找到自己喜欢的风格。

## 过滤器（Filter）
过滤器是指用来选择特定数据子集的规则。用户可以通过过滤器隐藏不需要的、不相关的、重复的数据。

## 切片器（Slicers）
切片器是指用来按特定维度划分数据子集的组件。切片器分为离散型和连续型两种类型。离散型切片器仅支持单个值，而连续型切片器则支持一段范围内的取值。

## 联轴（Linked Axes）
联轴是指两个或多个图表上的坐标轴共享相同的刻度范围及轴标签。联轴有助于比较各项指标之间的差异。

## 注释（Annotation）
注释是指用来添加辅助信息、说明文字、图像、链接等的组件。注释的作用主要是为了提升报告的易读性。

## 发布（Publish）
发布是指将制作好的报表和数据集分享给其他用户、团队、部门和企业的过程。发布完成后，就可以看到其他人的见解了。

## 发布选项（Publish Options）
发布选项是指用来设置一些发布参数、版本信息、权限管理等的组件。比如，可以设定刷新频率、电子邮件订阅、导出数据集、授权访问等。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 合并数据集
假设有一个产品销售数据集和订单数据集，需要将两个数据集合并为一个数据集才能做出销售产品的数据报表。首先打开两个数据集文件，然后将两张表合并。

![image-3.png](attachment:image-3.png)

点击“编辑关系”按钮，然后选择“连接表”。此时会弹出窗口询问用户是否要用标识符建立联系。选择“是”即可。然后设置唯一标识符为一个字段名，保存更改。

![image-4.png](attachment:image-4.png)

接下来，双击新建的数据集文件，就会看到两个表已经合并到了一起。

![image-5.png](attachment:image-5.png)

## 创建数据透视表
在数据集中创建数据透视表的方法如下：

1. 从菜单栏中依次选择“插入”->“表”，创建一个新的表。
2. 在左侧工具栏中选择“数据透视表”。
3. 在“层次”面板中选中想要展示的数据和对比维度。如果有日期作为层级，可以在“聚合”面板中选择合适的时间粒度。
4. 将想要展示的数据源设置为刚才合并的数据集。
5. 如果有多个维度，可以在“层次”面板中再增加一个层级。
6. 修改默认的行和列的排列顺序。
7. 在“数据”面板中选择要显示的数据，可以按照指定条件过滤。
8. 在“格式”面板中修改行列的颜色和文字大小。
9. 在“布局”面板中微调排版。

![image-6.png](attachment:image-6.png)

## 使用变量和度量值创建细化维度
数据透视表默认显示每一个层级的所有可能的组合，但是某些时候可能会希望细化某些层级。例如，在订单数据透视表中，如果只想看最热门的商品，就可以使用“商品名称”变量和“数量”度量值进行细化。

1. 打开订单数据透视表。
2. 在“层次”面板中选择想要细化的维度——“商品名称”。
3. 在“计算”面板中添加一个新的变量——“最大销量”。
4. 设置计算公式为“SUM(数量)”，然后将该变量设置为度量值。
5. 根据“最大销量”降序排列商品名称。
6. 右键单击某个商品，可以看到该商品的详细信息。

![image-7.png](attachment:image-7.png)

## 使用条件格式设置突出显示
条件格式是指将单元格的内容以特定的颜色和效果显示，以突出重点内容。Tableau允许用户基于一组条件设置单元格的格式，如颜色、对齐方式、数字格式等。

1. 打开订单数据透视表。
2. 在“数据”面板中选择“销售额”。
3. 在“格式”面板中选择“条件格式”。
4. 点击“创建规则”。
5. 输入规则名称，选择突出显示的条件——“>=200万元”。
6. 选择突出显示的效果——“填充”和“红色”。
7. 设置数值格式为“0.00亿元”。
8. 在“完善度”面板中选择想要显示的规则范围。

![image-8.png](attachment:image-8.png)

## 用堆积柱状图显示变化
在同样的数据集中，有一个销售产品的销售总量和年份数据，需要用堆积柱状图展示销售总量随着时间的变化。

1. 打开销售产品的数据集。
2. 选择“堆积柱状图”。
3. 把销售总量放在“x轴”、“y轴”两个维度上。
4. 添加一个“年份”维度，把销售总量放在这个维度上。
5. 把“年份”维度放在“分类轴”位置。
6. 调整“颜色”属性。
7. 调整堆积的方向。

![image-9.png](attachment:image-9.png)

# 4.具体代码实例和解释说明
## 生成一个含有颜色、大小、符号的散点图
```python
import random

data = [{"x":random.randint(1,10), "y":random.randint(1,10)} for i in range(10)] # 生成随机数据
marks = [] # 初始化视觉元素列表
for item in data:
    mark_dict = {"type":"circle",
                 "color":{'palette': 'tol',
                          'brightness': -0.5},
                 "size":item["x"]*2+item["y"],
                 "opacity":0.8}
    marks.append(mark_dict)
    
# 利用 Tableau Python API 生成散点图
from tableauscraper import TableauScraper as TS

ts = TS()
url = "" # 填写自己的URL
workbook = ts.getWorkbook(url=url)
worksheet = workbook.getWorksheet("Sheet1")
worksheet.clearAllFilters() # 清除所有筛选器
datasource = worksheet.datasources[0] # 获取数据集
worksheet.select() # 选中当前工作表
viz_page = ts.VizPage(viz="scatter plot")
viz_page.addDataSource(datasource) 
viz_page.setMarks(marks)
viz_page.show('scatter.html')
```

生成的散点图如下所示：

![image-10.png](attachment:image-10.png)

上述代码的逻辑是：

1. 生成10个随机数据点。
2. 为每一个数据点生成一个字典，表示对应的视觉元素的属性。
3. 对每一个视觉元素设置属性——颜色（使用tol色板），大小（x和y轴的值加起来乘以2），不透明度（0.8）。
4. 通过 Tableau Python API 创建一个 VizPage 对象，并设置数据源、视觉元素列表和输出文件名。
5. 调用 show 方法生成一个 HTML 文件。

## 用 Python 实现简单的机器学习模型训练
本例使用简单线性回归模型，拟合一个简单二维数据集。

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def generate_dataset():
    x = np.array([[-2], [0], [2]])
    y = np.dot(-2 * x + 1, 1/np.exp(abs((x - 2))))
    return x, y

if __name__ == '__main__':
    X, y = generate_dataset()

    reg = LinearRegression().fit(X, y)
    
    print('w = ', round(reg.coef_[0][0], 2))
    print('b = ', round(reg.intercept_, 2))
    
    xx = np.arange(-5, 5,.1).reshape((-1, 1))
    yy = reg.predict(xx)
    
    plt.plot(xx, yy, color='red', label='Predicted Line')
    plt.scatter(X, y, s=30, c='blue', marker='o', edgecolors='black', label='Dataset')
    plt.legend()
    plt.title('Simple Linear Regression')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()
```

运行上述代码可以训练出 w 和 b 参数，并画出拟合曲线。

