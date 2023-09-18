
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Seaborn是一个数据可视化库，可以用来快速创建高质量的统计图形，支持多种形式的数据表示，包括散点图、分布图、线性回归图、热力图、相关性矩阵等，并内置了一些用于绘制主题化的基础样式。
## 安装
Seaborn可以通过pip进行安装：
```python
!pip install seaborn
```
如果没有pip环境，也可以下载安装包手动安装。
## 使用示例
首先，导入Seaborn模块：
```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```
接下来，我们用Seaborn画一个随机生成的散点图：
```python
sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
ax = sns.scatterplot(x= "total_bill", y= "tip", data= tips)
plt.title('Total bill vs Tip')
plt.show()
```
这样就成功地画出了散点图。
## API介绍
### 设置样式
Seaborn允许我们设置很多默认的样式，比如使用灰色网格作为默认风格，或者使用白底黑字的颜色对比来呈现数据。我们可以使用set()函数设置各种风格：
```python
sns.set(style="darkgrid") # 设置灰色网格
sns.set(style="whitegrid") # 设置白色网格
sns.set(style="ticks") # 设置棒子样式
```
另外，还可以使用context参数来设置不同场景下的风格，比如notebook环境下使用notebook主题的风格，lab环境下使用poster主题的风格：
```python
sns.set(context='notebook', style='darkgrid')
```
### 数据集加载器
Seaborn提供了一个方便的数据集加载器来快速获取示例数据集，通过load_dataset()函数即可加载：
```python
iris = sns.load_dataset("iris")
tips = sns.load_dataset("tips")
flights = sns.load_dataset("flights")
titanic = sns.load_dataset("titanic")
```
### 创建图表对象
Seaborn提供了多个创建图表对象的函数，如scatterplot(), barplot(), countplot()等。每个函数都可以接收dataframe类型的数据，返回一个图形对象，我们可以使用set()函数设置不同的属性。举个例子，使用scatterplot()函数创建一个散点图：
```python
g = sns.scatterplot(data=tips, x= 'total_bill', y= 'tip')
g.set(xlabel='Total Bill ($)', ylabel='Tip ($)')
g.set_title('Total bill vs Tip')
```
### 设置样式属性
除了使用set()函数设置样式外，还可以直接在创建图表对象时传入属性设置，如g = sns.scatterplot(data=tips, x= 'total_bill', y= 'tip', hue='sex', size='size').
### 添加标注
Seaborn允许我们轻松添加标注，包括标示轴标签、标题、斜线、注释等。
#### 轴标签
我们可以使用xlabel和ylabel函数设置轴标签：
```python
g.set(xlabel='Total Bill ($)', ylabel='Tip ($)')
```
#### 标题
我们可以使用set_title()函数设置标题：
```python
g.set_title('Total bill vs Tip')
```
#### 折线
我们可以使用axhline()函数绘制一条水平线：
```python
g = sns.scatterplot(data=tips, x= 'total_bill', y= 'tip', hue='sex', s=75)
g.set(xlabel='Total Bill ($)', ylabel='Tip ($)')
g.set_title('Total bill vs Tip')
sns.despine() # 删除边框
g.axhline(y=tips['tip'].mean()) # 添加一条平均值的水平直线
```
#### 斜线
我们可以使用annotate()函数添加斜线标注：
```python
g = sns.scatterplot(data=tips, x= 'total_bill', y= 'tip', hue='sex', s=75)
g.set(xlabel='Total Bill ($)', ylabel='Tip ($)')
g.set_title('Total bill vs Tip')
sns.despine()
g.axhline(y=tips['tip'].mean())
g.annotate('$\\bar{y}=$'+str(round(tips['tip'].mean(), 2)), xy=(60, 20), color='red') # 添加斜线标注
```
### 分类变量
Seaborn允许我们通过hue参数对不同类别的样本进行区分，比如用颜色区分男女样本：
```python
sns.catplot(x="day", y="total_bill", hue="sex", kind="violin", split=True, data=tips)
```
上面的代码用 violin 描述了不同天气的总账单分布，通过颜色区分了男女样本。
### 连续变量
Seaborn允许我们绘制散点图、线性回归图、分布图等对连续型变量进行分析。
#### 分布图
我们可以使用displot()函数绘制分布图，如：
```python
sns.distplot(a=tips["total_bill"], kde=False)
```
上面的代码绘制了销售总账单的分布图，kde=False 表示不显示核密度估计曲线。
#### 线性回归图
我们可以使用lmplot()函数绘制线性回归图，如：
```python
sns.lmplot(x="total_bill", y="tip", data=tips)
```
上面的代码绘制了销售总账单与小费之间的线性回归关系图。