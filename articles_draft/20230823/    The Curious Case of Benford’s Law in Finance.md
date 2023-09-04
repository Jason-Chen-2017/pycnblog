
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1什么是“Benford's law”？
Benford's law, named after the first three digits of pi (π), states that in many naturally occurring collections of numbers, leading significant digit(s) tends to appear more frequently than less significant digit(s). The law was proposed by mathematician Charles Benford and is commonly used as a statistical tool for assessing data quality and detecting fraudulent activity.

It can be generalized to other number sequences that exhibit similar patterns but have different distributions or origins. It has been studied in various areas such as finance, economics, and biology, among others.

In this article, we will use Python to explore how Benford's law applies to financial data, specifically trading volumes from NASDAQ stock exchange. We will also present some interesting findings along with potential solutions for improving data quality.


## 1.2关于此文所用数据集

NASDAQ股票交易数据的获取，需要先注册一个账号并购买API key。我这里就不贴出来了，具体流程请参考网络教程或者官方文档。不过可以先通过下面这个链接下载一些NASDAQ数据作为演示用途。

链接: https://pan.baidu.com/s/1hX5yx8dHAzrY9nS_kJjsww 提取码: vhwv 

把文件放到任意文件夹下，然后创建一个名为nasdaq的目录，将这些文件复制进去即可。

nasdaq目录结构如下：

```
├── nasdaq
    ├── AAPL-2021-07-01.csv
    ├── AMZN-2021-07-01.csv
    ├── GOOGL-2021-07-01.csv
    └──...
```

每个文件里存储了一个股票交易日的所有行情信息，包括开盘价、收盘价、最高价、最低价等。每天的数据量都不大，所以可以一次性读取所有文件并进行处理。


# 2.背景介绍

## 2.1为什么要研究NASDAQ股票交易数据？

前面说过，NASDAQ是一个美国最大的上证指数交易所。无论是在股市还是经济界，NASDAQ都是少有的非寡头股市。尤其是在中国经济转型期间，NASDAQ上市公司的交易额占到了总成交额的很大比例，每年都会引起轰动。那么在这种情况下，我们如何评估交易数据呢？

目前股市存在着一项重要的标准，即“对冲基准”。也就是用某种指标来代表整个市场，比如市盈率PE，股息率DIV。通过对比不同对冲基准之间的股票价格走势，可以帮助投资者更好的判断公司表现是否合理，以及规避风险。但是，很多时候，由于各种原因（包括政策调整，战略方向等），无法准确得知股票的真实价值，因此就需要借助交易数据进行分析。

## 2.2NASDAQ股票交易数据中的噪声

交易数据一般来说，都会受到多方面的影响，其中有一类影响比较隐蔽，就是交易噪声。交易噪声的出现往往会造成极大的影响，因为它会使得数据质量受损。在投资领域，对于交易噪声的掌控非常重要。根据相关文献的统计，近几年国内对交易数据的研究主要集中在两个方面——1）市场成交量的分布、模式、变化；2）股票交易频次的变化。而在这两方面，现代交易系统已经做出了较大的改善，但噪声仍然是影响交易数据的关键因素之一。

例如，由于政策的变化、交易系统的完善、风险管理的优化，交易数据呈现出一条双曲线的分布——高频交易、低频交易和平衡态。为了降低交易噪声的影响，降低交易系统的延迟，企业经常提出降低报价水平和取消涨跌停限制等策略，以减少交易数据中的高频交易。

当然，也有一些企业试图通过引入人工神经网络（Artificial Neural Network, ANN）、随机森林（Random Forest）等机器学习算法来识别交易数据中的噪声，但效果并不理想。此外，还有些投资者也纠结于交易数据本身的特征（如时间序列、结构化、季节性等），来评估其价值的有效性。但从另一个角度看，由于数据及时更新，且相互依赖，要精准地预测市场走势，确实十分具有挑战性。

# 3.基本概念术语说明

## 3.1股票交易量

股票交易量是指每笔交易数量，以股（ shares ）为单位，与交易费用无关。它反映了市场对股票供需状况的关注程度。

交易量的大小除了直接影响股票价格外，还会对交易的活跃度产生重大影响，尤其是在股票交易量过大的情况下，买卖意愿可能会明显下降。所以，交易量本身也是市场的一个重要维度。

## 3.2对数、均值、算数平均数

对数：如果把原始数据变换为自然对数后再求均值，就会得到对数均值。假设我们有一组数据{x1, x2,..., xn}，则对数均值$log\overline{x}$等于：

$$ log\overline{x} = \frac{1}{n}\sum_{i=1}^{n}log(x_i) $$

算数平均数：在一个样本空间里，各个元素相加除以该样本空间的个数。

# 4.核心算法原理和具体操作步骤

## 4.1读取股票交易数据

首先，我们需要准备一个用来存放股票交易数据的目录。我们可以使用Python对每个文件的交易数据进行读取，并且存入一个列表中。

```python
import os

file_list = []
for file_name in os.listdir('nasdaq'):
    if '.csv' in file_name:
        file_list.append(os.path.join('nasdaq', file_name))
        
data = {}
for file_name in file_list:
    symbol = file_name.split('/')[-1].replace('.csv', '')
    df = pd.read_csv(file_name)[['Date', 'Volume']]
    df.columns = ['date', 'volume']
    df['symbol'] = symbol
    
    data[symbol] = df
    
all_df = pd.concat([v for k, v in data.items()], ignore_index=True)
```

## 4.2计算每日的交易量对数均值

计算每日的交易量对数均值可以简单地使用pandas库提供的groupby()方法。首先，对交易量列取对数后再进行groupby操作：

```python
all_df['ln_volume'] = np.log(all_df['volume'])
daily_mean_df = all_df[['symbol', 'date', 'ln_volume']].groupby(['symbol', 'date']).mean().reset_index()
```

这样就可以得到每只股票每天的交易量对数均值。

## 4.3计算每日的股票数目占比

类似的，我们也可以计算每日的股票数目占比。这里可以依据日期进行groupby，然后将交易量按正负进行分组，再计算每组交易量的比例。

```python
num_stocks_df = daily_mean_df.groupby('date')['symbol'].count().to_frame(name='num_stocks').reset_index()
num_stocks_df['pos_ratio'] = num_stocks_df['num_stocks']/len(file_list) * 2 - 1 # 正股占比
num_stocks_df['neg_ratio'] = len(file_list)/num_stocks_df['num_stocks'] * 2 - 1 # 负股占比
```

## 4.4绘制曲线图

最后，利用matplotlib库绘制曲线图。我们可以绘制股票数量占比图和交易量对数均值图，并加入点（point）、线（line）、填充（fill_between）图形，来直观显示出各个股票之间的差异。

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

sns.scatterplot(x='date', y='pos_ratio', hue='symbol', alpha=0.8, size='volume', sizes=(10, 100), 
                palette=['blue'], data=num_stocks_df[num_stocks_df['pos_ratio'] > 0], ax=axes[0])
axes[0].set_title('Positive Stocks')
axes[0].axhline(-1/len(file_list)*np.log(1/len(file_list)), color='red', linestyle='--')
axes[0].legend([], [], frameon=False)

sns.scatterplot(x='date', y='neg_ratio', hue='symbol', alpha=0.8, size='volume', sizes=(10, 100), 
                palette=['green'], data=num_stocks_df[num_stocks_df['neg_ratio'] < 0], ax=axes[1])
axes[1].set_title('Negative Stocks')
axes[1].axhline(1/(len(file_list)-1)*np.log((len(file_list)-1)/(len(file_list))), color='black', linestyle='--')
axes[1].legend([], [], frameon=False)

plt.show()
```

上述代码绘制出的结果如下图所示：


左图显示的是正股的数量占比，右边的图显示的是负股的数量占比。

蓝色的散点图表示正股交易量对数均值随时间的变化，红色虚线表示随机游走的对数均值曲线，白色的填充区域表示范围。可以看到，一些股票（如AAPL和GOOGL）的交易量偏离了均值很多，因此在数量上来说占优势；而一些股票（如AMZN和MSFT）的交易量处于均值附近，因此数量上没有明显优势。

绿色的散点图表示负股交易量对数均值随时间的变化，黑色虚线表示反向对数均值曲线（即负的的交易量比例远高于正的交易量比例），白色的填充区域表示范围。可以看到，有几个股票的数量占比远小于其他股票，这可能是因为负股数量的变化更剧烈一些。