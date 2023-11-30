import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.random.seed(123) # 设置随机种子
    data = np.random.randn(1000) # 生成标准正态分布数据
    plt.hist(data, bins=30, density=True, alpha=0.5, color='g') # 创建直方图
    plt.xlabel('Value') # x轴标签
    plt.ylabel('Frequency') # y轴标签
    plt.title('Histogram of Standard Normal Distribution') # 标题
    plt.show() # 显示图表