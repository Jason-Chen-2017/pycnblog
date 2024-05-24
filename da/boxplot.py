import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(123) # 设置随机种子
    data = [np.random.normal(0, std, 100) for std in range(1, 4)] # 生成三个正态分布数据
    plt.boxplot(data, labels=['1', '2', '3']) # 使用boxplot()函数绘制箱线图，并设置标签
    plt.show() # 显示图表