import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':


    np.random.seed(123) # 设置随机种子
    x = np.random.rand(100) * 4 - 2 # 在[-2, 2]区间内生成100个随机数
    y = (x ** 2 + np.random.randn(100) * 0.5).clip(-2, 2) # 以x^2+随机数的噪声为目标函数计算对应的值

    plt.scatter(x, y, c='r', marker='+') # 设置红色圆点，大小为+号
    plt.xlabel('Variable X') # x轴标签
    plt.ylabel('Variable Y') # y轴标签
    plt.title('Scatter Plot of Variable X and Y') # 标题
    plt.show() # 显示图表