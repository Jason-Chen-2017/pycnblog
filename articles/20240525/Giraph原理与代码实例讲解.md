Giraph 是一个开源的分布式大规模图计算系统，主要用于解决一些图形计算的问题，比如社交网络分析、图像识别、物联网、智能城市等。Giraph 提供了一个简单的编程模型，用户可以通过编写 MapReduce 函数来解决图计算问题。Giraph 的架构设计上，采用了 Master-Slave 模式，Master 负责分配任务给 Slave，Slave 负责执行任务。

Giraph 的核心组件有以下几个：

1. Master：Master 负责将图数据划分为多个小图，分别分配给 Slave 进行计算，然后将结果汇总到 Master 上。

2. Slave：Slave 负责执行 MapReduce 函数，对图数据进行处理并生成中间结果。

3. Graph：Graph 是 Giraph 中存储图数据的数据结构，包括顶点和边。

Giraph 的编程模型非常简单，用户只需要编写 MapReduce 函数来解决问题。MapReduce 函数分为两种，一种是 Map 函数，用于对图数据进行分割和处理，另一种是 Reduce 函数，用于对 Map 函数的结果进行汇总和合并。

下面是一个 Giraph 代码实例，用于计算图中每个顶点的度数：

```java
public class DegreeCalculator extends Computation {
    @Override
    public void map(Object key, int value, int context) {
        int degree = 0;
        // 对于每个顶点，遍历其所有邻接点，增加度数
        for (int i = 0; i < value; i++) {
            degree++;
        }
        context.write(key, degree);
    }

    @Override
    public void reduce(Object key, Iterable<int[]> values, int context) {
        // 对于每个顶点的度数求和
        int degreeSum = 0;
        for (int[] value : values) {
            degreeSum += value[1];
        }
        context.write(key, degreeSum);
    }
}
```

在这个例子中，我们编写了一个 DegreeCalculator 类，该类继承自 Giraph 的 Computation 类。我们实现了 map 和 reduce 函数，用于计算图中每个顶点的度数。map 函数遍历每个顶点的所有邻接点，增加度数；reduce 函数对每个顶点的度数求和。最后，我们将结果写入到 context 上。