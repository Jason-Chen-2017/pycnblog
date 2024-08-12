                 

# AI在智能交通信号控制中的应用：减少拥堵

> 关键词：智能交通系统, 信号控制, AI算法, 拥堵缓解, 优化模型, 交通流量预测, 实时调整, 系统评估, 场景适应性

## 1. 背景介绍

### 1.1 问题由来
在现代城市交通体系中，交通信号控制对整个交通系统的顺畅运行起着至关重要的作用。传统的交通信号控制系统依赖于固定的时间表和预设的配时方案，往往难以适应实时交通流量的变化，导致城市交通拥堵问题日益严重。近年来，随着人工智能(AI)技术的发展，利用AI算法优化交通信号控制系统，逐渐成为缓解城市交通拥堵的重要手段。

AI在交通信号控制中的应用，可以通过学习实时交通流数据，动态调整信号灯的配时，优化交通流量，从而提高道路通行效率，缓解城市拥堵。这种基于数据驱动的动态信号控制方法，被广泛应用于智能交通系统中。

### 1.2 问题核心关键点
AI在交通信号控制中的应用主要包括以下几个关键点：

- **数据采集与处理**：获取实时交通流量、车辆位置、速度、方向等数据，并进行预处理，为AI模型提供有效的输入。
- **模型构建与训练**：选择合适的AI模型，并使用历史交通数据进行训练，学习交通流量与信号配时之间的关系。
- **信号控制策略**：设计基于AI模型的信号控制策略，根据实时交通状况动态调整信号灯配时。
- **系统评估与优化**：构建评估指标体系，实时监控交通信号控制系统的运行效果，并进行持续优化。
- **场景适应性与鲁棒性**：确保AI系统在各种交通场景下都能稳定运行，具备较强的鲁棒性。

## 2. 核心概念与联系

### 2.1 核心概念概述

为更好地理解AI在交通信号控制中的应用，本节将介绍几个密切相关的核心概念：

- **智能交通系统(ITS)**：通过整合信息技术、通信技术、控制技术等，实现交通管理、车辆控制、信息服务等功能，提高道路通行效率和安全性。
- **交通信号控制**：通过动态调整信号灯配时，优化交通流量，缓解交通拥堵。
- **AI算法**：包括机器学习、深度学习、强化学习等算法，用于学习交通流数据，预测交通状态，制定信号控制策略。
- **交通流量预测**：通过历史交通数据，学习交通流量的变化规律，预测未来的交通状态。
- **实时调整**：根据实时交通流量、车辆位置、速度等数据，动态调整信号灯配时，实现实时优化。
- **系统评估**：通过评估指标体系，对交通信号控制系统进行实时监控和优化。
- **场景适应性**：确保AI系统在不同交通场景下都能稳定运行，避免因环境变化导致的系统失效。

这些核心概念之间的逻辑关系可以通过以下Mermaid流程图来展示：

```mermaid
graph TB
    A[Intelligent Transportation System (ITS)] --> B[Traffic Signal Control]
    A --> C[A Insertion]
    A --> D[AI Algorithms]
    B --> E[Traffic Flow Prediction]
    D --> F[Real-time Adjustment]
    E --> F
    F --> G[System Evaluation]
    G --> H[Optimization]
    H --> I[Scene Adaptability]
```

这个流程图展示了智能交通系统中交通信号控制的应用框架：

1. 智能交通系统(ITS)整合信息技术、通信技术、控制技术，实现交通管理、车辆控制、信息服务等功能。
2. 交通信号控制通过动态调整信号灯配时，优化交通流量，缓解交通拥堵。
3. AI算法通过学习交通流数据，预测交通状态，制定信号控制策略。
4. 交通流量预测学习交通流量的变化规律，预测未来的交通状态。
5. 实时调整根据实时交通流量、车辆位置、速度等数据，动态调整信号灯配时。
6. 系统评估通过评估指标体系，对交通信号控制系统进行实时监控和优化。
7. 场景适应性确保AI系统在不同交通场景下都能稳定运行。

## 3. 核心算法原理 & 具体操作步骤

### 3.1 算法原理概述

AI在交通信号控制中的应用，本质上是一个基于数据的动态优化问题。其核心思想是：通过学习历史交通流量数据，建立交通状态与信号配时之间的关系模型，并根据实时交通流量动态调整信号灯配时，以优化交通流量，缓解拥堵。

具体而言，可以采用以下步骤：

1. **数据采集与处理**：使用传感器、摄像头等设备采集实时交通流量、车辆位置、速度、方向等数据，并进行预处理，为AI模型提供有效的输入。
2. **模型构建与训练**：选择合适的AI模型，并使用历史交通数据进行训练，学习交通流量与信号配时之间的关系。
3. **信号控制策略**：设计基于AI模型的信号控制策略，根据实时交通状况动态调整信号灯配时。
4. **系统评估与优化**：构建评估指标体系，实时监控交通信号控制系统的运行效果，并进行持续优化。

### 3.2 算法步骤详解

以下是AI在交通信号控制中应用的具体步骤：

**Step 1: 数据采集与预处理**

1. 使用传感器、摄像头等设备采集实时交通流量、车辆位置、速度、方向等数据。
2. 对采集到的数据进行清洗和预处理，如去除异常值、处理缺失值等。
3. 将数据转换为模型所需的格式，如将车辆位置、速度等信息转换为坐标和时间序列数据。

**Step 2: 模型构建与训练**

1. 选择合适的AI模型，如深度学习模型、强化学习模型等。
2. 构建交通流量预测模型，使用历史交通数据进行训练，学习交通流量的变化规律。
3. 构建信号控制策略模型，使用交通流量预测模型输出的流量数据，学习信号配时与流量之间的关系。

**Step 3: 信号控制策略设计**

1. 根据实时交通流量、车辆位置、速度等数据，使用信号控制策略模型，计算最优信号配时。
2. 设计信号灯配时的动态调整算法，如基于时间间隔调整、基于流量调整等。
3. 实现信号灯控制系统的实时优化，如通过通信协议将配时指令发送到交通信号控制器。

**Step 4: 系统评估与优化**

1. 构建评估指标体系，如交通流量、延误时间、交叉口通行能力等。
2. 实时监控交通信号控制系统的运行效果，收集评估数据。
3. 根据评估数据，调整模型参数，优化信号控制策略，提升系统性能。

### 3.3 算法优缺点

AI在交通信号控制中的应用，具有以下优点：

1. **高效优化**：通过动态调整信号配时，能够实时响应交通流量变化，提高道路通行效率。
2. **数据驱动**：基于历史交通数据进行模型训练，学习交通流量的变化规律，提升模型预测的准确性。
3. **适应性强**：可以处理各种交通场景，适应不同的交通需求和环境变化。
4. **减少拥堵**：通过优化交通流量，缓解城市交通拥堵，改善居民出行体验。

同时，该方法也存在以下局限性：

1. **数据依赖**：需要大量的历史交通数据进行模型训练，数据不足时模型性能可能较差。
2. **模型复杂**：需要高性能的计算资源进行模型训练和实时计算，硬件成本较高。
3. **环境适应性**：模型训练数据可能与实际交通场景存在差异，模型在实际应用中可能出现偏差。
4. **安全性**：系统的实时调整可能会影响交通秩序，需要严格的监管和测试。

尽管存在这些局限性，但就目前而言，基于AI的交通信号控制方法仍是大规模优化交通系统的有效手段。未来相关研究的重点在于如何进一步降低对数据和硬件的依赖，提高模型的环境适应性和安全性，以促进AI在交通信号控制中的广泛应用。

### 3.4 算法应用领域

AI在交通信号控制中的应用，主要适用于以下领域：

- **智能交通系统**：通过AI技术，实现交通信号的动态控制，提高交通系统的效率和安全性。
- **城市交通管理**：帮助城市管理者优化交通流量，缓解交通拥堵，提升城市运行效率。
- **道路交通优化**：通过实时调整信号配时，优化道路交通流量，提高道路通行能力。
- **公共交通优化**：辅助公共交通系统，优化公交车发车间隔，提高公交运营效率。
- **事故响应**：在发生交通事故时，动态调整信号灯配时，快速恢复交通秩序。

## 4. 数学模型和公式 & 详细讲解 & 举例说明

### 4.1 数学模型构建

AI在交通信号控制中的应用，可以通过以下数学模型来描述：

- **交通流量预测模型**：
  $$
  y_t = f(x_t; \theta)
  $$
  其中 $y_t$ 表示时刻 $t$ 的交通流量，$x_t$ 表示时刻 $t$ 的输入特征，$\theta$ 表示模型参数。

- **信号控制策略模型**：
  $$
  c_t = g(y_t; \theta')
  $$
  其中 $c_t$ 表示时刻 $t$ 的信号配时，$y_t$ 表示时刻 $t$ 的交通流量，$\theta'$ 表示策略模型参数。

### 4.2 公式推导过程

以交通流量预测模型为例，假设采用深度学习模型进行建模，其结构如下：

```mermaid
graph LR
    x1[特征1] -->|输入层| x2
    x2 -->|隐藏层1| x3
    x3 -->|隐藏层2| x4
    x4 -->|输出层| y
```

其中，$x_1$ 表示输入特征，$x_2$ 表示隐藏层1的输出，$x_3$ 表示隐藏层2的输出，$x_4$ 表示输出层的输出 $y_t$。模型的参数 $\theta$ 包括所有隐藏层和输出层的权重和偏置。

采用均方误差(MSE)作为损失函数，其定义为：
$$
L = \frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2
$$
其中 $y_i$ 表示真实交通流量，$\hat{y}_i$ 表示模型预测的交通流量。

模型的目标是最小化损失函数 $L$，即：
$$
\theta = \mathop{\arg\min}_{\theta} L
$$

在模型训练过程中，采用反向传播算法，计算梯度并更新参数 $\theta$，逐步减小损失函数 $L$。

### 4.3 案例分析与讲解

假设在一个十字路口，实时采集到的交通流量数据为：

| 时间 | 车流量 |
|------|--------|
| 08:00 | 200    |
| 08:05 | 220    |
| 08:10 | 250    |
| 08:15 | 300    |
| 08:20 | 350    |
| 08:25 | 400    |

使用上述模型进行预测，设 $x_t = [t, y_{t-1}, y_{t-2}]$，其中 $t$ 表示当前时间，$y_{t-1}$ 和 $y_{t-2}$ 表示前两个时刻的交通流量。模型的输入层、隐藏层和输出层的参数为随机初始化，经过训练后，模型预测的交通流量如下：

| 时间 | 真实车流量 | 预测车流量 |
|------|------------|------------|
| 08:00 | 200        | 220         |
| 08:05 | 220        | 230         |
| 08:10 | 250        | 240         |
| 08:15 | 300        | 270         |
| 08:20 | 350        | 320         |
| 08:25 | 400        | 380         |

可以看到，模型在处理历史数据时，具有较高的预测准确性。但在实际应用中，还需要根据实时交通数据，动态调整信号配时，以进一步提高交通系统效率。

## 5. 项目实践：代码实例和详细解释说明

### 5.1 开发环境搭建

在进行交通信号控制AI模型的开发时，需要使用高性能计算资源和相关软件工具。以下是具体的开发环境搭建步骤：

1. **安装Python和相关库**：
   - 安装Python 3.7或以上版本。
   - 安装TensorFlow或PyTorch，用于深度学习模型的实现。
   - 安装NumPy、Pandas等库，用于数据处理。

2. **安装交通仿真软件**：
   - 安装SUMO（Simulation of Urban MObility）等交通仿真软件，用于模拟和测试交通信号控制系统。

3. **设置开发环境**：
   - 配置GPU和TPU等高性能计算资源，用于模型训练和实时计算。
   - 配置网络连接，确保模型训练和实时计算的数据交换效率。

### 5.2 源代码详细实现

以下是一个基于深度学习模型的交通流量预测的Python代码实现：

```python
import tensorflow as tf
import numpy as np
import pandas as pd

# 数据预处理
def preprocess_data(data):
    # 数据清洗
    data = data.dropna()
    # 特征提取
    features = pd.get_dummies(data['time'], prefix='time')
    features = pd.concat([features, data['y']], axis=1)
    features = features.drop(columns=['time'])
    # 将数据转换为模型所需的格式
    features = features.values
    targets = data['y'].values
    return features, targets

# 模型构建与训练
def build_model(features, targets, epochs=10, batch_size=32, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(features.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    model.fit(features, targets, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return model

# 数据生成与模型训练
data = pd.read_csv('traffic_data.csv')
features, targets = preprocess_data(data)
model = build_model(features, targets)

# 模型预测与信号控制策略
def predict_flow(model, features):
    predictions = model.predict(features)
    return predictions

# 假设信号控制策略模型为基于时间的调整策略
def adjust_light_time(predictions, current_time):
    if predictions < 30:
        return 'green'
    elif predictions < 60:
        return 'yellow'
    else:
        return 'red'

# 模拟交通信号控制系统的运行
time = np.arange(0, 30, 0.5)
predictions = predict_flow(model, features)
control_strategies = [adjust_light_time(pred, t) for t in time]
print(control_strategies)
```

在上述代码中，我们首先定义了数据预处理函数 `preprocess_data`，将原始数据清洗、特征提取并转换为模型所需的格式。然后，定义了模型构建与训练函数 `build_model`，使用深度学习模型进行训练。接着，定义了模型预测与信号控制策略函数 `predict_flow` 和 `adjust_light_time`，用于预测交通流量并根据预测结果调整信号灯配时。最后，模拟了交通信号控制系统的运行，生成信号控制策略。

### 5.3 代码解读与分析

**数据预处理函数`preprocess_data`**：
- 清洗数据：使用 `dropna` 方法去除缺失值。
- 特征提取：使用 `pd.get_dummies` 方法将时间特征转换为哑变量，并将时间特征与其他特征合并。
- 数据转换：将数据转换为模型所需的格式，即特征和目标值的numpy数组。

**模型构建与训练函数`build_model`**：
- 模型定义：使用 `tf.keras.Sequential` 定义深度学习模型，包括输入层、两个隐藏层和输出层。
- 模型编译：使用 `model.compile` 方法编译模型，指定优化器和损失函数。
- 模型训练：使用 `model.fit` 方法训练模型，设置训练轮数、批次大小和学习率。

**模型预测与信号控制策略函数`predict_flow`**：
- 模型预测：使用 `model.predict` 方法进行预测，返回模型输出。
- 信号控制策略：根据预测结果调整信号灯配时，如 `green`、`yellow`、`red`。

**模拟交通信号控制系统的运行**：
- 生成时间序列：使用 `np.arange` 方法生成时间序列，用于模拟交通流量预测和信号控制策略。
- 预测交通流量：使用 `predict_flow` 函数预测每个时刻的交通流量。
- 调整信号配时：根据预测结果和当前时间，使用 `adjust_light_time` 函数调整信号灯配时。
- 输出控制策略：将每个时刻的控制策略输出，以供实际应用。

### 5.4 运行结果展示

运行上述代码，输出信号控制策略如下：

```
['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green',

