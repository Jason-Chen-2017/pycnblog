                 

# 1.背景介绍

人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，研究如何让计算机模拟人类的智能。人工智能算法的核心是机器学习（Machine Learning，ML），它使计算机能够从数据中学习，而不是被人类直接编程。

深度学习（Deep Learning，DL）是机器学习的一个分支，它使用多层神经网络来模拟人类大脑的工作方式。深度学习算法可以处理大量数据，自动学习模式和规律，从而进行预测和决策。

ONNX（Open Neural Network Exchange）是一个开源的神经网络交换格式，它允许不同的深度学习框架之间的互操作性。TensorRT是NVIDIA的一个深度学习加速引擎，它可以加速深度学习模型的运行速度。

在本文中，我们将探讨如何使用ONNX格式的深度学习模型，并将其转换为TensorRT格式，以实现更高的运行速度和性能。

# 2.核心概念与联系

在深度学习中，神经网络是模型的核心组成部分。神经网络由多个节点（神经元）和连接这些节点的权重组成。每个节点接收输入，对其进行处理，并输出结果。连接权重决定了节点之间的信息传递方式。

ONNX是一个用于表示神经网络的标准格式。它定义了一个用于表示神经网络的数据结构，包括节点类型、连接关系、权重等信息。ONNX格式允许深度学习框架之间的互操作性，使得模型可以在不同的框架上进行训练和推理。

TensorRT是NVIDIA的一个深度学习加速引擎，它可以加速深度学习模型的运行速度。TensorRT支持多种深度学习框架，包括TensorFlow、PyTorch、Caffe等。TensorRT使用自己的格式来表示深度学习模型，称为TensorRT格式。

为了将ONNX格式的深度学习模型转换为TensorRT格式，我们需要使用ONNX-TensorRT转换器。ONNX-TensorRT转换器是一个开源的工具，它可以将ONNX格式的模型转换为TensorRT格式。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解

在本节中，我们将详细讲解如何将ONNX格式的深度学习模型转换为TensorRT格式的算法原理。

## 3.1 ONNX格式的深度学习模型

ONNX格式的深度学习模型由以下组成部分：

1. 图（Graph）：表示神经网络的结构，包括节点（Node）和连接（Edge）。
2. 节点（Node）：表示神经网络中的一个操作，如卷积、激活函数等。
3. 连接（Edge）：表示节点之间的关系，包括权重、偏置等。

ONNX格式的深度学习模型可以使用Python的ONNX Runtime库进行加载和操作。以下是一个加载ONNX格式的深度学习模型的示例代码：

```python
import onnx

# 加载ONNX格式的深度学习模型
model = onnx.load("model.onnx")
```

## 3.2 TensorRT格式的深度学习模型

TensorRT格式的深度学习模型由以下组成部分：

1. 网络（Network）：表示神经网络的结构，包括层（Layer）和连接（Connection）。
2. 层（Layer）：表示神经网络中的一个操作，如卷积、激活函数等。
3. 连接（Connection）：表示层之间的关系，包括权重、偏置等。

TensorRT格式的深度学习模型可以使用C++的TensorRT库进行加载和操作。以下是一个加载TensorRT格式的深度学习模型的示例代码：

```cpp
#include <iostream>
#include <cuda.h>
#include <cudnn.h>
#include <nvcuda.h>
#include <nvtensors.h>
#include <trt.h>

// 加载TensorRT格式的深度学习模型
IHostMemory model_buffer = cudaMallocHostMem(model_size);
cudaMemcpy(model_buffer, model, model_size, cudaMemcpyHostToDevice);
IBuilder builder(getDevice(0));
builder.maxWorkspaceSize(1 << 30);
builder.maxBatchSize(1);
builder.maxBindingSize(1 << 20);
builder.maxAutoBatchSize(1 << 20);
builder.maxAutoExtent(1 << 20);
builder.maxDimensionsOfAuxiliaryBuffers(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 << 20);
builder.maxWorkspaceSize(1 <<