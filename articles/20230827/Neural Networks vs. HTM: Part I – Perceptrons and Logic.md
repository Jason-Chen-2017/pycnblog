
作者：禅与计算机程序设计艺术                    

# 1.简介
  

前文中提到的神经网络（Neural Network）并不是唯一的一个可以用来学习数据的机器学习模型。另外一个流行的模型是专门用于处理高维数据的Hebbian Temporal Memory(HTM)模型。尽管HTM模型也在不断地发展中，但目前仍然被认为是一种更强大的模型。本文将探讨HTM的基本概念、术语和核心算法原理。

# 2.基本概念术语
## 2.1. Hebbian Learning
HTM的核心思想是Hebbian Learning，即“学而时习之”。根据Hebbian Learning的观点，两只竞争对手相互纠缠，一起进化出更好的策略。在HTM模型里，每一个突触都是一个连接到另一组突触上的权重，这些权重的值通过模拟“猜”或者“揣测”值的方式来训练。所以，每个单元(neuron)都可以看成是一个具有多个输入和输出的逻辑函数。

## 2.2. SDR
SDR是short-term memory(短期记忆)单元的缩写，它存储最近出现过的信息。SDRs可以分为持久性(permanent memory)SDR 和 暂时性(ephemeral memory)SDR两种。持久性SDRs保存在较长时间内保存下来的信息；而暂时性SDRs则在神经细胞死亡后丢失。HTM中的所有SDR共享同一个空间。

## 2.3. Dendrites
Dendrites是突触的主要部位。它们接收信号并传递给其他突触。每个突触都有一个与之对应的阈值，只有当其激活的电压超过了阈值时才会传递信号。HTM中的dendrites用来接收输入数据，并产生输出。

## 2.4. Axon hillock
Axon hillock 是主要的轴突，负责传递最终的信号。每个axon hillock都有一定的长度，并且可以通过STP(Spike Timing Dependent Plasticity)来适应突触的长度。

## 2.5. Synapse
Synapse 是突触之间的连接物。Synapse 的类型有不同的功能，例如 excitatory synapse, inhibitory synapse等。HTM 中的synapses 与神经元之间直接连接，并通过 dendritic 突触传导信息。

## 2.6. Time step/ Epoch
HTM模型中的时间步是指一次计算过程中的持续时间。在每次的时间步中，都会更新一次突触的权重，并根据新的突触权重重新计算神经元的输出。为了减少计算量，HTM采用了epoch 机制，一次epoch 中会进行多次时间步的计算。Epoch越多，模型精度越高，但是计算量也会增加。通常epoch取值为1000到10000。

# 3. Core Algorithms of the HTM Model
HTM 模型由以下几个基本组件构成：

- Input Layer：用于接收输入数据。
- HTM Cells：HTM单元是HTM模型的基本模块。每一个HTM单元包括几个输入，几个SDR，几个突触，和一个输出。这些单元连接起来形成层级结构，形成一个包含多层的网络。HTM单元根据输入信息和之前的单元输出信息，更新自己的SDR。然后，它把SDR的值作为输入传递到下一层的HTM单元中。
- Output layer：输出层就是最后的输出结果。由于输出层没有更新权重，所以不需要再进入下一层。但是，输出层可以由一些分类器层或回归层组成，如Softmax分类器或线性回归层。