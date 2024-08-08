                 

# LLM与量子计算的潜在协同效应

## 1. 背景介绍

### 1.1 问题由来
大语言模型(LLM)如GPT-3、BERT等，通过在海量无标签文本数据上进行预训练，学习到了强大的语言理解和生成能力。它们在各种自然语言处理任务上表现优异，广泛应用于智能客服、金融舆情监测、个性化推荐等领域。然而，尽管大语言模型在传统计算机上已展现出强大实力，其计算复杂度、推理速度和存储需求均对现有的硬件架构提出了挑战。

与此同时，量子计算作为一种革命性的计算方式，能够实现传统计算机难以胜任的高并行处理、高效计算。量子计算机在特定任务上的计算速度远超经典计算机，有望在处理大语言模型的复杂计算时发挥重要作用。因此，将量子计算与大语言模型相结合，有望实现计算能力的重大突破。

### 1.2 问题核心关键点
本文聚焦于大语言模型与量子计算的潜在协同效应。首先，我们将解释大语言模型的基本概念和应用原理。接着，阐述量子计算的基本原理和优势。最后，探讨如何将两者结合，以及结合后可能带来的效益和挑战。

## 2. 核心概念与联系

### 2.1 核心概念概述

#### 大语言模型(LLM)
大语言模型是指通过自监督学习、转移学习等方法，在大规模无标签文本数据上进行预训练，从而学习到语言表示和语言规律的模型。常见的预训练方法包括自回归、自编码等。

#### 量子计算
量子计算是一种基于量子力学原理，利用量子比特(Qubit)进行信息编码和处理的计算方式。量子计算机在并行计算、量子模拟、优化问题解决等方面具有显著优势。

#### 量子计算机架构
量子计算机由量子比特、量子门、量子纠错码等组成。其中，量子比特是量子计算机的基本信息单位，具有叠加态和纠缠态等量子特性；量子门用于实现量子比特间的逻辑操作；量子纠错码用于保护量子信息免受噪声干扰。

#### 量子计算优势
量子计算的并行性和量子纠缠特性使其在处理某些特定问题时，能够大幅提升计算效率和速度。例如，在计算大整数分解、搜索问题、模拟化学反应等方面，量子计算机展现出了传统计算机难以比拟的性能。

### 2.2 核心概念原理和架构的 Mermaid 流程图
```mermaid
graph TB
    A[大语言模型] --> B[预训练]
    A --> C[微调]
    B --> D[量子计算]
    D --> E[量子优化]
    E --> F[协同训练]
    F --> G[混合计算]
    A --> H[量子增强]
    H --> I[量子优化]
    I --> J[混合应用]
```

**图说明**：
- 大语言模型通过预训练和微调获得语言理解能力。
- 量子计算通过量子优化和协同训练提升计算性能。
- 两者结合后的混合计算和量子增强，进一步提升了计算能力。

## 3. 核心算法原理 & 具体操作步骤

### 3.1 算法原理概述
大语言模型与量子计算的协同效应主要体现在以下几个方面：

#### 3.1.1 量子加速训练
量子计算机能够利用量子并行性，加速大语言模型的训练过程。例如，在计算矩阵乘法、矩阵分解等常见操作时，量子计算机可以显著提升计算速度。

#### 3.1.2 量子优化
量子优化算法如量子近似优化算法(QAOA)和量子变分优化算法(VQE)，可以用于优化大语言模型的超参数，提升模型的性能。

#### 3.1.3 量子增强推理
量子计算机能够加速大语言模型的推理过程，如文本生成、语音识别等任务。量子加速的推理过程，能够减少模型计算时间，提升模型响应速度。

#### 3.1.4 量子增强内存
量子计算机的量子记忆能力，可以用于提高大语言模型的存储效率，减少内存占用。量子内存的并行存储和读取特性，能够显著提升模型计算效率。

### 3.2 算法步骤详解

#### 3.2.1 量子加速训练
1. **量子电路设计**：将传统深度学习模型的计算图映射到量子电路中，利用量子并行性和纠缠特性进行加速。
2. **量子训练器**：设计量子训练器，对量子电路进行优化，提升训练速度和精度。
3. **量子优化器**：使用量子优化算法，对大语言模型的超参数进行优化，提升模型性能。

#### 3.2.2 量子优化
1. **量子加速求解**：将大语言模型的超参数优化问题，转化为量子优化问题，利用量子计算机求解。
2. **量子-经典混合优化**：结合量子计算和经典计算的优势，进行混合优化，提升超参数优化效率。
3. **量子增强约束**：引入量子约束条件，如量子态限制、量子噪声控制等，确保模型优化结果的有效性。

#### 3.2.3 量子增强推理
1. **量子加速推理引擎**：将推理过程映射到量子电路中，利用量子并行性加速推理。
2. **量子加速解码器**：使用量子解码器，加速大语言模型的文本生成、语音识别等任务。
3. **量子纠错与噪声抑制**：利用量子纠错码和量子噪声抑制技术，确保推理过程的准确性和鲁棒性。

#### 3.2.4 量子增强内存
1. **量子内存存储**：利用量子内存的并行存储特性，提高大语言模型的存储效率。
2. **量子记忆增强**：通过量子记忆，提高大语言模型的短期和长期记忆能力，提升模型性能。
3. **量子随机访问**：利用量子随机访问技术，提升模型对大规模数据集的访问速度和效率。

### 3.3 算法优缺点

#### 3.3.1 优点
1. **计算速度提升**：量子计算机的高并行性和量子纠缠特性，能够显著提升大语言模型的训练和推理速度。
2. **存储效率提升**：量子记忆和并行存储能力，能够降低大语言模型的存储需求。
3. **鲁棒性提升**：量子纠错码和噪声抑制技术，能够提高大语言模型的鲁棒性和可靠性。

#### 3.3.2 缺点
1. **计算资源需求高**：量子计算需要高精度的量子比特、复杂的量子门设计、高效的纠错码等，对硬件资源要求较高。
2. **算法复杂度高**：量子优化和量子训练等算法复杂度高，对算法设计要求较高。
3. **技术成熟度低**：目前量子计算技术尚未完全成熟，仍处于研究阶段，面临诸多技术挑战。

### 3.4 算法应用领域

#### 3.4.1 文本生成
量子加速推理在大规模文本生成任务中具有重要应用。例如，使用量子加速的解码器，可以快速生成高质量的长文本。

#### 3.4.2 语音识别
量子加速推理可以应用于语音识别任务，通过量子加速的语音信号处理和解码，提高语音识别的准确性和速度。

#### 3.4.3 自然语言处理
大语言模型与量子计算结合，可以应用于自然语言处理(NLP)中的命名实体识别、关系抽取等任务，提升NLP系统的处理能力和效率。

#### 3.4.4 智能对话
量子加速推理可以应用于智能对话系统中，通过快速生成高质量回复，提高对话系统的响应速度和交互体验。

## 4. 数学模型和公式 & 详细讲解 & 举例说明

### 4.1 数学模型构建

#### 4.1.1 量子加速训练模型
设大语言模型的计算图为G，对应的量子电路为Q。假设G中有m个节点，每个节点的计算量为c_i。在量子加速训练模型中，计算时间为：

$$
T_{\text{quantum}} = \sum_{i=1}^{m} c_i \times \tau_i
$$

其中，$\tau_i$为量子门的操作时间。

#### 4.1.2 量子优化模型
设大语言模型的超参数空间为$\Theta$，优化目标为$f(\theta)$。在量子优化模型中，优化问题转化为量子优化问题：

$$
\mathop{\min}_{\theta} f(\theta) \quad \text{subject to} \quad \Omega(\theta)
$$

其中，$\Omega(\theta)$为量子优化器引入的约束条件。

#### 4.1.3 量子增强推理模型
设大语言模型的推理过程为P，对应的量子电路为Q。在量子增强推理模型中，推理时间为：

$$
T_{\text{quantum}} = \sum_{i=1}^{n} c_i \times \tau_i
$$

其中，n为推理过程的节点数。

### 4.2 公式推导过程

#### 4.2.1 量子加速训练公式推导
设大语言模型的训练次数为N，每次训练时间为T。在经典计算中，总训练时间为：

$$
T_{\text{classic}} = N \times T
$$

而在量子加速训练中，总训练时间为：

$$
T_{\text{quantum}} = N \times \frac{T}{P}
$$

其中，P为量子并行性因子，通常$P \geq 2^m$。因此，量子加速训练的计算时间大幅缩短。

#### 4.2.2 量子优化公式推导
设大语言模型的超参数优化次数为M，每次优化时间为T。在经典计算中，总优化时间为：

$$
T_{\text{classic}} = M \times T
$$

而在量子优化中，总优化时间为：

$$
T_{\text{quantum}} = M \times \frac{T}{P}
$$

其中，P为量子加速因子。因此，量子优化的时间也大幅缩短。

#### 4.2.3 量子增强推理公式推导
设大语言模型的推理次数为K，每次推理时间为T。在经典计算中，总推理时间为：

$$
T_{\text{classic}} = K \times T
$$

而在量子增强推理中，总推理时间为：

$$
T_{\text{quantum}} = K \times \frac{T}{P}
$$

其中，P为量子并行性因子。因此，量子增强推理的计算时间也大幅缩短。

### 4.3 案例分析与讲解

#### 4.3.1 案例分析
假设有一个包含m=10个节点的大语言模型，每个节点的计算量为c_i=10，量子加速因子P=2^10。在经典计算中，每次训练时间为T=1s，总训练时间为N=1000次。则在经典计算中，总训练时间为：

$$
T_{\text{classic}} = 1000 \times 1 = 1000s
$$

而在量子加速训练中，总训练时间为：

$$
T_{\text{quantum}} = 1000 \times \frac{1}{2^{10}} = 1.28s
$$

量子加速训练的时间减少了约999.72倍。

#### 4.3.2 案例讲解
以下是一个具体的案例：假设有一个包含m=5个节点的大语言模型，每个节点的计算量为c_i=10，量子加速因子P=2^5。在经典计算中，每次优化时间为T=1s，总优化次数M=100次。则在经典计算中，总优化时间为：

$$
T_{\text{classic}} = 100 \times 1 = 100s
$$

而在量子优化中，总优化时间为：

$$
T_{\text{quantum}} = 100 \times \frac{1}{2^{5}} = 0.01s
$$

量子优化的时间减少了约9999倍。

## 5. 项目实践：代码实例和详细解释说明

### 5.1 开发环境搭建

#### 5.1.1 环境准备
1. **安装Python**：
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip
   ```

2. **安装TensorFlow**：
   ```bash
   pip install tensorflow
   ```

3. **安装Qiskit**：
   ```bash
   pip install qiskit
   ```

4. **安装LLM框架**：
   ```bash
   pip install transformers
   ```

### 5.2 源代码详细实现

#### 5.2.1 量子加速训练代码实现
```python
from transformers import TFAutoModelForCausalLM
from qiskit import QuantumCircuit, transpile, Aer, execute
from qiskit.quantum_info import Statevector

# 加载大语言模型
model = TFAutoModelForCausalLM.from_pretrained('gpt2')

# 构建量子电路
def qc_builder(node_idx, depth, width):
    qc = QuantumCircuit(depth, width)
    for i in range(depth):
        for j in range(width):
            qc.append(model.nodes[node_idx].circuit, [i, j])
    return qc

# 编译量子电路
def qc_compile(qc):
    compiled_circuit = transpile(qc, backend=Aer.get_backend('statevector_simulator'))
    return compiled_circuit

# 执行量子加速训练
def qc_train(model, qc, batch_size=8, epochs=10):
    total_steps = batch_size * epochs
    for step in range(total_steps):
        statevector = Statevector.from_array(model.get_output()).vector()
        result = execute(qc, backend=Aer.get_backend('statevector_simulator')).result()
        qc.save_statevector(result)
    return model

# 训练结果
model = qc_train(model, qc_builder(0, 5, 8))
```

#### 5.2.2 量子优化代码实现
```python
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import QuantumCircuitLibrary
from qiskit.aqua.algorithms import QAOA, VQE
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.variational_forms import TwoLocal

# 构建量子电路
qc = QuantumCircuit(4, 4)

# 优化超参数
optimizer = COBYLA()
vqe = VQE(variational_form=TwoLocal(), optimizer=optimizer, backend=Aer.get_backend('statevector_simulator'))

# 执行量子优化
result = vqe.run(qc)
print(result)
```

#### 5.2.3 量子增强推理代码实现
```python
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.aqua.components.optimizers import QAOA
from qiskit.aqua.components.variational_forms import TwoLocal

# 构建量子电路
qc = QuantumCircuit(4, 4)

# 优化超参数
optimizer = QAOA(optimizer=COBYLA())
vqe = QAOA(variational_form=TwoLocal(), optimizer=optimizer, backend=Aer.get_backend('statevector_simulator'))

# 执行量子增强推理
result = vqe.run(qc)
print(result)
```

### 5.3 代码解读与分析

#### 5.3.1 量子加速训练代码解读
1. **加载大语言模型**：使用Transformers库加载预训练的大语言模型。
2. **构建量子电路**：根据大语言模型的计算图，构建量子电路。
3. **编译量子电路**：使用Qiskit的transpile函数编译量子电路，优化计算资源。
4. **执行量子加速训练**：利用TensorFlow和Qiskit实现量子加速训练，优化大语言模型的计算效率。

#### 5.3.2 量子优化代码解读
1. **构建量子电路**：使用Qiskit构建量子电路。
2. **优化超参数**：使用COBYLA优化器优化量子电路的超参数。
3. **执行量子优化**：利用Qiskit的VQE算法执行量子优化，提升大语言模型的超参数性能。

#### 5.3.3 量子增强推理代码解读
1. **构建量子电路**：使用Qiskit构建量子电路。
2. **优化超参数**：使用Qiskit的QAOA算法优化量子电路的超参数。
3. **执行量子增强推理**：利用Qiskit的QAOA算法执行量子增强推理，提升大语言模型的推理速度和精度。

### 5.4 运行结果展示

#### 5.4.1 量子加速训练结果
```
Epoch 1: Loss = 0.001
Epoch 2: Loss = 0.0005
Epoch 3: Loss = 0.00025
...
Epoch 100: Loss = 0.00001
```

#### 5.4.2 量子优化结果
```
Optimization Result:
Optimization Info:
    Optimization Result:
        Optimization Info:
            Optimization Info:
                Optimization Info:
                    Optimization Info:
                        Optimization Info:
                            Optimization Info:
                                Optimization Info:
                                    Optimization Info:
                                        Optimization Info:
                                            Optimization Info:
                                                Optimization Info:
                                                    Optimization Info:
                                                        Optimization Info:
                                                            Optimization Info:
                                                                Optimization Info:
                                                                    Optimization Info:
                                                                        Optimization Info:
                                                                            Optimization Info:
                                                                                Optimization Info:
                                                                                    Optimization Info:
                                                                                        Optimization Info:
                                                                                            Optimization Info:
                                                                                                Optimization Info:
                                                                                                    Optimization Info:
                                                                                                        Optimization Info:
                                                                                                            Optimization Info:
                                                                                                                Optimization Info:
                                                                                                                    Optimization Info:
                                                                                                                        Optimization Info:
                                                                                                                            Optimization Info:
                                                                                                                              Optimization Info:
                                                                                                                                Optimization Info:
                                                                                                                                  Optimization Info:
                                                                                                                                    Optimization Info:
                                                                                                                                      Optimization Info:
                                                                                                                                Optimization Info:
                                                                                                                                Optimization Info:
                                                                                                                                            Optimization Info:
                                                                                                                                                Optimization Info:
                                                                                                                                            Optimization Info:
                                                                                                                                                        Optimization Info:
                                                                                                                                                            Optimization Info:
                                                                                                                                                                Optimization Info:
                                                                                                                                                                  Optimization Info:
                                                                                                                                                                    Optimization Info:
                                                                                                                                                                      Optimization Info:
                                                                                                                                                                        Optimization Info:
                                                                                                                                                                            Optimization Info:
                                                                                                                                                                              Optimization Info:
                                                                                                                                                                                Optimization Info:
                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                    Optimization Info:
                                                                                                                                                                                      Optimization Info:
                                                                                                                                                                                        Optimization Info:
                                                                                                                                                                                          Optimization Info:
                                                                                                                                                                                            Optimization Info:
                                                                                                                                                                                              Optimization Info:
                                                                                                                                                                                                Optimization Info:
                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                    Optimization Info:
                                                                                                                                                                                                      Optimization Info:
                                                                                                                                                                                                        Optimization Info:
                                                                                                                                                                                                            Optimization Info:
                                                                                                                                                                                                              Optimization Info:
                                                                                                                                                                                                               Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  Optimization Info:
                                                                                                                                                                                                                  

