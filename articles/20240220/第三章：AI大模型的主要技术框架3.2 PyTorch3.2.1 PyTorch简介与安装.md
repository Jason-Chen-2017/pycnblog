                 

## 3.2 PyTorch-3.2.1 PyTorch简介与安装

PyTorch 是一个基于 Torch 库的开源 machine learning 框架，支持 GPU 加速训练和 inferencing，并且提供 Pythonic 风格的 API。它最初是由 Facebook AI Research 团队开发的，后来社区的贡献也越来越多。

PyTorch 具有以下特点：

* **Pythonic**：PyTorch 的 API 设计十分 pythonic，易于上手。
* **动态计算图**：PyTorch 采用动态计算图，这意味着张量操作会立即执行，而不是像 TensorFlow 那样先构建 computational graph，然后再执行。这使得 PyTorch 更加灵活，同时也更容易调试。
* **GPU 支持**：PyTorch 支持 CUDA 并行计算，因此可以利用 GPU 加速训练和 inferencing。
* **丰富的社区和生态系统**：PyTorch 拥有活跃的社区和丰富的生态系统，包括各种 tutorials、libraries 和 frameworks。

### 3.2.1 PyTorch 安装

#### 3.2.1.1 系统要求

PyTorch 支持 Linux, Windows 和 macOS 平台。具体来说，PyTorch 支持以下操作系统和硬件组合：

| 操作系统      | 处理器  | GPU       | CUDA 版本 |
| -------------- | -------- | ---------- | --------- |
| Linux         | x86\_64  | NVIDIA   | 10.2     |
| Windows       | x86\_64  | NVIDIA   | 10.2     |
| macOS         | x86\_64  | NVIDIA   | 10.2     |
| Linux         | ARM64   | NVIDIA   | 11.1     |
| macOS         | ARM64   | NVIDIA   | 11.1     |

注意：PyTorch 仅支持 NVIDIA GPU，因此如果您希望利用 GPU 进行训练和 inferencing，请确保您已经购买了 NVIDIA GPU。

#### 3.2.1.2 通过 pip 安装

你可以通过 pip 安装 PyTorch。首先，你需要确定你想安装哪个版本的 PyTorch。你可以通过 PyTorch 官方网站查看当前可用的版本。一旦选择了版本，你可以运行以下命令安装 PyTorch：
```bash
pip install torch torchvision -f https://download.pytorch.org/whl/cu102/torch_stable.html
```
注意：在上述命令中，`cu102` 表示支持 CUDA 10.2。如果你使用的是其他版本的 CUDA，请替换成对应的版本号。

#### 3.2.1.3 通过 conda 安装

你还可以通过 conda 安装 PyTorch。首先，你需要创建一个新的 conda 环境：
```bash
conda create -n pytorch_env python=3.9
```
接下来，激活该环境：
```bash
conda activate pytorch_env
```
最后，运行以下命令安装 PyTorch：
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
注意：在上述命令中，`cudatoolkit=10.2` 表示支持 CUDA 10.2。如果你使用的是其他版本的 CUDA，请替换成对应的版本号。

### 3.2.2 PyTorch 入门

接下来，我们将介绍 PyTorch 的基本概念和操作。

#### 3.2