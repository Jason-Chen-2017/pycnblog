# 手机Wi-Fi6技术的原理及性能优化

作者：禅与计算机程序设计艺术

## 1. 背景介绍

随着智能手机的广泛普及,手机上网已经成为现代生活中不可或缺的一部分。然而,随着用户数量和网络需求的不断增加,传统的Wi-Fi技术已经难以满足人们对于更高速、更稳定网络的需求。为了解决这一问题,Wi-Fi联盟在2019年推出了全新的Wi-Fi 6标准,也就是大家熟知的"802.11ax"。

Wi-Fi 6相比前代Wi-Fi标准,在网络速度、连接密度、功耗等方面都有了显著的提升,可以更好地满足当下智能手机用户的需求。作为Wi-Fi 6技术的主要应用场景之一,手机Wi-Fi 6的原理和性能优化成为了业界关注的热点话题。

## 2. 核心概念与联系

Wi-Fi 6的核心技术包括:

1. **OFDMA(Orthogonal Frequency Division Multiple Access)正交频分多址接入**
   - 可以将信道划分为多个子载波,实现多用户并发传输
   - 提高频谱利用率和网络容量

2. **MU-MIMO(Multi-User Multiple-Input Multiple-Output)多用户多输入多输出**
   - 可以同时服务于多个用户设备
   - 提高网络吞吐量

3. **1024-QAM(Quadrature Amplitude Modulation)1024阶正交幅度调制**
   - 提高单个子载波的数据传输率
   - 提高频谱利用率

4. **Target Wake Time(TWT)目标唤醒时间**
   - 可以让接入点和终端设备协商合适的唤醒时间
   - 减少终端设备的功耗

这些核心技术相互配合,共同构建了Wi-Fi 6的性能优势。

## 3. 核心算法原理和具体操作步骤以及数学模型公式详细讲解

### 3.1 OFDMA原理

OFDMA是Wi-Fi 6的核心技术之一,它通过将信道划分为多个子载波,使得多个用户设备可以并发地在同一信道上传输数据,从而提高频谱利用率和网络容量。

OFDMA的工作原理如下:

1. 将整个信道bandwidth $B$ 划分为 $N$ 个子载波,每个子载波的带宽为 $b = B/N$。
2. 将时间轴划分为多个时隙(Resource Unit, RU),每个时隙对应一个OFDMA符号。
3. 在每个时隙内,接入点可以根据用户设备的需求,动态分配不同数量的子载波给不同的用户设备。
4. 用户设备根据自身的信道质量,在接入点分配的子载波上进行数据传输。

OFDMA的数学模型如下:

假设系统带宽为 $B$, 子载波数为 $N$, 则每个子载波的带宽为:
$$ b = \frac{B}{N} $$

在第 $t$ 个时隙内,接入点分配给第 $i$ 个用户设备的子载波数为 $K_i(t)$,则该用户设备在该时隙内的传输速率为:
$$ R_i(t) = \sum_{k=1}^{K_i(t)} b \log_2(1 + \frac{P_i^k(t)G_i^k(t)}{\sigma^2}) $$

其中 $P_i^k(t)$ 是用户 $i$ 在第 $k$ 个子载波上的发射功率, $G_i^k(t)$ 是用户 $i$ 在第 $k$ 个子载波上的信道增益, $\sigma^2$ 是噪声功率谱密度。

接入点的目标是动态调整每个用户设备分配的子载波数 $K_i(t)$,以最大化系统的总吞吐量:
$$ \max \sum_{i=1}^M R_i(t) $$
其中 $M$ 是接入点服务的用户设备总数。

### 3.2 MU-MIMO原理

MU-MIMO是Wi-Fi 6另一项核心技术,它可以让接入点同时服务于多个用户设备,从而提高网络吞吐量。

MU-MIMO的工作原理如下:

1. 接入点配备有多个天线,可以同时向多个用户设备发送/接收数据流。
2. 用户设备也需要配备有多个天线,以支持MU-MIMO。
3. 接入点根据用户设备的信道状况,采用信号预编码技术,将数据流分配到不同的天线上发送。
4. 用户设备通过信号解码技术,可以从接收到的混合信号中提取出自己的数据流。

MU-MIMO的数学模型如下:

假设接入点有 $N_t$ 个发射天线,服务于 $K$ 个用户设备,每个用户设备有 $N_r$ 个接收天线。

接入点的发送信号矢量为 $\mathbf{x} \in \mathbb{C}^{N_t \times 1}$,用户 $k$ 的接收信号矢量为 $\mathbf{y}_k \in \mathbb{C}^{N_r \times 1}$,则有:
$$ \mathbf{y}_k = \mathbf{H}_k \mathbf{x} + \mathbf{n}_k $$

其中 $\mathbf{H}_k \in \mathbb{C}^{N_r \times N_t}$ 是用户 $k$ 的信道矩阵, $\mathbf{n}_k \in \mathbb{C}^{N_r \times 1}$ 是噪声矢量。

接入点的目标是设计预编码矩阵 $\mathbf{W} \in \mathbb{C}^{N_t \times K}$,使得每个用户设备的信噪比(Signal-to-Interference-plus-Noise Ratio, SINR)得到最大化:
$$ \max \min_{1 \leq k \leq K} \frac{|\mathbf{h}_k^H \mathbf{w}_k|^2}{\sum_{j \neq k}|\mathbf{h}_k^H \mathbf{w}_j|^2 + \sigma_k^2} $$

其中 $\mathbf{h}_k^H$ 是信道矩阵 $\mathbf{H}_k$ 的conjugate transpose, $\mathbf{w}_k$ 是预编码矩阵 $\mathbf{W}$ 的第 $k$ 列,$\sigma_k^2$ 是用户 $k$ 的噪声功率。

### 3.3 1024-QAM原理

1024-QAM是Wi-Fi 6的另一项核心技术,它通过提高单个子载波的调制阶数,从而提高单个子载波的数据传输率,进一步提高频谱利用率。

1024-QAM的工作原理如下:

1. 在OFDMA中,每个子载波可以采用不同的调制方式,如BPSK、QPSK、16-QAM、64-QAM等。
2. Wi-Fi 6采用了1024-QAM,即每个子载波可以携带10bit的信息,相比64-QAM(6bit/符号)有了显著提升。
3. 1024-QAM需要更高的信噪比才能可靠解调,因此结合MIMO技术可以进一步提高信噪比,从而支持1024-QAM的应用。

1024-QAM的数学模型如下:

假设子载波的复包络信号为 $s(t) = a(t) + jb(t)$,其中 $a(t)$ 和 $b(t)$ 分别为实部和虚部。

1024-QAM调制的信号constellation如下图所示:

$$ \begin{align*}
a(t) &= \sqrt{\frac{E_s}{10}} \cdot \left( \pm 1, \pm 3, \pm 5, \pm 7, \pm 9 \right) \\
b(t) &= \sqrt{\frac{E_s}{10}} \cdot \left( \pm 1, \pm 3, \pm 5, \pm 7, \pm 9 \right)
\end{align*} $$

其中 $E_s$ 是每个符号的能量。

对于AWGN信道,1024-QAM的比特错误率(Bit Error Rate, BER)可以表示为:
$$ BER = \frac{2(1-\frac{1}{\sqrt{1024}})}{log_2(1024)} \cdot Q\left(\sqrt{\frac{3E_b}{10N_0}}\right) $$

其中 $E_b$ 是每bit的能量, $N_0$ 是单边噪声功率谱密度, $Q(x)$ 是高斯Q函数。

### 3.4 TWT原理

TWT(Target Wake Time)是Wi-Fi 6的一项重要功能,它可以让接入点和终端设备协商合适的唤醒时间,从而减少终端设备的功耗。

TWT的工作原理如下:

1. 终端设备在加入网络时,会向接入点发送TWT请求,包括期望的唤醒周期、唤醒时间等参数。
2. 接入点根据终端设备的请求和网络状况,计算出最佳的TWT参数,并反馈给终端设备。
3. 终端设备根据接入点反馈的TWT参数,进入睡眠状态,在指定的时间唤醒进行数据收发。
4. 接入点也会根据终端设备的TWT参数,在指定时间唤醒终端设备,进行数据传输。

TWT的数学模型如下:

假设终端设备的唤醒周期为 $T$,唤醒时长为 $\Delta t$,那么终端设备的功耗可以表示为:
$$ P_{total} = P_{active} \cdot \Delta t + P_{sleep} \cdot (T - \Delta t) $$

其中 $P_{active}$ 是终端设备工作状态下的功耗, $P_{sleep}$ 是终端设备睡眠状态下的功耗。

接入点的目标是根据网络状况,动态调整每个终端设备的TWT参数 $T$ 和 $\Delta t$,使得总功耗 $P_{total}$ 最小化。

## 4. 具体最佳实践：代码实例和详细解释说明

### 4.1 OFDMA资源分配算法

基于前述的OFDMA数学模型,我们可以设计一种基于贪心算法的OFDMA资源分配算法,具体步骤如下:

```python
# 初始化
N = 256  # 子载波数量
M = 8    # 用户设备数量
RU = 10  # 时隙数量

# 用户设备信道状况
G = [[random.uniform(0, 1) for _ in range(N)] for _ in range(M)]

# 资源分配
R = [[0 for _ in range(RU)] for _ in range(M)]  # 用户吞吐量
K = [[0 for _ in range(RU)] for _ in range(M)]  # 分配子载波数

for t in range(RU):
    # 计算每个用户在当前时隙的最优子载波分配
    for i in range(M):
        sorted_g = sorted(enumerate(G[i]), key=lambda x: x[1], reverse=True)
        k = 0
        while sum(K[i]) < N//M:
            K[i][t] += 1
            R[i][t] += b * log2(1 + P_i * sorted_g[k][1] / sigma2)
            k += 1
    
    # 将剩余子载波平均分配给用户
    residual = N - sum(sum(K[i][t] for i in range(M)))
    for i in range(M):
        K[i][t] += residual // M
        R[i][t] += b * log2(1 + P_i * G[i][K[i][t]-1] / sigma2)

# 输出结果
print(K)
print(R)
```

该算法首先根据用户信道状况,为每个用户动态分配子载波,使得每个用户的速率得到最大化。然后将剩余的子载波平均分配给各用户。通过这种贪心策略,可以较好地平衡各用户的吞吐量,提高整体网络性能。

### 4.2 MU-MIMO预编码算法

基于前述的MU-MIMO数学模型,我们可以设计一种基于最小均方误差(Minimum Mean Square Error, MMSE)的预编码算法,具体步骤如下:

```python
# 初始化
N_t = 4  # 接入点天线数
N_r = 2  # 用户天线数
K = 2    # 用户设备数

# 信道矩阵
H = [np.random.randn(N_r, N_t) for _ in range(K)]

# MMSE预编码
W = np.zeros((N_t, K), dtype=complex)
for k in range(K):
    h_k = H[k]
    W[:, k] = np.linalg.pinv(np.sum([H[j].conj().T @ H[j] for j in range(K)]) + sigma2 * np.eye(N_t)) @ h_k.conj().T

# 计算SINR
SINR = [np.abs(h_k @ W