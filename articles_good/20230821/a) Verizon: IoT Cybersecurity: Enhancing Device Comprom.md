
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
作为移动终端以及云计算领域的重要组成部分，物联网（IoT）已经成为一个新的领域。随着物联网设备数量的不断增加、应用场景的多样化和技术的飞速发展，安全问题也越来越成为物联网系统面临的一大难题。对于物联网系统而言，从设备制造到部署，再到维护和运维阶段都面临着物联网设备和数据的泄露、恶意攻击、恶意程序入侵等严重威胁。因此，物联网系统的安全防御能力是其建设的关键之一。Verizon推出了Zigbee协会(ZCA)，该协会将建立在开放协议栈上运行的Zigbee网络视作一种独立实体，致力于增强其可靠性、可用性和安全性。基于这一观点，Verizon目前正在研究如何通过物联网(IoT)设备进行网络入侵检测、数据完整性验证、身份认证以及其他物联网安全相关的功能。Verizon近期推出的Zigbee智能传感器产品包括Zigbee智能插座、电子邮件通知灯、智能扬声器等，这些产品均由多个制造商生产。Verizon正在不断探索新型的物联网安全防护方式，并持续对IoT系统进行安全测试和调查，以保障其安全性。本文将结合实际案例介绍Verizon对IoT系统的安全防护能力建设。
## 主要亮点
- Zigbee协议入侵检测方法、ZigBee Sleepy End Devices的检测方法、安全资产发现方法；
- 数据完整性验证方法、设备认证及用户认证方法；
- 通过“虚构”安全事件对各项系统功能进行演练，有效提升系统的安全性。
## 结论
物联网系统的安全防御能力建设是一个复杂且艰巨的任务。本文介绍了Verizon对IoT系统的安全防御能力建设的方案，提供了系统流程、关键环节、具体方法以及技术工具，给读者留下了思考和参考的宝贵指导。希望能够提供帮助、启发、激励和动力，共同打造更安全、更可靠的物联网生态。
# 2.基本概念术语说明
## 2.1 基本概念
### 定义
**物联网（Internet of Things，IoT）** 是一种基于互联网的分布式的自动化的信息采集和处理技术，它使得各种物理对象（如家用电器、工业机器、环境监测系统等）通过数字连接，实现信息共享和交换，并通过云计算进行分析处理，形成独特的、实时的数据。 

**物联网安全（Internet of Things Security，IoTSec）** 是物联网系统和数据安全相关的工作，包括物联网系统的可靠性、可用性、隐私保护、数据安全、设备安全、恶意攻击防护等方面的保障措施。

### 特点
- 在物联网系统中，所有设备之间产生大量的数据，数据安全需要保证数据的完整性、正确性、可用性、访问权限控制和设备权限管理。
- 物联网安全研究将设备、终端、网络、应用程序、服务、安全机制等全生命周期安全问题综合考虑，并通过系统工程化的方法进行解决。
- 设备安全的主要目标是防止对设备进行非法入侵，主要手段是对设备进行数字签名验证、异常行为检测、威胁情报收集、实时威胁响应等。
- IoTSec是包括设备、终端、网络、应用程序、服务等的所有层面的安全建设。其中，设备安全侧重于确保设备不被恶意攻击、被篡改、暴露出去或泄漏个人信息，同时还要保证设备正常运行。
- IoTSec分三大支柱：传感网络安全、云计算安全、应用安全。
- 传感网络安全面临的主要风险来自恶意攻击、设备故障、恶意数据入侵等。
- 云计算安全面临的主要风险来自数据隐私泄露、数据篡改、业务逻辑泄露、操作系统漏洞、第三方平台攻击等。
- 应用安全面临的主要风险来自安全漏洞、业务漏洞、业务数据泄露等。

### 研究目标
IoTSec的研究目标如下：

1. 提高IoT设备的安全性能
2. 提升IoT设备的安全管理水平
3. 构建IoT平台安全体系结构
4. 提升IoT平台的安全可靠性、可用性、实时性、可追溯性和可控性
5. 促进IoT安全标准制定和推广
6. 改善IoT系统的安全开发与管理能力
7. 提供具有竞争力的IoT安全解决方案

## 2.2 术语
### 常用术语
- **主机（Host）**：硬件或者软件系统，通常可以执行指令来控制其上的硬件资源和软件功能。
- **低级攻击（Low-level attack）**：直接针对设备内部的攻击，例如修改芯片、拔掉电源线、插入恶意程序等。
- **中间人攻击（Man-in-the-middle attack）**：攻击者通过中间媒介控制通信通道，获取通信双方的任何消息，甚至可以伪装成通讯双方的请求。
- **物联网（IoT）设备**：具有物理标识和网络连接能力的计算机设备，可以实现与人、家庭、机关部门或其他计算机系统的物理通信。
- **物联网（IoT）垂直行业**：垂直行业是指经济领域，涉及的产品、服务、技术和流程高度相似，比如智能城市、智慧农业、智能医疗等。
- **物联网安全（IoTSec）**：指利用物联网技术提供安全管理服务，提供物联网平台安全体系、解决方案及相关标准，保障IoT设备、数据、用户及其他系统的安全。
- **信任边界（Trust Boundary）**：设备之间的关系图，描述各个设备的所属组织之间的信任关系。
- **秘钥（Key）**：用来加密、解密或签署数据的密码，设备通过网络传输秘钥来对接入数据进行加密。
- **低功耗设备（Low Power Devices，LQD）**：功耗较低的嵌入式系统，适用于连接频率较低、硬件成本较低的场景。
- **极端规模边缘计算（Extreme Scale Edge Computing，XSEC）**：一种超大规模集群架构，可以实现对海量数据的处理和分析。

## 2.3 Zigbee协议入侵检测方法
### 介绍
Zigbee协议是IEEE组织开发的一套无线通信协议，是在2011年发布的OSI模型中的第五层协议。Zigbee是一个低功耗低成本的无线通信系统，由Zigbee协会（ZCA）负责标准制定和推广。Zigbee协议运行在2.4GHz频段，每个设备都有一个唯一的设备地址，称为网关地址（coordinator address）。Zigbee支持两种模式：终端设备工作在终端模式，即单播通信；网关设备工作在网关模式，即组播通信。组播的目的是为了减少网络负载，当一个网关接收到一个命令后，会向整个网络发送该命令，而不是只向发送该命令的节点发送。 

#### 入侵检测方法概述
Zigbee协议基于IEEE 802.15.4的MAC层标准，采用无连接的点对点协议，因此不需要信任边界。但是，由于支持不同信道的通信，使得攻击者可以通过中间人攻击的方式进行窃听、篡改、欺骗、延迟或破坏通信过程。为了检测Zigbee协议的入侵，Verizon研究人员提出了以下两个入侵检测方法：
- MAC层特征检测：根据设备发送的MAC帧特征进行入侵检测。
- APSK特征检测：设备间采用对称加密，使用ASKEY加密算法和快速傅里叶变换FFT算法生成收发信信号，通过比较差值判断是否存在入侵行为。

### MAC层特征检测
MAC层特征检测是根据设备发送的MAC帧特征进行入侵检测，具体方法如下：

1. 每个设备的MAC层帧头部包含设备的网络地址、事务序列号、数据加密、传输速率、传感范围等信息，当设备发送数据包时，数据包中的MAC帧头部与之前发送的数据包相同。

2. 当设备接收到错误的MAC帧或没有收到应答时，设备会重发数据包，导致设备发生一定的抖动。因此，如果设备在一定时间内连续接收错误的MAC帧，那么就可以判定该设备存在入侵行为。

3. 使用动态检测技术可以动态地收集设备的MAC帧特征，例如，记录设备每秒钟发出的数据包个数、接收的数据包个数等信息，根据统计信息来判断设备是否存在入侵行为。

### APSK特征检测
APSK特征检测是根据设备间采用对称加密，使用ASKEY加密算法和快速傅里叶变换FFT算法生成收发信信号，通过比较差值判断是否存在入侵行为。具体方法如下：

1. 使用ASKEY加密算法对设备间通信数据进行加密。ASKEY是一种公钥加密算法，支持两类用户角色，协商出来的公钥加密算法可以用不同的私钥来解密。

2. 设备A首先随机选择一个素数q作为其私钥，然后计算它的公钥A=g^x mod p (p为一个质数)。x是A的私钥，g为G的取值范围[1,q-1]，G是公钥加密算法的初始值，公钥G=g^(p+1) mod p。

3. 设备A将自己的公钥A发送给设备B，设备B收到A的公钥后将自己的公钥B发送给设备A。

4. 设备A和B之间通信数据经过加密处理后分别发往对应的设备，如果通信过程中，A设备与B设备之间数据传输存在明显差异，那么就认为存在入侵行为。

5. 为了防止攻击者捕获设备间的通信数据，Verizon设计了网关作为中间节点，采用ASKEY加密算法加密网关与其他设备间的通信数据，只有网关才可以解密加密数据。这样做可以减小攻击者获取通信数据的可能性。

## 2.4 ZigBee Sleepy End Devices的检测方法
### 介绍
Zigbee协议是基于IEEE 802.15.4的MAC层协议，需要信任边界，所以它存在节点认证、数据完整性和访问控制等安全机制，但由于存在较大的功耗，设备进入睡眠状态时，会消耗较多的电能，影响系统整体的效率。因此，为了检测ZigBee Sleepy End Devices（ZSDB），Verizon提出了以下检测方法：

1. 使用I/Q测试法检测入侵行为：使用发射天线同时检测IQ信号，发现I波出现突然跳变、频谱窄带退化等电气特性变化，就说明存在入侵行为。

2. 使用设备特征检测技术检测入侵行为：通过分析设备发出的每条MAC帧的长度、载荷、种族码等特征信息，发现设备的配置、发射功率等微小差别，就会发现存在入侵行为。

3. 使用网络流量特征检测技术检测入侵行为：通过统计设备之间的网络数据流量特征，发现设备的抖动、错序等差异，就可以判断设备是否存在入侵行为。

4. 使用后台扫描检测入侵行为：使用日夜扫描技术扫描网络，发现设备的使用习惯、周末、节假日等特征，就可以发现设备的入侵行为。

5. 使用日志文件检测入侵行为：检查设备的日志文件，发现设备的使用习惯、周末、节假日等特征，就可以发现设备的入侵行为。

## 2.5 安全资产发现方法
### 介绍
安全资产就是那些可能会受到物联网设备入侵威胁的硬件、软件、设备等，当检测到发生入侵行为时，可以通过安全资产来确定入侵目标。Verizon提出了以下方法，来发现IoT系统中潜在的安全资产：

1. 使用嵌入式设备的固件检测方法：物联网设备一般会使用嵌入式系统，嵌入式设备的固件是存放在设备ROM（Read-Only Memory，只读存储器）中的，可以对其进行检测，检测到恶意修改和病毒等行为时，就可以判断该设备为安全资产。

2. 对关键数据的加密或签名校验：一些重要数据如用户名密码等，可以使用对称加密算法或签名算法进行加密或签名校验，如果加密后的结果和原始数据不一致，则可以判断该数据为安全资产。

3. 检测关键设备的配置参数：设备的配置参数中有很多敏感信息，如网络凭据、管理员账号密码等，如果设备的配置参数发生改变，就可以判断该设备为安全资产。

4. 对系统日志的分析：通过分析设备系统日志，发现一些异常行为，如系统异常、账户登录失败等，就可以判断该设备为安全资产。

5. 使用攻击和威胁建模技术检测安全资产：由于IoT系统面临着大量的攻击和威胁，因此可以使用攻击和威胁建模技术（ATM）来预测IoT系统中潜在的安全资产。

## 2.6 数据完整性验证方法
### 介绍
物联网（IoT）系统中数据传输的完整性、真实性和有效性是非常重要的，因为数据丢失、泄露或篡改都会造成严重的影响，因此需要有效的检测、跟踪、隔离、记录、清除等技术来保障数据完整性。Verizon提出了以下数据完整性验证方法：

1. 将数据分割成数据块：对要传输的数据进行分割，以便能够进行完整性验证。

2. 使用哈希函数对数据块进行散列：对分割的数据块进行散列，生成哈希值，将散列值进行传输。

3. 使用哈希链对数据块进行链接：使用哈希链对数据块进行链接，生成一条完整的数据链。

4. 使用事务确认：事务确认技术可以有效地检测数据丢失、重复、篡改等问题。

5. 使用可信时间戳：可信时间戳可以记录数据生成的时间，可以检测数据在传输过程中是否被篡改。

## 2.7 设备认证及用户认证方法
### 介绍
在物联网系统中，不同用户对设备的访问权限应该是受限的，只有授权的设备才能访问物联网数据。为此，需要对设备和用户进行身份验证，以确保设备的合法访问。Verizon提出了以下设备认证及用户认证方法：

1. 使用秘钥对加密：在设备和服务器之间使用加密来保证数据的机密性、完整性和身份验证。

2. 使用TPM模块进行设备认证：TPM（Trusted Platform Module，受信任的平台模块）是Intel推出的一种安全硬件，用于存储设备的相关信息，包括秘钥、证书等。

3. 使用VPN进行用户认证：使用VPN（Virtual Private Network，虚拟专用网络）可以在本地网络上建立安全通道，并对用户进行身份验证。

4. 使用TLS进行数据传输加密：使用TLS（Transport Layer Security，传输层安全）可以对设备之间的数据传输进行加密，保证数据的机密性、完整性和身份验证。

5. 使用IAM（Identity and Access Management，身份和访问管理）进行用户认证：使用IAM可以对用户进行身份验证，确保只有授权的用户才能访问系统资源。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 3.1 入侵检测原理
### IEEE 802.15.4 MAC层入侵检测
- 方法：通过监听设备之间的通信过程，检测是否存在MAC层入侵行为。
- 功能：通过分析设备发送的MAC帧特征，判断设备是否存在入侵行为。
- 关键环节：监听设备的MAC帧，分析MAC帧特征，判断设备是否存在入侵行为。
- 特点：容易受到中间人攻击。
- 操作步骤：
  1. 设置监听信道，监听设备之间的通信过程。
  2. 接收设备发送的MAC帧，分析MAC帧特征，判断是否存在入侵行为。
  3. 如果存在入侵行为，则向控制中心发出警报。

### FFT-based APSK入侵检测
- 方法：采用FFT-based APSK加密算法，检测设备间的通信过程是否存在明显差异。
- 功能：通过比较设备间通信数据加密后的差值，判断设备间是否存在明显差异。
- 关键环节：使用ASKEY加密算法，对设备间通信数据加密，比较加密后的结果。
- 特点：不需要信任边界，对所有设备和所有时间段均有效。
- 操作步骤：
  1. 设备A随机选择一个素数q作为其私钥，计算它的公钥A=g^x mod p (p为一个质数)。
  2. 设备A将自己的公钥A发送给设备B，设备B收到A的公钥后将自己的公钥B发送给设备A。
  3. 设备A和B之间通信数据经过加密处理后分别发往对应的设备，如果通信过程中，A设备与B设备之间数据传输存在明显差异，那么就认为存在入侵行为。
  4. 判断入侵类型：攻击类型包括加密前后数据差异、签名错误等。

## 3.2 ZigBee Sleepy End Devices检测方法
### I/Q测试法
- 方法：根据设备发送的I/Q信号进行入侵检测。
- 功能：根据设备发送的I/Q信号的差异，检测是否存在入侵行为。
- 关键环节：使用发射天线同时检测I/Q信号，检测是否存在入侵行为。
- 特点：由于设备进入睡眠状态时会消耗较多的电能，影响系统效率，不实时。
- 操作步骤：
  1. 配置发射天线，并设置发射功率。
  2. 使用发射天线同时检测I/Q信号。
  3. 根据检测到的I/Q信号差异，判断是否存在入侵行为。
  4. 识别入侵设备。

### MAC层特征检测法
- 方法：根据设备发送的MAC帧特征进行入侵检测。
- 功能：根据设备发送的MAC帧特征的差异，判断是否存在入侵行为。
- 关键环节：分析设备发送的MAC帧特征，判断是否存在入侵行为。
- 特点：难以检测设备的入侵类型。
- 操作步骤：
  1. 配置监听信道，监听设备发送的MAC帧。
  2. 接收设备发送的MAC帧，分析MAC帧特征，判断是否存在入侵行为。
  3. 如果存在入侵行为，则向控制中心发出警报。
  4. 识别入侵设备。

### 流量特征检测法
- 方法：根据设备间的网络数据流量特征进行入侵检测。
- 功能：根据设备间的网络数据流量特征的差异，判断是否存在入侵行为。
- 关键环节：统计设备间的网络数据流量特征，判断是否存在入侵行为。
- 特点：不精准。
- 操作步骤：
  1. 获取设备间的网络数据流量特征。
  2. 统计设备间的网络数据流量特征，判断是否存在入侵行为。
  3. 识别入侵设备。

### 后台扫描检测法
- 方法：通过日夜扫描技术，检测设备的活动时间。
- 功能：判断设备是否处于正常使用状态。
- 关键环节：扫描设备的使用习惯、周末、节假日等特征，判断是否存在入侵行为。
- 特点：设备通常不会在此刻一直处于使用状态。
- 操作步骤：
  1. 配置扫描频率，扫描设备网络的时间。
  2. 扫描设备的时间段，判断是否存在入侵行为。
  3. 如果存在入侵行为，则向控制中心发出警报。
  4. 识别入侵设备。

### 日志文件检测法
- 方法：分析设备日志文件，判断设备是否存在入侵行为。
- 功能：分析设备日志文件，发现异常行为，判断是否存在入侵行为。
- 关键环节：分析设备日志文件，判断是否存在入侵行为。
- 特点：不实时。
- 操作步骤：
  1. 从日志文件中读取设备日志。
  2. 分析设备日志文件，发现异常行为，判断是否存在入侵行为。
  3. 如果存在入侵行为，则向控制中心发出警报。
  4. 识别入侵设备。

# 4.具体代码实例和解释说明
## Python示例代码——I/Q检测法
```python
import numpy as np 
from scipy import signal

def iq_test(sig):
    """
    Input: sig -- list of complex numbers representing the time series
    
    Output: result -- boolean indicating whether or not the input signal is indicative of an intrusion detection test
        
    The iq_test function takes in a list of complex numbers representing a time series, 
    computes the discrete Fourier transform using the FFT algorithm provided by the SciPy package, 
    and returns True if the first significant frequency component exceeds its threshold for being suspected of a false positive; False otherwise.

    If the threshold value is set to zero, all components above the noise floor will be considered suspicious and will trigger alerting behavior. In this case, it's up to the user to determine how severe their network threat is based on other factors such as access point density, device count, etc.

    If the threshold value is greater than zero, only components with a power exceeding the threshold value are deemed suspicious and will trigger alerting behavior. This helps prevent spurious alerts from low-power devices that may not yet have been affected by a malicious event but could potentially at some point.
    """
    n = len(sig)
    yf = abs(np.fft.fft(sig)) # Compute the DFT of the signal
    
    max_idx = int((n + 1)/2) - 1 # Index of highest frequency component
    noise_floor = sum([abs(s)**2 for s in sig])/len(sig) # Estimate the noise floor
    
    thres = 0.1*noise_floor # Set the threshold for detecting intrusions
    
    return yf[max_idx]/noise_floor > thres # Return true if the maximum component exceeds the threshold
    
# Example usage: check if a single signal is suspicious based on the FFT methodology implemented here        
if __name__ == '__main__':
    sig = [complex(i,j) for j in range(10)] # Generate a sample signal with varying phase shift and amplitude
    print("Is signal {} indicative of an intrusion detection test? {}".format(sig,iq_test(sig)))
```

In the example code given above, we define a function called `iq_test` which accepts a list of complex numbers representing a time series. We then compute the Discrete Fourier Transform (DFT) of the signal using the `scipy.signal` module and extract the absolute values of each frequency component. We also estimate the noise floor of the signal based on the assumption that there should be relatively little energy present outside of these components due to various forms of background activity. Finally, we compare the largest frequency component with the noise floor and return `True` if it exceeds a certain threshold value.