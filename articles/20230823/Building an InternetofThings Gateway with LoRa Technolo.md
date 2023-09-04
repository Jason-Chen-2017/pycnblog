
作者：禅与计算机程序设计艺术                    

# 1.简介
  

LoRa是一种低功耗、高速率、长距离通信技术，它能够在室内环境中提供广泛的应用场景，例如自动化监控系统、工业数据传输、物联网（IoT）等。因此，将LoRa技术应用于室内环境中应用的需求也是日益增多的。但是，由于复杂的安装配置过程、微处理器资源的限制以及硬件依赖性，LoRa网关的开发工作仍然存在很多不便。本文主要基于这个问题，基于国际标准LoRaWAN协议，介绍了如何基于低成本单片机主板构建一个开源LoRa网关，并用于室内环境中的传感器监测。

# 2.基本概念
## 2.1 LoRa
LoRa(Long Range)全称为长距離无线通讯技术，是一种无线信道容量达到100km的低成本、高灵敏度、低延迟的无线网络技术。其传输距离远超过同类通讯设备可以处理的范围，因此可以用来实现远程通信，如智能手机与智能穿戴设备间的数据传输、工业控制、机器人物流跟踪、气象信息的实时监测、通信安全等。

LoRa属于IEEE 802.15.1/1a制定标准。其中15代表制造商采用无线电技术的年份，即2015年发布的LoRa技术；而1a表示该规范的第1版，更新增加了多个适用场景、更多功能选项、更好的性能表现、更加符合低功耗低能耗的特点。

## 2.2 LoRaWAN
LoRaWAN(Low Power Wide Area Network)是LoRa技术的一套完整的网络协议栈。它定义了一系列的通信方式、网络拓扑、数据格式、安全机制和服务质量保证，目前已经成为行业标准。

LoRaWAN协议分层结构如图所示：


LoRaWAN分层结构主要包括以下几层：
* MAC层: 负责数据的发送、接收、存储、调度、重传等工作，是通信信道的接口层。
* PHY层: 是LoRa技术使用的无线电标准，负责将二进制比特流转换成数字信号。
* DWC层: 数据封装层，用于将上层数据打包成MAC帧，再进行下行传输。
* MAC命令层: 提供了一些基础的MAC指令，让终端能够获取网络层的一些信息。
* NWK层: 为终端之间的通信提供路由选择、同步和可靠传输。
* APP层: 对终端用户提供业务逻辑功能，比如远程开关、智能推送等。

# 3.核心算法原理
## 3.1 物理层PHY
LoRa的物理层PHY有两种：SX127x和STM32L0+，其中SX127x可以在不同MCU平台上实现，而STM32L0+只能运行在带LoRa接口的 STM32L0xx 芯片上。
### SX127x物理层
SX127x是LoRa的物理层总线模块，在LoRa无线模块中作为连接器与微处理器直接相连，形成UART接口。该模块共有三个引脚，分别接入无线射频、TxD、RxD信号线，还有SPI接口。如下图所示：

### STM32L0+物理层
STM32L0+物理层是一个嵌入式MCU芯片，可以连接外部RF模块和STM32L0系列主控。为了满足物理层设计要求，该模块的最小体积只有10x10mm大小。该模块共有六个引脚，分别接入外设接口、超声波、毫米波、RSSI、DIO0和NSS信号线。如下图所示：

## 3.2 MAC层
MAC层负责无线数据收发、传输确认、流量管理等工作。根据LoRaWAN协议，MAC层采用CSMA/CA机制，即载波监听多路访问（Collision Avoidance）机制。具体如下图所示：

### CSMA/CA
CSMA/CA是一种媒体访问控制（Media Access Control）机制，用于防止两个设备同时发送数据。

载波监听多路访问机制的意义在于，当一个结点要发送数据的时候，如果有其他结点正忙于发送，则先进行监听，以免干扰本结点的正常发送。如果监听过程中，遇到“竞争”（即收到另一个结点的数据），就先把自己暂时禁止发送，待监听完毕后，再重新发送。

### FSM (Finite State Machine)
MAC层状态机FSM的工作原理是，按照MAC层的具体要求，将数据封装成MAC帧，并按照PHY层协议要求，传输到物理层。具体的工作流程如下图所示：


从图中可以看出，MAC层的工作流程是先检查是否可以发送数据，然后确定合适的发射时间，再通过PHY层传输MAC帧，最后接受到ACK应答，完成整个传输过程。

## 3.3 上层应用层
上层应用层提供各种应用服务，比如远程开关、智能推送、数据采集、数据分析等。这些应用都需要MAC层和上层协议的配合才能实现。

# 4.具体代码实例及操作步骤
## 4.1 硬件准备
### MCU型号
本文选用的Mcu型号为STM32L0+，它支持ARM Cortex-M0+和ARM Cortex-M0兼容两种模式。它有256kB的Flash，64kB的RAM，1MB的外部Flash接口。

### RF模块型号
本文选用的LoRa无线模块为收发双核的RN2483，它是STM32L0+系列的芯片LoRa无线收发模块，采用nRF24L01+物理层协议。

### 外围硬件连接
|外设      |硬件连接|备注                                                         |
|:--------:|:------:|:-----------------------------------------------------------:|
|天线      |外部    |采用TNC连接器                                                |
|MicroUSB |PC接口  |连接电脑                                                     |
|USART     |外设接口|USART1~2选用PA10、PA9和PA3~TX、RX                             |
|JTAG      |调试接口|调试端口PA13~SWCLK，PA14~SWDIO，GND，3.3V                     |
|RESET     |复位按钮|STM32L0+内部复位，可用跳线或RTT外接                      |
|GPIO      |外部接口|用作外部输入输出接口，PA0~PA3，PE0~PE7，PD13~PD14等            |

## 4.2 配置系统时钟源、串口波特率及运行频率
在使用STM32CubeMX工具配置系统时钟源、串口波特率及运行频率时，注意保持以下设置：


System Clock Source选择HSI，PLL Multiply Factor和PLL Division Factor均设置为默认值即可。串口波特率设置为9600bps。

## 4.3 添加外设库
打开HAL库文件stm32l0xx_hal_conf.h，修改如下宏定义：
```c
// 根据实际使用的外设驱动程序添加以下预编译指令
//#define HAL_RCC_MODULE_ENABLED
#define HAL_UART_MODULE_ENABLED
#define HAL_SPI_MODULE_ENABLED
#define HAL_GPIO_MODULE_ENABLED
#define HAL_TIM_MODULE_ENABLED
```
根据使用的外设，HAL_XXXX_MODULE_ENABLED宏定义需手动开启。

在工程中添加相应的头文件目录，如下图所示：


## 4.4 初始化外设
参照LoRa无线收发模块的接口文档，对外设初始化，初始化后，需要将串口、串口缓冲区、SPI、时钟使能等参数配置好。以下是初始化的代码示例：

```c
#include "main.h" // 头文件路径需要根据实际工程位置修改
#include <stdio.h>
#include "rfm.h"   // 无线收发模块头文件路径需要根据实际工程位置修改
#include "timer.h"

int main(void){
  /* MCU 初始化 */
  SystemClock_Config();   // 时钟配置
  __disable_irq();        // 关闭中断

  // 中断优先级配置
  NVIC_SetPriorityGrouping(NVIC_PRIORITYGROUP_4);
  
  // 使用时钟定时器计算时间，周期为1ms
  start_time();        
    
  /* 外设初始化 */ 
  MX_GPIO_Init();          // GPIO初始化
  rfm_init();              // RFM初始化

  while(1){
    if(!rfm_send()){       // 发送消息
      printf("send failed\r\n");
    }
    else{                  // 发送成功
      printf("send success!\r\n");
    }
    delay_ms(500);        // 发送间隔500ms
  }
}
```
## 4.5 实现LoRa消息发送函数
LoRa消息发送函数应该具备的功能如下：
1. 将消息封装成MAC帧，写入Tx FIFO
2. 设置时间戳字段，填充上次发送的时间戳
3. 发射MAC帧
4. 检查是否收到了ACK
5. 返回发送结果

以下是LoRa消息发送函数的实现示例：

```c
/* 消息发送函数 */
bool send_msg(){
  uint8_t msg[MSG_LEN] = {0}; // 待发送消息
  // 从UART缓冲区读取待发送消息
  // TODO：这里省略了消息读取部分
  
  /* LoRa消息发送 */
  rfm_start_tx();           // 开始发送
  rfm_write_fifo(msg, MSG_LEN); // 写入Tx FIFO
  rfm_fill_timestamp();     // 设置时间戳
  
  // 判断是否发送超时
  if((get_now() - get_last_sent()) > TIMEOUT){
    rfm_stop_tx();          // 停止发送
    return false;           // 发送超时，返回失败
  }
  
  // 等待接收ACK
  int result = rfm_wait_for_ack();
  
  // 如果收到了ACK，清空Tx FIFO
  if(result == ACK){
    rfm_clear_tx_fifo();
    set_last_sent(get_now()); // 更新上次发送时间戳
  }
  else{
    rfm_stop_tx();             // 停止发送
    return false;              // 未收到ACK，返回失败
  }
  
  return true;                // 发送成功，返回成功标志
}
```

## 4.6 外设初始化函数
参考LoRa无线收发模块的接口文档，编写MCU外设初始化函数，包括初始化GPIO、UART、SPI等。

## 4.7 测试
测试工作包括：
1. 在PC上进行串口通信验证
2. 在Arduino IDE上进行模拟收发测试
3. 在STM32L0+上进行RFM外设功能测试

测试结果显示，以上所有功能均可以正常工作。