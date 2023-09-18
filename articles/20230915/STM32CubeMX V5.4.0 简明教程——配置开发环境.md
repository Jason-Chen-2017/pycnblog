
作者：禅与计算机程序设计艺术                    

# 1.简介
  

　　STM32CubeMX是一个图形化的集成开发环境(IDE)，用于在各种主流单片机上快速生成项目、代码和仿真器等。它包含了丰富的外设驱动模板、工具和函数，并能够简化嵌入式软件应用开发流程，提升效率。本文主要介绍STM32CubeMX 5.4.0版本的安装和配置开发环境。
# 2.安装准备工作
## 2.1 安装java
　　STM32CubeMX需要java运行时环境才能正常运行。如果你的系统中没有安装java，你可以选择以下两种方式进行安装。

1. 通过包管理器安装

   a) Ubuntu/Debian:

     ```bash
     sudo apt install openjdk-11-jre
     ```

   b) Arch Linux:

     ```bash
     sudo pacman -S jre11-openjdk
     ```

   c) Fedora:

     ```bash
     sudo dnf install java-latest-openjdk-devel
     ```

   d) openSUSE:

     ```bash
     sudo zypper in java-1_11-openjdk-devel
     ```

2. 从Oracle官网下载并安装JDK

## 2.2 安装STM32CubeMX
　　目前STM32CubeMX最新版本为V5.4.0，可以到官方网站下载安装程序包。

```bash
wget https://www.st.com/content/ccc/resource/technical/software/mcu_software/stm32_software/stm32cube/stm32cubemx_v5/5.4.0/zip/stm32cubemx-setup-win32-5.4.0.exe
```

或

```bash
wget https://www.st.com/content/ccc/resource/technical/software/mcu_software/stm32_software/stm32cube/stm32cubemx_v5/5.4.0/zip/stm32cubemx-linux-5.4.0.tar.gz
```

根据你使用的操作系统下载对应的安装包，然后双击运行安装程序即可安装。

# 3 配置开发环境
　　经过安装之后，我们进入STM32CubeMX的开始菜单，点击Configure按钮。如下图所示：


　　1. 添加工具链目录

       如果你的工具链所在目录不是默认路径（比如D:/Program Files (x86)/STMicroelectronics/STM32H7xx SW tools/bin），你可以选择添加它的路径。右侧的“Add”按钮用来添加一个路径，删除一个路径则点击右边的“Remove”按钮。在弹出的窗口中输入要添加的路径地址，点击确定后会保存这个路径。
 

 
   2. 添加ST标准库目录

      ST标准库目录一般在：C:\ProgramData\st_server或者C:\Users\%username%\AppData\Local\st_server\st_standard_library下。如果你下载的工具链在默认位置，这里面的内容应该已经完成设置。
 
  3. 设置全局模板搜索路径
 
       模板文件是ST提供给用户使用的一些代码文件模版，存放在安装目录下的STMCubeMX\Templates目录。如果要修改当前项目模板文件的查找路径，可以在左上角点击“Options”选项卡，再选中“Global Template Search Path”，可以指定多个查找路径。
 
# 4 生成新工程
　　配置开发环境之后，就可以按照自己的需求创建新的工程了。

1. 创建新工程

   在开始菜单中的“New Project”选项卡，选择“GPIO-based Project”或其他项目模板，根据提示填写工程名称、目标芯片型号、输出文件夹及其它信息。


2. 添加外设驱动模块

   模块的驱动模板位于STM32CubeMX安装目录下的Drivers目录，可以将其拖动到工程中，也可以手动添加。

   对于某些外设，例如ADC、I2C、SPI等，你可能还需要额外添加一些组件，例如寄存器头文件。

3. 配置工程属性

   可以点击工具栏上的“Properties”按钮，设置工程的一些属性，如：设置路径、编译器优化级别、启动文件名等。

   某些属性可能依赖于编译器版本和MCU型号，所以需要在属性页面中查看详细的配置说明。

# 5 开始编写代码
　　完成以上步骤之后，我们可以开始编写应用程序代码了。下面以配置UART作为例来说明如何配置串口外设驱动模块，并发送数据到屏幕上。

1. 配置UART外设

   1. 在工程资源管理器中找到UART1。
   2. 右键单击，打开上下文菜单，选择“Add peripheral”。
   3. 在左侧的列表框中选择“UART”。
   4. 根据MCU型号，检查“Baud Rate Settings”中的速率是否合适，若不合适，可调整。
   5. 勾选“Enable Peripheral Clocks”选项，以使外设生效。

2. 使用宏定义控制UART外设开关

   1. 在文件顶部加入宏定义，示例代码如下：

```c++
#define USE_UART              //使能UART外设驱动功能
```

   2. 在初始化代码里判断USE_UART宏，只有当该宏被定义的时候才开启UART外设。示例代码如下：

```c++
void SysTick_Handler() {
#ifdef USE_UART
    static uint8_t count = 0;
    
    if (++count == 5) {
        HAL_UART_Transmit(&huart1, "Hello World!\n", strlen("Hello World!\n"), 10);   //发送一条字符串到UART
        count = 0;
    }
    
#endif 
}
```

3. 修改SystemClock_Config函数

   在SystemClock_Config函数里修改SYSCLK_FREQ参数的值，示例代码如下：

```c++
/** System Clock Configuration
*/
void SystemClock_Config(void)
{

  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  
  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}
  
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI|RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.MSICalibrationValue = RCC_MSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.MSIPLLMode = RCC_MSIPLLMODE_ENABLE;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 40;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct)!= HAL_OK)
  {
    Error_Handler();
  }
  
  /** Initializes the CPU, AHB and APB busses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4)!= HAL_OK)
  {
    Error_Handler();
  }
  
//修改SYSCLK_FREQ参数值为40000000
  HAL_RCC_MCOConfig(RCC_MCO1, RCC_MCO1SOURCE_PLLCLK, RCC_MCODIV_1);
  LL_RCC_SetSysClkFreq(40000000);
  
  /** Configure the Systick interrupt time
  */
  HAL_SYSTICK_Config(HAL_RCC_GetHCLKFreq()/1000);
  
  /** Enable Systick Interrupt
  */
  HAL_NVIC_SetPriority(SysTick_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(SysTick_IRQn);
}
```

4. 添加源文件头注释

   每个源文件都需要写好头注释，包括版权申明、作者姓名、日期等信息，示例代码如下：

```c++
/*******************************************************************************
 * @file    uart.c
 * @author  <NAME>
 * @version V1.0.0
 * @date    17-August-2021
 * @brief   This file provides code for the configuration of UART.
 *******************************************************************************/ 

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __UART_H
#define __UART_H
```

5. 将外设驱动文件包含进来

   在源文件开头加上引用头文件，示例代码如下：

```c++
#include "stm32h7xx_hal.h"      /* include stm32h7xx HAL headers */
#include "main.h"               /* include project header files */
#include "uart.h"                /* include custom header file */
```

6. 测试编译

   将源码文件编译成二进制文件，通过串口助手验证程序是否正确运行。