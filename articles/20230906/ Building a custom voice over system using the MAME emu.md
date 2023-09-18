
作者：禅与计算机程序设计艺术                    

# 1.简介
  

MAME (Multiple Arcade Machine Emulator)是一款开源的游戏机模拟器，它拥有丰富的游戏和系统平台支持，能够在许多不同平台上运行，主要应用于家用游戏机、掌上游戏机等电子设备。Vivado（Xilinx）集成开发环境是一个高级的硬件设计工具，可以用来生成可综合的FPGA芯片产品。

本文将阐述如何利用Vivado和MAME来制作一个定制化的音频输出系统。通过搭建该系统，用户可以在游戏中看到独特的声音效果，从而增强游戏体验。本文将首先详细介绍MAME的工作原理，然后介绍相关的Verilog HDL语言及其语法。最后，我们将详细地演示如何利用Vivado工具生成基于Verilog HDL的音频输出系统。

# 2.基本概念术语
## 2.1 VHDL
VHDL 是一种声明式语言，用于描述并创建可综合的 FPGA 硬件。它的语法类似于 Verilog，但有所区别，例如对信号类型的定义。VHDL 支持结构化编程，允许模块化设计。VHDL 的代码编译后得到的中间产物为.v 文件。

## 2.2 Verilog 模块
Verilog 模块是一种用于描述数字逻辑功能和信号通信的硬件描述语言（HDL）。它支持模块化设计，具有强大的硬件建模能力。每一个 Verilog 模块都包含以下三个主要组成部分：

1.端口定义——它指定了模块的接口，包括输入端口、输出端口、参数、常量等；
2.变量定义——它定义了模块中的状态变量和寄存器；
3.行为描述——它描述了模块内部信号的时序关系，即表示模块的逻辑功能。

## 2.3 MAME 主板
MAME 是一款开源的游戏机模拟器，它支持多种游戏机平台。MAME 使用一种名为 M680x0 处理器的指令集，其中包含一些特殊功能，例如 DMA 传输和双声道混音。MAME 提供了一系列的接口（如硬件和软件）让其它程序可以访问系统的硬件资源，并控制系统的各个部分。

MAME 主板是一个特殊的计算机芯片，它集成了多个 M680x0 处理器、多个采样率转换（ADC/DAC）芯片、RAM、DDR RAM、FLASH 闪存等，并提供各种接口和连接器，使其可以与外部设备交互。

## 2.4 AD9833
AD9833 是一款集成双混音单元芯片，通常配备在主板的音频引脚上。它的工作原理是通过将频率、占空比和方波占空比编码到三相、五相或七相半导体管的驱动电压，再通过串联电路实现声音的合成。

## 2.5 M68000
M68000 是由 Motorola 公司开发的一条指令集，它的功能覆盖了最常用的计算机运算、数据移动、内存访问、输入输出、定时器、外设控制等功能。

## 2.6 Game PCB
Game PCB 是用户在游戏机上安装自己喜爱的游戏软件的区域。通常，它会采用 Atmel 的 ATmega88PA 或 AVR32 微控制器，带有一个或两个 DAC 输出。


# 3.核心算法原理及实现步骤
在本节中，我们将介绍如何使用 Vivado 生成 Verilog HDL 代码，用于构建一个定制化的音频输出系统。

## 3.1 MAME 系统架构
MAME 的系统架构图如下所示。


如上图所示，MAME 系统由多个组件构成，这些组件包括：

1. CPU：M68000 处理器，用于执行各个游戏程序。
2. SRAM：系统内存，大小为 128KB，用于存储各类代码和数据。
3. GROM：全局读写存储器，大小为 2MB，用于存储游戏图像、音乐、声效等。
4. OPM：音频模拟器，通过 SPI 协议与 AD9833 芯片通信。
5. Input：用户输入设备，如键盘、鼠标、触摸屏等。
6. Output：视频显示设备，如 CRT 液晶、LCD 等。
7. Networking：网络传输接口，提供网络连接。
8. Serial：串行接口，提供 USB 和串口通讯。

## 3.2 Verilog HDL 语言概览
### 3.2.1 数据类型
Verilog HDL 有两种基础的数据类型：

- 整型（integer type）：整型类型代表整数值，有 signed 和 unsigned 两种。signed 表示带符号整数，unsigned 表示无符号整数。如果没有明确指出 signed/unsigned 属性，则默认为 unsigned。
- 实数型（real type）：实数类型代表浮点数值。实数类型可以分为两种：单精度浮点数 real，double。

```verilog
wire [3:0] wire_type; // 无符号的32位线宽
reg signed [15:0] reg_s_type; // 带符号的16位寄存器
real r_type; // 单精度浮点数
```

### 3.2.2 端口与变量
Verilog HDL 代码中最重要的是端口和变量。端口定义了一个信号的方向，也就是信号从哪里进入，又从何处退出。变量可以用来存储数据、计数器或其他类型的信息。

Port 语法格式如下：

```verilog
input/output [<width>:<range>] <name>;
wire [<width>:<range>] <name>;
inout [<width>:<range>] <name>;
parameter [<width>] <name> = value;
```

- input：表示输入端口，只能用于模块的输入端，不能驱动模块的输出端。
- output：表示输出端口，只能用于模块的输出端，不能接收模块的输入端。
- inout：表示双向端口，既能作为输入端口，也能作为输出端口。
- parameter：表示参数，是模块化设计的核心特征之一。参数的值在编译时设置，不可变。

Varible 语法格式如下：

```verilog
<data_type> <name>[<size>];
initial begin
    <name> = <value>;
end
```

例子如下：

```verilog
module sample_module(
   input clk, 
   input reset, 
   input enable,
   output [7:0] led_state, 
   output spi_sclk, spi_mosi);

   integer i, j;
   reg [3:0] count;
   real pi = 3.1415926535;

   initial begin
      for (i=0; i<8; i=i+1)
        led_state[i] <= '0';
   end
endmodule
```

如上示例，sample_module 模块包含 input、output、inout、integer、real、initial 关键字，端口有 clk、reset、enable、spi_sclk、spi_mosi 六个，变量有 led_state、count、pi 三个。initial 语句用于设置 led_state 为初始状态。

### 3.2.3 时序逻辑表达式
Verilog HDL 中的时序逻辑表达式主要有以下几种：

1. 赋值语句：assign 可以给一位或多位连续变量赋值。如 assign dff_Q = DFF_D | ~dff_Qn。
2. 条件表达式：if...else、case...endcase 等。
3. 运算符：unary operator (+, -, ~, &~)，binary operator (++, --,!, &, |, ^, &&, ||, ==,!=, >, >=, <, <=, <<, >>, +, -, *, /)。
4. 函数调用：$display、$random、$strobe、$readmemh 等。

### 3.2.4 过程调用和函数调用
过程调用可以看作是函数的一种，它们的作用是在代码的某个位置引用另一个过程的代码段。函数调用是过程的一个集合，它以参数形式传入数据，执行完之后可以返回结果。

Verilog HDL 内置了很多预定义的过程，以 $ 开头，如 $display、$random、$strobe、$fclose 等。我们也可以自定义过程。过程语法格式如下：

```verilog
task <name>(<arg1>,<arg2>,...,<argn>);
  statements to be executed;
endtask

function <ret_type> <name>(<arg1>,<arg2>,...,<argn>);
  statements to be executed;
  return <expression>;
endfunction
```

task 与 function 的区别在于，前者不返回值，只用于执行某些任务；后者可以返回计算结果，一般用于运算。

例子如下：

```verilog
module blinkled(clk, reset, en, out);
   input wire clk, reset, en;
   output logic out;

   task delay();
      #10ns;
   endtask

   always @(posedge clk or negedge reset)
     if (!reset)
       out <= '0';
     else if (en)
       out <= ~out after delay();
endmodule
```

如上示例，blinkled 模块包含 input、output 两类端口，而且包含一个过程 blinkled::delay()，用于延迟输出状态。模块还有 always 块，用于在posedge clk 和 negedge reset 时触发输出状态的变化。

# 4.代码实例与仿真
## 4.1 生成 Verilog HDL 代码
本节将详细介绍如何利用 Vivado 创建一个定制化的音频输出系统。我们将创建一个模块，它接受 MAME 的音频输出、计数器、复位信号、输出缓冲器的控制信号，并根据这些信号生成我们需要的定制声音。

### 4.1.1 准备工作
#### 安装 Vivado 
Vivado 可以从 Xilinx 官网下载安装，请参考以下链接：https://www.xilinx.com/support/download.html

#### 安装 Altera Quartus Prime Lite Edition 
Quartus Prime 是一款高端的硬件设计工具，它可以用来创建多种类型的 FPGA 产品。

#### 添加 Quartus Prime Lite Edition 路径 
在命令行窗口中输入以下命令，添加 Quartus Prime Lite Edition 路径：

```shell
export PATH=$PATH:<quartus prime lite edition path>/bin
```

#### 设置 Vivado 默认路径 
在命令行窗口中输入以下命令，设置 Vivado 默认路径：

```shell
echo "source <vivado path>/settings64.sh" >> ~/.bashrc
source ~/.bashrc
```

### 4.1.2 创建新的项目
在 Vivado 中点击菜单栏的 Tools -> Run Tcl Console，打开 TCL 命令行界面，输入以下命令创建新的 Vivado 项目：

```tcl
create_project audio_system audio_system –part xc7a35tftg256-1
set_property board_part www.digilentinc.com:pynq-z2:part0:1.0 [current_project]
```

这里 `-part` 参数是指选择的 FPGA 板型。这里使用的 `xc7a35tftg256-1` 是一个 Zybo-Z7，代号 A35。

### 4.1.3 生成时钟
为了方便仿真，我们先生成时钟。在 TCL 命令行中输入以下命令：

```tcl
create_bd_design “top”
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]
generate_bd_layout
startgroup
create_bd_port -dir I -from 0 -to 0 irq
connect_bd_net [get_bd_ports irq] [get_bd_pins processing_system7_0/IRQ_F2P]
endgroup
regenerate_bd_layout
```

这里 `create_bd_design` 命令新建了一个 `top` 设计文件，`create_bd_cell` 命令新建了一个 IP 核 `processing_system7`，并自动完成配置，`-vlnv` 参数是指版本、库名、实例名。`apply_bd_automation` 命令可以把外接的资源分配到 IP 核上，`-rule` 参数是指规则，`-config` 参数是指配置项，`Master` 和 `Slave` 参数设置为禁止模式。`create_bd_port` 命令建立一个 irq 输入端口，并连接到 `processing_system7` 上面的 IRQ_F2P 信号上。

### 4.1.4 在 Vivado 中导入 Verilog 模块
创建好了工程，就可以在 Vivado 中导入 Verilog 模块了。点击菜单栏 File -> Add Sources -> Import Sources，选取已有的 Verilog 文件。此例中的 Verilog 模块的文件名为 `audio_gen.sv`。

### 4.1.5 配置 Verilog 模块属性
导入 Verilog 模块后，右击选择 `audio_gen` ，在弹出的窗口中选择 `Project Manager` 标签，在 `Project Properties` 下的 `Sources` 选项卡中修改 `File Type` 为 `Verilog HDL`。


### 4.1.6 在 Vivado 中编辑 Verilog 模块
编辑完毕后，我们就可以在 Vivado 中编辑 Verilog 模块了。打开 `audio_gen.sv` 文件，删掉所有的内容，只保留以下代码：

```verilog
module audio_gen #(
    parameter SAMPLE_RATE      = 44100,   // sampling rate of sound card
    parameter NUM_CHANNELS     = 2       // number of channels in stereo mix
)
(
    input wire        CLK,             // clock signal from sound card
    input wire        RST,             // asynchronous reset

    input wire        EN_SINUSOID,     // Enable sinusoid waveform generation
    input wire        ENABLE_SOUND,    // Start generating sound

    output wire [15:0] SOUND_DATA       // generated data stream for stereo channel
);
```

这里的 `#()` 后的参数表明了这个模块的参数，分别是：

1. `SAMPLE_RATE`：音频采样率，单位是 Hz。
2. `NUM_CHANNELS`：同时播放的声道数量，可以是 1 或 2。

模块的接口有三个，分别是：

1. `CLK`：时钟信号，来自音频采样器。
2. `RST`：异步复位信号，当产生异步复位时，所有缓冲器重置。
3. `EN_SINUSOID`：使能正弦波形发生器，只有这个信号有效的时候，才会产生正弦波形。
4. `ENABLE_SOUND`：开始播放音乐信号，只有这个信号有效时，才会开始播放声音。
5. `SOUND_DATA`：Stereo 通道输出的数据流。

### 4.1.7 将 Verilog 模块加入 Design Suit
将 `audio_gen.sv` 文件导入 Vivado 后，可以把它作为一个 Design Suit 添加到当前的工程中。点击菜单栏 Tools -> Create Design Suit，然后在弹出的窗口中选择 `Add file` 按钮，选择刚刚导入的 `audio_gen.sv` 文件，在 `Name` 一栏输入 `audio_gen`，然后点击 OK 按钮。


### 4.1.8 生成 Verilog 模块 RTL 文件
点击菜单栏 Tools -> Synthesis -> Run Synthesis，生成 RTL 文件。点击左边的 Project Manager 标签，在导航窗格中双击 `audio_gen` ，然后在右侧面板中找到 `Synthesis Status` 标签，观察 RTL 文件是否生成成功。如果出现红色的错误提示，则说明 RTL 文件无法正确生成。


如果 RTL 文件生成成功，就可以继续进行下一步的仿真了。

## 4.2 仿真
我们已经生成了 Verilog 模块的 RTL 文件，我们可以对它进行仿真，验证它的正确性。点击菜单栏 Simulation -> Run Behavioral Simulation，在弹出的窗口中点击 OK 按钮即可开始仿真。

仿真过程可能会比较耗时，请耐心等待。仿真完成后，我们可以查看仿真结果。仿真结果会显示一些信号的激励时间，以及信号的变化趋势。点击左边的 Simulation Results 标签，点击底部的 Chart View 按钮，然后双击对应的信号，就可以查看它的波形。

## 4.3 将 Verilog 模块与 FPGA 相连接
我们已经生成了 Verilog 模块的 RTL 文件，并且仿真验证过其正确性，下面就是将它与 FPGA 相连接了。

点击菜单栏 Tools -> Generate Bitstream，生成 bit 文件。生成完成后，就可以下载到开发板上进行验证了。

下载完成后，就可以在开发板上测试我们的 Verilog 模块了。首先将音频输出引脚连接到开发板上的 SPDIF 接口，然后将输入的对应音频信号输入到声卡的 ADC 接口。这样就可以听到 Verilog 模块生成的声音了。