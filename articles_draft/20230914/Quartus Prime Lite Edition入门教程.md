
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Quartus Prime是一个FPGA开发工具套件，其可编程逻辑设备的特性可用于实时控制系统、数据处理等多种领域。Quartus Prime Lite Edition是Quartus Prime的一款精简版本，可以进行极简设计、仿真和验证。因此，很多学生或初学者都在学习如何用Quartus进行FPGA开发时被这个简化版吓到。因此，我打算写一篇Quartus Prime Lite Edition入门教程，帮助更多的人快速上手Quartus。本教程分为四个章节，分别介绍了Quartus基础知识、逻辑与电路设计、综合与仿真、验证与调试等环节。
# 2.Quartus基础知识
## 2.1 Quartus概述
Quartus是一个集成开发环境（IDE），可用于对数字系统进行高层次综合、设计、实现和测试。Quartus包含如下功能模块：

1. Logic Synthesis（逻辑合成）：Quartus提供各种组合逻辑、时序逻辑、触发器、寄存器等逻辑资源的建模和优化。

2. HDL Design（硬件描述语言设计）：Quartus提供了VHDL、Verilog和System Verilog的硬件描述语言支持。用户可以使用这些语言编写程序，其中VHDL是一种面向过程的语言，而Verilog则是一种基于事件驱动的语言。

3. FPGA Build Tools（FPGA构建工具）：Quartus集成了IHP、Xilinx ISE、Altera Quartus II和Lattice Diamond等FPGA构建工具。用户可以使用这些工具编译生成相应的FPGA配置片段。

4. FPGA Deployment Tools（FPGA部署工具）：Quartus提供JTAG、USB、BootROM等多种方式进行FPGA的部署。用户可以在板卡上看到对应的输出结果。

5. Modelsim/QuestaSim Simulation Tools （模型仿真工具）：Quartus提供了Modelsim和QuestaSim两种模型仿真工具，可以对RTL代码进行逻辑仿真、时序仿真、波形检查等。

6. Documentation Generation Tool（文档生成工具）：Quartus还提供了文档生成工具，可以根据用户的需求自动生成包括仿真报告、数据手册、用户指导书、约束文件等文档。
## 2.2 Quartus术语与概念
### 2.2.1 文件类型与扩展名
| 文件类型 | 扩展名 |
| --- | --- |
| 工程文件 |.qpf、.qsf |
| VHDL源文件 |.vhd |
| Verilog源文件 |.v、.vo |
| System Verilog源文件 |.sv |
| 库文件 |.lib |
| 模型文件 |.cdf |
| 可执行文件 |.sof |
| 数据文件 |.dat |
| 文档文件 |.pdf |
### 2.2.2 常用命令与快捷键
| 命令与快捷键 | 功能 |
| --- | --- |
| File - New Project | 创建新工程 |
| File - Open Project | 打开已有工程 |
| File - Save | 保存当前工程 |
| Edit - Undo | 撤销最后一个操作 |
| Edit - Redo | 恢复最后一次撤销操作 |
| Edit - Cut | 剪切选中的文字 |
| Edit - Copy | 复制选中的文字 |
| Edit - Paste | 粘贴剪切板的内容 |
| Edit - Delete | 删除选中的文本 |
| Edit - Find and Replace | 查找替换文本 |
| View - Zoom In | 放大视图 |
| View - Zoom Out | 缩小视图 |
| Window - Split Horizontally | 分割窗口水平 |
| Window - Split Vertically | 分割窗口竖直 |
| Run - Run Design | 运行设计 |
| Run - Stop Running | 中止正在运行的仿真 |
| Tools - Message Manager | 打开消息管理器 |
| Tools - Waveform Viewer | 打开波形查看器 |
| Help - Quartus Help | 打开Quartus帮助中心 |
| Help - Release Notes | 查看Quartus最新更新情况 |
### 2.2.3 逻辑块与信号
逻辑块是指逻辑功能单元，如加法器、多路选择器等；信号是指数据信号，如输入数据、输出结果、控制信号等。下表列出了一些常用的逻辑块及其所包含的信号。

| 逻辑块 | 信号 | 描述 |
| --- | --- | --- |
| D-type flip-flop | Q (output)、D (input)、CK (clock)、CLR (clear) | 时钟控制的非同步门控开关（锁存器）。 |
| JK-type flip-flop | J (join input)、K (join output)、CK (clock)、CLR (clear) | 时钟控制的与非门/或非门控开关。 |
| T-type flip-flop | T (triggle input)、Q (output)、CK (clock)、CLR (clear) | 时钟控制的触发器。 |
| SR-latch | S (set input)、R (reset input)、Q (output)、CLK (clock) | 时钟控制的SR型寄存器。 |
| DFFRAM | WE (write enable input)、DI (data input)、DQN (negative clock edge read data output)、DQP (positive clock edge read data output)、A[7:0] (address bus)、BA[1:0] (bank address)、CE (chip enable input)、OE (output enable input)、CLK (clock) | 时钟控制的DRAM。 |
| Mux | A (select input)、B (input 1)、C (input 2)、D (output) | 多路选择器。 |
| Adder | A[n-1:0] (data input 1)、B[n-1:0] (data input 2)、CI (carry input)、S[n+1:0] (sum output)、CO (carry output) | 加法器。 |
| Multiplier | A[n-1:0] (data input 1)、B[n-1:0] (data input 2)、P[m*n-1:0] (product output) | 乘法器。 |
| Register-transfer-level(RTL) | Inputs、Outputs、Wires、Assignments、Constants、Assignments| RTL语言的语句结构。 |
### 2.2.4 赋值运算符
Quartus中有三种赋值运算符：

1. <= : 从右向左赋值（通常表示赋值）。

2. => : 从左向右赋值（通常表示隐含地使用assign语句）。

3. = : 深层复制。

以下是一些赋值例子：

1. a <= b; // 将变量b的值赋给变量a。

2. c = d; // 对变量c的重新赋值，将d的深层拷贝赋给c。

3. if (en == '1') begin // 如果使能信号en为'1'，则执行以下语句：
   a => b + 1; // 将变量b加1的结果赋给变量a。
end else begin
   a => {others => '0'}; // 将变量a全置零。
end
### 2.2.5 时序逻辑语法
Quartus中时序逻辑语法如下图所示。


下面通过几个例子演示时序逻辑语法：

1. #5 reg q; // 在5ns内创建一个寄存器。

2. always @ (posedge clk or negedge rst) begin // 当posedge触发clk或negedge触发rst时进入always块。
    if (!rst) // 当rst为负时清除寄存器的值。
        q <= '0'; 
    else 
        q <= in; // 在其他情况下将in的值赋给寄存器。
end

3. assign out = (q > threshold); // 判断q是否大于阈值，并将判断结果赋给out。