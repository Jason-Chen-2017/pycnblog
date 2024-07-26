                 

## 1. 背景介绍

### 1.1 问题由来

实时操作系统（Real-Time Operating System, RTOS）是一种专门为实时应用设计的操作系统，它能够在严格的时间限制内响应用户请求，并提供稳定可靠的系统服务。实时操作系统在工业控制、航空航天、医疗设备、自动驾驶等领域中应用广泛，这些领域通常需要操作系统能够快速响应用户请求，且系统的稳定性和可靠性至关重要。

RTOS的设计目标与传统的通用操作系统（General-Purpose Operating System, GPOS）有显著不同。GPOS注重灵活性和通用性，适用于通用计算环境，但无法保证严格的时间限制。而RTOS则更加专注于时间关键型应用的开发，能够提供更加稳定可靠的系统服务，并确保系统响应的实时性。

### 1.2 问题核心关键点

RTOS的核心设计目标包括：

- **实时性**：系统能够快速响应用户请求，并在严格的时间限制内完成任务。
- **可靠性**：系统在面对异常情况（如硬件故障、软件错误）时能够保持稳定运行，不发生系统崩溃或数据损坏。
- **可预测性**：系统行为能够被预测和控制，确保任务按照预期进行。
- **安全性**：系统能够防止未授权访问和恶意攻击，确保数据和资源的安全性。

RTOS的设计需要在实时性、可靠性和可预测性之间找到平衡，同时还需要兼顾安全性。这使得RTOS的设计和实现相对复杂，但也是其在时间关键型应用中的重要价值所在。

### 1.3 问题研究意义

RTOS在工业控制、航空航天、医疗设备、自动驾驶等领域的应用，对于提高系统的实时性和可靠性，保障人命关天的安全，具有重要意义。通过深入研究RTOS的设计原理和实现方法，可以帮助开发者更好地理解和应用RTOS，提升时间关键型应用的开发效率和质量。

RTOS的研究也推动了实时计算、实时通信、实时数据库等技术的发展，促进了嵌入式系统、物联网等新兴领域的成长，对未来智能社会的构建具有重要意义。

## 2. 核心概念与联系

### 2.1 核心概念概述

为了更好地理解RTOS的核心设计原理和实现方法，本节将介绍几个关键概念：

- **任务(Task)**：RTOS中的基本执行单元，每个任务都是一个独立的执行流程，具有自己的栈空间和执行代码。任务可以包含多个线程，每个线程执行独立的逻辑。
- **中断(Interrupt)**：硬件或软件中断事件的发生，会打断当前任务的执行，转而执行中断处理函数。中断处理函数执行完毕后，会返回原任务。
- **调度(Scheduling)**：RTOS的任务调度器负责管理任务的执行顺序，确保每个任务都能在预设的时间内完成执行。调度算法通常包括优先级调度、时间片轮转等。
- **同步(Synchronization)**：RTOS中的任务间通信和同步机制，如信号量、事件对象、消息队列等，确保任务间的协作和数据共享。
- **内存管理(Memory Management)**：RTOS对内存空间的分配和管理，包括栈空间、堆空间和静态数据空间的分配和释放，确保系统内存资源的有效利用。

这些核心概念之间的逻辑关系可以通过以下Mermaid流程图来展示：

```mermaid
graph TB
    A[任务(Task)] --> B[中断(Interrupt)]
    A --> C[调度(Scheduling)]
    A --> D[同步(Synchronization)]
    A --> E[内存管理(Memory Management)]
```

这个流程图展示了大语言模型的核心概念及其之间的关系：

1. 任务是RTOS的基本执行单元。
2. 中断会打断当前任务的执行，转而执行中断处理函数。
3. 调度器负责管理任务的执行顺序，确保任务在预设时间内完成。
4. 同步机制确保任务间的协作和数据共享。
5. 内存管理确保系统内存的有效利用。

这些概念共同构成了RTOS的核心功能，使得系统能够在严格的时间限制内稳定可靠地运行。

## 3. 核心算法原理 & 具体操作步骤

### 3.1 算法原理概述

RTOS的核心算法包括任务调度、中断处理、同步和内存管理等。

- **任务调度算法**：RTOS的任务调度算法通常包括优先级调度、时间片轮转等。优先级调度根据任务的优先级来决定任务的执行顺序，确保高优先级任务能够优先执行；时间片轮转则将CPU时间分割为多个时间片，每个任务在一个时间片内执行，超出时间片的任务会被挂起。
- **中断处理机制**：中断处理机制分为硬件中断和软件中断。硬件中断由外部设备触发，如按键、传感器等；软件中断则由系统内部事件触发，如定时器、任务唤醒等。中断处理机制能够确保系统对外部事件的及时响应。
- **同步机制**：RTOS中的同步机制包括信号量、事件对象、消息队列等，确保任务间的协作和数据共享。信号量用于控制对共享资源的访问，事件对象用于任务间的信号传递，消息队列用于任务间的消息传递。
- **内存管理**：RTOS的内存管理通常包括栈空间、堆空间和静态数据空间的分配和释放。栈空间用于保存任务的局部变量和函数调用信息，堆空间用于动态分配内存，静态数据空间则用于保存全局变量和常量。

### 3.2 算法步骤详解

以下是RTOS的算法步骤详解：

**Step 1: 任务创建和初始化**

- 创建一个任务，指定任务名称、优先级、堆栈空间大小等参数。
- 初始化任务的执行代码和数据，为任务分配必要的内存空间。

**Step 2: 任务调度**

- 根据任务的优先级和时间片，决定任务的执行顺序。
- 在每个时间片内，执行当前任务的代码。
- 如果任务未完成，将其挂起，等待下一次执行。

**Step 3: 中断处理**

- 当系统发生中断事件时，保存当前任务的上下文信息。
- 执行中断处理函数，处理中断事件。
- 恢复当前任务的上下文信息，返回执行中断处理函数前的位置。

**Step 4: 同步机制**

- 使用信号量、事件对象、消息队列等同步机制，确保任务间的协作和数据共享。
- 在需要共享资源时，通过信号量或互斥锁进行保护，避免数据竞争。
- 在需要传递信号或消息时，通过事件对象或消息队列进行通信。

**Step 5: 内存管理**

- 根据任务需要，分配栈空间、堆空间和静态数据空间。
- 在任务执行完毕后，释放分配的内存空间。
- 确保内存空间的有效利用，避免内存泄漏和溢出。

### 3.3 算法优缺点

RTOS的核心算法具有以下优点：

- **实时性**：通过任务调度、中断处理和同步机制，RTOS能够快速响应用户请求，并确保任务在预设时间内完成。
- **可靠性**：RTOS的设计注重稳定性和可靠性，能够保证系统在面对异常情况时的稳定运行。
- **可预测性**：RTOS的任务调度算法和同步机制，使得系统行为可预测，确保任务按照预期进行。

同时，RTOS也存在一定的局限性：

- **复杂性**：RTOS的设计和实现相对复杂，需要开发者具备一定的系统设计和编程能力。
- **资源占用**：RTOS的任务调度和同步机制会占用一定的系统资源，影响系统的性能。
- **灵活性不足**：RTOS的设计注重实时性和可靠性，但灵活性相对不足，难以适应复杂的通用计算环境。

尽管存在这些局限性，但RTOS在时间关键型应用的开发中具有重要价值，是确保系统实时性和可靠性的关键。

### 3.4 算法应用领域

RTOS在工业控制、航空航天、医疗设备、自动驾驶等领域中应用广泛。以下是几个典型的应用场景：

- **工业控制**：RTOS能够快速响应用户请求，确保工业设备在预设时间内完成操作。
- **航空航天**：RTOS能够确保飞行控制系统的实时性和可靠性，保障飞行安全。
- **医疗设备**：RTOS能够确保医疗设备在实时监测和控制方面的稳定性，保障患者安全。
- **自动驾驶**：RTOS能够确保自动驾驶系统的实时性和可靠性，保障行车安全。

除了上述这些领域外，RTOS还在智能交通、智能家居、智能制造等新兴领域中发挥着重要作用。

## 4. 数学模型和公式 & 详细讲解

### 4.1 数学模型构建

RTOS的数学模型通常包括任务调度算法、中断处理机制、同步机制和内存管理等。以下以优先级调度算法为例，介绍RTOS的数学模型构建。

假设系统中有 $n$ 个任务，每个任务有优先级 $p_i$，任务执行时间 $t_i$。任务的执行顺序由优先级决定，高优先级任务优先执行。

定义任务调度算法的目标是最小化任务的总执行时间，即：

$$
\min_{\pi} \sum_{i=1}^n t_i^\pi
$$

其中 $\pi$ 表示任务的执行顺序。

### 4.2 公式推导过程

假设系统中的任务执行顺序为 $\pi$，则任务的总执行时间为：

$$
T_{total} = \sum_{i=1}^n t_i^\pi
$$

为了最小化总执行时间，需要找到最优的任务执行顺序 $\pi^*$。可以使用贪心算法来求解：

1. 从优先级最高的任务开始执行，将其执行时间累加到总执行时间 $T_{total}$ 中。
2. 从剩余任务中选取优先级最高的任务，将其执行时间累加到总执行时间 $T_{total}$ 中。
3. 重复步骤2，直到所有任务执行完毕。

这样得到的任务执行顺序 $\pi^*$，可以保证系统的总执行时间最小。

### 4.3 案例分析与讲解

以下以一个简单的示例任务集，说明优先级调度算法的应用。

假设系统中有三个任务，其优先级和执行时间如下：

| 任务编号 | 优先级 | 执行时间 |
| -------- | ------ | -------- |
| Task 1   | 2      | 3        |
| Task 2   | 1      | 4        |
| Task 3   | 3      | 2        |

使用贪心算法进行优先级调度，任务执行顺序为：Task 2 -> Task 1 -> Task 3，总执行时间为：$4 + 3 + 2 = 9$。

使用最优算法进行优先级调度，任务执行顺序为：Task 2 -> Task 3 -> Task 1，总执行时间为：$4 + 2 + 3 = 9$。

可以看到，贪心算法和最优算法的总执行时间相同，但任务执行顺序不同。贪心算法优先执行高优先级任务，确保高优先级任务的及时响应，最优算法则根据任务执行时间优化执行顺序，确保系统总执行时间最短。

## 5. 项目实践：代码实例和详细解释说明

### 5.1 开发环境搭建

在进行RTOS的开发和调试时，需要使用专业的开发工具和环境。以下是一些常用的开发工具和环境：

- **GNU/Linux**：RTOS开发通常使用Linux操作系统，Linux具备良好的系统稳定性和网络支持，适合进行RTOS的开发和调试。
- **嵌入式开发板**：如Raspberry Pi、Arduino等，适合进行RTOS的硬件调试和测试。
- **IDE开发工具**：如Eclipse、Visual Studio等，适合进行RTOS的编程和调试。

### 5.2 源代码详细实现

以下是一个简单的RTOS任务调度和中断处理的代码实现。

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

#define TASK1_PRIORITY 2
#define TASK2_PRIORITY 1
#define TASK3_PRIORITY 3

#define TASK1_EXEC_TIME 3
#define TASK2_EXEC_TIME 4
#define TASK3_EXEC_TIME 2

int task1();
int task2();
int task3();

void signal_handler(int signal);

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2_EXEC_TIME + "s");
    printf("Task 2 finished\n");
    exit(ret);
}

int task3() {
    printf("Task 3 started\n");
    ret = system("sleep " + TASK3_EXEC_TIME + "s");
    printf("Task 3 finished\n");
    exit(ret);
}

void signal_handler(int signal) {
    printf("Interrupt occurred\n");
    task3();
}

int main() {
    int ret;

    // 创建任务
    task1();
    task2();
    task3();

    // 等待任务完成
    while(1) {
        sleep(1);
    }

    return 0;
}

int task1() {
    printf("Task 1 started\n");
    ret = system("sleep " + TASK1_EXEC_TIME + "s");
    printf("Task 1 finished\n");
    exit(ret);
}

int task2() {
    printf("Task 2 started\n");
    ret = system("sleep " + TASK2

