
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Vulkan是一个基于Khronos组织开发的高性能跨平台图形API，其特点在于高度可移植性、异步计算、无需驱动支持且具有真正意义上的通用性。现在越来越多的游戏引擎也转向了Vulkan作为渲染API，使得3D图形渲染领域有望拥有统一的标准接口。基于Vulkan开发图形应用程序可以降低开发难度、提升开发效率，进而为广大的游戏玩家带来全新的视觉体验。
Vulkan的特点包括：

1. 可移植性: 在多个平台上都能运行，且可以和其他API同时共存。Vulkan最大的优点之一就是它的跨平台特性，能确保应用的兼容性，能够让开发者只编写一次代码就可以轻松部署到多个不同平台。

2. 高性能: 使用Vulkan构建出来的应用能提供高质量的渲染效果。Vulkan提供了各种高级功能，如图形管线管理、硬件加速计算等，让开发者能快速实现出精美、流畅的3D场景。

3. 异步计算: 异步计算（asynchronous computing）的概念可以让GPU执行指令的时间更长一些，并不会影响游戏画面显示的速度。Vulkan支持异步命令处理，它能充分利用CPU资源，提升渲染效率。

4. 真正意义上的通用性: Vulkan除了能进行3D图形渲染外，还可以用于其他类型的图形处理任务。例如，在游戏中还可以使用Vulkan实现动态光照、阴影、粒子系统等效果；再比如，在移动端设备上还可以利用Vulkan开发游戏引擎、渲染视频、图像处理等高性能图形加速应用。

总结一下，Vulkan是一种基于Khronos组织开发的高性能、跨平台图形API，具备高度可移植性、异步计算、真正意义上的通用性。目前，Vulkan已成为主流的图形API，并且有望成为下一个通用的图形API标准。

本文将围绕Vulkan图形学的基本知识和理论知识展开，重点介绍Vulkan图形学的基本概念、结构、用途及基本原理。对于希望了解Vulkan图形学底层原理及其技术应用的人士，本文也是一份必读的好文章。文章会从整体结构出发，先讨论Vulkan的历史发展、特性、架构设计以及适用范围，然后详细介绍Vulkan主要模块、概念及使用方法，最后将Vulkan的未来展望放在前面。

本文涉及到的主题有：

1.Vulkan概述
2.Vulkan架构设计
3.Vulkan工作原理及使用方法
4.Vulkan中的图形管线及流程
5.Vulkan中的物理渲染
6.Vulkan中的基于物理的渲染
7.Vulkan中的阴影映射
8.Vulkan中的基于物理的阴影渲染
9.Vulkan中的屏幕后期处理
10.Vulkan中的Vulkan GLES混合编程
11.Vulkan中的Vulkan VKSL着色语言
12.Vulkan中的Vulkan工具链及工具
13.Vulkan的未来发展方向
# 2.Vulkan概述

## 2.1 概述

Vulkan是一款由Khronos组织发布的开源图形API，是一套跨平台的3D图形渲染API标准。其目的是用来建立一套易于开发、易于使用的跨平台API，其中描述了图形应用所需的各种底层组件以及它们之间的交互关系。Vulkan API基于Vulkan规范（https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html），使用C语言编写。

Vulkan官方网站：https://www.khronos.org/vulkan

## 2.2 特点

### 2.2.1 高度可移植性

Vulkan的跨平台特性是最吸引人的特性之一。它不仅可以运行在多种不同的平台上，而且还可以在移动设备上跑起来。这样一来，开发者就不用自己重新造轮子了。 

Vulkan设计时就考虑到了移植性，所以它很好的解决了平台差异的问题。不需要通过驱动来实现跨平台支持。相反，它通过将所有的图形操作委托给显卡，使得应用能很好的和各种不同型号显卡互相配合，提升了用户的体验。

### 2.2.2 异步计算

Vulkan采用异步计算的方式，使得渲染操作能够更加高效。通过将命令提交给GPU，提交后的立即返回，之后命令会被放入一个队列中等待GPU的执行。因此，当多个命令需要同时执行的时候，由于渲染器只有一个线程来处理它们，因此执行效率就会变得更高。

### 2.2.3 真正意义上的通用性

Vulkan虽然不是唯一的一款图形API，但它已经成为主流的3D图形API标准。现在很多图形硬件厂商都开始支持Vulkan这种跨平台的API。通过这种方式，开发者不仅可以方便地移植自己的应用，而且也可以享受到社区及硬件厂商的成果。

另外，Vulkan不但支持3D图形渲染，还可以用于渲染其他类型的内容，如模拟类、可视化、动画等。

### 2.2.4 丰富的特性

Vulkan提供了诸如多样化的命令集、统一的内存模型、多级拓扑结构、跨越式采样、计算 shaders、渲染 passes、基于物理的渲染、阴影映射等诸多特性。

## 2.3 发展历史

Vulkan是在2016年由英伟达发起，2017年5月1日正式加入Khronos组织正式发布。它的第一版Vulkan规范于2016年6月发布，当前版本为1.0。在过去几年中，Vulkan经历了一系列的迭代，每个版本都推出了新特性和更新。Vulkan 1.1于2018年11月发布，引入了新的扩展接口VK_EXT_descriptor_indexing以及其他扩展。此外，在Vulkan 1.2中引入了新的Swapchain API，实现了应用程序无缝切换窗口大小、模式等内容。

目前，Vulkan的主流桌面平台有Windows，Linux，Android和macOS。其次，Google正在尝试把Vulkan加入安卓系统。苹果也计划在iOS上使用Vulkan来构建自己的框架Metal。微软正在探索将Vulkan带入Xbox One和HoloLens。

除此之外，还有许多的第三方图形API和引擎，如Godot，Unreal Engine，Ogre，Lumberyard，Urho3D，Nebula等，都基于Vulkan实现。

# 3.Vulkan架构设计

## 3.1 架构组成

Vulkan的架构主要分为三大部分：

1. 驱动层(Driver Layer): 负责驱动具体的硬件，包括设备的初始化、图形对象的创建、内存分配、数据传输等操作。
2. API层(Application Programming Interface): 实现了Vulkan API规范，定义了图形应用应该如何调用Vulkan的各个功能，包括图像创建、绘制、绑定资源、同步和查询等。
3. 运行时库层(Runtime Library): 实现了Vulkan各个功能，包括内存管理、设备管理、对象管理、窗口管理、状态机管理、编译器管理等功能。

## 3.2 设备驱动层

设备驱动层位于驱动程序和GPU之间，其作用是隐藏底层平台驱动和GPU硬件之间的差异，并提供跨平台的一致的图形接口。通常来说，驱动程序会负责从系统内存中分配缓冲区、图像和纹理对象，处理相关的底层操作，如创建窗口、命令缓冲区等。

## 3.3 API层

API层位于API客户端和驱动程序之间，其职责是实现Vulkan API规范。API层接收用户的请求，将它们转换为适合底层驱动的指令序列，并将这些指令序列传送至驱动层。API层还会管理底层资源，比如缓冲区、图像和纹理，以及将资源绑定到当前的渲染或计算操作。

## 3.4 运行时库层

运行时库层的职责就是实现Vulkan API规范中定义的所有功能，包括内存管理、设备管理、对象管理、窗口管理、状态机管理、编译器管理等功能。运行时库层提供了一系列的底层功能，让Vulkan API成为应用开发者熟悉的东西。比如，运行时库层对Vulkan API进行了封装，使得开发者使用起来更容易，并隐藏了内部的复杂性。运行时库层还提供了很多便利的接口，比如VkQueueSubmit()函数允许用户提交多个待执行的命令，并指定它们是否依赖于某些特定事件。

# 4.Vulkan工作原理及使用方法

Vulkan的使用主要分为如下几个步骤：

1. 创建实例：首先创建一个Vulkan的实例对象，它代表了一个Vulkan运行环境，并且管理着所有可用的Vulkan驱动程序和图形设备。
2. 检测设备：根据Vulkan驱动程序和系统的信息，查询可用的图形设备，选择一个适合的设备供应用使用。
3. 创建设备：创建一个Vulkan设备对象，它代表了一个图形处理单元(GPU)。
4. 创建提交队列：创建一个提交队列，它包含所有待执行的CommandBuffer。
5. 创建CommandBuffer：创建一个CommandBuffer，它代表了一段图形渲染指令序列。
6. 记录指令：将渲染指令记录到CommandBuffer中。
7. 提交CommandBuffer：提交CommandBuffer给对应的提交队列。
8. 执行渲染：当所有的CommandBuffer都被提交完毕后，图形设备会执行相应的指令序列。
9. 渲染结果：渲染完成后，Vulkan驱动程序会将结果数据从GPU复制回主机内存中，并显示到屏幕上。

接下来，我们将依次详细介绍每一步的功能。

## 4.1 创建实例

Vulkan的实例对象代表了整个运行环境。在创建实例之前，必须先确定系统中的Vulkan驱动程序是否可用。若驱动不可用，则无法创建实例。

实例对象的创建可以通过vkCreateInstance()函数来完成。这个函数需要传递一个VkInstanceCreateInfo结构参数，用于配置实例的属性。如图所示：

```c++
VkInstanceCreateInfo instanceInfo = {};
instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; // 指定结构类型
instanceInfo.pApplicationInfo = &appInfo;                    // 设置应用信息
// 添加需要启用的扩展名
const char *enabledExtensions[] = { "VK_KHR_surface",
                                    "VK_KHR_win32_surface" };
instanceInfo.ppEnabledExtensionNames = enabledExtensions;
instanceInfo.enabledExtensionCount = sizeof(enabledExtensions) /
                                      sizeof(enabledExtensions[0]);
// 创建实例对象
if (vkCreateInstance(&instanceInfo, nullptr, &instance)!= VK_SUCCESS) {
    throw std::runtime_error("failed to create instance!");
}
```

这里，我们创建了一个空的VkInstanceCreateInfo结构变量，设置了实例的类型为VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO。对于应用信息，我们传递了一个VkApplicationInfo结构。为了启用Win32窗口系统的支持，我们添加了VK_KHR_surface和VK_KHR_win32_surface两个扩展名。

接下来，我们调用vkCreateInstance()函数，传入刚才的实例创建信息和指针。如果函数执行成功，则会将创建的实例对象存储在instance变量中。

## 4.2 检测设备

Vulkan驱动程序检测到可用的设备后，就会生成一个数组，里面包含了设备的各种属性，如设备名称、类型、核数、属性等。每一个设备都有一个唯一标识符VkPhysicalDevice，我们可以通过vkGetPhysicalDeviceProperties()函数获取到该设备的详细属性。如图所示：

```c++
uint32_t deviceCount = 0;
vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);    // 获取设备数量
std::vector<VkPhysicalDevice> devices(deviceCount);               // 创建设备数组
vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());   // 获取设备列表
for (int i = 0; i < deviceCount; ++i) {                           // 遍历设备列表
    vkGetPhysicalDeviceProperties(devices[i], &deviceProps);     // 获取设备属性
    if (strstr(deviceProps.deviceName, "GeForce")) {            // 判断是否为GeForce
        physicalDevice = devices[i];                             // 保存GeForce设备
        break;                                                   // 跳出循环
    }
}
```

这里，我们通过vkEnumeratePhysicalDevices()函数来枚举系统中的设备，并得到设备的数量。然后，我们创建了一个设备数组，并使用vkGetPhysicalDeviceProperties()函数获取每一个设备的属性。在遍历设备列表的过程中，我们检查设备名称是否含有“GeForce”字符串，如果找到，就保存这一块设备，并退出循环。最后，我们将该设备存储在physicalDevice变量中。

## 4.3 创建设备

设备的创建可以通过vkCreateDevice()函数来完成。这个函数需要传递一个VkDeviceCreateInfo结构参数，用于配置设备的属性。如图所示：

```c++
VkDeviceCreateInfo deviceInfo = {};
deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;      // 指定结构类型
deviceInfo.queueCreateInfoCount = 1;                         // 配置1个队列
deviceInfo.pQueueCreateInfos = &queueCreateInfo;              // 设置队列信息
deviceInfo.pEnabledFeatures = &enabledFeatures;                // 设置可用特征
deviceInfo.enabledLayerCount = 0;                            // 不使用任何图层
if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr,
                    &device)!= VK_SUCCESS) {                  // 创建设备对象
    throw std::runtime_error("failed to create logical device!");
}
```

这里，我们创建了一个VkDeviceCreateInfo结构变量，设置了设备的类型为VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO。队列创建信息的数量设置为1，也就是创建一个队列。我们配置一个VkQueueCreateInfo结构，用于描述设备的渲染能力。启用了1个队列，且队列的属性设置为VK_QUEUE_GRAPHICS_BIT，表示它可以执行图形渲染任务。

接下来，我们调用vkCreateDevice()函数，传入设备创建信息和指针。如果函数执行成功，则会将创建的设备对象存储在device变量中。

## 4.4 创建提交队列

提交队列的创建可以通过vkCreateCommandPool()函数来完成。这个函数需要传递一个VkCommandPoolCreateInfo结构参数，用于配置命令池的属性。如图所示：

```c++
VkCommandPoolCreateInfo poolInfo = {};
poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;        // 指定结构类型
poolInfo.queueFamilyIndex = graphicsFamilyIdx;                     // 选择Graphics队列
poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // 复用命令缓冲区
if (vkCreateCommandPool(device, &poolInfo, nullptr,
                        &commandPool)!= VK_SUCCESS) {           // 创建命令池
    throw std::runtime_error("failed to create command pool!");
}
```

这里，我们创建了一个VkCommandPoolCreateInfo结构变量，设置了命令池的类型为VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO。队列族索引设置为graphicsFamilyIdx，表示要使用的图形队列。为了减少GPU资源的消耗，我们设置了VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT标志，这样当命令缓冲区过期时，它会自动重置。

接下来，我们调用vkCreateCommandPool()函数，传入命令池创建信息和指针。如果函数执行成功，则会将创建的命令池对象存储在commandPool变量中。

## 4.5 创建CommandBuffer

CommandBuffer的创建可以通过vkAllocateCommandBuffers()函数来完成。这个函数需要传递一个VkCommandBufferAllocateInfo结构参数，用于配置命令缓冲区的属性。如图所示：

```c++
VkCommandBufferAllocateInfo allocInfo = {};
allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;    // 指定结构类型
allocInfo.commandPool = commandPool;                                // 指定命令池
allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;                   // 为主CommandBuffer
allocInfo.commandBufferCount = 1;                                   // 分配1个CommandBuffer
if (vkAllocateCommandBuffers(device, &allocInfo,
                            &commandBuffer)!= VK_SUCCESS) {       // 创建命令缓冲区
    throw std::runtime_error("failed to allocate command buffers!");
}
```

这里，我们创建了一个VkCommandBufferAllocateInfo结构变量，设置了命令缓冲区的类型为VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO。命令池设置为commandPool，主CommandBuffer级别为VK_COMMAND_BUFFER_LEVEL_PRIMARY，表示它在首次提交时不能被重用。我们只分配1个CommandBuffer。

接下来，我们调用vkAllocateCommandBuffers()函数，传入命令缓冲区分配信息和指针。如果函数执行成功，则会将创建的CommandBuffer对象存储在commandBuffer变量中。

## 4.6 记录指令

CommandBuffer对象的创建并不意味着它能执行渲染指令。它只是代表了一段待执行的渲染指令序列，还没有实际提交给GPU去执行。

我们可以通过vkCmdXXX()系列函数来记录渲染指令。vkCmdBindPipeline()用于绑定一个渲染管线对象， vkCmdSetViewport()用于设置视口， vkCmdDraw()用于绘制顶点和索引。如图所示：

```c++
VkCommandBufferBeginInfo beginInfo = {};
beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;         // 指定结构类型
beginInfo.flags = 0;                                                  // 命令缓冲区初始状态
beginInfo.pInheritanceInfo = nullptr;                                 // 不使用继承信息
vkBeginCommandBuffer(commandBuffer, &beginInfo);                      // 启动CommandBuffer

vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                  pipeline);                                              // 绑定渲染管线
vkCmdSetViewport(commandBuffer, 0, 1, &viewport);                      // 设置视口
vkCmdSetScissor(commandBuffer, 0, 1, &scissor);                        // 设置裁剪矩形

vkCmdDraw(commandBuffer, 3, 1, 0, 0);                                  // 绘制一个三角形

vkEndCommandBuffer(commandBuffer);                                    // 结束CommandBuffer
```

这里，我们创建了一个VkCommandBufferBeginInfo结构变量，设置了命令缓冲区的类型为VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO。我们只需要用0作为flags字段值即可。

接下来，我们调用vkBeginCommandBuffer()函数，启动commandBuffer。在启动之后，我们调用vkCmdXXX()系列函数来记录渲染指令。命令缓冲区的结束由vkEndCommandBuffer()函数来完成。

## 4.7 提交CommandBuffer

CommandBuffer只能执行一次，所以它需要提交给GPU去执行才能产生实际的效果。Vulkan中，提交CommandBuffer有两种方式。第一种方式是直接提交，第二种方式是提交到一个VkFence对象上，然后等待该Fence信号，才会告诉GPU执行CommandBuffer。

提交CommandBuffer有两种方式。第一种方式是直接提交，第二种方式是提交到一个VkFence对象上，然后等待该Fence信号，才会告诉GPU执行CommandBuffer。

提交CommandBuffer的过程可以通过vkQueueSubmit()函数来完成。这个函数需要传递一个VkSubmitInfo结构参数，用于配置提交信息。如图所示：

```c++
VkSubmitInfo submitInfo = {};
submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;           // 指定结构类型
submitInfo.commandBufferCount = 1;                       // 待执行的CommandBuffer数量
submitInfo.pCommandBuffers = &commandBuffer;             // 设置CommandBuffer
submitInfo.signalSemaphoreCount = 0;                     // 不使用信号量
submitInfo.waitSemaphoreCount = 0;                       // 不使用等待信号
if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                 fence)!= VK_SUCCESS) {                    // 提交到Graphics队列
    throw std::runtime_error("failed to submit draw command buffer!");
}
```

这里，我们创建了一个VkSubmitInfo结构变量，设置了提交信息的类型为VK_STRUCTURE_TYPE_SUBMIT_INFO。待执行的CommandBuffer数量设置为1，commandBuffer指向刚才创建的那个CommandBuffer对象。我们不使用任何信号量或者等待信号。

接下来，我们调用vkQueueSubmit()函数，传入提交信息和Fence指针。如果函数执行成功，则会将CommandBuffer对象提交到图形队列。

## 4.8 执行渲染

所有的CommandBuffer对象都被提交到Graphics队列后，图形设备会按照提交顺序依次执行它们。当所有的CommandBuffer都被执行完毕后，会通知应用程序，并且开始渲染过程。

渲染完成后，渲染结果数据会被从GPU复制到主机内存中。Vulkan提供了vkMapMemory()和vkUnmapMemory()函数，用于映射和取消映射内存，从而访问和修改GPU内存中的数据。渲染结果数据将通过展示管线上传到屏幕上。

## 4.9 渲染结果

Vulkan驱动程序会将渲染结果数据从GPU复制到主机内存中。渲染结果数据将通过展示管线上传到屏幕上。展示管线会决定如何显示渲染结果数据。展示管线通常包含各种阶段，如输入 assembler、vertex shader、tessellation control shader、tessellation evaluation shader、geometry shader、rasterizer、fragment shader、color blender、output merger等。

渲染结果数据的展示可以由调用vkQueuePresentKHR()函数来完成。这个函数需要传递一个VkPresentInfoKHR结构参数，用于配置渲染结果的数据。如图所示：

```c++
VkSwapchainKHR swapChains[] = { presentInfo.swapchain };          // 使用的Swapchain
VkPresentInfoKHR presentInfo = {};
presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;          // 指定结构类型
presentInfo.waitSemaphoreCount = 0;                              // 不使用等待信号
presentInfo.swapchainCount = 1;                                  // Swapchain数量
presentInfo.pWaitSemaphores = nullptr;                           // 不使用等待信号
presentInfo.pResults = nullptr;                                  // 不使用结果值
presentInfo.pImageIndices = &imageIndex;                          // 当前帧图片索引
presentInfo.swapchain = swapChains[frameIndex % SWAPCHAIN_IMAGE_COUNT]; // 当前Swapchain
if (vkQueuePresentKHR(presentQueue,
                       &presentInfo) == VK_ERROR_OUT_OF_DATE_KHR ||
    framebufferResized) {                                       // Swapchain被重建
    recreateSwapchain = true;                                    // 需要重建Swapchain
    return;                                                      // 退出渲染循环
}
```

这里，我们创建了一个VkPresentInfoKHR结构变量，设置了渲染结果的类型为VK_STRUCTURE_TYPE_PRESENT_INFO_KHR。我们不使用任何等待信号。Swapchain数量设置为1，pImageIndices指向当前帧图片索引，swapchain指向当前Swapchain对象。如果出现错误或Swapchain被重建，则需要重建Swapchain。

接下来，我们调用vkQueuePresentKHR()函数，传入渲染结果信息。如果函数执行失败，且错误码为VK_ERROR_OUT_OF_DATE_KHR，则需要重建Swapchain。如果framebufferResized变量为true，则需要重建Swapchain。