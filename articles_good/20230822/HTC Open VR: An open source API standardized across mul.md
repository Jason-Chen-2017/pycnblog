
作者：禅与计算机程序设计艺术                    

# 1.简介
  

虚拟现实(VR)作为一项新兴的高科技产业，其产品种类繁多、技术水平参差不齐，各个厂商之间技术标准存在差异性，对开发者而言尤为复杂。
随着VR设备的逐渐普及，比如HTC Vive、Oculus Rift、Samsung Gear VR等等，以及各大游戏厂商纷纷推出支持VR游戏功能的SDK，使得VR游戏的开发成为可能，然而不同厂商提供的SDK兼容性较差，导致了开发者在集成VR功能时面临诸多挑战，如设备兼容性、性能限制、VR系统体验等等。
为了解决这些问题，美国HTC集团与多个VR设备制定了Open VR API标准，该API被许多开发者用于为不同的VR设备实现统一的VR游戏功能。通过Open VR API，开发者可以只需按照统一的规则编写代码，就可以轻松实现跨平台的VR游戏功能。
本文将以HTC Open VR API为例，阐述Open VR API的设计理念、实现机制、应用场景、优点以及局限性。
# 2.基本概念术语说明
## 2.1 虚拟现实(VR)
虚拟现实（Virtual Reality，VR）是一种利用计算机生成真实的三维环境的方法。它结合了模拟现实世界、计算机图形渲染和模拟人类的交互方式，让用户完全处于其创造出的数字环境中，真实地感受到周遭的物理世界、生物，甚至自我。在虚拟现实中，用户身处于一个虚拟空间中，并且周围的事物也在虚拟的画布上。用户能够在这个虚拟空间里自由移动、旋转和交互，同时获得与实际世界相似的体验。通过虚拟现实技术，我们可以更加直观地认识生活中的真实世界、探索未知领域、体验心灵鸡汤，还可作为一项新型的职业、娱乐活动或教育工具。VR虽然应用范围广泛，但目前仍处于起步阶段。截止2019年，全球已有超过两千万消费者接受过VR视频训练课程。

## 2.2 头戴式显示设备（HMD）
头戴式显示设备（Head Mounted Display, HMD）即为头部固定安装的电脑显示设备。它通常搭载于头部，帮助用户获得所需信息的视觉享受。采用头戴式显示设备能够提升用户的视觉能力和工作效率。除了在虚拟现实游戏中运用外，头戴式显示设备还可用于艺术表演、科研工作、远程会议等其他应用场景。

## 2.3 可穿戴设备（AR/MR设备）
可穿戴设备（Augmented Reality /Mixed Reality Device， AR/MR device）由硬件和软件组成，能够与现实世界进行互动，赋予人们新的想像力和沉浸式体验。以往AR/MR设备多由触屏控制，而最新研究表明，这种方式存在安全隐患，因此越来越多的企业和组织开始投入更多的精力在可穿戴设备的研发方面。

## 2.4 SteamVR
SteamVR是一个开源软件平台，能够运行在Windows PC、Android手机和Oculus Quest设备上的虚拟现实应用。SteamVR与HTC Vive一起出现后，推出了OpenVR API标准，HTC Open VR就是基于此API的开源项目。其开源授权协议为MIT License，具有跨平台、免费、开放的特点。

## 2.5 HTC Vive
HTC Vive是一款由HTC公司开发、生产的虚拟现实眼镜。它主要是为VR头戴设备，由两个光学头戴元素（一个为左眼，一个为右眼）和两个距离传感器组成。通过捕捉头顶上两侧的信息，Vive可以产生独有的3D坐标系，并将所有数据传输给PC。

## 2.6 SteamVR Driver
SteamVR驱动程序是一个为连接到SteamVR头戴显示设备的每个电脑配置的驱动程序。它负责处理底层的VR头戴设备相关接口请求，包括发送和接收VR数据包、呈现3D图像以及其他功能。SteamVR驱动程序通过虚拟现实头盔的摄像头捕捉到的头部姿态信息生成独有的空间坐标。

## 2.7 OpenVR SDK
OpenVR SDK是一个由Valve公司开发、维护的一套驱动和API，用来为不同类型的VR设备提供统一的接口。它包含了一系列模块，包括用于渲染VR环境的OpenVR compositor和用于处理输入事件的OpenVR input system。对于每个设备厂商来说，都需要根据OpenVR API规范去实现自己的驱动程序才能兼容OpenVR SDK。

## 2.8 SteamVR Home
SteamVR Home是一个由Valve公司开发的应用程序，支持运行在Windows PC上的虚拟现实平台。它是由SteamVR驱动、VR头盔和VR耳机设备组成，能够让用户在任何地方快速便捷地切换VR应用。

# 3.核心算法原理及操作步骤
Open VR提供了一套统一的API接口供开发者调用，其整体架构分为三个层级，分别是虚拟现实编程模型、设备驱动程序和主机软件。虚拟现实编程模型定义了一组抽象的函数和数据结构，供开发者调用，用于创建和管理VR应用。设备驱动程序负责向Open VR发送和接收VR数据包，并将其渲染到显示设备上。主机软件则是运行在用户的PC上，负责管理VR设备的连接、初始化和渲染。

## 3.1 虚拟现实编程模型
虚拟现实编程模型定义了一组抽象的函数和数据结构，供开发者调用，用于创建和管理VR应用。最核心的数据结构是VR_INTERFACE，它封装了各种不同设备的特定功能。开发者可以通过VR_INTERFACE对象获取各种功能，例如创建VR应用程序的框架、获取头部位置、获取手部数据等。

除了VR_INTERFACE之外，还有几个其他重要的抽象数据类型：

- VR_EVENT - 用于描述VR事件，例如设备被挂起或者键盘按键被触发。
- VR_ACTION - 描述一个可控变量，它的取值可以随时间变化。
- VR_COMPONENT - 提供了一个简单的接口，用来将组件嵌入到OpenVR应用程序中。
- VR_RENDERMODEL - 一种3D模型，通常用来表示某些实体，例如控制器、监视器和游戏物品。

每个VR头戴设备厂商都会创建一个名为openvr_api.h的头文件，包含一系列的函数接口和数据结构，供开发者调用。其中最重要的接口VR_IVRSystem，定义了一系列方法，用来查询和设置头戴设备的状态、参数和配置。

## 3.2 设备驱动程序
设备驱动程序是一个独立的进程，它与Open VR SDK通信，实现了从VR头戴设备获取VR数据包、处理VR事件、更新VR组件的状态。设备驱动程序是一个可执行文件，每台连接到Open VR的电脑上都有一个单独的设备驱动程序。设备驱动程序向Open VR发送VR数据包的方式有两种：

- 以定期的方式，按固定频率将渲染好的视图同步发送给Open VR。
- 当VR应用发生改变时，将通知Open VR更新渲染视图。

## 3.3 主机软件
主机软件又称为“框架”，运行在用户的PC上，负责管理VR设备的连接、初始化和渲染。它可以使用OpenVR SDK来调用Open VR提供的所有功能。当用户启动一个VR应用时，框架首先加载VR_IVRSystem，然后打开VR头戴设备。它通过VR驱动程序，与对应的设备驱动程序通信，初始化设备。在应用退出时，它关闭设备，清除资源。

# 4. 具体代码实例及解释说明
下面的例子展示了如何通过OpenVR SDK编写一个简单的VR程序。假设我们的应用需要展示一个空白的球体，并提供鼠标或触屏控制来旋转它。

## Step 1：准备开发环境
首先，安装好Visual Studio 2015及以上版本，并下载OpenVR SDK，解压后加入系统PATH路径。

## Step 2：新建工程
创建新的Win32 Console Application工程，命名为MyVRApp。


## Step 3：添加头文件
在MyVRApp工程目录下新建include文件夹，并添加如下头文件。

```c++
// openvr includes
#include <openvr.h>
#pragma comment(lib,"openvr_api") // link against the openvr_api dll
```

## Step 4：声明全局变量
在main函数之前声明一些全局变量，包括一个IVRSystem指针和一个bool变量表示是否初始化成功。

```c++
// global variables 
IVRSystem* vrsystem = nullptr;
bool isInitialized = false;
```

## Step 5：实现主循环
在while循环中，获取当前时间并计算两次连续调用的时间差，如果差值小于1秒，则等待0.01秒继续执行。

```c++
    int i=0; 
    auto start_time = std::chrono::high_resolution_clock::now();  

    while (true) {
        if (!isInitialized ||!vrsystem) {
            Sleep(1);    // sleep a bit before trying again 
            continue; 
        }

        auto now = std::chrono::high_resolution_clock::now();
        float elapsedTime = std::chrono::duration<float>(now - start_time).count();

        // do work here
        
        ++i;
        if (elapsedTime > 1.0f) {
            printf("FPS: %d\n", i);
            i = 0;
            start_time = std::chrono::high_resolution_clock::now();
        }

        Sleep(1);      // wait until next frame 
    }
}
```

## Step 6：初始化OpenVR
在初始化阶段，尝试初始化OpenVR系统。如果失败，则尝试重新初始化系统。由于初始化过程可能耗时，因此应该在必要时延迟并重试。

```c++
    // initialize OpenVR 
    vrsystem = VR_Init(&eError);

    if (vrsystem && eError == EVRInitError::None) {
        isInitialized = true;
        printf("OpenVR initialized.\n");
    } else {
        printf("Unable to initalize OpenVR. Error code: %d\n", eError);
        Sleep(5000);     // pause for 5 seconds and retry initialization 
        return 1;
    }
```

## Step 7：创建VR组件
创建一个VR组件用来代表我们的VR应用。我们这里只是简单创建一个球体，但是可以任意扩展成我们希望的任意3D模型。

```c++
    // create our virtual object 
    vr::TrackedDeviceIndex_t trackedController = k_unTrackedDeviceIndexInvalid;
    for (int i = 0; i < vrsystem->GetSortedTrackedDeviceIndicesOfClass( vr::TrackedDeviceClass_Controller, &trackedController, 1 ); i++) {
        char controllerName[128];
        vrsystem->GetStringTrackedDeviceProperty(trackedController, vr::Prop_RenderModelName_String, controllerName, sizeof(controllerName));
        const char* modelStr = "unknown";
        vr::RenderModel_t *model;
        vr::EVRRenderModelError error = vr::VRRenderModels()->LoadRenderModel_Async(controllerName, &model);
        if (error!= vr::VRRenderModelError_Loading) {
            modelStr = "loading failed";
        } else {
            while (!vr::VRRenderModels()->IsRenderModelLoaded(model)) {
                vSleep(1); //wait for loading
            }

            vr::RenderModel_TextureMap_t *texture;
            vr::VRRenderModels()->LoadTexture_Async(model->diffuseTextureId, &texture);
            while (!vr::VRRenderModels()->IsTextureLoaded(texture)) {
                vSleep(1); //wait for texture load
            }

            vr::Mat3x4_t matrix = {};
            vr::VRSystem()->GetMatrixForTrackedDevicePose(trackedController, vr::TrackingUniverseStanding, &matrix);
            vr::Vec3 center = { matrix.m[3][0], matrix.m[3][1], matrix.m[3][2] };
            vr::Quaternion orientation = { matrix.m[0][0], matrix.m[1][0], matrix.m[2][0], matrix.m[0][1] };
            CreateSphere(center, orientation, 0.2f, 10, 10);
        }

        break; // only use one controller for simplicity
    }
```

## Step 8：渲染VR组件
在渲染阶段，将我们的球体渲染到HMD的显示设备上。

```c++
    // render our virtual object 
    RenderStereoTargets();
    
    // render the sphere mesh onto the HMD
    static glProgramState program = NULL;
    if(!program) {
        program = GLProgramState::createFromFile("sphere.vs", "sphere.fs");
    }

    GL::Renderer::disableDepthTesting();
    Matrix4 projectionMatrix = getProjectionMatrix(vr::Eye_Left);
    program->setUniformValue("projection", projectionMatrix);
    DrawSphereMesh(*program);
```

## 完整源码
```c++
#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>

// openvr includes
#include <openvr.h>
#pragma comment(lib,"openvr_api") // link against the openvr_api dll


using namespace std;
using namespace chrono; 

void CreateSphere(const glm::vec3& position, const glm::quat& orientation, float radius, int slices, int stacks);
void CreateBox(const glm::vec3& position, const glm::quat& orientation, const glm::vec3& size);
glm::mat4 getViewMatrixFromPose(const vr::TrackedDevicePose_t &pose);
glm::mat4 getModelMatrixFromPose(const vr::TrackedDevicePose_t &pose);
glm::mat4 getProjectionMatrix(vr::EVREye eye);
void UpdateSphere(float timeSinceLastFrame);
void UpdatePoseData(std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount>& poses);

enum class Hand{ LeftHand, RightHand };

struct ButtonState { bool pressedPreviously : 1; bool pressedCurrent : 1; bool touchedPreviously : 1; bool touchedCurrent : 1; };

glm::vec3 g_positions[2]{};       // track left/right hand positions
ButtonState g_buttons[2]{};      // track left/right button states

// sphere object state data
glm::vec3 g_position{};          // position of our sphere
glm::quat g_orientation{};       // orientation of our sphere
glm::vec3 g_velocity{};          // velocity of our sphere
float g_angularVelocity{};       // angular velocity of our sphere

static double GetTimeInSeconds() {
    using namespace std::chrono;
    high_resolution_clock::time_point now = high_resolution_clock::now();
    duration<double> dtn = now.time_since_epoch();
    return dtn.count();
}

// global variables 
IVRSystem* vrsystem = nullptr;
bool isInitialized = false;

int main(int argc, char** argv) {
    // initialize OpenVR 
    vrsystem = VR_Init(nullptr);

    if (vrsystem) {
        isInitialized = true;
        cout << "OpenVR initialized." << endl;
    } else {
        cerr << "Unable to initalize OpenVR" << endl;
        return 1;
    }

    // create our virtual object 
    vr::TrackedDeviceIndex_t trackedController = k_unTrackedDeviceIndexInvalid;
    for (int i = 0; i < vrsystem->GetSortedTrackedDeviceIndicesOfClass( vr::TrackedDeviceClass_Controller, &trackedController, 1 ); i++) {
        char controllerName[128];
        vrsystem->GetStringTrackedDeviceProperty(trackedController, vr::Prop_RenderModelName_String, controllerName, sizeof(controllerName));
        const char* modelStr = "unknown";
        vr::RenderModel_t *model;
        vr::EVRRenderModelError error = vr::VRRenderModels()->LoadRenderModel_Async(controllerName, &model);
        if (error!= vr::VRRenderModelError_Loading) {
            modelStr = "loading failed";
        } else {
            while (!vr::VRRenderModels()->IsRenderModelLoaded(model)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1)); //wait for loading
            }

            vr::RenderModel_TextureMap_t *texture;
            vr::VRRenderModels()->LoadTexture_Async(model->diffuseTextureId, &texture);
            while (!vr::VRRenderModels()->IsTextureLoaded(texture)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1)); //wait for texture load
            }

            vr::Mat3x4_t matrix = {};
            vr::VRSystem()->GetMatrixForTrackedDevicePose(trackedController, vr::TrackingUniverseStanding, &matrix);
            vr::Vec3 center = { matrix.m[3][0], matrix.m[3][1], matrix.m[3][2] };
            vr::Quaternion orientation = { matrix.m[0][0], matrix.m[1][0], matrix.m[2][0], matrix.m[0][1] };
            CreateSphere(center, orientation, 0.2f, 10, 10);
        }

        break; // only use one controller for simplicity
    }

    while (true) {
        // update tracking information each frame
        std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> poses;
        vrsystem->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.0, poses.data(), static_cast<uint32_t>(poses.size()));

        UpdatePoseData(poses);

        // handle events
        vr::VREvent_t event;
        while (vrsystem->PollNextEvent(&event, sizeof(event))) {
            switch (event.eventType) {
            case vr::VREvent_ButtonPress:
                HandleButtonPress(event.data.controller.button, Hand::RightHand);
                break;
            case vr::VREvent_ButtonUnpress:
                HandleButtonUnpress(event.data.controller.button, Hand::RightHand);
                break;
            default:
                break;
            }
        }

        // update our VR components based on current pose data
        UpdateSphere(GetTimeInSeconds());

        // draw scene objects
        RenderStereoTargets();

        // render the sphere mesh onto the HMD
        static glProgramState program = NULL;
        if (!program) {
            program = GLProgramState::createFromFile("sphere.vs", "sphere.fs");
        }

        GL::Renderer::disableDepthTesting();
        Matrix4 projectionMatrix = getProjectionMatrix(vr::Eye_Left);
        program->setUniformValue("projection", projectionMatrix);
        DrawSphereMesh(*program);

        // flush rendering commands
        GL::Renderer::flushCommands();

        // wait for vertical sync signal
        WaitVerticalSync();
    }

    // clean up resources before exiting
    VR_Shutdown();
    return 0;
}

inline void UpdatePoseData(std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount>& poses) {
    // iterate through all devices in poses array
    for (auto i = 0u; i < poses.size(); ++i) {
        auto& pose = poses[i];

        // ignore any uninitialized or invalid devices
        if (!pose.bPoseIsValid) {
            continue;
        }

        // store position and orientation in global variable
        auto& p = g_positions[i];
        auto q = glm::quat(pose.qRotation.w, pose.qRotation.x, pose.qRotation.y, pose.qRotation.z);
        g_orientation = q;
        p.x = pose.mDeviceToAbsoluteTracking.m[3][0];
        p.y = pose.mDeviceToAbsoluteTracking.m[3][1];
        p.z = pose.mDeviceToAbsoluteTracking.m[3][2];
    }
}

inline glm::mat4 getModelMatrixFromPose(const vr::TrackedDevicePose_t &pose) {
    glm::mat4 modelMatrix = glm::translate(glm::mat4(1), glm::vec3(-pose.mDeviceToAbsoluteTracking.m[0][3], -pose.mDeviceToAbsoluteTracking.m[1][3], -pose.mDeviceToAbsoluteTracking.m[2][3]));
    glm::quat q = glm::quat(pose.qRotation.w, pose.qRotation.x, pose.qRotation.y, pose.qRotation.z);
    modelMatrix *= glm::toMat4(q);
    return modelMatrix;
}

inline glm::mat4 getViewMatrixFromPose(const vr::TrackedDevicePose_t &pose) {
    glm::mat4 viewMatrix = glm::inverse(getModelMatrixFromPose(pose));
    viewMatrix[3].w = 1; // fix w value for translation part
    return viewMatrix;
}

inline glm::mat4 getProjectionMatrix(vr::EVREye eye) {
    vr::HmdMatrix44_t proj = vrsystem->GetProjectionMatrix(eye, k_nearClipPlane, k_farClipPlane);
    glm::mat4 matProj = glm::transpose(glm::make_mat4x4((float*)&proj));
    return matProj;
}

void UpdateSphere(float timeSinceLastFrame) {
    constexpr float k_deltaT = 0.016f;        // fixed timestep
    static clock_t lastClockTick = 0;         // keep track of previous frame's time
    static float totalDeltaTime = 0.0f;      // accumulate time since start

    // calculate delta time
    clock_t currentClockTick = clock();
    float dt = ((float)(currentClockTick - lastClockTick)) / CLOCKS_PER_SEC;
    lastClockTick = currentClockTick;
    totalDeltaTime += dt;
    while (totalDeltaTime >= k_deltaT) {
        UpdateSpherePhysics(k_deltaT);
        totalDeltaTime -= k_deltaT;
    }

    // interpolate between current and previous pose to smooth motion
    float t = totalDeltaTime / k_deltaT;
    vr::TrackedDevicePose_t currPose = GetCurrentPose(Hand::RightHand);
    vr::TrackedDevicePose_t prevPose = GetPrevPose(Hand::RightHand);
    vr::Vector3_t vecPosPrev = glm::make_vec3(&prevPose.mDeviceToAbsoluteTracking.m[3][0]);
    vr::Quaternion_t quatPrev = glm::make_quat(&prevPose.qRotation.x);
    vr::Vector3_t vecPosCurr = glm::lerp(glm::make_vec3(&currPose.mDeviceToAbsoluteTracking.m[3][0]), glm::make_vec3(&vecPosPrev), t);
    vr::Quaternion_t quatCurr = glm::slerp(glm::normalize(glm::make_quat(&currPose.qRotation.x)), glm::normalize(glm::make_quat(&quatPrev.x)), t);
    g_position = glm::make_vec3(&vecPosCurr.v[0]);
    g_orientation = glm::normalize(glm::make_quat(&quatCurr.v[0]));
}

void HandleButtonPress(vr::EVRButtonId button, Hand hand) {
    if (hand == Hand::RightHand) {
        switch (button) {
        case vr::k_EButton_Grip:             // grab right-hand object
            SetActiveObject(g_handleSphere);
            break;
        case vr::k_EButton_Trigger:          // start teleportation gesture
            StartTeleport(g_position);
            break;
        }
    }
}

void HandleButtonUnpress(vr::EVRButtonId button, Hand hand) {
    if (hand == Hand::RightHand) {
        switch (button) {
        case vr::k_EButton_Grip:             // release right-hand object
            ClearActiveObject();
            break;
        case vr::k_EButton_TrackPadTouch:    // end teleportation gesture
            EndTeleport();
            break;
        }
    }
}

inline void CreateSphere(const glm::vec3& position, const glm::quat& orientation, float radius, int slices, int stacks) {
    MeshBuilder mb;
    mb.addSphere(mb.createPositionAttrib(position), mb.createNormalAttrib(glm::vec3()), radius, slices, stacks);
    unsigned short indices[] = {
        0, 1, 2, 2, 3, 0, // top cap
        4, 5, 6, 6, 7, 4, // bottom cap
        8, 9, 10, 10, 11, 8, // front face
        12, 13, 14, 14, 15, 12, // back face
        16, 17, 18, 18, 19, 16, // left side
        20, 21, 22, 22, 23, 20, // right side
        24, 25, 26, 26, 27, 24, // top face
        28, 29, 30, 30, 31, 28 // bottom face
    };
    mb.setIndices(indices, sizeof(indices)/sizeof(indices[0]));
    g_mesh = mb.create();
    g_handleSphere = AddComponent(g_mesh, Components::Transform{});
    GetComponent(g_handleSphere)->transform().setPosition(position);
    GetComponent(g_handleSphere)->transform().setRotation(orientation);
}

inline void CreateBox(const glm::vec3& position, const glm::quat& orientation, const glm::vec3& size) {
    MeshBuilder mb;
    mb.addBox(mb.createPositionAttrib(glm::vec3()), mb.createNormalAttrib(glm::vec3()), size.x, size.y, size.z);
    mb.rotate(mb.getVertexCount() - mb.createTexCoordAttrib({}).size(), 0.0f, 0.0f, 1.0f);
    mb.scale(mb.createPositionAttrib({}), size);
    unsigned short indices[] = {
        0, 1, 2, 2, 3, 0, // front face
        4, 5, 6, 6, 7, 4, // back face
        8, 9, 10, 10, 11, 8, // left side
        12, 13, 14, 14, 15, 12, // right side
        16, 17, 18, 18, 19, 16, // top face
        20, 21, 22, 22, 23, 20  // bottom face
    };
    mb.setIndices(indices, sizeof(indices)/sizeof(indices[0]));
    g_mesh = mb.create();
    g_handleBox = AddComponent(g_mesh, Components::Transform{});
    GetComponent(g_handleBox)->transform().setPosition(position);
    GetComponent(g_handleBox)->transform().setRotation(orientation);
}

inline void UpdateSpherePhysics(float dt) {
    // apply gravity force
    g_acceleration = glm::vec3(0.0f, -9.81f, 0.0f) * 10.0f; // simulate Earth surface gravity at roughly 10 units per second squared
    g_velocity += g_acceleration * dt;                      // update velocity by adding acceleration multiplied by time step
    g_position += g_velocity * dt;                           // update position by adding velocity multiplied by time step
}

// helper function to get current VR headset pose for specified hand
inline vr::TrackedDevicePose_t GetCurrentPose(Hand hand) {
    vr::TrackedDeviceIndex_t index = hand == Hand::RightHand? vr::k_ulRightControllerTrackerHandle : vr::k_ulLeftControllerTrackerHandle;
    vr::TrackedDevicePose_t result;
    memset(&result, 0, sizeof(result));
    vrsystem->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, &result, 1, &index);
    return result;
}

// helper function to get previously recorded VR headset pose for specified hand
inline vr::TrackedDevicePose_t GetPrevPose(Hand hand) {
    static std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> m_posesPrev;
    vr::TrackedDeviceIndex_t index = hand == Hand::RightHand? vr::k_ulRightControllerTrackerHandle : vr::k_ulLeftControllerTrackerHandle;
    m_posesPrev[index] = GetCurrentPose(hand);
    return m_posesPrev[index];
}

inline void StartTeleport(const glm::vec3& position) {
    g_teleportStartTime = GetTimeInSeconds();
    g_teleportStartPosition = position;
}

inline void EndTeleport() {
    if (GetTimeInSeconds() - g_teleportStartTime <= 0.5f) {
        glm::vec3 diff = g_position - g_teleportStartPosition;
        // TODO: perform actual teleport logic here!
    }
}

inline void ClearActiveObject() {
    g_activeObject = nullptr;
}

inline void SetActiveObject(GameObject* obj) {
    g_activeObject = obj;
}
```