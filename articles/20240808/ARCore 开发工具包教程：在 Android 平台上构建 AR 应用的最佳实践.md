                 

# ARCore 开发工具包教程：在 Android 平台上构建 AR 应用的最佳实践

> 关键词：ARCore, Android, AR应用, 工具包, 教程, 最佳实践

## 1. 背景介绍

增强现实（AR）技术通过在现实世界中叠加虚拟信息，创造出沉浸式的交互体验。随着AR设备如智能手机、AR眼镜的普及，AR应用在教育、医疗、旅游、娱乐等众多领域中迅速发展。Android平台作为全球主流的移动操作系统，自然成为AR应用开发的首选平台。

ARCore是由谷歌推出的官方AR开发平台，提供了强大的AR功能，帮助开发者构建高性能的AR应用。本文将详细介绍如何使用ARCore开发工具包在Android平台上构建AR应用，涵盖从环境搭建到具体实现的全过程。通过本教程，你将掌握ARCore的核心概念、开发技巧和最佳实践，成为AR应用的开发高手。

## 2. 核心概念与联系

### 2.1 核心概念概述

为了更好地理解ARCore开发工具包，本节将介绍几个关键概念：

- **ARCore**: 谷歌开发的AR开发平台，提供了基础的AR功能，如平面检测、跟踪、光照估计等，可用于构建高性能AR应用。
- **平面检测**: 检测并跟踪现实世界中的平面，AR应用可以通过平面进行空间定位和布局。
- **跟踪**: 在平面上跟踪虚拟对象，如球体、3D模型等，实现虚拟对象在现实空间中的交互。
- **光照估计**: 估计场景光照信息，优化虚拟对象在真实环境中的渲染效果。
- **AR场景理解**: 通过计算机视觉和机器学习技术，理解场景中物体的位置、姿态和运动等，为AR应用提供更丰富的信息。
- **AR体验定制**: 根据不同设备和用户的特性，定制AR应用体验，如动态AR效果、个性化内容等。

### 2.2 核心概念原理和架构的 Mermaid 流程图

```mermaid
graph LR
    A[平面检测] --> B[跟踪]
    B --> C[光照估计]
    C --> D[AR场景理解]
    D --> E[AR体验定制]
```

这个流程图展示了ARCore开发的基本流程。首先通过平面检测功能确定AR环境中的平面，然后在平面上进行物体跟踪，通过光照估计优化虚拟对象的渲染效果，最后利用AR场景理解，根据不同设备特性和用户偏好，定制个性化的AR体验。

## 3. 核心算法原理 & 具体操作步骤

### 3.1 算法原理概述

ARCore的核心算法原理可以归纳为以下几个方面：

- **平面检测**: 使用计算机视觉技术，如SIFT、ORB等，检测并跟踪现实世界中的平面。
- **跟踪**: 利用卡尔曼滤波器等算法，跟踪虚拟对象在平面上的位置和姿态。
- **光照估计**: 使用光照渲染模型，如光照感知渲染（LPA），根据场景的光照信息，优化虚拟对象的渲染效果。
- **场景理解**: 使用卷积神经网络（CNN）、深度学习等技术，理解场景中物体的位置、姿态和运动，为AR应用提供更丰富的信息。

### 3.2 算法步骤详解

以下详细介绍ARCore开发工具包的使用步骤：

**Step 1: 环境搭建**

1. **安装Android Studio**: 从官网下载并安装Android Studio，用于开发和调试Android应用。
2. **配置ARCore SDK**: 打开Android Studio，选择File -> Project Structure，选择App -> Modules -> arcore-app -> app -> build.gradle，添加ARCore依赖：

   ```gradle
   implementation 'com.google.ar.sceneform:sceneform-arcore:0.12.0'
   ```

**Step 2: 添加ARCore相关组件**

1. **初始化ARCore环境**: 在Activity的onCreate方法中添加ARCore环境初始化代码：

   ```java
   public class MainActivity extends AppCompatActivity {
       private ArFragment arFragment;
       @Override
       protected void onCreate(Bundle savedInstanceState) {
           super.onCreate(savedInstanceState);
           setContentView(R.layout.activity_main);
           arFragment = ArFragment.newInstance();
           arFragment.setCamera(0);
           arFragment.setWorldLightSource(ArFragment.ArLightSource.REAL_TIME_LIGHT);
           arFragment.setLightingSource(ArFragment.ArLightingSource.MIXED_LIGHT);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.ENABLED);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FAST);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.PRECISE);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.HIGH_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIGHEST_PRECISION);
           arFragment.setWorldLightingQuality(ArFragment.ArWorldLightingQuality.FCAST_HIG

