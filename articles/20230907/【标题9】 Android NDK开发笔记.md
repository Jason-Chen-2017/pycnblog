
作者：禅与计算机程序设计艺术                    

# 1.简介
  

NDK(Native Development Kit)是一种运行在Android系统上的编程接口，它允许您编写应用的底层代码，如C/C++，可以让您的APP具有更好的性能、可靠性及扩展性。本教程将详细地介绍了如何通过NDK开发Android应用程序，包括安装配置、基础语法、开发原理、方法调用与内存管理等内容。本文假设读者已经具备了一些编程经验，熟悉Java、Kotlin或其他面向对象的语言，并对JNI有一定了解。
## 1.1 本教程适用对象
- 有一定编程基础，熟练掌握Java、Kotlin等面向对象的语言。
- 对Android NDK有一定的了解。
- 想要学习NDK开发Android应用，掌握 JNI 开发流程及其原理。
## 1.2 阅读时间建议
本教程根据作者个人学习经验，所阅各章节相互独立，不涉及到太多基础知识，因此每章大约7-8页即可。
# 2.环境准备
## 2.1 安装JDK
如果你还没有安装JDK，请下载并安装JDK，版本建议选择最新版。
## 2.2 配置ANDROID_HOME环境变量
设置ANDROID_HOME指向你的Android SDK目录。例如:
```shell
export ANDROID_HOME=/Users/{yourUserName}/Library/Android/sdk
```
## 2.3 创建一个新项目
创建一个名为ndkDemo的新项目，命令如下:
```gradle
// 创建一个名为ndkDemo的新项目
$ mkdir ndkDemo
$ cd ndkDemo

// 在build.gradle文件中添加ndk的依赖关系
dependencies {
   ...
    implementation 'com.android.support:appcompat-v7:28.0.0'

    // 添加ndk相关依赖
    implementation "org.jetbrains.kotlin:kotlin-stdlib:$kotlin_version"
    implementation 'androidx.constraintlayout:constraintlayout:1.1.3'
    implementation("com.github.zhukic:nativehelper:1.3")
    
    // 指定ndk的最低支持版本
    android {
        defaultConfig {
            ndk {
                abiFilters 'armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64'
                
                // 设置生成so文件的存放路径
                moduleName "nativehelloworks"
                sharedLibFolder = file("$projectDir/app/src/main/jniLibs/${targetArch}")
            }
        }

        compileOptions {
            sourceCompatibility JavaVersion.VERSION_1_8
            targetCompatibility JavaVersion.VERSION_1_8
        }
        
        kotlinOptions {
            jvmTarget = '1.8'
        }
        
    }
}

task clean(type: Delete) {
    delete rootProject.buildDir
}
```
然后配置jni文件夹，创建CMakeLists.txt。
```gradle
// 创建jni文件夹
mkdir app/src/main/jni/
cd app/src/main/jni/

// 创建CMakeLists.txt文件
touch CMakeLists.txt
```
CMakeLists.txt 文件内容如下:
```cmake
add_library( # Sets the name of the library.
             nativehelloworks

             # Sources
             hello.cpp )
```
hello.cpp 文件内容如下:
```c++
#include <stdio.h>
#include <string.h>
#include <stdint.h>


extern "C" void hello() {
  printf("Hello from C++ code!\n");
}

```
## 2.4 配置依赖库
为了能够使我们的Gradle构建脚本正常工作，需要下载 Android 支持库（Google Play 服务）来支持 Android 特性，同时也推荐大家下载 Kotlin 来开发 Kotlin 应用。
```gradle
// 下载 AndroidX 的 AppCompat 模块，该模块提供了诸如按钮、标签、文本输入框等 UI 组件。
implementation 'androidx.appcompat:appcompat:1.0.2'

// 下载 Google Play 服务 SDK，它包含了许多 Android 特有的功能，例如广告、认证等。
implementation 'com.google.android.gms:play-services-auth:16.0.1' 

// 下载 Kotlin 编译器插件，该插件用于把 Kotlin 源码编译成字节码。
apply plugin: 'kotlin-android'

// 将 Kotlin 标准库添加为依赖项。
implementation "org.jetbrains.kotlin:kotlin-stdlib:${kotlin_version}"
```
最后修改 build.gradle 文件中的 minSdkVersion 和 applicationId 为自己项目的配置。
```gradle
defaultConfig {
    applicationId "com.example.ndkdemo"
    minSdkVersion 19
    targetSdkVersion 28
    versionCode 1
    versionName "1.0"
    testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    vectorDrawables.useSupportLibrary = true
}

dependencies {
    implementation fileTree(dir: 'libs', include: ['*.jar'])
    implementation 'androidx.appcompat:appcompat:1.0.2'
    implementation 'com.google.android.gms:play-services-auth:16.0.1'
    implementation "org.jetbrains.kotlin:kotlin-stdlib:${kotlin_version}"
    implementation 'com.github.zhukic:nativehelper:1.3'
}
```
注意：由于本地调试时Gradle构建依赖库可能会遇到网络连接问题导致失败，所以你可以尝试用国内镜像源替换掉官方源解决。
```gradle
allprojects {
    repositories {
        google()
        mavenCentral()
        jcenter() // https://developer.aliyun.com/mirror/jcenter
        maven { url 'https://maven.google.com/' } // https://developers.google.com/speed/public-dns/docs/using
        maven { url "http://maven.aliyun.com/nexus/content/groups/public/" } // https://yq.aliyun.com/articles/691430?spm=a2c4e.11153940.blogcont54665.6.7d9e2f1bEzgXXS
    }
}
```
# 3.基础语法
## 3.1 JNI基础
JNI（Java Native Interface）是一种用来在JVM和非JVM平台之间交换数据的机制。它主要提供三种功能：
- 使用Java API的Java程序可以调用非Java代码；
- 非Java代码可以使用JVM提供的各种函数，比如动态加载类、获取类的静态字段、调用方法等；
- JVM可以在运行期间将内部数据转换为非JVM平台的数据类型，或者将非JVM平台的数据类型转换为JVM内部的数据类型。
### 3.1.1 导入头文件
所有的JNI接口都定义在`jni.h`头文件中，使用以下方式导入头文件：
```c++
#include <jni.h>
```
### 3.1.2 JNIEnv指针
在调用JNI接口之前，必须先获得JNIEnv指针，JNIEnv指针保存了所有JNI接口的入口地址。JNIEnv指针可以通过JavaVM指针取得。JavaVM指针可以从JNIEnv指针取得。JNIEnv指针的获取方式如下：
```c++
JNIEnv *env;
int result = vm->GetEnv((void **) &env, JNI_VERSION_1_6);
if (result!= JNI_OK) {
    // handle error
}
```
其中vm是JavaVM指针，`JNI_VERSION_1_6`是请求的JNI版本号。
### 3.1.3 方法签名字符串
JNI中所有的方法都是通过方法签名字符串进行定义的。方法签名字符串由两部分组成：返回值类型和参数列表。参数列表里的参数类型用“;”分隔。例如："()V"表示无参无返回值的方法。
### 3.1.4 jclass类型
jclass是一个类型指针，用作存储Java类的引用。JNIEnv中有一个叫FindClass的方法，这个方法接收一个方法签名字符串作为参数，并查找对应的类。
### 3.1.5 jmethodID类型
jmethodID也是类型指针，用来标识某个方法。JNIEnv中有一个叫GetMethodID的方法，这个方法接收两个参数：类的引用和方法签名字符串。GetMethodID会查找指定的类里对应的方法并返回对应的方法ID。
### 3.1.6 jobject类型
jobject是一个类型指针，用来标识Java对象。JNIEnv中有一个叫NewObject的方法，这个方法接收三个参数：类的引用、构造方法ID和参数值数组。这个方法可以用来创建新的Java对象。
### 3.1.7 删除局部引用
当需要调用Java API的时候，不能直接使用本地堆栈上的数据，而应该转化为全局引用，或者是缓存起来供后续使用的全局引用。如果创建了一个全局引用之后，很可能在下一次调用的时候就会发生垃圾回收，那么就需要调用DeleteLocalRef方法删除该局部引用。
```c++
jstring stringValue = env->NewStringUTF("Hello from JNI!");
...
env->DeleteLocalRef(stringValue);
```
### 3.1.8 异常处理
如果一个JNI方法抛出了异常，那么可以通过Throw方法抛出给Java层。对于已经捕获到的异常，可以通过ExceptionOccurred方法检查是否有异常，如果有异常则可以通过ExceptionDescribe和ExceptionClear方法打印异常信息并清除异常状态。
```c++
try {
    // call some Java methods here
} catch (...) {
    LOGE("Caught exception during JNI call");
    env->ExceptionDescribe();
    return -1;
}
```
### 3.1.9 对象转换
Java和JNI之间的对象传递遵循以下规则：
- 如果传递的是基本数据类型（整数、浮点型、布尔型），它们的值被复制过去，也就是说JVM里的变量与JNI里面的变量不是同一个变量。
- 如果传递的是简单类实例（Primitive Class Instance），它只是复制了值，但是JVM里的变量还是原来的变量。
- 如果传递的是复杂类实例（Complex Class Instance），例如自定义类，那么JNIEnv中的NewObject或CallStaticMethod都会自动进行必要的类型转换。这种情况下，JNIEnv不会复制整个实例的内容，而只是复制其引用（reference）。
- 如果需要转换一个对象类型到另一个对象类型，可以使用JNIEnv中的CallObjectMethod、CallBooleanMethod、CallIntMethod等方法。这些方法会自动进行必要的类型转换。
- 可以使用IsInstanceOf方法判断一个对象是否是某种类型的实例。