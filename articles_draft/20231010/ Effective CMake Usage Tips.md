
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


CMake是一个跨平台构建工具，支持多种编程语言、多种编译器、多种集成开发环境等，使得项目构建配置管理变得简单、高效、可靠。然而，CMake的用法并不一定总是那么容易理解和掌握。在日益复杂的项目构建过程中，一些最基础的、经常使用的功能，如果没有好的使用方法，就会导致构建过程的不便捷、不可靠甚至错误。本文将详细阐述CMake的核心概念、用法技巧以及典型场景下的实践经验。希望能够对读者提供更加有效地使用CMake的建议，降低使用 CMake 的难度。
# 2.核心概念与联系
## CMake简介
CMake（Cross Platform Make）是一个跨平台的开源构建系统，其作用是从源文件到目标文件（可执行文件、库文件等）的自动化生成过程。CMake提供了方便的命令行接口和丰富的函数接口，可以实现对工程各种属性的设置，并根据指定规则生成对应需要的makefile或者工程文件，进而进行编译链接等流程。在CMake中，有一个名为CMakeLists.txt的文件，该文件是cmake用于描述项目信息的主要文件。CMake允许用户自定义变量、函数、宏等扩展语法，以便于用户灵活的定义自己的构建逻辑。
## CMake的核心概念
### 1.Build System
CMake是一个跨平台的构建工具，其工作流程包括三个主要阶段：
- 生成：预处理、扫描源码文件、产生中间态文件。这一步生成了用于构建的Makefile或工程文件。
- 编译：使用生成的中间态文件编译出目标文件。这一步通常涉及多个编译器，即所谓的编译器链（compiler chain）。
- 安装：安装生成的目标文件。这一步完成后，最终的软件包可以被用户使用。

### 2.Targets and Dependencies
CMake通过Target来表示构建单元。每个Target都代表一个特定的可执行文件、静态/动态库、头文件、资源文件等，它与其他目标之间存在依赖关系。依赖关系可以通过target_link_libraries、add_dependencies等命令定义。

### 3.Variables and Cache
CMake中的变量是由用户在CMakeLists.txt中自定义的，并且可以在整个工程范围内共享和使用。用户也可以在cmake-gui或ccmake命令行工具中查看和修改变量的值。

CMake的缓存存储着用户设置的所有CMake变量的当前值，并在cmake配置文件中保存为文本文件。CMake缓存是临时的，只能在一次CMake会话期间使用，不能持久化到磁盘上。当CMake重新运行时，这些缓存值将会失效。因此，CMake缓存应该只作为一次性的本地配置使用。如果想在不同的机器上共享缓存，可以使用CMAKE_CACHEFILE_DIR选项。

### 4.Generators
CMake可以生成多种类型的工程文件，其中包括VS工程、Xcode工程、Makefiles、Ninja构建脚本、Eclipse CDT工程等。选择正确的Generator对于优化编译速度、提升可移植性、减少生成时间至关重要。目前主流的Generator包括Unix Makefiles、Visual Studio系列、Xcode、Ninja、CodeBlocks等。

### 5.Commands and Functions
CMake提供了许多命令和函数用于控制构建过程。例如，add_executable、set、target_compile_options、find_package等命令和函数被用来控制项目的构建。许多第三方插件也支持CMake的扩展机制，让用户可以通过添加新的命令和函数来增强CMake的功能。

## 用法技巧
### 1.设置项目信息
#### 设置项目名称和版本号
项目信息的设置非常重要。通过PROJECT命令可以设置项目名称和版本号。设置完毕后，就可以在CMake中使用变量${PROJECT_NAME}获取项目名称，并使用变量${PROJECT_VERSION}获取项目版本。
```
project(MyProject VERSION 1.0)
```
#### 添加支持语言
CMake支持多种编程语言，可以通过设置CMAKE_CXX_COMPILER、CMAKE_C_COMPILER等来指定使用的编译器类型。可以通过ENABLE_LANGUAGE命令启用其他语言的支持，如Fortran。
```
enable_language(Fortran)
```
#### 指定源码文件路径
CMake默认情况下，查找源码文件的顺序是先在当前目录下查找，然后在CMakeLists.txt所在的父级目录下查找，最后才会到系统默认的搜索路径上查找。可以通过CMAKE_SOURCE_DIR变量指定源码文件的根目录。
```
set(CMAKE_SOURCE_DIR "/path/to/source")
```
#### 指定生成文件的路径
生成的工程文件、中间态文件、目标文件等都会放在构建目录（build directory）下。默认情况下，构建目录与源码文件同级。可以通过CMAKE_BINARY_DIR变量指定构建目录的路径。
```
set(CMAKE_BINARY_DIR "/path/to/build")
```
### 2.添加源文件和头文件
#### 添加单个源文件
CMake通过ADD_EXECUTABLE命令可以添加一个可执行目标。ADD_LIBRARY命令则用于添加一个库目标。可以直接指定源文件，如：
```
add_executable(myprogram main.cpp)
add_library(mylegacycode moduleA.cpp moduleB.cpp)
```
也可以给ADD_LIBRARY命令传递LINK_PUBLIC参数来让库的接口暴露给外部项目：
```
add_library(mylegacycode MODULE moduleA.cpp moduleB.cpp)
target_link_libraries(mylegacycode PUBLIC somelib)
```
#### 添加多个源文件
除了添加单个源文件外，还可以通过GLOB命令（全局匹配模式）或者GLOB_RECURSE命令（递归遍历目录）批量添加源文件。GLOB_RECURSE命令会递归遍历指定目录下的所有子目录，查找符合特定模式的文件。
```
glob(SRCS *.cpp)
add_executable(${PROJECT_NAME} ${SRCS})
```
#### 添加头文件
CMake通过include_directories命令来指定头文件搜索路径。它可以直接指定目录，也可以接受变量值。
```
include_directories(/path/to/headers)
include_directories($ENV{MY_HEADERS}/include)
```
### 3.设置编译选项
#### 设置编译器类型
CMake通过CMAKE_C_COMPILER、CMAKE_CXX_COMPILER等变量来指定使用的编译器类型。例如：
```
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
```
#### 设置编译器标志
CMake可以通过设置编译器标志来控制编译过程。可以通过target_compile_options命令设置编译目标级别的编译器标志，或者直接在add_executable、add_library等命令中设置编译单元级别的编译器标志。
```
target_compile_options(myapp PRIVATE "-Wall" "-Werror")
add_executable(myapp main.cpp)
set_property(TARGET myapp PROPERTY COMPILE_FLAGS "-std=c++11")
```
#### 设置链接器标志
CMake通过设置链接器标志来控制程序的连接过程。可以通过SET_PROPERTY、target_link_libraries命令设置链接器标志。
```
set_property(TARGET myapp PROPERTY LINK_FLAGS "-static")
target_link_libraries(myapp "${CMAKE_DL_LIBS}") # use dynamic linking for system libs
```
### 4.组织构建文件结构
#### 创建子目录
CMake可以创建子目录并将相关文件移动到相应的子目录中。可以使用add_subdirectory命令添加子目录。
```
add_subdirectory(utils)
```
#### 分割大的工程文件
对于较大规模的项目，可以考虑分割成几个小的CMakeLists.txt文件，再通过include命令来组合起来。这样可以简化构建脚本的维护和修改，也避免了因过长的CMakeLists.txt文件而造成的构建性能下降。
```
include(../utils/myUtils.cmake OPTIONAL)
```
### 5.指定依赖关系
#### 查找第三方库
CMake可以通过find_package命令来找到已安装的第三方库。这个命令会尝试在系统路径、CMake模块路径、或指定的搜索路径上查找第三方库的安装目录。可以使用REQUIRED参数让find_package命令失败时停止工程的构建。
```
find_package(OpenSSL REQUIRED)
target_link_libraries(myapp OpenSSL::SSL OpenSSL::Crypto)
```
#### 手动配置依赖关系
如果find_package命令无法满足项目的依赖需求，可以自己手动配置依赖关系。这种方式一般比较复杂，但是可以应对一些特殊的情况，如用一个不同版本的库替换掉系统默认版本。可以参考一下面的例子。
```
if(NOT MYLIB_FOUND)
    find_path(MYLIB_INCLUDE_DIR lib/mylib.h)
    if(MYLIB_INCLUDE_DIR)
        set(MYLIB_INCLUDE_DIRS ${MYLIB_INCLUDE_DIR})
        find_library(MYLIB_LIBRARIES NAMES mylib PATHS /usr/local/lib NO_DEFAULT_PATH)
        mark_as_advanced(MYLIB_INCLUDE_DIRS MYLIB_LIBRARIES)

        add_definitions(-DUSE_MYLIB) # define a macro to switch between using mylib or not
    endif()
endif()
```