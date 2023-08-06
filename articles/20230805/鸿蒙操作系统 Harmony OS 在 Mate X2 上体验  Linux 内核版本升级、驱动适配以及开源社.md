
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         在越来越多的手机及平板电脑的普及中，手机操作系统、UI界面、屏幕分辨率等方面逐渐趋于统一，而操作系统的更新换代速度也在加快，不仅改变了产业模式，也改变了用户对各个手机平台的依赖程度。移动端设备的种类繁多，从智能机、便携终端到大屏手机和平板电脑，面临着巨大的市场供需矛盾。而Android生态圈一直以来坚持稳定的发展方向，所以即使在Android上也有越来越多的第三方ROM厂商发布自己的定制系统，这些系统往往兼顾硬件定制能力和操作系统应用层扩展能力。然而Android这种闭源系统一直存在很多问题，比如安全性弱、性能差、耗电高、界面卡顿、相机性能差等。因此，华为、OPPO等一系列巨头也纷纷布局开源系统解决这个难题。Harmony OS 是华为和开源社区推出的开源移动端操作系统，它的目标是建立一个基于Linux内核的可信任的分布式操作系统，兼顾安全、性能、功能和生态。Harmony OS 的 Linux 内核版本目前已经更新到 5.10 ，并且支持常见的嵌入式设备、手机、平板等主流设备。这篇文章将以Mate X2为例，介绍Harmony OS在该平台上的体验。本文将主要介绍Harmony OS在Mate X2上的配置、适配过程、驱动移植过程、社区贡献、Harmony OS自身特色及未来的发展方向。
     
         本文分为两个章节，第一章介绍Harmony OS基本概念与核心组件，第二章以Mate X2为例详细介绍Harmony OS在该平台上的适配、移植、开发、社区贡献、自身特色、未来发展方向等环节。
     
         # 2.基本概念术语说明
         ## 2.1 操作系统
         
         操作系统（Operating System，OS）是管理计算机硬件资源、控制程序运行和提供抽象化服务的软硬结合系统，负责管理各种硬件和软件资源并向上提供应用程序接口。它主要包括三个部分：内核、Shell和应用程序。它们之间通过操作系统调用接口进行通信。操作系统的内核通常是指最基础、最核心的系统软件，负责资源分配、进程调度、内存管理等关键功能；Shell是一个命令行接口，用户可以通过它操作操作系统；应用程序则是操作系统上运行的各种程序。
         ## 2.2 Linux内核
         
         Linux 是一个开源的 Unix 操作系统，由林纳斯·托瓦兹和于谦两位创始人创立，其设计思想是“简单但功能强大”。Linux 以模块化和微内核的形式呈现，可以单独使用某些功能或全部使用，且可自由地定制或修改代码。Linux 发行版（Distribution）通常基于 Linux 内核，包含 Linux 内核和其它软件包，如文件管理器、文本编辑器、浏览器、命令行环境、图形界面、打印机客户端等。除了基于 Linux 内核之外，还有一些其他项目，如 Android、ChromeOS、Fire OS 和 Tizen 等，都是基于 Linux 内核构建。
         ## 2.3 Harmony OS
        
         Harmony OS 是华为开源社区推出的开源的移动端操作系统，可广泛应用于智能手机、平板电脑、穿戴设备、模拟设备等领域，具有轻量级、高效、可靠等特点。Harmony OS 基于开源项目 Linaro 提供的一套完整的操作系统，基于 Linux 内核，提供了一整套兼容 Linux API 的 API 框架，实现对系统硬件的完整控制，同时还融合了其他开源项目如 Chromium OS、Tizen OS、WebThings OS 的优势。Harmony OS 由系统内核、中间件、应用三部分组成。系统内核负责底层硬件的驱动、资源管理、进程调度等；中间件提供系统上层应用所需的各种服务，如网络、存储、安全等；应用则是操作系统上运行的各种程序，包括手机上的应用、桌面的应用、网页浏览等。Harmony OS 融合了各类开源项目的优点，开源社区可以很容易地参与进来，以便及早发现、解决系统中的问题。
     
         ## 2.4 Linaro
        
         Linaro 是基于 Linux 内核的开源项目，由国际团队合作开发，以打造一个全栈式的开源智能手机操作系统为目标。Linaro 关注开源、开放和透明，为开发者提供成熟的工具和方法论，以打造一个开源的、可信任的、可持续的、不断增长的、用户驱动的操作系统。Linaro 的主页 https://www.linaro.org/ 。
     
         ## 2.5 组件说明
         
         ### 2.5.1 系统服务框架(Saf)
         
         Saf (System Application Framework) 是一个用于快速构建系统应用的软件框架，以 Saffron 语言编写。Saf 中定义了一组系统服务，包括文件系统、窗口管理、音频、视频、多媒体、蓝牙、USB、Wi-Fi、触摸屏、传感器、传感器、位置、生物识别、电池状态、电源管理、通知、动画、资源管理等。
         
         ### 2.5.2 文件系统
         
         文件系统负责储存数据和元数据，包括用户数据、系统数据、日志、程序等。Harmony OS 支持 ext4 文件系统，其它文件系统可能不被支持。
         
         ### 2.5.3 进程管理
         
         进程管理负责创建和管理应用程序的执行环境，包括调度、分配系统资源、进程间通信等。Harmony OS 使用 LwIP 作为 TCP/IP 协议栈，可以实现类似 Windows 下的 Berkeley Sockets 接口，并集成了 FreeRTOS 实时操作系统。
         
         ### 2.5.4 运行时库(Runtime Library)
         
         运行时库负责提供操作系统所需的基本服务，如内存分配、输入输出、异常处理、线程同步、网络、文件、定时器、全局变量、调试信息等。Harmony OS 使用的是开源的 Glibc，其带有丰富的功能特性和安全保证。
         
         ### 2.5.5 用户接口(User Interface)
         
         用户接口是用来让用户与系统互动的各个部件。Harmony OS 提供了一个 Qt UI 框架，其中包括导航栏、状态栏、主窗口、消息框、对话框、菜单、按键、声音等。Harmony OS 的 UI 框架支持多个显示设备，包括 HDMI、LCD、AMOLED、e-ink、手写笔等。
         
         ### 2.5.6 驱动模型
         
         驱动模型负责管理系统的所有硬件资源，包括 CPU、外设、内存、网络等。Harmony OS 将驱动程序编译为动态链接库，加载到内存中后就可以被系统调用。驱动程序根据硬件特征分为不同的类型，如块设备驱动、字符设备驱动、网络设备驱动等。Harmony OS 支持 GPIO、I2C、SPI、UART、PWM 等接口，支持硬件加密、安全芯片等。
         
         # 3.Harmony OS在Mate X2上的适配与移植
         ## 3.1 配置与下载
         
         在Mate X2上搭建Harmony OS开发环境需要以下准备工作：
         
            1. 一台装有 Ubuntu 或 Debian 操作系统的 PC 或 云服务器
            2. 有一张 SD 卡，预先烧好镜像或者通过 USB 连接网盘下载合适的镜像文件，并且需要确保电脑与 Mate X2 通过数据线连接，确保数据传输没有问题。
            3. 下载适用于Mate X2的SDK，地址：https://github.com/OpenHarmony-mirror/build 
            4. 安装Docker，Docker是一个开源的应用容器引擎，Harmony OS官方推荐使用Docker来安装Harmony OS开发环境。
         当下载完成准备工作之后，就可以开始安装Harmony OS开发环境了。下面是详细的安装步骤。
         
         **注意**：这一步需要电脑配置较好的一方参与，建议大家一起协助，提前熟悉相关的配置与安装流程。
         
         1. 准备一台PC或云服务器，并配置好Ubuntu或Debian操作系统，安装VMWare或VirtualBox虚拟机环境，如果条件允许，也可以购买实体机。
         2. 在Windows主机下打开VMware或VirtualBox，选择新建虚拟机，指定操作系统为Ubuntu或Debian，确定。
         3. 设置虚拟机的内存大小，和CPU数量，这里建议内存大小大于8GB以上，CPU核心数量应该大于等于4个。
         4. 设置硬盘大小，建议至少8G以上。
         5. 安装VMware Tools，VMware Tools是在虚拟机上运行VMware软件的必备组件。安装方式如下：点击虚拟机中的“设置”选项，点击“安装VMware Tools”，等待完成安装。
         6. 从网盘中下载对应版本的Harmony OS SDK，解压压缩包到桌面目录，然后通过USB拷贝到VMware虚拟机的目录下。
         7. 配置SSH，Harmony OS SDK默认开启SSH远程登录，如果需要远程登录，请按照下面的步骤：
             a. 创建ssh密钥对，在Windows主机下打开Git Bash，输入以下命令：`ssh-keygen`，然后回车，会出现一个提示符，输入文件保存路径，比如`~/.ssh/id_rsa`，回车两次，密码为空。
             b. 把生成的公钥添加到VMware虚拟机的authorized_keys文件里，打开`.ssh`文件夹，把`id_rsa.pub`复制出来，然后双击打开VMware的终端窗口，输入命令`sudo nano authorized_keys`，粘贴进去，按Ctrl+X保存退出。
             c. 在VMware虚拟机上打开VMware终端窗口，输入`ls /root/.ssh`，确认是否有`authorized_keys`文件。
             d. 浏览器访问Mate X2的官网：http://mate.huawei.com/#phoneinfo 
             e. 点击Mate X2对应的型号，进入型号详情页面，找到“软件下载”栏目，点击“开发者工具下载”，跳转到开发者工具下载页面，点击“HarmonyOS-MateX2-Tools-linux-0.9.tar.gz”下载，下载完成后解压。
             f. 拷贝HarmonyOS-MateX2-Tools-linux-0.9.tar.gz到VMware虚拟机的根目录下。
             g. 通过USB将该SDK镜像文件拷贝到虚拟机的目录下。
             h. SSH连接到虚拟机，输入`./start.sh`,等待脚本执行完成。
             i. 在虚拟机中执行以下命令，安装支持ARM架构的交叉编译器：
             
            ```
            sudo apt install gcc-arm-none-eabi
            ```
            
             j. 安装华为Linux驱动，建议用命令安装：
             
            ```
            mkdir -p /home/$USER/bin
            wget http://openharmony-sig-ci-release.obs.cn-north-4.myhuaweicloud.com/resource/drivers/Hi3516DV300-linux-1.0.0.tar.gz
            tar xvf Hi3516DV300-linux-1.0.0.tar.gz && cp./lib/hisi_fb.* /lib/modules/$(uname -r)/kernel/drivers/video
            modprobe hisi_fb
            lsmod | grep hisi_fb
            echo "SUBSYSTEM==\"usb\", ATTR{bDeviceClass}==\"0xfe\",\
            MODE=\"0666\", GROUP=\"plugdev\"" > /etc/udev/rules.d/55-hisi-matex2.rules 
            ls /dev/ttyUSB*
            rm -rf Hi3516DV300-linux-1.0.0.tar.gz lib 
            ```
            
            k. 配置环境变量，在`.bashrc`或`.zshrc`文件末尾加入以下内容：
             
            ```
            export PATH=$PATH:/opt/hisi-objdir/usr/bin
            source build/envsetup.sh setarg
            export HARMONY_HOME=/root/HarmonyOs/build
            ```
            
            其中`HARMONY_HOME`变量的值指向Harmony OS SDK解压后的目录，`$PATH`变量追加`build/tools/flashlight`下的路径。
            
           如果需要正常启动，还需要添加一下启动参数：
             
            ```
            append loadramfs=init.rc initcall_debug androidboot.hardware=mate-ap exparam="consoleblank=0" console=ttyAMA0,115200n8 rootfstype=ext4 root=PARTUUID=$(blkid -s PARTUUID -o value /dev/mmcblk0p2)\; selinux=disabled debug_symbols=1 module_signatures=certs bootimg-partition=1
            ```
            
            l. 执行完以上步骤之后，即可正常使用Harmony OS开发环境。
            
         8. 测试Harmony OS，切换到虚拟机的用户账户下，打开终端，执行`hb-config`，查看Harmony OS的相关配置情况，确认无误后，输入`./start-developement.sh mate`。等到命令行提示符变成#，代表环境初始化成功。然后，输入`ping www.google.com`，测试网络是否正常。如果网络正常，则Harmony OS开发环境已经安装成功。
            
         
         ## 3.2 编译源码
         编译源码不需要做太多配置，只需要按照编译流程走一遍就好，这里我们以 helloworld 示例为例进行介绍。
         1. 获取源码：
         
            ```
            git clone https://gitee.com/openharmony/l-loader.git -b master
            cd l-loader/
            repo init -u https://gitee.com/openharmony/manifest.git -b refs/tags/OpenHarmony-v1.0.0-LTS -m hello_world.xml
            repo sync --no-clone-bundle --no-tags --optimized-fetch --force-sync -c
            ```
         2. 修改配置文件：打开`device/qemu/linux/harmony_hello_config.gni`，在`deps`数组中添加 `third_party/helloworld`依赖。

            ```
            deps = [
             ...
              "//third_party/helloworld:helloworld",
            ]
            ```
         3. 编译源码：使用`gn gen out/ohos-arm64 --args='target_cpu="arm64"'`，编译成功后，在out/ohos-arm64目录下看到生成的helloworld可执行文件。
         ## 3.3 烧录系统
        本例中，使用hb-config命令将系统烧录到Mate X2上。首先，需要设置IP地址和MAC地址，在命令行下输入：

        ```
        netcfg
        ```
        
        根据提示依次输入IP地址、子网掩码、网关地址、DNS服务器地址、MAC地址等信息，按回车继续。
        
        其次，按照以下操作顺序使用hb-config命令烧录HarmonyOS到Mate X2：
        
        ```
        hb-config
        devname qemuharf
        serialnum qemu_001
        connect
        compile
        image list
        image show 0
        writeimage
        reboot
        ```
        
        命令说明：
        
        * `hb-config`：初始HarmonyOS系统环境设置，若环境已设置过，可直接跳过此步骤。
        * `devname qemuharf`：设定设备名称为qemuharf。
        * `serialnum qemu_001`：设定序列号为qemu_001。
        * `connect`：连接设备，初次连接需要扫码认证。
        * `compile`：编译系统。
        * `image list`：列出所有系统镜像。
        * `image show 0`：查看当前使用的系统镜像。
        * `writeimage`：写入系统镜像到设备。
        * `reboot`：重启设备。
        
        当系统完成烧录，重启后，系统启动起来了，如果顺利的话，可以在串口终端中看到hello word的欢迎信息，表示helloworld项目已成功运行。
        