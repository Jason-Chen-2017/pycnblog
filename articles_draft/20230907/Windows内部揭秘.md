
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 作者信息
> 作者：田静（金山词霸创始人）、李石元（微软Windows系统工程师）。

# 2.正文
## 2.1 概念术语
### 2.1.1 文件系统层次结构
文件系统层次结构即Windows中重要的文件组织形式，它将用户文件按照逻辑分类，并确保易于管理和访问。下图展示了Windows中的目录树结构。


可以看到Windows目录分为四层：根目录、卷目录、文件夹和文件。根目录、卷目录固定不变，文件夹是用户存放数据的地方，而文件是存放在某个文件夹下的具体数据。

Windows的目录路径一般由以下几部分组成：
- \\ComputerName：表示网络地址，例如“\\computer1”，代表计算机名为“computer1”。如果当前连接的是本地计算机，则此处可省略。
- \\\\.\PhysicalDriveNumber：表示物理驱动器编号，例如“\\\\.\\PhysicalDrive0”代表硬盘C盘。
- DirectoryName：表示目录名称，例如“C:\Program Files\Internet Explorer”。
- Filename：表示文件名称，例如“notepad.exe”。
同时注意，目录名称可以包含冒号(:)，但文件名称不允许包含冒号(:)。

### 2.1.2 对象类型与对象ID
对象类型定义了文件或目录的种类，比如普通文件(File), 目录(Directory)，重定向点(SymbolicLink)等；对象ID用来唯一标识一个文件或目录。

Windows API都通过注册表(Registry)中存储了各种对象的相关信息，包括每个对象类型的默认属性值、子级列表、访问控制列表等。其中一些重要的键值对如下所示：
| Key | Value | 描述 |
|---|---|---|
| DefaultSecurityDescriptor | SDB | 默认的安全描述符 |
| DosDevices | device name : object ID | 所有设备对象的ID |
| Enum | object type : subkey list | 指定类型的所有对象的子项列表 |
| Filesystems | drive letter : volume name : filesystem signature | 文件系统的详细信息 |
| MountPoints | mount point : volume name | 已安装的硬盘驱动器的映射关系 |
| NetworkProvider | network provider name : subkeys with their values and types | 网络提供商的配置信息 |
| Services | service name : display name : image path : object ID : startup parameters | 服务相关的信息 |
| System | object type : object ID : subkey list | 内核对象、驱动程序、设备和服务的子项列表 |
| UserAssist | program name : executable path : last run time | 用户辅助组件运行时间记录 |
| Users | user name : SID | 当前登录的用户的信息 |
| Windows | component name : version number : subkeys with various information | 操作系统及其组件的版本信息 |

## 2.2 文件系统访问模型
### 2.2.1 请求处理流程
当应用程序或服务需要访问文件时，Windows会检查文件是否在本地磁盘上。如果在本地磁盘上，那么直接从本地硬盘读取文件即可；否则，会根据文件所在位置查找最近的一个距离目标最近的可用服务器，然后再从该服务器下载文件。读取文件的过程主要有三个步骤：

1. 文件名解析：客户端应用程序调用操作系统提供的API函数，传入指定文件路径，得到对应到文件的绝对路径。
2. 文件定位：若文件在本地磁盘上，则在本地硬盘查找并打开文件。若文件不在本地磁盘上，则在文件系统中搜索并打开最近的可用服务器上的文件副本。
3. 数据传输：读写请求的数据包通过网络发送给服务器，然后由服务器回传给客户端。

具体过程如图所示。


### 2.2.2 缓存机制
缓存就是将热点数据暂存在内存中，提高访问速度。Windows中有两种类型的缓存机制：页缓存（Page Cache）和系统缓存（System Cache）。

#### 2.2.2.1 页缓存
页缓存是指将某些文件的内容缓存在内存中，当应用程序请求读取这些文件时，就可以直接从内存中获取数据，避免磁盘I/O带来的延迟。

每个进程都有一个独立的页缓存，页缓存以页为单位进行大小分配。当进程访问一个文件时，首先判断这个文件是否在页缓存中，如果在，则可以直接从内存中读取数据；否则，先将相应页面调入内存，再从内存中读取数据。

当然，页缓存也有其局限性。首先，每次读取文件都会消耗内存资源，如果系统中同时打开的文件过多，或者系统内存不足，就会导致页缓存不够用，导致频繁换出和换入。其次，由于页缓存是针对每个进程的，因此同一个进程中的两个不同文件之间无法共享页缓存，造成缓存效率的降低。

#### 2.2.2.2 系统缓存
系统缓存是指将热点数据直接缓存在内存中，当应用程序请求读取这些文件时，可以直接从内存中获取数据，不需要再去磁盘上读取。

系统缓存又分为数据流缓存和反向快速打开（Reverse Accelerated Failure Detection，Raid）缓存。前者是在不经常访问的数据块中缓存文件，后者用于解决RAID阵列的数据同步问题。

## 2.3 I/O请求模型
### 2.3.1 同步/异步模型
同步I/O模型是指应用程序发起I/O请求后，必须等待I/O操作完成后才能继续执行。异步I/O模型是指应用程序发起I/O请求后，不必等待I/O操作完成后才能继续执行，只要操作完成就通知应用程序，这时应用程序自己负责对结果进行处理。

同步I/O模型中，应用程序的线程被阻塞，直至I/O操作完成，这样容易造成系统资源的浪费。异步I/O模型中，应用程序的线程不会被阻塞，I/O操作完成后，应用程序会收到信号通知，由回调函数进行处理，可以有效减少系统资源的占用。

Windows支持两种同步/异步模型：Windows NT和Win32。NT内核支持同步/异步I/O，并且在设备驱动程序接口（Device Driver Interface，DDK）中提供了两种同步/异步I/O接口。

| DDK API | 同步/异步模型 | 适用场景 |
|---|---|---|
| DeviceIoControl | 同步/异步 | 控制设备或端口 |
| ReadFileEx / WriteFileEx | 异步 | 读写文件 |
| SetEvent / ResetEvent | 同步 | 等待事件发生 |

### 2.3.2 阻塞/非阻塞模型
阻塞I/O模型是指应用程序发起I/O请求后，如果不能立刻获得结果，必须一直等待，直至I/O操作完成才返回结果。非阻塞I/O模型是指应用程序发起I/O请求后，如果不能立刻获得结果，可以尝试一下其他操作，如轮询或超时重试。

异步I/O模型和非阻塞I/O模型都是为了提高应用响应能力，但是不同的应用程序对响应能力的要求不同。对于那些对响应时间敏感的应用程序来说，异步I/O模型更加适合；对于那些对实时性要求较高的应用程序来说，非阻塞I/O模型更加适合。

Windows支持两种阻塞/非阻塞模型：同步I/O模型和系统调用接口（System Call Interface，Syscalls）。

| Syscall | 阻塞/非阻塞 | 参数 | 返回值 | 适用场景 |
|---|---|---|---|---|
| read | 阻塞 | int fd, void *buf, size_t count | ssize_t | 从文件中读取数据 |
| write | 阻塞 | int fd, const void *buf, size_t count | ssize_t | 将数据写入文件 |
| pread | 非阻塞 | int fd, void *buf, size_t count, off_t offset | ssize_t | 类似read，但可以从特定偏移处开始读取 |
| pwrite | 非阻塞 | int fd, const void *buf, size_t count, off_t offset | ssize_t | 类似write，但可以向特定偏移处写入 |

## 2.4 文件系统实现原理
### 2.4.1 NTFS文件系统
NTFS是Microsoft Windows平台最具特色的文件系统。NTFS是一个联机文件系统，能够快速地存储和检索大型、复杂的数据集。NTFS具有强大的容错特性，可以在许多磁盘错误情况下仍然保持完整性，从而保证文件的安全。

NTFS基于日志的体系结构，采用主文件表（Master File Table，MFT）组织文件。MFT记录文件信息，包括文件名、数据位置、属性、时间戳、访问权限和数据流信息等。NTFS还提供事务日志，支持崩溃恢复功能，确保文件系统的一致性。

NTFS支持非常大的磁盘空间，能够处理TB级数据，但磁盘容量有限。NTFS支持许多特性，如目录压缩、虚拟卷、加密、配额、审核、ACL和资源跟踪等，能够满足各种应用需求。

### 2.4.2 Ext3文件系统
Ext3是一种日志文件系统，可以兼顾性能和可靠性。Ext3使用只追加模式写入文件，因此在磁盘填满时可自动建立新的inode。同时，Ext3使用日志的方式来保护文件系统，确保文件系统的一致性。

Ext3支持动态扩展，可以随着文件和目录数量的增加而增长。Ext3支持多种特性，如数据块大小可设置，能够满足一些特殊应用的需求。

### 2.4.3 UDF文件系统
Universal Disk Format（UDF）是一种扩展的文件系统标准，旨在取代ISO9660及其他文件系统。UDF基于日志的设计，提供出色的容错能力。UDF支持文件系统和实体大小的动态调整，可以增减实体大小或添加新实体。

UDF文件系统支持可变长度的实体，且实体可按需申请。UDF可以应对各种类型的工作负载，如高速随机访问、大容量数据库、分层数据共享和多用户环境。

### 2.4.4 Btrfs文件系统
Btrfs（Better Than Raids）是一个新兴的文件系统，其设计理念是“性能优先”，基于块设备，通过合并与分离等技术提升文件系统的性能。

Btrfs使用数据聚簇和索引平衡技巧，通过自动平衡的方式提升整体性能，确保数据完整性和可靠性。Btrfs支持快照功能，可以保存文件系统的快照，甚至可以将多个快照合并，从而实现完整备份功能。

Btrfs支持多种特性，如对称群集、子volume、子卷装载、数据压缩、条带化、垃圾收集、和Copy-On-Write等。

## 2.5 文件系统管理工具
### 2.5.1 fsutil命令
fsutil命令是Windows中用于管理NTFS文件系统的工具。

```bash
fsutil [option] <object> <subobject>
```

| option | 描述 |
|---|---|
| attribute | 查看或修改NTFS文件系统的属性 |
| cache | 显示或清空NTFS文件系统缓存 |
| disk | 获取或设置磁盘配额 |
| file | 显示或修改NTFS文件系统中的文件 |
| fsinfo | 获取文件系统信息 |
|Hardlink|创建NTFS文件系统的硬链接|
| link | 创建NTFS文件系统的符号链接 |
| metrics | 查询NTFS文件系统统计信息 |
| sectorsize | 设置或查询NTFS文件系统的扇区大小 |
| sparse | 设置NTFS文件系统的稀疏属性 |
|usn|查看或修改NTFS文件系统的更新序列号记录|

### 2.5.2 NTFSSecurity命令
NTFSSecurity命令是Windows命令行工具，可用来管理NTFS文件系统的ACL和SD。

```bash
NTFSSecurity [option] [/T:driveletter] filename [securitydescriptorfile|-SDDL]
```

| option | 描述 |
|---|---|
| -A  | 添加ACE |
| -D  | 删除ACE |
| -L  | 显示所有ACL |
| -Q  | 检查ACL |
| -R  | 修改ACL |
| -S  | 修改安全描述符 |
| -X  | 导出安全描述符 |