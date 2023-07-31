
作者：禅与计算机程序设计艺术                    
                
                
许可证（License）是一个定义了使用者可以获得什么样的权利、义务、限制或者条件的法律文件。由于许可证对使用者权利与义务的约束性，使得第三方可以使用软件而不用担心侵犯版权或其他任何法律责任。许可证分为两个主要类型——专利授权协议（Patent License）和著作权许可协议（Copyright License）。
根据巴尔迪克特·桑斯坦（Brian Stansbury）的观点，Open Source Initiative (OSI) 组织定义了开源软件许可证（OSL），它是符合自由软件基金会（FSF）和开放源代码运动（Open Source Movement）精神的开源软件许可证之一。目前，开源软件已经成为当今最热门的话题，包括 Linux 操作系统、Git 源码管理工具、Python 和 Node.js 的软件包等。
今天，笔者将探讨两种开源许可证——BSD 和 OpenJDK 的区别。这两种许可证都属于 OSL，都是几十年前由 FSF 创建的许可证。因此，它们在许可证条款的原则上应该有相似性，但又存在着一些细微差别。这篇文章旨在阐述这些差异并分析它们之间的关联。
# 2.基本概念术语说明
## 2.1 BSD(Berkeley Software Distribution)
- Berkeley Software Distribution 是 AT&T Bell Laboratories 以 BSD 许可证授权发布的 Unix 操作系统。它采用了 BSD 授权条款，该授权条款赋予用户高度的自由权限，允许对其进行修改，再配合使用，但是也禁止对其进行任何商业目的的分发。
- 在 1970 年代中期至 2000 年代初期，BSD 许可证在美国和加拿大的使用者群体之间广泛流通。由于它采用简单宽松的许可协议，所以很快就受到各个领域的广泛关注。
- 它提供的是开发者友好的操作系统，具有高效率的实时性能，以及较低的系统开销。同时它也提供了适用于各种设备的丰富的应用编程接口（API）。
- BSD 兼容性：BSD 系列软件几乎可以在所有的类Unix平台上运行，包括 Linux，Solaris，FreeBSD，macOS 等。此外，多数系统都内置了 BSD 系列软件，因此只要满足相应的硬件需求，就可以轻易安装和使用。
## 2.2 OpenJDK
OpenJDK（Open Java Development Kit）是由 Oracle 于 2007 年推出的免费和开放源代码的 Java 开发环境。OpenJDK 可以运行在各种操作系统（如 Windows、Linux、macOS、FreeBSD 等）上。它支持 Java SE API 的所有特性，并实现了 Java EE、Java ME 等扩展规范。
- OpenJDK 是基于 GPL（General Public License）许可证版本 2 构建的，因此任何应用代码都可以利用OpenJDK 来运行，但不能直接分发到商业产品中。但是，如果需要将OpenJDK 重新分发到商业产品中，则必须遵守Oracle公司的商业许可条款。
- OpenJDK 也支持 Java Native Interface（JNI）功能，允许使用 Java 语言编写的程序调用本机应用程序库。
- OpenJDK 支持不同的 Java 虚拟机，其中包括 HotSpot VM（用于现代商用 Java 应用程序）、OpenJ9 VM（用于较新的企业级 Java 应用程序）、Zing VM（用于压缩后的垃圾回收 Java 应用程序）等。
- OpenJDK 的优势主要体现在以下几个方面：
    - OpenJDK 提供了完全兼容的 Java API，可用于编写运行在 OpenJDK 上面的应用；
    - OpenJDK 提供了跨平台能力，可用于运行在各种操作系统及硬件平台上；
    - OpenJDK 的配置灵活，可以方便地调节 JVM 的内存分配、线程模型、GC 方式等参数；
    - 开源的OpenJDK 许可协议鼓励应用开发者进行代码的开源共享，促进共同开发。

