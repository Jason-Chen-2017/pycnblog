
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Bastion 是微软 Azure 提供的一项服务，它提供了一个安全且无缝的访问方式，使管理员可以直接通过浏览器或通过 RDP 或 SSH 在 Azure 虚拟机上执行管理任务。它可以帮助防止暴力破解攻击、安全漏洞等攻击者对 Azure 环境的侵入。在本文中，我将向读者介绍什么是 Bastion 和如何使用它。

## 1.为什么需要使用 Bastion？
如果你是云计算领域的高级工程师或者系统架构师，或者你所在的企业有 Azure 的使用需求，那么你可能已经接触到 Azure 上的一些虚拟机（VM）。但是，当你想从外部访问这些 VM 时，可能会面临一些挑战。例如，你可能需要使用公网 IP 地址来访问它们，但这样会导致两个问题：

1. 公网 IP 地址是公开可用的，任何人都可以访问你的资源；
2. 如果攻击者获得了你的公网 IP 地址，他就可以直接连接到你的 VM，从而造成严重的安全威胁。

解决以上两个问题的一个方法就是使用 Bastion，它允许你在 Azure 上创建一个安全隧道，并通过它访问内部 VM。

Bastion 还可以通过以下方式提升你的工作效率：

- 通过控制板管理你的 Azure 环境；
- 使用基于 web 的端口转发功能，远程登录到 Azure VM；
- 配置磁盘加密；
- 在运行 Windows VM 时，保护你的 PowerShell 会话免受暴力攻击。

总之，如果你的组织希望减少对公网 IP 地址的依赖，可以使用 Bastion 来提升 Azure VM 的安全性。

## 2.Azure Bastion 是什么？
Azure Bastion 是 Azure 平台内的一个服务，它为非托管的虚拟机 (VM) 提供了一个安全且无缝的访问方式。你可以通过 Azure 门户、REST API、Azure CLI 或 Azure PowerShell 创建一个 Azure Bastion 主机，然后利用它来访问你的虚拟机。与其他 Azure 服务不同的是，Azure Bastion 不需要预先部署，只需创建一个新的资源即可，并且可以快速、轻松地与 Azure 资源集成。在创建 Bastion 后，它将自动关联到 Azure 虚拟网络中的子网，因此你可以选择要加入哪个子网。Bastion 主机有一个公共 IP 地址，可以通过它进行 RDP/SSH 访问。除了能够管理 Azure 虚拟机外，还可以通过它访问 Azure 虚拟网络中的所有资源。


## 3.如何使用 Bastion？
### 3.1 创建 Bastion 主机
首先，你需要创建一个新的 Azure Bastion 主机，如下图所示：

1. 打开 Azure 门户；
2. 选择“新建” > “网络” > “Bastion”。


在“创建 Bastion”页中，指定以下设置：

- **名称**：Bastion 主机的唯一名称。
- **订阅**：要在其中创建 Bastion 主机的 Azure 订阅。
- **资源组**：要在其中创建 Bastion 主机的资源组。
- **位置**：要在其中创建 Bastion 主机的区域。
- **虚拟网络**：Bastion 主机应附属于某个 VNet。你可以选择现有的 VNet，也可以创建一个新的 VNet。如果你选择创建新 VNet，则需要指定 VNet 的名称、地址空间、子网的名称及其范围。
- **子网**：VNet 中 Bastion 主机的子网。 你应该选择具有足够 IP 地址数量的子网 (/27 或更大的子网，以便容纳更多 VMs)。
- **公共 IP 地址**：Bastion 主机将分配到的静态公共 IP 地址。
- **公钥**: 如果想要使用密钥进行连接，可以选择此选项并上传公钥文件 (.pem 文件)，然后下载该密钥用于连接。

创建完成后，你的 Bastion 主机就会出现在资源组中。

### 3.2 创建 Azure 虚拟机
在配置 Bastion 之后，可以创建 Azure 虚拟机 (VM)。按照[在 Azure 上创建 Linux 虚拟机](~/articles/virtual-machines/linux/quick-create-cli.md)或[在 Azure 上创建 Windows 虚拟机](~/articles/virtual-machines/windows/quick-create-powershell.md)中的说明操作，在 VNet 中创建 VM。

### 3.3 连接到 Azure 虚拟机
创建 Bastion 主机和 VM 以后，就可以通过以下两种方式连接到 VM：

**选项 1: 使用 Azure 门户**

1. 在 Azure 门户中，找到你要连接到的 VM；
2. 在 VM 的页面上单击“连接”，然后选择“Bastion”。


**选项 2: 使用 RDP/SSH 客户端**

如果已安装适用于 Windows 的远程桌面客户端或 macOS 或 Linux 的命令行 SSH 客户端，则可以直接从本地计算机连接到 VM。若要获取连接信息，请参阅[管理 Azure Bastion - 获取信息](/azure/bastion/bastion-vm-manage#get-information-about-a-vm)。

### 3.4 使用 VPN 网关连接到本地网络
如果你想要通过 Azure 虚拟网络 (VNet) 中的 Bastion 主机访问本地网络资源，可以通过 VPN 网关来实现。首先，需要在本地网络中设置一个网关设备，再在 VNet 中创建一个网关子网。有关详细信息，请参阅[关于 VPN 网关](/azure/vpn-gateway/)。