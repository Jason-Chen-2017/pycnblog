
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Azure Stack Hub是微软推出的基于云的基础设施即服务（IaaS）产品，它可以让客户在自己的数据中心、公有云或私有云中部署和运行其工作负载。作为一个服务提供商，Microsoft Azure Stack Hub将提供基础设施即服务、软件即服务、应用平台即服务以及用户体验即服务等多种IaaS服务，帮助客户将其本地应用和服务迁移到云端。

近日，Azure Stack Hub更新了2个版本。新版的功能包括VM可用性集（AVS），高级网络、安全更新以及支持Ubuntu Server 20.04 LTS。本文将对新版本进行完整的了解并做出评估。

由于篇幅限制，本文仅讨论以下三个方面：

1. AVS功能

2. 高级网络功能

3. Ubuntu Server 20.04 LTS支持情况

对于其他功能特性暂不进行介绍，感兴趣的读者可以参考官方文档获取更多信息。
## 1. AVS 功能
AVS是Azure Stack Hub的一项新功能。顾名思义，它的作用就是提供可扩展性的云计算资源池。客户可以通过创建 Availability Sets (AVs) 将VM组成一个独立的可容错单元，在发生故障时使得虚拟机重新启动到另一个节点上，从而避免单点故障。这是Azure Stack Hub的一个独特之处，相比于传统的云计算服务，它提供了更好的可靠性保证。

### 1.1 创建和管理 AVS 

要创建一个 AVS，需要使用PowerShell cmdlet或REST API调用。下面是一个示例：
```powershell
New-AzsAvsGalleryItem -Location "redmond" `
                      -Name "myAvsGallery" `
                      -ResourceGroupName "system.local" `
                      -SubscriptionId "897d4a5b-0e4c-4b7a-bcaf-50ef571e204c" `
                      -OsType Windows
```
其中，`Location`参数指定了AVS的区域；`Name`参数给定了一个名称；`ResourceGroup`参数确定了AVS所属的资源组；`SubscriptionId`参数指定了用户使用的订阅ID；`OsType`参数指定了VM的OS类型，可以选择Windows或Linux。

当创建完毕后，我们就可以通过门户、PowerShell 或 CLI 来查看 AVS 的详细信息，也可以修改和删除它。如下图所示：

### 1.2 使用 AVS
创建好 AVS 之后，就可以像往常一样向里面添加 VM 了。唯一的区别就是，我们不能指定任何特定的主机来运行它们。如果该主机出现问题，系统会自动将其下所有 VM 从其它可用主机上调换过去。

如下图所示：


不过，如果系统发现某个 VM 的问题较严重，可能会停止该 VM 的自动故障转移，这时候就需要手动干预了。可以使用 PowerShell 或 REST API 来实现。例如，如果某台主机出现故障，我们可以使用 `Move-AzsVM` cmdlet 或 API 来把那台主机上的所有 VM 都移动到另一个主机上。

## 2. 高级网络功能
Azure Stack Hub提供了一种新的网络模型——高级网络。它综合了传统的“网络即服务”和“软件定义网络”的优点。相比于传统的网络模型，高级网络可以在不需要复杂的配置的情况下，将不同的服务整合起来。它具备完整的IPv6网络功能、NAT网关、VPN网关、Load Balancer和应用程序网关等。

### 2.1 配置高级网络
首先，我们需要在门户或者PowerShell中启用高级网络功能。然后，我们可以通过创建网络控制器和边界网关设备等方式，来创建网络。下面是一个示例：

```powershell
$adminUsername = 'azureuser'
$password = ConvertTo-SecureString '<PASSWORD>' -AsPlainText -Force
$location ='redmond'

New-AzsNetworkController -ResourceGroupName $env:ARM_RESOURCEGROUP -DeploymentType ASE_V2 -Location $location -Username $adminUsername -Password $password

$gatewaySubnet = Get-AzsLoadBalancerOutboundIpPool -Name PublicIPAddresses -Location $location | Where {$_.IpAddress -notmatch '/16'}[0]

New-AzsVpc -Name myVirtualNetwork -AddressPrefix 10.0.0.0/16 -DnsServer 10.0.0.4

New-AzsSubnet -Name GatewaySubnet -VirtualNetworkName myVirtualNetwork -AddressPrefix ($gatewaySubnet.IpAddress + "/28")

Add-AzsRouteTable -Name myRouteTable -Location $location

New-AzsLoadBalancer -Name myLoadBalancer -ResourceGroupName system.local -Location $location -Sku Standard -FrontEnd IpConfigs IPv4 PublicIPAddresses -BackEnd AddressPools BackendPool -InboundNatRule NatRules -RoutingRule RouteRules

New-AzsApplicationGateway -Name myAppGateway -ResourceGroupName system.local -Location $location -Tier WAF -InstanceCount 1 -BackendAddressPools BackendPool -FrontendIpConfigurations FrontendIPs -HttpSettingsCollection HttpSettings -Probes Probes -SslCertificates Certificates
```
其中，`$adminUsername`和`$password`是在部署过程中设置的管理员用户名和密码；`$location`指定了部署位置；`Get-AzsLoadBalancerOutboundIpPool` cmdlet 获取了可用于创建前端 IP 池的子网地址范围；`Add-AzsRouteTable` cmdlet 创建了一个空路由表。

这样，我们就完成了配置高级网络的过程。我们还可以通过门户或者PowerShell来查看网络、子网和路由表的详细信息，或者修改它们。