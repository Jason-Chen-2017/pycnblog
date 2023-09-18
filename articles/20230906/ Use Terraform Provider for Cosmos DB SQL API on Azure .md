
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Azure Cosmos DB 是一种完全托管的分布式多模型数据库服务，可快速缩放，通过全球分布、多区域写入和读取保证高可用性。它提供一个面向文档（Document）、图形（Graph）、键值对（Key-Value Pairs）及列系列（Column Family）的五种数据模型，并支持 MongoDB 的 wire 协议，使得客户端能够通过现有的工具和框架来连接到该服务。
在微软 Azure Stack Hub 中，Cosmos DB 作为基础设施即服务（IaaS）产品可以运行于私有云环境中，以支持混合、多云、本地部署等场景。而目前 Terraform 提供了很好的 IaaS 和 PaaS 产品管理能力，但对于 Cosmos DB 的支持却比较薄弱，这就要求开发者们需要自己手动编写或借助其他开源工具来处理，并且开发周期也比较长。为了方便 Terraform 用户更好地管理 Azure Stack Hub 中的 Cosmos DB，我们推出了一款名为 Terraform Provider for Cosmos DB (azurerm_cosmosdb) 的插件，用于在 Terraform 中进行 Cosmos DB 资源的创建、更新、删除等操作。本文将详细介绍该插件的安装、配置、使用方法和注意事项，并提供相关示例，帮助用户更快、更顺畅地管理 Azure Stack Hub 中的 Cosmos DB 资源。

# 2.核心概念
## 2.1.什么是 Terraform？
Terraform 是一个开源的自动化工具，可以用来管理和编排多个云服务平台的基础设施，如 Amazon Web Services (AWS)、Microsoft Azure、Google Cloud Platform (GCP)、vSphere、OpenStack 等，支持众多主流云厂商的 API 接口。其核心理念是通过声明式的配置文件，实现一次配置、多次部署，帮助用户降低运维成本和错误风险，提升效率，减少重复工作。

## 2.2.什么是 Azure Stack Hub？
Azure Stack Hub 是 Microsoft 在 Azure 上构建的私有云解决方案，由 Azure 云中的软件定义基础结构 (SFRP) 来支持，通过 SFRP 可以让客户在自己的内部数据中心内，或者 Azure 以外的其他云平台上部署、运行、管理容器应用、虚拟机、存储等基础设施，并获得同类 Azure 服务的完整集成。Azure Stack Hub 基于 Windows Server 2019 操作系统，兼容 Azure 服务的所有组件，包括计算、网络和存储。

## 2.3.什么是 Terraform Provider？
Terraform Provider 是 Terraform 插件，它是一个独立的 Go 语言库，通过封装各种云服务 RESTful API，提供统一的语法和接口，方便 Terraform 使用。例如 AWS provider 提供了一套资源类型 azurerm_instance，可以用来管理 AWS EC2 主机；azurerm_resource_group 提供了一套资源类型 azurerm_virtual_network，可以用来创建和管理 Azure VNet；azurestack_compute 提供了一套资源类型 azurestack_virtual_machine，可以用来管理 Azure Stack Hub 上的 VM 实例等。

## 2.4.什么是 Cosmos DB SQL API？
Cosmos DB 是一种分布式多模型数据库服务，可以通过 SDK 或 RESTful API 来访问其中的数据，包括文档、图形、键值对及列系列。Cosmos DB 提供了 SQL API 来支持查询和事务处理，Azure Stack Hub 支持 Cosmos DB SQL API 的版本是 2017-11-15。

# 3.安装 Terraform Provider for Cosmos DB
## 3.1.前置条件
- 已安装 Terraform >= v0.12.0 ，如果没有，参考官方文档安装；
- 安装 Git 命令行工具；
- 配置了 Azure CLI，并可以使用命令 az login 登录 Azure 账号；
- 拥有 Azure Stack Hub 环境，可以访问门户界面或执行 API 请求验证 Azure Stack Hub 是否正常；
- 了解 Terraform provider 的机制、安装和配置方法；

## 3.2.安装过程
1. 克隆 Terraform 官方源的代码仓库：

   ```bash
   git clone https://github.com/hashicorp/terraform-provider-azurerm.git 
   cd terraform-provider-azurerm
   ```

2. 查看当前目录下的子目录，确认是否有 vendor 文件夹，如果没有则执行 `go mod vendor` 命令。

3. 修改 Makefile 文件，注释掉所有插件的编译，保留 cosmosdbsqlapi 插件的编译，如下所示：
   
   ```Makefile
   #...
   
   providers:
     	@$(PACKER) build \
    	        $(PACKER_CONFIG)
    
    providers-core:
      	  CGO_ENABLED=0 GOOS=$(shell go env GOOS) GOARCH=$(shell go env GOARCH) go build \
                  -ldflags "-X github.com/hashicorp/terraform-plugin-sdk/version.ProviderVersion=$(VERSION)" \
                  -o bin/$(PROVIDER_BINARY)-$(OS)-$(ARCH)./main
      	  
    # cosmosdbsqlapi:
    #     cd cosmosdb && go build -o../../bin/providers/terraform-provider-azurerm_v2.0.0+$(GITSHA)
    # 
    # cosmosdbsqlapitests:
    #     cd test/cosmosdb && go test -v./...
    
   #...
   ```

4. 执行 `make tools` 命令，下载依赖包。

5. 执行 `make build` 命令，编译出 Terraform Provider for Cosmos DB。

6. 把编译后的文件复制到 Terraform 的 plugins 目录下，一般是 $HOME/.terraform.d/plugins/registry.terraform.io/hashicorp/azurerm/latest 下。

7. 将以下内容保存至 cosmosdb.tf 文件，作为示例代码之一：
   
   ```hcl
   resource "azurerm_cosmosdb_account" "example" {
     name                = "${random_string.prefix.dec}"
     location            = var.location
     resource_group_name = azurerm_resource_group.test.name
     offer_type          = "Standard"
     kind                = "GlobalDocumentDB"
     capabilities {
        name = "EnableAggregationPipeline"
     }
     enable_automatic_failover = true
     consistency_policy {
        consistency_level       = "BoundedStaleness"
        max_interval_in_seconds = 10
        max_staleness_prefix    = 200
     }
     geo_location {
        location          = "West US"
        failover_priority = 0
     }
     connection_strings {
        connection_string         = azurerm_cosmosdb_account.test.primary_connection_string
        connection_string_prefix  = "https://${azurerm_cosmosdb_account.test.name}.documents.azure.com:443/"
        default_consistency_level = "Session"
     }
   }
   
   variable "location" {}
   
   data "azurerm_client_config" "current" {}
   
   locals {
     prefix = format("%s%d", "cosmos-", timestamp())
   }
   
   resource "random_string" "prefix" {
     length           = 4
     special          = false
     upper            = false
     number           = false
     override_special = "-"
     keepers = {
       timestamp = timestamp()
     }
   }
   
   resource "azurerm_resource_group" "test" {
     name     = local.prefix
     location = var.location
   }
   ```

8. 通过命令 `terraform init`，初始化 Terraform 项目。

9. 通过命令 `terraform plan`，检查计划。

10. 通过命令 `terraform apply`，执行变更。

## 3.3.配置 Terraform Provider for Cosmos DB
配置文件中主要包含三个部分：

```yaml
provider "azurerm" {
  version = "=2.1.0"
  
  features {}
  
  skip_provider_registration = false

  subscription_id = "xxxxx-yyyyy-zzzzz"

  tenant_id = "ttt-ttt-ttt-ttt-ttt"

  client_id = "ccccc-bbbbb-aaaaa-fffffffff"

  client_secret = "<KEY>"

  environment = "AzureUSGovernmentCloud"   #(Optional) Default to PublicCloud

  msi_endpoint = ""                    #(Optional) Managed Service Identity endpoint

  use_msi = true                       #(Optional) Use Managed Service Identity Authentication
}


terraform {
  required_providers {
    azurerm = {
      source = "./terraform-provider-azurerm"

      # version = "2.0.0"

      # Configuration options
    }
  }
}
```

其中，`subscription_id`, `tenant_id`, `client_id`, `client_secret` 为必需参数，用于配置 Azure Stack Hub 的凭据信息，如果是在 Azure Stack Hub 上预配资源，则需要指定这些参数。`environment`(可选)，如果要预配的是 Azure US Government Cloud 的资源，需要设置为 `AzureUSGovernmentCloud`。

## 3.4.注意事项
由于 Azure Stack Hub 尚未支持 Key Vault，因此 Terraform Provider for Cosmos DB 还无法支持在 Azure Stack Hub 上预配 Key Vault 资源，这一点需要开发者自行承担。另外，由于 Azure Stack Hub 不支持虚拟网络（VNet），因此 Terraform Provider for Cosmos DB 会忽略设置 subnet_id 属性的行为，这可能会导致预配失败。