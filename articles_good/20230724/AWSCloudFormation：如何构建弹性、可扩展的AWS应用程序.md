
作者：禅与计算机程序设计艺术                    

# 1.简介
         
云计算（Cloud computing）已经成为IT界热门话题，越来越多的企业和开发者开始采用云计算服务来部署自己的应用。而Amazon Web Services(AWS)提供的云Formation(CFN)，则可以帮助用户在云上快速地创建、配置和管理多个资源，并确保这些资源能够按照预期的方式运行。本文将深入CFN内部，探讨其工作机制及关键概念，并通过实例讲述如何利用CFN编排云端环境。最后，结合社区和实际案例分析CFN的优势及局限性。
# 2.核心概念
## 2.1 CFN与CloudFormation
CloudFormation 是一种基于模板的声明式 Infrastructure as Code (IaC) 工具，它允许用户通过定义多个资源及其依赖关系描述整个云基础设施的架构，然后利用 CloudFormation API 或 AWS CLI 将所需的资源以预期的方式部署到云中。与传统的 IaC 工具如 Puppet、Chef、Ansible等相比，CloudFormation 更关注于资源层面上的自动化而不是应用层面的编排，因此更适用于复杂、动态、分布式系统的管理。CloudFormation 使用 YAML 或 JSON 文件作为模板，其中包含了用户希望在云上部署的各种资源及其属性设置。

![](./images/image-20201124215430792.png)


### 2.1.1 CFN和Terraform
CloudFormation 和 Terraform 都是 Cloud provider 提供的 IaC 工具。它们都支持多种编程语言比如 Python、Java、GoLang、JavaScript、Ruby、PowerShell等进行资源编排，但两者又存在一些差异。

1. 语法差异: CloudFormation 的模板是 JSON 或 YAML，但 Terraform 的模板是 HCL。
2. 模板复用: Terraform 可以从公共 Git 仓库或本地文件系统导入共享模块，但是 CloudFormation 不具备这种能力。
3. 生命周期管理: Terraform 可以管理现有的资源，并提供更高级的生命周期管理功能。比如，Terraform 支持 Terraform State，可以在不同阶段回滚之前的资源，而 CloudFormation 只提供对单个堆栈的生命周期管理。
4. 数据源接口: CloudFormation 不支持数据源接口，只能使用 AWS 提供的内置函数来查询资源状态。

综上所述，CloudFormation 更符合 AWS 的产品理念——提供简单的模板驱动方式来编排云资源，同时支持跨平台交互，适合构建分布式系统的架构设计。Terraform 在管理资源方面具有更强大的功能，但相对 CloudFormation 来说较为复杂，在日常的运维工作中需要做更多的配置工作。两者各有千秋，适合不同的场景选择不同的工具。

### 2.1.2 资源类型
CloudFormation 为用户提供了丰富的资源类型，包括 VPC、EC2、EBS、RDS、ELB、EFS、Lambda 等，这些资源类型可以满足用户对云端资源的需求。除了这些固定的资源类型外，用户还可以通过自定义资源来扩展 CloudFormation 的功能。Custom Resource 是一种特殊类型的资源，它由用户自定义的代码实现，并且可以与其他资源一起部署到 CloudFormation 中。比如，用户可以编写一个 Lambda 函数，该函数会接收 CloudFormation 发出的事件通知，当特定条件被触发时，该函数会调用相应的 API 或者执行其他逻辑。这样就可以在 CloudFormation 中利用 Custom Resource 实现更加灵活的自动化运维。

## 2.2 概念架构
CFN的架构比较简单，主要分成两个层次：第一层为云资源层，第二层为CFN Template层。

![](./images/image-20201124220432397.png)



### 2.2.1 云资源层
CFN 会在用户的账户下创建一个名为“CloudFormation”的虚拟机（即Stack），所有的资源都部署在这个虚拟机上。

![](./images/image-20201124220819614.png)



CloudFormation 通过不同的 API 请求向云资源层发送指令，请求包括创建、更新、删除资源、更新堆栈的状态等。云资源层的API将指令转译成底层的资源管理系统的指令，比如AWS EC2 API来实现实际的创建和删除操作。云资源层向 CloudFormation 返回指令执行结果，比如成功、失败或者中止信号。

### 2.2.2 CFN Template层
CloudFormation 的 Template 是一个文本文件，定义了整个堆栈的蓝图。Template 由四个主要部分组成：参数、资源、输出和堆栈设置。

**参数：**

参数用于定义堆栈中使用的变量，可以使得相同的模板可以部署到不同的环境中，例如测试、预生产、正式环境。参数的值可以直接在 Template 中指定也可以通过外部文件或命令行传入。

```json
{
  "Parameters" : {
    "ImageId": {
      "Description": "The AMI to use for the instance",
      "Type": "String",
      "Default": "ami-a1b2c3d4e5f"
    },
    "InstanceType": {
      "Description": "The size of the instance to launch",
      "Type": "String",
      "Default": "t2.micro",
      "AllowedValues": [
        "t2.nano",
        "t2.micro",
        "t2.small",
        "t2.medium",
        "t2.large"
      ],
      "ConstraintDescription": "must be a valid EC2 instance type."
    }
  }
}
```

**资源：**

资源是 CloudFormation 中的核心部分，这里面定义了实际要部署到的云资源信息。资源可以定义很多不同的属性，例如启动配置、安全组、网络配置、启动脚本等，都可以由 CloudFormation 管理。

```json
{
  "Resources" : {
    "MyEC2Instance" : {
      "Type" : "AWS::EC2::Instance",
      "Properties" : {
        "ImageId"        : {"Ref": "ImageId"},
        "InstanceType"   : {"Ref": "InstanceType"}
      }
    }
  }
}
```

**输出：**

输出是 CloudFormation 中的另一个重要部分，用于导出堆栈中的值，以便后续操作或者查看日志。

```json
{
   "Outputs":{
      "InstanceId":{
         "Value":{"Ref":"MyEC2Instance"}
      }
   }
}
```

**堆栈设置：**

堆栈设置包含用于控制堆栈行为的参数，比如超时时间、失败策略、角色等。

```json
{
  "AWSTemplateFormatVersion" : "2010-09-09",

  "Description"              : "This is an example template.",

  "Metadata"                 : {
    "Example": "Data can be placed here and accessed by Fn::GetAtt"
  },

  "Mappings" : {
    "RegionMap" : {
      "us-east-1"      : {"AMI": "ami-bbccddeeff"},
      "us-west-2"      : {"AMI": "ami-aaabbbccdd"}
    }
  },
  
 ...
  
   "TimeoutInMinutes": "60",

   "DisableRollback": "true",

   "UpdatePolicy": {
     "AutoScalingRollingUpdate": {
       "MinInstancesInService": "1",
       "MaxBatchSize": "1",
       "PauseTime": "PT5M"
     }
   },
   
  ...
   
    "RoleARN"                  : { "Fn::GetAtt" : ["IAMRole", "Arn"]},
    
    "NotificationARNs"         : [],

    "Capabilities"             : [ "CAPABILITY_NAMED_IAM" ]
}
```

## 2.3 工作原理
CFN的工作流程如下：

1. 用户使用CloudFormation API或CLI创建或更新模板，提交至CloudFormation服务。
2. 服务生成一个新的堆栈，包含整个模板，如果不存在就创建一个新的。
3. CloudFormation对模板进行解析，生成一个“Change Set”，作为创建或更新堆栈所需的内容。
4. 如果创建新堆栈，服务会创建对应的“CloudFormation虚拟机”VM。
5. CFN会在VM中启动指定数量的资源，并根据资源的依赖关系依次启动。
6. 当所有资源启动完成后，CFN会生成一个“SUCCESS”消息。

### 2.3.1 模板解析器
CFN服务会首先进行模板解析。解析器会读取模板并验证语法正确，并将其转换成内部模型。解析器对模板中的每个资源都会检查其名称是否唯一。

### ChangeSet
CFN在执行更改前，会先创建一个“ChangeSet”。这个“ChangeSet”是一个JSON文件，描述了所有要更改的资源及其属性，以方便用户确认更改内容。

### 执行器
CFN服务收到请求后，会创建一个执行器（Executor）。执行器会负责执行模板中的更改。当执行器收到指令后，它会与底层资源管理系统通信，请求创建或更新指定的资源。资源管理系统会根据资源的依赖关系依次处理这些请求。

### 等待器
CFN执行器创建好资源后，会等待资源的状态变成“CREATE_COMPLETE”或“UPDATE_COMPLETE”。如果一直处于“ROLLBACK_IN_PROGRESS”状态，CFN会终止部署过程。

### CFN栈
CFN服务在成功创建或更新堆栈后，会创建一个对应于堆栈的“CloudFormation虚拟机”VM。VM中包含了整个堆栈的配置，包括堆栈中的资源及其属性。如果堆栈存在，VM的ID会作为更新后的堆栈的ID返回给用户。

### CFN自定义资源
Custom Resource是在CFN堆栈模板中定义的一种资源。它由一个自定义的Lambda函数实现，可以通过定义接口来接收模板中的输入参数。资源在更新堆栈时，CFN会调用自定义资源的Lambda函数，并将其配置参数传送过去。Lambda函数通过接口返回相应的结果，这个结果会反映在CFN模板的输出参数中。

## 2.4 操作方法
### 2.4.1 创建堆栈
CFN提供了两种创建堆栈的方法：使用CFT API 或 CLI。

#### 方法一：CFT API
可以使用POST方法向cloudformation地址发送RESTful请求。URL为https://cloudformation.amazonaws.com， headers中包含了“Content-Type”：“application/json”、“X-Amz-Target”：“AWSCloudFormation_20100331.CreateStack”，body中的JSON字符串内容为一个模板。

```javascript
const aws = require('aws-sdk');

let cfn = new aws.CloudFormation();

cfn.createStack({
  StackName:'stack-name', // 堆栈名称
  TemplateBody: '{... }' // 模板内容
}).promise()
.then(() => console.log('stack created'))
.catch((err) => console.error(err));
```

#### 方法二：CLI
可以通过cf create命令来创建新的堆栈。该命令将会读取模板内容并将其发送至服务端，然后根据模板内容启动新创建的堆栈。

```bash
cf create-stack stack-name -t./template.yaml [--capabilities CAPABILITY1...CAPABILITYn] # --capabilities参数可选，指定了堆栈所需要的权限
```

### 2.4.2 更新堆栈
更新堆栈的方法也有两种，分别是CFT API 和 CLI。

#### 方法一：CFT API
可以使用PUT方法向cloudformation地址发送RESTful请求。URL为https://cloudformation.amazonaws.com， headers中包含了“Content-Type”：“application/json”、“X-Amz-Target”：“AWSCloudFormation_20100331.UpdateStack”，body中的JSON字符串内容为一个模板。

```javascript
const aws = require('aws-sdk');

let cfn = new aws.CloudFormation();

cfn.updateStack({
  StackName:'stack-name', // 堆栈名称
  TemplateBody: '{... }' // 模板内容
}).promise()
.then(() => console.log('stack updated'))
.catch((err) => console.error(err));
```

#### 方法二：CLI
可以通过cf update命令来更新已有的堆栈。该命令读取模板内容并将其发送至服务端。如果待更新的模板内容与当前堆栈的模板内容一致，则不会执行任何操作。

```bash
cf update-stack stack-name -t./template.yaml [-p parameter1=value1...] [--use-previous-template | --rollback-on-failure] #--use-previous-template参数表示使用当前堆栈的模板内容，不执行任何操作；--rollback-on-failure参数表示出现错误时，自动回滚堆栈的修改。
```

### 2.4.3 删除堆栈
删除堆栈的方法也有两种，分别是CFT API 和 CLI。

#### 方法一：CFT API
可以使用DELETE方法向cloudformation地址发送RESTful请求。URL为https://cloudformation.amazonaws.com， headers中包含了“Content-Type”：“application/json”、“X-Amz-Target”：“AWSCloudFormation_20100331.DeleteStack”，body中的JSON字符串内容为空。

```javascript
const aws = require('aws-sdk');

let cfn = new aws.CloudFormation();

cfn.deleteStack({
  StackName:'stack-name' // 堆栈名称
}).promise()
.then(() => console.log('stack deleted'))
.catch((err) => console.error(err));
```

#### 方法二：CLI
可以通过cf delete命令来删除已有的堆栈。

```bash
cf delete-stack stack-name 
```

### 2.4.4 查询堆栈
查询堆栈的方法有两种，分别是CFT API 和 CLI。

#### 方法一：CFT API
可以使用GET方法向cloudformation地址发送RESTful请求。URL为https://cloudformation.amazonaws.com，headers中包含了“Content-Type”：“application/json”、“X-Amz-Target”：“AWSCloudFormation_20100331.DescribeStacks”，body中的JSON字符串内容为空。

```javascript
const aws = require('aws-sdk');

let cfn = new aws.CloudFormation();

cfn.describeStacks({
  StackName:'stack-name' // 堆栈名称
}).promise()
.then((data) => console.log(data))
.catch((err) => console.error(err));
```

#### 方法二：CLI
可以通过cf describe命令来查询堆栈的信息。

```bash
cf describe-stacks stack-name
```

### 2.4.5 获取模板
获取模板的方法有两种，分别是CFT API 和 CLI。

#### 方法一：CFT API
可以使用GET方法向cloudformation地址发送RESTful请求。URL为https://cloudformation.amazonaws.com，headers中包含了“Content-Type”：“application/json”、“X-Amz-Target”：“AWSCloudFormation_20100331.GetTemplate”，body中的JSON字符串内容为空。

```javascript
const aws = require('aws-sdk');

let cfn = new aws.CloudFormation();

cfn.getTemplate({
  StackName:'stack-name' // 堆栈名称
}).promise()
.then((data) => console.log(data))
.catch((err) => console.error(err));
```

#### 方法二：CLI
可以通过cf get-template命令来获取堆栈模板内容。

```bash
cf get-template stack-name > template.yaml
```

### 2.4.6 查看堆栈事件
查看堆栈事件的方法有两种，分别是CFT API 和 CLI。

#### 方法一：CFT API
可以使用GET方法向cloudformation地址发送RESTful请求。URL为https://cloudformation.amazonaws.com，headers中包含了“Content-Type”：“application/json”、“X-Amz-Target”：“AWSCloudFormation_20100331.DescribeStackEvents”，body中的JSON字符串内容为空。

```javascript
const aws = require('aws-sdk');

let cfn = new aws.CloudFormation();

cfn.describeStackEvents({
  StackName:'stack-name' // 堆栈名称
}).promise()
.then((data) => console.log(data))
.catch((err) => console.error(err));
```

#### 方法二：CLI
可以通过cf events命令来查看堆栈事件。

```bash
cf events stack-name
```

