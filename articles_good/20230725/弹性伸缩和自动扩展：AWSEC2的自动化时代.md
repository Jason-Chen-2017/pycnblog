
作者：禅与计算机程序设计艺术                    

# 1.简介
         
随着云计算的兴起，传统的服务器及存储设备已经不再适应当前的业务需求，而基于云的服务则提供可弹性扩缩容的功能，使得应用能够根据需要自动调整资源。相对于传统的服务器，云主机价格低廉，弹性伸缩的配置简单，降低了成本，但是同时也带来了复杂度和管理难度。尤其是在运行多租户应用或容器化的应用时，容器编排引擎或调度平台等管理工具的出现促进了弹性伸缩的实现。

AWS EC2 是目前 AWS 提供的公共云计算服务之一，也是许多企业用户首选的基础设施即服务产品。EC2 具有弹性伸缩和自动扩展的能力，能够动态调整计算资源的利用率、自动添加或移除实例，通过 Amazon CloudWatch 服务进行监控并实时收费，是实现高可用性和节约成本的有效解决方案。作为企业用户，我们应该清楚 EC2 在自动伸缩方面的优势，理解 EC2 中的相关概念和术语，掌握如何使用 AWS CLI 或 API 来控制实例的启动、停止和重启，以及如何使用 Amazon CloudWatch 和 Amazon Elastic Load Balancing 服务来收集和分析实例的监控数据，以便对系统性能和业务需求进行优化。此外，还应该了解不同类型的 EBS（Elastic Block Store）卷在 EC2 中的作用，以及 EBS 的生命周期管理方法，以帮助我们合理分配磁盘空间并确保实例的持久性。最后，还要熟悉 AWS Auto Scaling 服务的工作原理，掌握如何设置规则和策略来触发伸缩活动，以及在不同的部署场景下使用的最佳实践。

# 2.基本概念术语说明
## EC2
Amazon Elastic Compute Cloud (EC2) 是一种基于 web 服务的计算服务，它允许您 rent virtual machines in the cloud, which are like physical computers but are hosted on a virtualized server rather than on dedicated hardware. You can use these instances to run applications and host databases, servers, or other computing needs, all the while being billed by the hour for what you actually use. EC2 provides many different types of instance sizes, ranging from small single-purpose computers to powerful GPU-powered high-performance clusters that can handle large amounts of workloads at once. Each EC2 instance is designed to be highly available, meaning it has redundant processing, memory, and networking capabilities to ensure your application continues to function even if one part of its infrastructure fails. Additionally, EC2 offers flexible pricing options based on usage rates and pre-configured software licensing that make it easy to pay only for what you need when you need it.

An EC2 instance runs on top of a virtual machine running inside an isolated environment called a VPC. This means each EC2 instance is independent and has its own set of resources such as CPU, RAM, storage space, and network connectivity. As with any other resource in the cloud, the cost of running an EC2 instance depends on factors including region, instance type, operating system, database size, and workload requirements. However, unlike traditional on-premises data centers where you have fixed costs associated with power, cooling, electricity, and network connections, there are no upfront commitments required when using EC2. Instead, you only pay for the EC2 instances you use while they are running, and you can choose from several payment options such as hourly billing or annual contracts.

## EBS
EBS (Elastic Block Store) 是一种块级存储，可以在 EC2 中作为根设备或者数据盘使用，可以支持高 I/O 吞吐量，非常适合于 I/O 密集型应用程序和高性能数据库。EBS 可以被映射到任何 EC2 实例，并且可以配置为自动扩展大小，以应对突然增加的存储需求。每一个 EBS 卷都是一个完整的文件系统，可以用于存放操作系统文件、数据库文件、日志文件等。当 EC2 实例关闭或者重启后，EBS 卷中的数据也将会丢失。因此，EBS 不适合作为长期的数据存储，仅限临时存放文件、数据库、缓存等。如果想要保存长期数据，建议使用 AWS EFS (Elastic File System)。

每个 EC2 实例可以挂载多个 EBS 卷，并且这些卷可以是相同的类型或不同的类型，取决于您的工作负载需要。例如，你可以创建一个带有系统盘和两个数据盘的 EC2 实例，其中系统盘用于安装操作系统，数据盘用于存放应用程序的源代码、日志文件和数据文件。另外，也可以使用 RAID 将多个 EBS 卷组合成一个更大的存储池，以提升存储性能和可靠性。

除了提供经济且高性能的存储之外，EBS 还有其他一些特性值得关注，如快照、加密、访问控制、监控和备份。我们可以通过 EBS 对磁盘进行快照，创建以前某个时间点的备份，也可以对整个磁盘或单个分区加密，以满足不同级别的安全要求。可以选择限制特定 IP 地址或子网对某个 EBS 卷的访问权限，或者监控 EBS 使用情况，防止过度占用资源。

## Instance Types
Instance Type 是 EC2 提供的一系列实例模板，包括各种配置和规格，比如 CPU 数量、内存大小、硬盘大小、网络速度等。EC2 为每一种实例类型提供了统一的、一致的操作系统，内核版本和软件配置，使得开发人员可以轻松地移植他们的应用程序到云中运行。常用的实例类型包括 t2 系列的低价廉的计算实例，c5 系列的高性能计算实例，m5 系列的通用计算实例，r5 系列的内存优化实例，g3 系列的高端 Graphics 加速实例等。

EC2 实例由两部分组成，分别是网络接口控制器 (NIC) 和实例存储。网络接口控制器负责处理网络通信，实例存储提供磁盘阵列以支持操作系统使用。实例类型中包括各种实例，其中 GPU 加速实例通常配有一个 NVIDIA CUDA 框架，可以利用 Nvidia Tesla GPUs 进行图形处理加速。

## Auto Scaling Group
Auto Scaling Group （缩放组） 是一种 Amazon Web Services 提供的服务，可以让用户根据实际需求自动增加或减少 EC2 实例的数量，从而实现应用的高可用性。当集群中某台服务器的负载过高时，Auto Scaling Group 会自动添加新的 EC2 实例以均衡负载；当集群中某台服务器的负载较低时，Auto Scaling Group 会自动减少 EC2 实例以节省成本。Auto Scaling Group 支持多种配置选项，包括按需伸缩、预置期和报警阈值等，可以灵活调整自动伸缩策略以满足用户的各种业务需求。

## Launch Template
Launch Template （启动模板） 是一种 Amazon EC2 提供的服务，它可以方便地批量创建 EC2 实例。启动模板可以指定实例类型、AMI、安全组、密钥对、IAM 角色等信息，并提供批量创建实例的能力，适合于启动大量的同类实例。

## Amazon CloudWatch
Amazon CloudWatch 是 Amazon Web Services 提供的一项实时性能监视服务，可以帮助用户跟踪和分析 AWS 资源、应用程序和第三方服务的性能指标，并设置通知和自动调整策略，提升资源利用率和应对突发事件。CloudWatch 可以从实例、系统组件、应用程序和第三方服务等多个维度获取性能指标，包括 CPU、内存、网络流量、磁盘读写次数、请求队列长度、错误日志数量等。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 弹性伸缩
弹性伸缩是一种云计算服务，提供动态增加或删除资源的能力，以应对业务增长或变化导致的负载波动。它能够根据当前的负载需求快速的增加或减少计算能力，从而实现应用的高可用性和节省资源开销。弹性伸缩的核心是横向扩展，通过增加计算节点的方式来提升应用的处理能力。弹性伸缩既可以用来水平扩展，也可以用来垂直扩展，比如从一个小型服务器集群扩展到一个更大的服务器集群，也可以从多核 CPU 扩展到多层次结构的并行计算集群。

弹性伸缩的关键在于自动识别并纠正业务模式变化，以保证应用程序始终能够提供足够的资源以满足需求。弹性伸缩一般采用两种方式：手动和自动。

### 手动伸缩
手动伸缩就是管理员定期查看系统状态，然后根据需要调整服务器的数量，以达到最佳的利用率和性能。这种方式简单易用，但不够灵活，不能应对不规则的业务需求，缺乏自动化的能力。因此，手动伸缩一般只适用于单体架构或新技术开发阶段。

### 自动伸缩
自动伸缩是一种通过计算机程序执行自动调整计算资源的能力，以响应不断变化的业务需求。自动伸缩的优点是简单、可靠、灵活，能够自动匹配业务变化并调整计算资源，因此能够兼顾效率和效益。

#### 垂直自动伸缩
垂直自动伸缩 (Vertical Autoscaling) 指的是自动增加或减少计算节点的数量，主要用于处理计算密集型应用，比如基于超算平台的大数据分析。由于超算平台通常具有多种不同型号的计算资源，因此需要根据应用的特点选择最合适的节点配置。自动伸缩的目标是增加或减少计算资源的利用率，而不是增加服务器的数量。

典型的场景如下：

1. 用户请求增加计算资源，超算平台会自动添加更多的计算节点，以提升应用的处理能力。

2. 如果用户请求减少计算资源，超算平台会自动减少计算节点，并释放空闲的计算资源，以节省资源开销。

垂直自动伸缩的过程如下：

1. 用户提交增加计算资源的申请。

2. 超算平台检测到增加资源请求，发送通知给用户。

3. 用户通过界面或 API 查询当前超算平台的可用计算资源。

4. 用户选择所需配置的计算资源类型，并输入相应的参数。

5. 超算平台根据用户输入参数自动调整计算资源配置，启动新增的计算节点。

6. 用户验证新增的计算节点是否正常运行。

7. 用户提交减少计算资源的申请。

8. 超算平台检测到减少资源请求，发送通知给用户。

9. 用户通过界面或 API 查询当前超算平台的可用计算资源。

10. 用户选择所需配置的计算资源类型，并输入相应的参数。

11. 超算平台根据用户输入参数自动调整计算资源配置，关闭不必要的计算节点，释放资源。

#### 横向自动伸缩
横向自动伸缩 (Horizontal Autoscaling) 指的是自动增加或减少服务器节点的数量，主要用于处理任务密集型应用。任务密集型应用的特点是有大量的并发任务需要处理，如果没有足够的服务器资源，它们很可能会拖慢整体的处理速度。因此，自动伸缩的目标是增加服务器的数量以支持任务的并发处理。

典型的场景如下：

1. 用户上传大型数据集，任务处理平台会自动添加更多的服务器节点，以提升数据处理能力。

2. 如果用户请求减少服务器资源，任务处理平台会自动减少服务器节点，并释放空闲的服务器资源，以节省资源开销。

横向自动伸缩的过程如下：

1. 用户上传大型数据集。

2. 数据处理平台检测到数据量增加，发送通知给任务处理平台。

3. 任务处理平台通过查询负载情况发现有大量任务等待处理，并自动增加服务器资源。

4. 任务处理平台启动更多的服务器节点，以提供处理数据的能力。

5. 当负载下降时，任务处理平台会自动减少服务器资源，释放空闲的服务器资源，以节省资源开销。

#### 混合自动伸缩
混合自动伸缩 (Hybrid Autoscaling) 是一种综合垂直和横向自动伸缩的方式，结合了这两种类型的自动伸缩策略。混合自动伸缩能够自动调整计算资源的利用率和数量，以便更好地满足用户的业务需求。

### 弹性伸缩流程
1. 创建 Auto Scaling Group 。用户创建一个新的 Auto Scaling Group ，定义组内 EC2 实例的配置，如实例类型、可用区、最大和最小实例数、生命周期 hook 配置等。

2. 启用 Auto Scaling 功能 。用户启用该组的 Auto Scaling 功能，开启自动扩展和自动缩容功能。

3. 设置自动伸缩策略 。用户设置自动伸缩策略，定义触发自动伸缩的条件，如资源使用率、请求队列长度、请求超时、外部事件等。

4. 测试 Auto Scaling 策略 。用户测试 Auto Scaling 策略，模拟应用负载变化，确认 Auto Scaling 策略能否正确触发自动伸缩操作。

5. 完成 Auto Scaling 配置 。用户完善 Auto Scaling 策略，设置生效条件和持续时间等参数，完成所有配置。

### 弹性伸缩原理
弹性伸缩的原理主要由以下三个方面构成：

1. 监测和预测。自动伸缩依赖于对系统的实时数据进行监测和预测，判断当前负载的趋势和可能出现的峰值。

2. 执行自动操作。当监测到负载过高或趋势改变时，自动伸缩会采取自动操作，提升或减少系统资源的利用率。

3. 自动运维。弹性伸缩与自动运维紧密结合，运维人员可以及时处理故障和异常，确保系统的持续稳定运行。

#### 弹性伸缩实例
弹性伸缩实例 (Elasticity Instance) 是 AWS 提供的一种按需计算服务，可以通过实例的启动和停止来动态分配计算资源，从而帮助客户避免昂贵的服务器购买，以满足业务的需求。

弹性伸缩实例的优点如下：

1. 成本效益。弹性伸缩实例的价格比购买新服务器划算，而且无需担心服务器停机或性能下降的问题。

2. 按需扩展。弹性伸缩实例可以自动根据应用的负载增减资源，无需手动管理服务器，保证资源的最佳利用率。

3. 灵活部署。弹性伸缩实例可部署在私有或公有云上，支持各种规模的应用。

4. 可伸缩性。弹性伸缩实例无需对应用进行任何改造即可实现自动扩展和缩容，具备高度的可伸缩性。

## 自动扩展
自动扩展是指云服务提供商根据应用的使用情况或配置要求，自动添加或删除服务器资源以处理日益增长的用户请求，从而帮助客户应对不断变化的业务需求。自动扩展可以有效减少服务器投入、降低运营成本、提高服务质量。在 EC2 上，自动扩展是通过 Amazon EC2 Auto Scaling 完成的，它通过一组独立的服务来实现弹性伸缩。

EC2 Auto Scaling 是一项 Amazon Web Services 提供的服务，可以自动添加或删除 EC2 实例来处理应用程序的负载增加或减少。它能根据指定的自动伸缩策略自动调整 EC2 实例的数量，从而处理实例负载增加或减少，保持服务的最佳性能和可用性。EC2 Auto Scaling 还提供监控功能，实时记录 EC2 实例的 CPU 使用率、网络连接数、磁盘 IO 等指标，并提供通知和警告机制，以便及时发现并诊断故障。

EC2 Auto Scaling 通过以下几个步骤来实现弹性伸缩：

1. 创建 Auto Scaling Group。首先，用户需要创建一个 Auto Scaling Group，定义该组中的 EC2 实例的配置、最大和最小实例数、健康检查设置、通知设置等。

2. 确定触发自动伸缩的条件。其次，用户需要确定自动伸缩的触发条件。比如，用户可以使用 Amazon CloudWatch 服务对 EC2 实例的 CPU 使用率、网络连接数等指标进行监控，当超过一定阀值时，触发自动伸缩。

3. 设置自动伸缩策略。第三步，用户设置自动伸缩策略，定义 EC2 实例的增加和减少的间隔时间、初始实例数、最大实例数、目标 CPU 使用率、网络连接数、磁盘 IO 等。

4. 测试并验证策略。第四步，用户测试 Auto Scaling 策略，模拟应用负载的变化，确认策略能否正确触发自动伸缩操作。

5. 验证并更新配置。第五步，用户验证 EC2 实例的自动扩展是否正确，根据需要修改策略和通知设置。

## 自动扩展原理
自动扩展的原理主要由以下几个方面构成：

1. 监控。自动扩展依赖于对系统的实时数据进行监测，实时获取用户的请求，并判断应用的容量是否能够满足用户的需要。

2. 分配资源。当应用的使用量增加时，自动扩展会主动添加服务器资源，提升服务能力。

3. 回收资源。当应用的使用量减少时，自动扩展会主动删除服务器资源，节省资源开销。

4. 配置服务器。自动扩展会根据用户的配置要求，部署指定数量的服务器，并自动安装和配置服务器软件，满足用户的应用部署需求。

# 4.具体代码实例和解释说明
## 安装 awscli
```bash
sudo yum install -y python-pip && sudo pip install --upgrade awscli
```

## AWS CLI 操作 EC2
### 获取 EC2 列表
```bash
aws ec2 describe-instances
```

示例输出:
```json
{
    "Reservations": [
        {
            "Groups": [], 
            "Instances": [
                {
                    "AmiLaunchIndex": 0, 
                    "ImageId": "ami-xxxxx", 
                    "InstanceId": "i-xxxxxx", 
                    "InstanceType": "t2.micro", 
                    "KeyName": "mykeypair", 
                    "LaunchTime": "YYYY-MM-DDTHH:MM:SSZ", 
                    "Monitoring": {
                        "State": "disabled"
                    }, 
                    "Placement": {
                        "AvailabilityZone": "us-west-2b", 
                        "GroupName": "", 
                        "Tenancy": "default"
                    }, 
                    "PrivateDnsName": "ip-10-xxx-xx-xx.ec2.internal", 
                    "PrivateIpAddress": "10.xxx.xx.xx", 
                    "ProductCodes": [], 
                    "PublicDnsName": "", 
                    "State": {
                        "Code": 16, 
                        "Name": "running"
                    }, 
                    "StateTransitionReason": "", 
                    "SubnetId": "subnet-xxxxx", 
                    "VpcId": "vpc-xxxxx", 
                    "Architecture": "x86_64", 
                    "BlockDeviceMappings": [], 
                    "ClientToken": "<PASSWORD>", 
                    "EbsOptimized": false, 
                    "EnaSupport": true, 
                    "Hypervisor": "xen", 
                    "IamInstanceProfile": null, 
                    "NetworkInterfaces": [
                        {
                            "Attachment": {
                                "AttachTime": "YYYY-MM-DDTHH:MM:SSZ", 
                                "AttachmentId": "eni-attach-xxxxx", 
                                "DeleteOnTermination": true, 
                                "DeviceIndex": 0, 
                                "Status": "attached"
                            }, 
                            "Description": "Primary network interface", 
                            "Groups": [...], 
                            "MacAddress": "xx:xx:xx:xx:xx:xx", 
                            "NetworkInterfaceId": "eni-xxxxxx", 
                            "OwnerId": "xxxxxxxxxx", 
                            "PrivateIpAddress": "10.xxx.xx.xx", 
                            "PrivateIpAddresses": [
                                {
                                    "Association": {
                                        "IpOwnerId": "xxxxxxxxxx", 
                                        "PublicDnsName": "ec2-xx-xx-xx-xx.compute-1.amazonaws.com", 
                                        "PublicIp": "192.168.127.12"
                                    }, 
                                    "Primary": true, 
                                    "PrivateDnsName": "ip-10-xxx-xx-xx.ec2.internal", 
                                    "PrivateIpAddress": "10.xxx.xx.xx"
                                }
                            ], 
                            "SourceDestCheck": true, 
                            "Status": "in-use", 
                            "SubnetId": "subnet-xxxxx", 
                            "VpcId": "vpc-xxxxx", 
                            "InterfaceType": "interface"
                        }
                    ], 
                    "RootDeviceName": "/dev/sda1", 
                    "RootDeviceType": "ebs", 
                    "SecurityGroups": [
                        {
                            "GroupName": "default", 
                            "GroupId": "sg-xxxxxxxxxxxxxxxxx"
                        }
                    ], 
                    "SourceDestCheck": true, 
                    "Tags": []
                }
            ]
        }
    ]
}
```

### 根据实例 ID 获取 EC2 详情
```bash
aws ec2 describe-instances --instance-ids i-xxxxxx
```

### 获取 EC2 最新公网 IP
```bash
aws ec2 describe-addresses | jq '.Addresses[] | select(.InstanceId=="i-xxxxxx")' |.PublicIp
```

### 修改 EC2 名称
```bash
aws ec2 create-tags --resources i-xxxxxx --tags Key=Name,Value="new name"
```

### 启动 EC2 实例
```bash
aws ec2 start-instances --instance-ids i-xxxxxx
```

### 停止 EC2 实例
```bash
aws ec2 stop-instances --instance-ids i-xxxxxx
```

### 重启 EC2 实例
```bash
aws ec2 reboot-instances --instance-ids i-xxxxxx
```

## AWS CLI 操作 EBS
### 获取 EBS 列表
```bash
aws ec2 describe-volumes
```

### 根据卷 ID 获取 EBS 详情
```bash
aws ec2 describe-volumes --volume-id vol-xxxxxx
```

### 创建 EBS
```bash
aws ec2 create-volume --availability-zone us-east-1d --size 20 --volume-type gp2
```

### 修改 EBS 名称
```bash
aws ec2 create-tags --resources vol-xxxxxx --tags Key=Name,Value="new name"
```

### 添加卷到实例
```bash
aws ec2 attach-volume --volume-id vol-xxxxxx --instance-id i-xxxxxx --device /dev/sdf
```

### 从实例中卸载卷
```bash
aws ec2 detach-volume --volume-id vol-xxxxxx
```

### 创建快照
```bash
aws ec2 create-snapshot --volume-id vol-xxxxxx --description "This is my snapshot."
```

### 查看快照列表
```bash
aws ec2 describe-snapshots
```

### 创建 AMI
```bash
aws ec2 create-image --name MyNewImage --source-volume vol-xxxxxx --no-reboot
```

### 删除 EBS
```bash
aws ec2 delete-volume --volume-id vol-xxxxxx
```

## AWS CLI 操作 Auto Scaling Group
### 创建 Auto Scaling Group
```bash
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name my-asg \
  --launch-template LaunchTemplateId=lt-xxxxxx,Version="$Latest" \
  --min-size 1 \
  --max-size 4 \
  --desired-capacity 1 \
  --vpc-zone-identifier subnet-xxxxxx,subnet-yyyyyyy,subnet-zzzzzz \
  --target-group-arns target-group/my-targets
```

### 更新 Auto Scaling Group
```bash
aws autoscaling update-auto-scaling-group \
  --auto-scaling-group-name my-asg \
  --launch-template LaunchTemplateId=lt-xxxxxx,Version="$Latest" \
  --min-size 2 \
  --max-size 6 \
  --desired-capacity 2
```

### 获取 Auto Scaling Group 列表
```bash
aws autoscaling describe-auto-scaling-groups
```

### 获取 Auto Scaling Group 详情
```bash
aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names my-asg
```

### 获取 Auto Scaling Group 通知列表
```bash
aws autoscaling describe-notification-configurations \
  --auto-scaling-group-name my-asg
```

### 设置 Auto Scaling Group 通知
```bash
aws autoscaling put-notification-configuration \
  --auto-scaling-group-name my-asg \
  --topic-arn arn:aws:sns:us-east-1:123456789012:my-topic \
  --notification-types NotificationType1,NotificationType2
```

### 获取 Auto Scaling Group 生命周期 Hooks 列表
```bash
aws autoscaling describe-lifecycle-hooks \
  --auto-scaling-group-name my-asg
```

### 设置 Auto Scaling Group 生命周期 Hooks
```bash
aws autoscaling put-lifecycle-hook \
  --auto-scaling-group-name my-asg \
  --lifecycle-hook-name scale-down-hook \
  --lifecycle-transition termination \
  --default-result ABANDON \
  --heartbeat-timeout 300 \
  --notification-metadata '{"message":"Scaling down due to low disk utilization"}' \
  --role-arn arn:aws:iam::123456789012:role/MySampleRole
```

### 清除 Auto Scaling Group 生命周期 Hooks
```bash
aws autoscaling complete-lifecycle-action \
  --auto-scaling-group-name my-asg \
  --lifecycle-hook-name scale-up-hook \
  --lifecycle-action-token <PASSWORD> \
  --lifecycle-action-result CONTINUE
```

### 设置 Auto Scaling Group 标签
```bash
aws autoscaling create-or-update-tags \
  --tags ResourceId=my-asg,ResourceType=auto-scaling-group,Key=Project,Value=MyApp \
  --tags ResourceId=my-asg,ResourceType=auto-scaling-group,Key=Environment,Value=Production
```

### 关闭 Auto Scaling Group
```bash
aws autoscaling suspend-processes \
  --auto-scaling-group-name my-asg \
  --scaling-processes Terminate,HealthCheck,ReplaceUnhealthy
```

### 开启 Auto Scaling Group
```bash
aws autoscaling resume-processes \
  --auto-scaling-group-name my-asg \
  --scaling-processes Terminate,HealthCheck,ReplaceUnhealthy
```

### 禁用 Auto Scaling Group
```bash
aws autoscaling enter-standby \
  --auto-scaling-group-name my-asg \
  --should-decrement-desired-capacity
```

### 启用 Auto Scaling Group
```bash
aws autoscaling exit-standby \
  --auto-scaling-group-name my-asg
```

### 设置 Auto Scaling Policy
```bash
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name my-asg \
  --policy-name my-scaling-policy \
  --policy-type StepScaling \
  --adjustment-type ChangeInCapacity \
  --scaling-adjustment 2 \
  --cooldown 300 \
  --metric-aggregation-type Average \
  --step-adjustments MetricIntervalLowerBound=0,MetricIntervalUpperBound=600,ScalingAdjustment=1 \
                  MetricIntervalLowerBound=600,MetricIntervalUpperBound=1200,ScalingAdjustment=-1
```

### 删除 Auto Scaling Policy
```bash
aws autoscaling delete-policy \
  --auto-scaling-group-name my-asg \
  --policy-name my-scaling-policy
```

### 告知 Auto Scaling Group 实例已准备就绪
```bash
aws autoscaling complete-lifecycle-action \
  --auto-scaling-group-name my-asg \
  --lifecycle-hook-name scale-up-hook \
  --lifecycle-action-token abcdefghijklmnopqrstuvwxyz \
  --lifecycle-action-result CONTINUE
```

### 告知 Auto Scaling Group 实例部署失败
```bash
aws autoscaling complete-lifecycle-action \
  --auto-scaling-group-name my-asg \
  --lifecycle-hook-name deploy-app-hook \
  --lifecycle-action-token abcdefghijklmnopqrstuvwxyz \
  --lifecycle-action-result ABANDON \
  --lifecycle-action-status Unsuccessful \
  --instance-id i-xxxxxx
```

## AWS CLI 操作 IAM
### 创建用户
```bash
aws iam create-user --user-name myusername
```

### 编辑用户属性
```bash
aws iam update-user --user-name myusername \
   --new-path /division_abc/department_xyz/ \
   --new-user-name newusername \
   --password NewPassword1! \
   --permissions-boundary policy-arn \
   --tags Key=environment,Value=production
```

### 为用户设置密码
```bash
aws iam create-login-profile --user-name myusername --password Password<PASSWORD>!
```

### 创建组
```bash
aws iam create-group --group-name mygroup
```

### 为组添加成员
```bash
aws iam add-user-to-group --user-name myusername --group-name mygroup
```

### 创建策略
```bash
aws iam create-policy \
  --policy-name AllowS3Access \
  --policy-document file://s3access.json
```

### 为用户、组或角色授权策略
```bash
aws iam attach-user-policy --user-name myusername --policy-arn arn:aws:iam::123456789012:policy/AllowS3Access
```

### 查看策略
```bash
aws iam list-policies
```

### 检查用户权限
```bash
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::123456789012:user/Alice \
  --action-names s3:* \
  --resource-arns "*"
```

