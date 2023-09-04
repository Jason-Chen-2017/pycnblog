
作者：禅与计算机程序设计艺术                    

# 1.简介
  

VPC(Virtual Private Cloud) 虚拟私有云，一种提供网络隔离、安全可靠的云计算服务。VPC 是用户在自己 AWS 账户下创建一个或者多个子网，然后创建 EC2 实例，配置相应的安全组等，从而实现自建数据中心的功能。

当两个不同 VPC 中的 EC2 需要互相通信时，需要通过 VPC peering 来建立连接。通过 VPC peering 可以将一个 VPC 中的资源连接到另一个 VPC 中的同样资源上，使得两 VPC 中的 EC2 实例可以互相访问和通信。本文将阐述如何在两个不同的 AWS 账户中配置 VPC peering connection。

# 2.VPC 相关概念

- VPC（Virtual Private Cloud）：虚拟私有云，提供了网络隔离、安全可靠的云计算服务。每个 VPC 中都存在着自己的网络地址空间，并且可以根据需求设置子网，分配不同 IP 范围。每台 EC2 实例都在这个 VPC 的某个子网中。

- Subnet （子网）：VPC 中的子网是一个虚拟网络环境，可以用来划分不同 AZ 中的 VPC 中的资源。在子网中可以部署资源如 EC2、RDS 等。每个 VPC 必须至少有一个子网。

- Route table （路由表）：路由表决定了数据包从源地址到目的地址经过哪些路由设备的过程。它包含一系列路由规则，每个规则指定了一个目的 CIDR 块，以及下一跳地址。路由表可以应用于 VPC 内部的子网，也可以应用于外部的网络，比如 Internet 。

- Security Group （安全组）：安全组允许或拒绝进入 VPC 中的实例的网络流量。它包含一系列安全规则，每个规则都有允许还是禁止某种协议、端口范围、方向等信息。安全组只能控制实例间的网络流量，不能控制对单个实例的访问。

# 3.VPC Peering 相关概念

- VPC Peering Connection （VPC 对等连接）：VPC 对等连接是一个逻辑上的连接，用于连接两个 VPC 之间的路由表和网络基础设施。两个 VPC 中的 EC2 实例之间可以通过 VPC peering 建立直接的连接，无需通过公网进行传输，可以有效降低网络延迟和成本。

- Requester VPC （请求方 VPC）：创建 VPC 对等连接的 VPC ，即申请 VPC peering 关系的 VPC 。

- Accepter VPC （接受方 VPC）：被请求方 VPC 请求建立 VPC peering 关系的 VPC 。

- Peer VPC CIDR block （对端 VPC CIDR 块）：对端 VPC 的网段，即 Requester VPC 或 Accepter VPC 没有的网段。

- Peer Role （对端角色）：是 Requester VPC 还是 Accepter VPC 。

- Cross Region Peering （跨区域对等）：跨区域对等是指跨越 AWS 区域的 VPC 之间的对等连接。跨区域对等会消耗额外的带宽费用，因此不建议大规模跨区域对等。

# 4.使用 VPC Peering 建立 VPC 连接的优点

使用 VPC peering 建立 VPC 连接的主要优点如下:

1.高可用性：如果两个 VPC 之间存在 VPC peering 连接，那么它们中的任何一个 VPC 中的实例都可以访问另外一个 VPC 中的实例，同时也能避免跨区域部署带来的复杂性和管理难度。

2.低延迟：通过 VPC peering 可以有效降低网络延迟，减少因数据包传输过程中的网络切换或其他因素导致的网络延迟。

3.降低成本：VPC peering 不会产生额外的网络流量费用，只会产生 VPC 连接相关的费用。

# 5.VPC Peering 连接的限制条件

使用 VPC peering 建立连接前，需要注意以下几个限制条件:

1.VPC 之间必须要有全通路由能力：这是因为 VPC peering 会利用 AWS 提供的 VPC 路由器进行通信。所以两个 VPC 必须要能够互相路由。

2.相同类型的 VPC 才能建立连接：即两个 VPC 都要属于同一个 AWS region，并且支持 VPC peering。

3.VPC 不能有重复的 CIDR 块：CIDR 块不能重叠。

4.ACL 不支持 VPC peering：由于 VPC peering 会直接利用 VPC 路由器进行通信，所以 VPC 之间的 ACL 是不生效的。

5.AWS Limitations：AWS 对 VPC peering 的限制很多。比如每个 VPC 可建立的 VPC peering 数量有限，超过限制后会报错等。


# 6.配置 VPC Peering Connection 

## 准备工作

1.在两个不同的 AWS 账号中分别创建一个 VPC 和子网。

Requester VPC：

```
aws ec2 create-vpc --cidr-block xxxx.xxxx.xxxx/x --instance-tenancy default --query 'Vpc.{VpcId: VpcId}'
aws ec2 create-subnet --vpc-id vpc-xxxxxx --cidr-block yyyy.yyyy.0.0/y --availability-zone ap-northeast-1a 
```

Accepter VPC：

```
aws ec2 create-vpc --cidr-block aaaa.aaaa.aaaa/z --instance-tenancy default --query 'Vpc.{VpcId: VpcId}'
aws ec2 create-subnet --vpc-id vpc-zzzzzz --cidr-block bbbb.bbbb.0.0/w --availability-zone us-east-1c
```

2.为 VPC 设置安全组并开启 ICMP、SSH 等网络访问权限。

```
aws ec2 create-security-group --group-name allow_icmp_ssh --description "allow icmp & ssh" \
    --vpc-id $REQUESTER_VPC_ID
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxx --protocol tcp --port 22 --cidr xx.xx.xx.xx/xx
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxx --protocol tcp --port 80 --cidr yy.yy.yy.yy/yy
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxx --protocol -1 --source-group sg-yyyyyyyy
```

其中 REQUESTER_VPC_ID 为 Requester VPC 的 ID；$ACCEPTER_VPC_ID 为 Accepter VPC 的 ID。

3.创建 IAM 用户并给予 VPCPeeringRole 权限。

```
aws iam create-user --user-name peeraccount
aws iam attach-user-policy --user-name peeraccount --policy-arn arn:aws:iam::aws:policy/AmazonVPCFullAccess
aws iam attach-user-policy --user-name peeraccount --policy-arn arn:aws:iam::aws:policy/AmazonRoute53FullAccess
aws iam put-user-policy --user-name peeraccount --policy-name VPCPeerPolicy --policy-document file://vpccreds.json
```

其中 vpccreds.json 文件内容类似于：

```
{
   "Version": "2012-10-17",
   "Statement":[
      {
         "Effect":"Allow",
         "Action":[
            "ec2:DescribeVpcs",
            "ec2:DescribeSubnets",
            "ec2:DescribeSecurityGroups",
            "ec2:CreateTags"
         ],
         "Resource":"*"
      },
      {
         "Effect":"Allow",
         "Action":[
            "ec2:ModifyInstanceAttribute",
            "ec2:CreateRoute",
            "ec2:DeleteRoute"
         ],
         "Resource":["arn:aws:ec2:*:*:route-table/*"]
      }
   ]
}
```

4.生成临时访问凭证，该凭证仅限于创建 VPC 对等连接的用户使用。

```
aws sts get-session-token --duration-seconds 900 --serial-number arn:aws:iam::your-peered-account-id:mfa/${USER_MFA_DEVICE_NAME}
```

生成的临时密钥存储于 ~/.aws/credentials 文件中，其内容类似于：

```
[peeraccount]
aws_access_key_id = AKIAXXXXXXXXXXXXXXXXX
aws_secret_access_key = sjGlt+xxxxxxxxxxxxxxxxxx=
aws_session_token = FwoGZXIvYXdzEFUtbWFzdGVyfHwxNTUxNDUyNzI3NjA3IMtAha5rPiUakF4OZSyHTic6qT+rjVbttrPNdyDnEVBrlYpUkVP7uCJu9DgdQgXilIJgL+/fgfQBrdxyRLqgUdfuyWHRJuvExiVChAF6cJrjFXGHLFnAqhcJHsxCJJudavBNWnk=
region = ap-northeast-1
```

## 配置 VPC Peering Connection

1.打开 Requester VPC 的终端窗口，输入以下命令创建和查看 VPC peering connection:

```
aws ec2 create-vpc-peering-connection --vpc-id $REQUESTER_VPC_ID --peer-vpc-id $PEERING_VPC_ID --peer-owner-id $PEERING_ACCOUNT_ID \
    --peer-region $REGION --query 'VpcPeeringConnection' > requester.txt
cat requester.txt | grep VpcPeeringConnectionId
```

其中 REQUESTER_VPC_ID 为 Requester VPC 的 ID；PEERING_VPC_ID 为 Accepter VPC 的 ID；PEERING_ACCOUNT_ID 为对端账户的 ID；REGION 为 Accepter VPC 在的区域。此时输出结果应包含 VPC peering connection 的 ID，如："VpcPeeringConnectionId": "pcx-xxxxxx"。

2.打开 Accepter VPC 的终端窗口，输入以下命令创建和查看 VPC peering connection:

```
aws ec2 accept-vpc-peering-connection --vpc-peering-connection-id $CONNECTION_ID --query 'VpcPeeringConnection' > accepter.txt
cat accepter.txt | grep VpcPeeringConnectionId
```

其中 CONNECTION_ID 为 Requester VPC 创建的 VPC peering connection 的 ID。此时输出结果应包含 VPC peering connection 的状态，如："Status": {"Code": "active"}。

至此，配置 VPC peering connection 完成！

# 7.总结

本文从 VPC 相关概念、VPC Peering 相关概念、VPC Peering 使用的优缺点及其限制等方面，详细地阐述了 AWS VPC Peering 连接的配置方法，包括生成临时访问凭证、配置 VPC Peering Connection 两步。文章简单易懂，值得初学者阅读。

希望本文对读者的学习与理解有所帮助！