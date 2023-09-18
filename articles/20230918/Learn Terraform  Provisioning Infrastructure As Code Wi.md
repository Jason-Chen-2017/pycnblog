
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Terraform是一个开源的基础设施即代码工具，可用于创建、更新和删除基础设施。Terraform可以管理现有的基础设施，也可以用于自动化地预配新的基础设施。
本文将主要讲述如何用Terraform配置EC2实例并运行基于CentOS的Web应用服务。本文不会教授Terraform的基础知识，只会带领读者安装并了解该工具，掌握其功能和命令的用法。如果你对Terraform的语法不熟悉，或者想学习基础知识，请阅读我的另一篇文章《Learn Terraform Syntax and Concepts - Understanding How to Use Terraform for Infrastructure Automation》。
# 2.环境准备
首先，您需要下载并安装Terraform。建议使用最新版本的Terraform进行部署。下载地址：https://www.terraform.io/downloads.html。另外，如果您已经安装过了，请确保已经升级至最新版本。
安装好 Terraform 后，您需要设置 AWS 的凭证信息。打开终端，输入以下命令：
```bash
$ export AWS_ACCESS_KEY_ID="your access key id"
$ export AWS_SECRET_ACCESS_KEY="your secret access key"
$ export TF_VAR_key_name="your SSH key name in AWS" # 如果没有SSH密钥，可以忽略此步
```
# 3.实例部署
## 3.1 配置文件编写
在准备好环境之后，我们就可以编写配置文件了。在本地目录下创建一个名为`main.tf`的文件，写入以下内容：
```terraform
provider "aws" {
  region = "us-east-1"
}

resource "aws_security_group" "web" {
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "my_security_group"
    Environment = "production"
  }
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0" # 使用的AMI ID
  instance_type = "t2.micro"
  vpc_security_group_ids = [aws_security_group.web.id]
  
  user_data = <<-EOF
              #!/bin/bash
              echo "Hello, World!" > index.html
              nohup python -m SimpleHTTPServer 80 &
              EOF
  
  root_block_device {
    volume_size = 10
  }

  tags = {
    Name        = "my_ec2_instance"
    Environment = "production"
  }
}
```
这个文件定义了两个资源：一个安全组，一个EC2实例。其中，安全组开放80端口，允许访问任意IP地址；EC2实例启动时，会把当前目录下的`index.html`文件的内容输出到浏览器显示。我们还指定了AMI的ID和实例类型。需要注意的是，由于云服务器磁盘空间小，所以调整大小为10GB。如果需要扩容，可以修改配置文件。
## 3.2 执行部署
然后，进入配置好的工作目录，执行以下命令：
```bash
$ terraform init # 初始化配置
$ terraform plan # 查看计划
$ terraform apply # 执行部署
```
执行完成后，查看AWS控制台，就会看到新建的EC2实例，以及相应的安全组。可以登录到EC2实例上，确认是否正常运行。
# 4.实例清除
最后，当不需要使用EC2实例的时候，可以通过以下命令停止并销毁它：
```bash
$ terraform destroy
```