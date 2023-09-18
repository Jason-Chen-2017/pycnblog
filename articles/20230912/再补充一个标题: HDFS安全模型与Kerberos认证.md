
作者：禅与计算机程序设计艺术                    

# 1.简介
  

HDFS是一个分布式文件系统，用于存储海量的数据，通过目录树结构进行管理；在企业数据中，数据主要存放在HDFS上，所以HDFS的安全性至关重要。HDFS采用了Kerberos认证机制实现用户访问权限控制。本文将结合HDFS的安全模型和Kerberos认证，阐述HDFS安全模型及Kerberos认证工作原理，并通过两个实例说明Kerberos认证对HDFS的影响。
# 2.核心概念、术语说明
## 2.1 Kerberos认证简介
Kerberos 是一种开放的认证授权协议，它定义了一套密钥管理系统，用于允许客户机或者服务器相互确认彼此的身份。Kerberos认证模型中包括如下几个组成部分：
- Ticket Granting Ticket (TGT)：
这是主体（如客户端）向服务器请求票据的一个凭证，用于身份验证和授权。TGT由Kerberos KDC颁发，有效期一般为10~20分钟。KDC除了发放票据之外，还负责票据的续约和撤销工作。
- Ticket-granting service (TGS)：
用于委托他人（如服务端）申请受限于特定资源的权限。TGS由KDC签署证书，用于指定客户端可访问哪些资源以及可使用的权限。TGS需要携带TGT才能获得授权。
- Key Distribution Center(KDC)：
用来管理整个认证过程。KDC分离了认证功能和票据的生成和维护，从而使得Kerberos成为一个独立的服务。KDC只需要知道密钥和策略即可管理用户的认证。
- Authentication Server：
用于处理客户端的请求，如账号密码登录等。Authentication Server根据配置好的策略对用户进行认证，生成票据返回给客户端。
- Principal and User Identification：
Kerberos认证基于身份认证，每个用户都有唯一的Principal名称，也称为UserID，这种名称可以由域名和用户名组合而成，如bob@HADOOP.COM。Kerberos还支持别名和映射，也就是说一个Principal可以对应多个用户ID。
- Encryption Key：
用于加密数据的密钥。用户的加密密钥由用户的密码加密得到。
- Session Key：
用以对会话进行加密的密钥，由TGT或TGS的加密密钥和客户端的密钥共同产生。Session key用于数据传输的加解密。
## 2.2 HDFS 安全模型
HDFS采用了Kerberos认证机制来实现用户访问权限控制。默认情况下，客户端不需要任何身份信息就可以访问HDFS。但是为了提供更高级别的安全保障，HDFS允许管理员为特定的用户设置访问权限。当某个用户访问HDFS时，Kerberos认证会检查其是否具有访问权限，如果拥有访问权限，则会给予其相应的权限；否则，则拒绝访问。
HDFS的安全模式分为两种类型：
### 2.2.1 简单模式（Simple Mode）
在Simple模式下，不启用安全认证。所有用户都可以匿名地访问HDFS，并且没有访问控制列表（ACL）。
### 2.2.2 安全模式（Secure Mode）
在Secure模式下，启动了安全认证，要求客户端必须提供认证信息才能访问HDFS。当客户端连接到NameNode时，首先必须发送一个Kerberos Token，然后才能够访问HDFS。因为启用了Kerberos认证，因此必须在集群上安装Kerberos相关组件。在Secure模式下，管理员可以为不同用户授予不同的访问权限，即使是匿名用户也可以控制访问权限。HDFS提供了以下方式来设定用户的访问权限：
#### ugi模式（User Group Information 模式）
这种模式依赖于已存在的Hadoop用户组信息，可以在配置文件中配置，也可以通过web界面设置。通过这种方式，可以将HDFS中的文件权限分配到对应的用户组，通过修改用户组中的用户，可以实现对文件的访问控制。
#### ACL模式（Access Control List 模式）
这种模式也是通过配置文件或web界面进行配置，管理员可以指定某些用户或用户组可以访问哪些目录或文件。这种模式最大的优点是细粒度的权限控制，而且可以通过命令行来管理。
## 2.3 Kerberized HDFS 概览

图1：Kerberized HDFS概览

1. 用户提交作业到ResourceManager，同时向RM提交一个Token，该Token包含了Kerberos TGT，并绑定到当前用户。
2. RM把该Token传递给NM，并等待NM向NN请求票据。
3. NN向KDC请求票据，并返回一个TGT给NM。
4. NM获取到TGT后，向Snnamenode发送请求获取block locations，Snnamenode收到请求后，向Zookeeper请求块位置信息，Zookeeper给出响应，NM把响应返回给RM。
5. RM根据NM的回复确定该作业所需的块位置信息，然后创建shuffle job，并提交给ApplicationMaster。
6. ApplicationMaster向MRAppMaster提交一个Token，该Token包含了Kerberos TGS，并绑定到ApplicationMaster所在的主机。
7. MRAppMaster获取到TGS后，向ShuffleHandler发送请求获取mapper task，ShuffleHandler收到请求后，向DataNode请求mapper的输入数据块，DataNode给出响应，ShuffleHandler把响应返回给MRAppMaster。
8. MRAppMaster拿到 mapper 的输入数据块后，启动自己的 Container，启动容器后，运行 MapTask，MapTask读取输入数据块，对其进行处理，生成中间结果，并将中间结果分片输出到ReduceTask所在的机器。
9. ReduceTask所在的机器的Container启动后，运行 ReduceTask ，ReduceTask 根据 mapper 分配到的分片数据，对其进行合并排序，并输出最终结果。
10. 当所有的任务完成后，ApplicationMaster通知RM作业执行完成。

# 3.具体操作步骤以及数学公式讲解
## 3.1 设置Kerberos环境变量
首先，要设置一些Kerberos环境变量：
```bash
export JAVA_HOME=/usr/java/jdk1.8.0_221
export PATH=$PATH:$JAVA_HOME/bin:/hadoop/bin
export KRB5_CONFIG=$JAVA_HOME/jre/lib/security/krb5.conf
export KERBEROS_AUTH_OPTIONS="-Djava.security.auth.login.config=/etc/hadoop/conf/krb5JAASLogin.conf -Djavax.security.auth.useSubjectCredsOnly=false"
```
其中，$JAVA_HOME指的是JDK的安装路径，$PATH是Java可执行程序的搜索路径，$KRB5_CONFIG是Kerberos的配置文件路径，$KERBEROS_AUTH_OPTIONS是用于开启Kerberos认证的选项，-Djava.security.auth.login.config表示Java登录配置文件的路径，-Djavax.security.auth.useSubjectCredsOnly=false表示使用subject身份验证，不使用keytab认证。

## 3.2 配置Kerberos相关文件
接着，配置Kerberos相关文件：
### 3.2.1 krb5.conf文件
编辑/etc/krb5.conf文件，增加如下内容：
```bash
[libdefaults]
  default_realm = EXAMPLE.COM
  dns_lookup_realm = false
  dns_lookup_kdc = false
  ticket_lifetime = 24h
  forwardable = true
  rdns = false

[realms]
  EXAMPLE.COM = {
    kdc = kdc1.example.com
    admin_server = kdc1.example.com
  }

[domain_realm]
 .example.com = EXAMPLE.COM
  example.com = EXAMPLE.COM
```
其中，default_realm表示Kerberos realm，dns_lookup_realm表示是否要通过DNS查找Realm信息，dns_lookup_kdc表示是否要通过DNS查找KDC地址，ticket_lifetime表示Ticket有效时间，forwardable表示是否可以转发Ticket，rdns表示是否采用反向解析（Reverse DNS lookup）。

admin_server是Kerberos域控制器（KDC）的地址，kdc1.example.com是示例的KDC服务器的地址。

### 3.2.2 krb5JAASLogin.conf文件
编辑/etc/hadoop/conf/krb5JAASLogin.conf文件，添加如下内容：
```bash
Client {
  com.sun.security.auth.module.Krb5LoginModule required
  useKeyTab=true
  storeKey=true
  useTicketCache=false
  principal="client@EXAMPLE.COM"
  keyTab="/home/myuser/.ssh/user.keytab";
};
```
其中，principal表示Kerberos用户名称，keyTab表示密钥文件路径。

### 3.2.3 user.keytab文件
生成user.keytab文件，命令如下：
```bash
kinit client
ktutil
addent -password -p client@EXAMPLE.COM -k 1 -e aes256-cts-hmac-sha1-96
wkt /home/myuser/.ssh/user.keytab
q
```
其中，-p表示Kerberos用户名称，-k表示第几个主密钥，-e表示加密算法。

## 3.3 配置HDFS
在/etc/hadoop/conf目录下，编辑core-site.xml文件，增加如下内容：
```bash
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://namenode:8020</value>
  </property>
  <property>
    <name>hadoop.security.authentication</name>
    <value>kerberos</value>
  </property>
  <property>
    <name>dfs.nameservices</name>
    <value>kerberos-ha</value>
  </property>
  <property>
    <name>dfs.ha.namenodes.kerberos-ha</name>
    <value>nn1,nn2</value>
  </property>
  <property>
    <name>dfs.namenode.shared.edits.dir</name>
    <value>qjournal://journalnode1:8485;journalnode2:8485;journalnode3:8485/kerberos-ha</value>
  </property>
  <property>
    <name>dfs.client.failover.proxy.provider.kerberos-ha</name>
    <value>org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider</value>
  </property>
  <!-- Directory used by DataNodes to store block replicas -->
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>/var/lib/hadoop/data</value>
  </property>
</configuration>
```
其中，fs.defaultFS表示HDFS的URI，hadoop.security.authentication表示安全认证的类型为Kerberos，dfs.nameservices表示HDFS集群的名称，nn1和nn2分别表示名字节点的主机名或IP地址。dfs.ha.namenodes.kerberos-ha是HA模式下名字节点的名称，dfs.namenode.shared.edits.dir是共享编辑日志的目录，dfs.client.failover.proxy.provider.kerberos-ha表示配置了故障切换代理的类。

## 3.4 配置YARN
在/etc/hadoop/conf目录下，编辑yarn-site.xml文件，增加如下内容：
```bash
<configuration>
  <property>
    <name>yarn.resourcemanager.ha.enabled</name>
    <value>true</value>
  </property>
  <property>
    <name>yarn.resourcemanager.cluster-id</name>
    <value>clutster_1</value>
  </property>
  <property>
    <name>yarn.resourcemanager.ha.rm-ids</name>
    <value>rm1,rm2</value>
  </property>
  <property>
    <name>yarn.resourcemanager.hostname.rm1</name>
    <value>resourcemanager1</value>
  </property>
  <property>
    <name>yarn.resourcemanager.hostname.rm2</name>
    <value>resourcemanager2</value>
  </property>

  <property>
    <name>yarn.log.server.url</name>
    <value>http://logserver:19888/jobhistory/logs/</value>
  </property>
  
  <!-- The address of the resource manager scheduler interface on the scheduler node. Used for communication between the application master and the scheduler during container allocation and release. -->
  <property>
    <name>yarn.resourcemanager.scheduler.address</name>
    <value>${yarn.resourcemanager.hostname.rm1}:8030</value>
  </property>
  
</configuration>
```
其中，yarn.resourcemanager.ha.enabled表示开启YARN HA模式，yarn.resourcemanager.cluster-id表示YARN集群的名称，rm1和rm2分别表示ResourceManager的主机名或IP地址。

## 3.5 启动HDFS和YARN集群
启动HDFS集群：
```bash
sbin/start-dfs.sh
```
启动YARN集群：
```bash
sbin/start-yarn.sh
```

## 3.6 测试Kerberized HDFS
下面，测试Kerberized HDFS的功能。
### 3.6.1 创建目录
创建目录/user：
```bash
hdfs dfs -mkdir /user
```
### 3.6.2 拷贝文件
拷贝文件到/user目录下：
```bash
hdfs dfs -put /etc/passwd /user/root
```
### 3.6.3 读文件
读取文件内容：
```bash
hdfs dfs -cat /user/root/passwd
```
### 3.6.4 删除文件
删除文件：
```bash
hdfs dfs -rm /user/root/passwd
```