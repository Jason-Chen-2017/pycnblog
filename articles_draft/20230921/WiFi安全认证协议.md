
作者：禅与计算机程序设计艺术                    

# 1.简介
  

WiFi安全认证协议(WPSP)是指在无线网络中建立安全信道、验证设备身份、保护数据完整性的一种安全机制。它可以提供对用户访问控制、服务质量保证、设备可靠性和隐私保护等方面的保证。
目前主流的WPSP协议包括WPA/WPA2和Wi-Fi Protected Access II (WAPI)。WPA/WPA2是一个由IEEE802.11i定义的加密算法和认证标准，用于保障无线网络通信安全。WAPI是在IEEE802.11i规范上扩展出来的一个安全协议，提供秘钥管理功能并可支持802.1X协议下的动态身份验证。
本文首先将介绍一下WiFi安全认证协议中的一些概念和术语，然后再具体阐述WPSP协议的实现过程。最后，我们会展示如何使用开源工具实现WPSP。

# 2. 基本概念和术语
## 2.1 认证
首先，需要对认证进行简单地介绍。在认证过程中，网络实体（如终端设备）要向授权实体（如网络管理者）提供自己的身份信息，确认其能够真实、合法、有效地访问网络资源。认证的目的就是为了让终端设备更加容易地确定其身份、判断是否有权访问某项资源。通常来说，由于各种各样的原因，网络实体不一定总是能够获得合法的身份信息。因此，认证过程还包括了多种手段来确保网络实体的合法身份信息，如采用双因素验证、密码重试机制等。

## 2.2 消息认证代码（MAC）
消息认证代码（Message Authentication Code，MAC）是一种基于密钥的一类校验和算法，用于检验消息完整性。它由发送者产生，并随着消息一起发送给接收者。接收者计算相同的密钥，并利用发送者发出的消息和自己计算的MAC值进行比较。如果两者匹配一致，则表明该消息没有被篡改或损坏。

MAC算法的一个重要特点是其计算复杂度很低。此外，因为MAC值的生成依赖于密钥，所以消息收发双方都必须事先共享同一个密钥，否则无法正确验证消息的完整性。

## 2.3 提供商认证（EAP）
提供商认证（Extensible Authentication Protocol，EAP）是一种通用的认证框架。它定义了一组认证方法，终端设备可以使用这些方法向网络管理器发起身份验证请求，从而完成网络实体认证过程。

## 2.4 基于密钥的协商过程
当两个参与方要建立连接时，由于通信双方之前可能不存在直接联系，因此需要交换密钥才能完成通信。这种密钥协商的过程称为密钥交换。目前最常见的密钥交换协议包括Diffie-Hellman密钥交换（DHE）、RSA密钥交换（RSA）、椭圆曲线密钥交换（ECDHE）等。DHE、RSA和ECDHE都是公钥加密系统，其中DHE、RSA和ECDHE都包含密钥交换算法和密钥生成算法。

## 2.5 数据加密标准（DES）
数据加密标准（Data Encryption Standard，DES），又称IBM高级加密标准（Advanced Encryption Standard，AES）、美国政府设计的加密标准，是一种分组对称加密算法。它具有56位的密钥长度，安全级别相对于3DES来说较低。

## 2.6 管理加密密钥的公钥基础设施（PKI）
管理加密密钥的公钥基础设施（Public Key Infrastructure，PKI）是一种独立的安全系统，用来处理密钥分配、管理、使用、记录和监控等关键环节。PKI主要包括证书中心、注册机构、证书颁发机构、目录服务器、密钥存储库等。

## 2.7 对称密钥加密算法
对称密钥加密算法是指通过对称密钥进行加密和解密的算法。对称加密算法包括DES、3DES、AES、Blowfish、IDEA、RC5、RC6等。对称加密算法的优点是运算速度快、加密效率高，缺点是安全性不够高。

## 2.8 公开密钥加密算法
公开密钥加密算法（Public-key cryptography algorithm）是指使用非对称密钥加密的方法，公钥和私钥成对出现。加密过程使用公钥，解密过程使用私钥。两种密钥之间存在一个数学关系，但这个关系并不是简单的计算，而是需要用复杂的算法进行计算。公开密钥加密算法包括RSA、ECC、DSA、ECDSA等。

# 3. Wi-Fi Protected Access II (WAPI)
WAPI是Wi-Fi Alliance提出的一种增强型的WLAN加密协议，其目标是在不破坏现有的IEEE 802.11标准的条件下，进一步增强无线网络的安全性。WAPI与WPA2不同之处在于它采用的是WPA2中使用的相同的加密算法，但是使用了新的身份验证方法。

## 3.1 WAPI工作流程
WAPI工作流程如下图所示。

1. 终端设备发送握手请求报文，即指示客户端应使用WAPI协议。
2. 接入点或网关发现终端设备支持WAPI协议，并且给予相应的支持，即加密算法应符合IEEE 802.11i规范中规定的要求。
3. 终端设备生成共享密钥K_enc和共享临时密钥K_mic。
4. 终端设备向接入点或网关发送握手响应报文，包括共享密钥K_enc和K_mic，以及相关的信息。
5. 接入点或网关与无线控制器进行会话协商，选择加密算法和身份验证算法。
6. 会话协商后，终端设备和无线控制器开始密钥派生。
7. 终端设备生成临时临时的机密密钥SK_tx。
8. 终端设备根据密钥派生函数KDF(KEY，R1 || R2)，将K_enc和SK_rx组合起来生成临时密钥K_tx。
9. 终端设备发送加密消息。
10. 无线控制器根据协商好的加密算法和身份验证算法，计算消息的MIC值，并与终端设备发送的MIC值进行比对。
11. 如果比对结果成功，无线控制器解密消息，并执行业务逻辑。

## 3.2 WAPI的身份验证方法
WAPI的身份验证方法包括Preshared Key，Message Integrity Code (MIC)验证，and Robust Header Compression。

### 3.2.1 Preshared Key
在采用Preshared Key方式进行身份验证前，需由认证实体（例如，管理者或者网关设备）预先分配一组共享密钥。终端设备在发送握手请求报文时，携带预共享密钥，网关设备或接入点使用该密钥对身份验证报文进行签名，终端设备验证签名是否正确。

### 3.2.2 MIC验证
采用MIC验证的方法类似于WPA2的TKIP方法，在密钥派生阶段，终端设备和接入点/网关设备配合产生一个序列号R1，并将R1发送给认证实体。认证实体使用相同的密钥对所有消息进行签名，并附加在消息中。终端设备接收到消息时，先检查消息的MIC值是否正确，然后解密消息。

### 3.2.3 Robust Header Compression
Robust Header Compression与Preshared Key、MIC方法相似，也是利用一种叫做头部压缩的方法减少握手包的大小。如果采用头部压缩的方法，则认证报文不会携带任何消息的内容，只会携带几个字节，比如随机数或序列号。终端设备发送消息时，会用压缩后的头部进行加密。在接收消息时，网关设备和接入点设备会将头部解压出来，然后使用相同的方法对消息的MIC值进行验证。如果验证成功，则会继续对消息进行解密。

# 4. WPA2/WPA3协议
## 4.1 WPA2/WPA3介绍
WPA2和WPA3协议都是由IEEE 802.11i定义的一种无线局域网加密协议。其目的是提供更安全的无线网络。WPA2是Wireless LAN Standard的第十二个修订版本，于2003年发布。WPA3是Wireless LAN Security的第三个更新版本，于2012年发布。两者均兼容IEEE 802.11标准。

## 4.2 WPA2协议
WPA2协议由IEEE 802.11i标准规定，该标准于2003年发布。其主要变化如下：

- 更安全的加密算法：支持CCMP、TKIP和WEP加密算法，并对它们进行了调整和优化。CCMP是Counter Mode with Cipher Block Chaining Message Authentication Code的简称，使用CTR模式进行加密，并结合CBC-MAC进行消息认证码（MAC）计算；TKIP是Temporal Key Integrity Protocol的简称，使用TKIP协议进行数据加密和认证；WEP加密也得到了增加，它的混杂模式可以抵御一些攻击，而且速度也非常快。
- 可配置的身份验证方法：WPA2协议可以同时使用PSK（Pre Shared Key）和EAP（Extensible Authentication Protocol）两种身份验证方法。PSK方式，即通过一串预共享密钥进行身份验证；EAP方式，则是通过支持的EAP方法进行身份验证。
- 可选的投票机制：WPA2协议提供了可选的投票机制，使得访问无线网络的终端设备可以集体决定采用哪种加密算法和身份验证方法，避免单独一个终端的配置造成网络性能下降。

## 4.3 WPA3协议
WPA3协议由IEEE 802.11ax标准规定，该标准于2012年发布。其主要变化如下：

- 使用新的加密算法和更安全的认证方式：WPA3协议引入了新的加密算法，名为GCMP，并支持SHA3-256哈希算法。另外，WPA3协议对TKIP协议进行了修改，加入了ANTICAST模式，提升了抗DoS攻击的能力。
- 支持更多的认证方式：WPA3协议支持TLS握手作为身份验证方式，并且支持多达16个EAP方法，这既满足了当前应用需求，又保障了网络的安全性。
- 支持更快的认证：WPA3协议采用了快速握手算法，可以缩短身份验证时间，并降低握手延迟。

# 5. 技术实现
## 5.1 网卡驱动
目前市面上主流的无线网卡驱动有Broadcom的BCM4329驱动、Intel的iwlwifi驱动、Realtek的rtl8xxxu驱动以及Mediatek的MT76x0u驱动等。其中iwlwifi驱动是Linux平台下最为知名的无线网卡驱动。iwlwifi驱动提供了完整的IEEE 802.11标准功能，它支持WPA2/WPA3、WPA、WMM、TDLS、RFKILL、Prism等众多WLAN功能。我们可以通过配置选项开启WPA2/WPA3、WPA、WMM等功能。

## 5.2 配置文件设置
配置文件的设置涉及到ssid、wpa_passphrase、wpspolicy等字段。以下示例为配置文件的参考模板：
```
country=US
interface=wlan0
driver=iwlwifi
ssid="mywifi"
hw_mode=g
channel=6
ieee80211n=1
wmm_enabled=1
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
wpa_passphrase="<PASSWORD>"
```
其中，ssid和wpa_passphrase为必填参数，其他字段根据具体需求进行选择。

## 5.3 无线连接命令
下面给出通过命令行进行无线连接的命令：

连接命令：`sudo nmcli device wifi connect [essid] password [password]`

断开连接命令：`sudo nmcli device disconnect wlan0`