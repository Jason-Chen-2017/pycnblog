
作者：禅与计算机程序设计艺术                    

# 1.简介
         

## 概要
加密是现代信息安全的一个重要部分。理解加密背后的基本原理可以帮助你更好地保护你的数据，保证数据在传输过程中没有被篡改、伪造或损坏。加密算法是使得这一切成为可能的关键。这本书尝试从广义上理解加密算法的工作原理，并对其进行分类，进而给出实际应用中的实现方法。每章都着重于一个特定的加密算法，如RSA，ElGamal，AES等，提供一个完整的描述，并包括相关的代码示例。最后，作者将这些算法应用到实际的问题中，包括数字签名、认证、密码管理、密码学协议以及加密货币的研究。

## 作者简介
Shai Vanderpoeldt（斯哈维德·范尔登特）是一名具有全球性影响力的加密专家和学者。他担任美国国家标准技术监督局（NIST）的信息科学技术委员会主席（ISC），并在过去几十年中致力于建立最先进的密码学算法。他还曾担任计算机科学教授和联邦学者。他现在是Facebook的高级工程师。

2017年，Shai与两位同事一起创立了Bitwise Solutions公司。公司为政府和金融机构提供加密服务，包括信息安全、电子合同、电子清算等。2019年，公司完成了第一轮融资，估值超过1亿美元。

# 2.基本概念术语说明
## 密码学(Cryptography)
密码学是指通过加密、解密、或者其他方式隐藏信息的一门学术分支。通常情况下，需要使用某种形式的秘钥对信息进行加解密，这样才能确保信息在传输过程中不被他人窃听或破译。密码学利用对称加密、公私钥加密、摘要函数、hash函数以及随机数生成器等多种技术来实现信息的保护。常用的密码学技术如下图所示：

## 对称加密(Symmetric Encryption)
对称加密是一种常用且简单的加密算法，它把一条消息加密成一条与原消息相同的密文，只有拥有正确的密钥才能进行解密。对称加密有两种主要的模式，ECB和CBC。它们分别对应于电子密码本模式和密码分组链接模式。ECB模式简单直接，适用于小块数据，但是容易受到副本攻击；CBC模式相对复杂些，适用于大量数据流，并且可以抵御副本攻击。常用的对称加密算法如下表所示：
| Algorithm | Key Length | Block Size | IV Required | Modes | Usecase |
|---|---|---|---|---|---|
| AES | 128/192/256 bits | 128 bits | Yes | EBC, CBC | Banking / Financial data |
| DES | 56 bits | 64 bits | No | EBC | Federal government |
| TripleDES | 168 bits | 64 bits | Yes | EBC | Internet banking / online transactions |
| RC4 | Variable length | 512 bits | No | Stream Cipher | TLS encryption / VPN security |

## 公钥加密(Asymmetric Encryption)
公钥加密也称非对称加密，是一种加密算法，其中同时存在两个密钥：公钥和私钥。公钥用于加密消息，私钥用于解密消息。与对称加密不同的是，公钥加密要求消息的接收方必须同时拥有公钥才可解密消息。公钥加密可以用来验证数据的真实性，确保数据来源不可被伪造，并提供数据机密性和完整性。常用的公钥加密算法如下表所示：
| Algorithm | Key Length | Signature Scheme | Usecase |
|---|---|---|---|
| RSA | > 1024 bits | PKCS #1 v1.5 or PSS padding | Online transactions / SSL certificates |
| ElGamal | Multiple sizes | NIST variant | Secure email / digital signatures |
| DSA | Multiple sizes | NIST variant | Digital signatures |
| ECDH | Variable sizes | NIST curves and Koblitz formulas | Authenticated key exchange protocols |

## 摘要函数(Hash Function)
摘要函数是一种单向算法，它接受输入数据，并输出固定长度的消息摘要。其目的是为了产生一种独特的输出结果，即使对于不同的输入数据得到的输出也是不同的。常用的摘要函数如下表所示：
| Algorithm | Output Size | Collision Resistance | Performance | Attack Model | Usage |
|---|---|---|---|---|---|
| SHA-1 | 160 bits | Depends on input size | Very fast | Little to no effort | Mostly deprecated |
| SHA-2 family | 224/256/384/512 bits | Probabilistically impossible | Slower than SHA-1 but more secure | Lots of effort | Common in cryptographic applications |
| MD5 | 128 bits | High probability of collisions | Fastest among all hash functions | Medium difficulty | Used for legacy systems |
| BLAKE2 | Variable sizes | Resistant against preimage attacks | Fast as compared to other hash functions | Easy attack model | More robust version of SHA-3 | 

## Hash-based message authentication codes(HMAC)
HMAC是一种基于哈希的消息认证码，它利用哈希函数对消息和密钥进行运算，然后把两个运算结果组合起来作为最终的消息认证码。HMAC有助于确保信息的完整性和真实性，因为如果有人修改了消息或密钥，那么它们的组合就会发生变化。常用的HMAC算法如下表所lide:
| Algorithm | Key Length | Usecase |
|---|---|---|
| HMAC-SHA-2 family | Same as used by the underlying hash function | Most common use case is MAC construction with symmetric keys |