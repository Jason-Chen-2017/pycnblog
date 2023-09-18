
作者：禅与计算机程序设计艺术                    

# 1.简介
  

目前，无线局域网（WLAN）已成为日常生活中不可或缺的一部分。Wireless Local Area Networks(WLANs) are becoming increasingly commonplace in everyday life, enabling a wide range of applications such as entertainment systems, remote control vehicles, medical monitoring devices, home automation systems, etc., to communicate with each other over the air (Bird, 2019). WLAN technology has become an essential component in providing wireless communication for Internet-connected devices like laptops, smartphones, tablets, smart TVs, game consoles, set-top boxes, etc. In addition, WLANs have been adopted by various industries and organizations for various purposes, including home appliances, medical imaging, manufacturing, transportation, healthcare, telecommunications, etc. The development of next-generation wireless technologies that offer better performance, reliability, security, scalability, and lower power consumption is driving WLANs into the future. However, the existing wireless encryption standards do not provide sufficient levels of protection against adversarial attacks or tampering with data during transmission, which can jeopardize the confidentiality, integrity, and availability of wireless networks. Therefore, there is a need for new wireless encryption protocols that address these issues.
One such protocol called Structured Authentication Enhanced Cryptography (SAE) offers stronger security than the widely used Wi-Fi Protected Access (WPA/WPA2) standard while still being compatible with the current deployment of legacy APs. This article will discuss about how to implement the Secure Key Exchange Protocol (SKES), Centralized Key Management System (CKMS), and the End-to-End Encryption using Elliptic Curve Cryptography (ECC) algorithms in order to establish secure wireless communications using SAE and ECC algorithms. The reader should be able to understand and follow along with the technical details, without any prior knowledge on the subject matter. 

# 2.相关术语
## 2.1 基本概念
### 2.1.1 概念定义
**Authentication**: Authentication refers to verifying the identity of the sender and receiver of a message through digital signatures or cryptographic keys. It ensures that messages are sent from the legitimate source, preventing unauthorized access to sensitive information.  
**Encryption**: Encryption refers to converting plain text messages into coded messages so that only authorized parties can read them. It provides privacy, ensuring that no one can listen in on conversations or intercept important messages.  

In a wireless LAN network, authentication helps to ensure that messages coming from different sources are genuine and authentic before they are allowed to transmit data. Encryption makes it difficult for attackers to intercept or manipulate messages transmitted between the nodes because all messages are encrypted and decrypted under the same key.

The following figure depicts the high-level architecture of a wireless LAN:


In a typical WLAN system, clients and servers communicate directly over the airwaves, bypassing the traditional infrastructure that connects them via routers and switches. When multiple users share a single radio link, collisions occur when two or more transmissions collide and interfere with each other. To avoid this problem, WLANs use techniques such as carrier sense multiple access (CSMA) and collision detection and resolution (CDCR) to detect and resolve such collisions. These mechanisms require specialized hardware components such as radios and antennas.

Once a wireless connection is established, the device must authenticate itself to ensure its identity and determine whether it can send and receive data traffic. This process involves exchanging keys with another node in the network, often known as mutual authentication or peer-to-peer (P2P) authentication. The challenge here is to ensure that each node in the network knows only the secret key required to authenticate itself, protecting the network from potential hackers who might try to eavesdrop on and modify communication packets. Once authenticated, nodes can begin communicating securely without having to worry about spoofing or replay attacks.

To further enhance security, wireless networks typically employ encryption methods to encrypt data streams before they are transmitted across the air. Data encryption enables secrecy and privacy within the network, while also allowing for reliable and trustworthy transfer of data. There are several types of encryption schemes available, such as stream ciphers, block cipher modes of operation, hybrid encryption methods based on public key cryptography, and private-key encryption schemes such as Diffie–Hellman key exchange algorithm. Each scheme provides a certain level of security depending on factors such as strength, efficiency, and implementation complexity.

However, even after successful authentication and encryption, vulnerabilities may remain hidden within the wireless LAN, making it susceptible to attacks such as man-in-the-middle attacks, replay attacks, packet injection, and flooding. One way to mitigate such threats is to enforce strict firewall policies that restrict incoming and outgoing traffic according to the desired security level. Another option is to deploy intrusion prevention systems (IPS) that continuously monitor the network activity and alert administrators if malicious activities are detected.

# 3.核心算法
This section presents the basic concepts behind SAE and ECC, which will help us understand how they work in practice. We start with the SKES algorithm followed by CKMS and finally ECC algorithms to build our understanding. Let’s get started!

## 3.1 SKES
Secure Key Exchange Protocol (SKES) is a protocol designed specifically to distribute symmetric session keys securely between pairs of wireless endpoint devices, which rely on elliptic curve cryptography (ECC) for key exchange. SKES uses prime field arithmetic, which allows for faster computation compared to conventional finite field arithmetic used in RSA encryption. Additionally, SKES avoids timing attacks by incorporating a random delay before sending the first bit of data, making it harder for an attacker to capture messages in transit.

At a high level, SKES works as follows:

1. Device A generates a private key kA randomly, and computes the corresponding public key PKA = kA * G. 

2. Device B receives PKA and selects a random value rB. 

3. Device A sends PKA, rB, and MAC(KDF(|PKA||rB|)), where MAC stands for Message Authentication Code, KDF stands for Key Derivation Function, |x| denotes the length of x in bits, and ^ denotes concatenation.

   - MAC function calculates a unique hash code of the concatenated values PKA and rB, thus proving the origin of the message and validating the integrity of the parameters.
   - KDF derives a shared key based on both public keys PKA and rB, which is later used for data encryption.
   
4. Device B validates MAC received from A and computes the shared key kAB as follows:
   
   kAB = (kA*PB^(-rB)) % p, where PB=G^(-rB)%p is the inverse of rB modulo p.
   
5. Both devices store the derived shared key kAB and continue the data encryption using the same key.
 
Note that SKES does not guarantee perfect forward secrecy since it relies solely on the latest public key obtained from the other party. Nonetheless, SKES provides significant advantages over most other secure key distribution protocols, especially when the number of devices involved reaches large scales. Nevertheless, there are many ways to improve SKES, some of which include:

1. Asymmetric key exchange instead of ECDHE for increased speed and flexibility.
2. Multiple iterations of KDF to increase entropy and reduce the impact of brute force attacks.
3. Better padding strategies for plaintext size to minimize chances of key recovery.

## 3.2 CKMS
Centralized Key Management System (CKMS) is a mechanism that stores all the secrets associated with client identities and distributes them securely to individual devices. Unlike centralized key management systems such as Active Directory or OpenLDAP, CKMS does not rely on external authorities for identity registration, enrollment, and authentication processes. Instead, CKMS leverages off-line key generation and storage mechanisms implemented within the wireless endpoints themselves. Here's what happens at a high level:

1. Client A registers their user credentials with the server, which assigns them an ID called RIDA.
2. Client A requests a certificate from the CA signed by a trusted Certificate Authority (CA). This certificate contains attributes such as the public key PA of client A, the serial number SN of the cert, and validity periods for the certificate.
3. Client A downloads the certificate to generate a session key pair SK_CA.
4. The server verifies the signature of the certificate provided by client A, and generates a shared key SK_ServerA using a combination of SK_CA and the server's secret key.
5. Server stores the SK_ServerA associated with RIDA and sends it back to client A.
6. Client A generates a session key pair SK_ClientA using SK_ServerA and the PIN entered during registration.
7. Client A forwards SK_ClientA to the appropriate server-side application for processing and stores it locally until disconnection.

Note that CKMS requires careful attention to manage the lifetime of certificates, since they can potentially be compromised if stolen or lost. Additionally, the deployment of distributed key management solutions adds additional layers of complexity due to the requirement of secure key distribution between nodes. Nonethethanol, CKMS may still provide valuable insights in reducing the risk of security breaches due to weak passwords and passphrases. Nevertheless, it is worth noting that recent studies show that wireless endpoints could eventually achieve similar levels of security and convenience as traditional endpoints through the integration of machine learning capabilities and biometric authentication systems (Davis et al., 2017; McCormick et al., 2019).

## 3.3 ECC
Elliptic Curve Cryptography (ECC) was originally introduced in the late 1980s as a replacement for Digital Signature Algorithm (DSS) used in SSL/TLS protocols. Today, ECC is commonly used for various cryptographic operations such as key exchange, digital signatures, encryption, and hashing. 

Instead of relying on traditional curves such as NIST P-192, SECP256R1, or BRAINPOOL P-256, ECC uses characteristic-two fields defined over finite fields of prime order, making it faster, cheaper, and easier to implement compared to non-EC cryptography approaches. Compared to DSS, ECC provides the advantage of higher security, resilience to small subgroup attacks, and efficient implementations compared to deterministic signature schemes such as HMAC-SHA1.

As mentioned earlier, SAE relies on ECDHE for key exchange and AES-GCM for data encryption. However, we need to note that ECC offers several benefits beyond these features:

1. Improved time complexity: Arithmetic operations on EC points can be performed efficiently thanks to modular exponentiation.
2. Reduced memory usage: Public keys and private keys can be stored efficiently by compressing point coordinates.
3. Flexibility: Different curves can be used depending on the desired security level and computational resources.
4. Simplicity: Implementations are relatively straightforward and well documented compared to traditional DH key exchange protocols.

However, deploying ECC in a wired environment poses several challenges, especially when dealing with mobility scenarios, intermittent connectivity, and low signal strength. Appropriate routing and forwarding policies need to be developed to optimize throughput and decrease latency. Also, optimal placement of antennae, multipath propagation effects, and power constraints need to be carefully considered to ensure good performance. Last but not least, careful maintenance of ECC implementation needs to be performed to stay up-to-date with the latest developments in the mathematical theory of elliptic curves, leading to potential security vulnerabilities. Despite these drawbacks, however, ECC has become a promising solution for implementing secure wireless communication protocols that require end-to-end encryption.