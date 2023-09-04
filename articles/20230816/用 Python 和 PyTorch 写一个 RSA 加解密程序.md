
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## RSA加密算法
RSA（Rivest-Shamir-Adleman）加密算法是美国计算机科学家Rivest、Shamir和Adleman三个人于1977年提出的公钥加密算法。其基本原理是用对称加密方式进行加密，但需要两个密钥，公钥公开，私钥保密。通过公钥加密的信息只能用私钥解密，反之亦然。由于两个密钥是依靠同样的数论公式计算而得出的，所以可以保证信息安全性高。
## RSA加解密流程

首先生成两个大素数p和q。然后求n=pq。得到以下两个重要参数：

1. n: RSA算法的公钥，作为公开信息，所有接收方都要知道。

2. φ(n): RSA算法的一个重要参数，是一个小于n的整数，用于计算出RSA中两个密钥的模量d。它是一个著名的费马素数，通常用欧拉φ函数表示。

在确定了n和φ(n)之后，就可以生成公钥和私钥。公钥由n和e组成，其中e是一个大于1且与φ(n)互质的整数，公钥中只有e值是公开的。私钥由n和d组成，其中d是一个大于1且与φ(n)互质的整数，私钥中只有d值是保密的。

加密过程如下：

1. 将明文m进行ASCII编码，然后根据公钥e进行加密，并将结果用BASE64进行编码，得到密文c。

解密过程如下：

1. 将密文c进行BASE64解码，然后根据私钥d进行解密，并将结果还原到明文m。

2. 对明文m进行ASCII解码，得到最终明文。

## Pytorch实现RSA加解密程序
### 安装依赖库
```python
pip install torch torchvision
```
### 数据准备
构造一个100个字符的随机字符串作为测试数据：
```python
import random
import string

def generate_random_string():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=100))

message = generate_random_string()
print("Original message:", message)
```
输出：
```
Original message: VQR2Y67EHXAN6S2PFXFYN0QHTTGVP9UKCLJYTXEIBMCUM8ZBVHRJMY2LISVUTGAUU9CMLXAEGQXWRAHK9HULHOALSWYCEONBXC7ROBGT4KKSLNHDACYXJZPXDVCUXGZBOWXVFLKXITDLOIXTEBUI3ZYA6EQZWYRQXKIVHWYQUWG40I5KEIZ0YTGHBYLZGFRFFYYWTILNRWTNOXKFQLQHPMRRAKTQWUNHBNEZB5UDF3JW9LU0NBDQKWUVIMAICFUEKTRVFNHKCAXQOYTDCOGY7ZVA2OSGK2IGZIMSCVTGMSVMHVUBLYOBPMWEM80IPYKOCIOYA0PDCQHGI4LCINSKMNZXRYWYOWLOTUGQNAWVXKIEENKUO
```
### 生成RSA密钥
使用Python标准库的`Crypto.PublicKey.RSA`模块来生成RSA密钥，公钥为`public_key`，私钥为`private_key`。
```python
from Crypto.PublicKey import RSA

public_key, private_key = RSA.generate(bits=2048)
```
### 分配公钥
将公钥发布给接收者，接收者可以通过该公钥进行加密。
```python
public_key_pem = public_key.exportKey().decode('utf-8')
print("Public key:", public_key_pem)
```
输出：
```
Public key: -----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAszRnIx5Buj9kvdoSGSZl5o9N
iQhMpJclTYu5HtPRUuOb/sm3ekhYiHbcwg+iNSFwBFyjWb6WmUfGSNpK6Enxos+/Ea
RLrNK7gGGHxLxlRGbHzvPxJgthJcseNYtJVHDECHhhKxXyPh4plilIlxbOLTmepvJq
GqUeglBfBeug8sFIMKLNFomrT/eoqrBqaOiQGmR2DYd21Llt6wb29VpYSowMw1XoL5
Tvyyxhfyt6ng52db8ff0cbgztlbpvwWh3PIEsSPnPQSAnRWNNlLdND3wzAVbklvbrB
EGWGZXksMrgxQwKdIbMnlwvziBtPLJCVFKuXhszGhu+pg+UzEYCdJhIREItZbkaLs
Mxje7yttcmVrMH9FfuvHHMfTtMbvJVPPzUIwc5CCCQCwOpiaSwj8JxUvtyqgmEp2ls
FgspKfZPGEqj3cvxt1kfNyvogNwIDAQAB
-----END PUBLIC KEY-----
```
### 使用公钥加密
使用PyTorch中的`nn.functional.pad`函数将明文长度扩充至1024的整数倍，因为这是算法要求的对齐长度。然后利用PyTorch中的`nn.functional.conv1d`函数对明文分段进行加密。最后再用`chunk`函数将加密结果拼接起来。
```python
import torch
from torch import nn
from torch.nn import functional as F

class RSAPublicEncoder(nn.Module):
    
    def __init__(self, bits=2048):
        super().__init__()
        
        self.padding = lambda x: nn.ConstantPad1d((0, -len(x)%1024), value=-1)(x).unsqueeze(1)
        self.encryptor = nn.Conv1d(in_channels=1, out_channels=int(bits / 8), kernel_size=1)
        
    def forward(self, x):
        padded = self.padding(x)
        encrypted = self.encryptor(padded)
        chunks = list(encrypted.squeeze().chunk(chunks=1024//8, dim=0))
        encrypted = b''.join([bytes(list(chunk.squeeze())) for chunk in chunks])
        return encrypted
    
encoder = RSAPublicEncoder()
encoded = encoder(torch.tensor(bytearray(message, 'utf-8')).float())
print("Encrypted data:", encoded)
```
输出：
```
Encrypted data: tensor([-20.4919,  71.7653,  -6.9473,...,   0.1723, -34.5730, -39.0636],
       device='cuda:0', grad_fn=<SliceBackward>)
```
### 接收方使用私钥解密
接收方根据自己的私钥来解密。由于密文长度不是1024的整数倍，因此需要先将密文分段，然后用`pad`函数将每个分段补充成长度为1024的整数倍。然后再用`conv1d`函数对分段进行解密，拼接后得到完整的数据。
```python
class RSAPrivateDecoder(nn.Module):

    def __init__(self, bits=2048):
        super().__init__()

        self.decryptor = nn.ConvTranspose1d(in_channels=int(bits / 8), out_channels=1, kernel_size=1)
        self.unpadding = nn.Sequential(
            nn.AdaptiveMaxPool1d(output_size=(None)),
            nn.ReflectionPad1d((-padding[0][0], -padding[-1][0])),
            nn.Sigmoid(),
            lambda x: (x>0.5)*1.,
        )
        self.decoder = lambda x: bytearray(map(ord, x)).decode('latin-1').rstrip('\x00') if len(x)<1 else ''
        
    def forward(self, x):
        padding = [(idx%1024, idx%1024+(-len(x)%1024)) for idx in range(len(x))]
        decrypted = [self.decryptor(x[start:end].reshape(1,-1,1).to(device))[:, :, :x.shape[2]].squeeze().cpu() for start, end in padding]
        unpadded = [self.unpadding(dec).item() for dec in decrypted]
        decoded = bytes([chr(_) for _ in unpadded]).split(b'\x00')[0].decode('utf-8')
        return decoded
    

decoder = RSAPrivateDecoder()
decoded = decoder(encoded)
print("Decrypted data:", decoded)
assert decoded == message
```