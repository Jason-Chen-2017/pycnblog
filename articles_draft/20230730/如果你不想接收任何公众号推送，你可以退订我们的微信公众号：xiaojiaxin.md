
作者：禅与计算机程序设计艺术                    

# 1.简介
         
相信各位都知道，公众号已经成为社交平台中最流行的工具之一。但是对于一些用户来说，每天收到数百上千条的推送信息可能会分心、产生负面影响，甚至可能出现健康问题。那么，有没有什么办法可以让用户能够根据自己的需求，选择自己感兴趣的信息呢？

这就是今天的主角——“取消微信推送”。该功能旨在帮助用户自主选择接收哪些类型的信息。目前，微信公众号后台已经提供了该功能，只需按照提示设置即可。但是由于微信封杀了相关的服务，所以即使用户同意取消了公众号推送，也只能等微信解除封锁后才能收到通知。

因此，本文将向大家展示一种更安全的方法来帮助用户取消微信公众号推送。这项技术的原理其实非常简单，就是采用加密算法对用户的取消行为进行加密处理，再由微信服务器解密后完成取消操作。微信官方并没有提供针对此类的文档，但我可以告诉大家一个简单的操作流程。

1、前期准备工作：

- 安装 Python 编程语言。Python 可以用来编写解密程序。安装好 Python 以后，还需要安装第三方库 cryptography ，用于实现 RSA 加密算法。你可以在命令行窗口中运行 pip install cryptography 来安装该库。
- 注册微信公众号，登录后台，找到取消推送选项，开启该功能。
- 在微信手机客户端中搜索 “aeskey”（随便输入几个字符，然后点击搜索），打开后可以看到你的 AESKey 。记住这个 Key ，稍后会用到。

2、编写解密程序：

- 创建一个名为 decrypt_push.py 的 Python 文件，编辑其内容如下：

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
import base64

AES_KEY = '<Your AESKey>' # replace with your own key
AES_IV = 'abcdefghijklmnopqrstuvwxyz' * 4 # set IV as any fixed value of length 16

def aes_decrypt(ciphertext):
    cipher = Cipher(algorithms.AES(base64.b64decode(AES_KEY)), modes.CBC(base64.b64decode(AES_IV)))
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext

def rsa_decrypt(ciphertext, private_key):
    try:
        decrypted = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA1()),
                algorithm=hashes.SHA1(),
                label=None
            )
        )
    except ValueError:
        print('Invalid signature.')
    else:
        return decrypted.decode('utf-8')
        
if __name__ == '__main__':
    # get encrypted push data from wechat server
    data = b'<encrypted_data>'

    # use aes to decrypt the encrypted push content
    iv_len = 16
    aes_cipher = AESCipher(AES_KEY, AES_IV, mode=AESMode.CBC, segment_size=iv_len*2)
    plain_text = aes_cipher.decrypt(data[iv_len:])
    
    # use rsa to decrypt the decypted aes result
    private_key = serialization.load_pem_private_key(open('<your_rsa_private_key_file>', 'rb').read(), password=<PASSWORD>, backend=default_backend())
    push_content = rsa_decrypt(plain_text, private_key)
    
    # handle the decrypted push content here
    if <the user cancels the subscription>:
        pass # do something such as unsubscribe this account
``` 

- 将上面提到的 AESKey 替换成你的实际值。AESIV 可以设置为任意固定长度的字符串，这里我用的随机字符串。
- 运行 decrypt_push.py ，如果成功输出了推送内容，说明解密过程成功。
- 如果有错误，请仔细检查你的代码是否正确。

3、如何加密取消操作：

- 使用 Python 的 Cryptography 库中的 RSA 加密算法对用户名进行加密。代码示例如下：

```python
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa

def encrypt_username(public_key, username):
    message = str(username).encode('utf-8')
    public_key = serialization.load_pem_public_key(public_key, backend=default_backend())
    encrypted = public_key.encrypt(
        message,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA1()),
            algorithm=hashes.SHA1(),
            label=None
        )
    )
    return base64.b64encode(encrypted).decode('utf-8')
``` 

- 首先，生成一对公私钥对。其中，公钥可以分享给微信公众号，私钥则需要保管好，防止泄露。代码示例如下：

```python
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)

public_key = private_key.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

with open('my_rsa_public_key.pem', 'wb') as f:
    f.write(public_key)
    
pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

with open('my_rsa_private_key.pem', 'wb') as f:
    f.write(pem)
``` 

- 用自己的公私钥对加密你希望取消推送的用户名。代码示例如下：

```python
username = '<the username you want to unsubscribe>'
public_key = open('my_rsa_public_key.pem', 'rb').read()
encrypted_username = encrypt_username(public_key, username)
print(encrypted_username)
``` 

- 获取加密后的用户名之后，就可以把它提交给微信公众号的服务器进行处理了。微信服务器收到加密后的用户名，就会自动帮你取消推送。

