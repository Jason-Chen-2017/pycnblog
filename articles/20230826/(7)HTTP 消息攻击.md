
作者：禅与计算机程序设计艺术                    

# 1.简介
  


随着互联网的迅速发展，网站用户越来越多，对网站的安全性要求也越来越高。因此，网络攻击成为一种重要的威胁。近年来，由于网站功能的丰富性、复杂性和易用性，黑客利用网站漏洞进行各种各样的攻击行为，其中最常见的就是 HTTP 消息注入攻击。

HTTP 消息注入（又称为 HTTP Response Splitting 或 CRLF Injection）指的是通过在 HTTP 请求或者响应的消息中插入特殊字符或制表符等控制字符，导致服务器端接收到的数据发生错误、改变、泄露甚至执行恶意代码。黑客可以通过构造特殊的 HTTP 请求或者响应数据包，引导受害者点击链接、提交表单或者其他方式，将控制字符注入进去，从而达到篡改服务器的目的。

# 2.概念术语说明

## 2.1 HTTP 消息

Hypertext Transfer Protocol （超文本传输协议）是用于从万维网上超媒体信息资源传递的协议，它是一个基于请求/响应模式的协议，由请求方（如浏览器）发送一个请求报文到服务器，服务器响应并返回一个响应报文给请求方，整个过程中所经过的中间节点都遵循HTTP协议。

HTTP 的工作流程可以分为两个阶段，第一阶段是建立连接，包括客户端和服务器之间的三次握手；第二阶段则是传输信息，也就是在浏览器端输入 URL 后，服务器向浏览器返回相应的文件，此过程也即是一次 HTTP 请求-响应过程。

HTTP 是无状态协议，它不维护会话状态，也就是说不会记录两次请求之间的联系，也就没有了身份验证这样的功能，这也是为什么黑客需要先登陆网站才能进入其页面的原因。

HTTP 消息由请求行、请求头、空行和请求数据四个部分构成。请求行通常由三个字段组成，分别是方法、URL 和 HTTP 版本。请求头提供关于请求的更多信息，比如客户端类型、语言偏好、Cookie 值、认证信息等。空行表示请求头结束，接下来的请求数据可以包含任意的内容，如表单数据、文件上传等。

HTTP 报文的结构如下图所示：


## 2.2 CRLF Injection

CRLF Injection 是 HTTP 消息注入攻击中最常用的攻击方式之一。它指的是通过在 HTTP 请求或者响应消息中插入换行符，使得服务器接收到的信息出现错误。常见的方法是利用 JavaScript 执行 HTTP POST 请求，可以自动填充表单信息。通过设置 Content-Length 标头的值，也可以模拟出不同长度的请求数据包，从而触发 HTTP 数据块校验机制，进一步导致服务器接收数据的混乱。

当服务器收到特殊的请求时，可能导致以下问题：

1. 无法正确处理请求，例如导致数据库异常、崩溃或无法正确显示页面。
2. 敏感信息泄露，包括用户名和密码、个人隐私等。
3. 服务器内核内存溢出或崩溃，可造成系统瘫痪。
4. 服务器执行恶意代码，导致拒绝服务攻击或远程执行命令。
5. 浏览器崩溃，导致浏览器卡死或浏览器被黑客操控。

# 3.算法原理及具体操作步骤

## 3.1 插入换行符

CRLF Injection 可以通过插入一些特殊字符（如\r\n），引导浏览器向服务器发起 HTTP 请求。这种攻击可以在请求前后增加一定的长度，引导服务器解析数据，从而达到修改数据的效果。

首先，可以用 Burpsuite 修改请求数据包，找到影响请求结果的关键点，然后再向前或向后追加一些特殊字符，如换行符、制表符、NULL 字符等。

Burp Suite 是一款强大的 Web 调试代理工具，它的特点是能够捕获、编辑和重放请求与响应数据，并且支持强大的断言功能，很容易用于测试网站安全性。

另外，可以使用抓包工具抓取请求数据包，然后在其中的某个位置添加一些控制字符，并通过修改 Content-Length 或者请求数据包大小的方式，触发漏洞。

```python
POST / HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36
Content-Type: application/x-www-form-urlencoded

username=%s&password=<PASSWORD>%s<script>alert("XSS")</script>&login=Login
```

可以看到，数据包最后的 `&login=Login` 之前添加了一个 `<script>` 标签，插入脚本使得服务器将其渲染为一个 alert 对话框。

## 3.2 设置 Content-Length 标头

另一种攻击方式是在请求头中设置 Content-Length 标头的值，使得服务器读取到的数据长度出现错误，进而触发数据块校验机制。这个方法比较灵活，可以设置较短或较长的 Content-Length 来干扰正常的数据流。

为了触发这一漏洞，可以用抓包工具获取正常请求数据包，然后将其截取掉，并在新请求中加入一些额外的字节，使得数据包长度变短。如下面例子所示，将 `Content-Length` 从 `46` 减小到 `3`，就产生了 Content-Length Injection 漏洞。

```python
GET /test.php?id=1 HTTP/1.1
Host: www.example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36
Connection: keep-alive
Upgrade-Insecure-Requests: 1
Sec-Fetch-Site: same-origin
Sec-Fetch-Mode: navigate
Sec-Fetch-User:?1
Accept-Encoding: gzip, deflate
Accept-Language: zh-CN,zh;q=0.9

XXXXXXX%c0%af
```

这里，除了最后的 `X` 以外，还增加了一个 `%c0%af` 字符，该字符采用两个字节编码 `\xC0\xAF`。在浏览器中打开这个请求，服务器会解析到数据长度只有两个字节，就会导致数据包被截断，导致请求无法正常完成。

# 4.代码实例与解释说明

上面主要介绍了 HTTP 消息注入的两种攻击方式——CRLF Injection 和 Content-Length Injection。下面给出具体的代码实例，阐述一下攻击的实现原理。

## 4.1 Python 实现

### 4.1.1 使用 CRLF Injection 攻击

使用 Python 的 `requests` 模块，可以方便地向目标站点发起 HTTP 请求。

```python
import requests

url = 'http://target.site'
data = {
    'user': 'admin',
    # 在 password 之后插入换行符
    'password': '<PASSWORD>' + '\r\n',
   'submit': 'Login',
}
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.post(url, data=data, headers=headers)
print(response.content)
```

运行以上代码，就可以触发 CRLF Injection 漏洞。通过查看服务器日志，可以发现 POST 请求的数据包中，`password` 参数的值中包含了换行符。服务器在解析数据的时候，如果遇到了换行符，可能会误认为这是另一条 HTTP 请求。

```log
[Thu Jul 05 14:38:02.183069 2020] [:error] [pid 13051] [client 127.0.0.1:56138] PHP Warning:  parse_str(): Cannot add an element to the array as the next element is already occupied in Unknown on line <number>, referer: http://target.site
```

为了防止 CRLF Injection 攻击，可以对传入的数据参数做检查，禁止把含有特殊字符的数据传递给服务器。

### 4.1.2 使用 Content-Length Injection 攻击

设置请求头的 `Content-Length` 标头的值，可以激活 Content-Length Injection 漏洞。

```python
import requests

url = 'http://target.site'
data = '{"user":"admin","password": "xxxxxxxxxxxxxx","submit": "Login"}'
headers = {
    'User-Agent': 'Mozilla/5.0',
    'Content-Length': len(data),  # 将请求体长度设置为最大值
}
response = requests.post(url, data=data, headers=headers)
print(response.content)
```

运行以上代码，服务器会判断数据长度是否和 `Content-Length` 标头一致，如果不一致，就会导致数据包被截断，从而导致请求失败。

为了防止 Content-Length Injection 攻击，可以使用流水线（pipeline）的形式发送请求，并在服务端判断数据长度是否合法。

```python
from queue import Queue
import threading

def worker():
    while not q.empty():
        url, data, headers = q.get()
        response = requests.post(url, data=data, headers=headers)
        print(response.content)
        q.task_done()

if __name__ == '__main__':
    urls = ['http://target.site'] * 100  # 创建100个相同的URL
    datas = ['{"user":"admin","password": "<PASSWORD>","submit": "Login"}'] * 100  # 创建100个相同的请求体
    headers = [{'User-Agent': 'Mozilla/5.0',
                'Content-Length': len(datas[i])} for i in range(len(urls))]

    q = Queue()
    for i in range(min(len(urls), len(datas))):
        q.put((urls[i], datas[i], headers[i]))
    
    for i in range(10):
        t = threading.Thread(target=worker)
        t.start()

    q.join()
```