
作者：禅与计算机程序设计艺术                    

# 1.简介
  

HTTP 是 HyperText Transfer Protocol（超文本传输协议）的缩写，是一个用于从 Web 服务器传输超文本到本地浏览器的协议。每当浏览器或者其他客户端向Web服务器发送请求时，都需要通过 HTTP 来建立连接、发送请求报文并接收响应报文。而 HTTP 状态码 (Status Code) 是 HTTP 协议返回给客户端的一种信息，它用来表示请求成功或失败以及相关的信息。在网络通信中，不同的状态码代表了不同的含义。本文旨在为刚刚接触 HTTP 状态码的读者提供一个简单的学习手册，帮助读者快速了解 HTTP 状态码的含义和各种分类方法，并且可以看到如何编写简单的 Python 代码来获取和处理 HTTP 状态码。本文着重于 HTTP/1.x 版本的状态码，也会涉及到一些较老旧的版本，例如：HTTP/0.9。如果您对这些状态码已经比较熟悉，但仍然想进一步了解更多的话，还可以在阅读过程中随时查阅相关资料。
# 2.基本概念术语说明
## 2.1 HTTP 方法
HTTP 请求可以使用以下几种方法之一: GET、POST、PUT、DELETE、HEAD、OPTIONS、TRACE 和 CONNECT。其中，GET、POST、HEAD、OPTIONS 和 TRACE 在最常用的 HTTP 操作中都会用到，其它的方法都是可选的扩展。
- **GET**：HTTP GET 方法被用于从指定的资源请求数据。该方法使用最简单的方式向服务器请求资源，但是会引起服务器压力，所以仅用于不安全环境且要求响应尽快的情况。
- **POST**：HTTP POST 方法被用于将实体提交到指定资源，它是常用的方法，可以向服务器传递数据。
- **PUT**：HTTP PUT 方法被用于上传指定资源，它也是常用的方法，相比 POST，PUT 更加安全，不会造成修改现有资源的问题。
- **DELETE**：HTTP DELETE 方法用于删除服务器上的资源。
- **HEAD**：HTTP HEAD 方法用于只获取 HTTP 头部信息，不能获得实体的主体部分。
- **OPTIONS**：HTTP OPTIONS 方法用于查询针对特定 URL 可以使用的请求方法，也可以查询服务器的性能。
- **TRACE**：HTTP TRACE 方法用于回显服务器收到的请求，主要用于测试或诊断。
- **CONNECT**：HTTP CONNECT 方法用于创建连接隧道，可以透明地转发连接。
## 2.2 HTTP 状态码
HTTP 状态码 (Status Code) 是指在 HTTP 协议下，服务器向客户端返回的请求结果状态码。HTTP 状态码是三位数字，第一个数字定义了响应类别，第二个数字定义了响应类型，第三个数字则用于进一步细化错误信息。常见的 HTTP 状态码如下表所示：
| 状态码 | 描述                                                         |
| ------ | ------------------------------------------------------------ |
| 1XX    | Informational（信息性状态码）：指示的状态信息，一切正常，继续请求。 |
| 2XX    | Success（成功状态码）：服务器已成功接收并理解请求的语法，并响应成功。 |
| 3XX    | Redirection（重定向状态码）：要完成请求必须进行更进一步的操作。        |
| 4XX    | Client Error（客户端错误状态码）：服务器无法处理请求，因为客户端提供了错误的请求数据或格式。 |
| 5XX    | Server Error（服务端错误状态码）：服务器处理请求出错，可能是程序错误、链接超时等。 |
除了常见状态码外，还有一些类似 403 Forbidden 的特殊状态码，它们只是一些常用的状态码的组合，不是具体的错误描述。
## 2.3 HTTP 状态码类别
一般来说，HTTP 状态码可以分为五个大类，分别为：信息性状态码（1XX），成功状态码（2XX），重定向状态码（3XX），客户端错误状态码（4XX），服务端错误状态码（5XX）。下面我们逐一看一下这些状态码类的含义：
### 2.3.1 信息性状态码（1XX）
- 100 Continue：服务器仅接收到了请求的一部分，客户应继续发送其余部分。
- 101 Switching Protocols：服务器转换协议。
- 102 Processing（WebDAV）：正在处理事务。
- 103 Early Hints（H2C）：提示先行（HTTP/2 服务器推送）。
### 2.3.2 成功状态码（2XX）
- 200 OK：请求成功。
- 201 Created：已创建新资源。
- 202 Accepted：已接受请求，但未处理完成。
- 203 Non-Authoritative Information（非权威信息）：文档已经更新，客户端应继续从他处获取最新信息。
- 204 No Content：服务器成功处理，但未返回任何内容。
- 205 Reset Content：服务器成功处理，但未清空网页的表单内容。
- 206 Partial Content：客户发送了一个带 Range 头的 GET 请求，服务器完成了范围请求，响应包括由 Range 指定的那些字节的内容。
- 207 Multi-Status（WebDAV）：多状态。
- 208 Already Reported（WebDAV）：告知客户端存在多个响应。
- 226 IM Used（HTTP Delta encoding）：客户端缓存通知。
### 2.3.3 重定向状态码（3XX）
- 300 Multiple Choices：多重选择。
- 301 Moved Permanently：永久移动。
- 302 Found（Moved temporarily）：临时移动。
- 303 See Other：查看其他位置。
- 304 Not Modified：未修改。
- 305 Use Proxy（从代理）：使用代理。
- 307 Temporary Redirect（重定向）：临时重定向。
- 308 Permanent Redirect（永久重定向）：永久重定向。
### 2.3.4 客户端错误状态码（4XX）
- 400 Bad Request：客户端请求的语法错误，服务器无法理解。
- 401 Unauthorized：请求要求身份验证。
- 402 Payment Required：保留有效使用，目前未使用。
- 403 Forbidden：服务器拒绝执行请求。
- 404 Not Found：服务器无法根据客户端的请求找到资源。
- 405 Method Not Allowed：禁止访问资源。
- 406 Not Acceptable：由于客户所拥有的权利数据库里没有找到合适的数据，服务器无法满足相应请求的Accept头信息。
- 407 Proxy Authentication Required：需要代理授权。
- 408 Request Timeout：请求超时。
- 409 Conflict：请求冲突。
- 410 Gone：资源不可用。
- 411 Length Required：Content-Length 未定义。
- 412 Precondition Failed：前置条件失败。
- 413 Payload Too Large：负载过大。
- 414 URI Too Long：URI 太长。
- 415 Unsupported Media Type：媒体类型不支持。
- 416 Range Not Satisfiable：所请求的范围无效。
- 417 Expectation Failed：期望失败。
- 418 I'm a teapot：服务不可用。
- 421 Misdirected Request（HTTP Delta encoding）：请求的目标跟进地址。
- 422 Unprocessable Entity（WebDAV）：语义错误。
- 423 Locked（WebDAV）：当前资源被锁定。
- 424 Failed Dependency（WebDAV）：由于之前的某个请求发生失败，导致当前请求失败。
- 426 Upgrade Required：客户端应升级到其他协商协议。
- 428 Precondition Required：要求先决条件。
- 429 Too Many Requests：请求次数过多。
- 431 Request Header Fields Too Large：头字段太大。
- 444 Connection Closed Without Response：连接关闭，服务器没有回应。
- 449 Retry With（Microsoft）：切换协议。
- 450 Blocked by Windows Parental Controls：当前客户端所在的 IP 地址被阻止。
- 451 Unavailable For Legal Reasons：由于法律原因，请求不可用。
- 499 Client Closed Request（IIS）：客户端关闭连接。
### 2.3.5 服务端错误状态码（5XX）
- 500 Internal Server Error：服务器内部错误，无法完成请求。
- 501 Not Implemented：尚未实施。
- 502 Bad Gateway：服务器作为网关或代理，从上游服务器收到无效的响应。
- 503 Service Unavailable：服务不可用。
- 504 Gateway Timeout：网关超时。
- 505 HTTP Version Not Supported：服务器不支持请求的 HTTP 协议版本。
- 506 Variant Also Negotiates（协商）：由网关选择变体协议。
- 507 Insufficient Storage（WebDAV）：存储空间不足。
- 508 Loop Detected（WebDAV）：循环检测。
- 509 Bandwidth Limit Exceeded（Apache）：带宽限制超出。
- 510 Not Extended（下一个版本的协议）：扩展错误。
- 511 Network Authentication Required（TLS/SSL）：网络认证失败。
- 598 Network read timeout error（Unknown）：网络读超时错误。
- 599 Network connect timeout error（Unknown）：网络连接超时错误。