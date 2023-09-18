
作者：禅与计算机程序设计艺术                    

# 1.简介
  

HTTP协议定义了很多状态码，用于表示HTTP请求或者响应的状态，各个状态码都有自己特定的含义。本文将对常用的HTTP状态码进行详细说明，并分享一下大家可能遇到的一些问题和解决方案。由于笔者不是HTTP专家，在写作过程中难免会出现错误或疏漏之处，还望海涵。
## 一、状态码的分类及意义
HTTP协议定义了7个类别中的状态码，它们分别是：

1xx信息性状态码-请求已经收到， continuing  
2xx成功状态码-请求正常处理完毕  
3xx重定向状态码-需要进行更进一步的操作才能完成请求  
4xx客户端错误状态码-服务器无法处理请求  
5xx服务器端错误状态码-服务器处理请求出错  
## 二、常用状态码列表
### 100 Continue 
　　该状态码表示客户端仍然发送其请求，但是服务器并没有接受到请求头部信息。客户端应当继续等待，直到收到服务器的回应。
### 101 Switching Protocols 
　　该状态码表示服务器根据客户端的请求切换到了新协议。只能切换到更高级的协议，例如，切换到TLS/1.0。
### 200 OK 
　　该状态码表示从客户端往服务器传输的数据完整，通常是在GET或POST方法下。
### 201 Created 
　　该状态码表示请求成功并且服务器创建了一个新的资源。
### 202 Accepted 
　　该状态码表示服务器已收到请求，但尚未处理。
### 203 Non-Authoritative Information 
　　该状态码表示客户端所获取的信息来自源站，但它可能不一定权威。
### 204 No Content 
　　该状态码表示服务器接收到了请求，但没有返回任何实体的内容。
### 205 Reset Content 
　　该状态码表示服务器已经成功处理了请求，且没有返回任何内容。
### 206 Partial Content 
　　该状态码表示服务器已经成功处理了部分 GET 请求。
### 300 Multiple Choices 
　　该状态码表示服务器可执行多种操作，请求的资源可包括多个位置。
### 301 Moved Permanently 
　　该状态码表示请求的网页已永久移动到新位置。
### 302 Found 
　　该状态码表示请求的网页临时从其他地方转移。
### 303 See Other 
　　该状态码表示由于 POST 没有被正确处理，所导致的请求失败，应该再次尝试。
### 304 Not Modified 
　　该状态码表示请求的资源未修改，服务器返回此状态码时，不会返回任何资源。
### 305 Use Proxy 
　　该状态码表示必须通过代理访问。
### 307 Temporary Redirect 
　　该状态码表示请求的资源临时从不同的URI响应请求，而且Future requests should use the original URI in future requests.
### 400 Bad Request 
　　该状态码表示请求报文存在语法错误。
### 401 Unauthorized 
　　该状态码表示发送的请求需要身份验证。
### 402 Payment Required 
　　该状态码表明在当前服务器上不能完成请求，客户应当在稍后重新发送请求。
### 403 Forbidden 
　　该状态码表示服务器拒绝请求。
### 404 Not Found 
　　该状态码表示请求的资源不存在。
### 405 Method Not Allowed 
　　该状态码表示请求中指定的方法不被允许。
### 406 Not Acceptable 
　　该状态码表示服务器无法满足客户端请求里的条件。
### 407 Proxy Authentication Required 
　　该状态码类似于401，表示客户端需要代理进行授权。
### 408 Request Timeout 
　　该状态码表示客户端没有在服务器预备等待的时间内完成请求。
### 409 Conflict 
　　该状态码表示请求的资源与资源的当前状态之间存在冲突。
### 410 Gone 
　　该状态码表示服务器上的资源被永久删除。
### 411 Length Required 
　　该状态码表示客户端需要先指定Content-Length。
### 412 Precondition Failed 
　　该状态码表示请求的Preconditions不满足。
### 413 Payload Too Large 
　　该状态码表示服务器拒绝接收超过指定大小的请求。
### 414 URI Too Long 
　　该状态码表示请求的URI过长（URI通常为网址）。
### 415 Unsupported Media Type 
　　该状态码表示服务器不支持请求中媒体类型。
### 416 Range Not Satisfiable 
　　该状态码表示客户端请求的范围无效。
### 417 Expectation Failed 
　　该状态码表示服务器无法满足Expect的请求标头字段。
### 418 I'm a teapot 
　　该状态码表示我是一个茶壶，并希望得到一个微茶。
### 421 There are too many connections from your internet address 
　　该状态码表示由于数量太多的连接引起的服务器暂时无法处理请求。
### 422 Unprocessable Entity 
　　该状态码表示服务器无法处理请求实体。
### 423 Locked 
　　该状态码表示服务器处理请求时发生冲突。
### 424 Failed Dependency 
　　该状态码表示由于之前的某个请求发生的故障，导致请求失败。
### 426 Upgrade Required 
　　该状态码表示客户端应当切换到一个不同版本的协议。
### 428 Precondition Required 
　　该状态码表示要求前提不能为空。
### 429 Too Many Requests 
　　该状态码表示用户在给定的时间段内发送太多请求。
### 431 Request Header Fields Too Large 
　　该状态码表示服务器不接受请求中特定的Header字段。
### 451 Unavailable for Legal Reasons 
　　该状态码表示该状态码由Internet Assigned Numbers Authority (IANA) 定义，代表请求被拒绝。
### 500 Internal Server Error 
　　该状态码表示服务器遇到错误，无法完成请求。
### 501 Not Implemented 
　　该状态码表示服务器不支持请求的功能。
### 502 Bad Gateway 
　　该状态码表示服务器作为网关或代理，从上游服务器收到无效的响应。
### 503 Service Unavailable 
　　该状态码表示服务暂时不可用，通常是由于维护或者过载。
### 504 Gateway Timeout 
　　该状态码表示上游服务器超时。
### 505 HTTP Version Not Supported 
　　该状态码表示服务器不支持HTTP协议版本。
### 506 Variant Also Negotiates 
　　该状态码表示服务器存在内部配置错误。
### 507 Insufficient Storage 
　　该状态码表示服务器无法存储完成请求所需的内容。
### 508 Loop Detected 
　　该状态码表示服务器在处理请求时陷入死循环。
### 510 Not Extended 
　　该状态码表示客户端试图扩展协议的愿望，但是服务器不支持。
### 511 Network Authentication Required 
　　该状态码表示客户端需要进行网络认证。