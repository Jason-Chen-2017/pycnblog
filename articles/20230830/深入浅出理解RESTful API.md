
作者：禅与计算机程序设计艺术                    

# 1.简介
  

REST（Representational State Transfer）即表述性状态转移，它是一个设计风格、约束条件和基于HTTP协议的Web服务之间互相通信的 architectural style。其思想是通过标准的HTTP方法如GET、POST、PUT、DELETE等对资源进行操作，从而改变或返回服务器上的资源的状态或行为。可以用一句话来概括REST的特点就是“client-server”“stateless”“cacheable”三个方面。

资源（Resource）：一般来说，Web服务中所描述的实体称为资源。比如在线电子商务网站中的用户信息、订单记录、产品信息等都是资源。

表现层（Representations）：采用特定的数据格式表示资源。比如JSON、XML、HTML等。不同的客户端可以通过不同的表现形式（representation）请求资源。

状态转移（State Transfer）：由于Web上资源的状态（State）可能会随时发生变化，因此需要一种机制保证客户端能够获取到最新的状态。而这种机制就是状态转移。REST则通过各种HTTP方法来实现资源的创建、查询、修改和删除，并通过状态码（Status Codes）来反映执行成功或者失败的结果。

统一接口（Uniform Interface）：REST最重要的一条要求就是统一的接口。通过定义一系列的标准的HTTP方法，使得客户端和服务器之间交流变得更加简单和一致。

缓存（Cacheability）：为了提升性能，通常情况下，服务器会将某些响应数据缓存起来，以便下次访问时直接提供给客户端。然而，也存在一些例外情况，比如用户的私密信息、安全性要求高的资源等。REST允许客户端和服务器通过缓存控制（Cache-Control）头来指定某个响应是否应该被缓存，如何被缓存，以及什么时候应该被更新。

RESTful API
RESTful API的全称是Representational State Transfer RESTful Architectural Style。它通过标准的HTTP方法，通过URL定位资源，并用HTTP的状态码来表示动作的成功或失败，并通过其他的约定俗成的方式来处理问题。因此，它可以帮助服务的开发者创建面向RESTful架构风格的API，让Web服务的架构师和前端工程师能更方便地进行前后端分离和交互。RESTful API是构建RESTful Web服务的基石。

# 2.基本概念及术语
## 请求（Request)
RESTful API的请求由三部分组成：请求行、请求头、请求体。下面举一个例子说明各个部分代表什么意义：
```http
GET /users HTTP/1.1
Host: example.com
Accept: application/json
Content-Type: application/json
{
  "name": "Alice",
  "age": 30
}
```
- 请求行：`GET /users HTTP/1.1`，用来指定HTTP方法以及请求路径。
- 请求头：`Host: example.com`，指定要请求的域名。
- 请求体：`{ "name": "Alice", "age": 30 }`，POST方法发送的数据。

## 响应（Response)
响应也是由三部分组成：响应行、响应头、响应体。下面举一个例子说明各个部分代表什么意义：
```http
HTTP/1.1 200 OK
Date: Sat, 09 Oct 2019 07:53:54 GMT
Server: Apache/2.4.29 (Ubuntu)
Last-Modified: Wed, 08 Sep 2019 10:39:24 GMT
ETag: "1f2e3d4a5b6c7890"
Content-Length: 100
Connection: close
Content-Type: application/json;charset=utf-8
[
  {
    "id": 1,
    "name": "Alice",
    "age": 30
  },
  {
    "id": 2,
    "name": "Bob",
    "age": 25
  }
]
```
- 响应行：`HTTP/1.1 200 OK`，用于返回HTTP版本号、状态码及描述。
- 响应头：包括了一些元数据，比如日期、服务器类型、最后修改时间、标记等。
- 响应体：包含了实际的响应内容，比如返回的用户信息列表。

## URI（Uniform Resource Identifier）
URI（Uniform Resource Identifier）是一个字符串，它唯一标识了一个资源。URI由以下几部分组成：
- Scheme：定义了该资源采用的协议类型，比如HTTP、FTP等。
- Host：指定了请求的主机名或IP地址。
- Port：如果不使用默认端口，可添加端口号。
- Path：表示具体的资源位置，比如`/path/to/resource`。
- Query String：用于传递查询参数，比如`?page=2&limit=10`。
- Fragment identifier：指向文档内部的某一位置，比如`#section1`。

## 方法（Methods）
HTTP协议提供了一系列的请求方法，比如GET、POST、HEAD、PUT、DELETE等。不同的方法对应不同的资源操作。

### GET
GET方法用来请求服务器发送指定的资源。比如，如果请求路径为`https://example.com/users`，GET方法将会得到该资源的内容。

### POST
POST方法用来向服务器提交数据，服务器接收到请求之后，就处理这个数据，然后返回响应。比如，当用户注册一个新账号时，服务器收到请求的数据后，会新建用户账户，然后返回响应。

### PUT
PUT方法用来上传指定的资源。比如，如果请求路径为`https://example.com/users/1`，PUT方法将把请求体里的数据上传到对应的资源中。

### DELETE
DELETE方法用来删除指定的资源。比如，如果请求路径为`https://example.com/users/1`，DELETE方法将删除相应的资源。

### HEAD
HEAD方法和GET类似，但是不会返回响应体，只返回响应行和响应头。

### OPTIONS
OPTIONS方法用来获取目的资源支持的HTTP方法。比如，对于一个网页，OPTIONS方法将会返回允许的HTTP方法，比如GET、POST、PUT、DELETE等。