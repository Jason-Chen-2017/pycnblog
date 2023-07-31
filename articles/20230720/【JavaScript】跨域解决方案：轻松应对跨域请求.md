
作者：禅与计算机程序设计艺术                    
                
                
在Web开发中，前端开发人员经常需要和后端开发人员协作完成业务需求。由于安全原因，浏览器同源策略限制了不同域名下的脚本的相互调用，也就是跨域请求（cross-origin request）。为了解决跨域问题，目前有多种跨域解决方案，比如通过JSONP、CORS或postMessage等方式，本文将介绍几种常用的跨域解决方案。下面我们就从不同的角度，逐一进行介绍。
# 2.基本概念术语说明
## （1）同源策略（Same-Origin Policy）
同源策略是一种约定，它是由 Netscape Navigator 的创始人 <NAME> 在 1995 年提出的，目的是防止一个站点可以访问另一个站点没有明确授权的资源。所谓同源指的是两个页面具有相同协议（HTTP or HTTPS），相同端口号，相同主机名。如果两个页面不满足同源条件，即便它们来自不同的域名，也会被禁止进行交互。

为了更好地理解同源策略，我们先来看以下示例：

1. https://www.example.com/pageA.html 和 http://www.example.com/dir/pageB.html

两者属于同源，因为它们都来自同一个根域名 example.com ，且使用相同的协议（HTTPS）和端口号（默认是443）。

2. https://www.example.com/pageA.html 和 http://sub.example.com/pageB.html

两者不属于同源，因为它们分别来自根域名 example.com 和子域名 sub.example.com 。协议不同，无法通信。

3. https://www.example.com/pageA.html 和 https://www.example.org/pageB.html

两者不属于同源，因为它们都来自不同根域名 example.com 和 example.org 。根域名不同，无法通信。

总结：

同源策略规定，AJAX 请求只能发送给同源服务器，否则会被拒绝。不同源服务器上的网页无法获取数据，除非用合适的接口获取数据并处理。

## （2）跨域请求
跨域请求是指浏览器的同源策略阻止的请求。也就是说，当前文档所在的域名和请求的 URL 域名不同。通常情况下，浏览器会限制 XMLHttpRequest、Fetch API 和 DOM 中的跨域请求。但是可以通过设置相关 HTTP 头信息来绕过同源策略。

什么是跨域？跨域是指两个或多个标签页之间的一种通信机制。如今，越来越多的应用需要处理跨域问题，所以各个浏览器厂商正在加强对跨域请求的支持。这里有一个著名的跨域攻击方式 —— JSONP 劫持漏洞（CVE-2014-4671），因此，在使用 JSONP 时，应当格外小心。此外，一些网站可能存在宽容策略，只允许其加载自己域名的资源，而不允许加载其他域的资源。因此，遇到类似的问题时，应该多做功课，了解哪些网站做出了宽容决定。

跨域请求的类型有三种：

- 简单请求（simple request）：该类型的请求允许 GET、HEAD 和 POST 请求方法，并且 Content-Type 首部字段的类型仅限于 application/x-www-form-urlencoded、multipart/form-data 或 text/plain。
- 预检请求（preflighted request）：该类型的请求要求先发送一个 OPTIONS 方法的请求到 CORS 预检，询问服务端是否允许跨域请求，具体的请求方法、自定义请求头等。
- Credentialed 请求（credentialed request）：该类型的请求包含用户凭证，例如 Cookie、HTTP 认证及 TLS Client Certificate，对于这些请求来说，浏览器必须使用特殊的机制才能发送它们。

简而言之，简单的请求就是不涉及第三方资源的请求，比如 GET 请求；复杂的请求包括预检请求和 credentialed 请求。

## （3）JSONP（跨域政策的一种选择）
JSONP (JSON with Padding) 是一种利用 script 标签的 src 属性，由网页操纵的一种手段，利用动态创建 script 标签的方式来实现跨域请求。它不是一种新的协议，只是利用现有的 HTTP 来传输 JavaScript 数据。

它利用一个回调函数，在响应的数据包中插入一段 JavaScript 函数调用语句。这样，网页就可以通过这个回调函数拿到相应的数据。

它的使用形式如下：

```javascript
// 创建script标签
var oScript = document.createElement("script");
// 设置src属性
oScript.src = "http://www.example.com:8080/api?callback=processData";
// 将script添加到body中
document.body.appendChild(oScript);

function processData(json){
    // 对json数据进行处理
}
```

JSONP 使用简单且易于实现，但是它有一个限制，只能发起 GET 请求。而且，这种方式只能从别的域名处获取数据，不能用来发送请求。

## （4）CORS（跨域资源共享）
CORS（Cross-Origin Resource Sharing，跨域资源共享）是 W3C 推荐的跨域解决方案，属于跨域请求的一类。它允许浏览器和服务器之间对通信进行负责的协商，实现跨域访问控制。CORS 通过检查 Access-Control-* 头信息来判断是否允许跨域请求。浏览器会自动处理 CORS 请求，不需要任何代码帮助。

在服务器上配置 CORS 之后，浏览器会把请求预检发给目标服务器，询问是否允许跨域请求。如果允许，则发起实际的请求；如果拒绝，则返回错误响应。如果 CORS 配置错误或者缺少必要的 Access-Control-* 头信息，则会出现跨域失败。

CORS 使用起来比较复杂，一般来说，后端接口要启用 CORS 支持，需要做以下配置：

- 服务端在响应头中加入 Access-Control-Allow-Origin。
- 如果要携带 cookie，那么还需要在响应头中加入 Access-Control-Allow-Credentials。
- 服务端响应时，可以使用自定义的响应头。

例如，下面的代码演示了一个最简单的 CORS 配置：

```python
@app.route('/test', methods=['POST'])
def test():
    # 获取客户端请求参数
    param = request.get_json()

    # 执行业务逻辑，假设只有 id 为 1 的数据可以访问
    if param['id'] == '1':
        data = {'name': 'John'}
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
        }
        return Response(json.dumps(data), headers=headers)
    else:
        abort(403)
```

## （5）postMessage（跨窗口通信）
postMessage 是 HTML5 中新增加的一个接口，用于不同窗口间的数据传递。它提供了一种安全的、跨源的方法，来确保不同窗口间的数据通信的完整性。

postMessage 接收两个参数：消息内容和目标窗口（可以为空字符串表示向所有窗口发送消息）。postMessage 会返回一个布尔值表示是否成功投递消息。

```javascript
window.opener.postMessage('Hello World!', '*');
```

在父窗口中监听 message 事件，接收消息并执行相应的代码：

```javascript
if (event.source!= window) {
  var jsonObj = event.data;

  // 此处省略对数据的处理代码...
}
```

postMessage 只能通过 DOM 编程来实现通信，既然 HTML5 提供了 postMessage，为什么还有很多人使用 Flash？主要原因在于 Flash 可以做很多事情，但它运行在一个独立的进程中，虽然不受同源策略的限制，但它并没有完全沙盒环境，可以访问完整的系统资源。并且，Flash 的兼容性较差，不同的浏览器对其行为可能会有所不同。所以，除非真的需要，尽量不要依赖 Flash 来实现跨窗口通信。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## （1）JSONP 跨域请求实现过程
JSONP 跨域请求的实现过程分为如下四步：

1. 创建一个<script>元素；
2. 设置<script>元素的src属性，值为跨域地址；
3. 当<script>元素的src跨域地址下载完成后，会触发onreadystatechange事件；
4. 通过传入的回调函数参数，解析JSON数据。

具体流程如下图所示：
![image](https://user-images.githubusercontent.com/50874703/125874923-d9b0c1ae-a1b5-44ea-ab0f-cdfe100db9dd.png)

举例：
假设前端要请求 https://www.example.com/api?callback=successCallback ，并获得相应的 JSON 数据：

1. 浏览器首先会创建一个<script>元素。
2. 设置<script>元素的src属性为 https://www.example.com/api?callback=successCallback 。
3. 当<script>元素的src跨域地址下载完成后，会触发onreadystatechange事件，执行指定的successCallback回调函数。
4. successCallback 函数的参数，将接收到服务器传回的 JSON 数据。

## （2）CORS 跨域请求实现过程
CORS 跨域请求的实现过程分为如下七步：

1. 检查是否有凭据（Cookie、HTTP 认证或 TLS 客户端证书）；
2. 检查请求是否符合简单请求或预检请求的条件；
3. 发起实际的请求；
4. 生成响应头；
5. 返回响应；
6. 浏览器检查响应头中的 Access-Control-Allow-Origin 是否正确；
7. 根据不同的状态码，显示不同的提示信息。

具体流程如下图所示：
![image](https://user-images.githubusercontent.com/50874703/125875236-0cf60e7f-b4da-4ad3-bbaa-d776a3cb5c25.png)

## （3）postMessage 跨域请求实现过程
postMessage 跨域请求的实现过程分为如下五步：

1. 新建一个iframe窗口，并设置为透明背景色。
2. 想要访问的目标窗口的文档写入如下的代码，来监听message事件：

   ```javascript
   window.addEventListener("message", function(event) {
     console.log(event.data);
   }, false);
   ```

3. 判断是否是在同源的情况，如果是，直接调用目标窗口的postMessage()方法。如果不是，那么需要使用JSONP的方法来实现跨域请求。
4. 把数据序列化成字符串，并通过传递给目标窗口的window.postMessage()方法。
5. 如果目标窗口允许接收消息，那么它就会收到一个message事件，通过event.data可以获取传递过来的数据。

具体流程如下图所示：
![image](https://user-images.githubusercontent.com/50874703/125875389-7ebfcdf7-b0e1-4cc3-9e12-2b2ce6de58e8.png)

## （4）CORS 与 JSONP 的区别
### 1. 定义不同
CORS（Cross-Origin Resource Sharing，跨域资源共享）是一种基于标准的网络技术，由Netscape公司1995年设计，W3C采纳并成为WEB标准。

JSONP（JSON with Padding，使用填充字符进行封装的数据格式），也叫JSON with Script。它是一种服务器加载数据的新型格式。它的优点是跨域请求能比Ajax更为方便，缺点是服务器必须支持JSONP。

### 2. 目的不同
CORS与JSONP的区别是，CORS是一种标准协议，JSONP是一种自定义的协议。

CORS旨在解决跨域请求问题，JSONP旨在解决跨域脚本加载问题。

### 3. 实现效果不同
对于JSONP，它不会像XMLHttpRequest一样发送一个完整的HTTP请求，而是利用script标签的src属性指向一个不同源服务器上的数据，从而可以在不同源之间传递数据。

而对于CORS，它是通过XMLHttpRequest向服务器发出HTTP OPTIONS请求来验证服务器是否允许跨域请求。如果允许，则会在HTTP请求中增加特定的请求头来实现跨域请求。

