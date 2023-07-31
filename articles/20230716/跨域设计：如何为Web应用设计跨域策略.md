
作者：禅与计算机程序设计艺术                    
                
                
当今互联网时代背景下，跨域问题成为了Web开发者的一大难题。跨域问题起源于不同域之间JavaScript脚本不能相互通信的问题，简单来说，跨域问题就是两个不同域名、不同端口的页面之间如何进行数据交换的问题。本文将从以下几个方面对Web跨域问题进行阐述和分析:

1.Web跨域问题由来
Web应用程序在浏览器中运行时，它所在的域名和协议都是一个独立的空间。同样，如果多个Web页面或Web应用需要共享相同的cookie、存储空间等资源，就会产生访问限制（Cross-Origin Resource Sharing）。由于不同源之间的资源只能通过特定的机制进行通信，因此就出现了跨域问题。典型的场景如：两个Web站点中的一个页面需要访问另一个站点中的某些资源文件、图片等，因为两者不同源，所以就涉及到跨域问题。

2.Web跨域方式分类
目前，Web应用程序一般采用两种不同的跨域方案：

1)JSONP跨域方案
JSONP(JSON with Padding)跨域方案主要用于解决主动请求的跨域问题，可以利用script标签src属性发送GET请求并将返回的数据作为参数传入指定的回调函数。这种方案的缺点是只能发送GET请求，不能发送POST请求，且发送的数据量受限于URL长度。但是其优点是在不影响用户体验的情况下实现了跨域访问，且实现起来较为容易。

2)CORS跨域方案
CORS全称是"Cross-origin resource sharing"（跨越来源资源分享），它是W3C制定的一种跨域访问标准。它允许服务器端设置Access-Control-Allow-Origin头部，以告知客户端是否允许其他域的请求，从而使得不同源下的Web应用程序可以共享数据、cookie、以及一些操作权限。该规范兼容所有现代浏览器，并可与任何类型的HTTP请求一起使用。目前，主流浏览器均已支持该规范，因此，CORS可以更好地保护Web应用程序的安全。

本文所要讨论的CORS跨域设计，属于第二种跨域方式。基于CORS设计的Web应用程序可以更灵活地处理跨域问题，包括以下几类需求:

1）GET、POST、PUT、DELETE请求类型
CORS设计旨在支持多种HTTP请求类型，包括GET、POST、PUT、DELETE等，但实际上只有GET和HEAD请求会被视为简单请求，其他请求都会被认为是复杂请求，需要遵循CORS协议。除此之外，为了提高安全性，CORS规定复杂请求必须遵循HTTP缓存机制，也就是说，对于复杂请求，浏览器需要先发送预检请求（OPTIONS方法），询问服务器是否允许该跨域请求，服务器若同意，则再返回响应信息；否则，服务器会拒绝该跨域请求，并返回一个错误响应。

2）自定义请求头
CORS允许客户端（比如浏览器）设置自定义的请求头，服务器可以通过相应的响应头控制是否允许这些自定义请求头。

3）响应结果过滤
有时候，服务器端需要根据请求头或者其他条件对响应结果进行过滤，这时候就可以用到preflight response。preflight response是指，服务器端在收到预检请求后，必须先进行服务器验证，然后才能确定是否可以返回指定的内容。

4）Cookie跨域传送
CORS允许在请求和响应过程中携带cookie，这样服务器就可以为不同源提供服务。但是需要注意的是，不同域下的cookie在不同浏览器中也存在一些限制，具体细节参阅RFC6265。另外，如果需要在跨域请求中发送cookie，可以在请求头中添加withCredentials选项，表示允许携带cookie。

5）凭证管理
CORS还提供了一种机制，可以让服务器通过HTTP认证（Basic、Digest、NTLM等）来验证客户端身份，也可以用于管理跨域请求所用的授权凭证。

6）WebSockets跨域通信
CORS也可以用于支持WebSockets跨域通信，只需配置服务器端和客户端的WebSocket连接即可，不需要额外的代码或配置项。

综合以上六条，Web应用程序应当考虑设计如下的跨域策略：

## 1）服务器端实现
### （1）CORS支持
服务器端必须确保其API符合HTTP CORS规范，并在请求中包含必要的响应头部字段，包括Access-Control-Allow-Origin、Access-Control-Allow-Methods、Access-Control-Max-Age等。

### （2）身份验证
服务器端可以选择性地支持基于HTTP认证的跨域访问，例如：Basic、Digest、NTLM等。

### （3）预检请求
对于复杂请求（非GET/HEAD请求），服务器应该先进行一次预检请求，询问浏览器是否允许跨域请求，如果允许的话才会继续发送真正的请求。预检请求的方法是OPTIONS，并且要求携带以下请求头：

Access-Control-Request-Method: HTTP方法（如GET/POST等）
Access-Control-Request-Headers: 请求头列表（如Content-Type、Authorization等）

服务器接收到预检请求后，需要检查请求头中是否包含Origin字段，以及Access-Control-Request-Method、Access-Control-Request-Headers是否有效，如果有效则返回200 OK状态码，并添加以下响应头：

Access-Control-Allow-Origin: 指定的来源域名（可以是*，表示任意来源）
Access-Control-Allow-Methods: 支持的HTTP方法列表（如GET/POST等）
Access-Control-Max-Age: 预检请求的最大存活时间（单位秒），建议设置为1800秒
Access-Control-Allow-Headers: 支持的自定义请求头列表
Access-Control-Expose-Headers: 需要暴露给客户端的响应头列表（通常用于浏览器调试）

浏览器收到预检请求的响应后，根据响应头中的相关信息决定是否发送真正的请求。

### （4）响应结果过滤
服务器端可以使用各种过滤规则对跨域请求的响应结果进行过滤，如：禁止特定域名的跨域请求、修改返回结果的大小写、去掉敏感信息等。

### （5）Cookie跨域传送
服务器端可以在请求中携带cookie，但必须正确设置响应头中的Set-Cookie字段，并指定Domain属性，以便让客户端接受这些cookie。

### （6）凭证管理
服务器端可以使用授权凭证管理机制来控制跨域请求使用的授权凭证，例如：OAuth2.0授权机制。

## 2）前端实现
### （1）AJAX请求
前端应该通过AJAX请求向服务器端发送跨域请求。

### （2）请求头设置
前端需要在Ajax请求头中设置Origin字段，值为当前Web应用所在的域名，这是CORS协议要求的。

### （3）预检请求
对于复杂请求，前端在发送真正请求之前，需要先发送预检请求，询问服务器是否允许跨域请求，如果允许的话，才会发送真正的请求。可以借助XMLHttpRequest对象的setRequestHeader()方法设置Access-Control-Request-Method、Access-Control-Request-Headers请求头。预检请求返回的响应中也包含预检请求成功后的响应头，前端可以根据该响应头设置后续真正请求的相关选项。

### （4）Cookie跨域传送
由于跨域请求不能携带cookie，因此需要在浏览器端设置特殊的跨域策略，比如：通过代理服务器等。

### （5）自定义响应头
服务器端可以通过响应头向前端传递一些必要的信息，前端可以在相应的响应头中获取这些信息，然后作出相应的处理。

