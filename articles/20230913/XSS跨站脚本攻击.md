
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
XSS(Cross Site Scripting)跨站脚本攻击，通常指的是恶意攻击者将恶意脚本注入到网页上，通过网站对其输出的正常网页内容和功能进行控制。
XSS漏洞往往伴随着严重的信息泄露、业务中断甚至攻击其他系统等严重后果。由于缺少对输入数据的过滤和验证，攻击者可利用XSS攻击将攻击者的恶意指令植入到网页上。攻击者可插入恶意脚本、HTML标签甚至是Flash动画，从而篡改页面内容并执行攻击者的任意操作。
目前已知的XSS攻击手法有三种：存储型XSS、反射型XSS和DOM型XSS。其中存储型XSS存在于Web表单，当用户输入的数据被存储在服务器端而不经过任何处理直接呈现时，就会发生存储型XSS攻击。反射型XSS则存在于URL参数、Cookie等前端传参中，其攻击方式更加隐蔽，只要攻击者能够进入目标网站，就可能触发XSS攻击。DOM型XSS是在JavaScript代码中插入恶意指令，通过修改页面结构达到恶意攻击目的。
## XSS攻击类型及防御策略
XSS攻击类型及防御策略总结如下表所示:
|类型|危害性|防御措施|
|---|---|---|
|存储型XSS|数据被存储在服务器端而不经过任何处理直接呈现时，可能会造成严重的安全隐患。|对于Web表单中的用户输入，必须进行转义或清除特殊字符；对于数据库中存储的用户输入，也必须进行合适的安全措施。<br>同时，服务器端也需要对存储在数据库中的数据进行有效的过滤，阻止恶意提交的攻击。|
|反射型XSS|XSS攻击可以利用恶意脚本，植入到链接、url地址、参数等多方面。<br>可以窃取用户信息、网站敏感信息、盗取用户cookie。|对于输入型、文本型、下拉列表框等易受攻击的场景，必须采用参数化查询或者白名单过滤机制。<br>采用验证码等增强验证机制。<br>|
|DOM型XSS|通过在前端代码中插入恶意指令，注入到服务器端，实现网页的控制，甚至篡改用户浏览行为。<br>特别是一些浏览器插件，例如Adobe Flash、Google Chrome等都容易受到DOM型XSS攻击。|尽量不要在前端代码中动态生成HTML代码，避免由恶意代码直接导致的DOM污染。<br>避免在DOM中写入属性值，以减少浏览器自动执行XSS攻击的可能性。<br>如果确实需要渲染前端变量，建议采用模板引擎或者其他技术来提升渲染效率和安全性。<br>|
## 浏览器安全配置
虽然XSS攻击已经成为网络安全领域的“热点事件”，但仍然是一种比较复杂的攻击形式。本文将主要讨论基于Web应用程序的XSS攻击，而非浏览器自身的攻击，因此以下讨论均基于实际案例。首先，本文将讨论常见的Web开发框架和工具，如Node.js、Ruby on Rails等，它们是否存在XSS漏洞，如何保护它们？接着，将介绍客户端浏览器中的XSS防御措施，如CSP（Content Security Policy）、SOP（Same Origin Policy）、X-Frame-Options等。最后，将讨论常用编程语言中对于XSS防御的支持情况，如PHP、Java、Python等。
# 2.基本概念术语说明
## DOM (Document Object Model)
文档对象模型（Document Object Model），缩写为DOM，是W3C组织推荐的处理可扩展置标语言的标准编程接口。它定义了用于处理和构造XML以及HTML文档的标准对象和函数。

通过DOM，Web开发人员可以通过脚本来操纵各种元素、属性和样式，并响应浏览器事件。DOM Level 1规范定义了访问HTML和XML文档的基础方法，包括获取元素、创建元素、删除元素、添加/修改/删除属性、样式设置、事件处理等。

DOM Level 2（又称Core DOM 2级规范）提供了更多的方法，如遍历文档树、事件监听器、自定义事件、范围（Range）、XMLHttpRequest、样式变化监控等。DOM Level 2规范是当前最新的版本，于2007年发布。

DOM Level 3（又称XML Core DOM 3级规范）增加了XPath和其他新特性，可用于处理XML文档。DOM Level 3规范也是W3C推荐标准，于2011年发布。

## BOM (Browser Object Model)
浏览器对象模型（Browser Object Model）是一个由不同浏览器供应商实现的规范，它为JavaScript（通常简称JS）提供了统一的窗口和文档对象，使得开发人员可以为不同的浏览器提供一致的用户体验。BOM定义了一系列描述Web浏览器窗口的对象，这些对象允许JavaScript与浏览器进行互动，控制浏览器显示的内容和行为，以及与用户交互。

除了Window对象外，BOM还定义了navigator对象、history对象、location对象、screen对象、console对象等。

## URL
Uniform Resource Locator （统一资源定位符）是一个用来标识互联网上的资源的字符串。它由五个部分组成：协议、域名、端口号、路径、参数。

URL语法格式如下：

```
protocol://username:password@host:port/path?query#fragment
```

协议部分：http/https/ftp/file/mailto等。
用户名、密码：访问该资源需要的凭证信息，可以省略。
主机名：网站的域名或IP地址。
端口号：访问该资源时使用的TCP/UDP端口号，可以省略，默认为http协议的默认端口80。
路径：指定服务器上的文件路径，如：index.html。
参数：传递给服务器的参数信息。
片段：浏览器定位到的位置，即锚点标记。

## CGI (Common Gateway Interface)
CGI 是一类计算机程序，它接收客户端发送的请求并产生相应的回应信息。它与HTTP服务器及其相关的应用软件配合工作，运行在服务器端。CGI 可以使用各种脚本语言编写，如 Perl、Python、C、Visual Basic 等。

## SSRF (Server Side Request Forgery)
服务端请求伪造（英语：Server-Side Request Forgery，简称SSRF），也叫做SSRF攻击，是一种由攻击者构造一个特殊的网址欺骗服务器执行恶意命令的攻击。这种攻击利用计算机网络的漏洞，冒充合法的身份，向内网或Internet上指定的网站发送一个请求。攻击者可以在这个请求里指定一个URL，指向他自己的服务器，然后让服务器发送一个请求到这个指定URL，将自己在同一个局域网内的敏感数据（如Cookie、Session ID等）或信任境外的第三方数据发送出去，从而获得访问权限、登录凭证、内部网络资源等。

## XSS Filter Evasion Techniques
为了躲避XSS检测，攻击者会采用各种手段来规避检测。下面列举几个常用的XSS过滤绕过方法：

1. Encoding：编码（编码是把不可显示的字符转换成可显示的字符的一个过程）。比如在Javascript代码中使用document.write()把数据插入HTML页面的时候，可以使用escape()函数进行编码。

2. Whitelisting：白名单过滤。通过白名单的方式，只允许特定字符和标签出现在用户输入的地方。比如在textarea中只允许特定标签，这样就可以防止用户输入非法的JavaScript代码。

3. Attribute Proxy：属性代理。通过隐藏的标签或属性，把用户输入的内容传送给后端的处理程序。比如在input标签上添加onblur事件，把用户输入的内容透传给后端服务器，再由后端服务器完成处理。

4. HTML Entity Encoding：实体编码。把不可显示的字符转换成实体字符。比如把"<"转换成"&lt;"，">'"转换成"&gt;'。

5. Style Sheets：使用CSS隐藏XSS攻击。比如使用display:none;样式隐藏攻击代码。