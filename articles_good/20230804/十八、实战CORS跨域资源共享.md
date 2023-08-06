
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 CORS（Cross-Origin Resource Sharing）跨域资源共享（Cross-origin resource sharing），是一种浏览器提供的一种机制，它允许一个网页从不同源（域名、协议、端口号等不同的源）获取资源，只要服务器端设置正确的响应头信息就可以实现这个功能。
         如果没有实现CORS，则当两个不相关的域的网站需要通信时，就存在跨域请求的问题，即一个域下的脚本向另一个域下的数据请求时，由于缺少必要的验证措施，就会导致数据被拦截。因此在实际开发中，一般都会考虑实现CORS，以解决这个跨域问题。本文将带领读者全面掌握CORS的知识体系及其运作方式。
         # 2.CORS跨域资源共享的基本概念
         ## 2.1 什么是CORS？
         CORS（Cross-Origin Resource Sharing）跨域资源共享（Cross-origin resource sharing），是一种基于HTTP标准的跨域协议。该协议允许浏览器向跨源服务器发送Ajax请求，从而克服了同源策略（SOP，Same Origin Policy)限制。

         ### 为什么需要CORS？
         目前的浏览器都采用同源策略来禁止Javascript脚本直接访问属于其他域的内容，否则会造成安全漏洞和信息泄露等问题。例如，A网站向B网站提交Ajax请求，如果两者不是同一域的话，就会遭到浏览器的拒绝。为了解决这个限制，W3C提出了CORS（跨域资源共享）标准，使得不同源之间的AJAX请求成为可能。

         ### 为什么需要用到XHR?
         早期的CORS并没有考虑到如何支持非简单请求（Non-simple requests）。也就是说，对于那些既有预检请求又有真正的XHR请求的复杂场景，CORS并不能很好地工作。比如PUT方法，对于上传文件来说是个不错的例子。因此，后续的版本也添加了对一些更复杂的方法的支持，如PUT、DELETE、OPTIONS、HEAD等。

         ### 浏览器对于CORS的支持情况
         根据Can I use CORS上述标准的最新版本，可以看到，IE9+，Firefox 4+，Chrome 4+和Opera 12+都已经支持了CORS。其它浏览器还处于试验阶段。另外，对于可以设置CORS的资源，服务端需要做好相应的配置才可以开启。

         
         # 3.CORS的工作流程
         当浏览器发起一个跨域AJAX请求的时候，首先会询问目标服务器是否允许跨域请求，如果允许，那么浏览器会发送两个HTTP请求：首先会发送一个“预检”请求，该请求是OPTION方法的，目的是判断目标服务器是否允许跨域请求；然后再发送真正的XHR请求。整个过程如下图所示：
         
                 |-----|                            Client                      |----------|
           Request from domain A          -->                                    Server
                                         (with Origin header)
                                  /|\ 
                                       |
                CROSS ORIGIN     ______________________________> [PREFLIGHT]
            REQUEST        /|\                                       OPTIONS
                                      |                                   Response from server
                                          |----|                                <----|
                                            <-|                                -|->
                                                    RESPONSE OK
                                    with Access-Control-* headers
                          |<-------->|
                            XHR request sent from domain B

            请求方式：

                     Request from Browser --->

                        OPTION Method
                        XHR Method
                      
               在这个过程中，如果服务器端接收到了预检请求并且返回了允许跨域请求的响应头，那么就正式发送了XHR请求。注意：虽然浏览器自身也会做一些关于CORS的优化，但是由于网络传输的特性，还是有可能出现某些情况下失败的情况。

         # 4.CORS的配置项
         ## 4.1 Access-Control-Allow-Credentials
         如果你把Access-Control-Allow-Credentials设为true，那么浏览器在发起CORS请求时，就不会自己 include cookies、authorization headers或者TLS客户端证书信息（即身份认证信息）了，而是让服务器决定是否允许带上这些信息。默认情况下，Access-Control-Allow-Credentials的值是false，表示不发送cookies、auth headers或TLS client certs。设置此值时，需要注意以下几点：

             * 只能在浏览器端使用，所以服务器不能解析它
             * 不能用于本站资源的请求
             * 设置了Access-Control-Allow-Credentials之后，如果后端的response没有Access-Control-Expose-Headers，则该header不会被暴露给前端

     
        ## 4.2 Access-Control-Allow-Headers
        如果要发送自定义HTTP请求头，就需要在XHR对象的setRequestHeader()方法调用之前，先使用setRequsetHeader()方法指定它们，否则请求头中的自定义头部信息将被忽略。如果要发送Cookie，Authorization 或 TLS客户端证书信息，这些信息也需要由服务器设置一下，具体怎么设置都要看服务端的接口规范。如果你想使用这些信息，可以在XHR对象的setRequestHeader()方法之前设置Access-Control-Allow-Headers属性。举例：

        ```javascript
        var xhr = new XMLHttpRequest();
        xhr.open('GET', 'http://somewhere.com/api');
        xhr.onload = function(){ /* do something */ };
        // Add custom headers to request before setting it in the open() method
        xhr.setRequestHeader("X-Custom-Header", "value");
        xhr.send(null);
        ```

        如果服务端设置了Access-Control-Allow-Headers: "X-Custom-Header"，则浏览器可以将"X-Custom-Header"加入到请求头中一起发送。你可以通过XHR对象的getAllResponseHeaders()方法获取服务器返回的所有headers，然后再根据需要筛选这些headers。

        
        ## 4.3 Access-Control-Allow-Methods
        如果请求的方法（method）不在这个列表里，则返回一个405错误。你可以通过设置Access-Control-Allow-Methods来指定哪些HTTP方法可以使用。
        
        ## 4.4 Access-Control-Max-Age
        指定在preflighted请求后的多少秒内，浏览器无须重新发出完整的预检请求即可进行跨域请求。默认情况下，这个值为60s。
        
        ## 4.5 Access-Control-Expose-Headers
        如果浏览器发现服务器返回了一个指定的header，而你没有在Access-Control-Expose-Headers属性里面指定它，那么默认情况下，这个header就无法被JavaScript代码读取到，除非服务器明确指出，希望浏览器这样做。

        # 5.实际操作演练
        ## 5.1 服务端配置
        以NodeJS为例，假设你有一个服务器，作用是处理API请求，你需要实现三个API：注册用户，登录用户，更新用户信息。分别对应的路由如下：

        ```javascript
        app.post('/register', registerUserHandler);
        app.post('/login', loginUserHandler);
        app.put('/updateUserInfo/:id', updateUserInfoHandler);
        ```

        每个API需要校验用户输入的参数，包括用户名、密码等，保证它们的合法性。为了防止CSRF攻击，还需在请求头（headers）中增加一个token字段，用来验证当前请求是否由合法的用户发起。另外，你还需要保证不同域的请求不会被误伤。

        服务端需要做以下几件事情：

        1. 安装cors模块，用于处理CORS中间件
        2. 配置路由，将所有需要校验参数的API设置成需要验证来源的权限，并在相应的响应头中设置Access-Control-Allow-Origin属性。下面给出更新的注册和登录路由的代码：

           ```javascript
           const cors = require('cors');
           const {verifyToken} = require('../utils/token');

           // 初始化中间件
           app.use(cors()); // 注入中间件

           // 注册用户
           router.post('/register', verifyToken(), async (req, res) => {
             try{
               await userService.createUser(req.body);
               res.json({message:'user created successfully'});
             } catch(err){
               console.error(err);
               res.status(500).end();
             }
           });

           // 登录用户
           router.post('/login', async (req, res) => {
             try{
               const token = await userService.authenticateUser(req.body);
               if (!token) return res.status(401).end();
               res.cookie('jwt', token, { httpOnly: true })
                .json({ message: `Login successful`});
             }catch(err){
               console.error(err);
               res.status(500).end();
             }
           });
           ```
        3. 配置路由，将所有允许跨域的API设置成允许特定来源的请求，并在相应的响应头中设置Access-Control-Allow-Origin属性。下面给出更新的用户信息更新路由的代码：

           ```javascript
           const cors = require('cors');

           // 初始化中间件
           app.use(cors()); // 注入中间件

           // 更新用户信息
           router.put('/updateUserInfo/:id', async (req, res) => {
              try{
                const userToUpdate = req.params.id;
                const updatedInfo = req.body;
                
                // 这里应该做一些验证和处理逻辑

                res.json({message:`updated ${userToUpdate}'s information`});
              } catch(err){
                console.error(err);
                res.status(500).end();
              }
            });
           ```

    ## 5.2 浏览器端配置
    浏览器端需要做以下几件事情：

    1. 使用XHR对象发起请求
    2. 判断是否跨域，若是，则发送预检请求
    3. 将正确的请求方法（GET、POST、PUT、DELETE等）设置在xhr.open()方法中
    4. 向xhr.setRequestHeader()方法传入自定义请求头
    5. 获取服务器的响应头，检查是否存在Access-Control-Allow-Origin，如果有，则设置xhr.withCredentials为true。否则，抛出错误。
    
    下面给出具体的代码：

    ```html
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <title>XHR Demo</title>
      </head>
      <body>
        <script>
          const url = 'http://localhost:3000';

          const xhr = new XMLHttpRequest();
          xhr.onreadystatechange = function () {
            if (this.readyState === 4 && this.status === 200) {
              document.getElementById('response').innerHTML = this.responseText;
            } else if (this.readyState === 4) {
              alert(`Error ${this.status}: ${this.statusText}`);
            }
          };

          // Check if cross origin request
          if (/^.+?:\/\//.test(url)) {
            const matches = /^(\w+:)\/\/([^:\/?#]+)(:\d*)?/.exec(url);
            if (matches[1]!== window.location.protocol || matches[2]!== window.location.host) {
              // Send preflight request first

              xhr.open('OPTIONS', url, true);
              xhr.setRequestHeader('Content-Type', 'application/json');
              xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');

              xhr.onreadystatechange = function () {
                if (this.readyState === 4) {
                  let accessAllowed = false;

                  // Check response status and get allowed methods for current route
                  const responseStatus = parseInt(this.status / 100);

                  if (responseStatus === 2 && this.getResponseHeader('access-control-allow-methods')) {
                    const allowMethodsStr = this.getResponseHeader('access-control-allow-methods');
                    const allowedMethods = allowMethodsStr.split(',');

                    switch (window.location.pathname) {
                      case '/register':
                        accessAllowed = ['POST'].includes(allowedMethods.toUpperCase());
                        break;
                      case '/login':
                        accessAllowed = ['POST'].includes(allowedMethods.toUpperCase());
                        break;
                      case '/updateUserInfo':
                        accessAllowed = ['PUT'].includes(allowedMethods.toUpperCase());
                        break;
                      default:
                        accessAllowed = false;
                    }
                  }

                  if (accessAllowed) {
                    xhr.open('PUT', `${url}/updateUserInfo/${userId}`, true);
                    xhr.withCredentials = true;
                    xhr.setRequestHeader('Accept', 'application/json');
                    xhr.send(JSON.stringify(data));
                  } else {
                    throw new Error('Cross origin request not allowed');
                  }
                }
              };

              xhr.send();
            }
          } else {
            xhr.open('GET', `${url}/users`, true);
            xhr.send();
          }
        </script>

        <div id='response'></div>
      </body>
    </html>
    ```