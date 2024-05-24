
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         ## 一、背景介绍
         
         对于Web开发者来说，SSL（Secure Sockets Layer，安全套接层）证书是一种经过认证的数字证书，可以确保网站提供的服务是加密且安全的。一般情况下，当用户打开访问网站时，浏览器会首先检查网站的域名是否与服务器的域名匹配，如果不匹配则提示警告信息，用户确认后才可继续访问；若域名与服务器名匹配，则浏览器会向服务器发出请求，通过建立连接、发送握手协议并进行证书验证，确认双方身份后，就可以传输数据。整个过程是通过SSL加密进行的，通信双方需要通过一定规则进行协商，交换相关的证书文件等，最后才能实现真正的通信。本文将从Nginx配置中详细讲述HTTPS/SSL证书的建立流程。
         
         ## 二、基本概念术语说明
         
         ### 1.SSL

         Secure Sockets Layer（SSL）是一种网络安全传输协议，它主要用来加密网页之间的通信，确保互联网上的信息安全。目前SSL由IETF(Internet Engineering Task Force)标准化，其目的是为互联网通信提供一个安全通道。使用SSL协议需要有一个CA机构颁发的数字证书，只有在安装了正确的证书的浏览器上，才能实现网站的正常访问。此外，还可以使用不同版本的SSL协议，比如TLS(Transport Layer Security)，即传输层安全性协议，它是SSL的更新版本，具有更高级的加密功能。
         
         ### 2.CA（Certificate Authority）

         CA即证书授权中心，是一个颁发证书的权威机构，负责对申请者进行身份核实，核实通过后再颁发证书。通常CA都设有颁发证书的CA证书，所有用户都应该信任这个根证书颁发机构。由于CA机构拥有颁发证书的绝对权力，任何滥用的行为都会带来严重的法律风险。所以，只有得到足够的授权才可以发放证书。
         
         ### 3.CSR（Certificate Signing Request）

         CSR（证书签名请求），是一个由用户提交给CA的申请表格，用于向CA证明自己身份及要求的服务器的相关信息。通过CSR生成证书之后，CA会签署证书并把它返回给用户。
         
         ### 4.OCSP（Online Certificate Status Protocol）

         OCSP（在线证书状态协议）是一种可以让客户端查询证书是否被吊销或失效的协议。它利用CA的分布式数据库来获取证书的有效期、吊销信息等。使用OCSP能够避免证书被伪造的问题。
         
         ### 5.SNI（Server Name Indication）

         SNI（Server Name Indication）是指在TLS协议中增加了一个扩展字段，该扩展字段允许客户端在建立TLS连接的时候，向服务器传递自己的域名，从而服务器能够区分多个虚拟主机，实现HTTPS功能。
         
         ### 6.DH（Diffie-Hellman Key Exchange）

         DH是一种密钥协商算法，基于整数的计算，它可以使得两端之间在密钥交换阶段建立临时的共享密码，相比RSA算法更加安全，更适合用于SSL/TLS协议。
         
         ### 7.ECDHE（Elliptic Curve Diffie-Hellman Ephemeral）

         ECDHE是ECDH的一种变体，它可以在不分享私钥的情况下进行密钥交换，相比DH算法更加快速并且安全，可以应用于高速的数据传输场景。
         
         ### 8.DH(E)（DHE or DHM）

         这是一种密钥协商算法，包括Diffie-Hellman和Elliptic Curve。Diffie-Hellman是一种密钥协商算法，基于整数的计算，可以使得两端之间在密钥交换阶段建立临时的共享密码，相比RSA算法更加安全，更适合用于SSL/TLS协议；Elliptic Curve是在ECDLP上添加了一项密钥交换阶段。在TLS1.2之前，DHE可以选择四种模式，分别是传统模式、独立模式、保留模式、和不安全模式。后续版本增加了DHE_PSK、ECDHE_PSK等新模式。
         
         ### 9.ECDSA（Elliptic Curve Digital Signature Algorithm）

         是一种基于椭圆曲线的数字签名算法，采用了ECC曲线群的参数作为私钥。可以保证私钥无法被第三方伪造，同时具有抗抵赖能力，并降低攻击成本。在TLS1.2之前，ECDSA只能用于签名证书，不能用于密钥协商。
         
         ### 10.TLS（Transport Layer Security）

         TLS是一种安全传输协议，由IETF(Internet Engineering Task Force)标准化，它的作用主要是为了实现两个应用程序之间的安全通信，即“端到端”安全。主要功能包括身份认证、数据完整性、加密和数据隐私保护。TLS1.3版本兼顾速度和安全性。TLS是SSL的升级版，TLS1.0～TLS1.2依然占据主导地位，但随着云计算、物联网、边缘计算、移动终端等新兴领域的兴起，越来越多的设备或系统不具备SSL证书，或者安装SSL证书的浏览器插件较旧，导致无法实现安全通信，因此TLS应运而生。
         
         ### 11.CN（Common Name）

         CN是X.509证书的公用名称属性，它代表了所签发证书的主体名称，该名称必须是唯一标识。
         
         ### 12.SAN（Subject Alternative Names）

         SAN是可选的 Subject Alternative Names 属性，它可以包含多个 DNS 或 IP 地址，表示同一证书可以用于多个域名或IP地址。
         
         ### 13.IP（Internet Protocol）

         IP是TCP/IP协议族中的网络层协议，它负责实现计算机之间的网络通信，它把数据包从源地址到目的地址传送。
         
         ### 14.URL（Uniform Resource Locator）

         URL（统一资源定位符）是通过因特网访问资源的路径，也称为网址或网页地址。
         
         ### 15.HTTPS（HyperText Transfer Protocol Secure）

         HTTPS（超文本传输安全协议）是以安全套接层（SSL或TLS）运行的HTTP协议，也就是说，HTTPS协议是由HTTP协议和SSL/TLS协议组合而成的可靠协议。通过它实现全站加密，保障个人信息安全。
         
         ### 16.CRT（Certificate Revocation and Management Tool）

         CRT（证书吊销与管理工具）是Windows系统下管理CA证书的工具。
         
         ### 17.PEM（Privacy Enhanced Mail）

         PEM（Privacy Enhanced Mail）是电子邮件的一种存储格式，它将编码的证书文件以ASCII码形式保存，以".pem"或".crt"为扩展名。
         
         ### 18.PFX（Personal Information Exchange File）

         PFX（个人信息交换文件）是微软IIS服务器的证书格式，它是包含证书私钥和证书链的文件，可以通过导入IIS服务器进行安装。
         
         ### 19.Apache HTTP Server

          Apache HTTP Server 是开源、免费、跨平台的HTTP Web服务器，它支持CGI（Common Gateway Interface）、FastCGI、SSI（服务器端嵌入）、SCGI（SCTP（Stream Control Transmission Protocol，流控制传输协议）Application Programming Interface，流控制传输层应用程序接口）。通过模块化的结构，Apache HTTP Server 可快速部署和管理各种Web站点。它还支持IPv6，包括SPDY协议，是最流行的Web服务器之一。
          
         ### 20.OpenSSL

          OpenSSL 是目前最流行的TLS/SSL协议库，它提供了丰富的命令行工具及编程接口，可用于实现各种安全协议，如TLS、DTLS、PKI、CA等。
          
         ### 21.Nginx

          Nginx 是一款高性能、轻量级的Web服务器和反向代理服务器，同时也是一款面向异步事件驱动的HTTP和反向代理服务器，也可以用作负载均衡器。Nginx是基于BSD许可协议进行分发的自由软件。

          ## 三、核心算法原理和具体操作步骤以及数学公式讲解

          1. 配置服务器证书

             在配置文件nginx.conf中设置ssl相关参数，其中常见的配置如下：
             
             ```
                ssl on;                      #开启ssl模块

                ssl_certificate /etc/nginx/cert/xxx.com.crt;      #指定证书位置

                ssl_certificate_key /etc/nginx/cert/xxx.com.key;   #指定证书对应的私钥位置

                ssl_session_timeout 5m;              #设置缓存时间，默认10分钟

                ssl_protocols TLSv1 TLSv1.1 TLSv1.2;     #指定协议版本，默认全部支持

                ssl_ciphers "HIGH:!aNULL:!MD5";        #指定加密方式
            ```

             

          2. 定义server块

            server块配置定义网站的虚拟主机，如监听端口、域名和位置：

             ```
                server {

                    listen      443;    #监听https请求

                    server_name www.example.com;  #绑定域名

                    root /var/www/html;  #网站根目录 

                    index index.html index.htm;   #默认主页文件

                }
            ```

  


           3. 配置域名和证书绑定关系

            在Nginx的配置文件中，我们需要配置域名和证书的对应关系，即证书文件(.crt)与域名的绑定关系。要实现这一功能，我们需要用到指令 `ssl_certificate` 和 `ssl_certificate_key`，它们分别指定证书文件的位置和对应的私钥文件的位置：

             ```
                 server {

                        listen      443;

                        server_name www.example.com;

                        root /var/www/html;

                        index index.html index.htm;

                        ssl on;

                        ssl_certificate /path/to/your/domain.crt;         

                        ssl_certificate_key /path/to/your/private.key;  

                 }
            ```

             上面的示例中，证书文件 `domain.crt` 和私钥文件 `private.key` 放在 `/path/to/` 下，相应的，域名 `www.example.com` 将与这些文件绑定。当然，可以根据实际情况配置证书文件和私钥文件的位置，Nginx会自动搜索它们。

          

          4. 请求证书

             当用户访问服务器时，他/她的浏览器首先会向服务器的443端口发出请求，然后，服务器就会检查用户请求的域名是否与证书绑定，如果匹配，则返回对应的证书文件；否则，就提示警告信息，并拒绝连接。因此，在浏览器中，还需做一些额外配置，以便用户接受证书。

             - Chrome 浏览器

                在Chrome浏览器中，需要先在地址栏输入 `chrome://flags/#allow-insecure-localhost`，然后找到 Allow invalid certificates for resources loaded from localhost 的选项，设置为启用。这样，浏览器就会接收来自本地服务器的HTTPS响应，因为它们没有绑定有效的证书。

             - Firefox 浏览器

                在Firefox浏览器中，需要点击菜单栏中的首选项 > 内容 > 网站权限 > 显示高级链接内容，勾选 “载入未知来源的图片”，这样，浏览器就会接收来自本地服务器的HTTPS响应，因为它们没有绑定有效的证书。

             - Safari 浏览器

                在Safari浏览器中，暂无配置方法。

             - Edge 浏览器

                在Edge浏览器中，需要先访问 `edge://settings/privacy`，然后，在 “Cookies and website data” 标签中，将 “阻止网站保存并查看cookie和其他网站数据” 设置为关闭，这样，浏览器就会接收来自本地服务器的HTTPS响应，因为它们没有绑定有效的证书。

          5. 建立连接

             用户的浏览器收到证书后，就会检查证书的有效期、颁发者、签名哈希值等信息，如果发现有问题，则显示警告信息，并拒绝连接；如果证书有效，则生成加密套件，并发送ClientHello消息，开始SSL/TLS连接。

             ClientHello消息中包含客户端支持的加密算法、压缩方法、随机数等信息，它会通知服务端，在接下来的SSL/TLS握手过程中，客户端希望使用的算法、密码套件和加密模式。

          6. 握手协商

             服务端收到ClientHello消息后，会产生一个新的随机数，并将它和客户端支持的加密套件一同发送给客户端。客户端收到之后，会和服务端的随机数进行比较，如果相同，则进入SSL/TLS连接；否则，显示警告信息，并拒绝连接。

             此时，客户端和服务端都拥有了相同的加密算法和密码套件，可以开始密钥交换过程。

          7. 生成密钥

             密钥交换过程会发生在服务端和客户端之间，服务端和客户端都各自产生一组随机数，并把自己的随机数发给对方，由另一方产生相同的随机数，两者握手完成之后，就生成一个相同的对称密钥。

          8. 传输数据

             SSL/TLS连接建立成功之后，就可以进行数据的传输。在传输数据时，首先会对数据进行加密，然后再传输。加密算法可以参考RFC文档，如AES、DES、3DES、RC4等。

          9. 断开连接

             当客户端和服务端的SSL/TLS连接已经完成，就可以断开连接了。断开连接的方式分为两种：主动断开连接和被动断开连接。主动断开连接是指客户端直接关闭连接，例如，用户关闭浏览器窗口；被动断开连接是指服务端超时、客户端主动关闭连接，例如，超时后，客户端关闭连接。

          

          10. OCSP（在线证书状态协议）

              如果服务器的证书是通过CA机构签发的，那么在建立SSL连接时，客户端还可以向CA机构发出证书吊销列表（CRL）查询请求，以判断证书是否被吊销，以防止证书泄露、恶意冒充等安全漏洞。但是，这种CRL查询请求可能会失败，因为CRL需要周期性更新。为了减少CRL查询请求失败导致连接失败的概率，可以启用OCSP功能。OCSP功能依赖于HTTP协议，通过GET方式向CA机构的OCSP服务器发出查询请求。如果CA机构的OCSP服务器能够响应查询请求，则证明证书有效，否则证明证书无效。

              在Nginx的配置文件中，我们可以配置OCSP功能，启用OCSP功能只需要在 `listen` 和 `server_name` 指令后面加上以下两条指令即可：

               ```
                   ocsp on;
                   ssl_stapling on;
               ```

              在 `ocsp on;` 指令启用OCSP功能，在 `ssl_stapling on;` 指令启用证书状态缓存机制。在Nginx启动时，会通过HTTP连接下载服务器的证书吊销列表，并缓存至文件中，在之后的连接中，客户端会通过缓存的OCSP响应来验证证书状态。

              通过OCSP验证后的证书状态缓存会在30天内保存，因此第一次验证之后，会在很短的时间内消除延迟，之后的连接验证则不需要再次连接到CA机构。

          

          11. 使用ECDHE（Elliptic Curve Diffie-Hellman Ephemeral）

              在TLS1.2之前，DHE可以选择四种模式，分别是传统模式、独立模式、保留模式、和不安全模式。在TLS1.2及以上版本，DHE除了默认的四种模式外，还新增了DHE_PSK、ECDHE_PSK、DHE_RSA1_2、EDH_RSA_DES_CBC3_SHA等五种模式。其中，DHE_PSK和ECDHE_PSK是为了支持基于PSK的密钥协商模式，而DHE_RSA1_2和EDH_RSA_DES_CBC3_SHA都是为了兼容旧版本。

              比较常用的就是默认的模式，即DHE_RSA_WITH_AES_128_GCM_SHA256。这种模式使用的是DHE密钥交换，即基于Diffie-Hellman的密钥协商算法，后面跟着的AES_128_GCM表示该算法使用AES_128加密算法，并且采用GCM模式。

              如果服务器配置了ECDHE模式，那么客户端和服务端都会使用ECDHE密钥交换算法，而非DHE算法。ECDHE算法支持的加密算法有两种类型：椭圆曲线加密（ECDHE-ECDSA）和椭圆曲线DH密钥交换（ECDHE-RSA）。对于较新的客户端（例如TLS1.2），可以使用ECDHE-ECDSA算法，这种算法使用的是椭圆曲线加密，速度更快；对于较老的客户端（例如TLS1.0~TLS1.1），可以使用ECDHE-RSA算法，这种算法使用的是椭圆曲线DH密钥交换，但是速度慢些。

              Ngnix的配置文件中，使用ECDHE密钥交换模式的配置如下：

                  ```
                      ssl_ecdh_curve secp384r1;             #指定椭圆曲线，默认值为prime256v1

                      ssl_prefer_server_ciphers on;           #优先选择服务端的加密套件

                      ssl_protocols TLSv1.2 TLSv1.1 TLSv1;     #指定协议版本

                      ssl_ciphers EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH;  #指定加密算法
                  ```

                  

              在上面的示例中，我们指定了ECDHE密钥交换使用的椭圆曲线为secp384r1，并通过指令 `ssl_prefer_server_ciphers on;` 指定优先选择服务端的加密套件，这样可以保证客户端的兼容性。通过指令 `ssl_ciphers` 指定了客户端和服务端的加密算法。ECDHE算法的优点是可以快速地生成密钥，缺点是对密钥长度有限制。

              需要注意的是，椭圆曲线加密算法虽然在某些场景下会比RSA算法更快，但是，它的适用性也有限。所以，只有在某些特定场景下，才会考虑使用它，比如，密钥交换、签名、数字摘要等等。

          ## 四、具体代码实例和解释说明

          1. 配置Nginx

            Nginx的配置文件 nginx.conf 可以放在不同的地方，例如 /etc/nginx/nginx.conf ，或者 /usr/local/nginx/conf/nginx.conf 。需要修改的文件包括：

            - 配置ssl_certificate和ssl_certificate_key指令

            - 配置server块
            
            - 配置域名和证书对应关系
            
            - 配置OCSP功能

            - 配置ECDHE密钥交换算法
              
          2. 配置ssl_certificate和ssl_certificate_key指令

              配置ssl_certificate和ssl_certificate_key指令需要指定证书文件所在位置和私钥文件所在位置。

              ```
                    server {
                       ...
                        
                            ssl on;

                            ssl_certificate /path/to/your/domain.crt; 
                            ssl_certificate_key /path/to/your/private.key; 
                            
                    }
              ```

          3. 配置server块

              配置server块需要指定监听端口，并配置域名和根目录。

              ```
                    server {
                        listen       443 default_server ssl http2;   #监听https请求，http2协议

                        server_name  yoursite.com www.yoursite.com;    #绑定域名

                        access_log logs/access.log combined;            #日志存放路径

                        error_log  logs/error.log;                      #错误日志存放路径

                        root   /home/project/public;                   #网站根目录

                        index  index.php index.html index.htm;          #默认主页文件
                    }
              ```

              默认情况下，nginx会使用http和https协议。如果只需要http协议，只需要将listen指令中的http和https注释掉即可。

          4. 配置域名和证书对应关系

              配置域名和证书对应关系需要在server块中使用指令 `ssl_certificate` 和 `ssl_certificate_key`。

              ```
                    server {
                       ...
                            
                            ssl on;

                            ssl_certificate /path/to/your/domain.crt;      #指定证书文件位置

                           ssl_certificate_key /path/to/your/private.key; #指定私钥文件位置

                     }
              ```

          5. 配置OCSP功能

              配置OCSP功能需要在server块中设置 `ocsp on;` 和 `ssl_stapling on;` 两个指令。

              ```
                    server {
                       ...
                        
                          ssl on;

                          ssl_certificate /path/to/your/domain.crt;  
                         
                          ssl_certificate_key /path/to/your/private.key; 

                          ocsp on;

                          ssl_stapling on;
                     }
              ```

          6. 配置ECDHE密钥交换算法

              配置ECDHE密钥交换算法需要在server块中设置 `ssl_ecdh_curve` 指令，指定使用的椭圆曲线。

              ```
                    server {
                       ...
                        
                            ssl_ecdh_curve secp384r1;               #指定椭圆曲线

                            ssl_prefer_server_ciphers on;          #优先选择服务端的加密套件

                            ssl_protocols TLSv1.2 TLSv1.1 TLSv1;    #指定协议版本

                            ssl_ciphers EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH;   #指定加密算法
                     }
              ```

              配置完毕后，服务器就可以正常工作了。

          ## 五、未来发展趋势与挑战

          1. 其他证书格式

              当前，Nginx仅支持PEM格式的证书。不过，为了更好的支持其他证书格式，OpenSSL提供了一个叫做 `c_rehash` 的工具，它可以帮助管理员自动创建符号链接，将证书文件映射到网站根目录下的不同路径。

              ```
                   $./openssl c_rehash /path/to/certs
                   $ ls /path/to/certs
                  ...
                   ca.crt  example.com.crt ->../../../../../etc/letsencrypt/live/example.com/fullchain.pem
                   ca.key  example.com.key ->../../../../../etc/letsencrypt/live/example.com/privkey.pem
              ```

              可以看到，除了PEM格式的证书文件，c_rehash工具还创建了符号链接，指向网站根目录下的letsencrypt目录下的live目录。这样就可以使用其他证书格式，比如PKCS#12格式的证书。

          2. 不安全的通信方式

              HTTPS使用SSL/TLS协议加密通信，虽然提供了对数据完整性的保护，但还是存在一些安全隐患。其中，主要有以下几种：

              - 伪装与篡改

                  由于HTTPS通信过程需要双向验证，中间人攻击者可以拦截双方的通信内容并进行篡改，这样，HTTPS加密通信就完全失去意义了。

              - 中间人攻击

                  由于通信使用了加密，通信双方要想建立连接就需要经过数十甚至上百个路由节点，攻击者可以窃听、修改通信内容，甚至篡改通信过程，获得通信内容的完整性。

              - 数据泄露

                  普通用户的所有敏感信息都应该使用HTTPS加密传输，但是，如果用户的浏览器设置不当，可能会把一些用户敏感的信息（例如，登录凭证）提交到服务器端，从而暴露这些信息。

              为解决这些问题，应该在TLS协议层面上增加更多的安全措施，比如Session Ticket、Forward Secrecy、Padding Oracle Attack等。另外，在应用层面上还应该引入更多的安全防护措施，比如验证码、滑动验证、二步认证等。

          ## 六、附录常见问题与解答

          1. Q：什么是CA？

             A：CA，即Certificate Authority，证书颁发机构，是一个权威机构，对申请者的身份进行认证，审核通过后颁发相应的数字证书。证书包括两部分：公钥和签名。公钥加密的内容只能用对应的私钥才能解密，而签名是证明内容的身份。CA通过认证，是确保通信双方身份的重要方法。

          2. Q：证书验证有哪些方式？

              A：常用的验证方式有两种：

　　　　　　　－ 验证机构(CA)

               　证书签名认证机构CA，是网络通信发展史上第一个认证的中心，主要职责就是验证数字证书的合法性、有效性，并为用户颁发数字证书。CA认证包含两个步骤：一是实体认证，即验证申请者（即网站或应用）的身份；二是服务认证，即验证申请者提供的服务是否符合服务条款。证书验证的结果主要取决于CA的审核级别和CA自身的信誉度。

               　验证机构认证的过程比较复杂，实体认证和服务认证都需要公共的信息，如CA的数字签名、CA根证书、CA机构的认证材料等。验证机构认证的速度比较慢，而且容易受到黑客攻击。

　　　　　　　－ 域名(DNS)

               　域名验证，通过域名系统解析服务验证网站域名的真实性。域名验证的步骤简单：

　　　　　　　　① 检查申请者域名的DNS记录是否指向申请者的服务器。

　　　　　　　　② 检查申请者服务器的SSL证书是否由CA机构签发。

               　域名验证的缺陷是无法检查证书中的私钥是否泄露。如果私钥泄露，证书也不会验证通过。

               　域名验证方式存在一个潜在的风险，即域名劫持攻击。域名劫持攻击是指攻击者通过域名劫持的方式，篡改网站的证书，或者插入恶意内容等，进一步恶意影响用户的访问。

          3. Q：为什么HTTPS仍然是网络世界上最安全的协议？

              A：HTTPS是建立在TLS/SSL协议基础之上的，它可以保障信息的安全，即在网络上传输的数据都是经过加密的。但是，HTTPS并不是一种免费的安全协议，它使用复杂的加密技术和认证机制，对计算机系统的资源消耗也比较大。

              HTTPS是一种非常成熟的技术，已经经历了多年的普及，并且得到了广泛的认可。它还依赖于其他协议的配合，比如DNSSEC、HTTP Strict Transport Security (HSTS)。因此，在HTTPS出现之前，也有很多安全的网站依旧使用HTTP协议。

          4. Q：证书的有效期是如何确定？

              A：证书的有效期是一个非常重要的问题。不同CA颁发的证书的有效期一般都不同。一般来说，对于个人使用的证书，有两种有效期策略：一是根据用户的使用频率自动延长证书有效期；二是手动指定证书的有效期。

              根据用户的使用频率自动延长证书有效期的方法，用户每次访问网站都可以从CA处重新下载新的证书。这种方式存在一个缺陷，就是用户可能不知道自己的证书是否已经过期，导致网站出现异常，出现访问故障。

              手动指定证书的有效期方法，要求用户提前指定好证书的有效期，然后在指定日期之前尽快安装。这种方式的弊端是用户需要不断的提醒自己续订证书，并在指定的日期之前完成安装。

              目前，大多数的浏览器都提供了一个显示证书到期日的功能，方便用户及时查看证书的有效期。

          5. Q：PEM文件和DER文件有什么区别？

              A：PEM（Privacy Enhanced Mail）和DER（Distinguished Encoding Rules）是两种不同的数据编码格式。PEM是纯文本格式，DER是二进制编码格式。PEM文件通常以".pem"或".crt"结尾，DER文件通常以".cer"或".der"结尾。

              PEM文件和DER文件都可以包含公钥、私钥、证书等信息，但是PEM文件通常采用纯文本格式，容易受到攻击；DER文件通常采用二进制格式，安全性更高。

              在实际应用中，PEM文件主要用于SSL/TLS证书的存储，DER文件则用于其他场合。