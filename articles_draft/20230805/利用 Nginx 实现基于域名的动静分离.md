
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　Nginx是一个开源的Web服务器/反向代理服务器及网页服务器。它也可以作为一个HTTP代理、负载均衡器、缓存服务器等。作为高性能的HTTP服务器，Nginx在很多网站提供静态页面的时候可以作为最好的选择。但是，当我们的项目中需要处理动态内容的时候，Nginx就无法满足需求了。于是在某种程度上，Nginx的静态资源处理能力被动了。因此，为了解决这个问题，我们引入了动静分离。
         # 2.动静分离(Static & Dynamic)
         　　什么叫做“静态”和“动态”？静态内容就是那些不经常发生变化的内容，比如图片、CSS文件、JavaScript脚本等等。这些内容可以预先生成后存储起来，当用户请求访问时，直接从硬盘读取并发送给用户，不需要执行复杂的程序或数据库查询，这样就可以极大的提升响应速度。而动态内容则相对来说更加难以处理，比如PHP、ASP、JSP等都是需要在每次访问时由程序生成内容并返回给用户的。

         　　为了实现基于域名的动静分离，我们需要做以下几步：

         （1）配置服务器：首先，我们要把服务器设置成支持多域名配置，并且配置好不同域名对应的目录（或者说虚拟路径）。
         （2）配置 Nginx 的 location 指令：然后，在 server 配置块里面定义多个 location，分别对应不同的 URL 模式，比如 /img/ 对应存放图片文件的目录，/html/ 对应存放 HTML 文件的目录，/php/ 对应存放 PHP 文件的目录等等。通过 location 指定不同的目录，我们就可以实现基于域名的动静分离。
         （3）配置重定向规则：最后一步，我们还需要添加一些重定向规则，比如 /images/ 可以重定向到 /img/ ，将所有访问“http://example.com/images/”的请求都重定向到“http://example.com/img/”，这样的话，我们就不会丢失任何的URL参数信息。

         # 3.原理及流程图


         # 4.操作步骤及注意事项

         ## 第一步：安装 Nginx

        ```
        sudo apt update && sudo apt install nginx -y
        ```

    　　　　　　
    　　安装成功后可以使用以下命令查看版本号：
    ```
    nginx -v
    ```
    ```
    nginx version: nginx/1.14.0 (Ubuntu)
    ```
       
     ## 第二步：配置 Nginx 以支持多域名配置

   ```
   sudo vim /etc/nginx/sites-enabled/default
   
   # 修改server_name 为你的域名，例如 www.example.com
   server {
       listen       80;
       server_name  example.com;

       root   /var/www/html;
       index  index.html index.htm;

       location / {
           try_files $uri $uri/ =404;
       }
   }
   ```

   此处的 `try_files` 指令用来指定哪个文件应该被尝试访问，`$uri` 表示客户端所请求的 URI，`=$uri/` 表示若找不到匹配的 URI 时，默认返回 `/index.html`。

   添加新域名只需复制以上配置，并修改 `listen` 和 `server_name`，再另行创建软链接到 `sites-enabled` 文件夹即可。

  ```
  sudo ln -s /etc/nginx/sites-available/example.com.conf /etc/nginx/sites-enabled/
  ```

   
## 第三步：配置 Nginx 的 location 指令

 ### a.配置静态文件存放位置

   创建文件夹用于存放静态文件，这里我以 `/var/www/html/img` 为例。

   ```
   sudo mkdir /var/www/html/img
   ```

 ### b.配置 PHP 文件存放位置

  在 `/var/www/html` 下创建一个名为 `php` 的文件夹，用于存放 PHP 文件。

  ```
  sudo mkdir /var/www/html/php
  ```

 ### c.配置不同 URL 模式的 location

   根据实际情况配置相应的 location。以下为配置示例。

   ```
   server {
      ...省略其他配置...
   
       location /img/ {
           alias /var/www/html/img/;
       }
   
       location ~ \.php$|\.php\?\S*$ {
           fastcgi_pass    unix:/run/php/php7.2-fpm.sock;
           fastcgi_index   index.php;
           include         fastcgi.conf;
           fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
       }
   }
   ```

   上述配置表明：

   * `/img/` 对应存放图片文件的目录，所以我们用 `alias` 指令将目录映射到 `/var/www/html/img/`。
   * `~ \.php$|\.php\?\S*$` 表示匹配所有以 `.php` 或带参数的`.php`结尾的 URI。
   * 将 PHP 文件通过 FastCGI 方式调用，并通过 socket 连接到 PHP-FPM 服务端。
   * 使用 `$document_root$fastcgi_script_name` 来指定脚本文件路径。

   **注意**：在 Windows 操作系统下，PHP-FPM 默认使用 `inetd` 模式，而非 `windows service` 模式，所以需要修改 PHP-FPM 配置文件 `/etc/php/7.2/fpm/pool.d/www.conf` 中的 `listen = 127.0.0.1:9000` 这一行，改为 `listen = npipe:///run/php/php7.2-fpm.sock`，即使用命名管道通信。

   
## 第四步：配置重定向规则

一般情况下，Apache 会自动从 `/images/` 重定向到 `/img/`，Nginx 也支持这种功能，可以在配置文件中添加如下条目。

```
rewrite ^/(.*)$ /$1 permanent;
```

此条目告诉 Nginx 每次收到客户端请求都会按当前配置进行重定向，将 `/images/` 替换成 `/img/`。`permanent` 参数表示永久重定向，即浏览器地址栏显示的是 `/images/`，但实际上会请求新的地址 `/img/`。

# 5.总结回顾

　　本文主要介绍了 Nginx 是如何实现基于域名的动静分离的，包括 Nginx 的配置文件的修改方法、location 指令的配置方法，以及通过重定向规则实现 URL 的重定向等内容。虽然不乏深度，但文章也算比较全面，可以帮助读者理解 Nginx 的相关知识。文章还有许多地方可以优化和完善，比如 Nginx 的其它模块的配置、使用场景、优缺点等，也欢迎读者补充补充。