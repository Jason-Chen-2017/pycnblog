
作者：禅与计算机程序设计艺术                    

# 1.简介
  

负载均衡（Load Balancing）是一种应用层的网络流量分担技术，通过将入站连接通过不同的服务器节点进行分配，从而达到优化系统资源利用率、提高网站响应速度、节省服务器成本等目的。一般来说，负载均衡可以实现以下三个主要功能：
1) 解决单点故障：当某个服务器节点出现故障时，其他服务器节点可以自动承担相应的流量，避免因为单点故障而导致整个服务不可用。

2) 提高服务器处理能力：通过将请求分布到多台服务器上，可以有效提高服务器的处理能力，提高网站的吞吐量。

3) 增加可靠性：在负载均衡器中加入冗余设备，可以提高服务器的可用性。

Nginx是一款免费开源的Web服务器/反向代理服务器，也是目前最热门的四层负载均衡器。它支持高度可扩展性和可靠性，是一个高性能、高可用的Web服务器。许多知名网站都在使用Nginx作为其负载均衡器。对于刚开始学习负载均衡的人来说，了解Nginx的工作原理以及如何安装、配置、管理、使用非常重要。
# 2. 基本概念术语说明
## 2.1 Nginx简介
Nginx 是一款自由及开源的 HTTP Web 服务器、反向代理服务器及电子邮件代理服务器。由俄罗斯程序员 <NAME> 博士开发，供俄国大型搜索引擎 Yandex 使用。其特色是占有内存少、稳定性高、并发能力强、高度模块化设计。

Nginx 以其高并发、高性能、高可扩展性闻名于世，是现今广泛使用的反向代理服务器、负载均衡器及缓存服务器。很多大型网站都在使用 NGINX 来作为自己的 Web 服务端。

## 2.2 Nginx 组成结构
Nginx 的架构由三大部分组成，分别是 event 模块、http 模块和 mail 模块。

- Event 模块：event 模块是 Nginx 内部的事件驱动机制，它负责对外界传来的各种请求进行响应。包括连接请求、读写事件、信号通知等。

- Http 模块：http 模块用于处理客户端到服务器端的 HTTP 请求。主要完成的内容包括 URI 解析、协议升级、缓存处理、日志记录、压缩传输、KeepAlive 支持、状态监测、限速访问控制、配置热更新、安全防护、访问控制等。

- Mail 模块：mail 模块用于处理客户端到服务器端的 POP3/IMAP 通信。包括接收和发送邮件、过滤、存储等。

Nginx 中还有一些独立的模块，如 ngx_lua、ngx_stream 等，它们可以通过配置加载到 Nginx 中来提供额外的功能，例如集成 Lua 语言环境、支持 TCP 和 UDP 流量转发。

## 2.3 Nginx 配置文件
Nginx 的配置文件是 nginx.conf 文件，默认情况下，该文件存放在 /etc/nginx/目录下，文件中包含了许多可以设置的参数。下面详细介绍一下该配置文件的各项参数。

### 2.3.1 events {} 指令
events {} 指令用来设定 Nginx 对外提供服务所依赖的事件模型。目前 Nginx 支持的事件模型有两种：epoll（Linux 下默认模型）和 kqueue（FreeBSD、Solaris 下默认模型）。

```bash
events {
    # 使用的事件模型，支持 epoll 和 kqueue 
    use [kqueue|epoll];

    # 每个 worker process 可以同时处理的最大连接数
    worker_connections 1024;
}
```

### 2.3.2 http {} 指令
http {} 指令是 Nginx 的核心模块，定义了 Nginx 服务器的全局配置信息。其中，include 可引入其它配置文件，使得一个配置文件可以包含多个 include 段。

```bash
http {
    # 设置缓冲区大小，单位为字节，建议设置为 8K 或以上
    client_body_buffer_size 8k;
    
    # 设置请求头的大小，单位为字节
    large_client_header_buffers 4 32k;
    
    # 请求超时时间，单位为秒
    keepalive_timeout 75s;
    
    # 指定日志路径，开启日志功能
    log_path /var/log/nginx/access.log;
    
    # 指定错误日志路径
    error_log /var/log/nginx/error.log info;
    
    # 是否开启连接超时
    sendfile on;
    
    # tcp_nopush on;
    
    # server_tokens off;
    
    # 禁止显示 Server 头部信息
    server_tokens off;
    
    # fastcgi_cache_path /data/nginx/fastcgi_cache levels=1:2 keys_zone=my_cache:10m inactive=60m;
    
    include enable-php.conf;
    include vhosts/*.conf;
}
```

#### 2.3.2.1 accept_mutex
accept_mutex 参数决定是否允许一个 worker process 接受新的连接，如果设为 on ，则只有 accept_mutex 为 off 的 worker process 可以接受新的连接。如果设为 off ，则所有 worker process 都可以接受新的连接，并且会受到限制。

```bash
http {
   ...
    accept_mutex on|off;
   ...
}
```

#### 2.3.2.2 aio
aio 参数指定异步 I/O 模式，它可以提升服务器的并发性能。

```bash
http {
   ...
    aio on|off;
   ...
}
```

#### 2.3.2.3 aio_write
aio_write 参数可以指定 Nginx 在向客户端返回数据时采用异步 I/O 方式，这样就可以更快地返回数据。

```bash
http {
   ...
    aio_write on|off;
   ...
}
```

#### 2.3.2.4 auth_basic
auth_basic 参数用于设定 HTTP 基础认证，它要求客户端提供用户名密码才能访问某些区域。

```bash
   root html;
   autoindex on;
   allow all;

   auth_basic "Restricted";
   auth_basic_user_file conf/htpasswd;
}
```

#### 2.3.2.5 buffer_size
buffer_size 参数用于设置读取 client request body 时一次读取的最大字节数。

```bash
http {
   ...
    server {
        listen       80 default_server;

        location /upload {
            client_max_body_size  10M;
            upload_tmp_dir /tmp/;
            upload_store /app/upload;
            upload_pass_form_field "";

            set $limit_rate '';
            if ($request_method = POST) {
                set $limit_rate 'limit_rate=$upload_rate';
            }

            proxy_pass      http://upstream/upload;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header Host $host:$server_port;
            proxy_set_header Content-Type $content_type;
            proxy_set_header Content-Length $content_length;
            proxy_set_header REMOTE-HOST $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto https;
            proxy_connect_timeout   90;
            proxy_send_timeout      90;
            proxy_read_timeout      90;
            send_timeout            60;
            $limit_rate;
        }
    }
   ...
}
```

#### 2.3.2.6 charset
charset 参数用于设置页面字符编码类型。

```bash
http {
   ...
    server {
        listen       80;

        index index.html index.htm;

        charset utf-8;

        location / {
           root  html;
           autoindex on;
           alias www;
        }
    }
   ...
}
```

#### 2.3.2.7 client_body_buffer_size
client_body_buffer_size 参数用于设置请求主体缓冲区大小，默认值为 8k。

```bash
http {
   ...
    client_body_buffer_size size;
   ...
}
```

#### 2.3.2.8 client_body_in_file_only
client_body_in_file_only 参数用于设置客户端请求主体是否缓存在磁盘上的临时文件里，默认为 off 。

```bash
http {
   ...
    client_body_in_file_only clean|on|off;
   ...
}
```

#### 2.3.2.9 client_body_temp_path
client_body_temp_path 参数用于设置客户端请求主体存储的临时文件的位置，默认为 client_body_temp。

```bash
http {
   ...
    client_body_temp_path path;
   ...
}
```

#### 2.3.2.10 connection_pool_size
connection_pool_size 参数用于设置每个 worker process 的连接池大小，默认为 64 。

```bash
http {
   ...
    connection_pool_size size;
   ...
}
```

#### 2.3.2.11 disable_symlinks
disable_symlinks 参数用于设置是否检查虚拟主机中的符号链接。

```bash
http {
   ...
    disable_symlinks if_not_owner|always;
   ...
}
```

#### 2.3.2.12 error_page
error_page 参数用于设置自定义的错误页，语法如下：

```bash
error_page code... uri;
```

code 表示错误码，可以是数字或者星号，表示匹配任意错误码；... 表示可选的参数，可以是任何东西；uri 表示错误页面的 URL 地址。

```bash
http {
   ...
    server {
        listen 80;
        
        error_page 404 http://example.com/errors/404.html;
        error_page 500 502 503 504 http://example.com/errors/default.html;
        
        location / {
            return 301 https://www.google.com;
        }
    }
   ...
}
```

#### 2.3.2.13 etag
etag 参数用于设置生成 etag 的方法，可以是“on”或“off”，默认为“on”。

```bash
http {
   ...
    etag on|off;
   ...
}
```

#### 2.3.2.14 geo
geo 参数用于根据指定的地理位置来选择相应的服务器进行负载均衡，该参数的值是一个嵌套的正则表达式定义的变量字典，键值对之间通过空格分隔。

```bash
http {
   ...
    geo $geoip_country country A1 B2 C3 D4 E5 F6 G7;
   ...
}
```

#### 2.3.2.15 gzip
gzip 参数用于启用/禁用基于 zlib 的 gzip 压缩输出。

```bash
http {
   ...
    gzip on|off;
   ...
}
```

#### 2.3.2.16 gzip_buffers
gzip_buffers 参数用于设置压缩过程中使用的缓冲区大小。

```bash
http {
   ...
    gzip_buffers number size;
   ...
}
```

#### 2.3.2.17 gzip_comp_level
gzip_comp_level 参数用于设置压缩级别，范围是 1 到 9，默认为 1 。

```bash
http {
   ...
    gzip_comp_level level;
   ...
}
```

#### 2.3.2.18 gzip_disable
gzip_disable 参数用于禁止基于特定上下文的压缩输出，可以是 “browser” （针对浏览器），“MSIE6” （针对 MSIE6），“MSIE7” （针对 MSIE7），“MSIE8” （针对 MSIE8），“MSIE9” （针对 MSIE9），“mobile” （针对移动端设备），“none” （不进行压缩），“all” （禁止所有的压缩）等。

```bash
http {
   ...
    gzip_disable none|browser|MSIE6|MSIE7|MSIE8|MSIE9|mobile|all;
   ...
}
```

#### 2.3.2.19 gzip_http_version
gzip_http_version 参数用于设置响应数据的压缩版本，可以是“1.0”或“1.1”，默认为“1.1”。

```bash
http {
   ...
    gzip_http_version version;
   ...
}
```

#### 2.3.2.20 gzip_min_length
gzip_min_length 参数用于设置进行压缩的数据最小长度，单位为字节，默认为 0 。

```bash
http {
   ...
    gzip_min_length length;
   ...
}
```

#### 2.3.2.21 gzip_proxied
gzip_proxied 参数用于指定哪些类型的请求需要进行压缩。可以是“any”、“expired”、“no-cache”、“no-store”、“private”、“no_last_modified”、“no_etag”、“auth”、“degradation”、“ratio”、“memcached”、“postgresql”、“gunzip”、“br”、“off”或按域划分的白名单。

```bash
http {
   ...
    gzip_proxied any|expired|no-cache|no-store|private|no_last_modified|no_etag|auth|degradation|ratio|memcached|postgresql|gunzip|br|off|domain=example.com;
   ...
}
```

#### 2.3.2.22 gzip_types
gzip_types 参数用于指定压缩的 MIME 类型，默认只压缩 text/plain 和 text/css 。

```bash
http {
   ...
    gzip_types type1 type2...;
   ...
}
```

#### 2.3.2.23 gzip_vary
gzip_vary 参数用于指定响应是否要根据 Accept-Encoding 中的 qvalue 进行vary，默认为“on”。

```bash
http {
   ...
    gzip_vary on|off;
   ...
}
```

#### 2.3.2.24 http2_idle_timeout
http2_idle_timeout 参数用于设置 HTTP/2 连接空闲超时时间，单位为秒，默认为 30 分钟。

```bash
http {
   ...
    http2_idle_timeout time;
   ...
}
```

#### 2.3.2.25 ignore_invalid_headers
ignore_invalid_headers 参数用于忽略无效的请求头字段，比如那些名称或格式错误的头字段，默认为“off”。

```bash
http {
   ...
    ignore_invalid_headers on|off;
   ...
}
```

#### 2.3.2.26 keepalive_disable
keepalive_disable 参数用于禁用长连接，默认为“off”。

```bash
http {
   ...
    keepalive_disable on|off;
   ...
}
```

#### 2.3.2.27 keepalive_requests
keepalive_requests 参数用于设置浏览器为保持活动状态而需要发出的请求次数。

```bash
http {
   ...
    keepalive_requests number;
   ...
}
```

#### 2.3.2.28 keepalive_timeout
keepalive_timeout 参数用于设置超时时间，单位为秒。

```bash
http {
   ...
    keepalive_timeout timeout;
   ...
}
```

#### 2.3.2.29 limit_conn
limit_conn 参数用于限制每个客户端IP地址的并发连接数量。

```bash
http {
   ...
    limit_conn address zone connections;
   ...
}
```

#### 2.3.2.30 limit_conn_log_level
limit_conn_log_level 参数用于设置触发日志记录的日志级别。

```bash
http {
   ...
    limit_conn_log_level level;
   ...
}
```

#### 2.3.2.31 limit_conn_status
limit_conn_status 参数用于设置超过限制时的返回状态码。

```bash
http {
   ...
    limit_conn_status code;
   ...
}
```

#### 2.3.2.32 limit_except
limit_except 参数用于限制除指定位置外的所有 URL 请求的并发连接数量。

```bash
    limit_except GET {
        deny all;
    }
}
```

#### 2.3.2.33 limit_rate
limit_rate 参数用于限制连接速率，单位为 bytes/second。

```bash
http {
   ...
    server {
        listen          80 default_server;
        server_name     example.com;
    
        access_log      logs/example.access.log;
        error_log       logs/example.error.log;
        
        location /download {
            root            /usr/share/nginx/html;
            add_header      Cache-Control private;
            
            limit_rate      1024k;
            types           application/octet-stream;
            default_type    application/octet-stream;
            expires         1h;
        }
    }
   ...
}
```

#### 2.3.2.34 lingering_close
lingering_close 参数用于设置 TCP 连接关闭时，主动关闭前等待的时间，默认为“on”。

```bash
http {
   ...
    lingering_close on|off;
   ...
}
```

#### 2.3.2.35 lingering_time
lingering_time 参数用于设置 TCP 连接关闭时，维护连接的时间，单位为秒，默认为“30s”。

```bash
http {
   ...
    lingering_time time;
   ...
}
```

#### 2.3.2.36 lingering_timeout
lingering_timeout 参数用于设置 TCP 连接关闭时，等待 FIN 包的时间，单位为秒，默认为“5s”。

```bash
http {
   ...
    lingering_timeout timeout;
   ...
}
```

#### 2.3.2.37 log_format
log_format 参数用于定义日志格式。

```bash
http {
   ...
    log_format myformat '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';
   ...
}
```

#### 2.3.2.38 max_ranges
max_ranges 参数用于设置可以在一个请求中指定多少个 Range 字段。

```bash
http {
   ...
    max_ranges ranges;
   ...
}
```

#### 2.3.2.39 merge_slashes
merge_slashes 参数用于合并多个斜杠为一个斜杠。

```bash
http {
   ...
    merge_slashes on|off;
   ...
}
```

#### 2.3.2.40 msie_padding
msie_padding 参数用于在 MSIE 6 和 IE 6 中添加 padding 以解决图片抖动的问题。

```bash
http {
   ...
    msie_padding on|off;
   ...
}
```

#### 2.3.2.41 msie_refresh
msie_refresh 参数用于在 MSIE 中添加 Refresh 报头以实现页面刷新。

```bash
http {
   ...
    msie_refresh on|off;
   ...
}
```

#### 2.3.2.42 open_file_cache
open_file_cache 参数用于打开文件缓存。

```bash
http {
   ...
    open_file_cache cache_size=number inactive=time;
   ...
}
```

#### 2.3.2.43 open_file_cache_errors
open_file_cache_errors 参数用于设置打开文件缓存出错时的行为，可以设置为“off”或“debug”。

```bash
http {
   ...
    open_file_cache_errors on|off|debug;
   ...
}
```

#### 2.3.2.44 open_file_cache_min_uses
open_file_cache_min_uses 参数用于设置打开文件缓存的最小使用次数。

```bash
http {
   ...
    open_file_cache_min_uses min_uses;
   ...
}
```

#### 2.3.2.45 open_file_cache_valid
open_file_cache_valid 参数用于设置打开文件缓存的过期时间，单位为秒。

```bash
http {
   ...
    open_file_cache_valid time;
   ...
}
```

#### 2.3.2.46 output_buffers
output_buffers 参数用于设置用于存储响应的缓冲区数量和大小。

```bash
http {
   ...
    output_buffers number size;
   ...
}
```

#### 2.3.2.47 port_in_redirect
port_in_redirect 参数用于设置重定向时，是否应该包含端口号。

```bash
http {
   ...
    port_in_redirect on|off;
   ...
}
```

#### 2.3.2.48 postpone_output
postpone_output 参数用于延迟输出，直到收到输入完整的请求体。

```bash
http {
   ...
    postpone_output on|off;
   ...
}
```

#### 2.3.2.49 read_ahead
read_ahead 参数用于预读一些数据，并缓存起来，后面可以直接使用，提高效率。

```bash
http {
   ...
    read_ahead size;
   ...
}
```

#### 2.3.2.50 recursive_error_pages
recursive_error_pages 参数用于递归显示错误页。

```bash
http {
   ...
    recursive_error_pages on|off;
   ...
}
```

#### 2.3.2.51 request_pool_size
request_pool_size 参数用于设置每个进程可以同时处理的请求数。

```bash
http {
   ...
    request_pool_size size;
   ...
}
```

#### 2.3.2.52 reset_timedout_connection
reset_timedout_connection 参数用于设置连接超时时，是否重新建立连接。

```bash
http {
   ...
    reset_timedout_connection on|off;
   ...
}
```

#### 2.3.2.53 resolver
resolver 参数用于设置 DNS 解析器，可以是内置的解析器 (google dns)、指定的文件 (which contains a list of nameservers to query for DNS queries)，也可以指定第三方的解析器 (eg. dnsmasq)。

```bash
http {
   ...
    resolver local=on|off|127.0.0.1:<port>|<ipv4>:<port>,<ipv6>:<port>;
   ...
}
```

#### 2.3.2.54 resolver_timeout
resolver_timeout 参数用于设置 DNS 解析器超时时间，单位为秒。

```bash
http {
   ...
    resolver_timeout seconds;
   ...
}
```

#### 2.3.2.55 root
root 参数用于指定访问根目录的路径。

```bash
http {
   ...
    server {
        listen          80;
        server_name     example.com;
        
        root            "/srv/www/htdocs";
    }
   ...
}
```

#### 2.3.2.56 satisfy
satisfy 参数用于设置客户端请求的满足条件。可以为“all”或按域名划分的白名单。

```bash
http {
   ...
    server {
        listen              80;
        server_name         example.com;
        
        root                "/srv/www/htdocs";
        
        satisfy             any;
    }
   ...
}
```

#### 2.3.2.57 sendfile
sendfile 参数用于发送文件，可以将文件内容发送给客户端，而不是等整个文件都传完再发送，适用于大文件下载。

```bash
http {
   ...
    sendfile on|off;
   ...
}
```

#### 2.3.2.58 sendfile_max_chunk
sendfile_max_chunk 参数用于设置最大数据块大小，单位为 bytes。

```bash
http {
   ...
    sendfile_max_chunk size;
   ...
}
```

#### 2.3.2.59 send_lowat
send_lowat 参数用于设置低水位标记，当发送数据积累到这个值时就发送数据。

```bash
http {
   ...
    send_lowat size;
   ...
}
```

#### 2.3.2.60 send_timeout
send_timeout 参数用于设置响应超时时间，单位为秒。

```bash
http {
   ...
    send_timeout timeout;
   ...
}
```

#### 2.3.2.61 server_name
server_name 参数用于指定虚拟主机的域名，可以使用通配符。

```bash
http {
   ...
    server {
        listen          80;
        server_name     *.example.com;
        
        root            "/srv/www/htdocs";
    }
   ...
}
```

#### 2.3.2.62 server_names_hash_bucket_size
server_names_hash_bucket_size 参数用于设置服务器名字的 hash 表的大小。

```bash
http {
   ...
    server_names_hash_bucket_size size;
   ...
}
```

#### 2.3.2.63 server_names_hash_max_size
server_names_hash_max_size 参数用于设置服务器名字的 hash 表的最大大小。

```bash
http {
   ...
    server_names_hash_max_size size;
   ...
}
```

#### 2.3.2.64 subrequest_output_buffer_size
subrequest_output_buffer_size 参数用于设置子请求输出的缓冲区大小。

```bash
http {
   ...
    subrequest_output_buffer_size size;
   ...
}
```

#### 2.3.2.65 tcp_nodelay
tcp_nodelay 参数用于启用/禁用 TCP NODELAY，用于减少延迟。

```bash
http {
   ...
    tcp_nodelay on|off;
   ...
}
```

#### 2.3.2.66 tcp_nopush
tcp_nopush 参数用于启动TCP协议的Nagle算法，即将多个小报文合并为一个大报文，减少网络流量。

```bash
http {
   ...
    tcp_nopush on|off;
   ...
}
```

#### 2.3.2.67 types
types 参数用于指定 MIME 类型。

```bash
http {
   ...
    types {
        text/html                             html htm shtml;
        text/css                              css;
        text/xml                              xml rss atom;
        image/gif                             gif;
        application/x-javascript              js;
        text/plain                            txt;
        text/vnd.sun.j2me.app-descriptor      jad;
        application/json                      json;
        font/woff                             woff;
        font/ttf                              ttf;
        font/otf                              otf;
    }
   ...
}
```

#### 2.3.2.68Types {}
Types {} 指令用于指定 MIME 类型相关的信息。

```bash
http {
   ...
    types {
        text/html                             html htm shtml;
        text/css                              css;
        text/xml                              xml rss atom;
        image/gif                             gif;
        application/x-javascript              js;
        text/plain                            txt;
        text/vnd.sun.j2me.app-descriptor      jad;
        application/json                      json;
        font/woff                             woff;
        font/ttf                              ttf;
        font/otf                              otf;
    }
   ...
}
```

#### 2.3.2.69 underscores_in_headers
underscores_in_headers 参数用于支持下划线作为 header 字段中的名称。

```bash
http {
   ...
    underscores_in_headers on|off;
   ...
}
```

#### 2.3.2.70 variables_hash_bucket_size
variables_hash_bucket_size 参数用于设置变量的 hash 表的大小。

```bash
http {
   ...
    variables_hash_bucket_size size;
   ...
}
```

#### 2.3.2.71 variables_hash_max_size
variables_hash_max_size 参数用于设置变量的 hash 表的最大大小。

```bash
http {
   ...
    variables_hash_max_size size;
   ...
}
```

#### 2.3.2.72 worker_cpu_affinity
worker_cpu_affinity 参数用于绑定 worker 进程运行的 CPU 核。

```bash
http {
   ...
    worker_cpu_affinity cpus;
   ...
}
```

#### 2.3.2.73 worker_priority
worker_priority 参数用于调整 worker 进程的优先级。

```bash
http {
   ...
    worker_priority number;
   ...
}
```

#### 2.3.2.74 worker_processes
worker_processes 参数用于设置 Nginx 服务器工作进程个数，一般设置为等于 CPU 核数。

```bash
http {
   ...
    worker_processes number|auto;
   ...
}
```

#### 2.3.2.75 worker_rlimit_core
worker_rlimit_core 参数用于指定 worker 进程的 core 文件大小。

```bash
http {
   ...
    worker_rlimit_core size;
   ...
}
```

#### 2.3.2.76 worker_shutdown_timeout
worker_shutdown_timeout 参数用于设置 worker 进程优雅停止的时间，单位为秒。

```bash
http {
   ...
    worker_shutdown_timeout seconds;
   ...
}
```