
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1.Supervisor 是 Python 下的一个开源进程管理器，它可以启动、停止、控制、和监控 Unix 和 Windows 平台上的应用程序，它还提供了强大的进程事件机制，可以响应系统上发生的各种事件并采取相应的动作。这在现代化 IT 环境中尤其重要，因为运维人员不仅需要了解复杂的 IT 系统内部运行情况，而且还要能够及时发现和解决问题。因此，Supervisor 提供了一种高效的方法来集中管理所有应用程序的运行状态，并提供全面的故障诊断能力。除了可靠性方面，Supervisor 也是一个非常实用的进程监控工具，可以帮助您更好地掌握应用服务的运行状态和性能指标。
         
         ## 为什么要使用 Supervisor? 
         在一个多应用的环境下，运维人员需要花费大量的时间来监控各个应用的运行状态，并对出现的问题进行诊断和处理。而如果将这些工作交给 Supervisor 来自动完成的话，那么整个运维工作流程就可以大大简化。Supervisord 是最著名的基于 Python 的进程监控器，它可以在 Unix/Linux 操作系统和 Windows 操作系统上运行。它是用 C++ 编写的，但是在 Python 中提供了接口支持。基于 Supervisor 的进程管理方案可以有效地减少人工操作的次数，提高工作效率。Supervisord 支持多种运行方式，包括守护进程模式和 inet 模式，还可以使用配置文件或 XML-RPC API 对进程进行控制。在日常运维中，我们经常会遇到一些意想不到的异常场景，比如应用突然挂掉或者某个任务卡住等。借助 Supervisor ，我们就可以快速定位和排查这些异常情况，防止它们导致系统瘫痪甚至崩溃。 

         ## 安装 Supervisor 
         1. Linux 
            
             ```bash
            sudo apt-get install supervisor 
            ```

             或 

             ```bash
            sudo yum install supervisor 
            ```
         
          2. Windows  

            如果没有安装过 Python ，则先安装 Python 然后再安装 Supervisor 。下载地址：https://www.python.org/downloads/release/python-379/  
            
            使用 pip 命令安装 Supervisor ，如下所示：  
             
            ```powershell
            python -m pip install supervisord
            ```
          
        ## 配置 Supervisor 
        在 /etc/supervisor/supervisord.conf 文件中配置 Supervisor 服务的监听端口和日志文件位置，默认情况下，Supervisor 默认的配置如下：

        ```ini
        [unix_http_server]
        file = /var/run/supervisor.sock   ; (the path to the socket file)
        chmod = 0700                       ; sockef file mode (default 0700)
        
        [inet_http_server]
        port = 127.0.0.1:9001              ; (ip_address:port specifier, *:port for all iface)
        
        [supervisord]
        logfile=/var/log/supervisord.log    ; main log file; default $CWD/supervisord.log
        pidfile=/var/run/supervisord.pid    ; supervisord PID file; default $CWD/supervisord.pid
        childlogdir=/var/log/               ; ('AUTO' or '/path/to/logdir') write child logs to a directory if it exists; default $TEMP
        
        [rpcinterface:supervisor]
        supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface
        
        [supervisorctl]
        serverurl=unix:///var/run/supervisor.sock ; use a unix:// URL  for a unix socket
    
    ```

    上述配置项中，`[unix_http_server]` 配置项指定了 Unix Socket 服务端的文件路径，`[inet_http_server]` 配置项设置了 HTTP 服务端监听端口，`[supervisord]` 配置项设置了主日志文件路径、`PID` 文件路径和子进程日志目录，`[supervisorctl]` 配置项指定了 Supervisor 服务端 Unix Socket 的地址。这里只需要关注一下 `[program:xxx]` 项。这是定义了一个名称为 `xxx` 的程序的配置，用来控制 `xxx` 程序的启动、停止、重启等操作。例如，如果我们想要监控 Nginx，就需按以下配置添加 `nginx` 程序的配置：

    ```ini
    [program:nginx]
    command=/usr/sbin/nginx   ; the program (relative uses PATH, can take args)
    autostart=true             ; start at supervisord startup
    autorestart=true           ; restart when process exits
    redirect_stderr=true       ; redirect stderr to stdout
    stopsignal=QUIT            ; send this signal to kill the process 
    user=nobody                ; setuid to this UNIX account
```

   此处只举例了几个关键的配置选项，具体含义参考官网文档即可。
   
   ## 启动 Supervisor 
   当配置完 supervisor.conf 文件后，可以使用如下命令启动 Supervisor 服务：

   ```bash
   supervisord
   ```

   成功启动之后，我们便可以通过浏览器访问 Supervisor 的 Web 页面查看进程信息、日志和状态。默认情况下，Supervisor 的 Web 页面绑定的是本地的 `localhost` 和 `9001` 端口。

  ![image.png](attachment:image.png)

   ## 配置 Nginx 程序的自动启动 
   
   在 Supervisor 配置中，每个程序都对应了一个配置项，用来控制该程序的启动、停止、重启等操作。我们可以将 Nginx 的配置项加入到 supervisor.conf 文件中，这样，当服务器启动时，Nginx 会自动启动：

   ```ini
   [program:nginx]
   command=/usr/sbin/nginx   ; the program (relative uses PATH, can take args)
   autostart=true             ; start at supervisord startup
   autorestart=true           ; restart when process exits
   redirect_stderr=true       ; redirect stderr to stdout
   stopsignal=QUIT            ; send this signal to kill the process 
   user=nobody                ; setuid to this UNIX account
   ```


   配置文件修改保存后，通过如下命令重新加载 Supervisor 配置：

   ```bash
   supervisorctl reload
   ```

   此时，当服务器启动时，Supervisor 将会自动启动 Nginx 。

   ## 测试 Nginx 自动启动 

   可以通过 `systemctl status nginx` 命令查看 Nginx 程序的运行状态，如果看到 `Active: active (running)` 表示已经正常启动。如果一切正常，则可以打开浏览器输入 `http://IP地址/` 看是否可以正确打开 Nginx 的欢迎界面。

  ![image.png](attachment:image.png)

   ## 查看进程信息 

   通过 `supervisorctl status` 命令可以查看 Supervisor 管理的所有程序的当前运行状态。

   ```bash
   supervisor> ps
      ngiNX                               RUNNING    pid 31752, uptime 0:00:23
     php-fpm                             STOPPED    May 26 14:44 PM
  inotifywait                          STOPPED    May 26 14:44 PM
 syslog-ng                            STOPPED    May 26 14:44 PM
   ```

   此时可以看到目前运行着的程序有 Nginx 和 php-fpm 。其中，Nginx 的状态为 `RUNNING`，php-fpm 的状态为 `STOPPED`。如果某些程序出现问题，可以执行 `supervisorctl start xxx`、`supervisorctl restart xxx`、`supervisorctl stop xxx` 命令进行控制。

   ## Nginx 程序的自动重启

   Supervisor 提供了很多便捷的方法来控制程序的运行，比如 `restart`、`stop`、`start`、`status` 等命令。我们也可以利用这些命令来实现一些复杂的功能，比如定时自动重启某个程序。

   以 Nginx 为例，假设我们希望每隔五分钟自动重启一次 Nginx ，那么我们可以在 Supervisor 配置文件中添加如下配置：

   ```ini
   [group:nginx]
   programs=nginx
   process_name=%(program_name)s_%(process_num)d
   priority=999
   autostart=false
   autorestart=unexpected

   [eventlistener:nginx_autorestart]
   command=touch %(directory)/nginx_autorestart
   events=TICK_5
   locks=nginx_autorestart
   listener_name=%(program_name)s_%(process_num)d
   options=
   ```

   此处的 `group` 定义了一个 `nginx` 组，用于同时管理 Nginx 的多个进程；`programs` 指定这个组下属的程序为 Nginx；`priority` 设置了优先级，值越小表示越优先启动；`autostart=false` 表示该组不会自动启动，由其他程序统一启动；`autorestart=unexpected` 表示只有当 Nginx 意外结束时才自动重启，避免无限重启；`eventlistener` 定义了一个事件监听器，每隔五秒产生一次 `TICK_5` 事件，调用命令 `touch %(directory)/nginx_autorestart`，实现了定时重启的效果。

   配置文件修改保存后，通过如下命令重新加载 Supervisor 配置：

   ```bash
   supervisorctl reread
   supervisorctl update
   ```

   此时，Supervisor 会识别到新增的 `eventlistener` 配置项，并自动启动定时重启 Nginx 的进程。

# 总结

本文主要介绍了 Supervisor 的安装、配置、使用方法，并且展示了如何监控 Nginx 程序，实现 Nginx 程序的自动重启。最后，通过示例阐述了 Supervisor 的配置文件结构、相关命令、事件监听器等知识点。此外，Supervisor 还提供了许多丰富的功能和特性，读者可以根据自己的需求选择合适的功能来使用。

