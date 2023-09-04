
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
session_start()是一个功能库，可以帮助开发者在PHP环境中实现会话管理功能，主要包括：

1.会话处理：session_start()函数负责创建新的或恢复现有的会话。通过PHP脚本的执行过程中，每个用户都可以访问相同的PHP变量和存储空间。

2.安全性保护：session_start()默认启用了Cookie的安全设置，从而确保会话数据只能通过加密和传输的方式进行交换。同时还提供了多种验证方式，如验证码、IP地址检测、HTTP Referer检测等，确保会话数据安全有效。

3.会话超时设置：开发者可以指定会话的生存周期，如果超过这个时间则自动销毁。

4.会话数据持久化存储：session_start()可以在PHP脚本结束时自动保存会话数据到服务器的磁盘上，或者在某个事件触发时临时保存会话数据。

5.会话作用域控制：开发者可以限制特定页面或目录下的所有会话，也可以细粒度地控制不同页面的会话权限。

## 安装与配置
session_start()可以直接下载安装包或通过PECL工具集进行安装。该扩展已经进入php官方扩展仓库，并不需要编译安装。

### PECL安装
PECL（PHP Extension Community Library）是一个第三方仓库，提供PHP扩展的发布、版本跟踪、下载和维护服务。可以使用下面的命令直接安装session_start()扩展：
```shell
pecl install session
```
然后在php.ini配置文件末尾添加以下几行代码启用session支持：
```ini
[Session]
extension=session.so
session.save_handler = files
session.save_path = "/tmp"
```
其中：
- `session.save_handler` 指定了会话的保存方式，可选的值有files、memcached、redis等，这里设置为文件存储。
- `session.save_path` 指定了会话文件保存路径，对于files方式，需要填写实际的文件夹路径；对于其他保存方式，根据自己的系统配置填写相应参数即可。

### 命令行安装
可以通过php源码安装或下载二进制文件，并将其拷贝到PHP安装目录的ext文件夹下，然后修改php.ini文件来启用扩展。以下以源码安装为例：
```bash
wget https://github.com/php/php-src/archive/php-7.4.zip # 下载源码压缩包
unzip php-7.4.zip && cd php-src-php-7.4
./configure --enable-fpm --with-mysqli --with-pdo-mysql --with-zlib --enable-mbstring \
  --enable-bcmath --with-curl --enable-soap --enable-sockets --enable-xml --disable-rpath \
  --enable-session --prefix=/usr/local/php7 --with-config-file-path=/etc/php7/ --with-config-file-scan-dir=/etc/php7/conf.d/
make ZEND_EXTRA_LIBS='-liconv' # 加上libiconv支持中文
sudo make install
```
然后在php.ini配置文件末尾添加以下几行代码启用session支持：
```ini
[Session]
extension=session.so
session.save_handler = files
session.save_path = "/tmp"
```
重新启动Web服务器使之生效。

## 使用方法
使用session_start()后，会自动创建一个名为SESSION的全局变量，此变量是一个数组，包含当前会话的所有数据。它具有以下属性：

1.session_id: 会话ID，每一次会话请求，都会自动分配一个唯一的ID值，用于标识当前用户的会话，并存储在客户端浏览器上的Cookie中。

2.$_SESSION: 会话数组，用来存储会话数据。

3.$_SESSION['name'] 或 $session->name: 可以用数组或对象的方式来访问会话的数据，也可以通过$_SESSION['name']获取，但最好还是用$session->name方式。

### 开启会话
调用session_start()函数，即可开启会话。例如：
```php
<?php
session_start();
// $_SESSION['user_id'] = 1; // 将用户ID存入会话数组
?>
```
当第一次请求该页面时，会自动创建新会话，并在浏览器中写入session_id Cookie。当再次请求页面时，如果发现存在session_id Cookie，就认为是同一个会话，就自动读取其对应的会话数据，并将其赋值给$_SESSION全局变量。所以无需担心多次请求重复创建会话。

### 关闭会话
一般情况下，只要用户关闭浏览器窗口，当前会话就会失效。也可以通过调用session_unset()、session_destroy()或session_write_close()函数来手动结束当前会话。例如：
```php
<?php
session_start();
// 一些处理代码……
session_unset(); // 清空当前会话数据
session_destroy(); // 删除当前会话
session_write_close(); // 将会话数据写入硬盘并关闭当前会话
header('Location: index.php'); // 跳转到首页
exit;
?>
```

### 设置超时时间
除了在配置文件中设置session.gc_maxlifetime参数外，还可以调用session_cache_expire()函数设置会话过期时间。例如：
```php
<?php
session_cache_expire(30); // 设置会话过期时间为30分钟
session_start();
// 处理代码……
session_write_close(); // 将会话数据写入硬盘并关闭当前会件
?>
```