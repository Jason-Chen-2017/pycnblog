
作者：禅与计算机程序设计艺术                    

# 1.简介
  


# 2.SESSION管理机制
PHP提供了一种称为“会话”（session）管理的机制，允许服务器跟踪用户在访问一个网站时的行为，从而为用户提供个性化的服务和信息。使用会话可以使得网站能够记住用户的信息，比如登录状态、购物车信息等。通过设置和读取SESSION变量，开发者可以将不同用户的数据分离开来，使得网站能够更好地满足用户需求。

一个SESSION是一个字典结构，存储了许多不同的键值对，其中包含了用户在同一会话中所做的各种动作、页面浏览路径等信息。PHP为每个用户维护了一个独立的SESSION，并且这个SESSION只能由相应的用户访问到。

当用户第一次访问一个网站时，就创建了一个新的SESSION，并为这个用户生成了一个唯一的ID，这个ID被称为SESSION ID。当用户再次访问这个网站时，浏览器就会自动发送这个SESSION ID，服务器就能够识别出这个用户，并将他从其他会话中踢下线。

# 3.SESSION设置方法

## 设置SESSION的两种方式

1. 使用全局变量$_SESSION：这是最简单的方法。只需在PHP脚本中声明一个名为$_SESSION的全局数组即可，然后就可以向此数组中添加或读取键值对。

   ```php
   <?php
   
       // 设置SESSION
       $_SESSION['name'] = 'John Doe';
       
       // 获取SESSION
       echo $_SESSION['name']; // Output: John Doe
   
  ?>
   ```
   
2. 使用session_start()函数启动会话：这种方法可以在PHP脚本的任何位置调用，并且不需要声明一个全局变量$_SESSION。当执行session_start()函数时，会话开始，并创建一个名为PHPSESSID的Cookie。该Cookie包含了当前会话的唯一标识符，它在浏览器上以文本形式存储，用于辨别属于哪个用户。

   在启用会话之后，可以通过$_SESSION超级全局变量来存取和操作会话数据。例如，以下代码用来设置和获取会话变量：

   ```php
   <?php
   
    session_start();
    
    // 设置SESSION变量
    $_SESSION['name'] = 'John Doe';
    
    // 获取SESSION变量
    echo $_SESSION['name']; // Output: John Doe
   
  ?>
   ```

## 配置PHP以启用SESSION COOKIE

默认情况下，PHP不会向客户端发送任何COOKIE，除非指定session.cookie_path参数，指定Cookie所在的文件夹。为了启用Cookie，需要在php.ini配置文件中设置session.use_cookies=1和session.use_only_cookies=1。这样，PHP才能向客户端发送名为PHPSESSID的Cookie，并在每次响应中返回它。

```
session.use_cookies = On
session.use_only_cookies = On
```

设置完这些配置后，浏览器应当会在HTTP请求头中看到Set-Cookie行，如：

```
Set-Cookie: PHPSESSID=obe9nvbvovmd5oi5hflqhnldj1; path=/
```

此处的PHPSESSID即为会话ID，它在浏览器上以文本形式存储，用于辨别属于哪个用户。对于某些浏览器，如IE，可能会出现权限问题，导致无法将Cookie发送至客户端，但其他浏览器应该均正常工作。

注意：Cookie被禁用的浏览器仍然可以使用SESSION机制，但需要额外的手段进行通信，如JSONP或Flash技术。

# 4.SESSION的生命周期

SESSION的生命周期指的是用户请求结束之后，SESSION持续存在的时间。过期时间可以在php.ini文件中修改。

# 5.SESSION的限制

由于服务器端的内存空间有限，因此不能无限扩充SESSION数量。为避免占用过多内存，服务器管理员可以设置最大SESSION数目、限制SESSION单个值的大小、回收旧SESSION等方式。

# 6.SESSION的防范

防范SESSION攻击的方式主要有三种：

1. 限制IP地址：服务器管理员可以将SESSION绑定到指定的IP地址，从而确保只有指定的IP才可以访问SESSION。

2. 设置超时时间：为了防止恶意用户长期盯着一个SESSION，服务器管理员可以设置超时时间，超时则自动销毁SESSION。

3. 使用加密传输：如果网站对用户数据的敏感程度较高，建议使用SSL加密传输，否则SESSION数据容易被窃听、篡改。

# 7.使用SESSION的一些注意事项

1. 在使用POST提交表单时，由于浏览器采用异步方式提交表单，因此可能会丢失表单数据。解决方案是在提交表单时将SESSION数据一起提交。

2. 如果SESSION数据量过大，可能影响网站的性能。因此，服务器管理员需要合理规划SESSION的数量和大小，选择合适的回收策略。

3. 为了防止跨站请求伪造（CSRF）攻击，服务器必须在每一次请求中都附带正确的SESSIONID，否则会拒绝该请求。

# 8.常见问题解答

**Q：什么是CSRF攻击？**

CSRF（Cross-site request forgery，跨站请求伪造），是一种攻击方式，通过伪装成合法用户的假象，利用受害者的无权访问某些功能或数据的长效链接，达到冒充受害者身份，取得个人信息或对网络目标发送恶意请求的目的。

**A：**CSRF攻击通常发生在第三方网站向你的网站发送请求，而不是真实的用户。CSRF的利用流程如下：

1. 用户登录A网站；
2. A网站会向B网站发送请求，但没有携带身份凭证（Cookie）。
3. B网站接受到请求，生成HTML页面，然后将HTML页面的URL通过电子邮件发送给用户。
4. 用户点击该URL，A网站中嵌入的页面加载完成后，B网站接收到了请求，根据请求参数，执行相应的操作。
5. 用户认为自己正在操作A网站，实际上他却是在操作B网站，因为他在自己的浏览器上执行的操作其实是给B网站发送请求。

为了防止CSRF攻击，服务器需要对用户请求进行验证，并附带一个特殊的Token，只允许请求携带有效的Token。

**Q：如果我关闭了浏览器的COOKIE功能，PHP又如何区分不同用户？**

如果你关闭了浏览器的COOKIE功能，那么PHP无法区分用户。PHP中SESSION管理依赖于Cookie来区分不同的用户，因此，如果浏览器禁用了COOKIE，PHP无法保证用户的访问记录。所以，为了能够正常使用SESSION，浏览器的COOKIE功能必须打开。

**Q：PHP默认关闭SESSION写入磁盘，为什么要开启？**

PHP默认关闭了SESSION写入磁盘，原因是很多WEB服务器是多进程模型，当WEB服务器重启的时候，所有进程都会被重新拉起，导致SESSION也会丢失。一般情况下，SESSION并不是作为系统资源要求很大的，所以PHP选择了默认关闭。但是，开启SESSION写入磁盘，可以解决WEB服务器崩溃的问题。当然，生产环境下的服务器一定要开启，不然的话，可能导致数据的丢失和泄漏。