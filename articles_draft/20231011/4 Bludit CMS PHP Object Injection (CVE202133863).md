
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Bludit CMS是一个基于PHP的CMS软件。它的最新版本（v3.9.2）发布于2020年7月1日。根据官方网站介绍，“Bludit is an Open Source flat-file CMS designed to be easy to use and very powerful.” Bludit属于开源、免费、简单易用、功能强大的Flat File CMS。其用户界面简洁、功能齐全，适用于个人或小型团队的博客网站。Bludit通过插件系统提供许多不同主题和插件供用户选择。Bludit的所有数据都存储在文件系统中，不需要数据库支持，所以它易于部署、管理和备份。

由于Bludit的插件机制，攻击者可以通过注入恶意代码到插件、主题、后台管理页面、任意用户上传的文件等，进而控制网站运行。最近，安全研究人员发现了一种新的Bludit漏洞——CVE-2021-33863。

此漏洞允许攻击者通过Bludit后台登录页面提交恶意代码，进一步导致远程执行命令、执行任意代码或篡改网站数据。该漏洞已得到CVSS评分为7.5分，高危级别。

本文作者将深入分析Bludit CMS PHP Object Injection漏洞的原理、利用方法、风险等方面，并提供专业的技术实践指南。希望能够帮助读者更好地理解和防范Bludit CMS的安全威胁。
# 2.核心概念与联系
## 漏洞类型
Bludit是一款开源的Flat File CMS，一般情况下，攻击者通过向Bludit后台页面发送恶意请求，来利用该漏洞可以执行任意代码。

Bludit漏洞类型主要包括PHP Object Injection和CSRF(Cross-Site Request Forgery)。

**PHP Object Injection**: 顾名思义，就是将恶意代码插入到服务器端代码执行的过程。比如，攻击者可以在POST请求的data参数里插入一个PHP对象，当访问受害者的网站时，这个对象会被执行，从而造成恶意行为。此类漏洞的危害可能会造成拒绝服务、信息泄露、权限提升等严重后果。例如，Bludit中的PHP Object Injection漏洞，就可以被用来攻击Bludit站点，对Bludit管理员账户及重要数据的造成损坏、恶意操作。

**CSRF**: CSRF（跨站请求伪造），又称为One Click Attack或Session Riding，指的是黑客冒充用户在网站上进行一系列动作，比如转账、投票等，而实际上用户并不会产生任何动作，因此遭到网站的恶意利用。与XSS攻击相比，XSS利用的是网站内的输入输出，而CSRF利用的是用户身份认证信息来欺骗服务器误认为是合法用户发起的请求。这种攻击方式通常是通过第三方网站合法渠道获取用户cookies，然后伪装成真正的用户请求网站，达到冒充用户的目的。

Bludit CVE-2021-33863漏洞，是一个PHP Object Injection漏洞，攻击者可以在Bludit的登录页面注入恶意代码。通过构造恶意的PHP object，攻击者可控制网站运行。这使得Bludit CMS成为远程执行命令、执行任意代码、篡改网站数据的重要目标。


## 影响范围
Bludit插件、主题、后台管理页面、任意用户上传的文件、登录页面等处。


## 修复方案
Bludit v4.0.0已经修复了此漏洞，建议升级到最新版。


# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
1.漏洞类型： PHP Object Injection

2.影响范围：Bludit插件、主题、后台管理页面、任意用户上传的文件、登录页面等处。

3.危害程度：高危。

Bludit插件、主题、后台管理页面、任意用户上传的文件、登录页面等处均存在 PHP Object Injection漏洞。Bludit不对用户提交的内容做过滤处理，导致攻击者可以注入任意的PHP代码，甚至能够读取服务器上的敏感文件。

4.漏洞详情：

Bludit v3.9.2版本之前的PHP Object Injection漏洞，存在较高的危害性。如果能够在特定位置构造出特定的PHP object，则可以控制服务器执行任意代码。

该漏洞主要涉及以下几个方面：

1.后台登录：攻击者可通过注入恶意的PHP object，通过构造恶意的链接地址、表单参数，绕过登录验证直接登录后台。

2.编辑器：攻击者可通过注入恶意的PHP object，提交表单、评论等内容。在提交评论时，恶意代码可以执行一些危险的命令或者操作网站数据。

3.文件上传：攻击者可通过上传包含恶意代码的图片文件，然后通过图片路径访问该图片，执行任意代码。

4.后台管理页面：攻击者可通过设置、添加插件、修改主题等操作，提交含有恶意代码的表单。这些恶意代码可以执行一些危险的操作，如控制网站、读取服务器文件、执行系统命令、篡改网站配置等。

5.插件和主题：攻击者可通过注入恶意的代码，提交含有恶意代码的插件或者主题包，然后在Bludit后台下载安装。这样，恶意代码将被加载到服务器，被解析执行，从而控制服务器的运行。

6.数据库查询：攻击者可以通过构造SQL语句，提交到后台数据库查询接口，从而读取服务器上的敏感文件。

通过构造PHP object，攻击者可控制服务器执行任意代码，比如命令执行、网站篡改、文件读取等。

为了确保Bludit站点的数据安全，Bludit不应该轻易给予普通用户上传文件的权限，并且需要限制普通用户上传文件的格式和大小。另外，在执行文件操作之前，还应做好相应的安全措施，如设置合理的目录权限、使用HTTPS加密传输等。


# 4.具体代码实例和详细解释说明

假设攻击者想通过Bludit登录后台，只需找到管理员账号对应的MD5值，构造如下的payload：

```php
<?php
    $md5 = '1f2d8fb5a7c5b1d0f6deaa9d0ebdaed8'; // 用户密码的MD5值
    $username = 'admin';
    if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['password']) && md5($_POST['password']) == $md5){
        echo "登录成功！";
    }else{
        echo "登录失败，请检查用户名或密码是否正确!";
    }
?>
```

接着，攻击者可以尝试提交以下内容作为`password`参数，以绕过登录验证：

```php
// username=admin&password=<script>alert("我是恶意代码")</script>&submit=Login
```

当受害者访问该链接时，会弹出JavaScript警告框：

```javascript
// 我是恶意代码
```

同时，也可以通过构造其他类型的文件上传漏洞，在上传文件的过程中注入PHP object，上传含有恶意代码的图片，通过图片路径访问该图片，执行恶意代码。


# 5.未来发展趋势与挑战
经过这次漏洞的披露，Bludit社区已经启动了针对该漏洞的调查和响应，正在积极跟踪并回复安全通报。随着Bludit在各个领域的推广应用，Bludit的安全性也越来越重要，越来越多的网站开始采用Bludit作为基础平台。 

针对Bludit的PHP Object Injection漏洞，安全专家们已经出台了多项解决方案，其中最佳的应对办法是强化服务器的安全配置。

一方面，要降低受攻击面，尽可能不要在网站上放置可执行文件，只允许白名单内的扩展名文件上传；另一方面，要通过减少外部输入，消除代码注入的可能性，实现最低限度的输入校验。

对于管理员，也应该加强管理权限，限制其登录的IP地址和网络段，并定期更换登录密码。

Bludit v4.0.0版本已经修复了此漏洞，但是目前仍然存在一定隐患，仍然有许多潜在的攻击点没有完全关闭。希望Bludit社区和厂商共同努力，共同打造更加安全的Bludit。


# 6.附录常见问题与解答

**Q: 为什么Bludit的漏洞可以执行任意代码？**

A: 在Bludit中，管理员账户的密码哈希值存储在数据库中，而不是明文存储。如果攻击者知道数据库的连接信息，他/她可以对密码进行解密，从而得到密码哈希值。根据哈希值的不同，攻击者可以构造不同的PHP object，从而控制服务器执行任意代码。

**Q: Bludit可以读取服务器上的敏感文件吗？**

A: 当用户上传了一个包含恶意代码的图片文件，管理员无法避免的需要保存到服务器的文件，导致文件被读取，出现恶意代码。即便管理员设置了权限限制，攻击者还是可以窃取到服务器上的敏感文件。

**Q: Bludit可以篡改网站配置吗？**

A: 在后台管理页面，管理员可以提交含有恶意代码的表单。虽然现代浏览器会对表单字段进行输入校验，但仍然有很大的安全风险。因此，建议管理员注意输入有效值。

**Q: Bludit可以执行命令吗？**

A: 在后台管理页面，管理员可以提交含有恶ence代码的插件、主题包。插件、主题包可能包含恶意代码，从而被加载到服务器，执行任意代码。

**Q: 是否所有的Bludit漏洞都是公开的?**

A: 不完全是，Bludit在发布漏洞时，会公布漏洞类型、影响范围、危害程度、修复方案、补丁等相关信息。并且，Bludit团队会持续跟踪并回复安全通报。