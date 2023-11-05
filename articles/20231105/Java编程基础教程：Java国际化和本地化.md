
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Java是一门面向对象的、类结构简单的编程语言。它提供了良好的性能及简单易用性，被广泛应用于各个行业领域，如移动互联网、企业级应用、游戏开发等。然而，作为一种高级语言，Java语言也存在很多局限性。其中一个重要的局限就是它的国际化支持不够完善，使得它在不同国家使用的用户体验差异较大。

为了解决这一问题，Sun公司推出了JRE（Java Runtime Environment）的国际化支持。JRE的国际化功能包括对日期、数字、货币、消息输出的支持。除此之外，还可以针对特定语言进行定制化翻译。

虽然Sun公司的JRE实现了较好的国际化支持，但其机制并非像一般的跨平台软件一样全面完备。当遇到特殊需求时，仍需要自己编写多套资源文件或使用复杂的国际化框架才能实现。本文将会讨论基于JRE的国际化和本地化机制，并通过一些实际例子介绍如何进行相应的国际化和本地化工作。

# 2.核心概念与联系
## 2.1 JRE国际化机制概览
JRE的国际化机制主要分为以下几步：

1. 设置默认语言环境：通过设置系统环境变量LANG或者JAVA_HOME/lib目录下的locale.properties文件来指定默认语言环境。

2. 根据用户设定的语言环境加载资源文件：JRE根据用户所指定的语言环境加载对应的资源文件，比如对于中文用户加载zh_CN.jar，英语用户加载en_US.jar。JRE提供ResourceBundle类用于加载资源文件。

3. 使用ResourceBundle加载资源对象：ResourceBundle类提供了loadResourceBundle()方法用于加载资源对象。该方法返回的对象是一个java.util.Map接口的实现，其中key为字符串，value为字符串数组。从返回的资源对象中可以获取相应的字符串值，比如：

   ResourceBundle rb = ResourceBundle.getBundle("mybundle");
   String greeting = rb.getString("greeting");
   System.out.println(greeting);
   
   此处的"mybundle"为资源文件的名称，"greeting"为资源键名，"hello world"则为对应的值。
   
4. 使用MessageFormat类进行消息格式化：MessageFormat类提供了format()方法，它可以格式化传入的对象数组，并返回格式化后的字符串。举例如下：

   double amount = 12345.67;
   String str = MessageFormat.format("{0} {1}", new Object[]{amount, "USD"});
   System.out.println(str);
   
   上述代码将格式化成"12345.67 USD"。
   
5. 支持动态语言切换：通过修改系统环境变量或调用ResourceBundle类的setLocale()方法可以实现动态语言切换。
   
总结来说，JRE的国际化机制可以说是非常简洁易用的。但是其缺点也是很明显的，比如只支持文本国际化，不支持图像、视频、声音等多媒体国际化；只支持文本文件的资源管理，不支持数据库、网络等更加灵活的资源管理方式；资源文件的命名规则比较固定，不能自定义；支持动态语言切换，但是切换后需要重新启动应用程序；等等。因此，相比其他的国际化和本地化框架，JRE的国际化机制还有很大的改进空间。

## 2.2 JRE本地化机制概览
JRE的本地化机制也分为以下几步：

1. 通过locale.setDefault()方法设置默认地区：通过该方法设置系统默认地区，一般情况下建议设置为跟系统一致。

2. 通过Locale.forLanguageTag()方法创建Locale对象：该方法可以根据RFC 4646定义的区域标签语法创建Locale对象。

3. 根据地区和语言环境加载资源文件：JRE根据用户指定的地区和语言环境加载相应的资源文件，比如对于中国大陆地区的英语用户加载zh_CN.jar，德国地区的德语用户加载de_DE.jar。如果指定的地区没有资源文件，则采用最适合的资源文件。JRE提供ResourceBundle类用于加载资源文件。

4. 使用ResourceBundle加载资源对象：ResourceBundle类提供了loadResourceBundle()方法用于加载资源对象。该方法返回的对象是一个java.util.Map接口的实现，其中key为字符串，value为字符串数组。从返回的资源对象中可以获取相应的字符串值，比如：

   Locale locale = new Locale("fr", "FR"); //French (France)
   ResourceBundle rb = ResourceBundle.getBundle("mybundle", locale);
   String message = rb.getString("message");
   System.out.println(message);
   
   此处的"mybundle"为资源文件的名称，"message"为资源键名，"bonjour le monde"则为对应的值。

5. 对资源文件进行拓展：可以通过创建新的资源文件来扩展现有的国际化机制。例如，可以创建一个本地化资源文件，用来替换国际化资源文件中的字符串。这样的话，就可以让软件更好地适应本地化需求。

6. 支持动态地区切换：通过修改系统地区或Locale对象的方法可以实现动态地区切换。

总结来说，JRE的本地化机制更加全面，但同时也要考虑到灵活性、可扩展性等因素，可能比JRE的国际化机制稍显复杂。不过，JRE的本地化机制的优点是可以支持各种多样化的本地化需求。