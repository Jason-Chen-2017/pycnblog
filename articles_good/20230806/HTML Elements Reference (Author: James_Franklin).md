
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 HTML（Hypertext Markup Language）是一种用于创建网页的标记语言。它的主要目的是用来描述网页文档结构及内容，并使其在浏览器上呈现出来。目前HTML已经成为互联网的一项重要组成部分。
          
          本文将介绍HTML文档结构及常用标签的用法，并详细介绍常见的元素用法及属性设置方法，包括基础的结构标签如div、span、img等，容器类标签如div、section、header、footer等，图片、视频、音频播放相关标签，表单类标签等。同时也会结合实际案例分享一些学习经验。
          
         # 2.基本概念和术语说明

         ## 2.1 HTML简介
         HyperText Markup Language，即超文本标记语言。它是一种基于SGML（标准通用标记语言Standard Generalized Markup Language的子集）制定的用于创建网页的标记语言。HTML定义了标签的语法和规则，网页的结构和内容。这些标签使得网页的内容易于搜索、索引和分类，并且可以应用CSS（层叠样式表Cascading Style Sheets）来美化页面。HTML还可以使用各种多媒体类型，例如图像、动画、声音、程序等，对网页的功能进行扩展。由于其简捷的学习难度和容易上手的特点，HTML被广泛使用于各个领域。

         

         ## 2.2 SGML简介
         简单网页标记语言（Standard Generalized Markup Language，SGML），是由美国国防部开发出来的国际标准化组织，是一种标准的通用标记语言。它是基于XML的强大的文本标记语言。它是一种基于文本的标记语言，用标签对文本进行区分和编排。通过约束标签的名称、用法和结构，用户可定义自己的标签。其中，DTD（Document Type Definition，文档类型定义）是SGML的核心，它规定了 SGML文档的结构和语法。

         

         ## 2.3 XML简介
         Extensible Markup Language（可扩展标记语言，XML），是W3C组织推荐的一种开放源代码的跨平台文件格式。它是一种简单的标记语言，结构上类似于SGML，但比之SGML更加强大。XML被设计用来传输和存储数据，因此它具备良好的扩展性。XML有很高的容错率和灵活性。

         

         ## 2.4 HTML文档结构
         HTML文档一般由以下几个部分组成：

         1. 文档类型声明：文档类型声明告诉浏览器这个文档使用哪种版本的HTML或XHTML规范来解析。

         2. 头部信息：包括网页的元数据，比如作者、描述、关键字、字符编码、viewport等。

         3. 文档正文：指的是展示给用户的内容，包括文字、图片、音乐、视频、Applet等。

         4. 链接：提供本页面和其他页面之间的链接，便于用户查找其他相关的信息。

         5. 脚本：脚本用来实现特效、动态交互、服务器通信等。

         6. 样式表：样式表提供统一的页面外观。

         7. 底部信息：通常显示关于作者、版权信息、联系方式等。

         下面是HTML文档结构图：




         ## 2.5 标签
         标签(tag)，是HTML中的一个标记符号，它是一个由尖括号包围的关键词，用来告知Web浏览器如何处理其后紧跟的文字或内容。所有HTML标签都有它们的特定含义和作用，标签之间可以构成复杂的网页结构。


         ### 2.5.1 概念
         在网页中，标签是用来定义文档结构、内容和外观的。标签提供了结构化的方式来表示网页的内容。HTML中的标签可以划分为以下几类：

         1. 块级元素：块级元素会在页面上独占一行。像div、p、h1~h6等标签都是块级元素。

         2. 内联元素：内联元素不会独占一行。像a、b、i、em、strong等标签都是内联元素。

         3. 空元素：没有内容的元素。像img、input、br等标签都是空元素。

         4. 注释标签：用于在HTML中添加注释。

         5. CDATA标签：CDATA标签用于在HTML中嵌入非HTML代码。


         ### 2.5.2 属性

         属性的作用主要有：

         1. 为标签增加语义：通过属性，可以让标签具有更多的含义，如 img 的 src 属性指定了图像文件的URL地址；a 的 href 属性则指定了链接指向的URL地址。

         2. 为标签提供更多的自定义能力：通过属性，可以让标签具备更丰富的定制能力，如 input 的 type 属性可以设定输入框的类型。


         ### 2.5.3 HTML5新标签
         HTML5引入了许多新的标签，如 `<article>`、`<aside>`、`<details>`、`<dialog>`、`<figcaption>`、`<figure>`、`<footer>`、`<header>`、`<main>`、`<mark>`、`<nav>`、`<progress>`、`<ruby>`、`<section>`、`<summary>`、`<time>`等。下表列出了HTML5新增的标签以及它们的作用：

         | 标签        | 描述                                    |
         | ----------- | --------------------------------------- |
         | article     | 定义文章                                 |
         | aside       | 定义侧边栏                               |
         | details     | 定义详细信息                             |
         | dialog      | 定义对话框                               |
         | figcaption  | 定义图片说明                             |
         | figure      | 定义图形                                 |
         | footer      | 定义文档的脚注                           |
         | header      | 定义文档的页眉                           |
         | main        | 定义主内容区域                           |
         | mark        | 定义突出显示                             |
         | nav         | 定义导航链接                             |
         | progress    | 定义任务进度                             |
         | ruby        | 定义俗语和古汉语的注音                   |
         | section     | 定义文档中的节                             |
         | summary     | 定义details元素的摘要                     |
         | time        | 定义日期和时间                           |



         # 3.元素详解

         ## 3.1 框架窗口框架frame

         ```html
         <!DOCTYPE html>
         <html lang="en">
            <head>
               <meta charset="UTF-8">
               <title>Frame Demo</title>
            </head>
            <body>
               
               <!-- 在这里添加框架窗口 -->
               <iframe src="demo.html"></iframe>

            </body>
         </html>
         ```

         iframe标签定义了一个包含另外一个文档的容器。默认情况下，iframe是一个内联框架窗口，但是可以通过设置frameborder属性来改变这一行为。

         frame的src属性指定加载进iframe的文档的URL。由于iframe内部的样式无法直接控制，所以建议通过修改iframe外部的样式来达到效果。


         ## 3.2 基本元素
         有些元素可以在多个地方使用，比如<div>和<span>。这些元素就属于基本元素。基本元素不需要额外属性就可以使用，只需要将它们作为页面的一部分即可。基本元素列表如下：

         1. div - 定义文档中的分隔区块

         2. span - 定义文档中的小片段

         3. a - 创建超链接

         4. p - 创建段落

         5. h1~h6 - 创建标题

         6. img - 定义图片

         7. hr - 创建水平线

         8. br - 创建换行符

         9. input - 创建输入字段

         ## 3.3 表单元素

         ### 3.3.1 text控件
         用于输入单行文本内容。输入文本内容时，按Enter键或Tab键可以换行。此控件只允许输入文本，不能包括特殊字符、数字、字母、中文字符等。在输入中文时，需要按Ctrl+Shift组合键。

         使用示例：

         ```html
         <label for="username">用户名：</label>
         <input type="text" id="username" name="username"><br><br>
         ```

         参数说明：

         1. type：输入类型为text，不可更改。

         2. id：设置控件唯一标识符。

         3. name：设置控件名称。

         4. label：为输入框添加一个关联的标签。


         ### 3.3.2 password控件
         此控件用于密码输入。当用户输入密码时，显示星号代替明文内容。此控件继承自text控件，支持与之相同的属性设置。

         使用示例：

         ```html
         <label for="password">密码：</label>
         <input type="password" id="password" name="password"><br><br>
         ```

         参数说明同上。

         ### 3.3.3 radio按钮
         用于选择单项内容，只能选择一个选项。

         使用示例：

         ```html
         <label for="gender1">男：</label>
         <input type="radio" id="gender1" name="gender" value="male">

         <label for="gender2">女：</label>
         <input type="radio" id="gender2" name="gender" value="female">
         ```

         参数说明：

         1. type：输入类型为radio，不可更改。

         2. id：设置控件唯一标识符。

         3. name：设置一组radio按钮的名称。

         4. value：设置当前选项的字符串值。

         5. label：为选项添加一个关联的标签。

         ### 3.3.4 checkbox按钮
         用于选择多项内容。可选多个选项。

         使用示例：

         ```html
         <label for="book1">《论语》</label>
         <input type="checkbox" id="book1" name="books" value="论语">

         <label for="book2">《孟子》</label>
         <input type="checkbox" id="book2" name="books" value="孟子">

         <label for="book3">《庄子》</label>
         <input type="checkbox" id="book3" name="books" value="庄子">

         <label for="book4">《荀子》</label>
         <input type="checkbox" id="book4" name="books" value="荀子">
         ```

         参数说明：

         1. type：输入类型为checkbox，不可更改。

         2. id：设置控件唯一标识符。

         3. name：设置一组checkbox按钮的名称。

         4. value：设置当前选项的字符串值。

         5. label：为选项添加一个关联的标签。

         6. checked：设置默认选项为已勾选状态。

         ### 3.3.5 select下拉框
         用户可从下拉菜单中选择一项或多项内容。

         使用示例：

         ```html
         <select multiple size="5">
             <option value="web">前端技术</option>
             <option value="app">移动应用技术</option>
             <option value="database">数据库技术</option>
             <option value="security">安全技术</option>
             <option value="java">Java技术</option>
             <option value="python">Python技术</option>
         </select>
         ```

         参数说明：

         1. multiple：允许用户选择多项内容。

         2. size：设置下拉框高度。

         3. option：为下拉框添加选项，每个选项可以设置value属性来保存选中的值。

         ### 3.3.6 file上传框
         用户可从本地硬盘选择一个或多个文件上传至服务器。

         使用示例：

         ```html
         <form enctype="multipart/form-data" action="upload.php" method="post">
             <label for="file">请选择文件：</label>
             <input type="file" id="file" name="file[]" multiple><br><br>
             <input type="submit" value="提交">
         </form>
         ```

         参数说明：

         1. enctype：设置表单数据的编码类型，值为"multipart/form-data"。

         2. action：设置接收上传文件的服务器端脚本的URL。

         3. method：设置提交方式为POST。

         4. input：设置类型为file，允许用户上传多个文件。

         ### 3.3.7 button按钮
         可以在表单中加入按钮，触发某些特定操作。

         使用示例：

         ```html
         <button type="submit">提交</button>
         ```

         参数说明：

         1. type：设置按钮类型，可选值为 submit、reset、button、image。

         ### 3.3.8 textarea文本域
         可用于输入多行文本内容。用户可输入文本，每行自动添加换行符。

         使用示例：

         ```html
         <textarea rows="4" cols="50"></textarea>
         ```

         参数说明：

         1. rows：设置文本域的行数。

         2. cols：设置文本域的宽度。

         ## 3.4 链接元素

        ### 3.4.1 a链接
        通过<a>标签，可以创建超链接。它允许你把指向其他网页或者其他位置的链接放在网页上的任何位置。你可以为<a>标签添加属性，设置链接目标、文本、颜色、悬停时的样式等。

        使用示例：

        ```html
        <a href="http://www.baidu.com">百度</a>
        ```

        参数说明：

        1. href：设置链接目标。

        2. target：设置链接打开方式，可选值为 "_blank"(在新页面打开)、"_self"(在当前页面打开)。默认值为"_self"。

        ### 3.4.2 base链接
        设置页面基准链接。当页面包含相对路径的资源时，可用该标签为这些资源设置一个基准链接。

        使用示例：

        ```html
        <base href="http://www.example.com/">
        ```

        参数说明：

        1. href：设置页面基准链接。

        ### 3.4.3 link样式表
        定义文档使用的外部样式表。

        使用示例：

        ```html
        <link rel="stylesheet" type="text/css" href="mystyle.css">
        ```

        参数说明：

        1. rel：描述样式表的关系，值必须设置为"stylesheet"。

        2. type：设置样式表的MIME类型。

        3. href：设置样式表的URL地址。

        ### 3.4.4 meta信息
        提供与当前页面的行为有关的元数据。

        使用示例：

        ```html
        <meta http-equiv="Content-Type" content="text/html;charset=utf-8">
        ```

        参数说明：

        1. http-equiv：设置元信息的HTTP响应头。

        2. content：设置元信息的内容。

        3. charset：设置网页的字符编码。

   
         # 4.常见问题与解答

         ## 4.1 如何在HTML中添加注释？
         HTML中注释的作用是为了帮助我们进行代码编写和维护。使用<!-- --> 来表示注释，并配合IDE的智能提示、代码格式化工具可以有效提高我们的编程效率。

         ```html
         <!-- 这是一条注释 -->
         ```

         ## 4.2 HTML常用的编码格式有哪些？
         HTML有三种常用的编码格式：

         1. UTF-8编码：Unicode字符集，使用最广泛的编码格式。

         2. GB2312、GBK编码：国标推荐的两种简体中文编码。

         3. Big5编码：繁体中文编码。

         ## 4.3 HTML中script和style标签的执行顺序？
         script和style标签可以出现在HTML页面的任意位置，但执行顺序与书写顺序有关。通常情况下，浏览器会先读取HTML文档的头部内容，然后按照顺序逐步读取剩余内容。如果script标签位于文档末尾，那么将不会阻塞浏览器的渲染过程，尽管这可能影响JavaScript代码的运行。而对于style标签来说，不同的浏览器可能会有不同程度的优化，但总的原则是优先渲染页面中存在的样式表，然后再渲染DOM树中生成的样式。

         ## 4.4 HTML的元素是否可以自闭合？
         不可以。HTML的元素只能有一个开始标签和一个结束标签，即使某个元素不需要标签闭合也可以省略结束标签。