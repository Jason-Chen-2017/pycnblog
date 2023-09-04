
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Valine 是一款快速、简洁且高效的无后端评论系统。它的特点是不依赖数据库，使用 GitHub Issues API 来存储评论数据。因此，它非常适合静态网站、个人博客等没有动态交互功能的场景。除此之外，Valine 提供了强大的管理后台，用户可以轻松地管理评论并设置相关权限。

本文将详细介绍Valine，阐述其使用方法及其背后的原理。希望能够帮助大家更好地理解和使用Valine这个优秀的开源项目。

# 2.基本概念和术语
## 2.1 Valine 的名称由来
Valine (from 'V' and 'W') 是中文意思中的 '神奇' 和 '冥王星'。由于'神奇'的意思容易让一些读者产生误解，所以Valine的名称由来实际上很简单。作者在寻找一种与众不同的名字时，先找到了'冥王星'这个词组。而'冥王星'最早出现于《星空记》中。这是一个三体人想象出来的奇妙世界。所以Valine的名字也取自这部经典电影。

## 2.2 为什么要做 Valine？
Valine 诞生于开源社区，并受到广大开发者的欢迎。这是因为：

1. 免费：Valine 是免费提供给所有人的。免费使得它有能力吸引更多的用户。

2. 易用：Valine 使用 GitHub Issue API 来存储评论数据，不需要用户自己搭建服务器。同时提供了丰富的管理后台，用户可以管理评论并设置相应的权限。

3. 可靠：Valine 使用 JavaScript 框架 Vue 进行前端渲染，并且已经拥有了完善的测试工具和文档，可以保证产品的稳定性。

4. 安全：Valine 默认采用 HTTPS 协议，不会保存用户提交的任何信息，确保数据的安全性。

5. 支持：Valine 提供对主流评论系统的兼容。你可以轻松地把 Valine 的评论嵌入到现有的网站或博客中。

总结来说，Valine 致力于解决现实生活中存在的各种各样的问题。通过建立一个基于 GitHub Issues API 的快速、简洁且免费的评论系统，它可以帮助你解决各种博客园、简书、Gitment、Disqus 这些烦人的评论插件难题，而且还支持自定义样式。

# 3.核心算法原理及操作步骤
## 3.1 数据结构
Valine 的核心数据结构是一个数组，其中每一项是一个对象，代表一条评论，具有以下属性：

- id: 每条评论的唯一标识符，也是 GitHub Issue 的 ID。
- content: 评论的内容。
- author: 评论的作者。
- email: 评论的邮箱地址。
- url: 评论的网址。
- ua: 用户代理（User Agent）信息。
- created_at: 创建时间戳。

另外，Valine 中还有一个配置文件 config.json，用于配置一些基础参数，如“是否启用昵称”、“Gravatar 头像”等。该文件中的某些参数也可以通过 URL 参数控制，例如：https://valinecdn.bili33.top/comment/count?id=xxx&title=xxxxx。

## 3.2 评论显示规则
当用户访问一个页面时，Valine 会通过 AJAX 请求获取该页面对应的评论数据。如果该页面启用了评论功能，则会向后端请求一个随机数（nonce），后端根据 nonce 生成评论列表的 HTML 代码。如果该页面没有启用评论功能，则返回的是一个空的 HTML 代码。

对于每个页面，Valine 会自动生成两个标签：<span id="vcomments"></span> 和 <script src="//cdn.jsdelivr.net/npm/@valine/valine/dist/valine.min.js"></script>。前者用于承载评论内容，后者用于加载 Valine 脚本。

Valine 根据用户配置或者 URL 中的参数，选择正确的评论系统模板。然后，在加载脚本之后，它会初始化一个新的 Valine 对象，传入必要的参数。比如说，它会读取配置文件中“应用 ID”和“评论框占位符”等参数。

在初始化完成之后，Valine 会发送一个 GET 请求到 GitHub API 获取指定页面的 GitHub Issue。如果当前页面启用了评论，那么就创建一条新的 issue，将评论作为 issue 的评论。如果当前页面没有启用评论，那么就从 GitHub API 获取 issue 信息，并根据 issue 信息填充评论表单。

然后，Valine 会将评论列表绑定到指定的 DOM 元素上，这样就能够在浏览器中看到评论内容。但是 Valine 在内部实现了很多优化技巧，比如说：分页展示，防止过多评论导致页面卡顿，提升性能，减少资源消耗等。

## 3.3 发表评论的流程
当用户填写完评论表单并点击“提交”按钮时，Valine 会首先判断用户输入的数据是否有效，比如说，检测邮箱是否符合格式，检测内容长度是否超过限制等。如果数据无误，则创建一条新的评论对象，并调用 GitHub API 将评论发布到对应的 GitHub Issue 上。

为了防止垃圾邮件的骚扰，Valine 会通过检查用户输入的邮箱地址和用户名是否相符，只有满足条件的才能正常发表评论。此外，Valine 可以设置验证码或其他验证方式，降低垃圾评论的风险。

# 4.代码实例及解释说明
## 4.1 创建评论表单
假设你的网站域名是 example.com，下面的示例代码演示了如何创建一个带有评论功能的 HTML 页面：

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <title>Example</title>
  </head>
  <body>
    <!-- 评论框 -->
    <div id="vcomments"></div>

    <!-- 引入 Valine 评论系统 -->
    <script src="//cdn.jsdelivr.net/npm/@valine/valine/dist/valine.min.js"></script>
    <script>
      new Valine({
        // 设置 appid (你创建的 APPID)
        appId: '<KEY>',

        // 设置路径 (可以不修改)
        appKey: '<KEY>',

        // 指定 el (也就是评论框的 selector)
        el: '#vcomments',

        // 指定语言
        language: 'zh-cn',

        // 是否开启头像
        avatar:'mm',

        // 是否需要昵称
        visitor: true,

        // 设置字段名
        fieldName: {
          // 作者名
          author: 'name',
          // 邮箱地址
          email: 'email',
          // 网址
          website: 'url'
        }
      })
    </script>
  </body>
</html>
```

这里，我们使用 jQuery 对评论表单进行简单的校验。如果用户输入的内容不符合要求，会弹出错误提示；否则，会触发提交事件，使用 Ajax 向 GitHub API 发起评论请求。完整的代码如下：

```javascript
$(document).ready(function() {

  $('#submit').click(function() {
    var name = $('input[name="author"]').val();
    var email = $('input[name="email"]').val();
    var url = $('input[name="website"]').val();
    if (!name ||!email ||!isValidEmail(email)) {
      alert('请输入正确的姓名和邮箱地址');
      return false;
    }
    $.ajax({
      type: "POST",
      url: "http://example.com/postComment",
      data: {'name': name, 'email': email, 'url': url},
      success: function(data){
        console.log("Success:" + JSON.stringify(data));
        $('#form')[0].reset();
      },
      error: function(){
        console.error("Failed to submit comment");
      }
    });
  });

  function isValidEmail(email) {
    var re = /\S+@\S+\.\S+/;
    return re.test(email);
  }
  
});
``` 

这里，我们通过 POST 请求提交评论数据，并重置表单，刷新页面。注意，这里的提交 URL 需要替换成你自己的服务器地址。

## 4.2 展示评论列表
同样地，我们可以通过 Ajax 请求得到评论数据，并用 jQuery 插入到页面中：

```javascript
$.getJSON("http://example.com/getComments", function(comments) {
  for (var i in comments) {
    var comment = comments[i];
    var tpl = [
      "<li>",
        "<h3>" + comment.name + "</h3>",
        "<p>" + comment.content + "</p>",
        "<small>Submitted on " + formatDate(comment.created_at) + "</small>",
      "</li>"
    ].join('');
    $("#comments").append($(tpl));
  }
});

function formatDate(timestamp) {
  var date = new Date(parseInt(timestamp));
  var months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  var year = date.getFullYear();
  var month = months[date.getMonth()];
  var day = date.getDate();
  return day + '-' + month + '-' + year;
}
``` 

这里，我们通过 GET 请求获取评论数据，并遍历它们，构造 HTML 片段插入到页面中。`formatDate()` 函数用来格式化日期字符串。

完整的代码可参考：https://github.com/xCss/Valine/blob/master/README-zh-CN.md