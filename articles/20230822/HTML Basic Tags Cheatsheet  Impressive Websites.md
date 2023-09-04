
作者：禅与计算机程序设计艺术                    

# 1.简介
  

HTML(Hypertext Markup Language) 是用于创建网页的标记语言。它是一种建立网站最基础也是最重要的工具之一。因为 HTML 的简洁、结构化、语义化的特点，使其能够很好的呈现多种不同的网页样式。

作为程序员，很长时间内我都把精力集中在编写代码上，然而编写 HTML 页面却屡屡被打断。于是决定通过博客的方式记录下自己学习和工作中的常用 HTML 标签和 CSS 样式，将自己的知识分享给其他需要帮助的人。

本文将总结一些有趣且实用的 HTML 标签，并带领读者了解这些标签背后的原理和意义。

# 2.基本概念和术语
## 2.1 HTML、CSS、JavaScript
HTML（超文本标记语言）：HTML 是一种用来描述网页的标记语言。网页的内容主要由 HTML 元素组成，每个元素表示网页的一小块内容，如文字、图片、视频等。

CSS（层叠样式表）：CSS（Cascading Style Sheets）是一种用来表现 HTML 或者 XML 文档样式的语言。它允许Web设计人员直接控制网页的布局、配色、效果甚至功能。

JavaScript：JavaScript 是一种动态脚本语言，它可以实现网页的各种动态特性，包括响应用户交互、动画效果、表单验证、可视化组件等。

## 2.2 DOM（文档对象模型）
DOM（Document Object Model），即“文档对象模型”。它定义了处理网页内容的方法、接口及规则。每一个 HTML 文档都有一个 DOM 树，它包含了该文档的全部内容，每个节点都是 DOM 对象。

## 2.3 Tag 和 Element
Tag：HTML 中的标签指的是尖括号 < > 中的内容，如 <html> 或 <body> 。


# 3.常用标签
## 3.1 Meta tag
Meta tags are used to provide information about the web page such as keywords, description, author of the page and other relevant data that is not visible on the webpage itself. The meta tags can be added in the head section of an HTML document using the following syntax:

```html
<meta name="description" content="This is a sample website for learning HTML.">
```

Here, `name` attribute specifies what kind of metadata it is (in this case, description), while the `content` attribute provides its value (in this case, "This is a sample website for learning HTML.")

Another example of meta tag could be:

```html
<meta charset="UTF-8">
```

This tells the browser to interpret the text encoding of the current document using UTF-8 character set. It should always appear at the beginning of your `<head>` element before any other meta tags or links.

## 3.2 Headings
Headings are defined with the `<h1>` to `<h6>` tags. These elements have different sizes depending on how large you want them to be. You can use these tags to create titles, subtitles, and sections within your webpage. For example:

```html
<h1>Welcome to my Website</h1>
<h2>About Me</h2>
<p>My name is John Doe and I am a student.</p>
```

In this code snippet, we first add an H1 heading for our webpage title. Then we add an H2 heading underneath for our About me section. Finally, we use a paragraph tag (`<p>`) to add some introductory text to our About Me section. We can also combine headings with paragraphs, lists, images etc. to make more complex websites. 

Note: Do NOT skip the correct heading level as it will cause hierarchy issues which may affect accessibility. A good practice is to start from h1 and then increment by one level after each heading until reaching the desired size.

## 3.3 Paragraphs
Paragraphs are defined with the `<p>` tag. This tag creates a block of text that stands alone and is separated from surrounding text by space. For example:

```html
<p>Hello! My name is John Doe and I am a student.</p>
<p>I enjoy playing guitar, reading books, and traveling around the world!</p>
```

We can insert multiple paragraphs into a webpage using this tag. However, keep in mind to only use paragraphs where necessary since they take up valuable space on the screen. If possible, try to break down longer blocks of text into smaller chunks.

## 3.4 Images
Images are included in HTML documents using the `<img>` tag. There are two ways to include images:

1. **Using source URL:** To display an image, simply specify the path or URL of the image file in the `src` attribute of the `<img>` tag. Here's an example:

   ```html
   ```
   
2. **Using data URI scheme:** Data URIs allow us to embed small images directly inside the HTML code without having to rely on external files. They consist of base64-encoded strings representing images encoded according to specified MIME types. Here's an example:

   ```html
   ```

Either way, it's important to add an alternative text (or `alt` text) to describe the image for users who cannot see it. Additionally, we can control the dimensions and alignment of the image using CSS styles.

## 3.5 Links
Links are created with the `<a>` tag. This tag defines a hyperlink which points to another web page, file or location. An anchor link consists of three parts:

1. The opening `<a>` tag with the href attribute specifying the target URL:

   ```html
   <a href="https://www.google.com/">Google</a>
   ```
   
2. The visible text between the opening and closing tags:

   ```html
   Google
   ```
   
3. Optionally, nested tags for additional styling or functionality.

To open the linked document in a new tab or window, add the `target="_blank"` attribute to the `<a>` tag:

```html
<a href="https://www.example.com/" target="_blank">Visit Example Domain</a>
```

## 3.6 Lists
Lists are commonly used to organize related items. In HTML, there are four main list types: ordered, unordered, definition and descriptive. Each type has slightly different formatting requirements but all can be easily achieved with their respective tags.

**Ordered List**: Ordered lists are numbered with numbers or letters. To create an ordered list, use the `<ol>` tag followed by individual `<li>` tags containing the list items:

```html
<ol>
  <li>Item 1</li>
  <li>Item 2</li>
  <li>Item 3</li>
</ol>
```

The numbers can be customized using CSS stylesheets or attributes if needed. Note that the list items need to be contained within an ordered list tag.

**Unordered List**: Unordered lists contain bullet points. To create an unordered list, use the `<ul>` tag followed by individual `<li>` tags containing the list items:

```html
<ul>
  <li>Item 1</li>
  <li>Item 2</li>
  <li>Item 3</li>
</ul>
```

By default, bullet points are displayed as disc symbols. However, you can customize this symbol by adding CSS rules. Similarly to ordered lists, list items need to be contained within an unordered list tag.

**Definition List**: Definition lists associate a term with a description. To create a definition list, use the `<dl>` tag followed by `<dt>` (definition terms) and `<dd>` (definitions) pairs:

```html
<dl>
  <dt>Apple</dt>
  <dd>A fruit</dd>
  <dt>Banana</dt>
  <dd>A yellow fruit</dd>
</dl>
```

You can further customize the appearance of definitions using CSS. Similarly to unordered and ordered lists, the `<dt>` and `<dd>` elements need to be contained within a definition list tag.

**Description List**: Description lists provide a concise way to define key-value pairs. To create a description list, use the `<dl>` tag followed by `<dt>` (term) and `<dd>` (description) pairs:

```html
<dl>
  <dt>Name:</dt>
  <dd>John Smith</dd>
  <dt>Age:</dt>
  <dd>30 years old</dd>
  <dt>Occupation:</dt>
  <dd>Software Engineer</dd>
</dl>
```

Similar to definition lists, you can further customize the appearance of descriptions using CSS. Again, the `<dt>` and `<dd>` elements need to be contained within a description list tag.