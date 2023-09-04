
作者：禅与计算机程序设计艺术                    

# 1.简介
  

HTML（超文本标记语言）是用于创建网页的一种标记语言。本文通过对HTML元素及其布局进行详细介绍，尝试阐述HTML编程知识、技巧、应用等方面的内容。
HTML是一门很古老的编程语言，它的诞生日期可以追溯到上个世纪90年代初。它的主要用途是作为网络文档的结构定义工具，目前已成为万维网世界中最重要的技术标准。HTML是一门基于标记语言的语言，结构上由一个个元素所构成，每个元素都具有独特的作用和属性。
HTML Elements是指HTML语言中的不同标签或元素，它分为几种类型：
- 块级元素：占据整个页面宽度，通常具有自带换行特性；
- 内联元素：只在一行内显示，不支持自带换行特性；
- 空元素：没有闭合标签；
- 容器元素：可以容纳其他HTML元素，如div、span等；
- 表单控件：用于收集和输入用户数据，如input、textarea、select等；
- HTML全局属性：应用于所有元素的属性，如id、class、style等。
HTML Elements的布局是通过CSS来实现的，它可以控制元素的大小、位置、颜色、背景色、边框、间距等样式。本文将会以最常用的HTML元素——段落、标题、链接、图片、列表、表格、表单等进行讲解。
# 2.Blocks
块级元素是指一组标签范围从开头到结尾都在视窗内完整显示的元素，如下所示：
```html
<h1>This is a Heading</h1>
<p>This is a paragraph.</p>
<ul>
    <li>Item 1</li>
    <li>Item 2</li>
    <li>Item 3</li>
</ul>
<table>...</table>
```
以上各个元素都是块级元素。注意这些元素在不同浏览器上的渲染效果可能略有差异。
## Block elements in Detail
### Headings (h1-h6)
`<h1>` to `<h6>` are headings for the document. The larger the number, the smaller the font size. They should be used sparingly as they define the overall structure of the page and there can only be one `<h1>` per page. It's recommended to use only `h2` or less on pages with more complex structures.
Example:
```html
<h1>Heading level 1</h1>
<h2>Heading level 2</h2>
<h3>Heading level 3</h3>
<h4>Heading level 4</h4>
<h5>Heading level 5</h5>
<h6>Heading level 6</h6>
```
### Paragraph (<p>)
The `<p>` tag defines a paragraph within a webpage. Generally, it contains text and optional inline tags like bold, italic, etc., which will be formatted according to CSS styles. A paragraph can have multiple lines and paragraphs may contain other block-level elements such as lists, images, tables, etc.
Example:
```html
<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed euismod ante et diam feugiat hendrerit vel at velit. Nam semper urna quis dui pharetra aliquam. Fusce consequat massa vitae nunc egestas pretium. </p>
```
### Horizontal rule (<hr>)
The `<hr>` tag creates a horizontal line that separates content vertically. It doesn't have any attributes and cannot contain any child elements.
Example:
```html
<p>Paragraph before the hr element.</p>
<hr>
<p>Paragraph after the hr element.</p>
```
### Preformatted Text (<pre>)
The `<pre>` tag preserves whitespace and newlines by wrapping them inside the tag. This makes code easier to read than using plain text formatting. Additionally, this element can also be styled using CSS to make it stand out from surrounding text.
Example:
```html
<pre><code>
  // Some example Javascript code
   function myFunction() {
       console.log("Hello World!");
   }

   myFunction();
</code></pre>
```
### Address Element (<address>)
The `<address>` tag specifies contact information for the author/owner of a document or an article. It usually includes the person's name, email address, physical address, phone number, etc. By default, browsers display this information alongside the rest of the content on the same line. However, you could add styling rules to move it to another location if needed.
Example:
```html
<header>
  <h1>My Website</h1>
  <nav>
    <ul>
      <li><a href="#">Home</a></li>
      <li><a href="#">About Us</a></li>
      <li><a href="#">Contact Us</a></li>
    </ul>
  </nav>
</header>
<main>
  <article>
    <h2>Welcome!</h2>
    <p>Thank you for visiting our website. We hope you find what you're looking for.</p>
    <footer>
      <address>
        Written by John Doe &lt;<EMAIL>&gt;.<br>
        Visit us at:<br>
        Example Street 123,<br>
        New York, NY 10001
      </address>
    </footer>
  </article>
</main>
```
### Blockquotes (<blockquote>)
The `<blockquote>` tag encloses long quotations. You can specify who said it by including an `<cite>` element inside the opening `<blockquote>` tag. Browsers usually format these quotes separately with indentation, but this can be customized using CSS. The closing `<blockquote>` tag must always come after the main quote text.
Example:
```html
<p>Someone once told me the world was gonna roll me, I ain't the sharpest tool in the shed.</p>
<blockquote cite="https://www.goodreads.com/author/quotes/56738">
    <p>I'm sorry Dave, I'm afraid I can't do that.</p>
</blockquote>
<p>Whenever someone finds something worth doing, they should take the time to work very hard at it.</p>
```
### Lists (<ul> and <ol>)
Lists are used to organize items into ordered or unordered groups. The `<ul>` tag defines an unordered list and the `<ol>` tag defines an ordered list. Both types of lists allow nested sublists by adding additional levels of nesting using the appropriate markup. Each item in a list should be wrapped in an `<li>` tag. Browsers render each list type differently, so choose carefully based on your needs.
Example:
```html
<ul>
  <li>Item 1</li>
  <li>Item 2</li>
  <li>Item 3
    <ul>
      <li>Subitem 1</li>
      <li>Subitem 2</li>
    </ul>
  </li>
  <li>Item 4</li>
</ul>

<ol>
  <li>First step</li>
  <li>Second step
    <ol>
      <li>Substep 1</li>
      <li>Substep 2</li>
      <li>Substep 3</li>
    </ol>
  </li>
  <li>Third step</li>
</ol>
```
### Definition List (<dl>)
A definition list (`<dl>`) consists of two parts - terms and descriptions. The term describes the thing being defined and appears in an `<dt>` tag while its description appears in a series of `<dd>` tags that follow. There can be multiple terms per definition group.
Example:
```html
<dl>
  <dt>Web Development</dt>
  <dd>- Programming Languages
  - Frameworks
  - Tools</dd>

  <dt>Graphic Design</dt>
  <dd>- Adobe Photoshop
  - Illustrator
  - InDesign</dd>
</dl>
```
### Division Element (<div>)
The `<div>` tag serves as a container for grouping elements together and applying CSS styles to all of them at once. It has no semantic meaning and can be used for anything from styling individual blocks to creating reusable components.
Example:
```html
<div class="container">
  <h1>Page Title</h1>
  <p>Page Content goes here</p>
</div>

<div class="card">
  <h2>Card Title</h2>
  <p>Card Content goes here</p>
</div>
```
# 3.Inline Elements
An inline element occupies only enough width to fit its content, whereas a block element takes up an entire line even when empty. Inline elements include most of the basic building blocks of HTML, such as links, emphasized text, images, spans, and others. Examples of inline elements include anchors (`<a>`), bold (`<b>`, `<strong>`), italics (`<i>`, `<em>`), and spans (`<span>`).
In contrast, examples of block-level elements include headings (`<h1>` through `<h6>`), paragraphs (`<p>`), divisions (`<div>`), and forms (`<form>`).
Here's some sample code to show how different elements would behave in different contexts:
```html
<p>Hello <a href="#">world</a>, how are <b>you</b>?</p>
<label for="name"><strong>Name:</strong></label>
<input type="text" id="name">
```