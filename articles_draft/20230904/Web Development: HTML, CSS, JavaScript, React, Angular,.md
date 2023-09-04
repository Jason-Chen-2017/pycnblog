
作者：禅与计算机程序设计艺术                    

# 1.简介
  

HTML, CSS, 和 JavaScript 是构建网页的基石。React 和 Angular 都是用于构建用户界面的库；VueJS 则是一个轻量级的前端框架，适合快速开发复杂单页应用（SPA）。Node JS 是一个基于 Chrome V8 引擎的JavaScript运行环境。如果要开发网站，那么这些技术都必不可少。本文将通过对其进行介绍，帮助读者理解并掌握它们。

# 2.HTML
## 2.1 什么是HTML？
HTML (Hypertext Markup Language) 是一种用来创建网页的标记语言。它定义了网页的内容结构、文本格式和链接关系等信息。网页中的所有内容均由HTML标签构成，如<body>、<head>、<title>、<h1>到<img>等。HTML页面由多种标签嵌套而成，使得网页具有层次性、表现力和互动性。HTML5是HTML的最新版本。

## 2.2 HTML5新特性
HTML5引入了许多新特性，包括语义化标签、本地存储、音频和视频播放、拖放功能、websocket协议等。其中语义化标签的重要意义不用我说，其他的则各有千秋。

## 2.3 HTML文档类型
HTML文档可以分为两类：过渡型文档类型（ Transitional Document Type）和严格型文档类型（ Strict Document Type）。

### 2.3.1 过渡型文档类型（ Transitional Document Type ）
过渡型文档类型是 HTML4 及更早版本使用的文档类型。它提供了较宽松的错误处理机制，适用于需要向后兼容的场景。一般情况下，推荐使用这种文档类型。

```html
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
```

### 2.3.2 普通型文档类型（ Normal Document Type ）
普通型文档类型是 HTML5 使用的文档类型。它提供较严格的错误处理机制，一般推荐使用该文档类型。

```html
<!DOCTYPE html>
```

### 2.3.3 混杂型文档类型（ Mixed Document Type ）
混杂型文档类型混合了普通型文档类型的一些特征和过渡型文档类型的一些特性。建议只在开发特定的老旧项目时使用。

```html
<!DOCTYPE html SYSTEM "about:legacy-compat">
<!--... -->
```

## 2.4 HTML常用标签汇总
HTML常用的标签如下所示：

- `<a>` - 创建超链接
- `<abbr>` - 缩写词或首字母缩略词
- `<address>` - 联系信息
- `<article>` - 文章内容
- `<aside>` - 侧边栏内容
- `<audio>` - 添加音频内容
- `<b>` - 粗体文字
- `<base>` - 指定页面基本 URL
- `<blockquote>` - 引用块
- `<body>` - 文档主体
- `<br>` - 换行符
- `<button>` - 创建按钮
- `<canvas>` - 在网页上绘制图形
- `<caption>` - 表格标题
- `<cite>` - 作品名称
- `<code>` - 计算机代码
- `<col>` - 为 table 定义列集
- `<colgroup>` - 为 table 定义组
- `<data>` - 数据列表
- `<datalist>` - 自动补全输入控件
- `<dd>` - 描述列表项目
- `<del>` - 删除线
- `<details>` - 折叠内容
- `<dfn>` - 定义一个术语
- `<dialog>` - 对话框
- `<div>` - 创建一个division(区块)
- `<dl>` - 描述列表
- `<dt>` - 描述列表标题
- `<em>` - 强调文字
- `<embed>` - 嵌入外部内容
- `<fieldset>` - 将表单元素分组
- `<figcaption>` - 插入图像题注
- `<figure>` - 插入图片描述
- `<footer>` - 页脚
- `<form>` - 创建一个表单
- `<frame>` - 子窗口
- `<frameset>` - 设置窗口框架
- `<h1>-<h6>` - 标题标签
- `<header>` - 页眉
- `<hr>` - 分割线
- `<i>` - 斜体文字
- `<iframe>` - 插入带有位置与尺寸的内联框架
- `<img>` - 插入图片
- `<input>` - 创建输入控件
- `<ins>` - 插入文本
- `<kbd>` - 表示键盘输入
- `<label>` - 为 input 元素添加标签
- `<legend>` - 为 fieldset 元素添加标题
- `<li>` - 描述列表项
- `<link>` - 链接外部文件
- `<main>` - 主要内容区域
- `<map>` - 定义客户端图像映射
- `<mark>` - 突出显示文本
- `<menu>` - 菜单列表
- `<meta>` - 描述网页元数据
- `<meter>` - 显示比例信息
- `<nav>` - 导航链接
- `<noscript>` - 当浏览器禁用脚本时，显示备用内容
- `<object>` - 插入外部资源
- `<ol>` - 有序列表
- `<optgroup>` - 将选项分组
- `<option>` - 下拉列表选项
- `<output>` - 显示计算结果
- `<p>` - 段落
- `<param>` - 描述对象参数
- `<picture>` - 提供多种不同源文件的不同版本
- `<pre>` - 预格式化文本
- `<progress>` - 显示进度信息
- `<q>` - 短引用
- `<rp>` - 替代中文输出
- `<rt>` - 提供对东亚文字的注释
- `<ruby>` - 支持日语、韩语等排版
- `<s>` - 中划线文字
- `<samp>` - 计算机代码示例
- `<script>` - 插入脚本
- `<section>` - 分段落或节
- `<select>` - 选择列表
- `<small>` - 小号字体
- `<source>` - 为媒体元素指定媒体资源
- `<span>` - 可变的容器标签
- `<strike>` - 中划线文字
- `<strong>` - 加粗文字
- `<style>` - 插入样式表
- `<sub>` - 上标文字
- `<summary>` - 详情元素的标题
- `<sup>` - 下标文字
- `<table>` - 创建表格
- `<tbody>` - 描述表格主体
- `<td>` - 描述单元格的数据
- `<template>` - 描述抽象内容
- `<textarea>` - 创建多行文本输入控件
- `<tfoot>` - 描述表格尾部
- `<th>` - 描述表头单元格
- `<thead>` - 描述表格头部
- `<time>` - 描述时间、日期或者时间间隔
- `<title>` - 网页标题
- `<tr>` - 描述表格行
- `<track>` - 为 media elements 指定字幕
- `<tt>` - 打字机文本
- `<u>` - 下划线文字
- `<ul>` - 无序列表
- `<var>` - 变量
- `<video>` - 添加视频内容