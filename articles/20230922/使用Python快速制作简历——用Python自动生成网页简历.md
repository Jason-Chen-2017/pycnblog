
作者：禅与计算机程序设计艺术                    

# 1.简介
  

首先给大家介绍一下这个项目的背景。我是一个软件工程师，在很多方面都比不上专业的机器学习、深度学习模型。但我想有一个地方可以展示自己的能力，分享一些与AI相关的内容。因此便萌生了制作一个简单的网页简历的想法。作为一个无聊的人类，我喜欢看小说和漫画，所以决定制作一份很简陋的网页简历。但后来发现没有用那么麻烦的东西，自己也不太会用，于是又动手实现了一套自动生成网页简历的工具。最后，我完成了这个工具，并且把它开源出来，希望能够帮助到其他需要制作简历的朋友。

在这篇文章中，我将介绍如何使用Python快速制作简历。整个过程分为如下几个步骤：

1. 安装依赖包
2. 生成模板文件
3. 数据填充进模板文件
4. 将HTML文档转换为PDF格式并保存
5. 更改字体样式及其他自定义设置

下面就让我们开始吧！
# 2. 基本概念术语说明
## 2.1 Python语言概述
Python 是一种高级编程语言，具有简洁、直观、可读性强，以及简单易用的特点。它的语法和 C++、Java、JavaScript 类似，可以在不同领域被应用。同时它还有很多第三方库支持，比如图像处理、数据分析等方面。

Python 应用广泛，比如云计算、web开发、科学计算、图像处理、游戏开发、人工智能等领域。Python 在国内还经常被用做数据分析工具，主要因为其易用、可靠、免费、跨平台等特点。

## 2.2 HTML、CSS、JavaScript概述
### 2.2.1 HTML(Hypertext Markup Language)
HTML (超文本标记语言) 是用于创建网页的标准标记语言。它用于定义页面的结构和内容。通过HTML文档，您可以创建网页的所有元素，如：文本、图片、表格、链接等。

### 2.2.2 CSS(Cascading Style Sheets)
CSS (层叠样式表) 是用于指定HTML文档的样式信息的语言。通过CSS，您可以控制网页的布局、颜色、字体、边框等外观效果。

### 2.2.3 JavaScript
JavaScript 是一种基于对象和事件驱动的动态脚本语言，它可以实现各种动态功能。在网页上实现各种动画效果、表单验证、富文本编辑器、地图接口、视频播放等功能时，JavaScript通常是必不可少的。

## 2.3 Markdown概述
Markdown 是一种轻量级的标记语言，它可以使用简单的纯文本语法编写文档。虽然 Markdown 比较简洁，但是它还是能实现复杂的排版。而且 GitHub 支持 Markdown 的语法，使得编写 README 文件更加方便。此外，很多网站和论坛也支持使用 Markdown 来撰写文章。

# 3.核心算法原理和具体操作步骤
## 3.1 安装依赖包
首先安装依赖包，pip install jinja2 html2pdf weasyprint pandas markdown

其中jinja2是python模板引擎，html2pdf用于转换html为pdf，weasyprint用于生成pdf，pandas用于csv读取，markdown用于将md文件转为html。

## 3.2 生成模板文件
创建templates文件夹，然后在该文件夹下创建一个template.html文件。

```html
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <title>{{ name }}</title>
  <link rel="stylesheet" href="{{ style }}">
</head>

<body>
  <div class="page">
    <!-- header -->
    {% block header %}
      <h1>{{ name }}</h1>
      <p>{{ title }}</p>
    {% endblock %}

    <!-- education section -->
    {% if 'education' in sections %}
      <h2>教育背景</h2>
      <ul>
        {% for school in educations %}
          <li><b>{{ school['degree'] }} : </b>{{ school['name'] }}, {{ school['start_year'] }} ~ {{ school['end_year'] }}</li>
        {% endfor %}
      </ul>
    {% endif %}

    <!-- experience section -->
    {% if 'experience' in sections %}
      <h2>工作经验</h2>
      <ul>
        {% for job in experiences %}
          <li><b>{{ job['position'] }} : </b>{{ job['company'] }}, {{ job['start_date'] }} ~
            {{ job['end_date'] }}</li>
        {% endfor %}
      </ul>

      <!-- projects section -->
      <h2>项目经验</h2>
      <ul>
        {% for project in projects %}
          <li><b>{{ project['name'] }} : </b>{{ project['description'] }}</li>
        {% endfor %}
      </ul>
    {% endif %}

    <!-- skill section -->
    {% if'skill' in sections %}
      <h2>技能清单</h2>
      <ul>
        {% for skill in skills %}
          <li>{{ skill }}</li>
        {% endfor %}
      </ul>
    {% endif %}

    <!-- footer -->
    {% block footer %}
      <hr />
      <small>&copy; {{ current_year }} {{ name }}</small>
    {% endblock %}
  </div>
</body>

</html>
```

## 3.3 数据填充进模板文件
我们需要填写的数据包括个人信息、教育背景、工作经历、项目经验、技能清单、Footer。在同级目录下的data.yaml文件里写入相应的字段即可。

```yaml
---
name: "张三" # 名字
current_year: "2021" # 年份
style: "style.css" # css文件名
title: "web前端工程师" # 职称
sections: [education, experience, skill] # 需要显示的段落
educations: 
  - degree: "本科"
    start_year: "2015"
    end_year: "2021"
    name: "中国科学院大学"
  - degree: "研究生"
    start_year: "2021"
    end_year: ""
    name: "上海交通大学"
experiences:
  - position: "Web前端开发"
    company: "深圳市文一软件股份有限公司"
    start_date: "2020/7"
    end_date: "至今"
  - position: "Python开发工程师"
    company: "北京京东方微电子股份有限公司"
    start_date: "2019/9"
    end_date: "2020/6"
  - position: "Python开发工程师"
    company: "北京优客逸宇信息科技股份有限公司"
    start_date: "2018/7"
    end_date: "2019/8"
projects:
  - name: "慕课商城"
    description: "慕课商城是一个在线商城系统，里面有许多实用的功能。用户可以通过手机、平板、PC访问。"
  - name: "智慧楼宇管理系统"
    description: "这是一款能够管理智慧楼宇的管理系统。可以统计各个区域的能耗，并对每月的用电、气用量进行分析。"
  - name: "沙箱环境监测系统"
    description: "这是一款能够实时的监测沙箱环境的系统。用户可以在线查看沙箱环境的温湿度、照度、VOC、甲醛等参数。"
skills: ["HTML", "CSS", "JavaScript", "jQuery", "Vue"] # 技能清单列表
footer: "<a href='mailto:<EMAIL>'>联系邮箱</a>" # 底部信息
...
```

## 3.4 将HTML文档转换为PDF格式并保存
导入`from weasyprint import HTML, CSS`, 创建`def generate_pdf()`方法，调用`HTML(string=html).write_pdf(target)`函数，传入模板文件的字符串以及输出路径即可保存为pdf格式。

```python
import os

from jinja2 import Template
from weasyprint import HTML, CSS


def generate_pdf():
    with open('data.yaml', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    template = get_template()
    rendered = template.render(**data)
    
    pdf_file_path = './output/' + data["name"] + '.pdf'
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    output_dir = os.path.join(dir_path, 'output/')
    
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
        
    HTML(string=rendered).write_pdf(pdf_file_path)


def get_template():
    path = os.path.join('./templates/', 'template.html')
    with open(path, encoding='utf-8') as file_:
        template = Template(file_.read())
    return template
```

## 3.5 更改字体样式及其他自定义设置
默认情况下，生成的简历字体样式可能无法满足你的需求。你可以根据自己的喜好添加字体样式或调整行距等参数。具体操作如下：

1. 修改字体样式：修改template.html中的<head>标签里面的css文件引用路径。
2. 添加自定义设置：你可以在generate_pdf()方法中添加自定义设置。举例来说，如果想要更改行距，可以在css文件中设置body的line-height属性，如：

```css
/* 设置行距 */
body {
  line-height: 1.6em; /* 1.6倍行距 */
}
```