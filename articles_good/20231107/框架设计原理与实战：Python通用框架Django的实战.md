
作者：禅与计算机程序设计艺术                    

# 1.背景介绍



作为一个具有经验丰富、职场经历丰富、工作积累丰富的技术人员，我相信每个技术人都曾深陷其中。无论是在解决具体的问题时，还是在架构某个项目、构建某款产品时，总会面临很多种技术上的问题。特别是对于初级技术人员来说，往往并不清楚各项技术背后的原理，只是知道如何去应用，很难真正理解其中的道理。而如果我们站在巨人的肩膀上，就能够把握全局，从而提升自己的能力，达到事半功倍的效果。因此，掌握某个领域的技术核心原理，对我们作为技术人员的进步至关重要。

作为Django的缔造者，它是一个非常著名的Python web框架，它的简洁、灵活、优雅让开发者的编码效率得到了很大的提升。但是，有时候我们也会觉得这个框架太过简单了，无法完全满足需求，甚至有些功能实现起来也比较复杂。那么，在这种情况下，我们该怎么办呢？有没有什么工具或方法能够帮助我们快速了解某个技术或框架背后所涉及的核心算法或原理，从而更加高效地利用这些技术呢？另外，除了文档和官方资料之外，是否还有其他更有效的方式呢？

基于以上原因，我相信了解各个Python技术框架背后的原理，对于我们进行技术选型、架构设计等工作都将产生深远的影响。所以，本文将通过分享个人学习研究的一些心得体会，介绍Django技术框架的核心算法和机制，希望能帮助读者在实际工作中更好地运用此框架。同时，也希望能够引起广泛的反响，促进更多技术人员的参与共建，共同推动Python技术的进步。


# 2.核心概念与联系

## 2.1 Python语言概述

Python是一种面向对象的动态脚本语言，被设计用来编写可维护的代码，并且易于学习和使用。Python语法类似于C++和Java，但又有自己独有的特性。它支持多种编程范式，包括面向对象编程、函数式编程和命令式编程。在使用过程中，可以将其解释器集成到各种应用程序中，用来创建丰富的数据处理程序。

## 2.2 Django概述


Django框架的主要组成包括：

- 模板层（Template layer）：Django使用模板系统（template system），这种系统使得Web页面的内容和布局可以根据用户输入生成。不同于传统的网页制作工具（比如Word或Adobe Dreamweaver），用户可以专注于页面内容的设计。模板系统通常使用HTML标记语言（HTML markup language）来定义页面内容，然后使用模板语言来填充标签，生成最终的网页。
- URL层（URL routing layer）：Django提供了路由功能，让用户可以自定义URL。用户可以定义各种URL规则，用以映射到相应的视图（view）。当用户访问特定URL时，Django将调用相应的视图，并渲染生成响应。
- 视图层（View layer）：视图层是Django应用的主要部分。它接收HTTP请求并返回HTTP响应，即显示给用户的内容。在Django中，视图通常都是类或者函数，它们负责处理用户的请求并返回相应的结果。
- ORM层（Object-relational mapping layer）：Django提供了一个ORM（object-relational mapping）系统，它将关系数据库表和对象数据类型转换成另一种形式，从而使开发者可以方便地访问数据库记录。
- Forms层（Forms layer）：Django提供了表单处理功能，它可以自动生成表单，将用户提交的数据绑定到对象上，并验证数据的合法性。
- 测试层（Test layer）：Django内置测试模块，它可以执行单元测试、功能测试和回归测试，帮助开发者发现代码错误。

总结一下，Django框架是一个面向对象的Web框架，使用Python语言开发，为开发者提供最基本的MVC模型和其他组件。Django框架的所有层都可以独立使用，也可以组合使用。

## 2.3 MVC模式

MVC模式（Model View Controller）是一个经典的Web应用程序设计模式。该模式描述了用户界面与应用程序逻辑之间的分离。模型（Model）代表数据，视图（View）代表用户界面，控制器（Controller）负责处理业务逻辑。使用MVC模式，Web应用程序的各部分可以轻松重用，修改和扩展。

- Model：它负责管理数据，存储和检索数据，模型数据可以通过多种方式呈现给用户。例如，数据库模型就是一个例子，它可以保存数据库记录和关系。Django框架使用ORM（object-relational mapping）将数据库映射到对象数据类型，这样就可以方便地查询和修改数据。
- View：它负责呈现给用户的用户界面。视图从模型获取数据，可以使用不同的格式，如XML、JSON、HTML、文本文件等，还可以包含图片、视频、音频、动画等媒体资源。Django框架使用模板系统将模型数据渲染成视图，并发送给浏览器。
- Controller：它负责处理客户端请求，通常它接收来自用户的请求，调度模型和视图进行通信，并响应用户的请求。Django框架采用WSGI（Web Server Gateway Interface）协议，可以与Web服务器集成。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解

## 3.1 请求处理流程

Django的请求处理流程如下图所示：


1. 当用户发出请求时，域名解析过程会将域名解析成IP地址；
2. Django接收到请求后，通过URLconf查找对应的处理函数，生成request对象；
3. Django调用中间件预处理中间件，对request进行预处理；
4. Django加载视图，对request进行处理，并生成response对象；
5. Django调用中间件后处理中间件，对response进行后处理；
6. Django将response返回给客户端。

## 3.2 URL配置

在Django中，URL配置指的是将用户请求的URL映射到指定的处理函数。Django提供的url()函数可以定义URL映射规则，该函数的语法如下：

```python
from django.urls import path

urlpatterns = [
    path('urlpattern/', viewfunction, name='optionalname'),
   ...
]
```

path()函数的第一个参数是URL模式，第二个参数是指向视图函数的路径字符串，第三个参数是可选项，用于指定名称。

举例如下：

```python
from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return HttpResponse("Hello, world!")

def hello(request, name):
    content = "Hello, %s!" % name
    return HttpResponse(content)

urlpatterns = [
    path('', index),    # root URL maps to the 'index' function
    path('hello/<str:name>/', hello),   # maps a URL with variable named 'name' to the 'hello' function
]
```

上面代码定义两个视图函数，分别处理根目录和带变量的URL请求。其中，index()函数直接返回"Hello, world!"。hello()函数接收名为'name'的GET参数，并输出greeting消息。

## 3.3 请求与响应对象

Django中，每一次请求都会生成一个request对象，该对象封装了用户的HTTP请求信息。当视图函数处理完请求之后，就会生成一个response对象，该对象封装了请求处理的结果。

### Request对象属性

Request对象包含以下属性：

- method：表示请求的方法，比如GET、POST等。
- scheme：表示协议方案，比如HTTP、HTTPS。
- get：是一个字典，用于获取GET请求的参数。
- post：是一个字典，用于获取POST请求的参数。
- FILES：是一个字典，用于获取上传的文件。
- COOKIES：是一个字典，用于获取cookie的值。
- session：一个属性，用于获取当前会话对象。
- META：是一个字典，用于获取一些元信息，比如HTTP头部信息。

### Response对象属性

Response对象包含以下属性：

- status_code：表示响应状态码，比如200 OK、404 Not Found等。
- content：表示响应内容，可能是字节流或字符串，取决于视图函数的返回值。
- headers：是一个字典，用于设置HTTP头部信息。
- cookies：是一个字典，用于设置cookie值。

## 3.4 会话对象

Session对象是Django提供的一个特殊对象，用于储存用户相关的数据。它可以在不同请求间保持一致，并且可以跨越多个请求持久化存储。

在视图函数中，可以通过request对象的session属性获取当前会话对象，该对象是一个字典，可以用来储存任意数据。可以设置和读取会话数据，也可以删除会话数据。

举例如下：

```python
from django.contrib.sessions.models import Session

def login(request):
    username = request.POST['username']
    password = request.POST['password']
    
    user = authenticate(username=username, password=password)
    if not user:
        return HttpResponseForbidden('Invalid credentials')

    # create new session or retrieve existing one based on user ID
    if request.user.is_authenticated():
        session = Session.objects.get(session_key=request.session.session_key)
        session.expire_date = timezone.now() + timedelta(days=1)
        session.save()
    else:
        request.session.set_expiry(timedelta(hours=1))
        request.session['username'] = username
        
    return redirect('/')
    
def logout(request):
    engine = import_module(settings.SESSION_ENGINE)
    sessions = Session.objects.filter(expire_date__lt=timezone.now())
    for session in sessions:
        engine.SessionStore(session.session_key).delete()
    
    request.session.flush()
    return redirect('/login/')
```

在上面的代码中，login()函数接收用户名和密码，并尝试验证用户身份。如果验证成功，则更新会话超时时间，并跳转到主页；否则，返回403错误。logout()函数清除所有过期的会话，并跳转到登录页面。

## 3.5 模板系统

Django中使用模板系统对网页内容进行渲染。模板系统允许开发者将HTML内容与动态内容分离，并在运行时将动态内容插入到模板中。

模板系统使用模板语言，该语言类似于Django的语法，可以嵌入Python表达式。在Django模板中，可以使用{%...%}语句包含控制结构和模板标签。模板标签可以控制模板的行为，比如条件判断和循环。

举例如下：

```html
<!DOCTYPE html>
<html>
  <head>
    {% block head %}
      <title>{{ title }}</title>
    {% endblock %}
  </head>

  <body>
    {% block body %}
      <h1>{{ header }}</h1>
      {{ message }}

      {% for item in items %}
        <li>{{ item }}</li>
      {% endfor %}
    {% endblock %}
  </body>
</html>
```

在上面的示例中，head和body是两个块标签，对应于HTML的head和body元素。title和header是两个变量，message是渲染后的值，items是一个列表。block标签可以包含一些默认内容，如果需要的话，可以通过重载父模板的方式来覆盖。

为了使用模板，我们需要创建一个模板的目录结构，并添加模板文件。在模板中，可以使用过滤器和上下文变量来修改变量的值。

举例如下：

views.py:

```python
from django.shortcuts import render

def myview(request):
    context = {
        'title': 'My Page',
        'header': 'Welcome!',
       'message': 'This is some dynamic text.',
        'items': ['apple', 'banana', 'cherry'],
    }
    return render(request,'mytemplate.html', context)
```

mytemplate.html:

```html
<!DOCTYPE html>
<html>
  <head>
    <title>{{ title|upper }}</title>
  </head>

  <body>
    <h1>{{ header|capfirst }}</h1>
    {{ message|linebreaksbr }}

    <ul>
      {% for item in items %}
        <li>{{ item }}</li>
      {% endfor %}
    </ul>
  </body>
</html>
```

在上面的示例中，我们通过传递一个上下文字典给render()函数来渲染模板。上下文变量title、header、message和items会被模板文件中的变量替换。title变量通过过滤器upper变为全大写字母，header变量通过过滤器capfirst首字母大写。message变量通过过滤器linebreaksbr将换行符转义为HTML的换行标签。

## 3.6 文件上传

在Django中，文件上传是通过表单实现的。表单可以包含文件字段，用户可以在浏览器上传文件。Django接受上传的文件后，会将文件保存到本地文件系统中。

上传的文件保存在django工程的media目录下。如果要修改上传文件的目录，需要设置MEDIA_ROOT参数。

举例如下：

forms.py:

```python
from django import forms
from django.core.files.storage import FileSystemStorage


class UploadFileForm(forms.Form):
    file = forms.FileField(label='选择文件', widget=forms.ClearableFileInput(attrs={'multiple': True}), help_text='支持多文件上传')


fs = FileSystemStorage(location='/tmp')  # 设置上传文件的临时目录


def handle_uploaded_file(f):
    filename = fs.save(f.name, f)  # 将上传的文件保存到临时目录
    uploaded_file_url = fs.url(filename)  # 获取上传的文件的URL路径
    return uploaded_file_url
```

views.py:

```python
from.forms import UploadFileForm
from django.shortcuts import render
from django.http import JsonResponse


def upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)

        if form.is_valid():
            for file in request.FILES.getlist('file'):
                uploaded_file_url = handle_uploaded_file(file)

            response = {'status':'success', 'data': {'url': uploaded_file_url}}
            return JsonResponse(response)
        else:
            response = {'status': 'error', 'errors': form.errors}
            return JsonResponse(response)
    else:
        form = UploadFileForm()

    return render(request, 'upload.html', {'form': form})
```

templates/upload.html:

```html
{% extends 'base.html' %}

{% load static %}

{% block content %}
  <div class="container">
    <div class="row">
      <div class="col-md-offset-3 col-md-6 well">
        <legend><center>文件上传</center></legend>
          <form id="upload-form" action="{% url 'upload' %}" enctype="multipart/form-data" method="post">
              {% csrf_token %}
              {{ form.as_p }}
              <button type="submit" class="btn btn-primary">上传</button>
          </form>
      </div>
    </div>
  </div>
{% endblock %}
```

在上面的示例中，我们定义了一个UploadFileForm表单，包含一个文件字段。文件上传后，我们保存上传的文件到本地临时目录并返回链接。前端展示上传表单，上传完成后，表单自动刷新并显示上传的文件链接。

## 3.7 CSRF防护

CSRF（Cross-site request forgery）跨站请求伪造是一种常用的攻击手段。攻击者通过伪装成受害者，向服务器发送恶意请求，冒充受害者的身份，盗取用户的敏感信息。Django提供CSRF防护功能，可以有效抵御CSRF攻击。

CSRF防护通过以下几点实现：

- 生成随机的CSRFToken。
- 在Cookie中发送CSRFToken。
- 通过POST请求时包含CSRFToken。
- 检查CSRFToken的正确性。

举例如下：

csrf_exempt装饰器用于排除某些不需要做CSRF检查的视图函数，并关闭CSRF防护功能。

settings.py:

```python
MIDDLEWARE += [
    'django.middleware.csrf.CsrfViewMiddleware',
]

CSRF_COOKIE_SECURE = True
CSRF_TRUSTED_ORIGINS = ['example.com']  # 允许来自指定域名的请求
```

views.py:

```python
@csrf_exempt
def update(request, pk):
    article = Article.objects.get(pk=pk)

    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        article.title = data.get('title')
        article.content = data.get('content')
        article.save()
        
        response = {'status':'success'}
        return JsonResponse(response)
    else:
        response = {'status': 'error','msg': 'Method not allowed.'}
        return JsonResponse(response)
```

在上面的示例中，update()函数未使用csrf_exempt装饰器，因此CSRF防护功能生效。

## 3.8 API接口

API（Application Programming Interface）应用程序编程接口是面向其它计算机软件与硬件的通信接口。在Django中，可以通过DRF（Django Rest Framework）、Tastypie等框架构建RESTful API。DRF提供了基于类的视图、序列化器等功能，可以快速构建RESTful API。

举例如下：

serializer.py:

```python
from rest_framework import serializers
from.models import Book


class BookSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Book
        fields = '__all__'
```

views.py:

```python
from rest_framework import generics
from.models import Book
from.serializer import BookSerializer


class BookListCreateView(generics.ListCreateAPIView):
    queryset = Book.objects.all()
    serializer_class = BookSerializer


class BookRetrieveUpdateDeleteView(generics.RetrieveUpdateDestroyAPIView):
    lookup_field = 'id'
    queryset = Book.objects.all()
    serializer_class = BookSerializer
```

在上面的示例中，我们定义了Book模型的序列化器，并定义了API的两个视图，分别对应Book模型列表和详情的操作。通过URL配置，我们可以将API暴露给外部系统。

# 4.具体代码实例和详细解释说明

## 4.1 安装与部署

安装Django环境：

```bash
pip install django==3.0.5
```

启动web服务器：

```bash
python manage.py runserver
```

打开浏览器，输入http://localhost:8000，如果看到欢迎页，则说明Django安装成功。

## 4.2 配置Django

配置文件是Django项目的核心。Django项目的配置文件一般放在project_name/settings.py文件中。

### 修改DEBUG设置

如果DEBUG设置为True，Django会输出错误堆栈信息，并且自动重新加载修改后的代码。如果设置为False，Django不会输出错误堆栈信息，并且禁止自动重新加载修改后的代码。

```python
DEBUG = True
```

### 修改SECRET_KEY设置

SECRET_KEY是一个安全密钥，用于加密session和CSRF令牌等数据。为了确保安全，务必将SECRET_KEY设置改为一个复杂且独一无二的值。

```python
SECRET_KEY = '<KEY>'
```

### 修改ALLOWED_HOSTS设置

ALLOWED_HOSTS是一个列表，用于指定哪些域名可以访问当前Django项目。默认为['localhost', '127.0.0.1']。

```python
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '*.<your domain>.com']
```

### 指定静态文件路径

STATIC_URL是静态文件的URL前缀，MEDIA_URL是媒体文件的URL前缀。我们可以通过STATICFILES_DIRS指定静态文件路径，MEDIA_ROOT指定媒体文件路径。

```python
STATIC_URL = '/static/'
STATICFILES_DIRS = (
    os.path.join(BASE_DIR,'static'),
)

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR,'media')
```

### 指定数据库连接

DATABASES是Django项目的数据库连接配置。

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME':'mydatabase',
        'USER': 'root',
        'PASSWORD': '',
        'HOST': 'localhost',
        'PORT': ''
    }
}
```

### 添加中间件

中间件（Middleware）是Django项目的功能组件，负责处理HTTP请求。我们可以自定义中间件来处理请求前后发生的事件。

```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
```

### 创建管理员账号

```python
from django.contrib.auth.models import User

admin = User.objects.create_superuser('admin', '<EMAIL>', '<PASSWORD>')
```

### 开启缓存

Django提供了缓存机制，可以减少数据库查询次数，提高性能。

首先，我们需要安装缓存依赖包：

```bash
pip install redis hiredis
```

然后，我们可以添加缓存配置：

```python
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION':'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'KEY_PREFIX':'myproject',  # 可选，默认为'django_cache'
    }
}

INSTALLED_APPS = [
   ...,
    'django_extensions',  # 用于检查缓存命中率
]
```

最后，我们在视图函数中引入缓存功能：

```python
from django.core.cache import cache

def test_cache(request):
    key = 'test_cache'
    value = cache.get(key)
    if value is None:
        value = 'Hello, World!'
        cache.set(key, value, timeout=30)
    return HttpResponse(value)
```

## 4.3 编写Django应用

Django项目是由一系列的Django应用构成。每个应用是一个python模块，包含一系列的Django组件，如模型、视图、模板、表单、插件等。

我们可以用manage.py命令行工具创建新的应用：

```bash
python manage.py startapp app_name
```

创建好应用后，我们需要修改配置文件，将新应用加入到INSTALLED_APPS列表中：

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'app_name',  # 新增的应用
]
```

在新应用下，我们可以创建models.py、admin.py、forms.py、views.py、urls.py等文件，编写Django应用。

### models.py

models.py文件定义了Django模型。

```python
from django.db import models


class MyModel(models.Model):
    field1 = models.CharField(max_length=100)
    field2 = models.IntegerField()

    def __str__(self):
        return self.field1
```

### admin.py

admin.py文件用于注册Django模型到后台管理页面。

```python
from django.contrib import admin
from.models import MyModel


class MyModelAdmin(admin.ModelAdmin):
    list_display = ('field1', 'field2')


admin.site.register(MyModel, MyModelAdmin)
```

### forms.py

forms.py文件定义了表单。

```python
from django import forms
from.models import MyModel


class MyModelForm(forms.ModelForm):
    class Meta:
        model = MyModel
        fields = ('field1', 'field2')
```

### views.py

views.py文件定义了视图函数。

```python
from django.shortcuts import render
from django.http import HttpResponse
from.models import MyModel
from.forms import MyModelForm


def index(request):
    obj_list = MyModel.objects.all()
    return render(request, 'app_name/index.html', {'obj_list': obj_list})


def add(request):
    if request.method == 'POST':
        form = MyModelForm(request.POST)
        if form.is_valid():
            instance = form.save()
            return HttpResponse('Success!')
    else:
        form = MyModelForm()

    return render(request, 'app_name/add.html', {'form': form})
```

### urls.py

urls.py文件定义了应用的URL配置。

```python
from django.urls import path
from.views import index, add


urlpatterns = [
    path('', index, name='index'),
    path('add/', add, name='add'),
]
```

### templates/app_name/index.html

```html
{% extends 'base.html' %}

{% block content %}
  <table border="1">
    <tr>
      <th>Field1</th>
      <th>Field2</th>
    </tr>
    {% for obj in obj_list %}
      <tr>
        <td>{{ obj.field1 }}</td>
        <td>{{ obj.field2 }}</td>
      </tr>
    {% empty %}
      <tr>
        <td colspan="2">No objects.</td>
      </tr>
    {% endfor %}
  </table>
{% endblock %}
```

### templates/app_name/add.html

```html
{% extends 'base.html' %}

{% block content %}
  <form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <input type="submit" value="Submit">
  </form>
{% endblock %}
```

## 4.4 使用Django模板语言

Django的模板语言是Django内置的模板引擎，使用Django模板语言，我们可以方便地在HTML页面中嵌入变量、条件判断语句、循环语句等功能。Django模板语言的语法与Python语言非常相似。

Django的模板文件以.html为后缀，可以包含多种类型的模板代码，如Django模板语言代码、HTML代码、CSS代码、JavaScript代码等。模板语言中支持的注释语法为{#... #}，可以使用缩进来表示代码块。

### 模板继承

Django模板支持模板继承，子模板可以继承父模板的布局样式和部分内容，从而提升代码复用率。

```html
<!-- base.html -->
<html>
  <head>
    <meta charset="UTF-8">
    <title>{% block title %}{% endblock %}</title>
  </head>
  <body>
    <nav>
      <a href="/">Home</a> | 
      <a href="/about/">About</a>
    </nav>
    <hr>
    {% block content %}{% endblock %}
  </body>
</html>
```

```html
<!-- child.html -->
{% extends 'base.html' %}

{% block title %}Child page{% endblock %}

{% block content %}
  <h1>Welcome!</h1>
  <p>Lorem ipsum dolor sit amet...</p>
{% endblock %}
```

### 变量

Django模板语言支持两种类型的变量：预定义变量和动态变量。

预定义变量是指一些固定值，比如常量、数据结构、字符串等。预定义变量可以使用{{ predefined_variable }}语法引用。

动态变量是指在模板渲染过程中才计算出的变量，比如模板变量、请求变量、查询字符串参数等。动态变量可以使用{% templatetag openvariable %}语法引用。

```html
<!-- index.html -->
{% extends 'base.html' %}

{% block title %}Index{% endblock %}

{% block content %}
  <h1>Homepage</h1>
  <ul>
    {% for product in products %}
      <li>{{ product.name }}, ${{ product.price }}</li>
    {% endfor %}
  </ul>
  Total price: $ {{ total_price }}
{% endblock %}
```

```html
<!-- detail.html -->
{% extends 'base.html' %}

{% block title %}Detail{% endblock %}

{% block content %}
  <h1>{{ object.name }}</h1>
  Price: $ {{ object.price }}
  Description: {{ object.description }}
{% endblock %}
```

### 条件判断

Django模板语言支持if/elif/else条件判断语句，可以根据条件判断是否渲染某些代码。

```html
{% if condition %}
  <!-- true code -->
{% elif condition2 %}
  <!-- false code -->
{% else %}
  <!-- other code -->
{% endif %}
```

### 循环

Django模板语言支持for/empty/else循环语句，可以遍历序列和其他可迭代对象，并渲染出内容。

```html
{% for var in sequence %}
  <!-- loop content -->
{% empty %}
  <!-- when no elements are available -->
{% endfor %}
```

### 函数

Django模板语言支持自定义函数，可以在模板中复用。

```python
@register.simple_tag
def multiply(x, y):
    return x * y
```

```html
<!-- index.html -->
{% extends 'base.html' %}

{% block title %}Multiply{% endblock %}

{% block content %}
  Result: {{ multiply(2, 3) }}
{% endblock %}
```