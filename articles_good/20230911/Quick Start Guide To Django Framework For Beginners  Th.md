
作者：禅与计算机程序设计艺术                    

# 1.简介
  
：Django是一个Python Web框架，它允许快速、简单地开发功能强大的网站。本教程将教会您如何从零开始安装和设置Django，创建您的第一个Web应用程序，并部署到云服务器。本指南适用于所有对Python编程感兴趣的人。

在开始学习Django之前，您需要做好以下准备工作：

1. 安装Python环境：您需要安装Python环境才能运行Django。推荐安装Anaconda Python 发行版，它是一个基于开源Python的科学计算平台。Anaconda提供了许多方便安装使用的包，包括Django等。您可以在其官网下载并安装Anaconda。

2. 安装virtualenv：如果您没有Python环境的管理工具virtualenv，则可以通过pip命令安装virtualenv：

   ```
   pip install virtualenv
   ```

3. 创建一个新的虚拟环境：创建并进入一个新目录，然后创建一个名为venv的文件夹，用于存放虚拟环境：

   ```
   mkdir myproject
   cd myproject
   virtualenv venv
   ```

   如果要退出虚拟环境，只需关闭终端窗口即可。如果要重新打开，可以用如下命令：

   ```
   source venv/bin/activate
   ```

4. 安装Django：安装Django有两种方式。第一种是通过conda安装：

   ```
   conda install django
   ```

   第二种是通过pip安装：

   ```
   pip install django
   ```

   有些Linux发行版还提供预编译好的包供下载，比如CentOS上可以使用yum或apt-get命令安装：

   ```
   yum install python3-django
   ```

   或

   ```
   apt-get install python3-django
   ```

   

5. 设置项目：设置项目前，需要先创建一个文件夹用于存放项目文件。在终端中输入以下命令：

   ```
   django-admin startproject mysite.
   ```

   此命令会在当前目录下创建一个名为mysite的文件夹，其中包含默认的配置文件和Django所需的所有文件。

   在这个文件夹里，有一个叫做manage.py的文件，它是Django的命令行工具，让我们能够运行各种各样的Django命令。使用以下命令运行服务器：

   ```
   python manage.py runserver
   ```

   这条命令会启动服务器，并监听本地的8000端口。在浏览器里访问http://localhost:8000，就可以看到默认的欢迎页面。

   恭喜！您已经成功地完成了Django的安装配置。接下来，您将学习如何创建Web应用程序、添加数据模型、定义路由、编写视图函数以及实现登录系统。

## 创建第一个Web应用程序

现在，假设你已经成功地安装了Django，接下来我们就要创建自己的第一个Web应用程序了。

1. 创建应用：创建一个新的子目录apps，用来存放你的Web应用程序。在终端中输入以下命令：

   ```
   python manage.py startapp polls
   ```

   此命令会在apps目录下创建一个名为polls的文件夹，其中包含三个Python模块：urls.py、views.py和models.py。这些模块将构成我们的Web应用程序。

2. 配置URL映射：在polls目录下编辑urls.py文件，添加以下代码：

   ```python
   from django.urls import path
   
   from. import views
   
   
   urlpatterns = [
       path('', views.index, name='index'),
   ]
   ```

   此代码导入path函数，并从.（当前目录）导入views模块。urlpatterns列表中的每一项都代表一条URL映射。第一个元素''表示根URL，第二个元素'index'是由views.index函数处理的请求，最后一个元素name='index'是URL别名。

3. 编写视图函数：编辑views.py文件，添加以下代码：

   ```python
   from django.shortcuts import render
   
   def index(request):
       return render(request, 'polls/index.html')
   ```

   此代码导入render函数，该函数接收两个参数，第一个参数是请求对象，第二个参数是模板文件的路径。函数返回渲染后的响应。

4. 添加HTML模板：在templates目录下创建一个名为polls的文件夹，再在其中创建一个名为index.html的文件。编辑index.html文件，添加以下代码：

   ```html
   <h1>Hello, world!</h1>
   ```

   保存文件后，重启服务器，刷新页面，应该就会看到"Hello, world!"的文字出现在屏幕上。恭喜！您已经成功地创建了一个Web应用程序。

## 数据模型

在真实世界的Web应用程序中，通常会存在很多的数据模型。每个数据模型对应着数据库中的一张表格，包含着相关的信息。例如，一个论坛应用程序可能包含用户信息、帖子信息、回复信息、板块信息等表格。Django提供了一个简洁易用的ORM（Object-Relational Mapping，对象关系映射），使得开发者可以轻松地与数据库进行交互。

1. 创建模型：编辑models.py文件，添加以下代码：

   ```python
   from django.db import models
   
   class Question(models.Model):
       question_text = models.CharField(max_length=200)
       pub_date = models.DateTimeField('date published')
       
   class Choice(models.Model):
       question = models.ForeignKey(Question, on_delete=models.CASCADE)
       choice_text = models.CharField(max_length=200)
       votes = models.IntegerField(default=0)
   ```

   此代码定义了两个模型：Question和Choice。Question模型有一个字段question_text用于存储提问的问题，pub_date用于记录发布日期；Choice模型有一个字段question用于关联Question模型，choice_text用于存储选项的内容，votes用于记录投票次数。

2. 执行迁移：执行以下命令，生成所需的数据库表：

   ```
   python manage.py makemigrations
   python manage.py migrate
   ```

3. 使用模型：回到views.py文件，修改index()函数，添加以下代码：

   ```python
   from django.shortcuts import render
   from.models import Question, Choice
   
   
   def index(request):
       latest_questions = Question.objects.order_by('-pub_date')[:5]
       context = {'latest_questions': latest_questions}
       return render(request, 'polls/index.html', context)
   ```

   此代码引入了刚才定义的两个模型类。调用Question.objects.order_by('-pub_date').all()方法，得到最新发布的前五个问题，并传递给模板文件作为上下文变量。

4. 修改HTML模板：编辑index.html文件，添加以下代码：

   {% for question in latest_questions %}
      <div>
          <h2>{{ question.question_text }}</h2>
          <p>{{ question.pub_date }}</p>
      </div>
   {% endfor %}

   {% for choice in question.choice_set.all %}
      <ul>
          <li>{{ choice.choice_text }} -- {{ choice.votes }} vote{{ choice.votes|pluralize }}</li>
      </ul>
   {% empty %}
      <p>No choices are available.</p>
   {% endfor %}

   此代码展示了最新发布的前五个问题及其对应的选项。{% for %}循环遍历latest_questions列表，{% empty %}标记显示当没有可用的选项时显示的提示文本。

## URL映射

在之前的例子中，我们添加了一个指向index视图函数的URL映射。Django的URL映射机制非常灵活，它支持各种形式的URL匹配模式，并允许定制URL正则表达式。

1. 创建URLs映射：编辑urls.py文件，添加以下代码：

   ```python
   from django.urls import path, re_path
   
   from. import views
   
   
   app_name = 'polls'
   
   urlpatterns = [
       path('', views.IndexView.as_view(), name='index'),
       # Detail view of a single poll
       path('<int:pk>/', views.DetailView.as_view(), name='detail'),
       # Results of a single poll
       path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
       # Voting form for a single poll
       path('<int:poll_id>/vote/', views.vote, name='vote'),
   ]
   ```

   此代码定义了四个映射：根URL、详情页、结果页、投票页。详情页和结果页采用的是动态URL模式，由<int:pk>指定。投票页采用的是静态URL模式，由/vote/结尾。

2. 编写视图类：编辑views.py文件，添加以下代码：

   ```python
   from django.views.generic import ListView, DetailView, FormView
   
   
   class IndexView(ListView):
       model = Question
       template_name = 'polls/index.html'
       
       def get_context_data(self, **kwargs):
           context = super().get_context_data(**kwargs)
           context['now'] = timezone.now()
           return context
       
   
   class DetailView(DetailView):
       model = Question
       template_name = 'polls/detail.html'
       
       def get_queryset(self):
           """
           Excludes any questions that aren't published yet.
           """
           queryset = super().get_queryset()
           return queryset.filter(pub_date__lte=timezone.now())
       
       def get_context_data(self, **kwargs):
           context = super().get_context_data(**kwargs)
           context['now'] = timezone.now()
           return context
       
   
   class ResultsView(DetailView):
       model = Question
       template_name = 'polls/results.html'
       
       def get_queryset(self):
           """
           Excludes any questions that aren't published yet.
           """
           queryset = super().get_queryset()
           return queryset.filter(pub_date__lte=timezone.now())
       
       def get_context_data(self, **kwargs):
           context = super().get_context_data(**kwargs)
           selected_choice = self.object.choice_set.get(pk=self.request.POST['choice'])
           selected_choice.votes += 1
           selected_choice.save()
           
           results = {}
           total_votes = 0
           for choice in self.object.choice_set.all():
               results[choice.choice_text] = choice.votes
               total_votes += choice.votes
               
           chart = {
               'chart': {'type': 'pie'},
               'title': {'text': ''},
              'series': [{'name': '', 'data': []}],
           }
           
           if total_votes > 0:
               chart['title']['text'] = f'{selected_choice.choice_text}: {total_votes} vote{"s" if total_votes!= 1 else ""}'
               data = [[label, value] for label, value in results.items()]
               chart['series'][0]['name'] = selected_choice.choice_text
               chart['series'][0]['data'] = data
           
           context['chart'] = json.dumps(chart)
           return context
       
   
   class PollForm(forms.ModelForm):
       class Meta:
           model = Choice
           fields = ['choice_text']
   
   
   
   class VoteView(FormView):
       template_name = 'polls/detail.html'
       form_class = PollForm
       
       def dispatch(self, request, *args, **kwargs):
           self.poll = get_object_or_404(Question, pk=kwargs['poll_id'])
           return super().dispatch(request, *args, **kwargs)
       
       def get_form_kwargs(self):
           kwargs = super().get_form_kwargs()
           kwargs['poll'] = self.poll
           return kwargs
       
       def form_valid(self, form):
           form.instance.question = self.poll
           form.instance.user = self.request.user
           form.save()
           
           return redirect(reverse('polls:results', args=(self.poll.pk,)))
       
       def get_success_url(self):
           messages.add_message(self.request, messages.INFO, "Your vote has been submitted.")
           return reverse('polls:detail', args=(self.poll.pk,))
   ```

   此代码定义了四个视图类：IndexView、DetailView、ResultsView和VoteView。它们分别负责显示首页、问题详情页、问题结果页、问题投票页。

   DetailView继承自DetailView类，用于显示特定问题的详细信息。为了防止未发布的问题被查看，get_queryset()方法过滤了查询集。另外，get_context_data()方法向上下文中添加了一个当前时间戳。

   ResultsView继承自DetailView类，用于显示特定问题的投票结果。get_context_data()方法向上下文中添加了投票图表数据。

   VoteView继承自FormView类，用于处理提交的投票表单。form_valid()方法处理表单提交后，将生成一个新的Choice对象并提交到数据库。form_invalid()方法处理表单验证失败的情况。

   注意：此处仅演示了基本的视图类编写方法，并不涉及数据库的实际操作。

3. 修改HTML模板：编辑index.html文件，添加以下代码：

   {% load staticfiles %}
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <title>{% block title %}{% endblock %}</title>
       <!-- Load the necessary scripts and stylesheets -->
       <script src="{% static 'polls/highcharts.js' %}"></script>
       <style type="text/css">
           body {font-family: sans-serif;}
           h1 {margin-bottom: 0.5rem;}
           ul {list-style: none; margin: 0; padding: 0;}
       </style>
   </head>
   <body>
       <header>
           <nav>
               <ul>
                   <li><a href="{% url 'polls:index' %}">Home</a></li>
               </ul>
           </nav>
       </header>
       <main>
           <h1>Polls</h1>
           {% if latest_questions %}
               <ol>
                   {% for question in latest_questions %}
                       <li><a href="{% url 'polls:detail' question.pk %}">{{ question.question_text }}</a></li>
                   {% endfor %}
               </ol>
           {% else %}
               <p>No polls are available.</p>
           {% endif %}
       </main>
   </body>
   </html>

   编辑detail.html文件，添加以下代码：

   {% extends 'polls/base.html' %}
   {% load crispy_forms_tags %}
   {% block content %}
       <article>
           <h2>{{ object.question_text }}</h2>
           <p>{{ object.pub_date }}</p>
           {% if error_message %}
               <p><strong>{{ error_message }}</strong></p>
           {% endif %}
           <form method="post">
               {% csrf_token %}
               {{ form|crispy }}
               <button type="submit">Vote</button>
           </form>
       </article>
   {% endblock %}

   编辑results.html文件，添加以下代码：

   {% extends 'polls/base.html' %}
   {% block content %}
       <article>
           <h2>{{ object.question_text }}</h2>
           <p>{{ object.pub_date }}</p>
           <figure id="chart"></figure>
           <table>
               <thead>
                   <tr>
                       <th>Choice</th>
                       <th>Votes</th>
                   </tr>
               </thead>
               <tbody>
                   {% for choice in object.choice_set.all %}
                       <tr>
                           <td>{{ choice.choice_text }}</td>
                           <td>{{ choice.votes }}</td>
                       </tr>
                   {% endfor %}
               </tbody>
           </table>
       </article>
   {% endblock %}

   此代码修改了首页、详情页和结果页的布局，并加载了必要的样式表和脚本。详情页使用Crispy Forms插件渲染了投票表单。结果页使用Highcharts.js库生成了一份简单的柱状图。

4. 测试应用程序：确保urls.py文件中的其他映射也正常工作，然后访问http://localhost:8000/polls/。应该会看到类似于这样的页面：


   点击任一问题链接，应该会跳转到详细页面，如图：


   提交正确的选项后，应该会看到投票结果页面，如图：


   可以看到，投票结果页面正确地显示了选定的选项，并更新了投票总数。