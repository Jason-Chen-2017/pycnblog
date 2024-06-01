
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2019年下半年是最好的年份，谷歌、Facebook、微软等一众科技巨头纷纷推出自己的服务器云服务平台，而作为程序员的我们更需要有一个地方可以快速搭建起一个属于自己的网站或者博客系统。因此，本文将带领读者构建自己的个人博客系统，学习更多关于Flask框架和MongoDB数据库的知识。
         
       # 2.基本概念及术语说明
       1.Flask框架
           Flask是一个轻量级Web应用框架，基于Python语言。它让开发者只需关注如何构造一个web应用，而不是其他诸如路由配置、数据库连接等繁琐任务。它的主要特点包括易用性、灵活性、可扩展性、可移植性等。
       
       2.RESTful API
           REST（Representational State Transfer）即表述性状态转化，它定义了客户端与服务器之间交互的标准方法。基于RESTful API的Web服务可以通过HTTP协议通信，支持不同类型的请求方法，例如GET、POST、PUT、DELETE等。
       
       3.MongoDB数据库
           MongoDB是一个开源NoSQL数据库，它提供了高性能的数据持久化存储能力。它是一个面向文档的数据库，能够处理复杂的数据结构。同时它也支持动态查询功能，使得用户能够方便地搜索、排序数据。
       
       4.HTML/CSS/JavaScript
           HTML（超文本标记语言）用于创建网页的内容，通过标签对网页元素进行编排；CSS（层叠样式表）用于美化网页的外观；JavaScript（Java 脚本）用于为网页添加功能。
       
       5.Bootstrap框架
           Bootstrap是Twitter公司推出的一个开源前端框架，用于快速、简单地设计响应式网页界面。它由HTML、CSS、JavaScript、jQuery组成，实现了前端开发中最常用的组件，并提供一系列设计模版，帮助开发人员快速搭建网页。
       
       6.部署与运维
           为了让自己的博客系统在互联网上得以访问，还需要考虑如何部署到服务器上，并进行必要的安全防护措施，确保系统的可用性和稳定性。
      
      # 3.核心算法原理及操作步骤
      首先，我们需要有一个博客的数据库模型。一般来说，博客数据库模型分为4个部分：
      
          1. 用户信息表(Users)
          2. 博客文章表(Posts)
          3. 评论表(Comments)
          4. 标签表(Tags)
          
      在这里，我假设我们已经有一个部署好并运行的MongoDB数据库。接着，我们要做的第一件事就是建立我们的博客的数据库模型。我们可以使用Python、Django或其他相关框架来编写我们的模型。比如，使用Django，我们可以创建一个`models.py`文件，然后定义如下所示的模型：
  
      ```python
      from django.db import models

      class User(models.Model):
          name = models.CharField(max_length=50)
          email = models.EmailField()
          password = models.CharField(max_length=50)
          created_at = models.DateTimeField(auto_now_add=True)
          updated_at = models.DateTimeField(auto_now=True)

          def __str__(self):
              return self.name

      class Post(models.Model):
          title = models.CharField(max_length=100)
          content = models.TextField()
          user = models.ForeignKey('User', on_delete=models.CASCADE)
          tags = models.ManyToManyField('Tag')
          created_at = models.DateTimeField(auto_now_add=True)
          updated_at = models.DateTimeField(auto_now=True)

          def __str__(self):
              return self.title

      class Comment(models.Model):
          post = models.ForeignKey('Post', related_name='comments', on_delete=models.CASCADE)
          author = models.CharField(max_length=50)
          content = models.TextField()
          created_at = models.DateTimeField(auto_now_add=True)

          def __str__(self):
              return f'{self.author}: {self.content[:20]}'

      class Tag(models.Model):
          name = models.CharField(max_length=50, unique=True)
          posts = models.ManyToManyField('Post', related_name='tags')

          def __str__(self):
              return self.name
      ```

      上面的代码定义了四种模型：
      
          1. `User`模型：用来存储用户信息。
          2. `Post`模型：用来存储博客文章。
          3. `Comment`模型：用来存储评论。
          4. `Tag`模型：用来存储文章标签。
          
      模型之间的关系可以用下图表示：
      

      下一步，我们就可以使用Flask框架来搭建我们的博客系统了。我们先安装Flask和相关依赖库：

      ```
      pip install flask pymongo dnspython markdown gunicorn flask-bootstrap
      ```

      安装完成后，我们可以新建一个名为`app.py`的文件，然后写入如下的代码：

      ```python
      from flask import Flask, render_template, request, redirect, url_for, flash
      from flask_bootstrap import Bootstrap
      from datetime import datetime
      from bson.objectid import ObjectId
      from werkzeug.security import generate_password_hash, check_password_hash
      from pymongo import MongoClient
      import os

      app = Flask(__name__)
      app.secret_key ='super secret key'
      bootstrap = Bootstrap(app)

      client = MongoClient(os.environ['MONGODB_URI'])
      db = client.flask_blog

      @app.route('/')
      def index():
          page = int(request.args.get('page', default=1))
          per_page = 10
          skips = (page - 1) * per_page
          total = db.posts.count()
          posts = list(db.posts.find().sort([('_id', -1)]).skip(skips).limit(per_page))
          for post in posts:
              if len(post['tags']):
                  tag_ids = [ObjectId(_id) for _id in post['tags']]
                  query = {'_id': {'$in': tag_ids}}
                  cursor = db.tags.find(query)
                  post['tag_names'] = [doc['name'] for doc in cursor]
              else:
                  post['tag_names'] = []
              del post['_id']
          prev_url = None if page == 1 else url_for('.index', page=page-1)
          next_url = None if skips+per_page >= total else url_for('.index', page=page+1)
          pagination = Pagination(page, per_page, total, css_framework='bootstrap4')
          return render_template('index.html', posts=posts,
                                 prev_url=prev_url, next_url=next_url, pagination=pagination)

      @app.route('/login', methods=['GET', 'POST'])
      def login():
          error = None
          if request.method == 'POST':
              email = request.form['email']
              password = request.form['password']
              user = db.users.find_one({'email': email})
              if not user or not check_password_hash(user['password'], password):
                  error = 'Invalid username or password.'
              else:
                  session['logged_in'] = True
                  session['username'] = user['name']
                  flash('You were logged in.')
                  return redirect(url_for('index'))
          return render_template('login.html', error=error)

      @app.route('/logout')
      def logout():
          session.pop('logged_in', None)
          session.pop('username', None)
          flash('You were logged out.')
          return redirect(url_for('index'))

      @app.route('/register', methods=['GET', 'POST'])
      def register():
          error = None
          if request.method == 'POST':
              name = request.form['name']
              email = request.form['email']
              password = request.form['password']
              confirm_password = request.form['confirm_password']
              user = db.users.find_one({'email': email})
              if user is not None:
                  error = 'This email address has already been registered.'
              elif password!= confirm_password:
                  error = 'Password does not match confirmation password.'
              else:
                  db.users.insert({
                      'name': name,
                      'email': email,
                      'password': generate_password_hash(password),
                      'created_at': datetime.utcnow(),
                      'updated_at': datetime.utcnow()
                  })
                  flash('Your account was created successfully.')
                  return redirect(url_for('login'))
          return render_template('register.html', error=error)

      @app.route('/create-post', methods=['GET', 'POST'])
      def create_post():
          error = None
          if not session.get('logged_in'):
              return redirect(url_for('login'))
          if request.method == 'POST':
              title = request.form['title']
              content = request.form['content']
              tag_names = request.form.getlist('tags[]')
              try:
                  post_id = db.posts.insert({
                      'title': title,
                      'content': content,
                      'user_id': ObjectId(session['user_id']),
                      'tags': [],
                      'created_at': datetime.utcnow(),
                      'updated_at': datetime.utcnow()
                  }, safe=True)['insertedId']
                  for tag_name in tag_names:
                      tag = db.tags.find_one({'name': tag_name})
                      if tag is None:
                          db.tags.insert({
                              'name': tag_name,
                              'posts': []
                          }, safe=True)
                      db.tags.update_many({}, {'$addToSet': {'posts': ObjectId(post_id)}})
                  flash('The post was created successfully.')
                  return redirect(url_for('view_post', post_id=str(post_id)))
              except Exception as e:
                  print(e)
                  error = 'Failed to save the post.'
          tags = [{'name': doc['name']} for doc in db.tags.find()]
          return render_template('create_post.html', error=error, tags=tags)

      @app.route('/edit-post/<post_id>', methods=['GET', 'POST'])
      def edit_post(post_id):
          error = None
          if not session.get('logged_in'):
              return redirect(url_for('login'))
          post = db.posts.find_one({'_id': ObjectId(post_id)})
          if post is None:
              return redirect(url_for('index'))
          if post['user_id']!= ObjectId(session['user_id']):
              return redirect(url_for('index'))
          if request.method == 'POST':
              title = request.form['title']
              content = request.form['content']
              tag_names = request.form.getlist('tags[]')
              try:
                  result = db.posts.update_one({'_id': ObjectId(post_id)}, {
                      '$set': {
                          'title': title,
                          'content': content,
                          'updated_at': datetime.utcnow()
                      }
                  })
                  assert result.matched_count == 1
                  db.tags.update_many({}, {'$pull': {'posts': ObjectId(post_id)}})
                  for tag_name in tag_names:
                      tag = db.tags.find_one({'name': tag_name})
                      if tag is None:
                          db.tags.insert({
                              'name': tag_name,
                              'posts': []
                          }, safe=True)
                      db.tags.update_one({'_id': ObjectId(tag['_id'])},
                                         {'$addToSet': {'posts': ObjectId(post_id)}})
                  flash('The post was edited successfully.')
                  return redirect(url_for('view_post', post_id=post_id))
              except AssertionError:
                  error = 'This post cannot be found.'
              except Exception as e:
                  print(e)
                  error = 'Failed to save the post.'
          tag_ids = post['tags']
          if isinstance(tag_ids, str):
              tag_ids = [tag_ids]
          tags = [{'name': doc['name'], '_id': str(doc['_id'])} for doc in
                  db.tags.find({'_id': {'$in': [ObjectId(_) for _ in tag_ids]}})]
          selected_tags = [t['name'] for t in tags]
          return render_template('edit_post.html', post=post, tags=tags,
                                 selected_tags=selected_tags, error=error)

      @app.route('/view-post/<post_id>')
      def view_post(post_id):
          post = db.posts.find_one({'_id': ObjectId(post_id)})
          comments = list(db.comments.find({'post_id': ObjectId(post_id)}).sort([('_id', 1)]))
          return render_template('view_post.html', post=post, comments=comments)

      @app.route('/comment/<post_id>', methods=['POST'])
      def comment(post_id):
          content = request.form['content']
          author = request.form['author']
          now = datetime.utcnow()
          db.comments.insert({
              'post_id': ObjectId(post_id),
              'author': author,
              'content': content,
              'created_at': now
          }, safe=True)
          return redirect(url_for('view_post', post_id=post_id))

      @app.route('/search')
      def search():
          keyword = request.args.get('keyword').strip()
          results = db.posts.find({'$text': {'$search': keyword}})
          return render_template('search_results.html', keyword=keyword, results=results)

      @app.route('/tags/<tag_name>')
      def show_tag(tag_name):
          page = int(request.args.get('page', default=1))
          per_page = 10
          skips = (page - 1) * per_page
          total = db.tags.find_one({'name': tag_name})['posts'].count()
          tag = db.tags.find_one({'name': tag_name})
          if tag is None:
              return redirect(url_for('index'))
          post_ids = tag['posts']
          posts = list(db.posts.find({'_id': {'$in': [ObjectId(_) for _ in post_ids]}}
                                     ).sort([('_id', -1)]).skip(skips).limit(per_page))
          for post in posts:
              if len(post['tags']):
                  tag_ids = [ObjectId(_id) for _ in post['tags']]
                  query = {'_id': {'$in': tag_ids}}
                  cursor = db.tags.find(query)
                  post['tag_names'] = [doc['name'] for doc in cursor]
              else:
                  post['tag_names'] = []
              del post['_id']
          prev_url = None if page == 1 else url_for('.show_tag', tag_name=tag_name, page=page-1)
          next_url = None if skips + per_page >= total else url_for('.show_tag', tag_name=tag_name, page=page+1)
          pagination = Pagination(page, per_page, total, css_framework='bootstrap4')
          return render_template('tag.html', tag=tag, posts=posts, prev_url=prev_url, next_url=next_url,
                                 pagination=pagination)

      if __name__ == '__main__':
          app.run(debug=True)
      ```

      大家可能会发现这个代码非常臃肿。不过不要担心，我们之后会逐渐优化这个代码，把一些重复的代码块合并起来，减少代码冗余。

      有了这个博客的数据库模型，我们就可以开始编写博客的前台页面了。我们可以先创建一个名为`templates`的文件夹，然后在里面分别创建三个文件夹：

      1. `layout`：用来存放整个网站的布局模板。
      2. `partial`：用来存放一些小组件。
      3. `pages`：用来存放各个页面的视图函数。

      ### 创建布局模板

      首先，我们需要创建`layout/base.html`模板，作为整个网站的布局模板。它应该包括以下内容：

      ```html
      <!DOCTYPE html>
      <html lang="en">
      <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          {% block head %}
          {% endblock %}
          <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
      </head>
      <body>
          {% block body %}
          {% endblock %}
          <script src="{{ url_for('static', filename='js/script.js') }}"></script>
      </body>
      </html>
      ```

      这是一个标准的HTML5文件，包括了`head`标签中的一些元数据和CSS引用。我们可以使用Jinja2模板语言来渲染这个模板。

      ### 创建首页模板

      接着，我们需要创建`pages/index.html`模板，作为网站的主页模板。它应该包括以下内容：

      ```html
      {% extends "layout/base.html" %}
      {% block head %}
          {{ super() }}
          <title>{{ config.BLOG_TITLE }}</title>
      {% endblock %}
      {% block body %}
          <header>
              <nav class="navbar navbar-expand-md navbar-dark bg-dark mb-4">
                  <div class="container">
                      <a class="navbar-brand" href="{{ url_for('index') }}">{{ config.BLOG_TITLE }}</a>
                      <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarCollapse" aria-controls="navbarCollapse" aria-expanded="false" aria-label="Toggle navigation">
                          <span class="navbar-toggler-icon"></span>
                      </button>
                      <div class="collapse navbar-collapse" id="navbarCollapse">
                          <ul class="navbar-nav mr-auto">
                              <li class="nav-item active">
                                  <a class="nav-link" href="{{ url_for('index') }}">Home</a>
                              </li>
                              {% if current_user.is_authenticated %}
                                  <li class="nav-item dropdown">
                                      <a class="nav-link dropdown-toggle" href="#" id="navbarDropdownMenuLink" role="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                                          Administration
                                      </a>
                                      <div class="dropdown-menu" aria-labelledby="navbarDropdownMenuLink">
                                          <a class="dropdown-item" href="#">New Post</a>
                                          <a class="dropdown-item" href="#">Edit Profile</a>
                                      </div>
                                  </li>
                              {% endif %}
                          </ul>
                          <ul class="navbar-nav ml-auto">
                              {% if current_user.is_anonymous %}
                                  <li class="nav-item"><a class="nav-link" href="{{ url_for('login') }}">Log In</a></li>
                                  <li class="nav-item"><a class="nav-link" href="{{ url_for('register') }}">Sign Up</a></li>
                              {% else %}
                                  <li class="nav-item dropdown">
                                      <a class="nav-link dropdown-toggle" href="#" id="navbarDropdownMenuLink" role="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                                          Welcome, {{ current_user.name }}
                                      </a>
                                      <div class="dropdown-menu dropdown-menu-right" aria-labelledby="navbarDropdownMenuLink">
                                          <a class="dropdown-item" href="{{ url_for('logout') }}">Logout</a>
                                      </div>
                                  </li>
                              {% endif %}
                          </ul>
                      </div>
                  </div>
              </nav>
          </header>
          <main class="container mt-4 mb-4">
              <h1>Blog Posts</h1>
              <hr>
              {% for post in posts %}
                  <article class="mb-4">
                      <h2><a href="{{ url_for('view_post', post_id=post['_id']) }}">{{ post['title'] }}</a></h2>
                      <p class="lead text-muted">{{ post['created_at'].strftime('%b %d, %Y') }} by {{ post['user']['name'] }}</p>
                      <div>{{ post['content'] | safe }}</div>
                      {% if post['tag_names'] %}
                          <footer>
                              <small class="text-muted">
                                  Tags:
                                  {% for tag_name in post['tag_names'] %}
                                      <a href="{{ url_for('show_tag', tag_name=tag_name) }}" class="badge badge-primary">{{ tag_name }}</a>
                                  {% endfor %}
                              </small>
                          </footer>
                      {% endif %}
                  </article>
              {% else %}
                  <p>There are no blog posts available.</p>
              {% endfor %}
              <nav>
                  <ul class="pagination justify-content-center">
                      {% if prev_url %}<li class="page-item"><a class="page-link" href="{{ prev_url }}">Previous</a></li>{% endif %}
                      {% for i in range((total//per_page)+min(1, total%per_page))+1 %}
                          {% if loop.index == page %}
                              <li class="page-item active" aria-current="page"><span class="page-link">{{ i }}</span></li>
                          {% else %}
                              <li class="page-item"><a class="page-link" href="{{ url_for('index', page=i) }}">{{ i }}</a></li>
                          {% endif %}
                      {% endfor %}
                      {% if next_url %}<li class="page-item"><a class="page-link" href="{{ next_url }}">Next</a></li>{% endif %}
                  </ul>
              </nav>
          </main>
      {% endblock %}
      ```

      这个模板继承自`layout/base.html`，并包含了一个导航栏。其中包含的链接都指向对应的视图函数。在渲染该模板时，还会根据当前用户是否已登录来显示不同的菜单项。

      使用`{{ super() }}`语句调用父模板的方法，从而继承`head`标签里的所有内容。在`{% block head %}`和`{% endblock %}`之间，包含了博客标题。

      `{% block body %}`和`{% endblock %}`之间的部分则是博客文章列表。这里我们循环遍历所有文章，并生成每篇文章的摘要、作者、创建日期和标签。如果文章没有标签，就显示无标签。

      如果有分页，就会显示分页导航条。

      ### 创建登陆注册模板

      接着，我们需要创建`pages/login.html`和`pages/register.html`模板，作为博客的登陆和注册页面。它们应该包括以下内容：

      ```html
      {% extends "layout/base.html" %}
      {% block head %}
          {{ super() }}
          <title>Login / Sign Up - {{ config.BLOG_TITLE }}</title>
      {% endblock %}
      {% block body %}
          <main class="container mt-4 mb-4">
              {% if error %}
                  <div class="alert alert-danger" role="alert">{{ error }}</div>
              {% endif %}
              <h1>Login / Sign Up</h1>
              <hr>
              <form method="post" action="{{ url_for('login') }}">
                  <div class="form-group">
                      <label for="email">Email:</label>
                      <input type="email" class="form-control" id="email" name="email" required>
                  </div>
                  <div class="form-group">
                      <label for="password">Password:</label>
                      <input type="password" class="form-control" id="password" name="password" required>
                  </div>
                  <button type="submit" class="btn btn-primary">Submit</button>
              </form>
              <br>
              <a href="{{ url_for('forgot_password') }}">Forgot Password?</a>
          </main>
      {% endblock %}
      ```

      和之前类似，这里也是继承自`layout/base.html`。如果存在错误提示，就会显示在页面上。否则，显示登录表单。如果用户忘记密码，还有一个找回密码的链接。

      ```html
      {% extends "layout/base.html" %}
      {% block head %}
          {{ super() }}
          <title>Register - {{ config.BLOG_TITLE }}</title>
      {% endblock %}
      {% block body %}
          <main class="container mt-4 mb-4">
              {% if error %}
                  <div class="alert alert-danger" role="alert">{{ error }}</div>
              {% endif %}
              <h1>Create an Account</h1>
              <hr>
              <form method="post" action="{{ url_for('register') }}">
                  <div class="form-group">
                      <label for="name">Name:</label>
                      <input type="text" class="form-control" id="name" name="name" required>
                  </div>
                  <div class="form-group">
                      <label for="email">Email:</label>
                      <input type="email" class="form-control" id="email" name="email" required>
                  </div>
                  <div class="form-group">
                      <label for="password">Password:</label>
                      <input type="password" class="form-control" id="password" name="password" required>
                  </div>
                  <div class="form-group">
                      <label for="confirm_password">Confirm Password:</label>
                      <input type="password" class="form-control" id="confirm_password" name="confirm_password" required>
                  </div>
                  <button type="submit" class="btn btn-primary">Create Account</button>
              </form>
          </main>
      {% endblock %}
      ```

      这是注册页面。同样，它也会显示错误提示，或者显示注册表单。

      ### 创建编辑博客文章模板

      最后，我们需要创建`pages/edit_post.html`模板，作为编辑博客文章页面。它应该包括以下内容：

      ```html
      {% extends "layout/base.html" %}
      {% block head %}
          {{ super() }}
          <title>Edit Post - {{ config.BLOG_TITLE }}</title>
      {% endblock %}
      {% block body %}
          <main class="container mt-4 mb-4">
              {% if error %}
                  <div class="alert alert-danger" role="alert">{{ error }}</div>
              {% endif %}
              <h1>Edit Post</h1>
              <hr>
              <form method="post" action="{{ url_for('edit_post', post_id=post['_id']) }}">
                  <div class="form-group">
                      <label for="title">Title:</label>
                      <input type="text" class="form-control" id="title" name="title" value="{{ post['title'] }}" required>
                  </div>
                  <div class="form-group">
                      <label for="content">Content:</label>
                      <textarea class="form-control" id="content" name="content" rows="10" required>{{ post['content'] }}</textarea>
                  </div>
                  <div class="form-group">
                      <label for="tags">Select Tags:</label>
                      <select multiple class="form-control" id="tags[]" name="tags[]" size="{{ config.MAX_TAGS }}">
                          {% for tag in tags %}
                              <option value="{{ tag['_id'] }}" {% if tag['name'] in selected_tags %}selected{% endif %}>{{ tag['name'] }}</option>
                          {% endfor %}
                      </select>
                  </div>
                  <button type="submit" class="btn btn-primary">Save Changes</button>
              </form>
          </main>
      {% endblock %}
      ```

      这个模板和之前一样，也是继承自`layout/base.html`。但是它增加了一部分额外的内容。首先，判断是否存在错误提示。其次，显示的是编辑博文的表单。表单包括文章标题、内容和标签。选择标签时，默认选中之前的标签。最后，按钮提交修改。

      ### 自定义样式

      此外，还需要创建一个`static/css/style.css`文件，用来自定义页面的样式。我们可以复制如下内容：

      ```css
      /* Global styles */
      body {
          font-family: sans-serif;
      }
      h1 {
          margin-top: 5rem;
      }
      hr {
          border-color: #ddd;
      }
      article {
          padding-bottom: 1rem;
      }
      footer {
          margin-top: 1rem;
      }

      /* Navigation bar */
     .navbar-brand {
          font-size: 2rem;
          font-weight: bold;
          color: white!important;
      }
     .navbar-nav li a {
          color: white!important;
      }
     .navbar-nav li a:hover {
          background-color: rgba(255, 255, 255, 0.1);
      }
     .navbar-nav li.active a {
          background-color: rgba(255, 255, 255, 0.1);
      }

      /* Article list */
      article h2 {
          font-size: 2rem;
          margin-bottom: 0.5rem;
      }
      article p.lead {
          font-size: 1.2rem;
      }

      /* Pagination */
      ul.pagination {
          display: flex;
          align-items: center;
          justify-content: center;
      }
      ul.pagination li {
          margin-left: 0.5rem;
          margin-right: 0.5rem;
      }
      ul.pagination li.disabled span {
          opacity: 0.7;
          pointer-events: none;
      }
  ```

      CSS文件里包含了许多全局的样式设置，比如字体、边距、颜色、间距等。导航栏的颜色、字体大小等设置也在这里。文章列表、分页导航条的样式也在这里定义。

      至此，所有的博客页面的模板和样式都已完成。

  ## 测试博客系统

  当我们运行`app.py`的时候，应该会看到一个欢迎消息和一个创建账户的表单。

  
  点击“Sign Up”按钮，填写表单信息，确认无误后，应该会跳转到登录页面。
  
  
  输入正确的用户名和密码，点击“Submit”按钮，会跳转到首页。
  
  
  页面右上角会显示当前用户的信息。点击“New Post”，会跳转到创建博客文章页面。
  
  
  可以输入文章标题、正文、选择标签。保存修改后，应该会跳转到查看新创建的文章页面。
  

  查看文章后，可以评论，点赞。
  