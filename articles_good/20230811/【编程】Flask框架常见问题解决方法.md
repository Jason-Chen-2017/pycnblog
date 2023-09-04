
作者：禅与计算机程序设计艺术                    

# 1.简介
         

## （一）课程简介
本节课主要是向大家分享一些常见的问题和相应的解决方案，助力同学们快速了解并掌握Flask相关知识和技能。

## （二）适用范围
本教程面向具有一定Python基础的学生群体，如果您刚刚接触到Python或是对Web开发不熟悉，建议先学习Python基础知识再来学习这个课程。

# 2.常见问题汇总
## （一）视图函数问题
### 2.1 为什么要分成多个视图函数？
- 提高代码可读性，方便管理
- 分离关注点，降低耦合度
- 可以实现函数级权限控制、缓存机制等扩展功能

### 2.2 url_for()函数怎么用的？
url_for()函数可以生成视图函数的URL地址，通过传入参数的方式指定需要调用的视图函数及其参数，例如：`url_for('sayHello', name='world')`会返回当前应用中的路由字典中对应名称为`sayHello`的视图函数的URL地址，该函数可以接收两个参数:
- 函数名作为字符串，也可以使用`name=func.__name__`来获取函数名；
- 参数作为关键字参数传递给函数。

```python
from flask import Flask, render_template, redirect, request, session, flash, url_for
app = Flask(__name__)


@app.route('/')
def index():
return 'hello'


@app.route('/login')
def login():
# 跳转到/welcome页面
return redirect(url_for('welcome'))


@app.route('/welcome')
def welcome():
user = {'username': 'admin'}
return render_template('welcome.html', user=user)


if __name__ == '__main__':
app.run(debug=True)
``` 

### 2.3 @app.before_request装饰器的作用？
before_request()装饰器是一个应用请求钩子函数，它在每一次请求之前被执行，通常用于设置默认值、检查登录状态、日志记录等。可以使用该装饰器实现以下功能：
- 设置cookie、session等；
- 检查用户登录信息，如果没有登录则重定向到登录页面；
- 添加CSRF防护；
- 处理响应结果的缓存。

```python
from flask import Flask, jsonify, abort
import functools

app = Flask(__name__)

def requires_auth(view):
"""检查用户是否登录"""
@functools.wraps(view)
def wrapper(*args, **kwargs):
if not g.current_user.is_authenticated:
abort(401)
return view(*args, **kwargs)
return wrapper

@app.route('/protected')
@requires_auth
def protected():
return "This is a protected page"

``` 

### 2.4 @app.context_processor()装饰器的作用？
context_processor()装饰器是一个应用上下文处理器函数，它可以将函数注册为上下文处理函数，当渲染模板时会自动调用这些函数。可以使用该装饰器实现以下功能：
- 在每个模板中添加全局变量；
- 将模板变量从数据库取出后赋值给模板变量。

```python
from flask import Flask, g, request
import json

app = Flask(__name__)

@app.context_processor
def inject_vars():
config = {}
with open("config.json", "r") as f:
config = json.load(f)
return dict(config=config)

@app.route("/")
def index():
return render_template("index.html")

if __name__ == "__main__":
app.run(debug=True)
``` 

### 2.5 Request对象属性和方法
#### 2.5.1 请求路径、请求方法、请求参数
| 属性 | 描述 |
| - | - |
| request.path | 请求路径，包含查询参数（query string）。 |
| request.method | HTTP请求方法。如GET、POST、PUT等。 |
| request.values | 通过表单发送的数据，类型为MultiDict。 |
| request.form | 仅包含请求数据的一个字典。对于POST请求来说，这个字典包含发送的数据。 |
| request.files | 文件上传信息。 |
| request.args | 查询字符串参数，类型为ImmutableMultiDict。 |
| request.headers | 请求头信息。 |
| request.cookies | 获取客户端Cookies。 |

#### 2.5.2 获取请求数据
| 方法 | 描述 |
| - | - |
| request.get_data() | 返回完整请求数据。 |
| request.get_json() | 返回JSON格式请求数据。 |

#### 2.5.3 请求上下文变量g
Flask使用上下文变量g来提供线程隔离的全局变量存储。该变量可以在视图函数间共享。以下是一些常用的g属性和方法：
| 属性/方法 | 描述 |
| - | - |
| g.cache | 缓存模块的代理。 |
| g.locale | 当前请求语言环境。 |
| g.request_id | 当前请求ID。 |
| g.session | 会话变量。 |
| g.user | 用户对象。 |

### 2.6 Response对象属性和方法
#### 2.6.1 Response对象的属性
| 属性 | 描述 |
| - | - |
| response.status_code | HTTP响应码。 |
| response.headers | HTTP响应头。 |
| response.content_type | 响应内容类型。 |
| response.charset | 字符集。 |
| response.mimetype | MIME类型。 |

#### 2.6.2 设置HTTP响应头
可以通过Response对象的headers属性设置HTTP响应头。例如：

```python
from flask import Flask, jsonify, make_response

app = Flask(__name__)

@app.route("/users/<int:uid>")
def get_user(uid):
resp = make_response(jsonify({"msg": "success"}))
resp.headers["X-User"] = str(uid)
return resp
``` 

#### 2.6.3 渲染HTML响应
可以通过render_template()函数直接渲染HTML响应。

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
user = {"name": "Alice", "age": 25}
return render_template("home.html", user=user)
``` 

# 3.Flask的常用扩展库
常用扩展库包括如下几个：
## （一）Flask-SQLAlchemy
Flask-SQLAlchemy是一个轻量级的数据库ORM扩展库，可以很方便地连接不同的数据库，并且支持多种关系型数据库，如MySQL、PostgreSQL等。它的主要特点如下：
- 使用Python上下文管理器，可以在不同请求之间保持数据库连接池的持久化；
- 内置的查询构造器可以方便地编写复杂的SQL语句；
- 模型继承、多态关联等特性使得数据库层面的交互更加方便；
- 支持事件回调，可以进行数据库访问前后的监听。

```python
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] ='mysql://root:123456@localhost:3306/test?charset=utf8mb4'
db = SQLAlchemy(app)

class User(db.Model):
id = db.Column(db.Integer, primary_key=True)
username = db.Column(db.String(50), unique=True)

def __repr__(self):
return '<User %r>' % self.username

@app.route('/users/')
def list_users():
users = User.query.all()
result = []
for u in users:
result.append({'id': u.id, 'username': u.username})
return jsonify(result)

@app.route('/users/', methods=['POST'])
def add_user():
data = request.get_json()
new_user = User(username=data['username'])
db.session.add(new_user)
db.session.commit()
return jsonify({'message': 'ok'})
``` 

## （二）Flask-WTF
Flask-WTF是基于WTForms的Flask扩展库，它提供了表单验证、 CSRF保护、文件上传等常用功能。它的主要特点如下：
- 对WTForms的核心功能进行了增强，增加了新特性，如自定义字段类型、字段验证链、CSRF保护等；
- 提供了CSRFProtect类，可以通过配置项开启CSRF保护；
- 提供CSRF令牌生成器，可以帮助开发者生成CSRF防护用的标记和隐藏字段；
- 提供CSRF保护的跨域请求伪造保护功能。

```python
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

app = Flask(__name__)
app.secret_key ='secret key'  # 安全密钥

class LoginForm(FlaskForm):
username = StringField('用户名', validators=[DataRequired(), Length(1, 20)])
password = StringField('密码', validators=[DataRequired()])
submit = SubmitField('登陆')

@app.route('/', methods=['GET', 'POST'])
def login():
form = LoginForm()
if form.validate_on_submit():
username = form.username.data
password = form.password.data
if username == 'admin' and password == '123456':
flash('登录成功！','success')
return redirect(url_for('home'))
else:
flash('用户名或密码错误！', 'danger')
return render_template('login.html', form=form)
``` 

## （三）Flask-Login
Flask-Login是实现用户登录、注销、权限控制等的扩展库。它的主要特点如下：
- 基于Flask-Principal扩展库，可以实现基于角色和权限的授权；
- 提供登录、注销、权限认证等流程；
- 支持多种登录方式，如OAuth、OpenID等；
- 提供用户模型扩展，可以自定义用户对象。

```python
from flask import Flask, render_template, redirect, url_for, request
from flask_login import LoginManager, login_required, logout_user, current_user, login_user

app = Flask(__name__)
login_manager = LoginManager(app)

# 用户登录信息存储
users = {
'admin': {'pw': '123456'},
'alice': {'pw': 'abcde'}
}

@login_manager.user_loader
def load_user(username):
if username in users:
user = User(username=username)
return user
else:
return None

@app.route('/')
@login_required
def home():
return 'Welcome back, %s!' % current_user.username

@app.route('/login', methods=['GET', 'POST'])
def login():
if request.method == 'POST':
username = request.form['username']
pw = request.form['pw']
if username in users and users[username]['pw'] == pw:
user = load_user(username)
login_user(user)
return redirect(url_for('home'))
return render_template('login.html')

@app.route('/logout')
def logout():
logout_user()
return redirect(url_for('login'))
``` 

## （四）Flask-RESTful
Flask-RESTful是一个轻量级的RESTful API扩展库，可以帮助开发者快速搭建RESTful API。它的主要特点如下：
- 提供了Resource基类，可以定义RESTful API资源；
- 提供了Api类，可以快速构建RESTful API；
- 提供了几种不同的装饰器，可以快速定义API端点；
- 内置了分页、认证等常用插件，可以满足大部分需求。

```python
from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
def get(self):
return {'hello': 'world'}

api.add_resource(HelloWorld, '/helloworld')
```