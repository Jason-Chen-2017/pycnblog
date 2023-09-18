
作者：禅与计算机程序设计艺术                    

# 1.简介
  


本教程从零开始，带领大家学习如何使用Nodejs和MongoDB快速搭建一个简单的Web应用程序。涵盖的内容包括：
- 使用Nodejs搭建本地服务器
- 安装MongoDB并连接到本地服务器
- 使用Express构建后端API接口
- 配置路由和中间件
- 使用Mongoose连接数据库和管理数据模型
- 使用模板引擎（EJS）渲染视图页面
- 用户注册、登录和身份验证
- 文件上传与下载功能
- 通过Socket.io实现实时通信

本教程适合于刚接触Nodejs和MongoDB的新手用户，而且对计算机相关知识和开发经验不要求太高。相反，教程以实际案例驱动，力求让读者真正动手实践。除此之外，还有很多进阶内容可供学习，比如：
- 在线聊天系统实战（聊天室功能）
- 用React或Angular构建前端界面（增加动态交互效果）
- 网站部署上云或使用PaaS平台（将网站上线）
- 更多需要深入研究的技术细节

# 2. 环境准备
为了能够完成本教程，你首先需要准备以下环境：
- 操作系统：建议选择MacOS或Linux系统，因为安装配置起来比较简单；Windows系统也能安装运行Nodejs和MongoDB，但可能存在一些兼容性问题。
- Nodejs：本教程基于Nodejs 14版本进行编写，你可以通过nvm或nvm-windows安装多个Nodejs版本。
- Visual Studio Code：如果你喜欢使用一个集成开发环境，可以使用Visual Studio Code。

# 3. 创建项目目录及文件
在任意位置打开终端窗口，输入如下命令创建项目目录：
```shell
mkdir node-mongo && cd $_
```
然后创建一个名为`package.json`的文件，输入如下命令初始化Nodejs项目：
```shell
npm init -y
```
这个命令会生成默认的package.json文件。

创建完项目目录之后，我们需要安装项目依赖包。输入如下命令：
```shell
npm install express mongoose ejs body-parser multer socket.io --save
```
以上命令将会安装Express框架、Mongoose数据库驱动、EJS模板引擎、body-parser中间件、multer文件上传库和socket.io实时通信库。

最后，创建如下目录结构：
```
├── app.js # 主程序文件
├── bin
│   └── www # 命令行工具脚本
├── controllers
│   ├── authController.js # 认证控制器
│   ├── fileController.js # 文件控制器
│   ├── messageController.js # 消息控制器
│   └── userController.js # 用户控制器
├── models
│   ├── FileModel.js # 文件模型
│   ├── MessageModel.js # 消息模型
│   └── UserModel.js # 用户模型
└── views
    ├── error.ejs # 错误页模板
    ├── index.ejs # 首页模板
    ├── login.ejs # 登录页模板
    ├── register.ejs # 注册页模板
    └── chat.ejs # 聊天室模板
```
其中bin目录存放的是启动程序的命令行工具脚本；controllers目录存放业务逻辑控制器；models目录存放数据模型；views目录存放页面模板。这些目录和文件都是我们下一步要创建的。

# 4. 项目配置
## 4.1 Express设置
打开`app.js`，写入以下代码：
```javascript
const express = require('express');
const app = express();

// 设置模板引擎
app.set('view engine', 'ejs');
app.set('views', './views'); // 指定模板文件的路径

// 解析POST请求参数
const bodyParser = require('body-parser');
app.use(bodyParser.urlencoded({extended: true})); 

// 设置静态文件托管
const path = require('path');
app.use('/static', express.static(path.join(__dirname, '/public')));

module.exports = app;
```
这里主要做了以下几件事情：
- 使用Express框架；
- 设置模板引擎为EJS；
- 解析POST请求参数；
- 设置静态文件托管；
- 将Express实例导出，方便其他模块调用。

## 4.2 Mongoose设置
创建`config.js`文件，写入以下代码：
```javascript
const mongoose = require('mongoose');
mongoose.connect('mongodb://localhost/node-mongo', {
  useNewUrlParser: true, 
  useUnifiedTopology: true
});

const db = mongoose.connection;
db.on('error', console.error.bind(console, 'connection error:'));
db.once('open', function() {
  console.log("Connected to database successfully!");
});

module.exports = mongoose;
```
这个文件定义了一个函数，用于连接到MongoDB数据库。它首先导入mongoose模块，然后连接到本地的node-mongo数据库。然后监听数据库连接状态，打印日志信息。最后再把mongoose实例导出，方便其他模块调用。

## 4.3 模型设置
### 4.3.1 文件模型
创建`FileModel.js`，写入以下代码：
```javascript
const mongoose = require('../config');
const Schema = mongoose.Schema;

let fileSchema = new Schema({
  name: String,
  type: String,
  size: Number,
  data: Buffer
});

module.exports = mongoose.model('File', fileSchema);
```
这里定义了一个文件数据模型，它有一个名称、类型、大小、数据字段。

### 4.3.2 消息模型
创建`MessageModel.js`，写入以下代码：
```javascript
const mongoose = require('../config');
const Schema = mongoose.Schema;

let messageSchema = new Schema({
  fromUser: {type: Schema.Types.ObjectId, ref: "User"},
  toUser: {type: Schema.Types.ObjectId, ref: "User"},
  content: String
});

module.exports = mongoose.model('Message', messageSchema);
```
这里定义了一个消息数据模型，它有发送者、接收者、内容三个字段。

### 4.3.3 用户模型
创建`UserModel.js`，写入以下代码：
```javascript
const mongoose = require('../config');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const Schema = mongoose.Schema;

let userSchema = new Schema({
  username: {type: String, unique: true},
  passwordHash: String,
  messages: [{type: Schema.Types.ObjectId, ref: "Message"}],
  files: [{type: Schema.Types.ObjectId, ref: "File"}]
});

userSchema.methods.comparePassword = async function(password){
  return await bcrypt.compareSync(password, this.passwordHash);
}

userSchema.methods.generateToken = async function(){
  let token = await jwt.sign({username:this.username},'secretKey', {expiresIn:'7d'});
  return token;
}

module.exports = mongoose.model('User', userSchema);
```
这里定义了一个用户数据模型，它有用户名、密码散列值、消息列表和文件列表等字段。

用户模型还定义了两个方法，用于校验密码和生成JWT令牌。

## 4.4 中间件设置
### 4.4.1 检查登录状态中间件
创建`authMiddleware.js`，写入以下代码：
```javascript
function checkAuth(req, res, next){
  if(!req.session ||!req.session.userId){
    req.flash('warning', '请先登录！');
    return res.redirect('/login');
  }
  next();
}

module.exports = checkAuth;
```
这个中间件检查当前请求是否已登录，如果没登录则跳转至登录页。

### 4.4.2 设置跨域中间件
创建`corsMiddleware.js`，写入以下代码：
```javascript
const cors = require('cors');

function allowCrossDomain(req, res, next){
  const whiteList = [
    /^\/.+\.html$/, // html页面
    /^\/api\/.+/ // API接口
  ];

  for(let i=0;i<whiteList.length;i++){
    if(whiteList[i].test(req.url)){
      next();
      return;
    }
  }
  
  cors({origin:"*"})(req,res,next);
}

module.exports = allowCrossDomain;
```
这个中间件设置允许跨域的白名单。对于HTML页面和API接口，允许跨域请求；对于其他资源，拒绝跨域请求。

## 4.5 路由设置
创建`routes.js`，写入以下代码：
```javascript
const express = require('express');
const router = express.Router();
const authMiddleware = require('./middleware/authMiddleware');
const fileController = require('./controller/fileController');
const messageController = require('./controller/messageController');
const userController = require('./controller/userController');

router.get('/', function(req, res){
  res.render('index');
});

router.get('/register', function(req, res){
  res.render('register');
});

router.post('/register', userController.register);

router.get('/login', function(req, res){
  res.render('login');
});

router.post('/login', userController.login);

router.all('*', authMiddleware); // 所有路由均需登录态

router.route('/files')
 .post(fileController.upload) // 文件上传接口
 .delete(fileController.remove); // 文件删除接口

router.route('/messages/:toUsername')
 .get(messageController.list) // 获取指定用户的消息列表
 .post(messageController.send); // 发送私信接口

router.route('/users/:username')
 .get(userController.profile); // 查看个人资料接口

router.route('/logout')
 .get(userController.logout); // 退出登录接口

module.exports = router;
```
这里定义了一系列路由规则。每个路由对应一个控制器，负责处理相应的业务逻辑。同时，我们定义了登录态检查中间件和跨域设置中间件，它们分别作用于各个路由。

## 4.6 控制器设置
### 4.6.1 认证控制器
创建`authController.js`，写入以下代码：
```javascript
const passport = require('passport');
const bcrypt = require('bcrypt');
const User = require('../models/UserModel');

const saltRounds = 10; // hash密码的迭代次数

async function validateRegister(req, res, next){
  try{
    let user = new User({
      username: req.body.username, 
      passwordHash: await bcrypt.hash(req.body.password, saltRounds)
    });

    await user.save();
    
    req.flash('success', `成功注册账号 ${user.username}!`);
    res.redirect('/login');
  }catch(err){
    req.flash('danger', err.message);
    res.redirect('/register');
  }
}

async function validateLogin(req, res, next){
  try{
    let user = await User.findOne({username:req.body.username}).exec();
    if(!user){
      throw new Error(`不存在该账号 ${req.body.username}`);
    }

    if(!(await user.comparePassword(req.body.password))){
      throw new Error(`密码错误`);
    }

    req.session.userId = user._id;
    res.redirect('/');
  }catch(err){
    req.flash('danger', err.message);
    res.redirect('/login');
  }
}

function logout(req, res){
  delete req.session.userId;
  req.flash('success', '成功注销登录！');
  res.redirect('/login');
}

module.exports = {validateRegister, validateLogin, logout};
```
这个文件定义了验证注册、登录、退出的方法。注意，这里用到了passport库。关于Passport的使用请自行查阅官方文档。

### 4.6.2 文件控制器
创建`fileController.js`，写入以下代码：
```javascript
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const User = require('../models/UserModel');
const File = require('../models/FileModel');

async function upload(req, res){
  try{
    let user = await User.findById(req.session.userId).populate('files').exec();
    let file = new File(req.file);
    file.owner = user._id;
    await file.save();
    user.files.push(file);
    await user.save();

    res.json(file);
  }catch(err){
    res.status(500).json({message: err.message});
  }
}

async function remove(req, res){
  try{
    let user = await User.findById(req.session.userId).populate('files').exec();
    let fileId = req.query.id;
    let fileToRemove = null;

    for(let i=0;i<user.files.length;i++){
      if(String(user.files[i]._id) === fileId){
        fileToRemove = user.files[i];
        break;
      }
    }

    if(!fileToRemove){
      throw new Error('未找到文件');
    }

    let filePath = `${__dirname}/../uploads/${fileToRemove.name}`;

    fs.unlinkSync(filePath);
    await File.findByIdAndRemove(fileId);

    res.json({message: '成功删除文件'});
  }catch(err){
    res.status(500).json({message: err.message});
  }
}

async function transform(req, res){
  try{
    let fileName = req.params.filename;
    let extName = path.extname(fileName);
    let targetSize = parseInt(req.query.size);

    let filePath = `${__dirname}/../uploads/${fileName}`;
    let targetPath = `${__dirname}/../thumbnails/${fileName}`;

    sharp(filePath).resize(targetSize).toFile(targetPath);

    let thumbnailStats = fs.statSync(targetPath);

    res.setHeader('Content-Type', 'image/' + extName.slice(1));
    res.setHeader('Content-Length', thumbnailStats.size);
    res.write(fs.readFileSync(targetPath), 'binary');
    res.end();
  }catch(err){
    res.status(500).json({message: err.message});
  }
}

module.exports = {upload, remove, transform};
```
这个文件定义了上传文件、删除文件、生成缩略图的方法。

### 4.6.3 消息控制器
创建`messageController.js`，写入以下代码：
```javascript
const Message = require('../models/MessageModel');

async function list(req, res){
  try{
    let user = await User.findById(req.session.userId).populate('messages').exec();
    let messages = [];

    for(let i=0;i<user.messages.length;i++){
      if(String(user.messages[i].toUser._id) === req.params.toUsername){
        messages.unshift(user.messages[i]);
      }else if(String(user.messages[i].fromUser._id) === req.params.toUsername){
        messages.push(user.messages[i]);
      }
    }

    res.json(messages);
  }catch(err){
    res.status(500).json({message: err.message});
  }
}

async function send(req, res){
  try{
    let sender = await User.findById(req.session.userId).exec();
    let receiver = await User.findOne({username: req.body.toUsername}).exec();

    if(!receiver){
      throw new Error(`不存在该用户 ${req.body.toUsername}`);
    }

    let msg = new Message({content: req.body.content, fromUser: sender, toUser: receiver});
    await msg.save();

    receiver.messages.push(msg);
    await receiver.save();

    sender.messages.push(msg);
    await sender.save();

    res.json(msg);
  }catch(err){
    res.status(500).json({message: err.message});
  }
}

module.exports = {list, send};
```
这个文件定义了获取私信列表和发送私信的方法。

### 4.6.4 用户控制器
创建`userController.js`，写入以下代码：
```javascript
const passport = require('passport');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const flash = require('connect-flash');

const User = require('../models/UserModel');

async function profile(req, res){
  try{
    let user = await User.findOne({_id:req.session.userId}).populate(['messages', 'files']).exec();
    res.render('profile', {user});
  }catch(err){
    res.status(500).json({message: err.message});
  }
}

async function editProfile(req, res){
  try{
    let user = await User.findByIdAndUpdate(req.session.userId, {$set:{username: req.body.username}}, {new:true}).exec();
    req.flash('success', `成功更新用户名为 ${user.username}`);
    res.redirect('/profile');
  }catch(err){
    req.flash('danger', err.message);
    res.redirect('/edit-profile');
  }
}

async function changePassword(req, res){
  try{
    let user = await User.findById(req.session.userId).exec();
    if(!(await user.comparePassword(req.body.oldPassword))){
      throw new Error('旧密码错误!');
    }

    if(req.body.newPassword!== req.body.confirmPassword){
      throw new Error('两次输入的新密码不一致!');
    }

    user.passwordHash = await bcrypt.hash(req.body.newPassword, saltRounds);
    await user.save();

    req.flash('success', '成功修改密码！');
    res.redirect('/profile');
  }catch(err){
    req.flash('danger', err.message);
    res.redirect('/change-password');
  }
}

function getResetToken(req, res){
  passport.authenticate('local')(req, res, () => {
    req.session.resetEmail = req.body.email;
    req.flash('info', '重置密码链接已发送到您的邮箱！请点击邮件中的链接来设置新密码。');
    res.redirect('/forgot-password');
  });
}

async function resetPassword(req, res){
  try{
    let user = await User.findOne({email: req.session.resetEmail}).exec();
    if(!user){
      throw new Error('无效的邮箱地址！');
    }

    user.passwordHash = await bcrypt.hash(req.body.password, saltRounds);
    await user.save();

    req.flash('success', '成功重置密码！');
    res.redirect('/login');
  }catch(err){
    req.flash('danger', err.message);
    res.redirect('/reset-password?token=' + req.query.token);
  }
}

async function generateToken(req, res){
  try{
    let user = await User.findOne({username:req.body.username}).exec();
    if(!user){
      throw new Error(`不存在该账号 ${req.body.username}`);
    }

    let token = await user.generateToken();
    res.json({token});
  }catch(err){
    res.status(500).json({message: err.message});
  }
}

module.exports = {
  profile,
  editProfile,
  changePassword,
  getResetToken,
  resetPassword,
  generateToken
};
```
这个文件定义了查看个人资料、编辑个人资料、修改密码、获取重置密码邮箱验证码的方法、重置密码的方法、生成JWT令牌的方法。注意，这里仍然用到了passport库。