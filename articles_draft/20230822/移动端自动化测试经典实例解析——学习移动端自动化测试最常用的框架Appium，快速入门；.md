
作者：禅与计算机程序设计艺术                    

# 1.简介
  

“移动端自动化测试”是一个比较新的领域，也是非常热门的一个研究方向。随着移动互联网的爆炸式增长，移动应用的使用场景越来越广泛，相应的移动端自动化测试也逐渐火爆起来。如何高效、自动地进行移动端自动化测试成为一个重要的话题。本文将以appium为例，带领大家了解移动端自动化测试的常用工具及其用法。通过掌握这个测试工具，读者可以快速上手编写移动端自动化测试脚本并解决一些日益凸显的自动化测试难点。以下为正文。
# 2.测试工具介绍
## 什么是Appium?
Appium是开源的自动化测试工具，支持iOS，Android等多个平台的移动端自动化测试。它基于开源项目Selenium WebDriver扩展了移动端的特性，能够更好地支持各种移动设备。它是基于Node.js开发的，其作者是开源项目Apache基金会的成员之一，他曾在Facebook工作过。
## Appium安装配置
### 安装
Appium安装需要先安装node.js环境。之后在终端执行以下命令安装Appium:
```
npm install -g appium
```
或者也可以选择使用yarn安装:
```
yarn global add appium
```
### 配置
首先需要启动Appium服务。在终端输入命令：
```
appium
```
然后打开浏览器访问http://localhost:4723/wd/hub，看到下图说明Appium服务已正常启动：
接下来就是对手机或模拟器进行设置，可以使用Appium自带的App设置Appium环境。设置完成后，就可以愉快地编写测试用例了！
## 测试用例
下面就以测试豆瓣FM登录页面登录功能为例，编写Appium自动化测试脚本。
### 登录页面元素定位
一般来说，移动端页面元素的定位都是比较复杂的，需要借助Appium提供的UIAutomatorHelper，XPathHelper等辅助类来定位页面元素。为了便于理解，这里只给出控件属性、控件名称的示例，具体定位方式还需要结合Appium App自带的UI检查工具获取。
| 属性 | 控件名称 | XPath路径 |
| --- | --- | --- |
| 用户名输入框 | username_input | //XCUIElementTypeTextField[1] |
| 密码输入框 | password_input | //XCUIElementTypeSecureTextField[1] |
| 登录按钮 | login_button | //XCUIElementTypeButton[@name="登 录"] |
### 编写脚本
下面是Appium测试用例的编写过程：
```javascript
const wd = require('webdriverio');
const assert = require('assert');

// 创建 webdriver 实例
let driver = await wd.remote({
    'hostname': 'localhost',
    'port': 4723,
    'path': '/wd/hub'
});

try {
    // 连接设备或模拟器
    let caps = {
        platformName: 'iOS',
        deviceName: 'iPhone XS Max Simulator',
        udid: '2D8F9B2A-DAB5-43C5-9F72-E9350BFE79EB',
        automationName: 'XCUITest',
        bundleId: 'fm.douban.doubanFM'
    };

    await driver.init(caps);

    console.log("driver session started...");
    
    // 等待页面加载完毕
    await driver.waitUntil(() => {
        return driver.isExisting('#username_input')
           .then((exist) => exist && driver.isExisting('#password_input'))
           .then((exist) => exist && driver.isExisting('#login_button'));
    }, 1000 * 60);

    console.log("page loaded.");

    // 获取元素对象
    const inputUsername = await driver.$("#username_input");
    const inputPassword = await driver.$("#password_input");
    const buttonLogin = await driver.$("#login_button");

    // 执行点击事件
    await buttonLogin.click();

    // 设置用户名和密码
    await inputUsername.setValue('your_username@douban.com');
    await inputPassword.setValue('your_password');

    // 执行点击事件
    await buttonLogin.click();

    // 判断是否登录成功
    try {
        await driver.waitForXpath("//XCUIElementTypeStaticText[contains(@label,'登录失败')]", 5000, true);

        throw new Error('login failed.');
    } catch (error) {}

    console.log('login succeeded!');
} finally {
    // 关闭浏览器
    await driver.quit();
}
```
脚本分为三步：
1. 创建 webdriver 实例
2. 连接设备或模拟器
3. 定位页面元素并执行相关操作

最后判断是否登录成功。如有报错，则抛出异常。
### 执行脚本
打开终端，切换到当前目录，执行如下命令运行脚本：
```
node test.js
```
输出结果应该类似于：
```
driver session started...
page loaded.
login succeeded!
```
如果登录成功，则输出“login succeeded!”，否则会报超时错误，请重新运行脚本。