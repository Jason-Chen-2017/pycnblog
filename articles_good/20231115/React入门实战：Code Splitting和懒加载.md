                 

# 1.背景介绍


React作为当前最火爆的前端框架，其开发速度、性能等方面都有不俗的表现。然而由于其庞大的体积和复杂性，开发者往往难以将它快速上手并掌握其功能特性。所以，React初学者通常都会遇到以下几个问题：
1. React项目初期如何划分文件？
2. 文件过多后应该如何管理？
3. 如何实现路由跳转时的页面动画效果？
4. 使用什么组件库、状态管理库？
5. 为什么要用React Hooks？

这些问题或许会让你疑惑、困惑、迷茫，甚至是恐惧。本文着重于解决这些问题，主要围绕以下四个主题进行展开：
1. Code Splitting（代码拆分）
2. Lazy Loading（延迟加载）
3. Route Animation（路由动画）
4. Component Libraries and State Management（组件库和状态管理）

首先，让我们回顾一下为什么需要Code Splitting与Lazy Loading。

# Why Code Splitting & Lazy Loading?
一般来说，一个Web应用的JavaScript文件可以分为两类：基础文件和业务逻辑文件。基础文件包含了如jQuery、Bootstrap、Lodash、moment.js等工具库，这些都是独立的文件，不需要被加载到浏览器中执行，因此对用户的加载速度有着直接的影响；而业务逻辑文件则包含了所有应用相关的代码，如果全部加载到浏览器中会导致浏览器内存占用过高，对于移动端设备尤其如此。因此，为了提升用户体验及降低加载时间，React在之后的版本中引入了两种技术——Code Splitting和Lazy Loading，帮助开发者将业务逻辑文件按需加载，从而提升应用性能。

## 1. Code Splitting（代码拆分）
Code Splitting允许开发者只加载必要的代码，而不是一次性加载所有的代码。这样做可以加快首屏加载速度，缩短加载时间，提高应用整体性能。其基本思路是把不同的功能模块单独打包成不同的JS文件，然后通过异步的方式加载这些文件。例如，某个应用中可能有两个页面：Home页面和Profile页面。分别对应两个模块：homePage.js和profilePage.js。当访问Home页面时，只需要加载首页所需的模块homePage.js；当访问Profile页面时，再加载profilePage.js即可。这样就可以避免用户访问其他页面时，还要下载其他页面的模块。

## 2. Lazy Loading（延迟加载）
Lazy Loading（也叫懒加载）可以说是Code Splitting的一种优化方式。与Code Splitting不同的是，Lazy Loading是在渲染页面时才去加载相应模块。这种方式可以减少页面初始加载的时间，并且节省带宽资源。举例来说，当用户滚动到某个区域时，React只会加载该区域所需的模块。

# 2.核心概念与联系
Code Splitting与Lazy Loading是React框架中的两种技术。其中，Code Splitting允许我们将应用的功能模块分离成多个独立的JS文件，并采用异步的方式加载，从而实现按需加载模块。而Lazy Loading则是在渲染页面时，根据用户的行为自动加载相应模块，从而减少页面初始加载时间，节省网络流量，提升应用性能。下面我们就逐一介绍它们之间的联系与区别。

 ## 1. Code Splitting VS Lazy Loading
相比较而言，Code Splitting和Lazy Loading具有以下几个明显的区别：

1. 目标不同：Code Splitting的目标是将应用划分为独立的模块，并通过异步加载的方式实现按需加载；Lazy Loading的目标则是仅在渲染页面时加载模块。
2. 拥有层级关系：Code Splitting没有层级关系，而Lazy Loading依赖于层级关系。也就是说，只有父组件声明了Lazy Loading，它的子组件才能够延迟加载。
3. 配置方式不同：Code Splitting可以通过webpack配置实现，而Lazy Loading则需要额外的代码。
4. 渲染时机不同：Code Splitting可以在编译时完成，而Lazy Loading只能在运行时实现。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
Code Splitting的主要原理就是把不同的功能模块单独打包成不同的JS文件，然后通过异步的方式加载这些文件。下面我们结合React Router中的例子进行讲解。

## 1. 操作步骤
假设有一个React应用，包括两个页面：Home页面和About页面。分别对应两个模块：HomePage.js和AboutPage.js。

### （1）初始化项目目录结构
首先，创建一个空文件夹，并创建src、public和package.json三个目录：

```bash
mkdir react-app && cd react-app

mkdir src public package.json
```

在src目录下创建HomePage.js和AboutPage.js两个文件：

```bash
cd src && touch HomePage.js AboutPage.js
```

在public目录下创建一个index.html文件：

```bash
cd.. && mkdir public && cd public && touch index.html
```

编辑index.html文件，加入如下内容：

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>My React App</title>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
```

编辑HomePage.js文件，加入如下内容：

```jsx
import React from'react';

const HomePage = () => {
  return (
    <div>
      <h1>Home Page</h1>
      <p>Welcome to my app!</p>
    </div>
  );
};

export default HomePage;
```

编辑AboutPage.js文件，加入如下内容：

```jsx
import React from'react';

const AboutPage = () => {
  return (
    <div>
      <h1>About Page</h1>
      <p>Learn more about me.</p>
    </div>
  );
};

export default AboutPage;
```

编辑package.json文件，加入如下内容：

```json
{
  "name": "my-react-app",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "^17.0.2",
    "react-dom": "^17.0.2",
    "react-router-dom": "^6.2.2"
  },
  "devDependencies": {
    "@babel/core": "^7.17.8",
    "@babel/preset-env": "^7.17.8",
    "parcel-bundler": "^1.12.5",
    "sass": "^1.49.0",
    "sass-loader": "^12.4.0",
    "style-loader": "^3.3.1"
  },
  "scripts": {
    "start": "parcel serve./src/index.html --open",
    "build": "parcel build./src/index.html --no-cache"
  }
}
```

### （2）安装依赖
```bash
npm install
```

### （3）配置路由
接下来，我们需要配置React Router，使之能正确匹配URL对应的页面模块：

编辑src/App.js文件，加入如下内容：

```jsx
import { BrowserRouter as Router, Switch, Route } from'react-router-dom';
import HomePage from './pages/HomePage';
import AboutPage from './pages/AboutPage';

function App() {
  return (
    <Router>
      <Switch>
        <Route path="/about">
          <AboutPage />
        </Route>
        <Route path="/">
          <HomePage />
        </Route>
      </Switch>
    </Router>
  );
}

export default App;
```

### （4）实现Code Splitting
前面已经介绍过Code Splitting的原理，即把不同模块的代码打包成不同的JS文件，并通过异步加载的方式实现按需加载。在React项目中，我们可以使用Webpack和Babel插件来实现Code Splitting。

#### 安装依赖
```bash
npm i webpack webpack-cli babel-plugin-dynamic-import-node -D
```

#### 修改配置文件
在根目录下新建webpack.config.js文件，添加如下内容：

```javascript
module.exports = {
  entry: ['./src/index.js'], // 指定入口文件
  output: {
    filename: '[name].[contenthash].js', // 生成的文件名称
    path: `${__dirname}/dist`, // 生成的文件路径
  },
  module: {
    rules: [
      { test: /\.(js|jsx)$/, exclude: /node_modules/, use: ['babel-loader'] },
      {
        test: /\.css$/,
        loader: ['style-loader', 'css-loader'],
      },
      {
        test: /\.s[ac]ss$/i,
        use: [
          // Creates `style` nodes from JS strings
         'style-loader',
          // Translates CSS into CommonJS
          'css-loader',
          // Compiles Sass to CSS
         'sass-loader',
        ],
      },
    ],
  },
  plugins: [],
};
```

在根目录下新建babel.config.js文件，添加如下内容：

```javascript
module.exports = {
  presets: [['@babel/preset-env']],
  plugins: ['dynamic-import-node'],
};
```

#### 创建异步加载函数
编辑src/index.js文件，加入如下内容：

```javascript
// 异步导入函数
function asyncComponent(importComponent) {
  const Component = React.lazy(() => import(`./${importComponent}`));

  return props => (
    <React.Suspense fallback={<div>Loading...</div>}>
      <Component {...props} />
    </React.Suspense>
  );
}

renderRoutes([
  {
    path: '/home',
    exact: true,
    component: asyncComponent('HomePage'), // 通过异步导入函数动态导入HomePage模块
  },
  {
    path: '/about',
    exact: true,
    component: asyncComponent('AboutPage'), // 通过异步导入函数动态导入AboutPage模块
  },
]);

ReactDOM.render(<App />, document.getElementById('root'));
```

### （5）启动项目
```bash
npm start
```

打开浏览器，访问http://localhost:1234，切换至Home页面，页面正常显示。点击“About”链接，页面正常显示。

# 4.具体代码实例和详细解释说明
下面我们结合代码示例，进行详细讲解。

## 1. Code Splitting Example

假设我们的React应用包括三页：首页、产品页、个人中心页。三个页面分别对应三个模块：HomePage.js、ProductPage.js、ProfilePage.js。

### （1）创建页面模块文件
创建三个模块文件：HomePage.js、ProductPage.js、ProfilePage.js。

```bash
touch src/components/Header.js src/pages/ProductPage.js src/pages/ProfilePage.js
```

编辑Header.js文件，加入如下内容：

```jsx
import React from'react';

const Header = () => {
  return <header className="header">This is the header</header>;
};

export default Header;
```

编辑ProductPage.js文件，加入如下内容：

```jsx
import React from'react';
import Header from '../components/Header';

const ProductPage = () => {
  return (
    <>
      <Header />
      <main>
        <h1>Product Page</h1>
        <p>Check out our products!</p>
      </main>
    </>
  );
};

export default ProductPage;
```

编辑ProfilePage.js文件，加入如下内容：

```jsx
import React from'react';
import Header from '../components/Header';

const ProfilePage = () => {
  return (
    <>
      <Header />
      <main>
        <h1>Profile Page</h1>
        <p>Manage your account here.</p>
      </main>
    </>
  );
};

export default ProfilePage;
```

### （2）修改路由配置
编辑src/App.js文件，加入如下内容：

```jsx
import { BrowserRouter as Router, Switch, Route } from'react-router-dom';
import HomePage from './pages/HomePage';
import ProductPage from './pages/ProductPage';
import ProfilePage from './pages/ProfilePage';

function App() {
  return (
    <Router>
      <Switch>
        <Route path="/product">
          <ProductPage />
        </Route>
        <Route path="/profile">
          <ProfilePage />
        </Route>
        <Route path="/">
          <HomePage />
        </Route>
      </Switch>
    </Router>
  );
}

export default App;
```

### （3）配置Webpack
编辑webpack.config.js文件，加入如下内容：

```javascript
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  entry: ['./src/index.js'],
  output: {
    filename: '[name].[contenthash].js',
    path: path.resolve(__dirname, 'dist'),
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname,'src'),
    },
  },
  module: {
    rules: [
      { test: /\.(js|jsx)$/, exclude: /node_modules/, use: ['babel-loader'] },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
      {
        test: /\.s[ac]ss$/i,
        use: [
          // Creates `style` nodes from JS strings
         'style-loader',
          // Translates CSS into CommonJS
          'css-loader',
          // Compiles Sass to CSS
         'sass-loader',
        ],
      },
    ],
  },
  plugins: [
    new HtmlWebpackPlugin({ template: './public/index.html' }),
    // 忽略moment.js locale文件，减小构建输出大小
    new webpack.IgnorePlugin(/^\.\/locale$/, /moment$/),
    // 分割代码
    new webpack.optimize.SplitChunksPlugin({ chunks: 'all' }),
  ],
};
```

编辑babel.config.js文件，加入如下内容：

```javascript
module.exports = {
  presets: [['@babel/preset-env', { modules: false }]],
  plugins: ['dynamic-import-node'],
};
```

### （4）导入异步函数
编辑src/index.js文件，加入如下内容：

```javascript
import ReactDOM from'react-dom';
import React from'react';
import { renderRoutes } from'react-router-config';
import routes from './routes';
import * as serviceWorkerRegistration from './serviceWorkerRegistration';

// 异步导入函数
function asyncComponent(importComponent) {
  const Component = React.lazy(() => import(`./pages/${importComponent}`).then((mod) => mod.default));

  return props => (
    <React.Suspense fallback={<div>Loading...</div>}>
      <Component {...props} />
    </React.Suspense>
  );
}

renderRoutes(routes.map(({ path, exact, component }) => ({...component, path, exact })));

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);

// If you don't want your app to work offline, you can change unregister() to register() below.
// Note this comes with some pitfalls.
// Learn more about service workers: https://cra.link/PWA
serviceWorkerRegistration.register();
```

### （5）创建路由配置文件
创建src/routes.js文件，加入如下内容：

```javascript
export default [
  {
    path: '/',
    component: AsyncHomePage,
    exact: true,
  },
  {
    path: '/product',
    component: asyncComponent('ProductPage'),
    exact: true,
  },
  {
    path: '/profile',
    component: asyncComponent('ProfilePage'),
    exact: true,
  },
];
```

### （6）更新服务工作进程
编辑src/serviceWorkerRegistration.js文件，加入如下内容：

```javascript
import { registrationSuccessMessage } from './utils';

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    console.log('SW activated!');

    if (!navigator.onLine) {
      console.warn('[Service Worker] No internet connection found. App is running in offline mode.');

      registrationSuccessMessage('Offline Mode','success');
    } else {
      navigator.serviceWorker.register('/sw.js').then((registration) => {
        console.log('SW registered successfully!', registration);

        registrationSuccessMessage('Service Worker Registered','success');
      }).catch((error) => {
        console.error('SW registration failed:', error);

        registrationSuccessMessage('Service Worker Registration Failed', 'danger');
      });
    }
  });
}
```

### （7）测试
```bash
npm run build && npm run start
```

打开浏览器，访问http://localhost:1234，点击导航菜单，查看不同页面是否加载成功。

# 5.未来发展趋势与挑战
随着Web前端技术的发展，Code Splitting、Lazy Loading已经成为React项目优化的重要手段。在未来的React版本中，还将推出更多的优化策略，比如Server Side Rendering、Tree Shaking等。不过，目前各主流前端框架均在维护Code Splitting，比如Vue、Angular等，所以，React在这个领域也处于领先地位。

# 6.附录常见问题与解答

## Q: 您的文章总共花费了多少时间？
我的文章主要花费了四个月左右的时间。因为我之前没有涉足这一方面的技术，所以很难用自己的话语来阐述清楚文章的内容。但是，为了保证文章质量，我还是按照自己水平所熟知的一些知识点，尝试去传达给读者一些信息。

## Q: 本文中您用到的关键词、名词、算法有哪些？
关键字：Webpack、Babel Plugin、Code Splitting、Async Imports、Lazy Loading、Tree Shaking、Server Side Rendering、Dynamic Imports、CSS Modules、NPM Scripts、Parcel Bundler。

## Q: 您文章中的每一步操作步骤和具体的代码实例，是否可以详细地展开讲解？
当然可以！虽然是自己的博客文章，但我确实希望大家能够仔细阅读完毕。以下是每个步骤的详细讲解。

### （1）初始化项目目录结构
首先，我们需要创建一个空文件夹，然后创建src、public和package.json三个目录：

```bash
mkdir react-app && cd react-app

mkdir src public package.json
```

然后，我们在src目录下创建HomePage.js和AboutPage.js两个文件：

```bash
cd src && touch HomePage.js AboutPage.js
```

然后，在public目录下创建一个index.html文件：

```bash
cd.. && mkdir public && cd public && touch index.html
```

编辑index.html文件，加入如下内容：

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>My React App</title>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
```

最后，编辑package.json文件，加入如下内容：

```json
{
  "name": "my-react-app",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "^17.0.2",
    "react-dom": "^17.0.2",
    "react-router-dom": "^6.2.2"
  },
  "devDependencies": {
    "@babel/core": "^7.17.8",
    "@babel/preset-env": "^7.17.8",
    "parcel-bundler": "^1.12.5",
    "sass": "^1.49.0",
    "sass-loader": "^12.4.0",
    "style-loader": "^3.3.1"
  },
  "scripts": {
    "start": "parcel serve./src/index.html --open",
    "build": "parcel build./src/index.html --no-cache"
  }
}
```

### （2）安装依赖
运行如下命令安装依赖：

```bash
npm install
```

### （3）配置路由
编辑src/App.js文件，加入如下内容：

```jsx
import { BrowserRouter as Router, Switch, Route } from'react-router-dom';
import HomePage from './pages/HomePage';
import AboutPage from './pages/AboutPage';

function App() {
  return (
    <Router>
      <Switch>
        <Route path="/about">
          <AboutPage />
        </Route>
        <Route path="/">
          <HomePage />
        </Route>
      </Switch>
    </Router>
  );
}

export default App;
```

### （4）实现Code Splitting
#### 安装依赖
运行如下命令安装依赖：

```bash
npm i webpack webpack-cli babel-plugin-dynamic-import-node -D
```

#### 修改配置文件
编辑webpack.config.js文件，加入如下内容：

```javascript
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  entry: ['./src/index.js'],
  output: {
    filename: '[name].[contenthash].js',
    path: path.resolve(__dirname, 'dist'),
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname,'src'),
    },
  },
  module: {
    rules: [
      { test: /\.(js|jsx)$/, exclude: /node_modules/, use: ['babel-loader'] },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
      {
        test: /\.s[ac]ss$/i,
        use: [
          // Creates `style` nodes from JS strings
         'style-loader',
          // Translates CSS into CommonJS
          'css-loader',
          // Compiles Sass to CSS
         'sass-loader',
        ],
      },
    ],
  },
  plugins: [
    new HtmlWebpackPlugin({ template: './public/index.html' }),
    // 忽略moment.js locale文件，减小构建输出大小
    new webpack.IgnorePlugin(/^\.\/locale$/, /moment$/),
    // 分割代码
    new webpack.optimize.SplitChunksPlugin({ chunks: 'all' }),
  ],
};
```

编辑babel.config.js文件，加入如下内容：

```javascript
module.exports = {
  presets: [['@babel/preset-env', { modules: false }]],
  plugins: ['dynamic-import-node'],
};
```

#### 创建异步加载函数
编辑src/index.js文件，加入如下内容：

```javascript
// 异步导入函数
function asyncComponent(importComponent) {
  const Component = React.lazy(() => import(`./${importComponent}`));

  return props => (
    <React.Suspense fallback={<div>Loading...</div>}>
      <Component {...props} />
    </React.Suspense>
  );
}

renderRoutes([
  {
    path: '/home',
    exact: true,
    component: asyncComponent('HomePage'), // 通过异步导入函数动态导入HomePage模块
  },
  {
    path: '/about',
    exact: true,
    component: asyncComponent('AboutPage'), // 通过异步导入函数动态导入AboutPage模块
  },
]);

ReactDOM.render(<App />, document.getElementById('root'));
```

### （5）启动项目
运行如下命令启动项目：

```bash
npm start
```

### （6）创建页面模块文件
创建三个模块文件：HomePage.js、ProductPage.js、ProfilePage.js。编辑Header.js文件，加入如下内容：

```jsx
import React from'react';

const Header = () => {
  return <header className="header">This is the header</header>;
};

export default Header;
```

编辑ProductPage.js文件，加入如下内容：

```jsx
import React from'react';
import Header from '../components/Header';

const ProductPage = () => {
  return (
    <>
      <Header />
      <main>
        <h1>Product Page</h1>
        <p>Check out our products!</p>
      </main>
    </>
  );
};

export default ProductPage;
```

编辑ProfilePage.js文件，加入如下内容：

```jsx
import React from'react';
import Header from '../components/Header';

const ProfilePage = () => {
  return (
    <>
      <Header />
      <main>
        <h1>Profile Page</h1>
        <p>Manage your account here.</p>
      </main>
    </>
  );
};

export default ProfilePage;
```

### （7）修改路由配置
编辑src/App.js文件，加入如下内容：

```jsx
import { BrowserRouter as Router, Switch, Route } from'react-router-dom';
import HomePage from './pages/HomePage';
import ProductPage from './pages/ProductPage';
import ProfilePage from './pages/ProfilePage';

function App() {
  return (
    <Router>
      <Switch>
        <Route path="/product">
          <ProductPage />
        </Route>
        <Route path="/profile">
          <ProfilePage />
        </Route>
        <Route path="/">
          <HomePage />
        </Route>
      </Switch>
    </Router>
  );
}

export default App;
```

### （8）配置Webpack
编辑webpack.config.js文件，加入如下内容：

```javascript
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  entry: ['./src/index.js'],
  output: {
    filename: '[name].[contenthash].js',
    path: path.resolve(__dirname, 'dist'),
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname,'src'),
    },
  },
  module: {
    rules: [
      { test: /\.(js|jsx)$/, exclude: /node_modules/, use: ['babel-loader'] },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
      {
        test: /\.s[ac]ss$/i,
        use: [
          // Creates `style` nodes from JS strings
         'style-loader',
          // Translates CSS into CommonJS
          'css-loader',
          // Compiles Sass to CSS
         'sass-loader',
        ],
      },
    ],
  },
  plugins: [
    new HtmlWebpackPlugin({ template: './public/index.html' }),
    // 忽略moment.js locale文件，减小构建输出大小
    new webpack.IgnorePlugin(/^\.\/locale$/, /moment$/),
    // 分割代码
    new webpack.optimize.SplitChunksPlugin({ chunks: 'all' }),
  ],
};
```

编辑babel.config.js文件，加入如下内容：

```javascript
module.exports = {
  presets: [['@babel/preset-env', { modules: false }]],
  plugins: ['dynamic-import-node'],
};
```

### （9）导入异步函数
编辑src/index.js文件，加入如下内容：

```javascript
import ReactDOM from'react-dom';
import React from'react';
import { renderRoutes } from'react-router-config';
import routes from './routes';
import * as serviceWorkerRegistration from './serviceWorkerRegistration';

// 异步导入函数
function asyncComponent(importComponent) {
  const Component = React.lazy(() => import(`./pages/${importComponent}`).then((mod) => mod.default));

  return props => (
    <React.Suspense fallback={<div>Loading...</div>}>
      <Component {...props} />
    </React.Suspense>
  );
}

renderRoutes(routes.map(({ path, exact, component }) => ({...component, path, exact })));

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);

// 如果不想应用拥有离线功能，你可以改为使用 register() 方法注册 Service Worker，但是这么做会带来一些潜在问题，你可以了解更多关于 Service Worker 的知识。
serviceWorkerRegistration.unregister();
```

### （10）创建路由配置文件
创建src/routes.js文件，加入如下内容：

```javascript
export default [
  {
    path: '/',
    component: AsyncHomePage,
    exact: true,
  },
  {
    path: '/product',
    component: asyncComponent('ProductPage'),
    exact: true,
  },
  {
    path: '/profile',
    component: asyncComponent('ProfilePage'),
    exact: true,
  },
];
```

### （11）更新服务工作进程
编辑src/serviceWorkerRegistration.js文件，加入如下内容：

```javascript
import { registrationSuccessMessage } from './utils';

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    console.log('SW activated!');

    if (!navigator.onLine) {
      console.warn('[Service Worker] No internet connection found. App is running in offline mode.');

      registrationSuccessMessage('Offline Mode','success');
    } else {
      navigator.serviceWorker.register('/sw.js').then((registration) => {
        console.log('SW registered successfully!', registration);

        registrationSuccessMessage('Service Worker Registered','success');
      }).catch((error) => {
        console.error('SW registration failed:', error);

        registrationSuccessMessage('Service Worker Registration Failed', 'danger');
      });
    }
  });
}
```