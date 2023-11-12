                 

# 1.背景介绍


React 是 Facebook 提出的一个用于构建用户界面的 Javascript 框架，很多公司如阿里巴巴、美团、腾讯等都在使用它。近年来 React 的热度也越来越高，其中 Parcel 作为 React 脚手架工具的最新玩家也备受青睐。本文将结合 Parcel 以及相关的 Webpack 技术，以入门级的视角带领读者从零开始，搭建自己的 React 项目，熟悉 Parcel 基本配置项及功能，提升开发效率。

# Parcel 简介
Parcel 可以说是目前最火的 React 脚手架工具了。它可以帮助开发者快速构建 React 应用，主要解决的是静态资源加载、编译打包、模块热更新等问题。它采用极速的速度、无配置的特性，帮助开发者零配置的快速上手。它由 JS Foundation 和 OpenJS 基金会联合维护，是 Web 应用的构建工具。

Parcel 在创建时就有很强烈的目标，就是要成为现代前端开发中的“零配置”，并且对所有主流工具链提供支持，包括 Babel、PostCSS、Sass/Less、TypeScript、CSS Modules、WebAssembly、Tree-shaking 等。基于这些特性，开发者可以更加专注于业务逻辑的实现。

总之，Parcel 无疑是一个极具革命性的工具。

# 2.核心概念与联系
## 2.1 基本用法
首先，先来看一下 Parcel 的基本用法。由于 Parcel 支持多种工程化环境，这里只讨论如何在 React 中使用 Parcel。Parcel 的使用非常简单，按照以下步骤即可完成 React 项目的初始化：

1. 安装 Parcel 脚手架工具：
```bash
npm install -g parcel-bundler
```
2. 初始化 React 项目：
```bash
mkdir my-project && cd my-project
npx create-react-app.
```
3. 使用 Parcel 启动项目：
```bash
parcel index.html
```
这样就启动了一个 React 项目，但实际上只有 HTML 文件被 Parcel 识别到了，因为我们并没有安装其他第三方依赖库。为了让 Parcel 更好地工作，我们还需要对其进行一些配置。

## 2.2 Parcel 配置
Parcel 的配置文件是 package.json 中的 scripts 对象下的 start 命令，位于 "start": "parcel index.html" 。此命令指定了 Parcel 执行的默认行为，即启动一个服务，监听源文件变化并自动刷新页面。除此之外，Parcel 还有许多可自定义的配置项，包括以下几类：

1. 入口文件：
默认情况下，Parcel 会从 src/index.js 或 src/index.html（如果存在）中寻找入口文件。但是，为了更好的组织项目结构，我们往往会把入口文件放在根目录下，例如 index.ts/tsx、index.jsx、index.html。所以，我们可以在 package.json 的 scripts 对象下添加 start 命令，修改其值为："parcel./src/index.html --out-dir build --public-url /dist/"。这里 --out-dir 指定了输出路径，--public-url 指定了打包后的资源的访问路径。

2. 插件：
Parcel 默认不开启任何插件，如果需要使用某个插件，则需要在命令前面增加 @babel/core、@babel/preset-env、@babel/plugin-transform-runtime、typescript 等，或者可以直接在配置文件中设置 plugins 字段。

3. 加载器：
Parcel 支持 Loader ，可以对不同类型的文件进行预处理。对于 JSX 和 TypeScript 文件，我们需要使用对应的加载器。比如，对于 JSX 文件，可以使用 react-hot-loader，这是个热更新插件。而对于 TypeScript 文件，则需要使用 ts-loader 来加载。所以，我们可以在 package.json 的 dependencies 字段中添加相应的 loader 依赖，然后在配置文件中设置 loaders 字段，如下所示：
```json
{
  "name": "my-project",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "start": "parcel./src/index.html --out-dir build --public-url /dist/",
    // 其他命令
  },
  "author": "",
  "license": "ISC",
  "dependencies": {
    "react": "^17.0.2",
    "react-dom": "^17.0.2"
  },
  "devDependencies": {
    "@types/node": "^16.9.6",
    "@types/react": "^17.0.34",
    "@types/react-dom": "^17.0.11",
    "parcel-bundler": "^1.12.5",
    "sass": "^1.38.1",
    "style-loader": "^2.0.0",
    "ts-loader": "^9.2.5",
    "typescript": "^4.4.4"
  },
  "browserslist": [
    ">0.2%",
    "not dead",
    "not op_mini all"
  ],
  "loaders": [
    {
      "test": "\\.ts|\\.tsx$",
      "exclude": "/node_modules/",
      "use": ["ts-loader"]
    }
  ]
}
```

4. 路径别名：
Parcel 通过 alias 属性来定义路径别名，使得导入文件或模块更方便。例如，我们可以给 src 路径取一个别名，如下所示：
```json
{
 ...
  "alias": {"@": "./src"}
 ...
}
```
这样就可以在任意位置使用 @ 表示 src 文件夹了。

5. 模块热替换（HMR）：
模块热替换（HMR）是指在运行过程中，Parcel 能够监测到代码变动并只重新执行变动的部分，而不是完全重载页面。当修改某处代码后，Parcel 只更新变动的代码，而不会影响其他部分，从而实现页面局部更新，提升开发效率。

总体来说，Parcel 提供了一系列便捷的配置方式，让开发者在日常开发中能够更加高效的完成任务。

## 2.3 路由配置
React Router 是一个独立的第三方模块，它提供声明式的路由配置语法。它的特点是简单灵活，同时提供了多种导航模式，例如 HashRouter 和 BrowserRouter。下面我们以 BrowserRouter 为例，介绍如何使用它在 React 项目中配置路由。

1. 安装依赖：
```bash
npm i react-router-dom
```

2. 创建路由组件：
创建一个 routes.ts 文件，并导出需要用到的所有路由组件。例如：
```javascript
import React from'react';
import { Route, Switch } from'react-router-dom';
import HomePage from './pages/HomePage';
import AboutPage from './pages/AboutPage';
import NotFoundPage from './pages/NotFoundPage';

const Routes = () => (
  <Switch>
    <Route exact path="/" component={HomePage} />
    <Route path="/about" component={AboutPage} />
    <Route component={NotFoundPage} />
  </Switch>
);

export default Routes;
```

3. 添加路由配置：
在 index.tsx 文件中引入路由组件，并使用 BrowserRouter 渲染出路由匹配结果：
```javascript
import ReactDOM from'react-dom';
import React from'react';
import { BrowserRouter } from'react-router-dom';
import Routes from './routes';

ReactDOM.render(
  <BrowserRouter>
    <Routes />
  </BrowserRouter>,
  document.getElementById('root')
);
```

4. 设置占位符路由：
当浏览器请求的 URL 不在路由列表内时，可以通过设置占位符路由来显示指定的 UI 内容，比如 404 Not Found 页面。

至此，React Router 配置已完成。

## 2.4 CSS 配置
Parcel 使用 SASS 来支持 CSS。我们可以在 package.json 文件中配置 scss 语法：
```json
{
 ...,
  "devDependencies": {
   ...,
    "sass": "^1.38.1",
    "style-loader": "^2.0.0"
  },
  "browserslist": [
    ">0.2%",
    "not dead",
    "not op_mini all"
  ],
  "moduleFileExtensions": ["ts", "tsx", "scss"],
  "resolver": "parcel-plugin-typescript",
  "transformers": {
    "*.{ts,tsx}": ["parcel-transformer-typescript-tsc", "parcel-transformer-pug"]
  },
 ...
}
```
这里我用到的 Parcel Transformer 有两个，分别是 typescript-tsc 和 pug。typescript-tsc 是用来编译 TypeScript 文件的 transformer，而 pug 是用来转换模板语言的 transformer。我们还需要安装这些依赖：
```bash
npm i parcel-transformer-typescript-tsc parcel-transformer-pug parcel-plugin-typescript
```
然后再配置.pug 文件的解析规则：
```json
{
  "name": "my-project",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "start": "parcel./src/index.html --out-dir build --public-url /dist/",
    // 其他命令
  },
  "author": "",
  "license": "ISC",
  "dependencies": {
    "react": "^17.0.2",
    "react-dom": "^17.0.2",
    "react-router-dom": "^5.2.0"
  },
  "devDependencies": {
    "@types/node": "^16.9.6",
    "@types/react": "^17.0.34",
    "@types/react-dom": "^17.0.11",
    "parcel-bundler": "^1.12.5",
    "sass": "^1.38.1",
    "style-loader": "^2.0.0",
    "ts-loader": "^9.2.5",
    "typescript": "^4.4.4",
    "parcel-transformer-typescript-tsc": "^1.0.2",
    "parcel-transformer-pug": "^1.0.2",
    "parcel-plugin-typescript": "^1.0.0"
  },
  "browserslist": [
    ">0.2%",
    "not dead",
    "not op_mini all"
  ],
  "moduleFileExtensions": ["ts", "tsx", "scss"],
  "resolver": "parcel-plugin-typescript",
  "transformers": {
    "*.{ts,tsx}": ["parcel-transformer-typescript-tsc", "parcel-transformer-pug"]
  },
  "alias": {"@": "./src"},
  "extends": "./tsconfig.json",
  "include": ["./src/**/*.ts", "./src/**/*.tsx", "./src/**/*.scss"]
}
```

最后，在项目根目录下新建 src/styles 目录，并编写 style.scss 文件。由于 Parcel 支持模块热替换，所以 CSS 文件不需要额外配置，只要保存后 Parcel 就会自动编译。