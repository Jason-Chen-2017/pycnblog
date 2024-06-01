
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在这篇教程中，我将带领您构建一个基于ReactJS的Ethereum DApp，您可以从零开始学习如何编写ReactJS代码，并创建自己的去中心化应用（DApp）。

如果您对React或者其他任何开发技术不了解，或对Solidity、Web3.js、Truffle等技术细节有疑问，请不要担心！本文提供了完整的流程，所以您可以一步步完成我们的目标。

# 2.准备工作

首先，您需要安装以下工具：


您还需要一些基本的前端开发知识，比如HTML、CSS、JavaScript、TypeScript等。本文不会涉及这些内容，但如果您想进阶，请参考相关资源。

# 3.项目结构

```
├── app
│   ├── src
│   │   └── components
│   │       ├── App.tsx    # 根组件
│   │       ├── Button.tsx # 按钮组件
│   │       ├── Card.tsx   # 卡片组件
│   │       └── Input.tsx  # 输入框组件
│   ├── index.css            # 全局样式文件
│   ├── index.html           # HTML模板文件
│   └── index.tsx            # 入口文件
└── test                      # 测试用例目录
    └── contracts             # Solidity合约源文件目录
        ├── SimpleStorage.sol # 简单的存储合约
```

我们将会创建一个React项目，这个项目主要由三个部分组成：

1. `src`目录：主要存放所有React组件源码；
2. `index.*`文件：主要存放静态页面模板、全局样式和项目入口文件；
3. `test`目录：存放测试用例，后面会详细介绍如何编写测试用例。

# 4.项目环境配置

接下来，我们将设置项目的开发环境。

## 创建项目

首先，我们需要创建一个空目录，然后进入该目录执行如下命令：

```bash
npm init -y
touch README.md package.json tsconfig.json.gitignore
mkdir app && cd app
mkdir src && touch src/.gitkeep
echo "module.exports = { \"compilerOptions\": { \"esModuleInterop\": true } }" > jsconfig.json
```

上述命令执行后，会生成如下文件：

- `package.json`：描述项目依赖和配置信息；
- `.gitignore`：定义Git忽略提交的文件列表；
- `README.md`：项目说明文档；
- `tsconfig.json`：TypeScript编译配置文件；
- `jsconfig.json`：针对JavaScript项目的编译配置文件；
- `app/`：项目源码目录；
  - `src/`：源码目录；
    - `.gitkeep`：占位符文件，防止Git认为目录为空而拒绝提交；
  - `index.[html|css]`：静态页面模板和全局样式文件；
  - `index.tsx`：入口文件，运行此文件将启动React项目。

## 安装依赖

为了使项目正常运行，我们需要安装一些必要的依赖包：

```bash
npm install --save react react-dom @types/react @types/react-dom typescript webpack webpack-cli html-webpack-plugin css-loader style-loader sass-loader node-sass @babel/core babel-loader @babel/preset-env eslint eslint-plugin-react @typescript-eslint/parser @typescript-eslint/eslint-plugin
```

上述命令会安装以下依赖包：

- `react`、`react-dom`: 管理UI组件；
- `@types/react`、`@types/react-dom`: 提供React类型定义；
- `typescript`: 编译器；
- `webpack`、`webpack-cli`: 模块打包工具；
- `html-webpack-plugin`: 生成HTML插件；
- `css-loader`/`style-loader`: 加载CSS文件；
- `sass-loader`/`node-sass`: 支持SCSS文件；
- `@babel/core`、`babel-loader`: ES6转ES5工具；
- `@babel/preset-env`: ES6转ES5规则集；
- `eslint`、`eslint-plugin-react`: JavaScript代码风格检查；
- `@typescript-eslint/parser`、`@typescript-eslint/eslint-plugin`: TypeScript代码风格检查；

## 初始化项目

接下来，我们初始化项目，新建一下配置文件：

```bash
npx truffle unbox metacoin
mv contracts/*./src/contracts
rm -rf build contracts migrations
sed's/SimpleStorage/MyEthereumDApp/' -i $(grep -rl SimpleStorage.)
```

上述命令执行后，会自动完成以下几件事情：

2. 将Solidity合约源文件移动到`./src/contracts`目录；
3. 删除多余文件；
4. 替换所有的`SimpleStorage`名称为`MyEthereumDApp`。

## 配置webpack

为了方便地进行项目打包，我们可以使用webpack。修改`webpack.config.js`文件如下：

```javascript
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  mode: 'development',
  entry: './src/index.tsx', // 入口文件
  output: {
    filename: '[name].bundle.js',
    path: path.resolve(__dirname, 'dist'), // 输出路径
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js'],
  },
  module: {
    rules: [
      {
        test: /\.(j|t)sx?$/,
        exclude: /node_modules/,
        use: ['babel-loader'],
      },
      {
        test: /\.scss$/,
        use: ['style-loader', 'css-loader','sass-loader']
      },
    ],
  },
  plugins: [new HtmlWebpackPlugin({ template: './public/index.html' })],
  devServer: {
    contentBase: './dist', // 静态文件目录
    open: true, // 启动浏览器
    hot: true, // 启用热更新
  },
};
```

上述配置指定了项目的入口文件，打包输出路径，并且通过babel-loader处理jsx语法，通过style-loader/css-loader/sass-loader加载scss样式表。

## 设置Babel

为了支持最新版本的JavaScript特性，我们需要使用Babel将ES6代码转换为ES5代码。编辑`.babelrc`文件如下：

```json
{
  "presets": ["@babel/preset-env", "@babel/preset-react"],
  "plugins": []
}
```

上述配置指定了ES6->ES5的转换规则和插件。

## 设置ESLint

为了保持JavaScript代码风格一致性，我们可以使用ESLint进行代码风格检查。编辑`.eslintrc.json`文件如下：

```json
{
  "env": {
    "browser": true,
    "commonjs": true,
    "es6": true
  },
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended"
  ],
  "globals": {},
  "parser": "@typescript-eslint/parser",
  "parserOptions": {
    "ecmaFeatures": {
      "jsx": true
    },
    "project": "./tsconfig.json"
  },
  "plugins": ["@typescript-eslint"]
}
```

上述配置指定了项目的运行环境、扩展的推荐规则集、解析器选项以及插件。

## 添加脚本命令

为了让大家更加便捷的运行各种任务，我们可以添加一些脚本命令到`package.json`文件里。编辑后的文件如下：

```json
{
  "name": "my-ethereum-dapp",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "start": "webpack serve",
    "build": "webpack --mode=production",
    "lint": "eslint src/**/*.{ts,tsx}"
  },
  "keywords": [],
  "author": "",
  "license": "ISC",
  "devDependencies": {
    "@babel/core": "^7.12.10",
    "@babel/preset-env": "^7.12.11",
    "@babel/preset-react": "^7.12.5",
    "@types/jest": "^26.0.20",
    "@types/node": "^14.14.19",
    "@types/react": "^16.9.53",
    "@types/react-dom": "^16.9.8",
    "babel-loader": "^8.2.2",
    "clean-webpack-plugin": "^3.0.0",
    "css-loader": "^5.0.2",
    "eslint": "^7.17.0",
    "eslint-plugin-react": "^7.22.0",
    "file-loader": "^6.2.0",
    "fork-ts-checker-webpack-plugin": "^6.1.0",
    "html-webpack-plugin": "^5.0.0",
    "identity-obj-proxy": "^3.0.0",
    "jest": "^26.6.3",
    "mini-css-extract-plugin": "^1.3.6",
    "node-sass": "^5.0.0",
    "optimize-css-assets-webpack-plugin": "^5.0.3",
    "postcss-loader": "^5.0.0",
    "react": "^17.0.1",
    "react-dom": "^17.0.1",
    "rimraf": "^3.0.2",
    "sass-loader": "^10.1.0",
    "style-loader": "^2.0.0",
    "terser-webpack-plugin": "^5.0.3",
    "ts-jest": "^26.4.4",
    "ts-loader": "^8.0.11",
    "typescript": "^4.1.3",
    "url-loader": "^4.1.1",
    "webpack": "^5.11.0",
    "webpack-cli": "^4.2.0",
    "webpack-dev-server": "^3.11.2"
  },
  "dependencies": {}
}
```

上述配置增加了两个新的命令：

1. `start`: 启动开发服务器，实时刷新浏览器；
2. `build`: 打包生产环境的代码。

# 5.编写React组件

现在，我们已经设置好了开发环境，接下来就可以编写React组件了。

## 编写Card组件

我们先编写一个简单的卡片组件，显示当前账户的地址和余额。

```typescript
import React, { useState, useEffect } from'react';
import Web3 from 'web3';

interface Props {
  account?: string;
}

function Card(props: Props) {
  const web3 = new Web3();

  return (
    <div className="card">
      <p>Address:</p>
      <code>{props.account}</code>
      <br />
      <p>Balance:</p>
      <code>{web3.utils.fromWei(String(props.balance), 'ether')} ETH</code>
    </div>
  );
}

export default Card;
```

这里，我们使用Web3.js获取当前账户的地址和余额。我们声明了一个名为`Props`的接口，传入的对象应该包含`account`属性。我们调用`Web3()`函数构造了一个Web3对象，然后在渲染组件的时候将其绑定到组件变量上。

## 编写Button组件

我们再编写一个按钮组件，用来连接钱包和发布合约。

```typescript
import React, { useState, useEffect } from'react';
import Web3 from 'web3';

interface Props {
  onClick: () => void;
}

function Button(props: Props) {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const web3 = new Web3(window.ethereum);

  useEffect(() => {
    async function connect() {
      if (!window.ethereum) {
        alert('Metamask is not installed!');
        return;
      }

      try {
        await window.ethereum.request({ method: 'eth_requestAccounts' });
        setIsConnected(true);
      } catch (error) {
        console.log(error);
      }
    }

    connect();
  }, []);

  return (
    <button disabled={!isConnected || props.onClick === undefined} onClick={() => props.onClick?.()}>
      Connect Wallet and Publish Contract
    </button>
  );
}

export default Button;
```

这里，我们定义了一个名为`Props`的接口，其中包含了一个`onClick`回调函数，用户点击按钮的时候调用。我们调用`Web3()`函数构造了一个Web3对象，这样才能够连接钱包。我们定义了一个useState hook，用于记录当前是否已连接钱包。在useEffect hook里面，我们尝试连接钱包，如果成功的话，则设置isConnected状态为true，否则打印错误日志。最后，我们返回一个按钮元素，disabled属性根据isConnected和onClick属性判断是否禁用按钮。当按钮被点击的时候，如果点击事件回调函数存在，则调用它。

## 编写Input组件

我们再编写一个输入框组件，用来保存合约发布者的名称。

```typescript
import React, { useState } from'react';

interface Props {
  value?: string;
  onChange: (event: React.FormEvent<HTMLInputElement>) => void;
}

function Input(props: Props) {
  return (
    <input type="text" value={props.value} placeholder="Enter a name for your contract" onChange={props.onChange} />
  );
}

export default Input;
```

这里，我们定义了一个名为`Props`的接口，其中包含了一个`value`属性和一个`onChange`回调函数，用于监听文本变化。我们返回一个输入框元素，placeholder属性定义了提示文字，onChange事件回调函数设置为props.onChange。

## 编写ContractInfo组件

我们再编写一个展示合约信息的组件。

```typescript
import React, { useState, useEffect } from'react';
import Web3 from 'web3';

interface Props {
  address: string | null;
}

function ContractInfo(props: Props) {
  const [contractName, setContractName] = useState('');
  const web3 = new Web3();

  useEffect(() => {
    if (!props.address) {
      return;
    }

    const getContractName = async () => {
      try {
        const provider = new web3.providers.WebsocketProvider('ws://localhost:8545');

        const instance = new web3.eth.Contract(
          JSON.parse(
            '{"abi":[{"inputs":[],"stateMutability":"nonpayable","type":"constructor"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"string","name":"","type":"string"}],"name":"SetMessage","type":"event"}]}'
          ),
          props.address,
          {
            from: (await provider.listAccounts())[0],
            gasPrice: web3.utils.toHex((await provider.getGasPrice()) * 1.5),
          }
        );

        const name = await instance.methods.getMessage().call();

        setContractName(name as any);
      } catch (error) {
        console.log(error);
      }
    };

    getContractName();
  }, [props]);

  return (
    <div className="card">
      <h2>Contract Info</h2>
      <p><strong>Address:</strong></p>
      <code>{props.address}</code>
      <br />
      <p><strong>Name:</strong></p>
      <code>{contractName}</code>
    </div>
  );
}

export default ContractInfo;
```

这里，我们定义了一个名为`Props`的接口，其中包含了一个`address`属性，表示要展示的信息对应的合约地址。我们声明了两个useState hook，分别用来记录合约名称和当前账户的余额。在useEffect hook里面，我们异步加载合约ABI数据和构造合约对象。然后我们调用合约的方法`getMessage`，读取存储在合约里面的字符串，并把结果存入contractName状态。最后，我们渲染出一个Card组件，显示合约地址和名称。