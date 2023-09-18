
作者：禅与计算机程序设计艺术                    

# 1.简介
  

TypeScript (TS) 是一种基于JavaScript的语言，它是一种开源、自由的编程语言。它的优点在于可以增强可读性、可维护性、可扩展性、可移植性等诸多方面。本文将介绍TypeScript的安装与配置方法。
# 2.基本概念术语说明
## Typescript
### 安装 TypeScript
TypeScript可以通过npm包管理器安装，直接在命令行中运行以下命令即可完成安装：
```
npm install -g typescript
```
如果你是开发人员，可以使用--save-dev参数将TypeScript作为devDependencies保存到你的package.json文件中：
```
npm install --save-dev typescript
```
如果已经成功安装了TypeScript，可以在终端执行tsc -v命令查看版本信息，如下所示：
```
tsc -v
Version 2.7.2
```

### 配置 TypeScript

比如，假设有一个项目结构如下：
```
myproject
  |- src
      |- app.ts
      |- lib
          |- math.ts
```
为了编译src下的所有TypeScript代码，并输出到dist目录下，我们可以设置如下的tsconfig.json文件：
```
{
  "compilerOptions": {
    "outDir": "./dist", // 编译输出目录
    "sourceMap": true // 生成相应的.map 文件
  },
  "include": ["./src/**/*"] // 编译 src 下的所有 TypeScript 文件
}
```
这样，运行`tsc`命令就会把src目录下的所有TypeScript文件编译成JavaScript文件，并输出到dist目录下，同时生成对应的.map文件供调试用途。


## Node.js with TypeScript
Node.js是JavaScript运行环境中的一个分支，其官方提供了TypeScript支持。Node.js版本为8或更新的版本支持TypeScript，因此建议使用最新版的Node.js。

首先，通过npm全局安装TypeScript：
```
npm install -g typescript
```
然后，创建一个TypeScript文件，例如hello.ts：
```
console.log("Hello, World!");
```
然后，编译这个TypeScript文件，转换成JavaScript文件，并输出到指定目录：
```
tsc hello.ts --outDir dist
```
这里的--outDir参数指定输出目录为dist。然后就可以运行node dist/hello.js命令运行JavaScript文件了，输出结果为："Hello, World!"。

除了简单的文件编译外，TypeScript还提供丰富的类型注解功能。你可以在变量、函数、类等声明时添加类型注解，使得编译器能够捕获一些错误。例如，可以给上面的hello.ts添加类型注解：
```
function greet(name: string): void {
  console.log(`Hello, ${name}!`);
}
greet('World');
```
然后编译运行这个TypeScript文件，会得到一个类型错误提示：
```
error TS2345: Argument of type '"World"' is not assignable to parameter of type'string'.
```
因为函数greet的参数name没有被正确地指定为字符串类型。
