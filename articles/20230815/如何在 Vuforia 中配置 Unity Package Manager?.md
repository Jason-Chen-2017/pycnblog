
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Vuforia开发包被分为了三个不同版本：Vuforia Engine SDK, Vuforia Augmented Reality SDK, Vuforia Extended Tracking SDK。为了使开发者能够更方便的管理这些不同的开发包及其依赖关系，Unity官方引入了Unity Package Manager(UPM)机制。而Vuforia中不支持UPM机制。因此本文将介绍如何将Vuforia开发包导入到UPM系统中，并对比两种配置方式的优缺点。

# 2.基本概念
- UPM: Unity Package Manager, Unity官方提供的新版本的管理外部资源的机制。它可以解决开发包之间的依赖关系、版本管理等。
- Unity工程：在Unity编辑器中打开的项目就是一个Unity工程。
- Package：一个完整的Unity工程或其他资源都是一个Package。包括AssetBundle, Unity工程中的各种文件（场景、模型、脚本等）。

# 3.方案选择
由于Vuforia不支持UPM机制，所以我们需要选取一种方案将Vuforia开发包导入到UPM系统中。我给出两种方案：
1. 将Vuforia开发包打包成一个独立的UPM package，然后再将这个package安装到Unity工程中。这种方式可以最大程度保持Vuforia开发包与其他package的一致性。
2. 将Vuforia各个模块分别作为单独的package发布，然后通过Git子模块的方式集成到Unity工程中。这种方式可以最大程度保留Vuforia自身的目录结构。

# 3.1 方法1：打包Vuforia开发包为独立的UPM package
首先下载Vuforia的最新版本的开发包，然后创建一个新的空白的UPM package。接着将Vuforia的所有必要的文件和文件夹从原始的Vuforia压缩包拷贝到该package对应的“Packages”文件夹下，并修改“manifest.json”文件，添加相应的“dependencies”。示例如下：

	{
	  "name": "com.yourcompany.vuforia",
	  "version": "1.0.0",
	  "displayName": "Vuforia Engine SDK for Unity",
	  "description": "The official Vuforia Engine SDK for Unity.",
	  "unity": "2019.4",
	  "keywords": [
		"Vuforia",
		"AR",
		"VR",
		"Virtual Reality",
		"Augmented Reality",
		"MR",
		"Mixed Reality",
		"AR Foundation"
	  ],
	  "author": {
		"name": "Your Company Name Here",
		"email": "<EMAIL>",
		"url": "https://www.yourcompanywebsitehere.com/"
	  },
	  "type": "tool",
	  "dependencies": {}
	}
	
最后将这个package导入到Unity工程中即可。此方法有如下优点：
- 整体的体积较小，只包含Vuforia所需的Assets和Plugins文件，不会影响其他package的体积。
- 可以与其他package共享某些Vuforia资源（如Models等），但不能使用任何非Vuforia的内容。
- 恢复Vuforia开发包之前的状态非常简单，无需特殊处理。

缺点：
- 与其他package共用Assets可能会造成冲突，需要注意避免。
- 需要额外手动配置并维护每一次更新。
- 如果Vuforia进行了大的更新或重构，会使得导入过程变得复杂。

# 3.2 方法2：将Vuforia各个模块分别作为单独的package发布，然后通过Git子模块的方式集成到Unity工程中。
这种方式相对于方法1来说，多了一个版本控制的问题。如果Vuforia进行了更新，则所有的package也需要跟随一起更新，否则Unity无法识别Vuforia的版本号。为了实现这一点，我们可以在Vuforia GitHub仓库中创建多个独立的submodule，每个submodule对应Vuforia的一个模块。然后在Unity项目中创建一个空白的UPM package，并在manifest.json中添加Git Submodules。配置完成后，就可以像使用其他package一样管理Vuforia了。

这种方式的优点：
- 每个模块都可以单独进行版本控制，可以追踪到Vuforia各个模块的变化。
- 在导入过程中不需要考虑冲突问题，因为所有package都是相互独立的。
- 更新Vuforia时，只需要更新对应的 submodule 即可。

缺点：
- 大量的package会占用磁盘空间，虽然可以忽略不计。
- 配置起来比较麻烦，需要多次提交才能完成。
- 在更新Vuforia时需要手动检查每个submodule是否有更新，有的话就重新checkout和merge。

# 4.具体实施
## 4.1 安装准备
在开始前，请先确保已经安装了以下软件：
1. Git
2. Unity Editor Version 2019.4 LTS or later

## 4.2 方法1：打包Vuforia开发包为独立的UPM package
### 4.2.1 创建一个空白的UPM package
创建方法很简单，只要创建一个名为`com.yourcompany.vuforia`的文件夹，然后在该文件夹内创建一个`manifest.json`文件，文件内容如下：

	{
	  "name": "com.yourcompany.vuforia",
	  "version": "1.0.0",
	  "displayName": "Vuforia Engine SDK for Unity",
	  "description": "The official Vuforia Engine SDK for Unity.",
	  "unity": "2019.4",
	  "keywords": [
		"Vuforia",
		"AR",
		"VR",
		"Virtual Reality",
		"Augmented Reality",
		"MR",
		"Mixed Reality",
		"AR Foundation"
	  ],
	  "author": {
		"name": "Your Company Name Here",
		"email": "youremail@domain.<EMAIL>",
		"url": "https://www.yourcompanywebsitehere.com/"
	  },
	  "type": "tool",
	  "dependencies": {}
	}

### 4.2.2 添加Vuforia开发包的文件到package中
下载Vuforia的最新版开发包，将“Assets”文件夹中的所有内容拷贝到新建的package的“Packages”文件夹中。

### 4.2.3 修改manifest.json文件，添加dependencies项
将以下依赖关系添加到`manifest.json`文件中。这样做的目的是告诉UPM系统，当前package需要依赖于其他哪几个package。

	{
	  "name": "com.yourcompany.vuforia",
	  "version": "1.0.0",
	  "displayName": "Vuforia Engine SDK for Unity",
	  "description": "The official Vuforia Engine SDK for Unity.",
	  "unity": "2019.4",
	  "keywords": [
		"Vuforia",
		"AR",
		"VR",
		"Virtual Reality",
		"Augmented Reality",
		"MR",
		"Mixed Reality",
		"AR Foundation"
	  ],
	  "author": {
		"name": "Your Company Name Here",
		"email": "youremail@domain.com",
		"url": "https://www.yourcompanywebsitehere.com/"
	  },
	  "type": "tool",
	  "dependencies": {
		"com.unity.render-pipelines.universal": "7.3.1",
		"com.unity.inputsystem": "1.0.0",
		"com.unity.xr.legacyinputhelpers": "2.1.4"
	  }
	}

其中，"com.unity.render-pipelines.universal"对应的是Vuforia中渲染管线相关的部分，"com.unity.inputsystem"对应的是用来与硬件交互的插件，"com.unity.xr.legacyinputhelpers"也是用来与硬件交互的插件。根据需要，可以添加更多的依赖关系。

### 4.2.4 将package导入到Unity工程中
在Unity编辑器中打开你的工程，在菜单栏依次点击`Window->Package Manager`。然后在左侧面板中点击"+"按钮，在弹出的窗口中选择刚才创建的`com.yourcompany.vuforia`，然后点击右上角的`Install`按钮。接下来等待一下，直到package安装成功，然后点击关闭窗口。

## 4.3 方法2：将Vuforia各个模块分别作为单独的package发布，然后通过Git子模块的方式集成到Unity工程中。
方法2适用于有多个developer同时参与开发Vuforia的情况。它的具体流程如下：

### 4.3.1 创建多个独立的package
最简单的方法就是分别克隆Vuforia仓库，然后在每个cloned repository目录下执行`git submodule add https://github.com/Vuforia/<submodule>`命令。这里`<submodule>`可以是：

1. VuforiaEngine
2. VuforiaSamples
3. VuforiaSamples-Srp
4....

这样就会在当前repository下生成对应的submodules，形如`.gitmodules`文件的形式。例如：

	[submodule "Vuforia/VuforiaEngine"]
		path = Vuforia/VuforiaEngine
		url = https://github.com/Vuforia/VuforiaEngine.git
	
	[submodule "Vuforia/VuforiaSamples-SRP"]
		path = Vuforia/VuforiaSamples-SRP
		url = https://github.com/Vuforia/VuforiaSamples-SRP.git
		
### 4.3.2 在你的Unity Project中创建一个空白的UPM package
同样地，创建一个名为`com.yourcompany.vuforia`的文件夹，然后在该文件夹内创建一个`manifest.json`文件，文件内容如下：

	{
	  "name": "com.yourcompany.vuforia",
	  "version": "1.0.0",
	  "displayName": "Vuforia Engine SDK for Unity",
	  "description": "The official Vuforia Engine SDK for Unity.",
	  "unity": "2019.4",
	  "keywords": [
		"Vuforia",
		"AR",
		"VR",
		"Virtual Reality",
		"Augmented Reality",
		"MR",
		"Mixed Reality",
		"AR Foundation"
	  ],
	  "author": {
		"name": "Your Company Name Here",
		"email": "youremail@domain.com",
		"url": "https://www.yourcompanywebsitehere.com/"
	  },
	  "type": "tool",
	  "dependencies": {},
	  "publishConfig": {
		"registry": "https://nuget.pkg.github.com/YourGitHubUsernameHere/index.json"
	  }
	}

请记住将`"publishConfig"`下的`"registry"`值替换为你的GitHub用户名。

### 4.3.3 通过Git Submodule管理Vuforia package
将各个package都添加到manifest.json的"dependencies"节点中，其中路径使用相对路径。例如：

	{
	  "name": "com.yourcompany.vuforia",
	  "version": "1.0.0",
	  "displayName": "Vuforia Engine SDK for Unity",
	  "description": "The official Vuforia Engine SDK for Unity.",
	  "unity": "2019.4",
	  "keywords": [
		"Vuforia",
		"AR",
		"VR",
		"Virtual Reality",
		"Augmented Reality",
		"MR",
		"Mixed Reality",
		"AR Foundation"
	  ],
	  "author": {
		"name": "Your Company Name Here",
		"email": "youremail@domain.com",
		"url": "https://www.yourcompanywebsitehere.com/"
	  },
	  "type": "tool",
	  "dependencies": {
		"com.yourcompany.vuforia.engine": "file:../Vuforia/VuforiaEngine",
		"com.yourcompany.vuforia.samples.srp": "file:../Vuforia/VuforiaSamples-SRP",
		...
	  },
	  "publishConfig": {
		"registry": "https://nuget.pkg.github.com/YourGitHubUsernameHere/index.json"
	  }
	}

其中"com.yourcompany.vuforia.engine"和"com.yourcompany.vuforia.samples.srp"分别对应于你克隆的两个Vuforia仓库中的Engine和Sample-SRP子模块。另外还可以通过相对路径指定某个版本，例如："com.yourcompany.vuforia.engine": "^1.200.0". 表示必须大于等于1.200.0且小于2.0.0。

### 4.3.4 初始化Git Submodule
初始化方法是在Unity Editor中打开你自己的Project，然后在Package Manager窗口中勾选你的Vuoforia package，在右边的详情界面找到"Add Packages from git urls..."按钮，在文本框中输入：

    - https://github.com/Vuforia/VuforiaEngine
    - https://github.com/Vuforia/VuforiaSamples-SRP
    -...
    
点击确定，就可以把所有的package克隆到本地。

### 4.3.5 使用UPM管理你的package
在完成初始化后，你可以像管理其他package那样管理Vuoforia packages。你只需要在上述步骤中配置好自己的nuget registry。当你更新Vuforia packages时，你只需要在Unity Editor中选择Vuoforia package并点击Update按钮即可。