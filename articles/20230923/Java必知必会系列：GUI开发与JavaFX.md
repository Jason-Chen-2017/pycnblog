
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在软件开发过程中，经常需要用图形用户界面（Graphical User Interface，简称GUI）来提升用户体验、降低使用成本等方面的效果。而JavaFX是Java平台的最新一款用于开发GUI界面的工具。对于想要学习JavaFX但是却苦于没有合适入门教程的初学者来说，这是一份值得推荐的入门材料。它涵盖了JavaFX的基础语法、组件、布局、事件处理、动画、多线程编程、部署等方面知识。通过阅读本文，可以让读者对JavaFX有个整体的认识以及对它的相关技术有所了解。

作者：孔令强

发布时间：2020-9-17

## 文章目录

1. 背景介绍
2. JavaFX概述
	2.1 概念及特点
	2.2 JDK版本要求
	2.3 Hello World示例
	2.4 SceneBuilder工具介绍
3. GUI编程概览
	3.1 GUI程序的组成
	3.2 Window类
	3.3 Stage类
	3.4 控件
	3.5 布局管理器
	3.6 主题
4. 画布和节点
	4.1 画布
	4.2 节点类型
	4.3 Node类
	4.4 各种节点及其属性介绍
		4.4.1 Button节点
		4.4.2 Label节点
		4.4.3 TextField节点
		4.4.4 PasswordField节点
		4.4.5 ChoiceBox节点
		4.4.6 CheckBox节点
		4.4.7 RadioButton节点
		4.4.8 ScrollPane节点
		4.4.9 ListView节点
		4.4.10 ImageView节点
		4.4.11 Separator节点
		4.4.12 ProgressBar节点
		4.4.13 Slider节点
		4.4.14 ToolTip节点
		4.4.15 MenuBar节点
		4.4.16 MenuItem节点
		4.4.17 ContextMenu节点
		4.4.18 Alert节点
		4.4.19 TableView节点
		4.4.20 TreeView节点
		4.4.21 Dialog节点
		4.4.22 TabPane节点
		4.4.23 DatePicker节点
		4.4.24 ColorPicker节点
		4.4.25 Region节点
		4.4.26 Pane节点
		4.4.27 Canvas节点
		4.4.28 Path节点
		4.4.29 Group节点
		4.4.30 Media播放器节点
5. 动画
	5.1 关于动画的概念
	5.2 KeyFrame类
	5.3 Timeline类
6. 绑定及事件
	6.1 什么是绑定？
	6.2 Property类
	6.3 事件监听机制介绍
		6.3.1 鼠标点击事件
		6.3.2 键盘事件
		6.3.3 拖放事件
		6.3.4 窗口变化事件
		6.3.5 对话框事件
		6.3.6 触控板事件
		6.3.7 场景中其他节点的事件
7. 多线程编程
	7.1 为什么要用多线程？
	7.2 创建新线程
	7.3 实现Runnable接口
	7.4 使用ExecutorService接口
	7.5 在后台运行的任务
8. 资源文件
	8.1 将资源文件打包到JAR或WAR中
	8.2 外部资源文件介绍
	8.3 FXML文档结构
	8.4 从FXML加载GUI节点
9. 部署应用
	9.1 如何生成可执行jar文件
	9.2 将JavaFX部署到Web容器中
	9.3 使用Maven插件进行JavaFX项目构建
10. 后记
# 2.JavaFX概述<|im_sep|>