
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


GUI(Graphical User Interface)即图形用户界面，是指利用人机交互的方式，将计算机内部的数据、资源及信息显示给终端用户的一种图形化的界面。简单来说，就是让计算机工作起来更方便、直观。从某种角度上来说，GUI技术可分为两类：命令式界面（Command-driven interface）和声明式界面（Declarative interface）。本教程主要面向Java开发者，介绍Java GUI编程的基础知识。
# 2.核心概念与联系
## 概念
- JFrame: 是Java AWT中的顶级容器类，所有应用程序窗口都需要扩展JFrame类来实现。它是一个顶级组件，负责管理整个窗口的内容及属性，包括窗口的标题、边框、大小、位置等；
- JPanel:JPanel是一个轻量级容器类，用于布局管理其他组件。可以理解为JPanel是一个画板，在JPanel上可以布置各种组件；
- JButton:JButton是一个标准按钮组件，继承自AbstractButton类；
- JLabel:JLabel是一个简单标签组件，用于显示文本或图像；
- JTextField:JTextField是一个输入字段组件，用于获取用户的单行文本输入；
- JComboBox:JComboBox是一个下拉列表组件，用于选择单项或多项值；
- JList:JList是一个列表组件，用于显示列表元素；
- JTable:JTable是一个表格组件，用于显示二维数据；
- JScrollPane:JScrollPane是一个滚动条组件，用以滚动其子组件以查看完整内容；
## 关系
下图是Java GUI组件之间的关系：

- JFrame: 可以容纳其他组件作为它的子组件，如JPanel、JButton、JLabel等；
- JPanel: 可以容纳其他组件作为它的子组件，如JButton、JLabel、JPanel等；
- JComponent: 可以被放在JFrame或者JPanel中，如JButton、JLabel、JTable等；
## 常用组件功能简介
### JFrame
创建并显示一个窗口，即使不添加任何组件，也会自动生成一个空白的窗口；
支持对窗口的标题、边框、大小、位置进行控制；
可以通过键盘快捷键关闭窗口；
当用户点击关闭按钮时，可以指定关闭事件的响应函数；
### JPanel
JPanel可以作为容器组件来布置其他组件，提供背景色、布局方式等属性；
JPanel可以包含多个子组件，包括JPanel、JButton、JLabel等；
JPanel默认布局管理器为BorderLayout布局，可以上下左右边界布局，还可以设置边界内的间距等；
JPanel一般不直接用来显示文字或图片等信息，而是作为容器来组织其他组件，然后再使用相应的JLabel、JTextArea、JEditorPane等组件显示；
### JButton
JButton是一个标准按钮组件，可以通过鼠标、键盘、方法调用等方式触发按钮的行为；
JButton可以显示文本、图标、或自定义样式；
JButton支持自定义点击事件处理，可以实现按钮单击、双击、悬浮等效果；
### JLabel
JLabel是一个简单的文字标签组件，可以在界面上显示一些简单的文字信息；
JLabel可以使用带格式文本的HTML标签显示富文本信息；
JLabel支持多行显示；
JLabel可以使用不同的字体颜色、背景色、字体风格等属性进行定制；
### JTextField
JTextField是一个输入字段组件，用户可以用鼠标或者键盘输入文本信息；
JTextField提供简单的校验功能，如最大长度限制、正则表达式校验等；
JTextField可以显示提示文本，用户可以快速了解该输入框期望输入的内容；
JTextField可以对密码进行隐藏，防止用户查看明文密码；
JTextField可以指定边框样式、边框颜色、宽度等；
### JComboBox
JComboBox是一个下拉列表组件，通过弹出菜单展示可选项列表，用户可以从中选择一个或多个值；
JComboBox提供了基于列表的选择功能，因此可以用来显示字符串、枚举类型的值；
JComboBox可以指定单选或多选模式；
JComboBox可以定制列表的高度、选择项前面的图标、选中项前面的图标等；
### JList
JList是一个列表组件，用户可以用鼠标或者箭头键浏览可选择的列表项目；
JList可以用来呈现对象集合、列表项、数组等数据；
JList可以允许单选或者多选模式；
JList可以显示复杂的列表项，比如图像、复选框等；
JList可以显示单元格编辑状态，允许用户对单元格数据进行修改；
### JTable
JTable是一个表格组件，用来呈现二维数据的表格视图；
JTable可以自定义列宽、行高、边框、背景色、字体颜色等；
JTable可以用作具有复杂结构的表格数据，比如树状表格；
JTable可以指定多种排序方式，并允许指定自定义排序函数；
JTable可以显示交替行色、虚线等特效；
### JScrollPane
JScrollPane是一个滚动条组件，可以整合在容器中，对组件进行滚动显示；
JScrollPane可以实现固定宽度或者高度的滚动条，也可以同时出现水平、垂直滚动条；
JScrollPane可以指定滚动区域的尺寸，这样就可以实现区域滚动显示；
JScrollPane还能跟踪滚动事件，如滚动条拖动、用户按下鼠标滚轮等；