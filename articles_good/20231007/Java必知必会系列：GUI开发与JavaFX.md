
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


在近几年，随着PC的普及，移动互联网应用的兴起，传统的客户端软件开发方式已经远远无法满足人们对快速响应、简洁易用的需求，基于Web技术的分布式计算模式也被广泛应用于各种应用程序中。对于高级技术人员而言，掌握Java语言、熟悉面向对象编程、设计模式、集合框架、多线程等有利于提升技术能力和个人竞争力。因此，作为Java生态圈中重要的一环，JavaFX（JavaFX for User Interface）是一款功能丰富、性能卓越的Java客户端GUI库。本系列教程将带领读者了解JavaFX，包括其基本用法、组件机制、事件处理、布局管理、动画效果等，并结合实际案例展示如何利用JavaFX开发出具有良好用户体验的桌面客户端应用。通过阅读本系列教程，读者可以学习到：

1. JavaFX基础知识：包括JavaFX入门、JavaFX控件、JavaFX组件、JavaFX事件处理、JavaFX布局管理、JavaFX动画、JavaFX主题和样式、JavaFX工具包和第三方库；
2. JavaFX特点及应用场景：包括JavaFX的优势、适用场景、历史渊源、现状和未来展望；
3. JavaFX实战案例：包括绘制基本图形、复杂表格视图、文件读取器、地图导航器、股票行情分析器、日历组件、登录界面等；
4. 深入理解JavaFX的底层实现原理：包括JavaFX的UI结构、渲染、布局、事件处理、线程调度、资源加载、图形音视频渲染等；
5. 掌握JavaFX性能优化技巧：包括使用后台线程加载数据、避免重绘与动画抖动、避免频繁垃圾回收、图片压缩、内存泄漏检测等。
欢迎关注微信公众号“Java千牛”，后续将推送更多JavaFX系列教程。另外，还可以通过添加本文的链接或转发朋友圈的方式推荐给需要的人。
# 2.核心概念与联系
首先，我们需要了解一下JavaFX中的一些基本概念与联系。
## 2.1 JavaFX UI
JavaFX UI是由一组可视化组件、容器以及其他控件构成的用户界面。JavaFX UI的主要组件包括：
- 窗体（Window）：用于显示应用程序的主要窗口。
- 面板（Pane）：是一个容器，用来放置各种组件。Pane可以看作是一个画布，可以容纳其他组件。
- 控件（Control）：如按钮、标签、文本框、列表、菜单、进度条等。
- 容器（Container）：用来容纳其它容器或者控件。
- 滚动条（ScrollBar）：用来滚动视窗内的元素。
- 列表视图（ListView）：一个只读的表格视图，用来显示元素的集合。
- 树视图（TreeView）：一个可编辑的树型视图，用来显示元素的集合以及它们之间的关系。
- 表格视图（TableView）：一个只读的表格视图，用来显示数据的多维表格。
除了这些主要的组件之外，还有许多组件都继承自Control类，如ToggleButton、Separator、ProgressBar、Slider、ChoiceBox等。
## 2.2 JavaFX App
JavaFX App是JavaFX的一个容器类，用于封装多个JavaFX组件、资源和行为，并提供运行时环境。App本身不直接显示任何内容，但它可以被嵌入到另一个容器组件中，例如：BorderPane、ScrollPane、Canvas、Stage、DialogPane等。App同样提供了生命周期方法，可以控制组件何时开始、停止、暂停或恢复运行。
## 2.3 FXML
FXML（FXML：File with Extended Markup Language）是一种声明性语言，可以定义JavaFX组件及其属性，并通过声明性代码和XML标记将其绑定到JavaFX控制器中。FXML支持分离视图逻辑（View）、模型逻辑（Model）、控制器逻辑（Controller），并允许视图和控制器使用继承和组合关系组织代码。
## 2.4 CSS
CSS（Cascading Style Sheets）是一种样式表语言，可以在浏览器端设置HTML页面的样式。CSS可以实现页面元素的动态效果和整体视觉效果。
## 2.5 SceneBuilder
SceneBuilder是一个图形界面的用户界面构建工具，可以用来创建JavaFX UI。SceneBuilder通过拖放组件，调整属性值来创建JavaFX UI，并能够导出FXML文件。
## 2.6 控制器（Controller）
控制器是一个Java类，负责处理用户输入、业务逻辑、更新组件状态，并且可以调用业务服务接口。它通常与FXML文件一起使用，可以监听FXML文件的变化并动态更新JavaFX UI。
## 2.7 业务逻辑（Business Logic）
业务逻辑指的是应用中用来处理应用程序核心任务的代码。它可以包含网络连接、数据库访问、业务规则等功能。它通常与模型层相连，并通过接口与模型层通信。
## 2.8 模型层（Model Layer）
模型层是用来存储和管理数据的层次结构，其中包含的数据应该是业务相关的信息。在JavaFX中，通常把该层称为实体（Entity）。通常情况下，模型层和业务逻辑应该是松耦合的，尽量减少模型层与业务逻辑的依赖关系。
## 2.9 JavaFX Event
JavaFX Event是发生在JavaFX UI上的一个事件。常见的JavaFX Event包括点击、拖动、按键、鼠标滚轮等。当一个事件发生时，相应的事件处理方法就会被调用。JavaFX Event通常与FXML文件配合使用，FXML文件可以指定JavaFX Event处理方法，并可以监听FXML文件的变化，从而动态更新JavaFX Event处理方法。
# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
我们接下来就可以详细讲解JavaFX的各个组件和特性了。首先，让我们看一下JavaFX中的布局管理。
## 3.1 布局管理
布局管理的目的是将JavaFX组件放置在窗体上。JavaFX提供了两种不同的布局管理策略：
### 3.1.1 AnchorPane
AnchorPane是JavaFX中最简单、最常用的布局管理策略，它的定位方式类似CSS中的position:absolute。通过设置anchorX、anchorY、leftAnchor、rightAnchor、topAnchor、bottomAnchor属性，可以使JavaFX组件在父组件内部进行定位。
### 3.1.2 GridPane
GridPane是一种二维的布局管理策略，它将子组件以行列的形式组织起来。通过设置row、columnIndex、rowIndex属性，可以使JavaFX组件按照指定的行列顺序排列。
除了以上两种布局管理策略之外，还有其他几种布局管理策略可以使用：VBox、HBox、StackPane、TilePane、FlowPane等。
布局管理涉及到的主要属性如下：
- alignment：用于指定组件在容器内部的位置。
- margin：用于设置组件之间的间距。
- padding：用于设置组件之间的内边距。
- prefWidth、prefHeight：用于设置组件的初始大小。
- maxWidth、maxHeight：用于设置组件的最大尺寸。
- minWidth、minHeight：用于设置组件的最小尺寸。
- hgrow、vgrow：用于设置组件的水平方向和垂直方向的伸缩性。
- columnConstraints、rowConstraints：用于设置GridPane中的行和列的约束条件。
除了布局管理之外，还需要了解JavaFX的组件机制。
## 3.2 JavaFX组件机制
JavaFX组件是指JavaFX UI中的基础部件，每个组件都有一个共同的父类javafx.scene.Node。Node提供一些共同的方法，如getBoundsInParent()、translateX()、getChildrenUnmodifiable()等，这些方法都用于处理组件的几何信息、位置变换、属性设置、子节点获取等。除了这些共同的方法之外，还有一些特定组件的额外方法。例如，Button类提供了fire()方法，可以触发按钮的事件处理。Label类提供了getText()和setText()方法，可以设置和获取文字内容。TextField类提供了clear()方法，可以清除输入框的内容。这些额外的方法都是JavaFX独有的，并没有出现在所有Node类的通用接口中。此外，每个组件都可以有自己的样式，可以定制它的外观。
除了Node类之外，还存在很多派生类，如AnchorPane、SplitPane、ToolBar、MenuBar等。这些类提供了一些独特的特性，如布局管理、动画效果、弹出式菜单等。
除了JavaFX组件机制之外，还需要了解JavaFX的事件处理机制。
## 3.3 JavaFX事件处理机制
JavaFX事件处理机制是指JavaFX UI中的用户交互行为，它是指当用户操作某个UI组件时，JavaFX组件对此做出的反应。JavaFX提供了两种不同类型的事件处理机制：
- 全局事件处理机制：这种机制可以处理整个应用程序的所有事件，它可以用addEventHandler()方法添加，也可以用removeEventHandler()方法移除。这种机制适用于那些全局性事件，比如鼠标点击、键盘按键等。
- 组件级别事件处理机制：这种机制可以处理单个组件的所有事件，它只能通过addComponentListener()方法添加，不能移除。这种机制适用于那些局部性事件，比如节点被添加到Pane或者节点的属性改变时。
事件处理机制涉及到的主要方法如下：
- setOnAction(): 设置组件触发时的事件处理方法。
- addEventFilter(): 添加一个过滤器，它可以截获该事件的传递，但不会影响该事件本身的传递。
- fireEvent(): 在树中发送事件，只有位于当前节点或子孙节点上的节点才会接收到这个事件。
除了事件处理机制之外，还需要了解JavaFX的动画机制。
## 3.4 JavaFX动画机制
JavaFX动画机制是指JavaFX UI中的视觉效果变化，它可以实现组件的动态效果。JavaFX提供了三种不同类型的动画效果：
- 播放列表（Timeline）：播放列表是一种多步动画效果，它可以同时播放多个动画效果。
- 路径动画（PathAnimation）：路径动画是一种使用路径来驱动组件位置的动画效果。
- 键帧动画（KeyFrameAnimation）：键帧动画是一种通过关键帧来驱动组件位置的动画效果。
除了动画机制之外，还需要了解JavaFX的工具包。
## 3.5 JavaFX工具包
JavaFX工具包是指围绕JavaFX的一些常用工具，它可以帮助开发者实现一些常见的功能，如日志记录、图像处理、数据库访问、单元测试、国际化、打印机支持等。JavaFX工具包目前有Maven Central Repository和JCenter。
# 4.具体代码实例和详细解释说明
接下来，我们将具体编写几个JavaFX案例来展示JavaFX的各种特性。
## 4.1 基本图形绘制
案例描述：利用JavaFX开发一个简单的绘制圆形的小程序。
要求：能完整编写代码并展示结果，不需要引用外部Jar包。
实现过程：1. 创建JavaFX应用程序。
           2. 初始化Scene对象。
           3. 创建画布Pane。
           4. 新建一个Circle对象，设置它的半径和颜色。
           5. 将Circle对象添加到Pane中。
           6. 设置应用的宽高。
           7. 设置Scene的根节点为Pane。
           8. 显示应用。
代码如下所示：

```
import javafx.application.Application;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.stage.Stage;

public class CircleDemo extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {

        //Step 1
        BorderPane root = new BorderPane();

        //Step 2
        Scene scene = new Scene(root);

        //Step 3
        Pane pane = new Pane();

        //Step 4
        Circle circle = new Circle(100, 100, 50);
        circle.setFill(Color.RED);

        //Step 5
        pane.getChildren().add(circle);

        //Step 6
        pane.setPrefSize(300, 300);

        //Step 7
        root.setCenter(pane);

        //Step 8
        primaryStage.setTitle("Circle Demo");
        primaryStage.setScene(scene);
        primaryStage.show();
    }


    public static void main(String[] args) {
        launch(args);
    }
}
```

输出结果：


## 4.2 文件读取器
案例描述：开发一个基于JavaFX的文件读取器。
要求：文件读取器具备打开、关闭、保存文件等功能。
实现过程：1. 创建JavaFX应用程序。
           2. 初始化Scene对象。
           3. 创建画布Pane。
           4. 新建一个TextArea对象，用于显示文件内容。
           5. 使用FileChooser类选择文件。
           6. 为打开按钮绑定onAction事件，用于打开文件并显示内容。
           7. 为关闭按钮绑定onAction事件，用于关闭应用程序。
           8. 为保存按钮绑定onAction事件，用于保存文件内容。
           9. 设置应用的宽高。
           10. 设置Scene的根节点为Pane。
           11. 显示应用。
代码如下所示：

```
import java.io.*;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.TextArea;
import javafx.scene.layout.BorderPane;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

public class FileReaderDemo extends Application {

    private String fileName;
    
    @Override
    public void start(Stage primaryStage) throws Exception {
        
        //Step 1
        BorderPane root = new BorderPane();
        
        //Step 2
        Scene scene = new Scene(root);
        
        //Step 3
        TextArea textArea = new TextArea();
        textArea.setEditable(false);
        
        //Step 4
        Button openBtn = new Button("Open...");
        Button closeBtn = new Button("Close");
        Button saveBtn = new Button("Save");
        
        //Step 5
        final FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Text files", "*.txt"),
                new FileChooser.ExtensionFilter("All Files", "*.*")
        );
        
        //Step 6
        EventHandler<ActionEvent> openHandler = new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                fileName = fileChooser.showOpenDialog(primaryStage).getAbsolutePath();
                
                if (fileName!= null &&!"".equals(fileName)) {
                    try {
                        BufferedReader br = new BufferedReader(new FileReader(fileName));
                        StringBuilder sb = new StringBuilder();
                        
                        String line;
                        while ((line = br.readLine())!= null) {
                            sb.append(line + "\n");
                        }
                        
                        br.close();
                        
                        textArea.setText(sb.toString());
                    } catch (Exception e) {
                        System.out.println(e);
                    }
                } else {
                    System.out.println("No file selected!");
                }
            }
        };
        
        openBtn.setOnAction(openHandler);
        
        //Step 7
        EventHandler<ActionEvent> closeHandler = new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                Platform.exit();
            }
        };
        
        closeBtn.setOnAction(closeHandler);
        
        //Step 8
        EventHandler<ActionEvent> saveHandler = new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                BufferedWriter bw = null;
                
                try {
                    if (!textArea.getText().trim().isEmpty()) {
                        if (fileName == null || "".equals(fileName)) {
                            fileName = fileChooser.showSaveDialog(primaryStage).getAbsolutePath();
                            
                        } 
                        bw = new BufferedWriter(new FileWriter(fileName));
                        bw.write(textArea.getText());
                    }
                    
                } catch (IOException ex) {
                    System.err.println(ex);
                } finally {
                    try {
                        if (bw!= null) {
                            bw.close();
                        }
                    } catch (IOException ex) {
                        System.err.println(ex);
                    }
                }
            }
        };
        
        saveBtn.setOnAction(saveHandler);
        
        
        //Step 9
        HBox btnBox = new HBox(5);
        btnBox.setAlignment(Pos.CENTER);
        btnBox.getChildren().addAll(openBtn, closeBtn, saveBtn);
        
        //Step 10
        VBox layout = new VBox(10);
        layout.getChildren().addAll(textArea, btnBox);
        layout.setAlignment(Pos.TOP_CENTER);
        
        //Step 11
        root.setCenter(layout);
        primaryStage.setWidth(600);
        primaryStage.setHeight(400);
        primaryStage.setScene(scene);
        primaryStage.setTitle("File Reader Demo");
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
    
}
```

输出结果：


## 4.3 股票行情分析器
案例描述：开发一个基于JavaFX的股票行情分析器。
要求：能够显示某支股票的日K线图，包括开盘价、收盘价、最高价、最低价、成交量、成交额。
实现过程：1. 创建JavaFX应用程序。
           2. 初始化Scene对象。
           3. 创建画布Pane。
           4. 新建一个TableView对象，用于显示股票信息。
           5. 新建一个XYChart对象，用于显示K线图。
           6. 从文件中读取股票信息并填充TableView。
           7. 根据股票信息生成K线图。
           8. 设置应用的宽高。
           9. 设置Scene的根节点为Pane。
           10. 显示应用。
代码如下所示：

```
import java.util.ArrayList;
import java.util.List;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.io.*;
import javafx.application.Application;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

public class StockAnalyzerDemo extends Application {

    private ObservableList<StockInfo> stockInfos = FXCollections.observableArrayList();
    
    private XYChart.Series<Integer, Double> series = new XYChart.Series<>();
    
    @Override
    public void start(Stage primaryStage) throws Exception {
        
        //Step 1
        BorderPane root = new BorderPane();
        
        //Step 2
        Scene scene = new Scene(root);
        
        //Step 3
        TableView table = new TableView<>();
        TableColumn<StockInfo, Integer> idCol = new TableColumn<>("ID");
        TableColumn<StockInfo, LocalDate> dateCol = new TableColumn<>("Date");
        TableColumn<StockInfo, LocalTime> timeCol = new TableColumn<>("Time");
        TableColumn<StockInfo, Double> openCol = new TableColumn<>("Open");
        TableColumn<StockInfo, Double> highCol = new TableColumn<>("High");
        TableColumn<StockInfo, Double> lowCol = new TableColumn<>("Low");
        TableColumn<StockInfo, Long> volumeCol = new TableColumn<>("Volume");
        TableColumn<StockInfo, Double> amountCol = new TableColumn<>("Amount");
        
        idCol.setCellValueFactory(cellData -> cellData.getValue().getIdProperty());
        dateCol.setCellValueFactory(cellData -> cellData.getValue().getDateProperty());
        timeCol.setCellValueFactory(cellData -> cellData.getValue().getTimeProperty());
        openCol.setCellValueFactory(cellData -> cellData.getValue().getOpenProperty());
        highCol.setCellValueFactory(cellData -> cellData.getValue().getHighProperty());
        lowCol.setCellValueFactory(cellData -> cellData.getValue().getLowProperty());
        volumeCol.setCellValueFactory(cellData -> cellData.getValue().getVolumeProperty());
        amountCol.setCellValueFactory(cellData -> cellData.getValue().getAmountProperty());
        
        table.setItems(stockInfos);
        table.getColumns().addAll(idCol, dateCol, timeCol, openCol, highCol, lowCol, volumeCol, amountCol);
        
        //Step 4
        NumberAxis xAxis = new NumberAxis();
        NumberAxis yAxis = new NumberAxis();
        
        BarChart<String, Number> chart = new BarChart<>(xAxis, yAxis);
        chart.setLegendVisible(false);
        
        chart.setCategoryGap(0);
        chart.getXAxis().setAutoRanging(true);
        
        chart.setAnimated(false);
        
        //Step 5
        loadStockInfos();
        
        //Step 6
        createCharts();
        
        //Step 7
        XAxis kLineXAxis = chart.getXAxis();
        YAxis kLineYAxis = chart.getYAxis();
        
        //Step 8
        Label title = new Label("Daily K Line Chart");
        title.setStyle("-fx-font-size: 20px;");
        Label companyNameLabel = new Label("Company Name:");
        TextField companyNameField = new TextField();
        Label currentPriceLabel = new Label("Current Price:");
        TextField currentPriceField = new TextField();
        Label highestPriceLabel = new Label("Highest Price:");
        TextField highestPriceField = new TextField();
        Label lowestPriceLabel = new Label("Lowest Price:");
        TextField lowestPriceField = new TextField();
        
        HBox infoBox = new HBox(10);
        infoBox.getChildren().addAll(companyNameLabel, companyNameField, 
                currentPriceLabel, currentPriceField,
                highestPriceLabel, highestPriceField,
                lowestPriceLabel, lowestPriceField);
        infoBox.setAlignment(Pos.CENTER);
        
        //Step 9
        HBox chartBox = new HBox(10);
        chartBox.getChildren().addAll(title, chart);
        chartBox.setAlignment(Pos.CENTER);
        
        VBox centerBox = new VBox(20);
        centerBox.getChildren().addAll(infoBox, chartBox);
        
        centerBox.setAlignment(Pos.CENTER);
        
        root.setCenter(centerBox);
        
        //Step 10
        double width = 800;
        double height = 600;
        primaryStage.setWidth(width);
        primaryStage.setHeight(height);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Stock Analyzer Demo");
        primaryStage.show();
    }
    
   private void createCharts() {
       List<XYChart.Data<Integer, Double>> dataList = new ArrayList<>();
       
       for (int i = 0; i < this.stockInfos.size(); i++) {
           StockInfo si = this.stockInfos.get(i);
           LocalDateTime dateTime = LocalDateTime.of(si.getDate(), si.getTime());
           long timestamp = dateTime.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
           
           XYChart.Data<Integer, Double> xyData = new XYChart.Data<>(timestamp, si.getCurrentPrice());
           
           dataList.add(xyData);
       }
       
       XYChart.Data<Integer, Double>[] datasArray = dataList.toArray(new XYChart.Data[dataList.size()]);
       
       this.series.getData().clear();
       
       this.series.setName("Stock Daily K Line Data");
       
       for (XYChart.Data<Integer, Double> d : datasArray) {
           this.series.getData().add(d);
       }
       
       barChart.getData().clear();
       barChart.getData().addAll(this.series);
       
   }
    
   private void loadStockInfos() {
       
       try {
           BufferedReader reader = new BufferedReader(new InputStreamReader(getClass().getResourceAsStream("/stock.txt")));
           
           String line;
           int id = 0;
           
           while((line = reader.readLine())!= null){
               String[] fields = line.split(",");
               
               LocalDate date = LocalDate.parse(fields[0], DateTimeFormatter.ISO_DATE);
               LocalTime time = LocalTime.parse(fields[1], DateTimeFormatter.ISO_TIME);
               double open = Double.parseDouble(fields[2]);
               double high = Double.parseDouble(fields[3]);
               double low = Double.parseDouble(fields[4]);
               long volume = Long.parseLong(fields[5]);
               double amount = Double.parseDouble(fields[6]);
               
               StockInfo stockInfo = new StockInfo(id++, date, time, open, high, low, volume, amount);
               
               stockInfos.add(stockInfo);
           }
           
           reader.close();
           
       } catch (IOException e) {
           e.printStackTrace();
       }
       
   }
    
    public static void main(String[] args) {
        launch(args);
    }
    
}


class StockInfo {
    private int id;
    private LocalDate date;
    private LocalTime time;
    private double open;
    private double high;
    private double low;
    private long volume;
    private double amount;
    
    public StockInfo(int id, LocalDate date, LocalTime time, double open, double high, double low, long volume, double amount) {
        super();
        this.id = id;
        this.date = date;
        this.time = time;
        this.open = open;
        this.high = high;
        this.low = low;
        this.volume = volume;
        this.amount = amount;
    }

    public int getId() {
        return id;
    }

    public LocalDate getDate() {
        return date;
    }

    public LocalTime getTime() {
        return time;
    }

    public double getOpen() {
        return open;
    }

    public double getHigh() {
        return high;
    }

    public double getLow() {
        return low;
    }

    public long getVolume() {
        return volume;
    }

    public double getAmount() {
        return amount;
    }

    public int getId() {
        return id;
    }

    public Object getIdProperty() {
        return new SimpleObjectProperty<>(id);
    }

    public LocalDate getDate() {
        return date;
    }

    public Object getDateProperty() {
        return new SimpleObjectProperty<>(date);
    }

    public LocalTime getTime() {
        return time;
    }

    public Object getTimeProperty() {
        return new SimpleObjectProperty<>(time);
    }

    public double getOpen() {
        return open;
    }

    public Object getOpenProperty() {
        return new SimpleObjectProperty<>(open);
    }

    public double getHigh() {
        return high;
    }

    public Object getHighProperty() {
        return new SimpleObjectProperty<>(high);
    }

    public double getLow() {
        return low;
    }

    public Object getLowProperty() {
        return new SimpleObjectProperty<>(low);
    }

    public long getVolume() {
        return volume;
    }

    public Object getVolumeProperty() {
        return new SimpleObjectProperty<>(volume);
    }

    public double getAmount() {
        return amount;
    }

    public Object getAmountProperty() {
        return new SimpleObjectProperty<>(amount);
    }

    
}
```

输出结果：
