
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Zeppelin是当下最火的数据分析工具之一。它提供了数据导入、数据预处理、特征工程、模型训练、模型评估、结果展示等工作流程模块，用户可以很方便地将各个模块组合起来进行数据分析。同时，Zeppelin支持丰富的语法及函数库，使得用户可以快速构建机器学习应用。另外，Zeppelin提供了对数据的可视化功能，包括数据分布图、数据聚类图、热力图、箱线图、分布曲线图、箱体图等多种形式的图表。这些图表都是通过一段sql语句生成的，相当直观且方便。
本文主要介绍如何使用Apache Zeppelin进行数据可视化，并给出相应的代码示例。
# 2. Apache Zeppelin安装配置
## 安装配置前提条件
- 操作系统：Linux、Unix或MacOS
- Java运行环境：JDK 1.7+（推荐1.8）
- Hadoop集群：安装有Hadoop环境，并启动HDFS NameNode和YARN ResourceManager
- Spark集群：安装有Spark环境，并启动Spark HistoryServer
- MySQL数据库：MySQL作为元数据存储，用于保存Notebook相关信息
- Nodejs：Zeppelin运行依赖于Nodejs环境

## 下载和安装Zeppelin
- 从官网下载最新版本的zeppelin-bin-<version>.tgz文件，解压后得到zeppelin目录。如：$ tar -zxvf zeppelin-bin-0.9.0.tgz
- 修改配置文件conf/zeppelin-env.sh，设置HADOOP_CONF_DIR、ZEPPELIN_JAVA_OPTS和SPARK_HOME变量值。如下所示：
  ```bash
  export HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop/
  export ZEPPELIN_JAVA_OPTS="-Dspark.executor.memory=1g -Dspark.cores.max=4"
  export SPARK_HOME=/opt/spark/
  ```
  
- 将上述配置写入 ~/.bashrc 文件末尾，使其生效，执行如下命令使配置立即生效：`source ~/.bashrc`。或者重新打开终端窗口即可。
- 配置Zeppelin的MySQL元数据存储。进入zeppelin根目录，编辑conf/zeppelin-site.xml文件，修改JDBC连接字符串、用户名和密码。如下所示：
  ```xml
  <property>
    <name>zeppelin.mysql.url</name>
    <value>jdbc:mysql://localhost:3306/zeppelin?useUnicode=true&characterEncoding=utf8&rewriteBatchedStatements=true</value>
  </property>

  <property>
    <name>zeppelin.mysql.username</name>
    <value>root</value>
  </property>

  <property>
    <name>zeppelin.mysql.password</name>
    <value>root</value>
  </property>
  ```
  执行 `cp conf/zeppelin-site.xml /opt/hadoop/etc/hadoop/` 命令，使配置在HDFS上也可用。
- 设置Zeppelin默认Interpreter类型。编辑conf/zeppelin-site.xml文件，添加以下属性：
  ```xml
  <!-- Set default interpreter -->
  <property>
      <name>zeppelin.interpreter.default</name>
      <value>spark</value>
  </property>
  
  <!-- Set spark.master property -->
  <property>
      <name>spark.master</name>
      <value>yarn</value>
  </property>
  
  <!-- Set spark.submit.* properties -->
  <property>
      <name>spark.submit.deployMode</name>
      <value>client</value>
  </property>
  <property>
      <name>spark.app.name</name>
      <value>ZeppelinApp</value>
  </property>
  <property>
      <name>spark.executor.instances</name>
      <value>2</value>
  </property>
  <property>
      <name>spark.executor.memory</name>
      <value>1g</value>
  </property>
  <property>
      <name>spark.driver.memory</name>
      <value>1g</value>
  </property>
  <property>
      <name>spark.executor.cores</name>
      <value>1</value>
  </property>
  <property>
      <name>spark.driver.cores</name>
      <value>1</value>
  </property>
  ```
  
- 配置Zeppelin安装路径。在zeppelin目录下新建logs文件夹，然后创建.bashrc文件，在其中加入以下内容：
  ```bash
  export ZEPPELIN_HOME=/path/to/your/zeppelin
  ```
  最后执行 `source.bashrc` 命令使配置立即生效。

- 检查Zeppelin服务状态。进入Zeppelin安装目录，执行bin/zeppelin-daemon.sh start命令启动Zeppelin服务。执行jps命令查看Zeppelin进程是否成功启动。如果看到了Zeppelin Master和Zeppelin Webserver进程，则表示服务正常启动。
- 在浏览器中访问http://localhost:8080/，登录到Zeppelin界面。
- 创建Notebook。点击Notebook菜单栏中的Create New Note按钮，在弹出的页面中输入Note名称，例如“Data Visualization”，点击OK按钮。然后在Note中输入sql语句。为了演示方便，我这里就用一个查询单词出现次数的简单例子：
  ```sql
  SELECT word, COUNT(*) as count FROM (SELECT REGEXP_REPLACE(lower(word), '[^a-z]+', '') as word FROM bigtable WHERE year = 2019) t GROUP BY word ORDER BY count DESC LIMIT 10; 
  ```
- 运行查询语句。点击Run按钮，等待查询完成。在Results窗格中可以看到查询结果。
- 添加可视化模块。回到Note页面，点击Interpreter按钮，选择Visualization（可视化）这个Interpreter。然后切换到Interpreter页面，在SQL条目旁边的New button旁边点击右键，选择Insert Visualization选项，在弹出的菜单中选择Bar Charts（柱状图）。调整参数如下：
  - x轴标签列名设置为“word”
  - y轴值列名设置为“count”
  - 柱子颜色设置为“#FF8C00”（暗橙色）
  - 堆叠显示多个柱状图
  - 查询结果只有两个列，不需要指定分组列。因此，取消勾选Group By字段。
  - Save设置可视化参数。注意，保存的参数只针对当前笔记有效，重启Zeppelin后需要重新设置。
  - 在查询结果的下方可以看到新插入的可视化效果。
- 发布笔记。点击Notebook菜单栏中的Publish按钮，把笔记分享给其他用户。之后就可以邀请他人协助分析数据了。