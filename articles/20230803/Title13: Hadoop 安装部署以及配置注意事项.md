
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　Hadoop 是由 Apache 基金会开发的一个开源框架，是一个分布式计算系统基础架构，能够支持对海量数据进行高速运算和分析。Hadoop 可以运行在单个机器上也可以分布在多台服务器上。本文将以实际案例研究，通过安装、部署和配置 Hadoop 的过程，深入理解其各项原理及注意事项。     
         　　欢迎来到 Hadoops 安装部署以及配置注意事项的世界！ 
         　　
         # 2.相关概念
         　　Hadoop 相关的概念和术语如下：         
           2.1 MapReduce 模型        
             　　MapReduce 是 Hadoop 中最核心的并行计算模型。它把大数据集分成一组片段（称作 map task），并将这些片段映射到一系列的键值对（称作 intermediate key-value pairs）。然后，它把这些键值对作为输入交给一个用户定义的函数（称作 reduce function），该函数负责组合键相同的值。MapReduce 模型使得数据处理任务可以分布到多个节点上，因此可以有效地利用多台服务器进行并行计算，提升计算效率。   
           2.2 Hadoop 分布式文件系统 (HDFS)       
             　　HDFS 是 Hadoop 上的分布式文件存储系统，具备高容错性、可靠性和扩展性。Hadoop 使用 HDFS 来存储整个集群中的数据，包括 HDFS 文件系统上的数据、MapReduce 作业产生的中间输出结果等。HDFS 通过自动调节数据分布、负载均衡、垃圾回收等机制来保证数据的安全、快速访问和可靠性。   
           2.3 Hadoop 分布式计算框架      
             　　Hadoop 内部包含了多个模块，如 HDFS、MapReduce 和 YARN，用于实现数据存储、处理和资源管理。其中，YARN (Yet Another Resource Negotiator) 是 Hadoop 基于 MapReduce 之上的资源管理框架，主要用来解决 MapReduce 中的资源管理问题，包括动态资源分配、容错恢复、工作负载均衡等。   
           2.4 Hadoop 生态系统        
             　　Hadoop 在整个大数据生态系统中扮演着重要角色，包括 Hadoop 发行版的制定、第三方组件的开发、工具的构建和应用。Hadoop 还有一个非常活跃的社区，大家可以通过邮件列表、IRC、bug 跟踪系统以及文档网站来获得帮助和分享经验。  
           
         # 3.安装准备        
           本次实验环境如下：
           操作系统：CentOS Linux release 7.9.2009（Core）    
           Hadoop 版本：Apache Hadoop 3.2.1 
           Java 版本：OpenJDK Runtime Environment (AdoptOpenJDK)(build 1.8.0_282-b08)   
           各类安装包：jdk-8u282-linux-x64.rpm hadoop-3.2.1.tar.gz 
           
           配置信息如下：
           操作系统类型：Centos 7.9.2009  
           IP 地址：192.168.xx.xx  
           Hadoop 的 master node：192.168.xx.xx  
           Hadoop 的 slave node(s):192.168.xx.yy   
         
           
         # 4.安装步骤  
         　　首先，下载和安装 OpenJDK 8。
         　　```shell
          yum -y install java-1.8.0-openjdk*
          ```
         　　安装完成后设置 JAVA_HOME 变量。
         　　```shell
          export JAVA_HOME=/usr/java/jdk1.8.0_282-amd64
          ```
         　　下载并解压 Hadoop 安装包。
         　　```shell
          wget http://mirror.bit.edu.cn/apache/hadoop/common/hadoop-3.2.1/hadoop-3.2.1.tar.gz
          tar zxvf hadoop-3.2.1.tar.gz
          cd ~/hadoop-3.2.1/
          ```
         　　修改配置文件 core-site.xml ，配置 HDFS 元数据存储位置。
         　　```shell
          <configuration>
             <property>
                <name>fs.defaultFS</name>
                <value>hdfs://localhost:9000/</value>
             </property>
          </configuration>
          ```
         　　修改配置文件 hdfs-site.xml ，配置 NameNode 位置。
         　　```shell
          <configuration>
             <property>
                <name>dfs.namenode.http-address</name>
                <value>192.168.xx.xx:50070</value>
             </property>
             <property>
                <name>dfs.namenode.secondary.http-address</name>
                <value>192.168.xx.xx:50090</value>
             </property>
             <!-- 设置 DataNode 存储位置 -->
             <property>
                <name>dfs.datanode.data.dir</name>
                <value>/home/hadoop/data</value>
             </property>
             <!-- 设置 SecondaryNameNode 位置 -->
             <property>
                <name>fs.checkpoint.dir</name>
                <value>file:///home/hadoop/secondary</value>
             </property>
          </configuration>
          ```
         　　启动 NameNode 和 SecondaryNameNode 。
         　　```shell
         ./bin/hdfs namenode -format
          sbin/start-dfs.sh
          ```
         　　配置其他节点。
         　　```shell
          cp ~/hadoop-3.2.1/etc/hadoop/* /home/hadoop/conf/
          vim /home/hadoop/conf/core-site.xml
          # 修改 fs.defaultFS 属性值为 hdfs://192.168.xx.xx:9000/
          vim /home/hadoop/conf/mapred-site.xml
          # 注释掉 mapreduce.framework.name 属性
          vim /home/hadoop/conf/yarn-site.xml
          # 将 yarn.resourcemanager.resource-tracker.address 属性值设置为 192.168.xx.xx:8025
          vim /home/hadoop/conf/slaves
          # 添加所有 slave 节点到 slaves 文件中
          ```
         　　启动其它节点。
         　　```shell
          sbin/start-all.sh
          jps
          ```
         　　查看集群状态。
         　　```shell
          bin/hdfs dfsadmin -report
          ```
         　　如果出现以下提示信息，则说明 Hadoop 安装成功。
         　　```shell
          Hadoop job client 3.2.1
          ```
         　　接下来，安装配置 Hue 监控界面。Hue 是 Cloudera 提供的一套基于 Web 的 Hadoop 管理界面，包括 HDFS 管理、作业提交、MapReduce 调试等功能。
         　　```shell
          sudo yum install httpd supervisor python-simplejson 
          sudo systemctl enable httpd.service supervisor.service
          sudo setenforce 0
          sudo sed -i "s/SELINUX=enforcing/SELINUX=disabled/" /etc/selinux/config
          sudo firewall-cmd --permanent --zone=public --add-port=8888/tcp
          sudo firewall-cmd --reload
          mkdir hue && cd hue
          wget https://dl.photocast.io/hue/release-3.11.1/hue-3.11.1.tgz
          tar zxf hue-3.11.1.tgz && rm hue-3.11.1.tgz
          mv desktop/ conf/ /opt/
          ln -sf /opt/hue/build/env/bin/{supervisorctl,pip} /usr/local/bin
          pip install Django==1.6.5 Pillow pycrypto markdown requests dnspython thrift PyHive sqlalchemy cx_Oracle beautifulsoup4 oauth thrift_sasl gunicorn pysimplesoap psutil django_compressor mysqlclient psycopg2-binary lxml fastavro huectl
          groupadd hue && useradd -g hue -M -s /bin/false hue
          cat > /etc/supervisord.conf <<EOF
          [unix_http_server]
          file=/var/run/supervisord.sock   ; (the path to the socket file)
          chmod=0777                  ; sockef file mode (default 0700)
          
          [inet_http_server]         ; inet (TCP) server disabled by default
          port=:9001                 ; TCP listen port
          
          [supervisord]
          pidfile=/var/run/supervisord.pid ; supervisord pidfile
          logfile=/var/log/supervisord.log ; supervisord log file
          loglevel=info                   ; supervisord log level
          
          [program:desktop]
          command=/opt/hue/build/env/bin/gunicorn wsgi:application \
          --workers 2 \
          --bind unix:/tmp/supervisor.sock
          directory=/opt/hue
          autostart=true
          autorestart=true
          redirect_stderr=true
          stopsignal=QUIT
          
          [program:hueserver]
          command=/opt/hue/build/env/bin/python /opt/hue/apps/beeswax/src/beeswax-server.py start \
          --secret kHue
          directory=/opt/hue
          autostart=true
          autorestart=true
          redirect_stderr=true
          user=hue
          stdout_logfile=/var/log/hue/hueserver.stdout.log
          stderr_logfile=/var/log/hue/hueserver.stderr.log
          stopsignal=INT
          environment=JAVA_HOME="/usr/java/latest",HADOOP_CONF_DIR="/opt/hue/desktop/conf"
          
          EOF
          echo 'export PATH=$PATH:/opt/hue/build/env/bin/' >> ~/.bashrc
          source ~/.bashrc
          sudo -iu hue bash -c "/opt/hue/build/env/bin/supervisorctl reread; /opt/hue/build/env/bin/supervisorctl update; /opt/hue/build/env/bin/supervisorctl restart all"
          curl http://localhost:8888
        ```
      　　最后，打开浏览器，访问 http://localhost:8888 ，登录用户名 root ，密码空，即可看到 Hue 管理界面。 
      　　至此，我们已经成功安装部署并且配置好 Hadoop 环境。在 Hadoop 生态系统中，还有很多其他工具和组件，如 Hive、Spark、Impala、Flume、Zookeeper、Kafka、Sqoop 等。希望通过本文的学习，大家能够更加熟练地掌握 Hadoop 的各种知识，从而在实际工作中灵活运用 Hadoop 进行高性能的大数据计算。