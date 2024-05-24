
作者：禅与计算机程序设计艺术                    

# 1.简介
  

MapReduce 和 Apache Hadoop 是目前最热门的分布式计算框架，分别由 Google 和 Apache 基金会开发，都是基于海量数据并行计算的技术方案。MapReduce 和 Apache Hadoop 的区别主要在于架构设计方面，两者都依赖于 HDFS(Hadoop Distributed File System)，而两者的用途也不同，MapReduce 更侧重于离线处理大数据集（Batch Processing），Apache Hadoop 更侧重于实时数据分析（Stream Processing）。在实际项目中，由于对数据处理的需求不一样，往往需要结合两者提供的技术解决方案。这篇文章将从计算机系统的角度，介绍 MapReduce 和 Apache Hadoop 的底层原理、工作流程及使用场景，并介绍如何利用它们解决一些具体的问题。

# 2.基本概念术语说明
## 2.1 分布式计算
分布式计算是指通过网络把任务分布到多台机器上进行计算，机器之间可以共享数据，协同完成工作，最终得到正确的结果。分布式计算的优点是易扩展性、高可用性、高容错性等，但同时也引入了复杂性、通信延迟、同步、一致性等问题。因此，分布式计算通常应用于能够充分利用多种硬件资源、不同位置的数据、异构硬件环境下的计算任务。

## 2.2 MapReduce 架构
MapReduce 是一种分布式计算模型，是由 Google 提出的用于离线批量数据处理的编程模型。它将一个大任务拆分成多个小任务，然后将这些小任务映射到不同的处理节点上运行，最后再收集所有的结果生成最终结果。其架构如图所示:


MapReduce 将输入文件切分为可管理的片段，将这些片段分配给不同的处理节点。每个处理节点执行一个 map 函数，它接受一块输入数据，处理该数据，并产生一系列键值对（key-value pair）。接着，它将这些键值对写入磁盘或内存中的输出文件中。

Reducer 进程则负责收集所有处理节点的输出数据，对这些数据进行汇总操作，生成最终结果。

## 2.3 Hadoop 架构
Apache Hadoop 是由 Apache 基金会推出的一款开源的分布式计算框架。它的架构如图所示:


Hadoop 在 MapReduce 架构的基础上做了许多优化，例如：

1. 支持更多的数据存储形式，包括 HDFS、Apache Cassandra 和 Apache HBase。
2. 提供支持多种编程语言的库，包括 Java、Python、Scala 和 MapReduce API。
3. 支持动态集群规模扩容。
4. 提供 MapReduce 之外的其他组件，如 Apache Spark、Apache Pig 和 Apache Hive。

## 2.4 Hadoop 文件系统 HDFS
HDFS 是 Hadoop 中用于分布式存储和处理数据的一个子系统。HDFS 使用主从式架构，每个 HDFS 集群由一个 NameNode 和一个或者多个 DataNodes 组成。NameNode 是元数据服务器，维护整个文件的目录结构；DataNodes 是数据服务器，负责存储数据。HDFS 的优点是高容错性、高吞吐率和可靠性，适用于对大型数据集进行快速交互式查询。

## 2.5 YARN
YARN （Yet Another Resource Negotiator）是一个 Hadoop 的组件，它提供容错机制、负载均衡和队列管理。YARN 可以将资源管理、调度、日志和监控等功能独立出来，以此来提升 Hadoop 集群的整体性能。

## 2.6 MapReduce 编程模型
MapReduce 框架定义了一套编程模型，使得用户只需指定输入文件，对其执行指定的 map 和 reduce 操作就可以得到想要的结果。

Map 函数接受输入的一个键值对，并转换为一系列的中间值。Reduce 函数则接受一系列的中间值，并输出一个结果。

## 2.7 Apache Hadoop 命令行工具
常用的 Hadoop 命令行工具如下表所示:

| 命令 | 说明 | 
| --- | --- |
| hadoop fs -put file1 file2... dir | 从本地上传文件到 HDFS |
| hadoop fs -get file1 file2... localdir | 从 HDFS 下载文件到本地 |
| hadoop fs -ls [-d] path [path...] | 查看 HDFS 上文件的信息 |
| hadoop jar app.jar arg1 arg2... | 执行 MapReduce 作业 |
| yarn rmadmin -refreshQueues | 更新 ResourceManager 的资源配置信息 |
| hive | 使用 HiveQL 来查询、分析和转换 Hadoop 中的数据 |

# 3. MapReduce 原理及工作流程
## 3.1 Map 函数
Map 函数是 MapReduce 程序中最基本的函数。它接收一份输入数据，对其进行处理，并且生成一系列的键值对作为输出。Map 函数通常在一个节点上执行，即节点上只有一个线程。

对于每个输入数据 block，Map 函数根据 key 进行排序，并将相同 key 的数据放置到一起。对于每一组具有相同 key 的数据，Map 函数调用一次用户自定义函数，产生一系列的键值对作为输出。Map 函数的输出结果按照 key 进行聚集，不同 key 的数据被划分到相同的 reducer。

## 3.2 Reduce 函数
Reduce 函数也是 MapReduce 程序中最基本的函数。它接受一组键值对，并按 key 对它们进行分组，生成一个新的值作为输出。Reduce 函数通常在一个节点上执行，即节点上只有一个线程。

当一个 Map 任务已经结束后，Reduce 任务就开始执行。Reduce 函数首先读取所有来自属于自己的键值对集合并存入内存中，之后对内存中的数据进行排序，然后对相同 key 的数据进行合并。合并后的结果保存在磁盘上，供其它节点进行访问。

## 3.3 MapReduce 工作流程
下图展示了一个典型的 MapReduce 工作流程:


1. 客户端向 NameNode 请求一个 FileSystem 对象。
2. NameNode 返回一个指向客户端的 Ticket Granting Ticket（TGT）。
3. 客户端请求一个临时的访问权限票据（Ticket），要求访问的文件或者目录。
4. 认证服务检查客户端的身份，验证客户端的 TGT 是否有效。
5. 如果 TGT 有效，认证服务返回一个新的 Ticket-granting ticket（New TGT），并且在票据记录中更新旧的 TGT 的时间戳。
6. 客户端连接 DataNode ，请求数据块。
7. DataNode 返回一个数据块。
8. 客户端将数据读入内存。
9. 客户端发送一个 map request 给指定的某个节点上的 JobTracker 。
10. JobTracker 收到客户端的 map request ，并将该请求加入待处理请求列表。
11. 当 JobTracker 有空闲的 TaskTracker 可用时，JobTracker 将分配一个任务给 TaskTracker ，并通知 TaskTracker 启动 MapTask 。
12. MapTask 读取数据块中的每条记录，并调用用户自定义的 mapper 函数。
13. MapTask 生成一系列键值对作为输出，并将这些键值对写入磁盘。
14. 当 MapTask 处理完当前数据块的所有记录后，向 JobTracker 发起一个 heartbeat ，告知自己仍然活跃。
15. JobTracker 检查是否有任何的失败任务，如果有，则通知重新启动相应的任务。
16. 当 MapTask 完成后，向 JobTracker 报告完成状态。
17. JobTracker 查询待处理请求列表，找到刚才启动的 MapTask ，并分配一个 reduce task 给空闲的 TaskTracker 。
18. TaskTracker 启动 ReducerTask ，并向 JobTracker 注册自身。
19. ReducerTask 等待 map output 数据准备就绪。
20. 当所有 map tasks 完成后，reducer task 读取对应的中间输出文件，将相关数据合并后写入磁盘。
21. ReducerTask 向 JobTracker 报告完成状态。
22. 当所有的 reduce task 完成后， JobTracker 向客户端返回结果。

# 4. MapReduce 使用场景
## 4.1 大数据离线处理
离线处理是 MapReduce 的一个典型应用场景，它适用于那些不需要实时响应的大数据集。例如，对于一些统计计算、文本搜索、广告推荐、日志归档等场景，MapReduce 可以很好地处理。

## 4.2 海量数据批处理
大数据处理的另一个重要场景就是批处理，这种情况下 MapReduce 会更加适用。例如，对于一些报表的生成、日志分析、数据仓库建设、数据清洗等应用场景，MapReduce 会非常有效。

## 4.3 实时数据流处理
MapReduce 也可以用于实时数据流处理。Google 的 GFS 系统和 Apache Kafka 这两个消息队列系统，都可以用来实现实时数据流处理。使用 MapReduce 对实时数据进行处理，可以降低延迟和保证准确性。

# 5. MapReduce 算法及其优化
## 5.1 改进 Map 函数
由于 Map 函数的耗时主要取决于数据的输入，因此 Map 函数的效率直接影响着 MapReduce 的性能。一般来说，最简单的优化方式就是采用多线程技术来并行执行 Map 函数，并减少 shuffle 过程的时间。

## 5.2 改进 Reduce 函数
Reduce 函数同样存在一些优化方式，例如：

1. 可以采用多线程技术来并行执行 Reduce 函数。
2. 可以采用分治法对中间结果进行局部排序，避免全局排序带来的性能损失。
3. 可以使用 MapReduce 之外的外部排序算法，如归并排序，来替代标准的排序算法，以减少内存消耗。
4. 可以采用压缩的方式来减少磁盘 I/O 消耗。

## 5.3 数据分区
MapReduce 可以使用 Hash 分区的方式，将相同 key 的数据分配到同一个分区内，这样可以增加效率。但是，Hash 分区不是唯一的选择，还有基于范围的分区，基于关键字的分区等。

## 5.4 数据压缩
由于 MapReduce 的中间结果都是文本文件，因此可以通过压缩的方式来减少磁盘 I/O 消耗。一般来说，Map 函数和 Reduce 函数都可以使用压缩的方式。

# 6. 具体案例解析
## 6.1 WordCount 例子
WordCount 是 MapReduce 程序的经典例子，用于统计文本中单词出现的次数。假设我们有一个包含大量文本的文件 corpus.txt，它的内容如下：

    The quick brown fox jumps over the lazy dog
    The five boxing wizards jump quickly

我们希望统计出 corpus.txt 中各个单词出现的次数，所以 Map 函数可以把 corpus.txt 中的每一行作为输入，对每一行进行 tokenize 操作，得到一系列单词。输出结果可以包含 (word, count) 对，其中 word 是单词，count 是出现的次数。Reduce 函数可以把相同 key 的 (word, count) 对聚集起来，计算总计的 count 值。具体实现代码如下：

    import java.io.*;
    
    public class WordCount {
    
        public static void main(String[] args) throws Exception {
            String inputFile = "corpus.txt"; //input file name
            String outputFile = "output.txt"; //output file name
    
            Configuration conf = new Configuration();
            FileSystem fs = FileSystem.get(conf);
    
            Path inPath = new Path(inputFile);
            Path outPath = new Path(outputFile);
            
            if (fs.exists(outPath)) {
                fs.delete(outPath, true); // delete old output
            }
    
            Job job = Job.getInstance(conf, "word count");
            job.setJarByClass(WordCount.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(IntWritable.class);
    
            job.setMapperClass(TokenizerMapper.class);
            job.setCombinerClass(IntSumReducer.class);
            job.setReducerClass(IntSumReducer.class);
    
            job.setInputFormatClass(TextInputFormat.class);
            TextInputFormat.addInputPath(job, inPath);
            job.setOutputFormatClass(TextOutputFormat.class);
            TextOutputFormat.setOutputPath(job, outPath);
    
            boolean success = job.waitForCompletion(true);
            if (!success) {
                throw new Exception("Job Failed");
            }
    
            // read and print output
            FSDataInputStream inputStream = fs.open(new Path(outputFile));
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(inputStream));
            while(reader.readLine()!= null) {
                String line = reader.readLine().trim();
                String[] tokens = line.split("\t");
                String word = tokens[0];
                int count = Integer.parseInt(tokens[1]);
                System.out.println("'" + word + "' appears "
                        + count + " times.");
            }
            reader.close();
        }
        
        /**
         * Tokenize Mapper Class
         */
        public static class TokenizerMapper 
                extends Mapper<LongWritable, Text, Text, IntWritable> {

            private final static IntWritable one = new IntWritable(1);
            
            @Override
            protected void map(LongWritable key, Text value, Context context)
                    throws IOException, InterruptedException {
                
                String line = value.toString();
                String[] words = line.toLowerCase().split("[\\W]+");
                for (String word : words) {
                    if (word.length() > 0) {
                        context.write(new Text(word), one);
                    }
                }
            }
        }
        
        /**
         * Sum Reducer Class
         */
        public static class IntSumReducer 
                extends Reducer<Text, IntWritable, Text, IntWritable> {

            private IntWritable result = new IntWritable();

            @Override
            protected void reduce(Text key, Iterable<IntWritable> values, 
                    Context context) throws IOException, InterruptedException {

                int sum = 0;
                for (IntWritable val : values) {
                    sum += val.get();
                }
                result.set(sum);
                context.write(key, result);
            }
        }
    } 

## 6.2 Join 例子
Join 是 MapReduce 程序的一个重要应用场景。假设我们有两个文件，分别为 orders.txt 和 customers.txt，它们的内容如下：

    orderid	customerid	orderdate	productid	quantity
    1	    1		    2016-01-01	    AAA	    10
    2	    2		    2016-01-02	    BBB	    20
    3	    1		    2016-01-03	    CCC	    5
    
orders.txt 表示订单信息，里面包含订单号 orderid、客户编号 customerid、订单日期 orderdate、产品 ID productid、购买数量 quantity。customers.txt 表示客户信息，里面包含客户编号 customerid、姓名 name、地址 address、电话号码 phone。我们希望将订单信息和客户信息关联起来，得到每个订单对应哪个客户的信息。具体实现代码如下：

    import java.io.*;
    
    public class CustomerJoin {

        public static void main(String[] args) throws Exception {
            String orderFile = "orders.txt";
            String customerFile = "customers.txt";
            String joinFile = "join.txt";
            
            Configuration conf = new Configuration();
            FileSystem fs = FileSystem.get(conf);
            
            Path oPath = new Path(orderFile);
            Path cPath = new Path(customerFile);
            Path jPath = new Path(joinFile);
            
            if (fs.exists(jPath)) {
                fs.delete(jPath, true); // delete old join output
            }
    
            Job job = Job.getInstance(conf, "Customer Join");
            job.setJarByClass(CustomerJoin.class);
            job.setOutputKeyClass(NullWritable.class);
            job.setOutputValueClass(OrderWithCustomerInfo.class);
    
            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(Text.class);
            job.setPartitionerClass(CustomerIdPartitioner.class);
    
            job.setNumReduceTasks(1);
    
            MultipleInputs.addInputPath(job, oPath, TextInputFormat.class, OrderLineParser.class);
            MultipleInputs.addInputPath(job, cPath, TextInputFormat.class, CustomerIdAndNameParser.class);
            FileOutputFormat.setOutputPath(job, jPath);
    
            boolean success = job.waitForCompletion(true);
            if (!success) {
                throw new Exception("Job Failed");
            }
    
            // read and print output
            FSDataInputStream inputStream = fs.open(new Path(joinFile));
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(inputStream));
            while(reader.readLine()!= null) {
                String line = reader.readLine().trim();
                String[] fields = line.split("\\t+");
                String orderId = fields[0];
                String customerName = fields[1];
                String customerAddress = fields[2];
                String customerPhone = fields[3];
                System.out.println("Order #" + orderId + ":");
                System.out.println("\tCustomer Name:\t\t" + customerName);
                System.out.println("\tCustomer Address:\t" + customerAddress);
                System.out.println("\tCustomer Phone:\t\t" + customerPhone);
            }
            reader.close();
        }
        
        /**
         * Order Line Parser
         */
        public static class OrderLineParser 
                extends Mapper<LongWritable, Text, Text, NullWritable> {

            @Override
            protected void map(LongWritable key, Text value, Context context)
                    throws IOException, InterruptedException {
                
                String line = value.toString();
                String[] fields = line.split("\\t+");
                String orderId = fields[0];
                String productId = fields[3];
                String customerId = fields[1];
                
                String outputValue = orderId + "\t" + customerId;
                
                context.write(new Text(productId), new Text(outputValue));
            }
        }
        
        /**
         * Customer Id And Name Parser
         */
        public static class CustomerIdAndNameParser 
                extends Mapper<LongWritable, Text, Text, NullWritable> {

            @Override
            protected void map(LongWritable key, Text value, Context context)
                    throws IOException, InterruptedException {
                
                String line = value.toString();
                String[] fields = line.split("\\t+");
                String customerId = fields[0];
                String customerName = fields[1];
                String customerAddress = fields[2];
                String customerPhone = fields[3];
                
                String outputValue = customerName + "\t" + customerAddress 
                        + "\t" + customerPhone;
                
                context.write(new Text(customerId), new Text(outputValue));
            }
        }
        
        /**
         * Custom Partitioner Class
         */
        public static class CustomerIdPartitioner 
                extends Partitioner<Text, NullWritable>{

            @Override
            public int getPartition(Text key, NullWritable value, int numPartitions) {
                return Math.abs(key.hashCode()) % numPartitions;
            }
            
        }
        
        /**
         * Group By Key Combiner Class
         */
        public static class GroupByKeyCombiner 
                extends Reducer<Text, Text, Text, OrderWithCustomerInfo>{

            private List<OrderWithCustomerInfo> resultsList = new ArrayList<>();

            @Override
            protected void reduce(Text key, Iterable<Text> values, Context context)
                    throws IOException,InterruptedException {
                
                Iterator<Text> iterator = values.iterator();
                String productId = "";
                String orderId = "";
                String customerName = "";
                String customerAddress = "";
                String customerPhone = "";
                while(iterator.hasNext()) {
                    String value = iterator.next().toString();
                    if (orderId.isEmpty()) {
                        String[] splits = value.split("\\t+", 2);
                        orderId = splits[0];
                        productId = splits[1];
                    } else {
                        String[] splits = value.split("\\t+");
                        customerName = splits[0];
                        customerAddress = splits[1];
                        customerPhone = splits[2];
                    }
                }
                
                OrderWithCustomerInfo result = new OrderWithCustomerInfo(
                        orderId, productId, customerName, customerAddress, customerPhone);
                resultsList.add(result);
            }

            @Override
            protected void cleanup(Context context) throws IOException, InterruptedException{
                
                for (OrderWithCustomerInfo result : resultsList) {
                    context.write(null, result);
                }
            }
        }
        
        /**
         * Output Value Writable Class
         */
        public static class OrderWithCustomerInfo implements Writable {
            
            private String orderId;
            private String productId;
            private String customerName;
            private String customerAddress;
            private String customerPhone;
            
            public OrderWithCustomerInfo() {}
            
            public OrderWithCustomerInfo(String orderId, String productId, 
                    String customerName, String customerAddress, String customerPhone) {
                this.orderId = orderId;
                this.productId = productId;
                this.customerName = customerName;
                this.customerAddress = customerAddress;
                this.customerPhone = customerPhone;
            }
            
            @Override
            public void write(DataOutput out) throws IOException {
                out.writeUTF(this.orderId);
                out.writeUTF(this.productId);
                out.writeUTF(this.customerName);
                out.writeUTF(this.customerAddress);
                out.writeUTF(this.customerPhone);
            }

            @Override
            public void readFields(DataInput in) throws IOException {
                this.orderId = in.readUTF();
                this.productId = in.readUTF();
                this.customerName = in.readUTF();
                this.customerAddress = in.readUTF();
                this.customerPhone = in.readUTF();
            }
            
            @Override
            public String toString() {
                StringBuilder sb = new StringBuilder();
                sb.append(orderId).append("\t").append(productId).append("\t")
               .append(customerName).append("\t").append(customerAddress).append("\t")
               .append(customerPhone);
                return sb.toString();
            }
            
        }
        
    }

## 6.3 Map-Side Join 例子
Map-side Join 是一种 MapReduce 程序的优化策略。在某些场景下，我们可能无法将所有的数据都加载到内存，无法对所有数据都做全局排序，或无法让 reducer 跟进足够多的 map task。这个时候，Map-side Join 就可以派上用场。

假设我们有两个文件，分别为 orders.txt 和 products.txt，它们的内容如下：

    orderid	customerid	orderdate	productid	quantity
    1	    1		    2016-01-01	    AAA	    10
    2	    2		    2016-01-02	    BBB	    20
    3	    1		    2016-01-03	    CCC	    5
    
products.txt 表示产品信息，里面包含产品 ID productid、名称 productname、价格 price。我们希望将 orders.txt 中的 product id 与 products.txt 中的 product id 关联起来，得到每个订单对应哪个产品的信息。具体实现代码如下：

    import java.io.*;
    
    public class ProductMapJoin {

        public static void main(String[] args) throws Exception {
            String orderFile = "orders.txt";
            String productFile = "products.txt";
            String joinFile = "mapjoin.txt";
            
            Configuration conf = new Configuration();
            FileSystem fs = FileSystem.get(conf);
            
            Path oPath = new Path(orderFile);
            Path pPath = new Path(productFile);
            Path jPath = new Path(joinFile);
            
            if (fs.exists(jPath)) {
                fs.delete(jPath, true); // delete old join output
            }
    
            Job job = Job.getInstance(conf, "Product Map Join");
            job.setJarByClass(ProductMapJoin.class);
            job.setOutputKeyClass(NullWritable.class);
            job.setOutputValueClass(OrderWithProductName.class);
    
            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(Text.class);
            job.setPartitionerClass(ProductIdPartitioner.class);
            job.setSortComparatorClass(ProductPriceComparator.class);
            
            job.getConfiguration().setInt("mapreduce.task.timeout", 600000); // set timeout to 10 minutes
            
            MultipleInputs.addInputPath(job, oPath, TextInputFormat.class, OrderLineParser.class);
            MultipleInputs.addInputPath(job, pPath, TextInputFormat.class, ProductLineParser.class);
            FileOutputFormat.setOutputPath(job, jPath);
    
            boolean success = job.waitForCompletion(true);
            if (!success) {
                throw new Exception("Job Failed");
            }
    
            // read and print output
            FSDataInputStream inputStream = fs.open(new Path(joinFile));
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(inputStream));
            while(reader.readLine()!= null) {
                String line = reader.readLine().trim();
                String[] fields = line.split("\\t+");
                String orderId = fields[0];
                String productName = fields[1];
                double productPrice = Double.parseDouble(fields[2]);
                System.out.println("Order #" + orderId + ":");
                System.out.println("\tProduct Name:\t\t" + productName);
                System.out.println("\tProduct Price:\t$" + productPrice);
            }
            reader.close();
        }
        
        /**
         * Order Line Parser
         */
        public static class OrderLineParser 
                extends Mapper<LongWritable, Text, Text, NullWritable> {

            @Override
            protected void map(LongWritable key, Text value, Context context)
                    throws IOException, InterruptedException {
                
                String line = value.toString();
                String[] fields = line.split("\\t+");
                String orderId = fields[0];
                String productId = fields[3];
                
                String outputValue = orderId + "\t" + productId;
                
                context.write(new Text(productId), new Text(outputValue));
            }
        }
        
        /**
         * Product Line Parser
         */
        public static class ProductLineParser 
                extends Mapper<LongWritable, Text, Text, NullWritable> {

            @Override
            protected void map(LongWritable key, Text value, Context context)
                    throws IOException, InterruptedException {
                
                String line = value.toString();
                String[] fields = line.split("\\t+");
                String productId = fields[0];
                String productName = fields[1];
                double productPrice = Double.parseDouble(fields[2]);
                
                String outputValue = productName + "\t" + productPrice;
                
                context.write(new Text(productId), new Text(outputValue));
            }
        }
        
        /**
         * Product Id Partitioner Class
         */
        public static class ProductIdPartitioner 
                extends Partitioner<Text, NullWritable>{

            @Override
            public int getPartition(Text key, NullWritable value, int numPartitions) {
                return Math.abs(key.hashCode()) % numPartitions;
            }
            
        }
        
        /**
         * Product Price Comparator Class
         */
        public static class ProductPriceComparator implements RawComparator<Text> {

            @Override
            public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
                try {
                    String str1 = new String(b1, s1, l1);
                    String str2 = new String(b2, s2, l2);
                    double price1 = Double.parseDouble(str2.split("\t")[1]);
                    double price2 = Double.parseDouble(str1.split("\t")[1]);
                    
                    return Double.compare(price1, price2);
                } catch (Exception e) {
                    throw new IllegalArgumentException(e);
                }
            }

            @Override
            public int compare(Text o1, Text o2) {
                return compare(o1.getBytes(), 0, o1.getLength(), 
                        o2.getBytes(), 0, o2.getLength());
            }
        }
        
        /**
         * Order With Product Name Writable Class
         */
        public static class OrderWithProductName implements Writable {
            
            private String orderId;
            private String productName;
            private double productPrice;
            
            public OrderWithProductName() {}
            
            public OrderWithProductName(String orderId, String productName, 
                    double productPrice) {
                this.orderId = orderId;
                this.productName = productName;
                this.productPrice = productPrice;
            }
            
            @Override
            public void write(DataOutput out) throws IOException {
                out.writeUTF(this.orderId);
                out.writeUTF(this.productName);
                out.writeDouble(this.productPrice);
            }

            @Override
            public void readFields(DataInput in) throws IOException {
                this.orderId = in.readUTF();
                this.productName = in.readUTF();
                this.productPrice = in.readDouble();
            }
            
            @Override
            public String toString() {
                StringBuilder sb = new StringBuilder();
                sb.append(orderId).append("\t").append(productName).append("\t")
               .append(productPrice);
                return sb.toString();
            }
            
        }
        
    }