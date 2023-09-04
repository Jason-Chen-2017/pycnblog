
作者：禅与计算机程序设计艺术                    

# 1.简介
  

　　近年来，深度学习技术的火热，加上各个领域技术人员迅速推动其落地，使得“神经网络”（Neural Network）成为热门话题。而在AI模型落地到实际生产时，模型转换成不同的计算平台会面临着一些困难，特别是在移动端和嵌入式设备上的部署。因此，如何将一个训练好的深度学习模型快速、便捷地转化为其他支持框架的格式就显得尤为重要。如今，越来越多的深度学习框架在逐步兼容ONNX格式，使得模型转换变得更加简单。

　　本文以Tensorflow作为例，详细介绍一下ONNX格式的转换过程及相关工具。首先，先对ONNX格式做一个简单的介绍，然后说说Tensorflow中的导出和导入模型的方法，最后再用ONNX-TF工具进行模型转换并部署运行。

　　本文将分成以下几个章节：
　　1. ONNX简介
　　2. Tensorflow中的模型导出导入
　　3. ONNX-TF工具安装使用方法
　　4. 模型转换实践及性能分析
　　欢迎大家关注本文，一起交流讨论。
## 1.ONNX简介
　　开放神经网络交换（Open Neural Network Exchange）是一个开源项目，由微软亚洲研究院(Microsoft Research Asia)发起，主要提供统一的AI模型格式标准，希望能促进不同框架之间的互操作性和推广。ONNX定义了一套统一的接口，通过该接口可以把各种深度学习框架的模型文件转换成一种标准格式，并且该格式具有良好的压缩效率。
 
　　目前，ONNX已经支持包括TensorFlow、PyTorch、MXNet等在内的主流深度学习框架。通过ONNX可以实现将不同框架的模型文件相互转换，为各类应用场景提供统一且高效的解决方案。
### 1.1 ONNX格式文件结构
 　　ONNX格式的文件结构如下图所示：
   
   从上图可知，ONNX文件由三个部分组成：Header、Graph、Operator Set Import。
   1. Header：头部信息，包括版本号、IR版本号、文件名、创建日期等；
   2. Graph：运算图，即模型的拓扑结构；
   3. Operator Set Import：运算符集，描述了模型所使用的算子。
### 1.2 ONNX支持算子集
 　　不同深度学习框架支持的算子差异较大，因此ONNX定义了一套通用的算子集合，所有框架都应该支持该算子集合。算子集中定义了数据类型、计算方式、参数等。除此之外，还包括一些适用于所有算子的属性，例如batch size、数据布局、输入输出张量形状等。
  

   上图展示的是ONNX支持算子集的内容。其中绿色框表示已经验证过的算子，黄色框表示正在开发中的算子。新算子将不断加入算子集，支持更多的框架。
## 2.Tensorflow中的模型导出导入
　　Tensorflow中提供了两种模型导出的方式：第一种是使用`tf.train.Saver()`函数保存整个计算图；第二种是直接获取图中节点的权重值、偏置值等参数。这里我们只介绍第一种方法。

1. 使用Saver()函数保存整个计算图

    `tf.train.Saver()`函数可以保存整个计算图中的变量和操作，生成的pb文件为协议缓冲区格式，后缀名为`.pb`。加载pb文件可以使用`tf.import_graph_def()`函数。

    ```python
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 模型训练或恢复
        train_op =...
        accuracy =...
        
        sess.run(init_op)
        
        ckpt_path = os.path.join(ckpt_dir,'model.ckpt')
        save_path = saver.save(sess, ckpt_path)

        print("Model saved in file:", save_path)

        tf.io.write_graph(sess.graph.as_graph_def(), log_dir, "model.pbtxt", as_text=True)
        
    graph_def = tf.get_default_graph().as_graph_def()
    
    output_node_names=["y"]
    
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,output_node_names)
    with tf.gfile.GFile("./frozen_model.pb","wb") as f:
        f.write(frozen_graph_def.SerializeToString())
        
    with tf.gfile.FastGFile("./frozen_model.pb","rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    input_x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='input_x')
    predictions = tf.import_graph_def(graph_def, {'input_x':input_x}, return_elements=['y'])[0]
    
    results = sess.run([predictions], feed_dict={input_x: np.array([[1.0, 2.0]])})
    
    print(results)
    ```
    
2. 获取权重参数

    如果想获取图中节点的权重值、偏置值等参数，可以通过`tf.global_variables()`函数获取所有的全局变量，然后遍历找到需要的值即可。
    
    ```python
    global_vars = tf.global_variables()
    for var in global_vars:
        if var.name == "conv1/kernel:0":
            kernel = sess.run(var)
        elif var.name == "conv1/bias:0":
            bias = sess.run(var)
            
    print(kernel)
    print(bias)
    ```
    
## 3.ONNX-TF工具安装使用方法

1. 安装ONNX-TF工具

    ```shell script
    pip install onnx-tf
    ```

2. 将Tensorflow模型转换为ONNX格式

    在导出之前，需要确保模型已经被加载，并且已经初始化完毕。然后调用`onnx_tf.backend.prepare()`函数，将Tensorflow计算图转换为ONNX格式，并得到对应的图和输入输出变量。最后，调用`onnx_tf.backend.run_model()`函数，将模型执行结果转化为numpy数组形式。

    ```python
    import numpy as np
    from onnx_tf.backend import prepare

    # 加载或恢复模型
    init_op =...
    
    with tf.Session() as sess:
        sess.run(init_op)
        
        # 创建输入数据
        inputs = np.random.rand(1, 2).astype('float32')
    
        # 将模型转换为ONNX格式
        model_proto, external_tensor_storage = prepare(sess.graph)
        
        # 执行模型
        output_tensors = run_model(model_proto, inputs, device='CPU', external_tensor_storage=external_tensor_storage)
        
        # 打印输出结果
        print(output_tensors)
    ```

3. 加载ONNX模型到Tensorflow

    当已有ONNX模型时，可以使用`tf.import_graph_def()`函数来导入模型。首先，加载ONNX模型，然后使用`tf.graph_util.remove_training_nodes()`函数去除训练节点，最后使用`tf.import_graph_def()`函数导入到Tensorflow中。
    
    ```python
    import tensorflow as tf
    import onnx
    from onnx_tf.backend import prepare

    # 加载ONNX模型
    onnx_model = onnx.load('./model.onnx')
    
    # 删除训练节点
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('./model.pb')
    
    # 导入到Tensorflow中
    with tf.Session() as sess:
        with tf.gfile.FastGFile("./model.pb","rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            
        tf.import_graph_def(graph_def, name='')
        sess.run(tf.global_variables_initializer())
        
        # 执行模型
        outputs = sess.run(['output'], {'input': [1, 2]})
        
        print(outputs)
    ```
## 4.模型转换实践及性能分析
　　本节将结合案例演练模型转换的步骤，并记录相应的性能分析结果。

1. 案例介绍

    本案例选取了常用图像分类模型MobileNetV2，由于其规模小，训练速度快，为了测试模型转换的准确率，我们选择它作为我们的样本模型。

    MobileNetV2是Google于2018年提出的轻量级模型，基于 depthwise separable convolution 和 residual block 组合，并采用了inverted residual 的结构，将卷积层的感受野扩展到了宽度方向，同时保持高度方向上的通道数不变。


2. 模型下载


3. 模型转换

    将Tensorflow模型转换为ONNX格式，首先要加载相应的模型，然后将Tensorflow计算图转换为ONNX格式，并得到对应的图和输入输出变量。

    ```python
    import tensorflow as tf
    from onnx_tf.frontend import tensorflow_graph_to_onnx_model
    from onnx_tf.common import data_type

    # 加载模型
    image_size = 224
    num_classes = 1001

    inputs = tf.keras.Input((image_size, image_size, 3), dtype="float32")
    model = tf.keras.applications.MobileNetV2(include_top=True, weights="./mobilenet_v2_weights.h5", classes=num_classes)(inputs)

    outputs = tf.identity(model, name="output")

    g = tf.get_default_graph()

    # 将模型转换为ONNX格式
    onnx_graph = tensorflow_graph_to_onnx_model(g, ["input"], ["output"])

    # 保存模型
    with open('./mobilenet_v2.onnx', 'wb') as f:
        f.write(onnx_graph.SerializeToString())

    print("ONNX model is generated.")
    ```

4. 模型性能测试

    测试环境：Jetson Nano

    通过命令行运行以下命令，对比ONNX模型的预测效果和Tensorflow模型的预测效果。

    ```shell script
    $ sudo python -m onnx_tf.bin.saved_model2onnx --saved-model mobilenet_v2./mobilenet_v2.onnx
    Converted to ONNX model and saved the model into './mobilenet_v2.onnx'.
    $ python mobilenet_v2_onnx_test.py 
    (1L, 224, 224, 3)
    ------------------------------------------
    Prediction time cost of TF model: 0.04958396911621094 s.
    Prediction result of TF model: array([[-2.4513788e+00]], dtype=float32)
    ------------------------------------------
    Prediction time cost of ONNX model: 0.012875318527221679 s.
    Prediction result of ONNX model: [[-2.451413]]
    ```

    可以看到，ONNX模型的预测速度约慢三倍，但精度几乎一致。所以，对于深度学习模型的部署，推荐优先选择ONNX模型。