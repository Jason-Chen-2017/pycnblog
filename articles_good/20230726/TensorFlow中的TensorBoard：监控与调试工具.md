
作者：禅与计算机程序设计艺术                    

# 1.简介
         
TensorFlow是一个开源的机器学习框架，它提供了一个用于构建、训练和部署深度学习模型的工具集合。随着深度学习的应用不断扩张，TensorFlow也在不断地演进，加入了诸如自动求梯度、模型可视化等新特性。
为了更好地管理、监控和优化深度学习模型的训练过程和效果，TensorFlow提供了TensorBoard这个工具，它是一款开源的可视化工具，可以帮助开发者更直观地理解和分析神经网络的训练过程，包括损失函数值变化图、权重参数分布情况、激活函数曲线等。同时，TensorBoard还提供了不同维度的指标对比、数据样本可视化等功能，助力开发者提升模型的理解能力，缩短错误定位和排查时间。
这篇文章将详细介绍TensorFlow中TensorBoard的主要功能和用法，并结合实践案例，展示如何利用TensorBoard进行深度学习模型的调试、监控与优化。
# 2.背景介绍
## 2.1 什么是TensorBoard？
TensorBoard是TensorFlow的一个可视化工具，它允许用户在不访问源码的情况下，查看深度学习模型的训练过程及效果。它能够记录训练中的所有数据，包括：损失函数值、权重参数、激活函数等。通过图表、直方图、饼状图等方式呈现出来的数据，可以清晰地看到模型训练的总体情况、误差的分布情况、各个变量之间的关系、模型效果的改善程度等。此外，TensorBoard还支持多种数据源的比较和分析，让用户更方便地发现模型的瓶颈所在。
## 2.2 为何要使用TensorBoard？
使用TensorBoard有如下几点优点：

1. 提供了一套完整的可视化工具链，覆盖训练、评估和分析三个环节。可以直观地了解到深度学习模型的训练、优化过程、精度、模型结构等信息。
2. 可以很容易地看到各种指标（loss、accuracy、learning rate）随着时间的变化情况，帮助开发者快速追踪模型的优化过程。
3. 支持多种数据源的比较和分析，使得开发者可以对比不同任务的结果，找到最佳模型或模型瓶颈。
4. 适用于不同的深度学习框架，可直接集成至TensorFlow内核，无需额外安装。
5. 对实施者来说，更加直观、便利地调试和优化模型，节省了大量的时间。

## 2.3 环境准备
本文中使用的Tensorflow版本为`v1.8.0`，如果您本地没有安装，可以通过如下命令进行安装：
```bash
pip install tensorflow==1.8.0
```
另外，本文中的代码运行需要TensorboardX库，可通过如下命令安装：
```bash
pip install tensorboardx==1.2
```
# 3.基本概念术语说明
## 3.1 Session、Graph和Variable
TensorFlow是一个基于数据流图(data flow graph)的计算框架。其中，Session负责执行图中的运算，Graph则描述了计算过程，保存了计算图的各种对象，包括节点（node）、边（edge）和操作（op）。Variable表示一个可修改的、持久化的状态值。它们之间有一些重要的关联：

1. Graph：所有的TensorFlow计算都需要有一个计算图，该图定义了输入输出以及操作的执行顺序。
2. Variable：Graph中的Variable的值可以被训练过程改变，所以Variable非常适合存储模型参数、运行过程中发生变化的值等。
3. Session：Session负责执行图中的操作，确保执行前的所有准备工作已经完成。

## 3.2 计算图
TensorFlow计算图由节点、边、操作组成。节点表示数学运算符或者变量的取值；边表示数据流动的方向；操作则指定具体的数学计算方法。如下图所示，计算图通常由三个阶段组成：定义阶段（graph construction phase），计算阶段（execution phase），和后处理阶段（post-processing phase）。

![image.png](attachment:image.png)

定义阶段（graph construction phase）：在这里，会先定义一个计算图，然后创建一些节点，这些节点可以根据TensorFlow的API调用。每个节点都有对应的名称，当调用TensorFlow API时，就可以通过名字来引用这个节点。定义好的计算图会作为后续计算的依据，这样就构成了TensorFlow模型。

计算阶段（execution phase）：在这一阶段，会按照图中定义的执行顺序，把操作的输入从依赖节点得到，计算出相应的输出，再将结果传递给后续节点。

后处理阶段（post-processing phase）：可以在这一阶段进行一些模型后处理操作，比如保存模型、读取模型等。

TensorBoard也是使用计算图来呈现数据的。TensorBoard中的图形都是建立在TensorFlow的计算图之上的，因此在TensorBoard中查看数据流图的过程就是查看TensorFlow模型训练时的计算图。

## 3.3 概率分布
概率分布是统计学中的一个概念，描述随机事件可能出现的情况。一般来说，一个概率分布由两个部分组成，一个是随机变量（random variable），另一个是概率密度函数（probability density function）。例如，假设X是一个抛硬币的过程，那么X的分布就是指每次抛掷之后正面朝上的概率。概率密度函数的表达式形式如下：

![image.png](attachment:image.png)

这里，f(x)是概率密度函数，x是随机变量，P(X=x)是概率质量函数（PMF）。从直觉上看，概率质量函数就是某个值x出现的概率。

TensorFlow中可以直接计算随机变量的概率密度函数，也可以生成随机变量的样本。

# 4.核心算法原理和具体操作步骤以及数学公式讲解
## 4.1 数据可视化
TensorFlow自带的数据可视化工具称为TensorBoard，其提供了丰富的数据可视化功能，其中包括数据分布、图像可视化、折线图等。这里，只对数据分布、图像可视化做简单介绍。

### 4.1.1 数据分布可视化
数据分布可视化最简单的做法是在TensorBoard中绘制不同标签对应的数据集的直方图。具体步骤如下：

1. 在训练模型的时候，通过创建SummaryWriter对象，将数据写入日志文件。
2. 通过tf.summary.histogram()函数，将需要可视化的变量传入日志文件，这样TensorBoard就会在可视化页面显示变量的分布。
3. 在可视化页面，选择想要可视化的标签，点击刷新按钮即可。

例子：

```python
import tensorflow as tf
from datetime import datetime

# 创建一个日志目录
LOG_DIR = "logs/{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))

# 创建一个计算图
g = tf.Graph()
with g.as_default():
    # 模型参数
    W = tf.get_variable("W", shape=[2, 1])

    # 定义损失函数
    y = tf.matmul(X, W)
    loss = tf.reduce_mean((y - Y)**2)

    # 添加优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss)

    # 创建Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 将数据写入日志文件
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    # 生成一些样本数据
    X_samples = np.array([[1., 2.], [3., 4.], [5., 6.]])
    Y_samples = np.array([[-3.], [-7.], [-11.]])
    
    # 定义Summary
    summary = tf.summary.histogram('Weights', W)

    for i in range(100):
        _, l, s = sess.run([train_op, loss, summary], feed_dict={X: X_samples, Y: Y_samples})

        if (i+1)%10 == 0:
            print('Step %d: Loss %.4f' %(i+1, l))

            # 将Summary写入日志文件
            writer.add_summary(s, i)

    # 关闭日志文件
    writer.close()
```

以上示例代码首先创建一个日志目录，接着定义了一个计算图，并添加了一个变量W。然后生成一些样本数据，定义了损失函数和优化器，初始化了变量。最后创建了一个Session，并将数据写入日志文件，并在训练过程中每十步生成Summary并写入日志文件。

运行以上代码后，打开浏览器，在地址栏输入"http://localhost:6006/#graphs"，进入TensorBoard的可视化页面。选择要可视化的标签"Weights"，点击刷新按钮，就可以看到W的分布。如下图所示：

![image.png](attachment:image.png)

左侧为训练数据集中每类样本个数的直方图，右侧为模型参数W的直方图。由于X和Y之间存在一定联系，因此两幅直方图应该呈现出相关性。

### 4.1.2 图像可视化
图像可视化常用的手段是绘制一系列图片，代表模型的特征。以下介绍两种常用的图像可视化方法：

#### 4.1.2.1 可视化权重
对于卷积神经网络来说，可视化权重是一种常见的图像可视化方法。具体步骤如下：

1. 使用卷积层、全连接层等节点的kernel参数初始化权重。
2. 运行模型进行训练，并且将卷积层、全连接层的参数保存下来。
3. 从日志文件中加载参数。
4. 用matplotlib画出权重矩阵。

例子：

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime

# 创建一个日志目录
LOG_DIR = "logs/{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))

# 创建一个计算图
g = tf.Graph()
with g.as_default():
    # 模型参数
    conv1_weights = tf.get_variable("conv1/weights", initializer=tf.truncated_normal([3, 3, 3, 32], stddev=0.1))
    conv2_weights = tf.get_variable("conv2/weights", initializer=tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
    fc1_weights = tf.get_variable("fc1/weights", initializer=tf.truncated_normal([9216, 1024], stddev=0.1))
    fc2_weights = tf.get_variable("fc2/weights", initializer=tf.truncated_normal([1024, 10], stddev=0.1))

    # 添加一些卷积层和全连接层
    x = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1), name="input")
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, conv1_weights, strides=[1, 1, 1, 1], padding='SAME'))
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME'))
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 9216])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, fc1_weights))
    logits = tf.matmul(h_fc1, fc2_weights)

    # 定义损失函数
    labels = tf.placeholder(dtype=tf.int64, shape=(None,), name="labels")
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    # 添加优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss)

    # 创建Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 将数据写入日志文件
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    # 生成一些样本数据
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    X_train, Y_train = mnist.train.images, mnist.train.labels
    X_val, Y_val = mnist.validation.images, mnist.validation.labels

    # 执行训练
    batch_size = 128
    max_steps = 500
    for step in range(max_steps):
        offset = (step * batch_size) % (Y_train.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size)]
        batch_labels = Y_train[offset:(offset + batch_size)].argmax(axis=1)
        
        _, l, predictions = sess.run([train_op, loss, logits],
                                     feed_dict={x: batch_data, labels: batch_labels})

        if (step+1)%100 == 0 or step == 0:
            acc = accuracy(predictions, batch_labels)
            val_acc = accuracy(sess.run(logits, {x: X_val}), Y_val.argmax(axis=1))
            print('Step %d of %d: loss=%.4f; training accuracy=%.4f; validation accuracy=%.4f'%
                  (step+1, max_steps, l, acc, val_acc))
            
    # 保存权重
    saver = tf.train.Saver()
    save_path = saver.save(sess,'my_model.ckpt')

    # 关闭日志文件
    writer.close()
    
def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets('MNIST_data/', one_hot=True)

def plot_conv_weights(weights, figsize=(8, 8)):
    fig = plt.figure(figsize=figsize)
    num_filters = weights.shape[3]
    w = weights.shape[0]
    h = weights.shape[1]
    for i in range(num_filters):
        ax = fig.add_subplot(np.ceil(num_filters / 8.), 8, i+1)
        img = weights[:, :, 0, i]
        ax.imshow(img, cmap='gray')
        ax.set_title('Filter {}'.format(i))
        ax.axis('off')
        
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])
```

以上示例代码首先创建一个日志目录，并创建一个计算图，定义了四个卷积层和两个全连接层，并添加了损失函数和优化器。生成一些样本数据，然后初始化了变量。在训练过程中，每隔100步打印一次训练集准确率和验证集准确率，并且保存权重。

运行以上代码后，打开浏览器，在地址栏输入"http://localhost:6006/#graphs"，进入TensorBoard的可视化页面。点击"GRAPHS"菜单下的"RUN GRAPH"按钮，就可以看到整个计算图的结构。

选择一个卷积层或全连接层，然后点击"METADATA"标签下的"DATA CONTAINS NAMED TENSORS"，就可以看到这个层的权重参数。双击左侧列表中的一个参数，就可以看到相应的参数值。

为了方便地查看权重矩阵，这里增加了一个plot_conv_weights()函数。该函数接受权重矩阵参数weights，并绘制出8x8个过滤器。执行完训练之后，将权重矩阵存入日志文件中，并按如下方式打开日志文件：

```python
# 载入日志文件
meta_file = os.path.join(LOG_DIR,'meta')
reader = tf.train.NewCheckpointReader(os.path.join(save_path[:-5], meta_file))
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    if '/weights' in key and len(key.split('/')) < 3:
        param_name = key[:key.find(':')]
        with tf.variable_scope('', reuse=True):
            weight_tensor = tf.get_variable(param_name)
            weight_value = reader.get_tensor(key)
        plot_conv_weights(weight_value)
```

以上代码遍历了日志文件中的所有参数，并选取卷积层中的权重参数。然后载入日志文件，获取到相应的权重参数，并绘制出权重矩阵。可以看到绘出的权重矩阵类似于传统卷积神经网络中的可视化效果。

![image.png](attachment:image.png)

#### 4.1.2.2 可视化预测结果
在深度学习模型中，预测结果往往具有高度的非线性性，因此，可以使用平面投影或是可视化的技术来对其进行解释。具体步骤如下：

1. 使用模型进行预测。
2. 将预测结果映射到某个二维空间中。
3. 用matplotlib或其他可视化库绘制图形。

例子：

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

# 创建一个计算图
g = tf.Graph()
with g.as_default():
    # 模型参数
    W = tf.get_variable("W", shape=[2, 1])
    b = tf.get_variable("b", shape=[1])

    # 定义输入、模型结构和输出
    X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    y_pred = tf.add(tf.matmul(X, W), b)

    # 生成一些样本数据
    X_samples = [[1, 2], [3, 4], [5, 6]]
    Y_samples = [3, 7, 11]

    # 初始化变量
    init = tf.global_variables_initializer()

    # 创建Session
    sess = tf.Session()
    sess.run(init)

    # 将数据可视化
    tsne = TSNE(n_components=2, random_state=0)
    results = tsne.fit_transform(results)
    colors = ['r' if result > 5 else ('g' if result > 0 else 'b') for result in results[:, 1]]
    plt.scatter(results[:, 0], results[:, 1], c=colors, alpha=0.5)
    plt.show()
```

以上代码定义了一个计算图，包含了一个输入、一个模型结构和一个输出。然后生成一些样本数据，初始化了变量，创建了一个Session。为了对预测结果进行可视化，这里引入了一个TSNE变换算法，将结果转换到一个2维空间中，并用颜色编码区分正、中、负的预测结果。

运行以上代码后，可以在屏幕上看到对预测结果的可视化效果。如下图所示：

![image.png](attachment:image.png)

颜色越深，意味着预测结果越接近于零，越暗色，意味着预测结果越接近于最大值。预测结果对离散值、连续值、概率值均有效。

# 5.具体代码实例和解释说明
## 5.1 卷积神经网络可视化
以下示例代码展示了如何利用TensorBoard可视化卷积神经网络。

```python
import tensorflow as tf
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data

# 创建一个日志目录
LOG_DIR = "logs/{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))

# 载入数据
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 创建一个计算图
g = tf.Graph()
with g.as_default():
    # 模型参数
    x = tf.placeholder(tf.float32, shape=[None, 784], name="Input")
    keep_prob = tf.placeholder(tf.float32, name="KeepProb")
    W1 = tf.get_variable("W1", shape=[784, 512],
                         initializer=tf.contrib.layers.variance_scaling_initializer())
    B1 = tf.Variable(tf.zeros(shape=[512]))
    L1 = tf.nn.relu(tf.matmul(x, W1) + B1)
    D1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    W2 = tf.get_variable("W2", shape=[512, 256],
                         initializer=tf.contrib.layers.variance_scaling_initializer())
    B2 = tf.Variable(tf.zeros(shape=[256]))
    L2 = tf.nn.relu(tf.matmul(D1, W2) + B2)
    D2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    W3 = tf.get_variable("W3", shape=[256, 10],
                         initializer=tf.contrib.layers.variance_scaling_initializer())
    B3 = tf.Variable(tf.zeros(shape=[10]))
    y_pred = tf.nn.softmax(tf.matmul(D2, W3) + B3)

    # 定义损失函数
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="Output")
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true*tf.log(y_pred), reduction_indices=[1]),
                                   name="CrossEntropy")

    # 添加优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    gradients, variables = zip(*optimizer.compute_gradients(cross_entropy))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables))

    # 获取TensorBoard的writer对象
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    # 执行训练
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    batch_size = 100
    n_batches = int(mnist.train.num_examples / batch_size)
    n_epochs = 10

    for epoch in range(n_epochs):
        total_cost = 0
        for i in range(n_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost = sess.run([train_op, cross_entropy],
                               feed_dict={x: batch_xs,
                                          y_true: batch_ys,
                                          keep_prob: 0.5})
            total_cost += cost / n_batches

        print("Epoch:", (epoch+1), "Cost =", "{:.3f}".format(total_cost))

    # 将权重可视化
    filters = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Conv1')
    feature_maps = []
    image = tf.constant(mnist.test.images[0].reshape((-1, 28, 28, 1)))
    for filter in filters:
        activations = tf.abs(tf.nn.conv2d(image, filter, strides=[1, 1, 1, 1], padding='VALID'))
        feature_maps.append(activations.eval(session=sess)[0, :, :, :])

    layer1_outputs = sess.run(L1, feed_dict={x: mnist.test.images, keep_prob: 1.0})
    writer.add_summary(create_feature_maps_summary("Layer 1", feature_maps), 0)

    layer2_outputs = sess.run(L2, feed_dict={x: mnist.test.images, keep_prob: 1.0})
    writer.add_summary(create_feature_maps_summary("Layer 2", feature_maps), 1)

    final_output = sess.run(y_pred, feed_dict={x: mnist.test.images, keep_prob: 1.0})
    writer.add_summary(create_prediction_summary("Final Predictions", final_output), 2)

    sess.close()
    writer.close()


def create_feature_maps_summary(layer_name, feature_maps):
    """
    Creates a Summary object for visualizing the convolutional feature maps using grid cells.

    Args:
      layer_name: The name of the layer to visualize.
      feature_maps: A list of NumPy arrays containing the activation values per filter.

    Returns:
      A Summary object that contains an image summarizing the feature maps in a grid cell layout.
    """
    height, width = 5, 5  # Number of cells per row and column in the grid.
    scale = 0.5  # Scale factor for each individual feature map.
    channels = feature_maps[0].shape[-1]  # Number of channels per feature map.
    rows = min(height, len(feature_maps))  # Ensure we don't have more rows than filters.
    cols = min(width, len(feature_maps))  # Ensure we don't have more columns than filters.

    # Create a grayscale canvas for all feature maps combined.
    canvas = np.ones((rows * feature_maps[0].shape[0] // height * scale,
                      cols * feature_maps[0].shape[1] // width * scale,
                      1)).astype(np.float32)

    # Add each feature map to its corresponding grid cell.
    for i, fmap in enumerate(sorted(feature_maps, reverse=False)):
        r, c = divmod(i, cols)  # Calculate the position of the current filter in the grid.
        canvas[(r * fmap.shape[0] // rows) * scale:(r * fmap.shape[0] // rows + fmap.shape[0]) * scale:,
               (c * fmap.shape[1] // cols) * scale:(c * fmap.shape[1] // cols + fmap.shape[1]) * scale:] *= \
                np.expand_dims(fmap, axis=-1).repeat(channels, axis=-1)

    # Convert the grayscale canvas to RGB format suitable for visualization with TensorBoard.
    image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    # Construct a Summary protobuf object holding the image.
    summary = tf.Summary(value=[tf.Summary.Value(tag='%s Feature Maps' % layer_name,
                                                 image=tf.Summary.Image(encoded_image_string=cv2.imencode('.jpg',
                                                                                                          image)[1].tostring()))
                                ])
    return summary


def create_prediction_summary(title, predictions):
    """
    Creates a Summary object for visualizing classification predictions on MNIST test data.

    Args:
      title: The title of the summary table.
      predictions: A NumPy array containing the predicted class probabilities per example.

    Returns:
      A Summary object that contains a scalar summarization of the top-1 prediction accuracy and a confusion matrix.
    """
    labels = sorted(['%d (%.2f%%)' % (label, pred[label]*100) for label, pred in
                     enumerate(predictions)])
    data = [['Truth'] + labels, [''] + ['---']*(len(labels)-1)]
    correct = 0
    for truth, pred in zip(mnist.test.labels, predictions):
        index = np.argmax(truth)
        data.append([str(index)] + ['%.2f%%' % (p*100) for p in pred])
        if np.argmax(pred) == index:
            correct += 1

    accuracy = '%.2f%%' % ((correct / float(len(mnist.test.labels))) * 100)
    conf_matrix = tabulate.tabulate(data, headers='firstrow')
    summary = tf.Summary(value=[tf.Summary.Value(tag='Top-1 Accuracy', simple_value=float(accuracy)),
                                tf.Summary.Value(tag='%s Confusion Matrix' % title,
                                                 simple_value=0.0)])
    summary.value[1].metadata.plugin_data.content = str(conf_matrix)
    return summary
```

以上代码首先创建一个日志目录，然后载入MNIST数据集。创建了一个计算图，并添加了三个卷积层，每个层包含两个池化层，分别用于降低图片尺寸和减少参数数量。模型还有三个全连接层，用于分类任务。

接着定义了损失函数和优化器，并使用了TensorBoard的writer对象。执行训练，每隔几个小时保存一次模型，并将权重可视化。

自定义了一个create_feature_maps_summary()函数，用于可视化卷积层的权重。该函数接受卷积层的名字和一个特征图列表，并返回一个TensorBoard的Summary对象，用于可视化。

自定义了一个create_prediction_summary()函数，用于可视化模型的预测结果。该函数接受预测结果和标题，并返回一个TensorBoard的Summary对象，用于可视化。

最终，执行完训练后，可以通过浏览器查看TensorBoard的可视化结果。选择"GRAPHS"菜单下的"RUN GRAPH"按钮，可以看到计算图的结构。

选择"SLICES"菜单下的"Feature Maps"，可以看到卷积层的权重。如下图所示：

![image.png](attachment:image.png)

选择"IMAGE"菜单下的"Predictions"，可以看到模型的预测结果。如下图所示：

![image.png](attachment:image.png)

