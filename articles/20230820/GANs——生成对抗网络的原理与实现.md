
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是GAN？
GAN（Generative Adversarial Networks）是由Ian Goodfellow等人于2014年提出的一种基于生成模型的深度学习方法，可以用于生成图像、音频或文本数据等多种模态的数据，并将其作为输入送入到判别器中进行评估，使得生成的结果逼真到达到指定的质量水平。GAN被广泛地应用在图像领域、视频编辑领域、游戏生成领域、自然语言处理领域、以及其他各个领域。
## 为什么要用GAN？
人类一直在努力寻找解决机器学习问题的新途径，而GAN是一个很好的开端。传统上，生成模型主要侧重于分析数据之间的关系，设计隐变量的映射函数，然后优化目标函数使得生成模型在特定任务上取得较好效果。而GAN是一种新型的机器学习方法，它利用了生成模型的特性，直接训练出生成数据的能力。也就是说，GAN可以理解为生成模型+判别器的组合，通过最大化判别器的损失来生成数据，同时也需要训练生成模型，从而促进生成数据的能力提高。这是因为生成模型所关注的问题是生成假数据的能力，而判别器所关注的问题则是区分真假数据的能力。因此，当两者能够共同完成这个过程时，GAN就可以在无监督学习、分类、回归、序列建模等众多领域得到应用。
## 生成模型与判别模型
为了更加清晰地阐述GAN的结构及其特点，我们先引入两个非常重要的概念，即生成模型G和判别模型D。生成模型G的作用就是根据给定的随机噪声z生成一组看起来像真实样本的输出x_fake，而判别模型D的作用则是通过已知样本x判断它们是真实样本还是虚假样本，并输出一个置信度y(x)表示它们属于第几类。
其中，x_real是真实样本，x_fake是生成的假样本，y_true是真实样本标签，y_fake是生成样本的标签。
假设我们有一个关于x的联合分布P(x)，既包括生成模型G生成的假样本，也包括真实样本x_real。在这种情况下，我们希望通过训练G，使得它的输出x_fake分布尽可能接近真实样本分布P(x)。而同时，我们也希望通过训练D，使得它的输出尽可能接近真实样本标签y_true和假样本标签y_fake的概率。我们可以通过以下的损失函数来衡量生成模型G和判别模型D的性能：
其中，L_adv是判别模型D的损失函数，L_con是生成模型G的损失函数。我们希望L_adv越小越好，意味着判别模型D在区分真假样本上的能力越强；而L_con越小越好，意味着生成模型G生成的假样本越真实越好。
# 2. 相关术语
## 深度学习
深度学习（deep learning）是指通过多层次的神经网络系统学习的一种机器学习技术。该方法的关键是由许多相互联系的简单单元组成的复杂神经网络，这些单元能够逐层抽象、分析和反映出复杂的数据模式。深度学习的研究者们通过不断增强神经网络的规模、深度、宽度、连接方式等参数，成功构建出能够有效处理各种任务的模型，并取得了极大的成功。
## 生成模型
生成模型是在给定某些条件下，根据定义的概率分布生成随机输出的模型。最常用的生成模型是二元线性回归模型，比如生成图片，首先输入一组参数，再生成图像的像素值。另一个常见的生成模型是循环神经网络模型，比如机器翻译、文本摘要、语音合成等任务。
## 对抗学习
对抗学习是深度学习的一个重要方向，其核心思想是通过训练两个模型，一个生成模型G，另一个判别模型D，让它们之间能够互相竞争，进而达到更好的模型性能。在GAN模型中，生成模型G的目标是生成尽可能真实的数据，而判别模型D的目标则是能够准确地识别出输入数据的真伪。通过极小化生成模型与判别模型之间的差距，即生成模型欺骗判别模型，最终达到学习到数据的本质目的。
# 3. 核心算法原理
## 概念
本节首先简要介绍一下GAN的基本概念及其核心算法，之后再详细介绍具体操作步骤和数学公式。
### 模型结构
GAN由生成模型G和判别模型D两部分组成，它们分别生成假数据x_fake和真实数据x_real，并由输入x以及生成数据与真实数据之间的区别信息来推断生成数据是否真实。其结构如下图所示：
生成模型G由一个编码器（Encoder）和一个解码器（Decoder）组成，编码器把潜在空间z映射到数据空间x_fake，解码器把潜在空间z映射回来。判别模型D由一个神经网络判别器（Discriminator）组成，它将输入数据x映射到一个概率值，表明数据来自真实样本的概率。
### 损失函数
GAN的损失函数包括生成模型G和判别模型D的损失。G的目标是使得生成的数据能够被判别模型D分辨出来，所以它的损失函数应该使得判别模型D的输出接近于1，即损失越低越好；而D的目标是尽可能推理真实数据，所以它的损失函数应该使得判别模型D的输出接近于0，即损失越高越好。基于此，GAN的损失函数一般包括以下三项：
* 判别模型D的损失函数：
$$\mathcal{L}_{\text {dis}} \left(\theta_{d}, \theta_{g}\right)=\mathbb{E}_{x^{(i)} \sim P_{\text {data}}}[\log D\left(x^{i} ; \theta_{d}\right)]+\mathbb{E}_{z^{(j)} \sim p(z)}\left[\log (1-D\left(G\left(z^{(j)} ; \theta_{g}\right) ; \theta_{d}\right))\right]$$
* 生成模型G的损失函数：
$$\mathcal{L}_{\text {gen}} \left(\theta_{d}, \theta_{g}\right)=\mathbb{E}_{z^{(j)} \sim p(z)}\left[\log D\left(G\left(z^{(j)} ; \theta_{g}\right) ; \theta_{d}\right)\right]$$
* 整体的损失函数：
$$\mathcal{L}_{GAN} \left(\theta_{d}, \theta_{g}\right)=\frac{1}{m} \sum_{i=1}^{m} \left[\left[D\left(x^{i} ; \theta_{d}\right)-1\right]+\left[\log (1-D\left(G\left(z^{(j+n/m)} ; \theta_{g}\right) ; \theta_{d}\right))\right]\right]$$
这里，$x^{(i)}, i=1,\cdots, m$ 是从数据集中随机采样的真实数据，$z^{(j)}, j=1,\cdots, n$ 是服从标准正态分布的潜在向量，$\theta_{d}$ 和 $\theta_{g}$ 分别是判别模型D和生成模型G的参数，$m$ 和 $n$ 分别是训练数据和潜在向量的数量，$D$ 和 $G$ 分别是判别模型D和生成模型G的前向传播运算。
### 优化算法
生成模型G和判别模型D都可以使用标准的梯度下降法来训练，但由于存在两个模型的博弈关系，所以不能同时进行训练。而是采用一种联合优化算法，即同时更新两个模型的参数。常用的联合优化算法包括梯度下降法、Adam、RMSProp等。在训练过程中，判别模型D的权重应当不断适当减少，而生成模型G的权重应当不断增加，这可以使得生成模型G产生更多更真实的数据，直至收敛到最优解。
## 生成模型
生成模型G的目的是生成尽可能真实的数据，所以其训练过程中的loss function应具有高熵的性质，即希望模型能够生成的数据分布尽可能均匀，并且生成的数据分布可以逼近原始数据分布。具体的，对于二维数据，我们可以考虑使用标准正太分布或者高斯分布来拟合生成的数据分布。除此之外，还可以引入限制条件来约束生成数据，比如限制数据范围或者要求满足数据之间的统计特性等。最后，可以尝试使用GAN来训练数据生成模型，从而获得真实数据的近似分布。
### 编码器与解码器
编码器用来将输入数据转换成潜在空间的表示，解码器用来将潜在空间的数据转换回数据空间。其结构如下图所示：
GAN中，生成模型G由一个编码器（Encoder）和一个解码器（Decoder）组成，编码器将输入数据编码成潜在空间的表示z，解码器将z恢复到数据空间，即生成新的样本x_fake。编码器与解码器共享权重，同时将编码器输出与判别模型D的输出合并，即将编码器输出的特征与真假样本标签合并一起送入到判别模型D中进行判别。这样做的目的是希望生成模型G在训练过程中生成的数据与真实数据有着更强的一致性。
### 生成器网络
生成器网络将潜在空间的表示z转换成数据空间的样本x。其结构如下图所示：
在GAN中，生成模型G由一个编码器（Encoder）和一个解码器（Decoder）组成，解码器将潜在空间的表示z转化成数据空间的样本x。解码器由几个全连接层（FC）构成，每层后面紧跟一个ReLU激活函数，最后一层是一个sigmoid函数，可以将数据空间的样本x缩放到0~1之间。生成器网络的目标是生成尽可能真实的数据，所以生成器应该产生符合高斯分布的数据。
#### WGAN
WGAN（Wasserstein Generative Adversarial Network）是GAN的改进版本。相比于GAN，WGAN的主要区别是使用Wasserstein距离来衡量数据之间的差异，而不是直接衡量判别模型D的预测值和实际标签之间的差异。WGAN可以提供更稳健的训练，并且可以保证生成器产生的样本更接近真实数据。具体的，WGAN的生成器和判别器的更新公式如下：
$$\min _{G} \max _{D} V\left(D, G\right)=\underset{x}{\operatorname{avg}}\left[-D\left(x ; \theta_{d}\right)+\overline{D}\left(x ; \theta_{g}-\lambda_{p}\nabla_{x} J_{D}\left(D, x ; \theta_{g}\right)\right]\right), \quad\quad \text { s.t. } \quad \underset{x \sim P_{r}(x)}{\mathop{\mathbb{E}}}[J_{D}\left(D, x ; \theta_{g}\right)>0], \quad \forall x \in X$$
其中，$\theta_{d}$ 和 $\theta_{g}$ 分别是判别模型D和生成模型G的参数，$\lambda_{p}>0$ 是正则化系数，$X$ 是潜在空间，$V$ 是Wasserstein距离，$\overline{D}$ 是目标函数。WGAN的优点是训练更稳健，并且可以保证生成器产生的样本更接近真实数据。
#### StyleGAN
StyleGAN（StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks）是一种基于风格的GAN模型。其基本思路是学习出一个能够控制生成样本风格的生成网络，而不是训练一个独立的生成网络。具体来说，StyleGAN有三个关键模块，即潮汐（Mapping Network）、颤动（Synthesis Network）和分离器（Discriminator）。潮汐网络将输入风格映射到潜在空间，颤动网络将潜在空间的表示转换回数据空间，即生成图片；而判别器通过判别生成数据和真实数据之间的差距，帮助生成网络更好地拟合数据分布。StyleGAN的缺陷是计算量很大，而且训练周期比较长。
## 判别模型
判别模型D的目的是判别输入数据是真实数据还是生成数据，所以其训练过程中的loss function应具有可辨识度的性质，即希望模型可以分辨出真实数据和生成数据，并准确标记它们。其目标是生成器产生的数据越接近真实数据，那么判别器就越能够区分两者，输出1，否则输出0。换句话说，判别器D的损失函数应该能够衡量生成模型G和真实模型D的区别，才能使得生成数据与真实数据之间的差异更小。
### 判别器网络
判别器网络通过输入数据x，将其映射到一个概率值y，即判别模型D的输出。判别器的结构如下图所示：
判别器的输入是数据x，输出是一个概率值y，该概率值可以认为是一个置信度。y越接近于1，表明判别模型D越认为数据来自真实数据，y越接近于0，表明判别模型D越认为数据来自生成数据。
### PatchGAN
PatchGAN（Perceptual Generative Adversarial Networks for Image Synthesis）是一种有创意的判别模型结构。其基本思路是训练一个卷积神经网络（CNN），该网络能够根据输入的图像样本来预测图像是否为真实图像，以及它们之间的差异程度。具体来说，输入图像经过卷积网络（CNN）得到特征图（Feature Map），然后将特征图展开成图像大小的向量，送入到判别器中进行判别。PatchGAN的好处是能够捕获图像细节，使得模型能够更精确地判别样本。
# 4. 具体操作步骤及代码实现
## 数据准备
本文选择MNIST手写数字数据集作为示例，下载相应的数据并进行预处理。
```python
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.
```
## 生成器网络搭建
生成器网络由一个编码器和一个解码器两部分组成，编码器将输入数据压缩成潜在空间的表示，解码器将表示恢复到数据空间，生成新的样本。
### 编码器网络
编码器网络由几个Conv2D层和BatchNormalization层组成。第一个Conv2D层的卷积核数设置为64，核大小设置为3x3，padding方式为same；第二个Conv2D层的卷积核数设置为128，核大小设置为4x4，strides设置为2，padding方式为same；第三个Conv2D层的卷积核数设置为256，核大小设置为4x4，strides设置为2，padding方式为same；第四个Conv2D层的卷积核数设置为1，核大小设置为7x7，strides设置为1，padding方式为valid。最后，使用Flatten层和Dense层将数据压缩到一个向量。
```python
from tensorflow.keras import layers
from tensorflow.keras import Model


class Encoder(Model):
    def __init__(self, input_shape=(28, 28, 1)):
        super(Encoder, self).__init__()
        self.conv1 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='enc1')
        self.bn1 = layers.BatchNormalization()(self.conv1)

        self.conv2 = layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', activation='relu', name='enc2')
        self.bn2 = layers.BatchNormalization()(self.conv2)

        self.conv3 = layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', activation='relu', name='enc3')
        self.bn3 = layers.BatchNormalization()(self.conv3)

        self.flatten = layers.Flatten()(self.bn3)
        self.fc = layers.Dense(units=input_shape[0]*input_shape[1]*1, activation='tanh')(self.flatten)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x
```
### 解码器网络
解码器网络由几个UpSampling2D层、Conv2D层和BatchNormalization层组成。第一个UpSampling2D层的大小为2，即将特征图的尺寸进行扩充；第二个Conv2D层的卷积核数设置为128，核大小设置为3x3，padding方式为same；第三个Conv2D层的卷积核数设置为64，核大小设置为3x3，padding方式为same；第四个Conv2D层的卷积核数设置为1，核大小设置为7x7，padding方式为valid。最后，使用Tanh层将数据拉伸到0~1之间。
```python
class Decoder(Model):
    def __init__(self, output_shape=(28, 28, 1)):
        super(Decoder, self).__init__()
        self.upsample1 = layers.UpSampling2D(size=2)(layers.Input(shape=(output_shape[0]//4, output_shape[1]//4, 256)))
        
        self.conv2 = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name='dec2')
        self.bn2 = layers.BatchNormalization()(self.conv2)

        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name='dec3')
        self.bn3 = layers.BatchNormalization()(self.conv3)

        self.conv4 = layers.Conv2D(filters=1, kernel_size=7, padding='valid', activation='tanh', name='dec4')
        
    def call(self, inputs):
        x = self.upsample1(inputs)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        return x
```
### 模型搭建
将编码器和解码器组装成GAN模型，然后编译模型。
```python
encoder = Encoder(input_shape=(28, 28, 1))
decoder = Decoder(output_shape=(28, 28, 1))

gan = Model([encoder.inputs], decoder(encoder.outputs))
gan.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
```
## 判别器网络搭建
判别器网络将输入数据转换成概率值，用于判断数据是否为真实数据。
### 潮汐网络
潮汐网络（Mapping Network）的作用是将输入风格（即输入的非结构化数据）映射到潜在空间的表示。其结构如下图所示：
在StyleGAN中，潮汐网络由五个全连接层（FC）组成，每层后面紧跟一个ReLU激活函数。第一个FC层的输出维度为512，第二个FC层的输出维度为512，第三个FC层的输出维度为1024，第四个FC层的输出维度为1024，第五个FC层的输出维度为style_dim，即风格向量的维度。其中，第五个FC层的输出与style_dim相同，可以为每个样本生成多个不同的风格向量。
```python
class MappingNetwork(Model):
    def __init__(self, style_dim):
        super(MappingNetwork, self).__init__()
        self.dense1 = layers.Dense(units=512, activation='relu', name='map1')
        self.dense2 = layers.Dense(units=512, activation='relu', name='map2')
        self.dense3 = layers.Dense(units=1024, activation='relu', name='map3')
        self.dense4 = layers.Dense(units=1024, activation='relu', name='map4')
        self.dense5 = layers.Dense(units=style_dim, name='map5')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return x
```
### Discriminator网络
判别器网络由两个卷积层、两个最大池化层、两个全连接层组成。第一个卷积层的卷积核数设置为64，核大小设置为3x3，strides设置为2，padding方式为same；第二个卷积层的卷积核数设置为128，核大小设置为3x3，strides设置为2，padding方式为same；第三个卷积层的卷积核数设置为256，核大小设置为3x3，padding方式为same；第四个卷积层的卷积核数设置为512，核大小设置为3x3，padding方式为same；第五个卷积层的卷积核数设置为1，核大小设置为3x3，padding方式为valid。然后，通过Flatten层和Dense层将数据压缩到一个向量，再将向量送入到最后的全连接层中，输出一个概率值。
```python
class Discriminator(Model):
    def __init__(self, input_shape=(28, 28, 1), patch_size=None):
        super(Discriminator, self).__init__()
        if not patch_size:
            patch_size = int(((input_shape[0]-2)/2)*(input_shape[1]-2)//16 + 1)*int(((input_shape[0]-2)/2)*(input_shape[1]-2)//16 + 1)
            
        self.patch_size = patch_size
        
        self.conv1 = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name='disc1')
        self.conv2 = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu', name='disc2')
        self.conv3 = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name='disc3')
        self.conv4 = layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name='disc4')
        self.conv5 = layers.Conv2D(filters=1, kernel_size=3, padding='valid', activation='relu', name='disc5')

        self.pooling1 = layers.AveragePooling2D(pool_size=[2, 2], strides=[2, 2])
        self.pooling2 = layers.AveragePooling2D(pool_size=[2, 2], strides=[2, 2])
        
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=1, activation='sigmoid', name='disc6')
        self.fc2 = layers.Dense(units=1, activation='sigmoid', name='disc7')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        patch = tf.image.extract_patches(x, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')
        x = self.flatten(patch)
        logits1 = self.fc1(x)
        logits2 = self.fc2(x)

        outputs = tf.concat([logits1, logits2], axis=-1)
        return outputs
```
## 模型训练
模型训练包括训练生成器G和判别器D。
### 生成器训练
生成器G的训练过程包括训练生成器G的损失函数，并更新G的参数。判别模型D的输出是假样本的概率，所以G的训练目标是使得判别模型D无法正确分辨假样本，即使得生成的数据具有真实的图像特征。具体的，我们定义一个loss函数为：
$$\mathcal{L}_{\text {con }}=\mathbb{E}_{x \sim P_{\text {data}}}[\log D\left(x^{i} ; \theta_{d}\right)]+\mathbb{E}_{z \sim p(z)}\left[\log (1-D\left(G\left(z ; \theta_{g}\right) ; \theta_{d}\right))\right]$$
然后，在训练G的过程中，每一步选取一个batch的真实样本，计算梯度来更新G的参数。
```python
for epoch in range(epochs):
  # 每轮迭代开始之前，先shuffle数据
  np.random.shuffle(train_images)
  
  for step in range(num_batches):
      batch_images = train_images[step * batch_size:(step+1) * batch_size]
      
      # 训练判别器D
      with tf.GradientTape() as tape:
          real_logits = discriminator(batch_images)
          
          z = tf.random.normal([batch_size, latent_dim])
          fake_images = generator(z)
          fake_logits = discriminator(fake_images)
          
          con_loss = cross_entropy(tf.ones_like(real_logits), real_logits) + cross_entropy(tf.zeros_like(fake_logits), fake_logits)
          
          gradient_of_con_loss = tape.gradient(con_loss, discriminator.trainable_variables)
          
          discriminator.optimizer.apply_gradients(zip(gradient_of_con_loss, discriminator.trainable_variables))

      # 训练生成器G
      z = tf.random.normal([batch_size, latent_dim])
      with tf.GradientTape() as tape:
          fake_images = generator(z)
          fake_logits = discriminator(fake_images)
          
          g_loss = cross_entropy(tf.ones_like(fake_logits), fake_logits)
          
          gradient_of_g_loss = tape.gradient(g_loss, generator.trainable_variables)
          
          generator.optimizer.apply_gradients(zip(gradient_of_g_loss, generator.trainable_variables))

  print("Epoch {}/{}".format(epoch+1, epochs), "Con Loss:", round(con_loss.numpy().mean(), 3), "G Loss:", round(g_loss.numpy().mean(), 3))
```
### 判别器训练
判别器D的训练过程包括训练判别器D的损失函数，并更新D的参数。生成模型G的输出是真样本的概率，所以D的训练目标是能够准确地识别真样本，即使得生成的数据具有真实的图像特征。具体的，我们定义一个loss函数为：
$$\mathcal{L}_{\text {dis }}=\mathbb{E}_{x \sim P_{\text {data}}}[\log D\left(x^{i} ; \theta_{d}\right)]+\mathbb{E}_{z \sim p(z)}\left[\log (1-D\left(G\left(z ; \theta_{g}\right) ; \theta_{d}\right))\right]$$
然后，在训练D的过程中，每一步选取一个batch的真实样本和假样本，计算梯度来更新D的参数。
```python
def calculate_gradient_penalty(real_images, fake_images, discriminator, lambd=10):
    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    
    interpolates = alpha * real_images + ((1 - alpha) * fake_images)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        disc_interpolates = discriminator(interpolates)
        
    gradients = tape.gradient(disc_interpolates, [interpolates])[0]
    
    gradient_penalty = tf.reduce_sum(tf.square(tf.norm(gradients, ord=2) - 1))
    gp = lambda_gp * gradient_penalty
    return gp

for epoch in range(epochs):
  # 每轮迭代开始之前，先shuffle数据
  np.random.shuffle(train_images)
  
  for step in range(num_batches):
      batch_images = train_images[step * batch_size:(step+1) * batch_size]
      
      # 训练判别器D
      with tf.GradientTape() as tape:
          real_logits = discriminator(batch_images)
          
          z = tf.random.normal([batch_size, latent_dim])
          fake_images = generator(z)
          fake_logits = discriminator(fake_images)
          
          d_loss = cross_entropy(tf.ones_like(real_logits), real_logits) + cross_entropy(tf.zeros_like(fake_logits), fake_logits)
          
          gradient_of_d_loss = tape.gradient(d_loss, discriminator.trainable_variables)
          
          discriminator.optimizer.apply_gradients(zip(gradient_of_d_loss, discriminator.trainable_variables))
        
      # 添加GP惩罚项
      if use_gp:
          gp = calculate_gradient_penalty(batch_images, fake_images, discriminator)
          
          gradient_of_d_loss += tape.gradient(gp, discriminator.trainable_variables)
          
      discriminator.optimizer.apply_gradients(zip(gradient_of_d_loss, discriminator.trainable_variables))

  print("Epoch {}/{}".format(epoch+1, epochs), "D Loss:", round(d_loss.numpy().mean(), 3))
```