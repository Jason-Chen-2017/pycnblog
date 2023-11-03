
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


在互联网和移动互联网蓬勃发展的今天，自动创作技术已经成为当下人工智能领域最热门、最火爆的研究方向之一。无论是对文字、图片、视频等媒体内容进行创作还是对音乐、美术作品进行创造，都可以在机器学习、强化学习、多种维度的优化中提升创作者的创作能力。例如，一款软件能够根据用户提供的内容自动生成一段完整的歌词，这就是一个智能创作的例子。或者可以用计算机视觉来帮助设计师完成场景布景、人物动效，甚至是风格迁移，这些都是人工智能智能创作的前沿探索。同时，越来越多的企业也在朝着这个方向努力迈进。
相对于传统的人工智能应用而言，智能创作带来的新机遇更加吸引人。首先，智能创作者不需要掌握复杂的编程语言、深厚的数学功底，只需要有一点创意、构思能力、想象力就可以创作出高质量的内容。其次，通过智能创作可以提升公司的知名度、营收，也可以让创作者实现价值共享、个人职场上的突破。最后，智能创作还能够为社会的发展添砖加瓦，促进科技创新的繁荣。因此，如何利用人工智能技术来创作出具有独特魅力、受关注的作品，已成为各行各业追求的方向。
那么，对于智能创作者而言，他们面临的最大挑战就是要用一种方法既能准确生成想要的结果又不失灵感。而这项工作的关键在于理解人类的语言、掌握自然语言处理的技巧、运用大量数据进行训练、选择正确的模型结构以及优化参数，从而达到“一分耕耘，十年建树”的效果。本文将详细阐述智能创作的基本概念、核心算法原理、具体代码实例和未来发展方向。希望能通过本文对读者有所启发，并能给予更多参考信息。
# 2.核心概念与联系
## 智能创作的概念定义及其与人工智能领域的关系
智能创作(Intelligent Creation)指的是基于特定领域的AI技术（如计算机视觉、自然语言处理、强化学习）产生的内容，通过一种简单的方式帮助用户生成符合需求的内容。一般来说，智能创作有以下几个方面的特征：
- 定制性：要求用户有丰富的创意经验或知识，有着特殊的想法；
- 个性化：智能创作者应该具备一定数量的创作素材、技能，可以通过分析用户习惯、偏好、风格等给予个性化推荐；
- 可控性：用户可以使用自己的语义表达能力、语法规则，以及个人喜好来控制生成的内容；
- 协同性：智能创作者之间可以进行有效地交流、协作，提升创作质量。
智能创作与人工智能领域密切相关，因为它需要依靠人工智能技术进行信息检索、图像理解、自然语言处理、决策支持等方面的功能。据统计，目前国内外智能创作相关的产品和服务企业超过了6亿个，每年新增超过三千万。从应用角度看，智能创作可用于广告、新闻、视频制作、互联网信息流、内容生产、互动社区等多个领域。

## 智能创作相关的主要研究领域
智能创作研究从不同的角度出发，将智能创作研究分成如下几个主要研究领域:
- 媒体创作：主要研究媒体内容智能生成的技术和应用。如视频、图像、文本等创作任务的自动生成；
- 故事生成：主要研究通过对话、语言等生成计算机生成的故事。如一个虚拟角色进行对话、唤起用户情绪、构建情境等；
- 艺术创作：包括美术创作、绘画创作、音乐创作、视频创作等，围绕创作者的创作、表达及其评论等进行研究。如通过机器学习、大数据分析等技术，帮助艺术创作者开发新的创作模式。
- 产品创意：包括手游、APP产品、电商、平板电脑、手机应用、游戏等应用领域的创意生成。如新冠疫情期间，国内许多创意产品都受到重视，通过多方合作、联动、迭代，提升产品的营收、用户满意度、竞争力等。
- 社会影响：通过研究智能创作对人类生活的影响，如社会效应、品牌宣传、个人成长等。如智能手机应用之后，消费者对生活节奏的调节、时间管理的提升、亲子关系的改善等得到了广泛的关注。
# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## 一、文本生成
### 生成式模型（Generative Model）
文本生成的基本思路是基于语言模型，即通过计算某些假设下的概率，来预测下一个词出现的可能性，再根据这些概率进行采样生成文本。概率计算的方法有很多种，这里采用Maximum Likelihood Estimation (MLE)方法。简单来说，就是给定当前词的上下文（context），模型预测下一个词的概率分布，再通过一定策略选取概率较大的词作为输出。

对于中文，通常使用HMM或BiLSTM+CRF模型，这里我们使用BERT模型（Bidirectional Encoder Representations from Transformers）。

### BERT模型（Bidirectional Encoder Representations from Transformers）
BERT(Bidirectional Encoder Representations from Transformers)是谷歌团队在2018年Nature上发表的一篇文章提出的预训练方法。这篇文章把transformer的encoder结构应用到NLP任务中，取得了非常好的结果，并且模型参数开源免费，是当前最流行的预训练模型。

BERT是一个双向的 transformer 模型，通过使用左右两边的 context 来表示单词的语义，相比于之前的左右独立的 RNN 或 CNN ，这种表示方式能够融合上下文信息，在很大程度上解决了 sequence labeling 和 machine reading comprehension 的两个难题。BERT 通过 fine-tuning 把通用的 transformer encoder 模型变成了特定任务的模型，使得模型在多个 NLP 数据集上做到了 state of the art 。

BERT 的模型架构如下图所示：


1. Tokenizing and Preprocessing
输入文本经过 tokenization 后会转换成 WordPiece Tokens ，这一步会对句子中的每个 token 进行切割，然后用 BERT 模型的字典查找表替换掉不能识别的词汇。

2. Input Embedding Layer
在此层，输入的 WordPiece tokens 会被映射到 BERT 模型的 embedding space 中，这里使用的嵌入方式是 one hot encoding ，即每个 token 会对应一个唯一的索引，并通过 softmax 函数进行归一化。

3. Attention Layer
这是 BERT 中的 attention layer，它的作用是计算输入的文本序列中每个词与其他词之间的相关性，并生成相应的权重系数。Attention layer 接收三个输入：query 表示单词表示，key 和 value 表示整体文本的编码。attention layer 使用 dot product 操作符计算 query 和 key 之间的相关性，并生成相应的权重系数。最后，使用 value 将 query 和权重系数进行点乘，得到最终的 context vector 。

4. Output Layer
输出层输出的是整个序列的表示向量，也就是 encoder 输出的最后一层隐藏状态。为了适配分类任务，最后接了一个 dense layer 进行分类。

模型训练时，使用最小化损失函数的方法更新模型参数。

### BERT模型推断过程简析
BERT模型推断的过程比较复杂，但本质上就是在最后一层输出层进行分类预测。BERT的输入是一个文本序列，首先经过 tokenization 和 preprocessing 操作，生成对应的 token 序列。然后输入到模型中，经过 encoder 网络得到 contextualized embeddings ，再进行 softmax 归一化，得到属于不同类别的概率分布。这时候可以直接用概率分布进行分类预测。但是如果只是简单预测，可能模型对某些类别的预测准确度不够。所以可以结合 decoder 网络对生成的文本进行进一步推断，包括 next sentence prediction 和 masked language modeling 。next sentence prediction 是判断输入的两个句子是否属于连贯的上下文。masked language modeling 是利用 mask 对输入文本进行预测。

## 二、文本续写
### SeqGAN（Sequence Generative Adversarial Network）
SeqGAN模型是一个深度学习模型，由两个生成器组成——生成器G和判别器D。生成器G的任务是在任意长度的文本序列上生成一系列连续的词语，判别器D的任务是区分生成的文本序列和真实文本序列的真伪，即判别生成的文本序列是否是原始文本序列的翻译版本。生成器G在训练过程中倾向于生成更加逼真的文本，判别器D则需要通过训练使生成器G无法欺骗判别器D。在SeqGAN模型中，两者的目标函数如下：

$$\min_{G} \max_{D} V(D,G) = E_{x~p_{data}(x)}[\log D(x)] + E_{z~p_{noise}(z)}[\log (1-D(G(z)))]$$

其中，$x$代表真实数据文本序列，$z$代表噪声（潜在空间中的随机变量），$p_{data}$代表数据分布，$p_{noise}$代表噪声分布。

SeqGAN模型的架构如下图所示：


#### 1. 判别器D
判别器D是一个二分类器，它通过一个多层神经网络接受输入序列x和输出0~1之间的概率值。判别器D的输入是一个序列$x=\{x_i^t| i=1...m, t=1...T\}$，它在每一个时刻都接收一个词向量$\phi(x_i^{t})$作为输入，其中$\phi(x)$是表示第t个时刻词向量的隐藏层。之后通过一个全连接层输出一个logit值$h_i=f(\sum_{j}^{}w_{ij}\phi(x_j^{t}))$，其中$f()$是一个非线性激活函数，$w_{ij}$代表由判别器D学习到的权重。最后使用sigmoid激活函数输出最终的概率$y=Sigmoid(h_i)=P(x_i^t\mid x_{\leq}^{t})$，表示输入序列x的第t个时刻的词是真实的概率。

#### 2. 生成器G
生成器G是一个自回归模型，它的目的是根据一个潜在变量z，生成一个文本序列$x^\prime=\{x^{\prime}_i^t| i=1...n, t=1...T^\prime\}$，这里的z代表先验知识，比如可能是一个随机生成的句子。生成器G的输入是一个潜在变量z，它会生成一个初始词向量$\theta(z)$作为第一个词的输入，随后循环生成剩余的词向量。在每一步，生成器G都会接收到上一步生成的词向量$v_i^{t-1}$、上一步生成的隐含状态$s_i^{t-1}$以及上一步生成的输入$x^{\prime}_{i-1}^{t-1}$作为条件输入。

生成器G的隐含状态$s_i^t$在每一步都由一个递归神经网络生成，它以$(x^{\prime}_i^{t-1}, s_{i}^{t-1})$作为输入，输出一个$(v_i^t,s_i^t)$作为输出。其中$v_i^t$是当前时刻生成的词向量，$s_i^t$是该时刻生成的隐含状态。

#### 3. 损失函数
SeqGAN的损失函数使用交叉熵作为度量标准，并使用以下损失函数：

$$L_D(x)=-\frac{1}{m}\sum_{i=1}^m[y_i\log P(x_i)\mathbf{1}_{\rm real}+\left(1-y_i\right)\log P(x_i)|\bar{x}]-\lambda H(q(z|x))$$

其中，$y_i=\begin{cases}1,&if\;x_i\in X\\0,&otherwise\end{cases}$代表标签，$\lambda>0$是拉普拉斯先验，$H(q(z|x))$是真实分布$p(z|x)$的熵。

$$L_G(z)=-\frac{1}{n}\sum_{i=1}^n\log p_{\theta}(x^{\prime}_i|\theta,\phi)+\beta H(q(z|x^{\prime}_{<}^{i}))$$

其中，$\theta$代表生成器的参数，$\beta > 0$是生成分布$p_{\theta}(z)$的熵。

## 三、图像生成
### GAN（Generative Adversarial Networks）
GAN是2014年提出的一种新的生成模型，其主要思想是由一个生成器生成一批图片，而另一个判别器则负责区分生成的图片是否是真实图片，从而让生成器生成真实looking的图片。如今，GAN已经成为生成高质量图像的主流方法。

GAN的核心思想是，生成器生成的假样本和真实样本一起训练一个神经网络，使生成器更快地模仿真实样本，而判别器则尽量欺骗生成器，使其误判生成样本为真实样本。

GAN的结构如下图所示：


GAN的损失函数为：

$$\min _{G} \max _{D} V(D, G)=E_{\boldsymbol{x} \sim p_{data}} [\log D (\boldsymbol{x})]+E_{\boldsymbol{z} \sim p_{noise}}[\log (1-D(G(\boldsymbol{z})))], \quad \text { where } D(\boldsymbol{x}):=\sigma (g(\boldsymbol{x})).$$ 

其中，$D$和$G$分别是判别器和生成器，$\sigma$是 sigmoid 函数，$g$是映射函数，$p_{data}$和$p_{noise}$代表真实样本分布和生成样本分布。

### CycleGAN（Cycle Consistency Loss for Image-to-Image Translation）
CycleGAN是一种基于Cycle-Consistancy GAN的图像到图像的域转换方法，其基本思想是利用Cycle-Consistency来消除域转换中风险，即判别器只关注输入的原始图像和转化后的图像之间的差异，而不关注其他特征。CycleGAN的两个主要问题是：缺少监督信号、如何生成对抗样本。

CycleGAN的结构如下图所示：


CycleGAN的损失函数如下所示：

$$\mathcal{L}_{cyc}(\theta_{A\rightarrow B}, \theta_{B\rightarrow A})={1\over2}\sum_{l\in{A,B}}\sum_{i=1}^{n_l}\sum_{j=1}^{n_l}(||\varphi_{l}(x_{il}^B)||_{2}-||\hat\varphi_{l}(x_{il}^A)-\tilde\varphi_{l}(x_{il}^B)||_{2})\mathcal{L}_{con}(A,B) \\ +\lambda_{id}\cdot{||\varphi_{l}(x_{il}^A)-\hat\varphi_{l}(x_{il}^A)||_{2}}^{2}$$

其中，$\mathcal{L}_{con}(A,B)$代表一致损失，$n_l$是第l个域的图像数量，$\varphi_l$和$\hat\varphi_l$分别是第l个域的特征提取器，$\tilde\varphi_l$是第l个域的映射函数，$\lambda_{id}$是一个调整参数。

# 4.具体代码实例和详细解释说明
## 文本生成代码实例（BERT+GPT-2）
### BERT模型代码
```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # or bert-large-uncased
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

input_ids = tokenizer("The cat is <mask>.", return_tensors="pt").input_ids
outputs = model(input_ids)[0]
predicted_index = torch.argmax(outputs[0, :]).item()
predicted_token = tokenizer.decode([predicted_index])
print(predicted_token)
```

### GPT-2模型代码
```python
import openai
openai.api_key = 'YOUR OPENAI API KEY'

response = openai.Completion.create(
    engine='davinci',
    prompt='The cat is ',
    temperature=0.7,
    max_tokens=5,
    top_p=1.0,
    n=1,
    stream=False,
    stop=['.', '!', '?'])

print(response['choices'][0]['text'].strip())
```

## 文本续写代码实例（SeqGAN）
### SeqGAN模型代码
```python
import os
import random

import nltk
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tqdm import trange

nltk.download('wordnet')

def preprocess_sentence(sent):
    sent = re.sub('\s+','', sent).strip()
    words = nltk.word_tokenize(sent)

    if len(words) == 0:
        raise ValueError("Can't process empty sentences")

    words = ['<start>'] + [word for word in words] + ['<end>']
    word_idx = dict([(w, idx) for idx, w in enumerate(set(words), start=1)])

    return words, word_idx

class TextGenerator():
    def __init__(self, batch_size=16, seq_len=32):
        self.batch_size = batch_size
        self.seq_len = seq_len

        data_path = '/content/'
        filelist = os.listdir(data_path)
        filelist = sorted([os.path.join(data_path, f) for f in filelist
                           if f.endswith('.txt')])
        
        texts = []
        for filename in filelist:
            with open(filename, 'r') as f:
                text = f.read().lower()
                texts += list(map(str.split, text.split('\n')))
                
        all_texts = []
        for i, text in enumerate(texts):
            try:
                _, word_idx = preprocess_sentence(text)
                all_texts.append((i, text, word_idx))
            except ValueError:
                continue

        self.all_texts = all_texts
        
    def generate_batches(self, mode='train'):
        while True:
            batches = []
            source_idxs = []
            target_idxs = []

            for i in range(self.batch_size // 2):
                index = random.choice(range(len(self.all_texts)))
                
                source_idx, text, word_idx = self.all_texts[index]

                prefix = '<start>' * 2
                prefix_idx = [word_idx[prefix]]
                prefix_idx += [random.randint(1, len(word_idx) - 1)
                                for _ in range(self.seq_len - 2)]

                suffix = '</end>' * 2
                suffix_idx = [word_idx[suffix]]
                suffix_idx += [random.randint(1, len(word_idx) - 1)
                                for _ in range(self.seq_len - 2)]

                input_seq = prefix_idx[:-1]
                output_seq = suffix_idx[:2] + [word_idx[k]
                                               for k in ''.join([''.join(text[-self.seq_len:])])[::-1]][:-1]
                
                batches.append([[word_idx[word]
                                 for word in input_seq],
                                output_seq])
                
            yield [[pad_sequences(batch[0]),
                    pad_sequences(batch[1])]
                   for batch in batches]
                
generator = TextGenerator()
        
vocab_size = len(generator.all_texts[0][2]) + 1
embedding_dim = 128
hidden_dim = 256
latent_dim = 128
dropout_rate = 0.5
num_layers = 3

class Generator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, num_layers, dropout_rate):
        super().__init__()

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size,
                                       embedding_dim,
                                       mask_zero=True),
            tf.keras.layers.GRU(hidden_dim,
                                 return_sequences=True,
                                 return_state=True,
                                 kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.RepeatVector(self.seq_len - 1),
            tf.keras.layers.GRU(hidden_dim,
                                 return_sequences=True,
                                 return_state=True,
                                 kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size, activation='softmax')
        ])

    def call(self, inputs, states=None, training=None):
        enc_outputs, enc_states = self.encoder(inputs, training=training)
        dec_inputs = tf.expand_dims([word_idx['<start>']] * self.batch_size, 1)

        predictions = []
        for t in range(self.seq_len - 1):
            dec_outputs, dec_states = self.decoder(dec_inputs,
                                                    initial_state=enc_states,
                                                    training=training)
            predicted_probas = tf.nn.softmax(dec_outputs[:, :, :] / temperature)
            predicted_token = tf.cast(tf.argmax(predicted_probas, axis=-1), tf.int32)

            predictions.append(predicted_token)
            dec_inputs = tf.concat([tf.expand_dims(predicted_token, 1),
                                    dec_inputs[:, :-1]], axis=-1)

        return tf.stack(predictions, axis=1)
    

class Discriminator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
        super().__init__()

        self.lstm = tf.keras.layers.LSTM(units=hidden_dim,
                                         return_sequences=True,
                                         return_state=True,
                                         dropout=dropout_rate,
                                         recurrent_dropout=dropout_rate)

        self.dense = tf.keras.layers.Dense(units=1,
                                           activation='sigmoid')

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim,
                                                   name='embedding')

    def call(self, inputs, lengths=None, states=None, training=None):
        embedded = self.embedding(inputs)
        lstm_outputs, h, c = self.lstm(embedded)
        logits = self.dense(lstm_outputs)

        return logits, None

    
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

@tf.function
def train_step(source_seq, target_seq):
    with tf.GradientTape() as tape:
        generator.trainable = False
        gen_seq = generator(target_seq)
        discriminator.trainable = True

        disc_logits, _ = discriminator(gen_seq)
        disc_loss = binary_crossentropy(tf.ones_like(disc_logits), disc_logits)

        true_logits, _ = discriminator(source_seq)
        true_loss = binary_crossentropy(tf.zeros_like(true_logits), true_logits)

        total_loss = (disc_loss + true_loss) / 2

    gradients = tape.gradient(total_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    
    return disc_loss


for epoch in trange(100):
    for source_seq, target_seq in generator.generate_batches():
        disc_loss = train_step(source_seq, target_seq)
```

## 图像生成代码实例（DCGAN）
### DCGAN模型代码
```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display

BUFFER_SIZE = 60000
BATCH_SIZE = 128
IMG_WIDTH = 28
IMG_HEIGHT = 28
OUTPUT_CHANNELS = 1

seed = tf.constant(np.random.normal(size=[BUFFER_SIZE, IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS]))

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model



generator = make_generator_model()
discriminator = make_discriminator_model()

decision = discriminator(generated_images)
print (decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(lr=1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
for epoch in range(EPOCHS):
  for images in dataset:
    train_step(images)

  # Produce images for the GIF as we go
  display.clear_output(wait=True)
  generate_and_save_images(generator, epoch + 1, seed)

  # Save the model every 15 epochs
  if (epoch + 1) % 15 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)