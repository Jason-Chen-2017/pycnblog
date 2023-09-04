
作者：禅与计算机程序设计艺术                    

# 1.简介
  

近年来，深度学习在自然语言处理领域扮演着越来越重要的角色。传统的基于规则和统计模型的分词、词性标注、命名实体识别、文本摘要等任务都已经被深度神经网络取代，并且取得了显著的效果。为了理解深度学习是如何帮助我们解决这些任务的，并探索其潜藏的巨大可能性，本文将从两个方面进行阐述——Markov Chain生成模型(MCG)和循环神经网络(RNN)。接下来，我们将讨论它们的优缺点，以及如何通过TensorFlow 2.x实现它们。最后，我们将利用这些模型来对文本生成，并做一些有意思的应用，比如阅读理解、机器翻译和情感分析。
# 2.基本概念
## 2.1.概率分布
在深度学习中，概率分布是一个非常重要的概念。它表示随机变量取值的概率。对于离散型随机变量，如字母表中的每个字母，我们可以用概率来表示该字母出现的概率。对于连续型随机变量，如图像中的每个像素，我们也可以用概率分布来描述其值的范围及概率密度函数（Probability Density Function）。
图1: 概率分布示意图

## 2.2.马尔可夫链
马尔可夫链（Markov chain）是由一个初始状态到另一个状态的随机过程，该过程具有以下特性：

1. 当前时刻只能依赖于过去时刻的信息；
2. 在当前时刻，状态只依赖于前一状态，不受其他影响；
3. 一旦到达终止状态或达到预定步长，则链断裂。

例如，在一个有限状态机（finite state machine, FSM）中，每个状态对应于一个动作或者事件，而转移概率则表示当系统处于某一状态时，采取某个动作转移到下一状态的概率。如果马尔可夫链处于状态$i$，则在时刻t+1状态必然是状态$j$(i<j)，且转移概率由马尔科夫转移矩阵M给出。马尔可夫链的特点决定了它的强大的生成能力。
图2: MCMC方法

## 2.3.马尔可夫决策过程
马尔可夫决策过程（Markov decision process, MDP），也称为马尔可夫奖赏过程，是指一个具有随机性的动态系统，其中状态是由行为空间的一个状态向量，行为空间中的每一个元素都有一个唯一标识符，动作是有限状态空间中的一个动作向量。在MDP中，环境是完全观测到的，即整个状态序列都已知，每一步都由环境确定，所以MDP定义了一种形式化的方法来研究一个智能体如何在一个环境中最大化它的收益。

状态转移函数定义了状态转移概率分布，即在状态$s_t$时选择动作$a_t$后下一状态$s_{t+1}$的条件概率分布。奖励函数定义了在执行动作$a_t$后，系统给予奖励的期望值，即$\sum_{i=0}^\infty \gamma^i r_i$，其中$\gamma$是折扣因子，r_i表示在第i个时间步获得的奖励。所以，MDP将带领我们逐渐了解如何通过回合制的方式，通过交互式地玩游戏，来学习如何在复杂的环境中找到最佳的策略。
图3: OpenAI Gym

## 2.4.循环神经网络
循环神经网络（Recurrent Neural Networks, RNN）是深度学习中的一种重要模型。它是一种特殊的深层神经网络，它能够处理序列数据，处理输入数据的顺序依次递进。RNN通过递归结构拥有记忆功能，使得它能够记住之前看到的数据，并根据过往经验提升决策效率。RNN有两种类型：单向RNN和双向RNN。单向RNN可以捕获信息仅仅一定的方向，即反向传递。双向RNN可以捕获信息在正向和反向两个方向上都有。
图4: RNN示意图

## 2.5.LSTM与GRU
LSTM（Long Short-Term Memory）和GRU（Gated Recurrent Unit）是RNN的变体。LSTM是一种更加复杂的RNN单元，它可以学习长期依赖关系，在很多任务中都有良好的表现。GRU相比LSTM可以减少参数个数，运算速度更快。LSTM中的遗忘门、输出门、更新门分别用于控制输入，输出，状态更新。
图5: LSTM和GRU结构示意图
# 3.算法原理
## 3.1.概率评估模型
在文本生成任务中，我们假设已知一个训练集，包括大量的文本样本，每个样本都是由一个有序序列组成。为了构造一个模型，使得它能够根据这个训练集生成新文本，我们需要设计一个概率评估模型。所谓的概率评估模型就是用来评估输入序列的下一个输出的概率分布。换句话说，就是输入序列和输出序列的联合概率分布。

通用的概率评估模型可以由三部分构成：

1. 发射概率模型（Emission Probability Model）：用于建模输入序列中的每个单词的概率分布。也就是每个输出词按照多大的概率出现在下一个输出词。

2. 跳转概率模型（Jump Probability Model）：用于建模输入序列中两个连续单词之间的概率分布。也就是模型知道当前词时，下一个词出现的概率分布。

3. 终止概率模型（Termination Probability Model）：用于建模生成结束的概率分布。也就是模型知道整个序列生成完成之后，生成新序列的概率分布。

这些概率模型之间存在什么联系呢？举例来说，考虑生成一个语句。假设我们已经得到了一个训练集：“你喜欢编程吗？”，那么，我们的目的就是根据这个训练集构造出一个概率评估模型，让它能够产生一个新的语句：“我也喜欢读书！”。这种模型的构造一般包括：

- Emission Probability Model：输入“你喜欢”时的发射概率分布，假设这个词出现的次数为$N_w$，那么我们就设置发射概率模型为：
$$P(o_i|h_t)=\frac{c(o_i,h_t)}{\sum_{l=1}^{V}c(l,h_t)}, l=1,...,V$$

- Jump Probability Model：假设下一个词只与当前词相关，则我们就设置跳转概率模型为：
$$P(h_t'|h_t)=\alpha_{h_t h_t'}^{z}, z=1,\cdots,n-1$$

- Termination Probability Model：假设整个序列生成完成之后，则设置终止概率模型为：
$$P(\text{end}|h_T)=\beta_h^u$$

其中$h_t$表示第$t$个隐藏节点的输出，$o_i$表示第$i$个输出节点的输出，$V$表示输出空间的大小，$c(l,h)$表示在隐藏节点$h$下的第$l$个输出$o_l$出现的次数。其他的参数$\alpha$, $\beta$等参数可以通过极大似然估计进行估计。

## 3.2.迭代采样算法
通过以上概率评估模型，我们已经构建出了可以生成新文本的模型。但实际上，我们还没有完全掌握生成文本的算法，还需要一种迭代算法。迭代算法是指每次从概率评估模型中抽样出一个新字符，然后将生成的字符添加到最终的结果中，一直迭代下去直到达到预定长度或者遇到终止符号才停止。

具体的迭代算法有很多种，这里我们介绍一种基于马尔可夫链的蒙特卡洛算法（Monte Carlo Sampling Algorithm）。蒙特卡洛算法是一种常用的生成模型采样方法。在每一次迭代中，我们先初始化一个状态$h_0$，然后对每一个输入$x_t$，根据如下公式计算下一个状态$h_t'$：

$$p(h_t'|h_t, x_t)\propto p(o_t|h_t)p(h_t')\\
= \alpha_{h_t o_t}^{z} c(o_t, h_t) \\
=\prod_{l=1}^{n}\left(\frac{\alpha_{h_tl}^z}{Z}\right)^tc_l^{zo}_{to}$$

其中$n$表示模型的隐含层结点个数，$\alpha_{h_tl}^z$表示在状态$h_t$下，第$l$个隐藏节点的值为$l$，且$t$时刻的输出是$o_t$的概率，$c_l^{zo}_{to}$表示在状态$h_t$下，输出为$o_t$情况下，第$l$个隐藏节点的值为$l$的次数，$Z$表示所有可能的路径的期望值。

根据贝叶斯公式，可以得到：

$$\ln P(H|\mathbf{x}) = \ln\left[ \frac{1}{Z} \prod_{t=1}^{T} p(h_t | h_{t-1}, x_t) \right]$$

其中$T$表示输入序列的长度，$H=(h_1,h_2,\cdots,h_T)$表示隐藏层的输出序列。

通过上面的公式，我们就可以通过蒙特卡洛算法对隐藏层的输出进行采样，得到符合统计规律的新文本。

## 3.3.预测算法
最后，为了预测新文本，我们需要通过对已有文本的分析来获取更多的上下文信息。通过对已有文本分析，我们可以了解到当前词的历史分布，比如当前词出现的次数、上文词出现的次数、下文词出现的次数等。这样的话，我们就可以根据历史信息来预测当前词的发射概率分布。

# 4.实践案例
本节，我们将通过几个例子展示如何通过TensorFlow 2.x实现基于马尔可夫链的生成模型。

## 4.1.机器翻译模型
机器翻译模型主要用于实现翻译任务，它可以自动把一种语言的句子转换为另外一种语言的句子。为了实现机器翻译模型，我们需要准备好大量的源语言句子和目标语言句子对。在训练过程中，模型可以学习到源语言句子和目标语言句子对的概率分布，并用此分布来生成目标语言句子。

### 数据集
我们可以使用经典的英语-法语数据集来训练机器翻译模型，该数据集包含了约7百万条英语句子和对应的法语句子。为了方便起见，我们只选取其中几百条作为训练集，验证集，测试集。

```python
import tensorflow as tf

BATCH_SIZE = 64
BUFFER_SIZE = 10000

train_dataset = tf.data.Dataset.from_tensor_slices((src_sentences, tgt_sentences)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset =... # same for validation dataset
test_dataset =... # same for test dataset
```

### 模型架构
对于一个翻译模型，我们通常采用三层的RNN：编码器、解码器和注意力机制。编码器接受源语言输入，使用RNN编码得到状态$h_t$，随后将状态$h_t$通过注意力机制压缩成固定维度的$c_t$。解码器接受$c_t$和当前词$y_{t-1}$，使用RNN生成目标语言词$y_t$，同时将$y_{t-1}$作为输入送入到下一个时间步。

```python
class Translator():
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def train_step(source_seq, target_seq):
        enc_output, enc_hidden = self.encoder(source_seq)
        dec_hidden = enc_hidden

        loss = 0

        for t in range(target_seq.shape[1]):
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(predictions, target_seq[:, t])

            dec_input = tf.expand_dims(target_seq[:, t], 1)
        
        return loss / int(target_seq.shape[1])
    
    @tf.function
    def inference(source_seq):
        enc_output, enc_hidden = self.encoder(source_seq)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([tokenizer_en.word_index['<start>']] * BATCH_SIZE, 1)

        translated_sentence = []

        for i in range(max_length):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_output)

            predicted_id = tf.argmax(predictions[i]).numpy()

            if tokenizer_fr.index_word[predicted_id] == '<end>':
                break
            
            output_word = tokenizer_fr.index_word[predicted_id]

            translated_sentence.append(output_word)

            dec_input = tf.expand_dims([predicted_id] * BATCH_SIZE, 1)

        return''.join(translated_sentence)
    
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
    
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)
        
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
    
        return context_vector, attention_weights
    
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)
        
    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights
```

### 训练
```python
model = Translator()

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

checkpoint_path = "./checkpoints/translator"
ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")

EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    total_loss = 0

    for (batch, (inp, tar)) in enumerate(train_dataset):
        batch_loss = model.train_step(inp, tar)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / num_batches))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
```

### 测试
```python
model = Translator()

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_path = "./checkpoints/translator"
ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")

while True:
    sentence = input("Enter a English sentence:\n")
    inputs = [tokenizer_en.word_index.get(i, UNK_token) for i in sentence.split()]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=MAX_LENGTH_ENGLISH, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = model.inference(inputs)
    print('Translation:', result)
```

## 4.2.情感分析模型
情感分析模型可以应用到自然语言处理中，它可以分析一个句子的情感倾向，判断它是积极还是消极，或者是中性的。为了实现情感分析模型，我们需要准备好大量的情感分析语料库，这些语料库包含了大量的积极文本，负面文本，和中性文本。

### 数据集
我们可以使用IMDB影评数据库来训练情感分析模型。该数据库包含了50,000条影评，其中12,500条作为训练集，12,500条作为测试集，剩余的25,000条作为验证集。

```python
import tensorflow as tf

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def preprocess_text(sent):
  sent = BeautifulSoup(sent, "html.parser").get_text()
  filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n'
  sent = ''.join(c for c in sent if c not in filters)
  tokens = nltk.tokenize.word_tokenize(sent)
  tokens = [token.lower() for token in tokens]
  stopwords = set(nltk.corpus.stopwords.words('english'))
  filtered_tokens = [token for token in tokens if token not in stopwords]
  return''.join(filtered_tokens)

def create_dataset(df):
    df = df[['review','sentiment']].sample(frac=1).reset_index(drop=True)
    positive_examples = df[df['sentiment']==1]['review'].tolist()[:int(len(df)*0.8)]
    negative_examples = df[df['sentiment']==0]['review'].tolist()[:int(len(df)*0.8)]
    neutral_examples = df[df['sentiment']==2]['review'].tolist()[:int(len(df)*0.8)]
    all_examples = positive_examples + negative_examples + neutral_examples
    labels = ([1]* len(positive_examples) + [0]* len(negative_examples) + [2]* len(neutral_examples))*5
    return pd.DataFrame({'text':all_examples, 'label':labels}).sample(frac=1).reset_index(drop=True)

df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/IMDB Dataset.csv').dropna().rename(columns={'review':'text'})
df['text'] = df['text'].apply(lambda x:preprocess_text(str(x)))
df = create_dataset(df)

train_size = int(len(df)*0.8)
valid_size = int(len(df)*0.1)

train_dataset = df[:train_size]
valid_dataset = df[train_size : train_size+ valid_size]
test_dataset = df[train_size+valid_size:]

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, lower=False, char_level=False, oov_token=None)
tokenizer.fit_on_texts(train_dataset['text'])

train_seqs = tokenizer.texts_to_sequences(train_dataset['text'])
valid_seqs = tokenizer.texts_to_sequences(valid_dataset['text'])
test_seqs = tokenizer.texts_to_sequences(test_dataset['text'])

vocab_size = len(tokenizer.word_index)+1

train_seqs = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding="post", maxlen=MAX_LEN)
valid_seqs = tf.keras.preprocessing.sequence.pad_sequences(valid_seqs, padding="post", maxlen=MAX_LEN)
test_seqs = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding="post", maxlen=MAX_LEN)

train_labels = np.array(train_dataset['label']).astype(np.float32)
valid_labels = np.array(valid_dataset['label']).astype(np.float32)
test_labels = np.array(test_dataset['label']).astype(np.float32)
```

### 模型架构
情感分析模型通常采用基于注意力的机制，其中通过注意力机制建立起输入文本和分类标签之间的关联。具体的模型结构为：词嵌入层->位置编码层->注意力层->全连接层->激活层->输出层。

```python
class TransformerSentimentAnalysis(tf.keras.Model):
    def __init__(self, d_model, num_heads,dff, rate=0.1):
        super(TransformerSentimentAnalysis, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_encoding = positional_encoding(max_len, d_model)
        self.dec_layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=rate)
        self.ffn = point_wise_feed_forward_network(d_model,dff)
        self.dense = tf.keras.layers.Dense(units=1, activation='sigmoid')
        
    def call(self, inputs, training, look_ahead_mask):
        seq_len = tf.shape(inputs)[1]
        attention_weights = {'decoder_layer{}_block{}'.format(idx,i):tf.zeros(((batch_size,num_heads,seq_len,seq_len))) for idx in range(num_layers) for i in range(1,num_blocks)}
        out = self.embedding(inputs) 
        out *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        out += self.pos_encoding[:, :seq_len, :]
        for i in range(num_layers):
            inp = out 
            out, block1, block2 = self.dec_layer([out, out, out, look_ahead_mask])  
            out = self.dropout(out, training=training) 
            attention_weights['decoder_layer{}_block1'.format(i+1)]=block1 
            attention_weights['decoder_layer{}_block2'.format(i+1)]=block2 
            out = self.ffn(out) 
            out = self.dropout(out, training=training) 
        
        outputs = self.dense(out) 
                
        return outputs, attention_weights
```

### 训练
```python
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(real, pred):
    accuracies = tf.equal(tf.round(pred), real)
    return tf.reduce_mean(tf.cast(accuracies, tf.float32))

train_dataset = tf.data.Dataset.from_tensor_slices((train_seqs, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_seqs, test_labels)).batch(BATCH_SIZE)
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_seqs, valid_labels)).batch(BATCH_SIZE)

model = TransformerSentimentAnalysis(d_model=512, num_heads=8, dff=2048)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.BinaryAccuracy(name='valid_accuracy')

@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    attentions_dict = {} 
    with tf.GradientTape() as tape:
        predictions, attention_weights = model(inp, training=True, look_ahead_mask=create_look_ahead_mask(tf.shape(tar)[1]))
        loss = loss_function(tar_real, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)      
    train_accuracy(tar_real, predictions)   

@tf.function
def test_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    attentions_dict = {}     
    predictions, attention_weights = model(inp, training=False, look_ahead_mask=create_look_ahead_mask(tf.shape(tar)[1]))
    loss = loss_function(tar_real, predictions)
    valid_loss(loss)      
    valid_accuracy(tar_real, predictions)  

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    
    for step,(inp, tar) in enumerate(train_dataset):
      train_step(inp, tar)
      
    for step,(inp, tar) in enumerate(valid_dataset):
      test_step(inp, tar)
      
    print ('Epoch {}, Training Loss: {:.4f}, Accuracy: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'
          .format(epoch+1, train_loss.result(), train_accuracy.result(), valid_loss.result(), valid_accuracy.result()))
```

### 测试
```python
def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = tokenizer.sequences_to_texts([sentence])[0]

    attention = tf.squeeze(attention['decoder_layer{}_block1'.format(layer)], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence)+2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result)-1.5, -0.5)

        ax.set_xticklabels(['<start>']+[tokenizer._index_word[i] for i in inp]+['<end>'],
                           rotation=90, fontdict=fontdict)

        ax.set_yticklabels([tokenizer._index_word[i] for i in tar_real], fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head+1))

    plt.tight_layout()
    plt.show()

def translate(sentence, plot=''):
    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']

    inputs = tf.expand_dims(tokenizer.texts_to_sequences([sentence])[0], 0)

    result = [[start_token]]
    attention_plot = np.zeros((MAX_LENGTH_FR, MAX_LENGTH_EN))

    for i in range(MAX_LENGTH_FR):
        predictions, attention_weights = model(inputs, training=False, look_ahead_mask=create_look_ahead_mask(tf.shape(inputs)[1]))

        attention_plot[i] = tf.squeeze(attention_weights['decoder_layer{}_block1'.format(num_layers)])

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

        result.append(predicted_id)

        if predicted_id == end_token or len(result[0]) > MAX_LENGTH_EN:
            break

        inputs = tf.expand_dims([predicted_id], 0)

    translation = tokenizer.sequences_to_texts(result)[0]

    if plot=='':
        return translation
    else:
        plot_attention_weights(attention_plot, sentence, result[-1:], num_layers)
        
translate("The movie was awesome!")
```