
作者：禅与计算机程序设计艺术                    

# 1.简介
  

BERT（Bidirectional Encoder Representations from Transformers）由谷歌提出，是一种基于transformer的自编码网络(Autoencoder)，其通过训练数据学习到不同位置之间的关系并进行信息编码。BERT能够自动提取输入序列的语义特征，使得无监督机器学习任务、文本分类、语言推断等都变得简单易用。近年来，BERT已被应用于众多领域，包括自然语言处理、情感分析、文本生成、问答系统、机器翻译、命名实体识别、摘要生成等多个NLP任务。本文将基于BERT模型结构，用tensorflow2.x实现了一个轻量级版本的BERT。


# 2.知识储备
- 对Transformer及BERT的了解；
- tensorflow2.x的相关基础知识；
- NLP的相关经验。

# 3.核心知识点解析
## 3.1 BERT概述

BERT(Bidirectional Encoder Representations from Transformers)模型最初由<NAME>，Jay Alammar，Devlin et al.在2018年发表，是一种自监督语言理解模型。它是一个基于Transformer的预训练语言模型，能够对深层上下文信息进行编码，可以用于各种下游任务，如文本分类、序列标注、问答等。其中Embedding层采用词嵌入矩阵，编码器层使用多头注意力机制(Multi-Head Attention)进行编码，使用双向RNN作为变换函数，通过全连接层输出预测结果。


BERT的主要特点有以下几点:
- 使用Transformer作为编码器模块，通过引入Self-Attention方法处理长距离依赖，能够捕获全局、局部信息并保证模型鲁棒性；
- 通过预训练的方式，能够学习到不同层次的上下文表示，提升模型的泛化能力；
- 可以训练任务特定参数，并结合微调来进一步提升性能；
- 模型大小小，可以在不同资源条件下快速部署，且可以支持丰富的下游任务。

## 3.2 BERT的结构

BERT模型由Encoder和Decoder两部分组成。Encoder负责编码输入序列，由词嵌入层、位置编码层、堆叠的Transformer层和投影层组成。每一层的输出是上一层的输入和当前层的输出做拼接后得到，最后得到整个句子或者段落的隐含表示。Decoder则用来进行预测或执行其他序列到序列的任务，根据不同的任务，Decoder的结构也会有所不同。但是总体来说，BERT中所有参数都是可学习的。





## 3.3 Transformer及BERT的工作方式

Transformer是Google 2017年提出的一种完全基于Attention的神经网络模型，它不仅可以处理序列建模，还可以看作是一种计算模型。基于这种计算模型的Transformer，可以有效解决长距离依赖的问题，并取得很好的效果。而BERT就是利用Transformer做预训练，进而训练出能同时兼顾通用性和特殊性的模型。BERT的预训练任务分为两步：Masked Language Modeling (MLM)和Next Sentence Prediction (NSP)。下面详细介绍一下BERT的预训练过程。


### Masked Language Modeling (MLM)

MLM的目标是使模型能够正确地预测原始输入序列中的缺失位置的单词。如下图所示，假设原始输入序列为“The quick brown fox jumps over the lazy dog”，在BERT模型训练之前，我们随机遮盖一些位置，如[MASK]替换成“The”。然后，BERT模型需要学习如何预测这些位置中的单词，也就是把它们还原成真实值。



MLM训练时，首先随机选择一批句子，然后按照一定概率替换掉其中某个词，让该词被预测成为[MASK]。例如，若随机选择的词是“fox”这个词，那么被替换的词可能是“jumps”、“lazy”或“dog”。这样做的好处是，MLM不会修改原始输入序列的内容，而只是让模型去预测缺失的单词。另外，为了防止模型过度拟合，每次MLM训练都会有一定的正负样本比例，从而更好地训练模型。


### Next Sentence Prediction (NSP)

NSP的目标是判断两个句子之间是否有连贯性，即前者的主干是不是后者的衍生。如下图所示，假设我们有两个句子A和B，BERT模型需要学习判断两个句子之间是否存在逻辑关系。NSP训练时，两个句子的顺序不做任何改动，但是将两句话合并成一个文本作为BERT模型的输入，其中句子A被打上了[CLS]标记，句子B被打上了[SEP]标记。



NSP训练时，也会随机选择一批句子，并组合成两个句子一起训练。例如，若随机选择的句子A是“The quick brown fox”，句子B是“The lazy dog was jumping”的话，则模型的输入是“[CLS] The quick brown fox [SEP] The lazy dog was jumping [SEP]”。这样做的目的是让模型更容易学到句子之间的相关性。另外，由于NSP训练时只有两句话的位置信息，所以模型并不能捕获全局信息。因此，训练结束后，将输入分开，分别输入模型，模型将会输出是否相连的信息。


### BERT模型的训练

以上，我们介绍了BERT的两种预训练任务——MLM和NSP。在预训练过程中，模型参数会随着时间不断更新，直至收敛。当模型训练完成之后，我们就可以用它来完成各个任务的预测。但这样做的代价是耗费更多的时间和计算资源，并且可能会降低模型的泛化能力。因此，一般情况下，我们会继续微调模型，将权重微调或裁剪，以适应特定任务。下面是BERT的训练过程：

- Step1: 在大规模语料库上预训练BERT模型，同时训练MLM和NSP两个任务；
- Step2: 在微调阶段，对MLM和NSP任务进行微调，并固定其他任务的参数；
- Step3: 用最终微调后的BERT模型预测任务。


# 4.代码实现

下面，我们基于BERT模型结构，用TensorFlow 2.x构建一个轻量级版本的BERT。下面给出基于TF2.x的BERT模型的定义及训练代码。


```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

class BertModel(keras.Model):
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 hidden_dim=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-12,
                 **kwargs):
        super(BertModel, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.epsilon = epsilon
        
        # Word embedding layer
        self.embedding = keras.layers.Embedding(input_dim=self.vocab_size+1,
                                                 output_dim=self.hidden_dim,
                                                 name="word_embeddings")
        # Positional encoding layer
        self.position_encoding = self._positional_encoding()
        
        # Encoder layers
        encoder_layers = []
        for i in range(self.num_hidden_layers):
            encoder_layers.append(
                keras.layers.Dense(units=self.hidden_dim,
                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                       stddev=self.initializer_range),
                                   name="dense_%d"%(i+1))
            )
            if i!= self.num_hidden_layers - 1:   # do not add a dropout layer after the last dense layer
                encoder_layers.append(
                    keras.layers.Dropout(rate=self.dropout_rate,
                                         name="dropout_%d"%(i+1)))
            else:
                pass    # we need to keep the original inputs for the first attention block
            
        self.encoder_layers = encoder_layers
        
        # Final linear transformation before softmax activation
        self.final_layer = keras.layers.Dense(units=self.vocab_size,
                                               use_bias=False,
                                               kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                   stddev=self.initializer_range),
                                               name="predictions")
        

    def call(self,
             input_ids,
             attention_mask=None,
             token_type_ids=None,
             training=True):
        
        batch_size = tf.shape(input_ids)[0]

        # Embeddings with positional encodings
        embeddings = self.embedding(input_ids) + self.position_encoding[:, :self.max_seq_len,:]

        # Run through each encoder layer and normalize the output of the transformer block
        for idx, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x, training=training)
            
        # Apply final linear transformation
        logits = self.final_layer(x, training=training)
                
        return logits


    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def _positional_encoding(self):
        position_enc = np.array([self._get_angles(pos, i, self.hidden_dim) for pos in range(self.max_seq_len) for i in range(self.hidden_dim)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # apply sin on even indices in the array
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # apply cos on odd indices in the array
        pad_row = np.zeros((1, self.hidden_dim))
        position_enc = np.concatenate([pad_row, position_enc], axis=0)
        return tf.constant(value=position_enc, shape=[self.max_seq_len+1, self.hidden_dim])

```

# 5.训练

## 5.1 数据准备

BERT模型的训练，需要大量的大规模语料库。作者采用的数据集为wikihow中文数据集。

```
wget https://www.dropbox.com/s/l0n9kp0stkg4zfd/wikiHowAll.txt.tar.gz?dl=0
tar zxf wikiHowAll.txt.tar.gz\?dl\=0 --strip-components=1
```

该数据集共计约54万篇文章，每篇文章平均长度约为1000字符。我们需要将文章转换成可以输入到BERT模型中的序列形式。


```python
def preprocess_text(sentence):
    sentence = ''.join(c for c in sentence if ord(c)<128) # remove non ASCII characters
    sentence =''.join(word.lower().replace("'", "") for word in sentence.split())
    words = sentence.split(' ')
    tokens = []
    
    for word in words:
        if word == '<unk>' or len(word)==0 or all(ord(char)>128 for char in word): continue
        tokens.append('[unused'+str(len(tokens))+']')
            
    return tokens
    
def create_examples(sentences):
    examples = []
    for sidx, sentence in enumerate(sentences):
        text = preprocess_text(sentence['title'])[:MAX_SEQ_LEN-2]
        label = int(sentence['id'].split('_')[0])
        
        example = InputExample(guid="%s-%d"%('train', sidx), text=text, label=label)
        examples.append(example)
        
    return examples
```

我们只保留标题部分，并将其截断为指定长度MAX_SEQ_LEN-2，并用[unusedi]表示第i个待预测的token。

```python
from sklearn.utils import shuffle
from collections import defaultdict

data = open('./wikiHowAll.txt').read().strip().split('\n')[:-1]
data = [{'title': line.split('\t')[0].strip(), 'id': line.split('\t')[1]} for line in data]

MAX_SEQ_LEN = 128

all_titles = [preprocess_text(line['title'])[:MAX_SEQ_LEN-2] for line in data]
all_labels = [int(line['id'].split('_')[0]) for line in data]
total_count = sum(1 for l in all_labels)
print("Total count:", total_count)

train_data = create_examples([{k: v for k, v in dict(zip(('title','id'), item)).items()} for item in zip(all_titles, all_labels)][:int(.8*total_count)], MAX_SEQ_LEN)
valid_data = create_examples([{k: v for k, v in dict(zip(('title','id'), item)).items()} for item in zip(all_titles, all_labels)][int(.8*total_count):int(.9*total_count)], MAX_SEQ_LEN)
test_data = create_examples([{k: v for k, v in dict(zip(('title','id'), item)).items()} for item in zip(all_titles, all_labels)][int(.9*total_count):], MAX_SEQ_LEN)
```

这里创建了三种类型的数据集：训练集、验证集和测试集，每个数据集中，包含一个标题和一个标签，标签对应于从1到10类别的10分类问题。

```python
print(len(train_data), "train examples created.")
print(len(valid_data), "validation examples created.")
print(len(test_data), "test examples created.")
```

## 5.2 训练脚本编写

首先，我们定义一些超参数。

```python
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
NUM_EPOCHS = 10
MAX_GRADIENT_NORM = 1.0
WARMUP_STEPS = int(0.1*len(train_data)/BATCH_SIZE)*NUM_EPOCHS
SAVE_EVERY_N_EPOCHS = 1
USE_TPU = False
```

然后，我们编写训练脚本。如果USE_TPU为True，则使用TPU训练，否则，使用GPU训练。

```python
if USE_TPU:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
else:
    strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = BertModel(vocab_size=VOCAB_SIZE,
                      max_seq_len=MAX_SEQ_LEN,
                      num_hidden_layers=CONFIG["num_hidden_layers"],
                      hidden_dim=CONFIG["hidden_size"],
                      num_attention_heads=CONFIG["num_attention_heads"])
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE, epsilon=1e-08)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(inputs):
    features, labels = inputs
    with tf.GradientTape() as tape:
        outputs = model(features, training=True)
        loss = loss_fn(y_true=labels, y_pred=outputs)
    grads = tape.gradient(loss, model.trainable_variables)
    gradients = [(tf.clip_by_norm(grad, MAX_GRADIENT_NORM)) for grad in grads]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def get_masked_lm_output(bert_inputs, bert_output, positions, label_ids, label_weights):
    """Get predictions for the masked LM objective."""
    bert_output = gather_indexes(bert_output, positions)
    log_probs = tf.nn.log_softmax(bert_output, axis=-1)
    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])
    
    one_hot_labels = tf.one_hot(label_ids, depth=bert_inputs.shape[-1], dtype=tf.float32)
    
    per_example_loss = -tf.reduce_sum(log_probs*one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights*per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator
    return loss


@tf.function
def get_next_sentence_output(bert_inputs, pooled_output, is_random):
    """Compute the loss for predicting whether two sentences are consecutive or not."""
    target_mapping = tf.cast(tf.not_equal(is_random, True), tf.float32)
    pred_logits = tf.matmul(pooled_output, pooled_output, transpose_b=True)
    mask = tf.linalg.band_part(tf.ones([tf.shape(target_mapping)[0], tf.shape(target_mapping)[0]]), -1, 0)
    inv_mask = tf.linalg.band_part(tf.ones([tf.shape(target_mapping)[0], tf.shape(target_mapping)[0]]), 0, -1)
    ignore_mask = tf.math.logical_or(tf.cast(tf.less(tf.range(tf.shape(inv_mask)[0]), tf.shape(inv_mask)[0]-tf.argmax(target_mapping,-1)), tf.bool), tf.eye(tf.shape(target_mapping)[0]))
    mask *= tf.expand_dims(tf.cast(tf.logical_and(~ignore_mask, ~tf.eye(tf.shape(target_mapping)[0])), tf.float32), -1)
    inverse_mask = tf.expand_dims(tf.where(tf.cast(tf.greater(target_mapping, 0.), tf.bool), tf.ones_like(target_mapping), tf.zeros_like(target_mapping)), -1)
    inverse_mask += tf.eye(tf.shape(inverse_mask)[0])*(-1.)
    inverse_mask /= tf.reduce_mean(inverse_mask, axis=-1, keepdims=True)+1e-5
    pred_logits -= 1e30*tf.cast(ignore_mask, tf.float32)
    
    next_sentence_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tf.cast(is_random, tf.int64),
                                                                           logits=pred_logits,
                                                                           weights=tf.squeeze(mask, -1)*tf.squeeze(inverse_mask, -1))
    next_sentence_accuracy = tf.keras.metrics.Accuracy()(tf.argmax(pred_logits, axis=-1, output_type=tf.int32),
                                                         tf.cast(is_random, tf.int32))*tf.reduce_mean(target_mapping)
    return next_sentence_loss, next_sentence_accuracy
```

这里，我们定义了三个输出：
- MLM loss，即masked language modeling的损失，用来预测缺失的token；
- NSP loss，即next sentence prediction的损失，用来判断两个句子是否具有连贯性；
- Accuracy，用来评估NSP任务的准确性。

```python
global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
optimizer = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=LEARNING_RATE, decay_steps=len(train_data)//global_batch_size*NUM_EPOCHS, end_learning_rate=0.0, power=1.0)

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(checkpoint, directory="./ckpt", max_to_keep=3)
if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print("[INFO] restore from %s."%manager.latest_checkpoint)
else:
    print("[INFO] initialize from scratch.")
    
for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    train_data = shuffle(train_data)
    train_dataset = tf.data.Dataset.from_tensor_slices(({k: v for k, v in e.__dict__.items() if type(v)!=list}, e.label) for e in train_data).repeat().shuffle(BUFFER_SIZE).batch(global_batch_size).map(lambda x,y: (x, tf.reshape(tf.convert_to_tensor(y), [-1])))#.prefetch(AUTO)
    valid_dataset = tf.data.Dataset.from_tensor_slices(({k: v for k, v in e.__dict__.items() if type(v)!=list}, e.label) for e in valid_data).repeat().batch(global_batch_size).map(lambda x,y: (x, tf.reshape(tf.convert_to_tensor(y), [-1])))#.prefetch(AUTO)

    dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    dist_valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)
    
    loss_metric = tf.keras.metrics.Mean()
    mlm_loss_metric = tf.keras.metrics.Mean()
    nsp_loss_metric = tf.keras.metrics.Mean()
    accuracy_metric = tf.keras.metrics.Mean()

    global_step = tf.Variable(0, trainable=False)
    learning_rate = optimizer(tf.cast(global_step, tf.float32))
    optimizer.assign(learning_rate)

    iterator = iter(dist_train_dataset)
    steps_per_epoch = len(train_data) // global_batch_size + 1
    for step in range(steps_per_epoch):
        try:
            inputs = next(iterator)
        except StopIteration:
            iterator = iter(dist_train_dataset)
            inputs = next(iterator)

        per_replica_loss = strategy.run(train_step, args=(inputs,))
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, None)
        
        loss_metric.update_state(loss)
        mlm_loss_metric.reset_states()
        nsp_loss_metric.reset_states()
        accuracy_metric.reset_states()

        if step > WARMUP_STEPS:
            elapsed_time = time.time()-start_time
            print("Epoch {}/{}, Step {}, Loss {:.4f} ({:.2f} sec/batch), Learning rate {}".format(epoch+1, NUM_EPOCHS, step+1, loss_metric.result().numpy(), elapsed_time/(step+1), learning_rate.numpy()))
            #if step % SAVE_EVERY_N_EPOCHS==0:
            #    save_path = manager.save()
            #    print("[INFO] Save ckpt file at %s"%save_path)

            with tf.device("/job:localhost"):
                for (batch, (features, labels)) in enumerate(dist_valid_dataset):
                    per_replica_mlm_loss, per_replica_nsp_loss, per_replica_accuracy = strategy.run(get_masked_lm_output, args=(features["input_ids"],
                                                                                                              features["mlm_positions"],
                                                                                                              labels))

                    current_mlm_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_mlm_loss, None)
                    current_nsp_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_nsp_loss, None)
                    current_accuracy = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accuracy, None)
                    
                    mlm_loss_metric.update_state(current_mlm_loss)
                    nsp_loss_metric.update_state(current_nsp_loss)
                    accuracy_metric.update_state(current_accuracy)

                    if batch>=100: break

                print("\tValidation MLM Loss {:.4f}".format(mlm_loss_metric.result().numpy()), flush=True)
                print("\tValidation NSP Loss {:.4f}, Accuracy {:.4f}".format(nsp_loss_metric.result().numpy(), accuracy_metric.result().numpy()))
                
            loss_metric.reset_states()
            mlm_loss_metric.reset_states()
            nsp_loss_metric.reset_states()
            accuracy_metric.reset_states()
```