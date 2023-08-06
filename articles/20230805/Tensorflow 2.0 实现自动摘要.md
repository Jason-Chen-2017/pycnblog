
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2020年是深度学习元年，它与机器学习、图像识别、自然语言处理等领域密切相关。而在深度学习的应用场景中，无论是文本理解、图像分类、视频分析还是音频分析，都离不开深度神经网络模型的帮助。为了加速文本理解过程，自动摘要算法应运而生。自动摘要（Automatic summarization）就是从一段长文档中生成短而精的摘要。简而言之，就是把长文档中的关键信息提炼出来，并输出给读者阅读或转发。因此，自动摘要具有很高的社会价值。
         
         在本文中，我们将会介绍一种基于Seq2Seq（序列到序列）模型的自动摘要方法—— Pointer-Generator Networks（指针生成网络），并基于TensorFlow 2.0框架实现自动摘要功能。我们首先回顾一下基本概念。
         
         # 2.基本概念和术语
         ## 2.1 Seq2Seq模型
         Seq2Seq模型（Sequence to Sequence，缩写为Seq2seq）是最简单且最流行的无监督学习模型，其特点是通过一个编码器将输入序列编码为固定长度的上下文向量表示，然后用解码器将上下文向量表示解码成目标序列。比如，在机器翻译任务中，输入序列为源语言句子，输出序列为目标语言句子。编码器的输出可以作为解码器的初始状态，对后续输出进行指导。如下图所示：
         
        
        Seq2seq模型主要由两个部分组成：编码器和解码器。编码器将输入序列编码为固定长度的上下文向量表示；解码器根据上下文向量表示生成目标序列。在训练过程中，编码器和解码器一起被训练，使得生成的输出序列尽可能贴近原始的输入序列。
         
        ## 2.2 Pointer Network
         另一种用于抽取上下文信息的方法是Pointer Network。该方法使用两种类型的注意力机制，即内容关注（content-based attention）和通用关注（generalized attention）。其中，内容关注利用输入序列中的当前元素及其之前的元素来重建原始序列；而通用关注则使用任意的历史元素来重建原始序列。以下图为例，说明这种方式的作用。
         
         
        如上图所示，左边的输入序列中有一个词“I”需要抽取上下文信息，右边的输出序列中同样也有一个词“I”。显然，左边的序列比较长，无法直接展示整个上下文信息，只能依次查看每个元素的重要性。使用内容关注时，仅考虑“I”之后的“am”，“s”等关键词，然后将这些词按重要性排列。在输出序列中生成对应的词。此外，还可以使用通用关注，即在每个元素的向量表示上添加一个权重向量，代表元素在历史上出现的概率，这样就可以直接计算出任意时刻元素的上下文分布情况，并选择对应时刻的上下文向量表示。
         
        # 3.核心算法原理和具体操作步骤
        ## 3.1 数据准备
        本项目采用维基百科数据集，该数据集共包括4个文件：
         1. `train.txt`：训练数据，包含若干篇带标签的维基百科文章
         2. `val.txt`：验证数据，包含若干篇带标签的维基百科文章
         3. `test.txt`：测试数据，包含若干篇带标签的维基百科文章
         4. `vocab.pkl`：词汇表文件，存储了所有出现过的单词及其索引
         
         
        假设我们的目的是做自动摘要任务，那么我们可以从`train.txt`中抽取一篇文章作为输入，生成它的摘要作为输出，使用`val.txt`中剩余的若干篇文章作为验证集，以确定模型是否过拟合。因此，输入序列就是`train.txt`的一篇文章，目标序列是自动生成的摘要。
        
        我们将上述四个文件放到指定目录下，并创建一个名为`data`的文件夹。接着，我们定义一些全局变量：
        
        ```python
        data_dir = 'data'   # 文件所在文件夹名称
        train_file = os.path.join(data_dir, 'train.txt')    # 训练数据文件路径
        val_file = os.path.join(data_dir, 'val.txt')      # 验证数据文件路径
        test_file = os.path.join(data_dir, 'test.txt')     # 测试数据文件路径
        vocab_file = os.path.join(data_dir, 'vocab.pkl')   # 词汇表文件路径
        max_len = 30          # 每个句子的最大长度
        padding_id = 0        # pad id
        start_id = 1          # start id
        end_id = 2            # end id
        num_steps = 30        # 每个batch的大小
        batch_size = 32       # batch大小
        embedding_dim = 50    # embedding维度
        hidden_units = 100    # LSTM隐藏单元个数
        dropout_rate = 0.2    # dropout率
        lr = 0.001            # 学习率
        epochs = 20           # epoch数
        save_dir = './save/'  # 模型保存文件夹
        ```
        
        
        ## 3.2 数据预处理
        在构建词典前，我们先对原始文本进行预处理。预处理包括：
         * 分割每一行文本成为句子列表
         * 过滤掉句子列表中较短的句子
         * 对每个句子进行词频统计，去除低频词
         * 创建一个词汇表字典，记录每个词及其对应的索引
        
        ```python
        def preprocess():
            sentences = []
            
            with open(train_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f]
                
            for line in lines:
                sentence_list = line.split('    ')[1].split('. ')
                if len(sentence_list[-1]) < 5 or len(sentence_list)<2:
                    continue
                
                sentences += sentence_list
                
            
            counter = Counter([word for sent in sentences for word in sent.lower().split()])
            words = sorted([word for word, cnt in counter.items() if cnt >= 5], key=lambda x: -counter[x])
            word_to_idx = {w: i+3 for i, w in enumerate(words)}
            
        
            return sentences, word_to_idx
                
        sentences, word_to_idx = preprocess()
        print('总句数:', len(sentences))
        print('总词数:', len(word_to_idx))
            
        ```
        
        上述代码首先读取`train.txt`，分割成句子列表，过滤掉较短的句子或者只有一句的句子，并对每个句子进行词频统计，保留次数大于等于5的单词。然后将单词映射到索引号，分别设置为pad为0，start为1，end为2。最后打印句子数和词数。
        
        
        ## 3.3 数据加载
        下面，我们定义了一个生成器函数，用来构造训练数据的batch，并且使用padding策略来确保每个句子的长度相同。
        
       ```python
        def gen():
            while True:
                X_sents, Y_sents = [], []

                for _ in range(num_steps):
                    
                    idx = np.random.randint(low=0, high=len(sentences), size=(batch_size,))

                    src_sents = [sentences[i][:max_len] + ['']*(max_len-min(map(len, sentences[i])))
                                 for i in idx]
                            
                        
                    trg_sents = [[word_to_idx[word] for word in sent.lower().split()] for sent in sentences[idx]]

                    input_ids = [[word_to_idx['<start>']] + ids[:max_len-2]+[word_to_idx['<end>']]
                                  for ids in src_sents]
                    


                    target_ids = [ids[:max_len-1]+[word_to_idx['<end>']] for ids in trg_sents]

                    input_mask = tf.cast((tf.math.not_equal(input_ids, padding_id)), dtype=tf.int32)[..., None]

                    yield (np.array(input_ids), np.array(target_ids), np.array(input_mask))
    

        dataset = tf.data.Dataset.from_generator(gen,
                                                 output_types=(tf.int32, tf.int32, tf.int32),
                                                 output_shapes=((None, max_len), (None, max_len), (None, max_len)))


        dataset = dataset.padded_batch(batch_size,
                                        padded_shapes=([None, max_len],[None, max_len],[None, max_len]), 
                                        padding_values=(padding_id, padding_id, 0))
        
    
        iterator = iter(dataset)
        
      
        ```
        
        此处，我们使用`tf.data.Dataset.from_generator()`函数生成一个数据集，其中包括三个张量：`input_ids`、`target_ids`、`input_mask`。`input_ids`存放的每个句子的token id，`target_ids`存放的每个句子的目标token id，`input_mask`存放的每个句子的有效token的mask。这里使用padding策略来确保每个句子的长度相同，其长度为`max_len`。
        
        当调用`dataset.padded_batch()`函数时，会自动填充所有的输入样本，使得它们的形状保持一致，所以不会出现`OutOfRangeError`错误。
        
        使用迭代器，将生成器转换为可迭代对象。
        
        
        
        ## 3.4 模型搭建
        在这一步，我们创建了一个`TransformerEncoder`类，该类继承于`tf.keras.layers.Layer`类，用于实现Transformer的encoder层，包括多头注意力模块和位置向量编码模块。在此基础上，我们创建了`TransformerDecoder`类，用于实现Transformer的decoder层，包括多头注意力模块和位置向量解码模块。
         
        ```python
        class TransformerEncoder(tf.keras.layers.Layer):

            def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
                super(TransformerEncoder, self).__init__(**kwargs)
                self.embed_dim = embed_dim
                self.dense_dim = dense_dim
                self.num_heads = num_heads
                self.attention = MultiHeadAttention(self.embed_dim, self.num_heads)
                self.dense_proj = tf.keras.layers.Dense(self.dense_dim, activation='relu')
                self.layernorm_1 = tf.keras.layers.LayerNormalization()
                self.layernorm_2 = tf.keras.layers.LayerNormalization()
                self.supports_masking = True
                

            def call(self, inputs, mask=None, training=True):
                
                seq_length = tf.shape(inputs)[1]
                
                attention_output = self.attention(inputs, inputs, attention_mask=mask)
                
                proj_input = self.layernorm_1(inputs + attention_output)
                
                proj_output = self.dense_proj(proj_input)
                
                return self.layernorm_2(proj_input + proj_output)
                
                
              
        class TransformerDecoder(tf.keras.layers.Layer):

            def __init__(self, embed_dim, latent_dim, num_heads, attn_dropout=0.2, projection_dropout=0.2, forward_expansion=1, **kwargs):
                super(TransformerDecoder, self).__init__(**kwargs)
                self.embed_dim = embed_dim
                self.latent_dim = latent_dim
                self.num_heads = num_heads
                self.attn_dropout = attn_dropout
                self.projection_dropout = projection_dropout
                self.forward_expansion = forward_expansion
                self.attention = MultiHeadAttention(self.embed_dim, self.num_heads)
                self.masked_attention = MultiHeadAttention(self.latent_dim, self.num_heads)
                self.embedding = tf.keras.layers.Embedding(input_dim=len(word_to_idx)+3, output_dim=self.embed_dim, name="dec_embedding")
                self.pos_encoding = PositionalEncoding(self.embed_dim, name='position_embedding')
                self.dense_proj = tf.keras.layers.Dense(self.latent_dim*self.forward_expansion, kernel_initializer='glorot_uniform', activation='linear')
                self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                self.ffn_dropout = tf.keras.layers.Dropout(self.attn_dropout)
                self.ffn_layer = tf.keras.layers.Dense(self.latent_dim, activation='relu')
                self.supports_masking = True
                


            def call(self, inputs, encoder_outputs, look_ahead_mask=None, padding_mask=None, decoder_history_mask=None, training=True):
                seq_length = tf.shape(inputs)[1]
                attention_weights = {}
                
                dec_embedding = self.embedding(inputs)
                
                position_embedding = self.pos_encoding(tf.range(start=0, limit=seq_length, delta=1))
                
                outputs = self.layernorm_1(dec_embedding + position_embedding)
                
                att_output = self.attention(query=outputs[:, :-1, :], value=outputs[:, :-1, :], key=outputs[:, :-1, :], attention_mask=look_ahead_mask)
                
                att_output = tf.concat([att_output,outputs[:, -1:, :]], axis=-2)
                
                out1 = self.layernorm_2(outputs + att_output)
                
                ffn_output = self.dense_proj(out1)
                
                ffn_output = self.ffn_layer(ffn_output)
                
                ffn_output = self.ffn_dropout(ffn_output, training=training)
                
                predictions = self.layernorm_2(ffn_output + out1)
                
                masked_tokens = decoder_history_mask[..., tf.newaxis]*predictions
                
                return predictions, masked_tokens, attention_weights
            
            

            
            
            
            
        ```
        以上代码实现了两层的Transformer Encoder和Transformer Decoder，并使用多头注意力机制实现自注意力和互注意力，并使用残差连接、层标准化和多层感知机完成编码器和解码器之间的连接。

        
        ## 3.5 Pointer Generator Network
        在Pointer-Generator Networks中，编码器的输出不再是固定的长度的上下文向量，而是变长的上下文向量的集合。解码器根据当前的词和已生成的词生成下一个词，但是只允许解码生成的词，而不是像传统的Seq2Seq模型一样完全生成目标序列。其基本思想是在解码过程中同时生成当前词和下一个词的概率，解码器根据这个概率选择生成当前词还是下一个词，并通过指针机制来帮助解码器准确地选择那个词。
        
        以机器翻译任务为例，在解码阶段，解码器生成当前词后，通过搜索的方式找到该词的候选词，并生成相应概率分布。如下图所示。
        
         
        
        由于PtrNet只允许生成已经生成的词，因此训练PtrNet通常比训练普通的Seq2Seq模型更困难。因此，在实际应用中，往往需要结合其他模型，例如语言模型或编码器来增强PtrNet的性能。例如，在中文机器翻译任务中，结合BERT模型提升PtrNet的性能。
        
        ```python
        class PointerGeneratorModel(tf.keras.Model):

            def __init__(self, embed_dim, latent_dim, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, backward_expansion, dropout_rate, vocab_size, tokenizer, **kwargs):
                super(PointerGeneratorModel, self).__init__(**kwargs)
                self.tokenizer = tokenizer
                self.encoder = TransformerEncoder(embed_dim, latent_dim, num_heads, num_layers=num_encoder_layers)
                self.decoder = TransformerDecoder(embed_dim, latent_dim, num_heads, attn_dropout=dropout_rate, projection_dropout=dropout_rate, forward_expansion=forward_expansion, num_layers=num_decoder_layers)
                self.enc_output = tf.keras.layers.Dense(latent_dim, activation=None)(self.encoder.output)
                self.logits_bias = tf.Variable([[0.] * vocab_size], dtype=tf.float32, name='logit_bias')
                self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[self.tokenizer.get_vocab_vector()], trainable=False, name='tgt_embedding')
                self.lm_head = LMHead(latent_dim, vocab_size, bias=False)
                self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')(self.enc_output)
                

            @staticmethod
            def create_masks(input_ids, padding_id):
                seq_length = tf.shape(input_ids)[1]
                
                look_ahead_mask = create_look_ahead_mask(tf.ones((seq_length, seq_length)), name='look_ahead_mask')
                
                dec_target_padding_mask = tf.cast(tf.math.equal(input_ids, padding_id), dtype=tf.int32)[..., tf.newaxis, :]
                
                combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
                
                return {'dec_target_padding_mask': dec_target_padding_mask, 
                        'combined_mask': combined_mask}
                

            def encode(self, inputs, training=True):
                enc_padding_mask = get_padding_mask(inputs)
                
                enc_outputs = self.encoder(inputs, mask=enc_padding_mask, training=training)
                
                return self.enc_output(enc_outputs), self.classifier(enc_outputs), enc_padding_mask
                
                
            def decode(self, tgt_ids, enc_output, memory_states, sos_id, eos_id, padding_id, teacher_forcing=False, training=True):
                state = MemoryState(memory_state=memory_states, prev_timestep=sos_id)
            
                logits = tf.zeros((tf.shape(tgt_ids)[0], 0, len(word_to_idx)))
                
                for t in range(tf.shape(tgt_ids)[1]):
                    dec_input = self.embedding(state.prev_timestep)
                    
                    context_vec, attention_weights = self.decoder(inputs=dec_input,
                                                                   encoder_outputs=enc_output,
                                                                   look_ahead_mask=create_look_ahead_mask(state.prev_timestep[..., None])[..., 0],
                                                                   padding_mask=create_padding_mask(tgt_ids, padding_id),
                                                                   decoder_history_mask=create_decoder_history_mask(tf.shape(tgt_ids)[1]-1, state.prev_timestep),
                                                                   training=training)
                    
                    pointer_probs = self.lm_head(context_vec).logits[:,:,:-3]/temperature + tf.nn.log_softmax(logits[:, :, :-3])
                    
                    next_token_probs = pointer_probs[:, -1, :]
                    
                    if training and teacher_forcing:
                        next_token = tgt_ids[:, t]
                    else:
                        next_token = tf.argmax(next_token_probs, axis=-1, output_type=tf.int32)
                        
                        if tf.reduce_all(tf.math.equal(next_token, sos_id)):
                            break
                            
                        elif tf.reduce_any(tf.math.equal(next_token, padding_id)):
                            non_padding_indices = tf.where(tf.math.not_equal(next_token, padding_id))[..., 0]
                            
                            next_token = tf.gather(next_token, non_padding_indices, axis=-1)
                            
                            if tf.shape(non_padding_indices)[0]==0:
                                raise ValueError("The generated sequence contains only padding tokens.")
                            
                            
                            
                            
                    
                    if not isinstance(next_token, int):
                        next_token = tf.expand_dims(next_token, axis=-1)
                        
                    token_probs = tf.gather(next_token_probs, next_token, axis=-1, batch_dims=1)*tf.constant([-1e10]*state.prev_timestep.shape[1], shape=[1,-1])*tf.reshape(teacher_probs[:, t], (-1, 1))
                    token_probs = tf.tensor_scatter_nd_update(token_probs, indices=tf.stack([tf.range(tf.shape(token_probs)[0]), tf.squeeze(next_token)], axis=-1), updates=tf.fill(shape=tf.shape(token_probs)[0], value=float('-inf')))
                    
                    new_state = MemoryState(prev_timestep=tf.concat([state.prev_timestep, next_token], axis=-1),
                                            memory_state=state.memory_state,
                                            attention_weights={k:v[:,t,:] for k, v in attention_weights.items()},
                                            log_probs=tf.concat([state.log_probs, token_probs], axis=-1))
                    
                    logits = tf.concat([logits, tf.one_hot(next_token, depth=len(word_to_idx)-3)], axis=-1)
                    
                    state = new_state
                    
                return state.memory_state, logits
                
                
            def predict(self, inputs, max_len=20, sos_id=1, eos_id=2, padding_id=0, temperature=1., inference='greedy'):
                enc_output, classifier_score, enc_padding_mask = self.encode(inputs, training=False)
                
                initial_output = tf.fill((tf.shape(inputs)[0], 1), sos_id)
                
                tgt_ids = tf.cond(tf.less(tf.random.uniform([]), 0.5),
                                   lambda: greedy_search(initial_output,
                                                         lambda y: self._decode_with_sampling(y,
                                                                                            enc_output,
                                                                                            self.memory_states,
                                                                                            sos_id,
                                                                                            eos_id,
                                                                                            padding_id,
                                                                                            temperature,
                                                                                            0.),
                                                         max_len=max_len,
                                                         padding_id=padding_id,
                                                         eos_id=eos_id),
                                   lambda: sample_sequence(initial_output,
                                                          lambda y: self._decode_with_sampling(y,
                                                                                             enc_output,
                                                                                             self.memory_states,
                                                                                             sos_id,
                                                                                             eos_id,
                                                                                             padding_id,
                                                                                             temperature,
                                                                                             0.),
                                                          max_len=max_len,
                                                          stop_token=padding_id,
                                                          eos_id=eos_id))
                preds = self.detokenize(tgt_ids)
                
                return preds
                
                
            def detokenize(self, ids):
                return tf.strings.reduce_join(self.tokenizer.convert_ids_to_tokens(ids), separator=' ', axis=-1)
                
                
                
        def compile(model, optimizer, loss_object):
            """Compiles the given model with the provided optimizer and loss object."""
            model.compile(optimizer=optimizer,
                          metrics=['accuracy'],
                          run_eagerly=False,
                          loss={'lm_loss':'sparse_categorical_crossentropy'},
                          loss_weights={'lm_loss':1.})
            
            
        def build_model(**config):
            model = PointerGeneratorModel(vocab_size=len(word_to_idx)+3,
                                          embed_dim=embedding_dim,
                                          latent_dim=hidden_units,
                                          num_heads=8,
                                          num_encoder_layers=4,
                                          num_decoder_layers=4,
                                          forward_expansion=4,
                                          backward_expansion=1,
                                          dropout_rate=dropout_rate,
                                          tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'))
            
            optimizer = tf.keras.optimizers.Adam(lr=lr, clipvalue=1.)
            
            loss_object = SparseCategoricalCrossentropyWithLogits(reduction=Reduction.NONE)
            
            compile(model, optimizer, loss_object)
            
            return model
            
            
        model = build_model()
        
        
        model.fit(dataset,
                  validation_data=(val_data,),
                  epochs=epochs)
                

                
             
        
        ```
        
        此处，我们定义了`PointerGeneratorModel`类，该类是一个`tf.keras.Model`的子类，包括以下组件：
        
         * `encoder`：Transformer编码器
         * `decoder`：Transformer解码器
         * `enc_output`：对编码器的输出进行线性变换
         * `logits_bias`：偏置项
         * `embedding`：目标词嵌入矩阵
         * `lm_head`：语言模型头部
         * `classifier`：分类器头部
         
         方法：
         
         1. `__init__()`：初始化方法
         2. `create_masks()`：创建注意力掩膜
         3. `encode()`：编码器
         4. `decode()`：解码器
         5. `predict()`：推断方法
         6. `detokenize()`：解码成字符串方法
         
         
        ## 3.6 模型训练
        在训练模型之前，我们首先对模型进行编译，其中包括设置优化器和损失函数。然后，我们开始训练模型。模型训练的时候，我们将使用的是NLLLoss来计算语言模型的损失。
        
       ```python
        def main():
            if args.mode == "train":
                train()
            elif args.mode == "eval":
                evaluate()
            elif args.mode == "infer":
                infer()
            else:
                raise ValueError("Invalid mode!")
                
                
                
                
        if __name__=="__main__":
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers(dest="mode", help="Mode of operation.")
    
            train_parser = subparsers.add_parser("train", help="Train the model.")
            train_parser.set_defaults(func=train)
            
            eval_parser = subparsers.add_parser("eval", help="Evaluate the trained model on dev set.")
            eval_parser.set_defaults(func=evaluate)
            
            infer_parser = subparsers.add_parser("infer", help="Infer using pre-trained model.")
            infer_parser.add_argument("--input", type=str, required=True, help="Input text file path.")
            infer_parser.add_argument("--output", type=str, default="", help="Output file path.")
            infer_parser.add_argument("--beam_width", type=int, default=1, help="Beam width used during decoding.")
            infer_parser.add_argument("--sample_temp", type=float, default=1., help="Temperature parameter used during sampling.")
            infer_parser.add_argument("--batch_size", type=int, default=1, help="Batch size used during inference.")
            infer_parser.set_defaults(func=infer)
            
            args = parser.parse_args()
            
            tf.random.set_seed(42)
            main()
            
            
            
            
        def compile(model, optimizer, loss_object):
            """Compiles the given model with the provided optimizer and loss object."""
            model.compile(optimizer=optimizer,
                          metrics=['accuracy'],
                          run_eagerly=False,
                          loss={'lm_loss':'sparse_categorical_crossentropy'},
                          loss_weights={'lm_loss':1.})
            
            
        def build_model(**config):
            model = PointerGeneratorModel(vocab_size=len(word_to_idx)+3,
                                          embed_dim=embedding_dim,
                                          latent_dim=hidden_units,
                                          num_heads=8,
                                          num_encoder_layers=4,
                                          num_decoder_layers=4,
                                          forward_expansion=4,
                                          backward_expansion=1,
                                          dropout_rate=dropout_rate,
                                          tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'))
            
            optimizer = tf.keras.optimizers.Adam(lr=lr, clipvalue=1.)
            
            loss_object = SparseCategoricalCrossentropyWithLogits(reduction=Reduction.NONE)
            
            compile(model, optimizer, loss_object)
            
            return model
                
            
        def train():
            model = build_model()
            
            callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
            
            history = model.fit(train_dataset, 
                                steps_per_epoch=len(train_sentences)//(batch_size*num_steps),
                                validation_data=(val_dataset, ),
                                validation_steps=len(val_sentences)//(batch_size*num_steps),
                                epochs=epochs,
                                callbacks=callbacks)
            
            
            
        def evaluate():
            model = load_model()
            
            pred_labels = model.predict(test_dataset)
            true_labels = [label for label, _, _ in test_examples]
            
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro")
            
            accuracy = sum([pred_labels[i]==true_labels[i] for i in range(len(pred_labels))])/len(pred_labels)
            
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print("Accuracy:", accuracy)
            
            
            
        def infer():
            beam_width = args.beam_width
            sample_temp = args.sample_temp
            batch_size = args.batch_size
            
            model = load_model()
            
            output_lines = []
            with open(args.input, "r", encoding="utf-8") as fin:
                for block in blocks(fin, batch_size):
                    inputs = [" ".join(["[CLS]"]+tokenizer.tokenize(line.strip())[:max_len-2]+["[SEP]"]) for line in block]
                    inputs = tokenizer.batch_encode_plus(inputs, add_special_tokens=False)["input_ids"]
                    
                    encodings = [{"input_ids": input_ids, "attention_mask": [1]*len(input_ids)} for input_ids in inputs]
                    
                    results = model.generate(encodings,
                                             use_cache=True,
                                             num_beams=beam_width,
                                             no_repeat_ngram_size=3,
                                             min_length=10,
                                             max_length=30,
                                             early_stopping=True,
                                             do_sample=True,
                                             top_p=0.9,
                                             temperature=sample_temp,
                                             length_penalty=1.0,
                                             no_repeat_ngram_size=3,)
                    
                    for result in results:
                        output_lines.append(tokenizer.decode(result.numpy(), skip_special_tokens=True)+"
")
                        
            if args.output:
                with open(args.output, "w", encoding="utf-8") as fout:
                    fout.writelines(output_lines)
            else:
                print("
".join(output_lines))
            
            
            
        if __name__=="__main__":
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers(dest="mode", help="Mode of operation.")
    
            train_parser = subparsers.add_parser("train", help="Train the model.")
            train_parser.set_defaults(func=train)
            
            eval_parser = subparsers.add_parser("eval", help="Evaluate the trained model on dev set.")
            eval_parser.set_defaults(func=evaluate)
            
            infer_parser = subparsers.add_parser("infer", help="Infer using pre-trained model.")
            infer_parser.add_argument("--input", type=str, required=True, help="Input text file path.")
            infer_parser.add_argument("--output", type=str, default="", help="Output file path.")
            infer_parser.add_argument("--beam_width", type=int, default=1, help="Beam width used during decoding.")
            infer_parser.add_argument("--sample_temp", type=float, default=1., help="Temperature parameter used during sampling.")
            infer_parser.add_argument("--batch_size", type=int, default=1, help="Batch size used during inference.")
            infer_parser.set_defaults(func=infer)
            
            args = parser.parse_args()
            
            tf.random.set_seed(42)
            main()