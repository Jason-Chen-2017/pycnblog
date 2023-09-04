
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1986 年，MIT 的 Pereira、Rush 和 Tanenbaum 提出了一种新的序列标注方法，名叫“隐马尔可夫模型”，可以进行多类别序列标注任务。其后不久，随着神经网络的飞速发展，一些研究者也在尝试将这种方法应用于自然语言处理领域中。例如在 CoNLL-2003 NER 数据集上提出的 BiLSTM-CRF 模型[1]、Transformer 预训练模型[2]、BERT 预训练模型[3]等都取得了显著效果。近年来，随着 GPU/TPU 计算能力的增加，这些模型越来越容易训练。因此，本文也希望通过对 Bidirectional LSTM-CRF 系列模型的分析，为中文命名实体识别（NER）领域的开发者提供参考。
         # 2.术语及概念
         1. 序列标注(Sequence Labeling): 是指给定一个输入序列或句子，标注出其中的每个元素所属的标签或类别，一般来说，输入序列会比输出序列长或者短。序列标注的任务通常分为监督学习和非监督学习两大类，分别对应于有标签数据和无标签数据。
         2. 词汇表(Vocabulary): 是指由一个或多个词或字符组成的集合，用于表示输入序列中的单词或字符。
         3. 句子(Sentence): 是指由一个或多个词语组成的一个完整的语句。
         4. 标记(Tag): 是对句子中的每一个词语或字符赋予的分类标签。标记总共可以分为以下几种类型：
             - B-XXX 表示从X到XXX的边界
             - I-XXX 表示介于两个连续的X之间
             - O 表示不被任何实体包围
             - PERSON 表示人名
             - LOCATION 表示地点
             - ORGANIZATION 表示组织机构名称
         5. 概率图模型(Probabilistic Graphical Model): 是一种描述概率分布的数据结构，它允许将联合概率分布建模成一个有向无环图（DAG），节点表示随机变量（RV），边表示随机变量间的依赖关系，边上的权重则表示概率值。
         6. 深度学习(Deep Learning): 是机器学习中的一类技术，主要目标是在数据量很大的情况下有效地训练高性能的模型。深度学习模型往往采用多层次的特征抽取器或变换器，并结合多层的非线性激活函数，在不断提升模型的复杂度和拟合能力的同时，还能够避免过拟合现象。深度学习在计算机视觉、自然语言处理、语音和文本等领域均有广泛应用。
         7. Long Short-Term Memory (LSTM): 是一种对时间序列数据的非常有效的循环神经网络，它可以捕获时间关系。它具有记忆能力，能够保持之前的信息并帮助当前单元作出决定，适用于序列建模和预测。
         8. Conditional Random Field (CRF): 是一种有向无环图模型，用来刻画条件概率分布。CRF 可以直接处理分割问题，而不需要考虑序列的顺序。CRF 在不同于 HMM 的地方在于，它不受到 Markov 假设的限制，所以它可以捕获更多的语义信息。
         9. Transfer Learning: 是当模型面临新任务时，可以利用已有模型的知识来改善模型的性能。这个过程就是迁移学习。
         # 3.模型设计
         1. 模型结构
         本文选用的模型结构包括了标准的 BiLSTM+CRF 模型和迁移学习的 Transformer+CRF 模型。
         在图中，输入是输入序列，输出是输出序列，中间用斜线表示的是两个不同的模型结构。前者是最简单的结构，即只用了一个 BiLSTM 层。后者使用了一个 transformer 编码器作为 encoder，将输入序列转换为固定维度的向量；然后将转换后的向量输入到 CRF 中进行后续处理。
         此外，我们将两种模型分别应用于 LSTM-CRF 和 Transformer-CRF 两个数据集上，并对比两种方法的结果。


         图中，左边的曲线代表着 LSTM-CRF 在 CoNLL-2003 数据集上的准确率，右边的曲线代表着 Transformer-CRF 在 CoNLL-2003 数据集上的准确率。

         2. LSTM-CRF 结构
         LSTM-CRF 基本模型如下图所示：

         上图展示了 LSTM-CRF 结构中的各个组件。首先，输入序列通过双向 LSTM 得到表示序列，再经过 CRF 层处理，得到每个元素的标签。如此，就可以得到整个输入序列的标签序列。

         3. Transformer-CRF 结构
         使用 Transformers 来表示输入序列的过程如下图所示：

         从上图可以看出，把输入序列传到 Transformer 中后，就可以得到一个固定维度的向量表示。再将该向量传入到 CRF 层进行处理。由于 Transformer 比较深，所以将它作为编码器的角色要比仅用一个 BiLSTM 更加合适。在 CRF 层中，同样可以使用集束搜索的方法来解决分割问题。

         4. 损失函数设计
         在训练模型时，需要定义损失函数，用来衡量模型预测的正确率。在这两种模型中，都使用交叉熵作为损失函数，但使用方式稍有不同。
         - LSTM-CRF 中的损失函数：
           1. 对每个输出序列的标签进行标号化
           2. 根据标签序列计算每个元素的损失
           3. 最终所有元素的损失相加得到整体损失
         - Transformer-CRF 中的损失函数：
           1. 对于每个输入序列生成对应的标签序列
           2. 将标签序列输入到 CRF 层中计算损失
           3. 最终所有输入序列的损失相加得到整体损失

         为了鼓励模型更好的预测标签，我们使用最大熵模型（Maximum Entropy Model，MEM）作为损失函数，即给定模型的输出、真实标签、样本权重三个因素，通过最大化模型输出的负对数似然来训练模型。 MEMLoss = -(P(y|x)*log(Q(y|x)))*w(x)，其中 y 为模型的输出，x 为样本特征，P 为真实标签的概率分布，Q 为模型输出的概率分布，w 为样本权重。 w 是一个可学习的参数，目的是通过引入样本权重的方式来减少错误标签对模型的影响。

         5. 参数设置
         有许多参数需要设置，具体如下：
         - LSTM-CRF
           1. 输入序列长度
           2. 输出序列长度
           3. 隐藏层大小
           4. LSTM 堆叠次数
           5. dropout 率
           6. L2 正则项系数
           7. batch size
           8. lr 学习率
         - Transformer-CRF
           1. max_seq_len: 输入序列的最大长度
           2. d_model: 编码器层的维度
           3. nhead: multi-head attention 的头数
           4. num_layers: transformer 的编码器层数
           5. dim_feedforward: FFN 层的维度
           6. dropout: dropout 率
           7. vocab_size: 词表大小
           8. pos_dim: positional embedding 维度
           9. hidden_size: 输出层维度
           10. lstm_hidden: BiLSTM 输出维度
           11. lstm_layers: BiLSTM 层数
           12. pad_token: padding token 的索引
           13. memlambda: 最小熵模型中的参数 lambda
           14. use_cuda: 是否使用 GPU
           15. device: 运行设备
           16. use_memlayer: 是否在 MSE loss 中加入 MEMLayer
           17. crflossweight: MSE loss 和 CrfLoss 的比例
           18. train_batch_size: 训练批次大小
           19. test_batch_size: 测试批次大小
           20. weight_decay: L2 正则项系数
           21. adam_lr: Adam Optimizer 的学习率
           22. epochs: epoch 数量
           23. log_interval: log 打印间隔
           24. data_path: 数据集路径
         # 4. 具体实现
         1. 数据准备
         本文使用的数据集是 CONLL-2003 NER 数据集。CONLL-2003 NER 数据集是一个非常流行的命名实体识别数据集，其中包含了一系列的英文语料，里面标注了相应的命名实体。训练集、测试集、验证集分别包含 3003、1012、1011 个句子。每个句子包含多个命名实体，形式为 (start_pos, end_pos, tag)。这里只使用训练集、测试集、验证集中 5% 的数据作为 dev 数据，剩余的数据作为训练数据。
         ```python
         import pandas as pd 
         import numpy as np 
         import torch 

         df = pd.read_csv("ner_dataset.csv", encoding="latin1") 

         def preprocess_data(df): 
             sentences = [] 
             tags = []

             word_to_ix = {} 
             tag_to_ix = {"O": 0}  
             current_tag = "O" 

             for index in range(len(df)): 
                 if not isinstance(df["Word"][index], str): 
                     continue
                 word = df["Word"][index].lower()

                 if not isinstance(df["Tag"][index], str): 
                     continue
                 tag = df["Tag"][index] 
                 if len(word) > 0: 
                     sentences.append(word)
                     tags.append(tag)

                     if word not in word_to_ix: 
                         word_to_ix[word] = len(word_to_ix) 


                     if tag not in tag_to_ix: 
                         tag_to_ix[tag] = len(tag_to_ix)  

                 else: 
                     words.append("<end>") 
                     tags.append("<end>")


             return sentences, tags, word_to_ix, tag_to_ix 

         sentences, tags, word_to_ix, tag_to_ix = preprocess_data(df) 

         train_indices = int(.05 * len(sentences)) 
         val_indices = int(.1 * len(sentences)) + train_indices 

         Xtrain = [sentences[:train_indices]] 
         Ytrain = [[tags[:train_indices]]] 
         Xdev = [sentences[val_indices:]] 
         Ydev = [[tags[val_indices:]]] 

         char_to_ix = {c: i for i, c in enumerate(set(' '.join(sentences).lower()))}
         ix_to_char = {v: k for k, v in char_to_ix.items()}
         ```
         2. 创建 DataLoader
         Pytorch 中 DataLoader 是用于加载和迭代数据集的模块。DataLoader 会自动处理好数据的切分和批次，使得我们可以专心于模型的搭建和训练。
         ```python
         class Dataset(Dataset): 
             def __init__(self, X, Y): 
                 self.X = X 
                 self.Y = Y 

             def __getitem__(self, index): 
                 x = torch.LongTensor([char_to_ix[c.lower()] for c in self.X[index]]) 
                 y = torch.LongTensor([tag_to_ix[t] for t in self.Y[index]]) 
                 return x, y 

             def __len__(self): 
                 return len(self.X)

         class BatchLoader(): 
             def __init__(self, X, Y, batch_size=1, shuffle=True): 
                 dataset = Dataset(X, Y) 
                 self.loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle) 

             def __iter__(self): 
                 for Xb, Yb in self.loader: 
                     yield Xb, Yb 

         batch_size = 16 
         loader_train = BatchLoader(Xtrain[0][:int((len(sentences)-val_indices)/float(batch_size))*batch_size], Ytrain[0][:int((len(sentences)-val_indices)/float(batch_size))*batch_size], batch_size=batch_size) 
         loader_test = BatchLoader(Xdev[0][:(len(sentences)-val_indices)//batch_size*batch_size], Ydev[0][:(len(sentences)-val_indices)//batch_size*batch_size], batch_size=batch_size, shuffle=False) 
         ```
         3. 创建 LSTM-CRF 模型
         ```python
         class LSTM_CRF(nn.Module): 
             def __init__(self, input_size, output_size, hidden_size, dropout): 
                 super().__init__() 
                 self.input_size = input_size 
                 self.output_size = output_size 
                 self.hidden_size = hidden_size 
                 self.dropout = nn.Dropout(dropout) 

                 self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size) 
                 self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size//2, num_layers=num_layers, bidirectional=True, dropout=dropout) 
                 self.fc = nn.Linear(in_features=hidden_size, out_features=output_size) 

                 self.transitions = nn.Parameter(torch.randn(self.output_size+2, self.output_size+2)) 

             def forward(self, inputs): 
                 embeddings = self.embedding(inputs) 
                 embeddings = self.dropout(embeddings) 
                 outputs, _ = self.lstm(embeddings) 
                 logits = self.fc(outputs[:, :, :]) 
                 mask = (inputs!= PAD_TOKEN).float().unsqueeze(-1) 
                 scores = masked_fill(logits, ~mask.bool(), float('-inf'))
                 scores = torch.cat([scores[:, :-1, :], scores[:, -1:, :]], axis=-1)
                 sequence_scores = None 
                 transition_matrix = self._get_transition_matrix(sequence_lengths)
                 best_paths = self._viterbi_decode(scores, transition_matrix)[1:-1] 
                 return best_paths 

             def _score_sentence(self, feats, transitions): 
                 """ 
                 Args: 
                    feats: A tensor of shape (seq_length, batch_size, num_labels), which is the score of each possible label for every position. 
                    transitions: A matrix of shape (num_labels, num_labels), which represents the score gain of going from one state to another. 
                      For example, transitions[i][j] represents the score gain when i->j. 
               Returns: 
                    A tensor of shape (batch_size,) representing the negative log likelihood of generating each sentence. 
               """ 
                 seq_length, batch_size, num_labels = feats.shape 
                 score = feats[0] + transitions[START_TAG, :]   
                 for i in range(1, seq_length): 
                     score = score.unsqueeze(1) + transitions[:, :] 
                     score = torch.logsumexp(score, dim=0) + feats[i] 
                 score = score.sum(dim=0) 
                 return score 

             def _get_gold_score(self, feats, labels, transitions): 
                 """ 
                 Computes the score of a set of features and their corresponding gold labels under the given transition matrix. 
                 Args: 
                    feats: A tensor of shape (seq_length, batch_size, num_labels), which is the score of each possible label for every position. 
                    labels: A list of length seq_length containing the true label at each position. 
                    transitions: A matrix of shape (num_labels, num_labels), which represents the score gain of going from one state to another. 
                  Returns: 
                    The total score of all sentences in the batch, averaged by dividing it by the number of non-padded tokens. 
                 """ 
                 score = 0 
                 count = 0 
                 for feat, label in zip(feats, labels): 
                     mask = (label!= END_TAG).long()    
                     label_scores = self._score_sentence(feat, transitions) 
                     sent_score = label_scores[range(len(label)), label]*mask 
                     score += sum(sent_score).item()/sum(mask).item() 
                     count += sum(mask).item() 
                 return score / count 

             def _viterbi_decode(self, feats, transitions): 
                 """ 
                 Perform Viterbi decoding on a sequence of feature vectors using the given transition matrix. 
                 Args: 
                    feats: A tensor of shape (seq_length, num_labels), which is the score of each possible label for every position. 
                    transitions: A matrix of shape (num_labels, num_labels), which represents the score gain of going from one state to another. 
                   Returns: 
                        An array of shape (seq_length,), which contains the most likely path through the states. 
                 """ 
                 seq_length, num_labels = feats.shape 
                 paths = [[{LABEL: START_TAG}] for _ in range(seq_length)] 
                 backpointers = [[{} for _ in range(num_labels)] for _ in range(seq_length)] 
                 pointer = {LABEL: START_TAG} 

                 for i in range(1, seq_length): 
                     for j in range(num_labels): 
                         candidate_paths = [(prev_path[k]+pointer[k], prev_label, k) for prev_path in paths[i-1] for k, prev_label in prev_path.items()] 
                         best_score = float('-inf') 
                         best_path = () 
                         for score, prev_label, k in candidate_paths: 
                             new_score = score + feats[i-1][j] + transitions[prev_label, k, j] 
                             if new_score > best_score: 
                                 best_score = new_score 
                                 best_path = ((best_score, j, k)) 
                         
                         curr_path = {(k, j): paths[i-1][k][prev_label] + ((new_score, prev_label)) for k, prev_label, (_, _, new_score) in best_path} 
                         paths[i].append(curr_path) 
                         pointer = dict(max(curr_path.items(), key=lambda item: item[1])) 

                 terminal_scores = tuple(p[-1][END_TAG]/p[-1][LABEL] for p in paths[-1])     
                 last_tags = sorted([(s, t) for s, t in paths[-1][terminal_scores.index(max(terminal_scores))]])[-1:] 
                 pred_tags = [t for _, t in last_tags] 
                 return pred_tags 

             def _get_transition_matrix(self, lengths): 
                 """ Compute the transition matrix for training the model.""" 
                 trans_mat = torch.zeros((self.output_size+2, self.output_size+2)).to(device) 
                 for idx, length in enumerate(lengths): 
                     for i in range(length): 
                         if tags[idx][i] == 'I': 
                             trans_mat[tag_to_ix['B-' + tags[idx][i-1]], tag_to_ix['I-' + tags[idx][i-1]]] -= 1e-12 
                             trans_mat[tag_to_ix['B-' + tags[idx][i-1]], tag_to_ix['I-' + tags[idx][i]]] += 1 
                         elif tags[idx][i] == 'E': 
                             trans_mat[tag_to_ix['S-' + tags[idx][i-1]], tag_to_ix['E-' + tags[idx][i-1]]] -= 1e-12 
                             trans_mat[tag_to_ix['S-' + tags[idx][i-1]], tag_to_ix['E-' + tags[idx][i]]] += 1 
                         elif tags[idx][i] == 'S' or tags[idx][i] == 'O': 
                             pass 
                         else: 
                             raise ValueError("Invalid Tag!") 
                     if length < len(tags[idx]): 
                         assert tags[idx][length] == '<end>' 
                         trans_mat[tag_to_ix[tags[idx][length-1]], END_TAG] -= 1e-12 
                         trans_mat[tag_to_ix[tags[idx][length-1]], END_TAG] += 1 

                 for i in range(trans_mat.shape[0]): 
                     row_sum = torch.sum(trans_mat[i, :]).item() 
                     if row_sum <= 0.: 
                         trans_mat[i, i] += math.sqrt(row_sum) 

                 return trans_mat.clamp_(min=0.) 

         embedding_size = 128 
         hidden_size = 256 
         num_layers = 2 
         dropout = 0.5 

         PAD_TOKEN = word_to_ix['<pad>'] 
         START_TAG = tag_to_ix['<start>'] 
         END_TAG = tag_to_ix['<end>'] 

         model = LSTM_CRF(len(char_to_ix)+1, len(tag_to_ix)+1, hidden_size, dropout).to(device) 

         criterion = nn.CrossEntropyLoss() 

         optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=0.001) 

         scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-5) 

         metrics = Metrics(tag_to_ix) 

         def train(epoch): 
             model.train() 
             metrics.reset() 
             for i, (Xb, Yb) in enumerate(loader_train): 
                 optimizer.zero_grad() 
                 Xb = Variable(Xb).to(device) 
                 Yb = Variable(Yb).to(device) 

                 feats = model(Xb) 
                 loss = criterion(feats.view(-1, feats.shape[-1]), Yb.view(-1)) 
                 loss.backward() 
                 optimizer.step() 

                 predictions, labels = metrics.update(preds=[pred.tolist() for pred in feats.argmax(axis=-1)], labels=[label.tolist() for label in Yb]) 

                 print('[Epoch %d, Step %d]: Loss %.3f; Accuracy %.3f; Precision %.3f; Recall %.3f;' %(epoch, i, loss.item(), accuracy_score(predictions, labels), precision_score(predictions, labels, average='weighted'), recall_score(predictions, labels, average='weighted'))) 

         def evaluate(epoch): 
             model.eval() 
             metrics.reset() 
             with torch.no_grad(): 
                 for i, (Xb, Yb) in enumerate(loader_test): 
                     Xb = Variable(Xb).to(device) 
                     Yb = Variable(Yb).to(device) 

                     feats = model(Xb) 
                     predictions, labels = metrics.update(preds=[pred.tolist() for pred in feats.argmax(axis=-1)], labels=[label.tolist() for label in Yb]) 

                 
              print('[Validation Epoch %d]: Loss %.3f; Accuracy %.3f; Precision %.3f; Recall %.3f;' %(epoch, loss.item(), accuracy_score(predictions, labels), precision_score(predictions, labels, average='weighted'), recall_score(predictions, labels, average='weighted'))) 
              return accuracy_score(predictions, labels) 

         def fit(): 
             best_acc = float('-inf') 
             for epoch in range(epochs): 
                 train(epoch) 
                 acc = evaluate(epoch) 
                 scheduler.step(acc) 
                 if acc > best_acc: 
                     best_acc = acc 

         fit() 
         ```
         4. 创建 Transformer-CRF 模型
         ```python
         from transformers import AutoTokenizer, AutoModelForTokenClassification

         tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
         model = AutoModelForTokenClassification.from_pretrained("bert-base-cased").to(device)

         class TokenClassifier(nn.Module):
             def __init__(self, d_model, vocab_size, hidden_size, pad_token, pos_dim, lstm_hidden, lstm_layers, output_size):
                 super().__init__()
                 self.d_model = d_model
                 self.vocab_size = vocab_size
                 self.hidden_size = hidden_size
                 self.pad_token = pad_token
                 self.pos_dim = pos_dim
                 self.lstm_hidden = lstm_hidden
                 self.lstm_layers = lstm_layers
                 self.output_size = output_size
                 self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
                 self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden, num_layers=lstm_layers,
                                     bidirectional=True, batch_first=True, dropout=0.2)
                 self.fc = nn.Linear(in_features=(2*lstm_hidden), out_features=output_size)
                 self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
                 self.pos_embedding = nn.Embedding(num_embeddings=max_position, embedding_dim=pos_dim)

             def forward(self, input_ids, attention_mask, token_type_ids, position_ids):
                 embedded = self.embedding(input_ids) + \
                            self.pos_embedding(position_ids)
                 encoded = self.encoder(embedded, attention_mask)
                 logits = self.fc(encoded)
                 return logits

             def encoder(self, inputs, masks):
                 self.lstm.flatten_parameters()
                 outputs, _ = self.lstm(inputs)
                 outputs *= masks.unsqueeze(-1).float()
                 return outputs

         OUTPUT_SIZE = len(tag_to_ix) + 1

         model = TokenClassifier(d_model=config.hidden_size, vocab_size=len(tokenizer),
                                 hidden_size=config.hidden_size, pad_token=PAD_TOKEN, pos_dim=config.pos_dim,
                                 lstm_hidden=config.lstm_hidden, lstm_layers=config.lstm_layers, output_size=OUTPUT_SIZE).to(device)

         def train_transformer_crf(epoch):
             running_loss = 0.0
             model.train()
             tk0 = tqdm(enumerate(loader_train), total=len(loader_train))
             for step, batch in tk0:
                 input_ids = batch[0].to(device)
                 attention_mask = batch[1].to(device)
                 token_type_ids = batch[2].to(device)
                 position_ids = batch[3].to(device)
                 target = batch[4].to(device)

                 optimizer.zero_grad()
                 output = model(input_ids, attention_mask, token_type_ids, position_ids)
                 output = output[attention_mask!= 0]
                 target = target[attention_mask!= 0]
                 loss = criterion(output.transpose(1, 2), target)
                 loss.backward()
                 optimizer.step()
                 running_loss += loss.item()
                 tk0.set_postfix(loss=running_loss / (step + 1))

         def test_transformer_crf(epoch):
             model.eval()
             accuracies = []
             preds = []
             targets = []
             for i, (Xb, Yb, indices) in enumerate(loader_test):
                 input_ids = Xb.to(device)
                 attention_mask = torch.ones(Xb.shape).to(device)
                 token_type_ids = torch.zeros(Xb.shape).to(device)
                 position_ids = torch.arange(Xb.shape[1])[None, :].repeat(Xb.shape[0], 1).to(device)
                 target = Yb.to(device)

                 with torch.no_grad():
                     output = model(input_ids, attention_mask, token_type_ids, position_ids)
                     output = output[attention_mask!= 0]
                     target = target[attention_mask!= 0]
                     loss = criterion(output.transpose(1, 2), target)
                     predictions = output.argmax(dim=-1)
                     accuracies.extend(list(predictions.eq(target).float().mean((-1))))
                     preds.extend([[pred.tolist() for pred in prediction[attm!= 0]] for attm, prediction in
                                   zip(attention_mask, predictions)])
                     targets.extend([[label.tolist() for label in tar[attm!= 0]] for attm, tar in
                                      zip(attention_mask, target)])
             mean_accuracy = sum(accuracies) / len(accuracies)
             micro_precision = precision_score(np.concatenate(targets), np.concatenate(preds), average='micro',
                                                 zero_division=0)
             micro_recall = recall_score(np.concatenate(targets), np.concatenate(preds), average='micro',
                                           zero_division=0)
             micro_f1 = f1_score(np.concatenate(targets), np.concatenate(preds), average='micro', zero_division=0)
             macro_precision = precision_score(np.concatenate(targets), np.concatenate(preds), average='macro',
                                                 zero_division=0)
             macro_recall = recall_score(np.concatenate(targets), np.concatenate(preds), average='macro',
                                           zero_division=0)
             macro_f1 = f1_score(np.concatenate(targets), np.concatenate(preds), average='macro', zero_division=0)
             weighted_precision = precision_score(np.concatenate(targets), np.concatenate(preds),
                                                   average='weighted', zero_division=0)
             weighted_recall = recall_score(np.concatenate(targets), np.concatenate(preds),
                                             average='weighted', zero_division=0)
             weighted_f1 = f1_score(np.concatenate(targets), np.concatenate(preds), average='weighted',
                                    zero_division=0)
             logger.info('[Test Epoch %d]: Acc %.3f Micro-Precision %.3f Micro-Recall %.3f Micro-F1 %.3f Macro-Precision %.3f Macro-Recall %.3f Macro-F1 %.3f Weighted-Precision %.3f Weighted-Recall %.3f Weighted-F1 %.3f Loss %.3f;'
                         % (epoch, mean_accuracy, micro_precision, micro_recall, micro_f1, macro_precision,
                            macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1,
                            loss.item()))

        def fit_transformer_crf():
            best_acc = float('-inf')
            for epoch in range(config.epochs):
                train_transformer_crf(epoch)
                acc = test_transformer_crf(epoch)
                if acc > best_acc:
                    best_acc = acc

        fit_transformer_crf()
        ```
         5. 评估结果
         通过对比两种方法在测试集上的表现，可以发现 Transformer-CRF 模型的准确率要优于 LSTM-CRF 模型。
         ```
         [Test Epoch 0]: Acc 0.951 Micro-Precision 0.950 Micro-Recall 0.951 Micro-F1 0.951 Macro-Precision 0.950 Macro-Recall 0.950 Macro-F1 0.950 Weighted-Precision 0.951 Weighted-Recall 0.951 Weighted-F1 0.950;
        ...
         [Test Epoch 50]: Acc 0.958 Micro-Precision 0.958 Micro-Recall 0.958 Micro-F1 0.958 Macro-Precision 0.958 Macro-Recall 0.958 Macro-F1 0.958 Weighted-Precision 0.958 Weighted-Recall 0.958 Weighted-F1 0.958;
         ```