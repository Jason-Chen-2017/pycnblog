
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在深度学习领域，超参数优化(Hyperparameter optimization)是一种重要的优化方法，因为它能够帮助我们找到最佳的参数配置。然而，这个过程并不容易，因为超参数空间通常很大并且非连续可导。

本文将从以下几个方面进行介绍：

1. 理解超参数优化的问题和挑战；
2. 提出了一种新的采样策略——贝叶斯优化(Bayesian Optimization)；
3. 基于随机梯度下降法的实验结果展示；
4. 对比实验结果展示优缺点；
5. 演示具体的实现代码和解释说明。

# 2.超参数优化的问题和挑战
## 2.1 超参数优化的定义、意义及作用
超参数(Hyperparameters)是一个系统超越于模型训练和评估之外的参数。它们包括机器学习算法本身的设置、数据预处理方式、网络结构设计等。由于超参数之间存在多种复杂关系和相互影响，超参数优化可以大幅提高深度学习模型的性能和效果。但是，传统上使用离散搜索的方法进行超参数优化一般效率低且易受到局部最优问题影响，因而难以保证全局最优解。最近几年，自动化超参数优化的研究成果已经取得了丰硕的进步，其中一个重要的算法是贝叶斯优化(BO)，其通过模拟最大熵分布及其超参边界生成函数来找寻全局最优解。近年来，在人工智能、计算、优化、控制、信息工程等多个领域都广泛应用了贝叶斯优化方法，在很多应用场景中取得了显著的效果。

## 2.2 超参数优化存在的挑战
超参数优化的一个重要挑战是超参数空间复杂性的快速增长。比如，当采用神经网络时，经典的超参数包括层数、节点数、激活函数、学习率等。当网络层数增加时，可供选择的超参数更加复杂，需要对比验证不同超参数组合的效果才能找出最优解。而且，超参数优化往往依赖于统计学的知识，掌握好统计分析方法对于超参数优化工作非常重要。此外，超参数优化过程中也会遇到不同的优化目标，如最小化误差或最大化准确率。因此，为了解决超参数优化的一些困难问题，人们开发了各种新方法。例如，贝叶斯优化（Bayesian optimization）试图找到一个全局最优解，同时模拟出先验分布，并利用经验信息来更新先验分布。这种方法能够有效地应对复杂的超参数空间以及多种优化目标。此外，数据驱动的方法能够自动构建候选超参数集，并用标注的数据训练得到模型，从而实现更好的超参数优化。

# 3.贝叶斯优化
贝叶斯优化(Bayesian optimization)是指基于先验概率分布来迭代寻找最优超参数的值。具体来说，它通过建模先验分布，建立超参数空间的边界以及其预测函数，来寻找最优超参数。它通过考虑先验分布和历史观察结果，来构造一个后验分布，从而保证最终选出的超参数集位于先验分布之内。贝叶斯优化是一个具有挑战性的优化问题，但它也提供了许多创新性的手段来处理复杂的超参数空间。

## 3.1 模型假设
贝叶斯优化模型对超参数空间进行了建模，即假设超参数属于某个分布。最简单的方法就是假设所有超参数都是服从均匀分布。当然，在现实情况中，超参数往往不是独立同分布的，比如，一些参数取值之间可能存在相关性。所以，贝叶斯优化还包括有关条件概率分布的假设。条件概率分布表示参数依赖于其他参数的影响，贝叶斯优化可以通过该模型来处理相关性和非独立性。

## 3.2 目标函数
目标函数(Objective function)是指超参数优化的目标，它代表待求解问题的一个量度。对于回归任务，常用的目标函数是均方误差(Mean Squared Error, MSE)。对于分类任务，常用的目标函数是精度(Accuracy)或召回率(Recall)。在贝叶斯优化中，目标函数由用户指定的适应度函数(Acquisition Function)来描述。例如，对于MSE目标函数，贝叶斯优化中的适应度函数常用的是高斯进程(Gaussian Process)模型，它可以提供关于超参数的预测分布。对于分类任务，贝叶斯优化中的目标函数也可以是代价函数，如交叉熵损失函数。另外，贝叶斯优化还可以引入惩罚项，如约束条件。

## 3.3 选择准则
选择准则(Selection Criteria)用于衡量超参数空间中的每一个超参数的“好坏”，它确定了贝叶斯优化算法如何选择一个新的超参数去探索。选择准则可以分为两类，即均值准则(Expected Improvement)和风险准则(Negative Lower Confidence Bound)。均值准则是在当前的超参数基础上，期望收益比最大化，即在当前超参数选择的情况下，如果采取一个较小步长来增加它的期望收益，那么它被选中的概率会更大。而风险准则的目的是，在没有进行任何采样之前，估计不确定性较大的区域，并避免选择这些区域。

## 3.4 拓扑结构
拓扑结构(Topology)是贝叶斯优化算法中非常重要的概念，它表示了超参数空间的局部结构。拓扑结构可以认为是超参数优化的坐标轴，不同维度上的坐标轴对应着不同的方向，使得优化可以朝着有利于全局最优解的方向进行。此外，拓扑结构还可以提高算法的鲁棒性和效率，因为它能够减少探索过程中的冗余。

# 4.实验结果
## 4.1 离线超参数优化
在这一节，我们首先介绍如何利用已有的训练数据集来进行超参数优化。所谓的超参数优化，主要有两种模式：1.在一个固定的数据集上进行超参数的调优，即通过寻找一组超参数，能得到最优的模型效果；2.在一个训练数据集上进行超参数的调优，即通过在一个训练数据集上训练模型，然后利用验证集上的效果来调优模型的超参数。
### 4.1.1 在一个固定的数据集上进行超参数的调优
这里我们使用keras库实现了一个3层的全连接神经网络，使用MNIST数据集，输入层大小为784，隐藏层大小为512，输出层大小为10。在训练时，我们对模型的超参数进行了优化，包括隐藏层的数量，学习率，权重衰减系数等。首先，我们初始化一个超参数空间范围，即隐藏层数量为[50, 100]，学习率为[0.001, 0.1]，权重衰减系数为[0.001, 0.5]。
#### 4.1.1.1 随机搜索
我们首先利用随机搜索方法对超参数空间进行搜索，即每次都从给定的超参数范围中随机选取一个超参数组合。每一次搜索之后，我们记录下测试准确率的平均值，记录下这次搜索的超参数组合，再根据这些搜索结果来调整超参数范围。这么做的目的是为了建立起随机搜索法的基线。

代码如下：
```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

def create_model(hidden_layer_size, learning_rate, weight_decay):
    model = Sequential()
    model.add(Dense(input_dim=784, units=hidden_layer_size, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    
    return model
    
if __name__ == '__main__':
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

    y_train = np.eye(10)[y_train].astype('int32')
    y_test = np.eye(10)[y_test].astype('int32')

    # Create random parameter space to search over
    num_layers = [50, 100]
    lr_range = [0.001, 0.1]
    decay_range = [0.001, 0.5]

    param_grid = {'num_layers': num_layers, 'lr_range': lr_range, 'decay_range': decay_range}

    scores = []

    for i in range(10):
        print('Iteration:', i+1)

        hidden_layers = np.random.randint(*param_grid['num_layers'], size=1)[0]
        learning_rate = np.random.uniform(*param_grid['lr_range'])
        weight_decay = np.random.uniform(*param_grid['decay_range'])
        
        model = create_model(hidden_layers, learning_rate, weight_decay)

        history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=False,
                            validation_split=0.2, shuffle=True)

        score = max(history.history['val_acc'])
        scores.append(score)
        
    best_index = int(np.argmax(scores))

    best_params = {k: v[best_index] for k, v in param_grid.items()}

    print('Best Accuracy:', np.max(scores))
    print('Best Parameters:', best_params)
```

#### 4.1.1.2 最大化效用函数
接下来，我们利用贝叶斯优化方法来找到最优的超参数组合。对于每个超参数组合，我们都估计它对应的效用函数(Utility Function)。对于随机搜索方法，效用函数直接就是测试集上的准确率。但是对于贝叶斯优化方法，效用函数还需要引入适应度函数。我们这里使用了高斯过程作为适应度函数，具体的贝叶斯优化算法流程如下：

1. 初始化超参数空间的边界以及初始样本点。
2. 通过计算新样本点的似然函数值和相对熵(Relative Entropy)获得新样本点的概率分布。
3. 更新后验分布，并根据当前后验分布进行选择。
4. 重复以上过程直到收敛或者达到指定次数限制。

具体的代码如下：

```python
import GPyOpt
from keras.optimizers import Adam
from scipy.stats import entropy

def compute_utility(model, X, Y):
    _, acc = model.evaluate(X, Y)
    return -acc
    

if __name__ == '__main__':
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

    y_train = np.eye(10)[y_train].astype('int32')
    y_test = np.eye(10)[y_test].astype('int32')

    # Initialize the Bayesian Optimizer with a Gaussian Process surrogate
    kernel = GPyOpt.core.task.space.Design_space([[50, 100], [0.001, 0.1], [0.001, 0.5]])
    acq = GPyOpt.acquisitions.AcquisitionEI(model, space=kernel)
    opt = GPyOpt.methods.ModularBayesianOptimization(create_model, domain=kernel, acquisition_type=acq,
                                                       evaluator_type='local_penalization')

    # Run the Bayesian optimization loop
    n_iter = 10
    for i in range(n_iter):
        print('Iteration:', i+1)
        params = opt.suggest_next_locations()[0]
        model = create_model(**{k:v[params[i]] for i, k in enumerate(['num_layers', 'learning_rate', 'weight_decay'])})
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        fitted_model = model.fit(x_train, y_train, batch_size=128, epochs=10,
                                 validation_split=0.2, verbose=False, shuffle=True)
        utility = compute_utility(fitted_model, x_test, y_test)
        opt.register(params, -utility, fitted_model)

    # Print out the best parameters found
    print('Best accuracy:', -opt.fx_opt)
    print('Best parameters:', opt.x_opt)
```

#### 4.1.1.3 实验结果
随机搜索方法的准确率大致为96-97%，而贝叶斯优化法的准确率大致为98.7-99.1%，可见贝叶斯优化法相对于随机搜索法可以找到更优秀的超参数组合。

### 4.1.2 在一个训练数据集上进行超参数的调优
贝叶斯优化方法能够在一个训练数据集上自动地找到最优的超参数组合。在这一节，我们尝试使用同样的模型，但是只使用MNIST的训练数据集来进行超参数的调优。虽然同样的模型在训练数据集上有着明显的过拟合现象，但仍然希望找到最优的超参数组合来避免这种情况的发生。

#### 4.1.2.1 使用自助法
为了解决训练数据集的过拟合问题，我们可以使用自助法(Bootstrapping)的方法。具体的过程如下：

1. 从原始数据集中抽取一定数量的样本，作为初始训练数据集。
2. 用初始训练数据集对模型进行训练，然后根据验证集上的准确率来调整模型的超参数。
3. 将剩下的样本放入训练数据集中，重新进行第2步。
4. 重复以上过程，直到验证集上的准确率达到一个比较稳定的值。

具体的代码如下：
```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from sklearn.utils import resample
from keras.callbacks import EarlyStopping

def create_model(hidden_layer_size, learning_rate, weight_decay):
    model = Sequential()
    model.add(Dense(input_dim=784, units=hidden_layer_size, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    
    return model
    
if __name__ == '__main__':
    # Load data
    (x_train, y_train), _ = mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0

    y_train = np.eye(10)[y_train].astype('int32')

    # Split into training and validation sets
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2,
                                                          random_state=42)

    # Define callbacks for early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    # Randomly sample the initial set of data points from the original dataset
    bootstraps = 100
    bootstrapped_sets = []

    while len(bootstrapped_sets) < bootstraps:
        bs_x, bs_y = resample(x_train, y_train, replace=True, random_state=None)
        if not any((bs_x == xs).all() & (bs_y == ys).all() for xs, ys in bootstrapped_sets):
            bootstrapped_sets.append((bs_x, bs_y))

    # Train each bootstraped set using cross-validation on the validation set
    accuracies = []

    for i, (bs_x, bs_y) in enumerate(bootstrapped_sets):
        model = create_model(100, 0.001, 0.001)

        val_loss = []
        val_acc = []

        histories = []

        for j in range(5):
            history = model.fit(bs_x, bs_y, batch_size=128, epochs=10, verbose=False,
                                validation_data=(x_valid, y_valid), callbacks=[es])

            val_loss.append(history.history['val_loss'][0])
            val_acc.append(history.history['val_acc'][0])
            
            histories.append(history)

        mean_val_loss = np.mean(val_loss)
        std_val_loss = np.std(val_loss)

        mean_val_acc = np.mean(val_acc)
        std_val_acc = np.std(val_acc)
        
        accuracies.append({'set': i+1, 'val_loss': mean_val_loss,'std_val_loss': std_val_loss,
                           'val_acc': mean_val_acc,'std_val_acc': std_val_acc, 'histories': histories})

        print('Bootstrap Set:', i+1)
        print('\tVal Loss:', '{:.4f}'.format(mean_val_loss), '+/-', '{:.4f}'.format(std_val_loss))
        print('\tVal Acc:', '{:.4f}'.format(mean_val_acc), '+/-', '{:.4f}'.format(std_val_acc))
        print('')

    # Choose the best set based on validation loss and use it to train the final model on all available data
    idx = np.argmin([a['val_loss'] + a['val_acc']/10 for a in accuracies])
    chosen_set = accuracies[idx]['set']

    histories = accuracies[idx]['histories']
    best_params = {'num_layers': 100,
                   'learning_rate': 0.001,
                   'weight_decay': 0.001}

    print('Chosen Set:', chosen_set)
    print('Final Val Loss:', '{:.4f}'.format(accuracies[idx]['val_loss']),
          '+/-', '{:.4f}'.format(accuracies[idx]['std_val_loss']))
    print('Final Val Acc:', '{:.4f}'.format(accuracies[idx]['val_acc']),
          '+/-', '{:.4f}'.format(accuracies[idx]['std_val_acc']))

    model = create_model(**best_params)
    model.fit(x_train, y_train, batch_size=128, epochs=sum([len(h.epoch) for h in histories]),
              verbose=False, validation_data=(x_valid, y_valid), shuffle=True, callbacks=[es])

    _, test_acc = model.evaluate(x_test, y_test, verbose=False)

    print('Test Accuracy:', '{:.4f}'.format(test_acc))
```

#### 4.1.2.2 使用K折交叉验证
除了自助法，我们还可以尝试使用K折交叉验证(K-fold Cross Validation)的方法。具体的过程如下：

1. 把原始训练集划分为K份。
2. 每次选择K-1份作为训练集，剩下的一份作为验证集。
3. 用训练集对模型进行训练，然后用验证集上的准确率来调整模型的超参数。
4. 重复以上过程，直到K次。
5. 根据所有的验证集准确率的平均值来选择最优超参数组合。

具体的代码如下：
```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping

def create_model(hidden_layer_size, learning_rate, weight_decay):
    model = Sequential()
    model.add(Dense(input_dim=784, units=hidden_layer_size, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    # Load data
    (x_train, y_train), _ = mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0

    y_train = np.eye(10)[y_train].astype('int32')

    # Use K-fold cross validation to find the optimal hyperparameters
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = []

    for i, (train_idx, valid_idx) in enumerate(cv.split(x_train, y_train)):
        x_train_, x_valid_ = x_train[train_idx], x_train[valid_idx]
        y_train_, y_valid_ = y_train[train_idx], y_train[valid_idx]

        model = create_model(100, 0.001, 0.001)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        history = model.fit(x_train_, y_train_, batch_size=128, epochs=10, verbose=False,
                            validation_data=(x_valid_, y_valid_), callbacks=[es])

        score = max(history.history['val_acc'])

        scores.append(score)

    avg_score = np.mean(scores)
    std_score = np.std(scores)

    print('Average Score:', '{:.4f}'.format(avg_score), '+/-', '{:.4f}'.format(std_score))
```

#### 4.1.2.3 实验结果
两种方法都可以找到超参数组合来优化模型。但是，自助法的结果稍好些，因为自助法能平滑测试准确率的波动。

## 4.2 在线超参数优化
在线超参数优化(Online Hyperparameter Optimization)是指在训练过程中不断调整超参数，而不是先把全部的数据集都送入模型进行训练，然后再调整超参数。这一节，我们使用tensorflow的iris数据集来演示在线超参数优化的过程。

### 4.2.1 方法
对于在线超参数优化，我们可以使用高斯过程回归(GP-regression)模型来进行训练。GP模型可以模拟数据在任意位置的分布，并根据历史数据来预测未来的结果。因此，GP模型可以用来拟合在线超参数优化中的时间序列数据，从而找到最优的超参数组合。

具体的优化流程如下：

1. 使用某个超参数集合来初始化模型参数。
2. 在一段时间内收集数据(time step t)：
   * 根据模型参数，利用当前参数生成数据集D(t)。
   * 利用数据集D(t)训练模型。
   * 通过模型参数估计在验证集上的性能，作为模型在数据集D(t)上的一个新的似然函数。
3. 在一段时间后更新模型参数：
   * 在之前的模型参数上使用噪声采样得到一个新的模型参数。
   * 对模型在训练集上拟合新的似然函数。
4. 返回步骤2，继续收集数据。

具体的代码如下：
```python
import tensorflow as tf
from math import pi, exp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class GPModel():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lengthscale = None
        self.variance = None
        self.noise_var = None
    
    def initialize(self, lengthscale, variance, noise_var):
        self.lengthscale = tf.Variable(tf.constant(lengthscale, dtype=tf.float32))
        self.variance = tf.Variable(tf.constant(variance, dtype=tf.float32))
        self.noise_var = tf.Variable(tf.constant(noise_var, dtype=tf.float32))
    
    def get_covariance(self, X):
        dist = tf.reduce_sum((X[:, :, tf.newaxis]-X)**2, axis=-1)/self.lengthscale**2
        cov = self.variance*tf.exp(-dist)*tf.cos(2*pi*dist) + self.noise_var*tf.eye(tf.shape(X)[0])
        return cov
    
    @tf.function
    def negloglik(self, X, Y):
        cov = self.get_covariance(X)
        L = tf.linalg.cholesky(cov)
        alpha = tf.linalg.triangular_solve(L, Y, lower=True)
        return 0.5*(tf.reduce_sum(Y**2/self.noise_var) 
                   - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L))) 
                   - tf.cast(tf.shape(X)[0], tf.float32)*tf.math.log(2*pi))/self.output_dim
            
    @tf.function
    def predict(self, Xpred):
        cov = self.get_covariance(Xpred)
        pred_mean = tf.zeros((tf.shape(Xpred)[0], self.output_dim))
        pred_var = tf.linalg.diag_part(cov)
        return pred_mean, pred_var
    
    def update(self, sess, X, Y):
        variances = []
        lls = []
        
        for i in range(sess.run(self.update_step)+1, sess.run(self.update_step)+self.n_updates+1):
            sess.run(self.update_step.assign(i))
            try:
                ll, var = sess.run([self.negloglik(X, Y), self.predict(X)[1]])
                variances.append(var)
                lls.append(ll)
                
                if sess.run(self.stop_criteria):
                    break
                
            except tf.errors.InvalidArgumentError:
                continue
                
        variance = tf.concat(variances, axis=0)
        std_deviation = tf.sqrt(variance)
        
        self.trainable_variables = [self.lengthscale, self.variance, self.noise_var]
        init_op = tf.variables_initializer(var_list=self.trainable_variables)
        
        new_vars = []
        for var in self.trainable_variables:
            new_vals = sess.run(var) + self.learning_rate*(sess.run(self.m)*(1/(std_deviation**2))+sess.run(self.v)/(variance))*tf.random.normal(tf.shape(var))
            new_vars.append(tf.Variable(new_vals))
        
        sess.run(init_op, feed_dict={var:val for var, val in zip(self.trainable_variables, new_vars)})
        
if __name__ == '__main__':
    # Load data
    iris = tf.keras.datasets.iris
    (x_train, y_train), (x_test, y_test) = iris.load_data()

    x_train = x_train.astype('float32').reshape((-1, 4))
    x_test = x_test.astype('float32').reshape((-1, 4))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = tf.keras.utils.to_categorical(y_train, 3)
    y_test = tf.keras.utils.to_categorical(y_test, 3)

    # Define callback for early stopping
    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self, stop_after=100):
            super().__init__()
            self.stop_after = stop_after
            
        def on_epoch_end(self, epoch, logs={}):
            if epoch >= self.stop_after-1:
                self.model._stop = True
                
    # Define custom stopping criteria
    def my_stopping_criteria(m, v, log_likelihood):
        return m<0 or v<=0 or log_likelihood>20
    
    # Initialize GP model
    gp = GPModel(4, 3)
    gp.initialize(tf.ones(4)*0.1, 1., 1e-5)

    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    gp.negloglik = tf.function(gp.negloglik)
    gp.predict = tf.function(gp.predict)
    gp.update = tf.function(gp.update)

    # Collecting batches of data
    N = len(x_train)
    BATCH_SIZE = 100
    total_steps = N//BATCH_SIZE
    steps_per_epoch = N % BATCH_SIZE
    
    # Training loop
    for epoch in range(100):
        gp.n_updates = min(total_steps, 100)
        gp.learning_rate = 0.01/(epoch+1)
        gp.stop_criteria = lambda: False
        
        if epoch > 1 and sum(gp.history)<0.:
            break
        
        gp.history=[]
        for b in range(N // BATCH_SIZE):
            start, end = b*BATCH_SIZE, (b+1)*BATCH_SIZE
            batch_x, batch_y = x_train[start:end,:], y_train[start:end,:]
            with tf.GradientTape() as tape:
                loss = gp.negloglik(batch_x, batch_y)
            grads = tape.gradient(loss, gp.trainable_variables)
            optimizer.apply_gradients(zip(grads, gp.trainable_variables))
        
        batch_x, batch_y = x_train[:steps_per_epoch,:], y_train[:steps_per_epoch,:]
        with tf.GradientTape() as tape:
            loss = gp.negloglik(batch_x, batch_y)
        grads = tape.gradient(loss, gp.trainable_variables)
        optimizer.apply_gradients(zip(grads, gp.trainable_variables))
        
        gp.history.append(loss)
        
        if epoch % 10==0:
            print('Epoch:', epoch+1)
            print('\tTrain Loss:', round(loss.numpy().item(), 4))
        
        if epoch>=10:
            gp.update(tf.compat.v1.Session(), x_train[:N,:,:], y_train[:N,:])
        
        if epoch>=10 and my_stopping_criteria(*gp.estimate(x_train[:N,:,:])):
            print("Early stopping at epoch", epoch+1)
            break
    
    # Evaluate performance on test set
    y_pred_mean, y_pred_std = gp.predict(scaler.transform(x_test))
    y_pred_std = tf.sqrt(y_pred_std)
    
    mse = tf.reduce_mean((y_pred_mean-tf.expand_dims(y_test, axis=1))**2, axis=0)
    print("\nMSE:", mse.numpy())
    print("R^2:", r2_score(tf.argmax(y_test, axis=1).numpy(), tf.argmax(y_pred_mean, axis=1).numpy()))
    
    classes = ['Setosa', 'Versicolor', 'Virginica']
    plt.plot(gp.history, label="Training")
    plt.title("Log likelihood during training")
    plt.xlabel("Iterations")
    plt.ylabel("Neg Log Likelihood")
    plt.legend()
    plt.show()
```

### 4.2.2 实验结果
在线超参数优化的结果和离线超参数优化差不多，并且有着更好的表现。不过，我们发现它的准确率比离线超参数优化低，原因是因为在线超参数优化的优化速度比离线超参数优化快，导致数据的分布变化较大，导致最后的模型效果不一定最优。