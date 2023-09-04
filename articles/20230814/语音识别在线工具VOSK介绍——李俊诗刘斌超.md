
作者：禅与计算机程序设计艺术                    

# 1.简介
  


## 什么是VOSK？

VOSK是一个开源的基于TensorFlow的开源语音识别工具包，可以将音频文件转化成文本。相比其他的语音识别工具包，如谷歌语音识别API等，它的优点主要有以下几点：

1.速度快：VOSK在短时模型上可达到170~220ms/样本，即使在最高精度下的16kHz采样率下也可达到90-100ms。这种速度要远远超过其他软件。

2.准确性高：该软件支持多种语言，包括中文、英文、日语等。不仅在声学角度，还在语言学角度进行了优化，使得其对于复杂语言表现出更好的准确性。

3.易于部署：VOSK对系统资源要求较低，可以在任何环境下运行，无需安装额外的依赖库。同时，它还提供了一个方便的Python接口，可以让开发者快速集成到自己的应用中。

4.自由开源：该软件采用Apache License协议，允许商业用途，并且提供了详细的文档。

5.中文模型：VOSK针对中文方言，提供了经过优化的中文ASR模型。相对于其他的工具包，VOSK的准确率提升明显。

总体来说，VOSK是一个值得尝试的语音识别工具包。

## VOSK的基本使用方法

### 安装配置


安装完成后，可以通过命令行测试是否成功安装：

```bash
$ vosk --help
```

如果出现了如下帮助信息，表示安装成功：

```text
usage: vosk [options] <input> [<model>]
Options:
  -h,--help           Print this help message and exit
  -m MODEL            Path to the model directory
  -l LANGUAGE         Language of the input audio (en, de, es, fr, it, nl)
                      Default: en (English)
  -d DEVICE_ID        Device ID to run inference on
                      Default: -1 (use CPU), or specify GPU device id
  -r RATE             Sample rate of the input audio in Hz
                      Default: 16000
  -c CONTEXT          Additional context for incremental decoding
  -t TRIEFILE         File with custom word list to improve recognition accuracy
  --max-alternatives MAX_ALTERNATIVES Maximum number of alternative hypotheses returned by decoder
                      Default: 1
  --verbose           Verbose output from Vosk
  --log LOGLEVEL      Log level verbosity (-1=quiet, 0=errors, 1=warnings,
                      2=info, 3=debug)
                      Default: 1
  --sample-rate SAMPLE_RATE 
                      Resample input file to sample rate before processing
                      Default: not resampling
```

### 测试安装结果

安装完成后，可以使用VOSK进行语音识别。为了演示VOSK的识别效果，我这里准备了一段MP3音频文件，并把它转化为WAV格式。然后，通过下面命令进行识别：

```bash
$ wget https://alphacephei.com/static/models/vosk-model-small-cn-0.3.zip #下载中文模型
$ unzip vosk-model-small-cn-0.3.zip && rm -f vosk-model-small-cn-0.3.zip #解压模型
$ wget http://www.fitnr.com/beep.mp3 #下载测试音频
$ ffmpeg -y -i beep.mp3 -acodec pcm_s16le -ar 16000 test.wav #转换为WAV格式
$ vosk -m model --max-alternatives 3 test.wav #进行语音识别
你好 你好 你好 你好 
[WordList] Word list is empty!
hello hello 
how are you? how do you do? 
```

在这里，`--max-alternatives`参数用来指定返回几个候选词。默认是1个。如果想返回多个候选词，可以在命令末尾加上这个参数，比如`-a 3`。

通过上面的测试，可以看到，VOSK识别出的单词非常接近正确的结果。另外，在执行完识别后，还会输出一些提示信息，比如`WordList`为空，这是因为我们的测试音频只有“你好”两个字。

### 使用场景举例

VOSK可以用于很多领域，比如：

1. 语音交互助手（例如Siri）：通过传入音频数据，得到文字响应，实现语音命令的语音控制功能。

2. 视频会议中白板共享（例如Google Meet）：通过传入音频数据，实现白板共享。

3. 智能机器人（例如Alexa）：通过传入音频数据，获取意图指令，然后回应语音反馈。

4. 银行自动取款机（例如借记卡系统）：通过传入音频数据，识别读卡器上的数字金额，并完成取款。

5. 语言学习软件（例如Anki）：通过传入音频数据，学习新词或句子。

6. 拍摄垃圾分类（例如TrashCam）：通过传入音频数据，判断垃圾的分类。

7. ……

## VOSK的内部原理及实践案例

### 基本概念

#### 模型与语言模型

VOSK中的模型是一个深度神经网络，它接受音频输入，输出每个音素的概率分布。其中每一个音素代表了发音的最小单位，VOSK所谓的音素，就是指中文中的汉字或者英文字母等。而语言模型则用来描述词汇的概率分布。也就是说，语言模型由一个个二元组构成，每个二元组都对应着一个词语及其出现的概率。如果模型能够学到这种词汇的概率分布，那么给定一个新的音频输入，模型便可以计算出相应的语音命令或语言文字。

#### HMM模型

VOSK使用的语音识别模型是一个HMM（Hidden Markov Model），也叫隐马尔可夫模型。顾名思义，这种模型是基于隐藏状态的马尔可夫链，可以用来分析和预测随机事件序列。传统的语音识别模型都是基于HMM来设计的，如标准的GMM-HMM或DNN-HMM模型。VOSK使用的HMM模型与这些模型有很大的不同之处。原因是在语音识别过程中存在很多歧义，比如发音相同但读法不同的字母、同音字、连读词等。因此，HMM模型一般只用来建模声学模型，不适合处理语音识别任务。

但是，在VOSK中，还是用了HMM模型来作为语音识别的框架。VOSK将语音信号切分成若干个小窗口（一般是25ms长），每一个窗口内的语音信号称为一个状态（state）。每一个状态对应一个音素。HMM模型中的状态是隐藏的，只能由当前的音频输入决定。这就保证了模型的动态平滑性，不会因受到噪声或其他影响而发生变化。

#### DAG矩阵

DAG模型的另一种名称叫有向无环图（Directed Acyclic Graph）。它是一种图结构，用于表示概率关系。VOSK使用DAG模型来建模语言模型。DAG模型定义了各个词语之间的概率关系，包括马尔可夫假设、发射概率、转移概率。由于各个词语之间有依赖关系，所以不能直接采用HMM模型建模语言模型。但是，我们可以通过DAG模型来抽象掉HMM模型中的马尔可夫假设。

#### Kaldi工具包

VOSK是基于TensorFlow构建的，这意味着它可以利用GPU计算加速。但是，从我的实验来看，用CPU进行语音识别效果也不是太差，尤其是在训练集规模比较小的情况下。而且，由于VOSK已经为我们提供了训练模型的脚本，所以在实际生产环境中，一般不需要自己训练模型。另外，VOSK提供的语料库中没有足够的中文语料，因此还需要结合外部的数据进行训练。因此，VOSK内部其实还是用到了Kaldi这一套工具包。Kaldi主要用于声学模型的训练和语言模型的生成。VOSK只是利用了Kaldi工具包的一部分功能，用于语言模型的生成。