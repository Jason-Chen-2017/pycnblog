
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## DAW (Digital Audio Workstation) 数字音频工作站
DAW（Digital Audio Workstation）即数字音频工作站，是指电脑或视频游戏主机中用于处理、编辑、制作各种数字音频文件的专用应用软件。其核心功能包括音乐创作、混音、合成、编曲、效果、视频编辑等。而目前较知名的DAW产品有FL Studio、Ableton Live、REAPER、Audacity、Sony Vegas Pro等。它们都属于商业化软件，用户需要付费购买使用，价格不菲，但其优秀的功能、高质量的音效、舒适的操作体验让许多音乐爱好者和主播倾倒。因此，了解如何创建自己的DAW，并根据自己的兴趣进行定制开发是一个有意义的事情。

## 博客网站介绍
DAW Publications, Inc. 是一家由人工智能专家组成的团队推出的新型科技型公司，致力于通过数据驱动的方式重新定义声音，解决音频工程领域中的痛点问题。我们将我们的探索与实践分享给广大的音频工程师和设计师，以期为更多的音频工作者提供更加便捷和有效的解决方案。

## DAW Publications, Inc. 概览
作为一家全球顶尖的音频工程师、设计师和开发者，我们一直秉持着“Be the best at what you do”的理念。因此，在日常的运营及各项活动中，我们会采用先进的管理模式、流程和工具，积极拓展我们的服务范围、扩大规模、提升品牌价值和社会影响力。

作为一家数字音频工程师，我们致力于提供一个专业的平台，为音频工程师、设计师和程序员提供无限可能。我们秉承“开放、互助、分享”的理念，不断向社区输出优秀的资源和教程。我们的知识产权策略鼓励大家自主创造，推动开源与共享，为广大音频工作者提供无限创造空间。

除了为音频工程师提供支持外，我们还提供一个公益性的平台，方便社会创业者参与其中，利用计算机视觉、机器学习等技术解决实际问题，促进科技进步。我们的目标是通过提供帮助，帮助更多的人实现自己的音频梦想。

# 2.核心概念与术语
## 核心概念
- Audio track：音轨，音频文件中的音频信息。
- Clip/region：剪辑/区域，从音轨中选择的一段音频片段。
- Automation：自动化，能够让音频工程师和设计师自动地完成重复性任务，节省时间、精力。如：节拍、调速、重置等。
- Envelope：envelope，即波形，声音的表现形式。它包括两个主要的属性：播放长度和动态范围。播放长度表示声音持续的时长；动态范围表示声音能承受的最大压力或压缩度。
- FX：效果器，是指音频工程师用来增强或改变音频特效的软件插件。例如，卷发器可以增加女声的原生力量，喇叭塔则可以发出令人神经紧张的噪音。
- Instrument/voice：乐器/音色，指具有独特的演奏方式和声音特征的乐器。例如，钢琴声音沿着音阶递减，吉他声音带有爆破感，巴洛克风格的音色吸引人群注意。
- Mastering：主音响，是指将不同音源合并到一起后，使之成为一个统一的音轨。它需要考虑混响、空间定位、精细化、人声分离等。
- MIDI：Musical Instrument Digital Interface，一种基于计算机的接口标准，允许不同的乐器接收输入信号，并产生对应的输出信号。它允许多种类型的乐器共存，并为音频制作提供了前所未有的能力。
- Mixer/track routing：混音器/轨道路由，指通过连接不同音轨、FX插件以及主音响，来创造出具有多样性的声音效果。
- Plugin/effect：插件/效果器，指的是一些特定功能的软件模块。例如，VSTi或AU插件可以实现特效化的声音效果，比如变声、降低幅度等。
- Reaper：REALLY ANIMATION EFFECTOR REPAIRER，是一种音频工程师可用的音频修复软件。它能够分析声音中的失真、混响、响度等，并根据设计师指定的效果对音轨进行调整。同时，Reaper还可以导出为许多主流格式，并兼容多种平台。
- Sequencer：音轨序列表，是指按照一定顺序播放音轨的一种功能。它可以使声音更有条理，也更容易被听众所接受。
- Speaker arrangement：扬声器排列，指将多个不同声音源定位、混合后扬出来，使之呈现出来的声音效果。
- Studio setup：工作室环境设置，包括音源、布局、混响效果、通道分配等。
- Tape delay：录音延迟，是指将一段录音文件的声音延迟多次之后再播放出来，得到延后的声音效果。它的作用类似于监听装置的收音机。
- Vocal effect：歌声效果，指调整歌声的声音大小、方向和速度，或者通过某些变声器发出类似古筝或民谣的声音。

## 术语
- ADAT：Advanced Dynamics and Acoustics Technology，即高级动态与声学技术。它是美国音频工程师协会（AASE）成立的一系列规格，被广泛应用于为高品质的立体声和多声道音频加上现场效果。
- AES/EBU：Audio Engineering Society/European Broadcasting Union，即音频工程师协会/欧洲广播联盟。这是国际音频工程师协会的前身，拥有著名的ACR和IEC规范。
- ARIS：Academy of Recording Arts and Sciences，即录音艺术与科学学院。这是欧洲的一个著名的学术机构。
- BRIO：Broadcast, Radio, Internet and Online，即广播、广播电台、互联网、网络等。是一门研究交流媒体应用和创新的艺术领域。
- DAW：数字音频工作站，通常指上述各个DAW产品。
- DCA：Digital Cinema Applications，即数字影像应用。是一种音频、视频、图像等数字化技术的集合，主要用于电影、电视、图文创作等领域。
- FDK：Fraunhofer Deutschland Kollaboration，即法兰克福-德国国际音频研讨会。它是德国的一个音频工程师协会。
- FFT：快速傅里叶变换，是一种信号处理技术。它可以把时域信号转换为频域信号。
- IEC：International Electrotechnical Commission，即国际电气电子委员会。
- ISDN：Integrated Services Digital Network，即集成服务数字网。这是一种短距通信技术，主要用于工业控制、车载通信等领域。
- LFE：Low Frequency Expansion，即低频伸缩。是一种音频工程技术，旨在在高架汽车的音频系统中增加低音频频率的响应。
- NRCS：National Rock and Country Music Standards，即国际音乐节和国家音乐标准。是一套由音乐家、制片人和音乐设备制造商共同制定的一系列标准，旨在使其制作出符合国际标准的专业音乐。
- NTS：Narrowband Telecommunications System，即窄带电信系统。它是一种采用频段覆盖率极低的通信技术，主要用于遥远地区的通信传输。
- OLBC：One Laptop per Child，即单机一子。是一种针对儿童数字学习的教育技术。
- PCM：Pulse Code Modulation，即脉冲编码调制。是一种编码方法，将模拟信号转化为数字信号。
- RECOTEC：Radio Electronic Conference On The Theory Of Communication，即理论通信的电台会议。它是欧洲理论通信学会举办的一系列学术会议。
- SLACS：Special Interest Group on Speech Analysis and Compression，即语音分析与压缩专题小组。这是德国语音信号处理协会（DVPM）的一项重要组别。
- SoX：Sound eXchange，即声音交换。是一个开源的音频处理工具箱，能够读取、写入、编辑多种类型音频文件。