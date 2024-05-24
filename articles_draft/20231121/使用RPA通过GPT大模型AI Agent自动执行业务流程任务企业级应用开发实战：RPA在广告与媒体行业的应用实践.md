                 

# 1.背景介绍



大家好，我是一名技术专家、程序员和软件系统架构师，公司CTO。近期，随着人工智能的热潮，我对这个新领域产生了浓厚的兴趣，看到很多公司都在采用一些机器学习的手段来进行产品的优化和改进，比如自动驾驶汽车，智能机器人等等。另外，我也看到一些公司正在使用RPA(Robotic Process Automation) 来完成重复性的工作，如企业内部办公自动化、ERP数据导入导出自动化、电子邮件自动回复等等。那么，RPA可以帮助公司节省时间成本、提高效率、降低风险，如何结合机器学习、大模型AI Agent，用一种简单可靠的方式来实现自动化呢？

为了回答上面的问题，笔者将带领大家以实战案例的方式，介绍如何使用RPA通过GPT大模型AI Agent自动执行业务流程任务企业级应用开发实战。

首先，让我们从广告与媒体行业出发，了解一下这个行业的特点和主要场景。通常，广告主需要投放不同的广告类型，例如，短视频广告、多媒体广告、电视广告等等。每种广告类型都会有不同的素材要求，包括创意、投放位置、创意冲击力、时效性等方面。投放时，广告主会面临各种困难和问题。这些困难和问题就需要使用RPA来解决。

对于媒体行业来说，它主要分为创作型媒体和传播型媒体，前者包括图片、影像、直播等；后者则包括新闻网站、广播台等。除此之外，还有各种视频制作平台，如YouTube、B站、抖音、头条号等。每种类型的媒体都有自己的特点、需求，这些需求往往不能完全匹配到所有用户，需要依据用户的偏好进行个性化推荐。

所以，相比于一般的电脑应用程序，广告与媒体行业需要更加精准的推荐系统，并且需要高度自动化才能达到最佳效果。因此，使用RPA来自动化这一过程就显得尤为重要。

# 2.核心概念与联系
## GPT-3语言模型（Generative Pre-trained Transformer）简介

GPT-3是一个基于transformer结构的大模型语言模型。它由微软亚洲研究院(Microsoft Asia Research Institute)和卡耐基梅隆大学联合研发。它的很多功能都依赖于语言模型。

- 训练数据的来源：语料库、互联网文本等
- 模型架构：transformer结构
- 生成性能：人类级别的写作水平，目前已达到或超过了78%的准确率。
- 可扩展性：模型参数量达到2.7亿，训练速度快。

## Rasa Open Source Chatbot Platform简介

Rasa是一款开源的聊天机器人平台，可以用于构建智能对话系统。它的主要特征如下：

1. 易用性：它的DSL（Domain Specific Language）语法类似于自然语言，使得初学者容易上手。同时，提供了丰富的组件供用户选择，可以快速构建满足特定需求的聊天机器人。
2. 灵活性：Rasa支持自定义训练模型和自定义策略，可以满足用户定制化的需求。而且，它还提供了RESTful API接口，可以方便地集成到现有系统中。
3. 性能优秀：它的响应速度非常快，能够处理并回复多个请求。同时，它的持久化机制也能保证消息记录的完整性。

## 基于Rasa的广告与媒体业务线RPA自动化实践

接下来，让我们一起看看在RPA中如何结合GPT-3语言模型和Rasa来完成广告与媒体业务线的自动化流程。

### 方案描述

业务流：

1. 用户填写信息
2. RPA发送问询及获取模板列表
3. 用户根据模板列表选择模板
4. RPA根据模板生成对应的广告内容并发送给用户
5. 用户确认是否要投放广告
6. 如果用户选择投放，则RPA将广告内容提交给媒体平台

### 技术实现

#### 配置环境

- 安装Anaconda Python环境，下载安装最新版Python3.9。
- 安装最新版rasa（>=2.3.0）。
- 在Anaconda Prompt中安装transformers库。
- 从https://huggingface.co/transformers/pretrained_models.html下载GPT-3预训练模型，并把目录路径复制到config文件中。

```yaml
language: "en"
pipeline:
- name: "ConveRTTokenizer"
  model_url: "/path/to/GPT-3 pretrain model directory/"
- name: "ConveRTFeaturizer"
  model_url: "/path/to/GPT-3 pretrain model directory/"
- name: "ConveRTEmbedder"
  model_url: "/path/to/GPT-3 pretrain model directory/"
...
policies:
- name: MemoizationPolicy
- name: TEDPolicy
  max_history: 5
  epochs: 500
  batch_size: 16
  learning_rate: 0.0001
...
```

#### 创建RASA项目

```bash
cd Documents\Projects
rasa init --no-prompt
```

#### 创建训练数据

训练数据格式：

- intent: ask_template
- text: "Which template do you want to use?"
- entities: null

```yml
nlu:
- intent: greeting
  examples: |
    - hello
    - hi there!
    - what's up?
    - good morning
    - hey

- intent: ask_template
  examples: |
    - Can I see the templates again?
    - Do you have any other templates available?
    - Please list all possible templates for me.
    - Let me know which one of these is right for my brand campaign.
    - Which template do you prefer?
```

#### 定义RASA配置文件

```yaml
language: en
pipeline:
- name: ConveRTTokenizer
  model_url: /path/to/GPT-3 pretrain model directory/
- name: ConveRTFeaturizer
  model_url: /path/to/GPT-3 pretrain model directory/
- name: ConveRTEmbedder
  model_url: /path/to/GPT-3 pretrain model directory/
policies:
- name: MemoizationPolicy
- name: TEDPolicy
  max_history: 5
  epochs: 500
  batch_size: 16
  learning_rate: 0.0001
```

#### 撰写自定义组件

编写一个自定义组件，继承Action类。这个组件用于生成GPT-3的response，并返回给RASA。

```python
from typing import Any, Text, Dict, List
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" #关闭多线程
from rasa.shared.actions import Action
import logging
import torch
from transformers import pipeline as hf_pipeline
logger = logging.getLogger(__name__)
class CustomResponseGenerator(Action):
    def name(self) -> Text:
        return "action_custom_response"

    async def run(
        self,
        output_channel: "Text",
        nlg_response: str,
        tracker: "DialogueStateTracker",
        domain: "DomainDict",
    ) -> List[Dict[Text, Any]]:
        if not hasattr(self,'model'):
            logger.info("Loading the conversation model...")
            try:
                self.model = hf_pipeline('text-generation', model='convbert-base')
            except Exception as e:
                raise Exception("Failed to load convbert-base model") from e
        input_text = "{}:".format(tracker.latest_message['intent']['name'])
        prompt_text = input_text + " " + tracker.get_slot('content')
        reply_text = ""
        response = {"recipient_id": tracker.sender_id,
                    "text": "", 
                    "elements": []}
        while len(reply_text)<len(nlg_response)+1 and len(input_text)<300:
            inputs = hf_pipeline('conversational', model="microsoft/DialoGPT-large")("<|im_sep|>"+reply_text+input_text+"