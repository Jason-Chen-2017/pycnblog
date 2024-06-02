# 【LangChain编程：从入门到实践】构造器回调

## 1.背景介绍

### 1.1 什么是LangChain?

LangChain是一个用于构建大型语言模型(LLM)应用程序的Python库。它旨在简化与LLM交互的过程,并提供了一系列工具和模块,使开发人员能够轻松构建各种基于LLM的应用程序。LangChain支持多种LLM提供商,如OpenAI、Anthropic、Cohere等,并且可以与各种数据源集成,如文件、网页、API等。

### 1.2 LangChain的主要组件

LangChain主要由以下几个核心组件组成:

- **Agents**:智能代理,用于定义任务和执行策略。
- **Memory**:存储代理的状态和上下文信息。
- **Tools**:各种工具,如网页查询、文件读写等,供代理调用以完成任务。
- **Chains**:将LLM、工具、内存等组件链接在一起的序列。
- **Prompts**:用于指导LLM输出的提示模板。

### 1.3 构造器回调在LangChain中的作用

构造器回调(Callback Handler)是LangChain中一个非常重要的概念,它允许开发人员在LLM生成响应时插入自定义的逻辑。通过构造器回调,开发人员可以实现诸如记录、过滤、修改等功能,从而更好地控制LLM的输出。

## 2.核心概念与联系

### 2.1 构造器回调的核心概念

构造器回调的核心概念是在LLM生成响应的过程中,提供一个钩子函数,允许开发人员在特定时间点执行自定义逻辑。LangChain提供了多种类型的构造器回调,包括:

- **BaseCallbackHandler**:基础回调处理器,提供最基本的回调功能。
- **CallbackManager**:用于管理和组合多个回调处理器。
- **StdOutCallbackHandler**:将LLM的响应输出到标准输出。
- **StreamingStdOutCallbackHandler**:将LLM的流式响应输出到标准输出。

### 2.2 构造器回调与LangChain其他组件的联系

构造器回调与LangChain的其他组件密切相关,例如:

- **Agents**:构造器回调可以用于记录代理的执行过程,或者在代理执行期间插入自定义逻辑。
- **Memory**:构造器回调可以用于在内存中存储或检索信息。
- **Chains**:构造器回调可以应用于整个链,或者链中的特定步骤。
- **Prompts**:构造器回调可以用于修改或优化提示模板。

通过将构造器回调与其他组件结合使用,开发人员可以构建出更加强大和灵活的LLM应用程序。

## 3.核心算法原理具体操作步骤

构造器回调的核心算法原理是在LLM生成响应的过程中,在特定时间点执行自定义逻辑。具体操作步骤如下:

1. **创建回调处理器**:首先,开发人员需要创建一个回调处理器,继承自`BaseCallbackHandler`或其他回调处理器类。在回调处理器中,开发人员可以定义自己的回调函数,用于执行特定的逻辑。

2. **注册回调处理器**:接下来,需要将创建的回调处理器注册到LangChain的其他组件中,如代理、链等。这可以通过组件的构造函数或其他方法来实现。

3. **执行回调函数**:当LLM生成响应时,LangChain会在特定时间点调用注册的回调函数。开发人员可以在回调函数中执行自定义逻辑,如记录、过滤、修改等。

4. **处理回调结果**:回调函数可以返回一个布尔值或者`BaseCallbackHandlerOutput`对象,用于指示是否继续执行LLM生成响应的过程。开发人员可以根据需要处理回调函数的返回值。

5. **组合多个回调处理器**:LangChain还支持使用`CallbackManager`来组合多个回调处理器,并按照指定的顺序执行它们。这为开发人员提供了更大的灵活性和可扩展性。

以下是一个简单的示例,展示如何创建和使用一个自定义的回调处理器:

```python
from langchain.callbacks import BaseCallbackHandler
from langchain import LLMChain, OpenAI

class CustomCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM started generating response...")

    def on_llm_new_token(self, token, **kwargs):
        print(token, end="")

    def on_llm_end(self, response, **kwargs):
        print("\nLLM finished generating response.")

llm = OpenAI(temperature=0.9)
handler = CustomCallbackHandler()
chain = LLMChain(llm=llm, callbacks=[handler])

query = "What is the capital of France?"
result = chain.run(query)
print(result)
```

在上面的示例中,我们创建了一个自定义的回调处理器`CustomCallbackHandler`,它在LLM开始生成响应时打印一条消息,在生成每个新的token时打印该token,并在LLM结束生成响应时打印另一条消息。然后,我们将这个回调处理器注册到`LLMChain`中,并运行一个查询。

## 4.数学模型和公式详细讲解举例说明

在LangChain中,构造器回调的核心算法原理并不涉及复杂的数学模型或公式。但是,在某些特定场景下,构造器回调可能需要与数学模型或公式相关联。例如,在使用LLM进行数学计算或符号推理时,构造器回调可以用于检查或修改LLM生成的数学表达式或公式。

以下是一个简单的示例,展示如何使用构造器回调来检查和修改LLM生成的数学表达式:

```python
import re
from langchain.callbacks import BaseCallbackHandler
from langchain import LLMChain, OpenAI

class MathExpressionCheckHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        if token == ")":
            expr = self.curr_expr + token
            try:
                result = eval(expr)
                if result < 0:
                    return "(-" + expr[1:-1] + ")"
            except:
                pass
        self.curr_expr += token

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.curr_expr = ""

    def on_llm_end(self, response, **kwargs):
        self.curr_expr = ""

llm = OpenAI(temperature=0.9)
handler = MathExpressionCheckHandler()
chain = LLMChain(llm=llm, callbacks=[handler])

query = "What is the value of (3 - 5) * (4 - 2)?"
result = chain.run(query)
print(result)
```

在上面的示例中,我们创建了一个自定义的回调处理器`MathExpressionCheckHandler`,它在LLM生成每个新的token时检查当前的数学表达式是否合法。如果表达式合法且结果为负数,则将其修改为正确的负数形式。我们将这个回调处理器注册到`LLMChain`中,并运行一个涉及数学计算的查询。

需要注意的是,这只是一个简单的示例,在实际应用中,处理数学表达式和公式可能需要更复杂的算法和技术。但是,构造器回调为开发人员提供了一种灵活的方式来插入自定义逻辑,从而更好地控制LLM的输出。

## 5.项目实践:代码实例和详细解释说明

在本节中,我们将通过一个实际项目来演示如何使用构造器回调来增强LangChain应用程序的功能。我们将构建一个简单的问答系统,它可以从Wikipedia上检索相关信息,并使用LLM生成回答。在这个过程中,我们将使用构造器回调来记录LLM的响应,并在必要时对响应进行修改。

### 5.1 项目设置

首先,我们需要安装所需的Python库:

```bash
pip install langchain wikipedia openai
```

接下来,我们导入所需的模块和库:

```python
import re
from langchain.callbacks import BaseCallbackHandler, CallbackManager
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import wikipedia
```

### 5.2 创建自定义回调处理器

我们将创建两个自定义的回调处理器:

1. `LoggingCallbackHandler`:用于记录LLM的响应。
2. `ResponseModifierCallbackHandler`:用于修改LLM的响应,确保它不包含任何不当内容。

```python
class LoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM started generating response...")

    def on_llm_new_token(self, token, **kwargs):
        print(token, end="")

    def on_llm_end(self, response, **kwargs):
        print("\nLLM finished generating response.")

class ResponseModifierCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        if token.lower() in ["bad", "inappropriate"]:
            return " "

    def on_llm_end(self, response, **kwargs):
        response.output_text = re.sub(r'\b(bad|inappropriate)\b', '*****', response.output_text, flags=re.IGNORECASE)
        return response
```

`LoggingCallbackHandler`在LLM开始生成响应时打印一条消息,在生成每个新的token时打印该token,并在LLM结束生成响应时打印另一条消息。

`ResponseModifierCallbackHandler`在LLM生成每个新的token时检查该token是否为"bad"或"inappropriate"。如果是,则将其替换为空格。在LLM结束生成响应时,它会使用正则表达式将响应中的"bad"或"inappropriate"替换为"*****"。

### 5.3 创建问答系统

接下来,我们将创建一个`RetrievalQA`链,用于从Wikipedia上检索相关信息并生成回答。我们将使用`FAISS`向量存储来存储Wikipedia文章的嵌入向量,并使用`OpenAIEmbeddings`作为嵌入模型。

```python
# 初始化嵌入模型
embeddings = OpenAIEmbeddings()

# 初始化向量存储
vectorstore = FAISS.from_texts(wikipedia.get_doc_texts(), embeddings)

# 初始化LLM
llm = OpenAI(temperature=0.7)

# 创建问答链
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)
```

### 5.4 注册回调处理器

现在,我们将创建一个`CallbackManager`来组合我们的自定义回调处理器,并将其注册到问答链中。

```python
# 创建回调管理器
callback_manager = CallbackManager([LoggingCallbackHandler(), ResponseModifierCallbackHandler()])

# 将回调管理器注册到问答链中
qa = qa.with_callback_manager(callback_manager)
```

### 5.5 运行问答系统

最后,我们可以运行问答系统并查看结果。

```python
query = "What is the capital of France and what are some inappropriate things about it?"
result = qa({"query": query})
print(result["result"])
```

在这个示例中,我们询问法国的首都以及一些不当的事情。由于我们使用了`ResponseModifierCallbackHandler`,任何不当的内容都将被替换为"*****"。同时,由于我们使用了`LoggingCallbackHandler`,LLM生成响应的整个过程都将被记录在控制台中。

通过这个项目实践,我们可以看到构造器回调如何为LangChain应用程序增加额外的功能和灵活性。开发人员可以根据自己的需求创建各种自定义的回调处理器,从而更好地控制LLM的输出和行为。

## 6.实际应用场景

构造器回调在LangChain中有广泛的应用场景,可以用于各种基于LLM的应用程序。以下是一些常见的应用场景:

### 6.1 日志记录和调试

构造器回调可以用于记录LLM的输入、输出和执行过程,从而帮助开发人员调试和优化应用程序。例如,可以创建一个回调处理器来记录LLM生成响应的每个步骤,或者记录代理执行任务的详细信息。

### 6.2 内容过滤和修改

构造器回调可以用于过滤或修改LLM生成的内容,以确保其符合特定的要求或标准。例如,可以创建一个回调处理器来过滤掉不当或有害的内容,或者修改LLM生成的文本以符合特定的格式或风格。

### 6.3 性能监控和优化

构造器回调可以用于监