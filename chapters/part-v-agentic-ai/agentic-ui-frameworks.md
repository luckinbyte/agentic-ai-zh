# 第 26 章 智能体 UI 框架

随着大语言模型(Large Language Model, LLM)从被动的文本生成器,演进为能够进行规划、工具使用和多步推理的主动智能体(agent),人类与之交互的界面也必须同步进化。传统的聊天界面——为单轮或短上下文对话而设计——已无法满足智能体工作流的需求:长时间运行的任务、分支化的决策树、并行的工具调用,以及对有效的人工监督的需要。本章综述了智能体 UI(用户界面)框架的全貌:那些支撑丰富、透明且值得信赖的人机协作的设计范式、组件库与实现模式。

## 26.1 动机:超越聊天框

### 为什么智能体需要专门的界面

一个聊天气泡传达的是一个结果;而一个智能体 UI 传达的是一个过程——推理过程、调用的工具、做出的决策,以及需要人工判断的节点。没有这种可见性,用户就无法信任、纠正或向智能体学习。

聊天界面与智能体界面之间的鸿沟,恰如自动售货机与熟练协作者之间的鸿沟。当智能体执行一个 20 步的研究任务、浏览网页、编写并运行代码、再综合成一份报告时,用户需要得到的答案,是简单的文本回复无法提供的:

- 智能体现在正在做什么?长时间运行的任务需要进度反馈;沉默滋生不信任。
- 智能体为什么做出这个决策?对推理过程的透明展示,能让用户尽早发现错误。
- 用了哪些工具,输入是什么?工具溯源(tool provenance)对于核实事实主张和审计行为至关重要。
- 我该在哪里介入?智能体必须主动呈现那些值得人工判断的决策点,而不是用每一个微决策淹没用户。
- 这一步能撤销吗?不可逆操作(发送邮件、修改文件、执行代码)需要明确的确认和回滚路径。

### 自动化偏见风险

关于人机自动化交互的研究一致表明,用户会过度信任自动化系统,尤其是当这些系统以自信且不带不确定性信号的方式呈现输出时 [398]。智能体 UI 必须通过呈现不确定性、展示推理过程,以及让用户能轻松地质疑或否决智能体决策,来主动抵消自动化偏见(automation bias)。

因此,智能体 UI 的设计处于人机交互(Human-Computer Interaction, HCI)、可解释 AI(Explainable AI, XAI)与软件工程三者的交汇处。其核心设计目标是:

1. **透明性(Transparency)**:让智能体的内部状态对用户清晰可读。
2. **可控性(Control)**:提供有意义的干预点,而不要求用户持续监督。
3. **信任校准(Trust Calibration)**:帮助用户建立关于智能体能力与局限的准确心智模型。
4. **高效性(Efficiency)**:最小化认知负荷;在恰当的时机呈现恰当的信息。
5. **可恢复性(Recoverability)**:让错误易于被发现并得以纠正。

## 26.2 智能体的 UI 范式

没有任何单一 UI 范式适用于所有智能体用例。合适的界面取决于任务时长、所需的人工参与程度、输出类型以及用户专业水平。其范围从完全对话式的聊天界面,一直延伸到几乎不需要人工交互的全自主仪表盘。

### 26.2.1 基于聊天的界面

聊天范式——消息气泡、文本输入框和滚动的历史记录——仍然是与 LLM 交互时最为人熟知的入口。它的优势在于学习成本低和自然语言的灵活性。面向智能体场景,聊天界面会通过以下方式增强:

- **流式响应(streaming responses)**:词元(token)随生成过程逐个出现,提供即时反馈并降低感知延迟。通过服务器推送事件(Server-Sent Events, SSE)或 WebSocket 实现。
- **行内工具指示器(inline tool indicators)**:消息流中嵌入小型徽标或可展开区域,显示工具调用的时机(例如 "[已搜索网络:climate change 2024]")。
- **输入指示器与状态消息**:"智能体正在思考……"、"正在运行 Python 代码……"、"正在获取结果……",在延迟间隙保持用户知情。
- **消息线程化(message threading)**:对于多轮智能体任务,可折叠的子线程可以容纳中间步骤,而不会使主对话显得杂乱。

**聊天界面用于智能体时的局限**

聊天界面会把本质上并行的过程串行化。当一个智能体同时扇出到五个工具时,线性的消息流会错误地表达实际的执行图。对于复杂的智能体工作流,聊天应当被更丰富的范式增强,或被其取代。

### 26.2.2 画布与产物式界面

画布(canvas)范式由 Claude Artifacts [^1] 与 ChatGPT Canvas [^2] 推广开来,它引入了一种分栏布局:左栏承载对话,右栏("画布"或"产物面板")则以一种实时、可编辑的产物(artifact)形式,展示生成的内容——代码、文档、图表、电子表格。

关键特征:

- **持久化产物**:生成的内容在各轮之间持久存在,可通过自然语言指令迭代精修("把图表改成蓝色"、"给函数加上错误处理")。
- **就地编辑**:用户可以直接编辑产物,智能体也能观察并对这些编辑做出响应。
- **版本历史**:产物维护修订历史,可回滚到任意先前状态。
- **多产物工作区**:高级实现支持多个同时存在的产物(例如一个代码文件、它的测试套件和一份文档页面)。

画布范式特别适合共创类任务:写作、编程、数据分析和设计——其产出是一份文档或产物,而非一段对话式回答。

[^1]: 译注:Claude Artifacts,Anthropic 推出的画布式产物功能。
[^2]: 译注:ChatGPT Canvas,OpenAI 推出的画布编辑界面。

### 26.2.3 工作流可视化

对于执行结构化计划——步骤序列或步骤图——的智能体,工作流可视化界面将计划显式化并使其可追踪。这一范式常见于:

- **智能体流水线(LangGraph、AutoGen、CrewAI)**:智能体的执行图被渲染为有向无环图(Directed Acyclic Graph, DAG)或流程图,节点表示步骤,边表示数据流或控制流。
- **任务分解视图**:智能体的高层计划以清单或甘特式时间线呈现,每个子任务可展开以揭示其自身步骤。
- **实时进度跟踪**:节点在执行时会改变颜色或显示旋转动画;完成的节点显示输出;失败的节点显示错误详情。

LangGraph Studio [^3] 是这一范式的典型代表,它为 LangGraph 智能体提供基于图的调试器和可视化器。用户可以检查每个节点的状态、回放执行过程,并注入修改后的状态以测试替代路径。

[^3]: 译注:LangGraph Studio,LangGraph 的桌面版可视化 IDE。

### 26.2.4 仪表盘与监控界面

对于长时间运行或生产级智能体,仪表盘界面提供运维视角:

- **实时状态**:哪些智能体在运行、空闲或失败;当前任务与步骤。
- **资源指标**:词元消耗、API 调用次数、延迟直方图、成本估算。
- **队列管理**:待处理任务、优先级排序、速率限制状态。
- **告警与异常检测**:异常行为(过多重试、成本激增、反复失败)以通知形式呈现。
- **历史分析**:任务完成率、平均时长、错误频率随时间的变化。

仪表盘界面通常使用 Grafana [^4]、定制的 React 仪表盘或 Streamlit 之类的工具构建,面向运维人员而非最终用户。

[^4]: 译注:Grafana,开源的可观测性与可视化平台。

### 26.2.5 协作式界面

协作式 UI 将智能体视为共享工作区——一份文档、一个代码库或一张设计画布——中与人类协作者并肩的贡献者。关键特性包括:

- **在场指示器(presence indicators)**:智能体在共享工作区中以具名的光标或头像形式出现。
- **变更归属(change attribution)**:智能体所做的编辑在视觉上区别于人类编辑(例如带颜色编码的 diff)。
- **行内建议**:智能体以追踪修订或评论的形式提出修改,人类可以接受、拒绝或修改。
- **冲突解决**:当智能体与人类同时编辑同一区域时,界面会呈现冲突并协助解决。

这一范式正在 Cursor [^5](带 AI 的协作式代码编辑)、Notion AI [^6] 以及集成 Gemini 的 Google Docs [^7] 等工具中兴起。

[^5]: 译注:Cursor,带 AI 协作的代码编辑器。
[^6]: 译注:Notion AI,Notion 集成的 AI 功能。
[^7]: 译注:Google Docs with Gemini,Google 文档集成的 Gemini 能力。

### 26.2.6 带检查点的自主运行

在自主性谱系的远端,一些智能体在很大程度上独立运行——浏览网页、编写代码、执行命令——仅在需要人工批准的预设检查点处浮现出来。这一范式用于:

- **计算机使用类智能体(computer-use agents)**(Anthropic Computer Use [^8]、OpenAI Operator [^9]):智能体控制浏览器或桌面;界面显示实时屏幕画面,并在不可逆操作前暂停等待批准。
- **带门控的自动化流水线**:类 CI/CD 风格的工作流,智能体完成一个阶段后等待人工"合并"才会继续。
- **定时智能体**:按计划运行的智能体,异步报告结果,配以基于通知的界面来审查输出并批准后续动作。

[^8]: 译注:Anthropic Computer Use,Anthropic 推出的计算机使用能力。
[^9]: 译注:OpenAI Operator,OpenAI 推出的浏览器/计算机使用智能体。

**检查点界面实践**

一个被指派"清理我的收件箱"任务的智能体,可能会自主地分类并归档 500 封邮件,然后暂停并呈现一份摘要:"我发现了 23 封来自邮件列表、而你 6 个月未曾打开过的邮件。我应该全部退订、部分退订,还是不退订?"用户审查一份清单、做出选择,智能体随即继续。这一模式——以人工决策点穿插自主执行——在效率与控制之间取得平衡。

## 26.3 智能体的关键 UI 组件

无论采用何种总体范式,智能体 UI 都共享一组反复出现的组件。本节编目其中最重要的若干组件,并为每个组件给出设计指引。

### 26.3.1 思维与推理展示

现代 LLM,尤其是经思维链(chain-of-thought)或扩展思考训练的模型(如 OpenAI o1/o3、启用扩展思考的 Anthropic Claude),在产出最终响应之前会生成大量内部推理。将这些推理呈现出来是一把双刃剑:它增加了透明度,却也可能用冗长的内心独白淹没用户。

最佳实践:

- **可折叠的推理块**:展示一个摘要("思考了 12 秒"),并为需要细节的用户提供展开切换。
- **渐进式披露(progressive disclosure)**:默认只显示最终结论;推理按需可得。
- **结构化推理**:若模型产出结构化的思维(假设、证据、结论),则用视觉层级来渲染,而非一大段文字墙。
- **推理与响应的区分**:在视觉上明确区分内部推理(可能包含错误或错误的起步)与最终响应。

### 26.3.2 工具使用可视化

工具调用是智能体与世界交互的主要机制。将其可视化对于建立信任和调试都不可或缺。

**工具调用的剖析**

每次工具调用都有四个值得展示的组成部分:(1)工具名称与图标,(2)输入参数(可能是很大的 JSON),(3)输出/结果(可能很大),(4)时序(延迟)。界面必须在完整性与可读性之间取得平衡。

工具可视化的设计模式:

- **行内工具卡片**:消息流中的紧凑卡片,显示工具名、输入的一行摘要以及状态(运行中/成功/出错)。可展开查看完整详情。
- **工具时间线**:一条横向时间线,展示一轮中所有工具调用及其耗时,便于识别瓶颈。
- **输入/输出 diff**:对于修改状态的工具(如文件编辑),展示前后差异。
- **工具图标与品牌**:为常见工具(网络搜索、代码执行、文件系统、API)采用可辨识的图标,便于快速扫视。
- **错误高亮**:失败的工具调用以红色展示,并附上错误消息及任何重试尝试。

### 26.3.3 进度指示器

多步智能体任务需要丰富的进度反馈:

- **步骤级进度**:一个带编号的计划步骤清单,每完成一步打上一个勾。对于动态计划,步骤可随智能体调整而增删。
- **词元流式指示器**:生成期间的闪烁光标或动画省略号;面向高级用户的每秒词元计数器。
- **预计完成时间**:在可行时,基于任务复杂度和历史表现给出 ETA。以恰当的不确定性显示("大约 2–5 分钟")。
- **子任务嵌套**:对于层次化任务,提供树状结构的进度视图,子任务可展开。
- **取消**:一个清晰可见的"停止"按钮,优雅地停止智能体并总结已完成的工作。

### 26.3.4 审批门

审批门(approval gates)是人在回路(human-in-the-loop, HITL)控制的主要机制。它们必须被设计得既有信息量(给用户足够上下文以做出好决策),又不会令人疲劳(对每一个琐碎动作都要求审批)。

**审批门中的告警疲劳**

若智能体过于频繁地请求审批,用户会开始不假思索地批准而不阅读——这恰恰违背了审批门的初衷。分层审批策略(见 26.7 节)对于维持有意义的监督至关重要。

审批门的界面元素:

- **动作摘要**:用平实语言描述智能体想做什么("向 john@example.com 发送一封附有报告的邮件")。
- **风险指示器**:动作可逆性的视觉信号(绿色 = 易于撤销,黄色 = 难以撤销,红色 = 不可逆)。
- **批准 / 拒绝 / 修改**:三选一界面;"修改"会在批准前为动作参数打开一个编辑器。
- **上下文面板**:可展开区域,展示智能体为何要采取此动作(相关推理、先前步骤)。
- **超时行为**:明确说明若用户不响应会发生什么(智能体暂停,而非继续)。

### 26.3.5 上下文展示

智能体维护着影响其行为的内部状态——记忆、可用工具、检索到的文档、对话历史。让这些状态可见,有助于用户理解并预测智能体的行为。

- **记忆面板**:展示智能体当前对用户、任务和先前交互"记得"什么。用户可编辑。
- **可用工具列表**:智能体当前可使用哪些工具,带启用/禁用开关。
- **检索到的上下文**:当前处于智能体上下文窗口(context window)中的文档或数据块,附来源引用。
- **词元预算指示器**:上下文窗口已消耗多少,帮助用户判断何时该开启新会话。

### 26.3.6 错误与恢复界面

智能体会失败——工具返回错误、模型产生幻觉、计划变得不可行。界面必须优雅地处理失败:

- **错误卡片**:行内展示失败,附错误类型、消息以及智能体的解读。
- **重试控件**:手动重试按钮,可选地调整参数。
- **替代方案**:当主方案失败时,智能体提出替代方案;界面以可选项的形式呈现。
- **部分结果**:若多步任务在中途失败,界面展示已完成步骤及其输出,保留部分价值。
- **升级路径**:当智能体无法继续时,清晰的人工支持或人工完成的路径。

### 26.3.7 置信度指示器

LLM 是具有校准良好(或校准不佳)不确定性的概率系统。呈现置信度有助于用户判断何时该信任、何时该核实:

- **言语型对冲展示**:高亮诸如"我不确定"或"你可能需要核实一下"之类的短语,以引起对低置信度主张的注意。
- **来源质量指示器**:对于检索到的信息,展示来源的时效性、权威性和相关性评分。
- **显式不确定性请求**:一个"你有多大把握?"按钮,促使智能体自我评估并解释其不确定性。
- **核实建议**:对于高风险输出,智能体主动建议核实步骤("我建议独立核对这个计算")。

## 26.4 框架与库

一个不断壮大的框架生态正在加速智能体 UI 的开发。我们按主要语言和用例,对其中采用最广泛的若干进行综述。

### 26.4.1 Vercel AI SDK

Vercel AI SDK [399] 是一个用于在 React、Next.js、Svelte 和 Vue 中构建流式 AI 界面的 TypeScript/JavaScript 库。它是生产级 Web 智能体 UI 中使用最广的框架。

核心抽象:

- `useChat`:一个 React hook,管理带流式支持、消息历史和加载状态的聊天对话。
- `useCompletion`:用于带流式的单轮文本补全的 hook。
- `useObject`:流式传输结构化 JSON 对象,支持复杂输出的渐进式渲染。
- `streamText` / `streamObject`:通过 HTTP 流式传输 LLM 响应的服务端函数。

**生成式 UI(AI SDK RSC)**:Vercel AI SDK 最具特色的特性是它通过 React 服务器组件(React Server Components, RSC)支持生成式 UI(generative UI)。LLM 不再返回文本,而是可以调用工具,工具的结果被渲染为任意的 React 组件——天气小部件、股票图表、预订表单——并直接流入界面。详见 26.5 节。

### 26.4.2 Chainlit

Chainlit [400] 是一个 Python 框架,用于以极少的样板代码构建生产级智能体 UI。它在 LangChain 与 LlamaIndex 生态中尤为流行。

关键特性:

- **步骤可视化**:Chainlit 原生地将 LangChain 与 LlamaIndex 的执行步骤渲染为一棵可折叠的树,展示每一次链调用、检索和工具调用。
- **多模态支持**:开箱即用的文件上传、图像展示、音频回放和 PDF 渲染。
- **认证与会话**:内置用户认证、持久化对话历史和多用户支持。
- **自定义元素**:React 组件可以从 Python 注册并渲染,实现丰富的自定义可视化。
- **反馈收集**:内置点赞/点踩反馈,可选附评论,并存入数据库。

```python
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


agent = create_react_agent(
    ChatOpenAI(model="gpt-4o"), tools=[search]
)


@cl.on_message
async def on_message(message: cl.Message):
    # 当使用回调处理器时,Chainlit 会自动将每个步骤
    # 渲染为一个可折叠的 UI 元素
    async with cl.Step(name="Agent", type="run") as step:
        step.input = message.content
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": message.content}]},
            config={"callbacks": [cl.LangchainCallbackHandler()]}
        )
        output = result["messages"][-1].content
        step.output = output
        await cl.Message(content=output).send()
```

清单 26.1:带步骤可视化的最小 Chainlit 智能体

### 26.4.3 Gradio

Gradio [401] 是一个 Python 库,用于快速构建机器学习 demo 和智能体界面。其 `gr.ChatInterface` 与 `gr.Blocks` API 使得用极少的代码快速原型化对话式智能体成为可能。

用于智能体 UI 的优势:

- **零配置部署**:通过 Hugging Face Spaces 一行命令分享。
- **自定义组件**:Gradio 自定义组件系统允许构建能与 Python 后端无缝集成的 React 组件。
- **多模态输入**:文件上传、图像、音频、视频和网络摄像头输入,配置极少。
- **流式**:原生支持基于生成器的流式响应。

局限:Gradio 的布局系统不如完整的 React 框架灵活,且其状态管理是会话级(session-scoped)的,这使得复杂的多智能体协调颇具挑战。

### 26.4.4 Streamlit

Streamlit [402] 是一个面向数据应用的 Python 框架,已被广泛用于智能体仪表盘和监控界面。其响应式执行模型——每次交互都重跑整个脚本——简单,但对于复杂的智能体工作流可能成为限制。

智能体用例:

- **智能体仪表盘**:使用 `st.metric`、`st.dataframe` 和 `st.status` 实现实时指标、任务队列和状态展示。
- **会话状态**:`st.session_state` 在重跑之间持久化智能体状态,支持多轮对话。
- **流式**:`st.write_stream` 渐进式渲染生成器输出。
- **片段**:`@st.fragment` 装饰器支持局部重跑,提升实时更新仪表盘的性能。

### 26.4.5 OpenAI Assistants Playground

OpenAI Assistants Playground 充当智能体 UI 设计的参考实现。它演示了:

- 基于线程(thread)的对话管理与持久化历史。
- 文件附件与检索可视化。
- 带输出展示(stdout、图像、文件)的代码解释器执行。
- 带输入/输出检查的函数调用展示。
- 展示模型调用与工具调用序列的运行步骤可视化。

虽然它并非用于构建自定义 UI 的框架,但 Playground 的设计模式被广泛效仿。

### 26.4.6 LangGraph Studio

LangGraph Studio [403] 是一款桌面应用,为 LangGraph 智能体提供可视化 IDE。它是目前最成熟的工具使用与工作流可视化环境。

特性:

- **图可视化**:交互式渲染智能体的状态机,节点表示智能体步骤,边表示转移。
- **状态检查**:在执行中的任意点,完整智能体状态(所有变量、记忆、工具结果)可作为结构化 JSON 检查。
- **时间旅行调试**:回放任意先前执行步骤,修改状态,并从该点重新运行。
- **人在回路集成**:可在任意节点设置断点;执行会暂停并等待人工输入后才继续。
- **多智能体支持**:可视化监督者-子智能体(supervisor-subagent)层级与智能体之间的消息传递。

### 26.4.7 框架对比

表 26.1 总结了上述框架的关键特征。

表 26.1:智能体 UI 框架对比。

| 框架 | 语言 | 流式 | 工具可视化 | 多智能体 | 生成式 UI | 生产级 |
|---|---|---|---|---|---|---|
| Vercel AI SDK | TypeScript | ✓ | 部分 | 部分 | ✓ | ✓ |
| Chainlit | Python | ✓ | ✓ | 部分 | 部分 | ✓ |
| Gradio | Python | ✓ | ◦ | × | ◦ | ✓ |
| Streamlit | Python | ✓ | ◦ | × | × | ✓ |
| OAI Playground | N/A(托管) | ✓ | ✓ | × | × | × |
| LangGraph Studio | Python/TS | ✓ | ✓ | ✓ | × | 部分 |

> 表中 ✓ = 完整支持,◦ = 部分支持,× = 不支持;OAI Playground = OpenAI Assistants Playground。

## 26.5 生成式 UI

### 生成式 UI 的概念

传统的 LLM 界面将模型输出渲染为文本或 markdown。生成式 UI 将此反转:模型的工具调用生成 UI 组件。模型不仅决定说什么,还决定如何呈现——作为图表、表单、地图或日历小部件——依据内容类型与用户意图。

生成式 UI 代表了 LLM 与界面之间关系的一次根本转变。开发者不再预先指定所有可能的 UI 状态,而是由模型动态地选择并参数化适合当前上下文的 UI 组件。

### 26.5.1 用于动态界面的 React 服务器组件

Vercel AI SDK 的 RSC(React 服务器组件 [^rscc])集成是生成式 UI 最成熟的实现。其架构如下:

[^rscc]: 译注:React Server Components,RSC。

1. 用户向一个 Next.js [^nx] 服务端动作(server action)发送消息。
2. 服务端携带一组工具调用 LLM,每个工具都关联一个 React 组件。
3. 当 LLM 调用某工具(如 `show_weather`)时,服务端将该工具输出作为 props 渲染对应的 React 组件。
4. 渲染后的组件作为 React 服务器组件流式传输到客户端,出现在聊天中行内位置。

[^nx]: 译注:Next.js,基于 React 的全栈框架。

```tsx
// app/actions.tsx (Server Action)
import { streamUI } from 'ai/rsc';
import { openai } from '@ai-sdk/openai';
import { WeatherCard } from '@/components/WeatherCard';
import { StockChart } from '@/components/StockChart';

export async function chat(userMessage: string) {
  const result = await streamUI({
    model: openai('gpt-4o'),
    messages: [{ role: 'user', content: userMessage }],
    tools: {
      show_weather: {
        description: 'Display current weather for a location',
        parameters: z.object({
          location: z.string(),
          unit: z.enum(['celsius', 'fahrenheit']),
        }),
        // 工具结果被渲染为一个 React 组件
        generate: async ({ location, unit }) => {
          const data = await fetchWeather(location, unit);
          return <WeatherCard data={data} />;
        },
      },
      show_stock: {
        description: 'Display stock price chart',
        parameters: z.object({ ticker: z.string() }),
        generate: async ({ ticker }) => {
          const data = await fetchStockData(ticker);
          return <StockChart ticker={ticker} data={data} />;
        },
      },
    },
  });
  return result.value;
}
```

清单 26.2:用 Vercel AI SDK RSC 实现的生成式 UI(TypeScript)

### 26.5.2 基于内容类型的自适应界面

生成式 UI 使界面能够适应所呈现内容的性质:

- 表格数据 → 可排序、可过滤、带导出选项的数据表。
- 地理数据 → 带标记和图层的交互式地图。
- 时间序列 → 带标注的可缩放折线图。
- 代码 → 带运行按钮的语法高亮编辑器。
- 文档 → 带标注工具的格式化文档查看器。
- 表单/结构化输入 → 动态生成的表单字段。

模型充当 UI 编排者,为每条信息选择最合适的呈现方式。这减少了开发者预先设想所有可能输出类型并预先构建相应组件的需要。

**生成式 UI 的局限**

有多少 UI 生成应当委托给模型?完全由模型驱动的 UI 有不一致、无障碍性失败和安全漏洞的风险(例如,模型生成一个把数据提交到非预期端点的表单)。在实践中,当模型从一个经过精心挑选、预先构建、可访问且安全的组件库中选择,而不是生成任意的 HTML 或 JSX 时,生成式 UI 的效果最好。

## 26.6 流式与实时模式

流式是智能体 UI 的基石:它把体验从"等待结果"转变为"观看智能体工作"。本节涵盖关键的流式模式及其实现考量。

### 26.6.1 词元流式

词元流式在词元生成时增量地交付 LLM 输出,而非等待完整响应。常用两种传输机制:

- **服务器推送事件(Server-Sent Events, SSE) [^sse]**:一种从服务端到客户端的单向 HTTP 流。每个事件携带一个词元块。SSE 简单,可在标准 HTTP/1.1 上工作,且由浏览器自动重连。它是 LLM 流式 API 的主流机制(OpenAI、Anthropic、Google 均使用 SSE)。
- **WebSocket**:双向持久连接。实现更复杂,但对于客户端需要在流式过程中途发送数据的交互式流式场景(如打断智能体、在生成中途提供反馈)是必需的。

[^sse]: 译注:Server-Sent Events。

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import json

app = FastAPI()
client = AsyncOpenAI()


async def token_stream(prompt: str):
    """生成 SSE 格式词元块的生成器。"""
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            # SSE 格式:"data: <json>\n\n"
            yield f"data: {json.dumps({'token': delta.content})}\n\n"
        elif chunk.choices[0].finish_reason:
            yield f"data: {json.dumps({'done': True})}\n\n"


@app.get("/stream")
async def stream_endpoint(prompt: str):
    return StreamingResponse(
        token_stream(prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

清单 26.3:用 FastAPI 实现 SSE 词元流式

### 26.6.2 工具调用流式

现代 LLM API 支持流式工具调用:工具名与参数被增量地流式传输,使界面能在工具被真正调用之前就显示"智能体正在调用 search_web,查询:'climate change 2024'……"。这需要解析不完整的 JSON,可借助流式 JSON 解析器完成。

工具调用流式的模式:

- **渐进式参数展示**:随参数流入逐步显示工具参数,即使在调用完成之前。
- **并行工具调用指示器**:当模型同时调用多个工具时,把它们全部显示为待处理,随后在每个结果到达时更新。
- **工具结果流式**:某些工具(如代码执行、网页抓取)本身可以流式产出结果;将这些结果渐进地透传到界面。

### 26.6.3 多智能体流式

在多智能体系统(multi-agent system)中,多个智能体可能同时产出输出。界面必须处理并行流:

- **带标签的智能体流**:每个流以智能体身份标记;界面将其渲染在不同的泳道或面板中。
- **流合并**:对于监督者-子智能体模式,监督者的流可能与子智能体的流交织;界面必须维持连贯的顺序。
- **背压(backpressure)**:若界面无法像流到达那样快地渲染(例如多个智能体同时生成),则必须有一种背压机制来防止缓冲区溢出。策略包括:丢弃中间词元(只显示最新)、批量更新,或暂停较慢的流。

### 26.6.4 乐观式 UI 更新

乐观式 UI 更新(optimistic UI updates)通过在服务端确认之前立即在界面中反映用户动作,来提升感知响应速度:

- 当用户发送一条消息时,它立即(乐观地)出现在聊天历史中,而请求仍在传输途中。
- 当一个审批门被接受时,界面立即把该动作显示为"已批准",并开始显示智能体的后续步骤,即使服务端尚未处理该批准。
- 若服务端返回错误,乐观式更新被回滚并显示错误状态。

### 26.6.5 背压处理

在高吞吐的智能体场景中,到达数据的速率可能超过界面的渲染能力。管理背压的策略:

- **词元批处理**:将词元缓冲 50–100ms 并批量渲染,而非逐个渲染,降低 DOM 更新频率。
- **虚拟滚动**:对于长输出,只渲染内容的可见部分,丢弃屏幕外的 DOM 节点。
- **节流更新**:对于指标与状态展示,无论到达数据速率如何,都以固定速率(如 10 Hz)更新。
- **渐进式细节**:在高吞吐期间显示摘要视图;完整细节按需可得。

## 26.7 人在回路的 UI 设计

人在回路(Human-in-the-loop, HITL)交互是智能体 UI 中最具影响力的设计挑战之一。其目标是维持有意义的人工监督,而又不制造抵消自动化效率优势的瓶颈。

### 26.7.1 何时打断智能体

并非所有智能体动作都值得人工审查。一套有原则的打断策略会考虑:

- **可逆性**:不可逆动作(删除文件、发送邮件、购买)总是值得审批。可逆动作(读文件、网络搜索)通常不需要。
- **范围**:影响外部系统或其他人的动作,比纯本地动作值得更多审视。
- **置信度**:当智能体对自身对用户意图的理解置信度低时,它应当请求澄清,而非贸然继续。
- **成本**:高成本动作(大量 API 调用、昂贵计算)值得审批。
- **新颖性**:在此情境下智能体此前未曾做过的动作,比例行动作值得更多审视。

### 26.7.2 分层审批工作流

一套分层审批策略在监督与效率之间取得平衡:

**三层审批模型**

- **第一层(自动批准)**:低风险、可逆、例行动作。示例:网络搜索、读文件、调用只读 API。智能体不打断地继续;动作被记录以备审计。
- **第二层(通知)**:中等风险动作。界面显示一个非阻塞式通知("智能体已将一封邮件草稿发送到你的草稿文件夹"),用户可异步审查。在一个短暂窗口(如 30 秒)内允许在动作最终确定前取消。
- **第三层(要求审批)**:高风险、不可逆或高成本动作。智能体暂停并呈现一个阻塞性审批门。用户必须明确地批准、拒绝或修改,智能体才会继续。

各层之间的阈值可由用户配置("发送邮件前总是询问"),也可从用户行为中学习(若用户总是批准网络搜索,则在将来自动批准它们)。

### 26.7.3 反馈机制

除审批门外,智能体 UI 还应提供丰富的反馈机制,帮助智能体随时间改进:

- **点赞/点踩**:对响应的简单二值反馈,被存储并用于 RLHF 微调或偏好学习。
- **行内更正**:用户可直接编辑智能体输出;原始输出与更正后输出之间的差异是一种训练信号。
- **偏好选择**:当智能体提供多个选项时,用户的选择是一种偏好信号。
- **显式指令**:"别再这样做"、"在 X 之前总是询问"、"在 Y 与 Z 之间偏好 Y"——这些自然语言指令更新智能体的行为策略。
- **带理由的评分**:可选的自由文本解释,伴随评分一同提供,比二值反馈给出更丰富的信号。

### 26.7.4 通过 UI 交互来教导智能体

最成熟的 HITL 界面把每一次交互都视为教导机会:

- **示范**:用户手动执行一项任务;智能体观察并学习偏好的方式。
- **带泛化的更正**:当用户更正一个智能体动作时,界面会问"我以后是否都该换种方式做?"以泛化该更正。
- **偏好引出**:周期性地提示用户比较两种智能体行为,并指出偏好哪一种。
- **行为画像**:界面维护一个可见的"偏好"画像供用户审查与编辑,使智能体学到的行为透明且可控。

## 26.8 无障碍性与信任

信任不是一项功能——它是一个始终如预期般行动、清晰地解释自身、并能从失败中优雅恢复的系统的涌现属性(emergent property)。智能体 UI 必须把信任作为头等关切来设计。

### 26.8.1 解释智能体的决策

智能体 UI 中的可解释性远不止展示思维链。它要求:

- **决策依据**:对于有后果的决策,智能体不仅应解释它决定了什么,还应解释为什么——考虑了哪些因素、拒绝了哪些备选、做出了哪些假设。
- **来源归属**:主张应链接到其来源;检索到的文档应可被引用。
- **反事实解释**:"如果你说的是 X 而非 Y,我会做 Z"——帮助用户理解智能体的决策边界。
- **不确定性量化**:对置信度的显式陈述,并附上驱动不确定性的因素。

### 26.8.2 展示置信度水平

置信度指示器必须是经过校准且富有意义的:

- **言语型置信度**:自然语言表达("我相当确信"、"我对这个不太确定")对大多数用户而言比数值概率更易解读。
- **视觉型置信度**:颜色编码(绿/黄/红)、图标变体或字重,可以在不增加文字的情况下编码置信度。
- **按主张的置信度**:对于多主张响应,逐主张的置信度指示器(如行内脚注)比单一响应级评分更有信息量。

### 26.8.3 撤销与回滚能力

在技术可行的情况下,每一个有后果的智能体动作都应可撤销:

- **带撤销的操作日志**:所有智能体动作的按时间顺序日志,每个可逆动作配一个"撤销"按钮。
- **基于快照的回滚**:对于有状态任务(如代码编辑、文档写作),周期性快照支持回滚到任意先前状态。
- **试运行模式**:在执行一个计划之前,智能体可以先模拟它并展示预测的状态变化,让用户在任何真实动作发生前批准或修改。
- **优雅降级**:当撤销不可能时(如邮件已发出),界面清晰地传达这一点,并提供最佳的可用替代(如发送一封后续邮件)。

### 26.8.4 界面中的审计轨迹

对于企业与受监管的用例,审计轨迹(audit trails)至关重要:

- **不可变的操作日志**:每一个智能体动作、工具调用和人工审批都被记录,附时间戳、用户身份和完整参数。
- **可导出的历史**:审计轨迹可作为 JSON、CSV 或 PDF 导出,用于合规报告。
- **diff 视图**:对于文档或代码修改,审计轨迹包含前后差异。
- **会话回放**:能够逐步回放整个智能体会话的能力,用于调试或合规审查。

### 26.8.5 管理用户预期

校准失当的预期是用户不信任的主要来源。智能体 UI 应当主动管理预期:

- **能力披露**:清晰、易访问的文档,说明智能体能做与不能做什么。
- **局限确认**:当智能体遇到超出其能力的任务时,它应明确说明,而不是尝试后默默失败。
- **不确定性沟通**:主动沟通不确定性,而非等待用户去发现错误。
- **一致的人设**:一致的智能体身份与沟通风格,能建立熟悉感与可预测性。

**通过透明建立信任:一个案例研究**

设想一个被指派预订航班的智能体。低信任的 UI 会呈现:"我已为你预订了航班。确认号:AA1234。"而高信任的 UI 会呈现:(1)所用搜索参数的摘要,(2)考虑过的备选方案以及为何选择了这一航班,(3)采取的确切动作(对预订系统的 API 调用),(4)带预订链接的确认详情,(5)一个在未来 24 小时内有效的撤销选项,(6)一条关于智能体不能做什么的说明(如"我无法修改这次预订;你需要直接致电航空公司")。第二种 UI 占据更多屏幕空间,却建立了用户对智能体行动正确的信心,并提供了在需要时核实与恢复所需的全部信息。

## 26.9 实现示例:一个全栈智能体 UI

下面我们给出一个具体实现示例,在 Python/React 技术栈中结合了流式、工具可视化与审批门。后端使用 FastAPI 与 LangGraph;前端使用 React,并把 Vercel AI SDK 的模式适配到定制后端。

### 26.9.1 后端:带流式与审批门的 FastAPI + LangGraph

```python
# backend/main.py
import asyncio
import json
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

app = FastAPI()

# -- 工具定义 ----------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [simulated results]"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. REQUIRES HUMAN APPROVAL."""
    return f"Email sent to {to} with subject '{subject}'"


@tool
def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"


# 需要审批的工具(第三层)
APPROVAL_REQUIRED_TOOLS = {"send_email"}

# -- 审批门存储(内存中;生产环境用 Redis) ------------------
approval_store: dict[str, asyncio.Event] = {}
approval_results: dict[str, dict] = {}

# -- LLM 设置 ----------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", streaming=True)
tools = [web_search, send_email, read_file]
llm_with_tools = llm.bind_tools(tools)


def should_request_approval(tool_name: str) -> bool:
    return tool_name in APPROVAL_REQUIRED_TOOLS


# -- 流式端点 --------------------------------------------------------
async def agent_stream(
    session_id: str,
    user_message: str,
) -> AsyncGenerator[str, None]:
    """以 SSE 流式传输智能体事件。"""
    def sse(event_type: str, data: dict) -> str:
        return f"data: {json.dumps({'type': event_type, **data})}\n\n"

    yield sse("status", {"message": "Agent starting ..."})
    # 模拟多步智能体执行
    steps = [
        ("thinking", {"content": "Analyzing the request ..."}),
        ("tool_call", {
            "tool": "web_search",
            "input": {"query": user_message},
            "tier": 1,  # 自动批准
        }),
        ("tool_result", {
            "tool": "web_search",
            "output": f"Results for: {user_message}",
            "duration_ms": 342,
        }),
    ]
    for event_type, data in steps:
        await asyncio.sleep(0.5)  # 模拟处理耗时
        yield sse(event_type, data)

    # 模拟一个需要审批的第三层动作
    approval_id = f"{session_id}_email_001"
    approval_event = asyncio.Event()
    approval_store[approval_id] = approval_event
    yield sse("approval_required", {
        "approval_id": approval_id,
        "tool": "send_email",
        "tier": 3,
        "risk": "irreversible",
        "action_summary": "Send summary email to user@example.com",
        "parameters": {
            "to": "user@example.com",
            "subject": f"Research results: {user_message}",
            "body": "Here are the findings ...",
        },
    })
    # 等待人工审批(5 分钟后超时)
    try:
        await asyncio.wait_for(approval_event.wait(), timeout=300)
        result = approval_results.get(approval_id, {})
        if result.get("approved"):
            yield sse("tool_call", {
                "tool": "send_email",
                "input": result.get("parameters", {}),
                "tier": 3,
                "approved_by": "human",
            })
            await asyncio.sleep(0.3)
            yield sse("tool_result", {
                "tool": "send_email",
                "output": "Email sent successfully",
                "duration_ms": 128,
            })
        else:
            yield sse("action_rejected", {
                "tool": "send_email",
                "reason": result.get("reason", "User rejected"),
            })
    except asyncio.TimeoutError:
        yield sse("approval_timeout", {
            "approval_id": approval_id,
            "message": "Approval timed out; action skipped",
        })

    # 最终响应
    yield sse("token", {"content": "I've completed the research. "})
    yield sse("token", {"content": "Here's a summary of what I found ..."})
    yield sse("done", {"total_tokens": 847, "duration_ms": 2341})


@app.get("/chat/stream")
async def chat_stream(session_id: str, message: str):
    return StreamingResponse(
        agent_stream(session_id, message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ApprovalRequest(BaseModel):
    approval_id: str
    approved: bool
    parameters: dict | None = None
    reason: str | None = None


@app.post("/chat/approve")
async def handle_approval(req: ApprovalRequest):
    if req.approval_id not in approval_store:
        raise HTTPException(status_code=404, detail="Approval not found")
    approval_results[req.approval_id] = {
        "approved": req.approved,
        "parameters": req.parameters,
        "reason": req.reason,
    }
    approval_store[req.approval_id].set()
    return {"status": "ok"}
```

清单 26.4:带流式与审批门的 FastAPI 后端

### 26.9.2 前端:带流式与工具可视化的 React

```tsx
// frontend/AgentChat.tsx
import { useState, useEffect, useRef } from 'react';

// -- 类型 -----------------------------------------------------------------
type AgentEvent =
  | { type: 'status'; message: string }
  | { type: 'thinking'; content: string }
  | { type: 'token'; content: string }
  | { type: 'tool_call'; tool: string; input: object; tier: number }
  | { type: 'tool_result'; tool: string; output: string; duration_ms: number }
  | { type: 'approval_required'; approval_id: string; tool: string;
      tier: number; risk: string; action_summary: string; parameters: object }
  | { type: 'action_rejected'; tool: string; reason: string }
  | { type: 'done'; total_tokens: number; duration_ms: number };

// -- 工具卡片组件 -------------------------------------------------------
function ToolCard({ event }: { event: AgentEvent & { type: 'tool_call' } }) {
  const [expanded, setExpanded] = useState(false);
  const tierColors = { 1: '#22c55e', 2: '#f59e0b', 3: '#ef4444' };
  const color = tierColors[event.tier as keyof typeof tierColors] || '#6b7280';
  return (
    <div style={{ border: `1px solid ${color}`, borderRadius: 8, padding: 8,
                  margin: '4px 0', fontSize: 13 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ color, fontWeight: 600 }}>[gear] {event.tool}</span>
        <span style={{ color: '#6b7280', fontSize: 11 }}>
          Tier {event.tier}. {event.tier === 1 ? 'Auto' : 'Approved'}
        </span>
        <button onClick={() => setExpanded(!expanded)}
                style={{ marginLeft: 'auto', fontSize: 11 }}>
          {expanded ? 'Hide' : 'Details'}
        </button>
      </div>
      {expanded && (
        <pre style={{ marginTop: 8, fontSize: 11, background: '#f3f4f6',
                      padding: 8, borderRadius: 4, overflow: 'auto' }}>
          {JSON.stringify(event.input, null, 2)}
        </pre>
      )}
    </div>
  );
}

// -- 审批门组件 ---------------------------------------------------
function ApprovalGate({
  event,
  onDecision,
}: {
  event: AgentEvent & { type: 'approval_required' };
  onDecision: (approved: boolean, params?: object) => void;
}) {
  const riskColors = { reversible: '#22c55e', 'hard-to-undo': '#f59e0b',
                       irreversible: '#ef4444' };
  const riskColor = riskColors[event.risk as keyof typeof riskColors] || '#6b7280';
  return (
    <div style={{ border: `2px solid ${riskColor}`, borderRadius: 8,
                  padding: 16, margin: '8px 0', background: '#fef9f0' }}>
      <div style={{ fontWeight: 700, color: riskColor, marginBottom: 8 }}>
        [!] Approval Required: {event.tool}
      </div>
      <div style={{ marginBottom: 8 }}>{event.action_summary}</div>
      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 12 }}>
        Risk level: <span style={{ color: riskColor }}>{event.risk}</span>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={() => onDecision(true, event.parameters)}
                style={{ background: '#22c55e', color: 'white', border: 'none',
                         borderRadius: 6, padding: '8px 16px', cursor: 'pointer' }}>
          [OK] Approve
        </button>
        <button onClick={() => onDecision(false)}
                style={{ background: '#ef4444', color: 'white', border: 'none',
                         borderRadius: 6, padding: '8px 16px', cursor: 'pointer' }}>
          [X] Reject
        </button>
      </div>
    </div>
  );
}

// -- 主聊天组件 -------------------------------------------------------
export function AgentChat() {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [response, setResponse] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [input, setInput] = useState('');
  const sessionId = useRef(`session_${Date.now()}`);

  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return;
    setEvents([]);
    setResponse('');
    setIsStreaming(true);
    const url = `/chat/stream?session_id=${sessionId.current}`
      + `&message=${encodeURIComponent(input)}`;
    const es = new EventSource(url);
    es.onmessage = (e) => {
      const event: AgentEvent = JSON.parse(e.data);
      if (event.type === 'token') {
        setResponse(prev => prev + event.content);
      } else if (event.type === 'done') {
        setIsStreaming(false);
        es.close();
      } else {
        setEvents(prev => [...prev, event]);
      }
    };
    es.onerror = () => { setIsStreaming(false); es.close(); };
    setInput('');
  };

  const handleApproval = async (
    approvalId: string,
    approved: boolean,
    parameters?: object,
  ) => {
    await fetch('/chat/approve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approval_id: approvalId, approved, parameters }),
    });
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 16 }}>
      <div style={{ minHeight: 400, border: '1px solid #e5e7eb',
                    borderRadius: 8, padding: 16, marginBottom: 16 }}>
        {events.map((event, i) => {
          if (event.type === 'tool_call')
            return <ToolCard key={i} event={event} />;
          if (event.type === 'approval_required')
            return (
              <ApprovalGate
                key={i} event={event}
                onDecision={(approved, params) =>
                  handleApproval(event.approval_id, approved, params)} />
            );
          if (event.type === 'status' || event.type === 'thinking')
            return (
              <div key={i} style={{ color: '#6b7280', fontSize: 12,
                                    fontStyle: 'italic', margin: '4px 0' }}>
                {event.type === 'thinking' ? event.content : event.message}
              </div>
            );
          return null;
        })}
        {response && (
          <div style={{ marginTop: 8, lineHeight: 1.6 }}>
            {response}
            {isStreaming && <span className="cursor-blink">|</span>}
          </div>
        )}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Ask the agent ..."
          style={{ flex: 1, padding: '8px 12px', borderRadius: 6,
                   border: '1px solid #d1d5db', fontSize: 14 }}
        />
        <button onClick={sendMessage} disabled={isStreaming}
                style={{ padding: '8px 16px', background: '#3b82f6',
                         color: 'white', border: 'none', borderRadius: 6,
                         cursor: isStreaming ? 'not-allowed' : 'pointer' }}>
          {isStreaming ? 'Running ...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
```

清单 26.5:带流式工具可视化与审批门的 React 前端

**此实现所演示的内容**

上述代码阐明了若干关键智能体 UI 模式的协同工作:

- **SSE 流式**:后端通过单一 HTTP 连接,流式传输不同类型的事件(状态、思考、工具调用、词元)。
- **类型化事件协议**:事件类型的可辨识联合(discriminated union),使前端能恰当地渲染每一类事件。
- **工具可视化**:`ToolCard` 渲染工具调用,带层级指示器和可展开的输入详情。
- **审批门**:`ApprovalGate` 阻塞流,并在智能体继续不可逆动作之前等待人工输入。
- **异步审批**:后端使用 `asyncio.Event` 在等待前端的审批 POST 请求时暂停流,干净地将审批 UI 与流式逻辑解耦。

## 26.10 小结

智能体 UI 框架代表人机交互的一个新前沿,要求从第一性原理重新思考界面设计。本节的关键洞见是:

1. **范式选择至关重要**:合适的 UI 范式(聊天、画布、工作流、仪表盘、协作、自主)取决于任务结构、所需的人工参与程度和输出类型。大多数生产系统组合了多种范式。
2. **透明性不可妥协**:用户无法信任他们看不见的东西。思维展示、工具可视化和上下文面板不是可选功能——它们是值得信赖的智能体系统的根基。
3. **流式是基线**:用户期望看到智能体实时工作。词元流式、工具调用流式和多智能体流式是基本能力(table-stakes)。
4. **审批门必须分层**:扁平的审批策略(全部批准或全部不批准)在实践中会失败。自动批准安全动作、为危险动作设置门控的分层策略,能在维持监督的同时不制造瓶颈。
5. **生成式 UI 是前沿**:LLM 不仅能生成文本,还能生成 UI 组件——图表、表单、地图、小部件——这使界面能够适应内容,而非把内容硬塞进固定模板。
6. **信任通过一致性与可恢复性赢得**:撤销能力、审计轨迹和经过校准的置信度指示器,对于建立用户信任而言,与原始能力同等重要。

**设计原则:作为透明协作者的智能体**

智能体 UI 设计的北极星,是那个透明的协作者:其行动始终可见、其推理始终可及、其错误始终可恢复、其能力与局限始终清晰的智能体。每一个 UI 决策都应以此标准来衡量。

本节所描述的框架与模式——Vercel AI SDK、Chainlit、Gradio、Streamlit、LangGraph Studio——提供了构建模块。实践者面临的挑战在于,以对其用户具体需求与其领域具体风险的深刻理解为指引,将它们深思熟虑地组合起来。
