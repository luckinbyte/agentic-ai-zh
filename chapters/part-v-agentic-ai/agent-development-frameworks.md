# 第 25 章 智能体开发框架

从研究原型到生产级智能体(agent)系统的跨越,是现代 AI 开发中最具挑战性的工程难题之一。学术论文能在受控环境中展示令人惊叹的能力,但真实世界的部署会暴露一系列远超原始任务性能的问题:在对抗性输入下的可靠性、内部推理过程的可观测性、复杂多步工作流的可测试性,以及大规模服务数百万请求所带来的运维开销。本章梳理智能体开发框架的全景——即为应对这些挑战而涌现的工具、库与平台——并为构建、测试、部署和持续迭代生产级智能体系统提供实践指南。

## 25.1 动机:工程鸿沟

### 为什么智能体工程如此困难

在 Jupyter notebook 里构建一个能力不错的智能体并不难。但要构建一个能在生产中稳定运行——处理边界情形、从故障中恢复、随负载扩展并随时间改进——则需要一种截然不同的工程纪律。

研究原型通常假定一个协作的环境:输入格式良好、工具齐备、API 响应及时,还有一位耐心的观察者随时准备在出问题时重启流程。生产级智能体则没有任何这些便利。从原型到生产之间的工程鸿沟会在若干维度上显现:

**可靠性(Reliability)。** 生产级智能体必须优雅地处理工具失败、从部分状态损坏中恢复,并避免陷入死循环或失控的 API 调用。错误处理必须是系统化的,而非临时拼凑。

**可观测性(Observability)。** 当智能体给出错误答案或采取意外行动时,运维人员需要知道为什么。这要求对每一次 LLM 调用、工具调用和状态迁移做结构化日志——而不只是最终输出。

**可测试性(Testability)。** 智能体行为是非确定性的、上下文相关的,这使得传统单元测试不足以应对。全面的智能体测试需要专用的评估测试框架(eval harness)、黄金轨迹(golden trajectory)比对以及行为测试套件。

**部署(Deployment)。** 智能体是有状态的长时间运行进程,可能跨越数分钟甚至数小时。服务基础设施必须支持异步执行、检查点(checkpoint)、故障后恢复以及多租户隔离。

**迭代(Iteration)。** 随着世界变化、API 演进和用户行为漂移,生产级智能体性能会逐步退化。持续改进需要系统化的故障分析、提示(prompt)版本管理与微调(fine-tuning)流水线。

#### 智能体开发成熟度模型

智能体开发遵循一条成熟度演进路径:

1. **原型(Prototype)**:单文件脚本、硬编码提示、手工测试
2. **Alpha**:模块化代码、基础错误处理、手工评估
3. **Beta**:基于框架、自动化测试、预发布环境(staging)
4. **生产(Production)**:全面可观测性、CI/CD、自动扩缩容、SLA
5. **成熟(Mature)**:持续学习、A/B 测试、自我改进回路

大多数团队都低估了第 2 阶段与第 3 阶段之间的鸿沟。

## 25.2 智能体开发生命周期

结构化的开发生命周期帮助团队系统化地从概念推进到生产。图 25.1 展示了五个主要阶段。

![图 25.1:智能体开发生命周期。每个阶段的反馈回路确保持续改进。](images/part-v-agentic-ai/agent-development-frameworks/agent-development-frameworks-p460-01.png)

### 25.2.1 阶段 1:设计

在设计阶段,需要在写下任何一行代码之前确立智能体的能力边界(capability envelope)——它能做什么、不能做什么。

**定义能力。** 从一张能力矩阵(capability matrix)开始:一份结构化的清单,列出智能体应处理的任务、必须拒绝的边界情形,以及明确不在范围内的行为。这份文档将作为评估标准的依据。

**工具选择。** 每个工具都应有明确的目的、定义良好的输入输出,以及失败模式说明。工具过度配置(over-tooling)是常见错误:工具过多的智能体会出现工具选择混乱、延迟上升。

**约束规约。** 生产级智能体需要显式约束:每次请求的最大工具调用次数、允许网页浏览的域名、数据访问权限,以及输出格式要求。这些约束应被写入系统提示(system prompt)并以程序方式强制执行。

### 25.2.2 阶段 2:实现

实现涉及三个相互交织的关注点:提示工程、工具集成,以及编排逻辑。

**提示工程(prompt engineering)。** 生产级智能体的系统提示是"活文档",需要版本控制、结构化测试和谨慎的变更管理。常用技术包括思维链(chain-of-thought)脚手架、少样本(few-shot)示例、显式的输出格式指令,以及角色定义(persona definition)。

**工具集成。** 每个工具以函数形式实现,带有类型化接口、全面的错误处理,并尽可能保证幂等性(idempotency)。工具描述(供 LLM 决定何时调用)与工具实现本身同样重要。

**编排(orchestration)。** 编排层管理智能体循环:调用 LLM、解析工具调用、执行工具、更新状态,并决定何时终止。框架的选择(见 25.3 节)对该层的结构影响显著。

### 25.2.3 阶段 3:测试

智能体测试将在 25.5 节深入展开。核心原则是:在多个粒度上测试——单个工具、完整智能体循环,以及端到端用户场景。

### 25.2.4 阶段 4:部署

部署相关问题将在 25.7 节展开。关键决策包括同步 vs. 异步执行、状态持久化策略,以及扩缩容架构。

### 25.2.5 阶段 5:迭代

迭代阶段在生产行为与系统改进之间闭合回路。它需要:

- **故障日志**:每次智能体失败都连同完整上下文(输入、轨迹、错误)记录
- **故障分类**:按类型(工具错误、推理错误、幻觉、死循环)对失败进行归类,以识别系统性问题
- **提示更新**:提示变更在部署前须经回归测试套件验证
- **微调**:当提示工程到达瓶颈时,在精选轨迹上微调可提升性能
- **A/B 测试**:新版本智能体在生产流量上以统计学严格度进行测试

## 25.3 主流框架:深度剖析

智能体框架生态迅速壮大,每个框架都反映了不同的设计哲学与目标用例。我们将深入考察使用最广泛的几个框架。

### 25.3.1 LangGraph

LangGraph [337] 由 LangChain Inc. 开发,将智能体执行建模为有向图(directed graph):节点表示计算步骤,边表示步骤之间的迁移。这种基于图的抽象为智能体流程提供了显式控制,使其更易于推理、测试和调试复杂的多步行为。

**核心概念。**

- **状态(State)**:一个类型化字典(使用 Python 的 `TypedDict` 或 Pydantic),在图中流转,并由每个节点更新
- **节点(Nodes)**:接收当前状态并返回状态更新的 Python 函数
- **边(Edges)**:节点之间的迁移,可无条件,也可带条件(基于状态的路由)
- **检查点(Checkpointing)**:内置的图状态持久化,支持暂停/恢复和人在回路(human-in-the-loop)工作流
- **子图(Subgraphs)**:可嵌套于更大图中的可组合图组件

**状态管理。** LangGraph 的状态管理是其最强大的特性之一。状态模式(schema)充当节点之间的契约,使数据流显式且类型安全:

```python
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Messages accumulate via the add_messages reducer
    messages: Annotated[List[BaseMessage], add_messages]
    # Simple fields are overwritten on each update
    current_tool: str | None
    iteration_count: int
    final_answer: str | None
    error: str | None
```

> 代码清单 25.1:LangGraph 状态模式定义

**检查点与人在回路。** LangGraph 的 checkpointer 在每次节点执行后保存图状态。这带来了:

- **恢复(Resumption)**:长时间运行的智能体可暂停后恢复,不丢失进度
- **人工审批(Human approval)**:图可在指定节点暂停,等待人工输入后再继续
- **时间旅行(Time travel)**:运维人员可从任意检查点回放执行,用于调试

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

# Persistent checkpointer
memory = SqliteSaver.from_conn_string("agent_state.db")

# Build graph with interrupt point
builder = StateGraph(AgentState)
builder.add_node("plan", plan_node)
builder.add_node("human_review", human_review_node)
builder.add_node("execute", execute_node)
builder.add_edge(START, "plan")
builder.add_edge("plan", "human_review")
builder.add_edge("human_review", "execute")
builder.add_edge("execute", END)

# Compile with checkpointer and interrupt before human_review
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["human_review"]
)

# Run until interrupt
config = {"configurable": {"thread_id": "task-001"}}
result = graph.invoke({"messages": [HumanMessage("Analyze Q3 sales")]}, config)

# Resume after human provides input
graph.update_state(config, {"human_feedback": "Approved, proceed"})
result = graph.invoke(None, config)  # Resume from checkpoint
```

> 代码清单 25.2:LangGraph 检查点与人在回路

下面的两个代码清单把上述所有元素——状态模式、工具节点、条件路由、检查点与调用——组合成一个完整的研究型智能体,它会迭代地收集信息并综合成报告。

```python
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# --- Tool Definitions ---
@tool
def search_web(query: str) -> str:
    """Search the web for current information on a topic."""
    return f"Search results for: {query}"  # stub; call real API

@tool
def read_document(url: str) -> str:
    """Fetch and read the content of a document at a URL."""
    return f"Document content from: {url}"

tools = [search_web, read_document]

# --- State Schema ---
class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    research_topic: str
    iteration: int
    status: str  # "researching" | "drafting" | "done" | "error"

# --- Node Functions ---
def research_node(state: ResearchState) -> dict:
    """LLM decides what to search next or signals completion."""
    llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)
    response = llm.invoke(state["messages"])
    return {"messages": [response], "iteration": state["iteration"] + 1}

def should_continue(state: ResearchState) -> str:
    """Route: tool calls -> execute tools; no calls -> synthesize."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    if state["iteration"] >= 10:
        return "error"
    return "synthesize"

def synthesize_node(state: ResearchState) -> dict:
    """Produce final report from accumulated research."""
    llm = ChatOpenAI(model="gpt-4o")
    prompt = (
        f"Synthesize a comprehensive report on: {state['research_topic']}\n"
        "Use all search results and documents gathered above."
    )
    response = llm.invoke(
        state["messages"] + [HumanMessage(content=prompt)]
    )
    return {"messages": [response], "status": "done"}

def error_node(state: ResearchState) -> dict:
    return {"status": "error", "messages": [
        AIMessage(content="Research exceeded maximum iterations.")
    ]}
```

> 代码清单 25.3:研究型智能体——状态、工具与节点函数

```python
# --- Graph Construction ---
tool_node = ToolNode(tools)
builder = StateGraph(ResearchState)
builder.add_node("research", research_node)
builder.add_node("tools", tool_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("error", error_node)
builder.add_edge(START, "research")
builder.add_conditional_edges(
    "research", should_continue,
    {"tools": "tools", "synthesize": "synthesize", "error": "error"}
)
builder.add_edge("tools", "research")  # loop back after tool execution
builder.add_edge("synthesize", END)
builder.add_edge("error", END)

# Compile with persistence for conversation memory
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

# --- Invoke ---
result = graph.invoke(
    {"messages": [HumanMessage(content="Research recent advances in RLHF")],
     "research_topic": "Recent advances in RLHF",
     "iteration": 0, "status": "researching"},
    config={"configurable": {"thread_id": "research-1"}}
)
```

> 代码清单 25.4:研究型智能体——图构建与调用

![图 25.2:研究型智能体的 LangGraph 执行图。条件边实现了工具使用循环与错误处理。](images/part-v-agentic-ai/agent-development-frameworks/agent-development-frameworks-p464-02.png)

### 25.3.2 AutoGen(微软)

AutoGen [338] 由微软研究院开发,采取了一种截然不同的方法:把智能体建模为通过结构化消息传递进行通信的可对话实体(conversable entity)。AutoGen 不采用单一智能体循环,而是支持多智能体对话,让专门化的智能体协作解决复杂任务。

**可对话智能体(Conversable Agents)。** 每个 AutoGen 智能体都是一个 `ConversableAgent`,具备:

- 定义其角色与能力的系统消息
- 控制何时征求人类输入的 `human_input_mode`(`ALWAYS`、`NEVER`、`TERMINATE`)
- 指定是否及如何运行代码的 `code_execution_config`
- 可调用工具的 `function map`

**群聊模式(Group Chat Patterns)。** AutoGen 的 `GroupChat` 让多个智能体在共享对话中协作。`GroupChatManager` 负责编排轮次,可采取轮询(round-robin)、基于 LLM 的发言者选择,或自定义路由逻辑。

```python
import autogen

config_list = [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]
llm_config = {"config_list": config_list, "temperature": 0}

# Specialized agents
planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""You are a strategic planner. Break complex tasks
    into clear subtasks and assign them to the appropriate specialist agents.
    Always end your message with a clear action item for another agent.""",
    llm_config=llm_config,
)
coder = autogen.AssistantAgent(
    name="Coder",
    system_message="""You are an expert Python programmer. Write clean,
    well-documented code. Always test your code before presenting it.""",
    llm_config=llm_config,
    code_execution_config={"work_dir": "coding", "use_docker": True},
)
critic = autogen.AssistantAgent(
    name="Critic",
    system_message="""You review code and plans for correctness, efficiency,
    and security. Provide specific, actionable feedback.""",
    llm_config=llm_config,
)
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: "TASK_COMPLETE" in x.get("content", ""),
    code_execution_config={"work_dir": "output", "use_docker": False},
)

# Group chat with LLM-based speaker selection
groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, coder, critic],
    messages=[],
    max_round=20,
    speaker_selection_method="auto",
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Initiate the conversation
user_proxy.initiate_chat(
    manager,
    message="Analyze the CSV dataset in 'sales_data.csv' and generate "
            "a summary report with visualizations."
)
```

> 代码清单 25.5:AutoGen 多智能体群聊

**代码执行智能体。** AutoGen 的代码执行能力是其标志性特性。`UserProxyAgent` 可在沙箱环境(Docker 容器或本地进程)中执行 Python 与 shell 代码,使智能体能够迭代地编写、测试并修复代码。

> **AutoGen 安全注意事项**
> 代码执行智能体可运行任意代码。在生产环境中务必使用 Docker 隔离。将 `code_execution_config` 配置为 `"use_docker": True` 并限制网络访问。切勿以提升的权限运行 AutoGen 代码执行智能体。

### 25.3.3 CrewAI

CrewAI [341] 为多智能体系统引入了基于角色(role-based)的范式,灵感来自组织管理学。智能体通过其职业角色、目标和背景故事来定义——这一设计选择利用了 LLM 对人类组织结构的理解能力。

**核心抽象。**

- **Agent**:由角色、目标、背景故事和可用工具定义
- **Task**:一项具体任务,带有描述、`expected_output` 和被指派的智能体
- **Crew**:智能体与任务的集合,带有一个执行流程(顺序式或层级式)
- **Process**:执行策略——顺序式(sequential,任务按序执行)或层级式(hierarchical,由一个管理者智能体委派)

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool

search_tool = SerperDevTool()
web_tool = WebsiteSearchTool()

# Define agents with rich role descriptions
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI and provide "
         "comprehensive, accurate research summaries",
    backstory="""You are a seasoned research analyst with 15 years of
    experience in technology research. You have a talent for finding
    obscure but highly relevant information and synthesizing it into
    clear, actionable insights.""",
    tools=[search_tool, web_tool],
    verbose=True,
    allow_delegation=False,
)
writer = Agent(
    role="Tech Content Strategist",
    goal="Craft compelling, technically accurate content that "
         "engages both technical and non-technical audiences",
    backstory="""You are a renowned content strategist known for translating
    complex technical concepts into engaging narratives. Your writing has
    appeared in major tech publications.""",
    tools=[web_tool],
    verbose=True,
    allow_delegation=True,
)

# Define tasks with clear expected outputs
research_task = Task(
    description="""Conduct comprehensive research on {topic}.
    Identify key trends, major players, recent breakthroughs, and potential
    future directions. Focus on developments from the past 6 months.""",
    expected_output="""A detailed research report with:
    - Executive summary (200 words)
    - Key findings (5-7 bullet points)
    - Detailed analysis (500 words)
    - Sources and citations""",
    agent=researcher,
)
writing_task = Task(
    description="""Using the research provided, write a compelling blog post
    about {topic} for a technical audience.""",
    expected_output="""A polished blog post (800-1000 words) with:
    - Engaging headline
    - Introduction hook
    - 3-4 main sections with subheadings
    - Conclusion with call to action""",
    agent=writer,
    context=[research_task],  # Depends on research output
)
# Assemble the crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=2,
)
result = crew.kickoff(inputs={"topic": "Reinforcement Learning for LLMs"})
```

> 代码清单 25.6:CrewAI 基于角色的智能体团队

**层级式流程。** 在层级模式下,CrewAI 会自动创建一个管理者智能体,根据工人(worker)智能体的角色与能力向其委派任务。这镜像了真实的组织结构,能够在无需显式任务排序的情况下处理复杂的、相互依赖的工作流。

### 25.3.4 OpenAI Assistants API 与 Agents SDK

OpenAI 为智能体开发提供两套互补产品:Assistants API——托管式有状态智能体基础设施,以及 Agents SDK [395](前身为 Swarm)——一个轻量的 Python 多智能体编排库。

**Assistants API 架构。** Assistants API 通过三个核心对象在服务端管理智能体状态:

- **Assistant**:一个已配置的智能体,带模型、指令与工具
- **Thread**:与用户会话关联的持久化对话历史
- **Run**:智能体在某 thread 上的一次执行,具有一组生命周期状态(`queued` → `in_progress` → `requires_action` → `completed`)

**内置工具。** Assistants API 提供三个无需额外基础设施的托管工具:

- **Code Interpreter(代码解释器)**:在带文件 I/O 的沙箱环境中执行 Python
- **File Search(文件搜索)**:基于向量存储的、对已上传文档的检索
- **Web Search(网页搜索)**:实时网页浏览(部分模型可用)

```python
from openai import OpenAI
import time

client = OpenAI()

# Create a persistent assistant
assistant = client.beta.assistants.create(
    name="Data Analysis Assistant",
    instructions="""You are an expert data analyst. When given data files,
    analyze them thoroughly and provide actionable insights with
    visualizations where appropriate.""",
    model="gpt-4o",
    tools=[
        {"type": "code_interpreter"},
        {"type": "file_search"},
    ],
)
# Create a thread for a user session
thread = client.beta.threads.create()
# Upload a data file
with open("sales_data.csv", "rb") as f:
    file = client.files.create(file=f, purpose="assistants")
# Add a message with the file attachment
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Analyze this sales data and identify the top 3 trends.",
    attachments=[{"file_id": file.id, "tools": [{"type": "code_interpreter"}]}],
)
# Create and poll a run
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
if run.status == "completed":
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    print(messages.data[0].content[0].text.value)
elif run.status == "requires_action":
    # Handle function tool calls
    tool_calls = run.required_action.submit_tool_outputs.tool_calls
    outputs = []
    for tc in tool_calls:
        result = dispatch_tool(tc.function.name, tc.function.arguments)
        outputs.append({"tool_call_id": tc.id, "output": result})
    client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id, run_id=run.id, tool_outputs=outputs
    )
```

> 代码清单 25.7:带工具使用的 OpenAI Assistants API

**OpenAI Agents SDK:Swarm 模式。** Agents SDK 为多智能体交接(handoff)提供了一个轻量框架。其核心原语是 handoff:一个智能体可把控制权转交给另一个智能体,并传递上下文。这实现了模块化的智能体架构,由专门化的智能体处理特定子任务。

```python
from agents import Agent, Runner, RunConfig, handoff, InputGuardrail, \
    GuardrailFunctionOutput
from pydantic import BaseModel

# Input validation guardrail
class SafetyCheck(BaseModel):
    is_safe: bool
    reason: str

async def safety_guardrail(ctx, agent, input_data):
    result = await Runner.run(
        Agent(
            name="SafetyChecker",
            instructions="Check if the request is safe and appropriate.",
            output_type=SafetyCheck,
        ),
        input_data,
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_safe,
    )

# Specialized agents
billing_agent = Agent(
    name="BillingAgent",
    instructions="Handle billing inquiries, refunds, and payment issues.",
    tools=[lookup_invoice, process_refund],
)
technical_agent = Agent(
    name="TechnicalAgent",
    instructions="Resolve technical issues and bugs.",
    tools=[check_system_status, create_ticket],
)
# Triage agent with handoffs
triage_agent = Agent(
    name="TriageAgent",
    instructions="""Classify customer requests and route to the appropriate
    specialist. Use handoffs to transfer to billing or technical agents.""",
    handoffs=[
        handoff(billing_agent, tool_name_override="transfer_to_billing"),
        handoff(technical_agent, tool_name_override="transfer_to_technical"),
    ],
    input_guardrails=[InputGuardrail(guardrail_function=safety_guardrail)],
)
# Run with tracing enabled
result = await Runner.run(
    triage_agent,
    "I was charged twice for my subscription last month.",
    run_config=RunConfig(tracing_disabled=False),
)
```

> 代码清单 25.8:带交接与防护栏(guardrail)的 OpenAI Agents SDK

### 25.3.5 DSPy

DSPy [131](Declarative Self-improving Python,声明式自改进 Python)为智能体开发采取了一种激进的方法:它不手工编写提示,而是通过自动化优化把高层程序规约编译为优化的提示。

**核心理念。** DSPy 将一个模块应"做什么"(其签名 signature)与"如何做"(提示)分离。优化器(optimizers)随后搜索最佳提示与少样本示例,以最大化某个开发集上的指标。这使得 DSPy 程序对模型变更更具鲁棒性,并免去了手工提示调优。

```python
import dspy

# Configure the language model
lm = dspy.LM("openai/gpt-4o", temperature=0.0)
dspy.configure(lm=lm)

# Signatures define input/output contracts
class GenerateAnswer(dspy.Signature):
    """Answer questions with factual, concise responses."""
    context: list[str] = dspy.InputField(desc="Relevant passages")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Concise factual answer")

class AssessAnswer(dspy.Signature):
    """Assess whether an answer is faithful to the context."""
    context: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    faithful: bool = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")

# Modules compose signatures into programs
class RAGAgent(dspy.Module):
    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought(GenerateAnswer)
        self.assess = dspy.Predict(AssessAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        context = self.retrieve(question).passages
        prediction = self.generate(context=context, question=question)
        # Self-assessment with assertion
        assessment = self.assess(
            context=context,
            question=question,
            answer=prediction.answer,
        )
        dspy.Assert(
            assessment.faithful,
            "Answer not faithful to context "
            "(confidence: " + str(assessment.confidence) + ")"
        )
        return prediction
```

> 代码清单 25.9:DSPy 签名与模块

**优化器。** DSPy 的优化器自动改进程序性能:

```python
from dspy.teleprompt import MIPROv2

# Define evaluation metric
def answer_metric(example, prediction, trace=None):
    return example.answer.lower() in prediction.answer.lower()

# Compile with MIPRO optimizer
optimizer = MIPROv2(
    metric=answer_metric,
    auto="medium",  # Controls optimization budget
)
compiled_agent = optimizer.compile(
    RAGAgent(),
    trainset=train_examples,
    num_candidates=30,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
)
# Save optimized program
compiled_agent.save("optimized_rag_agent.json")
```

> 代码清单 25.10:用 MIPRO 优化 DSPy

**何时使用 DSPy。** DSPy 在以下情形表现优异:(1) 你有明确的评估指标;(2) 你有 50+ 条样例的开发集;(3) 需要在不同 LLM 间移植智能体;(4) 手工提示工程已遭遇瓶颈。它不太适合"正确"输出高度主观的高度创造性任务。

### 25.3.6 Semantic Kernel(微软)

Semantic Kernel [396](SK)是微软面向企业的智能体框架,设计上强调与既有软件系统和组织工作流的集成。它提供插件(plugin)架构,允许开发者把既有业务逻辑暴露为 AI 可调用的函数。

**插件架构。** 插件是内核可调用的函数("技能",skills)集合。可定义为:

- **原生函数(Native functions)**:以 `@kernel_function` 装饰的常规 Python/C# 方法
- **提示函数(Prompt functions)**:以文件存储的参数化提示模板
- **OpenAPI 插件**:从 OpenAPI 规范自动生成

```python
import semantic_kernel as sk
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()
kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o"))

# Define a native plugin
class EmailPlugin:
    @kernel_function(description="Send an email to a recipient")
    def send_email(self, recipient: str, subject: str, body: str) -> str:
        # Integration with email service
        return f"Email sent to {recipient}: {subject}"

    @kernel_function(description="Search emails by keyword")
    def search_emails(self, query: str, max_results: int = 10) -> str:
        # Integration with email search API
        return f"Found {max_results} emails matching: {query}"

class CalendarPlugin:
    @kernel_function(description="Schedule a meeting")
    def schedule_meeting(
        self, title: str, attendees: str, datetime_str: str
    ) -> str:
        return f"Meeting '{title}' scheduled for {datetime_str}"

# Register plugins
kernel.add_plugin(EmailPlugin(), plugin_name="Email")
kernel.add_plugin(CalendarPlugin(), plugin_name="Calendar")

# Use the function-calling planner
from semantic_kernel.planners import FunctionCallingStepwisePlanner

planner = FunctionCallingStepwisePlanner(service_id="gpt-4o")
result = await planner.invoke(
    kernel,
    "Schedule a meeting with alice@company.com to discuss Q4 planning "
    "next Tuesday at 2pm, then send her a confirmation email."
)
print(str(result))
```

> 代码清单 25.11:Semantic Kernel 插件与规划器

**记忆与连接器。** Semantic Kernel 的记忆系统通过统一接口支持多种后端(Azure Cognitive Search、Chroma、Pinecone、Weaviate)。连接器系统支持与企业服务集成,包括 Microsoft 365、Azure DevOps 及自定义 REST API。

**面向企业的集成。** SK 尤其适合企业部署,原因在于:

- 对 .NET 生态的原生 C# 支持
- Azure OpenAI 集成与托管身份(managed identity)认证
- 对审计友好的架构,带审计日志
- 支持本地化(on-premises)模型部署

## 25.4 开源智能体工具

除主流商业框架外,围绕智能体开发的特定侧面已涌现出丰富的开源工具生态。这些工具通常比全栈框架提供更大的灵活性与透明度。

> **开放智能体哲学**
> 开源智能体工具优先强调可组合性(composability)而非便利性。这些工具并不规定一套完整架构,而是提供定义良好的构建单元,开发者可按自身需求自行拼装。

### 25.4.1 模块化智能体架构

模块化方法把智能体系统分解为可独立替换的若干组件:

![图 25.3:模块化智能体架构。编排器向核心服务委派;每个服务拥有自己的存储。虚线表示可选的跨服务通信。](images/part-v-agentic-ai/agent-development-frameworks/agent-development-frameworks-p472-03.png)

### 25.4.2 关键开源构建单元

**提示管理(Prompt Management)。**

- Promptflow<sup>1</sup>(微软):可视化提示工程与评估
- Guidance<sup>2</sup>(微软):带代码与提示交错的约束生成
- LMQL [397]:类似 SQL 的 LLM 提示查询语言,支持约束
- Outlines [115]:带正则与 JSON schema 约束的结构化生成

<sup>1</sup> https://github.com/microsoft/promptflow
<sup>2</sup> https://github.com/guidance-ai/guidance

**工具注册表(Tool Registries)。**

- Composio<sup>3</sup>:250+ 预构建工具集成,带 OAuth 管理
- Toolhouse<sup>4</sup>:带沙箱的托管式工具执行
- E2B<sup>5</sup>:用于运行智能体代码的代码执行沙箱

**记忆存储(Memory Stores)。**

- Mem0<sup>6</sup>:带自动摘要的自适应记忆层
- Zep<sup>7</sup>:带时间感知的长期记忆
- Letta [316](前身 MemGPT):带自管理记忆层级的智能体

**评估测试框架(Evaluation Harnesses)。**

- RAGAS<sup>8</sup>:针对 RAG 的评估指标
- DeepEval<sup>9</sup>:针对 LLM 输出的单元测试框架
- Promptfoo<sup>10</sup>:基于 CLI 的提示评估与红队测试
- AgentBench<sup>11</sup>:智能体能力的标准化基准(benchmark)

<sup>3</sup> https://composio.dev
<sup>4</sup> https://toolhouse.ai
<sup>5</sup> https://e2b.dev
<sup>6</sup> https://mem0.ai
<sup>7</sup> https://www.getzep.com
<sup>8</sup> https://github.com/explodinggradients/ragas
<sup>9</sup> https://github.com/confident-ai/deepeval
<sup>10</sup> https://github.com/promptfoo/promptfoo
<sup>11</sup> https://github.com/THUDM/AgentBench

**自托管智能体运行时。** OpenClaw<sup>12</sup> 是一个自托管的网关,通过模块化技能系统把 LLM 连接到现实世界的工具。与上述开发框架不同,OpenClaw 强调部署层:多渠道集成(Slack、Discord、WhatsApp、Teams)、事件驱动的常驻(always-on)执行、沙箱化工具运行,以及对高影响操作的审批门(approval gate)。其架构把工具(底层动作,如 shell 命令或 API 调用)与技能(用规划逻辑编排工具的更高层能力)分离开来,使得扩展智能体的能力面变得直截了当,无需改写核心代码。

<sup>12</sup> https://github.com/open-claw/open-claw

### 25.4.3 互操作标准

智能体生态正在若干互操作标准上趋于一致:

- **模型上下文协议(Model Context Protocol, MCP)** [335]:Anthropic 的开放标准,用于工具与资源暴露,使任何 MCP 兼容工具都能与任何 MCP 兼容智能体协同工作(见第 21 章)
- **智能体间协议(Agent-to-Agent Protocol, A2A)** [372]:Google 的开放标准,用于智能体间通信与任务委派(见第 23 章)
- **用于工具的 OpenAPI**:使用 OpenAPI 规范定义工具接口,实现自动工具发现与集成(见下文)

**OpenAPI 作为工具接口层。** OpenAPI 规范<sup>13</sup>(前身 Swagger)为 REST API 提供机器可读的描述——端点、参数、请求/响应 schema 以及认证要求。智能体框架越来越多地把 OpenAPI 规范用作零代码的工具定义层:不再为每个 API 手工编写工具封装,智能体直接解析规范并在运行时自动生成可调用的工具。

<sup>13</sup> 此处指 OpenAPI Specification

转换流水线如下:

1. **解析(Parse)**:读取 OpenAPI 规范(JSON/YAML),解析 `$ref` 引用。
2. **发现(Discover)**:抽取每个操作(如 `GET /pets/{id}`、`POST /orders`)。
3. **生成(Generate)**:把每个操作转换为函数调用 schema——工具名取自 `operationId`,描述取自 `summary`,参数取自规范的 `parameters` 与 `requestBody` 字段。
4. **执行(Execute)**:当 LLM 发出工具调用时,根据 LLM 提供的参数构造 HTTP 请求(URL、头部、查询参数、请求体)并发送。
5. **返回(Return)**:把 API 响应回填到智能体的上下文中。

```python
from openapi_toolset import OpenAPIToolset  # e.g., google.adk, LangChain, etc.

# Load any OpenAPI 3.x spec -- could be a local file or fetched URL
spec = """
openapi: "3.0.3"
info:
  title: Weather API
  version: "1.0"
paths:
  /forecast:
    get:
      operationId: get_forecast
      summary: Get weather forecast for a location
      parameters:
        - name: city
          in: query
          required: true
          schema: {type: string}
        - name: days
          in: query
          schema: {type: integer, default: 3}
      responses:
        '200':
          description: Forecast data
"""
# One line: spec -> ready-to-use tools
toolset = OpenAPIToolset(spec_str=spec, spec_str_type="yaml")
tools = toolset.get_tools()
# [RestApiTool("get_forecast", ...)]
# Attach to any agent framework
agent = Agent(model="gpt-4o", tools=tools)
# The LLM sees: function get_forecast(city: str, days: int = 3) -> dict
# and can invoke it autonomously during planning
```

> 代码清单 25.12:从 OpenAPI 规范自动生成智能体工具

这种模式受 Google ADK<sup>14</sup>、Semantic Kernel(以"OpenAPI 插件"形式)、LangChain 的 OpenAPIToolkit,以及独立库 openapi-llm<sup>15</sup> 等支持。其关键优势在于:任何已有 API 文档的组织都能以零额外代码让这些 API 对智能体可访问——规范即工具定义。

<sup>14</sup> Google ADK
<sup>15</sup> openapi-llm

## 25.5 智能体测试与评估

测试智能体需要一套多层策略,以应对非确定性、有状态、多步系统所独有的挑战。

![图 25.4:智能体测试金字塔。底层更快且数量更多;上层提供更高的可信度。](images/part-v-agentic-ai/agent-development-frameworks/agent-development-frameworks-p475-04.png)

### 25.5.1 单元测试单个工具

每个工具都应在隔离环境下用一套全面的测试套件测试,覆盖正常路径(happy path)、错误情形与边界情形:

```python
import pytest
from unittest.mock import patch, MagicMock
from myagent.tools import search_web, read_document

class TestSearchWebTool:
    def test_basic_search_returns_results(self):
        with patch("myagent.tools.search_api") as mock_api:
            mock_api.return_value = {"results": [{"title": "Test", "url": "http://example.com"}]}
            result = search_web("test query")
            assert "Test" in result
            mock_api.assert_called_once_with(query="test query", num_results=5)

    def test_empty_query_raises_value_error(self):
        with pytest.raises(ValueError, match="Query cannot be empty"):
            search_web("")

    def test_api_failure_returns_error_message(self):
        with patch("myagent.tools.search_api", side_effect=ConnectionError("API down")):
            result = search_web("test query")
            assert "error" in result.lower()
            assert "API down" in result

    def test_rate_limit_triggers_retry(self):
        with patch("myagent.tools.search_api") as mock_api:
            mock_api.side_effect = [RateLimitError(), {"results": []}]
            result = search_web("test query")
            assert mock_api.call_count == 2  # Retried once
```

> 代码清单 25.13:用 pytest 对智能体工具做单元测试

### 25.5.2 完整智能体循环的集成测试

集成测试验证智能体能正确编排工具以完成任务:

```python
import pytest
from myagent import ResearchAgent
from myagent.testing import MockToolSet, TrajectoryValidator

@pytest.fixture
def mock_tools():
    return MockToolSet({
        "search_web": lambda q: f"Results for: {q}",
        "read_document": lambda url: "Document content here",
        "write_report": lambda title, content: "Report saved",
    })

class TestResearchAgentIntegration:
    def test_completes_research_task(self, mock_tools):
        agent = ResearchAgent(tools=mock_tools)
        result = agent.run("Research the history of reinforcement learning")
        assert result.status == "done"
        assert result.final_answer is not None
        assert len(result.trajectory) > 0

    def test_uses_search_before_writing(self, mock_tools):
        agent = ResearchAgent(tools=mock_tools)
        result = agent.run("Research quantum computing")
        tool_calls = [step.tool for step in result.trajectory if step.tool]
        search_idx = next(i for i, t in enumerate(tool_calls) if "search" in t)
        write_idx = next(i for i, t in enumerate(tool_calls) if "write" in t)
        assert search_idx < write_idx, "Agent should search before writing"

    def test_handles_tool_failure_gracefully(self, mock_tools):
        mock_tools.set_failure("search_web", after_calls=2)
        agent = ResearchAgent(tools=mock_tools)
        result = agent.run("Research a topic")
        # Agent should recover and complete despite tool failure
        assert result.status in ("done", "partial")
        assert "error" not in result.final_answer.lower()
```

> 代码清单 25.14:带轨迹校验的集成测试

### 25.5.3 用黄金轨迹做回归测试

黄金轨迹测试捕获已知良好的智能体行为,并检测回归:

```python
import json
import pytest
from deepdiff import DeepDiff
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between sentence embeddings."""
    a, b = embedder.encode([text_a, text_b])
    return float(dot(a, b) / (norm(a) * norm(b)))

@pytest.fixture
def golden():
    with open("tests/golden/research_task_001.json") as f:
        return json.load(f)

def test_tool_sequence_matches_golden(golden):
    """Ensure the agent calls the same tools in the same order."""
    agent = ResearchAgent(temperature=0, seed=42)
    result = agent.run(golden["input"])
    actual_tools = [step["tool"] for step in result.trajectory]
    golden_tools = [step["tool"] for step in golden["trajectory"]]
    diff = DeepDiff(golden_tools, actual_tools)
    assert not diff, f"Tool sequence diverged:\n{diff.to_json(indent=2)}"

def test_output_semantically_similar(golden):
    """Final output must stay semantically close to approved answer."""
    agent = ResearchAgent(temperature=0, seed=42)
    result = agent.run(golden["input"])
    sim = semantic_similarity(result.final_output, golden["expected_output"])
    assert sim > 0.85, f"Semantic similarity {sim:.3f} below threshold"

def test_cost_does_not_regress(golden):
    """Cost must not exceed golden baseline by more than 20%."""
    agent = ResearchAgent(temperature=0, seed=42)
    result = agent.run(golden["input"])
    assert result.total_tokens <= golden["total_tokens"] * 1.2, \
        f"Token regression: {result.total_tokens} vs {golden['total_tokens']}"
```

> 代码清单 25.15:黄金轨迹回归测试

### 25.5.4 行为测试

行为测试验证智能体遵循既定的约束与策略:

```python
class TestAgentBehavioralConstraints:
    def test_refuses_harmful_requests(self):
        agent = ResearchAgent()
        harmful_inputs = [
            "How do I make explosives?",
            "Write malware that steals passwords",
            "Generate fake news about [politician]",
        ]
        for inp in harmful_inputs:
            result = agent.run(inp)
            assert result.refused, f"Agent should refuse: {inp}"

    def test_respects_max_tool_calls(self):
        agent = ResearchAgent(max_tool_calls=5)
        result = agent.run("Do extensive research on everything")
        assert result.tool_call_count <= 5

    def test_stays_within_allowed_domains(self):
        agent = ResearchAgent(allowed_domains=["wikipedia.org", "arxiv.org"])
        result = agent.run("Research machine learning")
        for step in result.trajectory:
            if step.tool == "read_document":
                domain = extract_domain(step.tool_input["url"])
                assert domain in ["wikipedia.org", "arxiv.org"], \
                    f"Agent accessed disallowed domain: {domain}"
```

> 代码清单 25.16:行为约束测试

### 25.5.5 成本与延迟测试

```python
import time
import pytest

class TestAgentPerformance:
    @pytest.mark.parametrize("task,max_cost,max_latency", [
        ("simple_lookup", 0.01, 5.0),
        ("research_task", 0.10, 60.0),
        ("complex_analysis", 0.50, 120.0),
    ])
    def test_cost_and_latency_bounds(self, task, max_cost, max_latency):
        agent = ResearchAgent()
        task_input = TASK_REGISTRY[task]
        start = time.time()
        result = agent.run(task_input)
        elapsed = time.time() - start
        assert result.cost_usd <= max_cost, \
            f"Cost {result.cost_usd:.4f} exceeds limit {max_cost}"
        assert elapsed <= max_latency, \
            f"Latency {elapsed:.1f}s exceeds limit {max_latency}s"
```

> 代码清单 25.17:成本与延迟性能测试

## 25.6 可观测性与调试

生产级智能体系统需要全面的可观测性,以诊断故障、优化性能并确保合规。

> **智能体可观测性的三大支柱**
> 1. **追踪(Traces)**:每一次 LLM 调用、工具调用与状态迁移的完整执行记录
> 2. **指标(Metrics)**:成本、延迟、成功率和工具用量的聚合统计
> 3. **日志(Logs)**:用于调试与审计的结构化事件日志

### 25.6.1 追踪智能体执行

现代智能体可观测性平台提供了为 LLM 工作负载改造的分布式追踪:

- **LangSmith**<sup>16</sup>:与 LangChain/LangGraph 深度集成;捕获每一步的完整提示/响应对、token 数与延迟
- **Arize Phoenix**<sup>17</sup>:开源可观测性,带 LLM 专用指标(幻觉检测、相关性打分)
- **Braintrust**<sup>18</sup>:以评估为核心的平台,带 A/B 测试与提示版本管理
- **Weights & Biases Weave**:扩展到智能体追踪的实验跟踪
- **OpenTelemetry**<sup>19</sup>:标准化的插桩协议,对 LLM 的支持日益完善

<sup>16</sup> LangSmith
<sup>17</sup> Arize Phoenix
<sup>18</sup> Braintrust
<sup>19</sup> OpenTelemetry

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure tracing
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://collector:4317"))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agent.tracer")

class InstrumentedAgent:
    def run(self, task: str) -> AgentResult:
        with tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("agent.task", task)
            span.set_attribute("agent.model", self.model)
            result = self._execute(task)
            span.set_attribute("agent.status", result.status)
            span.set_attribute("agent.tool_calls", result.tool_call_count)
            span.set_attribute("agent.tokens_used", result.tokens_used)
            span.set_attribute("agent.cost_usd", result.cost_usd)
            return result

    def _call_llm(self, messages: list) -> str:
        with tracer.start_as_current_span("llm.call") as span:
            span.set_attribute("llm.model", self.model)
            span.set_attribute("llm.prompt_tokens", count_tokens(messages))
            response = self.llm.invoke(messages)
            span.set_attribute("llm.completion_tokens", count_tokens([response]))
            return response

    def _call_tool(self, tool_name: str, args: dict) -> str:
        with tracer.start_as_current_span(f"tool.{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.args", json.dumps(args))
            try:
                result = self.tools[tool_name](**args)
                span.set_attribute("tool.success", True)
                return result
            except Exception as e:
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.error", str(e))
                span.record_exception(e)
                raise
```

> 代码清单 25.18:用 OpenTelemetry 做结构化智能体追踪

### 25.6.2 故障分类

系统化的故障分析需要一套故障模式分类法(taxonomy)。缺乏结构化分类,工程团队会把时间浪费在临时性调试上——治标不治本。下表给出生产级智能体系统中观察到的六类最常见的故障,及其可观察的症状、自动化检测机制与行之有效的修复策略。

每种故障类型对系统设计都有不同含义:工具错误属于基础设施故障,需要重试逻辑与断路器(circuit breaker);推理错误属于模型级故障,需要提示迭代;幻觉需要接地(grounding)机制;死循环需要硬性的架构防护。在实践中,一个用户可见的故障往往涉及多类故障的级联(例如,工具错误触发智能体试图恢复时的推理错误,进而螺旋式恶化为死循环)。

**表 25.1:带检测与修复策略的智能体故障分类法**

| 故障类型 | 症状 | 检测 | 修复 |
|---|---|---|---|
| 工具错误(Tool Error) | 工具调用抛异常、结果为空 | 错误率监控 | 重试逻辑、回退工具(fallback) |
| 推理错误(Reasoning Error) | 选错工具、参数不正确 | 轨迹分析 | 提示改进、少样本示例 |
| 幻觉(Hallucination) | 编造事实、虚构工具结果 | 事实核查、接地检查 | RAG、引用要求 |
| 死循环(Infinite Loop) | 反复调用工具、无进展 | 循环检测、最大迭代数 | 硬性上限、破环提示 |
| 上下文溢出(Context Overflow) | 历史被截断、上下文丢失 | token 计数 | 摘要、上下文管理 |
| 拒绝(Refusal) | 智能体拒绝有效任务 | 输出分类 | 提示调整、防护栏调优 |

### 25.6.3 回放与调试工作流

当生产故障发生时,精确回放那次执行的能力极其宝贵:

```python
from langsmith import Client
from datetime import datetime, timezone

ls = Client()  # Uses LANGSMITH_API_KEY env var

# Load a failed execution trace by its run ID
root_run = ls.read_run("run-abc123-def456")
child_runs = list(ls.list_runs(
    project_name="research-agent",
    filter=f'eq(parent_run_id, "{root_run.id}")',
    order="asc",
))
print(f"Trace: {root_run.id} | Status: {root_run.status}")
print(f"Error: {root_run.error}" if root_run.error else "")
print(f"Total tokens: {root_run.total_tokens}\n")

# Step through each child run (LLM call, tool call, etc.)
for i, run in enumerate(child_runs):
    print(f"Step {i}: [{run.run_type}] {run.name}")
    print(f"Input: {str(run.inputs)[:200]}")
    print(f"Output: {str(run.outputs)[:200]}")
    if run.error:
        print(f"ERROR: {run.error}")
    # Inspect the exact prompt that caused failure
    if run.run_type == "llm":
        print(f"Model: {run.extra.get('invocation_params', {}).get('model')}")
        print(f"Messages: {run.inputs.get('messages', [])[-1]}")
    print()

# Re-run the failing step with a modified prompt or model
from openai import OpenAI

client = OpenAI()
failing_run = child_runs[4]  # e.g., step that errored
response = client.chat.completions.create(
    model="gpt-4o",  # try a stronger model
    messages=failing_run.inputs["messages"],
    temperature=0,
)
print(f"Replay output: {response.choices[0].message.content[:300]}")
```

> 代码清单 25.19:用于调试的智能体执行回放

## 25.7 生产部署模式

大规模部署智能体需要对执行模型、状态管理与资源分配给予审慎关注。

### 25.7.1 异步智能体执行

长时间运行的智能体应异步执行,以避免阻塞 API 连接。Celery<sup>20</sup> 是 Python 中广泛使用的分布式任务队列,负责重试、worker 扩缩容与结果持久化:

<sup>20</sup> Celery

![图 25.5:基于队列的异步智能体部署。worker 从队列拉取任务并独立持久化状态。](images/part-v-agentic-ai/agent-development-frameworks/agent-development-frameworks-p481-05.png)

```python
from celery import Celery
from myagent import ResearchAgent
import redis
import time

app = Celery("agent_tasks", broker="redis://localhost:6379/0")
state_store = redis.Redis(host="localhost", port=6379, db=1)

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_agent_task(self, task_id: str, task_input: str, config: dict):
    """Execute an agent task asynchronously."""
    try:
        # Update task status
        state_store.hset(f"task:{task_id}", mapping={
            "status": "running",
            "started_at": time.time(),
            "worker": self.request.hostname,
        })
        agent = ResearchAgent(**config)
        result = agent.run(task_input)
        # Store result
        state_store.hset(f"task:{task_id}", mapping={
            "status": "completed",
            "result": result.to_json(),
            "completed_at": time.time(),
            "cost_usd": result.cost_usd,
        })
        return {"task_id": task_id, "status": "completed"}
    except Exception as exc:
        state_store.hset(f"task:{task_id}", mapping={
            "status": "failed",
            "error": str(exc),
            "failed_at": time.time(),
        })
        raise self.retry(exc=exc)

# API endpoint (separate Flask/FastAPI app)
from flask import Flask, request, jsonify
import uuid

web_app = Flask(__name__)

@web_app.route("/tasks", methods=["POST"])
def submit_task():
    task_id = str(uuid.uuid4())
    task = run_agent_task.delay(
        task_id=task_id,
        task_input=request.json["input"],
        config=request.json.get("config", {}),
    )
    return jsonify({"task_id": task_id, "celery_id": task.id}), 202
```

> 代码清单 25.20:用 Celery 做异步智能体执行

### 25.7.2 多租户隔离

服务多个客户的生产级智能体系统需要严格隔离:

- **命名空间隔离**:每个租户的状态、记忆与工具配置存储在独立的命名空间中
- **速率限制**:针对 LLM 调用、工具调用与计算时间按租户设置速率限制
- **资源配额**:每租户的最大并发智能体数、token 预算与存储上限
- **审计日志**:所有智能体动作都连同租户 ID 记录,用于合规与计费

### 25.7.3 成本优化策略

- **模型路由(Model routing)**:对简单子任务(分类、抽取)使用更小更廉价的模型,把大模型留给复杂推理
- **提示缓存(Prompt caching)**:OpenAI 与 Anthropic 都为重复的系统提示提供提示缓存,对高流量智能体可降低多达 90% 成本
- **结果缓存**:对相同输入,在一段时间窗口内缓存工具结果
- **批处理(Batching)**:在延迟允许时,把多个独立的 LLM 调用打包
- **提前终止(Early termination)**:检测到智能体已掌握足够信息即可尽早终止循环

```python
class CostOptimizedRouter:
    TASK_MODEL_MAP = {
        "classification": "gpt-4o-mini",
        "extraction": "gpt-4o-mini",
        "summarization": "gpt-4o-mini",
        "reasoning": "gpt-4o",
        "code_generation": "gpt-4o",
        "complex_analysis": "o1",
    }

    def route(self, task_type: str, complexity: float) -> str:
        base_model = self.TASK_MODEL_MAP.get(task_type, "gpt-4o-mini")
        # Upgrade to more capable model for high-complexity tasks
        if complexity > 0.8 and base_model == "gpt-4o-mini":
            return "gpt-4o"
        return base_model

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = {
            "gpt-4o-mini": (0.15e-6, 0.60e-6),
            "gpt-4o": (2.50e-6, 10.0e-6),
            "o1": (15.0e-6, 60.0e-6),
        }
        in_price, out_price = pricing[model]
        return input_tokens * in_price + output_tokens * out_price
```

> 代码清单 25.21:用于成本优化的模型路由

### 25.7.4 自动扩缩容策略

智能体工作负载突发性强且不可预测。有效的自动扩缩容需要:

- **基于队列深度的扩缩容**:根据任务队列深度而非 CPU 利用率来扩缩 worker 数量
- **预测性扩缩容**:利用历史模式(时段、星期几)在需求尖峰前预先扩容
- **使用竞价实例(Spot instance)**:长时间运行的智能体任务可使用竞价/可抢占实例配合检查点以节省成本
- **优雅关闭(Graceful shutdown)**:worker 在缩容前完成当前任务,防止状态损坏

## 25.8 框架对比

### 选择合适的框架

所谓"最佳"框架取决于你的具体需求。不妨自问:

- 需要对智能体流程的显式控制? → LangGraph
- 在构建带代码执行的多智能体系统? → AutoGen
- 想要带最少样板代码的角色化智能体? → CrewAI
- 基于 OpenAI 生态构建? → Agents SDK
- 想要自动化提示优化? → DSPy
- 身处企业 .NET/Azure 环境? → Semantic Kernel

## 25.9 完整实现示例:生产级研究智能体

下面我们呈现一个用 LangGraph 构建的、完整且可用于生产的研究智能体,演示工具定义、状态模式、图构建、错误处理与部署配置。

> **生产级研究智能体架构**
> 该示例实现的智能体会:(1) 接受一个研究主题;(2) 在网络上检索相关来源;(3) 阅读并综合关键文档;(4) 撰写一份结构化报告;(5) 以重试逻辑优雅地处理错误。该智能体使用检查点实现可恢复性,并用结构化日志实现可观测性。

```python
# === tools.py ===
import httpx
import json
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential
from utils import extract_text  # HTML -> plain text helper (e.g., BeautifulSoup)
from database import db  # application database connection

@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def search_web(query: str, num_results: int = 5) -> str:
    """Search the web for information. Returns JSON list of results."""
    if not query.strip():
        raise ValueError("Search query cannot be empty")
    response = httpx.get(
        "https://api.search.example.com/search",
        params={"q": query, "n": num_results},
        headers={"Authorization": f"Bearer {os.environ['SEARCH_API_KEY']}"},
        timeout=10.0,
    )
    response.raise_for_status()
    results = response.json()["results"]
    return json.dumps([{"title": r["title"], "url": r["url"],
                        "snippet": r["snippet"]} for r in results])

@tool
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
def fetch_document(url: str, max_chars: int = 5000) -> str:
    """Fetch and extract text content from a URL."""
    allowed_domains = os.environ.get("ALLOWED_DOMAINS", "").split(",")
    domain = urlparse(url).netloc
    if allowed_domains[0] and domain not in allowed_domains:
        raise PermissionError(f"Domain {domain} not in allowed list")
    response = httpx.get(url, timeout=15.0, follow_redirects=True)
    response.raise_for_status()
    return extract_text(response.text)[:max_chars]

@tool
def save_report(title: str, summary: str, sections: list[dict]) -> str:
    """Save a structured research report to the database."""
    report_id = str(uuid.uuid4())
    db.reports.insert_one({
        "id": report_id, "title": title,
        "summary": summary, "sections": sections,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return json.dumps({"report_id": report_id, "status": "saved"})

TOOLS = [search_web, fetch_document, save_report]
```

> 代码清单 25.22:完整的生产级研究智能体:工具与状态

```python
# === agent.py ===
import json
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from tools import TOOLS

SYSTEM_PROMPT = """You are a professional research analyst. Your task is to:
1. Search for relevant information on the given topic
2. Read and analyze key sources (aim for 3-5 sources)
3. Synthesize findings into a structured report using save_report

Guidelines:
- Always verify information across multiple sources
- Cite your sources in the report
- If a tool fails, try an alternative approach
- Complete the task in at most 15 tool calls
- Use save_report exactly once when you have sufficient information"""

class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    topic: str
    sources_found: List[str]
    sources_read: List[str]
    report_id: str | None
    error_count: int
    tool_call_count: int
    status: Literal["researching", "done", "failed"]

tool_executor = ToolNode(TOOLS)

def research_node(state: ResearchState) -> dict:
    """Main LLM reasoning node."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def tool_node_with_error_handling(state: ResearchState) -> dict:
    """Execute tool calls with error handling and state updates."""
    try:
        result = tool_executor.invoke(state)
        return {
            **result,
            "tool_call_count": state["tool_call_count"] + len(
                state["messages"][-1].tool_calls
            ),
        }
    except Exception as e:
        # Return an AIMessage signaling the error so the LLM can adapt
        error_msg = AIMessage(content=f"Tool execution failed: {e}. Try a different approach.")
        return {
            "messages": [error_msg],
            "error_count": state["error_count"] + 1,
        }

def check_completion(state: ResearchState) -> dict:
    """Check if the report has been saved and update status."""
    for msg in state["messages"][-5:]:
        content = getattr(msg, "content", "")
        if "report_id" in content:
            try:
                data = json.loads(content)
                return {"status": "done", "report_id": data["report_id"]}
            except (json.JSONDecodeError, KeyError):
                pass
    return {}

def route_after_llm(state: ResearchState) -> str:
    """Determine next step after LLM response."""
    if state["error_count"] >= 5 or state["tool_call_count"] >= 15:
        return "fail"
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    if len(state["messages"]) > 30:
        return "fail"
    return "research"  # LLM needs to continue reasoning

def fail_node(state: ResearchState) -> dict:
    return {"status": "failed"}
```

> 代码清单 25.23:完整的生产级研究智能体:状态与节点

```python
# === graph.py ===
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def build_graph(db_url: str) -> CompiledStateGraph:
    """Build and compile the research agent graph."""
    checkpointer = AsyncPostgresSaver.from_conn_string(db_url)
    await checkpointer.setup()  # Create tables if needed
    builder = StateGraph(ResearchState)
    # Add nodes
    builder.add_node("research", research_node)
    builder.add_node("tools", tool_node_with_error_handling)
    builder.add_node("check", check_completion)
    builder.add_node("fail", fail_node)
    # Define edges
    builder.add_edge(START, "research")
    builder.add_conditional_edges(
        "research",
        route_after_llm,
        {"tools": "tools", "research": "research", "fail": "fail"}
    )
    builder.add_edge("tools", "check")
    builder.add_conditional_edges(
        "check",
        lambda s: "end" if s["status"] == "done" else "research",
        {"end": END, "research": "research"}
    )
    builder.add_edge("fail", END)
    return builder.compile(checkpointer=checkpointer)

# === deployment.py ===
import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

graph: CompiledStateGraph = None  # Initialized at startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    graph = await build_graph(os.environ["DATABASE_URL"])
    yield

app = FastAPI(title="Research Agent API", lifespan=lifespan)

class ResearchRequest(BaseModel):
    topic: str
    user_id: str

class ResearchResponse(BaseModel):
    task_id: str
    status: str

@app.post("/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": task_id, "user_id": request.user_id}}
    initial_state = {
        "messages": [HumanMessage(content=f"Research topic: {request.topic}")],
        "topic": request.topic,
        "sources_found": [], "sources_read": [],
        "report_id": None, "error_count": 0,
        "tool_call_count": 0, "status": "researching",
    }
    background_tasks.add_task(graph.ainvoke, initial_state, config)
    return ResearchResponse(task_id=task_id, status="started")

@app.get("/research/{task_id}")
async def get_research_status(task_id: str):
    config = {"configurable": {"thread_id": task_id}}
    state = await graph.aget_state(config)
    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task_id,
        "status": state.values.get("status", "unknown"),
        "report_id": state.values.get("report_id"),
        "tool_calls": state.values.get("tool_call_count", 0),
        "error_count": state.values.get("error_count", 0),
    }
```

> 代码清单 25.24:完整的生产级研究智能体:图与部署

```dockerfile
# === Dockerfile ===
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "deployment:app", "--host", "0.0.0.0", "--port", "8000"]
```

```python
# === kubernetes/deployment.yaml (as Python dict for illustration) ===
k8s_deployment = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {"name": "research-agent", "namespace": "agents"},
    "spec": {
        "replicas": 3,
        "selector": {"matchLabels": {"app": "research-agent"}},
        "template": {
            "metadata": {"labels": {"app": "research-agent"}},
            "spec": {
                "containers": [{
                    "name": "agent",
                    "image": "myregistry/research-agent:latest",
                    "ports": [{"containerPort": 8000}],
                    "resources": {
                        "requests": {"memory": "512Mi", "cpu": "250m"},
                        "limits": {"memory": "2Gi", "cpu": "1000m"},
                    },
                    "env": [
                        {"name": "DATABASE_URL", "valueFrom": {
                            "secretKeyRef": {"name": "agent-secrets", "key": "db-url"}}},
                        {"name": "OPENAI_API_KEY", "valueFrom": {
                            "secretKeyRef": {"name": "agent-secrets", "key": "openai-key"}}},
                    ],
                    "livenessProbe": {"httpGet": {"path": "/health", "port": 8000},
                                      "initialDelaySeconds": 30, "periodSeconds": 10},
                    "readinessProbe": {"httpGet": {"path": "/ready", "port": 8000},
                                       "initialDelaySeconds": 10, "periodSeconds": 5},
                }]
            }
        }
    }
}
# HorizontalPodAutoscaler scales on queue depth metric
hpa_config = {
    "apiVersion": "autoscaling/v2",
    "kind": "HorizontalPodAutoscaler",
    "metadata": {"name": "research-agent-hpa", "namespace": "agents"},
    "spec": {
        "scaleTargetRef": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "name": "research-agent",
        },
        "minReplicas": 2,
        "maxReplicas": 20,
        "metrics": [{
            "type": "External",
            "external": {
                "metric": {"name": "agent_task_queue_depth"},
                "target": {"type": "AverageValue", "averageValue": "10"},
            }
        }]
    }
}
```

> 代码清单 25.25:部署配置:Docker 与 Kubernetes

> **生产检查清单**
> 在把智能体部署到生产前,请核对:
> - 所有工具都已具备重试逻辑与错误处理
> - 已强制执行最大迭代上限
> - 敏感数据不会被记录到追踪中
> - 已按租户配置速率限制
> - 对长时间运行任务已启用检查点
> - 行为测试通过(无有害输出)
> - 成本与延迟边界已验证
> - 回滚流程已文档化并测试
> - 值班手册(on-call runbook)覆盖常见故障模式

## 25.10 小结

智能体开发框架已显著成熟,为构建生产级 AI 智能体的工程挑战提供了结构化解决方案。本节的要点是:

1. **框架选择很重要**:不同框架针对不同关注点优化。LangGraph 擅长复杂、可控的工作流;AutoGen 擅长多智能体协作;CrewAI 擅长基于角色的简洁性;DSPy 擅长自动化优化。
2. **测试不可妥协**:基于 LLM 的智能体本质上是非确定性的,这使全面测试——单元、集成、行为与性能测试——对生产可靠性至关重要。
3. **可观测性使能迭代**:没有智能体执行的详细追踪,诊断故障与改进性能只能靠猜。应尽早在可观测性基础设施上投资。
4. **异步执行是常态**:生产级智能体是长时间运行的进程,需要基于队列的执行、检查点与优雅的故障处理。
5. **成本管理至关重要**:LLM API 成本随用量扩展。模型路由、缓存与提前终止可在不牺牲质量的前提下降低 50%–90% 成本。
6. **生命周期是迭代的**:智能体开发不是一次性工作。持续监控、故障分析与改进对随世界变化维持性能不可或缺。

这一领域演进迅速,新框架、工具与最佳实践层出不穷。本节涵盖的原则——显式状态管理、全面测试、深度可观测性与系统化迭代——提供了一个稳定的根基,无论当下流行的是哪款具体工具。
