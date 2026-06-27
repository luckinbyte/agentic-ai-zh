# 第 23 章 智能体间通信(Agent-to-Agent Communication, A2A)

随着大语言模型(Large Language Model, LLM)从孤立的单体助手演进为由专业化智能体(agent)构成的协作网络,"智能体之间如何相互对话"变得与"智能体如何在内部进行推理"同样重要。本章涵盖使多智能体系统(Multi-Agent System)得以协调、委派任务并集体求解单个智能体无法独立解决的问题所需的协议、模式与工程实践。

## 23.1 动机:智能体为何必须通信

### 专业化(专才)的必然要求

单个通才(generalist)智能体面临一个根本张力:知识广度与能力深度之间的取舍。现实世界的任务——法律文书审查、多步科研、企业级软件开发——往往对广度和深度都有要求。智能体间通信(Agent-to-Agent Communication, A2A)通过让一个由专才构成的网络协作来化解这一张力:每个专才贡献其所长,同时把不擅长的部分委派出去。

以下几股力量共同推动了对结构化智能体间通信的需求:

**认知负荷与上下文限制(Cognitive Load and Context Limits)。**
每个 LLM 都在一个有限的上下文窗口(context window)内运行。复杂工作流——横跨数百份文档、工具调用与推理步骤——会迅速超出单个智能体所能保留在记忆中的范围。通过把任务分解到多个智能体,每个智能体在可管理的上下文内运作,而编排智能体(orchestrating agent)只维护高层状态。

**专业化与专长(Specialization and Expertise)。**
不同的智能体可以针对特定领域进行微调(fine-tune)、定制提示或配备工具:例如一个能访问编译器与测试运行器的 `CodeAgent`,一个能访问判例数据库的 `LegalAgent`,一个配备统计库的 `DataAgent`。把子任务路由到合适的专才,既提升质量也提升效率。

**并行与吞吐(Parallelism and Throughput)。**
相互独立的子任务可以同时分派给多个智能体。一个科研编排者可以把文献检索扇出(fan out)到五个专才智能体上并行执行,再综合它们的结果——从而大幅缩短实际耗时(wall-clock time)。

**故障隔离与韧性(Fault Isolation and Resilience)。**
当某个智能体失败时,设计良好的多智能体系统可以换一个智能体重试、回退到更简单的方案,或上报给人工——而不会令整个工作流崩溃。

**委派与交接(Delegation and Handoff)。**
长时运行的任务可能需要随着上下文变化而在智能体之间交接。一个初始的 `PlannerAgent` 分解目标,把子任务交给 `ExecutorAgent`,最后由 `ReviewerAgent` 校验产出——每个智能体都只接收它恰好需要的上下文。

### A2A 通信的核心要求

1. **可发现性(Discoverability)**:智能体必须能够发现其他智能体并理解其能力。
2. **互操作性(Interoperability)**:由不同团队或厂商构建的智能体必须使用共同的协议交流。
3. **异步性(Asynchrony)**:长时运行的任务不得阻塞调用方;结果通过回调或轮询到达。
4. **安全性(Security)**:智能体之间必须相互认证,并强制执行授权边界。
5. **可观测性(Observability)**:每一次消息交换都必须可追踪,以便调试与审计。

## 23.2 Google A2A 协议

2025 年 4 月,Google(联合 50 余家技术合作伙伴)发布了 A2A 协议(Agent-to-Agent (A2A) Protocol)[372]——一份用于 AI 智能体之间互操作通信的开放规范。该协议随后捐赠给 Linux Foundation,截至 2026 年已获得超过 150 家组织的支持。A2A 围绕一组核心原则设计,使之有别于早期各种临时(ad-hoc)方案。

### 23.2.1 设计理念

A2A 规范阐明了五条指导原则(改编自官方规范 [372] 第 1.2 节):

**A2A 设计原则**

| 原则 | 含义 |
|---|---|
| **不透明执行(Opaque execution)** | 调用方智能体绝不窥探远端智能体的内部实现——只通过声明的接口交互。目标是 GPT-4、Gemini 还是基于规则的系统,对协议而言无关紧要,这使真正异构的智能体生态成为可能。 |
| **企业就绪(Enterprise readiness)** | 认证(OAuth 2.0、API keys、JWT)、审计日志与合规并非事后补丁——而是从协议层一开始就被纳入。 |
| **模态无关(Modality agnosticism)** | 单条消息可以混合文本、二进制文件与结构化 JSON 负载,无需协议扩展即可适配处理图像、音频、代码或文档的智能体。 |
| **以既有标准求简洁(Simplicity via existing standards)** | A2A 不发明新的传输层,而是复用 HTTP/HTTPS 承载 JSON-RPC 2.0 消息、用服务器推送事件(Server-Sent Events, SSE)做流式、以 gRPC 作为替代绑定——这些都是每个基础设施团队已在运营的技术。 |
| **异步优先的任务模型(Async-first task model)** | 长时运行操作是常态而非例外。推送通知与轮询都是一等机制,因此调用方永远不必为某个任务保持连接数小时。 |

### 23.2.2 智能体名片(Agent Cards)

A2A 可发现性的基础是 **Agent Card**(智能体名片)——一份机器可读的 JSON 清单,托管在一个知名端点(`/.well-known/agent.json`)上。它声明该智能体能做什么、如何认证、把任务发送到哪里——类似于 OpenAPI 规范,只不过面向自主智能体而非 REST 端点。

**Agent Card 结构**(托管于 `https://agent.example.com/.well-known/agent.json`):

```python
agent_card = {
    "name": "DataAnalysisAgent",
    "description": "Analyzes structured datasets, produces statistical summaries, "
                   "generates visualizations, and answers data questions.",
    "url": "https://agent.example.com/a2a",
    "version": "1.2.0",
    "capabilities": {
        "streaming": True,
        "pushNotifications": True,
        "stateTransitionHistory": True
    },
    "authentication": {
        "schemes": ["Bearer", "ApiKey"]
    },
    "skills": [
        {
            "id": "statistical-analysis",
            "name": "Statistical Analysis",
            "description": "Compute descriptive statistics, run hypothesis tests, "
                           "fit regression models on tabular data.",
            "tags": ["statistics", "data", "analysis", "regression"],
            "examples": [
                "What is the correlation between columns A and B?",
                "Run a t-test comparing these two groups.",
                "Fit a linear regression predicting sales from ad spend."
            ],
            "inputModes": ["text", "data"],
            "outputModes": ["text", "data", "file"]
        },
        {
            "id": "visualization",
            "name": "Data Visualization",
            "description": "Generate charts, plots, and dashboards from data.",
            "tags": ["charts", "plots", "visualization", "dashboard"],
            "examples": [
                "Create a bar chart of monthly revenue.",
                "Plot the distribution of customer ages."
            ],
            "inputModes": ["text", "data"],
            "outputModes": ["file", "text"]
        }
    ],
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"]
}
```

Agent Card 使基于能力的路由(capability-based routing)成为可能:编排智能体可以从注册表(registry)获取名片,将子任务语义化地匹配到最合适的智能体,并据此分派——这一切都无需硬编码的路由逻辑。

### 23.2.3 任务生命周期(Task Lifecycle)

A2A 把所有工作建模为 **任务(Task)**。一个任务沿着一个定义良好的状态机推进:

- **submitted(已提交)**:客户端已发送任务;服务端已确认收到。
- **working(处理中)**:智能体正在积极处理。客户端可轮询或等待 SSE 事件。
- **input-required(需要输入)**:智能体在继续之前需要来自用户或调用方智能体的额外信息(例如一个澄清问题、一份缺失的凭据)。
- **completed(已完成)**:任务成功结束;结果可在响应中获取。
- **failed(失败)**:发生不可恢复的错误;错误信息解释了原因。
- **rejected(已拒绝)**:智能体婉拒了任务(例如超出其能力范围或未获授权)。A2A v1.0 中新增。
- **canceled(已取消)**:任务被中止,可由客户端或服务端发起。

### 23.2.4 通过服务器推送事件流式传输(Streaming via Server-Sent Events)

对于会产生增量输出的任务(例如正在撰写的一份长报告、正在生成的一个代码文件),A2A 使用服务器推送事件(Server-Sent Events, SSE)。客户端打开一个持久的 HTTP 连接,并接收一个 JSON 事件流:

**SSE 事件流示例**:

```python
# 每个 SSE 事件携带一个 TaskStatusUpdateEvent 或 TaskArtifactUpdateEvent
# "撰写研究报告" 任务的示例流:

# 事件 1:状态更新
data: {
"id": "task-abc123",
"status": {"state": "working"},
"final": false
}

# 事件 2:部分产物(流式文本)
data: {
"id": "task-abc123",
"artifact": {
    "parts": [{"type": "text", "text": "## Introduction\n\nRecent advances in ..."}],
    "index": 0,
    "append": false,
    "lastChunk": false
},
"final": false
}

# 事件 3:追加更多文本
data: {
"id": "task-abc123",
"artifact": {
    "parts": [{"type": "text", "text": " reinforcement learning have shown ..."}],
    "index": 0,
    "append": true,   # 追加到已有产物
    "lastChunk": false
},
"final": false
}

# 最终事件:任务完成
data: {
"id": "task-abc123",
"status": {"state": "completed"},
"final": true
}
```

### 23.2.5 长时运行任务的推送通知(Push Notifications for Long-Running Tasks)

当一个任务可能耗时数分钟乃至数小时时,维持一个常开的 SSE 连接并不现实。A2A 支持推送通知:客户端注册一个 webhook URL,服务端在任务推进过程中向其 POST 状态更新。

```python
# 客户端在提交任务时注册一个推送通知端点
task_request = {
    "id": "task-xyz789",
    "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "Analyze Q3 sales data and produce a report."}]
    },
    "pushNotification": {
        "url": "https://my-orchestrator.example.com/webhooks/a2a",
        "token": "secret-hmac-token-for-verification",
        "authentication": {
            "schemes": ["Bearer"],
            "credentials": "eyJhbGciOiJIUzI1NiJ9..."
        }
    }
}
# 服务端会随任务状态迁移,把 TaskStatusUpdateEvent 对象 POST 到该 webhook URL
```

### 23.2.6 消息格式(Message Format)

A2A 消息由一个角色(role,user 或 agent)加上一组带类型的部分(parts:text、file 或结构化数据)构成。完整的消息模式(schema)、多模态示例与上下文传递指南将在 23.5 节展开。

### 23.2.7 认证与授权(Authentication and Authorization)

A2A 支持多种认证方案,在 Agent Card 中声明并按请求强制执行:

- **Bearer tokens(JWT/OAuth 2.0)**:企业部署的标准做法;令牌携带 scope(范围),用以限制调用方智能体被允许请求什么。
- **API keys**:用于内部或可信环境的更简单方案。
- **相互 TLS(Mutual TLS, mTLS)**:面向高安全部署的基于证书的认证。
- **OpenID Connect**:联邦身份(federated identity),支持跨组织的智能体通信。

**授权范围(Authorization Scope)的强制执行**

接收任务的智能体不仅必须核实"调用者是谁"(认证),还要核实"调用者被允许请求什么"(授权)。一个 `ReportingAgent` 可以接受任何已认证智能体的只读数据查询,但把写操作限制在持有特定 OAuth scope 的智能体。若不强制执行这一点,会在多智能体系统中制造权限提升(privilege escalation)漏洞。

## 23.3 通信模式(Communication Patterns)

多智能体系统依据任务性质、延迟要求与涉及智能体的数量,采用多种通信模式。

### 23.3.1 请求-响应(Request-Response)

最简单的模式:智能体 A 向智能体 B 发送任务并等待完整响应。适用于在继续之前就需要结果的、短小且定义明确的子任务。

### 23.3.2 流式(Streaming)

智能体 A 打开一个 SSE 连接;智能体 B 随着产出逐步返回部分结果。适合长篇生成(报告、代码)、实时协作或渐进式 UI 更新。

**流式模式用例**

编排者请求一个 `WritingAgent` 起草一份 10 页的技术文档。编排者无需等待 2 分钟拿回完整文档,而是随着每一节写完就流式获取,从而让 `ReviewAgent` 可以在后续章节仍在生成时就开始审阅前面的章节——这条流水线可把总延迟降低 40%–60%。

### 23.3.3 多轮交互(Multi-Turn Interaction)

某些任务需要迭代精化。智能体进入 `input-required` 状态,编排者提供澄清,任务随即恢复。这映射了人类协作流程:草稿 → 反馈 → 修订。

```python
# 多轮:编排者处理 input-required 状态
async def run_multiturn_task(client, initial_message):
    task = await client.send_task(message=initial_message)
    while task.status.state not in ("completed", "failed", "canceled"):
        if task.status.state == "input-required":
            # 智能体需要澄清
            clarification_needed = task.status.message
            print(f"Agent asks: {clarification_needed}")
            # 编排者生成或转发一条澄清响应
            user_reply = await get_clarification(clarification_needed)
            # 发送回复以继续任务
            task = await client.send_task(
                task_id=task.id,
                message={"role": "user",
                         "parts": [{"type": "text", "text": user_reply}]}
            )
        else:
            # 仍在处理中 —— 延迟后轮询
            await asyncio.sleep(2)
            task = await client.get_task(task.id)
    return task
```

### 23.3.4 广播(Broadcast)

编排者同时把同一条消息发送给多个智能体——适用于公告、分发共享上下文,或触发并行的独立工作流。

### 23.3.5 发布-订阅(Publish-Subscribe, Pub-Sub)

智能体订阅事件通道(例如 `new-document-uploaded`、`model-retrained`)。当某事件触发时,所有订阅的智能体都会收到通知。这解耦了生产者与消费者,支持响应式、事件驱动的架构。

### 23.3.6 协商(Negotiation)

两个智能体交换提议与反提议,以便就计划、资源分配或方法达成一致。在智能体有不同目标或约束的多智能体规划系统中很常见。

**协商模式示例**

`PlannerAgent` 提出一个 5 步科研计划。`ResourceAgent` 回应说第 3 步(运行一次大规模仿真)会超出算力预算。`PlannerAgent` 反提议一个缩小规模的仿真。`ResourceAgent` 同意。随后把达成一致的计划分派给执行智能体。

### 23.3.7 基于拍卖的任务分配(Auction-Based Task Allocation)

编排者宣布一个任务及其要求;候选智能体提交投标(估计完成时间、置信度、成本);编排者把任务授予中标者。这实现了跨智能体池的、动态的、基于市场的负载均衡。

**表 23.1:A2A 通信模式汇总。**

| 模式 | 延迟 | 最适合 |
|---|---|---|
| Request-Response(请求-响应) | 低 | 短小、定义明确的子任务 |
| Streaming(流式) | 低(首词) | 长篇生成、实时 UI |
| Multi-Turn(多轮) | 中 | 需要澄清的含糊任务 |
| Broadcast(广播) | 低 | 共享上下文分发 |
| Pub-Sub(发布-订阅) | 可变 | 事件驱动的响应式工作流 |
| Negotiation(协商) | 中–高 | 资源受限的规划 |
| Auction(拍卖) | 中 | 动态负载均衡 |

## 23.4 智能体发现与路由(Agent Discovery and Routing)

在一个智能体能与另一个通信之前,它必须先找到对方。智能体发现(discovery)就是定位能够处理给定任务的智能体的过程。

### 23.4.1 智能体注册表(Agent Registries)

智能体注册表是一种目录服务,为 Agent Card 建立索引并提供搜索与查找 API。存在两种部署模型:

**集中式注册表(Centralized Registry)**
单一权威注册表(例如企业服务目录)索引所有智能体。运营简单,但会构成单点故障,且未必能扩展到跨组织部署。

**联邦式注册表(Federated Registry)**
多个注册表,每个对其所在的领域或组织具有权威性,并配有跨注册表搜索协议。更具韧性也更利于隐私,但需要标准化的联邦协议。

### 23.4.2 基于能力的路由(Capability-Based Routing)

编排者不必硬编码智能体的 URL,而是执行基于能力的路由:它向注册表查询匹配所需技能的智能体,然后选择最佳匹配。

```python
class AgentRouter:
    """Routes tasks to agents based on capability matching."""
    # 基于能力匹配把任务路由到智能体

    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self._cache: dict[str, list[AgentCard]] = {}

    async def find_agents(self, required_skill: str,
                          tags: list[str] | None = None) -> list[AgentCard]:
        """Query registry for agents with the required skill."""
        # 按所需技能查询注册表
        params = {"skill": required_skill}
        if tags:
            params["tags"] = ",".join(tags)
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.registry_url}/agents", params=params)
            return [AgentCard(**card) for card in resp.json()["agents"]]

    async def route(self, task_description: str) -> AgentCard:
        """Semantically match a task description to the best available agent."""
        # 把任务描述语义化地匹配到最佳可用智能体
        # 嵌入任务描述
        task_embedding = await embed(task_description)
        # 获取所有已注册智能体
        all_agents = await self.find_agents(required_skill="*")
        # 按任务与智能体描述的余弦相似度为每个智能体打分
        scored = []
        for agent in all_agents:
            agent_embedding = await embed(agent.description)
            score = cosine_similarity(task_embedding, agent_embedding)
            scored.append((score, agent))
        # 返回得分最高的智能体
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
```

### 23.4.3 等价智能体间的负载均衡(Load Balancing Across Equivalent Agents)

当多个智能体提供相同能力时,路由器必须分配负载。常见策略:

- **轮询(Round-robin)**:在所有可用智能体间均匀分配任务。
- **最轻负载(Least-loaded)**:路由到活动任务最少的智能体(需要健康/指标端点)。
- **延迟感知(Latency-aware)**:路由到近期响应时间最低的智能体。
- **亲和性(Affinity-based)**:把相关任务路由到同一智能体,以利用缓存的上下文。

### 23.4.4 版本管理与兼容性(Version Management and Compatibility)

Agent Card 包含一个 version 字段。编排者应指定最低版本要求,并在仅有旧版本可用时优雅降级。推荐使用语义化版本(semantic versioning)[373](MAJOR.MINOR.PATCH):破坏性接口变更提升 MAJOR,新增能力提升 MINOR。

**长时运行系统中的版本偏移(Version Skew)**

在生产级多智能体系统中,不同智能体可能在不同时间被更新,从而造成版本偏移。一个按 Agent Card v2.1 编译的编排者可能遇到仍在运行 v1.3 的智能体。务必实现向后兼容的消息处理,并显式测试跨版本场景。

## 23.5 消息格式与模式(Message Formats and Schemas)

### 23.5.1 结构化 vs. 非结构化消息

A2A 支持一个从完全非结构化(纯文本)到完全结构化(带类型 JSON 模式)的谱系。恰当的选择取决于参与的智能体:

### 23.5.2 多模态消息(Multi-Modal Messages)

A2A 消息被结构化为一个角色(user 或 agent)加上一组带类型的部分(parts):

现代智能体越来越多地处理非文本模态。A2A 的 `FilePart` 支持任意 MIME 类型,从而支持丰富的多模态工作流:

**表 23.2:结构化与非结构化 A2A 消息的权衡。**

| 消息类型 | 优点 | 缺点 |
|---|---|---|
| 纯文本(Plain text) | 灵活、人可读、易生成 | 难以可靠解析,无模式校验 |
| 结构化 JSON(Structured JSON) | 机器可解析、可校验、有类型 | 需要模式约定,灵活性较低 |
| 混合(文本 + 数据)(Hybrid) | 人可读意图 + 机器可解析负载 | 构造与解析更复杂 |

**表 23.3:A2A 消息 part 类型(线上格式使用 `"type": "text"|"file"|"data"`)。**

| Part 类型 | 字段 | 用途 |
|---|---|---|
| TextPart | `text: string` | 自然语言指令、响应 |
| FilePart | `mimeType`, `uri` 或 `bytes` | 文档、图像、音频、代码文件 |
| DataPart | `data: object` | 结构化 JSON(工具结果、模式) |

**多模态 A2A 消息:数据分析**

```python
# 一条结合了文本指令、数据负载与文件的消息
message = {
    "role": "user",
    "parts": [
        {
            "type": "text",
            "text": "Analyze the attached CSV and the schema below. "
                    "Identify anomalies and produce a summary report."
        },
        {
            "type": "file",
            "mimeType": "text/csv",
            "uri": "https://storage.example.com/data/sales_q3.csv"
        },
        {
            "type": "data",
            "data": {
                "schema": {
                    "columns": ["date", "region", "product", "revenue", "units"],
                    "types": ["date", "string", "string", "float", "int"]
                },
                "expectedRowCount": 15000,
                "anomalyThreshold": 3.0   # z-score 阈值
            }
        }
    ]
}
```

**多模态 A2A 消息:图像分析**

```python
# 多模态消息:文本 + 图像 + 结构化数据
multimodal_message = {
    "role": "user",
    "parts": [
        {"type": "text",
         "text": "Describe what is wrong with this chart and suggest fixes."},
        {"type": "file",
         "mimeType": "image/png",
         "bytes": base64.b64encode(chart_image_bytes).decode()},
        {"type": "data",
         "data": {
            "chartType": "bar",
            "dataSource": "Q3 Revenue by Region",
            "knownIssues": ["y-axis does not start at zero",
                            "missing error bars"]
         }}
    ]
}
```

### 23.5.3 上下文传递:该分享什么 vs. 该保密什么

多智能体系统中一个关键设计决策是上下文作用域(context scoping):应把多少会话历史与内部状态传递给子智能体。

**上下文作用域原则**

- **最小上下文(Minimal Context)**:只传递子智能体完成任务所需的内容。减少 token 用量、延迟,以及泄露敏感信息的风险。
- **摘要化上下文(Summarized Context)**:与其传递原始会话历史,不如传递一份结构化摘要:目标、约束、已做出的决策,以及相关事实。
- **私有状态(Private State)**:内部推理、中间草稿与用户 PII(个人身份信息)通常不应转发给子智能体,除非明确需要。
- **关联 ID(Correlation IDs)**:始终传递一个 `correlationId`,使子智能体的动作能在日志与审计链路中回溯到发起工作流。

### 23.5.4 会话线程与会话关联 ID(Conversation Threading and Correlation IDs)

在复杂工作流中,可能同时有许多任务在途(in flight)。关联 ID 把跨智能体的相关任务联系在一起:

```python
import uuid

class WorkflowContext:
    """Carries correlation metadata through a multi-agent workflow."""
    # 在一个多智能体工作流中携带关联元数据

    def __init__(self, workflow_id: str | None = None):
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.span_id = str(uuid.uuid4())
        self.parent_span_id: str | None = None

    def child_context(self) -> "WorkflowContext":
        """Create a child context for a sub-task."""
        # 为子任务创建子上下文
        child = WorkflowContext(workflow_id=self.workflow_id)
        child.parent_span_id = self.span_id
        return child

    def to_metadata(self) -> dict:
        return {
            "x-workflow-id": self.workflow_id,
            "x-span-id": self.span_id,
            "x-parent-span-id": self.parent_span_id
        }

# 用法:附加到每一次 A2A 任务提交
ctx = WorkflowContext()
task = await client.send_task(
    message=message,
    metadata=ctx.to_metadata()
)
# 子任务使用子上下文以便追踪
sub_ctx = ctx.child_context()
```

## 23.6 协调协议(Coordination Protocols)

除了点对点通信,多智能体系统还能从更高层的协调协议中受益——这些结构化交互模式支持集体决策与问题求解。

### 23.6.1 合同网协议(Contract Net Protocol)

合同网协议(Contract Net Protocol, CNP)[374] 是一种经典的多智能体协调机制,被改造用于基于 LLM 的系统:

1. **公告(Announcement)**:管理者(manager)智能体向所有潜在承包者(contractor)智能体广播任务公告,包含任务要求与评估标准。
2. **投标(Bidding)**:承包者智能体依据自身能力评估任务,并提交包含估计完成时间、置信度与资源需求的投标。
3. **授标(Award)**:管理者选出中标投标(或为并行子任务选出多份投标)并授予合同。
4. **执行与汇报(Execution and Reporting)**:承包者执行任务并把结果回报给管理者。

**合同网协议实现**

```python
import dataclasses

class ContractNetManager:
    """Implements the Contract Net Protocol for task allocation."""
    # 为任务分配实现合同网协议

    async def allocate_task(self, task: Task,
                            candidate_agents: list[AgentCard]) -> AgentCard:
        # 阶段 1:向所有候选者公告任务
        announcement = {
            "type": "task-announcement",
            "task": dataclasses.asdict(task),
            "deadline": (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat(),
            "evaluationCriteria": ["confidence", "estimatedTime", "cost"]
        }
        # 阶段 2:收集投标
        bids = await asyncio.gather(*[
            self._request_bid(agent, announcement)
            for agent in candidate_agents
        ], return_exceptions=True)
        valid_bids = [(agent, bid) for agent, bid in zip(candidate_agents, bids)
                      if not isinstance(bid, Exception) and bid is not None]
        if not valid_bids:
            raise RuntimeError(f"No agents bid on task {task.id}")
        # 阶段 3:授予最高分投标者(最高置信度、最短时间)
        def score_bid(agent_bid):
            _, bid = agent_bid
            return bid["confidence"] - 0.1 * bid["estimatedSeconds"]
        winner_agent, winning_bid = max(valid_bids, key=score_bid)
        # 通知中标者与落标者
        await self._award_contract(winner_agent, task)
        await asyncio.gather(*[
            self._reject_bid(agent, task.id)
            for agent, _ in valid_bids if agent != winner_agent
        ])
        return winner_agent

    async def _request_bid(self, agent: AgentCard,
                           announcement: dict) -> dict | None:
        """Ask an agent to bid on a task."""
        # 请求一个智能体对任务投标
        try:
            result = await self.client.send_task(
                agent_url=agent.url,
                message={"role": "user",
                         "parts": [{"type": "data", "data": announcement}]}
            )
            return result.artifacts[0].parts[0]["data"]
        except Exception:
            return None
```

### 23.6.2 黑板系统(Blackboard Systems)

黑板系统(blackboard system)[321] 提供一个共享工作区("黑板"),智能体在上面发布部分解、观察与假设。其他智能体监视黑板,在能创造价值时贡献内容——这是一种机会主义式的问题求解方法。

黑板系统特别适合这类问题:求解路径事先未知,而不同智能体可能在不同阶段贡献力量——例如科学假设生成、复杂调试,或多来源情报分析。

### 23.6.3 共识协议(Consensus Protocols)

当多个智能体必须就一项决策达成一致时(例如执行哪个计划、某个结果是否正确),共识协议提供了结构化的投票机制:

**简单多数投票(Simple Majority Voting)**
每个智能体投票;得票超过 50% 的选项胜出。速度快,但如果智能体共用同一基础模型,则容易受相关错误(correlated errors)影响。

**加权投票(Weighted Voting)**
投票按智能体的置信度或历史准确度加权。更稳健,但需要校准良好的置信度估计。

**基于法定人数(Quorum-Based)**
一项决策需要至少 $n$ 个智能体中 $k$ 个同意。提供容错能力:最多 $n - k$ 个智能体可以失败或反对而不致阻塞。

**德尔菲法(Delphi Method)**
智能体投票,看到匿名化的结果,修订其投票,如此反复直至收敛。能降低锚定偏差(anchoring bias)并鼓励真正的审议。

```python
async def quorum_vote(agents: list[AgentCard], question: str,
                      options: list[str], quorum: int) -> str | None:
    """Run a quorum vote across agents. Returns winning option or None."""
    # 跨智能体运行法定人数投票,返回胜出选项或 None
    votes = await asyncio.gather(*[
        ask_agent_to_vote(agent, question, options)
        for agent in agents
    ])
    counts: dict[str, int] = {}
    for vote in votes:
        if vote in options:
            counts[vote] = counts.get(vote, 0) + 1
    # 返回第一个达到法定人数的选项
    for option, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count >= quorum:
            return option
    return None   # 未达到法定人数
```

### 23.6.4 领导者选举(Leader Election)

在动态多智能体系统中,可能需要在运行时选举一个领导者(编排者)——例如原编排者失效,或智能体在没有预先指定协调者的情况下自组织。经典的分布式系统算法(Bully、Ring)可改造用于智能体网络:智能体交换能力分数或优先级令牌,以选出最具能力的可用智能体作为领导者。

## 23.7 A2A vs. MCP:互补的协议

一个常见的困惑来源是 A2A 与模型上下文协议(Model Context Protocol, MCP)[335] 之间的关系。这两个协议是互补的,而非相互竞争:

**核心区别**

- **MCP 是纵向协议**:它把智能体向下延伸到数据库、API、文件系统与代码执行器的世界。只有智能体在推理;MCP 端点是确定性服务。
- **A2A 是横向协议**:它把一个推理智能体与另一个推理智能体联系起来。双方都是能够推理、规划与使用工具的智能参与者。

| 维度 | MCP | A2A |
|---|---|---|
| 参与方(Participants) | Agent ↔ 工具/资源 | Agent ↔ Agent |
| 智能性(Intelligence) | 一方(智能体)智能 | 双方都智能 |
| 有状态性(Statefulness) | 通常是无状态工具调用 | 带生命周期的有状态任务 |
| 流式(Streaming) | 有限(工具结果) | 一等公民的 SSE 流式 |
| 发现(Discovery) | 工具清单 | Agent Cards |
| 认证模型(Auth model) | 服务端控制 | 相互的、OAuth 2.0 |
| 典型延迟(Typical latency) | 毫秒级 | 秒到分钟级 |
| 用例(Use case) | "搜索网页"、"运行 SQL" | "委派给专家" |

### 23.7.1 何时用哪一个

- 当远端端点是一个**确定性函数**时使用 MCP:数据库查询、API 调用、代码执行沙箱。交互完全由智能体掌控。
- 当远端端点需要**推理**请求时使用 A2A:解读含糊指令、做出判断、使用自己的工具,或参与多轮对话。
- 在**同一系统**中两者并用:编排智能体用 A2A 委派给专家智能体,而每个专家智能体用 MCP 访问自己的工具。

### 23.7.2 组合架构(Combined Architecture)

在生产级多智能体系统中,A2A 与 MCP 在不同层次上协同:A2A 处理智能体之间的委派与协调(对等方之间的横向通信),而 MCP 处理每个智能体与其工具、数据源的连接(与能力的纵向集成)。这种关注点分离(separation of concerns)是构建可扩展智能体架构的关键。

![图 23.1:组合式 A2A + MCP 架构。编排者通过 A2A 委派给专家智能体;每个智能体通过 MCP 服务器访问自己的工具。](images/part-v-agentic-ai/agent-to-agent-communication-a2a/agent-to-agent-communication-a2a-p430-01.png)

- **用 A2A 做委派**:当一个智能体需要它所没有的能力时,它通过 A2A 任务消息委派给另一个智能体。每个智能体都是一个自包含的服务,拥有自己的 Agent Card。
- **用 MCP 做工具访问**:每个智能体通过 MCP 服务器连接到自己的工具。这意味着工具永远不会直接暴露给其他智能体——只能经由所属智能体的接口访问。
- **信任边界的分离**:编排者信任专家智能体(通过 A2A 认证核实);每个专家信任自己的 MCP 服务器(本地或已认证)。不存在传递性的工具访问。
- **独立扩缩**:偏重代码的负载可以扩缩 `CodeAgent` 实例;数据负载扩缩 `DataAgent`。编排者保持轻量。

## 23.8 多智能体系统中的安全与信任(Security and Trust)

多智能体系统引入了独特的安全挑战。当智能体 A 委派给智能体 B、B 又委派给智能体 C 时,这条信任链必须被谨慎管理。

### 23.8.1 智能体身份核验(Agent Identity Verification)

每个智能体都必须拥有可核验的身份。可选方案包括:

- 由可信身份提供方签发的 **JWT tokens**[375],携带智能体 ID、签发方与过期时间。接收方智能体使用该提供方的公钥加以核实。
- 由内部 CA 签发的 **mTLS 证书**[376],同时提供认证与传输加密。
- **去中心化标识符(Decentralized Identifiers, DIDs)**[377],适用于不存在单一可信权威机构的跨组织场景。

### 23.8.2 消息完整性与加密(Message Integrity and Encryption)

- 所有 A2A 通信应经由 **TLS 1.3**[378] 进行,以防窃听与中间人攻击。
- 对于敏感负载,端到端加密(例如 JWE)确保中间基础设施(负载均衡器、代理)无法读取消息内容。
- 消息签名(JWS)提供不可抵赖性(non-repudiation):接收方智能体可以证明某条具体消息来自某个具体发送方。

### 23.8.3 授权范围(Authorization Scopes)

并非每个智能体都应当能够请求任何其他智能体做任何事。OAuth 2.0 授权范围(authorization scopes)[379] 界定了这些边界:

```python
# DataAgent 的 OAuth 2.0 scope 示例
SCOPES = {
    "data:read": "Read data from connected databases",
    "data:write": "Write or modify data in connected databases",
    "data:export": "Export data to external systems",
    "analysis:run": "Execute statistical analyses",
    "analysis:schedule": "Schedule recurring analyses",
    "admin:config": "Modify agent configuration"
}
# 一个 ReportingAgent 可能只持有:data:read, analysis:run
# 一个 ETL pipeline 智能体可能持有:data:read, data:write, data:export
# 只有人类管理员持有:admin:config

class A2AServer:
    def verify_authorization(self, token: str, required_scope: str) -> bool:
        """Verify that the calling agent holds the required scope."""
        # 核实调用方智能体持有所需的 scope
        claims = jwt.decode(token, self.public_key, algorithms=["RS256"])
        granted_scopes = claims.get("scope", "").split()
        if required_scope not in granted_scopes:
            raise PermissionError(
                f"Caller lacks required scope '{required_scope}'. "
                f"Granted: {granted_scopes}"
            )
        return True
```

### 23.8.4 审计链路与问责(Audit Trails and Accountability)

**问责缺口(The Accountability Gap)**

在一条智能体委派链中,谁应对某个动作负责可能变得不清晰。如果智能体 A 请求智能体 B 删除一个文件,而 B 照做了,应由谁负责?每一次 A2A 交互都必须记录:调用方智能体的身份、任务描述、所使用的授权令牌、时间戳以及结果。这条审计链对事件响应、合规与调试都至关重要。

每个 A2A 服务端都应发出结构化审计日志:

```python
@dataclass
class A2AAuditEvent:
    timestamp: str            # ISO 8601
    workflow_id: str          # 顶层工作流的关联 ID
    span_id: str              # 本任务的 span
    parent_span_id: str       # 调用方任务的 span(用于委派链)
    caller_agent_id: str      # 调用方智能体的已核验身份
    callee_agent_id: str      # 本智能体的身份
    task_id: str
    skill_invoked: str
    authorization_scopes: list[str]
    outcome: str              # "completed" | "failed" | "rejected"
    duration_ms: int
    error_code: str | None
```

## 23.9 实现示例:多智能体科研工作流

下面这个示例演示了一个使用 A2A 的完整多智能体科研工作流:一个 `OrchestratorAgent` 分解一个科研问题,委派给专家智能体,并综合它们的结果。

```python
"""
Multi-agent research workflow using A2A protocol.
Demonstrates: Agent Cards, A2A client/server, task lifecycle,
multi-turn interaction, and agent handoffs.
"""
# 基于 A2A 协议的多智能体科研工作流
# 演示:Agent Card、A2A 客户端/服务端、任务生命周期、多轮交互与智能体交接

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# -- 数据模型 -----------------------------------------------------------------
class Part(BaseModel):
    type: str                 # "text" | "file" | "data"
    text: str | None = None
    data: dict | None = None
    mimeType: str | None = None
    uri: str | None = None

class Message(BaseModel):
    role: str                 # "user" | "agent"
    parts: list[Part]

class TaskStatus(BaseModel):
    state: str                # submitted | working | input-required | completed | failed
    message: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

class Artifact(BaseModel):
    parts: list[Part]
    index: int = 0
    append: bool = False
    lastChunk: bool = True

class Task(BaseModel):
    id: str
    status: TaskStatus
    messages: list[Message] = []
    artifacts: list[Artifact] = []
    metadata: dict = {}

# -- A2A 客户端(HTTP/REST 绑定) ---------------------------------------------
# 注意:A2A v1.0 定义了三种协议绑定:JSON-RPC 2.0、gRPC 与 HTTP+JSON/REST。
# 本示例为可读性使用 REST 绑定。
class A2AClient:
    """Client for sending tasks to A2A-compliant agents."""
    # 用于向 A2A 兼容智能体发送任务的客户端

    def __init__(self, agent_url: str, auth_token: str):
        self.agent_url = agent_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }

    async def get_agent_card(self) -> dict:
        """Fetch the agent's capability card."""
        # 获取智能体的能力名片
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.agent_url}/.well-known/agent.json",
                headers=self.headers
            )
            resp.raise_for_status()
            return resp.json()

    async def send_task(self, message: Message,
                        task_id: str | None = None,
                        metadata: dict | None = None) -> Task:
        """Submit a task and return the initial task object."""
        # 提交任务并返回初始任务对象
        payload = {
            "id": task_id or str(uuid.uuid4()),
            "message": message.model_dump(),
            "metadata": metadata or {}
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.agent_url}/tasks/send",
                json=payload,
                headers=self.headers,
                timeout=30.0
            )
            resp.raise_for_status()
            return Task(**resp.json())

    async def stream_task(self, message: Message,
                          metadata: dict | None = None) -> AsyncIterator[dict]:
        """Submit a task and stream SSE events."""
        # 提交任务并流式传输 SSE 事件
        payload = {
            "id": str(uuid.uuid4()),
            "message": message.model_dump(),
            "metadata": metadata or {}
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.agent_url}/tasks/sendSubscribe",
                json=payload,
                headers={**self.headers, "Accept": "text/event-stream"},
                timeout=300.0
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        event_data = json.loads(line[6:])
                        yield event_data
                        if event_data.get("final"):
                            break

    async def get_task(self, task_id: str) -> Task:
        """Poll for task status."""
        # 轮询任务状态
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.agent_url}/tasks/{task_id}",
                headers=self.headers
            )
            resp.raise_for_status()
            return Task(**resp.json())

    async def wait_for_completion(self, task: Task,
                                  poll_interval: float = 2.0) -> Task:
        """Poll until task reaches a terminal state."""
        # 轮询直到任务进入终态
        terminal_states = {"completed", "failed", "canceled"}
        while task.status.state not in terminal_states:
            await asyncio.sleep(poll_interval)
            task = await self.get_task(task.id)
        return task

# -- A2A 服务端(FastAPI) ----------------------------------------------------
class ResearchAgent:
    """
    A specialist research agent that searches literature
    and summarizes findings on a given topic.
    """
    # 一个专才科研智能体:检索文献并就给定主题总结发现

    AGENT_CARD = {
        "name": "ResearchAgent",
        "description": "Searches academic literature and synthesizes research findings.",
        "url": "https://research-agent.example.com/a2a",
        "version": "1.0.0",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True
        },
        "authentication": {"schemes": ["Bearer"]},
        "skills": [{
            "id": "literature-search",
            "name": "Literature Search",
            "description": "Search and summarize academic papers on a topic.",
            "tags": ["research", "literature", "academic", "papers"],
            "examples": [
                "Summarize recent papers on transformer attention mechanisms.",
                "What does the literature say about RLHF for code generation?"
            ],
            "inputModes": ["text"],
            "outputModes": ["text", "data"]
        }]
    }

    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self.app = FastAPI(title="ResearchAgent A2A Server")
        self._register_routes()

    def _register_routes(self):
        @self.app.get("/.well-known/agent.json")
        async def agent_card():
            return self.AGENT_CARD

        @self.app.post("/tasks/send")
        async def send_task(request: Request):
            body = await request.json()
            task = await self._create_and_run_task(body)
            return task.model_dump()

        @self.app.post("/tasks/sendSubscribe")
        async def send_subscribe(request: Request):
            body = await request.json()
            return StreamingResponse(
                self._stream_task(body),
                media_type="text/event-stream"
            )

        @self.app.get("/tasks/{task_id}")
        async def get_task(task_id: str):
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            return self.tasks[task_id].model_dump()

    async def _create_and_run_task(self, body: dict) -> Task:
        task_id = body.get("id", str(uuid.uuid4()))
        message = Message(**body["message"])
        task = Task(
            id=task_id,
            status=TaskStatus(state="submitted"),
            messages=[message],
            metadata=body.get("metadata", {})
        )
        self.tasks[task_id] = task
        # 异步运行
        asyncio.create_task(self._execute_task(task_id))
        return task

    async def _execute_task(self, task_id: str):
        task = self.tasks[task_id]
        task.status = TaskStatus(state="working")
        try:
            # 从消息中提取科研问题
            question = task.messages[0].parts[0].text
            # 模拟文献检索(生产环境替换为真实检索工具)
            await asyncio.sleep(1)   # 模拟延迟
            findings = await self._search_literature(question)
            # 产出产物
            task.artifacts = [Artifact(parts=[
                Part(type="text", text=findings["summary"]),
                Part(type="data", data={"papers": findings["papers"], "query": question})
            ])]
            task.status = TaskStatus(state="completed")
        except Exception as e:
            task.status = TaskStatus(state="failed", message=str(e))
        self.tasks[task_id] = task

    async def _search_literature(self, question: str) -> dict:
        """Placeholder: in production, calls a real search API."""
        # 占位实现:生产环境调用真实检索 API
        return {
            "summary": f"Based on a search of recent literature regarding "
                       f"'{question}', key findings include: ...",
            "papers": [
                {"title": "Attention Is All You Need", "year": 2017, "relevance": 0.95},
                {"title": "RLHF: Training Language Models to Follow Instructions",
                 "year": 2022, "relevance": 0.88}
            ]
        }

    async def _stream_task(self, body: dict) -> AsyncIterator[str]:
        task = await self._create_and_run_task(body)
        # 流式传输状态更新
        yield f"data: {json.dumps({'id': task.id, 'status': {'state': 'submitted'}, 'final': False})}\n\n"
        yield f"data: {json.dumps({'id': task.id, 'status': {'state': 'working'}, 'final': False})}\n\n"
        # 等待完成
        while task.status.state not in ("completed", "failed", "canceled"):
            await asyncio.sleep(0.5)
            task = self.tasks[task.id]
        # 流式传输产物
        if task.artifacts:
            for part in task.artifacts[0].parts:
                event = {
                    "id": task.id,
                    "artifact": {
                        "parts": [part.model_dump()],
                        "index": 0,
                        "append": False,
                        "lastChunk": True
                    },
                    "final": False
                }
                yield f"data: {json.dumps(event)}\n\n"
        # 最终状态
        yield f"data: {json.dumps({'id': task.id, 'status': task.status.model_dump(), 'final': True})}\n\n"

# -- 编排者:多智能体工作流 --------------------------------------------------
class ResearchOrchestrator:
    """
    Orchestrates a multi-agent research workflow:
    1. Decomposes the research question into sub-questions
    2. Dispatches each sub-question to a ResearchAgent
    3. Synthesizes results into a final report
    """
    # 编排一个多智能体科研工作流:
    # 1. 把科研问题分解为子问题
    # 2. 把每个子问题分派给一个 ResearchAgent
    # 3. 把结果综合为最终报告

    def __init__(self, research_agent_url: str, auth_token: str):
        self.research_client = A2AClient(research_agent_url, auth_token)
        self.workflow_id = str(uuid.uuid4())

    async def run(self, research_question: str) -> str:
        print(f"[Orchestrator] Starting workflow {self.workflow_id}")
        print(f"[Orchestrator] Question: {research_question}")
        # 步骤 1:分解为子问题
        sub_questions = self._decompose(research_question)
        print(f"[Orchestrator] Decomposed into {len(sub_questions)} sub-questions")
        # 步骤 2:并行分派子问题
        tasks = await asyncio.gather(*[
            self.research_client.send_task(
                message=Message(role="user", parts=[Part(type="text", text=q)]),
                metadata={"workflowId": self.workflow_id, "subQuestion": i}
            )
            for i, q in enumerate(sub_questions)
        ])
        # 步骤 3:等待所有任务完成
        completed_tasks = await asyncio.gather(*[
            self.research_client.wait_for_completion(task)
            for task in tasks
        ])
        # 步骤 4:检查失败
        failed = [t for t in completed_tasks if t.status.state == "failed"]
        if failed:
            print(f"[Orchestrator] Warning: {len(failed)} sub-tasks failed")
        # 步骤 5:综合结果
        findings = []
        for task, question in zip(completed_tasks, sub_questions):
            if task.status.state == "completed" and task.artifacts:
                summary = task.artifacts[0].parts[0].text
                findings.append(f"### {question}\n{summary}")
        report = self._synthesize(research_question, findings)
        print(f"[Orchestrator] Workflow complete. Report: {len(report)} chars")
        return report

    def _decompose(self, question: str) -> list[str]:
        """Decompose a complex question into focused sub-questions."""
        # 把复杂问题分解为聚焦的子问题
        # 生产环境:用 LLM 分解
        return [
            f"What are the foundational methods for: {question}?",
            f"What are the most recent advances in: {question}?",
            f"What are the open challenges and limitations in: {question}?"
        ]

    def _synthesize(self, question: str, findings: list[str]) -> str:
        """Synthesize sub-findings into a coherent report."""
        # 把子发现综合为一份连贯的报告
        # 生产环境:用 LLM 综合
        sections = "\n\n".join(findings)
        return f"# Research Report: {question}\n\n{sections}"

# -- 入口 ---------------------------------------------------------------------
async def main():
    orchestrator = ResearchOrchestrator(
        research_agent_url="https://research-agent.example.com/a2a",
        auth_token="eyJhbGciOiJSUzI1NiJ9..."
    )
    report = await orchestrator.run(
        "Reinforcement learning from human feedback for large language models"
    )
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
```

## 23.10 小结

**要点:智能体间通信**

1. **A2A 让大规模专业化成为可能**:通过把任务路由到专家智能体,多智能体系统同时获得深度与广度。(第 24 章深入讲解多智能体架构。)
2. **Google 的 A2A 协议**为可互操作的智能体通信提供了一个生产就绪的开放标准,涵盖 Agent Card、任务生命周期管理、SSE 流式传输与企业认证。
3. **通信模式**从简单的请求-响应延伸到复杂的协商与基于拍卖的分配——请依据任务复杂度与延迟需求来选择。
4. **A2A 与 MCP 互补**:A2A 连接智能体与智能体;MCP 连接智能体与工具。多数生产系统两者并用。
5. **安全不可妥协**:智能体身份核验、授权范围与审计链路在任何多智能体部署中都不可或缺。
6. **协调协议**(合同网、黑板、共识)在简单委派之外,提供了结构化的集体决策机制。
7. **通过关联 ID 实现可观测性**,对于调试和审计横跨众多智能体与工具的复杂多智能体工作流至关重要。

**A2A 中的开放研究问题**

- 在一个层级结构中,当来自多个编排者的指令相互冲突时,智能体应如何处理?哪些冲突解决机制最为有效?
- 智能体能否通过经验学会更好的路由与委派策略,而非依赖静态的能力声明?
- 如何防范提示注入(prompt injection)攻击——恶意智能体通过在消息中嵌入对抗性指令来操纵下游智能体?
- 上下文传递的隐私边界应当划在哪里——子智能体应看到多少会话历史?又如何在技术上强制执行这些边界?
- 随着智能体网络增长到数百乃至数千个智能体,如何在不产生瓶颈或一致性违例的前提下维护连贯的全局状态?
