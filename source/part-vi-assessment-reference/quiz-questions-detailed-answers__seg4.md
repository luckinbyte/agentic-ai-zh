<!-- page 555 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
27.24
MCP Protocol Questions
Q: Explain MCP’s N+M architecture and why it matters for the agent ecosystem
Answer: The N×M problem: Without MCP, N agent frameworks must each implement
integrations with M tools = N × M total integrations. Adding one new tool requires N imple-
mentations.
MCP’s N+M solution: Standardize the interface. Each agent implements one MCP client (N
total). Each tool implements one MCP server (M total). Total integrations = N + M.
Concrete example: 5 agent frameworks (LangChain/AutoGen/CrewAI/Claude/custom) × 20
tools (GitHub/Slack/DB/filesystem/. . . ) = 100 integrations without MCP. With MCP: 5 clients
+ 20 servers = 25 implementations.
Why it matters:
1. Tool reuse: Build a tool server once; use from any MCP-compatible agent
2. Agent portability: Switch from Claude to a custom agent without rewriting tool integra-
tions
3. Ecosystem growth: Lower barrier to adding new tools incentivizes the community to build
more
4. Composability: Connect multiple servers to one agent dynamically at runtime
Analogy: USB standardized peripheral connections. Before USB: every device had a proprietary
connector. After USB: one port fits all. MCP does the same for agent-tool connections.
Review: Chapter 20 (Model Context Protocol).
Q: What are MCP’s four core primitives and when do you use each?
Answer:
Primitive
Direction
Purpose
Example
Tools
Client →Server
Execute actions
create_issue; query_db
Resources
Client →Server
Read data
File contents; DB records
Prompts
Client →Server
Get templates
“Summarize this PR” template
Sampling
Server →Client
Request LLM gen
Server asks LLM to classify
Key distinctions:
• Tools vs Resources: Tools have side effects (create/modify/delete). Resources are read-
only. This distinction matters for safety — an agent can freely read resources but must get
approval for tools.
• Sampling reverses the direction: normally the client (agent) calls the server (tool). With
Sampling the server asks the client’s LLM for help. Use case: a code analysis server needs
the LLM to interpret a code snippet.
• Prompts are metadata (reusable templates) not execution. They help the agent formulate
better tool calls.
Review: Chapter 20 (Model Context Protocol).
555


<!-- page 556 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
27.25
Agent Communication (A2A) Questions
Q: How does Google’s A2A protocol differ from MCP and when do you need both?
Answer: Core distinction:
• MCP: Agent ↔Tool (structured function calls with defined schemas)
• A2A: Agent ↔Agent (opaque task delegation — you don’t know how the other agent
works)
Key A2A concepts:
• Agent Cards: JSON describing what an agent can do (like a resume). Discovery mechanism.
• Opaque execution: Requester doesn’t see internal reasoning of the delegate. Just sends
task and gets result.
• Task lifecycle: submitted →working →completed/failed (with streaming updates via
SSE)
When you need both:
1. An orchestrator agent uses A2A to delegate “research this topic” to a research agent
2. The research agent uses MCP to call web search and file read and database tools
3. Results flow back via A2A to the orchestrator
Architecture: A2A sits at the inter-agent layer; MCP sits at the agent-tool layer. A complete
system uses both: A2A for coordination between agents and MCP for each agent’s tool access.
Review: Chapters 20 and 22 (MCP; Agent-to-Agent Communication).
Q: What is the Contract Net Protocol and how does it apply to LLM agents?
Answer: The Contract Net Protocol (CNP) is a task allocation mechanism from distributed
AI:
Steps:
1. Announce: Manager broadcasts task description to all available agents
2. Bid: Agents assess their capability and submit bids (confidence; estimated cost; estimated
time)
3. Award: Manager selects best bid(s) based on criteria (capability/cost/availability)
4. Execute: Winning agent(s) perform the task
5. Report: Agent reports results back to manager
For LLM agents:
• Bidding = self-assessment: Each agent LLM evaluates “can I do this task well?” and
provides a confidence score. This requires calibrated self-knowledge.
• Specialization emerges: Code agents bid high on code tasks; research agents bid high on
research tasks. No central routing logic needed.
• Load balancing: If one agent is busy (high estimated time) others win the contract.
556


<!-- page 557 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Failure handling: If awarded agent fails then re-announce to remaining agents (automatic
failover).
Limitation for LLMs: LLMs often overestimate their capabilities (hallucinate confidence). Bids
should incorporate track record (historical success rate on similar tasks) not just self-reported
confidence.
Review: Chapters 22 and 23 (A2A; Multi-Agent Systems).
27.26
Multi-Agent Systems Questions
Q: Compare centralized vs decentralized multi-agent architectures for LLMs
Answer:
Centralized (Supervisor):
• One orchestrator LLM routes tasks to specialist workers
• Clear control flow; easy to debug (inspect supervisor decisions)
• Single point of failure; supervisor becomes token bottleneck
• Best for: well-defined workflows; small agent teams (3–5 agents)
Decentralized (Peer-to-Peer):
• Agents communicate directly; no central coordinator
• Resilient (no single point of failure); scales horizontally
• Hard to debug (emergent behavior); potential for conflicts and deadlocks
• Communication scales O(n2) without structure
• Best for: resilient systems; large agent populations; creative tasks where emergent behavior
is desired
Hybrid (Hierarchical): Tree structure with sub-managers. Combines benefits: local autonomy
within groups and global coordination at the top. Communication scales O(n log n).
Decision framework: Use centralized if you need predictability and auditability. Use decentral-
ized if you need resilience and creativity. Use hierarchical for large (>10 agent) systems.
Review: Chapter 23 (Multi-Agent Systems).
Q: What is CTDE and why is it important for training multi-agent LLM systems?
Answer: CTDE = Centralized Training; Decentralized Execution.
The problem: In multi-agent RL each agent’s environment is non-stationary (other agents are
changing their policies simultaneously). This makes independent training unstable.
CTDE solution:
• During training: A centralized critic has access to all agents’ observations and actions:
V (s1, s2, . . . , sn, a1, a2, . . . , an). This stabilizes training by removing non-stationarity from
the value function.
• During execution: Each agent acts based only on its own observation: ai = πi(oi). No
communication overhead at inference time.
For LLM agents: The centralized critic can be a reward model that evaluates the joint output
of all agents (e.g., did the team of agents produce a correct software system?) while each agent is
trained to maximize its contribution to the team reward using counterfactual credit assignment.
557


<!-- page 558 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Practical challenge: Full CTDE requires all agents to train simultaneously with shared state —
expensive for LLMs. Approximations: train agents in rounds (freeze others and train one) or use
population-based training with periodic synchronization.
Review: Chapter 23 (Multi-Agent Systems).
27.27
Agent Development Framework Questions
Q: Compare LangGraph vs AutoGen vs CrewAI for building multi-agent systems
Answer:
Dimension
LangGraph
AutoGen / CrewAI
Orchestration
Explicit state graph (nodes +
edges)
Implicit (conversation / role-based)
State mgmt
TypedDict schemas;
check-
pointing
Conversation history as state
Multi-agent
Graph with conditional rout-
ing
GroupChat / Crew
Debugging
Graph visualization; step re-
play
Chat logs
HITL
First-class (interrupt nodes)
Via approval tools
Production
LangGraph Cloud; persistence
Limited (AutoGen); growing (CrewAI)
Learning curve
High (graph concepts)
Low (AutoGen); Very low (CrewAI)
Choose LangGraph when: You need fine-grained control; complex conditional flows; production
deployment with persistence and human-in-the-loop.
Choose AutoGen when: Rapid prototyping of multi-agent conversations; code execution agents;
research experimentation.
Choose CrewAI when: Simple role-based teams; sequential task execution; quick demos;
minimal code.
Choose none (custom) when: You need maximum performance/control; don’t want framework
lock-in; or have non-standard orchestration patterns.
Review: Chapter 24 (Agent Development Frameworks).
Q: How do you test and evaluate an agent system in production?
Answer: Agent testing follows a testing pyramid:
Level 1 — Unit Tests (fast; many):
• Test individual tools in isolation (mock LLM; verify tool logic)
• Test prompt templates (given context; verify correct prompt construction)
• Test parsers (given LLM output; verify correct extraction)
Level 2 — Integration Tests (medium speed):
• Test complete agent loops with deterministic inputs
• “Golden trajectory” tests: known-good execution traces that must reproduce
• Tool chain tests: verify multi-tool sequences work end-to-end
Level 3 — Behavioral Tests (slow; few):
• Does the agent follow safety constraints? (adversarial inputs)
• Does it ask for clarification when appropriate?
558


<!-- page 559 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Does it stay within token/cost budgets?
Production evaluation:
• A/B testing: Route 5% of traffic to new agent version
• Shadow mode: Run new agent alongside old; compare outputs without serving
• LLM-as-judge: Automated quality scoring of agent responses
• User satisfaction: Thumbs up/down; task completion rate; time-to-resolution
Key metric: Task Success Rate (TSR) — fraction of tasks the agent completes correctly
without human intervention.
Review: Chapters 14 and 24 (LLM Evaluation; Agent Development Frameworks).
27.28
Agentic Environments Questions
Q: Design a reward function for a web browsing agent environment
Answer: For WebArena-style tasks (e.g., “find the cheapest flight from NYC to SF on Dec 15”):
Sparse reward (simple but hard to learn from):
r =
(
1
if final page/state matches ground truth
0
otherwise
Dense reward (better for training; harder to design):
1. Progress reward: +0.1 for each page that brings agent closer to goal (measured by text
similarity to target state)
2. Efficiency penalty: −0.01 per action (encourages shorter trajectories)
3. Milestone rewards: +0.3 for reaching intermediate goals (e.g., navigating to flight search
page)
4. Invalid action penalty: −0.05 for actions that produce errors (404; form validation
failures)
Potential-based shaping (preserves optimal policy):
rshaped(s, a, s′) = r(s, a, s′) + γΦ(s′) −Φ(s)
where Φ(s) = −min_steps_to_goal(s) (estimated by heuristic or learned value function).
Challenges: Partial observability (can’t always tell if you’re closer to goal); stochastic environ-
ments (page content changes); reward hacking (agent finds shortcuts that satisfy reward but not
user intent).
Review: Chapters 12 and 19 (LLM Agentic Training; Agentic Environments).
Q: What makes SWE-bench a particularly challenging agent benchmark?
Answer: SWE-bench tests agents on real GitHub issues from popular Python repositories:
Why it’s hard:
1. Repository-scale context: Agent must understand codebases with 100K+ lines. Cannot
fit in context window — must explore and search and navigate.
2. Underspecified tasks: Issues are written by humans with implicit context. Agent must
559


<!-- page 560 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
infer what’s actually needed.
3. Multi-file edits: Solutions often span multiple files with cascading dependencies.
4. Test verification: Must pass existing tests AND new tests that verify the fix.
5. No hand-holding: Unlike HumanEval (single function) SWE-bench requires full software
engineering workflow: read issue →explore code →localize bug →implement fix →verify.
State of the art (2024–2025): Best agents solve ∼50% of SWE-bench Verified (curated subset).
Full SWE-bench: ∼30%.
Key insight for training: SWE-bench exposes the gap between “coding ability” (writing correct
functions) and “software engineering ability” (understanding systems; navigating codebases; making
minimal changes). RL training on SWE-bench-style environments teaches agents exploration and
planning strategies not just code generation.
Review: Chapter 19 (Agentic Environments and Benchmarks).
27.29
Agentic UI Framework Questions
Q: Compare chat-based vs canvas-based UI paradigms for agents
Answer:
Chat-based (ChatGPT; Claude default):
• Linear message stream: user →assistant →user →. . .
• Pro: Familiar UX; natural for exploration and Q&A; easy to implement
• Con: Generated artifacts (code/documents) are buried in conversation. Hard to iterate on
a specific artifact. Context gets lost in long conversations.
Canvas/Artifact-based (Claude Artifacts; ChatGPT Canvas; Cursor):
• Side panel displays generated content; chat panel for instructions
• Agent can create and edit and iterate on persistent artifacts
• Pro: Artifacts persist independently of chat. Direct editing by user. Version history.
• Con: More complex UI; requires artifact type detection; harder to implement streaming to
both panels.
When to use which:
• Chat: brainstorming; Q&A; quick tasks; mobile interfaces
• Canvas: code generation; document writing; data analysis — any task with persistent output
that needs iteration
• Hybrid (most modern UIs): Chat by default; auto-elevate to canvas when detecting
code/document/visualization output
For agent training: The UI paradigm affects the reward signal. Canvas UIs provide explicit edit
feedback (user modifies the artifact) which can be used for online learning.
Review: Chapter 25 (Agentic UI Frameworks).
560


<!-- page 561 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q: How do you design approval gates for human-in-the-loop agent systems?
Answer: Approval gates pause agent execution at critical points for human review.
Three-tier model:
1. Auto-approve (no gate): Safe reversible actions. Read operations; searches; calculations.
2. Notify (soft gate): Potentially impactful but recoverable. Send email; create draft; modify
file. Agent proceeds but user is notified and can undo.
3. Block (hard gate): Irreversible or high-stakes. Delete data; send money; publish content;
execute code with side effects. Agent MUST wait for explicit approval.
Design principles:
• Minimize interruptions: Too many gates = user abandons the agent. The 3-tier model
lets most actions flow while catching dangerous ones.
• Show context: At approval gate display: what action; why (agent’s reasoning); what will
change; how to undo.
• Batch approvals: If agent needs 5 file writes present them together not one by one.
• Timeout handling: If user doesn’t respond within T minutes either retry notification or
proceed with safe default or abort gracefully.
• Learning from approvals: Track approval/rejection patterns. If users always approve a
certain action type consider auto-promoting it.
Implementation: Tool annotations (MCP’s destructiveHint and readOnlyHint) drive auto-
matic gate assignment. Custom rules can override based on context.
Review: Chapters 17 and 25 (Agent Harness; Agentic UI Frameworks).
27.30
RAG and Agentic RAG Questions
561


<!-- page 562 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q: Explain Reciprocal Rank Fusion (RRF) and why it works for hybrid retrieval
Answer: RRF combines rankings from multiple retrieval systems without needing score calibration:
RRF(d) =
X
r∈R
1
k + r(d)
where r(d) is the rank of document d in retriever r, and k = 60 is a constant that prevents
high-ranked documents from dominating.
Why it works:
1. No score normalization needed: BM25 scores are unbounded; dense similarity is in
[−1, 1]. RRF uses only ranks, making them directly comparable.
2. Robust to outliers: A single retriever giving anomalously high scores doesn’t dominate
because 1/(k + 1) ≈0.016 even for rank 1.
3. Complementary signals: BM25 catches exact keyword matches; dense retrieval catches
semantic similarity. Documents ranked highly by both get boosted.
Example: Document d is rank 3 in BM25 and rank 7 in dense. RRF score = 1/(60 + 3) + 1/(60 +
7) = 0.0159 + 0.0149 = 0.0308. A document at rank 1 in one but rank 100 in the other gets
1/61 + 1/160 = 0.0226 — lower despite having a top-1 ranking.
In practice: Hybrid (BM25 + dense + RRF) outperforms either alone on 85%+ of benchmarks.
Review: Chapter 15 (Retrieval-Augmented Generation).
Q: What is Agentic RAG and how does it differ from standard RAG?
Answer: Standard RAG follows a fixed pipeline: query →retrieve →generate. It has no
ability to:
• Decide whether retrieval is needed at all
• Evaluate if retrieved documents are sufficient
• Reformulate queries when retrieval fails
• Combine information from multiple retrieval steps
Agentic RAG treats retrieval as an action in the agent’s MDP:
• Retrieve-or-not decision: Agent assesses if it already knows the answer (skip retrieval for
factual questions in its training data)
• Query planning: Decomposes complex questions into sub-queries (“What year did X
happen?” + “Who was president then?”)
• Self-evaluation: After retrieval, grades relevance. If insufficient, reformulates query or tries
different source.
• Multi-hop reasoning: Retrieves →reasons →identifies knowledge gaps →retrieves again
• Source routing: Routes queries to appropriate knowledge bases (web for current events;
internal docs for company info; code search for programming)
Key architectural difference: Standard RAG = deterministic pipeline. Agentic RAG = state
machine with conditional transitions (LangGraph pattern with retrieve/grade/rewrite/generate
nodes).
Trade-off: Agentic RAG is more accurate on complex queries but adds latency (multiple LLM calls
for routing/grading). Use standard RAG for simple factual lookups; agentic RAG for multi-hop or
ambiguous queries.
Review: Chapters 15 and 17 (RAG; Agent Harness).
562


<!-- page 563 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q: Compare Self-RAG and CRAG approaches to improving retrieval quality
Answer:
Self-RAG (Asai et al., 2023):
• Trains special reflection tokens into the LLM vocabulary
• At inference, model outputs tokens like [Retrieve], [IsRel], [IsSup], [IsUse]
• Model decides when to retrieve (not every query needs it)
• After retrieval, model self-grades: Is the retrieved passage relevant? Does my answer follow
from it?
• Training: SFT on data augmented with reflection labels from GPT-4
• Pro: Single model handles everything. Con: Requires custom training.
CRAG (Corrective RAG, Yan et al., 2024):
• Uses a lightweight retrieval evaluator (separate model) to grade retrieved docs
• Three actions based on confidence: Correct (use as-is), Ambiguous (augment with web
search), Incorrect (discard; fallback to web)
• Adds a knowledge refinement step: extract only relevant sentences from retrieved docs
• Pro: Works with any frozen LLM. Con: Extra model for evaluation; added latency.
Key difference: Self-RAG embeds retrieval decisions into the LLM itself (requires training).
CRAG is a pipeline approach that wraps around any LLM (no training needed). Self-RAG is
more elegant; CRAG is more practical for production with existing models.
Review: Chapter 15 (Retrieval-Augmented Generation).
Q: What is the lost-in-the-middle problem and how do you mitigate it?
Answer: The problem: When retrieved context is long (many passages), LLMs disproportion-
ately attend to information at the beginning and end of the context, ignoring information in the
middle. If the answer is in passage 5 of 10, the model may miss it.
Empirical evidence: Liu et al. (2023) showed that for 20-document retrieval, accuracy drops by
15–20% when the relevant document is in positions 5–15 vs positions 1–3.
Mitigation strategies:
1. Re-rank and truncate: Use a cross-encoder to re-rank, then only include top-3 most
relevant passages (fewer = less lost-in-middle).
2. Strategic ordering: Place highest-relevance passages at the start AND end of context,
low-relevance in the middle.
3. Contextual compression: Summarize each passage to 1–2 sentences before insertion. Less
text = less position bias.
4. Map-reduce: Process each passage independently (map), then combine answers (reduce).
Eliminates position effects entirely.
5. Citation prompting: Ask model to cite which passage it used. This forces attention to all
passages.
6. Chunk size reduction: Smaller chunks mean fewer total chunks needed to cover the
answer.
563


<!-- page 564 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Best practice: Retrieve many (20+), re-rank to top 3–5, order by relevance (best first). This
sidesteps the problem entirely for most use cases.
Review: Chapter 15 (Retrieval-Augmented Generation).
564
