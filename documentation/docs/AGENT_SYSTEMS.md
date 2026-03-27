# Поддержка агентных систем

**Назначение:** полное и детальное описание того, **как фреймворк Иггдрасиль обеспечивает агентные системы** (оркестрация агентов, инструменты, субагенты, планирование, визуальные workflow, состояние и чекпоинт) в рамках единого движка гиперграфа. Документ задаёт **полный паритет по возможностям** с эталонными решениями: **n8n** (workflow-оркестрация, выполнение по стеку узлов, частичный запуск, EngineRequest/EngineResponse, типы узлов execute/trigger/poll/webhook), **LangChain** (create_agent, middleware), **LangGraph** (StateGraph, Pregel, каналы, чекпоинт, прерывания), **DeepAgents** (глубокий агент с инструментами, субагентами, backend, interrupt_on). Паритет достигается **полной интеграцией аналогичного функционала на базе движка Иггдрасиль** — без использования кода или исполнения этих платформ как основы; собственные гиперграфы, узлы-задачи и run принадлежат фреймворку. Документ описывает маппинг компонентов агентной системы на роли узлов-задач, типичные топологии графов, форматы данных (runData, pinData, tool_calls/tool_results), цикл выполнения и агентный цикл, конфигурацию, сериализацию и использование на уровне воркфлоу.

**Связь с каноном:** [01_FOUNDATION.md](01_FOUNDATION.md) — Block, Node, порты, run; [02_ABSTRACT_TASK_NODES.md](02_ABSTRACT_TASK_NODES.md) — семь ролей и контракты портов; [03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) — гиперграф задачи, циклы, agent_loop; [HYPERGRAPH_ENGINE.md](HYPERGRAPH_ENGINE.md) — планировщик, итеративная фаза, буферы; [LANGUAGE_MODELS.md](LANGUAGE_MODELS.md) — LLM как ядро агента, генерация, tool_calls; [DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) — обзор доменов.

**Язык:** русский.

**Паритет и превосходство по удобству.** Решения на базе Иггдрасиль должны не просто обеспечивать **паритет по функциональности и мощности** со всеми перечисленными state-of-the-art решениями (n8n, LangChain, LangGraph, DeepAgents), **но и** давать пользователю **более удобный, простой и интуитивно понятный интерфейс взаимодействия**: любое решение (агент с инструментами, workflow-оркестрация, субагенты, частичный запуск и т.д.) должно быть собираемо **в 3–5 строк кода** при сохранении возможности глубочайшей кастомизации. Это достигается продуманной логикой устройства фреймворка и разумной автоматизацией (фабрики гиперграфов, разумные умолчания, единый контракт run), без ущерба для расширяемости и полного контроля над каждым узлом и параметром.

---

## Важное замечание о текущем runtime

Этот документ описывает **целевую агентную архитектуру**, но текущий runtime репозитория стабилизирован только на уровне **minimal tool loop**:

- узел-агент определяется через `metadata["agent_node_ids"]`, `node.is_agent` или `graph_kind="agent"` для backbone-узлов;
- tool-узлы исключаются из основного execution plan и вызываются только через `tool_id_to_node_id`;
- между шагами agent-loop гарантированно передаётся только `tool_results`, а не произвольный state-thread;
- неизвестный `tool_id` по умолчанию пропускается; для fail-fast нужно явно задать `missing_tool_policy="error"`;
- per-tool interrupt/resume внутри `agent_loop` пока не следует считать стабильной гарантией.

Всё более широкое поведение, описанное ниже как parity с n8n / LangGraph / DeepAgents, нужно читать как **цель архитектуры**, а не как уже достигнутый runtime-факт.

## 1. Место агентных систем в фреймворке

### 1.1 Единая онтология

Агентные системы **не вводят отдельной онтологии**. Они реализуются **теми же сущностями**, что и остальные домены: узлы-задачи (Backbone, Conjector, Inner Module, Outer Module, Converter, Injector, Helper), гиперграф задачи, движок гиперграфа (планировщик, итеративная фаза, буферы), порты и run. Один гиперграф задачи может представлять **одну агентную задачу** (агент с инструментами, субагент, планировщик) или **один workflow** в стиле n8n (граф узлов с типами execute/trigger/poll/webhook); комбинирование нескольких агентов или workflow выполняется на уровне **воркфлоу**, где узлами служат целые гиперграфы. Движок не различает «тип домена» — только структуру графа, порты, циклы и опции run; агентный цикл (агент → tool_calls → выполнение инструментов → tool_results → агент) и выполнение workflow по стеку узлов выражаются одной и той же механикой итеративной фазы и планировщика ([HYPERGRAPH_ENGINE.md](HYPERGRAPH_ENGINE.md), [03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) §7.3).

### 1.2 Что такое «агентная система» в каноне

**Агентная система** в каноне Иггдрасиль — это конфигурация гиперграфа и опций run, при которой:

1. **Узел-агент** (или узел, возвращающий запрос на выполнение подузлов) выдаёт не только данные, но и **запрос на выполнение других узлов** (tool_calls, EngineRequest-подобный контракт). Движок по этому запросу выполняет указанные узлы (инструменты), собирает результаты и передаёт их обратно агенту; цикл повторяется до отсутствия запросов или до достижения лимита шагов (**agent_loop**).

2. **Workflow-оркестрация** (в духе n8n): граф узлов с явными связями; выполнение идёт по **стеку готовых к выполнению узлов** (nodeExecutionStack). На каждом шаге движок снимает узел со стека, выполняет его, записывает результат в буферы (runData) и добавляет в стек следующие узлы по исходящим рёбрам; при нескольких входах узел добавляется в стек только после появления данных по всем входам (waitingExecution). Поддерживаются **частичный запуск** (от выбранного узла до destination), **pinData** (подмена выхода узла зафиксированными данными), **триггеры и вебхуки** как узлы без входа, инициирующие выполнение.

3. **Состояние и персистенция:** общее состояние выполнения (runData по узлам, чекпоинт) сохраняется и загружается по опциям run (thread_id, checkpoint_id); возможны **прерывания** (interrupt_on) до/после указанного узла или инструмента с последующим **resume**.

4. **Субагенты и вложенные графы:** инструмент «task» или аналог вызывает **вложенный гиперграф** (субагент); движок выполняет run(субагент, inputs) и возвращает результат как tool_result главному агенту.

В рамках фреймворка всё это укладывается в гиперграф и движок: **агентный цикл** — частный случай итеративной фазы с проверкой выхода узла-агента на наличие tool_calls; **workflow по стеку** — альтернативная стратегия планировщика (stack-based execution вместо/в дополнение к dataflow по готовности портов); **частичный запуск** — выбор подмножества узлов (подграф от start до destination) и пересоздание плана выполнения из сохранённого runData; **триггеры/вебхуки** — узлы без обязательных входных портов, инициирующие выполнение при событии.

### 1.3 Связь с эталонными решениями

- **n8n** даёт эталон **визуального workflow**: граф узлов (INode, IConnections), выполнение по nodeExecutionStack, run/runPartialWorkflow2, pinData, runData, типы узлов (execute, trigger, poll, webhook), EngineRequest/EngineResponse для вызова инструментов агента, rewireGraph для выполнения узла как tool, DirectedGraph и findSubgraph для частичного запуска. В Иггдрасиль гиперграф задачи с опцией run в режиме «stack-based» или с планировщиком, учитывающим порядок и множественные входы, реализует ту же семантику; узлы-инструменты и узел-агент соответствуют узлам с execute и контрактом tool_calls/EngineRequest.

- **LangChain** и **LangGraph** дают эталон **агента и графа состояний**: create_agent (модель, инструменты, middleware, checkpointer, interrupt), StateGraph (State → Partial<State>), Pregel (цикл планирование → выполнение → запись в каналы → чекпоинт), каналы с редьюсерами, прерывания и resume. В Иггдрасиль этому соответствуют фабрика гиперграфа «агент + инструменты», буферы движка как каналы состояния, итеративная фаза и сериализация по [SERIALIZATION.md](SERIALIZATION.md).

- **DeepAgents** даёт эталон **глубокого агента**: встроенные инструменты (todo, файлы, execute, task для субагентов), skills, memory, backend (StateBackend, Sandbox), interrupt_on. В Иггдрасиль — те же узлы-инструменты (Helper), Conjector с подстановкой memory/skills, конфиг backend для узлов файловой системы и execute, субагенты как вложенные гиперграфы, interrupt_on в metadata или run options.

---

## 2. Полный паритет с эталонными решениями

Ниже перечислено **всё**, что фреймворк Иггдрасиль должен уметь поддерживать в части агентных систем, чтобы обеспечить **полный паритет** с:

- **n8n** — workflow-оркестрация, выполнение по стеку, частичный запуск, типы узлов, pinData, runData, EngineRequest/EngineResponse, rewire для tool-узлов.
- **LangChain** — create_agent (модель, инструменты, system_prompt, middleware, checkpointer, store, interrupt_before/after).
- **LangGraph** — StateGraph, Pregel, каналы состояния, чекпоинт, store, прерывания, resume.
- **DeepAgents** — глубокий агент (todo, файлы, субагенты, skills, memory, backend, interrupt_on, MCP).

**Важно:** паритет достигается **не за счёт использования кода или исполнения этих платформ**, а за счёт **полной интеграции аналогичного функционала в рамках фреймворка Иггдрасиль**. Гиперграфы, узлы-задачи (агент, инструменты, триггеры, поллеры, вебхуки), движок (планировщик, итеративная фаза, agent_loop, опционально stack-based execution), сериализация (конфиг + чекпоинт, runData) и API run принадлежат Иггдрасиль. Допускается совместимость по форматам (импорт/экспорт workflow, протоколы MCP); исполнение и архитектура остаются в границах фреймворка.

**Источники перечисления:** репозитории n8n-io/n8n, langchain-ai/langchain, langchain-ai/langgraph, langchain-ai/deepagents (актуальные на момент подготовки документа).

---

### 2.1 Паритет с n8n: workflow-оркестрация и выполнение

**Workflow как граф:**

- **Workflow** n8n: `id`, `name`, `nodes: INodes` (по имени), `connectionsBySourceNode`, `connectionsByDestinationNode`, `nodeTypes`, `settings`, `staticData`, `pinData`. В Иггдрасиль — **гиперграф задачи**: узлы с node_id, гиперрёбра по портам; аналог connections — рёбра (source_node, source_port) → (target_node, target_port); nodeTypes — реестр типов блоков; staticData — данные узлов или графа по конвенции (getStaticData(global) / getStaticData(node, node)); pinData — опция run или metadata (подмена выхода узла заданными данными для теста/ручного запуска).

**Выполнение по стеку (nodeExecutionStack):**

- n8n: `executionData.nodeExecutionStack: IExecuteData[]`; на каждом шаге `shift()` — снимается один элемент `{ node, data, source }`, выполняется узел, результаты записываются в resultData.runData; по исходящим связям следующие узлы добавляются в стек (или в waitingExecution при нескольких входах). В Иггдрасиль — **режим планировщика «stack-based»**: планировщик ведёт очередь готовых к выполнению узлов; после выполнения узла в очередь добавляются соседи по исходящим рёбрам (с учётом политики «все входы готовы» при множественных входах). Буферы движка играют роль runData (выход узла → запись в буфер по порту; следующий узел читает из буфера по входящему ребру).

**run() и runPartialWorkflow2():**

- **run(workflow, startNode?, destinationNode?, pinData?, ...)**: определение start node (или getStartNode(destinationNode)); при destinationNode — runNodeFilter = родители + опционально сам узел; инициализация nodeExecutionStack одним элементом (startNode, data, source: null); processRunExecutionData(workflow). В Иггдрасиль — run(hypergraph, inputs, start_node_id?, destination_node_id?, pin_data?, ...): при указании destination строится подграф (аналог findSubgraph) и фильтр узлов; начальное состояние стека/очереди — один или несколько стартовых узлов (триггеры или указанный start_node); выполнение до исчерпания очереди или до достижения destination.

- **runPartialWorkflow2(workflow, runData, pinData, dirtyNodeNames, destinationNode, agentRequest?)**: DirectedGraph из workflow; при tool-ноде rewireGraph (подмена на виртуальный ToolExecutor); findTriggerForPartialExecution; findSubgraph(trigger → destination); findStartNodes; handleCycles; cleanRunData; recreateNodeExecutionStack; processRunExecutionData. В Иггдрасиль — **частичный run**: run(hypergraph, inputs, run_data=..., pin_data=..., dirty_node_ids=[...], destination_node_id=...): загрузка сохранённого runData в буферы; построение подграфа от триггера/старта до destination; определение узлов для перезапуска (dirty); формирование начальной очереди выполнения из start nodes с учётом уже имеющихся данных; выполнение до destination. Циклы в графе обрабатываются конвенцией планировщика (handleCycles-аналог).

**Типы узлов (INodeType):**

- **execute** — основной тип: nodeType.execute(context, subNodeExecutionResults?) → NodeOutput. В Иггдрасиль — узел-задача с ролью, имеющей метод run (Backbone, Helper, Converter и т.д.); выполнение через движок по портам.
- **trigger** — узел без входа, инициирует выполнение; в manual режиме вызывается trigger, иначе данные пробрасываются. В Иггдрасиль — узел без обязательных входящих портов или с портом «event»; при run в режиме «от триггера» планировщик помещает триггерные узлы в начальную очередь; опционально отдельный block_type trigger.
- **poll** — опрос источника; в manual — вызов poll, иначе pass-through. В Иггдрасиль — узел с ролью, способной периодически выдавать данные (Helper или специализированный тип); при ручном запуске — один вызов, иначе данные из кэша/входа.
- **webhook** — узел вебхука; webhookMethods для setup. В Иггдрасиль — узел, активируемый внешним HTTP-запросом; при событии запроса данные помещаются на вход узла и он ставится в очередь.
- **supplyData** — поставка данных (например, для AI). В Иггдрасиль — Conjector или Helper, подающий данные по порту.

**ExecuteContext, IExecuteFunctions:**

- Контекст выполнения узла: workflow, node, additionalData, mode, runExecutionData, runIndex, connectionInputData, inputData, executeData, helpers (httpRequest, binary, dataTable, returnJsonArray, и т.д.), getNodeParameter. В Иггдрасиль — при выполнении узла движок передаёт в блок run контекст (доступ к буферам по входящим рёбрам, опции run, execution_id, run_index); getNodeParameter и helpers реализуются через конфиг узла и порты или через обёртку контекста выполнения.

**EngineRequest / EngineResponse:**

- Узел-агент возвращает не данные, а **EngineRequest** (список actions: nodeName, type, metadata); движок выполняет указанные узлы, собирает результаты в **EngineResponse** (actionResponses, metadata) и передаёт обратно агенту при следующем вызове. В Иггдрасиль — **agent_loop**: выход узла-агента содержит tool_calls; по таблице tool_id → node_id движок выполняет узлы-инструменты, формирует tool_results и снова вызывает агента; контракт tool_calls/tool_results эквивалентен EngineRequest/EngineResponse ([LANGUAGE_MODELS.md](LANGUAGE_MODELS.md) §6.2, [03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) §7.3).

**rewireGraph (tool как узел):**

- При частичном запуске с destination = узел-инструмент n8n переписывает граф так, что «виртуальный» ToolExecutor вызывает этот узел. В Иггдрасиль при agent_loop инструмент уже является отдельным узлом; вызов «только этот инструмент» с подставленными входами от агента реализуется запуском подмножества графа (агент + один инструмент) с передачей аргументов tool_calls на входы инструмента.

**DirectedGraph, findSubgraph, findStartNodes, cleanRunData, recreateNodeExecutionStack:**

- Представление графа для редактирования и частичного запуска; извлечение подграфа от trigger до destination; определение узлов старта для перезапуска; очистка runData от узлов вне подграфа; построение нового стека из start nodes и runData. В Иггдрасиль — операции над гиперграфом: getSubgraph(from_nodes, to_nodes), getStartNodes(run_data, pin_data, dirty_nodes), mergeRunDataIntoBuffers(run_data), buildExecutionPlan(subgraph, start_nodes) — возврат очереди/стека для планировщика.

**pinData, runData:**

- **pinData:** фиксированные выходы узлов (для теста); при выполнении если pinData[nodeName] задан, узел не выполняется, подставляются эти данные. В Иггдрасиль — run(..., pin_data={ node_id: value }) или metadata; движок при планировании узла проверяет pin_data и при наличии подставляет значение в буфер выхода узла без вызова run блока.
- **runData:** resultData.runData — по имени узла массив выполнений (каждый элемент — данные выхода, source, executionIndex, и т.д.). В Иггдрасиль — буферы движка по (node_id, port) или агрегированная структура run_data в опциях run и в сериализации чекпоинта.

**Множественные входы (waitingExecution):**

- Узел с несколькими входами добавляется в стек только когда по каждому входу получены данные; до этого данные накапливаются в waitingExecution[nodeName][runIndex]. В Иггдрасиль — политика «все входящие порты готовы»: планировщик ставит узел в очередь только когда по каждому входящему ребру в буфере есть значение (или явно null по конвенции).

**Execution order (v1 vs новый):**

- n8n: settings.executionOrder === 'v1' — unshift в стек (LIFO), иначе push (FIFO). В Иггдрасиль — опция run execution_order: "lifo" | "fifo" или аналог в metadata графа; планировщик использует её при добавлении следующих узлов в очередь.

Итог по §2.1: сценарий «workflow n8n с узлами execute/trigger/poll/webhook, полный и частичный запуск, pinData, runData, агент с EngineRequest» должен быть реализуем **гиперграфом Иггдрасиль** и движком с поддержкой stack-based (или эквивалентной) стратегии и agent_loop; реализация — собственные узлы и движок.

---

### 2.2 Паритет с LangChain: create_agent и middleware

(Кратко; детали в [LANGUAGE_MODELS.md](LANGUAGE_MODELS.md) §2.2.)

- **create_agent(model, tools, system_prompt, middleware, response_format, checkpointer, store, interrupt_before, interrupt_after, …)** → скомпилированный граф. В Иггдрасиль — **фабрика гиперграфа**: узел-агент (Backbone = LLM + парсинг tool_calls), узлы-инструменты, Conjector (system_prompt), опционально дополнительные узлы под middleware; checkpointer/store — сериализация и опции run (thread_id, checkpoint_id); interrupt_before/after — metadata или run options.
- **Middleware** (TodoList, Summarization, HITL, Filesystem, SubAgents, ToolRetry, и т.д.) — в Иггдрасиль выражаются дополнительными узлами (Helper, Conjector) или логикой внутри узла-агента и опциями run.

---

### 2.3 Паритет с LangGraph: StateGraph, Pregel, каналы, чекпоинт

(Кратко; детали в [LANGUAGE_MODELS.md](LANGUAGE_MODELS.md) §2.3.)

- **StateGraph:** узлы State → Partial<State>; состояние = буферы движка по портам; редьюсеры по ключам — конвенция порта или узел-агрегатор.
- **Pregel:** цикл планирование → выполнение узлов → запись в каналы → чекпоинт — **итеративная фаза движка** с сохранением состояния по опции.
- **Каналы (LastValue, reducer, ephemeral):** типы портов/буферов по конфигу.
- **Checkpoint, Store, interrupt, resume:** state_dict графа + пользовательское состояние; thread_id, checkpoint_id в run; interrupt_on в metadata/run.

---

### 2.4 Паритет с DeepAgents: глубокий агент, инструменты, субагенты

(Кратко; детали в [LANGUAGE_MODELS.md](LANGUAGE_MODELS.md) §2.4.)

- **create_deep_agent:** встроенные инструменты (write_todos, ls, read_file, write_file, edit_file, glob, grep, execute, task), subagents, skills, memory, backend (StateBackend, Sandbox), interrupt_on. В Иггдрасиль — гиперграф с узлами-инструментами (Helper: todo, filesystem_*, execute), субагенты как вложенные гиперграфы (task → run(subgraph)); backend_ref в конфиге узлов; Conjector + Helper для skills/memory; interrupt_on в metadata или run.
- **MCP tools:** узлы-инструменты с конфигом источника MCP.

---

### 2.5 Сводная таблица паритета (чеклист)

| Источник | Возможность | Выражение в Иггдрасиль |
|----------|-------------|------------------------|
| **n8n** | Workflow (nodes, connectionsBySource/Dest, settings, staticData, pinData) | Гиперграф: узлы, рёбра по портам; конфиг; pin_data в run/metadata |
| **n8n** | nodeExecutionStack, processRunExecutionData (цикл shift → runNode → addNodeToBeExecuted) | Планировщик в режиме stack-based: очередь узлов, выполнение, добавление следующих по рёбрам |
| **n8n** | run(workflow, startNode?, destinationNode?, pinData?) | run(hypergraph, inputs, start_node_id?, destination_node_id?, pin_data?) |
| **n8n** | runPartialWorkflow2(runData, pinData, dirtyNodeNames, destinationNode, agentRequest?) | run(..., run_data=..., pin_data=..., dirty_node_ids=..., destination_node_id=...) с подграфом и пересозданием плана |
| **n8n** | Типы узлов: execute, trigger, poll, webhook, supplyData | Роли узлов + опционально block_type trigger/poll/webhook; триггеры — узлы без обязательных входов |
| **n8n** | IExecuteData (node, data, source), IRunExecutionData (startData, resultData, executionData) | Контекст выполнения с буферами; run_data и execution_data в сериализации/опциях run |
| **n8n** | EngineRequest / EngineResponse (вызов подузлов агентом) | agent_loop: tool_calls → выполнение узлов-инструментов → tool_results → агент |
| **n8n** | rewireGraph для tool-узла, ToolExecutor | Вызов подмножества графа (агент + инструмент) с входами из tool_calls |
| **n8n** | DirectedGraph, findSubgraph, findStartNodes, cleanRunData, recreateNodeExecutionStack | getSubgraph, getStartNodes, mergeRunDataIntoBuffers, buildExecutionPlan |
| **n8n** | pinData (подмена выхода узла), runData (результаты по узлам) | pin_data в run; буферы движка = runData, сериализация в чекпоинт |
| **n8n** | waitingExecution при множественных входах | Политика «все входящие порты готовы» в планировщике |
| **n8n** | executionOrder (v1 unshift vs push) | run option execution_order: "lifo" | "fifo" |
| **LangChain** | create_agent, middleware, checkpointer, store, interrupt | Фабрика гиперграфа, доп. узлы/опции run ([LANGUAGE_MODELS.md](LANGUAGE_MODELS.md)) |
| **LangGraph** | StateGraph, Pregel, каналы, checkpoint, interrupt, resume | Буферы движка, итеративная фаза, сериализация, interrupt_on ([LANGUAGE_MODELS.md](LANGUAGE_MODELS.md)) |
| **DeepAgents** | Глубокий агент, инструменты, субагенты, skills, memory, backend, interrupt_on, MCP | Гиперграф с Helper/Conjector, вложенные графы, конфиг backend_ref, interrupt_on ([LANGUAGE_MODELS.md](LANGUAGE_MODELS.md)) |

Итог: **полный паритет** означает, что каждая перечисленная возможность выражается гиперграфом, узлами, буферами, опциями run или сериализацией и выполняется движком Иггдрасиль.

---

## 3. Маппинг компонентов агентной системы на роли узлов-задач

### 3.1 Сводная таблица

| Компонент | Роль в каноне | block_type (пример) | Назначение |
|-----------|----------------|----------------------|------------|
| Агент (LLM + парсинг tool_calls) | **Backbone** (или составной узел) | `backbone/llm`, агентный узел | (prompt, context, tool_results?) → response, tool_calls?; движок по tool_calls выполняет инструменты и повторяет вызов |
| Инструмент (tool) | Узел-задача (часто **Helper**) | `helper/api`, `helper/filesystem_read`, `helper/todo`, `helper/execute` | Входы = аргументы инструмента; выход = результат; tool_id ↔ node_id в конфиге агента |
| Триггер (запуск по событию) | Узел без обязательных входов | `trigger/schedule`, `trigger/webhook` | Инициирует выполнение; выход — данные события; планировщик помещает в начальную очередь |
| Поллер | Узел с периодическим/ручным опросом | `helper/poll`, специализированный тип | По запросу возвращает данные; при ручном run — один вызов |
| Вебхук | Узел, активируемый HTTP-запросом | `trigger/webhook` или `helper/webhook` | Внешний запрос → данные на вход узла → узел в очередь |
| Формат промпта, память, навыки | **Conjector**, **Helper** | `conjector/prompt_format`, `helper/memory` | system_prompt, подстановка memory/skills в контекст агента |
| Планировщик (что выполнять следующим) | Движок | — | По графу и буферам определяет следующую задачу (stack-based или по готовности портов); не отдельный узел |
| Состояние (runData, каналы) | Буферы движка | — | Запись выхода узла в буфер; чтение по входящим рёбрам; сериализация в чекпоинт |

### 3.2 Узел-агент

- Внутри: подграф или монолитный блок «LLM + парсинг tool_calls» (см. [LANGUAGE_MODELS.md](LANGUAGE_MODELS.md) §3.9). Входы: prompt/context, опционально tool_results. Выходы: response (text), tool_calls (список {tool_id, arguments}). При наличии tool_calls движок выполняет agent_loop: по таблице tool_id → node_id вызывает узлы-инструменты, собирает tool_results, снова вызывает агента.
- Конфиг графа: metadata tool_id_to_node_id (или tool_id → (node_id, port_mapping)); max_agent_steps в run или metadata.

### 3.3 Узлы-инструменты

- Каждый инструмент — узел с портами = аргументам инструмента и одним выходом (результат). block_type задаёт реализацию (HTTP, файлы, todo, execute, вызов субагента). backend_ref, sandbox_ref в конфиге узла для доступа к хранилищу или среде выполнения.

### 3.4 Триггеры, поллеры, вебхуки

- **Триггер:** узел без входящих рёбер (или с опциональным событием); при run в режиме «от триггера» стартовая очередь = [триггер(ы)]. Результат триггера — данные на исходящих портах, передаваемые по рёбрам следующим узлам.
- **Поллер:** узел с методом run, возвращающим данные опроса; при ручном запуске вызывается один раз.
- **Вебхук:** узел, регистрирующий HTTP-эндпоинт; при входящем запросе тело/заголовки подаются на вход узла и выполнение ставится в очередь (внешняя система вызывает run с соответствующими inputs).

### 3.5 Субагент

- Отдельный гиперграф (собственный агент с инструментами). В графе главного агента инструмент «task» — узел, который при run вызывает run(subgraph, inputs); inputs формируются из arguments вызова task; выход узла = выход субагента (tool_result). Реализация: либо узел-задача с конфигом subgraph_ref, либо воркфлоу-уровень, где один из узлов — целый гиперграф.

---

## 4. Типичные графы (топологии)

### 4.1 Простой агент с инструментами

Один узел-агент (Backbone = LLM + tool_calls) и несколько узлов-инструментов. Внешние входы: prompt/context. Движок: agent_loop до отсутствия tool_calls или max_steps. Выход: финальный response. Соответствует create_agent (LangChain).

### 4.2 Глубокий агент (DeepAgents-подобный)

Как §4.1, но инструменты: todo, read_file, write_file, edit_file, ls, glob, grep, execute, task (субагент). Conjector собирает system_prompt с memory/skills из Helper. Backend для файловых/execute узлов в конфиге. interrupt_on: при необходимости приостановка перед указанным инструментом.

### 4.3 Workflow в стиле n8n (линейная цепочка)

Узлы: Trigger → NodeA → NodeB → NodeC. Нет цикла агента. Выполнение: старт с Trigger, затем по одному узлу в порядке рёбер (или по стеку: Trigger → A → B → C). Данные передаются по буферам (runData по узлам). Поддержка pinData для подмены выхода любого узла.

### 4.4 Workflow с ветвлением и слиянием

Несколько исходящих рёбер от узла A (например, по условию) → узлы B1, B2; оба входят в C (множественные входы). Планировщик добавляет C в очередь только после появления данных от B1 и B2 (waitingExecution-аналог). Execution order (FIFO/LIFO) задаёт порядок обхода при нескольких готовых узлах.

### 4.5 Частичный запуск (re-run от узла)

Граф уже выполнялся; сохранён runData. Пользователь меняет параметры узла X и запрашивает выполнение от X до destination (например, до конца). Движок: подграф от X (или от триггера) до destination; dirty_nodes = [X]; start_nodes = узлы, от которых нужно пересчитать; буферы инициализируются из runData; очередь = start_nodes; выполнение до destination. Аналог runPartialWorkflow2.

### 4.6 Агент с субагентами

Главный граф: узел-агент + инструменты, среди которых узел «task» с subgraph_ref = субагент. Субагент — отдельный гиперграф (агент + свои инструменты). При tool_call task(name, input) движок выполняет run(субагент, { input }) и возвращает результат как tool_result.

---

## 5. Форматы данных на портах и в состоянии

### 5.1 runData (результаты по узлам)

- Структура: по node_id (или имени) — массив выполнений; каждый элемент: данные выхода (по портам), source (откуда пришли входы), executionIndex, startTime, и т.д. В Иггдрасиль — буферы движка по (node_id, port_name); при сериализации чекпоинта сохраняется как run_data для resume и частичного запуска.

### 5.2 pinData

- Словарь node_id → значение (или словарь порт → значение) для подстановки выхода узла без выполнения. Передаётся в run(..., pin_data=...) или в metadata; движок при планировании узла проверяет pin_data и при наличии пишет значение в буфер выхода и не вызывает блок.

### 5.3 tool_calls и tool_results

- **tool_calls:** список { id, type: "function", function: { name (tool_id), arguments } }. Выход узла-агента. **tool_results:** список { tool_call_id, content }. Передаются агенту на следующем шаге agent_loop. Формат совместим с OpenAI-подобным и LangChain/LangGraph.

### 5.4 Сообщения и контекст (чат-агент)

- messages: список { role, content [, tool_calls, tool_call_id ] }. Conjector формирует из них промпт для Backbone; после генерации ответ и tool_calls добавляются в историю; при resume (thread_id, checkpoint_id) история восстанавливается из чекпоинта.

### 5.5 IExecuteData-подобный элемент (n8n)

- { node, data: ITaskDataConnections (по типу связи массив массивов элементов), source }. В Иггдрасиль контекст выполнения узла содержит доступ к входящим данным по портам (аналог data) и к метаданным источника (source) для трассировки и отображения в UI.

---

## 6. Цикл выполнения и агентный цикл в движке

### 6.1 Workflow-выполнение (stack-based или по готовности)

- **Инициализация:** при run(hypergraph, inputs, start_node_id?, ...) начальная очередь заполняется стартовыми узлами (триггеры или start_node_id). Буферы внешних входов заполняются из inputs.
- **Шаг:** планировщик извлекает узел из очереди; движок выполняет узел (чтение из буферов по входящим рёбрам, вызов block.run); запись выхода в буферы по исходящим рёбрам; добавление в очередь узлов, для которых стали готовы все входы (если ещё не выполнены). Повтор до пустой очереди или до достижения destination_node_id.
- **Множественные входы:** узел попадает в очередь только когда для каждого входящего ребра в буфере есть значение (политика «все готовы»).

### 6.2 Агентный цикл (agent_loop)

- При выполнении узла-агента движок проверяет выход на наличие tool_calls. Если есть: по tool_id → node_id выполняются узлы-инструменты (входы = arguments); результаты собираются в tool_results; узел-агент вызывается снова с обновлёнными входами (context + tool_results). Цикл до отсутствия tool_calls или до max_agent_steps. Детали — [03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) §7.3, [LANGUAGE_MODELS.md](LANGUAGE_MODELS.md) §6.2.

### 6.3 Частичный запуск и resume

- **Частичный run:** run(..., run_data=..., dirty_node_ids=..., destination_node_id=...): загрузка run_data в буферы; построение подграфа; определение start_nodes; формирование очереди; выполнение до destination. **Resume:** run(..., thread_id=..., checkpoint_id=...) загружает чекпоинт (в т.ч. run_data, очередь при необходимости) и продолжает выполнение.

### 6.4 Прерывания (interrupt_on)

- metadata или run options: interrupt_on = [node_id] или [tool_id]. Движок при достижении указанного узла/инструмента приостанавливает выполнение и возвращает частичный результат и флаг «приостановлено»; следующий run с теми же thread_id/checkpoint_id продолжает с сохранённого состояния.

---

## 7. Конфигурация и вызов run

### 7.1 Конфиг гиперграфа (агент / workflow)

- **nodes:** node_id, block_type, config (для агента — checkpoint_ref, tool_id_to_node_id; для инструментов — backend_ref, sandbox_ref; для триггеров — тип и параметры).
- **edges:** (source_node, source_port) → (target_node, target_port).
- **exposed_inputs / exposed_outputs:** граница графа для run(inputs).
- **metadata:** execution_order ("fifo" | "lifo"), max_agent_steps, interrupt_on, checkpointer_ref, tool_id_to_node_id (для агента), destination_node_id (для частичного run по умолчанию — опционально).

### 7.2 Конфиг узлов (типичные поля)

- **Узел-агент:** как Backbone LLM ([LANGUAGE_MODELS.md](LANGUAGE_MODELS.md) §7.2) + tool_id_to_node_id, max_agent_steps.
- **Узел-инструмент:** аргументы по портам (имена соответствуют arguments инструмента); backend_ref, sandbox_ref; для task — subgraph_ref.
- **Триггер/вебхук:** тип (schedule, webhook), параметры (cron, path, method).

### 7.3 Вызов run

`outputs = run(hypergraph, inputs, **options)`

- **inputs:** по exposed_inputs (prompt, messages, или данные для триггера).
- **options:** start_node_id, destination_node_id, pin_data, run_data, dirty_node_ids, execution_order, thread_id, checkpoint_id, interrupt_on, max_agent_steps, stream, и др.
- **outputs:** по exposed_outputs после завершения (или частичный результат при interrupt).

---

## 8. Обучение

Агентные узлы (LLM как Backbone) обучаются по [LANGUAGE_MODELS.md](LANGUAGE_MODELS.md) §8 и [DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) §6. Инструменты без обучаемых параметров не требуют обучения; при необходимости дообучение инструмента (например, классификатор) задаётся конфигом trainable узлов и loss по выходам. Сериализация state_dict по node_id/block_id — по [SERIALIZATION.md](SERIALIZATION.md).

---

## 9. Сериализация

- **Конфиг гиперграфа:** nodes, edges, exposed_inputs, exposed_outputs, metadata (в т.ч. tool_id_to_node_id, interrupt_on, execution_order). Версионирование schema_version.
- **Чекпоинт:** state_dict по узлам (обучаемые параметры); опционально **run_data** (буферы по узлам/портам) и **execution_state** (очередь, текущий узел, шаг agent_loop) для resume и частичного запуска.
- **thread_id, checkpoint_id:** в run options для загрузки/сохранения чекпоинта сессии; при resume загружается последний чекпоинт по thread_id и выполнение продолжается.

По [SERIALIZATION.md](SERIALIZATION.md) — единый механизм конфиг + чекпоинт; для агентных workflow дополнительно сохраняются run_data и при необходимости execution_state.

---

## 10. Использование на уровне воркфлоу

- Цепочки: гиперграф «триггер + обработка» → гиперграф «агент с инструментами» → гиперграф «уведомление». Выход одного графа — вход следующего.
- Субагент как узел воркфлоу: один из узлов воркфлоу — целый гиперграф (субагент); вызов = run(subgraph, inputs).
- Импорт/экспорт: конфиг гиперграфа и при необходимости run_data в формате, совместимом с редакторами workflow (например, экспорт в формат n8n-подобный для визуализации).

---

## 11. Граничные случаи

- **Пустой tool_calls:** агент вернул ответ без вызовов — agent_loop завершается, выход = response.
- **Циклы в графе:** при stack-based выполнении цикл (A → B → C → A) обрабатывается конвенцией: узел может выполняться повторно при новом появлении данных на входах (runIndex в n8n); планировщик может ограничивать число проходов по циклу или требовать явной семантики итерации.
- **Триггеров несколько:** начальная очередь = все триггерные узлы; выполнение идёт от каждого (параллельно или последовательно по execution_order).
- **destination_node при частичном run:** выполнение останавливается после выполнения указанного узла; последующие узлы не выполняются.
- **pinData и runData вместе:** при частичном run run_data заполняет буферы; pin_data переопределяет выход указанных узлов и при наличии отменяет их выполнение.

---

## 12. Связь с другими документами

- [LANGUAGE_MODELS.md](LANGUAGE_MODELS.md) — LLM как ядро агента, генерация, tool_calls, create_agent, middleware, StateGraph, DeepAgents; данный документ фокусируется на **оркестрации агентных систем** и паритете с n8n/workflow-моделью.
- [03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) §7.3 — agent_loop в движке.
- [HYPERGRAPH_ENGINE.md](HYPERGRAPH_ENGINE.md) — планировщик, итеративная фаза, буферы; расширение на stack-based и частичный запуск.
- [02_ABSTRACT_TASK_NODES.md](02_ABSTRACT_TASK_NODES.md) — контракты портов ролей; здесь — маппинг на агента, инструменты, триггеры.
- [SERIALIZATION.md](SERIALIZATION.md) — конфиг + чекпоинт, run_data и execution_state для resume.
- [DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) — обзор поддержки доменов, развёртывание.

---

**Итог.** Документ задаёт **полную спецификацию поддержки агентных систем** в фреймворке Иггдрасиль и **полный паритет по возможностям** с n8n (workflow, стек выполнения, частичный запуск, pinData, runData, типы узлов, EngineRequest/EngineResponse), LangChain (create_agent, middleware), LangGraph (StateGraph, Pregel, каналы, чекпоинт), DeepAgents (глубокий агент, инструменты, субагенты, backend, interrupt_on). Паритет достигается **полной интеграцией аналогичного функционала на базе движка гиперграфа и узлов-задач Иггдрасиль** — без использования кода или исполнения этих платформ как основы. Все перечисленные возможности (оркестрация по стеку, частичный run, агентный цикл, триггеры/вебхуки, состояние и чекпоинт, прерывания и resume) выражаются конфигом гиперграфа, узлами и опциями run и выполняются движком фреймворка.
