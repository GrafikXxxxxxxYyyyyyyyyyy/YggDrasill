# Статус реализации YggDrasill

Этот документ фиксирует **текущее состояние репозитория** и отделяет:

- что уже существует в коде и тестах;
- что утверждено каноном, но пока не реализовано полностью;
- что доступно только как экспериментальная или доменная надстройка.

Если этот документ расходится с `documentation/docs/`, источником истины о runtime-состоянии считается именно он.

## 1. Реально реализовано сейчас

### 1.1. Базовое ядро

В коде уже существуют и используются:

- `AbstractBaseBlock` как материальное начало: конфиг, идентичность, состояние;
- `AbstractGraphNode` как идеальное начало: порты, позиция в графе, `run()`;
- абстрактные task-node роли в `yggdrasill/task_nodes/abstract.py`;
- исполняемый `Hypergraph`;
- исполняемый `Workflow`;
- validator / planner / executor / edge buffers;
- сериализация `config + checkpoint` для блоков, графов и workflow.

### 1.2. Исполняемая модель

Текущий runtime поддерживает:

- DAG-графы;
- один цикл исполнения с `num_loop_steps`;
- минимальный `agent_loop` для сценария `tool_calls -> tool_results -> повторный вызов`;
- partial run / suspend-resume через `run_data`, `interrupt_on`, `dirty_node_ids`, `destination_node_id`;
- `pin_data` и runtime kwargs на уровне executor;
- `run_stream` для inspection/observability с partial-run parity по `run_data`, `dirty_node_ids`, `destination_node_id`, `skip_node_ids`, но без отдельного suspend-resume контракта.

### 1.3. Доменные слои

В репозитории присутствуют рабочие доменные слои:

- diffusion / diffusers integration;
- минимальный diffusion training subsystem для `SD1.5 LoRA`;
- набор diffusion builders / presets / templates;
- базовые сценарии для agent- и LLM-подобных графов через общий hypergraph engine.

Но важно: наличие доменных документов в `documentation/docs/` **не означает**, что достигнут полный паритет с описанным там замыслом.

## 2. Канонически утверждено, но ещё не доведено до полной реализации

Следующие уровни описаны в каноне, но не являются завершённой production-ready runtime-реальностью в текущем репозитории:

- `Stage`;
- `World`;
- `Universe`;
- полный единый сериализационный слой для всех уровней 1-7;
- полный доменный паритет для diffusion / LLM / agent systems, заявленный в каноне.

Это означает:

- документы по этим уровням задают **целевую архитектуру**;
- кодовая база пока полноценно реализует главным образом уровни foundation -> hypergraph -> workflow.

## 3. Экспериментальное и частично реализованное

### 3.1. Diffusion layer

Diffusion-слой функционален, но остаётся **экспериментальной надстройкой над ядром**:

- API ещё дорабатывается;
- поддержка семейств и сценариев не равна всему объёму ambition-документов;
- ergonomics и high-level shortcuts допускают изменение при очистке архитектуры.
- training surface существует, но пока стабилизирован только как узкий `SD1.5 LoRA` recipe с отдельным trainer path, а не как общий graph-native diffusion training runtime.

### 3.2. Agent semantics

Текущий `agent_loop` следует трактовать как **минимальный tool loop**, а не как завершённый универсальный контракт для всех агентных систем.

Сейчас гарантируется:

- обнаружение agent-loop узла по `metadata["agent_node_ids"]`, `node.is_agent` или `graph_kind="agent"` для backbone-узлов;
- исключение tool-узлов из обычного execution plan и их вызов только через `tool_id_to_node_id`;
- обратная передача `tool_results` как единственного стабильного feedback-канала агенту;
- lenient default для missing tools: неизвестный `tool_id` по умолчанию пропускается; для fail-fast нужно явно задать `metadata["missing_tool_policy"] = "error"`.

Не гарантируется:

- автоматический state-threading сверх исходных входов агента и `tool_results`;
- stable per-tool interrupt/resume внутри `agent_loop`;
- полный parity со stateful agent runtimes из ambition-документов.

### 3.3. Auto-magic API

Автосвязывание по именам портов, builders и template APIs существуют ради удобства, но относятся к слою **ergonomic / partially experimental API**, а не к самому каноническому low-level ядру.

## 4. Как читать документацию правильно

Рекомендуемый порядок чтения:

1. `philosophy/` — зачем существует система.
2. `documentation/IMPLEMENTATION_STATUS.md` — что уже есть в runtime.
3. `documentation/API_SURFACE.md` — какие API считать stable, а какие experimental.
4. `documentation/Scheme.md` и `documentation/CANON.md` — целевая архитектурная карта.
5. `documentation/docs/` — детальные канонические спецификации по уровням и доменам.

## 5. Что считать стабильным сегодня

Относительно стабильными стоит считать:

- low-level block / node contracts;
- `Hypergraph` как основной исполняемый уровень;
- `Workflow` как hypergraph-of-hypergraphs;
- базовые save/load операции;
- executor surface для ядра.

Остальное следует считать либо evolving, либо experimental до отдельной стабилизации.
