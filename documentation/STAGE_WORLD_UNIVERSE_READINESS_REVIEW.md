# Readiness Review: Stage / World / Universe

Этот документ фиксирует, **насколько репозиторий реально готов** к переходу от текущего runtime (`foundation -> hypergraph -> workflow`) к верхним уровням `Stage`, `World`, `Universe`.

Он не запускает реализацию. Его задача — сохранить границу между:

- уже зрелым фундаментом;
- каноническим замыслом верхних уровней;
- тем, что ещё нужно стабилизировать до начала кодирования.

## 1. К чему проект уже готов

Репозиторий уже имеет полезный фундамент для следующего цикла:

- стабильный исполняемый core на уровнях `Hypergraph` и `Workflow`;
- validator / planner / executor / edge buffers;
- config/checkpoint serialization для нижних уровней;
- базовый partial-run / suspend-resume surface;
- более честно очерченные stable vs experimental boundaries в документации.

Это означает, что переход к верхним уровням можно планировать **архитектурно**, не изобретая движок с нуля.

## 2. Что пока не готово как runtime foundation

Несмотря на зрелость ядра, у верхних уровней пока нет зафиксированной runtime-основы в коде:

- в пакете `yggdrasill/` нет реализованных runtime-классов уровня `Stage`, `World`, `Universe`;
- нет test-backed executor contract для `run(stage, state)`, `run(world, state?, action?)`, `run(universe, world_inputs?)`;
- нет отдельного сериализационного слоя для `state`, world content и universe payloads;
- нет стабилизированной схемы инвариантов для `state_input_map`, `state_output_map`, `state_schema`, `payload_spec`, `ether`, entity transfer;
- нет отдельного validation layer для верхних уровней;
- нет policy для checkpoint aggregation и deduplication выше уровня workflow.

Главный вывод: **кодовая база готова к проектированию верхних уровней, но ещё не к немедленной реализации их полного runtime**.

## 3. Readiness по уровням

### 3.1 Stage

`Stage` ближе всего к реализации, потому что канонически это "workflow-of-workflows + state boundary".

Что уже есть:

- `Workflow` как исполняемый уровень;
- единый executor surface;
- сериализация конфигов и checkpoint-агрегации для нижних уровней.

Что ещё нужно зафиксировать до кодирования:

- минимальный runtime shape объекта `state`;
- контракт `state_input_map` и `state_output_map`;
- поведение `can_run(stage, state, context?)`;
- политика мутации state: in-place vs copy-on-write;
- граница между "данные внутри stage" и "state на внешней границе stage";
- validation rules для mapping-ов и required slots.

Readiness verdict:

- `Stage` можно брать первым кандидатом следующей реализации, но только после фиксации state-contract.

### 3.2 World

`World` опирается на `Stage`, поэтому его readiness ниже.

Что уже есть концептуально:

- канонический цикл `author -> environment -> creator`;
- разделение между исполняемой частью и world content;
- общее понимание `state_schema` и первой итерации без готового state.

Что критически не зафиксировано:

- точный runtime contract одного витка мира;
- storage contract для world updates;
- инварианты "полный state" vs "частичный state";
- правила взаимодействия `action`, `context`, `content`, `state`;
- checkpoint format мира как агрегата стадий;
- boundaries для `End World`.

Readiness verdict:

- `World` нельзя реализовывать до тех пор, пока `Stage` не получит рабочий state-boundary contract.

### 3.3 Universe

`Universe` сейчас наиболее далёк от безопасной реализации.

Что уже есть концептуально:

- чёткий канон про worlds-as-nodes;
- `payload_spec`, `ether`, `End World`, entity transfer;
- понимание, что это высший уровень и его нельзя размывать импровизацией.

Что не готово:

- исполнимый contract для universe traversal;
- формальный формат `payload_spec`;
- правила синхронизации и order/topology semantics;
- contract для `ether`;
- entity identity / ownership / round-trip semantics;
- serialization and validation model для universe edges и read-only worlds.

Readiness verdict:

- `Universe` пока годится только для отдельного design cycle, не для implementation-by-creep.

## 4. Главные архитектурные блокеры

До начала кодирования верхних уровней должны быть закрыты следующие блокеры:

- единый `state` contract, не привязанный к одному домену;
- mapping and validation primitives для `state -> workflow inputs -> state`;
- чёткая policy копирования и обновления state;
- serializable distinction между runtime structure, checkpoint и content;
- верхнеуровневый validator surface;
- test strategy для stateful multi-level execution.

## 5. Что нельзя делать преждевременно

Следующие шаги сейчас были бы ошибкой:

- добавлять `Stage`, `World`, `Universe` как thin wrappers без invariants;
- писать code-first реализацию `payload_spec` без отдельного spec document;
- смешивать world content serialization с обычным checkpoint runtime;
- переносить в верхние уровни временные ergonomics shortcuts из diffusion/agent layers;
- пытаться реализовать все три уровня одной волной.

## 6. Рекомендуемые gates перед стартом следующего цикла

Перед началом реализации верхних уровней нужно пройти такие gates:

1. Зафиксировать minimal `state` model и mapping invariants.
2. Согласовать serialization split: runtime structure / checkpoint / content.
3. Утвердить отдельный validator contract для `Stage`.
4. Подготовить test matrix сначала для `Stage`, потом для `World`, потом для `Universe`.
5. Только после этого начинать код уровня `Stage`.

## 7. Итоговый verdict

Текущий репозиторий **готов к следующей архитектурной программе**, но **не готов к безопасной немедленной реализации всех верхних уровней**.

Правильный следующий шаг:

- сначала отдельная программа `Stage-first`;
- затем `World` как следующий цикл;
- и только после этого `Universe`.
