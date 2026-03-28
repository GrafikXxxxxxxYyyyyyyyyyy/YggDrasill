# API Surface: stable vs experimental

Этот документ описывает, на каком слое API находится пользователь и какие границы стабильности действуют сейчас.

## Stable

Следующие части считаются базовым стабильным слоем проекта:

- `yggdrasill.Hypergraph`;
- `yggdrasill.hypergraph.structure.Hypergraph`;
- `yggdrasill.workflow.workflow.Workflow`;
- `AbstractBaseBlock`, `AbstractGraphNode`, базовые task-node абстракции;
- core executor contract: `inputs`, `num_loop_steps`, `seed`, `pin_data`, `run_data`, `dirty_node_ids`, `destination_node_id`, `interrupt_on`, `max_steps`;
- training on the same engine: `run_mode` (`"inference"` \| `"train"`), `build_training_plan` in `yggdrasill.engine.planner` next to `build_plan`;
- diffusion training UX: manual `Hypergraph` + `run(train)` → `DiffusionGraphBuilder.build_lora_training_hypergraph` (уровень 2) → именованные рецепты `from_template("sd15_lora_train" | …)` + `run` (уровень 3);
- config/checkpoint serialization primitives.

Эти API должны оставаться предсказуемыми и документируемыми как основной low-level путь.

## Experimental

Следующие части следует считать evolving / experimental:

- diffusion builders, templates и presets;
- top-level diffusion convenience APIs;
- auto-connect и другие слои неявной автоматизации;
- доменно-специфические ergonomic shortcuts;
- будущие уровни `Stage`, `World`, `Universe`.

Для этих зон допустимы контролируемые breaking changes, если они:

- делают архитектуру честнее;
- убирают скрытые side effects;
- сопровождаются тестами и обновлённой документацией.

К этому же experimental слою сейчас относятся diffusion notebooks и high-level example flows в `examples/`.

## Рекомендуемые слои использования

### 1. Low-level canonical path

Используйте вручную:

- блоки / узлы;
- `Hypergraph`;
- `Workflow`;
- явные рёбра;
- явные exposed inputs/outputs.

Это главный путь для архитектурно прозрачной сборки.

### 2. Builder / factory path

Используйте domain builders, если нужен ускоренный старт, но при этом важно понимать, что builder:

- может автоматически добавлять узлы;
- может автоматически связывать совместимые порты;
- может экспонировать стандартные входы и выходы.

Этот слой удобен, но не должен подменять понимание low-level структуры.

### 3. Template / high-level path

Templates и готовые пресеты предназначены для быстрого старта и smoke-сценариев.

Их следует считать самым высокоуровневым и наименее стабильным слоем API.

## Примеры и notebooks

- `examples/core_hypergraph.py` следует считать рекомендуемым stable-core примером.
- diffusion notebooks в `examples/` следует читать как experimental, environment-bound material.
- notebooks не должны использоваться как источник истины о стабильном API без сверки с `IMPLEMENTATION_STATUS.md` и этим документом.
