# Diffusion status and support map

Этот документ описывает не амбиции diffusion-канона, а **реально поддерживаемую поверхность** текущего репозитория.

## Базовый принцип

Diffusion-слой является **opt-in addon** над ядром:

- импорт `yggdrasill` не должен менять поведение core runtime;
- diffusion-функции подключаются явно через `yggdrasill.integrations.diffusers`;
- рекомендуемый высокоуровневый запуск diffusion-графов: `run_diffusion(graph, ...)`.

## Что поддерживается лучше всего

Наиболее зрелые сценарии в текущем репозитории:

- SD1.5 graph assembly и builders;
- SDXL graph assembly и builders;
- общая diffusion ergonomics вокруг `run_diffusion`;
- adapters: ControlNet, IP-Adapter, LoRA в текущем test-covered surface;
- family-aware `LoRA training` subsystem через `yggdrasill.integrations.diffusers.training`;
- diffusion builders / presets для smoke и development use.

## Training subsystem

Текущая training-поддержка уже отделена от inference-cycle и теперь организована как
**generic diffusers-family LoRA training layer** поверх training registry/contracts.

Поддерживаемая поверхность (stable-ориентир):

- **Engine:** `yggdrasill.engine.planner.build_training_plan` (рядом с `build_plan`), `run(..., run_mode="train")` для полного цикла обучения на графе с `metadata['training']`.
- **Шаблоны обучения (уровень 3):** `DiffusionGraphBuilder.from_template(..., task="train")`, `list_training_templates()` / `TRAINING_GRAPH_TEMPLATES` в `integrations.diffusers.training`.
- **Уровень 2:** `DiffusionGraphBuilder.build_lora_training_hypergraph(...)` (делегирует в `build_diffusion_lora_training_hypergraph`).

Legacy convenience (по-прежнему на модуле `integrations.diffusers.training`, но не в `__all__`):

- `train_diffusion_lora(config=TrainingConfig(...))`
- `train_sd15_lora(...)`, `train_sdxl_lora(...)`, `train_flux_lora(...)`
- registry-owned training dispatch:
  - backbone/component layout
  - conditioning builder
  - latent representation/objective
  - LoRA target attachment
  - export strategy

Поддерживаемые family/task proof points:

- `SD1.5`
  - `task="text2img" | "img2img" | "inpaint"`
  - local folder datasets (`image + .txt`) и manifest/Hugging Face dataset paths
  - `UNet LoRA` по умолчанию, optional `train_text_encoder=True`
- `SDXL`
  - `task="text2img" | "img2img" | "inpaint" | "refiner"`
  - dual tokenizer / dual text encoder path
  - pooled embeddings + `added_cond_kwargs` (`text_embeds`, `time_ids`)
  - optional `train_text_encoder` и `train_text_encoder_2`
- `FLUX`
  - текущий training proof point: `task="text2img"`
  - transformer backbone вместо `UNet`
  - packed latent representation + FLUX-specific text conditioning/export path

Общая поддержка включает:

- registry-driven trainer shell вместо hardcoded `sd15/sdxl` dispatch
- family-pluggable checkpoint/export path
- diffusers-compatible LoRA export contract для `SD1.5`, `SDXL` и `FLUX`
- compatibility wrappers: текущие `train_sd15_lora(...)` и `train_sdxl_lora(...)` работают поверх generic core

Что всё ещё вне поддерживаемой поверхности:

- `VAE training`
- `ControlNet training`
- distributed / multi-host training surface

**Graph-native training loop:** полный цикл — `engine.run(..., run_mode="train")`; шаг — `run_training_step` /
`build_diffusion_lora_training_hypergraph` (forward до loss-узла → backward в исполнителе → узлы `training/optim_step` / `training/lr_scheduler_step`).
Чекпоинты могут включать `training_graph.json` (подпись плана) рядом с `trainer_state.pt`.

Важно:

- training subsystem не использует inference denoising-cycle как train-loop; отдельный training-гиперграф описывает loss + пост-backward фазу;
- `run_diffusion(...)` и builders остаются inference-oriented слоем;
- training-export совместим с текущей LoRA inference-loading surface для `sd15`, `sdxl` и proof-point `flux`.

## Частично поддерживается / experimental

Следующие зоны остаются evolving:

- FLUX high-level presets;
- часть replacement / template ergonomics;
- неявная автоматизация вокруг builder auto-connect;
- сценарии, где поведение зависит от конкретных ограничений diffusers-семейства.

## Что нельзя трактовать как production guarantee

Нельзя автоматически считать production-гарантированными:

- полный паритет со всеми diffusion ambition-docs;
- одинаковую зрелость SD1.5, SDXL и FLUX;
- стабильность всех convenience shortcuts;
- полное покрытие всех комбинаций adapters / inpaint / control branches.

## Рекомендуемый способ использования

### Stable-er path

Если важна архитектурная прозрачность:

- собирать граф явно;
- использовать `Hypergraph` / `Workflow` напрямую;
- вызывать diffusion через `run_diffusion(...)`;
- считать builders удобным слоем поверх core, а не единственной истиной.

### High-level path

Если нужен быстрый старт:

- использовать `DiffusionGraphBuilder`;
- использовать templates / presets;
- воспринимать этот слой как более удобный, но менее стабильный.

## Текущее практическое разграничение

- Core runtime: stable-first.
- Diffusion addon: powerful, but still experimental in parts.
- FLUX family: especially worth treating as evolving surface.

## Примеры и notebooks

Текущие notebooks в `examples/` относятся именно к diffusion addon, а не к stable-core API:

- они предполагают подготовленное окружение (`torch`, `diffusers`, часто CUDA/GPU);
- они могут зависеть от внешних model repos и сетевого доступа;
- их следует читать как exploratory / experimental guidance, а не как production guarantee.
