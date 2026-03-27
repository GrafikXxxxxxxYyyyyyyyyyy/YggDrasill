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
- минимальный `SD1.5 LoRA training` subsystem через `yggdrasill.integrations.diffusers.training`;
- diffusion builders / presets для smoke и development use.

## Minimal training subsystem

Текущая training-поддержка намеренно узкая:

- реализован только отдельный training path для `SD1.5 LoRA`;
- первый recipe ориентирован на folder dataset (`image + .txt` или `metadata.jsonl` / `metadata.csv`);
- по умолчанию обучается `UNet LoRA`, а `text_encoder LoRA` включается явно;
- `VAE training`, `ControlNet training`, `SDXL training`, `FLUX training` и generic graph-wide training orchestration пока не входят в поддерживаемую поверхность.

Важно:

- training subsystem не использует существующий inference-cycle как основной train-loop;
- `run_diffusion(...)` и builders остаются inference-oriented слоем;
- training-export должен быть совместим с текущей LoRA inference-loading surface.

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
