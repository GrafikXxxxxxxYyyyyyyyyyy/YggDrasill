# Документация проекта Иггдрасиль — новый канон

**Назначение:** единая документация и **канон системы**, выстроенный на основе **текущей философии** ([../philosophy/](../philosophy/)). Здесь одновременно живут:

- целевая архитектурная карта проекта;
- честная карта того, что уже реализовано в текущем runtime;
- разграничение stable и experimental API.

**Якорь:** философия проекта (Часть 1, 2, 3) — неизменяемый ориентир. Канон и схема ей соответствуют; при противоречии пересматриваются канон и реализация, а не философия.

**Язык:** русский.

---

## Что здесь

| Документ / раздел | Содержание |
|-------------------|------------|
| **[README.md](README.md)** (этот файл) | Точка входа: назначение папки, навигация по канону. |
| **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** | Честная карта текущего состояния: что уже реализовано, что пока является каноном, а что остаётся экспериментальным. |
| **[API_SURFACE.md](API_SURFACE.md)** | Разделение stable / experimental API и рекомендуемых слоёв использования. |
| **[DIFFUSION_STATUS.md](DIFFUSION_STATUS.md)** | Честная карта поддерживаемого diffusion surface и его ограничений. |
| **[STAGE_WORLD_UNIVERSE_READINESS_REVIEW.md](STAGE_WORLD_UNIVERSE_READINESS_REVIEW.md)** | Отдельный readiness review для верхних уровней перед началом их реализации. |
| **[STAGE_WORLD_UNIVERSE_PROGRAM.md](STAGE_WORLD_UNIVERSE_PROGRAM.md)** | Следующая отдельная программа работ по `Stage -> World -> Universe`. |
| **[CANON.md](CANON.md)** | Состав канона, ключевые принципы, список всех документов. |
| **[Scheme.md](Scheme.md)** | Полная схема архитектуры — от фундамента до вселенной; уровни, контракты, цикл мира (три стадии). |
| **[docs/](docs/)** | Детальные документы по уровням: [01_FOUNDATION.md](docs/01_FOUNDATION.md) (фундамент), [02_ABSTRACT_TASK_NODES.md](docs/02_ABSTRACT_TASK_NODES.md) (абстрактные узлы-задачи), [03_TASK_HYPERGRAPH.md](docs/03_TASK_HYPERGRAPH.md) (гиперграф задачи), [04_WORKFLOW.md](docs/04_WORKFLOW.md) (воркфлоу), [05_STAGE.md](docs/05_STAGE.md) (стадия), [06_WORLD.md](docs/06_WORLD.md) (мир), [07_UNIVERSE.md](docs/07_UNIVERSE.md) (вселенная — последний уровень). Тематические: [SERIALIZATION.md](docs/SERIALIZATION.md) (сериализация), [GLOSSARY.md](docs/GLOSSARY.md) (глоссарий), [HYPERGRAPH_ENGINE.md](docs/HYPERGRAPH_ENGINE.md) (гиперграфовый движок), [DOMAINS_DEPLOYMENT_TRAINING.md](docs/DOMAINS_DEPLOYMENT_TRAINING.md) (домены, развёртывание, обучение), [DIFFUSION_MODELS.md](docs/DIFFUSION_MODELS.md) (поддержка диффузионных моделей — подробно), [LANGUAGE_MODELS.md](docs/LANGUAGE_MODELS.md) (поддержка языковых моделей — подробно), [AGENT_SYSTEMS.md](docs/AGENT_SYSTEMS.md) (поддержка агентных систем — подробно), [WIKI_AND_CONTENT_SOURCES.md](docs/WIKI_AND_CONTENT_SOURCES.md) (вики и внешние источники контента). |

---

## Иерархия канона

1. **Философия** ([../philosophy/](../philosophy/)) — зачем, почему, принципы. Не меняется.
2. **Scheme** — как устроена система в целом (схема, уровни, три стадии мира).
3. **CANON** — состав канона, правила, ссылки на документы.

---

## Как читать документацию

1. **Что уже есть в коде** — [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
2. **Какие API стабильны** — [API_SURFACE.md](API_SURFACE.md).
3. **Понять замысел** — [../philosophy/Philosophy-Part1.md](../philosophy/Philosophy-Part1.md), [Philosophy-Part2.md](../philosophy/Philosophy-Part2.md), [Philosophy-Part3.md](../philosophy/Philosophy-Part3.md).
4. **Увидеть схему целиком** — [Scheme.md](Scheme.md).
5. **Найти нужный канонический документ** — [CANON.md](CANON.md).

## Важная оговорка

`documentation/docs/` описывает не только то, что уже реализовано, но и **целевую архитектуру** проекта. Поэтому нельзя автоматически считать весь канон уже готовой runtime-реальностью: фактический статус фиксируется в [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

Канон выстроен в соответствии с текущей философией (в т.ч. **три стадии** мира: автор, среда, творец), но реализация некоторых верхних уровней всё ещё остаётся будущей программой работ.
