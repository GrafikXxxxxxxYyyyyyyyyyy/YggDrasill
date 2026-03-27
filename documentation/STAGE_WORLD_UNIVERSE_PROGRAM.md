# Program: Stage / World / Universe

Это **следующая отдельная программа работ** после второй очереди.

Её принцип: не реализовывать верхние уровни "одним прыжком", а двигаться слоями:

`Stage foundations -> Stage runtime -> World foundations -> World runtime -> Universe design prep`

## Волна 1. State contract foundation

### Цель

Зафиксировать единый, сериализуемый и доменно-нейтральный `state` contract, без которого верхние уровни будут рыхлыми.

### Что сделать

- определить minimal shape `state`;
- зафиксировать slot/path addressing rules;
- определить mapping contract для `state_input_map` и `state_output_map`;
- определить policy обновления state: copy-on-write vs in-place;
- зафиксировать required-slots semantics и `can_run`.

### Критерий выхода

- есть отдельная спецификация state-contract;
- по ней можно писать validator и tests, не угадывая semantics на ходу.

## Волна 2. Stage runtime

### Цель

Реализовать `Stage` как первый верхний уровень поверх зрелых `Workflow`.

### Что сделать

- ввести runtime-class уровня `Stage`;
- реализовать `run(stage, state, context?) -> state`;
- реализовать mapping state <-> workflow ports;
- добавить validator и serialization for stage structure;
- покрыть unit/integration tests для single-workflow и multi-workflow stage;
- отдельно проверить skip semantics при `can_run == False`.

### Критерий выхода

- `Stage` реально исполняем и test-backed;
- state boundary больше не является только канонической идеей.

## Волна 3. World foundations

### Цель

Подготовить мир как цикл стадий на уже работающем `Stage`.

### Что сделать

- зафиксировать runtime shape одного world-cycle;
- определить `state_schema`, `content`, `storage`, `action`, `context`;
- определить первую итерацию без предзаданного state;
- определить serialization split для world executable part vs world content;
- подготовить validator rules уровня мира.

### Критерий выхода

- world contract больше не зависит от неявных философских интерпретаций;
- можно безопасно переходить к реализации мира.

## Волна 4. World runtime

### Цель

Реализовать `World` как исполнимый цикл стадий.

### Что сделать

- ввести runtime-class уровня `World`;
- реализовать cycle traversal и `num_steps`/one-pass semantics;
- реализовать storage/content hooks;
- поддержать executable world vs `End World`;
- покрыть tests для first-pass, partial state, full state, content update.

### Критерий выхода

- `World` становится реальным исполняемым уровнем, а не только документом.

## Волна 5. Universe preparation

### Цель

Не реализовывать `Universe` сразу, а довести её до полноценного implementation-ready spec.

### Что сделать

- выделить formal spec для `payload_spec`;
- определить `ether` contract;
- определить entity transfer semantics;
- определить rules для read-only `End World` links;
- определить topology/order model и synchronization points;
- подготовить separate validation and serialization plan.

### Критерий выхода

- `Universe` имеет implementation-ready spec;
- команда понимает, какой минимальный runtime запускать первым, а какие части ещё нельзя брать.

## Общий порядок и ограничения

1. Не начинать `World` до завершения `Stage`.
2. Не начинать `Universe runtime` до завершения отдельного universe-spec cycle.
3. Не смешивать runtime structure, checkpoint и content serialization.
4. Не переносить experimental ergonomics в верхние уровни до фиксации их low-level contracts.

## Короткий итог

Следующая большая программа должна идти так:

1. `state` contract
2. `Stage`
3. `World foundations`
4. `World`
5. `Universe spec`

Только такой порядок сохраняет архитектурную чистоту и не размывает уже стабилизированное ядро.
