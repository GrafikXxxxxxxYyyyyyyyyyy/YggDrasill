# Поддержка диффузионных моделей

**Назначение:** глубокое и детальное описание того, **как использовать фреймворк Иггдрасиль для работы с диффузионными моделями**. Документ задаёт маппинг компонентов диффузионного пайплайна на роли узлов-задач, типичные графы (text-to-image, inpainting, upscale, image-to-video), форматы данных на портах, цикл деноизинга и интеграцию с движком, конфигурацию, обучение (LoRA, ControlNet, full fine-tune), сериализацию и использование на уровне воркфлоу. Канонический источник для реализации и использования диффузии в рамках фреймворка.

**Связь с каноном:** [01_FOUNDATION.md](01_FOUNDATION.md) — Block, Node, порты, run; [02_ABSTRACT_TASK_NODES.md](02_ABSTRACT_TASK_NODES.md) — семь ролей и контракты портов; [03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) — гиперграф задачи, циклы, run; [HYPERGRAPH_ENGINE.md](HYPERGRAPH_ENGINE.md) — планировщик, итеративная фаза; [DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) — обзор поддержки доменов.

**Язык:** русский.

**Паритет и превосходство по удобству.** Решения на базе Иггдрасиль должны не просто обеспечивать **паритет по функциональности и мощности** со всеми перечисленными state-of-the-art решениями (в данном домене — Diffusers и аналоги), **но и** давать пользователю **более удобный, простой и интуитивно понятный интерфейс взаимодействия**: любое диффузионное решение (text-to-image, inpainting, upscale, image-to-video и т.д.) должно быть собираемо **в 3–5 строк кода** при сохранении возможности глубочайшей кастомизации. Это достигается продуманной логикой устройства фреймворка и разумной автоматизацией (фабрики гиперграфов, разумные умолчания, единый контракт run), без ущерба для расширяемости и полного контроля над каждым узлом и параметром.

---

## 1. Место диффузионных моделей в фреймворке

### 1.1 Единая онтология

Диффузионные модели **не вводят отдельного уровня или отдельной онтологии**. Они реализуются **теми же сущностями**, что и остальные домены: узлы-задачи (роли Backbone, Conjector, Inner Module, Outer Module, Converter, Injector, Helper), гиперграф задачи, движок, порты и run. Один гиперграф задачи = **одна диффузионная задача** (text-to-image, upscale, inpainting и т.д.); комбинирование нескольких таких задач (например, text-to-image → upscale) выполняется на уровне **воркфлоу**, где узлами служат целые гиперграфы.

### 1.2 Что такое «диффузионная задача» в каноне

**Диффузионная задача** — это задача генерации или преобразования данных (изображение, видео, латенты), в которой центральным элементом является **итеративный процесс деноизинга**: начальное состояние (часто — шум) последовательно обновляется **K шагов** по правилу, задаваемому моделью (UNet, DiT и т.д.) и схемой дискретизации (DDIM, Euler, flow matching). Условие (текст, класс, изображение) может влиять на генерацию через кондиционирование. Вход и выход задачи могут быть в латентном пространстве или в пространстве пикселей; преобразование задаётся конвертерами (VAE encode/decode, токенизатор текста).

В рамках фреймворка эта задача **полностью укладывается** в гиперграф: цикл Backbone ↔ Inner Module (K итераций), Conjector (условие), Outer Module (начальный шум, расписание), Converter (текст→токены, токены→эмбеддинг; латенты→пиксели). Движок строит итеративный план и выполняет цикл без знания о том, что это «диффузия».

### 1.3 Изоляция семейств в коде интеграции

Реализации под конкретные линейки моделей живут в отдельных пакетах: `yggdrasill/integrations/diffusers/sd15/`, `sdxl/`, `flux/` и т.д. **Запрещены импорты между пакетами семейств** (например, `sdxl` не импортирует `sd15`). Так можно удалить или подключать семейство целиком, не таща чужой код.

Если у двух семейств совпадает алгоритм (типичный LDM-цикл: `set_timesteps` / `scheduler.step`, инициализация латентов), допускается **явное дублирование** исходного кода в пакете семейства; в шапке модуля фиксируется, что логика сознательно продублирована и при изменении контракта планировщика или latent init её при необходимости нужно синхронизировать вручную.

Папка `integrations/diffusers/common/` содержит **вспомогательные утилиты** (изображения, CFG, merge residual и т.п.), а не целые семейства; узлы семейства не выносятся в `common/` ради связи двух линеек друг с другом.

**AnimateDiff (реализация в репозитории):** графы с латентами `(B, C, F, H, W)`, компоненты `sd15.motionadapter` / `sdxl.motionadapter`, шаблоны `sd15_animatediff_*` и `sdxl_animatediff_text2img`, вспомогательные режимы FreeNoise в `common/animatediff_extras.py`. Актуальная матрица сценариев — в `documentation/DIFFUSION_STATUS.md`.

---

## 2. Что должен уметь фреймворк (паритет с Diffusers v0.37.0)

Ниже перечислено **всё**, что предоставляет библиотека Hugging Face Diffusers (v0.37.0) и что фреймворк Иггдрасиль должен уметь поддерживать и выражать через гиперграфы задач, блоки и движок. Это целевой объём функционала для реализации диффузионной поддержки.

**Важно:** паритет с Diffusers достигается **не за счёт использования решений или кода Diffusers**, а за счёт **полной интеграции аналогичного функционала в рамках фреймворка Иггдрасиль** — собственные реализации блоков (Backbone, Inner Module, Conjector, Converter и т.д.), движок, сериализация и API принадлежат фреймворку; совместимость по возможностям и форматам (например, загрузка весов из общепринятых форматов) допускается, но исполнение и архитектура остаются в границах Иггдрасиль.

**Источник:** полный анализ репозитория [huggingface/diffusers](https://github.com/huggingface/diffusers) (ветка/тег v0.37.0): структура `src/diffusers` (pipelines, schedulers, models, loaders, guiders, callbacks, image_processor, video_processor, optimization, configuration_utils, modular_pipelines, hooks, quantizers, experimental, free_init_utils, free_noise_utils).

---

### 2.1 Задачи и типы пайплайнов (Pipeline tasks)

Фреймворк должен позволять собирать гиперграфы под все перечисленные ниже **задачи** и **семейства пайплайнов**. Каждая задача — это один или несколько типовых гиперграфов с соответствующими Backbone, Inner Module (солвер), Conjector, Converter, Outer Module.

**Базовые диффузионные пайплайны:**

- **DDPM** — безусловная генерация изображений (шум → изображение).
- **DDIM** — то же с неявной схемой.
- **DiT** — диффузия на базе трансформера (класс-условие).
- **Latent Diffusion (LDM)** — текст-в-изображение, супер-разрешение; латентное пространство.
- **Dance Diffusion** — безусловная генерация аудио.
- **Consistency Models** — одношаговые/малошаговые consistency-модели.

**Stable Diffusion (1.x):**

- Text-to-Image, Image-to-Image, Inpainting, Instruct Pix2Pix, Depth2Img, Image Variation, Latent Upscale, Upscale (ESRGAN-подобный), UnCLIP (текст + приоритет), LDM3D (3D), Panorama, Safe (цензура), SAG, Attend-and-Excite, DiffEdit, GLIGEN, PAG (Prompt Adapter Guidance), варианты с ControlNet и T2I-Adapter.
- Онлайн-экспорт: **ONNX** (StableDiffusionOnnxPipeline, OnnxStableDiffusionPipeline, Img2Img, Inpaint, Upscale).

**Stable Diffusion XL:**

- Text-to-Image, Img2Img, Inpaint, Instruct Pix2Pix, ControlNet (в т.ч. Union), PAG (в т.ч. с ControlNet).

**Stable Diffusion 3:**

- Text-to-Image, Img2Img, Inpaint, ControlNet, PAG (SD3PAG, SD3PAGImg2Img).

**Flux / Flux2:**

- FluxPipeline (T2I), FluxImg2Img, FluxInpaint, FluxFill, FluxControl*, FluxPriorRedux, FluxKontext/KontextInpaint; Flux2Pipeline, Flux2Klein; поддержка ControlNet, LoRA, IP-Adapter.

**Kandinsky (1, 2.2, 3, 5):**

- Kandinsky: Prior + Decoder (T2I, Img2Img, Inpaint), отдельно Prior pipeline.
- Kandinsky 2.2: T2I, Img2Img, Inpaint, ControlNet (Img2Img).
- Kandinsky 3: T2I, Img2Img.
- Kandinsky 5: T2I, T2V, I2I, I2V.

**DeepFloyd IF:**

- IFPipeline (T2I), IFImg2Img, IFInpainting, IFSuperResolution, каскады супер-разрешения.

**unCLIP (DALL·E 2–стиль):**

- UnCLIPPipeline (текст + приоритет), UnCLIPImageVariation.

**Остальные семейства изображений:**

- **Amused** — T2I, Img2Img, Inpaint.
- **Bria / BriaFibo** — T2I, Edit.
- **Chroma** — T2I, Img2Img, Inpaint.
- **PixArt (Alpha/Sigma)** — T2I; PAG.
- **Sana** — T2I, Sprint (быстрый), ControlNet, Sprint Img2Img; PAG.
- **Stable Cascade** — Prior + Decoder (каскад).
- **Wuerstchen** — Prior + Decoder.
- **HunyuanImage** — T2I, Refiner; HunyuanDiT (T2I, PAG).
- **QwenImage** — T2I, Img2Img, Inpaint, Edit, EditPlus, EditInpaint, ControlNet, ControlNetInpaint, Layered.
- **Z Image** — T2I, Img2Img, Inpaint, Omni, ControlNet, ControlNetInpaint.
- **Lumina / Lumina2** — T2I.
- **PRX, Omnigen, OvisImage, HiDreamImage, GlmImage, LongCatImage** — T2I / Edit.
- **BLIP Diffusion, Paint-by-Example, PIA** — кондиционирование по изображению/примеру.
- **Kolors** — T2I, Img2Img (с sentencepiece).
- **ConsisID** — консистентная идентичность (требует OpenCV).
- **Visual Cloze** — генерация/дополнение.
- **Lucy, ChronoEdit** — редактирование.
- **Marigold** — depth, normals, intrinsics.
- **Shap-E** — 3D (T2I, Img2Img).
- **Semantic Stable Diffusion, LEdits++ (SD/SDXL)** — семантика / редактирование.
- **Stable Diffusion LDM3D** — 3D.

**Видео:**

- **Stable Video Diffusion** — image-to-video.
- **AnimateDiff** — анимация (T2V, с ControlNet, Sparse ControlNet, Video2Video, SDXL).
- **CogVideoX** — T2V, I2V, V2V, FunControl.
- **CogView3+, CogView4** — T2I/T2V, Control.
- **Hunyuan Video (1.0, 1.5)** — T2V, I2V, Skyreels, Framepack.
- **I2VGen-XL** — image-to-video.
- **LTX / LTX2** — видео, Condition, Latent Upsample, I2V, мульти-промпт.
- **Latte** — видео.
- **Sana Video** — T2V, I2V.
- **WAN** — T2V, I2V, V2V, VACE, Animate.
- **SkyReels V2** — T2V, I2V, V2V, Diffusion Forcing.
- **Mochi** — видео.
- **EasyAnimate** — T2V, Inpaint, Control.
- **Helios** — T2V, Pyramid.
- **Kandinsky 5** — T2V, I2V.
- **Text-to-Video (legacy)** — TextToVideoSDPipeline, TextToVideoZero, VideoToVideoSD.
- **Cosmos** — Text-to-World, Video-to-World, Transfer, 2.5 PredictBase/Transfer, 2TextToImage.

**Аудио:**

- **AudioLDM, AudioLDM2** — текст/условие → аудио.
- **MusicLDM** — музыка.
- **Stable Audio** — аудио-диффузия (DiT).

**Специализированные:**

- **T2I-Adapter** — Stable Diffusion / SDXL с адаптерами по карте (скетч, глубина и т.д.).
- **ControlNet** — SD, SDXL (в т.ч. Union), SD3, HunyuanDiT, ControlNetXS; PAG-комбинации.
- **ControlNet XS** — облегчённый ControlNet для SD/SDXL.
- **PAG (Prompt Adapter Guidance)** — стабильная диффузия, SDXL, SD3, PixArt, Sana, HunyuanDiT, Kolors, AnimateDiff, в т.ч. с ControlNet и Inpaint.
- **Aura Flow** — flow-based генерация.
- **Allegro** — 3D (молекулы/материи).
- **UniDiffuser** — мультимодальная генерация (текст + изображение в одном пайплайне; UniDiffuserPipeline, ImageTextPipelineOutput).

**Модульная сборка (Modular Pipelines):**

- **ModularPipeline, ComponentsManager** — сборка пайплайна из подменяемых компонентов (в Diffusers: Flux, Flux2, SDXL, WAN, QwenImage, Z Image). В Иггдрасиль — гиперграф с заменяемыми узлами и единым контрактом портов; состав задаётся конфигом.

**Бэкенд Flax (опционально):**

- **FlaxDiffusionPipeline, FlaxStableDiffusionPipeline, FlaxStableDiffusionImg2ImgPipeline, FlaxStableDiffusionInpaintPipeline, FlaxStableDiffusionXLPipeline, FlaxStableDiffusionControlNetPipeline** — для полного паритета при поддержке Flax/JAX как альтернативного бэкенда к PyTorch.

**Авто-подбор пайплайна:**

- **AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting** — выбор пайплайна по задаче и доступным компонентам (реализуется на уровне воркфлоу или фабрики гиперграфов).

Итог: фреймворк должен позволять описать и запускать **все перечисленные типы задач** в виде гиперграфов (один гиперграф — одна задача; комбинации — воркфлоу).

---

### 2.2 Солверы (Schedulers) — Inner Module

Фреймворк должен поддерживать **все** нижеперечисленные типы солверов как реализации **Inner Module** (шаг перехода latent + timestep + pred → next_latent, next_timestep). Замена солвера не должна менять топологию графа — только тип блока Inner Module.

**Дискретные (основные):**

- **DDIMScheduler** — DDIM.
- **DDPMScheduler** — DDPM.
- **DDIMInverseScheduler** — обратный DDIM (для inversion).
- **DDIMParallelScheduler, DDPMParallelScheduler** — параллельные схемы.
- **PNDMScheduler** — Pseudo Numerical Methods.
- **LMSDiscreteScheduler** — LMS (требует scipy).
- **EulerDiscreteScheduler, EulerAncestralDiscreteScheduler** — Euler.
- **HeunDiscreteScheduler** — Heun.
- **KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler** — K-DPM-2.
- **DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler** — DPM-Solver.
- **DPMSolverMultistepInverseScheduler** — обратный DPM-Solver.
- **EDMEulerScheduler, EDMDPMSolverMultistepScheduler** — EDM-семейство.
- **DEISMultistepScheduler** — DEIS.
- **UniPCMultistepScheduler** — UniPC.
- **IPNDMScheduler** — improved PNDM.
- **LCMScheduler** — Latent Consistency.
- **FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler, FlowMatchLCMScheduler** — flow matching.
- **TCDScheduler** — Consistency Distillation.
- **SASolverScheduler** — SA-Solver.
- **RePaintScheduler** — inpainting.
- **VQDiffusionScheduler** — VQ-Diffusion.
- **UnCLIPScheduler** — для unCLIP.
- **AmusedScheduler** — Amused.
- **DDPMWuerstchenScheduler** — Wuerstchen.
- **ConsistencyDecoderScheduler** — Consistency Decoder.
- **CMStochasticIterativeScheduler** — Consistency Models (stochastic iterative).
- **SCMScheduler** — SCM.
- **HeliosScheduler, HeliosDMDScheduler** — Helios.
- **LTXEulerAncestralRFScheduler** — LTX.
- **CogVideoXDDIMScheduler, CogVideoXDPMScheduler** — CogVideoX.

**Дополнительные (опциональные зависимости):**

- **CosineDPMSolverMultistepScheduler** — cosine DPM (требует torchsde).
- **DPMSolverSDEScheduler** — DPM-SDE (требует torchsde).
- **ScoreSdeVeScheduler, KarrasVeScheduler, ScoreSdeVpScheduler** — deprecated, но могут требоваться для воспроизводимости.

**Flax (если поддерживается бэкенд):** FlaxDDIM, FlaxDDPMScheduler, FlaxDPMSolverMultistepScheduler, FlaxEulerDiscreteScheduler, FlaxKarrasVeScheduler, FlaxLMSDiscreteScheduler, FlaxPNDMScheduler, FlaxScoreSdeVeScheduler.

В каноне Иггдрасиль каждый из них — вариант **Inner Module** с конфигом (num_train_timesteps, beta_schedule, prediction_type, и т.д.); движок вызывает один шаг step(latent, timestep, pred) → (next_latent, next_timestep).

---

### 2.3 Модели (Backbone, Converter, Conjector, Inner Module)

**Backbone (ядро предсказания):**

- **UNet2DModel, UNet2DConditionModel** — 2D UNet (безусловный и условный).
- **UNet3DConditionModel** — 3D (видео).
- **UNet1DModel** — 1D (аудио).
- **UNetSpatioTemporalConditionModel** — пространственно-временной.
- **I2VGenXLUNet, Kandinsky3UNet** — специализированные UNet.
- **StableCascadeUNet** — Stable Cascade.
- **UNetMotionModel, MotionAdapter** — AnimateDiff motion.
- **UVit2DModel** — UViT.
- **Transformer2DModel** — базовый 2D трансформер.
- **DiTTransformer2DModel** — DiT.
- **FluxTransformer2DModel, Flux2Transformer2DModel** — Flux.
- **SD3Transformer2DModel** — Stable Diffusion 3.
- **PixArtTransformer2DModel** — PixArt.
- **HunyuanDiT2DModel, HunyuanImageTransformer2DModel** — Hunyuan image.
- **HunyuanVideoTransformer3DModel, HunyuanVideo15Transformer3DModel, HunyuanVideoFramepackTransformer3DModel** — Hunyuan video.
- **LatteTransformer3DModel, LTXVideoTransformer3DModel, LTX2VideoTransformer3DModel** — видео-трансформеры.
- **CogVideoXTransformer3DModel, ConsisIDTransformer3DModel** — CogVideoX, ConsisID.
- **Kandinsky5Transformer3DModel** — Kandinsky 5.
- **SanaTransformer2DModel, SanaVideoTransformer3DModel** — Sana.
- **CosmosTransformer3DModel** — Cosmos.
- **Остальные трансформеры** (Bria, BriaFibo, Chroma, ChronoEdit, EasyAnimate, GlmImage, Helios, LuminaNextDiT2D, Mochi, OmniGen, OvisImage, PRX, QwenImage, SkyReelsV2, Wan, ZImage, AuraFlow, CogView3+/4, Dual, HiDreamImage, LongCatImage, Lumina2, Prior, StableAudioDiT, T5FilmDecoder и т.д.) — все должны быть выразимы как Backbone с соответствующими входами/выходами (latent, timestep, condition → pred).

**VAE / Autoencoder (Converter):**

- **AutoencoderKL** — стандартный KL-VAE (SD, SDXL и др.).
- **AsymmetricAutoencoderKL, AutoencoderDC, AutoencoderRAE, AutoencoderTiny** — варианты VAE.
- **AutoencoderKLAllegro, AutoencoderKLCogVideoX, AutoencoderKLCosmos, AutoencoderKLFlux2** — под конкретные пайплайны.
- **AutoencoderKLHunyuanImage, AutoencoderKLHunyuanImageRefiner, AutoencoderKLHunyuanVideo, AutoencoderKLHunyuanVideo15** — Hunyuan.
- **AutoencoderKLLTXVideo, AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio** — LTX.
- **AutoencoderKLMagvit, AutoencoderKLMochi, AutoencoderKLQwenImage** — Magvit, Mochi, Qwen.
- **AutoencoderKLTemporalDecoder, AutoencoderKLWan** — временной декодер, WAN.
- **AutoencoderOobleck, ConsistencyDecoderVAE** — Oobleck, Consistency Decoder.
- **VQModel** — VQ-VAE.

**ControlNet (Inner Module или Injector по конфигу):**

- **ControlNetModel, MultiControlNetModel** — классический ControlNet.
- **ControlNetUnionModel, MultiControlNetUnionModel** — объединение карт.
- **FluxControlNetModel, FluxMultiControlNetModel** — Flux.
- **SD3ControlNetModel, SD3MultiControlNetModel** — SD3.
- **HunyuanDiT2DControlNetModel, HunyuanDiT2DMultiControlNetModel** — Hunyuan DiT.
- **QwenImageControlNetModel, QwenImageMultiControlNetModel** — Qwen.
- **SanaControlNetModel, CosmosControlNetModel** — Sana, Cosmos.
- **SparseControlNetModel** — разрежённый контроль.
- **ControlNetXSAdapter, UNetControlNetXSModel** — ControlNet XS.
- **ZImageControlNetModel** — Z Image.

**Адаптеры (T2I-Adapter и др.):**

- **T2IAdapter, MultiAdapter** — карты-адаптеры (скетч, глубина и т.д.) как дополнительный вход в цикл (Inner Module или Conjector по реализации).

Все перечисленные модели должны быть выразимы как блоки с ролями Backbone, Converter, Conjector или Inner Module и корректными портами.

---

### 2.4 Загрузчики и адаптеры (Loaders)

Фреймворк должен поддерживать загрузку и применение следующих механизмов; они отображаются на **Injector** (LoRA, встраивание весов), **Conjector** (IP-Adapter — условие по изображению) или на конфиг/run (загрузка из одного файла).

**LoRA:**

- Загрузка LoRA для: Stable Diffusion, SDXL, SD3, Flux, Flux2, Amused, AuraFlow, CogVideoX, CogView4, HunyuanVideo, Kandinsky, HiDreamImage, SkyReelsV2, QwenImage, ZImage, Lumina2, Wan, Helios, Mochi, LTX, LTX2, Sana.
- Поддержка **нескольких LoRA** с весами и масштабами.
- **PEFT** (PeftAdapterMixin) — универсальная интеграция с библиотекой PEFT.

**IP-Adapter:**

- **IPAdapterMixin** — базовый (SD/SDXL и др.).
- **FluxIPAdapterMixin, SD3IPAdapterMixin** — для Flux и SD3.
- **ModularIPAdapterMixin** — модульный вариант.
- Семантика: изображение → condition для Backbone (в каноне — **Conjector**).

**Textual Inversion:**

- **TextualInversionLoaderMixin** — загрузка токенов-эмбеддингов и подстановка в текст/эмбеддинги (влияет на Conjector или на подготовку condition).

**Single-file и оригинальные форматы:**

- **FromSingleFileMixin** — загрузка пайплайна из одного .safetensors/.ckpt файла.
- **FromOriginalModelMixin** — загрузка из оригинальных форматов (ComfyUI, A1111 и т.д.).
- **UNet2DConditionLoadersMixin, FluxTransformer2DLoadersMixin, SD3Transformer2DLoadersMixin** — загрузка весов в UNet/трансформер из разных форматов.

Реализация в Иггдрасиль: LoRA/Textual Inversion — через блоки Injector/Conjector и единый механизм загрузки весов (сериализация по block_id); single-file — через фабрику гиперграфов или Helper, который разбирает файл и заполняет конфиг/чекпоинт.

---

### 2.5 Guiders (направление генерации)

Diffusers предоставляет несколько **guiders** (классов, управляющих тем, как условие влияет на предсказание). В каноне они могут быть реализованы внутри Backbone (два прохода + комбинация), через отдельный узел между Conjector и Backbone, или через конфиг CFG/guidance_scale.

- **ClassifierFreeGuidance** — классический CFG (условный и безусловный предсказания, масштаб).
- **ClassifierFreeZeroStarGuidance** — вариант с zero-star трюком.
- **AdaptiveProjectedGuidance (APG), AdaptiveProjectedGuidanceMix** — адаптивная проекция.
- **FrequencyDecoupledGuidance** — развязка по частотам.
- **MagnitudeAwareGuidance** — учёт величины.
- **PerturbedAttentionGuidance** — возмущённая attention.
- **SkipLayerGuidance** — пропуск слоёв.
- **SmoothedEnergyGuidance** — сглаженная энергия.
- **TangentialClassifierFreeGuidance** — тангенциальный CFG.
- **AutoGuidance** — авто-выбор типа guidance.

Фреймворк должен позволять задавать тип guidance и его параметры (например, guidance_scale) в конфиге гиперграфа или в опциях run и реализовывать выбранную схему в узлах (Backbone/Conjector или отдельный узел-агрегатор pred).

---

### 2.6 Callbacks (обратные вызовы по шагам)

- **PipelineCallback** — базовый интерфейс: вызов на каждом шаге цикла с (pipeline, step_index, timestep, callback_kwargs).
- **MultiPipelineCallbacks** — композиция нескольких колбэков.
- **SDCFGCutoffCallback, SDXLCFGCutoffCallback, SDXLControlnetCFGCutoffCallback, SD3CFGCutoffCallback** — отключение CFG после заданного шага (cutoff_step_ratio / cutoff_step_index).
- **IPAdapterScaleCutoffCallback** — обнуление масштаба IP-Adapter после шага.

В Иггдрасиль: callbacks передаются в **run(..., callbacks=...)** и вызываются движком на каждой итерации цикла (после Backbone и/или Inner Module); сигнатура должна быть совместима с передачей step_index, timestep и текущих тензоров (latent, pred и т.д.) в callback_kwargs.

---

### 2.7 Обработка изображений и видео (Image/Video Processor)

- **Image processor** — препроцессинг/постпроцессинг изображений (resize, crop, normalize, форматы для CLIP, VAE, ControlNet и т.д.). В каноне это часть блоков Converter или Helper (например, подготовка входа для Conjector или VAE).
- **Video processor** — то же для видео (кадры, размерности, батчи по времени). Должен поддерживаться в видео-гиперграфах (Converter/Helper).
- **Free Init / Free Noise** (free_init_utils, free_noise_utils) — техники модификации начального латента или шума в цикле (например FreeU и аналоги). Выражаются через Outer Module (начальное состояние) или через узел/параметр в цикле; фреймворк должен допускать подстановку таких правил без смены топологии.

Фреймворк должен допускать подключение таких процессоров в узлах (например, converter/preprocess, helper/postprocess) без жёсткой привязки к одному формату.

---

### 2.8 Конфигурация и оптимизация

- **ConfigMixin, register_to_config** — сохранение/загрузка конфигов компонентов (в Иггдрасиль уже есть to_config/from_config для блоков и гиперграфов).
- **Загрузка из Hub и локальных путей** — from_pretrained(id_or_path), save_pretrained(path); поддержка model_index.json (или эквивалента в конфиге гиперграфа) для состава компонентов.
- **Оптимизация** — FP16/BF16, компиляция (torch.compile), расчёт на CPU/CUDA/MPS, offload (CPU offload для экономии VRAM). Должны быть выразимы через опции run или конфиг устройства/точности для узлов.
- **Hooks (хуки)** — в Diffusers: кэширование (faster_cache, first_block_cache, mag_cache, taylorseer_cache), group_offloading, context_parallel, layer_skip, layerwise_casting, pyramid_attention_broadcast, smoothed_energy_guidance_utils. Фреймворк должен допускать аналоги: кэширование промежуточных активаций, перенос компонентов на CPU между шагами, пропуск слоёв, побайтовое приведение по слоям — через опции run или конфиг блоков/движка.
- **Quantizers (квантизация весов)** — в Diffusers: quantizers (base, auto, bitsandbytes, gguf, modelopt, quanto, torchao, quantization_config, pipe_quant_config). Поддержка загрузки и запуска квантизованных моделей (int8, int4, GGUF и т.д.) для экономии памяти; в Иггдрасиль — через конфиг блоков и загрузчики весов (Converter/Helper или отдельный механизм квантизации).
- **Зависимости** — опциональные (transformers, accelerate, safetensors, onnx, sentencepiece, opencv, librosa, note_seq и т.д.) не должны быть обязательными для ядра; доменные блоки подключают их по необходимости.
- **Experimental** — в Diffusers: experimental/rl (RL для диффузии) и прочие экспериментальные API. Для полного паритета фреймворк должен допускать реализацию таких сценариев поверх того же движка и графов (обучение с RL, кастомные шаги и т.д.).

---

### 2.9 Итог по паритету с Diffusers

Фреймворк Иггдрасиль должен уметь:

1. **Собирать гиперграфы** под любую из перечисленных задач (T2I, I2I, Inpaint, T2V, I2V, аудио, 3D, специализированные варианты, UniDiffuser, модульные пайплайны) с нужным Backbone, солвером, VAE, кондиционированием и опциональными ControlNet/Adapter.
2. **Поддерживать все перечисленные солверы** как Inner Module с единым контрактом step(latent, timestep, pred) → (next_latent, next_timestep).
3. **Поддерживать все перечисленные типы моделей** (UNet, DiT, трансформеры, VAE, ControlNet, T2I-Adapter) в виде блоков с ролями Backbone, Converter, Conjector, Inner Module.
4. **Интегрировать LoRA, IP-Adapter, Textual Inversion, single-file и PEFT** через механизмы Injector/Conjector и загрузки весов.
5. **Поддерживать различные стратегии guidance** (CFG и расширения) через конфиг и/или узлы.
6. **Вызывать callbacks** на каждом шаге цикла с передачей step_index, timestep и тензоров.
7. **Включать пре/пост-обработку изображений и видео**, а также техники Free Init/Free Noise в узлы Converter/Helper/Outer Module.
8. **Обеспечивать конфигурируемость и оптимизацию** (точность, устройство, offload, hooks/кэширование, квантизация весов, from_pretrained/save_pretrained) без изменения топологии графа.
9. **Допускать модульную сборку** (подмена компонентов при сохранении контракта) и при необходимости **Flax-бэкенд** и **экспериментальные сценарии** (RL и др.) поверх того же движка.

Реализация может быть поэтапной (сначала ядро задач и солверов, затем расширение по спискам выше), но **целевая спецификация** — полный паритет с возможностями Diffusers v0.37.0 за счёт **собственной реализации аналогичного функционала** в рамках Иггдрасиль (гиперграфы, блоки, движок), а не за счёт вызова или оборачивания библиотеки Diffusers.

---

### 2.10 Проверка полноты паритета (чеклист по структуре Diffusers v0.37.0)

Следующие модули и возможности Diffusers должны быть учтены; ниже — соответствие документу и канону.

| Модуль/возможность Diffusers | Учтено в §2 | Как в Иггдрасиль |
|------------------------------|------------|-------------------|
| **pipelines** (все пайплайны и задачи) | §2.1 | Гиперграфы задач; перечислены все семейства, включая UniDiffuser, модульные, Flax. |
| **schedulers** (все солверы) | §2.2 | Inner Module; перечислены все классы, включая Flax и опциональные. |
| **models** (UNet, трансформеры, VAE, ControlNet, адаптеры) | §2.3 | Backbone, Converter, Conjector, Inner Module. |
| **loaders** (LoRA, IP-Adapter, TI, single-file, PEFT) | §2.4 | Injector/Conjector, загрузка весов, сериализация. |
| **guiders** (CFG, APG и др.) | §2.5 | Конфиг/узлы guidance. |
| **callbacks** | §2.6 | run(..., callbacks=...), вызов на каждом шаге цикла. |
| **image_processor, video_processor** | §2.7 | Converter/Helper. |
| **free_init_utils, free_noise_utils** | §2.7 | Outer Module / узел в цикле. |
| **optimization, configuration_utils** | §2.8 | Опции run, конфиг, to_config/from_config. |
| **hooks** (кэш, offload, layer_skip и т.д.) | §2.8 | Опции run или конфиг блоков/движка. |
| **quantizers** (квантизация весов) | §2.8 | Конфиг блоков, загрузчики весов. |
| **modular_pipelines** | §2.1 | Гиперграф с заменяемыми узлами, конфиг состава. |
| **experimental** (в т.ч. RL) | §2.8 | Сценарии поверх того же движка. |
| **commands** (CLI) | — | Реализация поверх API фреймворка (вне ядра). |
| **training_utils** (обучение в Diffusers) | §8 документа | Обучение блоков, trainable, state_dict; детали в [DOMAINS_DEPLOYMENT_TRAINING](DOMAINS_DEPLOYMENT_TRAINING.md). |

Итог: **полный паритет** означает, что для каждой перечисленной сущности Diffusers v0.37.0 в документе указано, как она отображается на гиперграфы, блоки, порты или опции run в Иггдрасиль; при этом паритет обеспечивается **полной интеграцией аналогичного функционала внутри фреймворка Иггдрасиль**, а не использованием решений Diffusers. Реализация покрывает все пункты чеклиста собственной кодовой базой; при появлении новых пайплайнов/моделей в Diffusers их добавляют в соответствующий подраздел §2 и в чеклист.

---

## 3. Маппинг компонентов пайплайна на роли узлов-задач

### 3.1 Сводная таблица

| Компонент диффузионного пайплайна | Роль в каноне | Типичный block_type | Назначение |
|-----------------------------------|---------------|----------------------|------------|
| Модель предсказания шума/скорости (UNet, DiT, трансформер) | **Backbone** | `backbone/unet2d`, `backbone/dit` | Один шаг предсказания: (latent, timestep, condition) → pred (шум ε, x0 или v). |
| Солвер (DDIM, Euler, flow matching) | **Inner Module** | `inner_module/ddim`, `inner_module/euler` | Один шаг перехода: (latent, timestep, pred) → next_latent, next_timestep. Выполняется **внутри цикла** на каждой итерации. |
| Кондиционирование по тексту (CLIP, T5 и т.д.) | **Conjector** | `conjector/clip` | Текст (или token_ids) → condition (эмбеддинг). Стоит **рядом с ядром**, подаёт условие в Backbone. |
| Токенизатор текста | **Converter** | `converter/tokenizer` | Текст → token_ids (для Conjector). |
| Генератор начального шума, расписание таймстепов | **Outer Module** | `outer_module/noise`, `outer_module/schedule` | До цикла: начальный latent (шум), расписание (timesteps). **Вне цикла** — подготовка входа в цикл. |
| VAE (encode/decode) | **Converter** | `converter/vae` | Пиксели ↔ латенты. Encode — до цикла (если вход в pixel-space); decode — после цикла (латент → изображение). |
| ControlNet, адаптеры по карте | **Inner Module** или **Injector** | `inner_module/controlnet`, `injector/control` | ControlNet как отдельный узел в цикле: control map + latent → вклад в pred или в backbone. Либо Injector, встроенный в Backbone (инжекция по карте). |
| LoRA, стиль-инжекция по весам | **Injector** | `injector/lora` | Встраивается **внутрь Backbone**; влияет на генерацию через веса (модификация слоёв, добавление матриц). |
| IP-Adapter (кондиционирование по изображению) | **Conjector** | `conjector/ip_adapter` | Стоит **рядом с ядром**: изображение → condition (эмбеддинг для cross-attention). Подаёт условие в Backbone так же, как CLIP подаёт текстовое условие. |
| Поиск по корпусу для промпта, метрики, загрузка изображений | **Helper** | `helper/rag`, `helper/metrics` | Вспомогательные функции: RAG для подсказок, PSNR/SSIM, загрузка/сохранение. |

### 3.2 Backbone (ядро диффузии)

- **Входы:** `latent` (текущее латентное состояние, тензор), `timestep` (текущий шаг шума, скаляр или тензор), `condition` (эмбеддинг от Conjector; опционально).
- **Выход:** `pred` — предсказание модели (шум ε в параметризации по шуму, или x0, или v в flow matching).
- **Семантика:** один вызов = один forward pass модели. Не выполняет шаг обновления латентов — это делает Inner Module (солвер).
- **Типичные реализации:** UNet 2D (Stable Diffusion, SDXL), DiT (трансформер для диффузии), UNet 3D (видео). Размерности латентов и condition задаются конфигом и должны согласовываться с Outer Module (шум) и Conjector (эмбеддинг).

### 3.3 Inner Module (солвер и контроль внутри цикла)

- **Роль солвера:** на каждой итерации цикла получает от Backbone выход `pred` и текущие `latent`, `timestep`; вычисляет **следующее состояние** `next_latent` и опционально `next_timestep` по выбранной схеме (DDIM, Euler, flow matching).
- **Входы:** `latent`, `timestep`, `pred`; опционально `control` (для ControlNet-подобных).
- **Выходы:** `next_latent`, опционально `next_timestep`.
- **Связь с движком:** цикл в графе — это именно Backbone и Inner Module: на шаге k движок вызывает Backbone(latent_k, timestep_k, condition) → pred_k, затем Inner Module(latent_k, timestep_k, pred_k) → next_latent, next_timestep. На следующей итерации next_latent и next_timestep подаются снова в Backbone. Число итераций K задаётся `num_loop_steps` (конфиг гиперграфа или опция run).
- **ControlNet:** может быть реализован как отдельный Inner Module в цикле (получает control map и latent, выдаёт добавку к pred или модифицированный latent) или как Injector внутри Backbone.

### 3.4 Conjector (кондиционирование)

- **Вход:** текст (prompt), token_ids от Converter, или изображение (для кондиционирования по картинке).
- **Выход:** `condition` — эмбеддинг (или словарь encoder_hidden_states, attention_mask и т.д.) для Backbone.
- **Семантика:** выполняется **один раз до цикла** (или один раз на весь run). Стоит **рядом с ядром** и подаёт условие в Backbone; на каждой итерации цикла Backbone получает один и тот же condition.
- **Типичные реализации:** CLIP text encoder (текст → condition), T5 encoder; **IP-Adapter** (изображение → condition для cross-attention, по сути конжектор: отдельный узел, подающий дополнительное условие в Backbone); опционально два энкодера (CLIP + T5) с объединением по конфигу (два Conjector или один с двумя входами). Стиль-инжекция, реализованная как отдельный источник condition (вход → condition), тоже относится к Conjector; если же она реализована как встраивание весов в слои Backbone — то Injector.

### 3.5 Outer Module (граница цикла)

- **Генератор шума:** выдаёт начальный `latent` (тензор шума) и опционально начальный `timestep` или полное расписание. Выполняется **до входа в цикл**. Размерность латента должна совпадать с ожидаемой Backbone и VAE (если используется латентное пространство).
- **Расписание (schedule):** может быть отдельным узлом или частью генератора шума: выдаёт список/тензор таймстепов для K шагов. Inner Module (солвер) использует это расписание для определения next_timestep на каждой итерации.
- **Связь с графом:** выход Outer Module (initial_latent, initial_timestep или schedule) подаётся на **вход первого шага цикла** (в Backbone и/или Inner Module). После выхода из цикла результат (финальный latent) идёт в Converter (VAE decode) или во внешний выход гиперграфа.

### 3.6 Converter (преобразование форматов)

- **Токенизатор:** текст → token_ids. Выход идёт в Conjector (CLIP и т.д.). Один вызов до цикла.
- **VAE encode:** изображение (пиксели) → латенты. Нужен, если вход задачи — изображение (image-to-image, inpainting с маской в pixel-space). Выполняется до цикла; выход — начальный latent, который может смешиваться с шумом по конфигу (например, inpainting).
- **VAE decode:** латенты → изображение. Выполняется **после выхода из цикла**. Вход — финальный latent с последней итерации; выход — внешний выход гиперграфа (image).

### 3.7 Injector (встраивание в ядро)

- **LoRA:** дообученные веса, встраиваемые в слои Backbone (линейные, attention). Не имеет отдельного «выхода» в графе в простейшем случае — Backbone при run использует свои веса + веса LoRA. Condition для LoRA (например, стиль) опционально подаётся по порту.
- **Стиль-инжекция по весам:** если реализована как модификация внутренних слоёв/активаций Backbone (а не как отдельный узел, подающий condition), относится к Injector.
- В графе Injector может быть зарегистрирован как узел с связью condition → Injector; Backbone при конфигурации указывает, что использует данный Injector (по node_id или block_id). Реализация вызова (вызов инжектора внутри forward Backbone) задаётся конкретной реализацией Backbone и Injector. **Примечание:** IP-Adapter по своей сути — **Conjector** (изображение → condition для Backbone), а не Injector: он подаёт условие в ядро, а не встраивается в его слои.

### 3.8 Helper (вспомогательные функции)

- RAG по корпусу для расширения/улучшения промпта; загрузка и сохранение изображений; вычисление метрик (PSNR, SSIM) после генерации. Подключаются по необходимости; не входят в обязательную цепочку encode → цикл → decode.

### 3.9 Три способа создания диффузионной задачи

Фреймворк предлагает **три уровня API** для сборки диффузионного графа:

**1. Низкоуровневый способ** — ручная сборка Hypergraph. Максимальная гибкость и адаптивность.

- `add_node("id", type="sd15/unet", config={...})` или `add_node("id", node_object)`
- Явное `add_edge(Edge(source, port, target, port))` для каждого ребра
- `expose_input`, `expose_output` для внешних портов
- Загрузка компонентов через `ModelStore.load_components_by_keys()`

Пример: `examples/diffusion_sd15.ipynb` (ячейка «Самый честный способ»).

**2. DiffusionGraphBuilder** — компоненты с автосвязыванием по именам портов.

- `add_component("unet", "sd15.unet", pretrained="runwayml/stable-diffusion-v1-5")` — загрузка и добавление узла
- Рёбра создаются автоматически через `apply_port_name_auto_connect`
- `expose_default_io()` — экспорт стандартных входов/выходов (prompt, output_image)
- `replace_component("unet", "sd15.unet", pretrained=...)` — замена компонента в существующем графе

Пример: `examples/diffusion_sd15.ipynb` (ячейка «Создание через DiffusionGraphBuilder»).

**3. Высокоуровневый способ** — одна строка кода.

- `graph = Hypergraph.from_template("sd15_text2image", device="cuda")`

Шаблоны: `sd15_text2img`, `sd15_img2img`, `sd15_inpaint`, `sdxl_*`, `flux_*`, `flux_controlnet_text2img`. Возвращается полностью собранный и готовый к run граф.

---

## 4. Типичные графы (топологии)

### 4.1 Text-to-image (текст → изображение)

**Поток данных:**

1. Внешний вход: `prompt` (текст).
2. Converter (tokenizer): prompt → token_ids.
3. Conjector (CLIP): token_ids → condition.
4. Outer Module (noise): → initial_latent, расписание (timesteps).
5. **Цикл (K итераций):** Backbone(latent, timestep, condition) → pred; Inner Module (солвер)(latent, timestep, pred) → next_latent, next_timestep. На следующей итерации next_latent и next_timestep становятся latent и timestep.
6. Converter (VAE decode): финальный latent → image.
7. Внешний выход: `image`.

**Узлы:** tokenizer (Converter), clip (Conjector), noise_schedule (Outer Module), unet (Backbone), solver (Inner Module), vae_decode (Converter). Опционально: Injector (LoRA), второй Conjector (T5 для длинного текста), Conjector (IP-Adapter для кондиционирования по изображению).

**Внешние порты гиперграфа:** вход `prompt` (на порт tokenizer или clip, в зависимости от того, где объявлен expose_input); выход `image` (с порта vae_decode).

### 4.2 Image-to-image (стилизация, перевод домена)

Отличие от text-to-image: есть входное изображение. Варианты:

- **Латентная инициализация от VAE encode:** внешний вход `image` → VAE encode → latent_init; затем latent_init смешивается с шумом по формуле (например, alpha * latent_init + (1-alpha) * noise) в Outer Module или в отдельном узле; результат — initial_latent для цикла. Остальное как в text-to-image (condition от текста, цикл, VAE decode).
- **Inpainting:** входы `image`, `mask` (область, которую нужно перегенерировать). VAE encode(image) → latent; маска в латентном пространстве или в pixel-space задаёт, где подставлять шум; цикл заполняет только замаскированную область (семантика задаётся реализацией Backbone/Inner Module и передачей mask по порту).

**Узлы:** добавляются VAE encode, опционально узел смешивания латентов; остальное как в §4.1.

### 4.3 Upscale (увеличение разрешения)

Может быть реализовано как отдельный гиперграф: вход — изображение низкого разрешения (или латенты), выход — изображение высокого разрешения. Внутри — модель upscale (диффузионная или не диффузионная). Если диффузионная — та же схема (condition от низкоразрешённого изображения через Conjector/Converter, цикл, decode). На уровне **воркфлоу** типична цепочка: гиперграф text-to-image → гиперграф upscale (вход первого — prompt, выход первого — image; вход второго — image, выход второго — image_hr).

### 4.4 Image-to-video

Один гиперграф: вход — image (опционально + text), выход — video. Внутри: Backbone — 3D UNet или видео-трансформер; цикл деноизинга в латентном пространстве видео; Conjector — условие от изображения и/или текста; Outer Module — начальный шум для видео-латентов; Converter — видео-VAE decode. Топология та же, что text-to-image; меняются размерности и типы блоков.

### 4.5 Варианты без цикла (одношаговая модель)

Если используется одношаговая модель (например, один шаг диффузии или GAN-подобная генерация), цикл может отсутствовать: Backbone вызывается один раз, Inner Module (солвер) не нужен или выполняет один шаг. Граф остаётся DAG; движок строит план без итеративной фазы. Конфиг такого гиперграфа не задаёт num_loop_steps или задаёт 1.

---

## 5. Форматы данных на портах

### 5.1 Латенты (latent)

- **Тип:** тензор (например, `[B, C, H, W]` в NCHW). Размерности C, H, W задаются конфигом модели (VAE, Backbone) и должны быть согласованы между узлами.
- **Семантика:** в латентном пространстве диффузии (например, Stable Diffusion: C=4, H=height/8, W=width/8). На входе цикла — начальный шум или результат VAE encode; на выходе цикла — деноизированный латент для VAE decode.

### 5.2 Таймстеп (timestep)

- **Тип:** скаляр или одномерный тензор (batch). Обычно нормализован (0…1 или 0…1000 в зависимости от расписания).
- **Семантика:** текущий шаг шума; солвер по нему и по pred вычисляет next_latent и next_timestep. Расписание (список таймстепов для K шагов) формируется Outer Module или задаётся конфигом (linear, cosine и т.д.).

### 5.3 Условие (condition)

- **Тип:** тензор эмбеддингов (например, `[B, seq_len, dim]`) или словарь (encoder_hidden_states, attention_mask, и т.д.) по соглашению реализации.
- **Семантика:** текст/класс/изображение в виде представления для cross-attention или concat в Backbone.

### 5.4 Предсказание (pred)

- **Тип:** тензор той же формы, что latent (или x0/v в flow matching). Семантика зависит от параметризации: шум ε, x0 или v. Солвер использует pred для вычисления next_latent по выбранной схеме.

### 5.5 Изображение (image)

- **Тип:** тензор пикселей (например, `[B, 3, H, W]` в диапазоне 0…1 или нормализованный). На границах гиперграфа (внешние входы/выходы) может быть представление в формате, удобном для приложения (например, PIL, numpy); конвертация при необходимости в блоке Converter или в обвязке run.

---

## 6. Цикл деноизинга и движок

### 6.1 Как движок видит цикл

Гиперграф содержит рёбра: выход Inner Module (next_latent, next_timestep) снова подаётся на вход Backbone (latent, timestep) и на вход Inner Module (latent, timestep для следующего шага). Это образует **цикл** в графе. Планировщик движка ([HYPERGRAPH_ENGINE.md](HYPERGRAPH_ENGINE.md)) обнаруживает сильно связную компоненту (SCC) с циклом и строит **итеративный план**: начальная фаза (узлы до цикла: tokenizer, Conjector, Outer Module) → **циклическая фаза** (повторить K раз: Backbone, Inner Module) → конечная фаза (узлы после цикла: VAE decode).

### 6.2 Задание числа шагов K

- **num_loop_steps:** задаётся в метаданных гиперграфа (metadata) или в опциях вызова `run(hypergraph, inputs, num_loop_steps=K)`. Движок выполняет циклическую фазу ровно K раз; после K итераций выход Inner Module (next_latent, next_timestep) передаётся в конечную фазу (VAE decode).
- **Расписание:** если K не совпадает с длиной расписания (например, расписание задано для 50 шагов, а K=20), реализация Outer Module или Inner Module может подвыборку/интерполяцию расписания (по конфигу или по соглашению).

### 6.3 Инициализация буферов для цикла

Перед первой итерацией цикла буферы входов Backbone и Inner Module заполняются выходами узлов начальной фазы: initial_latent и initial_timestep (или первый таймстеп из расписания) от Outer Module; condition от Conjector. На каждой итерации после run(Backbone) и run(Inner Module) буферы обновляются: next_latent и next_timestep становятся новыми значениями latent и timestep для следующей итерации.

### 6.4 Выход из цикла

После последней (K-й) итерации значение next_latent из Inner Module передаётся по ребру в Converter (VAE decode) или во внешний выход. Движок не различает «последний шаг» в логике узлов — просто после K повторов выполняется конечная фаза, и в буферы конечной фазы попадают выходы последней итерации цикла.

---

## 7. Конфигурация и вызов run

### 7.1 Конфиг гиперграфа задачи (диффузия)

- **nodes:** для каждого узла — node_id, block_type (например, `backbone/unet2d`, `inner_module/ddim`, `conjector/clip`), config (параметры блока: размерности, пути к весам, тип расписания, num_train_timesteps и т.д.).
- **edges:** пары (source_node, source_port, target_node, target_port) в соответствии с топологией (§4).
- **exposed_inputs:** например, (tokenizer, input) или (clip, input) с name `prompt`; для image-to-image — ещё (vae_encode, image) с name `image`.
- **exposed_outputs:** (vae_decode, output) с name `image`.
- **metadata:** num_loop_steps (K), опционально seed, guidance_scale (если реализовано в узлах через конфиг).

### 7.2 Конфиг блоков (типичные поля)

- **Backbone (UNet/DiT):** in_channels, out_channels, block_out_channels, attention_head_dim, num_layers, sample_size; путь к чекпоинту или pretrained.
- **Inner Module (DDIM):** num_train_timesteps, beta_schedule, clip_sample, set_alpha_to_one; опционально prediction_type (epsilon, v_prediction, sample).
- **Conjector (CLIP):** max_length, projection_dim; путь к чекпоинту.
- **Converter (VAE):** scaling_factor, latent_channels; путь к чекпоинту.
- **Outer Module (noise):** latent_channels, height, width (или из конфига Backbone); schedule_type (linear, cosine).

### 7.3 Вызов run

`outputs = run(hypergraph, inputs, num_loop_steps=K, **options)`

- **inputs:** словарь по внешним входам, например `{"prompt": "a cat"}` или `{"prompt": "...", "image": tensor}` для image-to-image.
- **outputs:** словарь по внешним выходам, например `{"image": tensor}`.
- **options:** device, seed, callbacks; num_loop_steps может переопределять значение из metadata.

---

## 8. Обучение (LoRA, ControlNet, full fine-tune)

### 8.1 Принцип

Все обучаемые параметры принадлежат **блокам** (узлам-задачам). Гиперграф агрегирует state_dict по узлам; сериализация и загрузка чекпоинта — по [SERIALIZATION.md](SERIALIZATION.md). Обучение на уровне гиперграфа задаётся флагом training и выбором trainable узлов ([DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) §6).

### 8.2 LoRA

- LoRA реализуется как **Injector** (injector/lora), встроенный в Backbone. Обучаемые параметры — только веса LoRA; Backbone может быть заморожен (trainable=False). При backward градиенты проходят через LoRA в слой Backbone; оптимизатор обновляет только параметры LoRA. state_dict LoRA сохраняется в чекпоинте отдельно или вместе с state_dict Backbone (по соглашению: один узел с двумя блоками или два узла с привязкой).

### 8.3 ControlNet

- Если ControlNet — отдельный **Inner Module** с весами, его параметры помечаются trainable; при обучении обновляются веса ControlNet, Backbone может быть заморожен. Если ControlNet реализован как Injector внутри Backbone — аналогично LoRA.

### 8.4 Full fine-tune Backbone

- Backbone (и при необходимости Conjector, VAE) помечаются trainable; оптимизатор получает параметры из hypergraph.trainable_parameters(). Чекпоинт сохраняет полный state_dict всех trainable узлов.

### 8.5 Дедупликация

- Если один и тот же блок (например, один и тот же Backbone) используется в нескольких узлах по block_id, в чекпоинте хранится **один** state_dict на block_id; при загрузке он подставляется во все узлы с этим block_id ([SERIALIZATION.md](SERIALIZATION.md), [03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) §9).

---

## 9. Сериализация

### 9.1 Конфиг + чекпоинт

- **Конфиг гиперграфа:** to_config() возвращает структуру (nodes, edges, exposed_inputs, exposed_outputs, graph_id, metadata, schema_version). В nodes для каждого узла — node_id, block_type, config; веса в конфиг не входят.
- **Чекпоинт:** state_dict() гиперграфа — агрегат state_dict всех узлов (по node_id; при общем block_id — одна запись на block_id). Формат чекпоинта (файл, директория) по [SERIALIZATION.md](SERIALIZATION.md).
- **Воспроизводимость:** при одинаковых конфиге, чекпоинте, inputs и num_loop_steps результат run должен быть воспроизводим (при фиксированном seed в Outer Module или в run options).

### 9.2 Сохранение и загрузка

- save(path) / save_config(path), save_checkpoint(path); load(path), load_from_checkpoint(path). Отдельное сохранение только диффузионного гиперграфа не отличается от общего правила сериализации гиперграфа задачи ([03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) §9, [SERIALIZATION.md](SERIALIZATION.md)).

---

## 10. Использование на уровне воркфлоу

### 10.1 Цепочки задач

- **Text-to-image → Upscale:** воркфлоу из двух гиперграфов. Узел 1 — гиперграф text-to-image (внешний вход prompt, внешний выход image). Узел 2 — гиперграф upscale (внешний вход image, внешний выход image_hr). Ребро воркфлоу: (graph_1, image) → (graph_2, image). Внешние порты воркфлоу: вход prompt (expose от graph_1), выход image_hr (expose от graph_2).
- **Text-to-image → Image-to-video:** аналогично; выход первого гиперграфа (image) подаётся на вход второго (image + опционально prompt).

### 10.2 Опции run на уровне воркфлоу

- num_loop_steps может относиться к разным гиперграфам: по соглашению конфига воркфлоу или опций run можно передавать num_loop_steps по graph_id (например, num_loop_steps: {graph_1: 30, graph_2: 1}). Движок воркфлоу при вызове run(hypergraph_1, …) передаёт соответствующий num_loop_steps в опциях.

### 10.3 Обучение в воркфлоу

- Тренируемыми могут быть один или несколько гиперграфов в воркфлоу; остальные заморожены. state_dict воркфлоу — объединение state_dict всех вложенных гиперграфов; чекпоинт воркфлоу сохраняет/загружает их по graph_id ([04_WORKFLOW.md](04_WORKFLOW.md), [DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md)).

---

## 11. Примеры конфигов (схема)

### 11.1 Минимальный text-to-image (схема узлов и рёбер)

```
nodes:
  - node_id: tokenizer
    block_type: converter/tokenizer
    config: { vocab_path: "...", max_length: 77 }
  - node_id: clip
    block_type: conjector/clip
    config: { pretrained: "..." }
  - node_id: noise
    block_type: outer_module/noise
    config: { latent_channels: 4, height: 64, width: 64 }
  - node_id: unet
    block_type: backbone/unet2d
    config: { in_channels: 4, out_channels: 4, ... }
  - node_id: solver
    block_type: inner_module/ddim
    config: { num_train_timesteps: 1000 }
  - node_id: vae_decode
    block_type: converter/vae
    config: { pretrained: "...", mode: decode }

edges:
  - [tokenizer, output, clip, input]
  - [clip, condition, unet, condition]
  - [noise, output, unet, latent]
  - [noise, timestep, unet, timestep]
  - [noise, output, solver, latent]
  - [noise, timestep, solver, timestep]
  - [unet, pred, solver, pred]
  - [solver, next_latent, unet, latent]
  - [solver, next_timestep, unet, timestep]
  - [solver, next_latent, vae_decode, input]
  - [solver, next_timestep, solver, timestep]   # для следующей итерации

exposed_inputs: [{ node_id: tokenizer, port_name: input, name: prompt }]
exposed_outputs: [{ node_id: vae_decode, port_name: output, name: image }]
metadata: { num_loop_steps: 20 }
```

(Примечание: в реальной реализации точные имена портов и способ задания расписания next_timestep между итерациями задаются контрактом Inner Module и движком; схема приведена для иллюстрации топологии.)

### 11.2 Добавление LoRA

- Добавляется узел injector/lora с config (target_modules, rank, path to lora weights). В конфиге Backbone (unet) указывается использование инжектора (lora_node_id или injectors: [lora_node_id]). Condition для LoRA опционально; при обучении trainable только узел lora.

### 11.3 Практические примеры SDXL

В репозитории добавлены отдельные примеры для текущего SDXL-layer:

- `examples/diffusion/sdxl_text2img.py` — high-level factory для `SDXL text2img`.
- `examples/diffusion/sdxl_img2img.py` — high-level factory для `SDXL img2img`.
- `examples/diffusion/sdxl_base_refiner.py` — двухстадийный `base -> refiner` workflow.
- `examples/diffusion/sdxl_manual_graph.py` — ручная сборка SDXL-графа через `ModelStore` и preset builder.

У всех примеров единая задача:

- показать канонический graph-first API для SDXL;
- показать, какие внешние порты и артефакты публикует граф;
- сохранить `config.json` для дальнейшей сериализации и повторного поднятия структуры;
- явно вывести результат `validate(...)`, чтобы пользователь видел текущее состояние графа до запуска.

Минимальный high-level сценарий для SDXL text-to-image:

```python
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes

register_diffusion_nodes()

graph = Hypergraph.from_template(
    "sdxl_text2img",
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    device="cuda",
    torch_dtype="float16",
    config={
        "height": 1024,
        "width": 1024,
        "guidance_scale": 7.5,
        "num_inference_steps": 30,
        "output_type": "pil",
    },
)
```

Минимальный сценарий для `base + refiner`:

```python
from yggdrasill.workflow.workflow import Workflow
from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes

register_diffusion_nodes()

workflow = Workflow.from_template(
    "sdxl_base_refiner",
    base_repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    refiner_repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
    device="cuda",
    torch_dtype="float16",
    denoising_end=0.8,
)
```

**Inpaint по умолчанию.** Вызов `build_sdxl_pipeline(task="inpaint")` без `repo_id` подставляет чекпоинт
`diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (9-канальный UNet с concat маски в Diffusers).
Для остальных задач по-прежнему используется `stabilityai/stable-diffusion-xl-base-1.0`.

**Паритет с SD1.5.** Префикс `build_sdxl_*` и ручной `DiffusionGraphBuilder` для стека SDXL (включая
`try_complete_sdxl_universal_diffusion`) повторяют топологию цикла `latent_init → scheduler_step ↔ unet`,
img2img с `strength`, inpaint: 9 ch — маска на UNet; 4 ch — узел `common/inpaint_latent_blend` после шага
планировщика. При `run(width=…, height=…)` обновляются и поля `original_size` / `target_size` у
`sdxl/added_conditioning`.

---

## 12. Граничные случаи и варианты

### 12.1 Latent-space vs pixel-space

- Если диффузия идёт в пространстве пикселей (например, некоторые модели), VAE encode/decode не используются; Outer Module выдаёт шум в pixel-space, Backbone и Inner Module работают с пикселями. Топология та же; меняются типы блоков и размерности.

### 12.2 Classifier-free guidance (CFG)

- CFG реализуется либо внутри Backbone (два прохода: conditioned и unconditioned, затем комбинация pred), либо через два Conjector (condition и null_condition) и узел комбинирования pred по scale. В каноне не задаётся единственный способ; реализация выбирается в блоках и в конфиге.

### 12.3 Разные солверы и предсказания

- Солвер может ожидать pred как ε, x0 или v; конфиг Inner Module и Backbone должны согласовываться (prediction_type). Расписание (linear, cosine, sigmoid) задаётся в Outer Module или в конфиге Inner Module.

### 12.4 Батчинг

- Все тензоры на портах поддерживают batch-измерение (B). run(hypergraph, inputs) может принимать batch промптов; condition и латенты имеют размерность [B, ...]. Реализации блоков должны корректно обрабатывать batch.

### 12.5 Устройство (device)

- hypergraph.to(device) переносит все блоки на заданное устройство; run(..., device=...) может передаваться в опциях. Согласование device между узлами — ответственность реализации (все на одном device или распределённое выполнение по конфигу).

---

## 13. Связь с другими документами

- **Обзор доменов:** [DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) §3 — краткое описание поддержки диффузии; данный документ расширяет его до полной спецификации использования.
- **Роли узлов:** [02_ABSTRACT_TASK_NODES.md](02_ABSTRACT_TASK_NODES.md) §§4–10 — контракты портов Backbone, Inner Module, Conjector, Outer Module, Converter, Injector; здесь приведён маппинг на конкретные реализации диффузии.
- **Гиперграф и циклы:** [03_TASK_HYPERGRAPH.md](03_TASK_HYPERGRAPH.md) §2.2, §6–7 — одна задача = один гиперграф, цикл и num_loop_steps; [HYPERGRAPH_ENGINE.md](HYPERGRAPH_ENGINE.md) §2, §4 — итеративный план и буферы.
- **Сериализация:** [SERIALIZATION.md](SERIALIZATION.md) — конфиг + чекпоинт, дедупликация по block_id.
- **Обучение и развёртывание:** [DOMAINS_DEPLOYMENT_TRAINING.md](DOMAINS_DEPLOYMENT_TRAINING.md) §5 (API), §6 (обучение), §7 (развёртывание) применимы к диффузионным гиперграфам без изменений.

---

**Итог.** Документ задаёт **полную спецификацию использования фреймворка для диффузионных моделей**: маппинг компонентов пайплайна на роли, типичные топологии графов, форматы данных на портах, цикл деноизинга и интеграцию с движком, конфигурацию, обучение (LoRA, ControlNet, full fine-tune), сериализацию и использование в воркфлоу. Реализация диффузии в коде опирается на этот документ и на общий канон уровней 01–04 и движка.
