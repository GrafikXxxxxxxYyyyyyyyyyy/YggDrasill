# Examples

This directory currently contains two different kinds of material:

- Stable core example: `core_hypergraph.py`
- Experimental diffusion notebooks: `diffusion_sd15.ipynb`, `diffusion_sdxl.ipynb`, `diffusion_train_lora.ipynb`

## Recommended starting point

If you want the most stable and architecture-transparent entrypoint, start with:

```bash
python examples/core_hypergraph.py
```

That example uses only the core `Hypergraph` API and does not require `torch`, `diffusers`, CUDA, or model downloads.

## Diffusion notebooks

The notebooks in this directory are useful as exploratory material, but they should be treated as experimental:

- they depend on the diffusion addon, not just the core runtime;
- they often assume a GPU-ready environment with `torch`, `diffusers`, and enough VRAM;
- they may rely on external model downloads and network access;
- they are not the source of truth for the stable API surface.

## Notebook policy

When adding or updating notebooks in this repository, follow this policy:

- mark whether the notebook is `stable-core` or `experimental-diffusion`;
- state the expected environment near the top of the notebook;
- mention external prerequisites explicitly: CUDA/GPU, model repos, authentication, datasets, or network access;
- prefer clean or minimally relied-on outputs; committed outputs are illustrative, not a compatibility guarantee;
- keep the root `README.md` and `documentation/API_SURFACE.md` aligned with the current notebook classification.
