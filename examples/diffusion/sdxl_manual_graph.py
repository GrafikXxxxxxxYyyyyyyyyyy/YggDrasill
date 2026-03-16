"""Load SDXL components manually and assemble a graph."""
from __future__ import annotations

from pathlib import Path
from pprint import pprint

import torch

from yggdrasill.diffusion.presets.sdxl import build_sdxl_text2img_graph
from yggdrasill.engine.validator import validate
from yggdrasill.integrations.diffusers.model_store import ModelStore
from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes


def main() -> None:
    register_diffusion_nodes()

    store = ModelStore.default()
    sdxl_keys = ["tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2", "unet", "vae", "scheduler"]
    components = store.load_components_by_keys(
        "sdxl", sdxl_keys, "stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16", torch_dtype=torch.float16,
    )

    graph = build_sdxl_text2img_graph(
        tokenizer=components["tokenizer"],
        tokenizer_2=components["tokenizer_2"],
        text_encoder=components["text_encoder"],
        text_encoder_2=components["text_encoder_2"],
        unet=components["unet"],
        vae=components["vae"],
        scheduler=components["scheduler"],
        config={
            "height": 1024,
            "width": 1024,
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
            "output_type": "pil",
        },
    )

    result = validate(graph)

    print(f"graph_id: {graph.graph_id}")
    print(f"nodes: {sorted(graph.node_ids)}")
    print(f"valid: {result.valid}")
    if result.errors:
        print("validation_errors:")
        pprint(result.errors)

    out_dir = Path("artifacts/sdxl_manual_graph")
    graph.save_config(out_dir, filename="config.json")
    print(f"saved config to: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
