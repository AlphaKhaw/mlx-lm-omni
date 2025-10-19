from pathlib import Path
from typing import Optional, Tuple, Type

import mlx.nn as nn
from mlx_lm.utils import hf_repo_to_path, load_adapters, load_model, load_tokenizer

import mlx_lm_omni.models.qwen_omni.model as qwen_omni
import mlx_lm_omni.models.ultravox.model as ultravox


def get_model_classes(config: dict) -> Tuple[Type[nn.Module], Type]:
    match config["model_type"]:
        case "qwen2_5_omni":
            return qwen_omni.Model, qwen_omni.ModelArgs
        case "ultravox":
            return ultravox.Model, ultravox.ModelArgs
        case _:
            raise ValueError(f"Model type {config['model_type']} not supported")


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    thinker_adapter_path: Optional[str] = None,
    lazy: bool = False,
):
    if Path(path_or_hf_repo).exists():
        # Local path
        model_path = Path(path_or_hf_repo)
    else:
        # HuggingFace repo ID - resolve to local cache path
        model_path = Path(hf_repo_to_path(path_or_hf_repo))

    model, config = load_model(
        model_path,
        lazy,
        model_config=model_config,
        strict=False,
        get_model_classes=get_model_classes,
    )
    if thinker_adapter_path is not None:
        model.thinker = load_adapters(model.thinker, thinker_adapter_path)
        model.eval()
    tokenizer = load_tokenizer(
        model_path, tokenizer_config, eos_token_ids=config.get("eos_token_id", None)
    )
    tokenizer = model.build_custom_tokenizer(tokenizer)
    return model, tokenizer
