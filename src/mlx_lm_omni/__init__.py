from mlx_lm import generate as _original_generate

from .utils import load


def generate(model, tokenizer, prompt, **kwargs):
    """
    Wrapper around mlx_lm.generate that filters out parameters not supported by generate_step.

    This fixes the issue where temperature and other parameters cause:
    TypeError: generate_step() got an unexpected keyword argument 'temperature'
    """
    # Parameters that are known to cause issues with generate_step in MLX-LM 0.28.3
    problematic_params = [
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "repetition_context_size",
    ]

    # Filter out problematic parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in problematic_params}

    # Log if we're filtering out any parameters
    filtered_out = {k: v for k, v in kwargs.items() if k in problematic_params}
    if filtered_out:
        print(
            f"Warning: Filtering out unsupported parameters: {list(filtered_out.keys())}"
        )
        print(
            "These parameters are not supported in MLX-LM 0.28.3 due to a bug in generate_step"
        )

    return _original_generate(model, tokenizer, prompt, **filtered_kwargs)


__all__ = ["load", "generate"]
