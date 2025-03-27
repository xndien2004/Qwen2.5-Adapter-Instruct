import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from qwen2_5_adapter_v1 import Qwen2AdapterV1Config, Qwen2AdapterV1ForCausalLM
from qwen2_5_adapter_v2 import Qwen2AdapterV2Config, Qwen2AdapterV2ForCausalLM


def Qwen2_5_Adapter(model_name: str, adapter_len: int = 64, adapter_layer: int = 4, is_type_qwen_adapter: str = "v1", use_model_origin: int = 0) -> Tuple[torch.nn.Module, AutoTokenizer]:
    assert is_type_qwen_adapter in ["v1", "v2"], "is_type_qwen_adapter must be 'v1' or 'v2'."
    
    if is_type_qwen_adapter == "v1":
        config_class, model_class = Qwen2AdapterV1Config, Qwen2AdapterV1ForCausalLM
    else:
        config_class, model_class = Qwen2AdapterV2Config, Qwen2AdapterV2ForCausalLM

    # Load configuration and model
    if use_model_origin:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
    else:
        config = config_class.from_pretrained(model_name)
        config.adapter_len = adapter_len
        config.adapter_layer = adapter_layer
        if is_type_qwen_adapter == "v2":
            config.add_bias = True
            config.add_scale = True
        config._attn_implementation = "eager"

        model = model_class.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16
        ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if use_model_origin == 0:
        for name, param in model.named_parameters():
            requires_grad = (
                "adapter_query" in name
                or "gate_adapter" in name
                or name.endswith(".added_bias")
                or name.endswith(".added_scale")
            )

            if requires_grad:
                param.data = param.data.float() 
                param.requires_grad = True
            else:
                param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model '{model_name}' (Adapter {is_type_qwen_adapter}) loaded successfully.")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model, tokenizer
