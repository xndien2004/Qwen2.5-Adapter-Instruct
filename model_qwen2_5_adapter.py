from typing import Tuple
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from fairscale.nn.model_parallel import initialize
# from fairscale.nn.data_parallel import ShardedDataParallel

from qwen2_5_adapter_v1 import Qwen2AdapterV1Config, Qwen2AdapterV1ForCausalLM
from qwen2_5_adapter_v2 import Qwen2AdapterV2Config, Qwen2AdapterV2ForCausalLM

# world_size = 4
# rank = dist.get_rank()  

# dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
# initialize.initialize_model_parallel(2)
def Qwen2_5_Adapter(model_name: str, adapter_len: int = 64, adapter_layer: int = 4, is_type_qwen_adapter: str = "v1") -> Tuple[torch.nn.Module, AutoTokenizer]:
    assert is_type_qwen_adapter in ["v1", "v2"], "is_type_qwen_adapter must be 'v1' or 'v2'."

    if is_type_qwen_adapter == "v1":
        config_class, model_class = Qwen2AdapterV1Config, Qwen2AdapterV1ForCausalLM
    else:
        config_class, model_class = Qwen2AdapterV2Config, Qwen2AdapterV2ForCausalLM

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
    # if is_type_qwen_adapter == "v2":
    #     model = ShardedDataParallel(model, sharding_strategy="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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
