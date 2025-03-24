from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AdaptionPromptConfig, get_peft_model

from qwen2_5_adapter_v1 import Qwen2AdapterV1Config, Qwen2AdapterV1ForCausalLM

def Qwen2_5_Adapter(model_name: str, adapter_len: int = 64, adapter_layer: int = 4) -> Tuple[Qwen2AdapterV1ForCausalLM, AutoTokenizer]:
    # config = Qwen2AdapterV1Config.from_pretrained(model_name)
    # config.adapter_len = adapter_len
    # config.adapter_layer = adapter_layer
    # config._attn_implementation = "eager"

    # model = Qwen2AdapterV1ForCausalLM.from_pretrained(
    #     model_name,
    #     config=config,
    #     trust_remote_code=True,
    #     ignore_mismatched_sizes=True,
    #     torch_dtype=torch.float16
    # ).to("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    for param in model.parameters():
        param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if "adapter_query" in name or "gate_adapter" in name:
    #         param.requires_grad = True

    config = AdaptionPromptConfig(
        adapter_layers=30,  # Ví dụ: Số lớp trên cùng để áp dụng adapter
        adapter_len=10,     # Ví dụ: Độ dài của các prompt thích ứng
        # ... các tham số liên quan khác ...
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded. Trainable parameters: {total_params}")

    return model, tokenizer