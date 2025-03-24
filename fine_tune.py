import argparse
import torch
import pandas as pd
import numpy as np
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback
) 
from data_utils import FactVerificationDataset
from model_qwen2_5_adapter import Qwen2_5_Adapter 

class GradientClippingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        return control

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2 Adapter Model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Pretrained model name")
    parser.add_argument("--adapter_layer", type=int, default=4, help="Adapter layer")
    parser.add_argument("--adapter_len", type=int, default=64, help="Adapter length")
    parser.add_argument("--max_length", type=int, default=1024, help="Max input length")
    parser.add_argument("--train_file", type=str, default="/kaggle/input/semviqa-data/data/evi/viwiki_train.csv", help="Path to training data")
    parser.add_argument("--val_file", type=str, default="/kaggle/input/semviqa-data/data/evi/viwiki_test.csv", help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="1_5B_adapter4", help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)  # Giảm learning rate để giảm độ lớn của gradient
    return parser.parse_args()


def get_training_args(args) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        fp16=False,
        evaluation_strategy="epoch",  
        save_strategy="epoch",           
        save_total_limit=3,
        logging_steps=10,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,   
        max_grad_norm=0.5,  
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        report_to=["none"]
    )


def main():
    args = parse_args()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, tokenizer = Qwen2_5_Adapter(args.model_name, adapter_layer=args.adapter_layer, adapter_len=args.adapter_len)

    # Load datasets
    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)
    train_dataset = FactVerificationDataset(train_df, tokenizer, max_len=args.max_length)
    val_dataset = FactVerificationDataset(val_df, tokenizer, max_len=args.max_length)

    training_args = get_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[GradientClippingCallback()],
    )

    model.config.use_cache = False
 
    def compute_gradients(model):
        for param in model.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, max_norm=0.5)  

    trainer.add_callback(compute_gradients)   

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
