import argparse
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from data_utils import FactVerificationDataset
import pandas as pd
from model_qwen2_5_adapter import Qwen2_5_Adapter

def load_model_and_tokenizer(model_path):
    model, tokenizer = Qwen2_5_Adapter(model_name=model_path)
    return model, tokenizer

def evaluate_model(model, tokenizer, eval_file, max_len=1024, batch_size=2):
    # Load evaluation data
    eval_df = pd.read_csv(eval_file)
    eval_dataset = FactVerificationDataset(eval_df, tokenizer, max_len=max_len)
    
    trainer = Trainer(
        model=model,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            per_device_eval_batch_size=batch_size,
            output_dir='./eval_output',
            do_train=False,
            do_eval=True
        )
    )
     
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
    return eval_results

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2 Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to the evaluation file")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum input length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation")
    return parser.parse_args()

def main(): 
    args = parse_args()
     
    model, tokenizer = load_model_and_tokenizer(args.model_path)
     
    eval_results = evaluate_model(model, tokenizer, args.eval_file, max_len=args.max_len, batch_size=args.batch_size)
     
    print(f"Evaluation completed. Results: {eval_results}")

if __name__ == "__main__":
    main()
