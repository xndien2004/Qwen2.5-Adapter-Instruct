from torch.utils.data import Dataset
import torch

class FactVerificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Định nghĩa mapping cho labels
        self.label_map = {
            "SUPPORTED": 0,
            "REFUTED": 1,
            "NEI": 2
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Format chat template cho Qwen
        messages = [
            {"role": "system", "content": "You are an AI assistant specializing in verifying the accuracy of information in Vietnamese."},
            {"role": "user", "content": f"""You are tasked with verifying the accuracy of a statement based on the provided context in Vietnamese.

Requirements:
We provide you with a claim and a context.
Your task is to classify the claim into one of the following three labels:
SUPPORTED: If the claim is fully supported by the information in the context.
REFUTED: If the claim contradicts the information in the context.
NEI (Not Enough Information): If the context does not provide sufficient information to either support or refute the claim.
Your answer must include the classification label and a complete sentence from the context as evidence to justify your decision.
Note: The evidence must be a full sentence, not a partial sentence or a fragment.
Answer format:
Answer: The claim is classified as {item['verdict']}. The evidence is: {item['evidence']}.

Provided data:
Context: {item['context']}
Claim: {item['claim']}
"""}
        ]
 
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  
            add_generation_prompt=True  
        )

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_text = f"Answer: The claim is classified as {item['verdict']}. The evidence is: {item['evidence']}"
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = target_encoding["input_ids"].squeeze()
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels
        }