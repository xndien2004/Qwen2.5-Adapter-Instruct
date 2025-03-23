from torch.utils.data import Dataset
import torch
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def preprocess_fact_verification(dataframe, tokenizer, max_length: int):
    input_ids_list, attention_mask_list, label_ids_list = [], [], []

    for idx, row in dataframe.iterrows():
        context, claim, verdict, evidence = row['context'], row['claim'], row['verdict'], row['evidence']

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
Answer: The claim is classified as <LABEL>. The evidence is: <EVIDENCE>.

Provided data:
Context: {context}
Claim: {claim}
"""}
        ]

        label_text = f"Answer: The claim is classified as {verdict}. The evidence is: {evidence}."

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) + label_text

        full = tokenizer(prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = full["input_ids"].squeeze()
        attention_mask = full["attention_mask"].squeeze()
        labels = input_ids.clone()

        # Mask prompt
        prompt_ids = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                               max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"].squeeze()
        prompt_len = (prompt_ids != tokenizer.pad_token_id).sum().item()
        labels[:prompt_len] = IGNORE_TOKEN_ID

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        label_ids_list.append(labels)

    return dict(
        input_ids=torch.stack(input_ids_list),
        attention_mask=torch.stack(attention_mask_list),
        labels=torch.stack(label_ids_list),
    )


class FactVerificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=1024):
        self.examples = preprocess_fact_verification(dataframe, tokenizer, max_length)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples["input_ids"][idx],
            "attention_mask": self.examples["attention_mask"][idx],
            "labels": self.examples["labels"][idx],
        }