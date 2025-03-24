from torch.utils.data import Dataset
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from typing import Dict

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def preprocess_fact_verification(
    dataframe,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are an AI assistant specializing in verifying the accuracy of information in Vietnamese."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    if hasattr(tokenizer, 'im_start_id'):
        im_start = tokenizer.im_start_id
        im_end = tokenizer.im_end_id
    else:
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    input_ids_list, targets_list = [], []

    for _, row in dataframe.iterrows():
        context, claim, verdict, evidence = row['context'], row['claim'], row['verdict'], row['evidence']

        input_id, target = [], []

        # System prompt
        system_prompt = (
            "You are tasked with verifying the accuracy of a statement based on the provided context in Vietnamese.\n\n"
            "Requirements:\n"
            "We provide you with a claim and a context.\n"
            "Your task is to classify the claim into one of the following three labels:\n"
            "SUPPORTED: If the claim is fully supported by the information in the context.\n"
            "REFUTED: If the claim contradicts the information in the context.\n"
            "NEI (Not Enough Information): If the context does not provide sufficient information to either support or refute the claim.\n"
            "Your answer must include the classification label and a complete sentence from the context as evidence to justify your decision.\n"
            "Note: The evidence must be a full sentence, not a partial sentence or a fragment.\n"
            "Answer format:\n"
            "Answer: The claim is classified as <LABEL>. The evidence is: <EVIDENCE>.\n\n"
            f"Provided data:\nContext: {context}\nClaim: {claim}"
        )

        # Add system
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens

        # Add user prompt
        # print("user:", tokenizer(roles["user"]).input_ids)
        # print("im_start:", im_start)
        user_msg = [im_start] + nl_tokens + tokenizer(system_prompt).input_ids + [im_end] + nl_tokens
        input_id += user_msg
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(user_msg) - 3) + [im_end] + nl_tokens

        # Add assistant response
        assistant_msg = f"Answer: The claim is classified as {verdict}. The evidence is: {evidence}."
        assistant_token = tokenizer(roles["assistant"]).input_ids + nl_tokens + tokenizer(assistant_msg).input_ids + [im_end] + nl_tokens
        input_id += assistant_token

        _target = (
            [im_start] +
            [IGNORE_TOKEN_ID] * len(tokenizer(roles["assistant"]).input_ids) +
            tokenizer(assistant_msg).input_ids +
            [im_end] + nl_tokens
        )
        target += _target

        # Padding
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))

        input_ids_list.append(input_id[:max_len])
        targets_list.append(target[:max_len])

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.int)
    targets_tensor = torch.tensor(targets_list, dtype=torch.long)

    return dict(
        input_ids=input_ids_tensor,
        labels=targets_tensor,
        attention_mask=input_ids_tensor.ne(tokenizer.pad_token_id),
    )


class FactVerificationDataset(Dataset):
    """Dataset for fact verification fine-tuning."""

    def __init__(self, dataframe, tokenizer: transformers.PreTrainedTokenizer, max_len=1024):
        super(FactVerificationDataset, self).__init__()

        print("Formatting fact verification dataset...")
        self.data_dict = preprocess_fact_verification(dataframe, tokenizer, max_len)

        self.input_ids = self.data_dict["input_ids"]
        self.labels = self.data_dict["labels"]
        self.attention_mask = self.data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )
