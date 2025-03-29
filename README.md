# Qwen2.5 Adapter Instruct: Fine-tuning Qwen2.5 with the Llama-Adapter Method for Fact-Checking  

This project fine-tunes the Qwen2.5 language model using the Llama-Adapter method for fact-checking. This approach enables efficient training of large language models with low computational costs.  

## Introduction  

Qwen2.5 is a large language model developed by Alibaba Cloud, while Llama-Adapter is an efficient fine-tuning method originally designed for Llama models. This project integrates both to create an effective fine-tuning solution for Qwen2.5, specifically for fact-checking tasks.  

## Fact-Checking Task  

This project focuses on training the model to:  
- Analyze and assess the accuracy of claims  
- Identify reliable information sources  
- Detect misinformation and fake news  
- Provide evidence and explanations for conclusions  

## Installation  

```bash
git clone https://github.com/xndien2004/Qwen2.5-Adapter-Instruct.git  
cd Qwen2.5-Adapter-Instruct  

pip install -r requirements.txt  
```  

## Usage  

### 1. Prepare Data  

Place fact-checking training data in the `data/` directory following this format:  

```json
{
     "messages":[
         {"role": "system", "content": "You are an AI assistant specializing in verifying the accuracy of information in Vietnamese."},
         {"role": "user", "content": "Question: You are tasked with verifying the correctness of the following statement.\n We provide you with a claim and a context. Please classify the claim into one of three labels:\n - SUPPORTED: the evidence supports the claim;\n - REFUTED: the evidence contradicts the claim;\n - NEI: not enough information to decide.\n Your answer should include the classification label and the most relevant evidence sentence from the context.\n Remember, the evidence must be a full sentence, not part of a sentence or less than one sentence. Given a claim and context as follows:\n Context: {context}\n Claim: {claim}\n Answer: The claim is classified as <LABEL>. The evidence is: <EVIDENCE>"},
         {"role": "assistant", "content": "Answer: The claim is classified as {verdict}. The evidence is: {evidence}"},
    ],
    "format": "chatml"
}
```  
For the SFT datasets, the raw JSONLINE file follows the following format:
```bash
{"messages": [sample1...], "format": "chatml"}
{"messages": [sample2...], "format": "chatml"}
{"messages": [sample3...], "format": "chatml"}
```

### 2. Start Training  

```bash
python3 -m fine_tune \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --adapter_layer 12 \
    --adapter_len 128 \
    --max_length 768\
    --train_file "/kaggle/input/viwikifc-process/chatml_viwikifc_train.jsonl" \
    --val_file "/kaggle/input/viwikifc-process/chatml_viwikifc_test.jsonl" \
    --output_dir "output" \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --epochs 4 \
    --learning_rate 5e-4 \
    --is_type_qwen_adapter "v2" \
    --use_model_origin 0
```  

## License  

MIT License  

## References  

1. **Qwen2.5 Technical Report (2025)**  
   - Authors: Qwen Team  
   - arXiv: [2412.15115](https://arxiv.org/abs/2412.15115)  
   - Description: Detailed technical report on the Qwen2.5 model  

2. **LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention (2023)**  
   - Authors: Zhang et al.  
   - arXiv: [2303.16199](https://arxiv.org/abs/2303.16199)  
   - Description: Efficient fine-tuning method using zero-init attention  

3. **SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking (2025)**  
   - Authors: Nguyen et al.  
   - arXiv: [2503.00955](https://arxiv.org/abs/2503.00955)  
   - Description: A semantic question-answering system for Vietnamese fact-checking