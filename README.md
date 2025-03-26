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
    "claim": "Claim to be verified",
    "context": "Relevant background information",
    "label": "SUPPORTS/REFUTES/NEI",
    "evidence": "Evidence supporting the conclusion"
}
```  

### 2. Start Training  

```bash
python3 -m fine_tune \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --adapter_layer 12 \
    --adapter_len 64 \
    --max_length 512 \
    --train_file "/kaggle/input/semviqa-data/data/evi/viwiki_train.csv" \
    --val_file "/kaggle/input/semviqa-data/data/evi/viwiki_test.csv" \
    --output_dir "adapter" \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --epochs 3 \
    --learning_rate 5e-4 \
    --is_type_qwen_adapter "v2"
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