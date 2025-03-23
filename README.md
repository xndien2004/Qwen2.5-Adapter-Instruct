# Fine-tuning Qwen2.5 với phương pháp Llama-Adapter cho Fact Checking

Dự án này thực hiện fine-tuning mô hình ngôn ngữ Qwen2.5 sử dụng phương pháp Llama-Adapter cho nhiệm vụ fact checking (kiểm tra tính chính xác của thông tin). Phương pháp này cho phép huấn luyện hiệu quả các mô hình ngôn ngữ lớn với chi phí tính toán thấp.

## Giới thiệu

Qwen2.5 là một mô hình ngôn ngữ lớn được phát triển bởi Alibaba Cloud, trong khi Llama-Adapter là một phương pháp fine-tuning hiệu quả được phát triển cho mô hình Llama. Dự án này kết hợp cả hai để tạo ra một giải pháp fine-tuning hiệu quả cho Qwen2.5, tập trung vào nhiệm vụ fact checking.

## Nhiệm vụ Fact Checking

Dự án này tập trung vào việc huấn luyện mô hình để:
- Phân tích và đánh giá tính chính xác của các tuyên bố
- Xác định nguồn thông tin đáng tin cậy
- Phát hiện thông tin sai lệch và tin giả
- Cung cấp bằng chứng và giải thích cho các kết luận

## Cài đặt

```bash
git clone https://github.com/xndien2004/Qwen2.5-Adapter-Instruct.git
cd Qwen2.5-Adapter-Instruct

pip install -r requirements.txt
```

## Cấu trúc dự án

```
.
├── data/                   # Thư mục chứa dữ liệu training
│   ├── train.json         # Dữ liệu training cho fact checking
│   └── validation.json    # Dữ liệu validation cho fact checking
├── src/                    # Source code
│   ├── adapter/           # Implementation của LoRA Adapter
│   ├── model/             # Model architecture và configurations
│   └── training/          # Training scripts
├── configs/               # Configuration files
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Cách sử dụng

### 1. Chuẩn bị dữ liệu

Đặt dữ liệu training cho fact checking trong thư mục `data/` theo định dạng:
```json
{
    "claim": "Tuyên bố cần kiểm tra",
    "context": "Bối cảnh và thông tin liên quan",
    "label": "true/false/partially_true",
    "evidence": "Bằng chứng hỗ trợ kết luận",
    "explanation": "Giải thích chi tiết về kết luận"
}
```

### 2. Cấu hình training

Chỉnh sửa file cấu hình trong `configs/training_config.yaml`:
```yaml
model:
  name: "Qwen/Qwen2.5-7B"
  adapter_config:
    rank: 8
    alpha: 16
    dropout: 0.1

training:
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 3
  gradient_accumulation_steps: 4
  task_type: "fact_checking"
  metrics:
    - accuracy
    - f1_score
    - evidence_retrieval_score
```

### 3. Bắt đầu training

```bash
python src/training/train.py --config configs/training_config.yaml
```

## Kiến trúc Adapter

Adapter được triển khai dựa trên kiến trúc của Llama-Adapter với các thành phần chính:

1. **LoRA Layers**: Các lớp ma trận hạng thấp được thêm vào các layer attention
2. **Adapter Gates**: Cơ chế điều khiển luồng thông tin qua adapter
3. **Zero-init Attention**: Khởi tạo attention weights với giá trị 0 để đảm bảo training ổn định
4. **Fact Checking Head**: Layer đặc biệt cho việc phân tích và đánh giá tính chính xác

## Kết quả và Đánh giá

- **Hiệu suất Fact Checking**: 
  - Độ chính xác trên tập test
  - F1-score cho các lớp true/false/partially_true
  - Khả năng trích xuất bằng chứng
- **Thời gian training**: Giảm đáng kể so với fine-tuning truyền thống
- **Dung lượng model**: Chỉ cần lưu trữ adapter weights (~100MB) thay vì toàn bộ model

## Các điểm cần lưu ý

1. Đảm bảo đủ GPU memory (tối thiểu 16GB)
2. Sử dụng gradient checkpointing để tiết kiệm memory
3. Điều chỉnh batch size và learning rate phù hợp với hardware
4. Cân bằng dữ liệu training giữa các lớp true/false/partially_true
5. Đảm bảo chất lượng và độ đa dạng của dữ liệu training

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## License

MIT License 

## References

1. Qwen2.5 Technical Report (2025)
   - Tác giả: Qwen Team
   - arXiv: [2412.15115](https://arxiv.org/abs/2412.15115)
   - Mô tả: Báo cáo kỹ thuật chi tiết về mô hình Qwen2.5

2. LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention (2023)
   - Tác giả: Zhang et al.
   - arXiv: [2303.16199](https://arxiv.org/abs/2303.16199)
   - Mô tả: Phương pháp fine-tuning hiệu quả với zero-init attention

3. SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking (2025)
   - Tác giả: Nguyen et al.
   - arXiv: [2503.00955](https://arxiv.org/abs/2503.00955)
   - Mô tả: Hệ thống hỏi đáp ngữ nghĩa cho fact checking tiếng Việt 