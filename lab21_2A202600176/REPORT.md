# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Mạnh Dũng
**Mã học viên**: 2A202600176
**Ngày nộp**: 2026-05-07
**Submission option**: B (HF Hub)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 rounded up)
- **GPU**: Tesla T4, 16 GB VRAM
- **Training cost**: Free (Google Colab T4)
- **HF Hub link**: [DuNGuyen1806/qwen2.5-3b-vi-lab21-r16](https://huggingface.co/DuNGuyen1806/qwen2.5-3b-vi-lab21-r16)

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 4.11 min   | 11.22 GB  | 1.5577    | 4.7479     |
| 16   | 3,686,400       | 3.98 min   | 10.62 GB  | 1.5161    | 4.5544     |
| 64   | 14,745,600      | 3.80 min   | 12.00 GB  | 1.4768    | 4.3790     |
| Base | -               | -          | -         | -         | -          |

*Lưu ý: Thời gian training có thể biến động nhẹ do môi trường Colab dùng chung.*

## 3. Loss Curve Analysis
- Quan sát: Loss giảm đều đặn qua các epoch. Không có dấu hiệu overfitting rõ rệt trên tập eval nhỏ (20 samples) trong 3 epoch. Eval loss của rank 64 là thấp nhất, cho thấy việc tăng rank giúp mô hình khớp dữ liệu tốt hơn trên domain tiếng Việt này.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là một bộ môn khoa học tính toán nhằm học từ dữ liệu... (Hơi dài dòng, chưa ngắt câu tốt)
**Fine-tuned (r=16)**: Machine learning là một bộ môn khoa học máy tính giúp máy tự học và cải thiện từ dữ liệu... (Định nghĩa trực diện hơn, cấu trúc câu tự nhiên hơn)
**Nhận xét**: Improved.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: Cung cấp giải pháp đệ quy nhưng giải thích hơi lủng củng.
**Fine-tuned (r=16)**: Cung cấp code Python sạch hơn, có kiểm tra điều kiện đầu vào (ValueError).
**Nhận xét**: Improved.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: Liệt kê các nguyên tắc chung chung.
**Fine-tuned (r=16)**: Có cấu trúc đánh số rõ ràng, nội dung tập trung vào trải nghiệm người dùng.
**Nhận xét**: Improved structure.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: Giải thích cơ bản về việc thay thế trọng lượng.
**Fine-tuned (r=16)**: Nhấn mạnh vào việc sử dụng các "hạt nhân nhỏ" (low-rank matrices) và hiệu quả tính toán.
**Nhận xét**: Same/Slightly Improved.

### Example 5
**Prompt**: Viết một ví dụ về unit test trong Python cho một hàm cộng hai số.
**Base**: Sử dụng pytest nhưng code bị cắt nửa chừng.
**Fine-tuned (r=16)**: Cung cấp ví dụ trực quan hơn (mặc dù vẫn bị giới hạn độ dài trong bảng so sánh).
**Nhận xét**: Improved formatting.

## 5. Conclusion về Rank Trade-off

- **Rank nào cho ROI tốt nhất trên dataset này? Tại sao?**
  Rank 16 dường như mang lại ROI tốt nhất. Nó cân bằng giữa số lượng tham số huấn luyện (3.6M) và hiệu quả giảm perplexity so với rank 8. Mặc dù rank 64 có perplexity thấp nhất (4.3790), nhưng sự chênh lệch so với rank 16 không quá lớn trong khi số lượng tham số tăng lên gấp 4 lần (14.7M), dẫn đến tiêu tốn VRAM nhiều hơn khi load mô hình full.
  
- **Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?**
  Dấu hiệu diminishing returns bắt đầu xuất hiện khi tăng từ rank 16 lên 64. Eval loss chỉ giảm khoảng 0.04 (từ 1.51 xuống 1.47), trong khi trainable parameters tăng mạnh. Với một dataset nhỏ (200 samples), việc tăng rank quá cao có thể dẫn đến lãng phí tài nguyên mà không mang lại sự khác biệt đáng kể trong output thực tế.

- **Recommendation: nếu deploy production, bạn chọn rank nào? Tại sao?**
  Tôi chọn Rank 16. Đây là mức rank tiêu chuẩn được khuyến nghị trong nhiều bài báo (như LoRA gốc), đủ để học các pattern đặc thù của domain (tiếng Việt Alpaca) mà vẫn giữ cho adapter nhẹ nhàng, dễ dàng merge hoặc swap linh hoạt trong môi trường production có giới hạn tài nguyên.

## 6. What I Learned
- Tăng rank (r) giúp mô hình có khả năng học được nhiều chi tiết hơn từ dataset, nhưng đồng thời làm tăng chi phí tính toán và bộ nhớ.
- QLoRA (4-bit) kết hợp với Unsloth cực kỳ hiệu quả để fine-tune các model 3B+ trên GPU phổ thông như T4.
- Việc chuẩn bị dữ liệu chất lượng (Alpaca format) quan trọng hơn số lượng dữ liệu; chỉ với 200 samples, model đã có sự cải thiện rõ rệt về phong cách trả lời.
