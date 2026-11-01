# 03. THIẾT KẾ THỰC NGHIỆM VÀ HỆ THỐNG ĐÁNH GIÁ

Tài liệu này quy định tiêu chuẩn thiết lập môi trường phần cứng, cấu hình siêu tham số (Hyperparameters), kịch bản thực nghiệm đối chứng, và hệ thống chỉ số đánh giá khoa học cho bài toán Phân loại Ngôn từ thù ghét với PhoBERT-base trên bộ dữ liệu ViHSD.

---

## 1. Môi trường Thực nghiệm Phần cứng & Phần mềm

### 1.1. Hạ tầng tính toán (Google Colab Free Tier)
- **Bộ xử lý đồ họa (GPU):** NVIDIA Tesla T4 (16GB GDDR6 VRAM, Kiến trúc Turing, Compute Capability 7.5).
- **Hệ điều hành:** Linux x86_64 (Ubuntu dựa trên Colab Runtime).
- **Cơ chế tăng tốc Mixed Precision (FP16):**
  - **Kỹ thuật:** Sử dụng chế độ tính toán độ chính xác nửa (Half Precision - 16-bit Floating Point) kết hợp với Dynamic Loss Scaling.
  - **Lợi ích:**
    - Giảm mức tiêu thụ bộ nhớ đồ họa (VRAM) xấp xỉ **40% - 50%**.
    - Tăng tốc độ tính toán ma trận thông qua các Tensor Cores của GPU T4 (tốc độ tăng từ 2x đến 2.5x so với FP32).
    - Đảm bảo an toàn không bị tràn số (underflow/overflow) nhờ cơ chế scaled gradient.

### 1.2. Thư viện phần mềm cốt lõi
- **PyTorch:** $\ge 2.0.0$ (Hỗ trợ native AMP - Automatic Mixed Precision).
- **Hugging Face Transformers:** $\ge 4.30.0$ (Quản lý mô hình PhoBERT và AutoTokenizer).
- **Hugging Face Datasets / Accelerate:** Tối ưu luồng dữ liệu và nạp batch.
- **Scikit-learn:** Tính toán trọng số lớp (`compute_class_weight`) và tính toán độ đo (`classification_report`, `f1_score`, `confusion_matrix`).
- **PyVi / VnCoreNLP:** Tách từ tiếng Việt chuẩn hóa tiền xử lý.

---

## 2. Thiết lập Siêu tham số (Hyperparameter Configurations)

Các siêu tham số được căn chỉnh nhằm tối ưu hóa hiệu năng mô hình trong giới hạn tài nguyên của Google Colab:

| Siêu tham số (Hyperparameter) | Giá trị thiết lập | Cơ sở khoa học & Lý do kỹ thuật |
|:---|:---:|:---|
| **Pretrained Model** | `vinai/phobert-base-v2` | Phiên bản RoBERTa tối ưu mới nhất cho tiếng Việt, độ ổn định cao hơn v1. |
| **Max Sequence Length** | **128** | Theo thống kê tập ViHSD, hơn 98.5% bình luận có độ dài dưới 128 subwords. Giới hạn này triệt tiêu chi phí tính toán cho padding thừa, giúp tiết kiệm bộ nhớ GPU. |
| **Batch Size** | **16** | Đảm bảo nạp vừa vặn vào 16GB VRAM của T4 khi bật FP16 mà không gặp lỗi hết bộ nhớ (CUDA Out of Memory - OOM). |
| **Số lượng Epochs** | **3 - 4** | Đối với mô hình Pretrained Transformer lớn như PhoBERT, 3 đến 4 epoch là điểm hội tụ lý tưởng; huấn luyện quá dài dễ gây hiện tượng Catastrophic Forgetting và Overfitting trên tập Train. |
| **Tốc độ học (Learning Rate)** | **$2 \times 10^{-5}$** ($2e-5$) | Tốc độ học chuẩn cho Fine-tuning các mô hình họ BERT/RoBERTa; tránh phá vỡ trọng số đã tiền huấn luyện. |
| **Bộ tối ưu (Optimizer)** | **AdamW** | Khắc phục nhược điểm L2 Regularization của Adam truyền thống bằng Weight Decay độc lập ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$). |
| **Trọng số suy giảm (Weight Decay)** | **0.01** | Hạn chế Overfitting các trọng số trong Classification Head. |
| **Chiến lược điều chỉnh LR (Scheduler)** | **Linear Warmup with Decay** | Khởi động nhẹ trong 10% tổng số bước đầu (Warmup ratio = 0.1), sau đó giảm tuyến tính về 0. |
| **Hạt giống ngẫu nhiên (Random Seed)** | **42** | Cố định ngẫu nhiên (`torch`, `numpy`, `random`) để tái lập 100% kết quả thực nghiệm. |

---

## 3. Kịch bản Thực nghiệm Đối chứng (Experimental Scenarios)

Để chứng minh tính hiệu quả của phương pháp giải quyết mất cân bằng lớp, dự án triển khai 2 kịch bản thực nghiệm chính (và 1 kịch bản mở rộng):

```
                                  Bộ dữ liệu ViHSD
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
      [Thực nghiệm 1: Baseline]                      [Thực nghiệm 2: Đề xuất]
         PhoBERT-base v2                                 PhoBERT-base v2
                 +                                               +
    Standard Cross-Entropy Loss                      Weighted Cross-Entropy Loss
  (Không áp dụng Class Weight)                         (Balanced Class Weight)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                            Đối chiếu trên tập TEST
                 ┌───────────────────────────────────────────────┐
                 │ 1. Macro F1-score (MF1)                       │
                 │ 2. Weighted F1-score (WF1)                    │
                 │ 3. Per-class F1 & Confusion Matrix            │
                 │ 4. Độ dịch chuyển Recall trên HATE/OFFENSIVE  │
                 └───────────────────────────────────────────────┘
```

### 3.1. Thực nghiệm 1: Baseline Model (PhoBERT-base + Standard Cross Entropy)
- **Mục đích:** Thiết lập mốc cơ sở (benchmark) để so sánh.
- **Hàm mất mát:** `nn.CrossEntropyLoss()` không truyền tham số trọng số `weight`.
- **Dự báo lý thuyết:**
  - Accuracy tổng thể cao (khoảng 85% - 88%) do đoán trúng lớp `CLEAN`.
  - Recall của lớp `OFFENSIVE` và `HATE` thấp (thường dưới 40% - 50%), nhiều mẫu thù ghét bị bỏ sót và gán nhãn thành `CLEAN`.
  - Macro F1 bị kéo tụt xuống mức trung bình.

### 3.2. Thực nghiệm 2: Proposed Model (PhoBERT-base + Class Weight)
- **Mục đích:** Giải quyết triệt để vấn đề mất cân bằng lớp bằng phạt tổn thất có trọng số.
- **Hàm mất mát:** `nn.CrossEntropyLoss(weight=class_weights)`.
- **Trọng số:** Được tính toán tự động từ tập Train:
  ```python
  from sklearn.utils.class_weight import compute_class_weight
  import numpy as np
  import torch

  class_weights = compute_class_weight(
      class_weight="balanced",
      classes=np.unique(train_labels),
      y=train_labels
  )
  weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
  ```
- **Dự báo lý thuyết:**
  - Recall của lớp `OFFENSIVE` và `HATE` tăng vọt.
  - Tỷ lệ bỏ sót ngôn từ thù ghét giảm rõ rệt.
  - Chỉ số **Macro F1-score** cải thiện đáng kể từ 3% đến 8% so với Baseline.

### 3.3. Thực nghiệm 3 (Tùy chọn mở rộng): PhoBERT-base + Focal Loss
- **Mục đích:** Kiểm nghiệm cơ chế tự động giảm trọng số của các mẫu `CLEAN` dễ phân loại và tập trung vào các mẫu ranh giới phức tạp.
- **Tham số:** $\alpha = \text{class\_weights}$, $\gamma = 2.0$.

---

## 4. Hệ thống Chỉ số Đánh giá (Evaluation Metrics)

Do tính chất phân bố dữ liệu cực kỳ mất cân bằng, **tuyệt đối không sử dụng Accuracy** làm chỉ số chính. Hệ thống đánh giá dựa trên 4 trụ cột định lượng sau:

### 4.1. Macro F1-score (MF1) - Thước đo Trọng yếu nhất
- **Công thức:**
  $$MF1 = \frac{F1_{\text{CLEAN}} + F1_{\text{OFFENSIVE}} + F1_{\text{HATE}}}{3}$$
- **Ý nghĩa:** Macro F1 tính trung bình cộng không trọng số F1-score của cả 3 lớp. Nó coi trọng lớp `OFFENSIVE` (6.7%) và lớp `HATE` (10.5%) ngang hàng với lớp đa số `CLEAN` (82.7%).
- **Vai trò:** Đây là chỉ số chuẩn mực quốc tế trên bảng xếp hạng các nghiên cứu NLP về phát hiện ngôn từ độc hại (SemEval, EMNLP).

### 4.2. Weighted F1-score (WF1)
- **Công thức:**
  $$WF1 = \sum_{c=0}^{2} \left( \frac{N_c}{N_{\text{total}}} \times F1_c \right) = 0.83 \cdot F1_0 + 0.07 \cdot F1_1 + 0.10 \cdot F1_2$$
- **Ý nghĩa:** Phản ánh hiệu năng tổng quát tính theo xác suất xuất hiện thực tế của các loại bình luận trên mạng xã hội.

### 4.3. Per-class Precision, Recall và F1-score
Đối với từng lớp $c \in \{0, 1, 2\}$, đánh giá bộ ba chỉ số:

| Chỉ số | Công thức | Ý nghĩa thực tiễn trong bài toán Kiểm duyệt |
|:---|:---:|:---|
| **Precision ($P_c$)** | $\frac{TP_c}{TP_c + FP_c}$ | Tỷ lệ những bình luận mô hình **báo là vi phạm** thực sự là vi phạm. Nếu Precision thấp $\rightarrow$ Báo động giả nhiều, gây ức chế cho người dùng bình thường. |
| **Recall ($R_c$)** | $\frac{TP_c}{TP_c + FN_c}$ | Tỷ lệ những bình luận **thực sự vi phạm** được mô hình tóm được. Nếu Recall thấp $\rightarrow$ Bỏ lọt ngôn từ thù ghét, độc hại lan tràn trên mạng xã hội. |
| **F1-score ($F1_c$)** | $2 \cdot \frac{P_c \cdot R_c}{P_c + R_c}$ | Trung bình điều hòa giữa Precision và Recall của lớp $c$. |

> **Quy luật Đánh đổi (Trade-off):** Khi đưa Class Weight vào huấn luyện, mô hình sẽ chấp nhận hy sinh một lượng nhỏ Precision của lớp thiểu số để đổi lấy sự tăng trưởng vượt bậc của **Recall**. Trong bài toán kiểm duyệt an toàn nội dung, **ưu tiên Recall cao là mục tiêu sống còn** nhằm không bỏ lọt vi phạm.

### 4.4. Ma trận Nhầm lẫn (Confusion Matrix)
Ma trận $3 \times 3$ biểu diễn trực quan mối tương quan giữa nhãn thực tế (Ground Truth) và nhãn dự đoán (Predicted Label):

```
                          NHÃN DỰ ĐOÁN (PREDICTED)
                     CLEAN (0)    OFFENSIVE (1)    HATE (2)
                  ┌─────────────┬───────────────┬────────────┐
        CLEAN (0) │  TP(CLEAN)  │   FP(OFF_0)   │  FP(HATE_0)│
NHÃN THỰC         ├─────────────┼───────────────┼────────────┤
(GROUND OFFENSIVE │  FN(OFF_0)  │ TP(OFFENSIVE) │  FN(OFF_2) │
 TRUTH)   (1)     ├─────────────┼───────────────┼────────────┤
         HATE (2) │  FN(HATE_0) │   FN(HATE_1)  │  TP(HATE)  │
                  └─────────────┴───────────────┴────────────┘
```

**Các vị trí nhầm lẫn quan trọng cần phân tích:**
1. **$FN(HATE\_0)$:** Mẫu thực tế là HATE nhưng mô hình đoán là CLEAN. Đây là lỗi nghiêm trọng nhất của hệ thống an toàn thông tin (bỏ lọt thù ghét).
2. **$FP(HATE\_0)$:** Mẫu thực tế là CLEAN nhưng mô hình đoán là HATE (phạt oan người dùng sạch).
3. **$FN(OFF\_2)$ & $FN(HATE\_1)$:** Sự nhầm lẫn giữa hai lớp độc hại (OFFENSIVE $\leftrightarrow$ HATE). Sự nhầm lẫn này ít nguy hiểm hơn việc nhầm lẫn với CLEAN, vì cả hai đều thuộc nhóm nội dung cần kiểm soát.
