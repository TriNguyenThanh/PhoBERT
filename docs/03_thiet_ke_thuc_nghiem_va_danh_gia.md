# 03. THIẾT KẾ THỰC NGHIỆM VÀ HỆ THỐNG ĐÁNH GIÁ

Tài liệu này quy định tiêu chuẩn thiết lập môi trường phần cứng, cấu hình siêu tham số (Hyperparameters), quy trình thực nghiệm đối chứng A/B, chiến lược kiểm định đa seed và hệ thống chỉ số đánh giá khoa học cho bài toán Phát hiện ngôn từ thù ghét trên bộ dữ liệu ViHSD.

---

## 1. Môi trường Thực nghiệm Phần cứng & Phần mềm

### 1.1. Hạ tầng Tính toán (Google Colab GPU T4)
- **Bộ xử lý đồ họa (GPU):** NVIDIA Tesla T4 (16GB GDDR6 VRAM, kiến trúc Turing, Compute Capability 7.5).
- **Hệ điều hành:** Linux x86_64 (Ubuntu môi trường máy ảo Colab).
- **Cơ chế tăng tốc độ chính xác hỗn hợp (FP16 - Mixed Precision):**
  - **Kỹ thuật:** Sử dụng tính toán độ chính xác 16-bit (Half Precision Floating Point) kết hợp Dynamic Loss Scaling thông qua PyTorch AMP (`accelerate`).
  - **Lợi ích kỹ thuật:**
    - Giảm tiêu thụ bộ nhớ đồ họa (VRAM) khoảng **40% - 50%**.
    - Tăng tốc độ tính toán ma trận thông qua các nhân Tensor Cores trên GPU T4 (tốc độ xử lý nhanh hơn 2x so với FP32 thông thường).
    - Ngăn ngừa hiện tượng tràn số dưới (underflow) trong quá trình lan truyền ngược gradient.

### 1.2. Thư viện Phần mềm Cốt lõi
- **PyTorch:** $\ge 2.0.0$ (hỗ trợ native Mixed Precision và GPU CUDA).
- **Hugging Face Transformers:** $\ge 4.38.0$ (quản lý mô hình PhoBERT-base-v2 và bộ Tokenizer BPE).
- **Hugging Face Accelerate:** $\ge 0.33.0$ (tối ưu hóa luồng nạp batch và phân bổ tài nguyên bộ nhớ).
- **Scikit-learn:** Tính toán trọng số lớp cân bằng (`compute_class_weight`) và trích xuất chỉ số đánh giá (`classification_report`, `f1_score`, `confusion_matrix`).
- **PyVi:** Tách từ tiếng Việt (`ViTokenizer`) gộp từ ghép bằng ký tự `_`.

---

## 2. Thiết lập Siêu tham số Huấn luyện (Hyperparameters)

Các siêu tham số được căn chỉnh nhằm bảo đảm tính hội tụ tối ưu của mô hình Transformer trên tập ngữ liệu ViHSD:

| Siêu tham số (Hyperparameter) | Giá trị thiết lập | Cơ sở khoa học & Lý do kỹ thuật |
|:---|:---:|:---|
| **Mô hình nền tảng (Backbone)** | `vinai/phobert-base-v2` | Phiên bản RoBERTa tối ưu mới nhất cho tiếng Việt (12 layers, 768 hidden, 135M tham số). |
| **Độ dài chuỗi tối đa (Max Length)** | **128** | Thống kê ViHSD cho thấy hơn 98.5% bình luận có độ dài dưới 128 subwords. Giới hạn này triệt tiêu chi phí đệm thừa và tối ưu bộ nhớ GPU. |
| **Batch Size** | **16** | Đệm động theo từng batch (`DataCollatorWithPadding`), bảo đảm không bị tràn bộ nhớ VRAM (OOM) trên GPU T4. |
| **Số Epochs huấn luyện** | **4** | Điểm hội tụ lý tưởng của PhoBERT trên tập dữ liệu ~21k mẫu; tránh hiện tượng học vẹt (Overfitting) khi huấn luyện quá dài. |
| **Tốc độ học (Learning Rate)** | **$2 \times 10^{-5}$** ($2e-5$) | Tốc độ học chuẩn khi tinh chỉnh (Fine-tuning) mô hình Transformer, tránh phá vỡ trọng số tiền huấn luyện. |
| **Bộ tối ưu (Optimizer)** | **AdamW** | Cải tiến từ Adam với cơ chế phân tách suy giảm trọng số độc lập (`weight_decay = 0.01`, $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$). |
| **Chiến lược điều chỉnh LR (Scheduler)**| **Linear Warmup with Decay** | Khởi động nhẹ trong 10% tổng số bước đầu (Warmup ratio = 0.1), sau đó giảm tuyến tính về 0. |
| **Cắt gradient (Gradient Clipping)** | **1.0** | Giới hạn chuẩn gradient tối đa (`max_grad_norm = 1.0`) để ổn định quá trình tối ưu. |
| **Chiến lược chọn mô hình tốt nhất** | **Best Macro F1 trên tập Dev** | Đánh giá sau mỗi epoch trên tập Validation (2,672 mẫu), lưu lại checkpoint có Macro F1 cao nhất để suy luận trên Test set. |

---

## 3. Kịch bản Thực nghiệm Đối chứng A/B

Để chứng minh tính hiệu quả của phương pháp giải quyết mất cân bằng lớp, đề tài triển khai hai kịch bản thực nghiệm đối chứng nghiêm ngặt:

```
                            Tập dữ liệu ViHSD (33,400 mẫu)
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
 [Tập Train sạch (21,098 mẫu)]                               [Tập Test chuẩn (6,680 mẫu)]
            │                                                           │
            ├─────────────────────────────┬─────────────────────────────┤
            ▼                             ▼                             ▼
  [Kịch bản 1: Mô hình A]       [Kịch bản 2: Mô hình B]         [Đánh giá Độc lập]
      PhoBERT-base-v2               PhoBERT-base-v2                     │
            +                             +                             │
   Standard Cross-Entropy        Balanced Class Weight                  │
     (Mô hình Đối chứng)         (Sản phẩm Đề xuất Chính)               │
            │                             │                             │
            └──────────────┬──────────────┘                             │
                           ▼                                            ▼
               [4 Random Seeds: 42, 123, 456, 789] ────────► [Báo cáo: Mean ± Std (μ ± σ)]
```

### 3.1. Kịch bản 1: Mô hình A (Baseline Đối chứng)
- **Cấu hình:** `PhoBERT-base-v2` + Tiền xử lý teencode 414 từ + Tách từ `PyVi` + **Standard Cross-Entropy Loss** (không gán trọng số lớp).
- **Hàm mất mát:** `nn.CrossEntropyLoss()` mặc định.
- **Vai trò trong nghiên cứu:** Thiết lập mốc đối chứng cơ sở nhằm chứng minh hạn chế của hàm mất mát tiêu chuẩn khi đối mặt với dữ liệu mất cân bằng nhãn.
- **Dự báo lý thuyết:** Accuracy tổng thể cao (~88%) do bắt trúng lớp đa số `CLEAN`, nhưng chỉ số Recall của hai lớp thiểu số `OFFENSIVE` và `HATE` thấp do bị áp đảo gradient.

### 3.2. Kịch bản 2: Mô hình B (Sản phẩm Nghiên cứu Đề xuất Cốt lõi)
- **Cấu hình:** `PhoBERT-base-v2` + Tiền xử lý teencode 414 từ + Tách từ `PyVi` + **Balanced Class Weighting**.
- **Hàm mất mát:** `nn.CrossEntropyLoss(weight=weights_tensor)`.
- **Bộ trọng số cân bằng:** Được tính toán độc quyền từ tập Train sạch ($N = 21,098$):
  $$w_c = \frac{N}{3 \times N_c} \implies w = [w_{\text{CLEAN}} \approx 0.41, \; w_{\text{OFFENSIVE}} \approx 4.86, \; w_{\text{HATE}} \approx 2.94]$$
- **Vai trò trong nghiên cứu:** Đây là **sản phẩm cốt lõi** của đề tài nhằm giải quyết triệt để vấn đề mất cân bằng dữ liệu.
- **Dự báo lý thuyết:** Mức phạt mất mát cao cho các lớp thiểu số buộc mô hình phải dịch chuyển ranh giới quyết định, giúp chỉ số **Recall của HATE và OFFENSIVE tăng vọt**, giảm thiểu tối đa các ca bỏ sót nội dung độc hại nguy hiểm.

---

## 4. Chiến lược Đánh giá Đa Seed (Multi-Seed Robustness Verification)

Nhằm đảm bảo tính khoa học và loại trừ hoàn toàn yếu tố "may mắn ngẫu nhiên" (Lucky Seed), đề tài thực hiện quy trình kiểm định trên **4 hạt giống ngẫu nhiên độc lập**:

| Hạt giống (Seed) | Thành viên phụ trách | Mục tiêu thực nghiệm | Thư mục đầu ra |
|:---:|:---|:---|:---|
| **42** | **Trí** | Mốc thực nghiệm chuẩn (Baseline Reference) | `outputs/seed42_teencode_414/` |
| **123** | **Ngân** | Kiểm định độ ổn định & Thực nghiệm bóc tách (Ablation) | `outputs/seed123_teencode_414/` |
| **456** | **Hiếu** | Kiểm chứng độc lập đa seed | `outputs/seed456_teencode_414/` |
| **789** | **Tài** | Kiểm chứng độc lập đa seed | `outputs/seed789_teencode_414/` |

**Phương pháp báo cáo:** Toàn bộ các chỉ số được tổng hợp dưới dạng **Giá trị trung bình và Độ lệch chuẩn ($\mu \pm \sigma$)** qua 4 lần chạy:
$$\mu = \frac{1}{K} \sum_{k=1}^{K} m_k, \quad \sigma = \sqrt{\frac{1}{K-1} \sum_{k=1}^{K} (m_k - \mu)^2} \quad (K = 4)$$

---

## 5. Hệ thống Chỉ số Đánh giá Định lượng (Evaluation Metrics)

Do tập dữ liệu ViHSD mất cân bằng nghiêm trọng (lớp `CLEAN` chiếm >82.7%), **Accuracy là chỉ số có tính chất gây nhiễu**, không phản ánh đúng năng lực nhận diện của mô hình. Hệ thống đánh giá của đề tài dựa trên 4 trụ cột định lượng chuẩn mực:

### 5.1. Macro F1-score (MF1) — Thước đo Khoa học Trọng yếu
- **Công thức:**
  $$MF1 = \frac{F1_{\text{CLEAN}} + F1_{\text{OFFENSIVE}} + F1_{\text{HATE}}}{3}$$
- **Ý nghĩa:** Macro F1 tính trung bình cộng không trọng số giữa 3 lớp, gán mức độ quan trọng ngang nhau cho cả 3 lớp: `CLEAN` (82.7%), `OFFENSIVE` (6.8%) và `HATE` (10.5%).
- **Vai trò:** Thước đo quyết định để đánh giá chất lượng mô hình trên các bảng xếp hạng NLP quốc tế (SemEval, EMNLP, ACL).

### 5.2. Weighted F1-score (WF1)
- **Công thức:**
  $$WF1 = \sum_{c=0}^{2} \left( \frac{N_c}{N_{\text{total}}} \times F1_c \right) \approx 0.83 \cdot F1_0 + 0.07 \cdot F1_1 + 0.10 \cdot F1_2$$
- **Ý nghĩa:** Phản ánh hiệu năng tổng thể của hệ thống tính theo xác suất xuất hiện thực tế của các loại bình luận trên mạng xã hội.

### 5.3. Bộ ba Chỉ số Từng Lớp: Precision, Recall và F1-Score
Đối với từng lớp $c \in \{\text{CLEAN, OFFENSIVE, HATE}\}$:

| Chỉ số | Công thức | Ý nghĩa thực tiễn trong bài toán Kiểm duyệt mạng xã hội |
|:---|:---:|:---|
| **Precision ($P_c$)** | $\frac{TP_c}{TP_c + FP_c}$ | Tỷ lệ những bình luận mô hình **báo là vi phạm** thực sự có vi phạm. Precision thấp dẫn đến báo động giả, gây khó chịu cho người dùng bình thường. |
| **Recall ($R_c$)** | $\frac{TP_c}{TP_c + FN_c}$ | Tỷ lệ những bình luận **thực tế vi phạm** được mô hình phát hiện thành công. Recall thấp đồng nghĩa với việc bỏ lọt ngôn từ thù ghét ra không gian mạng. |
| **F1-score ($F1_c$)** | $2 \cdot \frac{P_c \cdot R_c}{P_c + R_c}$ | Trung bình điều hòa giữa Precision và Recall của lớp $c$. |

> **Quy luật Đánh đổi (Precision-Recall Trade-off):** Khi đưa Class Weight vào huấn luyện, mô hình chủ động chấp nhận giảm nhẹ một lượng Precision để đổi lấy sự tăng trưởng vượt bậc của **Recall**. Trong bài toán kiểm duyệt an toàn nội dung, **ưu tiên Recall cao là mục tiêu sống còn** nhằm giảm thiểu tối đa các phát ngôn thù ghét độc hại bị bỏ lọt.

### 5.4. Ma trận Nhầm lẫn (Confusion Matrix $3 \times 3$)
Trực quan hóa chi tiết mối tương quan giữa nhãn thực tế (Ground Truth) và nhãn dự đoán (Predicted Label) trên 6,680 mẫu Test:

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

**Các vị trí nhầm lẫn then chốt cần phân tích:**
1. **$FN(HATE\_0)$:** Mẫu thực tế là HATE nhưng mô hình đoán nhầm thành CLEAN (bỏ lọt ngôn từ thù ghét — lỗi nguy hiểm nhất).
2. **$FP(HATE\_0)$:** Mẫu thực tế là CLEAN nhưng mô hình đoán nhầm thành HATE (nghi ngờ sai người dùng sạch).
3. **$FN(OFF\_2)$ & $FN(HATE\_1)$:** Nhầm lẫn nội bộ giữa hai lớp vi phạm ($OFFENSIVE \leftrightarrow HATE$). Đây là dạng nhầm lẫn chấp nhận được trong kiểm duyệt, vì cả hai lớp đều được hệ thống gắn cờ vi phạm để tiếp tục xử lý.
