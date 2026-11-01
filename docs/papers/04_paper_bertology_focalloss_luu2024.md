# 04. PIPELINE & PHƯƠNG PHÁP: FOCAL LOSS & BERTOLOGY TRÊN ViHSD (LUU ET AL., 2023/2024)

---

## 1. Thông tin Tổng quan Bài báo
- **Tên bài báo:** *An approach of data augmentation to improve the performance of BERTology models for Vietnamese hate speech detection*
- **Tác giả:** Son T. Luu, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen (UIT-NLP Lab, Trường ĐH Công nghệ Thông tin - ĐHQG TP.HCM).
- **Tạp chí công bố:** *Multimedia Tools and Applications* (Springer, 2024; trực tuyến 2023).
- **Tập & Trang:** Vol. 83, pp. 56763–56783.
- **Chỉ số khoa học:** ISI / Scopus Q1/Q2, Impact Factor ~3.6.
- **DOI:** [10.1007/s11042-023-16968-5](https://doi.org/10.1007/s11042-023-16968-5)
- **Bản tiền ấn bản (Preprint liên quan):** *Impacts of Transformer-Based Language Models and Imbalanced Data for Hate Speech Detection...* (Research Square, DOI: [10.21203/rs.3.rs-2242843/v1](https://doi.org/10.21203/rs.3.rs-2242843/v1)).

---

## 2. Mục tiêu Nghiên cứu & Bối cảnh Học thuật
Bài báo này là công trình nghiên cứu toàn diện nhất của chính nhóm tác giả ViHSD nhằm trả lời hai câu hỏi phương pháp luận lớn:
1. Các kỹ thuật tiền xử lý (như xóa stopword) và tăng cường dữ liệu truyền thống (EDA) có thực sự hiệu quả đối với các mô hình Transformer hiện đại (họ BERTology) trên tiếng Việt hay không?
2. Trong bài toán phân loại đa lớp mất cân bằng (ViHSD), việc điều chỉnh hàm mất mát ở cấp độ thuật toán bằng **Focal Loss** mang lại hiệu quả vượt trội như thế nào so với Cross-Entropy tiêu chuẩn?

---

## 3. Sơ đồ Pipeline Thực nghiệm của Nghiên cứu

```
[Bộ dữ liệu ViHSD (33,400 mẫu)]
               │
               ▼
[Khảo sát Tiền xử lý: Bảo toàn Ngữ pháp vs. Xóa Stopword/Dấu câu]
   (Kết luận: KHÔNG xóa stopword, KHÔNG xóa dấu câu)
               │
               ▼
[Đánh giá Phương pháp Xử lý Mất cân bằng Lớp]
   ├──────────────────────────────┬──────────────────────────────┐
   ▼                              ▼                              ▼
[Nhánh 1: Baseline]       [Nhánh 2: Data-level]       [Nhánh 3: Algorithm-level]
Standard Cross-Entropy    Modified EDA Augmentation   Multi-class Focal Loss (gamma=2.0)
   │                              │                              │
   └──────────────────────────────┼──────────────────────────────┘
                                  ▼
[Khảo sát trên các Mô hình BERTology Tiếng Việt & Đa ngữ]
   - PhoBERT-base & PhoBERT-large (Nguyen & Nguyen, 2020)
   - viBERT4news (Hà & cộng sự)
   - vELECTRA (VinAI Research)
   - XLM-RoBERTa (Conneau et al., 2020)
                                  │
                                  ▼
[Đánh giá Đa chiều: Macro F1, Weighted F1, Per-class F1 trên ViHSD]
```

---

## 4. Các Khám phá Phương pháp luận Đột phá

### 4.1. "Nghịch lý" Tiền xử lý: Tại sao không được xóa Stopwords?
Nghiên cứu chỉ ra sự khác biệt sâu sắc giữa mô hình Deep Learning truyền thống (Text-CNN, GRU) và mô hình Transformer:
- **Với Text-CNN / GRU:** Xóa từ dừng (stopwords) giúp giảm chiều không gian từ vựng và loại bỏ nhiễu, làm tăng nhẹ hiệu năng.
- **Với các mô hình BERTology (PhoBERT):** Việc xóa từ dừng **làm giảm hiệu năng mô hình**:
  - Cơ chế Self-Attention phụ thuộc vào trật tự từ và các hư từ quan hệ (*"nhưng"*, *"thì"*, *"là"*, *"mà"*) để giải mã mối quan hệ phụ thuộc xa giữa chủ ngữ và vị ngữ.
  - Xóa từ dừng làm đứt gãy mạch cú pháp của câu, khiến mô hình không phân biệt được câu tự trách mình với câu công kích người khác.

### 4.2. Giới hạn của Tăng cường Dữ liệu (EDA) trên Transformer
- Thao tác xóa từ ngẫu nhiên (Random Deletion) hoặc tráo từ (Random Swap) trong EDA nguyên bản làm phá vỡ biểu diễn ngữ cảnh tiền huấn luyện mà PhoBERT đã học từ 20GB văn bản tiếng Việt.
- Chỉ khi áp dụng phương pháp EDA cải tiến có kiểm soát ngữ nghĩa (Contextual EDA), Macro F1 mới tăng thêm được khoảng **~1.5%**.

---

## 5. Cơ sở Lý thuyết & Cài đặt Multi-class Focal Loss

### 5.1. Công thức Toán học
Để giải quyết sự mất cân bằng ở cấp độ thuật toán, tác giả đưa **Focal Loss** vào bài toán phân loại đa lớp:

$$\mathcal{L}_{\text{Focal}} = - \frac{1}{B} \sum_{i=1}^B \alpha_{y_i} (1 - p_{i, y_i})^\gamma \log(p_{i, y_i})$$

Trong đó:
- $p_{i, y_i} \in [0, 1]$: Xác suất dự đoán của mô hình cho nhãn đúng $y_i$.
- $\gamma \ge 0$: Tham số tiêu điểm (Focusing parameter). Nghiên cứu khẳng định $\gamma = 2.0$ là giá trị tối ưu.
- $\alpha_{y_i}$: Trọng số cân bằng lớp.

### 5.2. Tại sao Focal Loss đặc biệt vượt trội trên Phân loại Đa lớp (ViHSD)?
- **Cơ chế tự động triệt tiêu mất mát mẫu dễ:** Lớp `CLEAN` chiếm 82.7% dữ liệu. Sau vài trăm bước huấn luyện, PhoBERT đã phân loại đúng hầu hết các câu CLEAN đơn giản với xác suất $p_t \ge 0.98$. Khi đó:
  $$(1 - p_t)^2 = (1 - 0.98)^2 = 0.0004$$
  Đóng góp mất mát của mẫu dễ này bị giảm đi **2,500 lần**!
- **Dồn năng lực tối ưu vào mẫu khó:** Các câu thuộc lớp `OFFENSIVE` (từ lóng mới) hoặc `HATE` (thù ghét ẩn dụ, châm biếm) có xác suất $p_t$ thấp ($p_t \approx 0.15$), nhân tử $(1 - 0.15)^2 = 0.72$ giữ nguyên gradient lớn, buộc mô hình phải tập trung học ranh giới phân tách các lớp này.
- **Kết luận của bài báo:** Focal Loss thể hiện tính bền vững và hiệu quả trên bài toán phân loại đa lớp (ViHSD) cao hơn rất nhiều so với khi áp dụng cho bài toán phân loại nhị phân.

---

## 6. Kết quả Thực nghiệm Điển hình trên ViHSD

| Kiến trúc Mô hình | Hàm mất mát / Chiến lược | Macro F1 (%) |
|:---|:---|:---:|
| PhoBERT-base | Standard Cross-Entropy | 63.85 |
| PhoBERT-base | EDA Tăng cường dữ liệu | 65.35 |
| **PhoBERT-base** | **Multi-class Focal Loss ($\gamma = 2.0$)** | **66.85** |
| PhoBERT-large | Standard Cross-Entropy | 64.92 |
| **PhoBERT-large** | **Multi-class Focal Loss ($\gamma = 2.0$)** | **67.62** |

---

## 7. Bài học Kế thừa Trực tiếp cho Đề tài của chúng ta

1. **Khẳng định tính đúng đắn của phương pháp cấp độ thuật toán:** Việc sử dụng hàm mất mát có trọng số (Class Weight / Focal Loss) thay vì nhân bản dữ liệu nhân tạo (EDA) là hoàn toàn phù hợp với lý thuyết hiện đại về mô hình ngôn ngữ lớn tiếng Việt.
2. **Quy tắc tiền xử lý sạch:** Trong code pipeline, chúng ta chỉ chuẩn hóa Unicode NFC, thu gọn ký tự kéo dài và tách từ bằng PyVi; tuyệt đối không xóa stopword hay bỏ dấu câu, đúng theo khuyến nghị của Luu et al. (2024).
3. **Cơ sở học thuật bảo vệ đồ án:** Toàn bộ lý giải về việc tại sao chọn Class Weight / Focal Loss trong báo cáo của bạn đều có thể viện dẫn trực tiếp từ bài báo này trên tạp chí quốc tế Springer.
