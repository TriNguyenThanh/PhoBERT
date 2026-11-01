# 02. KIẾN TRÚC MÔ HÌNH VÀ PHƯƠNG PHÁP XỬ LÝ MẤT CÂN BẰNG LỚP

Tài liệu này trình bày chi tiết về kiến trúc mô hình ngôn ngữ **PhoBERT**, cơ chế tách từ tiếng Việt (**Word Segmentation**), và cơ sở lý thuyết toán học của các phương pháp giải quyết bài toán mất cân bằng lớp (**Class Imbalance**) trong đề tài Phân loại Ngôn từ thù ghét trên bộ dữ liệu ViHSD.

---

## 1. Tổng quan Kiến trúc PhoBERT

### 1.1. Mô hình tiền huấn luyện PhoBERT
- **Nguồn gốc:** PhoBERT ([Nguyen & Nguyen, 2020](https://aclanthology.org/2020.findings-emnlp.92/)) là mô hình ngôn ngữ tiền huấn luyện theo kiến trúc Transformer Encoder dựa trên **RoBERTa** (Robustly Optimized BERT Approach), được tối ưu hóa đặc biệt cho tiếng Việt.
- **Dữ liệu huấn luyện:** PhoBERT được tiền huấn luyện trên kho ngữ liệu tiếng Việt quy mô lớn khoảng 20GB văn bản (khoảng 3 tỷ từ đơn/từ ghép) thu thập từ Báo Mới và Wikipedia tiếng Việt.
- **Phiên bản sử dụng:** `vinai/phobert-base-v2` (hoặc `vinai/phobert-base`):
  - Số lớp ẩn (Hidden layers - $L$): 12
  - Kích thước vector ẩn (Hidden size - $H$): 768
  - Số đầu Attention (Self-attention heads - $A$): 12
  - Tổng số tham số (Parameters): ~135M tham số.

### 1.2. Thách thức ngôn ngữ và Kỹ thuật Tách từ (Word Segmentation)
Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ vựng không đồng nhất với khoảng trắng:
- Ví dụ: cụm từ *"học sinh học sinh học"* bao gồm hai từ ghép: *"học sinh"* (danh từ) và *"sinh học"* (danh từ/tính từ).
- **Yêu cầu bắt buộc của PhoBERT:** PhoBERT áp dụng kỹ thuật tách từ ở cấp độ từ (word-level) trước khi áp dụng thuật toán mã hóa Byte-Pair Encoding (BPE). Do đó, các từ ghép tiếng Việt cần được nối với nhau bằng ký tự gạch dưới `_`.
- **Ví dụ biến đổi:**
  $$\text{"Học sinh đi học"} \xrightarrow{\text{Word Segmentation}} \text{"Học_sinh đi học"}$$
- **Công cụ tách từ tiêu chuẩn:**
  1. `RDRSegmenter` trong bộ công cụ **VnCoreNLP** (chính xác nhất, theo chuẩn của tác giả PhoBERT).
  2. Thư viện `pyvi` (`ViTokenizer.tokenize`) - giải pháp gọn nhẹ, tốc độ cao, không phụ thuộc Java trên môi trường Google Colab.

### 1.3. Cấu trúc Mô hình Phân loại (Classification Architecture)
Mô hình hoàn chỉnh cho bài toán 3 lớp gồm hai thành phần:

1. **Backbone (PhoBERT-base Encoder):**
   - Đầu vào là chuỗi token: `<s> w_1 w_2 ... w_n </s>`.
   - Đầu ra là ma trận biểu diễn ngữ cảnh $H \in \mathbb{R}^{T \times 768}$ (với $T \le 128$).
   - Vector biểu diễn câu lấy từ token phân loại đặc biệt `<s>` (tương đương token `[CLS]` trong BERT): $\mathbf{h}_{\text{<s>}} \in \mathbb{R}^{768}$.

2. **Classification Head:**
   - **Dropout layer:** Tỷ lệ $p = 0.1$ nhằm giảm thiểu hiện tượng Overfitting.
   - **Linear Projection:** Ánh xạ từ không gian đặc trưng ẩn 768 chiều sang không gian 3 chiều tương ứng 3 nhãn:
     $$\mathbf{z} = \mathbf{W} \cdot \text{Dropout}(\mathbf{h}_{\text{<s>}}) + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{3 \times 768}, \mathbf{b} \in \mathbb{R}^{3}$$
   - **Softmax Layer:** Chuyển đổi logits $\mathbf{z}$ thành phân phối xác suất dự đoán $\hat{\mathbf{y}} = [\hat{y}_0, \hat{y}_1, \hat{y}_2]$:
     $$\hat{y}_c = P(y = c \mid x) = \frac{\exp(z_c)}{\sum_{k=0}^{2} \exp(z_k)}$$

---

## 2. Lý thuyết Xử lý Mất cân bằng Lớp

Trên tập dữ liệu ViHSD, tỷ lệ phân bố nhãn rất chênh lệch:
- `CLEAN (0)`: ~82.7%
- `OFFENSIVE (1)`: ~6.8%
- `HATE (2)`: ~10.5%

### 2.1. Giới hạn của hàm mất mát Cross-Entropy tiêu chuẩn
Hàm mất mát Cross-Entropy (CE) thông thường tính trung bình lỗi trên toàn bộ $N$ mẫu trong mini-batch:

$$\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=0}^{2} y_{i,c} \log(\hat{y}_{i,c}) = - \frac{1}{N} \sum_{i=1}^{N} \log(\hat{y}_{i, y_i})$$

Trong đó:
- $y_i \in \{0, 1, 2\}$ là nhãn thực tế của mẫu thứ $i$.
- $\hat{y}_{i, y_i}$ là xác suất dự đoán của mô hình dành cho đúng nhãn thực tế $y_i$.

**Hệ quả khi mất cân bằng:**
Khi 83% mẫu thuộc lớp 0, tổng gradient $\nabla_{\mathbf{W}} \mathcal{L}_{\text{CE}}$ bị áp đảo bởi các mẫu lớp 0:
$$\nabla_{\mathbf{W}} \mathcal{L}_{\text{CE}} \approx 0.83 \cdot \nabla \mathcal{L}_{\text{CLEAN}} + 0.07 \cdot \nabla \mathcal{L}_{\text{OFFENSIVE}} + 0.10 \cdot \nabla \mathcal{L}_{\text{HATE}}$$
Điều này làm mô hình học thiên lệch về việc dự đoán mọi văn bản đều là `CLEAN`, dẫn đến chỉ số **Recall** của các lớp độc hại `OFFENSIVE` và `HATE` rơi xuống mức rất thấp.

---

### 2.2. Phương pháp 1: Weighted Cross-Entropy Loss (Class Weight)

Weighted Cross-Entropy gán một hệ số phạt $w_c > 0$ cho từng lớp $c$. Khi mô hình dự đoán sai một mẫu thuộc lớp thiểu số, tổn thất bị nhân lên gấp nhiều lần, buộc gradient cập nhật trọng số mạnh mẽ hơn để nhận diện các lớp này.

#### Công thức hàm mất mát có trọng số:
$$\mathcal{L}_{\text{WCE}} = - \frac{1}{N} \sum_{i=1}^{N} w_{y_i} \log(\hat{y}_{i, y_i})$$

#### Cách tính trọng số cân bằng (Balanced Class Weighting)
Trọng số được tính toán dựa trên tần suất nghịch đảo của các lớp trên tập huấn luyện (Train set) theo chuẩn thư viện `scikit-learn`:

$$w_c = \frac{N_{\text{total}}}{C \times N_c}$$

Trong đó:
- $N_{\text{total}}$: Tổng số mẫu trên tập Train ($N_{\text{total}} = 24,048$).
- $C$: Tổng số lớp ($C = 3$).
- $N_c$: Số lượng mẫu của lớp $c$ trên tập Train.

#### Bảng giá trị trọng số tính toán từ tập Train ViHSD:

1. **Trên tập Train thô ban đầu ($N_{\text{total}} = 24,048$ mẫu):**

| Lớp ($c$) | Tên nhãn | Số mẫu ($N_c$) | Tỷ lệ (%) | Công thức tính $w_c$ | Trọng số lý thuyết ($w_c$) |
|:---:|:---|:---:|:---:|:---|:---:|
| **0** | CLEAN | 19,886 | 82.69% | $\frac{24048}{3 \times 19886}$ | **~ 0.4031** |
| **1** | OFFENSIVE | 1,606 | 6.68% | $\frac{24048}{3 \times 1606}$ | **~ 4.9913** |
| **2** | HATE | 2,556 | 10.63% | $\frac{24048}{3 \times 2556}$ | **~ 3.1365** |

2. **Trên tập Train sạch sau tiền xử lý & khử trùng lặp/rò rỉ ($N_{\text{clean}} = 21,098$ mẫu):**

| Lớp ($c$) | Tên nhãn | Số mẫu sạch ($N_c$) | Tỷ lệ (%) | Trọng số thực nghiệm ($w_c$) | Mức phạt so với CLEAN |
|:---:|:---|:---:|:---:|:---:|:---:|
| **0** | CLEAN | 17,258 | 81.80% | **0.4075** | $1.0\times$ (Mốc cơ sở) |
| **1** | OFFENSIVE | 1,448 | 6.86% | **4.8568** | **~11.9 lần** ($4.8568 / 0.4075$) |
| **2** | HATE | 2,392 | 11.34% | **2.9401** | **~7.2 lần** ($2.9401 / 0.4075$) |

**Nhận xét:**
- Một mẫu sai sót ở lớp `OFFENSIVE` sẽ bị phạt nặng gấp **~11.9 lần** so với một mẫu lớp `CLEAN`.
- Một mẫu sai sót ở lớp `HATE` bị phạt nặng gấp **~7.2 lần** so với mẫu lớp `CLEAN`.
- Cơ chế này buộc mô hình phải đặc biệt chú trọng việc phát hiện các câu thuộc 2 lớp thiểu số độc hại, hạn chế tối đa việc bỏ sót (tối ưu hóa chỉ số Recall).

---

### 2.3. Phương pháp 2: Focal Loss (Lựa chọn mở rộng)

Focal Loss ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)) được thiết kế để giải quyết cả hai bài toán: mất cân bằng lớp và mất cân bằng độ khó của mẫu (Hard vs. Easy examples).

#### Công thức toán học:
$$\mathcal{L}_{\text{FL}}(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$

Trong đó:
- $p_t$: Xác suất dự đoán của mô hình cho nhãn đúng ($p_t = \hat{y}_{y_i}$).
- $\alpha_t$: Trọng số cân bằng lớp (tương tự như class weight).
- $\gamma \ge 0$: Tham số điều chỉnh tiêu điểm (Focusing parameter), thường chọn $\gamma = 2.0$.
- $(1 - p_t)^\gamma$: Hệ số điều biến (Modulating factor):
  - Với mẫu dễ phân loại ($p_t \to 1$): $(1 - p_t)^\gamma \to 0$, hàm mất mát triệt tiêu, gradient không bị lãng phí cho mẫu đã học tốt.
  - Với mẫu khó phân loại ($p_t \to 0$): $(1 - p_t)^\gamma \to 1$, hàm mất mát giữ nguyên giá trị cao, tập trung cập nhật gradient vào mẫu khó.

---

## 3. Ma trận So sánh Kỹ thuật Giải quyết Mất cân bằng Lớp

| Tiêu chí | Cross-Entropy tiêu chuẩn | Weighted Cross-Entropy (Class Weight) | Focal Loss |
|:---|:---|:---|:---|
| **Cơ chế chính** | Đối xử bình đẳng mọi mẫu | Phạt nặng theo tần suất nghịch đảo lớp | Phạt theo lớp + Giảm trọng số mẫu dễ |
| **Độ phức tạp cài đặt** | Tích hợp sẵn (`nn.CrossEntropyLoss`) | Truyền tensor trọng số vào `weight` | Cần viết custom Loss function |
| **Số siêu tham số bổ sung** | 0 | 0 (tính trực tiếp từ tập Train) | 2 ($\alpha, \gamma$) |
| **Tác động đến Recall lớp hiếm** | Thấp (thường bỏ sót HATE/OFFENSIVE) | **Tăng mạnh** | Tăng mạnh |
| **Tác động đến Precision lớp hiếm** | Cao (chỉ đoán khi rất chắc chắn) | Giảm nhẹ (chấp nhận đoán nhầm để bao phủ) | Cân bằng hơn |
| **Độ phù hợp với bài toán này** | Baseline đối chứng | **Phương pháp đề xuất chính** | Phương pháp mở rộng nâng cao |

---

## 4. Tóm tắt Luồng Xử lý Dữ liệu và Mô hình

```
Văn bản thô (Raw text)
       │
       ▼
[Chuẩn hóa ký tự, URLs, khoảng trắng]
       │
       ▼
[Tách từ tiếng Việt: PyVi / RDRSegmenter]
       │
       ▼
[PhoBERT Tokenizer: BPE Tokenization, max_length=128, padding, truncation]
       │
       ▼
[PhoBERT-base Encoder: 12 Transformer Layers]
       │
       ▼
[Pooled Representation: Vector h_<s> (768 chiều)]
       │
       ▼
[Dropout (p=0.1) + Linear Layer (768 -> 3)]
       │
       ▼
[Logits z: 3 giá trị]
       │
       ├─────────────────────────────────────────┐
       ▼ (Giai đoạn Huấn luyện)                  ▼ (Giai đoạn Đánh giá/Dự đoán)
[Weighted Cross-Entropy Loss]             [Softmax -> Argmax]
       │                                         │
       ▼                                         ▼
[Gradient Backpropagation & AdamW]         [Nhãn dự đoán: 0, 1 hoặc 2]
```
