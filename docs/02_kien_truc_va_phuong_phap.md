# 02. KIẾN TRÚC MÔ HÌNH VÀ PHƯƠNG PHÁP XỬ LÝ MẤT CÂN BẰNG LỚP

Tài liệu này trình bày chi tiết về kiến trúc mô hình ngôn ngữ **PhoBERT-base-v2**, cơ chế tách từ tiếng Việt (**Word Segmentation**), quy trình tiền xử lý dữ liệu và cơ sở lý thuyết toán học của kỹ thuật gán trọng số lớp (**Balanced Class Weighting**) trong bài toán Phát hiện ngôn từ thù ghét trên bộ dữ liệu ViHSD.

---

## 1. Tổng quan Kiến trúc PhoBERT-base-v2

### 1.1. Mô hình Tiền huấn luyện PhoBERT
- **Nguồn gốc:** PhoBERT ([Nguyen & Nguyen, 2020](https://aclanthology.org/2020.findings-emnlp.92/)) là mô hình ngôn ngữ tiền huấn luyện theo kiến trúc Transformer Encoder dựa trên **RoBERTa** (Robustly Optimized BERT Approach), được tối ưu hóa đặc biệt cho tiếng Việt.
- **Dữ liệu huấn luyện nền tảng:** PhoBERT được tiền huấn luyện trên kho ngữ liệu tiếng Việt quy mô 20GB văn bản (khoảng 3 tỷ từ đơn/từ ghép) thu thập từ Báo Mới và Wikipedia tiếng Việt.
- **Phiên bản sử dụng:** `vinai/phobert-base-v2`:
  - Số lớp ẩn (Hidden layers - $L$): 12
  - Kích thước vector ẩn (Hidden size - $H$): 768
  - Số đầu tự chú ý (Self-attention heads - $A$): 12
  - Tổng số tham số (Parameters): ~135M tham số.

### 1.2. Thách thức Ngôn ngữ và Kỹ thuật Tách từ (Word Segmentation)
Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ vựng không đồng nhất với khoảng trắng:
- Ví dụ: Cụm từ *"học sinh học sinh học"* bao gồm hai từ ghép: *"học sinh"* (danh từ) và *"sinh học"* (danh từ/tính từ).
- **Yêu cầu bắt buộc của PhoBERT:** PhoBERT thực hiện mã hóa Byte-Pair Encoding (BPE) trên cấp độ từ (word-level). Do đó, các từ ghép tiếng Việt bắt buộc phải được nối với nhau bằng ký tự gạch dưới `_` trước khi đưa vào bộ tokenizer.
- **Ví dụ biến đổi:**
  $$\text{"Học sinh đi học"} \xrightarrow{\text{Word Segmentation (PyVi)}} \text{"Học_sinh đi học"}$$
- **Công cụ tách từ:** Sử dụng thư viện `pyvi` (`ViTokenizer.tokenize`) — giải pháp tách từ tiếng Việt gọn nhẹ, tốc độ xử lý nhanh và tương thích hoàn toàn trên môi trường Google Colab GPU.

### 1.3. Cấu trúc Mô hình Phân loại (Classification Architecture)
Hệ thống phân loại 3 lớp gồm hai thành phần chính:

1. **Backbone (PhoBERT-base-v2 Encoder):**
   - Đầu vào là chuỗi token: `<s> w_1 w_2 ... w_n </s>` với độ dài tối đa $T = 128$.
   - Đầu ra là ma trận biểu diễn ngữ cảnh $H \in \mathbb{R}^{T \times 768}$.
   - Vector biểu diễn ngữ cảnh toàn câu lấy từ token phân loại đặc biệt `<s>`: $\mathbf{h}_{\text{<s>}} \in \mathbb{R}^{768}$.

2. **Classification Head:**
   - **Dropout layer:** Tỷ lệ $p = 0.1$ nhằm giảm thiểu hiện tượng Overfitting.
   - **Linear Projection:** Ánh xạ từ không gian vector ẩn 768 chiều sang không gian 3 chiều tương ứng 3 lớp:
     $$\mathbf{z} = \mathbf{W} \cdot \text{Dropout}(\mathbf{h}_{\text{<s>}}) + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{3 \times 768}, \mathbf{b} \in \mathbb{R}^{3}$$
   - **Softmax Layer:** Chuyển đổi logits $\mathbf{z}$ thành phân phối xác suất dự đoán $\hat{\mathbf{y}} = [\hat{y}_0, \hat{y}_1, \hat{y}_2]$:
     $$\hat{y}_c = P(y = c \mid x) = \frac{\exp(z_c)}{\sum_{k=0}^{2} \exp(z_k)}$$

---

## 2. Lý thuyết Xử lý Mất cân bằng Lớp

Phân bố dữ liệu trên tập ViHSD có sự chênh lệch lớn:
- `CLEAN (0)`: ~82.7%
- `OFFENSIVE (1)`: ~6.8%
- `HATE (2)`: ~10.5%

### 2.1. Giới hạn của Hàm Mất mát Cross-Entropy Tiêu chuẩn (Mô hình A - Baseline)
Hàm mất mát Cross-Entropy (CE) thông thường tính trung bình lỗi trên toàn bộ $N$ mẫu trong mini-batch:

$$\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=0}^{2} y_{i,c} \log(\hat{y}_{i,c}) = - \frac{1}{N} \sum_{i=1}^{N} \log(\hat{y}_{i, y_i})$$

Trong đó:
- $y_i \in \{0, 1, 2\}$ là nhãn thực tế của mẫu thứ $i$.
- $\hat{y}_{i, y_i}$ là xác suất dự đoán của mô hình dành cho đúng nhãn thực tế $y_i$.

**Hệ quả khi huấn luyện trên dữ liệu mất cân bằng:**
Tổng gradient $\nabla_{\mathbf{W}} \mathcal{L}_{\text{CE}}$ bị áp đảo bởi các mẫu thuộc lớp đa số `CLEAN`:
$$\nabla_{\mathbf{W}} \mathcal{L}_{\text{CE}} \approx 0.83 \cdot \nabla \mathcal{L}_{\text{CLEAN}} + 0.07 \cdot \nabla \mathcal{L}_{\text{OFFENSIVE}} + 0.10 \cdot \nabla \mathcal{L}_{\text{HATE}}$$

Hệ quả là mô hình có xu hướng thiên lệch về việc dự đoán mọi văn bản đều là `CLEAN` để cực tiểu hóa hàm mất mát nhanh chóng. Điều này khiến chỉ số **Recall** của các lớp độc hại `OFFENSIVE` và `HATE` rất thấp, dẫn đến việc bỏ sót nhiều vi phạm.

---

### 2.2. Phương pháp Đề xuất: Weighted Cross-Entropy Loss (Mô hình B - Proposed)

Để khắc phục hiện tượng trên, **Mô hình B** áp dụng hàm mất mát có trọng số (**Weighted Cross-Entropy Loss**). Mô hình gán một hệ số phạt $w_c > 0$ cho từng lớp $c$ theo nguyên lý nghịch đảo tần suất. Khi mô hình phân loại sai một mẫu thuộc lớp thiểu số, mức tổn thất sẽ bị nhân lên nhiều lần, buộc gradient cập nhật mạnh hơn cho các đặc trưng nhận diện lớp đó.

#### Công thức Hàm Mất mát có Trọng số:
$$\mathcal{L}_{\text{WCE}} = - \frac{1}{N} \sum_{i=1}^{N} w_{y_i} \log(\hat{y}_{i, y_i})$$

#### Công thức Tính Trọng số Lớp Cân bằng (Balanced Class Weighting):
Trọng số được tính toán dựa trên tần suất nghịch đảo của các lớp trên tập huấn luyện sạch ($N_{\text{clean}} = 21,098$) theo công thức chuẩn:

$$w_c = \frac{N_{\text{clean}}}{C \times N_c}$$

Trong đó:
- $N_{\text{clean}}$: Tổng số mẫu trên tập Train sạch ($N_{\text{clean}} = 21,098$).
- $C$: Tổng số lớp phân loại ($C = 3$).
- $N_c$: Số lượng mẫu của lớp $c$ trên tập Train sạch.

#### Bảng Giá trị Trọng số Lớp Thực nghiệm:

| Lớp ($c$) | Tên nhãn khoa học | Số mẫu sạch ($N_c$) | Tỷ lệ (%) | Trọng số thực nghiệm ($w_c$) | Mức phạt tương đối so với CLEAN |
|:---:|:---|:---:|:---:|:---:|:---:|
| **0** | **CLEAN** *(Sạch / Lành mạnh)* | 17,258 | 81.80% | **0.4075** (~0.41) | $1.0\times$ (Mốc chuẩn) |
| **1** | **OFFENSIVE** *(Xúc phạm / Thô tục)* | 1,448 | 6.86% | **4.8568** (~4.86) | **Gấp ~11.9 lần** ($4.8568 / 0.4075$) |
| **2** | **HATE** *(Ngôn từ thù ghét)* | 2,392 | 11.34% | **2.9401** (~2.94) | **Gấp ~7.2 lần** ($2.9401 / 0.4075$) |

**Phân tích tác động:**
- Mẫu phân loại sai thuộc lớp `OFFENSIVE` bị phạt gấp **~11.9 lần** so với mẫu lớp `CLEAN`.
- Mẫu phân loại sai thuộc lớp `HATE` bị phạt gấp **~7.2 lần** so với mẫu lớp `CLEAN`.
- Cơ chế này thúc đẩy mô hình chủ động mở rộng vùng nhận diện cho hai lớp thiểu số, giúp tăng mạnh chỉ số **Recall** trên các bình luận độc hại.

---

### 2.3. So sánh Cơ chế: Mô hình A vs. Mô hình B

| Tiêu chí so sánh | Mô hình A (Baseline Đối chứng) | Mô hình B (Sản phẩm Đề xuất Cốt lõi) |
|:---|:---|:---|
| **Hàm mất mát** | Standard Cross-Entropy ($\mathcal{L}_{\text{CE}}$) | Weighted Cross-Entropy ($\mathcal{L}_{\text{WCE}}$) |
| **Bộ trọng số lớp ($w$)** | Không dùng ($w_c = 1, \forall c$) | Balanced Weights ($w = [0.41, 4.86, 2.94]$) |
| **Chi phí tính toán** | Tương đương | Tương đương (không tăng thêm tham số mô hình) |
| **Độ nhạy (Recall) lớp hiếm** | Thấp (thường bỏ lọt nhiều câu thù ghét/xúc phạm) | **Tăng vượt bậc** (+5.2% HATE, +6.3% OFFENSIVE) |
| **Độ chính xác thô (Accuracy)**| Cao hơn (~88.1%) do thiên lệch lớp CLEAN | Giảm nhẹ có chủ đích (~86.9%) để tăng độ nhạy vi phạm |
| **Mục đích trong nghiên cứu** | Làm mốc đối chứng (Benchmark Reference) | **Giải pháp chính giải quyết mất cân bằng nhãn** |

---

## 3. Quy trình Tiền xử lý Dữ liệu Tập trung (Data-centric Pipeline)

Bên cạnh cải tiến hàm mất mát, quy trình tiền xử lý đóng vai trò quyết định giúp PhoBERT phát huy tối đa năng lực hiểu ngữ cảnh mạng xã hội tiếng Việt:

```
Văn bản mạng xã hội thô (Raw Comments)
                 │
                 ▼
[1. Chuẩn hóa Unicode dựng sẵn (NFC)]
                 │
                 ▼
[2. Loại bỏ Emoji (tránh vỡ subword và lỗi token)]
                 │
                 ▼
[3. Loại bỏ URLs, Email và ký tự rác]
                 │
                 ▼
[4. Chuẩn hóa Teencode / Tiếng lóng qua Từ điển 414 từ]
                 │
                 ▼
[5. Giữ nguyên từ dừng (No Stopwords - bảo toàn Self-Attention)]
                 │
                 ▼
[6. Tách từ tiếng Việt bằng PyVi (ViTokenizer: từ ghép nối bằng '_')]
                 │
                 ▼
[7. Đệm động theo batch: DataCollatorWithPadding (max_length = 128)]
```

### Các bước Tiền xử lý Chi tiết:
1. **Chuẩn hóa Unicode NFC:** Đảm bảo tính nhất quán của các ký tự tiếng Việt có dấu.
2. **Loại bỏ Emoji:** Tránh việc bộ mã hóa BPE băm emoji thành các chuỗi byte vụn vặt gây nhiễu biểu diễn không gian vector.
3. **Loại bỏ URLs và Emails:** Loại bỏ các thành phần nhiễu không mang giá trị ngữ nghĩa phân loại.
4. **Chuẩn hóa Teencode (414 từ):** Ánh xạ các từ viết tắt, tiếng lóng mạng (*ko* $\rightarrow$ *không*, *dc* $\rightarrow$ *được*, *vcl* $\rightarrow$ *vãi cả lồn*) về từ vựng tiếng Việt chuẩn, hạn chế tối đa từ ngoài từ điển (OOV).
5. **Bảo toàn từ dừng (No Stopwords):** Kế thừa kết quả nghiên cứu của Luu et al. (2024), việc giữ lại từ dừng giúp cơ chế Self-Attention của Transformer nắm bắt đầy đủ quan hệ cú pháp và cấu trúc ngữ nghĩa câu.
6. **Tách từ tiếng Việt (`PyVi`):** Nhận diện ranh giới từ vựng ghép trước khi đưa vào mô hình BPE.

---

## 4. Sơ đồ Luồng Hoạt động Tổng thể của Hệ thống

```
                               Văn bản thô (Facebook Comments)
                                              │
                                              ▼
                             [Pipeline Tiền xử lý Chuẩn hóa]
                       (Unicode NFC, Emoji, URL, Teencode 414, PyVi)
                                              │
                                              ▼
                             [PhoBERT Tokenizer (BPE, max=128)]
                                              │
                                              ▼
                            [PhoBERT-base-v2 Backbone (12 Layers)]
                                              │
                                              ▼
                           [Vector Biểu diễn Câu: h_<s> (768 chiều)]
                                              │
                                              ▼
                                 [Dropout (p = 0.1)]
                                              │
                                              ▼
                            [Linear Classification Head (768 -> 3)]
                                              │
                                              ▼
                                    [Logits z (3 giá trị)]
                                              │
                   ┌──────────────────────────┴──────────────────────────┐
                   ▼ (Huấn luyện Mô hình A)                              ▼ (Huấn luyện Mô hình B - Đề xuất)
       [Standard Cross-Entropy Loss]                         [Weighted Cross-Entropy Loss]
        (Loss không gán trọng số)                              (Trọng số w = [0.41, 4.86, 2.94])
                   │                                                     │
                   └──────────────────────────┬──────────────────────────┘
                                              ▼ (Đánh giá trên Test 6,680 mẫu)
                               [Softmax -> Argmax: Nhãn 0, 1 hoặc 2]
                                              │
                                              ▼
                               [So sánh: Macro F1, Recall, CM]
```
