# 03. PIPELINE & PHƯƠNG PHÁP: PHOBERT-CNN TRÊN ViHSD (TRAN ET AL., 2022)

---

## 1. Thông tin Tổng quan Bài báo
- **Tên bài báo:** *Vietnamese Hate and Offensive Detection using PhoBERT-CNN and Social Media Streaming Data*
- **Tác giả:** Nguyen-An Tran, Kiet Van Nguyen, Phu Gia Hoang, Son T. Luu, Trong-Hop Do, Ngan Luu-Thuy Nguyen (UIT-NLP Lab, Trường ĐH Công nghệ Thông tin - ĐHQG TP.HCM).
- **Tạp chí công bố:** *Neural Computing and Applications* (Springer, 2022).
- **Chỉ số khoa học:** ISI / Scopus Q1, Impact Factor ~5.0.
- **DOI:** [10.1007/s00521-022-07521-w](https://doi.org/10.1007/s00521-022-07521-w) | **arXiv:** [2206.00524](https://arxiv.org/abs/2206.00524)

---

## 2. Bối cảnh & 4 Điểm nghẽn Được Xác định
Nghiên cứu của Tran et al. (2022) chỉ ra 4 hạn chế cốt lõi của các công trình trước đó trên bộ dữ liệu ViHSD:
1. **Thiếu sót trong tiền xử lý văn bản:** Bỏ qua các đặc thù mạng xã hội tiếng Việt như teencode, từ viết tắt, tiếng lóng và biểu tượng cảm xúc (emojis).
2. **Thờ ơ với sự mất cân bằng dữ liệu nghiêm trọng:** Tỷ lệ nhãn 12.4 : 1 : 1.6 khiến mô hình bị thiên lệch nặng nề về lớp CLEAN.
3. **Hiệu năng mô hình còn khiêm tốn:** Bài báo gốc (2021) chỉ đạt 62.69% Macro F1 với m-BERT cased.
4. **Thiếu khả năng triển khai thực tế:** Chưa có nghiên cứu nào thử nghiệm phân loại theo luồng thời gian thực (Streaming Data) từ mạng xã hội.

---

## 3. Sơ đồ Pipeline Kiến trúc Lai PhoBERT-CNN

```
[Bình luận mạng xã hội (Streaming / ViHSD)]
                      │
                      ▼
[Quy trình Tiền xử lý chuyên sâu cho MXH tiếng Việt]
   - Chuẩn hóa viết tắt / Teencode bằng từ điển
   - Xử lý và chuyển đổi biểu tượng cảm xúc (Emojis)
   - Tách từ ghép tiếng Việt bằng VnCoreNLP (RDRSegmenter)
                      │
                      ▼
[Tăng cường Dữ liệu cấp độ dữ liệu (Data Augmentation - EDA)]
   - Chỉ áp dụng trên 2 lớp thiểu số: OFFENSIVE và HATE
   - Train set: 24,048 mẫu gốc  ──>  46,882 mẫu sau EDA
                      │
                      ▼
[Mã hóa BPE Tokenizer của PhoBERT-base (max_length = 128)]
                      │
                      ▼
[Backbone: PhoBERT-base Encoder (12 Transformer Layers)]
   - Trích xuất chuỗi vector ngữ cảnh: H in R^{T x 768}
                      │
                      ▼
[Text-CNN Module: 1D Convolution với Multi-scale Filter Sizes]
   - Kích thước kernel: 2, 3, 4, 5 (Bắt n-gram cục bộ mang tính công kích)
                      │
                      ▼
[Global Max-over-time Pooling + Concatenation]
                      │
                      ▼
[Dropout (p = 0.5) + Fully Connected Layer (Dense)]
                      │
                      ▼
[Softmax -> Dự đoán 3 nhãn: CLEAN (0), OFFENSIVE (1), HATE (2)]
```

---

## 4. Phương pháp Tiền xử lý & Tăng cường Dữ liệu (Preprocessing & EDA)

### 4.1. Tiền xử lý chuyên sâu cho Mạng xã hội tiếng Việt
Nghiên cứu chứng minh rằng **riêng bước tiền xử lý chuẩn hóa đã giúp tăng +4.66% Macro F1**:
- **Từ điển ánh xạ từ viết tắt:** Chuyển đổi các biến thể teencode phổ biến về dạng chuẩn (`vcl`, `đm`, `dcm`, `clgt`, `ko`, `hok` $\rightarrow$ dạng từ chuẩn).
- **Xử lý Emojis:** Ánh xạ các biểu tượng cảm xúc thành các từ ngữ biểu thị sắc thái tình cảm tương đương.
- **Tách từ tiếng Việt:** Sử dụng công cụ đạt chuẩn SOTA là **VnCoreNLP** (RDRSegmenter) để nối các âm tiết thành từ ghép chuẩn.

### 4.2. Tăng cường Dữ liệu EDA (Easy Data Augmentation)
Để giải quyết bài toán mất cân bằng lớp ở **cấp độ dữ liệu (Data-level)**, tác giả áp dụng 4 thao tác biến đổi trên 2 lớp thiểu số:
1. *Synonym Replacement (SR):* Thay thế từ ngẫu nhiên bằng từ đồng nghĩa tiếng Việt.
2. *Random Insertion (RI):* Chèn thêm từ đồng nghĩa vào vị trí ngẫu nhiên trong câu.
3. *Random Swap (RS):* Hoán đổi vị trí của 2 từ bất kỳ trong câu.
4. *Random Deletion (RD):* Xóa ngẫu nhiên một từ với xác suất $p$.

**Hiệu quả cân bằng số lượng mẫu trên tập Train:**
- **CLEAN (0):** 19,886 mẫu $\rightarrow$ **Giữ nguyên 19,886 mẫu**.
- **OFFENSIVE (1):** 1,606 mẫu $\rightarrow$ **Tăng lên 10,147 mẫu** (tăng gấp 6.3 lần).
- **HATE (2):** 2,556 mẫu $\rightarrow$ **Tăng lên 16,849 mẫu** (tăng gấp 6.6 lần).
- **Tổng tập Train:** Tăng từ **24,048 mẫu** lên **46,882 mẫu**.

---

## 5. Kiến trúc Mô hình PhoBERT-CNN & Phương pháp Huấn luyện

### 5.1. Cơ chế phối hợp giữa PhoBERT và Text-CNN
- **Tại sao cần Text-CNN trên đỉnh PhoBERT?**
  - PhoBERT giỏi nắm bắt ngữ nghĩa toàn câu (Global Context) nhưng đôi khi phân tán sự chú ý khi câu có nhiều mệnh đề.
  - Tầng tích chập 1D (Text-CNN) với các bộ lọc kích thước nhỏ ($k \in \{2, 3, 4, 5\}$) hoạt động như các bộ quét cục bộ, phát hiện cực kỳ nhạy bén các cụm $n$-gram mang tính chửi thề hoặc miệt thị đặc thù (ví dụ: *"đồ súc vật"*, *"lũ mọi rợ"*).
- **Pooling & Phân loại:**
  - Áp dụng Max-over-time pooling trên từng feature map để chọn ra đặc trưng kích hoạt mạnh nhất.
  - Ghép các vector đặc trưng lại, đi qua Dropout ($p=0.5$) và tầng Dense 3 chiều.

### 5.2. Siêu tham số Huấn luyện
- **Tốc độ học (Learning rate):** $2 \times 10^{-5}$ cho các tầng PhoBERT, $1 \times 10^{-3}$ cho các tầng CNN.
- **Batch size:** 16.
- **Epochs:** 4 epochs.
- **Hàm mất mát:** Cross-Entropy trên dữ liệu đã cân bằng bằng EDA.

---

## 6. Kết quả Thực nghiệm Đạt Kỷ lục SOTA trên ViHSD

| Mô hình | Tiền xử lý | Tăng cường dữ liệu | Macro F1 (%) | Cải thiện so với Baseline gốc |
|:---|:---:|:---:|:---:|:---:|
| m-BERT cased (Luu et al., 2021) | Cơ bản | Không | 62.69 | Mốc cơ sở |
| PhoBERT-base đơn thuần | Cơ bản | Không | 63.85 | +1.16% |
| PhoBERT-base | Nâng cao | Không | 65.42 | +2.73% |
| **PhoBERT-CNN (Tran et al., 2022)** | **Nâng cao** | **EDA (46k mẫu)** | **67.46** | **+4.77% (Kỷ lục mới)** |

---

## 7. Phân tích So sánh & Ứng dụng vào Đề tài của chúng ta

1. **So sánh hướng tiếp cận giải quyết Mất cân bằng lớp:**
   - **Cách của Tran et al. (2022) - Data-level:** Nhân bản dữ liệu bằng EDA lên gần 47,000 mẫu. Nhược điểm: Tăng gấp đôi thời gian huấn luyện và có nguy cơ làm nhiễu ngữ pháp câu tiếng Việt nếu từ chèn vào không hợp ngữ cảnh.
   - **Cách của Đề tài chúng ta - Algorithm-level (Class Weight / Focal Loss):** Giữ nguyên tập Train 24,048 mẫu nguyên bản, điều chỉnh đạo hàm của hàm mất mát. Ưu điểm: Huấn luyện nhanh hơn gấp đôi, bảo toàn 100% ngữ pháp tự nhiên của câu.
2. **Mốc đánh giá (Benchmark):**
   - Đề tài của chúng ta với **PhoBERT-base + Class Weight** đạt Macro F1 kỳ vọng **~66.0% - 67.0%**, tiệm cận rất sát mức SOTA 67.46% của PhoBERT-CNN mà không cần phải chạy EDA nhân bản dữ liệu phức tạp.
