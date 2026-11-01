# 01. PIPELINE & PHƯƠNG PHÁP: BÀI BÁO GỐC ViHSD (LUU ET AL., 2021)

---

## 1. Thông tin Tổng quan Bài báo
- **Tên bài báo:** *A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts*
- **Tác giả:** Son T. Luu, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen (UIT-NLP Lab, Trường ĐH Công nghệ Thông tin - ĐHQG TP.HCM).
- **Hội nghị công bố:** The 34th International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems (IEA/AIE 2021).
- **Nhà xuất bản:** Springer (LNAI, Vol. 12798, pp. 415–426).
- **DOI:** [10.1007/978-3-030-79457-6_35](https://doi.org/10.1007/978-3-030-79457-6_35) | **arXiv:** [2103.11528](https://arxiv.org/abs/2103.11528)
- **Mã nguồn & Dữ liệu:** [GitHub sonlam1102/vihsd](https://github.com/sonlam1102/vihsd)

---

## 2. Mục tiêu Nghiên cứu & Định nghĩa Bài toán
1. Xây dựng bộ dữ liệu quy mô lớn đầu tiên đạt chuẩn nghiên cứu cho bài toán phát hiện ngôn từ thù ghét tiếng Việt (**ViHSD** - 33,400 bình luận), giải quyết sự thiếu hụt tài nguyên ngữ liệu tiếng Việt.
2. Thiết lập quy chuẩn gán nhãn 3 lớp không chồng lấp:
   - **CLEAN (0):** Bình luận sạch, mang tính xây dựng, trung tính hoặc tích cực/tiêu cực không có yếu tố xúc phạm/thù ghét.
   - **OFFENSIVE (1):** Bình luận thô tục, chửi thề bộc phát nhưng không nhắm vào đối tượng định danh cụ thể.
   - **HATE (2):** Ngôn từ thù ghét, xúc phạm danh dự, kỳ thị vùng miền, dân tộc, tôn giáo, giới tính có chủ đích.
3. Thiết lập hệ thống mô hình đối chứng cơ sở (Baselines) gồm Deep Learning truyền thống và Multilingual Pretrained Transformers.

---

## 3. Sơ đồ Pipeline Tổng thể (Overall Pipeline Architecture)

```
[Thu thập dữ liệu mạng xã hội: Facebook & YouTube]
                      │
                      ▼
[Tiền xử lý ẩn danh: Ẩn thực thể tên riêng, URLs, SĐT, Email]
                      │
                      ▼
[Quy trình gán nhãn thủ công: 4 Annotators độc lập, Kappa = 0.52]
                      │
                      ▼
[Phân chia tập dữ liệu cố định: Train (70%), Dev (10%), Test (20%)]
                      │
                      ▼
   ┌──────────────────┴──────────────────┐
   ▼ (Nhánh Deep Learning)               ▼ (Nhánh Pretrained Transformers)
[Token hóa từ vựng Word-level]       [Subword Tokenization (BPE/WordPiece)]
   │                                     │
   ▼                                     ▼
[Text-CNN / GRU Model]               [m-BERT / XLM-R / DistilBERT]
   │                                     │
   └──────────────────┬──────────────────┘
                      ▼
[Hàm mất mát: Standard Cross-Entropy (Không trọng số)]
                      │
                      ▼
[Đánh giá trên Test Set: Accuracy, Macro F1, Weighted F1]
```

---

## 4. Phương pháp Xử lý Dữ liệu (Data Processing Methodology)

### 4.1. Thu thập và Lọc dữ liệu thô
- **Nguồn dữ liệu:** Cào (crawling) từ các bài đăng có lượng tương tác và bình luận cao trên Facebook và YouTube tại Việt Nam (chủ đề giải trí, sao nổi tiếng, bóng đá, chính trị, xã hội).
- **Ẩn danh hóa (Anonymization):**
  - Để bảo vệ quyền riêng tư người dùng theo đạo đức nghiên cứu, toàn bộ tên người, số điện thoại, email và đường link URL ngoài đều được thay thế bằng token định danh ẩn danh.

### 4.2. Quy trình Gán nhãn & Kiểm định Chất lượng
- **Người gán nhãn:** 4 annotators độc lập (A1, A2, A3, A4) được đào tạo theo hướng dẫn gán nhãn nghiêm ngặt.
- **Độ tin cậy gán nhãn:** Đánh giá trên tập đối chứng 202 bình luận ngẫu nhiên, đạt hệ số **Cohen's Kappa $\kappa = 0.52$** (dao động từ $0.46$ đến $0.65$ giữa từng cặp người gán nhãn).
- **Phân chia tập chuẩn (Split):**
  - **Train:** 24,048 mẫu (CLEAN: 19,886; OFFENSIVE: 1,606; HATE: 2,556)
  - **Dev:** 2,672 mẫu (CLEAN: 2,190; OFFENSIVE: 212; HATE: 270)
  - **Test:** 6,680 mẫu (CLEAN: 5,548; OFFENSIVE: 444; HATE: 688)

---

## 5. Kiến trúc Mô hình & Phương pháp Huấn luyện trong Bài báo

Nhóm tác giả thử nghiệm hai nhóm kiến trúc chính:

### 5.1. Nhóm Deep Learning truyền thống
1. **Text-CNN (Kim, 2014):**
   - Bộ nhúng từ (Embedding layer) kích thước 300 chiều.
   - Các bộ lọc tích chập 1D với kích thước filter: 2, 3, 4, 5 (100 feature maps mỗi loại).
   - Max-over-time pooling kết hợp Dropout (0.5) và Fully Connected layer.
2. **GRU (Cho et al., 2014):**
   - Bi-directional GRU với 128 hidden units, kết hợp Dropout và tầng phân loại Dense.

### 5.2. Nhóm Pretrained Transformers Đa ngữ
1. **m-BERT cased (`bert-base-multilingual-cased`):** 12 layers, 768 hidden, 12 heads, phân biệt hoa/thường.
2. **m-BERT uncased (`bert-base-multilingual-uncased`):** Không phân biệt hoa/thường.
3. **XLM-RoBERTa (`xlm-roberta-base`):** 12 layers, 768 hidden, pretrain trên 100 ngôn ngữ.
4. **DistilBERT multilingual (`distilbert-base-multilingual-cased`):** 6 layers nén từ m-BERT.

### 5.3. Thiết lập Tối ưu hóa & Hàm mất mát
- **Hàm mất mát:** **Standard Cross-Entropy Loss** thông thường:
  $$\mathcal{L} = - \frac{1}{N} \sum_{i=1}^N \log p(y_i \mid x_i)$$
  *(Bài báo gốc hoàn toàn KHÔNG áp dụng Class Weight hay Resampling)*.
- **Optimizer:** AdamW với Learning rate $2 \times 10^{-5}$ cho Transformers, Adam ($1 \times 10^{-3}$) cho CNN/GRU.
- **Batch size:** 16 hoặc 32.
- **Epochs:** 3 - 5 epochs (dừng sớm theo Dev loss).

---

## 6. Kết quả Thực nghiệm trong Bài báo Gốc

Kết quả đánh giá trên tập **TEST (6,680 mẫu)**:

| Mô hình | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) |
|:---|:---:|:---:|:---:|
| **GRU** | 85.57 | 58.74 | 84.62 |
| **Text-CNN** | 86.69 | 61.11 | 85.48 |
| **DistilBERT** | 85.87 | 59.98 | 84.90 |
| **XLM-R base** | 86.41 | 61.90 | 85.56 |
| **m-BERT uncased** | 86.44 | 61.42 | 85.61 |
| **m-BERT cased (Best Baseline)** | **86.88** | **62.69** | **85.92** |

---

## 7. Phân tích Hạn chế & Bài học Kế thừa cho Đề tài của chúng ta

1. **Chưa sử dụng PhoBERT:** Bài báo gốc (2021) mới chỉ khảo sát mô hình đa ngữ (`m-BERT`, `XLM-R`). Tiếng Việt trong mô hình đa ngữ bị chia sẻ không gian từ vựng với 100 ngôn ngữ khác, dẫn đến token hóa bị phân mảnh (oversplitting). Việc đưa **PhoBERT-base** vào là bước nâng cấp tự nhiên và vượt trội.
2. **Thiếu xử lý mất cân bằng lớp:** Do chỉ dùng Cross-Entropy tiêu chuẩn, mô hình `m-BERT cased` đạt Accuracy 86.88% nhưng Macro F1 chỉ dừng ở 62.69% vì bỏ sót rất nhiều mẫu HATE và OFFENSIVE.
3. **Mốc đối chứng thực nghiệm (Baseline Benchmark):** Mức **62.69% Macro F1** của `m-BERT cased` chính là mốc chuẩn để chúng ta chứng minh tính ưu việt của PhoBERT + Class Weight trong đề tài hiện tại.
