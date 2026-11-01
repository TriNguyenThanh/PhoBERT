# 02. KIẾN TRÚC & PHƯƠNG PHÁP: BÀI BÁO GỐC PHOBERT (NGUYEN & NGUYEN, 2020)

---

## 1. Thông tin Tổng quan Bài báo
- **Tên bài báo:** *PhoBERT: Pre-trained language models for Vietnamese*
- **Tác giả:** Dat Quoc Nguyen, Anh Tuan Nguyen (VinAI Research).
- **Hội nghị công bố:** Findings of the Association for Computational Linguistics: EMNLP 2020 (The 2020 Conference on Empirical Methods in Natural Language Processing).
- **Trang:** 1037–1042 | **Năm công bố:** 2020.
- **ACL Anthology:** [2020.findings-emnlp.92](https://aclanthology.org/2020.findings-emnlp.92/) | **arXiv:** [2003.00744](https://arxiv.org/abs/2003.00744)
- **Kho lưu trữ chính thức:** [GitHub VinAIResearch/PhoBERT](https://github.com/VinAIResearch/PhoBERT) | [Hugging Face vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2)

---

## 2. Mục tiêu Nghiên cứu & Đóng góp Cốt lõi
1. Xây dựng mô hình ngôn ngữ tiền huấn luyện Transformer đơn ngữ (**Monolingual Pre-trained Model**) quy mô lớn đầu tiên dành riêng cho tiếng Việt.
2. Chứng minh tầm quan trọng sống còn của việc **Tách từ tiếng Việt (Word Segmentation)** trước khi áp dụng mã hóa Byte-Pair Encoding (BPE), khắc phục triệt để nhược điểm phân mảnh từ vựng của các mô hình đa ngữ (`m-BERT`, `XLM-R`).
3. Thiết lập kỷ lục mới (State-of-the-Art - SOTA) trên 4 tác vụ NLP tiếng Việt: POS Tagging, Named Entity Recognition (NER), Dependency Parsing và Natural Language Inference (XNLI).

---

## 3. Sơ đồ Pipeline Huấn luyện & Ứng dụng PhoBERT

```
[Ngữ liệu văn bản tiếng Việt thô (20GB - ~3 tỷ từ)]
                      │
                      ▼
[Tách từ tiếng Việt cấp độ từ: RDRSegmenter (VnCoreNLP)]
   Ví dụ: "công nghệ thông tin" -> "công_nghệ thông_tin"
                      │
                      ▼
[Mã hóa Byte-Pair Encoding (BPE): Từ điển 64,000 subwords]
                      │
                      ▼
[Tiền huấn luyện RoBERTa: Masked Language Modeling (MLM 15%)]
   (Không sử dụng Next Sentence Prediction - NSP)
                      │
                      ▼
   ┌──────────────────┴──────────────────┐
   ▼                                     ▼
[PhoBERT-base: 135M params]          [PhoBERT-large: 370M params]
   (12 layers, 768 hidden, 12 heads)    (24 layers, 1024 hidden, 16 heads)
                      │
                      ▼
[Fine-tuning trên bài toán cụ thể: ViHSD Hate Speech Detection]
   (PhoBERT Encoder + Dropout + Linear Classification Head)
```

---

## 4. Phương pháp Xử lý Dữ liệu & Mã hóa Từ vựng (Tokenization)

### 4.1. Ngữ liệu Tiền huấn luyện (Pre-training Corpus)
- **Quy mô:** Khoảng **20GB** văn bản tiếng Việt không nén (~3 tỷ từ vựng).
- **Nguồn:**
  - Wikipedia tiếng Việt (~1GB).
  - Kho ngữ liệu tin tức Báo Mới (~19GB) bao hàm đa dạng chủ đề (kinh tế, chính trị, thể thao, văn hóa, giải trí).

### 4.2. Cơ chế Tách từ Tiếng Việt (Word Segmentation) - Bước bắt buộc
- **Bản chất ngôn ngữ học:** Tiếng Việt là ngôn ngữ đơn lập. Một từ có thể gồm 1 âm tiết (*nhà*, *đi*) hoặc nhiều âm tiết tạo thành từ ghép (*học_sinh*, *trí_tuệ_nhân_tạo*). Khoảng trắng trong văn bản tiếng Việt chỉ phân cách âm tiết chứ không đồng nhất với ranh giới từ.
- **Hạn chế của mô hình đa ngữ:** `m-BERT` và `XLM-R` chỉ xem dấu cách là phân tách từ, dẫn đến việc xem *"học"* và *"sinh"* là hai từ độc lập, làm sai lệch ngữ nghĩa.
- **Giải pháp của PhoBERT:** Sử dụng công cụ tách từ **RDRSegmenter** từ bộ thư viện **VnCoreNLP** để phát hiện từ ghép và nối bằng ký tự gạch dưới `_`.
- **Mã hóa BPE:** Sau khi đã tách từ ghép, thuật toán FastBPE được áp dụng để trích xuất từ điển gồm **64,000 subword types** dành riêng cho tiếng Việt.

---

## 5. Kiến trúc Mô hình & Thiết lập Tiền huấn luyện

PhoBERT kế thừa kiến trúc tối ưu hóa **RoBERTa** (Liu et al., 2019):

### 5.1. Hai biến thể kiến trúc
| Cấu hình | Số lớp ẩn ($L$) | Kích thước ẩn ($H$) | Số Attention Heads ($A$) | Tổng số tham số |
|:---|:---:|:---:|:---:|:---:|
| **PhoBERT-base** | **12** | **768** | **12** | **~135 Triệu** |
| **PhoBERT-large** | 24 | 1024 | 16 | ~370 Triệu |

### 5.2. Mục tiêu Huấn luyện (Pre-training Objective)
- **Masked Language Model (MLM):** Che ngẫu nhiên 15% subword tokens trong câu bằng cơ chế Dynamic Masking (80% thay bằng token `<mask>`, 10% giữ nguyên, 10% thay bằng token ngẫu nhiên).
- **Loại bỏ Next Sentence Prediction (NSP):** Huấn luyện trên các chuỗi văn bản dài liên tục (lên tới 512 tokens), chứng minh việc bỏ tác vụ NSP giúp cải thiện đáng kể biểu diễn ngữ cảnh.
- **Tối ưu hóa:** Bộ tối ưu AdamW ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-6}$), Peak learning rate $5 \times 10^{-4}$ (base) và $2 \times 10^{-4}$ (large), linear warmup kết hợp linear decay.

---

## 6. Quy tắc Fine-tuning PhoBERT cho Tác vụ Phân loại Chuỗi (Sequence Classification)

Khi áp dụng PhoBERT vào bài toán phân loại đa lớp (như ViHSD):
1. **Đầu vào chuẩn:** Chuỗi token hóa phải bắt đầu bằng `<s>` và kết thúc bằng `</s>`:
   $$\text{Input} = \text{<s> } w_1 \text{ } w_2 \text{ } ... \text{ } w_T \text{ </s>}$$
2. **Biểu diễn phân loại:** Vector ẩn tại vị trí token đầu tiên $\mathbf{h}_{\text{<s>}} \in \mathbb{R}^{768}$ đóng vai trò là vector biểu diễn toàn bộ ngữ cảnh của câu.
3. **Classification Head:**
   $$\mathbf{z} = \mathbf{W} \cdot \text{Dropout}(\mathbf{h}_{\text{<s>}}, p=0.1) + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{C \times 768}$$
4. **Siêu tham số Fine-tuning khuyến nghị từ tác giả:**
   - Learning rate: $1 \times 10^{-5}$ đến $3 \times 10^{-5}$ (chuẩn: $2 \times 10^{-5}$).
   - Batch size: 16 hoặc 32.
   - Epochs: 3 - 4 epochs.

---

## 7. Bài học Kế thừa Trực tiếp cho Đề tài của chúng ta

1. **Tuân thủ quy trình tách từ:** Bắt buộc phải chạy `ViTokenizer.tokenize()` (hoặc `RDRSegmenter`) trước khi gọi `tokenizer()` của PhoBERT. Nếu bỏ qua bước này, hiệu năng mô hình sẽ sụt giảm nghiêm trọng từ 4% - 7%.
2. **Lựa chọn phiên bản mô hình:** Đề tài sử dụng `vinai/phobert-base-v2` (bản nâng cấp ổn định, cập nhật weights) với 135M tham số — kích thước hoàn hảo để huấn luyện mượt mà trên GPU T4 của Google Colab (chỉ mất ~4GB VRAM khi bật FP16).
