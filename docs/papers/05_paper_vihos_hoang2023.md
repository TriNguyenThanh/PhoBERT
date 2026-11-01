# 05. PIPELINE & PHƯƠNG PHÁP: BÀI TOÁN NHẬN DIỆN KHOẢNG TỪ THÙ GHÉT ViHOS (HOANG ET AL., 2023)

---

## 1. Thông tin Tổng quan Bài báo
- **Tên bài báo:** *ViHOS: Hate Speech Spans Detection for Vietnamese*
- **Tác giả:** Phu Gia Hoang, Canh Duc Luu, Linh Thi-Van Tran, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen (UIT-NLP Lab, Trường ĐH Công nghệ Thông tin - ĐHQG TP.HCM).
- **Hội nghị công bố:** The 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2023).
- **Trang:** 1530–1542 | **Năm công bố:** 2023.
- **Phân hạng:** Hội nghị chuyên ngành NLP hàng đầu thế giới (CORE A / ACL Anthology).
- **ACL Anthology:** [2023.eacl-main.114](https://aclanthology.org/2023.eacl-main.114/) | **arXiv:** [2301.10186](https://arxiv.org/abs/2301.10186)
- **Mã nguồn & Dữ liệu:** [GitHub phu-gia/ViHOS](https://github.com/phu-gia/ViHOS)

---

## 2. Mục tiêu Nghiên cứu & Định nghĩa Bài toán
1. **Chuyển dịch từ phân loại toàn câu sang cấp độ khoảng từ (Span-level):**
   - Bộ dữ liệu ViHSD (2021) chỉ gán 1 nhãn duy nhất cho toàn bộ câu bình luận. Trong hệ thống kiểm duyệt thực tế, quản trị viên cần biết chính xác **những từ/cụm từ cụ thể nào** mang tính độc hại để bôi đỏ hoặc che mờ (tính giải thích được - Explainability).
2. **Xây dựng bộ dữ liệu ViHOS:**
   - Bộ ngữ liệu đầu tiên cho tiếng Việt gồm **11,059 bình luận** với **26,476 khoảng từ thù ghét/xúc phạm** được khoanh vùng ở cấp độ ký tự và từ vựng.
3. **Phát hiện và khắc phục thiên kiến dữ liệu (Data Bias):**
   - Chỉ ra rằng nếu chỉ huấn luyện mô hình trên các câu độc hại, mô hình sẽ bị thiên kiến (báo động giả trên cả các câu sạch). Bổ sung thêm **5,528 bình luận CLEAN** giúp tăng vọt độ chính xác.

---

## 3. Sơ đồ Pipeline Phát hiện Khoảng từ Thù ghét (Spans Detection)

```
[Bình luận mạng xã hội tiếng Việt]
               │
               ▼
[Tách từ tiếng Việt bằng VnCoreNLP]
               │
               ▼
[Mã hóa chuỗi: BIO Scheme (Beginning, Inside, Outside)]
   Ví dụ:  "Nó        là      thằng      khốn_nạn"
            O          O        B-HATE    I-HATE
               │
               ▼
[Mã hóa BPE với PhoBERT Tokenizer]
               │
               ▼
[Backbone Transformer: PhoBERT-base / PhoBERT-large]
   (Trích xuất vector ngữ cảnh h_i cho từng token)
               │
               ▼
[Tầng Phân loại Chuỗi (Token Classification Head / CRF)]
   Linear Projection (768 -> Số nhãn BIO)
               │
               ▼
[Hàm mất mát: Cross-Entropy Loss ở cấp độ Token]
               │
               ▼
[Đầu ra: Tập hợp các khoảng ký tự bắt đầu và kết thúc của từ thù ghét]
```

---

## 4. Phương pháp Xử lý Dữ liệu & Quy chuẩn Gán nhãn BIO

### 4.1. Kế thừa từ ViHSD và Mở rộng
- Tác giả trích xuất các bình luận từ tập ViHSD gốc, tập trung vào các câu có chứa từ ngữ xúc phạm hoặc thù ghét.
- Để giải quyết hiện tượng thiên lệch (Over-triggering) trên câu bình thường, nhóm tác giả bổ sung thêm **5,528 câu CLEAN** mới thu thập từ các hội nhóm giải trí.

### 4.2. Định dạng Gán nhãn BIO
Mỗi token được gán một trong các nhãn:
- `O` (Outside): Từ ngữ trung tính, sạch.
- `B-HATE` / `I-HATE`: Vị trí bắt đầu / bên trong của khoảng từ thù ghét.
- `B-OFFENSIVE` / `I-OFFENSIVE`: Vị trí bắt đầu / bên trong của khoảng từ xúc phạm.

---

## 5. Kiến trúc Mô hình & Thiết lập Huấn luyện

Nghiên cứu đối chiếu 3 nhóm kiến trúc chính:
1. **BiLSTM-CRF kết hợp PhoW2V:** Mạng nơ-ron hồi quy hai chiều kết hợp trường ngẫu nhiên có điều kiện (CRF) trên từ nhúng PhoW2V 300 chiều.
2. **PhoBERT-base + Token Classification:** 12 layers Transformer + Linear Head phân loại từng token.
3. **PhoBERT-large + Token Classification:** 24 layers Transformer (370M tham số).
4. **XLM-RoBERTa (base & large):** Mô hình Transformer đa ngữ để đối sánh.

**Siêu tham số huấn luyện:**
- Learning rate: $5 \times 10^{-5}$ cho PhoBERT.
- Batch size: 16 hoặc 32.
- Epochs: 5 epochs (lưu checkpoint dựa trên F1 ở cấp độ span trên tập Dev).

---

## 6. Kết quả Thực nghiệm Điển hình trên ViHOS

Độ đo đánh giá là **Span-level F1-score** (được tính dựa trên độ trùng khớp chính xác của khoảng từ):

| Mô hình | Huấn luyện chỉ trên câu độc hại (F1 %) | Huấn luyện có bổ sung 5,528 câu CLEAN (F1 %) | Độ tăng trưởng ($\Delta$) |
|:---|:---:|:---:|:---:|
| BiLSTM-CRF + PhoW2V | 56.42 | 65.18 | +8.76% |
| XLM-R base | 62.15 | 71.40 | +9.25% |
| **PhoBERT-base** | **67.20** | **75.69** | **+8.49%** |
| **PhoBERT-large** | **68.65** | **77.16** | **+8.51%** |

---

## 7. Bài học Kế thừa cho Đề tài của chúng ta

1. **Hiểu rõ bức tranh toàn cảnh của bài toán:** Bài toán trong đề tài của chúng ta (ViHSD) là **Sentence-level Classification** (Phân loại toàn câu), đóng vai trò là tầng phòng thủ số 1 để sàng lọc nhanh bình luận. Bài toán ViHOS là **Span-level Detection** (Phát hiện vị trí vi phạm), đóng vai trò là tầng phòng thủ số 2 để phục vụ giải trình minh bạch cho người dùng.
2. **Ý tưởng viết phần Hướng nghiên cứu tương lai (Future Work):** Bạn có thể viết trong báo cáo rằng: *"Sau khi hoàn thiện mô hình phân loại toàn câu bằng PhoBERT-base kết hợp Class Weight trên ViHSD, hướng phát triển tự nhiên tiếp theo là mở rộng kiến trúc sang bài toán nhận diện khoảng từ (Span Detection) theo chuẩn ViHOS (Hoang et al., 2023) để hỗ trợ tính năng bôi đỏ từ khóa vi phạm cho đội ngũ kiểm duyệt viên."*
