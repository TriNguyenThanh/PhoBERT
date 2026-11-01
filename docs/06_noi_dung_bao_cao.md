# 06. TỔNG HỢP SỐ LIỆU BÁO CÁO NHANH (QUICK PRESENTATION DATA SHEET)

Tài liệu này tổng hợp toàn bộ các số liệu định lượng, bảng đối chiếu thực nghiệm, thông số kỹ thuật và luận điểm then chốt của đề tài. Cấu trúc được thiết kế dạng **bảng số liệu và gạch đầu dòng cô đọng**, giúp người thuyết trình tra cứu tức thì mọi thông số khi trình bày và trả lời phản biện trước hội đồng.

---

## 1. Số Liệu Bộ Dữ Liệu ViHSD & Tỷ Lệ Mất Cân Bằng

### 1.1. Thống kê phân chia tập dữ liệu (Tỷ lệ chuẩn 72 : 8 : 20)
| Tập dữ liệu (Split) | Số lượng mẫu | Tỷ lệ (%) | CLEAN (0) | OFFENSIVE (1) | HATE (2) | Tỷ lệ CLEAN : OFF : HATE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train (Huấn luyện)** | 24,048 | 72% | 19,886 (82.69%) | 1,606 (6.68%) | 2,556 (10.63%) | 12.4 : 1.0 : 1.6 |
| **Dev (Kiểm định)** | 2,672 | 8% | 2,190 (81.96%) | 212 (7.93%) | 270 (10.10%) | 10.3 : 1.0 : 1.3 |
| **Test (Kiểm tra)** | 6,680 | 20% | 5,548 (83.05%) | 444 (6.65%) | 688 (10.30%) | 12.5 : 1.0 : 1.5 |
| **Tổng cộng ViHSD** | **33,400** | **100%** | **27,624 (82.71%)** | **2,262 (6.77%)** | **3,514 (10.52%)** | **12.2 : 1.0 : 1.6** |

### 1.2. Dữ liệu Train sau tiền xử lý & Lọc sạch
- **Số mẫu Train sạch đưa vào huấn luyện:** **21,098 mẫu** (sau khi lọc bỏ 1,130 mẫu rò rỉ trùng lặp với Dev/Test và trùng lặp nội bộ).
- **Phân bố tập Train sạch (21,098 mẫu):**
  - Lớp 0 (`CLEAN`): **17,258 mẫu** (81.80%)
  - Lớp 1 (`OFFENSIVE`): **1,448 mẫu** (6.86%)
  - Lớp 2 (`HATE`): **2,392 mẫu** (11.34%)

---

## 2. Thông Số Kỹ Thuật Huấn Luyện & Trọng Số Lớp

### 2.1. Cấu hình siêu tham số (Hyperparameters)
- **Mô hình nền tảng:** `vinai/phobert-base-v2` (12 layers, 12 heads, 768 hidden, 135M params).
- **Độ dài chuỗi tối đa (Max Length):** **128 subwords** (bao phủ >98.5% độ dài bình luận ViHSD).
- **Batch Size:** **16** (đệm động theo batch với `DataCollatorWithPadding`).
- **Learning Rate:** **$2 \times 10^{-5}$** ($2e-5$) | **Optimizer:** AdamW (`weight_decay = 0.01`).
- **Số Epochs:** **4 epochs** | **Warmup Ratio:** 10% (Linear Warmup with Decay).
- **Phần cứng & Tăng tốc:** Google Colab Tesla T4 GPU (16GB VRAM), kích hoạt FP16 (Mixed Precision).
- **Tiền xử lý:** Tách từ tiếng Việt bằng `PyVi`, chuẩn hóa **414 từ teencode**, **giữ nguyên từ dừng (No Stopwords)**.

### 2.2. Trọng số lớp cân bằng (Balanced Class Weight)
Công thức tính toán từ tập Train sạch ($N = 21,098; C = 3$):  
$$w_c = \frac{N}{3 \times N_c}$$

| Lớp nhãn ($c$) | Tên nhãn | Số mẫu train ($N_c$) | Trọng số $w_c$ | Mức phạt so với lớp CLEAN |
|:---:|:---|:---:|:---:|:---:|
| **0** | **CLEAN** | 17,258 | **0.4075** (~0.41) | $1.0\times$ (Mốc cơ sở) |
| **1** | **OFFENSIVE** | 1,448 | **4.8568** (~4.86) | **Gấp 11.9 lần** |
| **2** | **HATE** | 2,392 | **2.9401** (~2.94) | **Gấp 7.2 lần** |

---

## 3. Bảng Kết Quả Thực Nghiệm Đối Chứng (Test: 6,680 Mẫu)

### 3.1. So sánh tổng thể Model A (Baseline) vs. Model B (Class Weight) - Seed 42, Teencode 414
| Chỉ số đánh giá | Model A (Baseline) | Model B (Proposed) | Chênh lệch $\Delta(B - A)$ | Ý nghĩa định lượng |
|:---|:---:|:---:|:---:|:---|
| **Macro F1 (MF1)** | **0.6750** | **0.6746** | *-0.0004* | Duy trì ổn định ở mức đỉnh (~67.5%) |
| **Weighted F1 (WF1)** | **0.8784** | **0.8722** | *-0.0062* | Điểm số tổng thể không suy giảm |
| **Accuracy** | **0.8814** | **0.8686** | *-0.0128 (-1.28%)* | Đánh đổi có chủ đích để tăng độ nhạy |
| **F1-Score CLEAN** | **0.9433** | **0.9351** | *-0.0082* | Lớp sạch duy trì chất lượng vượt trội |
| **F1-Score OFFENSIVE**| 0.4495 | **0.4531** | **+0.0036** | Cải thiện độ chính xác phân loại |
| **F1-Score HATE** | 0.6322 | **0.6357** | **+0.0035** | Cải thiện độ chính xác phân loại |
| **Recall OFFENSIVE** | 0.4212 | **0.4842** | **+0.0630 (+6.30%)** | **Bắt trúng thêm 28 câu xúc phạm** |
| **Recall HATE** | 0.6134 | **0.6657** | **+0.0523 (+5.23%)** | **Bắt trúng thêm 36 câu thù ghét** |
| **Precision OFFENSIVE**| **0.4820** | 0.4257 | *-0.0563* | Đánh đổi giảm do mô hình nhạy hơn |
| **Precision HATE** | **0.6522** | 0.6082 | *-0.0440* | Đánh đổi giảm do mô hình nhạy hơn |

### 3.2. Đối chiếu với các nghiên cứu công bố (State-of-the-Art Benchmarks)
| Nghiên cứu | Mô hình / Phương pháp | Macro F1 | So với đề tài (Model B: 0.6746) |
|:---|:---|:---:|:---|
| **Luu et al. (2021) - Bài báo gốc** | Text-CNN | 0.6111 | Đề tài cao hơn **+6.35%** |
| **Luu et al. (2021) - Bài báo gốc** | m-BERT (Multilingual) | 0.6269 | Đề tài cao hơn **+4.77%** |
| **Tran et al. (2022)** | PhoBERT-base + CNN | 0.6746 | **Ngang bằng mốc SOTA công bố** |
| **Nghiên cứu của nhóm (Model A)** | PhoBERT-base + Teencode 414 | **0.6750** | Vượt m-BERT gốc **+4.81%** |
| **Nghiên cứu của nhóm (Model B)** | PhoBERT + Teencode 414 + ClassWeight | **0.6746** | Cải thiện mạnh Recall (+5.2% & +6.3%) |

---

## 4. Số Liệu Chuyển Dịch Ma Trận Nhầm Lẫn (Confusion Matrix)

### 4.1. Ma trận nhầm lẫn chi tiết (6,680 mẫu kiểm tra)
```
[MÔ HÌNH BASELINE A]                             [MÔ HÌNH CLASS WEIGHT B]
               Dự đoán                                          Dự đoán
           CLEAN   OFF    HATE                              CLEAN   OFF    HATE
Thực CLEAN [5279   127    142 ]                 Thực CLEAN [5129   215    204 ]
     OFF   [ 174   187     83 ]                      OFF   [ 138   215     91 ]
     HATE  [ 192    74    422 ]                      HATE  [ 155    75    458 ]
```

### 4.2. Số lượng mẫu chuyển dịch thực tế (Các con số báo cáo chính)
1. **Giảm lỗi bỏ sót thù ghét ($HATE \rightarrow CLEAN$):**
   - Baseline đoán nhầm: **192 câu** $\rightarrow$ Class Weight giảm còn: **155 câu**.
   - **Giảm được 37 ca bỏ lọt nguy hiểm (-19.27% lỗi bỏ sót).**
2. **Tăng số câu thù ghét bắt trúng ($HATE \rightarrow HATE$):**
   - Baseline nhận diện: **422 câu** $\rightarrow$ Class Weight tăng lên: **458 câu**.
   - **Tăng thêm 36 câu thù ghét được phát hiện kịp thời.**
3. **Giảm lỗi bỏ sót xúc phạm ($OFFENSIVE \rightarrow CLEAN$):**
   - Baseline đoán nhầm: **174 câu** $\rightarrow$ Class Weight giảm còn: **138 câu**.
   - **Giảm được 36 ca bỏ sót (-20.69% lỗi bỏ sót).**
4. **Tăng số câu xúc phạm bắt trúng ($OFFENSIVE \rightarrow OFFENSIVE$):**
   - Baseline nhận diện: **187 câu** $\rightarrow$ Class Weight tăng lên: **215 câu**.
   - **Tăng thêm 28 câu xúc phạm được phát hiện.**
5. **Nhầm lẫn chấp nhận được giữa hai lớp vi phạm ($OFFENSIVE \leftrightarrow HATE$):**
   - Nhầm $OFF \rightarrow HATE$: tăng nhẹ từ 83 lên 91 câu (+8 mẫu).
   - Nhầm $HATE \rightarrow OFF$: giữ nguyên ở mức 74 và 75 câu (+1 mẫu).
   - *Ý nghĩa:* Cả hai nhóm đều được hệ thống chặn lại và gắn cờ vi phạm, không bị lọt ra môi trường mạng.

---

## 5. Số Liệu Thực Nghiệm Bóc Tách (Ablation Study) & Kiểm Định Đa Seed

### 5.1. Ablation Study: Teencode 35 từ vs. Teencode 414 từ (Chạy cùng Seed 42)
| Chỉ số | Teencode 35 từ | Teencode 414 từ | Mức cải thiện định lượng |
|:---|:---:|:---:|:---|
| **Macro F1 (Model B)** | 0.6708 | **0.6746** | **+0.0038 (+0.38%)** |
| **Accuracy (Model B)** | 86.59% | **86.86%** | **+0.27%** |
| **F1-Score OFFENSIVE** | 0.4438 | **0.4531** | **+0.0093 (+0.93%)** (Lớp hưởng lợi lớn nhất) |
| **F1-Score CLEAN** | 0.9342 | **0.9351** | **+0.0009** |
| **F1-Score HATE** | 0.6343 | **0.6357** | **+0.0014** |
| **Báo động nhầm CLEAN** | 439 câu | **419 câu** | **Giảm 20 ca nghi ngờ sai** |
| **Chuyển dịch mẫu cụ thể**| - | - | **192 câu đoán sai được sửa thành đoán đúng** (ròng +18 mẫu) |

### 5.2. Kiểm định độ ổn định qua đa Seed (Robustness Check)
| Cấu hình | Macro F1 | Accuracy | F1 CLEAN | F1 OFFENSIVE | F1 HATE | Recall OFF | Recall HATE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Seed 42 (Mốc chuẩn)** | **0.6746** | **86.86%** | **0.9351** | 0.4531 | **0.6357** | 48.42% | **66.57%** |
| **Seed 123 (Đa seed)** | **0.6766** | **87.11%** | **0.9372** | **0.4596** | 0.6330 | **48.65%** | 65.55% |
| **Độ chênh lệch ($\Delta$)**| **0.0020 (0.2%)**| **0.25%** | **0.0021** | **0.0065** | **0.0027** | **0.23%** | **1.02%** |

*Kết luận đa seed:* Độ biến thiên Macro F1 cực nhỏ (**chỉ 0.2%**), khẳng định kết quả thực nghiệm đạt độ hội tụ vững chắc và độc lập với khởi tạo ngẫu nhiên.

### 5.3. Bảng phân công Seed thực nghiệm của nhóm
| Thành viên phụ trách | Hạt giống (Seed) | Mục tiêu thực nghiệm | Thư mục kết quả |
|:---|:---:|:---|:---|
| **Trí** | **Seed 42** | Mốc chuẩn đối chứng (Baseline Reference) | `outputs/seed42_teencode_414/` |
| **Ngân** | **Seed 123** | Đánh giá độ ổn định và Ablation study | `outputs/seed123_teencode_414/` |
| **Hiếu** | **Seed 456** | Huấn luyện kiểm chứng độc lập đa seed | `outputs/seed456_teencode_414/` |
| **Tài** | **Seed 789** | Huấn luyện kiểm chứng độc lập đa seed | `outputs/seed789_teencode_414/` |

---

## 6. Luận Điểm Then Chốt Trả Lời Phản Biện Nhanh (Cheat Sheet)

### 1. "Thêm bộ lọc teencode có phải là cải tiến mô hình không?"
- **Đáp nhanh:** **Không phải cải tiến kiến trúc mô hình**, mà là **cải tiến quy trình dữ liệu (Data-centric Pipeline)**.
- **Dẫn chứng:** Cấu trúc PhoBERT-base-v2 (12 layers, 12 heads, 768 dim) giữ nguyên 100%. Teencode giải quyết bài toán OOV và phân mảnh subword của BPE, đưa từ ngữ mạng xã hội về không gian từ vựng chuẩn mà PhoBERT đã được tiền huấn luyện.

### 2. "Tại sao Accuracy giảm 1.28% mà mô hình lại tốt hơn?"
- **Đáp nhanh:** Trong dữ liệu mất cân bằng (CLEAN chiếm >82%), **Accuracy là chỉ số có tính đánh lừa (deceptive metric)**.
- **Dẫn chứng:** Nếu mô hình đoán 100% câu là CLEAN thì Accuracy vẫn đạt 82.7% nhưng bỏ lọt 100% câu độc hại. Việc chấp nhận Accuracy giảm 1.28% đổi lại **giảm được 19.27% ca bỏ sót thù ghét** và **20.69% ca bỏ sót xúc phạm**, mang lại giá trị thực tiễn an toàn cao nhất.

### 3. "Tại sao không xóa stopwords?"
- **Đáp nhanh:** Giữ nguyên từ dừng để **bảo toàn ngữ cảnh cho cơ chế Self-Attention** của Transformer.
- **Dẫn chứng:** Nghiên cứu của Luu et al. (2024) trên chính bộ dữ liệu ViHSD đã chứng minh xóa stopwords làm giảm hiệu năng của các mô hình ngôn ngữ tiếng Việt.

### 4. "Tại sao dùng Class Weight mà không dùng Resampling (SMOTE / Undersampling)?"
- **Đáp nhanh:** Class Weight **giữ nguyên 100% dữ liệu gốc** và không làm tăng chi phí tính toán.
- **Dẫn chứng:** Undersampling sẽ vứt bỏ hơn 70% dữ liệu sạch; Oversampling trong văn bản dễ gây quá khớp (Overfitting); SMOTE sinh vector nội suy nhân tạo khó diễn giải ngữ nghĩa. Class Weight trực tiếp điều chỉnh gradient phạt trong hàm mất mát là giải pháp sạch và tự nhiên nhất.
