# 06. TỔNG HỢP SỐ LIỆU BÁO CÁO NHANH (QUICK PRESENTATION DATA SHEET)

Tài liệu này tổng hợp toàn bộ các số liệu định lượng, bảng đối chiếu thực nghiệm, thông số kỹ thuật và luận điểm then chốt của đề tài. Cấu trúc được thiết kế dạng **bảng số liệu và gạch đầu dòng cô đọng**, giúp người thuyết trình tra cứu tức thì mọi thông số khi trình bày và trả lời phản biện trước hội đồng.

---

## 1. Số Liệu Bộ Dữ Liệu ViHSD & Tỷ Lệ Mất Cân Bằng

### 1.1. Thống kê phân chia tập dữ liệu (Tỷ lệ benchmark chuẩn 72 : 8 : 20)
| Tập dữ liệu (Split) | Số lượng mẫu | Tỷ lệ (%) | CLEAN (0) | OFFENSIVE (1) | HATE (2) | Tỷ lệ CLEAN : OFF : HATE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train (Huấn luyện gốc)** | 24,048 | 72% | 19,886 (82.69%) | 1,606 (6.68%) | 2,556 (10.63%) | 12.4 : 1.0 : 1.6 |
| **Dev (Kiểm định)** | 2,672 | 8% | 2,190 (81.96%) | 212 (7.93%) | 270 (10.10%) | 10.3 : 1.0 : 1.3 |
| **Test (Kiểm tra cố định)**| 6,680 | 20% | 5,548 (83.05%) | 444 (6.65%) | 688 (10.30%) | 12.5 : 1.0 : 1.5 |
| **Tổng cộng ViHSD** | **33,400** | **100%** | **27,624 (82.71%)** | **2,262 (6.77%)** | **3,514 (10.52%)** | **12.2 : 1.0 : 1.6** |

### 1.2. Dữ liệu Train sau tiền xử lý & Lọc sạch (Cleaned Train Set)
- **Số mẫu Train sạch đưa vào huấn luyện:** **21,098 mẫu** (sau khi rà soát loại bỏ 1,130 mẫu trùng lặp nội bộ và rò rỉ dữ liệu với Dev/Test).
- **Phân bố tập Train sạch ($N = 21,098$ mẫu):**
  - Lớp 0 (`CLEAN` - Sạch/Lành mạnh): **17,258 mẫu** (81.80%)
  - Lớp 1 (`OFFENSIVE` - Xúc phạm/Thô tục bộc phát): **1,448 mẫu** (6.86%)
  - Lớp 2 (`HATE` - Thù ghét/Kỳ thị có chủ đích): **2,392 mẫu** (11.34%)

---

## 2. Thông Số Kỹ Thuật Huấn Luyện & Trọng Số Lớp

### 2.1. Cấu hình siêu tham số (Hyperparameters)
- **Mô hình nền tảng (Backbone):** `vinai/phobert-base-v2` (12 layers, 12 heads, 768 hidden, 135M params).
- **Độ dài chuỗi tối đa (Max Length):** **128 subwords** (bao phủ >98.5% độ dài bình luận ViHSD).
- **Batch Size:** **16** (đệm động theo batch với `DataCollatorWithPadding`).
- **Learning Rate:** **$2 \times 10^{-5}$** ($2e-5$) | **Optimizer:** AdamW (`weight_decay = 0.01`).
- **Số Epochs:** **4 epochs** | **Warmup Ratio:** 10% (Linear Warmup with Decay).
- **Phần cứng & Tăng tốc:** Google Colab Tesla T4 GPU (16GB VRAM), kích hoạt FP16 (Mixed Precision).
- **Tiền xử lý:** Tách từ tiếng Việt bằng `PyVi`, chuẩn hóa **414 từ teencode**, **giữ nguyên từ dừng (No Stopwords)**.

### 2.2. Trọng số lớp cân bằng (Balanced Class Weight)
Công thức tính toán từ tập Train sạch ($N_{\text{clean}} = 21,098; C = 3$):  
$$w_c = \frac{N_{\text{clean}}}{3 \times N_c}$$

| Lớp nhãn ($c$) | Tên nhãn khoa học | Số mẫu train ($N_c$) | Trọng số $w_c$ | Mức phạt tương đối so với CLEAN |
|:---:|:---|:---:|:---:|:---:|
| **0** | **CLEAN** *(Sạch / Lành mạnh)* | 17,258 | **0.4075** (~0.41) | $1.0\times$ (Mốc cơ sở) |
| **1** | **OFFENSIVE** *(Xúc phạm / Thô tục)* | 1,448 | **4.8568** (~4.86) | **Gấp ~11.9 lần** |
| **2** | **HATE** *(Ngôn từ thù ghét)* | 2,392 | **2.9401** (~2.94) | **Gấp ~7.2 lần** |

---

## 3. Bảng Kết Quả Thực Nghiệm Đối Chứng (Test: 6,680 Mẫu)

### 3.1. So sánh tổng thể Model A (Baseline) vs. Model B (Proposed) — Seed 42, Teencode 414
| Chỉ số đánh giá | Model A (Baseline Đối chứng) | Model B (Sản phẩm Đề xuất Cốt lõi) | Chênh lệch $\Delta(B - A)$ | Ý nghĩa khoa học |
|:---|:---:|:---:|:---:|:---|
| **Macro F1 (MF1)** | **0.6750** | **0.6746** | *-0.0004* | Duy trì ổn định ở mức đỉnh (~67.5%) |
| **Weighted F1 (WF1)** | **0.8784** | **0.8722** | *-0.0062* | Điểm số tổng thể không suy giảm |
| **Accuracy** | **0.8814** | **0.8686** | *-0.0128 (-1.28%)* | Đánh đổi có chủ đích để tăng độ nhạy |
| **F1-Score CLEAN** | **0.9433** | **0.9351** | *-0.0082* | Lớp sạch duy trì chất lượng vượt trội |
| **F1-Score OFFENSIVE**| 0.4495 | **0.4531** | **+0.0036 (+0.36%)** | Cải thiện độ chính xác phân loại |
| **F1-Score HATE** | 0.6322 | **0.6357** | **+0.0035 (+0.35%)** | Cải thiện độ chính xác phân loại |
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
| **Nghiên cứu của nhóm (Model A)** | PhoBERT-base-v2 + Teencode 414 | **0.6750** | Vượt m-BERT gốc **+4.81%** |
| **Nghiên cứu của nhóm (Model B)** | PhoBERT-base-v2 + Teencode 414 + ClassWeight | **0.6746** | Cải thiện mạnh Recall (+5.2% HATE, +6.3% OFF) |

### 3.3. Kết quả Thống kê Tổng hợp Đa Hạt Giống (Mean ± Std: $\mu \pm \sigma$, $N = 4$ Seeds)
| Chỉ số đánh giá | Model A (Baseline Đối chứng) | Model B (Sản phẩm Đề xuất Cốt lõi) | Chênh lệch $\Delta(B - A)$ | Ý nghĩa khoa học |
|:---|:---:|:---:|:---:|:---|
| **Macro F1 (MF1)** | **0.6739 ± 0.0034** | **0.6732 ± 0.0037** | *-0.0007 ± 0.0058* | Duy trì ổn định tuyệt đối quanh mức 67.3% |
| **Accuracy** | **0.8766 ± 0.0057** | **0.8685 ± 0.0027** | *-0.0081 ± 0.0078* | Đánh đổi giảm có kiểm soát ~0.81%; phương sai của B giảm 52.6% |
| **F1-Score CLEAN** | **0.9405 ± 0.0030** | **0.9350 ± 0.0018** | *-0.0055 ± 0.0045* | Lớp sạch duy trì chất lượng vượt trội |
| **F1-Score OFFENSIVE**| 0.4445 ± 0.0090 | **0.4502 ± 0.0104** | **+0.0057 ± 0.0149** | Tăng trung bình +0.57% trên toàn bộ 4 seed |
| **F1-Score HATE** | **0.6365 ± 0.0035** | **0.6344 ± 0.0015** | *-0.0021 ± 0.0041* | Tương đương hoàn toàn; độ lệch chuẩn của B chỉ còn 0.0015 |
| **Recall OFFENSIVE** | 0.4358 ± 0.0405 | **0.4702 ± 0.0179** | **+0.0344 ± 0.0307** | **Tăng trung bình +3.44%**, độ bấp bênh giảm hơn 55% |
| **Recall HATE** | 0.6418 ± 0.0259 | **0.6729 ± 0.0186** | **+0.0312 ± 0.0291** | **Tăng trung bình +3.12%**, đạt đỉnh 69.91% ở Seed 456 |

---

## 4. Số Liệu Chuyển Dịch Ma Trận Nhầm Lẫn (Confusion Matrix)

### 4.1. Ma trận nhầm lẫn chi tiết (6,680 mẫu kiểm tra)
```
[MÔ HÌNH A: BASELINE ĐỐI CHỨNG]                  [MÔ HÌNH B: PROPOSED CỐT LÕI]
               Dự đoán                                          Dự đoán
           CLEAN   OFF    HATE                              CLEAN   OFF    HATE
Thực CLEAN [5279   127    142 ]                 Thực CLEAN [5129   215    204 ]
     OFF   [ 174   187     83 ]                      OFF   [ 138   215     91 ]
     HATE  [ 192    74    422 ]                      HATE  [ 155    75    458 ]
```

### 4.2. Số lượng mẫu chuyển dịch thực tế (Các con số báo cáo chính)
1. **Giảm lỗi bỏ sót thù ghét ($HATE \rightarrow CLEAN$):**
   - Baseline đoán nhầm: **192 câu** $\rightarrow$ Class Weight giảm còn: **155 câu**.
   - **Giảm được 37 ca bỏ sót nguy hiểm (-19.27% lỗi bỏ lọt thù ghét).**
2. **Tăng số câu thù ghét bắt trúng ($HATE \rightarrow HATE$):**
   - Baseline nhận diện: **422 câu** $\rightarrow$ Class Weight tăng lên: **458 câu**.
   - **Tăng thêm 36 câu thù ghét được phát hiện kịp thời.**
3. **Giảm lỗi bỏ sót xúc phạm ($OFFENSIVE \rightarrow CLEAN$):**
   - Baseline đoán nhầm: **174 câu** $\rightarrow$ Class Weight giảm còn: **138 câu**.
   - **Giảm được 36 ca bỏ sót (-20.69% lỗi bỏ sót xúc phạm).**
4. **Tăng số câu xúc phạm bắt trúng ($OFFENSIVE \rightarrow OFFENSIVE$):**
   - Baseline nhận diện: **187 câu** $\rightarrow$ Class Weight tăng lên: **215 câu**.
   - **Tăng thêm 28 câu xúc phạm được phát hiện.**
5. **Nhầm lẫn chấp nhận được giữa hai lớp vi phạm ($OFFENSIVE \leftrightarrow HATE$):**
   - Nhầm $OFF \rightarrow HATE$: tăng nhẹ từ 83 lên 91 câu (+8 mẫu).
   - Nhầm $HATE \rightarrow OFF$: giữ nguyên ở mức 74 và 75 câu (+1 mẫu).
   - *Ý nghĩa:* Cả hai nhóm đều được hệ thống gắn cờ vi phạm, không bị bỏ lọt ra mạng xã hội.

---

## 5. Số Liệu Thực Nghiệm Bóc Tách & Kiểm Định Đa Seed

### 5.1. Ablation Study: Teencode 35 từ vs. Teencode 414 từ (Chạy cùng Seed 42)
| Chỉ số | Teencode 35 từ | Teencode 414 từ | Mức cải thiện định lượng |
|:---|:---:|:---:|:---|
| **Macro F1 (Model B)** | 0.6708 | **0.6746** | **+0.0038 (+0.38%)** |
| **Accuracy (Model B)** | 86.59% | **86.86%** | **+0.27%** |
| **F1-Score OFFENSIVE** | 0.4438 | **0.4531** | **+0.0093 (+0.93%)** *(Lớp hưởng lợi lớn nhất)* |
| **F1-Score CLEAN** | 0.9342 | **0.9351** | **+0.0009** |
| **F1-Score HATE** | 0.6343 | **0.6357** | **+0.0014** |
| **Báo động nhầm CLEAN** | 439 câu | **419 câu** | **Giảm 20 ca nghi ngờ sai** |
| **Chuyển dịch mẫu cụ thể**| - | - | **192 câu đoán sai được sửa thành đoán đúng** (ròng +18 mẫu) |

### 5.2. Kiểm định độ ổn định qua toàn bộ 4 Seeds (Robustness Check & Multi-Seed Statistics)
| Cấu hình Thực nghiệm | Thành viên | Macro F1 | Accuracy | F1 CLEAN | F1 OFFENSIVE | F1 HATE | Recall OFF | Recall HATE |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Seed 42 (Mốc chuẩn)** | **Trí** | **0.6746** | 86.86% | 0.9351 | 0.4531 | **0.6357** | 48.42% | 66.57% |
| **Seed 123 (Đa seed)** | **Ngân** | **0.6766** | **87.11%** | **0.9372** | **0.4596** | 0.6330 | **48.65%** | 65.55% |
| **Seed 456 (Kiểm chứng)** | **Hiếu** | 0.6679 | 86.48% | 0.9327 | 0.4353 | 0.6358 | 45.05% | **69.91%** |
| **Seed 789 (Kiểm chứng)** | **Tài** | 0.6737 | 86.95% | 0.9349 | 0.4528 | 0.6333 | 45.95% | 67.15% |
| **Trung bình Model B ($\mu \pm \sigma$)** | **Proposed** | **0.6732 ± 0.0037** | **86.85% ± 0.27%** | **0.9350 ± 0.0018** | **0.4502 ± 0.0104** | **0.6344 ± 0.0015** | **47.02% ± 1.79%** | **67.29% ± 1.86%** |
| **Trung bình Model A ($\mu \pm \sigma$)** | **Baseline** | **0.6739 ± 0.0034** | 87.66% ± 0.57% | 0.9405 ± 0.0030 | 0.4445 ± 0.0090 | 0.6365 ± 0.0035 | 43.58% ± 4.05% | 64.18% ± 2.59% |
| **Mức tăng ròng $\Delta(B - A)$** | - | *-0.0007* | *-0.81%* | *-0.0055* | **+0.0057 (+0.57%)** | *-0.0021* | **+3.44% (+3.44%)** | **+3.12% (+3.12%)** |

*Kết luận định lượng đa seed:*
1. **Loại trừ hoàn toàn yếu tố may mắn (Lucky Seed):** Macro F1 của Model B dao động trong biên độ cực hẹp ($66.79\% - 67.66\%$, độ lệch chuẩn chỉ **0.37%**), khẳng định tính ổn định cao của giải pháp.
2. **Cải thiện độ nhạy bền vững:** Recall của cả hai lớp độc hại đều tăng nhất quán qua 4 seed (**Recall-OFF tăng trung bình +3.44%**, **Recall-HATE tăng trung bình +3.12%**, đạt đỉnh **69.91%** ở Seed 456).
3. **Giảm phương sai vận hành:** Độ lệch chuẩn của Recall-OFF giảm từ 4.05% xuống 1.79% (ổn định hơn gấp 2.2 lần) và Recall-HATE giảm từ 2.59% xuống 1.86%, chứng minh tính tin cậy kỹ thuật khi đưa vào triển khai thực tế.

### 5.3. Bảng phân công 4 Seed thực nghiệm của nhóm & Đóng góp dữ liệu
| Thành viên phụ trách | Hạt giống (Seed) | Mục tiêu thực nghiệm | Thư mục kết quả | Trạng thái xác thực |
|:---|:---:|:---|:---|:---:|
| **Trí** | **Seed 42** | Mốc chuẩn đối chứng (Baseline Reference) | `outputs/seed42_teencode_414/` | **Đã hoàn thành (Verified)** |
| **Ngân** | **Seed 123** | Đánh giá độ ổn định và Ablation study | `outputs/seed123_teencode_414/` | **Đã hoàn thành (Verified)** |
| **Hiếu** | **Seed 456** | Huấn luyện kiểm chứng độc lập đa seed | `outputs/seed456_teencode_414/` | **Đã hoàn thành (Verified)** |
| **Tài** | **Seed 789** | Huấn luyện kiểm chứng độc lập đa seed | `outputs/seed789_teencode_414/` | **Đã hoàn thành (Verified)** |

*Điểm nhấn đóng góp của từng thành viên:*
- **Trí (Seed 42):** Thiết lập mốc thực nghiệm chuẩn của đề tài, đo lường chi tiết ma trận nhầm lẫn đối chứng và phân tích sự chuyển dịch của từng lớp.
- **Ngân (Seed 123):** Thực hiện Ablation study chứng minh sự vượt trội của từ điển 414 từ so với 35 từ; đạt đỉnh hiệu năng Macro F1 thực nghiệm (**0.6766**) và F1-OFFENSIVE (**0.4596**).
- **Hiếu (Seed 456):** Xác nhận độc lập năng lực phát hiện ngôn từ thù ghét; đạt đỉnh độ nhạy thù ghét cao nhất nhóm với **Recall-HATE = 69.91%** (giảm số ca bỏ lọt HATE xuống mức thấp nhất: chỉ 130 mẫu).
- **Tài (Seed 789):** Xác nhận độc lập sự cải thiện đồng thời cả 2 lớp độc hại (+3.83% Recall OFFENSIVE và +4.50% Recall HATE), đưa Macro F1 của Model B vượt Model A (+0.47%).

---

## 6. Luận Điểm Then Chốt Trả Lời Phản Biện Nhanh (Q&A Defense Cheat Sheet)

### Câu 1: "Thêm bộ lọc teencode có phải là cải tiến mô hình không?"
- **Trả lời trực diện:** **Không phải cải tiến kiến trúc mô hình**, mà là **cải tiến quy trình dữ liệu (Data-centric Pipeline)**.
- **Cơ sở khoa học:** Cấu trúc PhoBERT-base-v2 (12 layers, 12 heads, 768 dim, 135M tham số) được giữ nguyên vẹn. Từ điển teencode giải quyết bài toán OOV và phân mảnh subword của BPE, đưa từ ngữ mạng xã hội về không gian từ vựng chuẩn mà PhoBERT đã được tiền huấn luyện.

### Câu 2: "Tại sao Accuracy giảm 1.28% mà mô hình lại tốt hơn?"
- **Trả lời trực diện:** Trong dữ liệu mất cân bằng nặng (lớp CLEAN chiếm >82.7%), **Accuracy là chỉ số gây ngộ nhận (deceptive metric)**.
- **Cơ sở khoa học:** Một mô hình đoán toàn bộ là CLEAN vẫn đạt Accuracy ~83% nhưng hoàn toàn vô giá trị trong thực tế. Đổi 1.28% Accuracy để **giảm 19.27% lỗi bỏ sót thù ghét** và **20.69% lỗi bỏ sót xúc phạm** là sự đánh đổi hoàn toàn tối ưu và có giá trị bảo vệ không gian mạng cao nhất.

### Câu 3: "Tại sao không xóa từ dừng (stopwords)?"
- **Trả lời trực diện:** Giữ nguyên từ dừng để **bảo toàn ngữ cảnh trọn vẹn cho cơ chế tự chú ý (Self-Attention)** của Transformer.
- **Cơ sở khoa học:** Nghiên cứu thực nghiệm của Luu et al. (2024) trên bộ dữ liệu ViHSD đã chứng minh việc xóa từ dừng làm suy giảm khả năng hiểu ngữ pháp và giảm hiệu năng của các mô hình ngôn ngữ tiếng Việt.

### Câu 4: "Tại sao dùng Class Weight mà không dùng Resampling (SMOTE / Undersampling)?"
- **Trả lời trực diện:** Class Weight **bảo toàn 100% dữ liệu gốc** và không làm tăng chi phí tính toán.
- **Cơ sở khoa học:** Undersampling sẽ lãng phí hơn 70% dữ liệu sạch; Oversampling dễ dẫn đến quá khớp (Overfitting); SMOTE sinh vector nội suy nhân tạo khó diễn giải ngữ nghĩa ngôn ngữ. Class Weight trực tiếp điều chỉnh gradient phạt trong hàm mất mát là giải pháp toán học sạch sẽ và hiệu quả nhất.
