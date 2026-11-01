# Quy trình Phân loại Đa lớp ViHSD với PhoBERT-base (Thực nghiệm Đối chứng Class Weight)

Quy trình chuẩn hóa và thực nghiệm đối chứng giữa hai mô hình:
- **Mô hình A (Baseline):** PhoBERT-base huấn luyện với hàm mất mát Entropy chéo tiêu chuẩn (Standard Cross-Entropy Loss), không sử dụng trọng số lớp.
- **Mô hình B (Proposed):** PhoBERT-base huấn luyện với hàm mất mát Entropy chéo có trọng số lớp (Weighted Cross-Entropy Loss), nhằm xử lý hiện tượng mất cân bằng lớp nghiêm trọng.

Cả hai mô hình sử dụng cùng một phiên bản dữ liệu sau khi chuẩn hóa, cùng kiến trúc mạng, siêu tham số tối ưu và quy trình chọn mô hình; điểm khác biệt duy nhất nằm ở việc áp dụng trọng số lớp trong hàm mất mát.

---

## 1. Cấu hình Thí nghiệm Chuẩn tắc

| Thành phần | Đặc tả kỹ thuật | Ghi chú & Ràng buộc |
| :--- | :--- | :--- |
| **Mô hình nền tảng** | `vinai/phobert-base-v2` | Kiến trúc RoBERTa-base tiền huấn luyện cho tiếng Việt; Fine-tune toàn bộ tham số. |
| **Bộ dữ liệu** | **ViHSD** (Luu et al., 2021) | 33,400 mẫu; Tỷ lệ phân chia chuẩn: **72% Train - 8% Dev - 20% Test** (Stratified). |
| **Hệ thống nhãn** | `0 = CLEAN, 1 = OFFENSIVE, 2 = HATE` | Nhãn phân loại đơn (Single-label multi-class). |
| **Hạ tầng tính toán** | Google Colab GPU Tesla T4 (16GB VRAM) | Tính toán dấu phẩy động nửa chính xác (`fp16=True`); Lưu trữ bền vững trên Google Drive. |
| **Độ dài chuỗi đầu vào** | `max_length = 128` | Cắt cụt (truncation) tối đa 128 tokens; Đệm động theo batch (`dynamic padding`). |
| **Kích thước nhóm mẫu** | Train Batch Size = 16; Eval Batch Size = 32 | Batch size 16 cho train nhằm cập nhật gradient ổn định; 32 cho eval nhằm tối ưu tốc độ. |
| **Số vòng học (Epochs)** | 4 vòng (`epochs = 4`) | Đảm bảo mô hình đạt trạng thái hội tụ tối ưu mà không bị overfitting nặng. |
| **Bộ tối ưu & Tốc độ học** | AdamW; `learning_rate = 2e-5`; `weight_decay = 0.01` | Thiết lập chuẩn cho tinh chỉnh kiến trúc Transformer. |
| **Lịch điều chỉnh tốc độ** | Tuyến tính (Linear warmup & decay) | Tăng dần trong 10% tổng số bước đầu (warmup ratio = 0.1), sau đó giảm đều về 0. |
| **Cắt xén Gradient** | `max_grad_norm = 1.0` | Ngăn chặn hiện tượng bùng nổ gradient, đặc biệt ở mô hình B có trọng số phạt lớn. |
| **Hàm mất mát Mô hình A** | Standard Cross-Entropy (`reduction='mean'`) | Lấy trung bình số học theo số mẫu trong batch: $\mathcal{L}_A = \frac{1}{B} \sum_{i=1}^B \mathcal{L}_i$. |
| **Hàm mất mát Mô hình B** | Weighted Cross-Entropy (`reduction='mean'`) | Tính trọng số cân bằng từ tập Train; chuẩn hóa theo tổng trọng số: $\mathcal{L}_B = \frac{\sum w_{y_i} \mathcal{L}_i}{\sum w_{y_i}}$. |
| **Hạt giống ngẫu nhiên** | Phân tán 4 Seeds: `[42, 123, 456, 789]` | 4 thành viên trong nhóm chạy độc lập 4 seeds song song trên 4 môi trường Colab. |
| **Tiêu chí chọn Checkpoint** | **Macro F1 cao nhất trên tập Dev** | Làm tròn 4 chữ số thập phân; nếu bằng điểm (tie), chọn checkpoint có `dev_loss` nhỏ hơn. |

---

## 2. Chi tiết Quy trình Thực thi (End-to-End Pipeline)

```
[Bộ dữ liệu ViHSD (33,400 mẫu)] 
       │
       ▼
[1. Phân chia Dữ liệu]             ──► Tỷ lệ chuẩn: 72% Train (24,048) : 8% Dev (2,672) : 20% Test (6,680)
       │
       ▼
[2. Tiền xử lý & Chuẩn hóa Dữ liệu] ──► Khử trùng lặp, Chuẩn hóa Teencode, Xóa Emoji, Chuẩn hóa NFC
       │
       ▼
[3. Tách từ Tiếng Việt & Mã hóa]   ──► RDRSegmenter (VnCoreNLP) / PyVi ──► PhoBERT Tokenizer (max_len=128)
       │
       ▼
[4. Tính toán Trọng số Lớp]        ──► scikit-learn Balanced Class Weight (chỉ tính trên Train sạch)
       │
       ├───► [5A. Huấn luyện Mô hình A] (Standard CE) ──► Đánh giá Dev mỗi epoch ──► Lưu Best Checkpoint A
       │
       └───► [5B. Huấn luyện Mô hình B] (Weighted CE) ──► Đánh giá Dev mỗi epoch ──► Lưu Best Checkpoint B
                                                               │
                                                               ▼
[6. Đánh giá Đa chiều trên Test]   ◄───────────────────────────┴── Suy luận argmax trên Test Set
       │
       ├──► Macro F1, Weighted F1, Per-class Precision / Recall / F1
       ├──► Confusion Matrix (Tuyệt đối & Chuẩn hóa)
       └──► Xuất `predictions_A.csv`, `predictions_B.csv` & Báo cáo đối chứng
```

### Bước 1: Phân chia Bộ dữ liệu & Tỷ lệ Chia (Dataset Partitioning & Split Ratio)

Bộ dữ liệu ViHSD bao gồm **33,400 bình luận Facebook** được phân chia sẵn theo tỷ lệ chuẩn tắc của bài báo gốc (Luu et al., EMNLP 2021):

| Phân tập (Split) | Tỷ lệ (%) | Số lượng mẫu | CLEAN (0) | OFFENSIVE (1) | HATE (2) | Tỷ lệ phân bố lớp |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (Huấn luyện)** | **72.0%** | 24,048 | 19,886 (82.69%) | 1,606 (6.68%) | 2,556 (10.63%) | 12.4 : 1.0 : 1.6 |
| **Validation / Dev (Chọn mô hình)** | **8.0%** | 2,672 | 2,190 (81.96%) | 212 (7.93%) | 270 (10.10%) | 10.3 : 1.0 : 1.3 |
| **Test (Kiểm tra độc lập)** | **20.0%** | 6,680 | 5,548 (83.05%) | 444 (6.65%) | 688 (10.30%) | 12.5 : 1.0 : 1.5 |
| **Tổng cộng** | **100.0%** | **33,400** | **27,624 (82.71%)** | **2,262 (6.77%)** | **3,514 (10.52%)** | **12.2 : 1.0 : 1.6** |

**Nguyên tắc và Cơ chế phân chia:**
1. **Phân chia phân tầng (Stratified Splitting):** Bộ dữ liệu được chia theo cơ chế phân tầng (Stratified Sampling), đảm bảo tỷ lệ mất cân bằng giữa 3 lớp nhãn (`CLEAN` ~82.7%, `OFFENSIVE` ~6.8%, `HATE` ~10.5%) được bảo toàn đồng nhất trên cả 3 tập Train, Dev và Test.
2. **Quy tắc bảo toàn phân chia chuẩn (Benchmark Preservation):**
   - Đề tài **bắt buộc sử dụng đúng 3 phân tập `train.csv`, `dev.csv`, `test.csv` gốc** do nhóm tác giả ViHSD cung cấp. 
   - Không tự ý gộp rồi chia ngẫu nhiên lại từ đầu vì việc giữ nguyên tập Test chuẩn (6,680 mẫu) là điều kiện tiên quyết để đối chiếu công bằng với các công trình nghiên cứu đã công bố (Luu et al. 2021, Tran et al. 2022).
3. **Quy định trong trường hợp tự chia lại (Optional Cross-Validation / Re-split):**
   - Nếu có yêu cầu thực nghiệm phân chia lại, phải áp dụng hàm `train_test_split` của `scikit-learn` với tham số `stratify=y`, giữ đúng tỷ lệ 70:10:20 (hoặc 72:8:20) và cố định hạt giống `random_state=42`.

### Bước 2: Kiểm toán Toàn vẹn và Làm sạch Dữ liệu (Data Auditing & Cleaning)

1. **Chuẩn hóa lược đồ (Schema):**
   - Đảm bảo cả 3 tập Train, Dev, Test đồng nhất 3 cột: `id` (int/str), `text` (str), `label` (int: 0, 1, 2).
   - Kiểm tra ràng buộc nhãn: giá trị `label` bắt buộc thuộc tập $\{0, 1, 2\}$. Dừng pipeline và báo lỗi nếu xuất hiện nhãn bất hợp lệ.

2. **Quy tắc xử lý trùng lặp và rò rỉ dữ liệu (Deduplication & Data Leakage):**
   - **Trùng lặp nội bộ tập Train:**
     - Nếu phát hiện các dòng trùng lặp 100% về văn bản và cùng nhãn: giữ lại 1 bản ghi duy nhất, xóa các bản ghi trùng lặp để tránh overfit.
     - Nếu phát hiện trùng lặp văn bản nhưng mâu thuẫn nhãn (do nhiều người gán nhãn khác nhau): áp dụng quy tắc bỏ phiếu đa số (Majority Voting) nếu có số đông; nếu tỷ lệ mâu thuẫn ngang nhau (1:1), loại bỏ hoàn toàn các mẫu này khỏi tập Train để loại bỏ nhiễu nhãn.
   - **Rò rỉ dữ liệu giữa Train và Dev/Test (Data Leakage):**
     - Tuyệt đối bảo toàn tập Dev và Test. Nếu phát hiện câu trong tập Train trùng văn bản với câu trong Dev hoặc Test, loại bỏ câu đó khỏi tập Train.
   - **Bảo toàn tập Dev và Test:**
     - Tuyệt đối **không xóa mẫu, không gộp và không thay đổi nhãn** của tập Dev và Test nhằm đảm bảo tính công bằng và khả năng so sánh chuẩn mực (benchmark comparability) với các nghiên cứu công bố quốc tế.

3. **Quy chuẩn làm sạch văn bản (Text Normalization Pipeline):**
   - **Chuẩn hóa Unicode:** Áp dụng Unicode chuẩn dựng sẵn **NFC** (`unicodedata.normalize('NFC', text)`) để đồng nhất các biến thể dấu tiếng Việt.
   - **Xóa Emoji & Ký tự tượng hình:** Loại bỏ toàn bộ ký tự emoji và biểu tượng đồ họa Unicode (`emoji.replace_emoji(text, replace='')` hoặc regex Unicode Blocks) nhằm loại bỏ nhiễu ký tự lạ trước khi tách từ.
   - **Xử lý URL và địa chỉ mạng:** Thay thế toàn bộ liên kết (`http\S+|www\S+`) và email bằng khoảng trắng, tránh việc tokenizer băm link thành chuỗi subword rác.
   - **Chuẩn hóa Teencode & Viết tắt (Teencode Normalization - theo Tran et al., 2022):**
     - Áp dụng từ điển ánh xạ các từ viết tắt, tiếng lóng và biến thể teencode phổ biến trên mạng xã hội về dạng tiếng Việt chuẩn:
       - Từ thông dụng: `ko, k, hổng` $\rightarrow$ `không`; `dc, đc` $\rightarrow$ `được`; `vs` $\rightarrow$ `với`; `ng` $\rightarrow$ `người`; `bt` $\rightarrow$ `biết`; `j` $\rightarrow$ `gì`; `ntn` $\rightarrow$ `như thế nào`.
       - Ký tự cố tình chèn dấu né bộ lọc: loại bỏ các dấu chấm, gạch ngang xen giữa các chữ cái nhằm ngụy trang từ ngữ công kích (ví dụ: `đ.ụ` $\rightarrow$ `đụ`, `đ-ĩ` $\rightarrow$ `đĩ`).
       - Giúp khôi phục ngữ nghĩa nguyên bản trước khi tách từ, tránh việc tokenizer băm các từ teencode thành subword `<unk>` (tăng đáng kể chỉ số Macro F1 theo thực nghiệm của Tran et al., 2022).
   - **Xử lý khoảng trắng và ký tự điều khiển:** Chuyển đổi các ký tự xuống dòng (`\n`, `\r`), tab (`\t`) và khoảng trắng liên tiếp `\s+` thành một khoảng trắng đơn, sau đó `.strip()`.
   - **Xử lý câu rỗng sau làm sạch:**
     - Trên tập Train: Nếu một câu sau khi xóa emoji và URL trở thành rỗng (`""`), loại bỏ dòng đó khỏi tập Train.
     - Trên tập Dev/Test: Nếu câu trở thành rỗng, giữ nguyên dòng đó và thay thế bằng chuỗi ký tự giữ chỗ an toàn (placeholder, ví dụ `"trống"`) để mô hình vẫn sinh ra dự đoán hợp lệ, bảo đảm đủ số lượng mẫu test đầu ra.

### Bước 3: Tách từ Tiếng Việt và Mã hóa Tokenizer (Word Segmentation & Tokenization)

1. **Tách từ tiếng Việt (Word Segmentation):**
   - PhoBERT hoạt động trên đơn vị từ tiếng Việt (các từ ghép có nghĩa nối với nhau bằng dấu gạch dưới `_`, ví dụ: `ngôn_từ`, `thù_ghét`).
   - Sử dụng công cụ tách từ chuẩn khuyến nghị: `RDRSegmenter` (từ `py_vncorenlp` hoặc VnCoreNLP). (Trong trường hợp hạn chế môi trường Java trên Colab, có thể dùng `pyvi` nhưng phải cố định nhất quán cho cả 2 lượt A và B).
   - Lưu trữ bản trung gian sau khi tách từ (`train_segmented.csv`, `dev_segmented.csv`, `test_segmented.csv`) để tránh việc chạy lại tách từ gây tốn thời gian.

2. **Mã hóa Tokenizer (Tokenization):**
   - Tải tokenizer chuẩn từ `vinai/phobert-base-v2` bằng `AutoTokenizer.from_pretrained(..., use_fast=False)`.
   - Cắt cụt (truncation) với `max_length = 128` (bao gồm cả 2 token đặc biệt `<s>` và `</s>`).
   - Sử dụng **Đệm động theo batch (`dynamic padding`)** thông qua `DataCollatorWithPadding` khi đưa vào DataLoader, giúp tối ưu bộ nhớ GPU và tăng tốc độ xử lý hơn so với đệm cố định 128 tokens.

### Bước 4: Tính toán Trọng số Lớp (Class Weight Formulation)

Trọng số lớp được tính toán **chỉ từ tập Train thực tế** sau khi đã hoàn thành làm sạch và xóa trùng lặp:

1. **Công thức Balanced Class Weight:**
   $$w_c = \frac{N}{K \cdot N_c}$$
   *Trong đó:*
   - $N$: Tổng số mẫu trong tập Train sạch.
   - $K$: Tổng số lớp ($K = 3$).
   - $N_c$: Số lượng mẫu của lớp $c \in \{0, 1, 2\}$.

2. **Kiểm tra tính hợp lệ và chuẩn hóa:**
   - Kiểm tra điều kiện: $N_c > 0$ với mọi $c$, $w_c > 0$ và hữu hạn (không chứa `NaN` hay `Inf`).
   - Lưu trữ mảng trọng số theo đúng thứ tự nhãn: $[w_0, w_1, w_2]$ tương ứng `[CLEAN, OFFENSIVE, HATE]`.
   - Chuyển đổi thành PyTorch Tensor: `torch.tensor([w0, w1, w2], dtype=torch.float).to(device)`.

### Bước 5: Thiết kế Huấn luyện Đối chứng A/B Độc lập

Quy trình thực hiện tuần tự trên GPU Tesla T4 của Google Colab (được lặp lại qua danh sách hạt giống cố định, ví dụ `SEEDS = [42, 123, 2024]` để tính trung bình và độ lệch chuẩn $\mu \pm \sigma$):

1. **Mô hình A (Baseline):**
   - Khởi tạo seed $s \in \text{SEEDS}$ (`seed_everything(s)`).
   - Nạp mô hình gốc `vinai/phobert-base-v2` với một Classification Head mới (Linear layer ngẫu nhiên).
   - Huấn luyện với hàm mất mát `torch.nn.CrossEntropyLoss(reduction='mean')`.
   - Đánh giá sau mỗi epoch trên tập Dev; lưu lại checkpoint có Macro F1 cao nhất.
   - Kết thúc lượt A: Giải phóng bộ nhớ GPU (`del model, trainer`, gọi `torch.cuda.empty_cache()` và thu gom rác bộ nhớ `gc.collect()`).

2. **Mô hình B (Proposed):**
   - Khởi tạo lại cùng seed $s$ (`seed_everything(s)`) và đặt lại Generator của DataLoader để đảm bảo thứ tự lấy mẫu mini-batch giống hệt lượt A.
   - Nạp lại mô hình gốc `vinai/phobert-base-v2` từ pre-trained ban đầu (tuyệt đối không dùng mô hình A đã học).
   - Huấn luyện với hàm mất mát `torch.nn.CrossEntropyLoss(weight=class_weights_tensor, reduction='mean')`.
   - Đánh giá sau mỗi epoch trên tập Dev; lưu lại checkpoint có Macro F1 cao nhất.

3. **Chiến lược Checkpoint & Phục hồi (Checkpointing & Resumption):**
   - Lưu trữ trực tiếp lên thư mục Google Drive:
     - Checkpoint tốt nhất (Best Checkpoint): lưu trọng số mô hình đạt Macro F1 cao nhất trên tập Dev.
     - Checkpoint gần nhất (Latest Checkpoint): lưu trạng thái vòng học gần nhất (bao gồm weights, optimizer, lr_scheduler, scaler FP16 và RNG states) kèm cấu hình `save_total_limit = 1` để không gây tràn ổ đĩa Google Drive.
   - Nếu phiên làm việc Colab bị ngắt kết nối đột ngột, mã nguồn tự động kiểm tra và tiếp tục huấn luyện từ checkpoint gần nhất thông qua `trainer.train(resume_from_checkpoint=True)`.

### Bước 6: Đánh giá Đa chiều và Suy luận Kiểm tra (Evaluation Framework)

Sau khi hoàn tất cả 4 vòng học của cả A và B:
1. Cố định checkpoint tốt nhất của A và B đã được chọn thông qua tập Dev.
2. Thực hiện suy luận (Inference) không tính gradient (`torch.no_grad()`) trên tập Test gốc bằng hàm lấy nhãn xác suất cực đại $\hat{y} = \arg\max(\text{logits})$.
3. **Hệ thống chỉ số đánh giá toàn diện:**
   - **Macro-averaged F1 (Chỉ số quyết định chính - Primary Metric):** Trung bình không trọng số của F1 cả 3 lớp:
     $$\text{Macro F1} = \frac{\text{F1}_{\text{CLEAN}} + \text{F1}_{\text{OFFENSIVE}} + \text{F1}_{\text{HATE}}}{3}$$
   - **Chỉ số đánh giá chi tiết từng lớp (Per-Class Precision, Recall, F1):**
     - Đánh giá cụ thể mức độ đánh đổi giữa Precision và Recall khi áp dụng Class Weight trên từng lớp, đặc biệt là hai lớp thiểu số `OFFENSIVE` và `HATE`.
     - Tính mức chênh lệch: $\Delta \text{Recall} = \text{Recall}_B - \text{Recall}_A$ và $\Delta \text{Precision} = \text{Precision}_B - \text{Precision}_A$.
   - **Weighted-averaged F1 (Chỉ số tham khảo phụ - Secondary Metric):** Tính theo số mẫu thực tế của tập Test (đặt `zero_division = 0`).
   - **Ma trận nhầm lẫn (Confusion Matrix):**
     - Ma trận số lượng tuyệt đối (Absolute Counts) để kiểm tra tổng số mẫu dự đoán khớp đúng kích thước tập Test.
     - Ma trận chuẩn hóa theo hàng (Row-normalized / Recall-oriented) để nhìn rõ tỷ lệ dự đoán chính xác của từng lớp.

---

## 3. Đặc tả Sản phẩm Bàn giao (Deliverables Specification)

1. **Tệp dự đoán nộp bài `predictions.csv` (Yêu cầu bắt buộc):**
   - **Tệp nộp bài chính thức:** `predictions.csv` chứa toàn bộ kết quả dự đoán trên 6,680 mẫu của tập Test từ mô hình tốt nhất (Mô hình B - có áp dụng Class Weight).
   - Tệp có cấu trúc chính xác gồm 4 cột theo đúng đặc tả đề bài:
     - `id`: Mã định danh mẫu test gốc (giữ nguyên thứ tự dòng).
     - `text`: Văn bản gốc ban đầu của mẫu test (trước khi tách từ).
     - `ground_truth`: Nhãn thực tế dạng chuỗi (`CLEAN`, `OFFENSIVE`, `HATE`).
     - `predicted_label`: Nhãn mô hình dự đoán dạng chuỗi (`CLEAN`, `OFFENSIVE`, `HATE`).
   - **Các tệp dự đoán đối chứng theo Seed:** `predictions_A_seed{s}.csv` (Baseline không trọng số) và `predictions_B_seed{s}.csv` (Class Weight) cho 4 hạt giống `s` $\in \{42, 123, 456, 789\}$.
   - Tệp số liệu đánh giá trung gian `metrics_seed{s}.json` và tệp tổng hợp kết quả đối chứng: `aggregated_metrics.csv` (chứa đầy đủ $\mu \pm \sigma$ nộp kèm).

2. **Mô hình và Tài nguyên lưu trữ trên Google Drive:**
   - Thư mục lưu mô hình tốt nhất của A và B: chứa `model.safetensors` (hoặc `pytorch_model.bin`), `config.json`.
   - Tokenizer đầy đủ: `vocab.txt`, `bpe.codes`, `tokenizer_config.json`, `special_tokens_map.json`.
   - Tệp lưu trữ siêu tham số và trọng số lớp: `class_weights.json`.

3. **Báo cáo Khoa học Đối chứng (Comparative Technical Report):**
   - Bảng phân bố dữ liệu trước và sau khi làm sạch; giá trị trọng số lớp đã tính.
   - Bảng đối chiếu định lượng chi tiết giữa Mô hình A và B trên tập Test:
     - Precision, Recall, F1 riêng biệt cho từng lớp: CLEAN, OFFENSIVE, HATE.
     - Macro F1, Weighted F1, Accuracy.
     - Cột chênh lệch $\Delta (B - A)$ để kết luận tác động của Class Weight.
   - Hình ảnh trực quan hóa Ma trận nhầm lẫn đối chiếu cạnh nhau (Side-by-side Confusion Matrix).
   - Phân tích định tính các ca lỗi điển hình: Phân loại các trường hợp câu chửi thề (OFFENSIVE) bị nhầm sang thù ghét (HATE), câu sạch có từ nhạy cảm (False Positive), và đánh giá sự đánh đổi Precision-Recall.

---

## 4. Kiểm tra Chất lượng Bàn giao (Verification Checklist)

Trước khi nghiệm thu toàn bộ pipeline, bắt buộc hoàn thành các bước kiểm tra tự động sau:
- [ ] Đủ 4 bộ file dự đoán tương ứng 4 seeds: `predictions_A_seed{s}.csv` và `predictions_B_seed{s}.csv` (`s` $\in \{42, 123, 456, 789\}$), mỗi file đúng 6,680 dòng của `test.csv` gốc.
- [ ] Thứ tự và giá trị cột `id` trong mọi file xuất khớp 100% với tệp test gốc.
- [ ] Không có giá trị `NaN` hoặc rỗng trong cột `predicted_label`.
- [ ] Tổng các phần tử trong ma trận nhầm lẫn của mỗi lần chạy bằng đúng kích thước tập Test (6,680).
- [ ] Báo cáo tổng hợp đầy đủ các giá trị Trung bình và Độ lệch chuẩn ($\mu \pm \sigma$) của Macro F1, Weighted F1, và F1 từng lớp trên cả 4 seeds.
- [ ] Mô hình lưu trên Google Drive có thể nạp lại độc lập bằng `AutoModelForSequenceClassification.from_pretrained` và thực hiện suy luận ra đúng kết quả trong file predictions tương ứng.

---

## 5. Cơ sở Khoa học & Tài liệu Tham khảo

- **Bộ dữ liệu ViHSD:** Luu et al., *"A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts"*, EMNLP 2021.
- **Mô hình PhoBERT:** Nguyen & Nguyen, *"PhoBERT: Pre-trained language models for Vietnamese"*, Findings of EMNLP 2020.
- **PhoBERT trên ViHSD:** Tran et al., *"A deep learning approach for hate speech detection on Vietnamese social media texts"*, Neural Computing and Applications, 2022.
- **Xử lý mất cân bằng lớp & Trọng số:** 
  - King & Zeng, *"Logistic Regression in Rare Events Data"*, Political Analysis, 2001 (Cơ sở thuật toán `compute_class_weight` trong scikit-learn).
  - PyTorch Documentation: `torch.nn.CrossEntropyLoss` (Cơ chế tính toán mất mát có trọng số và chuẩn hóa mean reduction).
- **Quy chuẩn tinh chỉnh Transformer:** Devlin et al., *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019 (Cơ sở cho tốc độ học `2e-5`, warmup 10% và AdamW weight decay 0.01).
