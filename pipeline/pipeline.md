### BƯỚC 1: Nạp & Phân chia Bộ dữ liệu ViHSD (33,400 mẫu)

#### 1. Chi tiết bên trong:

Bộ dữ liệu gồm 33,400 bình luận Facebook được chia sẵn thành 3 tệp (`train.csv`, `dev.csv`, `test.csv`).

- **Train (24,048 mẫu - 72%):** Dùng để mô hình "học" (cập nhật trọng số mạng nơ-ron).
- **Validation / Dev (2,672 mẫu - 8%):** Dùng để "chấm thi thử" sau mỗi vòng học nhằm **chọn ra bản mô hình tốt nhất (Best Checkpoint)**. Tuyệt đối không dùng tập này để cập nhật trọng số.
- **Test (6,680 mẫu - 20%):** Dùng để "thi tốt nghiệp" lần cuối cùng. Mô hình hoàn toàn chưa từng thấy tập này trong suốt quá trình học.

#### 2. Ý nghĩa các thông số & Khái niệm:

- **Tỷ lệ phân tầng (Stratified Split):** Dữ liệu bị lệch nặng:
  - `CLEAN (0)`: chiếm ~82.7%
  - `HATE (2)`: chiếm ~10.5%
  - `OFFENSIVE (1)`: chỉ chiếm ~6.8%
    $\rightarrow$ "Phân tầng" nghĩa là khi chia thành 3 tập, **tỷ lệ 83% - 10% - 7% này được giữ nguyên y hệt trên cả Train, Dev và Test**, không để xảy ra tình trạng tập Dev hay Test bị thiếu mất một lớp nào.
- **Tại sao bắt buộc giữ nguyên số lượng mẫu gốc?**
  Nếu bạn tự ý gộp lại rồi chia ngẫu nhiên (ví dụ Test còn 5,000 mẫu), kết quả F1 của bạn sẽ **không thể so sánh** với bài báo quốc tế của Luu et al. (2021) vì đề thi của bạn đã khác đề thi chuẩn.

---

### BƯỚC 2: Tiền xử lý & Làm sạch Dữ liệu

#### 1. Chi tiết bên trong:

Văn bản mạng xã hội chứa rất nhiều ký tự rác, viết tắt, biểu tượng cảm xúc và liên kết web. Cần dọn dẹp trước khi đưa vào mô hình ngôn ngữ.

#### 2. Ý nghĩa từng kỹ thuật:

- **Xóa Emoji & Icon (`emoji.replace_emoji`):**
  - _Tại sao phải xóa?_ Công cụ tách từ tiếng Việt (`RDRSegmenter`) được xây dựng trên từ điển báo chí. Khi gặp emoji dính liền chữ (ví dụ: `ghét😀`), nó sẽ bị lỗi (crash Java) hoặc ghép dính thành một từ dị dạng `ghét_😀`, làm PhoBERT không hiểu được.
- **Lọc URL & Email:** Các link web dài ngoằng (`http://fb.com/...`) không mang ý nghĩa ngôn từ thù ghét. Nếu để nguyên, PhoBERT sẽ băm link đó thành 30 - 40 mảnh từ vô nghĩa, làm tràn giới hạn độ dài của câu.
- **Chuẩn hóa Unicode NFC:**
  - Tiếng Việt có 2 kiểu gõ dấu: **Tổ hợp** (chữ `a` + dấu `sắc`) và **Dựng sẵn - NFC** (chữ `á` duy nhất).
  - Nếu không đưa về chuẩn **NFC**, cùng là chữ "á" nhưng máy tính lại thấy 2 mã nhị phân khác nhau, khiến mô hình tưởng đó là hai từ xa lạ.
- **Chuẩn hóa Teencode & Viết tắt (theo nghiên cứu Tran et al., 2022):**
  - *Tại sao phải chuẩn hóa?* Người dùng mạng xã hội thường dùng từ viết tắt (`ko, k, hổng -> không`, `dc, đc -> được`, `vs -> với`, `ng -> người`) hoặc cố tình chèn dấu để ngụy trang né kiểm duyệt (`đ.ụ`, `đ-ĩ`, `đ_m`).
  - *Tác dụng:* Ánh xạ về từ ngữ chuẩn giúp PhoBERT hiểu trọn vẹn ngữ nghĩa, không bị băm thành các mã con vô nghĩa `<unk>`, giúp tăng đáng kể chỉ số Macro F1 trên tập kiểm tra.
- **Chống rò rỉ dữ liệu (Data Leakage):**
  - Nếu một câu chửi xuất hiện ở cả tập Train và tập Test $\rightarrow$ Mô hình chỉ việc "học vẹt" câu đó ở Train là sang Test làm đúng ngay (gian lận). Loại bỏ câu trùng lặp này khỏi tập Train giúp đánh giá đúng năng lực suy luận thực tế của mô hình.

---

### BƯỚC 3: Tách từ Tiếng Việt & Tokenize

#### 1. Chi tiết bên trong:

Khác với tiếng Anh (mỗi từ cách nhau bằng dấu cách: _I love cat_ = 3 từ), tiếng Việt là ngôn ngữ đơn lập, có **từ ghép** gồm nhiều âm tiết (_"ngôn từ"_, _"thù ghét"_).

#### 2. Ý nghĩa các thông số:

- **Tách từ ghép bằng dấu gạch dưới `_` (`RDRSegmenter` / `PyVi`):**
  - Nếu để nguyên _"ngôn từ"_, PhoBERT hiểu là từ _"ngôn"_ riêng và _"từ"_ riêng.
  - Sau khi tách từ: thành _"ngôn_từ"_ $\rightarrow$ PhoBERT hiểu đây là một khái niệm trọn vẹn duy nhất.
- **`max_length = 128` (Độ dài chuỗi tối đa):**
  - Một câu sau khi mã hóa được giới hạn tối đa 128 "mảnh từ" (tokens), bao gồm cả 2 token đặc biệt bắt buộc của PhoBERT là `<s>` (bắt đầu) và `</s>` (kết thúc).
  - _Tại sao là 128?_ Khảo sát trên ViHSD cho thấy >98% bình luận Facebook dài dưới 100 từ. Chọn 128 là vừa bao trọn hầu hết bình luận, vừa giúp GPU chạy nhanh gấp 4 lần so với đặt 512.
- **Đệm động theo nhóm (`Dynamic Padding`):**
  - Thay vì ép mọi câu ngắn dài đều phải đệm số 0 cho đủ 128 (lãng phí bộ nhớ), Dynamic Padding chỉ đệm các câu trong cùng một nhóm (batch) bằng độ dài của câu dài nhất nhóm đó. Điều này giúp tiết kiệm 40% VRAM GPU.

---

### BƯỚC 4: Tính toán Trọng số Lớp (Class Weight)

#### 1. Chi tiết bên trong:

Đây là **trọng tâm kỹ thuật của đề tài**. Bộ dữ liệu có 83% là CLEAN, nếu mô hình cứ đoán bừa mọi câu đều là CLEAN thì độ chính xác (Accuracy) đã đạt 83%, nhưng bỏ sót 100% câu HATE (vô dụng). Kỹ thuật này sinh ra để "phạt nặng" thói quen lười biếng đó.

#### 2. Ý nghĩa công thức toán học:

Công thức tính trọng số cân bằng (Balanced Weight) của scikit-learn:
$$w_c = \frac{N}{3 \times N_c}$$

- $N$: Tổng số mẫu tập Train (gốc là 24,048 câu; sau khi khử trùng lặp và chống rò rỉ còn 21,098 mẫu sạch).
- $3$: Số lượng nhãn (CLEAN, OFFENSIVE, HATE).
- $N_c$: Số lượng mẫu của lớp $c$.

**Ý nghĩa các con số sinh ra (tính trên tập Train sạch thực tế):**

- **$w_{\text{CLEAN}} \approx 0.41$ ($0.4075$, nhỏ hơn 1):** Vì lớp này quá nhiều mẫu (17,258 câu - 81.8%), khi mô hình đoán sai câu CLEAN, điểm phạt bị **giảm bớt đi hơn một nửa**.
- **$w_{\text{HATE}} \approx 2.94$ ($2.9401$, gấp ~7.2 lần CLEAN):** Lớp thù ghét có 2,392 mẫu (11.3%), nếu mô hình bỏ sót 1 câu HATE, nó bị **phạt nặng gấp 7.2 lần** so với bỏ sót câu CLEAN.
- **$w_{\text{OFFENSIVE}} \approx 4.86$ ($4.8568$, gấp ~11.9 lần CLEAN):** Lớp này cực hiếm (1,448 mẫu - 6.9%), nên mức phạt là **cao nhất (~11.9 lần)** để buộc mạng nơ-ron phải chú ý học đặc trưng của nó.

---

### BƯỚC 5: Huấn luyện Đối chứng A/B (4 Seeds ngẫu nhiên)

#### 1. Chi tiết bên trong:

Huấn luyện 2 nhánh độc lập để so sánh công bằng:

- **Nhánh A (Baseline):** Dùng hàm mất mát thông thường `nn.CrossEntropyLoss()` $\rightarrow$ Phạt đều mọi câu như nhau.
- **Nhánh B (Proposed):** Dùng hàm mất mát có trọng số `nn.CrossEntropyLoss(weight=class_weights)` $\rightarrow$ Áp dụng mức phạt đã tính ở Bước 4.

#### 2. Ý nghĩa từng siêu tham số (Hyperparameters):

- **`learning_rate = 2e-5` ($0.00002$):** Tốc độ học. Vì PhoBERT đã được huấn luyện sẵn trên 20GB báo chí tiếng Việt (nó đã rất thông minh), ta chỉ đang tinh chỉnh nhẹ (fine-tuning) cho bài toán Hate Speech. Nếu đặt tốc độ học quá lớn (ví dụ $10^{-3}$), toàn bộ tri thức cũ của PhoBERT sẽ bị phá hủy ("Catastrophic Forgetting").
- **`batch_size = 16`:** Mỗi bước cập nhật trọng số sẽ đọc 16 câu. Con số 16 vừa vặn với bộ nhớ 16GB của GPU T4, giúp gradient ổn định.
- **`epochs = 4`:** Cho toàn bộ tập Train chạy qua mạng 4 lần. Transformer thường hội tụ rất nhanh, chạy 3 - 4 epochs là đạt đỉnh, nếu chạy 10 - 20 epochs mô hình sẽ bị "học vẹt" (Overfitting).
- **`fp16 = True` (Nửa độ chính xác 16-bit):** Thay vì dùng số thực 32-bit (`float32`), chuyển sang 16-bit (`float16`). Tác dụng: Giảm 50% dung lượng VRAM GPU và tăng tốc độ học lên 2.5 lần mà không suy giảm chất lượng.
- **`max_grad_norm = 1.0` (Cắt xén gradient):** Vì mô hình B bị nhân hệ số phạt lên tới $\approx 5.0$, gradient có thể bị vọt lên quá lớn làm nổ mạng (Exploding Gradient). Tham số này giữ cho độ lớn gradient không vượt quá 1.0.
- **4 Seeds ngẫu nhiên (`42, 123, 456, 789`):**
  - Đổi seed = đổi góc khởi tạo ngẫu nhiên ban đầu của mạng.
  - 4 thành viên chạy 4 seed khác nhau để chứng minh: **Mô hình B tốt hơn A là do thuật toán thực sự hiệu quả, chứ không phải do may mắn chọn trúng 1 seed đẹp.**

---

### BƯỚC 6: Đánh giá Đa chiều trên Test & Bàn giao

#### 1. Chi tiết bên trong:

Sau khi có mô hình tốt nhất, cho chạy dự đoán trên 6,680 câu của tập Test (chưa từng dùng trong lúc học) và tính điểm.

#### 2. Ý nghĩa các chỉ số đánh giá:

- **Macro F1-score (MF1 - Chỉ số quan trọng nhất):**
  $$\text{Macro F1} = \frac{\text{F1}_{\text{CLEAN}} + \text{F1}_{\text{OFFENSIVE}} + \text{F1}_{\text{HATE}}}{3}$$
  - _Tại sao nó là chỉ số chính?_ Vì nó chia đều trọng số cho cả 3 lớp. Dù lớp HATE ít câu, nó vẫn đóng góp 1/3 tổng điểm. Mô hình chỉ đạt Macro F1 cao khi nó đoán tốt **cả 3 lớp**.
- **Weighted F1-score (WF1 - Chỉ số phụ):** Tính trung bình F1 nhưng nhân với tỷ lệ số câu của từng lớp. Vì lớp CLEAN chiếm 83%, chỉ số này thường rất cao (>85%) và dễ gây ngộ nhận (ảo tưởng mô hình quá tốt dù bỏ sót câu HATE).
- **Per-Class Precision & Recall (Bản chất của Class Weight):**
  - **Recall của HATE/OFFENSIVE tăng lên:** Chứng minh mô hình đã dũng cảm bắt được nhiều câu thù ghét hơn, ít bỏ sót hơn.
  - **Precision giảm nhẹ:** Chấp nhận đánh đổi, đôi khi mô hình bắt nhầm một câu sạch thành câu nhạy cảm (thà bắt nhầm còn hơn bỏ sót).
- **Ma trận nhầm lẫn (Confusion Matrix - Công cụ trực quan hóa lỗi):**
  - Vì có 3 lớp nên ma trận có dạng bảng kích thước $3 \times 3$ (Hàng: nhãn thật; Cột: nhãn mô hình đoán).
  - *Đường chéo chính:* Số lượng câu mô hình đoán đúng của từng lớp (`CLEAN -> CLEAN`, `OFFENSIVE -> OFFENSIVE`, `HATE -> HATE`).
  - *Ý nghĩa then chốt:* 
    - Thấy rõ mô hình đang "bắt nhầm" câu thuộc lớp nào sang lớp nào (ví dụ: có bao nhiêu câu thù ghét `HATE` bị mô hình lười biếng đoán bừa thành câu sạch `CLEAN`?).
    - Cho phép vẽ 2 ma trận của **Mô hình A (Baseline)** và **Mô hình B (Class Weight)** cạnh nhau để giáo viên nhìn thấy ngay số lượng câu thù ghét `HATE` đoán đúng tăng lên rõ rệt như thế nào.
    - Được xuất ra dưới 2 dạng: Ma trận số lượng câu tuyệt đối (tổng 9 ô = đúng 6,680 mẫu test) và Ma trận chuẩn hóa theo phần trăm hàng (Normalized Recall).
- **Độ lệch chuẩn ($\mu \pm \sigma$):** Ví dụ kết quả là $64.40\% \pm 0.32\%$. Con số $\pm 0.32\%$ càng nhỏ chứng tỏ mô hình chạy rất ổn định qua các lần thử nghiệm khác nhau.
- **Tệp bàn giao `predictions.csv`:** Chứa 4 cột bắt buộc: `id` (mã câu), `text` (nội dung gốc), `ground_truth` (nhãn thực tế), `predicted_label` (nhãn mô hình đoán: CLEAN, OFFENSIVE, HATE) để giáo viên chạy script chấm điểm tự động.
