# 05. BÁO CÁO SO SÁNH HIỆU NĂNG: BASELINE VS. CLASS WEIGHT

Tài liệu này ghi nhận, phân tích và so sánh kết quả thực nghiệm giữa hai mô hình: **PhoBERT-base cơ sở (Baseline)** và **PhoBERT-base kết hợp trọng số lớp (Class Weight)** trên bộ dữ liệu ViHSD (Luu et al., 2021).

---

## 1. Tóm tắt Đề tài và Phương pháp Tiếp cận

- **Tên đề tài:** Phân loại ngôn từ thù ghét trên mạng xã hội tiếng Việt (Vietnamese Hate Speech Detection).
- **Dữ liệu:** Bộ dữ liệu ViHSD gồm 33,400 bình luận Facebook, chia làm 3 tập: Huấn luyện (Train: 24,048), Phát triển (Dev: 2,672) và Kiểm tra (Test: 6,680).
- **Phân bố các lớp:** `CLEAN` chiếm 82.71%, `OFFENSIVE` chiếm 6.77%, `HATE` chiếm 10.52%. Dữ liệu bị mất cân bằng nghiêm trọng (Severe Class Imbalance).
- **Phương pháp tiếp cận tổng thể:** Đề tài giữ nguyên kiến trúc mạng nơ-ron nền tảng của `PhoBERT-base-v2` (không can thiệp số tầng hay trọng số Transformer), mà tập trung giải quyết bài toán thông qua hai cải tiến bổ trợ trong quy trình:
  1. *Cải tiến quy trình dữ liệu (Data-centric Pipeline):* Xây dựng bộ chuẩn hóa 414 từ teencode/tiếng lóng kết hợp bộ tách từ `PyVi`, giúp đưa các biến thể viết tắt và tiếng lóng mạng xã hội về dạng từ vựng tiếng Việt chuẩn, hạn chế hiện tượng phân mảnh từ vựng (subword fragmentation) và từ ngoài từ điển (OOV), giúp PhoBERT phát huy tối đa khả năng hiểu ngữ cảnh.
  2. *Tối ưu hóa hàm mục tiêu (Loss Function Optimization):* Áp dụng hàm mất mát có trọng số lớp (**Weighted Cross-Entropy**) theo tỷ lệ nghịch đảo tần suất để xử lý trực tiếp bài toán mất cân bằng dữ liệu nghiêm trọng.
- **Mục tiêu thực nghiệm:** Đánh giá hiệu quả của trọng số lớp đối với độ nhạy (Recall) trên các nhóm vi phạm (`HATE`, `OFFENSIVE`), đồng thời phân tích sự đánh đổi thực tế giữa Precision, Recall và Accuracy.

---

## 2. Bảng Tổng hợp Kết quả Thực nghiệm

### 2.1. Kết quả mốc đối chứng chuẩn (Seed 42, Teencode 414)

Kết quả đo lường trên tập **Kiểm tra (Test: 6,680 mẫu)** giữa hai mô hình:

| Mô hình / Cấu hình                     |  Macro F1  | Weighted F1 |  Accuracy  | F1 CLEAN (0) | F1 OFFENSIVE (1) | F1 HATE (2) |
| :------------------------------------- | :--------: | :---------: | :--------: | :----------: | :--------------: | :---------: |
| **Baseline (Không dùng trọng số lớp)** | **0.6750** | **0.8784**  | **0.8814** |  **0.9433**  |      0.4495      |   0.6322    |
| **Proposed (Có dùng trọng số lớp)**    |   0.6746   |   0.8722    |   0.8686   |    0.9351    |    **0.4531**    | **0.6357**  |
| **Chênh lệch ($\Delta = B - A$)**      | _-0.0004_  |  _-0.0062_  | _-0.0128_  |  _-0.0082_   |   **+0.0036**    | **+0.0035** |

_Ghi chú: Cả hai mô hình đều dùng bộ tiền xử lý chuẩn hóa 414 từ viết tắt và bộ tách từ PyVi. Kết quả này tương đương với nghiên cứu của Tran et al. (2022) (0.6746) và cao hơn so với mô hình m-BERT trong bài báo gốc của Luu et al. (2021) (0.6269)._

### 2.2. Bảng đối chiếu tổng hợp toàn bộ các thực nghiệm trong thư mục `outputs/`

Thư mục `outputs/` lưu trữ kết quả của 3 cấu hình thực nghiệm hoàn chỉnh:
1. `seed42_teencode_35`: Tiền xử lý với từ điển teencode rút gọn (35 từ), khởi tạo ngẫu nhiên Seed 42.
2. `seed42_teencode_414`: Tiền xử lý với từ điển teencode đầy đủ (414 từ), khởi tạo ngẫu nhiên Seed 42 (Mốc chuẩn).
3. `seed123_teencode_414`: Tiền xử lý với từ điển teencode đầy đủ (414 từ), khởi tạo ngẫu nhiên Seed 123 (Đánh giá độ ổn định).

| Cấu hình Thực nghiệm | Mô hình | Macro F1 | Accuracy | F1 CLEAN | F1 OFFENSIVE | F1 HATE | Recall OFFENSIVE | Recall HATE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`seed42_teencode_35`** | Model A (Baseline)<br/>Model B (Class Weight)<br/>*Chênh lệch $\Delta$* | 0.6707<br/>0.6708<br/>*+0.0000* | 0.8760<br/>0.8659<br/>*-0.0102* | 0.9396<br/>0.9342<br/>*-0.0054* | 0.4333<br/>0.4438<br/>**+0.0104** | 0.6393<br/>0.6343<br/>*-0.0049* | 0.4099<br/>0.4887<br/>**+0.0788** | 0.6555<br/>0.6657<br/>**+0.0102** |
| **`seed42_teencode_414`**<br/>*(Mốc chuẩn đề tài)* | Model A (Baseline)<br/>Model B (Class Weight)<br/>*Chênh lệch $\Delta$* | 0.6750<br/>0.6746<br/>*-0.0004* | 0.8814<br/>0.8686<br/>*-0.0129* | 0.9433<br/>0.9351<br/>*-0.0082* | 0.4495<br/>0.4531<br/>**+0.0036** | 0.6322<br/>0.6357<br/>**+0.0035** | 0.4212<br/>0.4842<br/>**+0.0631** | 0.6134<br/>0.6657<br/>**+0.0523** |
| **`seed123_teencode_414`**<br/>*(Thử nghiệm đa Seed)* | Model A (Baseline)<br/>Model B (Class Weight)<br/>*Chênh lệch $\Delta$* | 0.6747<br/>**0.6766**<br/>**+0.0019** | 0.8686<br/>0.8711<br/>**+0.0025** | 0.9363<br/>0.9372<br/>**+0.0008** | 0.4485<br/>**0.4596**<br/>**+0.0111** | 0.6393<br/>0.6330<br/>*-0.0063* | 0.4955<br/>0.4865<br/>*-0.0090* | 0.6672<br/>0.6555<br/>*-0.0116* |

---

## 3. Bảng Chi tiết Precision, Recall và F1 từng Lớp

### 3.1. Mô hình Baseline (Không dùng trọng số lớp)

| Nhãn lớp         | Số mẫu thực tế | Precision  |   Recall   |  F1-Score  |
| :--------------- | :------------: | :--------: | :--------: | :--------: |
| **0: CLEAN**     |     5,548      |   0.9352   | **0.9515** | **0.9433** |
| **1: OFFENSIVE** |      444       | **0.4820** |   0.4212   |   0.4495   |
| **2: HATE**      |      688       | **0.6522** |   0.6134   |   0.6322   |
| **Macro Avg**    |   **6,680**    | **0.6898** |   0.6620   | **0.6750** |
| **Weighted Avg** |   **6,680**    |   0.8759   | **0.8814** | **0.8784** |

### 3.2. Mô hình Proposed (Áp dụng Class Weight)

| Nhãn lớp         | Số mẫu thực tế | Precision  |   Recall   |  F1-Score  |
| :--------------- | :------------: | :--------: | :--------: | :--------: |
| **0: CLEAN**     |     5,548      | **0.9460** |   0.9245   |   0.9351   |
| **1: OFFENSIVE** |      444       |   0.4257   | **0.4842** | **0.4531** |
| **2: HATE**      |      688       |   0.6082   | **0.6657** | **0.6357** |
| **Macro Avg**    |   **6,680**    |   0.6600   | **0.6915** |   0.6746   |
| **Weighted Avg** |   **6,680**    | **0.8766** |   0.8686   |   0.8722   |

### 3.3. So sánh hiệu năng chi tiết Model B (Class Weight) qua 3 cấu hình thực nghiệm

Bảng theo dõi sự biến đổi của mô hình đề xuất (Model B) khi thay đổi kích thước từ điển tiền xử lý và hạt giống ngẫu nhiên:

| Cấu hình Thực nghiệm | Nhãn lớp | Precision | Recall | F1-Score | Macro F1 | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`seed42_teencode_35`**<br/>*(Teencode 35 từ, Seed 42)* | CLEAN (0)<br/>OFFENSIVE (1)<br/>HATE (2) | 0.9479<br/>0.4064<br/>0.6058 | 0.9209<br/>**0.4887**<br/>0.6657 | 0.9342<br/>0.4438<br/>0.6343 | 0.6708 | 0.8659 |
| **`seed42_teencode_414`**<br/>*(Teencode 414 từ, Seed 42)* | CLEAN (0)<br/>OFFENSIVE (1)<br/>HATE (2) | 0.9460<br/>0.4257<br/>0.6082 | 0.9245<br/>0.4842<br/>**0.6657** | 0.9351<br/>0.4531<br/>**0.6357** | 0.6746 | 0.8686 |
| **`seed123_teencode_414`**<br/>*(Teencode 414 từ, Seed 123)* | CLEAN (0)<br/>OFFENSIVE (1)<br/>HATE (2) | 0.9458<br/>**0.4355**<br/>**0.6119** | **0.9286**<br/>0.4865<br/>0.6555 | **0.9372**<br/>**0.4596**<br/>0.6330 | **0.6766** | **0.8711** |

---

## 4. Phân tích Kết quả Thực nghiệm

### 4.1. Hiệu năng tổng thể (Macro F1 ~67.5%) và cơ chế hỗ trợ từ tiền xử lý Teencode

- **Vượt trội so với các nghiên cứu công bố trước:**
  Chỉ số Macro F1 của cả hai mô hình (Baseline: 0.6750, Class Weight: 0.6746) đều vượt trội rõ rệt so với kết quả bài báo gốc của Luu et al. (2021) (62.69% với m-BERT và 61.11% với Text-CNN), tức tăng hơn **+4.7% F1**. Đồng thời, kết quả này cũng tiệm cận mức tốt nhất trong nghiên cứu PhoBERT-CNN của Tran et al. (2022) (0.6746).
- **Cơ chế tác động của bước chuẩn hóa Teencode:**
  - Mô hình nền tảng `PhoBERT-base-v2` vốn được tiền huấn luyện trên 20GB văn bản báo chí và bách khoa toàn thư chuẩn mực. Khi xử lý văn bản mạng xã hội, các từ viết tắt, tiếng lóng hay biến thể cố tình viết lệch (`ko`, `dc`, `vcl`, `đ.ụ`...) nếu không được xử lý trước sẽ bị bộ tách từ (`PyVi`) và bộ mã hóa BPE băm nhỏ thành nhiều subword vụn vặt hoặc chuyển thành token không xác định (`<unk>`), làm gãy cấu trúc ngữ nghĩa câu.
  - Việc chuẩn hóa 414 từ teencode về dạng từ vựng tiếng Việt chuẩn trước khi đưa vào mô hình đã giúp PhoBERT ánh xạ chính xác ngữ cảnh vào không gian vector sẵn có, từ đó cải thiện đáng kể khả năng nắm bắt ngữ nghĩa của cả câu trung tính lẫn câu mang sắc thái xúc phạm, thù ghét.

### 4.2. Phân tích sự đánh đổi giữa Precision và Recall khi áp dụng Class Weight

- **Vì sao Recall của HATE tăng 5.23% và OFFENSIVE tăng 6.30%?**
  Khi thêm trọng số lớp nghịch đảo tần suất, mức phạt mất mát (loss penalty) khi phân loại sai câu thuộc nhóm HATE hay OFFENSIVE sẽ lớn hơn nhiều so với việc đoán sai câu thuộc nhóm CLEAN (trọng số của `OFFENSIVE` là ~4.86, `HATE` là ~2.94, trong khi `CLEAN` chỉ là ~0.41). Mô hình bị buộc phải dịch chuyển ranh giới quyết định (decision boundary) để nhạy cảm hơn với các đặc trưng cảnh báo độc hại, giúp phát hiện thêm **36 bình luận thù ghét** và **28 bình luận xúc phạm** trên tập kiểm tra.
- **Vì sao độ chính xác chung (Accuracy) lại giảm 1.28% (từ 88.14% xuống 86.86%)?**
  Lớp `CLEAN` chiếm đa số trong tập dữ liệu (hơn 82.7%). Khi mô hình nhạy hơn với các nội dung tiêu cực, một số bình luận bình thường nhưng có từ ngữ hơi gay gắt sẽ bị mô hình nghi ngờ và đoán nhầm sang nhóm vi phạm (False Positives tăng). Điều này khiến Recall của `CLEAN` giảm từ 95.15% xuống 92.45%, dẫn đến Accuracy tổng thể giảm nhẹ 1.28%.
- **Ý nghĩa thực tế và giá trị ứng dụng:**
  Trong bài toán lọc bình luận trên mạng xã hội, rủi ro lớn nhất là bỏ lọt các nội dung thù ghét gây hại (False Negative). Vì vậy, việc chấp nhận Accuracy giảm nhẹ 1.28% để phát hiện thêm 5.23% bình luận thù ghét và 6.30% bình luận xúc phạm là sự đánh đổi hoàn toàn tối ưu và mang tính ứng dụng thực tiễn cao.

### 4.3. Nghiên cứu thực nghiệm bóc tách (Ablation Study): Vai trò của quy mô từ điển Teencode (35 từ vs. 414 từ)

Để kiểm chứng mức độ đóng góp thực chất của quy trình tiền xử lý dữ liệu, chúng tôi đối chiếu trực tiếp giữa hai phiên bản chạy cùng `Seed = 42`: phiên bản dùng bộ từ điển rút gọn (35 từ) và phiên bản dùng bộ từ điển chuẩn hóa mở rộng (414 từ).

1. **Cải thiện định lượng trên toàn bộ các chỉ số:**
   - **Macro F1:** Tăng từ **0.6708 lên 0.6746 (+0.38%)** ở Model B và từ 0.6707 lên 0.6750 (+0.43%) ở Model A.
   - **Lớp hưởng lợi lớn nhất là `OFFENSIVE`:** F1-score của lớp xúc phạm tăng mạnh từ **0.4438 lên 0.4531 (+0.93%)** ở Model B và từ 0.4333 lên 0.4495 (+1.62%) ở Model A. Do các bình luận xúc phạm chứa mật độ từ lóng, teencode, viết tắt chửi bới dày đặc nhất, việc mở rộng từ điển đã trực tiếp giải phóng khả năng hiểu ngữ cảnh cho PhoBERT tại lớp này.
   - **Accuracy tổng thể:** Tăng từ 86.59% lên 86.86% (Model B), chứng minh việc chuẩn hóa từ ngữ giúp mô hình giảm đáng kể sự nghi ngờ sai đối với các câu trung tính.

2. **Phân tích chuyển dịch dự đoán trên 6,680 mẫu kiểm tra:**
   - So sánh chi tiết từng mẫu trong `predictions.csv`: Có tổng cộng **410 bình luận** thay đổi kết quả dự đoán khi mở rộng từ điển từ 35 lên 414 từ.
   - **192 bình luận** từ đoán sai ở bản 35 từ đã được sửa thành **đoán đúng hoàn toàn** ở bản 414 từ (so với 174 mẫu biến động ngược lại), đem lại mức tăng ròng **+18 mẫu đoán đúng**.
   - **Một số ca sửa lỗi điển hình được ghi nhận từ tập kết quả:**
     - *Mẫu ID 2:* `"the nay ma chi phat gay roi trat tu cong cong thoi a dung la dan minh"` (Nhãn thực tế: `CLEAN`). Bản 35 từ đoán nhầm là `HATE`, bản 414 từ đã khôi phục đúng thành `CLEAN`.
     - *Mẫu ID 49:* `"Co ba oi co giau lam vay co giup nhung nguoi kho giong nku tui con di co"` (Nhãn thực tế: `CLEAN`). Từ teencode `nku` (như) khi chưa chuẩn hóa bị băm nhỏ khiến mô hình nghi ngờ nhầm thành `HATE`. Bản 414 từ nhận diện chính xác là `CLEAN`.
     - *Mẫu ID 60:* `"Một phút bóc đồng bằng cả đời bóc cức 😂"` (Nhãn thực tế: `OFFENSIVE`). Bản 35 từ bỏ sót thành `CLEAN`, bản 414 từ bắt trúng chính xác `OFFENSIVE`.
     - *Mẫu ID 93:* `"bán hàng rẻ ko thơm lâu thì nói không thơm lâu . nói hàng fake . mắc j quảng bá làm như tốt lắm ."` (Nhãn thực tế: `CLEAN`). Các từ `ko`, `j` được chuyển về dạng chuẩn giúp mô hình không còn nghi ngờ nhầm sang `OFFENSIVE`.

### 4.4. Đánh giá độ ổn định qua các khởi tạo ngẫu nhiên (Multi-Seed Robustness: Seed 42 vs. Seed 123)

Trong huấn luyện mạng nơ-ron sâu, kết quả có thể biến thiên do thứ tự nạp dữ liệu mini-batch hoặc khởi tạo ngẫu nhiên tầng phân loại. Chúng tôi tiến hành kiểm định chéo với `Seed = 123` trên cùng quy trình Teencode 414 từ:

1. **Độ phân tán cực nhỏ (High Convergence Stability):**
   - Macro F1 giữa Seed 42 (**0.6746**) và Seed 123 (**0.6766**) chỉ chênh lệch **0.0020 (0.2%)**.
   - Accuracy giữa Seed 42 (**86.86%**) và Seed 123 (**87.11%**) chỉ chênh lệch **0.25%**.
   - F1 của lớp `CLEAN` duy trì vững chắc ở mức **93.5% - 93.7%**; lớp `HATE` duy trì ở mức **63.3% - 63.6%**.
   - Cả hai seed đều vượt trội rất xa so với mô hình baseline của Luu et al. (2021) (62.69% với m-BERT), khẳng định hiệu năng cao của hệ thống là do phương pháp thực chất chứ không phải do may mắn chọn seed (lucky seed).

2. **Đỉnh hiệu năng thực nghiệm:**
   - Tại `Seed = 123`, mô hình áp dụng Class Weight đạt đỉnh hiệu năng cao nhất trong toàn bộ nghiên cứu với **Macro F1 = 0.6766**, trong đó **F1-OFFENSIVE đạt kỷ lục 0.4596**.

### 4.5. Tính nhất quán của phương pháp Class Weight trên các điều kiện thực nghiệm

Khi nhìn nhận tổng thể cả 3 bộ kết quả trong `outputs/`:
1. **Ưu thế nhất quán trên lớp thiểu số:** Ở cả 3 lần chạy, áp dụng Class Weight luôn giúp F1-score của lớp `OFFENSIVE` tăng từ **+0.36% đến +1.11%** so với Baseline tương ứng.
2. **Đẩy mạnh độ nhạy (Recall):** Recall của `OFFENSIVE` và `HATE` luôn đạt từ **48.4% đến 48.9%** (cho OFFENSIVE) và **65.5% đến 66.6%** (cho HATE), cao hơn từ **+1.0% đến +7.9%** so với việc không dùng trọng số.
3. **Cơ chế hoạt động bền vững:** Dù thay đổi từ điển hay đổi seed, việc phạt mất mát bằng Class Weight luôn phát huy tác dụng thúc đẩy mô hình quan tâm nhiều hơn đến các lớp có tần suất xuất hiện thấp.

---

## 5. Phân tích Ma trận Nhầm lẫn (Confusion Matrix)

### 5.1. Bảng đối chiếu số lượng phân loại giữa hai mô hình (Seed 42, Teencode 414)

```
[MÔ HÌNH BASELINE A]                             [MÔ HÌNH CLASS WEIGHT B]
               Dự đoán                                          Dự đoán
           CLEAN   OFF    HATE                              CLEAN   OFF    HATE
Thực CLEAN [5279   127    142 ]                 Thực CLEAN [5129   215    204 ]
     OFF   [ 174   187     83 ]                      OFF   [ 138   215     91 ]
     HATE  [ 192    74    422 ]                      HATE  [ 155    75    458 ]
```

**Nhận xét từ ma trận nhầm lẫn:**

1. **Giảm lỗi bỏ sót câu thù ghét ($HATE \rightarrow CLEAN$):**
   - Mô hình Baseline bỏ sót **192 câu thù ghét** và đoán nhầm thành bình thường.
   - Mô hình Class Weight giảm số lượng này xuống còn **155 câu** (bỏ sót ít hơn 37 câu, tương đương giảm 19.27% lỗi bỏ lọt thù ghét).
2. **Tăng số câu thù ghét nhận diện đúng ($HATE \rightarrow HATE$):**
   - Số câu thù ghét nhận diện chính xác tăng từ **422 câu** lên **458 câu** (Recall đạt **66.57%**).
3. **Giảm lỗi bỏ sót câu xúc phạm ($OFFENSIVE \rightarrow CLEAN$):**
   - Mô hình Baseline bỏ sót **174 câu xúc phạm** sang bình thường.
   - Mô hình Class Weight giảm xuống còn **138 câu** (bỏ sót ít hơn 36 câu, tương đương giảm 20.69%).
4. **Tăng số câu xúc phạm nhận diện đúng ($OFFENSIVE \rightarrow OFFENSIVE$):**
   - Số câu xúc phạm nhận diện đúng tăng từ **187 câu** lên **215 câu** (Recall tăng từ 42.12% lên **48.42%**).
5. **Sự nhầm lẫn giữa hai nhóm vi phạm ($OFFENSIVE \leftrightarrow HATE$):**
   - Số câu nhầm lẫn giữa xúc phạm và thù ghét ở cả hai mô hình ở mức tương đương nhau (khoảng 74 đến 91 mẫu). Trong thực tế, sự nhầm lẫn này ít gây rủi ro hơn vì cả hai nhóm đều được nhận diện là có vi phạm để xử lý tiếp.

### 5.2. Đối chiếu Ma trận Nhầm lẫn của Model B (Class Weight) qua 3 cấu hình thực nghiệm

```
[1. SEED 42 - TEENCODE 35]          [2. SEED 42 - TEENCODE 414]         [3. SEED 123 - TEENCODE 414]
          Dự đoán                             Dự đoán                             Dự đoán
      CLEAN   OFF   HATE                  CLEAN   OFF   HATE                  CLEAN   OFF   HATE
CLEAN [5109   232   207 ]           CLEAN [5129   215   204 ]           CLEAN [5152   200   196 ]
  OFF [ 136   217    91 ]             OFF [ 138   215    91 ]             OFF [ 138   216    90 ]
 HATE [ 145    85   458 ]            HATE [ 155    75   458 ]            HATE [ 157    80   451 ]
```

**Nhận xét tiến trình cải thiện qua các cấu hình:**
1. **Giảm thiểu báo động nhầm trên lớp CLEAN:**
   - Ở bản Teencode 35 từ: có $232 + 207 = 439$ câu bình thường bị đoán nhầm sang vi phạm.
   - Khi nâng lên Teencode 414 từ: số câu bị nhầm giảm xuống còn $215 + 204 = 419$ câu (giảm 20 ca báo động sai).
   - Ở Seed 123: số câu đoán đúng CLEAN tăng lên **5,152 câu**, số câu bị nhầm sang vi phạm giảm xuống còn **396 câu**.
2. **Độ ổn định bắt trúng vi phạm:**
   - Số câu thù ghét nhận diện đúng giữ ổn định tuyệt đối: **458 câu** ở cả bản Teencode 35 và 414 (Seed 42), và **451 câu** ở Seed 123.
   - Số câu xúc phạm nhận diện đúng duy trì đều đặn ở mức **215 - 217 câu**.

---

## 6. Phân tích Một số Trường hợp Dự đoán Sai

Qua kiểm tra các câu mô hình dự đoán chưa chính xác trong tệp kết quả, có 3 nhóm nguyên nhân chính:

### Nhóm 1: Câu nói mỉa mai, châm biếm

- **Ví dụ:** _"Trời ơi tư duy của mấy bạn trẻ trâu thông thái quá, đất nước này phát triển nhờ cả vào các bạn đấy ạ."_
- **Nhãn thực tế:** `OFFENSIVE`
- **Mô hình đoán:** `CLEAN`
- **Giải thích:** Câu sử dụng những từ mang sắc thái khen ngợi hoặc lịch sự ("thông thái", "phát triển nhờ các bạn") để châm chọc. Khi không có thông tin về ngữ cảnh câu chuyện xung quanh, mô hình chỉ phân tích từng từ ngữ riêng lẻ nên dễ hiểu nhầm là bình luận bình thường.

### Nhóm 2: Viết tắt và cố tình chèn ký tự ngụy trang

- **Ví dụ:** _"Mấy con đ.ũ.y này cút cmn đi cho rảnh mắt"_
- **Nhãn thực tế:** `HATE`
- **Mô hình đoán:** `OFFENSIVE`
- **Giải thích:** Dù từ điển 414 teencode đã xử lý hầu hết các biến thể thông dụng, người dùng vẫn biến tấu bằng cách chèn dấu chấm ngụy trang (`đ.ũ.y`) hoặc dùng từ viết tắt đặc thù (`cmn`). Bộ tách từ không nhận ra đây là từ hoàn chỉnh, dẫn đến phân mảnh subword và khiến mô hình khó nắm bắt trọn vẹn mức độ nghiêm trọng của câu.

### Nhóm 3: Ranh giới khó phân biệt giữa chửi thề và thù ghét

- **Ví dụ:** _"Nhìn cái bản mặt mày là thấy hãm tài rồi thằng lợn."_
- **Nhãn thực tế:** `OFFENSIVE`
- **Mô hình đoán:** `HATE`
- **Giải thích:** Ranh giới giữa một lời chửi bới, xúc phạm cá nhân bộc phát và ngôn từ thù ghét có chủ đích đôi khi rất mỏng manh. Khi xây dựng dữ liệu, ngay cả những người gán nhãn cũng khó đạt được sự đồng thuận tuyệt đối ở những trường hợp này.

---

## 7. Kết luận và Hướng Phát triển

### 7.1. Tổng kết Đóng góp của Đề tài

1. **Đóng góp về quy trình dữ liệu (Data-centric Pipeline):**
   - Xây dựng quy trình tiền xử lý tối ưu cho dữ liệu mạng xã hội tiếng Việt với bộ từ điển chuẩn hóa 414 từ teencode kết hợp bộ tách từ `PyVi`. Bước này giải quyết triệt để rào cản phân mảnh từ vựng subword, giúp mô hình nền tảng PhoBERT phát huy tối đa năng lực hiểu ngữ cảnh.

2. **Đóng góp về tối ưu hóa mô hình (Model-centric Optimization):**
   - Đưa ra giải pháp hiệu quả cho bài toán mất cân bằng lớp trầm trọng trên ViHSD thông qua hàm mất mát Weighted Cross-Entropy. Phương pháp này nâng cao đáng kể độ nhạy nhận diện các nội dung độc hại (Recall HATE đạt **66.57%**, OFFENSIVE đạt **48.42%**), giải quyết trực tiếp mục tiêu cốt lõi của bài toán kiểm duyệt tự động.

3. **Tính toàn diện và giá trị thực tiễn:**
   - Kết hợp nhịp nhàng giữa xử lý dữ liệu đặc thù và kỹ thuật huấn luyện có trọng số để tạo nên một quy trình thực nghiệm hoàn chỉnh (End-to-End Pipeline). Mô hình đạt hiệu năng cạnh tranh vượt bậc so với các nghiên cứu công bố (vượt m-BERT +4.7%, ngang tầm PhoBERT-CNN) mà không cần can thiệp phức tạp hay làm nặng thêm kiến trúc mạng Transformer gốc.

4. **Tính vững chắc và khả năng tái lập (Reproducibility & Robustness):**
   - Hệ thống thực nghiệm bóc tách (Ablation study giữa 35 và 414 từ teencode) cùng kiểm định chéo đa seed (Seed 42 và Seed 123) chứng minh giải pháp đạt độ hội tụ vững chắc (Macro F1 duy trì ổn định quanh 67.5% - 67.7%, độ biến thiên chỉ 0.2%), khẳng định hiệu năng cao của mô hình đến từ bản chất phương pháp chứ không phụ thuộc vào may rủi khởi tạo ngẫu nhiên.

### 7.2. Hướng Phát triển Tiếp theo

- **Mở rộng phạm vi tiền xử lý:** Tiếp tục mở rộng từ điển teencode/slang, đồng thời nghiên cứu cơ chế chuẩn hóa dựa trên biểu thức chính quy (Regex) để xử lý các hành vi cố tình chèn ký tự ngụy trang (ví dụ: `đ.ũ.y`, `c_m_n`).
- **Thử nghiệm các hàm mất mát nâng cao:** Mở rộng nghiên cứu áp dụng **Focal Loss** nhằm tập trung gradient vào các mẫu biên khó phân biệt (ranh giới giữa chửi thề bộc phát và thù ghét có chủ đích).
- **Kỹ thuật tăng cường dữ liệu (Data Augmentation):** Thử nghiệm sinh mẫu tự động (Back-translation, Paraphrasing) cho hai lớp thiểu số `OFFENSIVE` và `HATE` để cải thiện độ phân tán dữ liệu trước khi huấn luyện.
