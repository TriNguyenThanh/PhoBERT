# 01. TỔNG QUAN ĐỀ TÀI VÀ BỘ DỮ LIỆU ViHSD

## 1. Giới thiệu Đề tài

### 1.1. Bối cảnh
Sự bùng nổ của các nền tảng mạng xã hội (Facebook, TikTok, YouTube, Threads,...) tại Việt Nam đã tạo ra môi trường giao tiếp mở, đa chiều và có tốc độ lan truyền nhanh chóng. Tuy nhiên, đi kèm với đó là sự gia tăng của các nội dung tiêu cực, đặc biệt là **ngôn từ thù ghét (Hate Speech)** và **ngôn từ xúc phạm, khiếm nhã (Offensive Speech)**.

Các nội dung này gây ảnh hưởng tâm lý tiêu cực cho người dùng, kích động thù hằn và làm suy giảm chất lượng môi trường số. Do khối lượng bình luận phát sinh hàng ngày rất lớn, việc kiểm duyệt thủ công bằng con người không thể đáp ứng về mặt chi phí và thời gian. Do đó, việc xây dựng hệ thống **Tự động phát hiện ngôn từ thù ghét trên mạng xã hội tiếng Việt (Vietnamese Hate Speech Detection - HSD)** ứng dụng công nghệ Xử lý Ngôn ngữ Tự nhiên (NLP) và Học sâu (Deep Learning) là một nhu cầu cấp thiết.

> **Lưu ý phân biệt thuật ngữ khoa học:** Đề tài nghiên cứu bài toán **Phát hiện ngôn từ thù ghét (Hate Speech Detection)** nhằm xác định mức độ vi phạm chuẩn mực nội dung, hoàn toàn phân biệt với bài toán **Phân loại cảm xúc (Sentiment Analysis)** vốn chỉ nhận diện thái độ Tích cực/Tiêu cực/Trung tính.

### 1.2. Định nghĩa Bài toán
Bài toán được phát biểu dưới dạng **Phân loại văn bản đa lớp (Multi-class Text Classification)**:
- **Đầu vào ($X$):** Một đoạn văn bản (bình luận, bài đăng) bằng tiếng Việt trên mạng xã hội.
- **Đầu ra ($Y$):** Một nhãn duy nhất thuộc tập 3 lớp:
  $$C = \{\text{CLEAN (0 - Sạch/Lành mạnh)}, \text{OFFENSIVE (1 - Xúc phạm/Thô tục bộc phát)}, \text{HATE (2 - Thù ghét/Kỳ thị có chủ đích)}\}$$

### 1.3. Mục tiêu và Trọng tâm Nghiên cứu
Đề tài tập trung vào việc giải quyết triệt để vấn đề **mất cân bằng nhãn nghiêm trọng** trong ngữ liệu mạng xã hội tiếng Việt thông qua quy trình thực nghiệm A/B đối chứng khoa học:

1. **Sản phẩm cốt lõi (Mô hình B - Proposed Model):** Xây dựng mô hình phân loại dựa trên nền tảng **PhoBERT-base-v2** kết hợp quy trình tiền xử lý chuẩn hóa 414 từ teencode, tách từ ngữ pháp `PyVi` và kỹ thuật **Balanced Class Weighting** (gán trọng số nghịch đảo tần suất lớp vào hàm mất mát Cross-Entropy).
2. **Mô hình đối chứng (Mô hình A - Baseline Model):** Huấn luyện PhoBERT-base-v2 với hàm mất mát Cross-Entropy tiêu chuẩn (không trọng số) để làm cơ sở so sánh, chứng minh tính hiệu quả và sự vượt trội của Mô hình B.
3. **Đánh giá đa chiều:** Đo lường toàn diện các chỉ số: **Macro F1-score (MF1)**, **Weighted F1-score (WF1)**, **Per-class Precision/Recall/F1** và ma trận nhầm lẫn (Confusion Matrix).
4. **Kiểm định tính ổn định thống kê:** Thực hiện đối chứng trên **4 hạt giống ngẫu nhiên độc lập (Seeds: 42, 123, 456, 789)** để báo cáo kết quả trung bình và độ lệch chuẩn ($\mu \pm \sigma$).

---

## 2. Nghiên cứu Tham chiếu và Bộ Dữ liệu ViHSD

### 2.1. Nghiên cứu Gốc
- **Tên bài báo:** *A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts*
- **Tác giả:** Son T. Luu, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen (Trường ĐH Công nghệ Thông tin - ĐHQG TP.HCM, UIT-VNUHCM).
- **Hội nghị công bố:** The 34th International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems (IEA/AIE 2021).
- **Mã nguồn & Dữ liệu:** [GitHub sonlam1102/vihsd](https://github.com/sonlam1102/vihsd) | [Hugging Face ViHSD](https://huggingface.co/datasets/visolex/ViHSD)

### 2.2. Định nghĩa 3 Lớp Nhãn Chuẩn trong ViHSD
Theo chuẩn gán nhãn của Luu et al. (2021), tập dữ liệu phân biệt rõ ràng giữa 3 mức độ nội dung:

| Mã nhãn | Tên nhãn | Định nghĩa khoa học | Ví dụ minh họa |
|:---:|:---:|:---|:---|
| **0** | **CLEAN**<br>*(Sạch / Lành mạnh)* | Bình luận sạch, mang tính xây dựng, trung tính hoặc thể hiện quan điểm thông thường mà không mang tính công kích, xúc phạm. | *"Phim này xem giải trí rất ổn, diễn viên đóng đạt."* |
| **1** | **OFFENSIVE**<br>*(Xúc phạm / Thô tục)* | Chứa từ ngữ thô tục, chửi thề bộc phát (profanity) nhưng **không nhắm mục tiêu công kích cá nhân hay nhóm đối tượng được bảo vệ**. | *"Xem trận bóng ức chế vãi, đá đấm chán thật sự."* |
| **2** | **HATE**<br>*(Ngôn từ thù ghét)* | Ngôn từ thù ghét: công kích nhân phẩm, đe dọa, kích động phân biệt đối xử hoặc kỳ thị dựa trên các đặc điểm định danh: dân tộc, vùng miền, tôn giáo, giới tính,... | *"Bọn [vùng miền/dân tộc] này chỉ biết ăn bám, biến khỏi đây đi."* |

### 2.3. Phân chia Tập Dữ liệu và Thống kê Mất cân bằng Lớp
Bộ dữ liệu chuẩn ViHSD gồm **33,400 mẫu** bình luận tiếng Việt trên Facebook, được chia theo tỷ lệ benchmark cố định 72 : 8 : 20:

| Tập dữ liệu (Split) | Số lượng mẫu | Tỷ lệ (%) | CLEAN (0) | OFFENSIVE (1) | HATE (2) | Tỷ lệ CLEAN : OFF : HATE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train (Gốc)** | 24,048 | 72% | 19,886 (82.69%) | 1,606 (6.68%) | 2,556 (10.63%) | 12.4 : 1.0 : 1.6 |
| **Validation (Dev)** | 2,672 | 8% | 2,190 (81.96%) | 212 (7.93%) | 270 (10.10%) | 10.3 : 1.0 : 1.3 |
| **Test (Đánh giá)** | 6,680 | 20% | 5,548 (83.05%) | 444 (6.65%) | 688 (10.30%) | 12.5 : 1.0 : 1.5 |
| **Tổng cộng ViHSD** | **33,400** | **100%** | **27,624 (82.71%)** | **2,262 (6.77%)** | **3,514 (10.52%)** | **12.2 : 1.0 : 1.6** |

### 2.4. Tập Huấn luyện Sạch (Cleaned Training Set)
Trong quy trình tiền xử lý dữ liệu chuẩn hóa của đề tài, tập Train gốc (24,048 mẫu) được rà soát và loại bỏ **1,130 mẫu** trùng lặp nội bộ và rò rỉ dữ liệu (data leakage giữa tập Train với Dev/Test). Tập Train sạch còn lại **21,098 mẫu**:

| Lớp nhãn | Số mẫu sau làm sạch ($N_c$) | Tỷ lệ phần trăm (%) | Trọng số cân bằng ($w_c$) |
|:---|:---:|:---:|:---:|
| **0: CLEAN** | 17,258 | 81.80% | **0.4075** (~0.41) |
| **1: OFFENSIVE** | 1,448 | 6.86% | **4.8568** (~4.86) |
| **2: HATE** | 2,392 | 11.34% | **2.9401** (~2.94) |
| **Tổng cộng** | **21,098** | **100.00%** | - |

### 2.5. Phân tích Hiện tượng Mất cân bằng Lớp (Class Imbalance)
- **Mức độ chênh lệch cực đoan:** Lớp `CLEAN` chiếm tới hơn **82.7%** toàn bộ ngữ liệu, trong khi `OFFENSIVE` chỉ chiếm chưa đầy **6.8%** và `HATE` chiếm khoảng **10.5%**.
- **Hệ quả đối với mô hình thông thường (Mô hình A):**
  1. Nếu mô hình có xu hướng dự đoán toàn bộ là `CLEAN`, độ chính xác tổng thể (Accuracy) vẫn đạt tới **~83%**, nhưng mô hình hoàn toàn mất khả năng nhận diện các nội dung độc hại.
  2. Gradient của hàm mất mát Cross-Entropy tiêu chuẩn bị áp đảo hoàn toàn bởi lớp đa số `CLEAN`, khiến mô hình ít cập nhật trọng số cho các đặc trưng của `OFFENSIVE` và `HATE`, dẫn đến chỉ số **Recall** của hai lớp này bị tụt giảm nghiêm trọng.

---

## 3. Thách thức Đặc thù của Tiếng Việt trên Mạng Xã hội

Khi xử lý bài toán phát hiện ngôn từ thù ghét trên mạng xã hội tiếng Việt, mô hình phải xử lý các đặc thù ngôn ngữ sau:

1. **Ranh giới từ phức (Word Segmentation):**
   - Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ vựng không trùng khớp với dấu cách (ví dụ: *học sinh*, *công nghệ thông tin*).
   - Nếu không tách từ đúng ngữ pháp bằng công cụ chuyên dụng (`PyVi`), bộ mã hóa subword (BPE) của PhoBERT sẽ chia nhỏ các âm tiết độc lập, làm sai lệch cấu trúc ngữ nghĩa câu.
2. **Teencode, tiếng lóng và biến thể ngụy trang (Obfuscation):**
   - Người dùng thường sử dụng từ viết tắt (*ko, dc, vcl, đm*), thay thế ký tự tương đồng hoặc cố tình chèn dấu (*đ.ụ, đ-ĩ*) nhằm vượt qua các bộ lọc từ khóa đơn giản.
   - Đề tài sử dụng bộ từ điển chuẩn hóa **414 từ teencode** để ánh xạ các biến thể này về từ vựng tiếng Việt chuẩn trước khi đưa vào mô hình.
3. **Bảo toàn ngữ cảnh (Không loại bỏ từ dừng - No Stopwords):**
   - Các nghiên cứu ngữ liệu tiếng Việt (Luu et al., 2024) đã chứng minh việc loại bỏ stopwords làm suy giảm khả năng chú ý (Self-Attention) của mô hình Transformer. Vì vậy, quy trình tiền xử lý giữ nguyên toàn bộ cấu trúc câu tự nhiên.
4. **Hiện tượng mỉa mai, châm biếm (Sarcasm):**
   - Bình luận sử dụng từ ngữ trung tính hoặc khen ngợi nhưng mang hàm ý công kích sâu sắc, đòi hỏi mô hình phải có khả năng học biểu diễn ngữ cảnh tầng sâu.

---

## 4. Cấu trúc Sản phẩm Đầu ra của Đề tài

Hệ thống bàn giao các sản phẩm hoàn chỉnh theo chuẩn nghiên cứu:

1. **Mô hình cốt lõi (Mô hình B - Proposed Model):**
   - Checkpoint mô hình PhoBERT-base-v2 kết hợp tiền xử lý 414 teencode, tách từ `PyVi` và Balanced Class Weighting.
2. **Tệp dự đoán trên tập kiểm tra (`predictions.csv`):**
   - Kết quả suy luận của Mô hình B trên toàn bộ 6,680 mẫu của tập Test ViHSD.
   - Định dạng chuẩn 4 cột: `id`, `text`, `ground_truth`, `predicted_label`.
3. **Báo cáo định lượng so sánh (`aggregated_metrics.csv`):**
   - Bảng đối chiếu chi tiết giữa Mô hình A (Baseline) và Mô hình B (Proposed) trên các độ đo Macro F1, Weighted F1, Per-class F1 và Recall.
4. **Biểu đồ trực quan ma trận nhầm lẫn (`confusion_matrix.png`):**
   - Hình ảnh so sánh ma trận nhầm lẫn $3 \times 3$ giữa Mô hình A và Mô hình B.
5. **Mã nguồn thực nghiệm:**
   - Notebook hoàn chỉnh: [PhoBERT_ViHSD_ClassWeight.ipynb](PhoBERT_ViHSD_ClassWeight.ipynb), sẵn sàng thực thi trên Google Colab GPU T4 với cơ chế lưu trữ tự động lên Google Drive.
6. **Hệ thống tài liệu kỹ thuật đồng bộ:**
   - [01_tong_quan_de_tai.md](docs/01_tong_quan_de_tai.md): Tổng quan đề tài & tập dữ liệu ViHSD.
   - [02_kien_truc_va_phuong_phap.md](docs/02_kien_truc_va_phuong_phap.md): Kiến trúc PhoBERT & Phương pháp Class Weight.
   - [03_thiet_ke_thuc_nghiem_va_danh_gia.md](docs/03_thiet_ke_thuc_nghiem_va_danh_gia.md): Thiết kế thực nghiệm đối chứng & Hệ thống độ đo.
   - [04_huong_dan_trien_khai_colab.md](docs/04_huong_dan_trien_khai_colab.md): Hướng dẫn triển khai thực nghiệm trên Google Colab.
   - [05_mau_bao_cao_so_sanh.md](docs/05_mau_bao_cao_so_sanh.md): Báo cáo phân tích đối chiếu kết quả thực nghiệm.
   - [06_noi_dung_bao_cao.md](docs/06_noi_dung_bao_cao.md): Bảng số liệu thuyết trình & luận điểm bảo vệ đề tài.
