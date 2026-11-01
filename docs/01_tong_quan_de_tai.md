# 01. TỔNG QUAN ĐỀ TÀI VÀ BỘ DỮ LIỆU ViHSD

## 1. Giới thiệu đề tài

### 1.1. Bối cảnh
Sự bùng nổ của các nền tảng mạng xã hội (Facebook, TikTok, YouTube, Threads,...) tại Việt Nam đã tạo ra một môi trường giao tiếp mở, đa chiều và tốc độ lan truyền chóng mặt. Tuy nhiên, đi kèm với đó là sự gia tăng nghiêm trọng của các nội dung tiêu cực, đặc biệt là **ngôn từ thù ghét (Hate Speech)** và **ngôn từ xúc phạm, khiếm nhã (Offensive Speech)**.

Các nội dung này gây tổn thương tinh thần cho cá nhân, kích động xung đột sắc tộc, tôn giáo, giới tính, và làm ô nhiễm không gian mạng. Do khối lượng dữ liệu phát sinh hàng ngày là hàng triệu lượt bình luận và bài viết, việc kiểm duyệt thủ công bằng con người là bất khả thi về mặt chi phí và thời gian. Vì vậy, việc xây dựng các hệ thống **Tự động nhận diện và phân loại ngôn từ thù ghét (Automated Hate Speech Detection - HSD)** ứng dụng công nghệ Xử lý Ngôn ngữ Tự nhiên (NLP) và Học sâu (Deep Learning) trở thành một yêu cầu cấp thiết.

### 1.2. Định nghĩa bài toán
Bài toán được phát biểu dưới dạng **Phân loại văn bản đa lớp (Multi-class Text Classification)**:
- **Đầu vào ($X$):** Một đoạn văn bản (bình luận, trạng thái, câu viết) bằng tiếng Việt trên mạng xã hội.
- **Đầu ra ($Y$):** Một nhãn duy nhất thuộc tập $C = \{\text{CLEAN (0)}, \text{OFFENSIVE (1)}, \text{HATE (2)}\}$.

### 1.3. Mục tiêu nghiên cứu
1. Huấn luyện mô hình ngôn ngữ tiền huấn luyện dành riêng cho tiếng Việt (**PhoBERT-base**) trên bộ dữ liệu chuẩn hóa **ViHSD**.
2. Giải quyết bài toán **mất cân bằng lớp nghiêm trọng (Severe Class Imbalance)** bằng kỹ thuật **Class Weight** (hoặc Focal Loss).
3. Đánh giá đa chiều hiệu năng mô hình thông qua các độ đo: **Macro F1-score (MF1)**, **Weighted F1-score (WF1)**, **Per-class F1-score** và **Confusion Matrix**.
4. Phân tích đối chiếu thực nghiệm giữa hai nhánh: **Baseline (Cross Entropy tiêu chuẩn)** và **Class Weight (Weighted Cross Entropy)** để làm rõ sự đánh đổi giữa Precision và Recall trên các lớp thiểu số (HATE, OFFENSIVE).

---

## 2. Nghiên cứu tham chiếu và Bộ dữ liệu ViHSD

### 2.1. Nghiên cứu gốc
- **Tên bài báo:** *A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts*
- **Tác giả:** Son T. Luu, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen (Trường ĐH Công nghệ Thông tin - ĐHQG TP.HCM, UIT-VNUHCM).
- **Công bố tại:** Hội nghị quốc tế EMNLP 2021 (The 2021 Conference on Empirical Methods in Natural Language Processing).
- **Mã nguồn & Dữ liệu:** [GitHub repository](https://github.com/sonlam1102/vihsd) | [Hugging Face](https://huggingface.co/datasets/visolex/ViHSD)

### 2.2. Định nghĩa các lớp nhãn trong ViHSD
Theo chuẩn gán nhãn của Luu et al. (2021), tập dữ liệu phân biệt rõ ràng giữa 3 mức độ nội dung:

| Mã nhãn | Tên nhãn | Định nghĩa khoa học | Ví dụ minh họa |
|:---:|:---:|:---|:---|
| **0** | **CLEAN** | Bình luận sạch, mang tính xây dựng, trung tính hoặc thể hiện cảm xúc tích cực/tiêu cực thông thường mà không mang tính xúc phạm, công kích. | *"Phim này xem giải trí ổn ghê, diễn viên đóng tròn vai."* |
| **1** | **OFFENSIVE** | Chứa ngôn từ thô tục, chửi thề, tục tĩu (profanity), thô thiển nhưng **không nhắm mục tiêu thù ghét** cụ thể vào một nhóm đối tượng được bảo vệ (dân tộc, tôn giáo, giới tính, xuất thân,...). | *"Xem trận bóng ức chế vãi chưởng, đá đấm như hạch."* |
| **2** | **HATE** | Ngôn từ thù ghét: công kích, hạ thấp nhân phẩm, đe dọa bạo lực, kích động phân biệt đối xử hoặc kỳ thị dựa trên các đặc điểm định danh: chủng tộc, vùng miền, tôn giáo, xu hướng tính dục, khuyết tật,... | *"Bọn mọi rợ [vùng miền/dân tộc] này chỉ biết ăn bám, biến khỏi đây đi."* |

### 2.3. Phân chia tập dữ liệu và Thống kê mất cân bằng
Dữ liệu chuẩn ViHSD gồm **33,400 mẫu** được thu thập từ hơn 100 trang fanpage và nhóm thảo luận công khai trên mạng xã hội Facebook tại Việt Nam.

Phân bố chi tiết theo từng tập (Split) chuẩn của bài báo:

| Tập dữ liệu (Split) | Số lượng mẫu | CLEAN (0) | OFFENSIVE (1) | HATE (2) | Tỷ lệ CLEAN : OFFENSIVE : HATE |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Train** | 24,048 | 19,886 (82.69%) | 1,606 (6.68%) | 2,556 (10.63%) | 12.4 : 1.0 : 1.6 |
| **Validation (Dev)** | 2,672 | 2,190 (81.96%) | 212 (7.93%) | 270 (10.10%) | 10.3 : 1.0 : 1.3 |
| **Test** | 6,680 | 5,548 (83.05%) | 444 (6.65%) | 688 (10.30%) | 12.5 : 1.0 : 1.5 |
| **Tổng cộng** | **33,400** | **27,624 (82.71%)** | **2,262 (6.77%)** | **3,514 (10.52%)** | **12.2 : 1.0 : 1.6** |

### 2.4. Phân tích hiện tượng mất cân bằng lớp (Class Imbalance)
- **Độ lệch cực đoan:** Lớp `CLEAN` chiếm tới hơn **82.7%** toàn bộ dữ liệu, trong khi lớp `OFFENSIVE` chỉ chiếm chưa đầy **6.8%** và `HATE` chiếm khoảng **10.5%**.
- **Hệ quả đối với mô hình chuẩn:**
  1. Nếu một mô hình chỉ dự đoán toàn bộ dữ liệu là `CLEAN`, mô hình đó đã đạt độ chính xác thô (Accuracy) lên tới **83%** nhưng hoàn toàn vô dụng vì bỏ sót 100% các câu thù ghét và xúc phạm.
  2. Hàm mất mát Cross Entropy tiêu chuẩn sẽ bị chi phối bởi các mẫu thuộc lớp đa số `CLEAN`. Gradient cập nhật sẽ kéo trọng số mô hình về phía ưu tiên dự đoán lớp `CLEAN` để giảm thiểu tổng mất mát, dẫn tới chỉ số **Recall** của lớp `OFFENSIVE` và `HATE` cực kỳ thấp.

---

## 3. Thách thức đặc thù của tiếng Việt trên mạng xã hội

Khi xử lý bài toán phát hiện ngôn từ thù ghét trên dữ liệu mạng xã hội tiếng Việt, các kỹ sư NLP phải đối mặt với nhiều rào cản ngôn ngữ phức tạp:

1. **Ranh giới từ phức (Word Segmentation):**
   - Khác với tiếng Anh (ngăn cách từ bằng dấu cách), tiếng Việt là ngôn ngữ đơn lập, ranh giới từ vựng không đồng nhất với khoảng trắng (ví dụ: *học sinh*, *công nghệ thông tin*).
   - Nếu không tách từ đúng ngữ pháp, mô hình sẽ xem *"học"* và *"sinh"* là hai thực thể độc lập, làm sai lệch ngữ nghĩa sâu sắc.
2. **Teencode và cố tình viết biến thể né kiểm duyệt (Obfuscation & Camouflage):**
   - Người dùng thường cố tình biến dạng từ ngữ độc hại để vượt qua bộ lọc từ khóa tĩnh: thay ký tự tương đồng (*đ* $\rightarrow$ *d*, *@* $\rightarrow$ *a*), xen dấu chấm/khoảng trắng (*đ.ụ*, *đ-i-ế-m*), viết tắt (*vcl*, *đm*, *dcm*).
3. **Hiện tượng mỉa mai, châm biếm (Sarcasm & Irony):**
   - Câu văn có thể sử dụng các từ ngữ hoàn toàn lịch sự, hoa mỹ trên bề mặt ngữ nghĩa nhưng hàm ý miệt thị sâu sắc đối tượng bên trong (ví dụ: *"Các bạn nữ quyền thông thái thế này thì xã hội sớm muộn cũng được nhờ"*).
4. **Ngữ cảnh ngắn và phụ thuộc văn hóa địa phương:**
   - Các bình luận trên mạng thường rất ngắn, thiếu chủ ngữ, mang đậm phong cách nói (Spoken style), sử dụng tiếng lóng theo xu hướng (slang) và từ ngữ phân biệt vùng miền đặc thù.

---

## 4. Cấu trúc sản phẩm đầu ra của đề tài

Theo yêu cầu chuẩn của đề tài nghiên cứu, hệ thống hoàn chỉnh bàn giao các sản phẩm sau:

1. **File dự đoán `predictions.csv`:**
   - Tập hợp kết quả suy luận trên toàn bộ 6,680 mẫu của tập TEST.
   - Định dạng chuẩn 4 cột: `id`, `text`, `ground_truth`, `predicted_label`.
2. **Báo cáo so sánh thực nghiệm:**
   - Chi tiết tại [docs/05_mau_bao_cao_so_sanh.md](docs/05_mau_bao_cao_so_sanh.md).
   - Đối chiếu tường minh giữa mô hình **Baseline (không Class Weight)** và mô hình **Class Weight** dựa trên các tiêu chí khoa học (MF1, WF1, Per-class F1, Confusion Matrix).
3. **Mã nguồn hoàn chỉnh:**
   - Notebook Google Colab: [pipeline/PhoBERT_ViHSD_ClassWeight_Colab.ipynb](pipeline/PhoBERT_ViHSD_ClassWeight_Colab.ipynb).
   - Chạy độc lập trên môi trường Google Colab GPU T4 (FP16), hỗ trợ tự động lưu checkpoint và sản phẩm lên Google Drive.
4. **Bộ tài liệu kỹ thuật hoàn chỉnh:**
   - [docs/01_tong_quan_de_tai.md](docs/01_tong_quan_de_tai.md): Tổng quan đề tài & tập dữ liệu ViHSD.
   - [docs/02_kien_truc_va_phuong_phap.md](docs/02_kien_truc_va_phuong_phap.md): Kiến trúc PhoBERT & Phương pháp Class Weight.
   - [docs/03_thiet_ke_thuc_nghiem_va_danh_gia.md](docs/03_thiet_ke_thuc_nghiem_va_danh_gia.md): Thiết kế thực nghiệm & Chỉ số đánh giá.
   - [docs/04_huong_dan_trien_khai_colab.md](docs/04_huong_dan_trien_khai_colab.md): Hướng dẫn chi tiết triển khai Colab.
   - [docs/05_mau_bao_cao_so_sanh.md](docs/05_mau_bao_cao_so_sanh.md): Mẫu báo cáo so sánh kết quả thực nghiệm.
