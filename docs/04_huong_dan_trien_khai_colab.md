# 04. HƯỚNG DẪN TRIỂN KHAI VÀ HUẤN LUYỆN TRÊN GOOGLE COLAB

Tài liệu này cung cấp hướng dẫn kỹ thuật chi tiết từng bước (Step-by-step) dành cho các thành viên trong nhóm nghiên cứu để triển khai huấn luyện **Mô hình A (Baseline)** và **Mô hình B (Proposed - Sản phẩm cốt lõi)** trên môi trường Google Colab GPU T4.

Tài liệu bao gồm 2 phương thức thực thi:
1. **Phương pháp 1:** Huấn luyện trực tiếp trên giao diện Web của **Google Colab**.
2. **Phương pháp 2:** Huấn luyện qua **Extension Google Colab trong VSCode** (kết nối tài khoản Colab PRO 5TB).

Đồng thời, tài liệu hướng dẫn cụ thể quy trình chuẩn bị dữ liệu, phân công hạt giống ngẫu nhiên (Random Seed) độc lập giữa các thành viên, và chia sẻ thư mục sản phẩm đầu ra trên Google Drive.

---

## 1. Mục Tiêu Thực Nghiệm & Phân Công Nhiệm Vụ Nhóm

Để đảm bảo tính khách quan và khoa học, mỗi mô hình được huấn luyện độc lập qua **4 hạt giống ngẫu nhiên (Random Seeds: 42, 123, 456, 789)** nhằm đo lường giá trị trung bình và độ lệch chuẩn ($\mu \pm \sigma$):

```
                         Phân Công Huấn Luyện Độc Lập
                                       │
        ┌──────────────┬───────────────┴───────────────┬──────────────┐
        ▼              ▼                               ▼              ▼
  [Thành viên: Trí]  [Thành viên: Ngân]          [Thành viên: Hiếu]  [Thành viên: Tài]
     Seed: 42          Seed: 123                     Seed: 456         Seed: 789
  (Mốc Đối Chứng)   (Kiểm Định Đa Seed)          (Thực Nghiệm 3)    (Thực Nghiệm 4)
        │              │                               │              │
        └──────────────┴───────────────┬───────────────┴──────────────┘
                                       ▼
                     Tổng hợp kết quả báo cáo (Mean ± Std)
```

### 1.1. Bảng phân công chi tiết theo thành viên

| Thành viên phụ trách | Hạt giống (Random Seed) | Cấu hình biến trong Notebook | Thư mục kết quả (`outputs/`) | Trạng thái thực hiện |
|:---|:---:|:---|:---|:---:|
| **Trí** | **42** | `SEED = 42` | `outputs/seed42_teencode_414/` | Đã hoàn thành (Mốc chuẩn) |
| **Ngân** | **123** | `SEED = 123` | `outputs/seed123_teencode_414/` | Đã hoàn thành (Đa Seed) |
| **Hiếu** | **456** | `SEED = 456` | `outputs/seed456_teencode_414/` | Đã hoàn thành (Kiểm chứng độc lập) |
| **Tài** | **789** | `SEED = 789` | `outputs/seed789_teencode_414/` | Đã hoàn thành (Kiểm chứng độc lập) |

### 1.2. Thao tác cấu hình trước khi chạy
Tại **Cell số 2** của notebook [PhoBERT_ViHSD_ClassWeight.ipynb](PhoBERT_ViHSD_ClassWeight.ipynb), thành viên chỉ cần cập nhật giá trị biến `SEED` theo đúng phân công:
- **Ngân:** `SEED = 123`
- **Hiếu:** `SEED = 456`
- **Tài:** `SEED = 789`

*Lưu ý:* Toàn bộ các đường dẫn lưu trữ checkpoint và thư mục đầu ra trong notebook đã được tự động gắn mã seed (`f"seed{SEED}_teencode_414"`), tránh ghi đè kết quả của nhau.

---

## 2. Chuẩn Bị Dữ Liệu và Dự Án Trên Google Drive

Trước khi bắt đầu huấn luyện, các thành viên cần đồng bộ bộ dữ liệu ViHSD và từ điển teencode lên Google Drive cá nhân.

### 2.1. Cấu trúc thư mục chuẩn trên Google Drive
Hãy tạo cấu trúc thư mục trên Google Drive của bạn chính xác như sau:

```
Google Drive của tôi (MyDrive)
└── PhoBERT_ViHSD/
    ├── PhoBERT_ViHSD_ClassWeight.ipynb    # Notebook chính để chạy thực nghiệm
    └── data/                              # Thư mục dữ liệu bắt buộc
        ├── train.csv                      # Tập huấn luyện gốc (24,048 mẫu)
        ├── dev.csv                        # Tập kiểm định gốc (2,672 mẫu)
        ├── test.csv                       # Tập kiểm tra gốc (6,680 mẫu)
        └── teencode_414.json              # Từ điển chuẩn hóa 414 teencode
```

### 2.2. Các bước tải dữ liệu lên Google Drive
1. Truy cập [Google Drive](https://drive.google.com/).
2. Tại thư mục **My Drive (Drive của tôi)**, tạo thư mục mới đặt tên là: `PhoBERT_ViHSD`.
3. Mở thư mục `PhoBERT_ViHSD`, tạo tiếp thư mục con tên là `data`.
4. Tải 4 tệp dữ liệu từ thư mục `data/` của dự án lên:
   - `train.csv`
   - `dev.csv`
   - `test.csv`
   - `teencode_414.json`
5. Tải tệp notebook [PhoBERT_ViHSD_ClassWeight.ipynb](PhoBERT_ViHSD_ClassWeight.ipynb) vào thư mục `PhoBERT_ViHSD`.

---

## 3. PHƯƠNG PHÁP 1: Huấn Luyện Bằng Google Colab Web (Trình Duyệt)

Phương pháp này phù hợp khi thực thi trực tiếp trên trình duyệt Web (Chrome, Edge, Firefox) mà không cần cài đặt phần mềm phụ trợ.

### Bước 1: Mở Notebook trên Google Colab Web
1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Chọn tab **Google Drive** $\rightarrow$ Tìm và mở tệp `PhoBERT_ViHSD/PhoBERT_ViHSD_ClassWeight.ipynb`.
3. *(Hoặc tại giao diện Google Drive, nhấp chuột phải vào file `.ipynb` $\rightarrow$ Chọn **Mở bằng** $\rightarrow$ **Google Colaboratory**).*

### Bước 2: Thiết lập GPU T4 và Kiểm tra Phần cứng
1. Trên thanh menu, chọn: **Runtime (Thời gian chạy)** $\rightarrow$ **Change runtime type (Thay đổi loại thời gian chạy)**.
2. Tại mục **Hardware accelerator (Phần cứng tăng tốc)**:
   - Chọn **T4 GPU** (hoặc chọn **A100 / V100 GPU** nếu dùng Colab Pro).
3. Nhấn **Save (Lưu)**.
4. Chạy cell kiểm tra GPU:
   ```python
   !nvidia-smi
   ```
   *Xác nhận: Hệ thống hiển thị GPU Tesla T4 (khoảng 15GB VRAM).*

### Bước 3: Mount Google Drive
Tại cell kết nối Drive của notebook, chạy đoạn mã:

```python
from google.colab import drive
drive.mount('/content/drive')
```
- Chọn **Kết nối với Google Drive (Connect to Google Drive)**.
- Đăng nhập tài khoản Google và cấp quyền truy cập.
- Khi màn hình xuất hiện thông báo `Mounted at /content/drive` là đã thành công.

### Bước 4: Sao chép dữ liệu về máy ảo Colab
Để quá trình đọc dữ liệu và huấn luyện đạt tốc độ I/O cao nhất, sao chép thư mục dữ liệu về ổ đĩa cục bộ `/content/` của máy ảo:

```python
import os
import shutil

# Tạo thư mục data cục bộ trên máy ảo Colab
os.makedirs('./data', exist_ok=True)

# Sao chép dữ liệu từ Drive vào máy ảo
!cp -r /content/drive/MyDrive/PhoBERT_ViHSD/data/* ./data/

# Kiểm tra danh sách tệp sẵn sàng
print("Các tệp dữ liệu sẵn sàng trong máy ảo:")
print(os.listdir('./data'))
```

### Bước 5: Cập nhật Seed theo phân công
Tìm đến **Cell số 2** và cập nhật giá trị biến `SEED`:
- Ngân: `SEED = 123`
- Hiếu: `SEED = 456`
- Tài: `SEED = 789`

### Bước 6: Khởi chạy quá trình thực nghiệm
- Trên thanh menu chọn **Runtime** $\rightarrow$ **Run all (Chạy tất cả)** hoặc nhấn tổ hợp phím `Ctrl + F9`.
- Quá trình chạy diễn ra tự động qua các bước:
  1. Tiền xử lý dữ liệu (chuẩn hóa 414 teencode, lọc 1,130 mẫu trùng lặp và rò rỉ, tách từ PyVi).
  2. Huấn luyện **Mô hình A (Baseline - Standard Cross-Entropy)**: 4 epochs (~15-18 phút).
  3. Huấn luyện **Mô hình B (Proposed - Balanced Class Weighting)**: 4 epochs (~15-18 phút).
  4. Đánh giá suy luận trên tập Test cố định (6,680 mẫu), xuất `predictions.csv`, `aggregated_metrics.csv` và `confusion_matrix.png`.
  5. Đồng bộ toàn bộ sản phẩm tự động về Google Drive.
- Tổng thời gian thực thi: **khoảng 35 - 45 phút**.

---

## 4. PHƯƠNG PHÁP 2: Huấn Luyện Dùng Extension Colab Trong VSCode

Phương pháp này cho phép thành viên chạy và kiểm soát notebook trực tiếp trong trình soạn thảo **Visual Studio Code (VSCode)** trên máy cá nhân, nhưng vẫn tận dụng tài nguyên GPU từ Google Colab.

### Bước 1: Cài đặt Extension trên VSCode
1. Mở VSCode.
2. Nhấn `Ctrl + Shift + X` (Windows/Linux) hoặc `Cmd + Shift + X` (macOS) để mở tab **Extensions**.
3. Cài đặt hai extension: **Google Colab** và **Jupyter**.

### Bước 2: Mở file Notebook và Chọn Kernel Colab
1. Trong VSCode, mở tệp [PhoBERT_ViHSD_ClassWeight.ipynb](PhoBERT_ViHSD_ClassWeight.ipynb).
2. Ở góc trên bên phải giao diện Notebook, nhấn nút **Select Kernel (Chọn Kernel)**.
3. Chọn **Google Colab**.

### Bước 3: Xác thực tài khoản Colab PRO (Dung lượng 5TB)
1. VSCode sẽ mở trình duyệt yêu cầu đăng nhập tài khoản Google.
2. Đăng nhập bằng tài khoản **Google Colab PRO**.
3. Cấp quyền truy cập cho VSCode.
4. Sau khi hoàn tất, góc trên bên phải VSCode sẽ hiển thị:  
   `Google Colab: Connected (T4 GPU / Premium GPU)`.

### Bước 4: Kiểm tra kết nối và tải thư mục Data
1. Chạy cell đầu tiên (`!nvidia-smi`) để kích hoạt phiên làm việc trên máy chủ Colab.
2. Tại cây thư mục Explorer bên trái của VSCode, nhấp chuột phải vào thư mục `data` $\rightarrow$ Chọn **Upload to Colab**.
3. Xác nhận trên máy ảo Colab đã có đủ 4 tệp: `train.csv`, `dev.csv`, `test.csv`, `teencode_414.json`.

### Bước 5: Cập nhật Seed và Bấm Run All
1. Tại Cell số 2, sửa biến `SEED` theo phân công (123, 456, hoặc 789).
2. Nhấn nút **Run All** trên thanh công cụ notebook.
3. Khi notebook thực thi đến cell cuối, cấp quyền OAuth Google Drive để hệ thống tự động đồng bộ kết quả.

---

## 5. Danh Mục Sản Phẩm Đầu Ra và Chia Sẻ Kết Quả

Sau khi hoàn tất quá trình huấn luyện, toàn bộ kết quả thực nghiệm và trọng số mô hình đã được tự động lưu vào Google Drive tại đường dẫn:

```
MyDrive/PhoBERT_ViHSD/outputs/seed{SEED}_teencode_414/
MyDrive/PhoBERT_ViHSD/checkpoints/seed{SEED}_teencode_414/
```

### 5.1. Danh mục các sản phẩm bắt buộc trong thư mục output
Tại thư mục `outputs/seed{SEED}_teencode_414/`, hãy kiểm tra đầy đủ 3 tệp sau:

1. **`predictions.csv`**: Tệp dự đoán chính thức của Mô hình B trên tập Test ViHSD:
   - Đúng **6,680 dòng** (không tính header).
   - Đúng **4 cột**: `id`, `text`, `ground_truth`, `predicted_label`.
   - Không chứa bất kỳ giá trị `NaN` hay `Null`.
2. **`aggregated_metrics.csv`**: Bảng tổng hợp đối chiếu chỉ số định lượng giữa Mô hình A và Mô hình B (Macro F1, Accuracy, Per-class F1, Recall).
3. **`confusion_matrix.png`**: Biểu đồ trực quan so sánh Ma trận nhầm lẫn giữa Mô hình A và Mô hình B.

### 5.2. Chia sẻ thư mục kết quả trên Google Drive
Để gửi kết quả cho nhóm tổng hợp:
1. Mở [Google Drive](https://drive.google.com/).
2. Điều hướng đến: **My Drive** $\rightarrow$ **`PhoBERT_ViHSD`** $\rightarrow$ **`outputs`**.
3. Nhấp chuột phải vào thư mục `seed{SEED}_teencode_414` $\rightarrow$ Chọn **Share (Chia sẻ)**.
4. Tại mục **General access**, chọn **Anyone with the link (Bất kỳ ai có liên kết)** với quyền **Viewer**.
5. Nhấn **Copy link** và gửi vào kênh thảo luận của nhóm nghiên cứu.

---

## 6. Xử Lý Sự Cố Kỹ Thuật Thường Gặp (Troubleshooting)

### 6.1. Lỗi hết bộ nhớ GPU (CUDA Out of Memory - OOM)
- **Nguyên nhân:** Mô hình A chưa được dọn dẹp giải phóng hoàn toàn khỏi VRAM trước khi khởi tạo Mô hình B.
- **Khắc phục:** Đảm bảo cell giải phóng bộ nhớ giữa 2 mô hình đã được chạy:
  ```python
  import gc, torch
  del model_A
  gc.collect()
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
  ```
- Nếu vẫn gặp lỗi, giữ nguyên `max_length = 128` và giảm `batch_size` xuống `8` (kết hợp `gradient_accumulation_steps = 2` để bảo toàn effective batch size).

### 6.2. Lỗi `FileNotFoundError: ./data/train.csv`
- **Nguyên nhân:** Chưa sao chép tệp từ Drive vào thư mục `./data/` trên máy ảo Colab.
- **Khắc phục:** Chạy lại lệnh copy dữ liệu:
  ```bash
  !mkdir -p ./data
  !cp -r /content/drive/MyDrive/PhoBERT_ViHSD/data/* ./data/
  !ls -la ./data
  ```

### 6.3. Cảnh báo Secret `HF_TOKEN` khi tải PhoBERT
- **Hiện tượng:** Xuất hiện cảnh báo `UserWarning: Error while fetching HF_TOKEN secret value...`.
- **Giải thích:** Đây là cảnh báo thông thường của Hugging Face Hub khi tải mô hình công khai `vinai/phobert-base-v2`. Mô hình vẫn được tải xuống bình thường và **hoàn toàn không ảnh hưởng** đến quá trình huấn luyện.

---

## 7. Checklist Kiểm Tra Hoàn Thành Nhiệm Vụ

- [ ] Đã đồng bộ thư mục `data/` và notebook lên `MyDrive/PhoBERT_ViHSD/`.
- [ ] Đã kiểm tra kết nối GPU (`!nvidia-smi` hiển thị Tesla T4).
- [ ] Đã cập nhật biến `SEED` theo đúng phân công:
  - [x] Trí: `SEED = 42` (Đã hoàn thành)
  - [x] Ngân: `SEED = 123` (Đã hoàn thành)
  - [x] Hiếu: `SEED = 456` (Đã hoàn thành)
  - [x] Tài: `SEED = 789` (Đã hoàn thành)
- [ ] Đã hoàn thành huấn luyện cả Mô hình A (Baseline) và Mô hình B (Proposed).
- [ ] Đã xác nhận tệp `predictions.csv` đủ 6,680 dòng và định dạng chuẩn.
- [ ] Đã đồng bộ sản phẩm đầu ra về Google Drive và gửi liên kết chia sẻ cho nhóm.
