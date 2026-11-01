# 04. HƯỚNG DẪN TRIỂN KHAI VÀ HUẤN LUYỆN TRÊN GOOGLE COLAB

Tài liệu này cung cấp hướng dẫn kỹ thuật chi tiết từng bước (Step-by-step) dành cho các thành viên trong nhóm nghiên cứu để triển khai huấn luyện mô hình **PhoBERT-base** kết hợp **Class Weight** trên bộ dữ liệu **ViHSD**. 

Tài liệu bao gồm 2 phương thức thực thi:
1. **Phương pháp 1:** Huấn luyện trực tiếp trên giao diện Web của **Google Colab**.
2. **Phương pháp 2:** Huấn luyện qua **Extension Google Colab trong VSCode** (kết nối tài khoản Colab PRO 5TB).

Đồng thời, tài liệu hướng dẫn cụ thể quy trình chuẩn bị dữ liệu, phân công seed ngẫu nhiên độc lập giữa các thành viên, và chia sẻ thư mục sản phẩm đầu ra trên Google Drive.

---

## 1. Mục Tiêu Thực Nghiệm & Phân Công Nhiệm Vụ Nhóm

Để đảm bảo tính khách quan và khoa học theo chuẩn các hội nghị NLP quốc tế (ACL, EMNLP), mỗi mô hình cần được huấn luyện độc lập qua nhiều **hạt giống ngẫu nhiên (Random Seeds)** nhằm đo lường giá trị trung bình và độ lệch chuẩn ($\mu \pm \sigma$).

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

| Thành viên thực hiện | Hạt giống (Random Seed) | Cấu hình biến trong Notebook | Thư mục kết quả trên Drive (`outputs/`) | Trạng thái |
|:---|:---:|:---|:---|:---:|
| **Trí** | **42** | `SEED = 42` | `outputs/seed42_teencode_414/` | Đã hoàn thành (Mốc đối chứng) |
| **Ngân** | **123** | `SEED = 123` | `outputs/seed123_teencode_414/` | Đã hoàn thành (Kiểm định đa Seed) |
| **Hiếu** | **456** | `SEED = 456` | `outputs/seed456_teencode_414/` | **Cần thực hiện** |
| **Tài** | **789** | `SEED = 789` | `outputs/seed789_teencode_414/` | **Cần thực hiện** |

### 1.2. Thao tác bắt buộc trước khi bấm chạy
Tại **Cell số 2** của notebook `PhoBERT_ViHSD_ClassWeight.ipynb`, thành viên sửa giá trị biến `SEED` theo đúng phân công:
- **Ngân:**
  ```python
  SEED = 123
  ```
- **Hiếu:**
  ```python
  SEED = 456
  ```
- **Tài:**
  ```python
  SEED = 789
  ```

*Ghi chú:* Toàn bộ các đường dẫn lưu trữ checkpoint và tệp đầu ra (`OUTPUT_DIR`, `CHECKPOINT_DIR`, `DRIVE_OUTPUT_DIR`) trong notebook đã được tự động gắn mã seed (`f"seed{SEED}_teencode_414"`), không ghi đè lẫn nhau.

---

## 2. Chuẩn Bị Dữ Liệu và Dự Án Trên Google Drive

Trước khi bắt đầu huấn luyện, các thành viên cần đồng bộ bộ dữ liệu ViHSD và từ điển teencode lên Google Drive cá nhân.

### 2.1. Cấu trúc thư mục chuẩn trên Google Drive
Hãy tạo cấu trúc thư mục trên Google Drive của bạn chính xác như sau:

```
Google Drive của tôi (MyDrive)
└── PhoBERT_ViHSD/
    ├── PhoBERT_ViHSD_ClassWeight.ipynb    # Notebook chính để chạy huấn luyện
    └── data/                              # Thư mục dữ liệu bắt buộc
        ├── train.csv                      # Tập huấn luyện gốc (24,048 mẫu)
        ├── dev.csv                        # Tập kiểm định gốc (2,672 mẫu)
        ├── test.csv                       # Tập kiểm tra gốc (6,680 mẫu)
        └── teencode_414.json              # Từ điển chuẩn hóa 414 teencode
```

### 2.2. Các bước tải dữ liệu lên Google Drive
1. Truy cập [Google Drive](https://drive.google.com/).
2. Tại thư mục **My Drive (Drive của tôi)**, tạo một thư mục mới đặt tên là: `PhoBERT_ViHSD`.
3. Mở thư mục `PhoBERT_ViHSD`, tạo tiếp thư mục con tên là `data`.
4. Kéo thả 4 tệp dữ liệu từ thư mục `data/` của dự án lên:
   - `train.csv`
   - `dev.csv`
   - `test.csv`
   - `teencode_414.json`
5. Tải tệp notebook `PhoBERT_ViHSD_ClassWeight.ipynb` vào thư mục `PhoBERT_ViHSD`.

---

## 3. PHƯƠNG PHÁP 1: Huấn Luyện Bằng Google Colab Web (Trình Duyệt)

Phương pháp này phù hợp khi chạy trực tiếp trên trình duyệt Web (Chrome, Edge, Firefox, Brave) không cần cài đặt phần mềm phụ trợ.

### Bước 1: Mở Notebook trên Google Colab Web
1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Chọn tab **Google Drive** $\rightarrow$ Tìm và mở tệp `PhoBERT_ViHSD/PhoBERT_ViHSD_ClassWeight.ipynb`.
3. (Hoặc tại giao diện Google Drive, nhấp chuột phải vào file `.ipynb` $\rightarrow$ Chọn **Mở bằng** $\rightarrow$ **Google Colaboratory**).

### Bước 2: Thiết lập GPU T4 và Kiểm tra Phần cứng
1. Trên thanh menu, chọn: **Runtime (Thời gian chạy)** $\rightarrow$ **Change runtime type (Thay đổi loại thời gian chạy)**.
2. Tại mục **Hardware accelerator (Phần cứng tăng tốc)**:
   - Chọn **T4 GPU** (hoặc chọn **A100 / V100 GPU** nếu tài khoản sử dụng Colab Pro).
3. Nhấn **Save (Lưu)**.
4. Chạy cell kiểm tra GPU:
   ```python
   !nvidia-smi
   ```
   *Xác nhận: Hiển thị GPU Tesla T4 (khoảng 15GB VRAM).*

### Bước 3: Mount Google Drive
Tại cell đầu tiên của notebook hoặc tạo cell mới, chạy đoạn mã để kết nối Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```
- Colab sẽ hiển thị hộp thoại hoặc thông báo yêu cầu cấp quyền: chọn **Kết nối với Google Drive (Connect to Google Drive)**.
- Đăng nhập tài khoản Google và cấp quyền đọc/ghi.
- Khi màn hình xuất hiện thông báo `Mounted at /content/drive` là đã thành công.

### Bước 4: Thiết lập và đổi đường dẫn dữ liệu (`/content/...`)
Sau khi mount Drive, toàn bộ tệp trong Drive của bạn sẽ nằm tại `/content/drive/MyDrive/`. 

Để quá trình đọc dữ liệu và huấn luyện đạt tốc độ cao nhất (tránh độ trễ mạng khi đọc I/O từ Drive), nhóm khuyến nghị **sao chép thư mục data về ổ đĩa cục bộ `/content/`** của máy ảo Colab:

```python
import os
import shutil

# Tạo thư mục data cục bộ trên máy ảo Colab
os.makedirs('./data', exist_ok=True)

# Sao chép dữ liệu từ Drive vào máy ảo
!cp -r /content/drive/MyDrive/PhoBERT_ViHSD/data/* ./data/

# Kiểm tra danh sách tệp đã copy
print("Các tệp dữ liệu sẵn sàng trong máy ảo:")
print(os.listdir('./data'))
```

> **Ghi chú thay thế (Nếu không muốn sao chép file):**  
> Nếu không dùng lệnh sao chép trên, bạn có thể chỉnh trực tiếp đường dẫn đọc dữ liệu tại **Cell số 2 và 3** của notebook trỏ thẳng vào Drive:
> ```python
> DATA_DIR = "/content/drive/MyDrive/PhoBERT_ViHSD/data"
> train = pd.read_csv(f"{DATA_DIR}/train.csv")
> dev   = pd.read_csv(f"{DATA_DIR}/dev.csv")
> test  = pd.read_csv(f"{DATA_DIR}/test.csv")
> with open(f"{DATA_DIR}/teencode_414.json", "r", encoding="utf-8") as f:
>     TEENCODE_DICT = json.load(f)
> ```

### Bước 5: Cấu hình Seed theo phân công
Tìm đến **Cell số 2** và cập nhật giá trị `SEED`:
- Ngân đổi thành: `SEED = 123`
- Hiếu đổi thành: `SEED = 456`
- Tài đổi thành: `SEED = 789`

### Bước 6: Khởi chạy toàn bộ quá trình huấn luyện
- Trên thanh menu chọn **Runtime** $\rightarrow$ **Run all (Chạy tất cả)** hoặc nhấn tổ hợp phím `Ctrl + F9`.
- Quá trình chạy gồm 2 giai đoạn tự động:
  1. Huấn luyện **Mô hình A (Baseline)**: ~4 epoch (~15-18 phút).
  2. Huấn luyện **Mô hình B (Class Weight)**: ~4 epoch (~15-18 phút).
  3. Dự đoán trên tập Test (6,680 mẫu), vẽ Confusion Matrix, xuất `predictions.csv`.
  4. Đồng bộ kết quả tự động về Google Drive.
- Tổng thời gian chạy ước tính: **khoảng 35 - 45 phút**.

---

## 4. PHƯƠNG PHÁP 2: Huấn Luyện Dùng Extension Colab Trong VSCode

Phương pháp này cho phép thành viên chạy và kiểm soát notebook trực tiếp trong trình soạn thảo **Visual Studio Code (VSCode)** trên máy cá nhân, nhưng vẫn tận dụng sức mạnh tính toán GPU từ Google Colab.

### Bước 1: Cài đặt Extension Colab trên VSCode
1. Mở VSCode trên máy tính cá nhân.
2. Nhấn tổ hợp phím `Ctrl + Shift + X` (Windows/Linux) hoặc `Cmd + Shift + X` (macOS) để mở tab **Extensions**.
3. Nhập từ khóa: `Google Colab` (hoặc `Jupyter`).
4. Tìm extension **Google Colab** (chính thức từ Google / hỗ trợ Colab kernel) và extension **Jupyter** (từ Microsoft) $\rightarrow$ Nhấn **Install**.

### Bước 2: Mở file Notebook và Chọn Kernel Colab
1. Trong VSCode, chọn `File` $\rightarrow$ `Open File...` $\rightarrow$ Mở tệp `PhoBERT_ViHSD_ClassWeight.ipynb`.
2. Ở góc trên cùng bên phải của giao diện Notebook, click vào nút **Select Kernel (Chọn Kernel)**.
3. Trong danh sách hiện ra, chọn **Google Colab** (hoặc chọn **Existing Jupyter Server...** và dán URL phiên Colab).

### Bước 3: Xác thực tài khoản Colab PRO (Dung lượng 5TB)
Để có hiệu năng cao nhất, không bị giới hạn 12h và sở hữu dung lượng lưu trữ đám mây lên đến **5TB**:
1. Khi chọn Kernel Colab, VSCode sẽ mở trình duyệt yêu cầu đăng nhập tài khoản Google.
2. Đăng nhập bằng tài khoản **Google Colab PRO**.
3. Cấp quyền truy cập cho VSCode kết nối tới tài khoản Google Cloud / Colab của bạn.
4. Sau khi hoàn tất xác thực, góc trên bên phải VSCode sẽ hiển thị trạng thái kết nối:  
   `Google Colab: Connected (T4 GPU / Premium GPU)`.

### Bước 4: Chạy Test Cell Đầu Tiên Để Kiểm Tra Kết Nối
1. Chạy cell đầu tiên của notebook (cell cài đặt thư viện `!pip install ...` hoặc tạo một cell lệnh `!nvidia-smi`) để kích hoạt kết nối tới máy chủ Colab.
2. Kiểm tra xem cell có thực thi bình thường và trả về kết quả hay không. Thao tác này nhằm đảm bảo phiên làm việc (Session) đã được khởi tạo thành công trên máy chủ Colab và GPU đã sẵn sàng.

### Bước 5: Upload Thư Mục Data Lên Máy Ảo Colab
Vì notebook đọc dữ liệu từ đường dẫn cục bộ `./data/`, bạn cần đưa dữ liệu từ máy cá nhân lên phiên làm việc Colab:
1. Trên cây thư mục bên trái của VSCode (Explorer), tìm đến thư mục **`data`** của dự án.
2. **Nhấp chuột phải vào thư mục `data`** $\rightarrow$ Chọn **Upload to Colab**.
3. Chờ quá trình tải lên hoàn tất trong vài giây.
4. Mở tab quản lý tệp của Colab (biểu tượng thư mục Colab trên thanh bên VSCode hoặc mục Colab Files) $\rightarrow$ **Kiểm tra xem thư mục `data` đã xuất hiện trên máy ảo Colab chưa**, và xác nhận bên trong đã có đủ 4 tệp:
   - `train.csv`
   - `dev.csv`
   - `test.csv`
   - `teencode_414.json`

### Bước 6: Cấu hình Seed theo phân công nhiệm vụ
Tìm đến **Cell số 2** và sửa biến `SEED` theo đúng người phụ trách:
- Ngân: `SEED = 123`
- Hiếu: `SEED = 456`
- Tài: `SEED = 789`

### Bước 7: Khởi chạy toàn bộ Source và Lưu trữ về Drive
1. Sau khi đã kiểm tra thư mục `data` đã có đầy đủ trên Colab, nhấn nút **Run All** ở thanh công cụ phía trên notebook (hoặc bấm chạy tuần tự các cell).
2. Khi tiến trình chạy đến cell cuối (Cell lưu trữ về Google Drive):
   - Một thông báo xác thực (OAuth authorization) sẽ hiển thị hoặc mở tab trình duyệt.
   - Đăng nhập tài khoản Google và nhấn **Allow (Cho phép)** để cấp quyền mount Drive.
   - Notebook sẽ tự động sao chép các tệp kết quả (`predictions.csv`, `aggregated_metrics.csv`, `confusion_matrix.png`, checkpoint) vào Google Drive của bạn.

---

## 5. Mở Drive và Chia Sẻ Thư Mục Output

Sau khi chạy xong cell cuối cùng của Notebook, toàn bộ kết quả thực nghiệm và trọng số mô hình đã được tự động lưu vào Google Drive tại đường dẫn:

```
MyDrive/PhoBERT_ViHSD/outputs/seed{SEED}_teencode_414/
MyDrive/PhoBERT_ViHSD/checkpoints/seed{SEED}_teencode_414/
```

### 5.1. Danh mục các sản phẩm đầu ra bắt buộc phải có
Tại thư mục `outputs/seed{SEED}_teencode_414/`, hãy kiểm tra để đảm bảo có đầy đủ 3 tệp sau:

1. **`predictions.csv`**: Tệp dự đoán chính thức của Mô hình B trên tập Test (6,680 mẫu):
   - Đủ **6,680 dòng** (không tính header).
   - Đúng **4 cột**: `id`, `text`, `ground_truth`, `predicted_label`.
   - Không chứa bất kỳ giá trị `NaN` hay `Null`.
2. **`aggregated_metrics.csv`**: Bảng tổng hợp đối chiếu chỉ số định lượng ($F_1$-Macro, Accuracy, Per-class F1, Recall).
3. **`confusion_matrix.png`**: Biểu đồ đối chiếu trực quan Ma trận nhầm lẫn giữa Mô hình A và Mô hình B.

### 5.2. Các bước Chia Sẻ Thư Mục Kết Quả trên Google Drive
Để gửi kết quả cho Trưởng nhóm tổng hợp:

1. Mở [Google Drive](https://drive.google.com/) trên trình duyệt.
2. Điều hướng đến thư mục: **My Drive** $\rightarrow$ **`PhoBERT_ViHSD`** $\rightarrow$ **`outputs`**.
3. Bạn sẽ thấy thư mục kết quả tương ứng với seed của mình:
   - Trí: Thư mục `seed42_teencode_414`
   - Ngân: Thư mục `seed123_teencode_414`
   - Hiếu: Thư mục `seed456_teencode_414`
   - Tài: Thư mục `seed789_teencode_414`
4. **Nhấp chuột phải** vào thư mục đó $\rightarrow$ Chọn **Share (Chia sẻ)** $\rightarrow$ **Share (Chia sẻ)**.
5. Tại mục **General access (Quyền truy cập chung)**:
   - Chuyển từ *Restricted (Bị hạn chế)* sang **Anyone with the link (Bất kỳ ai có đường liên kết)**.
   - Vai trò: Chọn **Viewer (Người xem)** hoặc **Editor (Người chỉnh sửa)**.
   *(Hoặc nhập trực tiếp địa chỉ email của các thành viên trong nhóm).*
6. Nhấn nút **Copy link (Sao chép đường liên kết)**.
7. Gửi link thư mục kèm ảnh chụp ma trận nhầm lẫn vào nhóm thảo luận dự án.

---

## 6. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

### 6.1. Lỗi Hết Bộ Nhớ GPU (CUDA Out of Memory - OOM)
- **Nguyên nhân:** Mô hình A chưa được dọn dẹp khỏi VRAM trước khi khởi tạo Mô hình B.
- **Khắc phục:** Đảm bảo notebook có cell giải phóng bộ nhớ giữa 2 mô hình:
  ```python
  del model_A
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
  ```
- Nếu vẫn gặp lỗi, hạ `per_device_train_batch_size = 16` xuống `8` (hoặc kiểm tra `max_length = 128`).

### 6.2. Lỗi `FileNotFoundError: ./data/train.csv`
- **Nguyên nhân:** Chưa sao chép tệp từ Drive vào thư mục `/content/data/` hoặc đường dẫn chưa khớp.
- **Khắc phục:** Chạy lại lệnh copy dữ liệu:
  ```bash
  !mkdir -p ./data
  !cp -r /content/drive/MyDrive/PhoBERT_ViHSD/data/* ./data/
  !ls -la ./data
  ```

### 6.3. Lỗi cảnh báo Secret `HF_TOKEN` khi tải PhoBERT
- **Hiện tượng:** Xuất hiện cảnh báo `UserWarning: Error while fetching HF_TOKEN secret value...`.
- **Giải thích:** Đây chỉ là cảnh báo thông thường của Hugging Face Hub khi tải mô hình công khai `vinai/phobert-base-v2`. Mô hình vẫn được tải xuống bình thường và **hoàn toàn không ảnh hưởng** đến kết quả huấn luyện.

---

## 7. Tóm Tắt Checklist Hoàn Thành Nhiệm Vụ

- [ ] Đã upload bộ dữ liệu `data/` và notebook lên `MyDrive/PhoBERT_ViHSD/`.
- [ ] Đã kiểm tra GPU hoạt động (`!nvidia-smi` hiển thị Tesla T4).
- [ ] Đã cập nhật biến `SEED`:
  - [x] Trí: `SEED = 42` (Đã hoàn thành)
  - [x] Ngân: `SEED = 123` (Đã hoàn thành)
  - [ ] Hiếu: `SEED = 456`
  - [ ] Tài: `SEED = 789`
- [ ] Đã chạy hoàn tất cả 2 mô hình (Mô hình A và Mô hình B).
- [ ] Đã kiểm tra tính toàn vẹn của tệp `predictions.csv` (đủ 6,680 dòng).
- [ ] Đã chạy cell cuối để đồng bộ toàn bộ file sang Google Drive.
- [ ] Đã mở Google Drive và chia sẻ link thư mục `outputs/seed{SEED}_teencode_414` cho nhóm.
