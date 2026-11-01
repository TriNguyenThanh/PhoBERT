# PHÂN LOẠI ĐA LỚP NGÔN TỪ THÙ GHÉT TRÊN MẠNG XÃ HỘI TIẾNG VIỆT
### Huấn luyện PhoBERT-base kết hợp Class Weight trên Bộ dữ liệu ViHSD (Luu et al., 2021)

---

## 1. Giới thiệu Dự án

Dự án này tập trung giải quyết bài toán **Phân loại đa lớp Ngôn từ thù ghét (Hate Speech Detection)** trên dữ liệu văn bản mạng xã hội tiếng Việt bằng mô hình ngôn ngữ **PhoBERT-base** kết hợp kỹ thuật **Class Weight** nhằm khắc phục hiện tượng mất cân bằng lớp nghiêm trọng (Severe Class Imbalance).

- **Nghiên cứu tham chiếu:** *A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts* (Luu et al., EMNLP 2021).
- **Bộ dữ liệu chuẩn:** **ViHSD** (Vietnamese Hate Speech Dataset) gồm 33,400 bình luận Facebook được gán nhãn 3 lớp:
  - `0: CLEAN` (82.71%): Bình luận bình thường, sạch.
  - `1: OFFENSIVE` (6.77%): Ngôn từ xúc phạm, chửi thề nói chung, không nhắm vào đối tượng định danh.
  - `2: HATE` (10.52%): Ngôn từ thù ghét nhắm vào định danh (dân tộc, vùng miền, tôn giáo, giới tính,...).
- **Hạ tầng thực thi:** Google Colab GPU NVIDIA Tesla T4 với cơ chế tăng tốc tính toán nửa độ chính xác **FP16** (`fp16=True`).

---

## 2. Bản đồ Bộ Tài liệu Kỹ thuật (Documentation Map)

Bộ tài liệu của đề tài được tổ chức theo tiêu chuẩn nghiên cứu khoa học và kỹ thuật phần mềm, bao gồm các tài liệu chuyên đề sau:

| STT | Tài liệu | Nội dung chính |
|:---:|:---|:---|
| 1 | [docs/01_tong_quan_de_tai.md](docs/01_tong_quan_de_tai.md) | Bối cảnh, phát biểu bài toán, chi tiết bộ dữ liệu ViHSD, phân tích hiện tượng mất cân bằng lớp và các thách thức xử lý ngôn ngữ tiếng Việt. |
| 2 | [docs/02_kien_truc_va_phuong_phap.md](docs/02_kien_truc_va_phuong_phap.md) | Kiến trúc mạng PhoBERT-base, cơ chế tách từ tiếng Việt (Word Segmentation), cơ sở toán học của Cross-Entropy, Weighted Cross-Entropy (Class Weight) và Focal Loss. |
| 3 | [docs/03_thiet_ke_thuc_nghiem_va_danh_gia.md](docs/03_thiet_ke_thuc_nghiem_va_danh_gia.md) | Thiết lập môi trường Google Colab GPU T4, cơ chế FP16, bảng cấu hình siêu tham số (`max_length=128`, `batch_size=16`, `epochs=3-4`, `lr=2e-5`), và hệ thống độ đo (MF1, WF1, Per-class F1, Confusion Matrix). |
| 4 | [docs/04_huong_dan_trien_khai_colab.md](docs/04_huong_dan_trien_khai_colab.md) | Hướng dẫn chi tiết từng bước nạp dữ liệu ViHSD, tách từ bằng PyVi, tính class weight bằng `scikit-learn`, huấn luyện bằng Custom Trainer, xuất tệp `predictions.csv` và lưu checkpoint vào Google Drive. |
| 5 | [docs/05_mau_bao_cao_so_sanh.md](docs/05_mau_bao_cao_so_sanh.md) | Mẫu báo cáo khoa học đối chiếu định lượng và định tính giữa mô hình Baseline (không Class Weight) và mô hình Proposed (có Class Weight), phân tích đánh đổi Precision-Recall và phân loại các ca lỗi điển hình. |
| 6 | [docs/papers/01_paper_vihsd_luu2021.md](docs/papers/01_paper_vihsd_luu2021.md) | Phân tích pipeline bài báo gốc ViHSD (Luu et al., IEA/AIE 2021): thu thập dữ liệu, 4 annotators, baselines m-BERT, Text-CNN. |
| 7 | [docs/papers/02_paper_phobert_nguyen2020.md](docs/papers/02_paper_phobert_nguyen2020.md) | Phân tích pipeline bài báo gốc PhoBERT (Nguyen & Nguyen, EMNLP 2020): tiền huấn luyện RoBERTa 20GB, BPE 64k, RDRSegmenter. |
| 8 | [docs/papers/03_paper_phobert_cnn_tran2022.md](docs/papers/03_paper_phobert_cnn_tran2022.md) | Phân tích pipeline PhoBERT-CNN (Tran et al., NCAA 2022): trích xuất đặc trưng cục bộ, tiền xử lý teencode, tăng cường dữ liệu EDA đạt SOTA 67.46%. |
| 9 | [docs/papers/04_paper_bertology_focalloss_luu2024.md](docs/papers/04_paper_bertology_focalloss_luu2024.md) | Phân tích pipeline Focal Loss & BERTology (Luu et al., MTAP 2024): nghịch lý xóa stopword, chứng minh Focal Loss vượt trội trên ViHSD. |
| 10 | [docs/papers/05_paper_vihos_hoang2023.md](docs/papers/05_paper_vihos_hoang2023.md) | Phân tích pipeline ViHOS Spans Detection (Hoang et al., EACL 2023): gán nhãn BIO nhận diện khoảng từ thù ghét, khắc phục thiên kiến dữ liệu. |
| 11 | [docs/papers/06_paper_focal_loss_lin2017.md](docs/papers/06_paper_focal_loss_lin2017.md) | Phân tích lý thuyết Focal Loss (Lin et al., ICCV 2017): cơ chế điều biến động $(1-p_t)^\gamma$, triệt tiêu mất mát mẫu dễ, code PyTorch đa lớp. |

---

## 3. Cấu trúc Mã nguồn và Kho lưu trữ

```
.
├── README.md                                         <- Tổng quan đề tài và chỉ mục tài liệu
├── docs/                                             <- Thư mục tài liệu kỹ thuật
│   ├── 01_tong_quan_de_tai.md                        <- Tổng quan đề tài và bộ dữ liệu ViHSD
│   ├── 02_kien_truc_va_phuong_phap.md                <- Kiến trúc PhoBERT & Lý thuyết Class Weight
│   ├── 03_thiet_ke_thuc_nghiem_va_danh_gia.md        <- Thiết kế thực nghiệm và hệ thống độ đo
│   ├── 04_huong_dan_trien_khai_colab.md              <- Hướng dẫn từng bước chạy trên Colab
│   ├── 05_mau_bao_cao_so_sanh.md                     <- Báo cáo đối chứng Baseline vs Class Weight
│   └── papers/                                       <- Phân tích chi tiết pipeline từng bài báo
│       ├── 01_paper_vihsd_luu2021.md                 <- Luu et al. (2021) - ViHSD Dataset
│       ├── 02_paper_phobert_nguyen2020.md             <- Nguyen & Nguyen (2020) - PhoBERT
│       ├── 03_paper_phobert_cnn_tran2022.md          <- Tran et al. (2022) - PhoBERT-CNN SOTA
│       ├── 04_paper_bertology_focalloss_luu2024.md   <- Luu et al. (2024) - Focal Loss & BERTology
│       ├── 05_paper_vihos_hoang2023.md               <- Hoang et al. (2023) - ViHOS Spans Detection
│       └── 06_paper_focal_loss_lin2017.md            <- Lin et al. (2017) - Focal Loss Foundation
├── paper/                                            <- Các tệp PDF gốc của các bài báo
│   ├── 2017_Lin_Focal_Loss.pdf
│   ├── 2020_Nguyen_PhoBERT.pdf
│   ├── 2021_Luu_ViHSD_A_Large_Scale_Dataset.pdf
│   ├── 2022_Tran_PhoBERT_CNN_ViHSD.pdf
│   └── 2023_Hoang_ViHOS_Hate_Speech_Spans.pdf
└── pipeline/                                         <- Mã nguồn thực thi
    └── PhoBERT_ViHSD_ClassWeight_Colab.ipynb         <- Jupyter Notebook hoàn chỉnh chạy trên Colab
```

---

## 4. Tóm tắt Cấu hình Siêu tham số (Hyperparameter Specifications)

| Tham số | Giá trị | Giải thích lý do thiết lập |
|:---|:---:|:---|
| **Backbone Model** | `vinai/phobert-base-v2` | Mô hình ngôn ngữ tiền huấn luyện tối ưu nhất cho tiếng Việt. |
| **Max Sequence Length** | **128** | Bao phủ >98.5% độ dài bình luận ViHSD; tối ưu bộ nhớ GPU. |
| **Batch Size** | **16** | Đảm bảo an toàn không bị tràn VRAM trên GPU T4. |
| **Training Epochs** | **3 - 4** | Đảm bảo hội tụ tối ưu cho mô hình tiền huấn luyện lớn. |
| **Learning Rate** | **$2 \times 10^{-5}$** | Chuẩn tinh chỉnh mô hình Transformer; bộ tối ưu AdamW. |
| **Hardware & Precision** | **GPU T4 + FP16** | Giảm 50% VRAM, tăng tốc độ huấn luyện 2.5 lần. |
| **Xử lý mất cân bằng** | **Balanced Class Weight** | Trọng số phạt tính theo tần suất nghịch đảo từ tập Train bằng `sklearn`. |

---

## 5. Hướng dẫn Chạy Nhanh trên Google Colab (Quickstart)

1. Tải tệp notebook [pipeline/PhoBERT_ViHSD_ClassWeight_Colab.ipynb](pipeline/PhoBERT_ViHSD_ClassWeight_Colab.ipynb) lên [Google Colab](https://colab.research.google.com/).
2. Chọn **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Chọn **T4 GPU** $\rightarrow$ Lưu lại.
3. Chạy tuần tự các cell từ trên xuống dưới:
   - Cell 1-3: Kết nối Google Drive và kiểm tra GPU.
   - Cell 4-5: Cài đặt thư viện (`transformers`, `accelerate`, `scikit-learn`, `pyvi`, `datasets`).
   - Cell 6-8: Nạp dữ liệu ViHSD, tách từ tiếng Việt, và tính toán Class Weight.
   - Cell 9-11: Huấn luyện mô hình PhoBERT-base kết hợp Class Weight với FP16.
   - Cell 12-14: Đánh giá trên tập Test, hiển thị Ma trận nhầm lẫn (Confusion Matrix).
   - Cell 15-16: Xuất bản tệp dự đoán `predictions.csv` và lưu Checkpoint vào Google Drive.

---

## 6. Danh mục Sản phẩm Bàn giao (Deliverables Checklist)

- [x] **Tài liệu tổng quan và phân tích dữ liệu:** [docs/01_tong_quan_de_tai.md](docs/01_tong_quan_de_tai.md)
- [x] **Tài liệu kiến trúc và phương pháp toán học:** [docs/02_kien_truc_va_phuong_phap.md](docs/02_kien_truc_va_phuong_phap.md)
- [x] **Tài liệu thiết kế thực nghiệm & chỉ số đánh giá:** [docs/03_thiet_ke_thuc_nghiem_va_danh_gia.md](docs/03_thiet_ke_thuc_nghiem_va_danh_gia.md)
- [x] **Tài liệu hướng dẫn triển khai Google Colab:** [docs/04_huong_dan_trien_khai_colab.md](docs/04_huong_dan_trien_khai_colab.md)
- [x] **Mẫu báo cáo khoa học đối chiếu hiệu năng:** [docs/05_mau_bao_cao_so_sanh.md](docs/05_mau_bao_cao_so_sanh.md)
- [x] **Mã nguồn Notebook Colab hoàn chỉnh:** [pipeline/PhoBERT_ViHSD_ClassWeight_Colab.ipynb](pipeline/PhoBERT_ViHSD_ClassWeight_Colab.ipynb)
- [x] **Định dạng file dự đoán:** Tệp `predictions.csv` chuẩn 4 cột (`id`, `text`, `ground_truth`, `predicted_label`).
