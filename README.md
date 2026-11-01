<div align="center">

# Phân Loại Đa Lớp Ngôn Từ Thù Ghét Trên Mạng Xã Hội Tiếng Việt
### Tối Ưu Hóa PhoBERT Với Trọng Số Lớp Và Chuẩn Hóa Teencode Trên Bộ Dữ Liệu ViHSD

[![Original Repo](https://img.shields.io/badge/Original%20Repo-sonlam1102%2Fvihsd-181717.svg?logo=github)](https://github.com/sonlam1102/vihsd)
[![Dataset](https://img.shields.io/badge/Dataset-ViHSD%20(UIT--NLP)-green.svg)](https://huggingface.co/datasets/sonlam1102/vihsd)
[![Backbone](https://img.shields.io/badge/Backbone-VinAI%20PhoBERT--base--v2-yellow.svg?logo=huggingface)](https://github.com/VinAIResearch/PhoBERT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.38%2B-orange.svg)](https://huggingface.co/docs/transformers)
[![Hardware](https://img.shields.io/badge/Hardware-Tesla%20T4%20(FP16)-76B900.svg?logo=nvidia)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

<br/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

<p align="center">
  Dự án kế thừa và cải tiến phương pháp từ bài báo của <i>Luu et al. (2021)</i> trên bộ dữ liệu ViHSD.<br/>
  Trọng tâm là giải quyết vấn đề <b>dữ liệu mất cân bằng</b> và <b>ngôn ngữ mạng xã hội (teencode, viết tắt)</b> bằng cách kết hợp <b>PhoBERT-base-v2</b> với <b>trọng số lớp (Class Weight)</b>.
</p>

---

</div>

## 1. Đề Tài

### 1.1. Bài toán
Phát hiện và phân loại bình luận trên mạng xã hội tiếng Việt thành 3 nhóm riêng biệt:
- `0: CLEAN`: Bình luận sạch, mang tính xây dựng hoặc bình thường.
- `1: OFFENSIVE`: Bình luận thô tục, khiếm nhã, chửi thề bộc phát nhưng không nhắm vào đối tượng cụ thể.
- `2: HATE`: Ngôn từ thù ghét, xúc phạm danh dự hoặc kỳ thị vùng miền, dân tộc, tôn giáo, giới tính có chủ đích.

### 1.2. Vấn đề cần giải quyết
- **Dữ liệu bị lệch rất lớn (Class Imbalance):** Tỷ lệ phân bố là `12.2 : 1.0 : 1.6` (`CLEAN` 82.71%, `OFFENSIVE` 6.77%, `HATE` 10.52%). Nếu chỉ dùng hàm mất mát Cross-Entropy thông thường, mô hình sẽ có xu hướng đoán phần lớn là `CLEAN` để lấy độ chính xác (Accuracy) cao, nhưng lại bỏ sót các bình luận thù ghét nguy hiểm.
- **Đặc trưng ngôn ngữ mạng xã hội:** Bình luận chứa nhiều từ viết tắt, teencode (`ko`, `dc`, `vcl`, `chug`...) và biểu tượng cảm xúc (emoji). Nếu không xử lý trước, bộ tách từ sẽ bị lỗi hoặc băm nhỏ câu thành các mảnh vô nghĩa (`<unk>`), làm mô hình khó hiểu ngữ cảnh.

---

## 2. Bài Báo Gốc & Nguồn Dữ Liệu

> [!NOTE]
> **Về nguồn gốc và bản quyền:** Dự án này là nghiên cứu mở rộng dựa trên bài báo và dữ liệu của nhóm tác giả Luu et al. (2021). Chúng tôi không tự nhận quyền sở hữu đối với tập dữ liệu ViHSD hay mô hình ngôn ngữ PhoBERT. Toàn bộ mã nguồn gốc của bài báo được lưu tại thư mục [`original_paper_notebooks/`](original_paper_notebooks/) để đối chứng khách quan.

### 2.1. Bài báo nền tảng & Bộ dữ liệu ViHSD
- **Tên bài báo:** *A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts*
- **Tác giả:** Son T. Luu, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen (UIT-NLP Lab, Trường ĐH Công nghệ Thông tin - ĐHQG-HCM).
- **Xuất bản:** Hội nghị IEA/AIE 2021 (Springer LNAI 12798, pp. 415–426, DOI: [10.1007/978-3-030-79457-6_35](https://doi.org/10.1007/978-3-030-79457-6_35), arXiv: [2103.11528](https://arxiv.org/abs/2103.11528)).
- **Kho mã nguồn gốc:** [github.com/sonlam1102/vihsd](https://github.com/sonlam1102/vihsd)
- **Tập dữ liệu ViHSD (33,400 bình luận Facebook):**
  - Giữ đúng phân tập chuẩn: Train (24,048 mẫu - 72%), Dev (2,672 mẫu - 8%), Test (6,680 mẫu - 20%).
  - Dữ liệu chính thức: [Hugging Face sonlam1102/vihsd](https://huggingface.co/datasets/sonlam1102/vihsd).

### 2.2. Mô hình nền tảng PhoBERT
- **Công trình gốc:** *PhoBERT: Pre-trained language models for Vietnamese* (Nguyen & Nguyen, EMNLP 2020) của VinAI Research.
- **Kho lưu trữ:** [github.com/VinAIResearch/PhoBERT](https://github.com/VinAIResearch/PhoBERT) | Hugging Face: [`vinai/phobert-base-v2`](https://huggingface.co/vinai/phobert-base-v2).

---

## 3. Cải Tiến Đã Làm Được So Với Bài Báo

### 3.1. Bảng so sánh kỹ thuật

| Nội dung | Bài báo gốc (Luu et al., 2021) | Cải tiến của đề tài này | Hiệu quả thực tế |
| :--- | :--- | :--- | :--- |
| **Xử lý dữ liệu lệch** | Dùng Cross-Entropy thông thường; không có cơ chế bù trừ trọng số lớp. | Áp dụng **Balanced Class Weight** ($w_{\text{CLEAN}} \approx 0.40, w_{\text{OFF}} \approx 4.99, w_{\text{HATE}} \approx 3.14$), phạt nặng hơn khi đoán sai lớp thiểu số. | **Bắt đúng thêm +22.67% bình luận HATE** và **+26.12% bình luận OFFENSIVE** trên tập kiểm tra. |
| **Chuẩn hóa Teencode** | Không có bước xử lý teencode; từ viết tắt bị băm thành token lạ (`<unk>`). | Xây dựng 2 từ điển chuẩn hóa: `teencode_35.json` (35 từ cơ bản) và `teencode_414.json` (414 từ mở rộng). | Giúp PhoBERT hiểu đúng nghĩa của từ viết tắt trên mạng; giảm hiện tượng từ ngoài từ điển (OOV). |
| **Xử lý Emoji & Link** | Để nguyên emoji; dễ gây crash bộ tách từ hoặc tạo ra từ ghép dị dạng. | Xóa toàn bộ emoji và đường link (URL) trước khi tách từ; chuẩn hóa bảng mã Unicode NFC. | Bảo vệ tokenizer hoạt động ổn định và giữ nguyên cấu trúc câu tiếng Việt. |
| **Bộ tách từ tiếng Việt** | Dùng `VnCoreNLP` phụ thuộc Java JVM, cài đặt lâu và dễ quá tải bộ nhớ trên Colab. | Chuyển sang `PyVi` thuần Python nhẹ, kết hợp nâng cấp `transformers>=4.38.0` và `accelerate`. | Chạy mượt mà, không bị lỗi đứt phiên (crash session) trên Google Colab. |
| **Tối ưu VRAM & Tốc độ** | Cố định độ dài chuỗi 256/512 tĩnh; tốn nhiều bộ nhớ GPU. | Rút gọn `max_length = 128` kết hợp đệm động theo nhóm (**Dynamic Padding**) và tính toán nửa chính xác (**FP16**). | **Tiết kiệm 50% VRAM**, huấn luyện **nhanh gấp 2.5 lần** trên GPU Tesla T4; bao trọn 98.5% bình luận. |
| **Cách đo lường thực nghiệm** | Đánh giá thử nghiệm 1 lần (single-seed). | Chạy thực nghiệm A/B đối chứng trên **4 random seeds (`42, 123, 456, 789`)**. | Đảm bảo kết quả khách quan và ổn định thống kê ($\mu \pm \sigma$), tránh yếu tố may rủi. |
| **Hiệu năng tổng thể** | Baseline m-BERT cased đạt Macro F1: **62.69%**. | PhoBERT-base-v2 + Class Weight đạt Macro F1: **67.85%**. | **Tăng +5.16%** so với baseline bài báo gốc; **tăng +5.70%** so với PhoBERT không dùng trọng số. |

### 3.2. Kết quả kiểm định trên tập Test (6,680 mẫu)

So sánh đối chứng giữa mô hình huấn luyện thông thường (Mô hình A) và mô hình có áp dụng trọng số lớp (Mô hình B) trên cùng tập dữ liệu:

| Chỉ số đánh giá | Mô hình A (Không trọng số) | Mô hình B (Có Class Weight) | Mức cải thiện ($\Delta$) | Tác động thực tiễn |
| :--- | :---: | :---: | :---: | :--- |
| **Macro F1-Score** | 0.6215 | **0.6785** | **+5.70%** | Chất lượng nhận diện chung cả 3 lớp tăng rõ rệt |
| **Độ bao phủ HATE (Recall)** | 43.75% | **66.42%** | **+22.67%** | Phát hiện thêm 156 bình luận thù ghét bị bỏ sót |
| **Độ bao phủ OFFENSIVE (Recall)** | 31.98% | **58.10%** | **+26.12%** | Tăng mạnh khả năng nhận diện bình luận khiếm nhã |
| **Lỗi bỏ sót HATE sang CLEAN** | 341 mẫu | **183 mẫu** | **-46.33%** | Giảm gần một nửa số ca bỏ sót nguy hiểm nhất |
| **Weighted F1-Score** | 0.8650 | **0.8682** | **+0.32%** | Duy trì tốt hiệu năng phân loại trên toàn bộ dữ liệu |
| **Độ chính xác (Accuracy)** | **87.80%** | 85.40% | *-2.40%* | Chấp nhận giảm nhẹ để bắt được nhiều nội dung độc hại hơn |

> [!TIP]
> **Vì sao Accuracy giảm nhẹ nhưng mô hình lại hiệu quả hơn?**
> Vì lớp `CLEAN` chiếm hơn 82% dữ liệu, nếu mô hình đoán bừa mọi câu đều là `CLEAN` thì Accuracy vẫn đạt trên 82% nhưng hoàn toàn vô dụng vì không lọc được câu thù ghét nào. Việc giảm nhẹ 2.4% Accuracy để cứu lại hơn 22% câu thù ghét là sự đánh đổi hoàn toàn tối ưu trong bài toán kiểm duyệt nội dung mạng xã hội.

### 3.3. Mã nguồn thực thi
- [`PhoBERT_ViHSD_ClassWeight.ipynb`](PhoBERT_ViHSD_ClassWeight.ipynb): Notebook hoàn chỉnh chạy trên Google Colab GPU T4, thực hiện tiền xử lý teencode (414 từ), tách từ tiếng Việt bằng PyVi, tính class weight, huấn luyện đối chứng Model A vs Model B và xuất file dự đoán chuẩn `predictions.csv`.
- [`docs/05_mau_bao_cao_so_sanh.md`](docs/05_mau_bao_cao_so_sanh.md): Báo cáo đối chứng khoa học, phân tích ma trận nhầm lẫn và khảo sát các nhóm ca lỗi điển hình.

---

## Trích Dẫn Bài Báo Gốc (Citation)

Nếu sử dụng lại dữ liệu hoặc tham khảo phương pháp, vui lòng trích dẫn các bài báo nền tảng gốc:

```bibtex
@inproceedings{luu-etal-2021-large,
  title     = {A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts},
  author    = {Luu, Son T. and Nguyen, Kiet Van and Nguyen, Ngan Luu-Thuy},
  booktitle = {Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices (IEA/AIE 2021)},
  pages     = {415--426},
  year      = {2021},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-79457-6_35},
  url       = {https://github.com/sonlam1102/vihsd}
}

@inproceedings{nguyen-nguyen-2020-phobert,
  title     = {{PhoBERT}: Pre-trained language models for {V}ietnamese},
  author    = {Nguyen, Dat Quoc and Nguyen, Anh Tuan},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  pages     = {1037--1042},
  year      = {2020},
  doi       = {10.18653/v1/2020.findings-emnlp.92},
  url       = {https://github.com/VinAIResearch/PhoBERT}
}
```

---

<div align="center">

**Nghiên Cứu Mở Rộng Trên Bộ Dữ Liệu ViHSD Benchmark**  
Tôn trọng bản quyền học thuật và đóng góp kỹ thuật mã nguồn mở

</div>
