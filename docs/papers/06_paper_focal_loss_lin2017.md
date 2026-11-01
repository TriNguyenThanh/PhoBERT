# 06. NỀN TẢNG TOÁN HỌC & PHƯƠNG PHÁP: FOCAL LOSS (LIN ET AL., 2017)

---

## 1. Thông tin Tổng quan Bài báo
- **Tên bài báo:** *Focal Loss for Dense Object Detection*
- **Tác giả:** Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár (Facebook AI Research - FAIR).
- **Hội nghị công bố:** IEEE International Conference on Computer Vision (ICCV 2017) - Đạt giải thưởng danh giá **Marr Prize (Best Student Paper Award)**.
- **Trích dẫn khoa học:** Hơn **40,000 lượt trích dẫn** trên Google Scholar (thuộc nhóm các công trình kinh điển nhất của học sâu thế giới).
- **arXiv:** [1708.02002](https://arxiv.org/abs/1708.02002)

---

## 2. Bối cảnh Ra đời & Bản chất Vấn đề Cần Giải quyết
Trong các mô hình phân loại đối mặt với sự mất cân bằng cực đoan (như nhận diện vật thể hoặc phân loại văn bản hiếm):
- Số lượng mẫu thuộc lớp đa số (âm tính/dễ phân loại) áp đảo hoàn toàn các mẫu thuộc lớp thiểu số (dương tính/khó phân loại) với tỷ lệ từ $1:100$ đến $1:10,000$.
- **Bẫy tối ưu hóa của Cross-Entropy truyền thống:**
  Dù mỗi mẫu dễ chỉ sinh ra một giá trị mất mát rất nhỏ (ví dụ $\text{loss} \approx 0.001$), nhưng khi nhân với hàng chục nghìn mẫu dễ, tổng mất mát tích lũy của chúng áp đảo hoàn toàn các mẫu khó. Gradient cập nhật bị cuốn theo hướng làm vừa lòng các mẫu dễ, khiến mô hình không học được ranh giới của các lớp thiểu số.

---

## 3. Sơ đồ So sánh Cơ chế Đạo hàm: Cross-Entropy vs. Focal Loss

```
HÀM MẤT MÁT CROSS-ENTROPY TIÊU CHUẨN:
   Loss(p_t) = - log(p_t)
   Khi p_t = 0.99 (mẫu rất dễ): Loss = 0.01  (Vẫn đóng góp gradient đáng kể khi số lượng lớn)

HÀM MẤT MÁT FOCAL LOSS (gamma = 2.0):
   Loss(p_t) = - (1 - p_t)^2 * log(p_t)
   Khi p_t = 0.99 (mẫu rất dễ):
      Nhân tử điều biến = (1 - 0.99)^2 = 0.0001
      Loss = 0.0001 * 0.01 = 0.000001  (Giảm đi 10,000 lần -> Triệt tiêu gradient rác)
   Khi p_t = 0.10 (mẫu khó, phân loại sai):
      Nhân tử điều biến = (1 - 0.10)^2 = 0.81
      Loss = 0.81 * 2.30 = 1.86  (Giữ nguyên 81% giá trị gradient để sửa lỗi)
```

---

## 4. Công thức Toán học & Cơ chế Điều biến Động (Modulating Factor)

### 4.1. Định nghĩa Xác suất $p_t$
Đặt $y \in \{+1, -1\}$ là nhãn thực tế và $p \in [0, 1]$ là xác suất dự đoán của mô hình:

$$p_t = \begin{cases} p & \text{nếu } y = 1 \\ 1 - p & \text{ngược lại} \end{cases}$$

### 4.2. Công thức Focal Loss Nguyên bản
$$\mathcal{L}_{\text{FL}}(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$

Trong đó:
1. **$\log(p_t)$:** Hàm mất mát Cross-Entropy gốc.
2. **$(1 - p_t)^\gamma$ (Modulating Factor - Nhân tử điều biến động):**
   - Khi một mẫu bị phân loại sai và $p_t \to 0$, nhân tử tiến tới $1$, độ lớn mất mát được giữ nguyên.
   - Khi mô hình tự tin phân loại đúng ($p_t \to 1$), nhân tử tiến về $0$, tự động làm suy giảm trọng số đóng góp của mẫu dễ.
3. **$\gamma \ge 0$ (Focusing Parameter - Tham số tập trung):**
   - Điều chỉnh tốc độ làm mịn mất mát của mẫu dễ.
   - Khi $\gamma = 0$: Trở về Cross-Entropy tiêu chuẩn.
   - Khi $\gamma = 2$: Cân bằng tối ưu giữa việc triệt tiêu mẫu dễ và bảo tồn thông tin mẫu biên.
4. **$\alpha_t \in [0, 1]$ (Alpha Balance):** Hệ số cân bằng tần suất lớp.

---

## 5. Mở rộng Focal Loss cho Bài toán Phân loại Văn bản Đa lớp (Multi-class NLP)

Trong bài toán phân loại đa lớp trên bộ dữ liệu **ViHSD** (3 lớp: CLEAN, OFFENSIVE, HATE):

### 5.1. Công thức Đa lớp (Multi-class Formulation)
Với mini-batch gồm $B$ mẫu, nhãn $y_i \in \{0, 1, 2\}$, và xác suất dự đoán sau tầng Softmax $p_{i, y_i} = \frac{\exp(z_{i, y_i})}{\sum_{c=0}^2 \exp(z_{i, c})}$:

$$\mathcal{L}_{\text{Multi-class Focal}} = - \frac{1}{B} \sum_{i=1}^B \alpha_{y_i} \left(1 - p_{i, y_i}\right)^\gamma \log\left(p_{i, y_i}\right)$$

### 5.2. Đoạn mã Cài đặt Chuẩn bằng PyTorch
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha      # Tensor trọng số [C]
        self.gamma = gamma      # Thường đặt gamma = 2.0
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Trích xuất log(p_t) và p_t tại nhãn thực tế targets
        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)

        # Tính nhân tử điều biến (1 - p_t)^gamma
        focal_term = (1.0 - pt).pow(self.gamma)
        loss = -focal_term * log_pt

        # Kết hợp trọng số lớp alpha_t nếu có
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha.gather(0, targets.squeeze(1))
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

---

## 6. Tại sao Focal Loss Vượt trội hơn Class Weight Tĩnh trên ViHSD?

| Tiêu chí | Weighted Cross-Entropy (Class Weight) | Multi-class Focal Loss (Lin et al., 2017) |
|:---|:---|:---|
| **Cấp độ điều chỉnh** | **Class-level (Cấp độ lớp)**: Mọi mẫu thuộc cùng một lớp đều chịu chung một trọng số tĩnh. | **Instance-level (Cấp độ mẫu)**: Đánh trọng số động theo độ khó và độ tự tin của từng mẫu riêng biệt. |
| **Hành vi với mẫu dễ** | Vẫn phạt nặng các mẫu OFFENSIVE/HATE dễ, dẫn tới mô hình dự đoán tràn lan (tăng False Positives). | **Triệt tiêu tự động** mất mát của mọi mẫu đã học tốt (kể cả mẫu CLEAN lẫn HATE dễ). |
| **Hành vi với mẫu khó** | Không phân biệt được câu chửi thề trực tiếp với câu thù ghét ẩn dụ, châm biếm phức tạp. | **Dồn tối đa gradient** vào các mẫu biên khó phân biệt (thù ghét ngầm, teencode). |
| **Độ nhạy siêu tham số** | Không có siêu tham số phụ (tính từ Train set). | Cần chọn $\gamma$ (thường cố định $\gamma = 2.0$). |

---

## 7. Bài học Kế thừa Trực tiếp cho Đề tài của chúng ta

1. **Khắc phục nhược điểm của Class Weight:** Nếu nhận thấy mô hình PhoBERT kết hợp Class Weight tĩnh dự đoán nhầm quá nhiều câu sạch có chứa từ lóng sang `OFFENSIVE` (làm giảm Precision), việc chuyển sang **Focal Loss ($\gamma=2.0$)** chính là bước nâng cấp thuật toán hoàn hảo nhất đã được chứng minh trong [search/deep_research.md](search/deep_research.md) và Luu et al. (2024).
2. **Cơ sở toán học chuẩn mực:** Toàn bộ công thức tính toán và mã nguồn PyTorch ở trên có thể đưa trực tiếp vào chương Phương pháp nghiên cứu trong báo cáo khoa học của bạn.
