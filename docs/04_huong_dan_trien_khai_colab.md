# 04. HƯỚNG DẪN TRIỂN KHAI VÀ HUẤN LUYỆN TRÊN GOOGLE COLAB

Tài liệu này cung cấp hướng dẫn kỹ thuật chi tiết từng bước (Step-by-step) để thiết lập môi trường, nạp dữ liệu ViHSD, tiền xử lý tách từ, huấn luyện mô hình PhoBERT-base kết hợp Class Weight với cơ chế FP16, và xuất bản các sản phẩm đầu ra (`predictions.csv`, model checkpoints) lên Google Drive.

---

## 1. Chuẩn bị Môi trường Google Colab

### 1.1. Kích hoạt GPU Tesla T4
1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Tạo Notebook mới hoặc mở notebook của dự án.
3. Trên thanh công cụ, chọn: **Runtime** $\rightarrow$ **Change runtime type** (Thay đổi loại phần cứng).
4. Tại mục **Hardware accelerator**, chọn **T4 GPU** $\rightarrow$ Nhấn **Save**.
5. Kiểm tra thông số GPU bằng lệnh:
   ```python
   !nvidia-smi
   ```
   *Kết quả xác nhận: GPU hiển thị Tesla T4 với ~15,360 MiB VRAM.*

### 1.2. Kết nối Google Drive để lưu trữ bền vững
Trên môi trường Colab miễn phí, session có thể bị ngắt kết nối bất kỳ lúc nào làm mất toàn bộ dữ liệu trên ổ đĩa tạm `/content`. Do đó, bắt buộc phải mount Google Drive và tạo thư mục lưu checkpoint:

```python
import os
from google.colab import drive

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Định nghĩa thư mục làm việc trên Drive
PROJECT_DIR = '/content/drive/MyDrive/PhoBERT_ViHSD'
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Đã thiết lập thư mục lưu trữ: {PROJECT_DIR}")
```

### 1.3. Cài đặt các thư viện cần thiết
```python
!pip install -q transformers==4.38.2 accelerate==0.28.0 datasets==2.18.0 scikit-learn pyvi seaborn matplotlib
```

---

## 2. Nạp và Chuẩn hóa Bộ dữ liệu ViHSD

Bộ dữ liệu ViHSD có thể được tải trực tiếp từ kho lưu trữ chính thức của tác giả:

```python
import io
import urllib.request
import zipfile
import pandas as pd

# URL dữ liệu gốc dạng tệp zip từ repository tác giả Luu et al. (2021)
ZIP_URL = "https://raw.githubusercontent.com/sonlam1102/vihsd/main/data/vihsd.zip"

req = urllib.request.Request(ZIP_URL, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
        train_df = pd.read_csv(z.open('vihsd/train.csv'))
        dev_df   = pd.read_csv(z.open('vihsd/dev.csv'))
        test_df  = pd.read_csv(z.open('vihsd/test.csv'))

# Kiểm tra và đồng bộ tên cột chuẩn: id, text, label_id
def standardize_df(df):
    # Xác định cột text và label
    text_col = 'free_text' if 'free_text' in df.columns else ('text' if 'text' in df.columns else df.columns[1])
    label_col = 'label_id' if 'label_id' in df.columns else ('label_id' if 'label_id' in df.columns else 'label')
    
    df = df.rename(columns={text_col: 'text', label_col: 'label'})
    if 'id' not in df.columns:
        df['id'] = range(len(df))
    
    # Ép kiểu dữ liệu và lọc bỏ NaN
    df['text'] = df['text'].fillna('').astype(str)
    df['label'] = df['label'].astype(int)
    return df[['id', 'text', 'label']]

train_df = standardize_df(train_df)
dev_df   = standardize_df(dev_df)
test_df  = standardize_df(test_df)

print(f"Kích thước Train: {train_df.shape}, Dev: {dev_df.shape}, Test: {test_df.shape}")
print("Phân bố nhãn tập Train:\n", train_df['label'].value_counts())
```

---

## 3. Tiền xử lý Dữ liệu và Tách từ Tiếng Việt (Word Segmentation)

PhoBERT yêu cầu các từ ghép tiếng Việt phải được nối với nhau bằng dấu gạch dưới `_`. Chúng ta sử dụng thư viện `pyvi` để thực hiện tách từ với tốc độ cao:

```python
import re
from pyvi import ViTokenizer

def preprocess_and_segment(text: str) -> str:
    # 1. Chuẩn hóa khoảng trắng và ký tự xuống dòng
    text = re.sub(r'\s+', ' ', text).strip()
    # 2. Ẩn URLs
    text = re.sub(r'http\S+|www\S+', '[URL]', text)
    # 3. Tách từ tiếng Việt với PyVi
    segmented_text = ViTokenizer.tokenize(text)
    return segmented_text

# Áp dụng cho cả 3 tập dữ liệu
train_df['processed_text'] = train_df['text'].apply(preprocess_and_segment)
dev_df['processed_text']   = dev_df['text'].apply(preprocess_and_segment)
test_df['processed_text']  = test_df['text'].apply(preprocess_and_segment)

print("Ví dụ trước và sau khi tách từ:")
print("Gốc :", train_df['text'].iloc[0])
print("Xử lý:", train_df['processed_text'].iloc[0])
```

---

## 4. Tính toán Class Weight từ Tập Train

Sử dụng `scikit-learn` để tính toán trọng số cân bằng:

```python
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

train_labels = train_df['label'].values
classes = np.unique(train_labels)

# Tính toán balanced class weight: w_c = N / (C * N_c)
calculated_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_labels
)

# Chuyển đổi sang PyTorch Float Tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights_tensor = torch.tensor(calculated_weights, dtype=torch.float).to(device)

print("Các lớp:", classes)
print("Trọng số tính toán:", calculated_weights)
for c, w in zip(classes, calculated_weights):
    label_name = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}[c]
    print(f" - Lớp {c} ({label_name:<9}): {w:.4f}")
```

---

## 5. Xây dựng Dataset và Tokenizer

Sử dụng `AutoTokenizer` với mô hình `vinai/phobert-base-v2` và tham số `max_length = 128`:

```python
from transformers import AutoTokenizer
from torch.utils.data import Dataset

MODEL_NAME = "vinai/phobert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

class ViHSDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = ViHSDDataset(train_df['processed_text'], train_df['label'], tokenizer, max_len=128)
dev_dataset   = ViHSDDataset(dev_df['processed_text'], dev_df['label'], tokenizer, max_len=128)
test_dataset  = ViHSDDataset(test_df['processed_text'], test_df['label'], tokenizer, max_len=128)
```

---

## 6. Xây dựng Custom Trainer với Weighted Cross-Entropy Loss

Để áp dụng Class Weight một cách tự nhiên trong hệ sinh thái Hugging Face Trainer, ta kế thừa `Trainer` và override hàm `compute_loss`:

```python
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_recall_fscore_support

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Hàm tính toán chỉ số đánh giá trong quá trình huấn luyện
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, labels=[0, 1, 2])
    
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "f1_clean": f1[0],
        "f1_offensive": f1[1],
        "f1_hate": f1[2]
    }
```

---

## 7. Cấu hình Huấn luyện (TrainingArguments với FP16)

Thiết lập các tham số huấn luyện theo đúng yêu cầu đề bài:
- `max_length = 128`
- `batch_size = 16`
- `epochs = 3` hoặc `4`
- Bật `fp16 = True` để tăng tốc trên GPU T4.

```python
# Cấu hình siêu tham số
training_args = TrainingArguments(
    output_dir=os.path.join(CHECKPOINT_DIR, "phobert_class_weight"),
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=True,                       # Kích hoạt FP16 trên Colab GPU T4
    logging_steps=100,
    save_total_limit=2,
    seed=42,
    report_to="none"
)

# Khởi tạo mô hình PhoBERT cho 3 nhãn
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

# Khởi tạo Trainer với Class Weight
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights_tensor  # Truyền trọng số đã tính
)

# Bắt đầu quá trình huấn luyện
print("Bắt đầu huấn luyện mô hình PhoBERT-base với Class Weight...")
trainer.train()
```

---

## 8. Đánh giá trên Tập Test & Trực quan hóa Ma trận Nhầm lẫn

Sau khi huấn luyện, tiến hành suy luận trên tập Test:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Dự đoán trên tập Test
predictions_output = trainer.predict(test_dataset)
test_logits = predictions_output.predictions
test_preds = np.argmax(test_logits, axis=-1)
test_labels = test_df['label'].values

# 1. Báo cáo định lượng chi tiết
target_names = ["CLEAN (0)", "OFFENSIVE (1)", "HATE (2)"]
print("=== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ===")
print(classification_report(test_labels, test_preds, target_names=target_names, digits=4))

# 2. Vẽ Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label (Nhãn dự đoán)')
plt.ylabel('True Label (Nhãn thực tế)')
plt.title('Confusion Matrix - PhoBERT với Class Weight')
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_class_weight.png")
plt.savefig(cm_path, bbox_inches='tight', dpi=300)
plt.show()
print(f"Đã lưu biểu đồ ma trận nhầm lẫn tại: {cm_path}")
```

---

## 9. Tạo và Lưu trữ Tệp `predictions.csv` vào Google Drive

Yêu cầu tệp dự đoán chuẩn gồm chính xác 4 cột: `id`, `text`, `ground_truth`, `predicted_label`:

```python
# Tạo DataFrame kết quả dự đoán
predictions_df = pd.DataFrame({
    'id': test_df['id'],
    'text': test_df['text'],
    'ground_truth': test_labels,
    'predicted_label': test_preds
})

# Đường dẫn lưu trữ trên Google Drive
predictions_file_path = os.path.join(OUTPUT_DIR, "predictions.csv")
predictions_df.to_csv(predictions_file_path, index=False, encoding='utf-8')

print(f"Đã xuất tệp dự đoán thành công: {predictions_file_path}")
print("Xem trước 5 dòng đầu tiên:")
print(predictions_df.head())
```

---

## 10. Lưu trữ Checkpoint Mô hình Hoàn chỉnh

```python
FINAL_MODEL_DIR = os.path.join(PROJECT_DIR, "final_phobert_class_weight_model")
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"Mô hình và Tokenizer đã được lưu trữ bền vững tại: {FINAL_MODEL_DIR}")
```
