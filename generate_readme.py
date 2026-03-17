import json
import os
from datetime import datetime

# Load metrics
with open("kaggle_output/metrics_resnet.json", "r") as f:
    metrics_resnet = json.load(f)

with open("kaggle_output/metrics_swin.json", "r") as f:
    metrics_swin = json.load(f)

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# Tentukan model terbaik
best_model = "ResNet50" if metrics_resnet['val_f1'] > metrics_swin['val_f1'] else "Swin Transformer"
best_f1 = max(metrics_resnet['val_f1'], metrics_swin['val_f1'])

readme = f"""# 🗑️ Trashnet Image Classification

> Auto-generated report · Last updated: **{now}**

Proyek klasifikasi sampah menggunakan dua arsitektur deep learning yang dilatih dan dievaluasi secara otomatis via GitHub Actions + Kaggle GPU.

---

## 🏆 Best Model

| | |
|---|---|
| **Model** | {best_model} |
| **F1 Score** | {best_f1:.4f} |

---

## 📊 Hasil Training

### ResNet50

| Metric | Value |
|--------|-------|
| ✅ Val Accuracy | `{metrics_resnet['val_accuracy']:.4f}` |
| 🎯 Val F1 Score | `{metrics_resnet['val_f1']:.4f}` |
| 🔍 Val Precision | `{metrics_resnet['val_precision']:.4f}` |
| 🔁 Val Recall | `{metrics_resnet['val_recall']:.4f}` |
| 📈 Best Epoch | `{metrics_resnet['best_epoch']}` |

### Swin Transformer

| Metric | Value |
|--------|-------|
| ✅ Val Accuracy | `{metrics_swin['val_accuracy']:.4f}` |
| 🎯 Val F1 Score | `{metrics_swin['val_f1']:.4f}` |
| 🔍 Val Precision | `{metrics_swin['val_precision']:.4f}` |
| 🔁 Val Recall | `{metrics_swin['val_recall']:.4f}` |
| 📈 Best Epoch | `{metrics_swin['best_epoch']}` |

---

## 📈 Perbandingan Model

| Metric | ResNet50 | Swin Transformer | Winner |
|--------|----------|-----------------|--------|
| Accuracy | `{metrics_resnet['val_accuracy']:.4f}` | `{metrics_swin['val_accuracy']:.4f}` | {"ResNet50 🏆" if metrics_resnet['val_accuracy'] > metrics_swin['val_accuracy'] else "Swin 🏆"} |
| F1 Score | `{metrics_resnet['val_f1']:.4f}` | `{metrics_swin['val_f1']:.4f}` | {"ResNet50 🏆" if metrics_resnet['val_f1'] > metrics_swin['val_f1'] else "Swin 🏆"} |
| Precision | `{metrics_resnet['val_precision']:.4f}` | `{metrics_swin['val_precision']:.4f}` | {"ResNet50 🏆" if metrics_resnet['val_precision'] > metrics_swin['val_precision'] else "Swin 🏆"} |
| Recall | `{metrics_resnet['val_recall']:.4f}` | `{metrics_swin['val_recall']:.4f}` | {"ResNet50 🏆" if metrics_resnet['val_recall'] > metrics_swin['val_recall'] else "Swin 🏆"} |

---

## 🖼️ Visualisasi

### Confusion Matrix

| ResNet50 | Swin Transformer |
|----------|-----------------|
| ![ResNet50 Confusion Matrix](results/confusion_matrix_resnet.png) | ![Swin Confusion Matrix](results/confusion_matrix_swin.png) |

### Akurasi Per Kelas

| ResNet50 | Swin Transformer |
|----------|-----------------|
| ![ResNet50 Accuracy](results/accuracy_per_class_resnet.png) | ![Swin Accuracy](results/accuracy_per_class_swin.png) |

### Sample Per Kelas

| ResNet50 | Swin Transformer |
|----------|-----------------|
| ![ResNet50 Samples](results/sample_per_class_resnet.png) | ![Swin Samples](results/sample_per_class_swin.png) |

### Prediksi Benar vs Salah

| ResNet50 | Swin Transformer |
|----------|-----------------|
| ![ResNet50 Predictions](results/prediction_results_resnet.png) | ![Swin Predictions](results/prediction_results_swin.png) |

---

## 🗂️ Dataset

| Split | Kelas |
|-------|-------|
| `train/` | cardboard, glass, metal, paper, plastic, trash |
| `val/` | cardboard, glass, metal, paper, plastic, trash |
| `test/` | cardboard, glass, metal, paper, plastic, trash |

---

## 🤗 Model Links

| Model | Link |
|-------|------|
| ResNet50 | [ziyadazz/trashnet-resnet50](https://huggingface.co/ziyadazz/trashnet-resnet50) |
| Swin Transformer | [ziyadazz/trashnet-swin](https://huggingface.co/ziyadazz/trashnet-swin) |

---

## 📁 Project Structure
```
.
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── kaggle_notebook/
│   ├── kernel.py
│   └── kernel-metadata.json
├── results/
│   ├── confusion_matrix_resnet.png
│   ├── confusion_matrix_swin.png
│   ├── sample_per_class_resnet.png
│   ├── sample_per_class_swin.png
│   ├── prediction_results_resnet.png
│   ├── prediction_results_swin.png
│   ├── accuracy_per_class_resnet.png
│   └── accuracy_per_class_swin.png
├── model_training_resnet.py
├── model_training_swin.py
├── validate_model_resnet.py
├── validate_model_swin.py
├── generate_readme.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

1. **Clone repository**:
```bash
   git clone https://github.com/zzazzz/trashnet.git
   cd trashnet
```

2. **Install dependencies**:
```bash
   pip install -r requirements.txt
```

3. **Jalankan training**:
```bash
   python model_training_resnet.py
   python model_training_swin.py
```

4. **Jalankan validasi**:
```bash
   python validate_model_resnet.py
   python validate_model_swin.py
```

---

## 🔄 CI/CD Pipeline

Pipeline otomatis berjalan setiap push ke `main`:
```
Push ke main
    │
    ▼
Push script ke Kaggle dataset
    │
    ▼
Tunggu dataset ready
    │
    ▼
Trigger Kaggle notebook (GPU)
    ├── Training ResNet50
    ├── Training Swin Transformer
    ├── Validasi ResNet50
    └── Validasi Swin Transformer
    │
    ▼
Download output dari Kaggle
    │
    ▼
Generate README otomatis
    │
    ▼
Commit results + README ke GitHub
```

---

## 📦 Requirements
```
torch
torchvision
transformers
datasets
wandb
huggingface_hub
scikit-learn
seaborn
matplotlib
Pillow
```
"""

os.makedirs("results", exist_ok=True)

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)

print("README.md generated successfully!")
print(f"  Best model: {best_model} (F1: {best_f1:.4f})")
print(f"  ResNet50  — Acc: {metrics_resnet['val_accuracy']:.4f} | F1: {metrics_resnet['val_f1']:.4f}")
print(f"  Swin      — Acc: {metrics_swin['val_accuracy']:.4f} | F1: {metrics_swin['val_f1']:.4f}")