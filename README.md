# 🗑️ Trashnet Image Classification

> Auto-generated report · Last updated: **2026-03-19 06:56 UTC**

Proyek klasifikasi sampah menggunakan dua arsitektur deep learning yang dilatih dan dievaluasi secara otomatis via GitHub Actions + Kaggle GPU.

---

## 🏆 Best Model

| | |
|---|---|
| **Model** | Swin Transformer |
| **F1 Score** | 0.9894 |

---

## 📊 Hasil Training

### ResNet50

| Metric | Value |
|--------|-------|
| ✅ Val Accuracy | `0.9725` |
| 🎯 Val F1 Score | `0.9723` |
| 🔍 Val Precision | `0.9728` |
| 🔁 Val Recall | `0.9725` |
| 📈 Best Epoch | `0` |

### Swin Transformer

| Metric | Value |
|--------|-------|
| ✅ Val Accuracy | `0.9895` |
| 🎯 Val F1 Score | `0.9894` |
| 🔍 Val Precision | `0.9896` |
| 🔁 Val Recall | `0.9895` |
| 📈 Best Epoch | `0` |

---

## 📈 Perbandingan Model

| Metric | ResNet50 | Swin Transformer | Winner |
|--------|----------|-----------------|--------|
| Accuracy | `0.9725` | `0.9895` | Swin 🏆 |
| F1 Score | `0.9723` | `0.9894` | Swin 🏆 |
| Precision | `0.9728` | `0.9896` | Swin 🏆 |
| Recall | `0.9725` | `0.9895` | Swin 🏆 |

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
