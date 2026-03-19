# 🗑️ Trashnet Image Classification

> Auto-generated report · Last updated: **2026-03-19 12:39 UTC**

Proyek klasifikasi sampah otomatis menggunakan dua arsitektur deep learning — **ResNet50** dan **Swin Transformer** — yang dilatih dan dievaluasi secara penuh via GitHub Actions + Kaggle GPU.

---

## 🏆 Best Model

| | |
|---|---|
| **Model** | Swin Transformer |
| **F1 Score** | `0.9934` |

---

## 📊 Hasil Evaluasi

### ResNet50

> Convolutional Neural Network klasik dengan residual connections. Cepat, stabil, dan proven untuk image classification.

| Metric | Value |
|--------|-------|
| ✅ Accuracy | `0.9830` |
| 🎯 F1 Score | `0.9829` |
| 🔍 Precision | `0.9831` |
| 🔁 Recall | `0.9830` |
| 📈 Best Epoch | `20` |

---

### Swin Transformer

> Vision Transformer berbasis Shifted Window. Unggul dalam menangkap long-range dependency antar patch gambar.

| Metric | Value |
|--------|-------|
| ✅ Accuracy | `0.9934` |
| 🎯 F1 Score | `0.9934` |
| 🔍 Precision | `0.9935` |
| 🔁 Recall | `0.9934` |
| 📈 Best Epoch | `13` |

---

## 📈 Perbandingan Model

| Metric | ResNet50 | Swin Transformer | Winner |
|--------|----------|-----------------|--------|
| Accuracy | `0.9830` | `0.9934` | Swin 🏆 |
| F1 Score | `0.9829` | `0.9934` | Swin 🏆 |
| Precision | `0.9831` | `0.9935` | Swin 🏆 |
| Recall | `0.9830` | `0.9934` | Swin 🏆 |
| Best Epoch | `20` | `13` | - |

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

6 kelas sampah dari dataset [TrashNet](https://github.com/garythung/trashnet):

| Split | Jumlah |
|-------|--------|
| `train/` | ~3.500 gambar |
| `val/` | ~756 gambar |
| `test/` | ~763 gambar |

**Kelas:** `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`

---

## 🤗 Model Links

| Model | HuggingFace |
|-------|-------------|
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

## ⚙️ Setup

```bash
git clone https://github.com/zzazzz/trashnet.git
cd trashnet
pip install -r requirements.txt
python model_training_resnet.py
python model_training_swin.py
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
Trigger Kaggle notebook (GPU T4)
    ├── Training ResNet50      → metrics_resnet.json
    ├── Training Swin          → metrics_swin.json
    ├── Validasi ResNet50      → *_resnet.png
    └── Validasi Swin          → *_swin.png
    │
    ▼
Download output dari Kaggle
    │
    ▼
Copy gambar ke results/
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
