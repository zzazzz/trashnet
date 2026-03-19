import os
import json
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_recall_fscore_support)
from transformers import SwinForImageClassification
from huggingface_hub import snapshot_download
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading model from Hugging Face Hub...")
snapshot_download(repo_id="ziyadazz/trashnet-swin", local_dir="model_hf")

# Load label mappings
with open("model_hf/id2label.json", "r") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

num_classes = len(id2label)
class_names = [id2label[i] for i in range(num_classes)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Swin model
model = SwinForImageClassification.from_pretrained("model_hf/swin_best")
model = model.to(device)
model.eval()

# Transforms (ImageNet normalization)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class TestDataset(Dataset):
    def __init__(self, test_dir, id2label, transform=None):
        self.samples = []
        self.raw_samples = []
        self.transform = transform
        for class_idx, class_name in id2label.items():
            class_folder = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for image_name in os.listdir(class_folder):
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                image_path = os.path.join(class_folder, image_name)
                self.samples.append((image_path, class_idx))
                self.raw_samples.append((image_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

test_dir = "data/test"
test_dataset = TestDataset(test_dir, id2label, transform=val_transforms)
print(f"Total test samples: {len(test_dataset)}")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Inference
true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(pixel_values=images)
        preds = outputs.logits.argmax(dim=1).cpu().numpy()
        pred_labels.extend(preds)
        true_labels.extend(labels.numpy())

true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

# Classification Report
report = classification_report(true_labels, pred_labels, target_names=class_names)
print("Classification Report:")
print(report)

# ── 1. Confusion Matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Swin Transformer")
plt.tight_layout()
plt.savefig("confusion_matrix_swin.png", dpi=150)   # ✅ suffix _swin
plt.close()
print("Confusion matrix saved to confusion_matrix_swin.png")

# ── 2. 5 Sample Per Kelas ────────────────────────────────────────────────────
SAMPLES_PER_CLASS = 5
n_classes = len(class_names)

class_image_paths = {class_name: [] for class_name in class_names}
for image_path, class_idx in test_dataset.raw_samples:
    class_name = id2label[class_idx]
    if len(class_image_paths[class_name]) < SAMPLES_PER_CLASS:
        class_image_paths[class_name].append(image_path)

fig, axes = plt.subplots(
    n_classes, SAMPLES_PER_CLASS,
    figsize=(SAMPLES_PER_CLASS * 3, n_classes * 3)
)
fig.suptitle("5 Sample Per Kelas - Test Set (Swin Transformer)", fontsize=16, fontweight="bold", y=1.01)

for row_idx, class_name in enumerate(class_names):
    paths = class_image_paths[class_name]
    for col_idx in range(SAMPLES_PER_CLASS):
        ax = axes[row_idx][col_idx]
        if col_idx < len(paths):
            img = Image.open(paths[col_idx]).convert("RGB").resize((224, 224))
            ax.imshow(img)
        else:
            ax.imshow(np.ones((224, 224, 3), dtype=np.uint8) * 200)
            ax.text(112, 112, "N/A", ha="center", va="center", fontsize=12, color="gray")
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel(class_name, fontsize=12, fontweight="bold",
                          rotation=0, labelpad=80, va="center")

plt.tight_layout()
plt.savefig("sample_per_class_swin.png", dpi=150, bbox_inches="tight")  # ✅ suffix _swin
plt.close()
print("Sample visualization saved to sample_per_class_swin.png")

# ── 3. Prediksi Benar vs Salah ───────────────────────────────────────────────
fig, axes = plt.subplots(
    n_classes, SAMPLES_PER_CLASS,
    figsize=(SAMPLES_PER_CLASS * 3, n_classes * 3)
)
fig.suptitle("Contoh Prediksi Per Kelas - Swin Transformer (Hijau=Benar, Merah=Salah)",
             fontsize=14, fontweight="bold", y=1.01)

class_results = {class_name: [] for class_name in class_names}
for i, (image_path, true_idx) in enumerate(test_dataset.raw_samples):
    if i >= len(pred_labels):
        break
    class_name = id2label[true_idx]
    pred_name = id2label[pred_labels[i]]
    is_correct = (true_idx == pred_labels[i])
    if len(class_results[class_name]) < SAMPLES_PER_CLASS:
        class_results[class_name].append((image_path, pred_name, is_correct))

for row_idx, class_name in enumerate(class_names):
    results = class_results[class_name]
    for col_idx in range(SAMPLES_PER_CLASS):
        ax = axes[row_idx][col_idx]
        if col_idx < len(results):
            image_path, pred_name, is_correct = results[col_idx]
            img = Image.open(image_path).convert("RGB").resize((224, 224))
            ax.imshow(img)
            color = "green" if is_correct else "red"
            label = f"✓ {pred_name}" if is_correct else f"✗ {pred_name}"
            ax.set_title(label, fontsize=8, color=color, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
        else:
            ax.imshow(np.ones((224, 224, 3), dtype=np.uint8) * 200)
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel(class_name, fontsize=11, fontweight="bold",
                          rotation=0, labelpad=80, va="center")

plt.tight_layout()
plt.savefig("prediction_results_swin.png", dpi=150, bbox_inches="tight")  # ✅ suffix _swin
plt.close()
print("Prediction results saved to prediction_results_swin.png")

# ── 4. Akurasi Per Kelas ─────────────────────────────────────────────────────
per_class_acc = []
for i in range(len(class_names)):
    mask = true_labels == i
    if mask.sum() > 0:
        acc = (pred_labels[mask] == true_labels[mask]).mean()
        per_class_acc.append(acc * 100)
    else:
        per_class_acc.append(0)

colors = ["#2ecc71" if a >= 80 else "#e67e22" if a >= 60 else "#e74c3c"
          for a in per_class_acc]

plt.figure(figsize=(10, 5))
bars = plt.bar(class_names, per_class_acc, color=colors, edgecolor="white", linewidth=1.2)
plt.ylim(0, 110)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Akurasi Per Kelas - Swin Transformer", fontsize=14, fontweight="bold")
plt.xticks(fontsize=11)

for bar, acc in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1.5,
             f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("accuracy_per_class_swin.png", dpi=150)  # ✅ suffix _swin
plt.close()
print("Accuracy per class saved to accuracy_per_class_swin.png")


# ── 5. Grafik Training History ───────────────────────────────────────────────
history_path = "history_swin.json"
if os.path.exists(history_path):
    with open(history_path, "r") as hf:
        history = json.load(hf)

    ep_list = list(range(1, len(history["train_loss"]) + 1))

    # best_epoch sudah terbaca dari metrics di atas
    best_ep = best_epoch

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training History - Swin Transformer", fontsize=16, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(ep_list, history["train_loss"], label="Train Loss", color="#3498db", linewidth=2)
    ax.plot(ep_list, history["val_loss"],   label="Val Loss",   color="#e74c3c", linewidth=2)
    if best_ep and best_ep <= len(ep_list):
        ax.axvline(x=best_ep, color="#2ecc71", linestyle="--", linewidth=1.5, label=f"Best ({best_ep})")
    ax.set_title("Loss per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(ep_list, [v * 100 for v in history["train_acc"]], label="Train Acc", color="#3498db", linewidth=2)
    ax.plot(ep_list, [v * 100 for v in history["val_acc"]],   label="Val Acc",   color="#e74c3c", linewidth=2)
    if best_ep and best_ep <= len(ep_list):
        ax.axvline(x=best_ep, color="#2ecc71", linestyle="--", linewidth=1.5, label=f"Best ({best_ep})")
    ax.set_title("Accuracy per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1
    ax = axes[2]
    ax.plot(ep_list, [v * 100 for v in history["train_f1"]], label="Train F1", color="#3498db", linewidth=2)
    ax.plot(ep_list, [v * 100 for v in history["val_f1"]],   label="Val F1",   color="#e74c3c", linewidth=2)
    if best_ep and best_ep <= len(ep_list):
        ax.axvline(x=best_ep, color="#2ecc71", linestyle="--", linewidth=1.5, label=f"Best ({best_ep})")
    ax.set_title("F1 Score per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history_swin.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Training history chart saved to training_history_swin.png")
else:
    print(f"WARNING: {history_path} not found, skipping training history chart")

# ── 6. Metrics ───────────────────────────────────────────────────────────────
val_acc = accuracy_score(true_labels, pred_labels)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average='weighted'
)

# best_epoch diambil dari metrics training (ditulis training script di sesi yang sama)
try:
    with open("metrics_swin.json", "r") as f:
        train_metrics = json.load(f)
    best_epoch = train_metrics.get("best_epoch", 0)
except FileNotFoundError:
    best_epoch = 0

# Tulis ulang metrics dengan hasil test set + best_epoch dari training
metrics = {
    "val_accuracy": float(val_acc),
    "val_f1": float(val_f1),
    "val_precision": float(val_precision),
    "val_recall": float(val_recall),
    "best_epoch": best_epoch  # dipertahankan dari training
}

with open("metrics_swin.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSemua output tersimpan:")
print("  - confusion_matrix_swin.png")
print("  - sample_per_class_swin.png")
print("  - prediction_results_swin.png")
print("  - accuracy_per_class_swin.png")
print("  - metrics_swin.json")
print("  - training_history_swin.png")
print(f"\nTest Accuracy : {val_acc:.4f}")
print(f"Test F1 Score : {val_f1:.4f}")
print(f"Best Epoch    : {best_epoch}")