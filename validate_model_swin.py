import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from transformers import SwinForImageClassification, SwinImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# 🔥 OPTIONAL: clear cache biar ga pakai config lama
import shutil
shutil.rmtree("/root/.cache/huggingface", ignore_errors=True)

print("Loading model from Hugging Face Hub...")
snapshot_download(repo_id="ziyadazz/trashnet-swin", local_dir="model_hf")

# Load label mappings
with open("model_hf/id2label.json", "r") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

num_classes = len(id2label)
class_names = [id2label[i] for i in range(num_classes)]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = SwinForImageClassification.from_pretrained("model_hf/swin_best")
model = model.to(device)
model.eval()

# ✅ FIX: load processor (WAJIB)
processor = SwinImageProcessor.from_pretrained("model_hf/swin_best")

# Custom Dataset
class TestDataset(Dataset):
    def __init__(self, test_dir, id2label, processor):
        self.samples = []
        self.raw_samples = []
        self.processor = processor

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

        # ✅ pakai processor (bukan transform manual)
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return pixel_values, label


# Load test dataset
test_dir = "data/test"
test_dataset = TestDataset(test_dir, id2label, processor=processor)
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
print("\nClassification Report:\n")
print(report)

# ── 1. Confusion Matrix ─────────────────────────
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Swin Transformer")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("Confusion matrix saved")

# ── 2. Sample per class ─────────────────────────
SAMPLES_PER_CLASS = 5
n_classes = len(class_names)

class_image_paths = {class_name: [] for class_name in class_names}

for image_path, class_idx in test_dataset.raw_samples:
    class_name = id2label[class_idx]
    if len(class_image_paths[class_name]) < SAMPLES_PER_CLASS:
        class_image_paths[class_name].append(image_path)

fig, axes = plt.subplots(n_classes, SAMPLES_PER_CLASS,
                         figsize=(SAMPLES_PER_CLASS * 3, n_classes * 3))

for row_idx, class_name in enumerate(class_names):
    paths = class_image_paths[class_name]
    for col_idx in range(SAMPLES_PER_CLASS):
        ax = axes[row_idx][col_idx]

        if col_idx < len(paths):
            img = Image.open(paths[col_idx]).convert("RGB").resize((224, 224))
            ax.imshow(img)
        else:
            ax.imshow(np.ones((224, 224, 3), dtype=np.uint8) * 200)

        ax.axis("off")

        if col_idx == 0:
            ax.set_ylabel(class_name, rotation=0, labelpad=60)

plt.tight_layout()
plt.savefig("sample_per_class.png", dpi=150)
plt.close()

print("Sample visualization saved")

# ── 3. Accuracy per class ───────────────────────
per_class_acc = []

for i in range(len(class_names)):
    mask = true_labels == i
    if mask.sum() > 0:
        acc = (pred_labels[mask] == true_labels[mask]).mean()
        per_class_acc.append(acc * 100)
    else:
        per_class_acc.append(0)

plt.figure(figsize=(10, 5))
plt.bar(class_names, per_class_acc)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy per Class")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("accuracy_per_class.png", dpi=150)
plt.close()

print("Accuracy chart saved")

# ── 4. Metrics ──────────────────────────────────
val_acc = accuracy_score(true_labels, pred_labels)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average='weighted'
)

metrics = {
    "val_accuracy": float(val_acc),
    "val_f1": float(val_f1),
    "val_precision": float(val_precision),
    "val_recall": float(val_recall),
    "best_epoch": 0
}

# ✅ FIX nama file
with open("metrics_swin.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nMetrics saved to metrics_swin.json")

print("\nDONE ✅")