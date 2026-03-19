import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import SwinForImageClassification, SwinConfig, SwinImageProcessor
from huggingface_hub import HfApi
from PIL import Image
import numpy as np
import json
import os

wandb.login()
wandb.init(project="trashnet-classification", entity="ziyad-azzufari")

config = wandb.config
config.learning_rate = 1e-4
config.batch_size = 32
config.epochs = 50
config.patience = 5

data_dir = "data"
ds = load_dataset("imagefolder", data_dir=data_dir)

labels = ds["train"].features["label"].names
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
num_classes = len(labels)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

class TrashDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = TrashDataset(ds["train"], transform=train_transforms)
val_dataset = TrashDataset(ds["validation"], transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

# ✅ Fix: load via SwinConfig dulu, lalu from_pretrained dengan config
# Ini menghindari auto feature extractor loading di transformers versi baru
# Load image_processor dari microsoft supaya preprocessor_config.json yang
# ter-upload ke HF selalu benar dan linked ke microsoft Swin
image_processor = SwinImageProcessor(
    do_resize=True,
    size={"height": 224, "width": 224},
    resample=3,
    do_rescale=True,
    rescale_factor=1/255,
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
)

swin_config = SwinConfig.from_pretrained("microsoft/swin-base-patch4-window7-224")
swin_config.num_labels = num_classes
swin_config.id2label = id2label
swin_config.label2id = label2id

model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-base-patch4-window7-224",
    config=swin_config,
    ignore_mismatched_sizes=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
total_steps = len(train_loader) * config.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

best_val_f1 = 0.0
patience_counter = 0
best_epoch = 0
os.makedirs("model_swin", exist_ok=True)

for epoch in range(config.epochs):
    model.train()
    train_loss = 0.0
    all_preds, all_labels = [], []

    for images, batch_labels in train_loader:
        images, batch_labels = images.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, batch_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        preds = outputs.logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    _, _, train_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    model.eval()
    val_loss = 0.0
    val_preds, val_labels_list = [], []

    with torch.no_grad():
        for images, batch_labels in val_loader:
            images, batch_labels = images.to(device), batch_labels.to(device)
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, batch_labels)
            val_loss += loss.item()
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels_list.extend(batch_labels.cpu().numpy())

    val_acc = accuracy_score(val_labels_list, val_preds)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels_list, val_preds, average='weighted')

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss / len(train_loader),
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "val_loss": val_loss / len(val_loader),
        "val_accuracy": val_acc,
        "val_f1": val_f1,
    })

    print(f"Epoch {epoch+1}/{config.epochs} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1
        patience_counter = 0
        model.save_pretrained("model_swin/swin_best")
        image_processor.save_pretrained("model_swin/swin_best")
        print(f"  -> Best model saved (val_f1={val_f1:.4f})")
    else:
        patience_counter += 1
        print(f"  -> No improvement ({patience_counter}/{config.patience})")
        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best val_f1: {best_val_f1:.4f}")
            break

with open("model_swin/label2id.json", "w") as f:
    json.dump(label2id, f)
with open("model_swin/id2label.json", "w") as f:
    json.dump(id2label, f)

metrics = {
    "val_accuracy": float(val_acc),
    "val_f1": float(val_f1),
    "val_precision": float(val_precision),
    "val_recall": float(val_recall),
    "best_epoch": best_epoch
}

with open("metrics_swin.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved to metrics_swin.json")
print("Training complete. Uploading to Hugging Face Hub...")

api = HfApi()
repo_id = "ziyadazz/trashnet-swin"
api.create_repo(repo_id=repo_id, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    folder_path="model_swin",
    commit_message="Upload Swin Transformer trashnet model"
)

print(f"Model uploaded to: https://huggingface.co/{repo_id}")
wandb.finish()