import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup
from huggingface_hub import HfApi
from PIL import Image
import numpy as np
import json
import os

# Initialize wandb
wandb.login()
wandb.init(project="trashnet-classification", entity="ziyad-azzufari")

config = wandb.config
config.learning_rate = 1e-4
config.batch_size = 32
config.epochs = 50        # naikkan epoch, biar early stopping yang handle
config.patience = 5       # stop kalau 5 epoch tidak improve

# Load dataset
data_dir = "/kaggle/input/datasets/ziyadmuhammad/trashnet-data"
ds = load_dataset("imagefolder", data_dir=data_dir)

labels = ds["train"].features["label"].names
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
num_classes = len(labels)

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Custom Dataset
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

# Load pretrained ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
total_steps = len(train_loader) * config.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training loop
best_val_f1 = 0.0
patience_counter = 0
os.makedirs("model", exist_ok=True)

for epoch in range(config.epochs):
    # Train
    model.train()
    train_loss = 0.0
    all_preds, all_labels = [], []

    for images, batch_labels in train_loader:
        images, batch_labels = images.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    _, _, train_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # Validation
    model.eval()
    val_loss = 0.0
    val_preds, val_labels_list = [], []

    with torch.no_grad():
        for images, batch_labels in val_loader:
            images, batch_labels = images.to(device), batch_labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
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

    # Save best model & early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), "model/resnet50_best.pth")
        print(f"  -> Best model saved (val_f1={val_f1:.4f})")
    else:
        patience_counter += 1
        print(f"  -> No improvement ({patience_counter}/{config.patience})")
        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best val_f1: {best_val_f1:.4f}")
            break

# Save label mappings
with open("model/label2id.json", "w") as f:
    json.dump(label2id, f)
with open("model/id2label.json", "w") as f:
    json.dump(id2label, f)

print("Training complete. Uploading to Hugging Face Hub...")

# Upload to Hugging Face Hub
api = HfApi()
repo_id = "ziyadazz/trashnet-resnet50"

api.create_repo(repo_id=repo_id, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    folder_path="model",
    commit_message="Upload ResNet50 trashnet model"
)

print(f"Model uploaded to: https://huggingface.co/{repo_id}")
wandb.finish()