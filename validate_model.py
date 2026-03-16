import os
import json
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
from huggingface_hub import snapshot_download
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading model from Hugging Face Hub...")
snapshot_download(repo_id="ziyadazz/trashnet-resnet50", local_dir="model_hf")

# Load label mappings
with open("model_hf/id2label.json", "r") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

num_classes = len(id2label)
class_names = [id2label[i] for i in range(num_classes)]

# Load ResNet50 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("model_hf/resnet50_best.pth", map_location=device))
model = model.to(device)
model.eval()

# Transforms
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
        self.transform = transform
        for class_idx, class_name in id2label.items():
            class_folder = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                self.samples.append((image_path, class_idx))

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
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Inference
true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        pred_labels.extend(preds)
        true_labels.extend(labels.numpy())

true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

# Classification Report
report = classification_report(true_labels, pred_labels, target_names=class_names)
print("Classification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - ResNet50')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved to confusion_matrix.png")