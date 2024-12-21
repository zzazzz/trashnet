import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import Swinv2ForImageClassification, AutoImageProcessor
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, GaussianBlur, RandomAdjustSharpness, RandomEqualize, ToTensor

# Load model and processor
model = "model"
model = Swinv2ForImageClassification.from_pretrained(model)
image_processor = AutoImageProcessor.from_pretrained(model)

# Load the dataset
data_dir = "data"
ds = load_dataset("imagefolder", data_dir=data_dir)

# Preprocessing
_transforms = Compose([
    Resize((200, 200)),
    GaussianBlur(kernel_size=(1, 5)),
    RandomAdjustSharpness(sharpness_factor=2),
    RandomEqualize(),
    ToTensor()
])

def preprocess_test(example):
    example["pixel_values"] = _transforms(example["image"].convert("RGB"))
    return example

test_ds = ds["test"].map(preprocess_test, remove_columns=["image"])

# Function for predictions
def predict(model, image):
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx

# Evaluate the model
def evaluate_model(test_ds):
    true_labels = []
    pred_labels = []

    for sample in test_ds:
        pixel_values = sample['pixel_values']
        true_label = sample['label']
        
        # Get prediction for the image
        predicted_label = predict(model, pixel_values)
        
        true_labels.append(true_label)
        pred_labels.append(predicted_label)
    
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)

    return cm, report

# Evaluate and plot the confusion matrix
cm, report = evaluate_model(test_ds)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("Classification Report:\n", report)
