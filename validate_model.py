import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import Swinv2ForImageClassification, AutoImageProcessor

# Load model and processor
model = "swin"
processor = 'processor'
model = Swinv2ForImageClassification.from_pretrained(model)
image_processor = AutoImageProcessor.from_pretrained(processor)

# Fungsi untuk memprediksi satu gambar
def predict_single_image(image):
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx

# Direktori dataset test
# Membuat id2label dan label2id berdasarkan struktur folder
test_dir = 'data/test'  # Direktori dataset test
class_names = sorted(os.listdir(test_dir))  # Mengurutkan nama folder
id2label = {idx: name for idx, name in enumerate(class_names)}
label2id = {name: idx for idx, name in id2label.items()}

true_labels = []
pred_labels = []

# Iterasi semua subfolder dalam direktori test
for class_idx, class_name in id2label.items():  # id2label harus didefinisikan sebagai {0: "cardboard", 1: "glass", ...}
    class_folder = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_folder):
        continue  # Lewati jika bukan folder

    for image_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_name)
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Predict label
            predicted_label = predict_single_image(image)
            
            # Simpan true dan predicted label
            true_labels.append(class_idx)
            pred_labels.append(predicted_label)
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Konversi ke numpy array
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)

# Classification Report
report = classification_report(true_labels, pred_labels, target_names=id2label.values())
print("Classification Report:")
print(report)

# Plot Confusion Matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Plot confusion matrix
plot_confusion_matrix(cm, list(id2label.values()))