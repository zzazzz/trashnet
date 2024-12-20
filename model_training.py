import wandb
import torch
from transformers import Swinv2ForImageClassification, Swinv2ImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, GaussianBlur, RandomAdjustSharpness, RandomEqualize, ToTensor
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Initialize wandb
wandb.login()
wandb.init(project="trashnet_classification", entity="ziyad-azzufari")

# Load the dataset
data_dir = "trashnet_split"
ds = load_dataset("imagefolder", data_dir=data_dir)

labels = ds["train"].features["label"].names
label2id, id2label = dict(), dict()

for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# Preprocessing images
_transforms = Compose([
    Resize((200, 200)),
    GaussianBlur(kernel_size=(1, 5)),
    RandomAdjustSharpness(sharpness_factor=2),
    RandomEqualize(),
    ToTensor()
])

# Preprocess images for the dataset
def preprocess_train(example):
    example["pixel_values"] = _transforms(example["image"].convert("RGB"))
    return example

train_ds = ds["train"].map(preprocess_train, remove_columns=["image"])
val_ds = ds["validation"].map(preprocess_train, remove_columns=["image"])

# Load the SwinV2 model
model = Swinv2ForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window7-224",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
model.to("cuda")

# Create Trainer
training_args = TrainingArguments(
    output_dir="swinv2_output",
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=10,
    eval_steps=100,
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    report_to="wandb"
)

# Define metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("model")
wandb.finish()
