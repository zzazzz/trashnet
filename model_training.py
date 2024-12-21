import wandb
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, GaussianBlur, RandomAdjustSharpness, RandomEqualize, ToTensor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup
from transformers import Swinv2ForImageClassification, AutoImageProcessor, TrainingArguments, Trainer
from transformers.integrations import WandbCallback

# Initialize wandb
wandb.login()
wandb.init(project="trashnet-classification", entity="ziyad-azzufari")

# Membuat konfigurasi untuk eksperimen
config = wandb.config
config.learning_rate = 1e-4
config.batch_size = 32
config.epochs = 20

# Load the dataset
data_dir = "data"
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

# Memuat model SwinV2 dari Hugging Face
model_name = "microsoft/swinv2-tiny-patch4-window8-256"
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Load the SwinV2 model
model = Swinv2ForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Optimizer dan Scheduler
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Tentukan argumen pelatihan
training_args = TrainingArguments(
    output_dir="swinv2_output",
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=10,
    eval_steps=100,
    learning_rate=config.learning_rate,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    num_train_epochs=config.epochs,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    report_to="wandb"  # Menggunakan Wandb untuk melacak metrik
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

total_steps = len(train_ds) // training_args.per_device_train_batch_size * training_args.num_train_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Membuat Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    tokenizer=image_processor,
    optimizers=(optimizer, scheduler),
    callbacks=[WandbCallback()]  # Menambahkan callback Wandb untuk pelatihan
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("model")
image_processor.save_pretrained('processor')
wandb.finish()
