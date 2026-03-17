import os
import subprocess
import wandb
from huggingface_hub import login
import shutil

DATASET_PATH = "/kaggle/input/datasets/ziyadmuhammad/trashnet-data"
SCRIPT_PATH = "/kaggle/input/datasets/ziyadmuhammad/trashnet-training-script"

subprocess.run([
    "pip","install","-q",
    "wandb","huggingface_hub",
    "datasets","transformers",
    "torchvision","scikit-learn",
    "seaborn","matplotlib"
], check=True)

wandb.login(key="WANDB_KEY_PLACEHOLDER")
login(token="HF_KEY_PLACEHOLDER")

# Copy scripts ke working directory
scripts = [
    "model_training_resnet.py",
    "model_training_swin.py",
    "validate_model_resnet.py",
    "validate_model_swin.py",
    "requirements.txt"
]

for s in scripts:
    shutil.copy(f"{SCRIPT_PATH}/{s}", s)

print("Dataset path:", DATASET_PATH)

# Training ResNet
print("\n========== TRAINING RESNET50 ==========")
exec(open("model_training_resnet.py").read())

# Training Swin
print("\n========== TRAINING SWIN TRANSFORMER ==========")
exec(open("model_training_swin.py").read())

# Validation
print("\n========== VALIDASI RESNET50 ==========")
exec(open("validate_model_resnet.py").read())

print("\n========== VALIDASI SWIN ==========")
exec(open("validate_model_swin.py").read())