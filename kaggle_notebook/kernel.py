import os
import subprocess
import wandb
from huggingface_hub import login
import shutil

DATASET_PATH = "/kaggle/input/trashnet-data"
SCRIPT_PATH = "/kaggle/input/trashnet-training-script"

print("Dataset path:", DATASET_PATH)
print("Script path:", SCRIPT_PATH)

# ✅ Fix: symlink supaya data_dir="data" di training script bisa resolve dengan benar
if not os.path.exists("data"):
    os.symlink(DATASET_PATH, "data")
    print("Symlink created: ./data ->", DATASET_PATH)
else:
    print("./data already exists, skipping symlink")

subprocess.run([
    "pip","install","-q",
    "wandb",
    "huggingface_hub",
    "datasets",
    "transformers",
    "torchvision",
    "scikit-learn",
    "seaborn",
    "matplotlib"
], check=True)

wandb.login(key="WANDB_KEY_PLACEHOLDER")
login(token="HF_KEY_PLACEHOLDER")

scripts = [
    "model_training_resnet.py",
    "model_training_swin.py",
    "validate_model_resnet.py",
    "validate_model_swin.py",
    "requirements.txt"
]

for s in scripts:
    src = f"{SCRIPT_PATH}/{s}"
    if os.path.exists(src):
        shutil.copy(src, s)
        print(f"Copied {s}")
    else:
        raise FileNotFoundError(src)

print("\nTraining ResNet50")
exec(open("model_training_resnet.py").read())

print("\nTraining Swin Transformer")
exec(open("model_training_swin.py").read())

print("\nValidasi ResNet")
exec(open("validate_model_resnet.py").read())

print("\nValidasi Swin")
exec(open("validate_model_swin.py").read())

print("Training pipeline completed.")