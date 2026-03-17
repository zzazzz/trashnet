import os
import subprocess
import wandb
from huggingface_hub import login
import shutil

# Path dataset Kaggle (BENAR)
DATASET_PATH = "/kaggle/input/trashnet-data"
SCRIPT_PATH = "/kaggle/input/trashnet-training-script"

print("Dataset path:", DATASET_PATH)
print("Script path:", SCRIPT_PATH)

# Install dependencies
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

# Login menggunakan environment variable (AMAN)
wandb.login(key=os.environ["WANDB_API_KEY"])
login(token=os.environ["HF_API_KEY"])

# Copy scripts dari dataset ke working directory
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
        raise FileNotFoundError(f"{src} not found")

print("\nAll scripts copied successfully\n")

# =========================
# TRAINING
# =========================

print("\n========== TRAINING RESNET50 ==========")
exec(open("model_training_resnet.py").read())

print("\n========== TRAINING SWIN TRANSFORMER ==========")
exec(open("model_training_swin.py").read())

# =========================
# VALIDATION
# =========================

print("\n========== VALIDASI RESNET50 ==========")
exec(open("validate_model_resnet.py").read())

print("\n========== VALIDASI SWIN ==========")
exec(open("validate_model_swin.py").read())

print("\nTraining pipeline completed.")