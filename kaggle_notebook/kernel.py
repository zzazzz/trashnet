import os
import subprocess
import shutil
import wandb
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

# =========================
# Dataset paths
# =========================

DATASET_PATH = "/kaggle/input/trashnet-data"
SCRIPT_PATH = "/kaggle/input/trashnet-training-script"

print("Dataset path:", DATASET_PATH)
print("Script path:", SCRIPT_PATH)

# =========================
# Install dependencies
# =========================

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

# =========================
# Load Kaggle Secrets
# =========================

user_secrets = UserSecretsClient()

try:
    wandb_key = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=wandb_key)
    print("W&B login success")
except:
    print("WANDB_API_KEY not found, skipping W&B login")

try:
    hf_key = user_secrets.get_secret("HF_API_KEY")
    login(token=hf_key)
    print("HuggingFace login success")
except:
    print("HF_API_KEY not found, skipping HF login")

# =========================
# Copy training scripts
# =========================

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

print("\nAll scripts copied successfully")

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

print("\n========== VALIDASI RESNET ==========")
exec(open("validate_model_resnet.py").read())

print("\n========== VALIDASI SWIN ==========")
exec(open("validate_model_swin.py").read())

print("\nTraining pipeline completed.")