import os
import subprocess

subprocess.run(["pip", "install", "-q", "wandb", "huggingface_hub",
                "datasets", "transformers", "torchvision", "scikit-learn",
                "seaborn", "matplotlib"], check=True)

import wandb
from huggingface_hub import login

wandb.login(key="WANDB_KEY_PLACEHOLDER")
login(token="HF_KEY_PLACEHOLDER")

import shutil

# Auto-detect path script
script_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'model_training.py' in files:
        script_path = root
        break

print(f"Scripts found at: {script_path}")
shutil.copy(f"{script_path}/model_training.py", "model_training.py")
shutil.copy(f"{script_path}/requirements.txt", "requirements.txt")
shutil.copy(f"{script_path}/validate_model.py", "validate_model.py")

# Auto-detect path data
data_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train' in os.listdir(root) and 'test' in os.listdir(root):
        data_path = root
        break

print(f"Data found at: {data_path}")
shutil.copytree(data_path, "data")

# Jalankan training
print("=== Starting Training ===")
exec(open("model_training.py").read())

# Jalankan validasi
print("=== Starting Validation ===")
exec(open("validate_model.py").read())