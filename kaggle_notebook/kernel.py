import os
import subprocess

subprocess.run(["pip", "install", "-q", "wandb", "huggingface_hub",
                "datasets", "transformers", "torchvision",
                "scikit-learn", "seaborn", "matplotlib"], check=True)

import wandb
from huggingface_hub import login

wandb.login(key="WANDB_KEY_PLACEHOLDER")
login(token="HF_KEY_PLACEHOLDER")

import shutil

print("Input directory:", os.listdir("/kaggle/input/"))
print("trashnet-data contents:", os.listdir("/kaggle/input/trashnet-data/"))

shutil.copy("/kaggle/input/trashnet-training-script/model_training_resnet.py", "model_training_resnet.py")
shutil.copy("/kaggle/input/trashnet-training-script/model_training_swin.py", "model_training_swin.py")
shutil.copy("/kaggle/input/trashnet-training-script/validate_model_resnet.py", "validate_model_resnet.py")
shutil.copy("/kaggle/input/trashnet-training-script/validate_model_swin.py", "validate_model_swin.py")
shutil.copy("/kaggle/input/trashnet-training-script/requirements.txt", "requirements.txt")
shutil.copytree("/kaggle/input/trashnet-data/data", "data")

print("data/ contents:", os.listdir("data/"))

# Training ResNet50
print("\n========== TRAINING RESNET50 ==========")
exec(open("model_training_resnet.py").read())

# Training Swin Transformer
print("\n========== TRAINING SWIN TRANSFORMER ==========")
exec(open("model_training_swin.py").read())

# Validasi ResNet50
print("\n========== VALIDASI RESNET50 ==========")
exec(open("validate_model_resnet.py").read())

# Validasi Swin Transformer
print("\n========== VALIDASI SWIN TRANSFORMER ==========")
exec(open("validate_model_swin.py").read())