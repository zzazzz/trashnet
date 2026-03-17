import os
import subprocess

subprocess.run(["pip", "install", "-q", "wandb", "huggingface_hub",
                "datasets", "transformers", "torchvision",
                "scikit-learn", "seaborn", "matplotlib"], check=True)

import wandb
from huggingface_hub import login
import shutil
from kaggle_secrets import UserSecretsClient

# Ambil secret dari Kaggle Secrets
secrets = UserSecretsClient()
wandb_key = secrets.get_secret("WANDB_API_KEY")
hf_key = secrets.get_secret("HF_API_KEY")

wandb.login(key=wandb_key)
login(token=hf_key)

shutil.copy("/kaggle/input/trashnet-training-script/model_training_resnet.py", "model_training_resnet.py")
shutil.copy("/kaggle/input/trashnet-training-script/model_training_swin.py", "model_training_swin.py")
shutil.copy("/kaggle/input/trashnet-training-script/validate_model_resnet.py", "validate_model_resnet.py")
shutil.copy("/kaggle/input/trashnet-training-script/validate_model_swin.py", "validate_model_swin.py")
shutil.copy("/kaggle/input/trashnet-training-script/requirements.txt", "requirements.txt")
shutil.copytree("/kaggle/input/trashnet-data/data", "data")

print("\n========== TRAINING RESNET50 ==========")
exec(open("model_training_resnet.py").read())

print("\n========== TRAINING SWIN TRANSFORMER ==========")
exec(open("model_training_swin.py").read())

print("\n========== VALIDASI RESNET50 ==========")
exec(open("validate_model_resnet.py").read())

print("\n========== VALIDASI SWIN TRANSFORMER ==========")
exec(open("validate_model_swin.py").read())