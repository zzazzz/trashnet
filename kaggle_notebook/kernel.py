import os
import subprocess

subprocess.run(["pip", "install", "-q", "wandb", "huggingface_hub", 
                "datasets", "transformers", "torchvision"], check=True)

import wandb
from huggingface_hub import login

wandb.login(key="WANDB_KEY_PLACEHOLDER")
login(token="HF_KEY_PLACEHOLDER")

import shutil
shutil.copy("/kaggle/input/trashnet-training-script/model_training.py", "model_training.py")
shutil.copy("/kaggle/input/trashnet-training-script/requirements.txt", "requirements.txt")
shutil.copy("/kaggle/input/trashnet-training-script/validate_model.py", "validate_model.py")
shutil.copytree("/kaggle/input/trashnet-data", "data")

# Jalankan training
exec(open("model_training.py").read())

# Jalankan validasi setelah training
exec(open("validate_model.py").read())