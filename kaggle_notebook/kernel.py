import os
import subprocess

# Install dependencies
subprocess.run(["pip", "install", "-q", "wandb", "huggingface_hub", 
                "datasets", "transformers", "timm"], check=True)

import wandb
from huggingface_hub import login

# Login menggunakan Kaggle Secrets
wandb.login(key=os.environ.get("WANDB_API_KEY"))
login(token=os.environ.get("HF_API_KEY"))

# Copy script dari dataset yang diupload
import shutil
shutil.copy("/kaggle/input/trashnet-training-script/model_training.py", "model_training.py")
shutil.copy("/kaggle/input/trashnet-training-script/requirements.txt", "requirements.txt")

# Jalankan training
exec(open("model_training.py").read())