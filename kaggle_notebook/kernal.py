import os
import subprocess

# Install dependencies
subprocess.run(["pip", "install", "-q", "wandb", "huggingface_hub", "datasets", "transformers"], check=True)

# Set environment variables dari Kaggle Secrets
# (tambahkan WANDB_API_KEY dan HF_API_KEY di Kaggle notebook secrets)
import wandb
from huggingface_hub import login

wandb.login(key=os.environ.get("WANDB_API_KEY"))
login(token=os.environ.get("HF_API_KEY"))

# Copy script dari dataset yang diupload
import shutil
shutil.copy("/kaggle/input/trashnet-training-script/model_training.py", "model_training.py")

# Jalankan training
exec(open("model_training.py").read())