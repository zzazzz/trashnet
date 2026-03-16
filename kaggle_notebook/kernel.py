import os
import subprocess

subprocess.run(["pip", "install", "-q", "wandb", "huggingface_hub", 
                "datasets", "transformers", "torchvision"], check=True)

import wandb
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
wandb_key = secrets.get_secret("WANDB_API_KEY")
hf_key = secrets.get_secret("HF_API_KEY")

wandb.login(key=wandb_key)
login(token=hf_key)

import shutil
shutil.copy("/kaggle/input/trashnet-training-script/model_training.py", "model_training.py")
shutil.copy("/kaggle/input/trashnet-training-script/requirements.txt", "requirements.txt")

exec(open("model_training.py").read())