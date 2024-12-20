from huggingface_hub import HfApi

# Define model name and upload directory
model_name = "ziyadazz/trashnet-swinTransformers"
model_dir = "model"  # Path to the directory where the model is saved

# Log in to Hugging Face
api = HfApi()

# Upload the model
api.upload_folder(
    repo_id=model_name,
    folder_path=model_dir,
)

print(f"Model uploaded to Hugging Face Hub: {model_name}")
