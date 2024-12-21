from huggingface_hub import HfApi

# Define model name and upload directory
model_name = "ziyadazz/trashnet-swinTransformers"
model_dir = "model" 
processor= "processor"

# Log in to Hugging Face
api = HfApi()

# Upload the model
api.upload_folder(
    repo_id=model_name,
    folder_path=model_dir,
)

# Upload the second folder
api.upload_folder(
    repo_id=model_name,
    folder_path=processor,
)

print(f"Model uploaded to Hugging Face Hub: {model_name}")
