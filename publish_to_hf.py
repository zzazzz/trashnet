from huggingface_hub import HfApi

# Define model name and upload directory
model_name = "ziyadazz/trashnet-swinTransformers"
model_dir = "model" 
processor_dir = "processor"

# Log in to Hugging Face
api = HfApi()

# Upload the processor folder (this will overwrite any existing files with the same name)
api.upload_folder(
    repo_id=model_name,
    folder_path=processor_dir,
    commit_message="Update processor files"  # Optional: add a commit message for clarity
)

# Upload the model folder (this will overwrite any existing model files with the same name)
api.upload_folder(
    repo_id=model_name,
    folder_path=model_dir,
    commit_message="Update model files"  # Optional: add a commit message for clarity
)

print(f"Model uploaded and files overwritten at Hugging Face Hub: {model_name}")