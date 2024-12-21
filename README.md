# SwinV2 Image Classification using Hugging Face and WandB

This project demonstrates image classification using the SwinV2 transformer model from Hugging Face Transformers library. The model is trained on a custom dataset for the task of classifying images into multiple categories.

## Model Description

### SwinV2 Transformer
- **Model Architecture**: SwinV2 (Shifted Window Transformer V2) is a vision transformer designed for high-performance image recognition tasks.
- **Pretrained Model**: The model `microsoft/swinv2-tiny-patch4-window8-256` is used as the base model and fine-tuned for the classification task.
- **Custom Adaptation**: The number of output labels is adjusted to match the number of categories in the dataset, and label-to-ID mappings are defined for proper encoding.

### Training Pipeline

1. **Dataset Preparation**:
   - The dataset is structured into training and validation splits.
   - Images are preprocessed with transformations like resizing, Gaussian blur, sharpness adjustment, and histogram equalization.

2. **Optimizer and Scheduler**:
   - Optimizer: Adam, with a learning rate of 1e-4.
   - Scheduler: Linear warmup followed by decay to ensure smooth optimization.

3. **Metrics for Evaluation**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score

4. **Training Configuration**:
   - Batch size: 32
   - Epochs: 20
   - Evaluation and model saving are performed at specified intervals.

5. **Logging and Tracking**:
   - All training metrics and evaluations are logged using WandB (Weights and Biases), allowing real-time monitoring and analysis.

### Output
- The fine-tuned SwinV2 model is saved after training and can be loaded for inference or further fine-tuning.

## Project Structure

```
.
.github/
└── workflows/
│   └── ci-cd-pipeline.yml
data/
├── test/
├── train/
└── val/
model/
├── config.json
├── model.safetensors
├── preprocessor_config.json
└── training_args.bin
wandb/
.gitattributes
model_training.py
notebook.ipynb
publish_to_hf.py
README.md
requirements.txt
validate_model.py
```

## Setup Instructions

Follow these steps to set up the project locally:

1. **Clone the repository**:
   ```bash
   https://github.com/zzazzz/trashnet.git
   cd trashnet
   ```

2. **Create a virtual environment**:
   - On Ubuntu/Mac:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install dependencies**:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the training script**:
   ```bash
   python model_training.py
   ```

5. **Validate the model**:
   ```bash
   python validate_model.py
   ```

6. **Publish the model**:
   ```bash
   python publish_to_hf.py
   ```

## CI/CD Pipeline

This project includes a GitHub Actions pipeline defined in `.github/workflows/ci_cd_pipeline.yml`. The pipeline consists of the following stages:

1. **Model Training**: Trains the deep learning model.
2. **Validation**: Validates the model on a validation dataset.
3. **Publishing**: Publishes the trained model to the Hugging Face Hub.

